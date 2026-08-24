"""Determine what Kalshi's order-book `seq` counter is actually scoped to.

This question has now been wrong twice, so it gets a tool instead of a guess.

  - Per market?  An early probe said no: subscribing to 8 markets produced
    snapshots at seq 1-8 and then deltas at 9, 10, 11 regardless of market.
  - Per connection?  That was the next assumption, and the depth tier then ran
    with 3,000+ books held and ZERO synced -- because the agents keep issuing
    *new* subscribe commands as they discover markets, and something about that
    resets the sequence.

Each Kalshi message carries a `sid` (subscription id) alongside `seq`. This
probe subscribes, waits, subscribes again, and reports whether the sequence is
contiguous per-sid or globally -- which settles it.

    python -m tools.probe_sequence
"""

import os
import sys
import asyncio
import logging
import collections
import contextlib

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalshi import auth, stream, universe  # noqa: E402


async def main():
    logging.disable(logging.CRITICAL)
    load_dotenv()

    uni = universe.Universe()
    uni.refresh()
    tickers = [m.ticker for m in
               sorted(uni.tradeable(), key=lambda m: -(m.open_interest or 0))[:20]]

    seen = []

    def on_message(message):
        mtype = message.get("type") or ""
        if mtype.startswith("orderbook"):
            seen.append((message.get("sid"), message.get("seq"), mtype,
                         (message.get("msg") or {}).get("market_ticker")))

    key_id = os.getenv("KALSHI_KEY_ID")
    private_key = auth.load_private_key(os.getenv("KALSHI_PRIVATE_KEY_PATH"))
    s = stream.Stream(key_id, private_key, on_message,
                      broad_channels=[], depth_tickers=tickers[:10])

    task = asyncio.create_task(s.run())
    await asyncio.sleep(12)
    first_batch = len(seen)
    print(">>> issuing a SECOND subscribe command for 10 more markets")
    s.watch_depth(tickers[10:20])
    await asyncio.sleep(15)
    s.stop()
    task.cancel()
    with contextlib.suppress(BaseException):
        await task

    by_sid = collections.defaultdict(list)
    for sid, seq, _, _ in seen:
        by_sid[sid].append(seq)

    print(f"\ntotal order-book messages: {len(seen)}  "
          f"(before second subscribe: {first_batch})")
    print(f"distinct sids: {sorted(k for k in by_sid if k is not None)}")
    for sid in sorted(by_sid, key=lambda k: (k is None, k)):
        seqs = by_sid[sid]
        contiguous = all(b == a + 1 for a, b in zip(seqs, seqs[1:]))
        print(f"  sid={sid}: n={len(seqs):5d} first={seqs[0]:6d} "
              f"last={seqs[-1]:6d} contiguous={contiguous}")

    flat = [q for _, q, _, _ in seen]
    breaks = sum(1 for a, b in zip(flat, flat[1:]) if b != a + 1)
    print(f"\nignoring sid entirely, sequence breaks: {breaks}")

    print("\naround the second subscribe:")
    for row in seen[max(first_batch - 3, 0): first_batch + 8]:
        print(f"    sid={row[0]} seq={row[1]} {row[2]:20s} {(row[3] or '')[:34]}")

    verdict = ("PER-SID -- track one counter per subscription id"
               if len(by_sid) > 1 and breaks > 0
               else "single counter across the connection")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
