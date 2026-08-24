"""Measure the Kalshi firehose before committing to an architecture on top of it.

PRD 15 requires flagging anything likely to grow large *before* building it. The
design streams every market live and holds order books in memory, so three
numbers decide whether that is viable:

  1. message + byte rate   -> what raw logging would cost (we don't log raw, but
                              this sets the retention budget)
  2. distinct markets      -> how many live order books we'd maintain
  3. process RSS growth    -> whether it fits in RAM in one Python process

It also settles a question the docs don't answer: whether order-book `seq` is
global to the connection or per-market. That decides whether a gap means
"resync one market" or "resync everything" -- the difference between a system
that survives for weeks and one that thrashes.

Usage:
    python -m tools.measure_firehose --seconds 600 --depth 500 --books
"""

import os
import sys
import time
import asyncio
import argparse
import collections

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalshi import auth, stream, universe  # noqa: E402
from kalshi.orderbook import OrderBook  # noqa: E402


def _rss_mb():
    """Resident memory of this process in MB. Windows, stdlib only."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        return float("nan")


class Meter:
    """Accumulates everything worth knowing about the stream."""

    def __init__(self, keep_books):
        self.keep_books = keep_books
        self.by_type = collections.Counter()
        self.bytes_by_type = collections.Counter()
        self.quote_markets = set()     # seen on ticker: the visible universe
        self.book_markets = set()      # seen on orderbook_*: the depth tier
        self.books = {}
        self.seq_gaps = 0
        self.seq_ok = 0
        self.global_seq_monotonic = True
        self._last_global_seq = None
        self._last_seq = {}

    def on_message(self, message):
        mtype = message.get("type") or "?"
        self.by_type[mtype] += 1
        msg = message.get("msg") or {}
        ticker = msg.get("market_ticker")

        if mtype == "ticker" and ticker:
            self.quote_markets.add(ticker)
        if mtype in ("orderbook_snapshot", "orderbook_delta") and ticker:
            self.book_markets.add(ticker)

        # Sequence semantics. Track both views: if seq is per-market the global
        # view looks like chaos while the per-market view is clean.
        seq = message.get("seq")
        if seq is not None and ticker and mtype.startswith("orderbook"):
            if self._last_global_seq is not None and seq != self._last_global_seq + 1:
                self.global_seq_monotonic = False
            self._last_global_seq = seq

            prev = self._last_seq.get(ticker)
            if prev is not None:
                if seq == prev + 1:
                    self.seq_ok += 1
                else:
                    self.seq_gaps += 1
            self._last_seq[ticker] = seq

        if self.keep_books and ticker:
            if mtype == "orderbook_snapshot":
                book = self.books[ticker] = OrderBook()
                try:
                    book.apply_snapshot(message)
                except Exception:
                    pass
            elif mtype == "orderbook_delta":
                book = self.books.get(ticker)
                if book is not None:
                    try:
                        book.apply_delta(message)
                    except Exception:
                        pass


def active_tickers(universe, limit):
    """The `limit` most-liquid tradeable markets, by open interest.

    Stands in for what the agents will actually pull into depth coverage:
    markets they hold or are evaluating. Open interest is a proxy for "somewhere
    an agent would plausibly be looking" -- the real system picks by Q-value.
    """
    rows = [(m.open_interest or 0.0, m.ticker) for m in universe.tradeable()]
    rows.sort(reverse=True)
    return [t for _, t in rows[:limit]]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("--depth", type=int, default=500,
                    help="markets to pull into full order-book depth")
    ap.add_argument("--books", action="store_true",
                    help="maintain live order books (measures real RAM cost)")
    args = ap.parse_args()

    load_dotenv()
    key_id = os.getenv("KALSHI_KEY_ID")
    private_key = auth.load_private_key(os.getenv("KALSHI_PRIVATE_KEY_PATH"))

    print("sweeping the universe via /events ...", flush=True)
    uni = universe.Universe()
    swept = uni.refresh()
    stats = uni.stats()
    depth = active_tickers(uni, args.depth)
    print(f"  {swept['total']:,} markets in {swept['seconds']:.1f}s  |  "
          f"tradeable {stats['tradeable']:,}  two-sided {stats['quoted']:,}")
    print(f"  depth tier = {len(depth):,} markets")

    meter = Meter(keep_books=args.books)
    rss_start = _rss_mb()

    def on_message(message):
        # Both consumers run inline on the receive loop, exactly as they will in
        # production -- so the measured rate is the real achievable rate, not an
        # optimistic one taken with the universe update stubbed out.
        uni.on_message(message)
        meter.on_message(message)

    s = stream.Stream(key_id, private_key, on_message, depth_tickers=depth)

    print(f"\nmeasuring for {args.seconds}s (books={'on' if args.books else 'off'}), "
          f"RSS start {rss_start:.0f} MB")
    print("   elapsed    msgs/s     MB/s    quoted     depth    RSS MB")

    task = asyncio.create_task(s.run())
    t0 = time.time()
    last_msgs, last_bytes, last_t = 0, 0, t0
    try:
        while time.time() - t0 < args.seconds:
            await asyncio.sleep(30)
            now = time.time()
            dt = now - last_t
            print(f"  {now - t0:8.0f}s {(s.messages - last_msgs) / dt:9.0f} "
                  f"{(s.bytes_in - last_bytes) / dt / 1e6:8.3f} "
                  f"{len(meter.quote_markets):9,d} {len(meter.book_markets):9,d} "
                  f"{_rss_mb():9.0f}", flush=True)
            last_msgs, last_bytes, last_t = s.messages, s.bytes_in, now
    finally:
        s.stop()
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    elapsed = time.time() - t0
    bytes_s = s.bytes_in / elapsed
    raw_gb_day = bytes_s * 86400 / 1e9
    rss_end = _rss_mb()

    print("\n" + "=" * 64)
    print("FIREHOSE MEASUREMENT")
    print("=" * 64)
    print(f"  elapsed             {elapsed:.0f}s   reconnects={s.reconnects} errors={s.errors}")
    print(f"  messages            {s.messages:,}  ({s.messages / elapsed:,.0f}/s)")
    print(f"  bytes               {s.bytes_in / 1e6:,.1f} MB  ({bytes_s / 1e6:.3f} MB/s)")
    print(f"  markets quoted      {len(meter.quote_markets):,}   (broad tier, every market)")
    print(f"  universe unknown    {uni.unknown_ticker_updates:,} ticker updates for "
          f"un-materialized markets (the auto-combo space)")
    print(f"  lifecycle events    {uni.lifecycle_events:,}")
    print(f"  markets with depth  {len(meter.book_markets):,}   (depth tier)")
    print(f"  RSS                 {rss_start:.0f} -> {rss_end:.0f} MB (+{rss_end - rss_start:.0f})")
    if args.books:
        print(f"  order books held    {len(meter.books):,}")
        if meter.books:
            print(f"  RAM per book        ~{(rss_end - rss_start) * 1024 / len(meter.books):.1f} KB")
    print()
    print(f"  RAW LOG WOULD COST  {raw_gb_day:,.1f} GB/day   "
          f"gzipped ~{raw_gb_day / 10:,.1f} GB/day")
    print("  (we do not log raw -- this is why)")
    print()
    print("  order-book sequence semantics")
    print(f"    per-market in-order {meter.seq_ok:,}")
    print(f"    per-market gaps     {meter.seq_gaps:,}")
    print(f"    global monotonic    {meter.global_seq_monotonic}")
    print("    verdict             " + (
        "PER-MARKET -> resync one market on a gap"
        if not meter.global_seq_monotonic else
        "GLOBAL -> a gap means resync everything"))
    print()
    print("  message mix")
    for mtype, n in meter.by_type.most_common(12):
        print(f"    {n:>11,}  {100 * n / max(s.messages, 1):5.1f}%  {mtype}")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
