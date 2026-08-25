"""Force a WebSocket disconnection and confirm the system recovers.

PRD 11 requires graceful handling of disconnects -- "a long-running system will
definitely drop connections". That claim is easy to make in a docstring and easy
to get wrong in practice, so this actually severs the socket and checks what
happens next.

What recovery has to mean here, beyond "it reconnected":

  - Backoff with jitter, not a hot retry loop.
  - Every order book is dropped. A new connection restarts the sequence, so any
    book carried across the gap would be silently wrong -- which is worse than
    having no book at all, because the agent would trade on it.
  - The depth subscription is rebuilt, not merely re-requested for new markets.

    python -m tools.test_reconnect
"""

import os
import sys
import time
import asyncio
import logging
import contextlib

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kalshi import auth, stream, universe  # noqa: E402
from kalshi.books import BookRegistry  # noqa: E402


async def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    load_dotenv()

    uni = universe.Universe()
    uni.refresh()
    depth = [m.ticker for m in
             sorted(uni.tradeable(), key=lambda m: -(m.open_interest or 0))[:400]]

    books = BookRegistry()
    s = stream.Stream(os.getenv("KALSHI_KEY_ID"),
                      auth.load_private_key(os.getenv("KALSHI_PRIVATE_KEY_PATH")),
                      lambda m: (uni.on_message(m), books.on_message(m)),
                      depth_tickers=depth)
    books.on_desync = s.request_resync
    # Wire it the way run.py does: a new connection restarts the sequence, so
    # every book carried across the gap would be silently wrong.
    s.on_reconnect = books.reset

    task = asyncio.create_task(s.run())
    await asyncio.sleep(25)

    before = books.stats()
    print(f"\nBEFORE  connected={s.connected} books={before['synced']}/"
          f"{before['books']} gaps={before['gaps']} reconnects={s.reconnects}")
    assert before["synced"] > 0, "no books synced before the test even started"

    print(">>> severing the socket")
    severed_at = time.time()
    await s._ws.close(code=1011, reason="forced")

    # Wait for it to come back AND rebuild most of the depth tier -- merely
    # reconnecting is not recovery if the books never come back.
    deadline = severed_at + 90
    recovered_in = None
    while time.time() < deadline:
        await asyncio.sleep(2)
        if (s.connected and s.reconnects > 0
                and books.stats()["synced"] >= 0.9 * len(depth)):
            recovered_in = time.time() - severed_at
            break

    after = books.stats()
    connected = s.connected                 # capture BEFORE stopping: the
    reconnects = s.reconnects               # shutdown path clears `connected`
    recovered_in = recovered_in or (time.time() - severed_at)
    print(f"AFTER   connected={connected} books={after['synced']}/"
          f"{after['books']} gaps={after['gaps']} reconnects={reconnects}")
    print(f"        depth tier rebuilt in {recovered_in:.1f}s")

    s.stop()
    task.cancel()
    with contextlib.suppress(BaseException):
        await task

    ok = (reconnects >= 1 and connected
          and after["synced"] >= 0.9 * len(depth)
          and recovered_in < 90)
    print("\nVERDICT:", "RECOVERED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
