"""Order books for the depth tier, and connection-level gap detection.

## The finding that shapes this module

Kalshi's `seq` on the orderbook channel is **a single counter for the whole
subscription**, not one per market. Measured directly: subscribing to 8 markets
produced 8 snapshots with seq 1-8, then deltas numbered 9, 10, 11... regardless
of which market each belonged to. Across a 10-minute run of 500 markets the
global sequence had zero breaks, while per-market sequences were gappy by
construction.

This matters because `kalshi/orderbook.py` -- ported verbatim from the earlier
project -- checks `seq != self.last_seq + 1` **per book** and reports a desync
when it fails. That check is exactly right in its original setting, where one
market was subscribed and the connection counter and the market counter were the
same number. At multi-market scale it is wrong in the worst way: every book sees
constant gaps and screams desync on nearly every delta. The 10-minute run
recorded 111,574 such phantom gaps against 105,421 "in-order" messages.

So gap detection belongs **here**, at the connection level, which is where Kalshi
actually puts it. The registry owns one sequence counter; the books own only the
ladder mutation.

## What a real gap means

A break in the connection-wide sequence means we missed a message, but the
sequence number does not say *which market* it belonged to. So a gap invalidates
the whole depth tier and every book must be re-snapshotted.

That is affordable precisely because the depth tier is small -- the markets an
agent holds or is evaluating, hundreds not tens of thousands. It is another
reason not to put all 46k active markets into depth coverage even though the
subscription would be accepted.
"""

import time
import logging

from .orderbook import OrderBook

log = logging.getLogger(__name__)


class BookRegistry:
    """Live order books for the depth tier, keyed by market ticker.

    Not thread-safe by design: this is written only from the stream's receive
    loop. Readers take `snapshot_of()`, which returns plain data.
    """

    def __init__(self, on_desync=None):
        self.books = {}
        self.on_desync = on_desync

        # The connection-wide orderbook sequence. See the module docstring.
        self.last_seq = None
        self.gaps = 0
        self.snapshots = 0
        self.deltas = 0
        self.desynced_at = None
        self.updated_at = {}

    # ---------- ingest ----------

    def on_message(self, message):
        """Apply one orderbook frame. Returns True if a gap was detected."""
        mtype = message.get("type")
        if mtype == "orderbook_snapshot":
            return self._on_snapshot(message)
        if mtype == "orderbook_delta":
            return self._on_delta(message)
        return False

    def _check_seq(self, seq):
        """Track the connection-wide counter. True when a message was missed.

        A snapshot resets the counter rather than being checked against it: after
        a resubscribe Kalshi restarts the sequence, so treating that restart as a
        gap would put us in a resync loop that never converges.
        """
        if seq is None:
            return False
        if self.last_seq is not None and seq != self.last_seq + 1:
            self.gaps += 1
            self.desynced_at = time.time()
            return True
        self.last_seq = seq
        return False

    def _on_snapshot(self, message):
        ticker = message["msg"].get("market_ticker")
        if not ticker:
            return False
        book = self.books.get(ticker)
        if book is None:
            book = self.books[ticker] = OrderBook()
        book.apply_snapshot(message)
        # A snapshot is ground truth, so adopt its sequence rather than
        # validating against the old one.
        self.last_seq = message.get("seq", self.last_seq)
        self.snapshots += 1
        self.updated_at[ticker] = time.time()
        return False

    def _on_delta(self, message):
        seq = message.get("seq")
        gapped = self._check_seq(seq)
        if gapped:
            log.warning("orderbook sequence gap at seq=%s; depth tier is stale", seq)
            if self.on_desync:
                self.on_desync(list(self.books))
            # Everything is suspect until fresh snapshots arrive.
            for book in self.books.values():
                book.synced = False
            self.last_seq = seq
            return True

        ticker = message["msg"].get("market_ticker")
        book = self.books.get(ticker)
        if book is None or not book.synced:
            # A delta for a market we have no snapshot for is not an error --
            # it arrives in the window between subscribing and the snapshot.
            return False

        # `OrderBook.apply_delta` re-checks the sequence per book, which is
        # meaningless here (see the module docstring) and would reject nearly
        # every delta. Align the book's counter so its check always passes and
        # it does only what we want from it: mutate the ladder. Real gap
        # detection already happened above, against the connection counter.
        book.last_seq = seq - 1
        book.apply_delta(message)
        self.deltas += 1
        self.updated_at[ticker] = time.time()
        return False

    # ---------- lifecycle ----------

    def reset(self):
        """Drop everything. Called on reconnect, when all books are invalid."""
        self.books.clear()
        self.updated_at.clear()
        self.last_seq = None

    def forget(self, tickers):
        for ticker in tickers:
            self.books.pop(ticker, None)
            self.updated_at.pop(ticker, None)

    # ---------- reads ----------

    def get(self, ticker):
        """The live book for `ticker`, or None if we have no synced copy."""
        book = self.books.get(ticker)
        return book if book is not None and book.synced else None

    def has_depth(self, ticker):
        return self.get(ticker) is not None

    def age_of(self, ticker, now=None):
        """Seconds since this book last changed, or None if never."""
        stamp = self.updated_at.get(ticker)
        return ((now or time.time()) - stamp) if stamp else None

    def stats(self):
        synced = sum(1 for b in self.books.values() if b.synced)
        return {
            "books": len(self.books),
            "synced": synced,
            "snapshots": self.snapshots,
            "deltas": self.deltas,
            "gaps": self.gaps,
            "last_seq": self.last_seq,
            "desynced_at": self.desynced_at,
        }
