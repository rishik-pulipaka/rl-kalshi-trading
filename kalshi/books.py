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

## What `seq` actually counts

This took several wrong answers to pin down, so the evidence lives here rather
than in a commit message.

`seq` is a **connection-wide counter over every sequenced frame**, whatever
channel or subscription it belongs to. It is not per market, and it is not per
subscription -- `sid` merely labels which subscription a frame came from.

The measurements that settle it:

  - Subscribe to 8 markets with no other channels: snapshots arrive at seq 1-8,
    then deltas continue 9, 10, 11 regardless of market. So: not per market.
  - Add a second subscribe command: the acknowledgement itself consumes a seq,
    and the following data frame continues the same run. So: not per
    subscription, and control frames count.
  - Run the full system: gaps appeared on sid=4 at seq 5,558 then 8,892 then
    11,687, climbing to 1,062,749 in four minutes -- against 1,182,205 total
    messages received. The order-book "sequence" was tracking the entire
    connection's traffic, because `trade` and lifecycle frames consume numbers
    too.

So the registry must observe **every** sequenced frame, not just order-book
ones. Feed it only order-book messages and it sees a gap on almost every
delta, which is precisely what happened: a live run held 3,119 books with
**zero** synced, every fill silently fell back to top-of-book instead of the
real ladder, and the only symptom in the logs was a climbing gap count.

`tools/probe_sequence.py` reproduces the measurement.

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

        # One sequence counter per subscription id. See the module docstring:
        # each `subscribe` command gets its own sid and its own sequence, and
        # the broad tier's subscriptions run their own counters in parallel.
        self.last_seq = {}
        self.gaps = 0
        self.snapshots = 0
        self.deltas = 0
        self.desynced_at = None
        self.updated_at = {}

    # ---------- ingest ----------

    def on_message(self, message):
        """Apply one sequenced frame. Returns True if a gap was detected.

        Every frame that carries a `seq` advances its sid's counter, including
        `subscribed` acknowledgements -- those consume a sequence number, and
        ignoring them makes the following data frame look like a loss.
        """
        mtype = message.get("type")
        if mtype == "orderbook_snapshot":
            return self._on_snapshot(message)
        if mtype == "orderbook_delta":
            return self._on_delta(message)
        # A frame from some other channel, or an acknowledgement. It still
        # consumes a sequence number, so the counter has to follow it -- but it
        # is never order-book evidence, and a gap detected here is somebody
        # else's problem, not a reason to invalidate the books.
        self._check_seq(message.get("sid"), message.get("seq"))
        return False

    def _check_seq(self, sid, seq):
        """Track one subscription's counter. True when a message was missed."""
        if seq is None:
            return False
        previous = self.last_seq.get(sid)
        gapped = previous is not None and seq != previous + 1
        if gapped:
            self.gaps += 1
            self.desynced_at = time.time()
        self.last_seq[sid] = seq
        return gapped

    def _on_snapshot(self, message):
        """A fresh book for one market.

        The snapshot is applied whatever happened around it -- it is ground
        truth for its own market. A gap alongside it only invalidates the others.
        """
        ticker = message["msg"].get("market_ticker")
        if not ticker:
            return False

        gapped = self._check_seq(message.get("sid"), message.get("seq"))

        book = self.books.get(ticker)
        if book is None:
            book = self.books[ticker] = OrderBook()
        book.apply_snapshot(message)
        self.snapshots += 1
        self.updated_at[ticker] = time.time()

        if gapped:
            for other, other_book in self.books.items():
                if other != ticker:
                    other_book.synced = False
            if self.on_desync:
                self.on_desync([t for t in self.books if t != ticker])
        return gapped

    def _on_delta(self, message):
        seq = message.get("seq")
        if self._check_seq(message.get("sid"), seq):
            log.warning("orderbook gap on sid=%s at seq=%s; depth tier stale",
                        message.get("sid"), seq)
            for book in self.books.values():
                book.synced = False
            if self.on_desync:
                self.on_desync(list(self.books))
            return True

        ticker = message["msg"].get("market_ticker")
        book = self.books.get(ticker)
        if book is None or not book.synced:
            # A delta for a market we have no snapshot for is not an error --
            # it arrives in the window between subscribing and the snapshot.
            return False

        # `OrderBook.apply_delta` re-checks the sequence per book, which is
        # meaningless here and would reject nearly every delta. Align its
        # counter so it does only what we want: mutate the ladder. Real gap
        # detection already happened above.
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
        self.last_seq.clear()

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
            "sids": len(self.last_seq),
            "desynced_at": self.desynced_at,
        }
