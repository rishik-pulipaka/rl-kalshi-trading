"""Local Kalshi order book maintained from `orderbook_snapshot` + `orderbook_delta`
messages. The book has two sides: YES bids and NO bids. A NO bid at price p is
equivalent to a YES ask at (1 - p), so the YES ask ladder is derived from the NO
side. Prices arrive as 4-decimal dollar strings ("0.5100"); we key levels by
`round(price * 10000)` (integer ten-thousandths of a dollar) to avoid float keys.

Each message carries a top-level `seq` that increments by 1; a gap means we
missed messages and must resubscribe to get a fresh snapshot."""


def price_key(price_dollars):
    return round(float(price_dollars) * 10000)


class OrderBook:
    def __init__(self):
        self.yes = {}   # price_key -> size (YES bids)
        self.no = {}    # price_key -> size (NO bids)
        self.last_seq = None
        self.synced = False

    def _load(self, target, rows):
        target.clear()
        for price, size in rows:
            s = float(size)
            if s > 0:
                target[price_key(price)] = s

    def apply_snapshot(self, message):
        msg = message["msg"]
        self._load(self.yes, msg.get("yes_dollars_fp", []))
        self._load(self.no, msg.get("no_dollars_fp", []))
        self.last_seq = message["seq"]
        self.synced = True

    def apply_delta(self, message):
        """Apply an incremental change. Returns False on a sequence gap, meaning
        the caller must resubscribe and rebuild from a fresh snapshot."""
        seq = message["seq"]
        if self.last_seq is not None and seq != self.last_seq + 1:
            self.synced = False
            return False
        msg = message["msg"]
        book = self.yes if msg["side"] == "yes" else self.no
        k = price_key(msg["price_dollars"])
        # Round to 2 decimals (the sizes' native precision). Without this, float
        # drift over thousands of deltas leaves tiny positive residues at levels
        # that should be exactly zero — phantom levels that can cross the book.
        book[k] = round(book.get(k, 0.0) + float(msg["delta_fp"]), 2)
        if book[k] <= 0:
            book.pop(k, None)
        self.last_seq = seq
        return True

    def best_yes_bid(self):
        return max(self.yes) / 10000 if self.yes else None

    def best_yes_ask(self):
        # Lowest YES ask = 1 - highest NO bid.
        return (10000 - max(self.no)) / 10000 if self.no else None

    def yes_bid_levels(self, n):
        keys = sorted(self.yes, reverse=True)[:n]
        return [(k / 10000, self.yes[k]) for k in keys]

    def yes_ask_levels(self, n):
        # Highest NO bids become the lowest YES asks.
        keys = sorted(self.no, reverse=True)[:n]
        return [((10000 - k) / 10000, self.no[k]) for k in keys]

    def yes_bid_total(self):
        return sum(self.yes.values())

    def yes_ask_total(self):
        return sum(self.no.values())
