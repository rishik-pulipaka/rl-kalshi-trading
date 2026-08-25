"""The live, in-memory view of every market on Kalshi.

This is what an agent "sees". It is built once from a REST sweep and then kept
current by the broad WebSocket tier -- `ticker` updates quotes, `trade` updates
last-trade info, `market_lifecycle_v2` adds and retires markets.

## Why /events and not /markets

Both endpoints reach the same data, but they are not remotely equal in cost
(measured, not assumed):

    /markets?status=open                 1,286,907 markets   1,287 pages   ~4 min
    /events?with_nested_markets=true        96,315 markets      59 pages   ~12 s

The gap is Kalshi's auto-generated cross-category parlay space -- ~1.19M
machine-produced combinations that `/markets` returns and `/events` does not.
`/events` also carries the category and event grouping we need anyway for the
PRD's market-preference analytics, which `/markets` does not include.

So the sweep is cheap enough to re-run every few minutes, which is what keeps
the universe honest without any incremental-diffing machinery.

## Auto-generated combos

Those ~1.19M parlays are still *reachable*: they stream in on `ticker` like
everything else, a bounded sample of the ones actually quoting is kept in
`_unknown`, and `adopt_many()` pulls a batch into the universe on demand -- at
which point an agent evaluates a Kalshi combo exactly as it evaluates any other
market. That is the "take a multi-leg combo" half of PRD 2.

They are not materialized up front because holding 1.29M market objects to
represent combinations of markets we already hold is waste, not freedom. Every
market a human would recognize as a market is here from the start, in every
category, with no filtering.

## Prices

Everything is stored as integer ten-thousandths of a dollar (0.4200 -> 4200),
matching `orderbook.py`'s convention. Kalshi sends prices as decimal strings;
parsing once on ingest keeps floats out of the hot path and out of comparisons.
"""

import time
import random
import collections
import threading

from . import rest

# A market whose quote hasn't updated in this long is stale -- present on the
# exchange, but nothing is happening. Used for reporting, not for exclusion.
STALE_AFTER_S = 900.0

# Kalshi's market status vocabulary. An open, tradeable market is "active" --
# NOT "open", which is the status of the *event*. Getting this wrong silently
# yields an empty candidate pool, since every market fails the filter.
# Observed values: initialized, active, inactive, closed, determined, finalized.
TRADEABLE_STATUS = "active"

# Prices are integer ten-thousandths, so a contract spans 0..10000. A side
# quoted at exactly 0 or 10000 is not a real quote -- it means nobody is there.
PRICE_MIN = 0
PRICE_MAX = 10000

# How many un-materialized tickers to keep a handle on. These are Kalshi's
# auto-generated parlays: ~1.19M exist, so this is a sampling window rather
# than an index.
UNKNOWN_RESERVOIR = 4000

# Batch size for `adopt_many`. The /markets?tickers=a,b,c form is verified.
ADOPT_BATCH = 100


def to_price_key(value):
    """'0.4200' -> 4200. None/'' -> None. Integer ten-thousandths of a dollar."""
    if value is None or value == "":
        return None
    try:
        return round(float(value) * 10000)
    except (TypeError, ValueError):
        return None


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class Market:
    """One tradeable market. `__slots__` because there are ~96k of these."""

    __slots__ = ("ticker", "event_ticker", "series", "category", "title",
                 "subtitle", "close_ts", "yes_bid", "yes_ask", "yes_bid_size",
                 "yes_ask_size", "last_price", "volume", "open_interest",
                 "updated_at", "created_at", "status")

    def __init__(self, ticker, event_ticker=None, series=None, category=None,
                 title=None, subtitle=None, close_ts=None):
        self.ticker = ticker
        self.event_ticker = event_ticker
        self.series = series or rest.series_of(ticker)
        self.category = category
        self.title = title
        self.subtitle = subtitle
        self.close_ts = close_ts
        self.status = TRADEABLE_STATUS

        self.yes_bid = None
        self.yes_ask = None
        self.yes_bid_size = None
        self.yes_ask_size = None
        self.last_price = None
        self.volume = None
        self.open_interest = None

        self.created_at = time.time()
        self.updated_at = None

    # ---------- derived quantities the agent's features are built from ----------

    @property
    def mid(self):
        """Mid price in ten-thousandths, or None if either side is missing."""
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) // 2

    @property
    def spread(self):
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    @property
    def can_buy_yes(self):
        """Someone is offering YES, so the agent could take it."""
        return self.yes_ask is not None and PRICE_MIN < self.yes_ask < PRICE_MAX

    @property
    def can_buy_no(self):
        """Someone is bidding YES, which is the same as offering NO at 1-p."""
        return self.yes_bid is not None and PRICE_MIN < self.yes_bid < PRICE_MAX

    @property
    def is_quoted(self):
        """A genuine two-sided market: you could enter on either side.

        Note a `yes_bid` of 0 is not a bid -- it is the absence of one. Kalshi
        sends "0.0000" rather than omitting the field, so a naive None-check
        counts every untraded market as quoted.
        """
        return self.can_buy_yes and self.can_buy_no

    @property
    def is_tradeable(self):
        """At least one side can be entered. Weaker than `is_quoted`, and the
        right test for the agent's candidate pool -- a one-sided market is still
        a market you can take a position in."""
        return self.can_buy_yes or self.can_buy_no

    def seconds_to_close(self, now=None):
        if not self.close_ts:
            return None
        return self.close_ts - (now or time.time())

    def is_stale(self, now=None):
        if self.updated_at is None:
            return True
        return (now or time.time()) - self.updated_at > STALE_AFTER_S

    def __repr__(self):
        return f"<Market {self.ticker} {self.yes_bid}/{self.yes_ask}>"


class Universe:
    """Thread-safe registry of every market, plus the live quote state.

    The WebSocket loop writes; agent threads and the dashboard read. A single
    lock is enough because writes are short dict/attribute assignments -- there
    is no long-held work under the lock.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._markets = {}
        self.last_sweep_at = None
        self.last_sweep_seconds = None
        # Ticker updates for markets we never materialized (the auto-combo
        # space). Counted so we can report the real size of what's out there.
        self.unknown_ticker_updates = 0
        self.lifecycle_events = 0
        # A bounded sample of those un-materialized tickers, so `adopt_many`
        # has something to pull from. Bounded because there are ~1.19M of them
        # and holding every ticker string would defeat the point of not
        # materializing them. Only ones that are actually trading get in --
        # a parlay nobody has quoted is not a market an agent could take.
        self._unknown = collections.OrderedDict()

    # ---------- population ----------

    def refresh(self, session=None):
        """Re-sweep the universe from REST. Cheap enough to run on a timer.

        Adds new markets and updates metadata on existing ones, but never
        overwrites live quote state -- the stream is fresher than the sweep.
        """
        started = time.time()
        seen = set()
        added = 0
        for row in _iter_event_markets(session=session):
            ticker = row.get("ticker")
            if not ticker:
                continue
            seen.add(ticker)
            with self._lock:
                market = self._markets.get(ticker)
                if market is None:
                    market = self._markets[ticker] = Market(ticker)
                    added += 1
                _apply_rest_row(market, row)

        with self._lock:
            # Markets that vanished from the open sweep have closed.
            for ticker, market in self._markets.items():
                if ticker not in seen and market.status == TRADEABLE_STATUS:
                    market.status = "closed"
            self.last_sweep_at = time.time()
            self.last_sweep_seconds = self.last_sweep_at - started
        return {"total": len(seen), "added": added,
                "seconds": self.last_sweep_seconds}

    def adopt(self, ticker, session=None):
        """Materialize one market we didn't sweep -- e.g. an auto-generated combo.

        This is how the ~1.19M parlay space stays reachable without being held
        in memory: an agent that wants one asks for it by ticker.
        """
        with self._lock:
            if ticker in self._markets:
                return self._markets[ticker]
        data = rest._get("/markets/" + ticker, session=session)
        row = data.get("market") or {}
        with self._lock:
            market = self._markets.get(ticker) or Market(ticker)
            _apply_rest_row(market, row)
            self._markets[ticker] = market
            return market

    def unknown_sample(self, n, rng=None):
        """A sample of tradeable markets we have seen but not materialized.

        This is how the ~1.19M auto-generated parlay space stays reachable
        (PRD 2: agents may "take" multi-leg combos). Only markets that have
        actually quoted are in here.
        """
        with self._lock:
            pool = [t for t in self._unknown if t not in self._markets]
        if not pool:
            return []
        n = min(n, len(pool))
        return (rng or random).sample(pool, n)

    def adopt_many(self, tickers, session=None):
        """Materialize several markets at once, batched.

        Used to pull listed parlays into the universe so they become ordinary
        candidates -- the agent then evaluates a Kalshi combo exactly as it
        evaluates anything else, which is the point.
        """
        tickers = [t for t in tickers if t]
        added = 0
        for i in range(0, len(tickers), ADOPT_BATCH):
            batch = tickers[i:i + ADOPT_BATCH]
            try:
                data = rest._get("/markets", {"tickers": ",".join(batch),
                                              "limit": ADOPT_BATCH},
                                 session=session)
            except Exception:
                continue
            for row in data.get("markets", []):
                ticker = row.get("ticker")
                if not ticker:
                    continue
                with self._lock:
                    market = self._markets.get(ticker)
                    if market is None:
                        market = self._markets[ticker] = Market(ticker)
                        added += 1
                        market.category = market.category or "Combo"
                    _apply_rest_row(market, row)
                    self._unknown.pop(ticker, None)
        return added

    # ---------- live stream application ----------

    def on_message(self, message):
        """Apply one WebSocket frame. Called on the stream's receive loop.

        Must stay cheap: this runs for every message on the firehose.
        """
        mtype = message.get("type")
        msg = message.get("msg") or {}
        ticker = msg.get("market_ticker")
        if not ticker:
            return

        if mtype == "ticker":
            self._on_ticker(ticker, msg)
        elif mtype == "trade":
            self._on_trade(ticker, msg)
        elif mtype == "market_lifecycle_v2":
            self._on_lifecycle(ticker, msg)

    def _on_ticker(self, ticker, msg):
        with self._lock:
            market = self._markets.get(ticker)
            if market is None:
                # An auto-generated combo, or a market opened since the last
                # sweep. Not materialized -- see the module docstring -- but
                # remembered so an agent can pull it in on demand.
                self.unknown_ticker_updates += 1
                ask = to_price_key(msg.get("yes_ask_dollars"))
                if ask is not None and PRICE_MIN < ask < PRICE_MAX:
                    self._unknown[ticker] = None
                    if len(self._unknown) > UNKNOWN_RESERVOIR:
                        self._unknown.popitem(last=False)
                return
            market.yes_bid = to_price_key(msg.get("yes_bid_dollars"))
            market.yes_ask = to_price_key(msg.get("yes_ask_dollars"))
            market.yes_bid_size = _f(msg.get("yes_bid_size_fp"))
            market.yes_ask_size = _f(msg.get("yes_ask_size_fp"))
            price = to_price_key(msg.get("price_dollars"))
            if price is not None:
                market.last_price = price
            volume = _f(msg.get("volume_fp"))
            if volume is not None:
                market.volume = volume
            oi = _f(msg.get("open_interest_fp"))
            if oi is not None:
                market.open_interest = oi
            market.updated_at = time.time()

    def _on_trade(self, ticker, msg):
        with self._lock:
            market = self._markets.get(ticker)
            if market is None:
                return
            price = to_price_key(msg.get("yes_price_dollars"))
            if price is not None:
                market.last_price = price
            market.updated_at = time.time()

    def _on_lifecycle(self, ticker, msg):
        """Markets opening, closing, and having their close time moved.

        This is what keeps the universe current between REST sweeps.
        """
        self.lifecycle_events += 1
        event = msg.get("event_type")
        with self._lock:
            market = self._markets.get(ticker)
            if market is None:
                return
            if event in ("close_date_updated", "opened", "reopened", "activated"):
                close_ts = msg.get("close_ts")
                if close_ts:
                    market.close_ts = close_ts
                if event in ("opened", "reopened", "activated"):
                    market.status = TRADEABLE_STATUS
            elif event in ("closed", "settled", "determined"):
                market.status = "closed"

    # ---------- reads ----------

    def get(self, ticker):
        with self._lock:
            return self._markets.get(ticker)

    def __len__(self):
        with self._lock:
            return len(self._markets)

    def tickers(self):
        with self._lock:
            return list(self._markets)

    def tradeable(self, now=None):
        """Every open market with a live two-sided quote.

        This is the agent's candidate pool: no category filter, no series filter,
        no human deciding what is worth looking at.
        """
        now = now or time.time()
        with self._lock:
            return [m for m in self._markets.values()
                    if m.status == TRADEABLE_STATUS and m.is_tradeable
                    and (m.close_ts is None or m.close_ts > now)]

    def stats(self):
        """Summary for the dashboard and for startup logging."""
        now = time.time()
        with self._lock:
            markets = list(self._markets.values())
        quoted = sum(1 for m in markets if m.is_quoted)
        tradeable = sum(1 for m in markets if m.status == TRADEABLE_STATUS
                        and m.is_tradeable
                        and (m.close_ts is None or m.close_ts > now))
        fresh = sum(1 for m in markets if not m.is_stale(now))
        categories = {}
        for m in markets:
            categories[m.category or "?"] = categories.get(m.category or "?", 0) + 1
        return {
            "markets": len(markets),
            "quoted": quoted,
            "tradeable": tradeable,
            "fresh": fresh,
            "categories": categories,
            "last_sweep_at": self.last_sweep_at,
            "last_sweep_seconds": self.last_sweep_seconds,
            "unknown_ticker_updates": self.unknown_ticker_updates,
            "unknown_reservoir": len(self._unknown),
            "lifecycle_events": self.lifecycle_events,
        }


# ---------- helpers ----------

def _apply_rest_row(market, row):
    """Copy metadata from a REST market dict onto a Market.

    Quote fields are only filled in when the market has never had a live update:
    a 12-second-old sweep must not clobber a 200ms-old stream quote.
    """
    market.event_ticker = row.get("event_ticker") or market.event_ticker
    market.category = row.get("category") or market.category
    market.series = row.get("series_ticker") or market.series
    market.title = row.get("title") or market.title
    market.subtitle = row.get("subtitle") or row.get("yes_sub_title") or market.subtitle
    if row.get("status"):
        market.status = row["status"]

    close_ts = _parse_iso(row.get("close_time"))
    if close_ts:
        market.close_ts = close_ts

    if market.updated_at is None:
        market.yes_bid = to_price_key(row.get("yes_bid_dollars"))
        market.yes_ask = to_price_key(row.get("yes_ask_dollars"))
        market.last_price = to_price_key(row.get("last_price_dollars"))
        market.volume = _f(row.get("volume_fp"))
        market.open_interest = _f(row.get("open_interest_fp"))


def _parse_iso(value):
    """Kalshi ISO-8601 -> unix seconds. Tolerant of fractional seconds."""
    if not value:
        return None
    import datetime as dt
    text = value.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _iter_event_markets(session=None):
    """Yield every open market, carrying its event's category down onto it.

    Category lives on the event, not the market, but the agent reasons about
    markets -- so it gets denormalized here, once, at ingest.
    """
    for page in rest._paginate("/events",
                               {"status": "open", "with_nested_markets": "true",
                                "limit": 200}, session=session):
        for event in page.get("events", []):
            category = event.get("category")
            series = event.get("series_ticker")
            event_ticker = event.get("event_ticker")
            title = event.get("title")
            for market in event.get("markets") or []:
                row = dict(market)
                row.setdefault("event_ticker", event_ticker)
                row["category"] = category
                row["series_ticker"] = series
                row["title"] = title
                yield row
