"""Phase A: learning basic mechanics from resolved markets, fast (PRD 3.1).

Live learning runs at one episode per real day. That is fine for discovering
*strategy* but hopeless for discovering *mechanics* -- an agent should not need
three weeks of calendar time to work out that a cheaper contract pays more per
dollar risked. So before going live, each agent runs through many compressed
"days" of markets that have already resolved.

PRD 3.1 is specific about the goal and the limit: this exists to get agents past
"doesn't understand the game at all", not to pre-solve the problem. The majority
of eventual competence is supposed to come from live experience, and the handoff
is clean -- once pretraining ends, no historical training happens in the
background ever again.

## The leakage trap, which is the whole difficulty here

A settled market's `last_price` is the **post-settlement print**: 0.9990 for the
winning side, 0.0010 for the loser. Confirmed live. Training on it does not
teach an agent to forecast -- it hands over the answer, produces an agent that
looks brilliant in backtest and clueless the moment it goes live, and gives no
warning that anything is wrong.

So every price here comes from **candlesticks**, and only from candles that
closed strictly before the market's own close time. Nothing else is allowed to
touch a price on a settled market. This trap has already cost this codebase's
author twice on previous Kalshi projects, which is why the guard is a function
with its own tests rather than a comment.

## What it trains on

Exactly what live trading trains on: a feature vector built the same way by
`agent/features.py`, and a realized return computed the same way by
`agent/reward.trade_target`. The only difference is where the price came from.
Memory accumulates too, so an agent arrives at its first live day with beliefs
already formed.

Usage:
    python -m agent.pretrain --agent stan --series KXBTC15M --days 30
"""

import os
import json
import time
import random
import logging
import datetime as dt

from sim import money, fills
from sim.fills import YES, NO
from kalshi import rest
from .features import build as build_features, entry_price
from .reward import trade_target

log = logging.getLogger(__name__)

# Both the candle interval and the lookahead buffer scale with how long a market
# actually lived. Kalshi markets span fifteen minutes (KXBTC15M) to months
# (elections), and fixed values break at one end or the other.
#
# The first version used a flat hourly candle and a flat one-hour buffer, and
# produced ZERO training examples from 1,000 resolved BTC markets: the buffer
# was four times longer than the market's entire lifetime, so every candle was
# discarded as potentially leaked. It failed silently -- no error, just nothing.
LOOKAHEAD_FRACTION = 0.20      # discard the last fifth of a market's life
MIN_BUFFER_S = 60
MAX_BUFFER_S = 3600


def candle_minutes(duration_s):
    """Candle granularity for a market that lived `duration_s`.

    Kalshi accepts 1, 60, or 1440 minutes. Anything under a few hours needs
    minute candles or there is nothing to look at.
    """
    if duration_s <= 6 * 3600:
        return 1
    if duration_s <= 30 * 86400:
        return 60
    return 1440


def lookahead_buffer(duration_s):
    """How much of the end of a market's life to treat as unusable."""
    if not duration_s or duration_s <= 0:
        return MIN_BUFFER_S
    return max(MIN_BUFFER_S,
               min(MAX_BUFFER_S, int(duration_s * LOOKAHEAD_FRACTION)))


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def candle_price(candle):
    """Mid of a candle's closing bid/ask, falling back to its close price."""
    def field(section, key):
        value = (candle.get(section) or {}).get(key)
        return _f(value)

    bid = field("yes_bid", "close_dollars")
    ask = field("yes_ask", "close_dollars")
    if bid is not None and ask is not None:
        return round((bid + ask) / 2 * money.ONE_DOLLAR)
    close = field("price", "close_dollars")
    return round(close * money.ONE_DOLLAR) if close is not None else None


def safe_candles(candles, close_ts, buffer_s=MIN_BUFFER_S):
    """Only candles that ended safely before the market resolved.

    This is the guard the whole module exists around. A candle whose window
    extends past the cutoff may already reflect the outcome, and one leaked
    candle is enough to teach an agent that it can see the future.
    """
    if not close_ts:
        return []
    cutoff = close_ts - buffer_s
    out = []
    for candle in candles:
        end = candle.get("end_period_ts") or candle.get("ts")
        if end is not None and end <= cutoff:
            out.append(candle)
    return out


class HistoricalMarket:
    """A resolved market, presented the way the live `Market` is.

    Duck-typed rather than subclassed so `agent/features.py` needs no
    special case: pretraining must build features by exactly the same code
    path as live trading, or it is training on a different problem.
    """

    __slots__ = ("ticker", "series", "category", "title", "subtitle",
                 "close_ts", "yes_bid", "yes_ask", "yes_bid_size",
                 "yes_ask_size", "last_price", "volume", "open_interest",
                 "status", "updated_at", "result", "_now")

    def __init__(self, row, price, observed_ts):
        self.ticker = row.get("ticker")
        self.series = rest.series_of(self.ticker)
        self.category = row.get("category") or "Historical"
        self.title = row.get("title")
        self.subtitle = row.get("yes_sub_title") or row.get("subtitle")
        self.close_ts = row.get("_close_ts")
        self.result = (row.get("result") or "").strip().lower()
        self.status = "active"
        self._now = observed_ts

        # A synthetic one-cent spread around the observed price. The candle
        # gives a mid, not a book; assuming a tight two-sided market is the
        # conservative direction -- it makes entry slightly more expensive than
        # a mid-price fill would be, never cheaper.
        half = money.ONE_DOLLAR // 200
        self.yes_bid = max(1, price - half)
        self.yes_ask = min(money.ONE_DOLLAR - 1, price + half)
        self.yes_bid_size = self.yes_ask_size = 500.0
        self.last_price = price
        self.volume = _f(row.get("volume")) or 100.0
        self.open_interest = _f(row.get("open_interest")) or 100.0
        self.updated_at = observed_ts

    # The handful of Market properties features.py touches.
    @property
    def mid(self):
        return (self.yes_bid + self.yes_ask) // 2

    @property
    def spread(self):
        return self.yes_ask - self.yes_bid

    @property
    def can_buy_yes(self):
        return 0 < self.yes_ask < money.ONE_DOLLAR

    @property
    def can_buy_no(self):
        return 0 < self.yes_bid < money.ONE_DOLLAR

    @property
    def is_quoted(self):
        return self.can_buy_yes and self.can_buy_no

    @property
    def is_tradeable(self):
        return self.can_buy_yes or self.can_buy_no

    def seconds_to_close(self, now=None):
        """Time remaining AS OF THE OBSERVATION, not as of today.

        Using wall-clock now would make every historical market read as long
        expired, and `log_time_to_close` -- a real feature -- would be constant
        zero across the whole pretraining set.
        """
        return max(0.0, (self.close_ts or 0) - self._now)

    def is_stale(self, now=None):
        return False


def _parse_iso(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(
            value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


OBSERVATIONS_PER_MARKET = 3

# Candles are cached under data/pretrain/ so a second run costs nothing. One
# request per market is the expensive part of Phase A -- a settled series can
# hold thousands of markets, and fetching candles for all of them before
# honouring the limit is how the first version of this ran for five minutes
# without producing a single training example.
CACHE_DIRNAME = "pretrain"


def _cache_path(data_dir, series_ticker, ticker):
    directory = os.path.join(data_dir, CACHE_DIRNAME, series_ticker)
    os.makedirs(directory, exist_ok=True)
    safe = ticker.replace("/", "_")
    return os.path.join(directory, f"{safe}.json")


def cached_candles(series_ticker, ticker, open_ts, close_ts,
                   data_dir=None, session=None):
    """Candles for one market, read from disk when we already have them."""
    path = _cache_path(data_dir, series_ticker, ticker) if data_dir else None
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except (ValueError, OSError):
            pass
    candles = rest.candlesticks(series_ticker, ticker, open_ts, close_ts,
                                period_interval=candle_minutes(close_ts - open_ts),
                                session=session)
    if path:
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(candles, handle)
        except OSError:
            pass
    return candles


def load_series(series_ticker, limit=400, session=None, page_cap=1,
                data_dir=None, rng=None):
    """Resolved markets for one series, with a leakage-safe observed price.

    Yields `(HistoricalMarket, winning_side)`. Markets whose candles all fall
    inside the lookahead buffer are skipped rather than guessed at.

    The market list is truncated BEFORE any candles are fetched. Each market
    costs an HTTP request, so filtering afterwards means paying for thousands of
    requests to use a hundred of them.
    """
    markets = rest.settled_markets(series_ticker, session=session,
                                   page_cap=page_cap)
    resolved = [m for m in markets
                if (m.get("result") or "").strip().lower() in ("yes", "no")]
    log.info("%s: %d settled, %d resolved", series_ticker, len(markets),
             len(resolved))

    # Sample rather than take the head: consecutive markets in a series are
    # near-duplicates of each other (the same event an hour later), and a
    # contiguous block would train the agent on one slice of history.
    needed = max(1, limit // OBSERVATIONS_PER_MARKET)
    if len(resolved) > needed:
        resolved = (rng or random).sample(resolved, needed)

    produced = 0
    for row in resolved:
        if produced >= limit:
            return
        result = (row.get("result") or "").strip().lower()
        if result not in ("yes", "no"):
            continue                       # unresolved or voided: nothing to learn

        close_ts = _parse_iso(row.get("close_time"))
        open_ts = _parse_iso(row.get("open_time")) or (close_ts - 7 * 86400
                                                       if close_ts else None)
        if not close_ts or not open_ts:
            continue

        try:
            candles = cached_candles(series_ticker, row["ticker"],
                                     open_ts, close_ts,
                                     data_dir=data_dir, session=session)
        except Exception:
            continue

        usable = safe_candles(candles, close_ts,
                              buffer_s=lookahead_buffer(close_ts - open_ts))
        if not usable:
            continue

        for candle in usable[-OBSERVATIONS_PER_MARKET:]:
            price = candle_price(candle)
            if price is None or not (0 < price < money.ONE_DOLLAR):
                continue
            observed = candle.get("end_period_ts") or candle.get("ts") or open_ts
            row = dict(row, _close_ts=close_ts)
            yield HistoricalMarket(row, price, observed), result
            produced += 1
            if produced >= limit:
                return


def pretrain(agent, series, markets_per_series=200, session=None, rng=None,
             data_dir=None):
    """Run one agent through resolved markets. Returns a summary dict.

    Trades are simulated but never touch the agent's real bankroll: pretraining
    is about learning the shape of the problem, not about carrying a P&L into
    live trading. The agent starts its first live day with $100 and a head full
    of opinions, which is exactly what PRD 3.1 describes.
    """
    rng = rng or random.Random(0)
    started = time.time()
    seen = wins = losses = 0

    for series_ticker in series:
        for market, winner in load_series(series_ticker,
                                          limit=markets_per_series,
                                          session=session,
                                          data_dir=data_dir, rng=rng):
            side = YES if rng.random() < 0.5 else NO
            vector = build_features(market, side, memory=agent.memory,
                                    has_depth=False,
                                    now=market.updated_at)
            if vector is None:
                continue

            price = entry_price(market, side)
            levels = fills.levels_from_quote(market, side)
            fill = fills.buy(levels, 10, side)
            if not fill.filled:
                continue

            won = (winner == side)
            payout = money.payout(fill.contracts) if won else 0
            pnl = payout - fill.cost

            outcome = {
                "ticker": market.ticker, "series": market.series,
                "category": market.category, "side": side,
                "entry_price": fill.avg_price, "contracts": fill.contracts,
                "stake": fill.cost, "pnl": pnl,
                "return_on_risk": pnl / fill.cost if fill.cost else 0.0,
                "outcome": "win" if won else "loss", "won": won,
                "held_seconds": None,
            }
            agent.memory.record(market, outcome, day="pretrain",
                                side=side, price=fill.avg_price,
                                now=market.updated_at)

            # Same target function as live trading, including this agent's own
            # risk shaping -- so Kyle and Cartman draw different lessons from
            # the same history, exactly as they will once live.
            class _P:
                realized_pnl = pnl
                cost = fill.cost
                stake_fraction = fill.cost / agent.p.starting_bankroll

            target = trade_target(_P(), agent.p.reward)
            if target is not None:
                agent.policy.update(vector, target)

            seen += 1
            wins += int(won)
            losses += int(not won)

    agent.save()
    summary = {
        "agent": agent.name,
        "observations": seen,
        "wins": wins,
        "losses": losses,
        "updates": agent.policy.updates,
        "mean_abs_error": agent.policy.mean_abs_error,
        "patterns": agent.memory.stats()["patterns"],
        "seconds": round(time.time() - started, 1),
    }
    log.info("pretrained %s on %d observations in %.1fs",
             agent.name, seen, summary["seconds"])
    return summary


def main():
    import os
    import argparse
    from dotenv import load_dotenv
    from .personality import load_all
    from .loop import Agent

    ap = argparse.ArgumentParser(description="Phase A historical pretraining")
    ap.add_argument("--agent", default="stan")
    ap.add_argument("--series", default="KXBTC15M,KXHIGHNY,KXNBA",
                    help="comma-separated series tickers")
    ap.add_argument("--markets", type=int, default=150)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(message)s")
    load_dotenv()

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    personality = load_all()[args.agent]
    agent = Agent(personality, data_dir)
    agent.load()

    print(f"pretraining {personality.display_name} on "
          f"{args.series} ({args.markets} markets each)")
    summary = pretrain(agent, [s.strip() for s in args.series.split(",")],
                       markets_per_series=args.markets, data_dir=data_dir)

    print(f"\n  observations   {summary['observations']}")
    print(f"  win/loss       {summary['wins']}/{summary['losses']}")
    print(f"  weight updates {summary['updates']}")
    print(f"  mean abs error {summary['mean_abs_error']}")
    print(f"  memory patterns{summary['patterns']:>4}")
    print(f"  elapsed        {summary['seconds']}s")
    print("\ntop learned weights:")
    for name, weight in agent.policy.top_weights(8):
        print(f"    {name:22s} {weight:+.4f}")
    agent.close()


if __name__ == "__main__":
    main()
