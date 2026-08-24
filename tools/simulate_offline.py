"""Run an agent against a synthetic market world, fast.

Live learning is inherently slow -- PRD 13 is blunt that "up and running" and
"visibly competent" are months apart. That makes it impossible to tell, from
live data alone, whether the learning machinery actually works or is quietly
broken.

So this harness builds a world with a KNOWN truth (some categories are
systematically overpriced), runs an agent through thousands of compressed days,
and checks whether it finds it. If an agent cannot learn a rule this blatant, it
will certainly not learn anything from real markets, and the bug is ours.

    python -m tools.simulate_offline --agent stan --days 400
"""

import os
import sys
import time
import random
import argparse
import tempfile
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import money
from kalshi.universe import Market, Universe
from agent.personality import load_all
from agent.loop import Agent
from store.db import Store

# The hidden truth: a per-category multiplier on the quoted price. Above 1.0 the
# category is systematically CHEAP (buying YES is profitable); below 1.0 it is
# expensive (buying NO is profitable); 1.0 is fair, so the spread and fees make
# it a slow loser. Nothing tells the agent any of this.
#
# The edges are deliberately SMALL -- a few percent, which is about what a real
# mispricing looks like. An earlier version used 0.70/1.15, and the agent
# correctly farmed it from $100 to $1.2M in 400 simulated days. That proved the
# machinery works but proved nothing about whether it works on a realistic
# signal, which is the only question worth asking here.
EDGE = {"Crypto": 1.00, "Sports": 1.00, "Politics": 0.96, "Weather": 1.04}


class FakeUniverse(Universe):
    """A universe of synthetic markets with a known generating process."""

    def __init__(self, rng, n=300):
        super().__init__()
        self.rng = rng
        self.truth = {}
        for i in range(n):
            category = rng.choice(list(EDGE))
            series = f"KX{category.upper()[:4]}"
            fair = rng.uniform(0.08, 0.92)
            ticker = f"{series}-{i}"
            m = Market(ticker)
            m.category, m.series = category, series
            m.subtitle = f"{category} subject {i % 25}"
            m.title = f"Synthetic {category} market {i}"
            # Quoted price sits above fair value by the category's edge factor.
            quoted = min(0.97, max(0.03, fair / EDGE[category]))
            m.yes_bid = int(quoted * money.ONE_DOLLAR) - 100
            m.yes_ask = int(quoted * money.ONE_DOLLAR) + 100
            m.yes_bid_size = m.yes_ask_size = 5000.0
            m.volume, m.open_interest = 500.0, 300.0
            m.close_ts = time.time() + 86400
            m.updated_at = time.time()
            self._markets[ticker] = m
            self.truth[ticker] = fair

    def reprice(self):
        """New day, new prices. Fair value is redrawn; the edge persists."""
        for ticker, m in self._markets.items():
            fair = self.rng.uniform(0.08, 0.92)
            self.truth[ticker] = fair
            quoted = min(0.97, max(0.03, fair / EDGE[m.category]))
            m.yes_bid = int(quoted * money.ONE_DOLLAR) - 100
            m.yes_ask = int(quoted * money.ONE_DOLLAR) + 100
            m.close_ts = time.time() + 86400

    def resolve(self, ticker):
        """Did YES happen? Drawn against the market's true probability."""
        return self.rng.random() < self.truth[ticker]


class FakeRest:
    """Resolves every held market immediately, as the day's settlement."""

    def __init__(self, universe):
        self.u = universe

    def _get(self, path, params=None, session=None):
        tickers = (params or {}).get("tickers", "").split(",")
        return {"markets": [
            {"ticker": t, "result": "yes" if self.u.resolve(t) else "no"}
            for t in tickers if t in self.u._markets]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="stan")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--ticks", type=int, default=12, help="decisions per day")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    universe = FakeUniverse(rng)
    rest = FakeRest(universe)

    workdir = tempfile.mkdtemp(prefix="rlk_sim_")
    store = Store(os.path.join(workdir, "activity.db"))
    personality = load_all()[args.agent]
    agent = Agent(personality, workdir, store=store, seed=args.seed)
    # Sleep is a real-clock concept; a compressed simulation ignores it.
    agent.is_asleep = lambda when=None: False

    print(f"{personality.display_name}: {args.days} days x {args.ticks} decisions, "
          f"start {money.fmt(agent.portfolio.bankroll)}")
    print(f"  hidden truth -> {EDGE}")
    print()
    print(f"  {'day':>5} {'equity':>10} {'trades':>7} {'skips':>6} "
          f"{'winrate':>8} {'error':>7}")

    equity_curve = []
    for day in range(args.days):
        agent.day = f"sim-{day:04d}"
        agent.trades_today = 0
        universe.reprice()
        skips = 0
        for _ in range(args.ticks):
            decision = agent.tick(universe, books=None)
            if decision is not None and not decision.acted:
                skips += 1
        agent.check_exits(universe, books=None)
        agent.settle(universe, rest)
        agent.close_day(agent.day)
        equity_curve.append(agent.portfolio.equity())

        if day % max(1, args.days // 12) == 0 or day == args.days - 1:
            stats = agent.portfolio.stats()
            wr = stats["win_rate"]
            err = agent.policy.mean_abs_error
            print(f"  {day:5d} {money.fmt(stats['equity']):>10} "
                  f"{agent.trades_today:7d} {skips:6d} "
                  f"{(f'{wr:.1%}' if wr else '--'):>8} "
                  f"{(f'{err:.3f}' if err else '--'):>7}")

    print()
    stats = agent.portfolio.stats()
    print(f"FINAL  equity {money.fmt(stats['equity'])}  "
          f"trades {stats['trades_opened']}  "
          f"win rate {stats['win_rate']:.1%}  "
          f"bankruptcies {stats['bankruptcies']}")
    print(f"       weight updates {agent.policy.updates}  "
          f"mean abs error {agent.policy.mean_abs_error:.3f}  "
          f"exploration {agent.policy.exploration_rate:.1%}")

    print("\nDid it find the hidden edge? Positions taken per category:")
    rows = store.category_breakdown(agent.name)
    total = sum(r["n"] for r in rows) or 1
    for row in sorted(rows, key=lambda r: -r["n"]):
        share = row["n"] / total
        wr = (row["wins"] / row["resolved"]) if row["resolved"] else 0
        print(f"  {row['category'] or '?':10s} {row['n']:5d} ({share:5.1%})  "
              f"win {wr:5.1%}  pnl {money.fmt(row['pnl']):>10}  "
              f"[true edge {EDGE.get(row['category'], 1.0):.2f}]")

    print("\nTop learned weights:")
    for name, weight in agent.policy.top_weights(8):
        print(f"  {name:22s} {weight:+.4f}")

    print("\nMemory bank, most-encountered patterns:")
    for belief in agent.memory.top(limit=8, order="encounters"):
        wr = f"{belief.win_rate:.0%}" if belief.win_rate is not None else "--"
        print(f"  {belief.label[:34]:36s} n={belief.encounters:4d} win={wr:>4s} "
              f"roi={belief.roi:+.2f}" if belief.roi is not None else "")

    print(f"\nstore: {store.stats()}")
    print(f"workdir: {workdir}")
    agent.close()
    store.close()


if __name__ == "__main__":
    main()
