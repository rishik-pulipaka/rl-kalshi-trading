"""Tests for multi-leg combos -- both kinds (PRD 2).

PRD 2 asks that agents be able to "construct or take multi-leg combos/parlays".
Those are two different capabilities and this file covers both:

  TAKE      -- Kalshi's ~1.19M auto-generated cross-category parlays are real
               all-or-nothing instruments. Agents reach them by materializing a
               rotating sample into the universe and evaluating them like any
               other market.
  CONSTRUCT -- an agent commits to several legs as one decision. That is a
               basket, not a parlay: legs settle independently.
"""

import time

import pytest

from sim import money
from sim.fills import YES, NO
from kalshi import universe as universe_module
from kalshi.universe import Market, Universe
from agent.personality import load_all
from agent.loop import Agent
from tests.test_loop import FakeUniverse, FakeRest


# ---------- TAKE: reaching Kalshi's listed parlays ----------

def _ticker_frame(ticker, bid="0.0400", ask="0.0600"):
    return {"type": "ticker", "msg": {
        "market_ticker": ticker, "yes_bid_dollars": bid,
        "yes_ask_dollars": ask, "yes_bid_size_fp": "100.00",
        "yes_ask_size_fp": "100.00", "price_dollars": "0.0500",
        "volume_fp": "10.00", "open_interest_fp": "5.00"}}


def test_quoting_combos_are_remembered_without_being_materialized():
    """The reservoir is how the parlay space stays reachable.

    Materializing all ~1.19M up front would defeat the point of sweeping
    /events; forgetting them entirely would make PRD 2's "take a combo"
    impossible.
    """
    u = Universe()
    u.on_message(_ticker_frame("KXMVECROSSCATEGORY-SHARD1-ABC"))
    assert len(u) == 0                       # not materialized
    assert u.stats()["unknown_reservoir"] == 1


def test_an_unquoted_combo_is_not_worth_remembering():
    """A parlay nobody has priced is not a market an agent could take."""
    u = Universe()
    u.on_message(_ticker_frame("KXMVECROSSCATEGORY-SHARD1-DEAD", ask="0.0000"))
    assert u.stats()["unknown_reservoir"] == 0


def test_the_reservoir_is_bounded():
    """There are 1.19M of these; holding every ticker string defeats the point."""
    u = Universe()
    for i in range(universe_module.UNKNOWN_RESERVOIR + 500):
        u.on_message(_ticker_frame(f"KXMVECROSSCATEGORY-SHARD1-{i}"))
    assert u.stats()["unknown_reservoir"] == universe_module.UNKNOWN_RESERVOIR


def test_adopting_makes_a_combo_an_ordinary_tradeable_market():
    """Once adopted, an agent evaluates a Kalshi parlay like anything else."""
    u = Universe()
    ticker = "KXMVECROSSCATEGORY-SHARD1-XYZ"
    u.on_message(_ticker_frame(ticker))

    class FakeRestModule:
        series_of = staticmethod(lambda t: (t or "").split("-")[0])

        @staticmethod
        def _get(path, params=None, session=None):
            return {"markets": [{
                "ticker": ticker, "status": "active",
                "yes_bid_dollars": "0.0400", "yes_ask_dollars": "0.0600",
                "close_time": "2030-01-01T00:00:00Z",
                "title": "Three things all happen"}]}

    universe_module.rest, original = FakeRestModule, universe_module.rest
    try:
        assert u.adopt_many([ticker]) == 1
    finally:
        universe_module.rest = original

    market = u.get(ticker)
    assert market is not None
    assert market.is_tradeable
    assert market in u.tradeable()
    assert u.stats()["unknown_reservoir"] == 0    # no longer merely "seen"


def test_a_sample_is_drawn_only_from_markets_not_already_held():
    u = Universe()
    for i in range(5):
        u.on_message(_ticker_frame(f"KXMVECROSSCATEGORY-SHARD1-{i}"))
    sample = u.unknown_sample(3)
    assert len(sample) == 3
    assert len(set(sample)) == 3


def test_sampling_an_empty_reservoir_is_harmless():
    assert Universe().unknown_sample(10) == []


# ---------- CONSTRUCT: building a basket ----------

@pytest.fixture
def agent(tmp_path):
    a = Agent(load_all()["cartman"], str(tmp_path), seed=5)
    a.is_asleep = lambda when=None: False
    yield a
    a.memory.close()


def test_an_agent_can_construct_a_multi_leg_basket(agent):
    agent.p.trading.combo_appetite = 1.0          # always bundle
    agent.p.trading.combo_legs = 3
    agent.tick(FakeUniverse(n=10))

    positions = agent.portfolio.open_positions()
    assert len(positions) == 3
    basket_ids = {p.basket_id for p in positions}
    assert len(basket_ids) == 1 and None not in basket_ids


def test_basket_legs_are_distinct_markets(agent):
    """Two legs on one market is just a bigger single bet wearing a hat."""
    agent.p.trading.combo_appetite = 1.0
    agent.p.trading.combo_legs = 4
    agent.tick(FakeUniverse(n=10))
    tickers = [p.ticker for p in agent.portfolio.open_positions()]
    assert len(tickers) == len(set(tickers))


def test_a_basket_splits_one_stake_rather_than_multiplying_it(agent):
    """Bundling is a way to bet on a conjunction, not a way to bet more."""
    agent.p.trading.combo_appetite = 1.0
    agent.p.trading.combo_legs = 3
    before = agent.portfolio.bankroll
    agent.tick(FakeUniverse(n=10))
    spent = before - agent.portfolio.bankroll

    single = Agent(load_all()["cartman"],
                   str(agent.dir).rsplit("agents", 1)[0] + "solo", seed=5)
    single.is_asleep = lambda when=None: False
    single.p.trading.combo_appetite = 0.0
    solo_before = single.portfolio.bankroll
    single.tick(FakeUniverse(n=10))
    solo_spent = solo_before - single.portfolio.bankroll
    single.memory.close()

    # Same order of magnitude -- a basket is not 3x the exposure.
    assert spent < solo_spent * 2


def test_basket_legs_settle_independently(agent):
    """Not a parlay: two winning legs pay out even when a third loses."""
    agent.p.trading.combo_appetite = 1.0
    agent.p.trading.combo_legs = 3
    universe = FakeUniverse(n=10)
    agent.tick(universe)

    legs = agent.portfolio.open_positions()
    results = {}
    for i, leg in enumerate(legs):
        won = i < 2
        results[leg.ticker] = ("yes" if leg.side == YES else "no") if won \
            else ("no" if leg.side == YES else "yes")
    agent.settle(universe, FakeRest(results))

    pnls = [leg.realized_pnl for leg in legs]
    assert sum(1 for p in pnls if p > 0) == 2
    assert sum(1 for p in pnls if p < 0) == 1


def test_each_basket_leg_trains_the_model_separately(agent):
    """Legs resolve independently, so each is its own training example."""
    agent.p.trading.combo_appetite = 1.0
    agent.p.trading.combo_legs = 3
    universe = FakeUniverse(n=10)
    agent.tick(universe)
    legs = agent.portfolio.open_positions()
    before = agent.policy.updates

    agent.settle(universe, FakeRest(
        {leg.ticker: ("yes" if leg.side == YES else "no") for leg in legs}))

    assert agent.policy.updates == before + len(legs)


def test_an_agent_with_no_appetite_never_bundles(agent):
    """Kyle's 0.02 is nearly never; 0.0 is never."""
    agent.p.trading.combo_appetite = 0.0
    for _ in range(5):
        agent.tick(FakeUniverse(n=10))
    assert all(p.basket_id is None for p in agent.portfolio.open_positions())


def test_cartman_is_the_most_drawn_to_combos_and_kyle_the_least():
    """PRD 4: Cartman is pulled toward combos and long shots."""
    agents = load_all()
    appetite = {n: p.trading.combo_appetite for n, p in agents.items()}
    assert appetite["cartman"] == max(appetite.values())
    assert appetite["kyle"] == min(appetite.values())


def test_a_basket_is_logged_as_one_event(tmp_path):
    """The dashboard should show a basket as a single decision, not N."""
    from store.db import Store
    store = Store(str(tmp_path / "a.db"))
    a = Agent(load_all()["cartman"], str(tmp_path), store=store, seed=5)
    a.is_asleep = lambda when=None: False
    a.p.trading.combo_appetite = 1.0
    a.tick(FakeUniverse(n=10))

    events = [e for e in store.recent_events() if e["kind"] == "basket"]
    assert len(events) == 1
    assert events[0]["detail"]["legs"] >= 2
    a.close()
    store.close()
