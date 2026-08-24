"""End-to-end tests for the agent decision loop and the activity store.

The loop is where every real bug in this project has been: the churn trap, the
side-blind memory, exits recorded as neither win nor loss. So these tests drive
a whole agent against a synthetic universe rather than poking at parts.
"""

import os
import time

import pytest

from sim import money
from sim.fills import YES, NO
from kalshi.universe import Market, Universe
from agent.personality import load_all
from agent.loop import Agent
from store.db import Store


class FakeUniverse(Universe):
    """A handful of synthetic markets with controllable quotes."""

    def __init__(self, n=12, bid=4000, ask=4200, close_in=86400):
        super().__init__()
        for i in range(n):
            m = Market(f"KXTEST-{i}")
            m.category = "Sports" if i % 2 else "Crypto"
            m.series = "KXTEST"
            m.subtitle = f"Subject {i}"
            m.title = f"Test market {i}"
            m.yes_bid, m.yes_ask = bid, ask
            m.yes_bid_size = m.yes_ask_size = 10000.0
            m.volume, m.open_interest = 500.0, 300.0
            m.close_ts = time.time() + close_in
            m.updated_at = time.time()
            self._markets[m.ticker] = m

    def set_quotes(self, bid, ask):
        for m in self._markets.values():
            m.yes_bid, m.yes_ask = bid, ask


class FakeRest:
    """Settles named markets with a fixed result."""

    def __init__(self, results=None):
        self.results = results or {}

    def _get(self, path, params=None, session=None):
        wanted = (params or {}).get("tickers", "").split(",")
        return {"markets": [{"ticker": t, "result": self.results.get(t, "")}
                            for t in wanted]}


@pytest.fixture
def agent(tmp_path):
    a = Agent(load_all()["stan"], str(tmp_path), store=None, seed=3)
    a.is_asleep = lambda when=None: False      # sleep is a wall-clock concept
    yield a
    a.memory.close()


# ---------- candidates ----------

def test_candidates_cover_both_sides_of_each_market():
    """Buying YES and buying NO are different bets and both must be scoreable."""
    a = Agent(load_all()["stan"], os.devnull.replace(os.devnull, "."), store=None)
    try:
        candidates = a.build_candidates(FakeUniverse(n=5))
        assert len(candidates) == 10
        assert {c.side for c in candidates} == {YES, NO}
    finally:
        a.memory.close()


def test_markets_about_to_close_are_skipped(agent):
    """Entering a market seconds before resolution is not a decision."""
    assert agent.build_candidates(FakeUniverse(n=5, close_in=10)) == []


def test_markets_already_held_are_not_re_entered(agent):
    universe = FakeUniverse(n=3)
    agent.tick(universe)
    held = {p.ticker for p in agent.portfolio.open_positions()}
    assert held
    tickers = {c.market.ticker for c in agent.build_candidates(universe)}
    assert not (tickers & held)


def test_an_unquoted_market_yields_no_candidate(agent):
    universe = FakeUniverse(n=3, bid=0, ask=money.ONE_DOLLAR)
    assert agent.build_candidates(universe) == []


# ---------- acting ----------

def test_a_fresh_agent_actually_trades(agent):
    """Optimistic initialization must be enough to get off the ground.

    An agent that never acts never observes an outcome and never learns.
    """
    decision = agent.tick(FakeUniverse())
    assert decision.acted is True
    assert len(agent.portfolio.open_positions()) == 1
    assert agent.trades_today == 1


def test_a_pessimistic_agent_deliberately_does_nothing(agent):
    """PRD 2: choosing not to trade stays a legitimate strategy."""
    agent.policy.weights[:] = 0
    agent.policy.weights[0] = -1.0          # bias: everything looks bad
    agent.policy.epsilon = 0.0
    decision = agent.tick(FakeUniverse())
    assert decision.acted is False
    assert decision.skipped_reason == "nothing_worth_doing"
    assert agent.portfolio.open_positions() == []


def test_an_agent_at_its_position_limit_stops_opening(agent):
    agent.p.trading.max_open_positions = 2
    universe = FakeUniverse(n=20)
    for _ in range(8):
        agent.tick(universe)
    assert len(agent.portfolio.open_positions()) <= 2


def test_a_position_records_the_features_it_was_opened_on(agent):
    """Without this the resolution has no (state, action) to train."""
    agent.tick(FakeUniverse())
    position = agent.portfolio.open_positions()[0]
    assert position.entry_features is not None
    assert position.stake_fraction > 0


def test_entering_asks_for_order_book_depth(agent):
    """The depth tier follows real agent interest rather than a fixed list."""
    agent.tick(FakeUniverse())
    assert agent.depth_wanted


# ---------- the churn trap ----------

def test_a_position_is_not_exited_before_its_minimum_hold(agent):
    """The bug that left an agent with $3.45 after 300 simulated days.

    A pessimistic model exits everything the instant it opens it, paying the
    spread plus two fees and never discovering how anything resolves.
    """
    universe = FakeUniverse()
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]

    agent.policy.weights[:] = 0
    agent.policy.weights[0] = -5.0          # everything now looks terrible

    assert agent.check_exits(universe) == []
    assert position.is_open


def test_a_position_is_exited_once_it_is_old_enough_and_looks_bad(agent):
    universe = FakeUniverse()
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]
    position.opened_at -= agent.p.trading.min_hold_seconds + 1

    agent.policy.weights[:] = 0
    agent.policy.weights[0] = -5.0

    exited = agent.check_exits(universe)
    assert exited == [position]
    assert not position.is_open
    assert position.result == "exit"


def test_a_position_that_still_looks_good_is_held(agent):
    universe = FakeUniverse()
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]
    position.opened_at -= agent.p.trading.min_hold_seconds + 1

    agent.policy.weights[:] = 0
    agent.policy.weights[0] = 5.0           # everything looks great

    assert agent.check_exits(universe) == []
    assert position.is_open


def test_the_exit_margin_creates_hysteresis(agent):
    """Anything barely worth entering must not be instantly worth exiting."""
    universe = FakeUniverse()
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]
    position.opened_at -= agent.p.trading.min_hold_seconds + 1

    # Sit just below the entry threshold but above the exit threshold.
    agent.policy.weights[:] = 0
    agent.policy.weights[0] = agent.p.policy.act_threshold - \
        agent.p.trading.exit_margin / 2

    assert agent.check_exits(universe) == []
    assert position.is_open


def test_kenny_bails_far_sooner_than_kyle():
    """PRD 4's hold-time trait, expressed purely in config."""
    agents = load_all()
    assert agents["kenny"].trading.min_hold_seconds < \
        agents["stan"].trading.min_hold_seconds < \
        agents["kyle"].trading.min_hold_seconds


# ---------- settlement and learning ----------

def test_settling_a_win_trains_the_model_and_the_memory(agent):
    universe = FakeUniverse()
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]
    before_updates = agent.policy.updates

    result = "yes" if position.side == YES else "no"
    agent.settle(universe, FakeRest({position.ticker: result}))

    assert position.result == "win"
    assert agent.policy.updates == before_updates + 1
    assert agent.memory.get(f"series:KXTEST|side:{position.side}") is not None


def test_a_void_teaches_nothing(agent):
    """No prediction was tested, so training on a zero return would be a lie."""
    universe = FakeUniverse()
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]
    before = agent.policy.updates

    agent.settle(universe, FakeRest({position.ticker: "void"}))

    assert position.result == "void"
    assert agent.policy.updates == before
    assert agent.memory.get("series:KXTEST") is None


def test_an_exit_is_remembered_as_a_win_or_a_loss(agent):
    """Exits once counted as neither, leaving beliefs with a None win rate --
    which made them invisible to the policy entirely."""
    universe = FakeUniverse()
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]
    position.opened_at -= agent.p.trading.min_hold_seconds + 1

    agent.policy.weights[:] = 0
    agent.policy.weights[0] = -5.0
    agent.check_exits(universe)

    belief = agent.memory.get(f"series:KXTEST|side:{position.side}")
    assert belief is not None
    assert belief.resolved == 1              # counted as a win or a loss
    assert belief.win_rate is not None


# ---------- the daily episode ----------

def test_closing_the_day_produces_a_reward_and_a_breakdown(agent):
    agent.tick(FakeUniverse())
    reward, parts = agent.close_day("2026-08-24")
    assert isinstance(reward, float)
    assert parts
    assert sum(parts.values()) == pytest.approx(reward)


def test_a_day_with_no_trades_scores_slightly_negative(agent):
    reward, parts = agent.close_day("2026-08-24")
    assert "inaction" in parts
    assert reward < 0


def test_the_day_rolls_over_and_resets_the_trade_count(agent):
    agent.tick(FakeUniverse())
    agent.day = "2020-01-01"                 # pretend it is yesterday
    agent.trades_today = 5
    agent.roll_day_if_needed()
    assert agent.day != "2020-01-01"
    assert agent.trades_today == 0


# ---------- persistence (PRD 11) ----------

def test_an_agent_resumes_everything_after_a_restart(tmp_path):
    """A crash or reboot must not wipe learning progress."""
    universe = FakeUniverse()
    first = Agent(load_all()["stan"], str(tmp_path), seed=3)
    first.is_asleep = lambda when=None: False
    for _ in range(3):
        first.tick(universe)
    for _ in range(20):
        first.policy.update(first.portfolio.open_positions()[0].entry_features, 0.4)
    open_count = len(first.portfolio.open_positions())
    bankroll = first.portfolio.bankroll
    prediction = first.policy.q(first.portfolio.open_positions()[0].entry_features)
    first.close()

    second = Agent(load_all()["stan"], str(tmp_path), seed=3)
    assert second.load() is True
    assert second.portfolio.bankroll == bankroll
    assert len(second.portfolio.open_positions()) == open_count
    restored = second.portfolio.open_positions()[0]
    assert second.policy.q(restored.entry_features) == pytest.approx(prediction)
    assert restored.stake_fraction > 0       # needed by the risk-shaped target
    second.close()


def test_a_restored_position_can_still_settle_and_train(tmp_path):
    """A resumed position must remain a usable training example."""
    universe = FakeUniverse()
    first = Agent(load_all()["stan"], str(tmp_path), seed=3)
    first.is_asleep = lambda when=None: False
    first.tick(universe)
    ticker = first.portfolio.open_positions()[0].ticker
    side = first.portfolio.open_positions()[0].side
    first.close()

    second = Agent(load_all()["stan"], str(tmp_path), seed=3)
    second.load()
    before = second.policy.updates
    second.settle(universe, FakeRest({ticker: "yes" if side == YES else "no"}))
    assert second.policy.updates == before + 1
    second.close()


def test_a_corrupt_state_file_starts_fresh_rather_than_crashing(tmp_path):
    agent = Agent(load_all()["stan"], str(tmp_path))
    with open(agent.state_path, "w", encoding="utf-8") as handle:
        handle.write("{ this is not json")
    assert agent.load() is False
    assert agent.portfolio.bankroll == money.STARTING_BANKROLL
    agent.close()


def test_sleeping_agents_do_not_act(tmp_path):
    agent = Agent(load_all()["stan"], str(tmp_path))
    agent.is_asleep = lambda when=None: True
    assert agent.tick(FakeUniverse()) is None
    assert agent.portfolio.open_positions() == []
    agent.close()


# ---------- the activity store ----------

@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "activity.db"))
    yield s
    s.close()


def test_decisions_including_non_actions_are_logged(store, tmp_path):
    """PRD 9 wants deliberate inaction in the feed, not hidden."""
    agent = Agent(load_all()["stan"], str(tmp_path), store=store, seed=3)
    agent.is_asleep = lambda when=None: False
    agent.tick(FakeUniverse())

    agent.policy.weights[:] = 0
    agent.policy.weights[0] = -5.0
    agent.policy.epsilon = 0.0
    agent.tick(FakeUniverse())

    rows = store.recent_decisions("stan")
    assert len(rows) == 2
    assert any(r["acted"] == 0 for r in rows)
    assert any(r["acted"] == 1 for r in rows)
    agent.close()


def test_positions_are_written_on_open_and_updated_on_close(store, tmp_path):
    universe = FakeUniverse()
    agent = Agent(load_all()["stan"], str(tmp_path), store=store, seed=3)
    agent.is_asleep = lambda when=None: False
    agent.tick(universe)
    position = agent.portfolio.open_positions()[0]

    assert len(store.open_positions("stan")) == 1

    agent.settle(universe, FakeRest({position.ticker: "yes"}))
    assert store.open_positions("stan") == []
    row = store.recent_positions("stan")[0]
    assert row["status"] == "settled"
    assert row["realized_pnl"] is not None
    agent.close()


def test_the_daily_row_is_upserted_not_duplicated(store):
    store.log_daily("2026-08-24", "stan", reward=0.1, trades=3)
    store.log_daily("2026-08-24", "stan", reward=0.2, trades=5)
    rows = store.daily_series("stan")
    assert len(rows) == 1
    assert rows[0]["reward"] == 0.2


def test_category_breakdown_answers_which_markets_it_prefers(store, tmp_path):
    universe = FakeUniverse(n=10)
    agent = Agent(load_all()["stan"], str(tmp_path), store=store, seed=3)
    agent.is_asleep = lambda when=None: False
    for _ in range(6):
        agent.tick(universe)
    rows = store.category_breakdown("stan")
    assert rows
    assert sum(r["n"] for r in rows) == agent.portfolio.trades_opened
    agent.close()


def test_pruning_drops_old_decisions_but_keeps_positions(store):
    store._db.execute(
        "INSERT INTO decisions (ts, day, agent, acted, explored, considered) "
        "VALUES (?,?,?,?,?,?)", (0.0, "2000-01-01", "stan", 0, 0, 0))
    store._db.commit()
    assert store.prune() == 1
    assert store.recent_decisions("stan") == []


def test_the_store_reports_its_own_disk_usage(store):
    """PRD 15: be mindful of disk, and be able to see it."""
    stats = store.stats()
    assert stats["bytes_on_disk"] > 0
    assert set(stats) >= {"decisions", "positions", "daily", "events"}
