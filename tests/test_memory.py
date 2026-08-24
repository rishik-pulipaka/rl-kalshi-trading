"""Tests for the explicit memory bank.

PRD 3.3 makes this the centerpiece: the primary window into how each agent is
learning. So the properties tested here are the ones that decide whether that
window shows anything true -- that a pattern accumulates correctly, that
confidence distinguishes three encounters from two hundred, and that the memory
survives a restart.
"""

import os
import tempfile

import pytest

from agent import memory as mem
from agent.memory import MemoryBank, Belief, pattern_keys, price_bucket, ttr_bucket
from kalshi.universe import Market


@pytest.fixture
def bank(tmp_path):
    b = MemoryBank(str(tmp_path / "memory.db"))
    yield b
    b.close()


def _market(ticker="KXNBA-26-GSW", category="Sports", series="KXNBA",
            subtitle="Stephen Curry", bid=4000, ask=4400, close_in=1800):
    import time
    m = Market(ticker)
    m.category, m.series, m.subtitle = category, series, subtitle
    m.yes_bid, m.yes_ask = bid, ask
    m.close_ts = time.time() + close_in if close_in else None
    return m


def _outcome(pnl, result, stake=4200, price=4200, ticker="KXNBA-26-GSW"):
    return {"ticker": ticker, "side": "yes", "entry_price": price, "stake": stake,
            "pnl": pnl, "return_on_risk": pnl / stake, "outcome": result}


# ---------- bucketing ----------

def test_price_buckets_are_finer_at_the_extremes():
    """Long shots and near-certainties are structurally different bets.

    Lumping everything under 20c into one band would hide exactly the region
    Cartman is supposed to be drawn to.
    """
    assert price_bucket(300) == "0-5c"
    assert price_bucket(700) == "5-10c"
    assert price_bucket(5000) == "40-60c"
    assert price_bucket(9700) == "95-100c"


def test_price_bucket_covers_the_whole_range():
    for price in (0, 1, 4999, 9999, 10000):
        assert price_bucket(price) is not None
    assert price_bucket(None) is None


def test_ttr_buckets_span_minutes_to_months():
    assert ttr_bucket(60) == "under_1h"
    assert ttr_bucket(7200) == "1-6h"
    assert ttr_bucket(3 * 86400) == "1-7d"
    assert ttr_bucket(90 * 86400) == "over_1m"
    assert ttr_bucket(-5) is None


# ---------- pattern extraction ----------

def test_one_market_produces_several_nested_patterns():
    """Broad keys fill up fast and help early; specific keys are the payoff."""
    keys = pattern_keys(_market(), side="yes", price=4200)
    kinds = {kind for _, kind, _ in keys}
    assert {"category", "series", "series_price", "entity", "entity_side",
            "ttr"} <= kinds


def test_entity_comes_from_the_subtitle():
    """The subtitle names the specific outcome within an event.

    Parsing titles instead would need per-category rules for thousands of series
    and would fail silently on the ones it did not know about.
    """
    keys = dict((k, label) for k, _, label in pattern_keys(_market(), side="yes"))
    assert keys["entity:Stephen Curry"] == "Stephen Curry"


def test_a_market_with_no_subtitle_still_produces_patterns():
    keys = pattern_keys(_market(subtitle=None), side="yes", price=4200)
    assert keys
    assert not any(kind.startswith("entity") for _, kind, _ in keys)


def test_pattern_keys_are_stable_across_calls():
    """Tomorrow's lookup must find yesterday's memory."""
    a = [k for k, _, _ in pattern_keys(_market(), side="yes", price=4200)]
    b = [k for k, _, _ in pattern_keys(_market(), side="yes", price=4200)]
    assert a == b


# ---------- accumulating ----------

def test_recording_creates_a_belief_per_pattern(bank):
    market = _market()
    bank.record(market, _outcome(500, "win"), day="d1")
    beliefs = bank.recall(market, side="yes", price=4200)
    assert len(beliefs) == len(pattern_keys(market, side="yes", price=4200))
    assert all(b.encounters == 1 for _, b in beliefs)


def test_wins_and_losses_accumulate(bank):
    market = _market()
    for pnl, result in [(500, "win"), (500, "win"), (-4200, "loss")]:
        bank.record(market, _outcome(pnl, result), day="d1")
    belief = bank.get("series:KXNBA")
    assert belief.encounters == 3
    assert belief.wins == 2 and belief.losses == 1
    assert belief.win_rate == pytest.approx(2 / 3)
    assert belief.net_pnl == 500 + 500 - 4200


def test_a_high_win_rate_can_still_be_unprofitable(bank):
    """The lesson the PRD most wants an agent to be able to learn.

    Winning often while losing money -- small wins, large losses -- is the
    classic trap, and the memory bank has to be able to show it rather than
    reporting a cheerful 75% win rate and nothing else.
    """
    market = _market()
    for _ in range(3):
        bank.record(market, _outcome(300, "win"), day="d1")
    bank.record(market, _outcome(-4200, "loss"), day="d1")

    belief = bank.get("series:KXNBA")
    assert belief.win_rate == 0.75
    assert belief.net_pnl < 0
    assert belief.roi < 0


def test_voids_do_not_count_as_resolved(bank):
    """A cancelled market taught the agent nothing; it must not move win rate."""
    market = _market()
    bank.record(market, _outcome(500, "win"), day="d1")
    bank.record(market, _outcome(0, "void"), day="d1")

    belief = bank.get("series:KXNBA")
    assert belief.encounters == 2
    assert belief.voids == 1
    assert belief.resolved == 1
    assert belief.win_rate == 1.0        # not 0.5


def test_avg_entry_price_tracks_where_the_agent_buys(bank):
    market = _market()
    bank.record(market, _outcome(0, "loss", price=2000), day="d1")
    bank.record(market, _outcome(0, "loss", price=6000), day="d1")
    assert bank.get("series:KXNBA").avg_entry_price == pytest.approx(4000)


def test_best_and_worst_outcomes_are_remembered(bank):
    market = _market()
    for pnl in (100, 5000, -3000):
        bank.record(market, _outcome(pnl, "win" if pnl > 0 else "loss"), day="d1")
    belief = bank.get("series:KXNBA")
    assert belief.best_pnl == 5000
    assert belief.worst_pnl == -3000


# ---------- confidence ----------

def test_confidence_distinguishes_thin_evidence_from_thick():
    """"60% over 3 tries" and "60% over 200 tries" must not look the same.

    Kyle's preference for well-established patterns is a preference over exactly
    this number.
    """
    thin = Belief(wins=3, losses=2, encounters=5)
    thick = Belief(wins=120, losses=80, encounters=200)
    assert thin.win_rate == pytest.approx(thick.win_rate, abs=0.01)
    assert thin.confidence() < 0.4
    assert thick.confidence() > 0.9


def test_confidence_is_zero_with_no_resolved_encounters():
    assert Belief(wins=0, losses=0, encounters=0).confidence() == 0.0


def test_win_rate_is_none_rather_than_zero_without_data():
    """No data is not a 0% win rate -- treating it as one would rank an
    untried pattern below a genuinely bad one."""
    assert Belief(wins=0, losses=0, encounters=0).win_rate is None


# ---------- recency weighting ----------

def test_recency_weighting_favours_recent_outcomes(bank):
    """Cartman's "over-repeats whatever recently worked" is this one number."""
    market = _market()
    for _ in range(5):
        bank.record(market, _outcome(-4200, "loss"), day="d1")
    old = bank.get("series:KXNBA").ewma_return
    for _ in range(3):
        bank.record(market, _outcome(8400, "win"), day="d2")
    new = bank.get("series:KXNBA").ewma_return
    assert new > old


def test_a_high_alpha_reacts_faster_than_a_low_one(tmp_path):
    """The dial itself: Cartman (0.5) vs Kyle (0.1) on identical evidence."""
    market = _market()
    results = []
    for alpha in (0.1, 0.5):
        bank = MemoryBank(str(tmp_path / f"m{alpha}.db"), ewma_alpha=alpha)
        for _ in range(5):
            bank.record(market, _outcome(-4200, "loss"), day="d1")
        bank.record(market, _outcome(8400, "win"), day="d2")
        results.append(bank.get("series:KXNBA").ewma_return)
        bank.close()
    slow, fast = results
    assert fast > slow


def test_the_ewma_is_seeded_by_the_first_observation(bank):
    """Starting from zero would make a first big win look mediocre.

    That is a bias with nothing behind it, and it would take several more
    encounters to work off.
    """
    bank.record(_market(), _outcome(8400, "win"), day="d1")
    assert bank.get("series:KXNBA").ewma_return == pytest.approx(2.0)


# ---------- the episodic log ----------

def test_every_observation_is_logged_for_the_dashboard(bank):
    """PRD 9 wants to show how a belief changed over time, not just its value."""
    market = _market()
    for _ in range(4):
        bank.record(market, _outcome(500, "win"), day="d1")
    history = bank.history("series:KXNBA")
    assert len(history) == 4
    assert all(row["pattern_key"] == "series:KXNBA" for row in history)
    assert history[0]["outcome"] == "win"


# ---------- persistence ----------

def test_memory_survives_a_restart(tmp_path):
    """PRD 11: a crash or reboot must not wipe learning progress."""
    path = str(tmp_path / "memory.db")
    market = _market()

    first = MemoryBank(path)
    for _ in range(7):
        first.record(market, _outcome(500, "win"), day="d1")
    first.close()

    second = MemoryBank(path)
    belief = second.get("series:KXNBA")
    assert belief.encounters == 7
    assert belief.wins == 7
    second.close()


def test_agents_do_not_share_memory(tmp_path):
    """PRD 2: four genuinely separate learning trajectories."""
    market = _market()
    stan = MemoryBank(str(tmp_path / "stan.db"))
    kyle = MemoryBank(str(tmp_path / "kyle.db"))

    stan.record(market, _outcome(500, "win"), day="d1")

    assert stan.get("series:KXNBA").encounters == 1
    assert kyle.get("series:KXNBA") is None
    stan.close()
    kyle.close()


# ---------- reads ----------

def test_recall_returns_only_patterns_seen_before(bank):
    """An agent with no memory of a market genuinely knows nothing about it."""
    assert bank.recall(_market(), side="yes", price=4200) == []


def test_top_ranks_patterns_for_the_dashboard(bank):
    for series, pnl in (("KXA", 5000), ("KXB", -2000), ("KXC", 9000)):
        bank.record(_market(series=series, subtitle=None),
                    _outcome(pnl, "win" if pnl > 0 else "loss"), day="d1")
    best = bank.top(limit=3, kind="series")
    assert [b.label for b in best] == ["KXC", "KXA", "KXB"]


def test_stats_summarizes_the_whole_bank(bank):
    bank.record(_market(), _outcome(500, "win"), day="d1")
    stats = bank.stats()
    assert stats["patterns"] > 0
    assert stats["encounters"] == stats["patterns"]   # one encounter each
    assert "series" in stats["by_kind"]
