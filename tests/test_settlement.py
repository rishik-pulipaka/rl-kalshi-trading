"""Tests for resolution detection and settlement.

Offline: a fake REST module stands in for Kalshi, so these run in milliseconds
and exercise the cases that are hard to provoke live -- voids, partial
resolution, API failures mid-poll.
"""

import pytest

from sim import money, fills, settlement
from sim.fills import YES, NO
from sim.portfolio import Portfolio, SETTLED


class FakeRest:
    """Stands in for `kalshi.rest`. Records calls so batching can be asserted."""

    def __init__(self, results=None, fail=False):
        self.results = results or {}
        self.fail = fail
        self.calls = []

    def _get(self, path, params=None, session=None):
        self.calls.append(params)
        if self.fail:
            raise RuntimeError("kalshi is down")
        wanted = (params or {}).get("tickers", "").split(",")
        return {"markets": [{"ticker": t, "result": self.results.get(t, "")}
                            for t in wanted if t in self.results]}


def _portfolio_with(*specs, bankroll=money.STARTING_BANKROLL):
    """specs are (ticker, side, price, contracts)."""
    p = Portfolio("stan", bankroll=bankroll)
    positions = []
    for ticker, side, price, contracts in specs:
        fill = fills.buy([(price, 10000)], contracts, side)
        positions.append(p.open_position(ticker, side, fill, day="d1"))
    return p, positions


# ---------- reading Kalshi's result field ----------

def test_resolved_side_maps_results_to_sides():
    assert settlement.resolved_side("yes") == YES
    assert settlement.resolved_side("no") == NO
    assert settlement.resolved_side("YES") == YES       # case-insensitive
    assert settlement.resolved_side(" no ") == NO       # tolerant of whitespace


def test_an_empty_result_means_still_live():
    assert settlement.resolved_side("") is None
    assert settlement.resolved_side(None) is None


def test_void_results_are_recognised():
    for value in ("void", "voided", "cancelled", "canceled"):
        assert settlement.resolved_side(value) == "void"


# ---------- settling ----------

def test_a_yes_holder_wins_when_the_market_resolves_yes():
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    rest = FakeRest({"A": "yes"})

    settled = settlement.settle_open_positions(p, "d2", rest)

    assert settled == [(pos, "win")]
    assert pos.status == SETTLED
    assert pos.realized_pnl > 0
    assert p.realized_today("d2") == pos.realized_pnl


def test_a_yes_holder_loses_when_the_market_resolves_no():
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    settled = settlement.settle_open_positions(p, "d2", FakeRest({"A": "no"}))
    assert settled == [(pos, "loss")]
    assert pos.realized_pnl == -pos.cost


def test_a_no_holder_wins_when_the_market_resolves_no():
    """The mirror case -- easy to get backwards, and silent when you do."""
    p, (pos,) = _portfolio_with(("A", NO, 4000, 10))
    settled = settlement.settle_open_positions(p, "d2", FakeRest({"A": "no"}))
    assert settled == [(pos, "win")]
    assert pos.realized_pnl > 0


def test_a_no_holder_loses_when_the_market_resolves_yes():
    p, (pos,) = _portfolio_with(("A", NO, 4000, 10))
    settled = settlement.settle_open_positions(p, "d2", FakeRest({"A": "yes"}))
    assert settled == [(pos, "loss")]


def test_unresolved_positions_are_left_alone():
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    assert settlement.settle_open_positions(p, "d2", FakeRest({"A": ""})) == []
    assert pos.is_open


def test_only_the_resolved_positions_settle():
    p, (a, b) = _portfolio_with(("A", YES, 4000, 10), ("B", YES, 4000, 10))
    settled = settlement.settle_open_positions(p, "d2", FakeRest({"A": "yes", "B": ""}))
    assert [pos for pos, _ in settled] == [a]
    assert b.is_open


def test_pnl_is_credited_to_the_resolution_day():
    """PRD 7, enforced end to end through the settlement path."""
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    settlement.settle_open_positions(p, "2026-09-01", FakeRest({"A": "yes"}))
    assert p.realized_today("d1") == 0
    assert p.realized_today("2026-09-01") == pos.realized_pnl


# ---------- voids ----------

def test_a_void_refunds_the_stake_and_is_not_a_loss():
    """A cancelled game is not a decision the agent got wrong.

    Booking it as a loss would punish an agent for something outside its control
    and inject pure noise into the learning signal.
    """
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    before = p.bankroll

    settled = settlement.settle_open_positions(p, "d2", FakeRest({"A": "void"}))

    assert settled == [(pos, "void")]
    assert pos.result == "void"
    assert pos.realized_pnl == 0
    assert p.bankroll == before + pos.cost      # stake and fee both returned
    assert p.realized_today("d2") == 0


# ---------- robustness ----------

def test_nothing_open_means_no_api_call():
    rest = FakeRest()
    assert settlement.settle_open_positions(Portfolio("stan"), "d2", rest) == []
    assert rest.calls == []


def test_an_api_failure_settles_nothing_and_does_not_raise():
    """A long-running system must survive Kalshi being briefly unavailable."""
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    assert settlement.settle_open_positions(p, "d2", FakeRest(fail=True)) == []
    assert pos.is_open


def test_a_market_missing_from_the_response_is_left_open():
    """One bad ticker must not stop the other positions from settling."""
    p, (a, b) = _portfolio_with(("A", YES, 4000, 10), ("GONE", YES, 4000, 10))
    settled = settlement.settle_open_positions(p, "d2", FakeRest({"A": "yes"}))
    assert [pos for pos, _ in settled] == [a]
    assert b.is_open


def test_lookups_are_batched_not_one_request_per_position():
    """Thirty positions must not cost thirty HTTP requests every poll."""
    specs = [(f"T{i}", YES, 4000, 1) for i in range(250)]
    # 250 positions cannot fit in the standard $100 stake; the point of this
    # test is the request count, so fund it generously.
    p, _ = _portfolio_with(*specs, bankroll=money.dollars(1000))
    rest = FakeRest({f"T{i}": "" for i in range(250)})

    settlement.settle_open_positions(p, "d2", rest)

    assert len(rest.calls) == 3            # 250 tickers at BATCH=100
    assert all(len(c["tickers"].split(",")) <= settlement.BATCH for c in rest.calls)


def test_duplicate_tickers_are_looked_up_once():
    """Two positions in the same market should not double the request size."""
    p, _ = _portfolio_with(("A", YES, 4000, 5), ("A", NO, 4000, 5))
    rest = FakeRest({"A": ""})
    settlement.settle_open_positions(p, "d2", rest)
    assert rest.calls[0]["tickers"] == "A"


def test_both_sides_of_one_market_settle_oppositely():
    """Holding YES and NO on the same market: exactly one side wins."""
    p, (yes_pos, no_pos) = _portfolio_with(("A", YES, 4000, 10), ("A", NO, 4000, 10))
    settled = dict(settlement.settle_open_positions(p, "d2", FakeRest({"A": "yes"})))
    assert settled[yes_pos] == "win"
    assert settled[no_pos] == "loss"


# ---------- memory hand-off ----------

def test_outcome_for_memory_reports_return_on_risk():
    """Raw P&L hides bet size; the memory bank needs the ratio.

    A $2 profit on a $2 stake and a $2 profit on a $50 stake are very different
    lessons and must not look identical in memory.
    """
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    settlement.settle_open_positions(p, "d2", FakeRest({"A": "yes"}))

    row = settlement.outcome_for_memory(pos, "win")
    assert row["won"] is True
    assert row["entry_price"] == pos.entry_price
    assert row["return_on_risk"] == pytest.approx(pos.realized_pnl / pos.cost)
    assert row["stake"] == pos.cost


def test_outcome_for_memory_handles_a_loss():
    p, (pos,) = _portfolio_with(("A", YES, 4000, 10))
    settlement.settle_open_positions(p, "d2", FakeRest({"A": "no"}))
    row = settlement.outcome_for_memory(pos, "loss")
    assert row["won"] is False
    assert row["return_on_risk"] == pytest.approx(-1.0)   # total loss of stake
