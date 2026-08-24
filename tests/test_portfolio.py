"""Tests for bankroll, positions, and settlement.

The rule these exist to protect is PRD 7: **P&L counts toward the day it
resolves, not the day the position was opened.** Get that wrong and the reward
signal is attributed to the wrong episode -- which does not crash, does not warn,
and quietly makes every learning signal noise.
"""

import pytest

from sim import money, fills, portfolio
from sim.fills import YES, NO
from sim.portfolio import Portfolio, InsufficientFunds, OPEN, CLOSED, SETTLED


def _buy(price=5000, contracts=10, size=1000):
    """A filled buy at a flat price, for setting up positions."""
    return fills.buy([(price, size)], contracts, YES)


def _portfolio(bankroll=money.STARTING_BANKROLL):
    return Portfolio("stan", bankroll=bankroll)


# ---------- opening ----------

def test_opening_deducts_capital_immediately():
    """PRD 7: the bankroll drops when the trade is made, not when it resolves."""
    p = _portfolio()
    fill = _buy(price=5000, contracts=10)
    before = p.bankroll
    pos = p.open_position("M", YES, fill, day="2026-08-24")

    assert p.bankroll == before - fill.cost
    assert pos.status == OPEN
    assert pos.contracts == 10
    assert pos.cost == fill.gross + fill.fee


def test_opening_records_entry_features_for_credit_assignment():
    """The state at entry is what the settlement trains against."""
    p = _portfolio()
    features = {"price_bucket": 5, "memory_winrate": 0.6}
    pos = p.open_position("M", YES, _buy(), day="2026-08-24", features=features)
    assert pos.entry_features == features


def test_cannot_open_a_position_you_cannot_afford():
    p = _portfolio(bankroll=money.dollars(1.00))
    with pytest.raises(InsufficientFunds):
        p.open_position("M", YES, _buy(price=5000, contracts=100),
                        day="2026-08-24")
    assert p.bankroll == money.dollars(1.00)   # untouched


def test_an_unfilled_order_opens_nothing():
    p = _portfolio()
    before = p.bankroll
    assert p.open_position("M", YES, fills.buy([], 10, YES), day="d") is None
    assert p.bankroll == before


# ---------- settlement ----------

def test_a_winning_position_pays_one_dollar_per_contract():
    p = _portfolio()
    fill = _buy(price=4000, contracts=10)
    pos = p.open_position("M", YES, fill, day="2026-08-24")
    after_open = p.bankroll

    pnl = p.settle_position(pos, won=True, day="2026-08-25")

    assert pos.status == SETTLED
    assert pos.result == "win"
    assert p.bankroll == after_open + money.dollars(10.00)   # 10 contracts x $1
    assert pnl == money.dollars(10.00) - fill.cost
    assert pnl > 0


def test_a_losing_position_pays_nothing():
    p = _portfolio()
    fill = _buy(price=4000, contracts=10)
    pos = p.open_position("M", YES, fill, day="2026-08-24")
    after_open = p.bankroll

    pnl = p.settle_position(pos, won=False, day="2026-08-25")

    assert pos.result == "loss"
    assert p.bankroll == after_open          # nothing comes back
    assert pnl == -fill.cost                 # the whole stake, fee included


def test_pnl_lands_on_the_resolution_day_not_the_open_day():
    """The core PRD 7 rule, and the one worth failing loudly on.

    A position opened Monday and resolved Wednesday belongs to Wednesday's
    reward, because Wednesday is when the outcome became knowable.
    """
    p = _portfolio()
    pos = p.open_position("M", YES, _buy(), day="2026-08-24")
    p.settle_position(pos, won=True, day="2026-08-26")

    assert p.realized_today("2026-08-24") == 0
    assert p.realized_today("2026-08-26") == pos.realized_pnl
    assert pos.opened_day == "2026-08-24"
    assert pos.closed_day == "2026-08-26"


def test_settling_a_closed_position_is_a_no_op():
    p = _portfolio()
    pos = p.open_position("M", YES, _buy(), day="d1")
    p.settle_position(pos, won=True, day="d2")
    assert p.settle_position(pos, won=True, day="d3") is None


def test_settlement_charges_no_fee():
    """Kalshi charges on trades, not on resolution."""
    p = _portfolio()
    pos = p.open_position("M", YES, _buy(), day="d1")
    p.settle_position(pos, won=True, day="d2")
    assert pos.exit_fee == 0


# ---------- voluntary exit ----------

def test_selling_before_resolution_realizes_pnl_today():
    """PRD 7: a voluntary exit counts toward the day it happened."""
    p = _portfolio()
    entry = _buy(price=4000, contracts=10)
    pos = p.open_position("M", YES, entry, day="2026-08-24")

    exit_fill = fills.sell([(6000, 1000)], 10, YES)
    pnl = p.close_position(pos, exit_fill, day="2026-08-25")

    assert pos.status == CLOSED
    assert pos.result == "exit"
    assert pnl == exit_fill.cost - entry.cost
    assert pnl > 0                                    # bought at 40c, sold at 60c
    assert p.realized_today("2026-08-25") == pnl
    assert p.realized_today("2026-08-24") == 0


def test_buy_low_sell_high_is_profitable_after_fees():
    """The behaviour PRD 2 explicitly wants available: buy-low-sell-high."""
    p = _portfolio()
    entry = _buy(price=2000, contracts=100)
    pos = p.open_position("M", YES, entry, day="d1")
    exit_fill = fills.sell([(8000, 1000)], 100, YES)
    pnl = p.close_position(pos, exit_fill, day="d1")
    assert pnl > 0


def test_a_flat_round_trip_loses_both_fees():
    """Churning at an unchanged price is strictly negative. Kenny will find this."""
    p = _portfolio()
    entry = _buy(price=5000, contracts=50)
    pos = p.open_position("M", YES, entry, day="d1")
    exit_fill = fills.sell([(5000, 1000)], 50, YES)
    pnl = p.close_position(pos, exit_fill, day="d1")
    assert pnl == -(entry.fee + exit_fill.fee)
    assert pnl < 0


def test_closing_an_already_closed_position_is_a_no_op():
    p = _portfolio()
    pos = p.open_position("M", YES, _buy(), day="d1")
    p.close_position(pos, fills.sell([(5000, 1000)], 10, YES), day="d1")
    assert p.close_position(pos, fills.sell([(5000, 1000)], 10, YES), "d1") is None


# ---------- exposure and equity ----------

def test_exposure_tracks_capital_tied_up():
    p = _portfolio()
    a = p.open_position("A", YES, _buy(price=5000, contracts=10), day="d1")
    b = p.open_position("B", YES, _buy(price=3000, contracts=10), day="d1")
    assert p.exposure() == a.cost + b.cost
    assert p.largest_open_exposure() == max(a.cost, b.cost)


def test_equity_is_bankroll_plus_open_positions():
    p = _portfolio()
    start = p.bankroll
    p.open_position("M", YES, _buy(), day="d1")
    # Valued at cost with no live price supplied: nothing has been won or lost yet.
    assert p.equity() == start


def test_mark_to_market_uses_live_prices():
    p = _portfolio()
    pos = p.open_position("M", YES, _buy(price=4000, contracts=10), day="d1")
    # Price doubled: the position is worth 10 x 0.80 = $8.00
    assert p.mark_to_market(lambda _: 8000) == money.dollars(8.00)
    assert pos.unrealized_pnl(8000) == money.dollars(8.00) - pos.cost


def test_unquoted_positions_are_marked_at_cost_not_zero():
    """An illiquid market has no price -- that is not the same as being worthless.

    Marking to zero would fire the drawdown penalty on every thinly-traded
    holding and teach agents to avoid illiquid markets for a reason that is an
    artifact of our own bookkeeping.
    """
    p = _portfolio()
    pos = p.open_position("M", YES, _buy(), day="d1")
    assert p.mark_to_market(lambda _: None) == pos.cost


# ---------- drawdown ----------

def test_drawdown_is_zero_at_a_new_high():
    p = _portfolio()
    p.update_peak(money.dollars(150))
    assert p.drawdown(money.dollars(150)) == 0.0


def test_drawdown_measures_decline_from_the_peak():
    p = _portfolio()
    p.update_peak(money.dollars(200))
    assert p.drawdown(money.dollars(150)) == pytest.approx(0.25)


def test_peak_never_moves_down():
    p = _portfolio()
    p.update_peak(money.dollars(200))
    p.update_peak(money.dollars(50))
    assert p.peak_equity == money.dollars(200)


# ---------- bankruptcy ----------

def test_being_fully_invested_is_not_bankruptcy():
    """No cash but live positions is reckless, not ruined.

    Cartman will reach this state constantly. Firing the terminal penalty here
    would punish a position that might still win.
    """
    p = _portfolio(bankroll=money.dollars(10))
    p.open_position("M", YES, _buy(price=5000, contracts=19), day="d1")
    assert p.bankroll < money.dollars(1)
    assert p.is_bankrupt() is False


def test_bankruptcy_when_equity_falls_below_the_floor():
    """A ruined agent holds dust, not exactly zero -- so the test is a floor.

    Without it the bankruptcy counter the PRD wants on the dashboard would sit
    at zero forever while agents were plainly wiped out.
    """
    p = _portfolio(bankroll=money.dollars(10))
    pos = p.open_position("M", YES, _buy(price=5000, contracts=19), day="d1")
    p.settle_position(pos, won=False, day="d1")

    assert 0 < p.equity() < money.BANKRUPTCY_FLOOR   # dust, not zero
    assert p.is_bankrupt() is True


def test_an_agent_just_above_the_floor_is_not_bankrupt():
    p = _portfolio(bankroll=money.dollars(2))
    assert p.is_bankrupt() is False


def test_reset_restores_the_stake_and_counts_the_event():
    """PRD 5: reset to $100 and log it -- bankruptcy count is interesting data."""
    p = _portfolio(bankroll=money.dollars(10))
    pos = p.open_position("M", YES, _buy(price=5000, contracts=19), day="d1")

    assert p.reset_after_bankruptcy() == 1
    assert p.bankroll == money.STARTING_BANKROLL
    assert p.peak_equity == money.STARTING_BANKROLL
    assert pos.status == SETTLED and pos.result == "bankrupt"
    assert p.open_positions() == []


def test_reset_abandons_open_positions_rather_than_settling_them_later():
    """A reset agent must not later be credited for a pre-reset position."""
    p = _portfolio(bankroll=money.dollars(10))
    pos = p.open_position("M", YES, _buy(price=5000, contracts=19), day="d1")
    p.reset_after_bankruptcy()
    assert p.settle_position(pos, won=True, day="d2") is None


# ---------- baskets ----------

def test_a_basket_groups_legs_chosen_together():
    p = _portfolio()
    basket = p.open_basket(day="d1", label="three-leg")
    for ticker in ("A", "B", "C"):
        p.open_position(ticker, YES, _buy(contracts=5), day="d1",
                        basket_id=basket.id)
    assert len(basket) == 3
    assert all(p.positions[pid].basket_id == basket.id
               for pid in basket.position_ids)


def test_basket_legs_settle_independently():
    """A basket is not a parlay: two winning legs pay out even if a third loses.

    Buying legs separately cannot produce all-or-nothing payoff, and pretending
    otherwise would invent a payoff structure the exchange does not offer.
    Genuine parlays come from Kalshi's own listed combo markets instead.
    """
    p = _portfolio()
    basket = p.open_basket(day="d1")
    legs = [p.open_position(t, YES, _buy(price=5000, contracts=10), day="d1",
                            basket_id=basket.id) for t in ("A", "B", "C")]

    p.settle_position(legs[0], won=True, day="d2")
    p.settle_position(legs[1], won=True, day="d2")
    p.settle_position(legs[2], won=False, day="d2")

    assert legs[0].realized_pnl > 0
    assert legs[1].realized_pnl > 0
    assert legs[2].realized_pnl < 0
    # Net is positive -- an all-or-nothing parlay would have paid zero.
    assert sum(leg.realized_pnl for leg in legs) > 0


# ---------- reporting ----------

def test_stats_reports_win_rate_over_resolved_positions_only():
    p = _portfolio()
    won = p.open_position("A", YES, _buy(price=4000, contracts=10), day="d1")
    lost = p.open_position("B", YES, _buy(price=4000, contracts=10), day="d1")
    p.open_position("C", YES, _buy(price=4000, contracts=10), day="d1")  # open

    p.settle_position(won, won=True, day="d1")
    p.settle_position(lost, won=False, day="d1")

    s = p.stats()
    assert s["wins"] == 1 and s["losses"] == 1
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["open_positions"] == 1


def test_win_rate_is_none_before_anything_resolves():
    """Nothing resolved is not a 0% win rate -- it is no data.

    Reporting 0.0 would put every agent at the bottom of the head-to-head view
    on day one and make the comparison meaningless.
    """
    assert _portfolio().stats()["win_rate"] is None


def test_fees_accumulate_across_entry_and_exit():
    p = _portfolio()
    entry = _buy(price=5000, contracts=20)
    pos = p.open_position("M", YES, entry, day="d1")
    exit_fill = fills.sell([(5000, 1000)], 20, YES)
    p.close_position(pos, exit_fill, day="d1")
    assert p.total_fees == entry.fee + exit_fill.fee
