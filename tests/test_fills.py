"""Tests for money arithmetic and simulated execution.

These are the numbers that decide whether an agent got richer or poorer, so they
are tested hard. A bug here does not crash anything -- it silently teaches four
agents the wrong lesson for weeks.
"""

import pytest

from sim import money, fills
from sim.fills import YES, NO
from kalshi.orderbook import OrderBook


# ---------- money ----------

def test_one_dollar_is_ten_thousand_units():
    assert money.ONE_DOLLAR == 10000
    assert money.dollars(1.00) == 10000
    assert money.dollars(0.006) == 60
    assert money.STARTING_BANKROLL == 1_000_000


def test_money_roundtrips_through_dollars():
    for amount in (0, 1, 4200, 1_000_000):
        assert money.dollars(money.to_dollars(amount)) == amount


def test_deci_cent_prices_are_representable():
    """Kalshi's ladder steps by 0.0010 below 10c. Integer cents cannot hold that.

    This is the whole reason the money unit is ten-thousandths and not cents.
    """
    assert money.dollars(0.0010) == 10
    assert money.dollars(0.0060) == 60
    assert money.dollars(0.0995) == 995


def test_no_price_is_complementary():
    assert money.no_price(4200) == 5800
    assert money.no_price(money.no_price(4200)) == 4200
    assert money.no_price(0) == money.ONE_DOLLAR


def test_fee_matches_kalshi_published_formula():
    """fee = ceil(0.07 * contracts * p * (1-p)), rounded up to the next cent."""
    # 100 contracts at 50c: 0.07 * 100 * 0.5 * 0.5 = $1.75 -> $1.76 after ceiling
    assert money.trade_fee(100, 5000) == money.dollars(1.76)


def test_fee_peaks_at_the_midpoint_and_falls_at_the_extremes():
    """p*(1-p) means uncertainty is expensive and long shots are cheap.

    Worth locking down: it is a real structural quirk of the exchange, and one
    Cartman's long-shot preference could plausibly discover.
    """
    mid = money.trade_fee(100, 5000)
    longshot = money.trade_fee(100, 500)
    favourite = money.trade_fee(100, 9500)
    assert mid > longshot
    assert longshot == favourite       # symmetric in p and (1-p)


def test_fee_is_never_rounded_down():
    """An agent that could round its way to a free trade would find that hole."""
    for contracts in range(1, 40):
        for price in (60, 500, 2500, 5000, 7500, 9900):
            fee = money.trade_fee(contracts, price)
            assert fee % money.CENT == 0, "fees settle in whole cents"
            exact = money.FEE_RATE * contracts * (price / money.ONE_DOLLAR) * \
                (1 - price / money.ONE_DOLLAR) * money.ONE_DOLLAR
            assert fee >= exact - 1e-6


def test_fee_is_zero_at_the_boundaries():
    assert money.trade_fee(0, 5000) == 0
    assert money.trade_fee(100, 0) == 0
    assert money.trade_fee(100, money.ONE_DOLLAR) == 0


@pytest.mark.parametrize("price", [60, 500, 2500, 4200, 5000, 9500])
def test_affordable_contracts_never_exceed_budget(price):
    budget = money.STARTING_BANKROLL
    n = money.max_contracts_affordable(budget, price)
    assert n * price + money.trade_fee(n, price) <= budget
    # And it is the *most* affordable: one more would break the budget.
    assert (n + 1) * price + money.trade_fee(n + 1, price) > budget


def test_affordable_contracts_handles_an_empty_wallet():
    assert money.max_contracts_affordable(0, 4200) == 0
    assert money.max_contracts_affordable(100, 4200) == 0


# ---------- walking a ladder ----------

def test_walk_fills_from_the_best_price_first():
    levels = [(4200, 100), (4300, 100), (4400, 100)]
    filled, gross, walked = fills.walk(levels, 150)
    assert filled == 150
    assert gross == 100 * 4200 + 50 * 4300
    assert walked == [(4200, 100), (4300, 50)]


def test_walk_partially_fills_when_the_ladder_runs_out():
    filled, gross, walked = fills.walk([(4200, 10)], 100)
    assert filled == 10
    assert gross == 10 * 4200


def test_walk_floors_fractional_sizes():
    """Kalshi sends sizes as fractional strings; contracts are whole units.

    Flooring rather than rounding means an agent can never take size that was
    not actually offered.
    """
    filled, _, _ = fills.walk([(4200, 10.9)], 100)
    assert filled == 10


def test_walk_skips_empty_levels():
    filled, gross, _ = fills.walk([(4200, 0), (4300, 5)], 10)
    assert filled == 5 and gross == 5 * 4300


def test_walk_on_an_empty_ladder_fills_nothing():
    assert fills.walk([], 10) == (0, 0, [])


# ---------- buying ----------

def test_buy_charges_the_ladder_plus_a_fee():
    fill = fills.buy([(4200, 100)], 10, YES)
    assert fill.contracts == 10
    assert fill.avg_price == 4200
    assert fill.gross == 42000
    assert fill.fee == money.trade_fee(10, 4200)
    assert fill.cost == fill.gross + fill.fee
    assert fill.complete is True
    assert fill.slippage == 0


def test_buy_across_levels_produces_slippage():
    """The point of simulating depth: over-sizing a thin market costs real money."""
    fill = fills.buy([(4200, 10), (4500, 10), (5000, 10)], 30, YES)
    assert fill.contracts == 30
    assert fill.avg_price == (10 * 4200 + 10 * 4500 + 10 * 5000) // 30
    assert fill.slippage > 0
    assert len(fill.levels_walked) == 3


def test_buy_more_than_the_book_holds_is_a_partial_fill():
    fill = fills.buy([(4200, 5)], 100, YES)
    assert fill.contracts == 5
    assert fill.filled is True
    assert fill.complete is False


def test_buy_with_no_liquidity_fills_nothing_and_costs_nothing():
    fill = fills.buy([], 10, YES)
    assert fill.contracts == 0
    assert fill.cost == 0
    assert fill.filled is False


def test_buy_zero_or_negative_is_a_no_op():
    for n in (0, -5):
        fill = fills.buy([(4200, 100)], n, YES)
        assert fill.contracts == 0 and fill.cost == 0


# ---------- selling ----------

def test_sell_proceeds_are_net_of_the_fee():
    """Kalshi charges on the way out too -- which is what makes churn expensive."""
    fill = fills.sell([(4200, 100)], 10, YES)
    assert fill.contracts == 10
    assert fill.gross == 42000
    assert fill.cost == fill.gross - fill.fee      # proceeds, not outlay
    assert fill.fee > 0


def test_a_round_trip_at_a_flat_price_loses_money():
    """Buy and immediately sell at the same price: you are down two fees.

    Encodes the fact that doing nothing beats churning at a flat price -- which
    is exactly the pressure the PRD wants inaction to be able to win against.
    """
    bought = fills.buy([(5000, 100)], 50, YES)
    sold = fills.sell([(5000, 100)], 50, YES)
    assert sold.cost < bought.cost
    assert bought.cost - sold.cost == bought.fee + sold.fee


# ---------- liquidity sources ----------

def _book():
    b = OrderBook()
    b.apply_snapshot({"type": "orderbook_snapshot", "seq": 1, "msg": {
        "market_ticker": "M",
        "yes_dollars_fp": [["0.4700", "300.00"], ["0.4600", "100.00"]],
        "no_dollars_fp": [["0.5200", "200.00"], ["0.5100", "60.00"]]}})
    return b


def test_buying_yes_lifts_the_yes_ask_ladder():
    levels = fills.levels_from_book(_book(), YES)
    # Best YES ask = 1 - best NO bid (0.52) = 0.48
    assert levels[0][0] == 4800


def test_buying_no_lifts_the_mirrored_yes_bid_ladder():
    """A YES bid at p is somebody offering NO at (1 - p)."""
    levels = fills.levels_from_book(_book(), NO)
    assert levels[0][0] == money.no_price(4700)   # 5300


def test_yes_and_no_best_prices_do_not_cross():
    """Sanity: buying both sides must cost more than $1, never less.

    If it were less, the agents would find a risk-free arbitrage that exists only
    in our simulation -- the most dangerous kind of reward-function bug.
    """
    book = _book()
    yes = fills.levels_from_book(book, YES)[0][0]
    no = fills.levels_from_book(book, NO)[0][0]
    assert yes + no >= money.ONE_DOLLAR


class _Quote:
    """Minimal stand-in for kalshi.universe.Market."""

    def __init__(self, bid, ask, bid_size=100.0, ask_size=100.0):
        self.yes_bid, self.yes_ask = bid, ask
        self.yes_bid_size, self.yes_ask_size = bid_size, ask_size

    @property
    def can_buy_yes(self):
        return self.yes_ask is not None and 0 < self.yes_ask < money.ONE_DOLLAR

    @property
    def can_buy_no(self):
        return self.yes_bid is not None and 0 < self.yes_bid < money.ONE_DOLLAR


def test_quote_liquidity_offers_exactly_one_level():
    """The broad tier knows the touch and nothing beyond it.

    Inventing depth here would let agents book profits on liquidity that was
    never there, so a large order partially fills instead.
    """
    levels = fills.levels_from_quote(_Quote(4700, 4800), YES)
    assert levels == [(4800, 100.0)]
    fill = fills.buy(levels, 500, YES)
    assert fill.contracts == 100 and fill.complete is False


def test_quote_liquidity_is_empty_when_a_side_is_unquoted():
    assert fills.levels_from_quote(_Quote(0, 4800), NO) == []
    assert fills.levels_from_quote(_Quote(4700, money.ONE_DOLLAR), YES) == []


def test_exit_levels_mirror_entry_levels():
    """Selling YES hits the YES bid; selling NO hits the mirrored YES ask."""
    book = _book()
    assert fills.exit_levels_from_book(book, YES)[0][0] == 4700
    assert fills.exit_levels_from_book(book, NO)[0][0] == money.no_price(4800)


def test_exit_costs_less_than_entry_on_a_spread():
    """Round-tripping across the spread loses the spread plus both fees.

    The basic reason a churning agent bleeds even when it is right about direction.
    """
    book = _book()
    entry = fills.buy(fills.levels_from_book(book, YES), 50, YES)
    exit_ = fills.sell(fills.exit_levels_from_book(book, YES), 50, YES)
    assert exit_.cost < entry.cost
