"""Money and price arithmetic for the simulation.

## One unit, everywhere: ten-thousandths of a dollar

Every monetary quantity in this project -- prices, stakes, bankrolls, fees, P&L
-- is an **integer number of ten-thousandths of a dollar**. $1.00 is 10000,
$100.00 is 1_000_000, and a contract quoted at "0.0060" is 60.

Two reasons, and they are not stylistic:

1. **Kalshi quotes to four decimals.** Its `tapered_deci_cent` price ladder steps
   by 0.0010 below 10c and above 90c. Integer *cents* cannot represent a 0.0060
   contract at all, so cents are simply the wrong unit for this exchange.

2. **Floats drift.** These agents place thousands of trades over weeks and the
   bankroll is a running sum. Float dollars accumulate representation error until
   a bankroll that should read 100.00 reads 99.99999999999997 -- and the PRD
   makes bankruptcy (bankroll <= 0) a terminal event with a strong penalty. A
   drifting bankroll means an agent can be bankrupted by rounding.

The same convention is already used by `kalshi/orderbook.py`, which keys book
levels by `round(price * 10000)`, so the whole codebase speaks one unit.

## Contract mechanics

A Kalshi contract settles at $1 if its side is correct and $0 if not.

  - Buying YES at price p costs p and returns ONE_DOLLAR if the market resolves
    YES, 0 otherwise.
  - Buying NO at price q costs q and returns ONE_DOLLAR if it resolves NO.
  - The two sides are complementary: no_price = ONE_DOLLAR - yes_price. So a NO
    ask is derived from the YES bid, which is why `kalshi/orderbook.py` builds
    the YES ask ladder out of the NO bid side.

The agents start knowing none of this (PRD 2). They discover that a cheaper
contract pays more per dollar risked by losing money until the pattern shows up
in their memory bank.
"""

import math

# One dollar, in the project's money unit.
ONE_DOLLAR = 10000

# What each agent starts with, and what a bankruptcy reset restores (PRD 5).
STARTING_BANKROLL = 100 * ONE_DOLLAR

# Equity below this counts as bankruptcy (PRD 5: "if bankroll hits zero").
#
# Taken literally, "zero" almost never happens: a losing agent is left holding
# dust -- three cents, half a cent -- so an `equity <= 0` test would essentially
# never fire and the bankruptcy counter the PRD wants on the dashboard would sit
# at zero forever while agents were plainly ruined.
#
# $1.00 is the practical floor. An agent down from $100 to under a dollar has
# lost 99% of its stake and cannot construct a position that matters. Exposed
# here rather than inlined because, like every other threshold in this project,
# it is expected to need tuning once there is real behaviour to look at.
BANKRUPTCY_FLOOR = 1 * ONE_DOLLAR

# Kalshi's published trading fee: 7% of the notional "risk" of the trade,
# rounded UP to the next cent. In dollars that is
#     fee = ceil(0.07 * contracts * price * (1 - price))
# The price*(1-price) term means fees peak at 50c (maximum uncertainty) and fall
# toward zero at the extremes -- so long-shot contracts are cheap to trade, which
# is exactly the kind of structural quirk we want Cartman to be able to discover.
FEE_RATE = 0.07
CENT = ONE_DOLLAR // 100


def dollars(amount):
    """Float dollars -> money units. For config and display boundaries only."""
    return round(amount * ONE_DOLLAR)


def to_dollars(units):
    """Money units -> float dollars. For display and logging only."""
    return units / ONE_DOLLAR


def fmt(units):
    """Money units -> '$12.34' for logs and the dashboard."""
    return f"${units / ONE_DOLLAR:,.2f}"


def no_price(yes_price):
    """The complementary side. Buying NO at (1 - p) is buying against YES at p."""
    return ONE_DOLLAR - yes_price


def trade_fee(contracts, price):
    """Kalshi's fee for `contracts` at `price`, in money units, rounded up to a cent.

    `price` is the price of the side being bought, in money units. The fee is
    symmetric in the two sides because p*(1-p) is.
    """
    if contracts <= 0 or price <= 0 or price >= ONE_DOLLAR:
        return 0
    p = price / ONE_DOLLAR
    fee_dollars = FEE_RATE * contracts * p * (1.0 - p)
    fee_units = fee_dollars * ONE_DOLLAR
    # Round up to the next whole cent, as Kalshi does. Never round a fee down --
    # an agent that could round its way to a free trade would find that loophole.
    return int(math.ceil(fee_units / CENT)) * CENT


def payout(contracts):
    """What `contracts` winning contracts pay out. Each settles at $1."""
    return contracts * ONE_DOLLAR


def max_contracts_affordable(budget, price, include_fee=True):
    """How many contracts `budget` buys at `price`, fees included.

    Solved by estimate-then-correct rather than algebraically: the fee's ceiling
    to the next cent makes the exact relationship a step function, and one or two
    correction passes is clearer than inverting it.
    """
    if price <= 0 or budget <= 0:
        return 0
    contracts = budget // price
    if not include_fee:
        return int(contracts)

    def total(n):
        return n * price + trade_fee(n, price)

    # Shrink by the shortfall rather than one at a time, so a large budget with
    # a cheap contract doesn't spin for thousands of iterations.
    while contracts > 0 and total(contracts) > budget:
        over = total(contracts) - budget
        contracts -= max(1, over // price)
    contracts = max(contracts, 0)

    # The jump above can overshoot: the fee is a step function of the count, so
    # landing below the true maximum is possible. Walk back up until one more
    # would break the budget. Without this the agents are quietly under-sized,
    # which looks like risk aversion rather than the arithmetic bug it is.
    while total(contracts + 1) <= budget:
        contracts += 1
    return int(contracts)
