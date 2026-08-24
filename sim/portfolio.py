"""An agent's simulated money: bankroll, open positions, and realized P&L.

One `Portfolio` per agent. The four agents share no state whatsoever (PRD 2), so
nothing here is a class attribute and nothing is global.

## Settlement rules, straight from PRD 7

  - Opening a position **immediately** deducts the staked capital. The bankroll
    drops the moment the trade is made, not when it resolves.
  - P&L is credited **on resolution**, and counts toward the reward of the day it
    resolved -- not the day the position was opened. Resolution day is when the
    learnable outcome actually becomes known, so that is where the learning
    signal belongs.
  - Positions may be held across days. Those are unrealized until they settle.
  - Selling before resolution is fully supported and realizes P&L immediately,
    counting toward that day's reward.

## Baskets are not parlays, and the difference is real

PRD 2 asks for the ability to "construct or take multi-leg combos/parlays".
Those are two different things and this module keeps them apart:

  **Take** -- Kalshi lists ~1.19M auto-generated cross-category parlay markets.
  Those are genuine all-or-nothing instruments, and they are just markets: an
  agent buys one the same way it buys anything else. True parlay payoff.

  **Construct** -- an agent buys several legs as one atomic decision. This is a
  *basket*, not a parlay: if two of three legs win, it collects on those two.
  You cannot synthesize all-or-nothing payoff by buying legs separately, and
  pretending otherwise would invent a payoff structure that does not exist.

So a `Basket` here groups legs for tracking, sizing, and attribution -- the
agent decided on them together, so they are evaluated together -- while each leg
settles on its own merits. Cartman's pull toward combos is served by the listed
parlays; the basket is what gives "these five things all happen" a mechanical
form the agent can express.

## Credit assignment

Every position carries `entry_features`: the feature vector as it was when the
agent decided to open it. On settlement or exit, that vector plus the realized
return is one training example. This is what turns ~7 learning signals a week
into dozens or hundreds (see the plan's learning-signal decision).
"""

import time
import itertools

from . import money
from .fills import YES, NO

OPEN = "open"
CLOSED = "closed"      # voluntarily exited before resolution
SETTLED = "settled"    # held to resolution

_ids = itertools.count(1)


def _new_id(prefix):
    return f"{prefix}{next(_ids)}"


class Position:
    """One side of one market, held by one agent."""

    __slots__ = ("id", "agent", "ticker", "side", "contracts", "entry_price",
                 "entry_gross", "entry_fee", "opened_at", "opened_day",
                 "basket_id", "status", "exit_price", "exit_proceeds",
                 "exit_fee", "closed_at", "closed_day", "realized_pnl",
                 "entry_features", "category", "series", "close_ts", "result",
                 "stake_fraction")

    def __init__(self, agent, ticker, side, fill, day, features=None,
                 basket_id=None, category=None, series=None, close_ts=None,
                 stake_fraction=0.0):
        self.id = _new_id("p")
        self.agent = agent
        self.ticker = ticker
        self.side = side
        self.contracts = fill.contracts
        self.entry_price = fill.avg_price
        self.entry_gross = fill.gross
        self.entry_fee = fill.fee
        self.opened_at = time.time()
        self.opened_day = day
        self.basket_id = basket_id
        self.category = category
        self.series = series
        self.close_ts = close_ts
        # How much of the bankroll this bet represented at the moment it was
        # made. Recorded at entry because it cannot be reconstructed later --
        # the bankroll has moved on -- and the risk-shaped learning target
        # needs it. See `agent/reward.trade_target`.
        self.stake_fraction = stake_fraction

        self.status = OPEN
        self.exit_price = None
        self.exit_proceeds = None
        self.exit_fee = None
        self.closed_at = None
        self.closed_day = None
        self.realized_pnl = None
        self.result = None

        # The state the agent saw when it chose this. The whole point of
        # per-trade credit assignment.
        self.entry_features = features

    @property
    def cost(self):
        """Total capital committed, fees included. Already out of the bankroll."""
        return self.entry_gross + self.entry_fee

    @property
    def is_open(self):
        return self.status == OPEN

    @property
    def max_payout(self):
        """What this returns if it wins. Each contract settles at $1."""
        return money.payout(self.contracts)

    def value_at(self, price):
        """Mark-to-market at `price` (the price of THIS side), before fees."""
        return self.contracts * price

    def unrealized_pnl(self, price):
        """Paper P&L if closed at `price` right now. Ignores the exit fee.

        Deliberately fee-free: this is a display and drawdown number, not a
        decision number. The agent's actual exit decision prices the fee in via
        `sim.fills.sell`.
        """
        return self.value_at(price) - self.cost

    def settle(self, won, day):
        """Resolve at $1 or $0 per contract. Returns the cash to credit back."""
        proceeds = self.max_payout if won else 0
        self.status = SETTLED
        self.result = "win" if won else "loss"
        self.exit_price = money.ONE_DOLLAR if won else 0
        self.exit_proceeds = proceeds
        self.exit_fee = 0          # Kalshi charges no fee on settlement
        self.closed_at = time.time()
        self.closed_day = day
        self.realized_pnl = proceeds - self.cost
        return proceeds

    def close(self, fill, day):
        """Voluntarily exit. `fill.cost` is proceeds net of the exit fee."""
        proceeds = fill.cost
        self.status = CLOSED
        self.result = "exit"
        self.exit_price = fill.avg_price
        self.exit_proceeds = proceeds
        self.exit_fee = fill.fee
        self.closed_at = time.time()
        self.closed_day = day
        self.realized_pnl = proceeds - self.cost
        return proceeds

    def to_row(self):
        """Flat dict for logging and the dashboard."""
        return {
            "id": self.id, "agent": self.agent, "ticker": self.ticker,
            "side": self.side, "contracts": self.contracts,
            "entry_price": self.entry_price, "cost": self.cost,
            "entry_fee": self.entry_fee, "opened_at": self.opened_at,
            "opened_day": self.opened_day, "basket_id": self.basket_id,
            "category": self.category, "series": self.series,
            "close_ts": self.close_ts, "status": self.status,
            "stake_fraction": self.stake_fraction,
            "exit_price": self.exit_price, "exit_proceeds": self.exit_proceeds,
            "exit_fee": self.exit_fee, "closed_at": self.closed_at,
            "closed_day": self.closed_day, "realized_pnl": self.realized_pnl,
            "result": self.result,
        }

    def __repr__(self):
        return (f"<Position {self.id} {self.agent} {self.side} {self.contracts}x "
                f"{self.ticker} @ {self.entry_price} [{self.status}]>")


class Basket:
    """Several legs the agent chose as one decision. See the module docstring.

    Not a parlay: each leg settles independently. What the basket provides is
    atomic sizing and joint attribution -- the agent committed to these together,
    so the dashboard and the memory bank treat them as one act.
    """

    __slots__ = ("id", "agent", "position_ids", "opened_at", "opened_day",
                 "entry_features", "label")

    def __init__(self, agent, day, features=None, label=None):
        self.id = _new_id("b")
        self.agent = agent
        self.position_ids = []
        self.opened_at = time.time()
        self.opened_day = day
        self.entry_features = features
        self.label = label

    def __len__(self):
        return len(self.position_ids)


class InsufficientFunds(Exception):
    """Raised when a trade would cost more than the agent has."""


class Portfolio:
    """One agent's money. Not thread-safe; each agent owns its own loop."""

    def __init__(self, agent, bankroll=money.STARTING_BANKROLL,
                 bankruptcy_floor=money.BANKRUPTCY_FLOOR):
        self.agent = agent
        self.bankroll = bankroll
        self.bankruptcy_floor = bankruptcy_floor
        self.positions = {}            # id -> Position (open and historical)
        self.baskets = {}              # id -> Basket

        # Realized P&L keyed by the day it was realized, per PRD 7. This is the
        # dict the daily reward reads from.
        self.realized_by_day = {}

        self.peak_equity = bankroll    # for the drawdown penalty
        self.bankruptcies = 0
        self.trades_opened = 0
        self.trades_closed = 0
        self.total_fees = 0

    # ---------- opening ----------

    def can_afford(self, cost):
        return cost <= self.bankroll

    def open_position(self, ticker, side, fill, day, features=None,
                      basket_id=None, category=None, series=None,
                      close_ts=None):
        """Commit capital to a filled order. Deducts immediately (PRD 7)."""
        if not fill.filled:
            return None
        if not self.can_afford(fill.cost):
            raise InsufficientFunds(
                f"{self.agent} needs {money.fmt(fill.cost)} but holds "
                f"{money.fmt(self.bankroll)}")

        # Computed BEFORE the deduction: the fraction of what the agent had.
        capital = max(self.bankroll, 1)
        position = Position(self.agent, ticker, side, fill, day,
                            features=features, basket_id=basket_id,
                            category=category, series=series, close_ts=close_ts,
                            stake_fraction=fill.cost / capital)
        self.bankroll -= fill.cost
        self.total_fees += fill.fee
        self.positions[position.id] = position
        self.trades_opened += 1
        if basket_id and basket_id in self.baskets:
            self.baskets[basket_id].position_ids.append(position.id)
        return position

    def open_basket(self, day, features=None, label=None):
        basket = Basket(self.agent, day, features=features, label=label)
        self.baskets[basket.id] = basket
        return basket

    # ---------- closing ----------

    def close_position(self, position, fill, day):
        """Voluntary exit before resolution. Realizes P&L today (PRD 7)."""
        if not position.is_open or not fill.filled:
            return None
        # A partial exit is not modelled: an agent either holds or exits. Keeping
        # positions atomic keeps credit assignment unambiguous, since a position
        # maps to exactly one (state, action) pair.
        proceeds = position.close(fill, day)
        self.bankroll += proceeds
        self.total_fees += fill.fee
        self.trades_closed += 1
        self._record_realized(day, position.realized_pnl)
        return position.realized_pnl

    def settle_position(self, position, won, day):
        """Resolution. P&L lands on the day it resolved, not the day it opened."""
        if not position.is_open:
            return None
        proceeds = position.settle(won, day)
        self.bankroll += proceeds
        self.trades_closed += 1
        self._record_realized(day, position.realized_pnl)
        return position.realized_pnl

    def _record_realized(self, day, pnl):
        self.realized_by_day[day] = self.realized_by_day.get(day, 0) + pnl

    # ---------- state ----------

    def open_positions(self):
        return [p for p in self.positions.values() if p.is_open]

    def exposure(self):
        """Capital currently tied up in open positions."""
        return sum(p.cost for p in self.open_positions())

    def largest_open_exposure(self):
        """Biggest single open position. Feeds the risk-adjusted penalty (PRD 5)."""
        open_ = self.open_positions()
        return max((p.cost for p in open_), default=0)

    def mark_to_market(self, price_of):
        """Value of open positions. `price_of(position)` returns this side's price.

        Falls back to entry cost when a market has no live quote, which values
        an unquoted position at what was paid rather than at zero. Marking to
        zero would make every illiquid holding look like a catastrophic loss and
        would fire the drawdown penalty on markets that simply have not traded.
        """
        total = 0
        for position in self.open_positions():
            price = price_of(position)
            total += position.value_at(price) if price is not None else position.cost
        return total

    def equity(self, price_of=None):
        """Bankroll plus the value of open positions -- the agent's real worth."""
        if price_of is None:
            return self.bankroll + self.exposure()
        return self.bankroll + self.mark_to_market(price_of)

    def realized_today(self, day):
        return self.realized_by_day.get(day, 0)

    def all_time_realized(self):
        return sum(self.realized_by_day.values())

    # ---------- drawdown and bankruptcy ----------

    def update_peak(self, equity):
        if equity > self.peak_equity:
            self.peak_equity = equity
        return self.peak_equity

    def drawdown(self, equity):
        """Fractional decline from the peak. 0.0 at a new high."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def is_bankrupt(self, price_of=None):
        """Ruined: equity has fallen below the viable-stake floor.

        Two decisions are baked in here.

        **Equity, not bankroll.** An agent with no cash but $40 of open positions
        is not bankrupt -- it is fully invested, which is a legitimate (if
        reckless) state and one Cartman will reach constantly. Declaring that
        bankruptcy would fire the terminal penalty on positions that might still
        win, and would teach exactly the wrong lesson.

        **A floor, not zero.** See `money.BANKRUPTCY_FLOOR`: a ruined agent is
        left holding dust rather than exactly nothing, so a `<= 0` test would
        never fire in practice.
        """
        return self.equity(price_of) < self.bankruptcy_floor

    def reset_after_bankruptcy(self, bankroll=money.STARTING_BANKROLL):
        """Restore the starting stake and count the event (PRD 5).

        Open positions are abandoned rather than force-closed: they are already
        worthless by definition, and settling them later would credit money to an
        agent that has since been reset.
        """
        for position in self.open_positions():
            position.status = SETTLED
            position.result = "bankrupt"
            position.realized_pnl = -position.cost
            position.closed_at = time.time()
        self.bankruptcies += 1
        self.bankroll = bankroll
        self.peak_equity = bankroll
        return self.bankruptcies

    # ---------- reporting ----------

    def stats(self, price_of=None):
        equity = self.equity(price_of)
        settled = [p for p in self.positions.values()
                   if p.status in (SETTLED, CLOSED) and p.realized_pnl is not None]
        wins = sum(1 for p in settled if p.realized_pnl > 0)
        return {
            "agent": self.agent,
            "bankroll": self.bankroll,
            "equity": equity,
            "exposure": self.exposure(),
            "open_positions": len(self.open_positions()),
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown(equity),
            "all_time_realized": self.all_time_realized(),
            "trades_opened": self.trades_opened,
            "trades_closed": self.trades_closed,
            "wins": wins,
            "losses": len(settled) - wins,
            "win_rate": (wins / len(settled)) if settled else None,
            "total_fees": self.total_fees,
            "bankruptcies": self.bankruptcies,
        }
