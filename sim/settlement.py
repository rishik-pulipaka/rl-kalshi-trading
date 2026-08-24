"""Detecting resolution and crediting P&L on the day it happened.

Positions can be held across days (PRD 7). This module is what notices that a
market an agent holds has finally resolved, works out whether the agent's side
won, and hands the result to the portfolio so the P&L lands on **today's**
episode -- the day the outcome became knowable -- rather than the day the
position was opened.

## How resolution is detected

A market's `result` field is `""` while it is live and becomes `"yes"` or `"no"`
once determined. Markets are looked up in batches (`/markets?tickers=a,b,c`,
verified working), because an agent holding thirty positions should not cost
thirty HTTP requests every poll.

## Voided markets

Kalshi can void a market -- a game cancelled, an event that never happened. A
void is not a loss: the stake comes back. Treating a void as a loss would punish
agents for something no decision of theirs caused, and would inject pure noise
into the learning signal.

## The leakage trap, one more time

Nothing here reads `last_price`. On a settled market that field is the
post-settlement print (0.9990 for the winning side, 0.0010 for the loser) -- it
is the answer, not a forecast. It is safe to use `result` because we are
settling a position that is already open, but any *historical* price must come
from candlesticks. See `kalshi/rest.py`.
"""

import logging

from . import money
from .fills import YES, NO

log = logging.getLogger(__name__)

# Kalshi's `result` vocabulary.
RESULT_YES = "yes"
RESULT_NO = "no"
# A void/cancelled market refunds rather than paying out. Kalshi has used a few
# spellings here over time, so all are treated the same.
VOID_RESULTS = frozenset({"void", "voided", "cancelled", "canceled", "all_no"})

# Markets are looked up in batches; 100 keeps the query string well short of any
# URL length limit while making the request count negligible.
BATCH = 100


def resolved_side(result):
    """Map a Kalshi `result` to the winning side, or None if unresolved.

    Returns "void" for a cancelled market, which the caller refunds.
    """
    if not result:
        return None
    value = str(result).strip().lower()
    if value == RESULT_YES:
        return YES
    if value == RESULT_NO:
        return NO
    if value in VOID_RESULTS:
        return "void"
    return None


def fetch_results(tickers, rest_module, session=None):
    """`{ticker: result}` for the given markets, batched.

    Missing markets are simply absent from the mapping rather than raising -- a
    ticker can disappear from the API, and one bad ticker must not stop the
    other twenty-nine positions from settling.
    """
    tickers = [t for t in tickers if t]
    results = {}
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        try:
            data = rest_module._get("/markets",
                                    {"tickers": ",".join(batch), "limit": BATCH},
                                    session=session)
        except Exception as exc:
            log.warning("settlement lookup failed for %d markets: %s",
                        len(batch), exc)
            continue
        for market in data.get("markets", []):
            results[market.get("ticker")] = market.get("result")
    return results


def settle_open_positions(portfolio, day, rest_module, session=None):
    """Settle every open position whose market has resolved.

    Returns a list of `(position, outcome)` for logging, where outcome is one of
    "win", "loss", or "void".
    """
    open_positions = portfolio.open_positions()
    if not open_positions:
        return []

    tickers = {p.ticker for p in open_positions}
    results = fetch_results(tickers, rest_module, session=session)

    settled = []
    for position in open_positions:
        winner = resolved_side(results.get(position.ticker))
        if winner is None:
            continue                       # still live; hold it
        if winner == "void":
            refund_position(portfolio, position, day)
            settled.append((position, "void"))
            continue
        won = (winner == position.side)
        portfolio.settle_position(position, won=won, day=day)
        settled.append((position, "win" if won else "loss"))

    if settled:
        log.info("%s settled %d positions on %s", portfolio.agent, len(settled), day)
    return settled


def refund_position(portfolio, position, day):
    """Return the stake on a voided market. Not a win and not a loss.

    The fee is refunded too: the trade is being unwound as though it never
    happened, which is what a void means.
    """
    position.status = "settled"
    position.result = "void"
    position.exit_price = position.entry_price
    position.exit_proceeds = position.cost
    position.exit_fee = 0
    position.closed_day = day
    position.realized_pnl = 0
    portfolio.bankroll += position.cost
    portfolio.trades_closed += 1
    portfolio._record_realized(day, 0)
    return 0


def classify(outcome, pnl):
    """Reduce an outcome label to what memory records: win, loss, or void.

    A voluntary exit is a win or a loss depending on whether it made money.
    Recording it as its own third thing was a real bug: exits counted as neither
    win nor loss, so a memory row could sit at 46 encounters with a win rate of
    None -- which then made the belief invisible to the policy, because
    `summarize_memory` skips beliefs with no win rate. An agent that mostly
    exits positions would learn nothing at all.
    """
    if outcome == "void":
        return "void"
    if outcome == "exit":
        return "win" if pnl > 0 else "loss"
    return outcome


def outcome_for_memory(position, outcome):
    """Compact summary of how a position turned out, for the memory bank.

    Return-on-risk rather than raw P&L: a $2 profit on a $2 stake and a $2
    profit on a $50 stake are very different lessons, and the memory bank is
    where an agent needs to be able to tell them apart.
    """
    pnl = position.realized_pnl or 0
    outcome = classify(outcome, pnl)
    return {
        "ticker": position.ticker,
        "series": position.series,
        "category": position.category,
        "side": position.side,
        "entry_price": position.entry_price,
        "contracts": position.contracts,
        "stake": position.cost,
        "pnl": pnl,
        "return_on_risk": (pnl / position.cost) if position.cost else 0.0,
        "outcome": outcome,
        "won": outcome == "win",
        "held_seconds": ((position.closed_at or 0) - position.opened_at)
                        if position.closed_at else None,
    }
