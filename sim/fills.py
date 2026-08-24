"""Simulated order execution against real Kalshi liquidity.

No order ever reaches Kalshi. These functions take the live book (or the live
top-of-book quote) and work out what *would* have happened, so the agents face
honest constraints: finite size at each price, slippage when they ask for more
than the top level holds, and real fees.

## Why this matters for the learning problem

The lazy version of this -- "you always fill your whole order at the mid" --
quietly hands the agents free money. An agent that learns to size up on a thin
market would, under that model, be rewarded for it. Under this one it eats the
ladder and pays for it, which is the lesson we actually want it to be able to
learn.

## Two liquidity sources

The stream is two-tier (see `kalshi/stream.py`), so liquidity comes in two
fidelities and the fill code handles both through one shape -- a list of
`(price, size)` levels, best price first:

  DEPTH tier -- the real ladder from `kalshi/orderbook.py`. Multi-level, so
                walking it produces genuine slippage.
  BROAD tier -- one level, from the `ticker` frame's best bid/ask and its size.
                Everything beyond that level is unknown, so we refuse to fill
                past it rather than inventing depth that may not exist.

Refusing to invent depth is the conservative choice in the direction that
matters: it under-fills rather than over-fills, so an agent can never book a
profit on liquidity that was never there.

All prices and money are integer ten-thousandths of a dollar (see `sim/money.py`).
"""

from . import money

# Sides. YES and NO are the two halves of a Kalshi contract.
YES = "yes"
NO = "no"


class Fill:
    """The result of a simulated execution.

    `contracts` may be less than requested -- that is a partial fill, and it is
    the normal case on a thin market.
    """

    __slots__ = ("side", "contracts", "avg_price", "gross", "fee", "cost",
                 "levels_walked", "requested")

    def __init__(self, side, contracts, gross, fee, levels_walked, requested):
        self.side = side
        self.contracts = contracts
        self.gross = gross                    # before fees
        self.fee = fee
        self.cost = gross + fee               # what leaves the bankroll on a buy
        self.avg_price = (gross // contracts) if contracts else 0
        self.levels_walked = levels_walked
        self.requested = requested

    @property
    def filled(self):
        return self.contracts > 0

    @property
    def complete(self):
        return self.contracts == self.requested

    @property
    def slippage(self):
        """How far the average price landed from the best available price.

        Zero on a fill that never left the top level. This is the number that
        makes over-sizing visibly expensive on the dashboard.
        """
        if not self.levels_walked:
            return 0
        return self.avg_price - self.levels_walked[0][0]

    def __repr__(self):
        return (f"<Fill {self.side} {self.contracts}/{self.requested} "
                f"@ {self.avg_price} cost {money.fmt(self.cost)}>")


def walk(levels, contracts):
    """Consume up to `contracts` from a price ladder, best price first.

    Returns `(filled, gross, walked)` where `walked` lists the
    `(price, size_taken)` pairs actually consumed -- kept so the caller can show
    exactly where the slippage came from.

    Sizes on Kalshi arrive as fractional strings ("3000.00"); contracts are whole
    units, so each level's capacity is floored. Flooring is deliberate: rounding
    up would let an agent take size that was never offered.
    """
    remaining = int(contracts)
    filled = 0
    gross = 0
    walked = []
    for price, size in levels:
        if remaining <= 0:
            break
        available = int(size)
        if available <= 0:
            continue
        take = min(remaining, available)
        filled += take
        gross += take * price
        walked.append((price, take))
        remaining -= take
    return filled, gross, walked


def _units(levels):
    """Convert an OrderBook ladder from dollars to money units.

    `kalshi/orderbook.py` is ported verbatim from the earlier project and its
    public accessors return **float dollars** (0.47), while everything in this
    simulation is integer ten-thousandths (4700). Converting at this single
    boundary keeps that difference from leaking anywhere else -- a mixed-unit
    price does not raise, it just quietly makes every P&L wrong by 10,000x.
    """
    return [(round(price * money.ONE_DOLLAR), size) for price, size in levels]


def levels_from_book(book, side, depth=10):
    """Price ladder for buying `side`, from a live order book.

    Buying YES means lifting the YES ask ladder. Buying NO means lifting the NO
    ask ladder, which is the YES *bid* ladder mirrored: a YES bid at p is
    somebody offering NO at (1 - p).
    """
    if side == YES:
        return _units(book.yes_ask_levels(depth))
    return [(money.no_price(price), size)
            for price, size in _units(book.yes_bid_levels(depth))]


def levels_from_quote(market, side):
    """Single-level ladder from a top-of-book `ticker` quote.

    Everything past the touch is unknown at this fidelity, so exactly one level
    is offered. An agent asking for more gets a partial fill.
    """
    if side == YES:
        if not market.can_buy_yes:
            return []
        size = market.yes_ask_size
        return [(market.yes_ask, size)] if size else []
    if not market.can_buy_no:
        return []
    size = market.yes_bid_size
    return [(money.no_price(market.yes_bid), size)] if size else []


def buy(levels, contracts, side):
    """Simulate buying `contracts` of `side` against `levels`."""
    requested = int(contracts)
    if requested <= 0:
        return Fill(side, 0, 0, 0, [], requested)
    filled, gross, walked = walk(levels, requested)
    if filled == 0:
        return Fill(side, 0, 0, 0, [], requested)
    # Fees are charged on the average price actually paid, not the touch.
    fee = money.trade_fee(filled, gross // filled)
    return Fill(side, filled, gross, fee, walked, requested)


def sell(levels, contracts, side):
    """Simulate selling `contracts` of `side` back into the book.

    Selling YES hits the YES bid ladder; selling NO hits the NO bid ladder,
    which is the mirrored YES ask side. Proceeds are gross minus fee -- Kalshi
    charges the same fee on the way out, which is why churning is expensive and
    why Kenny's high-turnover personality has something real to run into.
    """
    requested = int(contracts)
    if requested <= 0:
        return Fill(side, 0, 0, 0, [], requested)
    filled, gross, walked = walk(levels, requested)
    if filled == 0:
        return Fill(side, 0, 0, 0, [], requested)
    fee = money.trade_fee(filled, gross // filled)
    fill = Fill(side, filled, gross, fee, walked, requested)
    # On a sale the fee comes out of the proceeds rather than being added on.
    fill.cost = gross - fee
    return fill


def exit_levels_from_book(book, side, depth=10):
    """Price ladder for selling `side` back to the market."""
    if side == YES:
        return _units(book.yes_bid_levels(depth))
    return [(money.no_price(price), size)
            for price, size in _units(book.yes_ask_levels(depth))]


def exit_levels_from_quote(market, side):
    """Single-level exit ladder from a top-of-book quote."""
    if side == YES:
        if not market.can_buy_no:      # selling YES needs a YES bid
            return []
        size = market.yes_bid_size
        return [(market.yes_bid, size)] if size else []
    if not market.can_buy_yes:         # selling NO needs a YES ask
        return []
    size = market.yes_ask_size
    return [(money.no_price(market.yes_ask), size)] if size else []
