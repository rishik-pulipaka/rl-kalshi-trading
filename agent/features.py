"""Turning a market plus memory into a named feature vector.

Every feature has a name, and the policy keeps one weight per name. That is the
whole reason for choosing linear function approximation (PRD 3.4): you can print
the model and read it. `w["mem_winrate"] = +0.31` is a sentence about what the
agent believes. A hidden layer is not.

## A candidate is (market, side), not just a market

Buying YES at 42c and buying NO at 58c are different bets on the same market,
with different prices, different payoffs, and different memory. So features are
computed per candidate and the policy scores each one, rather than computing one
market vector and asking a three-headed model about it.

## What the model predicts

`Q(features)` estimates **return on risk** -- profit per dollar staked -- not raw
P&L. Two reasons:

  - It is scale-free. A $2 profit on a $2 stake and a $2 profit on a $50 stake
    are very different outcomes and must not train the same target.
  - It has a natural zero. Q > 0 means "worth doing", Q <= 0 means "not worth
    doing", so **inaction falls out of the model rather than needing a special
    case** (PRD 2, 5). An agent that finds nothing with positive expected value
    does nothing, which is exactly the human behaviour the PRD wants preserved.

The floor is -1.0 (lose the entire stake); the ceiling depends on entry price --
a 5c contract that wins returns about +19.

## Scaling

Everything lands roughly in [-1, 1] so a single learning rate works across all
weights. Where a quantity is unbounded (volume, time to close) it is compressed
with a log and then scaled. Where it is a probability it is used directly.
Unknown values become 0.0 rather than being dropped, so the vector is always the
same length in the same order.
"""

import math

from sim import money
from sim.fills import YES, NO

# The feature vector, in order. Index i in the weight array is FEATURE_NAMES[i].
# Adding a feature means appending here -- never inserting -- so that saved
# weights from earlier runs stay aligned with the names they were trained under.
FEATURE_NAMES = (
    "bias",
    # --- price shape -------------------------------------------------------
    "price",              # entry price, 0..1
    "price_longshot",     # under 20c
    "price_even",         # 40c..60c
    "price_favourite",    # over 80c
    "spread",             # bid/ask gap, scaled
    # --- market state ------------------------------------------------------
    "log_volume",
    "log_open_interest",
    "log_time_to_close",
    "closing_soon",       # under an hour
    "has_depth",          # full order book available, not just the touch
    # --- memory: the agent consulting itself -------------------------------
    # Kept at three separate granularities rather than blended into one number.
    # Blending was tried and it destroyed the signal: a category whose YES side
    # was -34% EV and whose NO side was +62% averaged out to "roughly
    # break-even", and the agent kept taking the losing side. See
    # `summarize_memory`.
    "mem_seen",             # any memory at all for this market's patterns
    "mem_confidence",       # evidence behind the most-supported belief
    "mem_broad_winrate",    # this category, this side
    "mem_broad_roi",
    "mem_series_winrate",   # this market type, this side
    "mem_series_roi",
    "mem_entity_seen",      # have we backed this subject this way before
    "mem_entity_winrate",
    "mem_entity_roi",
    "mem_ewma",             # recency-weighted return, the personality dial
)

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)

# Memory kinds grouped by how specific they are, most specific first. Within a
# tier the side-aware key is preferred, because "which way to bet on this" is
# exactly what a side-blind key cannot express.
BROAD_KINDS = ("category_side", "category")
SERIES_KINDS = ("series_price", "series_side", "series")
ENTITY_KINDS = ("entity_side", "entity")

# Clip range for ratio features. A single 19x win on a 5c long shot would
# otherwise dominate every weight update that touched it.
ROI_CLIP = 2.0


def _clip(value, limit):
    return max(-limit, min(limit, value))


def _log_scale(value, divisor):
    """log1p compression into roughly 0..1. None and negatives become 0."""
    if not value or value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / divisor)


def entry_price(market, side):
    """What this side costs to buy right now, or None if it cannot be bought."""
    if side == YES:
        return market.yes_ask if market.can_buy_yes else None
    return money.no_price(market.yes_bid) if market.can_buy_no else None


def _pick(by_kind, kinds):
    """The most specific belief available from an ordered preference list.

    `kinds` runs most-specific first, so a side-aware key wins over a side-blind
    one and a price-banded key wins over a bare series key.
    """
    for kind in kinds:
        belief = by_kind.get(kind)
        if belief is not None and belief.win_rate is not None:
            return belief
    return None


def summarize_memory(recalled):
    """Collapse `[(kind, Belief)]` into the numbers the policy uses.

    Memory is reported at three granularities -- broad (category), series
    (market type), and entity (specific subject) -- rather than blended into a
    single confidence-weighted average.

    The blended version was tried first and it silently destroyed the signal. In
    a synthetic world where one category's YES side was -34% EV and its NO side
    was +62%, blending averaged those into "roughly break-even" and the agent
    went on taking the losing side for four hundred simulated days. A linear
    model cannot recover a distinction its inputs have already thrown away.

    Within each tier the *most specific* available belief wins, preferring
    side-aware keys.
    """
    summary = {
        "seen": 0.0, "confidence": 0.0, "ewma": 0.0,
        "broad_winrate": 0.0, "broad_roi": 0.0,
        "series_winrate": 0.0, "series_roi": 0.0,
        "entity_seen": 0.0, "entity_winrate": 0.0, "entity_roi": 0.0,
    }
    if not recalled:
        return summary

    summary["seen"] = 1.0
    by_kind = {}
    best_confidence = 0.0
    for kind, belief in recalled:
        by_kind.setdefault(kind, belief)
        best_confidence = max(best_confidence, belief.confidence())
    summary["confidence"] = best_confidence

    ewma_source = None
    for tier, kinds in (("broad", BROAD_KINDS), ("series", SERIES_KINDS),
                        ("entity", ENTITY_KINDS)):
        belief = _pick(by_kind, kinds)
        if belief is None:
            continue
        # Centred on 0.5 so "no opinion" reads as 0.0 rather than an arbitrary
        # 0.5 that the bias weight would have to cancel out.
        summary[tier + "_winrate"] = belief.win_rate - 0.5
        summary[tier + "_roi"] = _clip(belief.roi or 0.0, ROI_CLIP)
        if tier == "entity":
            summary["entity_seen"] = 1.0
        # The recency signal comes from the most specific tier that has one.
        if belief.ewma_return is not None:
            ewma_source = belief.ewma_return

    if ewma_source is not None:
        summary["ewma"] = _clip(ewma_source, ROI_CLIP)
    return summary


def build(market, side, memory=None, has_depth=False, now=None):
    """Feature vector for one candidate, as a plain list of floats.

    Returns None when the side cannot be bought -- there is nothing to score.
    """
    price = entry_price(market, side)
    if price is None:
        return None

    recalled = memory.recall(market, side=side, price=price, now=now) if memory else []
    mem = summarize_memory(recalled)

    seconds = market.seconds_to_close(now)
    spread = market.spread
    price_frac = price / money.ONE_DOLLAR

    values = {
        "bias": 1.0,
        "price": price_frac,
        "price_longshot": 1.0 if price < 2000 else 0.0,
        "price_even": 1.0 if 4000 <= price < 6000 else 0.0,
        "price_favourite": 1.0 if price >= 8000 else 0.0,
        # A 10c spread is already very wide for a prediction market, so scale
        # against that rather than against the full dollar.
        "spread": min(1.0, (spread or 0) / 1000.0),
        "log_volume": _log_scale(market.volume, 12.0),
        "log_open_interest": _log_scale(market.open_interest, 12.0),
        # ~14 is log(1 week in seconds); a week out reads about 1.0.
        "log_time_to_close": _log_scale(seconds, 14.0),
        "closing_soon": 1.0 if (seconds is not None and seconds < 3600) else 0.0,
        "has_depth": 1.0 if has_depth else 0.0,
        "mem_seen": mem["seen"],
        "mem_confidence": mem["confidence"],
        "mem_broad_winrate": mem["broad_winrate"],
        "mem_broad_roi": mem["broad_roi"],
        "mem_series_winrate": mem["series_winrate"],
        "mem_series_roi": mem["series_roi"],
        "mem_entity_seen": mem["entity_seen"],
        "mem_entity_winrate": mem["entity_winrate"],
        "mem_entity_roi": mem["entity_roi"],
        "mem_ewma": mem["ewma"],
    }
    return [values[name] for name in FEATURE_NAMES]


def named(vector):
    """`{name: value}` for logging and the dashboard."""
    return dict(zip(FEATURE_NAMES, vector))


def describe(vector, top=6):
    """The features that are actually non-zero, largest first. For debugging."""
    pairs = [(name, value) for name, value in zip(FEATURE_NAMES, vector) if value]
    pairs.sort(key=lambda kv: -abs(kv[1]))
    return pairs[:top]
