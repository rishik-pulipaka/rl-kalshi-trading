"""The daily shaped reward (PRD 5).

## What this is, and what it is not

There are two signals in this project and they do different jobs:

  **Per-trade** (`trade_target`, consumed by `agent/policy.py`) -- each resolved
  position trains the Q weights. This is the ONLY thing that changes the model,
  and it fires dozens of times a week rather than seven.

  **Per-day** (`daily_reward`) -- one shaped number per agent per episode. It is
  the scoreboard: what the dashboard plots and what a human reads to judge an
  agent's conduct over a day.

The split matters, and getting it wrong is easy. Terms that describe *one
trade's* quality -- was it well sized, was the loss painful -- belong in
`trade_target`, because otherwise they never reach the learner at all and PRD 5
becomes decoration. Terms that describe *portfolio conduct across a day* --
drawdown, overtrading, sitting out entirely -- belong in `daily_reward`, because
blaming a single position for them would attribute behaviour it did not cause.

## Every weight is config, none are constants

PRD 5 says outright that reward shaping will need adjustment after watching real
behaviour, and predicts the agents will find at least one loophole. So
`RewardWeights` is a plain dataclass loaded from `config/agents.yaml`, and
nothing in the maths below is hardcoded.

## The failure mode this is designed around

The known trap is an agent that discovers inaction is the safest route to a
non-negative reward and stops trading forever. Three choices guard against it:

  - Loss aversion is **mild** (~1.15x, not 2x). Enough to teach that capital
    preservation matters, not enough to make every bet look bad.
  - A no-trade day scores slightly negative, not zero. Doing nothing is a
    legitimate strategy (a real thing a human does) but it should not be a
    comfortable equilibrium.
  - Penalties scale with *fractions of bankroll*, not absolute dollars, so they
    cannot swamp the P&L term as the bankroll grows.
"""

from dataclasses import dataclass, asdict

from sim import money


@dataclass
class RewardWeights:
    """Every tunable in the reward function. Loaded from config, never inlined."""

    # --- core P&L ---------------------------------------------------------
    # P&L is divided by the starting bankroll, so a $10 day on $100 scores 0.10
    # and the number stays comparable as bankrolls diverge between agents.
    pnl_scale: float = 1.0
    # Losses count slightly more than equivalent gains. Mild on purpose.
    loss_aversion: float = 1.15

    # --- risk shaping -----------------------------------------------------
    # Penalty on the largest single position as a fraction of bankroll, applied
    # even on a winning day: a reckless win should not reinforce as strongly as
    # a well-sized one (PRD 5). Heaviest for Kyle, lightest for Cartman.
    exposure_penalty: float = 0.30
    # Exposure below this fraction of bankroll is free. Above it, penalised.
    exposure_free_fraction: float = 0.20

    # Peak-to-trough decline, encouraging survival over volatility.
    drawdown_penalty: float = 0.50

    # --- friction ---------------------------------------------------------
    # Charged per trade regardless of outcome, on top of Kalshi's real fees
    # (which are already modelled in sim/fills.py). This is the discouragement
    # of spam-betting rather than a second attempt at modelling costs.
    per_trade_penalty: float = 0.002

    # --- inaction ---------------------------------------------------------
    # A day with no trades. Slightly negative: not punished into forced
    # trading, not rewarded into permanent paralysis.
    inaction_reward: float = -0.01
    # Inaction while genuinely holding positions is not idleness -- the agent
    # has capital at work and is waiting, which is a real strategy.
    inaction_reward_holding: float = 0.0

    # --- terminal ---------------------------------------------------------
    bankruptcy_penalty: float = -5.0

    @classmethod
    def from_dict(cls, data):
        """Build from config, ignoring unknown keys so config can carry extras."""
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in (data or {})}
        return cls(**known)

    def to_dict(self):
        return asdict(self)


def daily_reward(weights, *, realized_pnl, trades, drawdown,
                 largest_exposure, bankroll, open_positions=0,
                 went_bankrupt=False, starting_bankroll=None):
    """Score one agent's day. Returns `(reward, breakdown)`.

    The breakdown is returned alongside the number because a single scalar is
    useless for diagnosing a reward loophole. PRD 5 expects at least one, and
    finding it means being able to see which term the agent is farming.

    All money arguments are in the project's money units (see `sim/money.py`).
    """
    base = starting_bankroll or money.STARTING_BANKROLL
    parts = {}

    # --- P&L, normalized by the starting stake ---------------------------
    pnl_fraction = realized_pnl / base
    if pnl_fraction < 0:
        parts["pnl"] = weights.pnl_scale * weights.loss_aversion * pnl_fraction
    else:
        parts["pnl"] = weights.pnl_scale * pnl_fraction

    # --- inaction ---------------------------------------------------------
    # Handled before the friction terms: a day with no trades has no exposure
    # and no per-trade cost, so those would all be zero anyway.
    if trades == 0:
        parts["inaction"] = (weights.inaction_reward_holding if open_positions
                             else weights.inaction_reward)
        total = sum(parts.values())
        if went_bankrupt:
            parts["bankruptcy"] = weights.bankruptcy_penalty
            total += weights.bankruptcy_penalty
        return total, parts

    # --- oversized single bets -------------------------------------------
    # Measured against bankroll + what is already committed, so an agent cannot
    # dodge the penalty by having already spent everything.
    capital = max(bankroll + largest_exposure, 1)
    exposure_fraction = largest_exposure / capital
    excess = max(0.0, exposure_fraction - weights.exposure_free_fraction)
    parts["exposure"] = -weights.exposure_penalty * excess

    # --- drawdown ---------------------------------------------------------
    parts["drawdown"] = -weights.drawdown_penalty * max(0.0, drawdown)

    # --- overtrading ------------------------------------------------------
    parts["overtrading"] = -weights.per_trade_penalty * trades

    total = sum(parts.values())

    if went_bankrupt:
        parts["bankruptcy"] = weights.bankruptcy_penalty
        total += weights.bankruptcy_penalty

    return total, parts


def trade_target(position, weights=None):
    """The per-trade learning target: risk-adjusted return on risk.

    This is what `LinearQ.update` trains against, and it is the ONLY signal that
    changes the model. That matters, because it means anything PRD 5 wants the
    agents to actually learn has to be expressed here -- not only in the daily
    reward, which is a scoreboard.

    Two shaping terms are folded in, both from PRD 5:

      **Loss aversion.** A losing trade is weighted slightly more heavily than
      an equivalent win, so capital preservation carries real weight. Mild on
      purpose -- harsh loss aversion is how an agent learns that never betting
      is optimal.

      **Exposure.** A win on a bet that risked 40% of the bankroll trains a
      smaller target than the same-percentage win on a 5% bet. PRD 5 asks for
      exactly this ("a reckless win should not be reinforced as strongly as a
      well-sized win"), and without it the shaping has no path to the learner:
      return on risk is scale-free, so a reckless win and a careful one with the
      same ratio are otherwise literally the same training example.

    Note what this does to the meaning of Q. It is no longer a pure prediction
    of expected return -- it is a prediction of *this agent's utility* for the
    trade, which is what the agent should be maximizing. Kyle's heavy exposure
    penalty and Cartman's light one make their Q functions answer genuinely
    different questions about the same market, which is the point.

    Returns None for a position that has not resolved, or one whose stake was
    zero (nothing was risked, so there is nothing to learn).
    """
    if position.realized_pnl is None or not position.cost:
        return None

    roi = position.realized_pnl / position.cost
    if weights is None:
        return roi

    if roi < 0:
        roi *= weights.loss_aversion

    # Penalise the slice of the bankroll that went beyond the "free" allowance.
    fraction = getattr(position, "stake_fraction", 0.0) or 0.0
    excess = max(0.0, fraction - weights.exposure_free_fraction)
    return roi - weights.exposure_penalty * excess


def summarize_day(agent, day, reward, parts, portfolio_stats):
    """One row for the daily log and the dashboard's learning curve."""
    return {
        "agent": agent,
        "day": day,
        "reward": reward,
        "reward_parts": parts,
        "realized_pnl": portfolio_stats.get("all_time_realized"),
        "bankroll": portfolio_stats.get("bankroll"),
        "equity": portfolio_stats.get("equity"),
        "drawdown": portfolio_stats.get("drawdown"),
        "open_positions": portfolio_stats.get("open_positions"),
        "win_rate": portfolio_stats.get("win_rate"),
        "bankruptcies": portfolio_stats.get("bankruptcies"),
    }
