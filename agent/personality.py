"""Loading the four personalities from config.

PRD 4 requires personality to be **mechanically implemented, not cosmetic**. The
way that claim is kept honest here is structural: this module turns
`config/agents.yaml` into parameters, and nothing anywhere in the project
branches on an agent's name. All four run the same policy, the same reward
function, the same decision loop. Stan and Cartman differ only in numbers.

That also makes the claim falsifiable, which matters more. If Cartman ends up
behaving like Stan, the config is wrong -- there is no hidden code path where
his greed might be lurking.

## What each trait actually is

  Kyle "prefers patterns he trusts"     -> low epsilon, high act_threshold
  Cartman "over-repeats what worked"    -> high memory ewma_alpha
  Cartman "swings big"                  -> high sizing fractions, low exposure penalty
  Kenny "constantly tries new markets"  -> very high epsilon
  Kenny "in and out fast"               -> short exit_check_interval
  Kyle "steady small wins"              -> heavy drawdown and exposure penalties

Bet sizing is sampled per trade from `[min_fraction, max_fraction]` of bankroll.
A wide band is erratic sizing (Kenny); a narrow low band is disciplined sizing
(Kyle). The distribution *is* the trait.
"""

import os
import copy
import random
from dataclasses import dataclass, field

import yaml

from sim import money
from .reward import RewardWeights
from .sleep import SleepSchedule

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "agents.yaml")


def _deep_merge(base, override):
    """Override wins, but only for keys it actually sets.

    So an agent's config states just its deviations from `defaults` and the rest
    is inherited -- which keeps each personality block readable as a list of
    what makes that agent different.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class Sizing:
    """How large a bet this agent makes, as a fraction of its bankroll."""

    min_fraction: float = 0.02
    max_fraction: float = 0.10
    absolute_max_fraction: float = 0.35

    def sample(self, rng, bankroll):
        """Stake for one trade, in money units.

        Sampled rather than fixed because erratic sizing is itself a
        personality trait (Kenny). The absolute cap is a backstop against a
        config typo emptying an agent in a single trade.
        """
        low = min(self.min_fraction, self.max_fraction)
        high = max(self.min_fraction, self.max_fraction)
        fraction = rng.uniform(low, high)
        fraction = min(fraction, self.absolute_max_fraction)
        return int(bankroll * fraction)

    @classmethod
    def from_dict(cls, data):
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in (data or {})}
        return cls(**known)


@dataclass
class Trading:
    """Pace and limits. How often the agent looks, and how much it holds."""

    decision_interval_seconds: int = 60
    candidates_per_decision: int = 40
    max_open_positions: int = 25
    min_seconds_to_close: int = 120
    exit_check_interval_seconds: int = 300
    # Shortest time a position is held before an exit is even considered.
    # This is PRD 4's "average hold time" trait made mechanical, and it also
    # closes a churn trap -- see `Agent.check_exits`.
    min_hold_seconds: int = 3600
    # How far below the entry threshold the score must fall before exiting.
    # Without this gap, anything barely worth entering is instantly worth
    # exiting, and the agent round-trips the spread for nothing.
    exit_margin: float = 0.15
    # Chance of committing to several legs at once instead of a single bet.
    # PRD 2 wants agents able to "construct" multi-leg combos; this is how
    # often they choose to. Cartman is drawn to them; Kyle is not.
    combo_appetite: float = 0.10
    # Legs in a constructed basket.
    combo_legs: int = 3

    @classmethod
    def from_dict(cls, data):
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in (data or {})}
        return cls(**known)


@dataclass
class PolicyParams:
    learning_rate: float = 0.01
    epsilon: float = 0.15
    optimism: float = 0.15
    act_threshold: float = 0.0

    @classmethod
    def from_dict(cls, data):
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in (data or {})}
        return cls(**known)


@dataclass
class Personality:
    """Everything that makes one agent different from the other three."""

    name: str
    display_name: str = ""
    blurb: str = ""
    starting_bankroll: int = money.STARTING_BANKROLL
    policy: PolicyParams = field(default_factory=PolicyParams)
    sizing: Sizing = field(default_factory=Sizing)
    trading: Trading = field(default_factory=Trading)
    reward: RewardWeights = field(default_factory=RewardWeights)
    sleep: SleepSchedule = field(default_factory=SleepSchedule)
    memory_ewma_alpha: float = 0.20

    def rng(self, seed=None):
        """This agent's own random source. Never shared (PRD 2)."""
        return random.Random(seed if seed is not None else hash(self.name) & 0xFFFF)

    def summary(self):
        """Compact view for the dashboard's per-agent header."""
        return {
            "name": self.name,
            "display_name": self.display_name or self.name.title(),
            "blurb": self.blurb,
            "epsilon": self.policy.epsilon,
            "act_threshold": self.policy.act_threshold,
            "optimism": self.policy.optimism,
            "memory_ewma_alpha": self.memory_ewma_alpha,
            "bet_size_range": [self.sizing.min_fraction, self.sizing.max_fraction],
            "max_open_positions": self.trading.max_open_positions,
            "sleep": self.sleep.to_dict(),
            "reward": self.reward.to_dict(),
        }


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build(name, config):
    """One personality, with `defaults` merged under its own overrides."""
    merged = _deep_merge(config.get("defaults", {}),
                         (config.get("agents", {}) or {}).get(name, {}))
    return Personality(
        name=name,
        display_name=merged.get("display_name", name.title()),
        blurb=merged.get("blurb", ""),
        # Config states dollars for readability; the system runs on money units.
        starting_bankroll=money.dollars(merged.get("starting_bankroll", 100.0)),
        policy=PolicyParams.from_dict(merged.get("policy")),
        sizing=Sizing.from_dict(merged.get("sizing")),
        trading=Trading.from_dict(merged.get("trading")),
        reward=RewardWeights.from_dict(merged.get("reward")),
        sleep=SleepSchedule.from_dict(merged.get("sleep")),
        memory_ewma_alpha=(merged.get("memory") or {}).get("ewma_alpha", 0.20),
    )


def load_all(path=CONFIG_PATH):
    """Every configured agent, in file order. `{name: Personality}`."""
    config = load_config(path)
    return {name: build(name, config) for name in (config.get("agents") or {})}


def load_one(name, path=CONFIG_PATH):
    config = load_config(path)
    if name not in (config.get("agents") or {}):
        raise KeyError(f"no agent named {name!r} in {path}")
    return build(name, config)
