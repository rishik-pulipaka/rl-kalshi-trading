"""Linear Q-learning over named features, with epsilon-greedy exploration.

PRD 3.4 asks for the most explainable approach that works, not the most
sophisticated. This is online linear regression trained on realized outcomes:

    prediction = w . f                      one dot product
    error      = target - prediction
    w         += lr * error * f             the LMS / delta rule

That is the whole learning algorithm. Every weight has a name, so the trained
model can be printed and read as sentences about what the agent believes, and
`explain()` breaks any single decision into per-feature contributions. There is
no hidden state to interpret and nothing to visualize with a saliency map.

## The cold-start problem, and optimistic initialization

Q estimates return on risk, so "act only when Q > 0" makes inaction fall out of
the model for free (see `agent/features.py`). But with weights starting at zero,
Q is 0 for everything, the agent never acts, never observes an outcome, never
learns, and sits still forever. The reward function is not the problem here --
the *initialization* is.

The fix is textbook optimism in the face of uncertainty: start the bias weight
slightly positive. The agent begins believing that anything is mildly worth
trying, acts on that, and gets argued out of it by real losses. Optimism decays
on its own as evidence arrives -- no exploration schedule to hand-tune, and no
special case telling the agent to trade when it does not want to.

## Exploration

Epsilon-greedy over candidates. With probability epsilon the agent takes a
random candidate instead of its best one, which is how it ends up in markets its
memory says nothing about. Kenny's very high epsilon and Kyle's very low one
(PRD 4) are this single number, not different code.

Exploration is recorded per decision because PRD 9 wants the
exploration-vs-exploitation ratio plotted over time.

## What it trains on

One settled or exited position is one training example: the feature vector
captured at entry, and the realized return on risk as the target. This is a
contextual bandit rather than full temporal-difference learning -- there is no
bootstrapping from a successor state, because a resolved market has no successor.
Saying so plainly is more honest than dressing it up as deep RL, and it is the
right shape for the problem.
"""

import os
import json
import random
import logging

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES, FEATURE_INDEX

log = logging.getLogger(__name__)

# How optimistic a fresh agent is: it expects a 15% return on anything until
# evidence says otherwise. Large enough to get it trading on day one, small
# enough that a handful of real losses overrides it.
DEFAULT_OPTIMISM = 0.15

# Outlier guard on the TARGET, not on the error.
#
# This distinction was measured, and it matters. Clipping the *error* biases the
# estimator badly: a 5c winner has a true return of +19, so its error clips to
# +1 while a losing trade's error of -1 passes through untouched. Wins get
# throttled and losses do not, and the model drifts pessimistic. In a synthetic
# world where fair bets have exactly zero expected value, error clipping priced
# them at -0.03 and priced long shots at -0.98 against a true -0.45 -- more than
# twice as negative as reality. That would have made every agent look like Kyle
# and left Cartman's long-shot preference with nothing to find.
#
# Clipping the target instead is unbiased over the normal range. The bound is
# deliberately generous: a 5c contract that wins returns +19 and passes through
# untouched. It only bites on sub-5c lottery tickets, where a single 999x return
# (Kalshi's ladder goes down to 0.0010) really would swamp every other trade the
# agent has ever made.
TARGET_CLIP = 20.0

# Scores within this of the best count as tied, and one of them is chosen at
# random. Two candidates whose predicted return differs by less than 1e-9 are
# the same bet as far as the model is concerned, and picking the earlier one
# would be a preference for list order rather than for the market.
TIE_TOLERANCE = 1e-9


class Decision:
    """What the agent chose, and enough context to explain it later."""

    __slots__ = ("candidate", "q", "explored", "considered", "best_q",
                 "skipped_reason")

    def __init__(self, candidate=None, q=0.0, explored=False, considered=0,
                 best_q=None, skipped_reason=None):
        self.candidate = candidate
        self.q = q
        self.explored = explored
        self.considered = considered
        self.best_q = best_q
        self.skipped_reason = skipped_reason

    @property
    def acted(self):
        return self.candidate is not None

    def __repr__(self):
        if not self.acted:
            return f"<Decision skip ({self.skipped_reason}) of {self.considered}>"
        mode = "explore" if self.explored else "exploit"
        return f"<Decision {mode} q={self.q:+.3f} of {self.considered}>"


class LinearQ:
    """One agent's model. Not shared with any other agent (PRD 2)."""

    def __init__(self, learning_rate=0.01, epsilon=0.15,
                 optimism=DEFAULT_OPTIMISM, act_threshold=0.0, seed=None):
        self.weights = np.zeros(N_FEATURES, dtype=np.float64)
        # See the module docstring: this is what gets a fresh agent moving.
        self.weights[FEATURE_INDEX["bias"]] = optimism

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.act_threshold = act_threshold

        self.random = random.Random(seed)
        self.updates = 0
        self.decisions = 0
        self.explorations = 0
        self.abs_error_sum = 0.0

    # ---------- scoring ----------

    def q(self, vector):
        """Predicted return on risk for taking this candidate."""
        return float(np.dot(self.weights, vector))

    def select(self, candidates):
        """Choose one candidate to act on, or none.

        `candidates` are objects with a `.features` list. Returns a `Decision`.

        Skipping is a real outcome here, not a failure: if nothing scores above
        the threshold the agent deliberately does nothing, which PRD 2 requires
        to remain a legitimate strategy.
        """
        self.decisions += 1
        if not candidates:
            return Decision(considered=0, skipped_reason="no_candidates")

        scores = [self.q(c.features) for c in candidates]
        best_q = max(scores)
        # Break ties at random rather than taking the first maximum.
        #
        # This matters far more than it looks. A fresh agent has all-zero
        # weights, so EVERY candidate scores exactly the optimism prior and the
        # whole list is one big tie. `np.argmax` returns the first index, so the
        # agent deterministically bought whatever the sampler happened to list
        # first -- which is always the same half of `_sample_candidates`. Its
        # opening choices were an artifact of list order, not of the model, and
        # it stayed that way until enough weight updates arrived to break the
        # ties on their own. Seen live: two agents kept buying long-dated
        # markets after sampling was changed to favour fast-resolving ones,
        # because the reweighted half was never reached.
        tied = [i for i, s in enumerate(scores) if s >= best_q - TIE_TOLERANCE]
        best_index = (tied[0] if len(tied) == 1
                      else self.random.choice(tied))

        # Explore: take something other than the best. This is how an agent ends
        # up in markets its memory says nothing about.
        if self.random.random() < self.epsilon:
            index = self.random.randrange(len(candidates))
            self.explorations += 1
            return Decision(candidate=candidates[index], q=scores[index],
                            explored=True, considered=len(candidates),
                            best_q=best_q)

        if best_q <= self.act_threshold:
            return Decision(considered=len(candidates), best_q=best_q,
                            skipped_reason="nothing_worth_doing")

        return Decision(candidate=candidates[best_index], q=best_q,
                        explored=False, considered=len(candidates), best_q=best_q)

    # ---------- learning ----------

    def update(self, vector, target):
        """One gradient step toward the realized return. Returns the error.

        `target` is realized return on risk: -1.0 for a total loss, +0.5 for a
        50% gain. This is the only place the model changes.
        """
        vector = np.asarray(vector, dtype=np.float64)
        # Bound the target, never the error -- see TARGET_CLIP.
        target = max(-TARGET_CLIP, min(TARGET_CLIP, float(target)))
        prediction = float(np.dot(self.weights, vector))
        error = target - prediction

        self.weights += self.learning_rate * error * vector

        self.updates += 1
        self.abs_error_sum += abs(error)
        return error

    @property
    def mean_abs_error(self):
        """Rolling accuracy. Falling over time is what "learning" looks like."""
        return (self.abs_error_sum / self.updates) if self.updates else None

    @property
    def exploration_rate(self):
        """Realized fraction of decisions that explored (PRD 9 wants this plotted)."""
        return (self.explorations / self.decisions) if self.decisions else 0.0

    # ---------- explanation ----------

    def explain(self, vector, top=8):
        """Per-feature contributions to one score, largest magnitude first.

        This is the payoff for choosing a linear model: any decision decomposes
        exactly into `weight * value` per named feature, and the parts sum to
        the score with nothing left over.
        """
        contributions = [(name, float(self.weights[i] * vector[i]),
                          float(self.weights[i]), float(vector[i]))
                         for i, name in enumerate(FEATURE_NAMES) if vector[i]]
        contributions.sort(key=lambda row: -abs(row[1]))
        return contributions[:top]

    def named_weights(self):
        """`{feature_name: weight}` -- the model, readable."""
        return {name: float(self.weights[i]) for i, name in enumerate(FEATURE_NAMES)}

    def top_weights(self, n=10):
        items = sorted(self.named_weights().items(), key=lambda kv: -abs(kv[1]))
        return items[:n]

    # ---------- persistence ----------

    def save(self, path):
        """Weights plus the names they belong to (PRD 11: survive restarts).

        Names are stored alongside the numbers so a future feature-set change is
        detected on load instead of silently misaligning every weight -- which
        would not raise, it would just make the agent behave randomly.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(path, weights=self.weights, names=np.array(FEATURE_NAMES),
                 meta=np.array([json.dumps({
                     "learning_rate": self.learning_rate,
                     "epsilon": self.epsilon,
                     "act_threshold": self.act_threshold,
                     "updates": self.updates,
                     "decisions": self.decisions,
                     "explorations": self.explorations,
                     "abs_error_sum": self.abs_error_sum,
                 })]))

    def load(self, path):
        """Restore weights. Returns False if there was nothing to restore."""
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=False)
        names = [str(n) for n in data["names"]]
        if names != list(FEATURE_NAMES):
            log.warning(
                "feature set changed since these weights were saved "
                "(%d saved vs %d current); starting fresh rather than "
                "misaligning them", len(names), N_FEATURES)
            return False
        self.weights = data["weights"].astype(np.float64)
        meta = json.loads(str(data["meta"][0]))
        self.learning_rate = meta.get("learning_rate", self.learning_rate)
        self.epsilon = meta.get("epsilon", self.epsilon)
        self.act_threshold = meta.get("act_threshold", self.act_threshold)
        self.updates = meta.get("updates", 0)
        self.decisions = meta.get("decisions", 0)
        self.explorations = meta.get("explorations", 0)
        self.abs_error_sum = meta.get("abs_error_sum", 0.0)
        return True

    def stats(self):
        return {
            "updates": self.updates,
            "decisions": self.decisions,
            "explorations": self.explorations,
            "exploration_rate": self.exploration_rate,
            "mean_abs_error": self.mean_abs_error,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "top_weights": self.top_weights(8),
        }
