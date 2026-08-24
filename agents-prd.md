# PRD — Multi-Agent RL Trading Sandbox ("South Park Agents")

## 1. Project overview

Build a sandbox in which independent reinforcement-learning agents learn to
trade Kalshi prediction markets from scratch, using **simulated money against
real live market data**. The agents are observed, not directed — the point of
the project is to watch how different agent "personalities" independently
discover (or fail to discover) profitable behavior over time.

Four agents, named after South Park characters — **Stan, Kyle, Cartman,
Kenny** — each with a distinct, mechanically-implemented personality. They
live entirely separate lives: separate bankrolls, separate memory, separate
learning, no shared experience or knowledge between them. A live dashboard
displays their activity and evolution in real time.

**This is an observational/experimental project, not a profit tool.** No agent
ever touches real money or places a real Kalshi order. Success is measured by
whether the system produces interesting, interpretable learning behavior —
not by whether the agents get rich.

### Explicitly out of scope (do not build)
- Any live order execution against a real Kalshi account
- A fifth "Timmy" agent that inherits the other four's knowledge (parked for
  the distant future — do not build, do not scaffold for it now)
- Human-readable reasoning traces / natural-language explanations of decisions
  (see §6)
- Any paid cloud compute — everything must run locally and free (see §12)

---

## 2. Core concept and constraints

- **Simulated money only.** Each agent starts with **$100 in virtual funds**.
  Kalshi is used *exclusively* as a live market-data source. No real orders,
  ever. The user's real Kalshi account and balance must never be touched or
  modified by this system.
- **Real live prices.** Agents see and act against real Kalshi prices via the
  existing WebSocket/REST integration (already built and working in this
  project — reuse it, do not rebuild).
- **Full action space from day one.** Any action a human can take on Kalshi,
  an agent can take: buy Yes/No on any market, sell/exit a position before
  resolution (buy-low-sell-high), construct or take multi-leg combos/parlays,
  and — importantly — **choose to do nothing at all**. Inaction is a
  first-class action, not an absence of one.
- **Full market freedom.** Agents are NOT restricted to one category. They may
  trade any Kalshi market available through the API — sports, crypto, weather,
  economics, politics, anything. A core research question of this project is
  *which markets each agent gravitates toward on its own*, so any hardcoded
  market restriction would defeat the purpose.
- **Zero starting knowledge.** Agents do not begin knowing that a lower price
  means a bigger payout, or what any market means. They discover mechanics
  through experience.
- **Independent lives.** No shared memory, no shared experience replay, no
  shared model weights, no communication between agents. Four genuinely
  separate learning trajectories.

---

## 3. Learning architecture

### 3.1 Two phases, clean handoff

**Phase A — Historical pretraining (fast, compressed).**
Each agent trains against historical resolved Kalshi market data, running
through many simulated "days" as fast as compute allows. Purpose: let agents
discover basic mechanics (price ↔ payout relationship, resolution, position
sizing) without burning weeks of real calendar time on fundamentals.

**Phase B — Live learning (real-time, one real day = one episode).**
A clean handoff: once pretraining completes, the agent switches to learning
from live markets in real time. **No continued historical training in the
background** — from handoff onward, all learning comes from live experience.

The majority of each agent's eventual competence should come from Phase B.
Phase A exists to get them past "doesn't understand the game at all," not to
pre-solve the problem.

### 3.2 Episode definition
**One episode = one real trading day** (in live mode). Reward is computed at
day's end from that day's resolved outcomes.

### 3.3 Explicit memory (required)
Each agent maintains a **persistent, inspectable memory store** — not just
neural network weights. This memory is a structured, queryable log the agent
consults when making decisions, e.g.:

```
{
  "market_pattern": "NBA player 3PM over",
  "entity": "Stephen Curry",
  "encounters": 7,
  "wins": 5,
  "avg_entry_price": 0.44,
  "net_pnl": +12.40
}
```

Two reasons this is required:
1. It's closer to how a human bettor actually operates (consulting remembered
   specifics rather than retraining a whole brain).
2. **It is the primary window into how each agent is learning**, and is
   directly surfaced on the dashboard. The memory bank is the observable
   artifact of the project.

Memory should accumulate learnings keyed on meaningful entities/patterns
(specific players, teams, market types, price ranges, time-to-resolution
buckets) — the exact schema is an implementation decision, but it must be
readable and dashboard-displayable.

### 3.4 Model complexity guidance
Deliberately keep the approach **understandable, not maximally sophisticated**.
The user intends to read this code, learn from it, and explain it in job
interviews. Prefer well-established, explainable RL approaches (e.g.
tabular/approximate Q-learning or a modest policy-gradient method with a small
network) over large deep-RL architectures with many exotic components. It must
work — but between two approaches that both work, always choose the one that's
easier to explain. Comment the code accordingly, especially around the reward
computation, the action-selection step, and the memory update step.

---

## 4. Agent personalities

Personalities must be **mechanically implemented**, not cosmetic labels — each
maps to concrete parameters that produce genuinely different, observable
behavior. They should also be recognizable as their South Park counterparts.

| Agent | Personality | Mechanical implementation |
|---|---|---|
| **Stan** | The balanced everyman. The control group. | Moderate risk tolerance, moderate exploration rate, no market-type bias, standard bet sizing. Serves as the baseline against which the others' quirks are measured. |
| **Kyle** | Cautious, analytical, rule-following. | Low risk tolerance; reward function additionally penalizes variance/drawdown, so he's pushed toward steady small wins. Low exploration rate (exploits known-good patterns). Smaller average bet sizing. Prefers markets where his memory has high confidence. |
| **Cartman** | Greedy, overconfident, swings big. | High risk tolerance; reward function weights large payouts more heavily, so he's drawn to long-shot/high-multiplier bets and combos. Larger bet sizing. Tends to over-repeat whatever recently worked (higher recency weighting in memory). |
| **Kenny** | Chaotic, unpredictable, high-variance. | Very high exploration rate — constantly tries new/unfamiliar markets rather than exploiting. Erratic bet sizing. Shortest average hold time (in and out fast). Most likely of the four to churn through many small positions. |

All four share the same underlying learning algorithm and action space —
personality is expressed through hyperparameters (exploration rate, risk
weighting in the reward function, bet-sizing distribution, memory recency
weighting), not through different codebases.

---

## 5. Reward and punishment system

Design goal: **realistic and balanced.** Reward genuine skill, punish
recklessness, but do not punish so heavily that agents learn "never bet" is
optimal (a known RL failure mode — an agent that discovers inaction is the
safest path to a non-negative reward will simply stop trading forever).

### Core reward signal
- **Primary: realized P&L per episode (day).** Positive day → positive reward
  proportional to profit; negative day → negative reward proportional to loss.
- **Asymmetry:** losses weighted somewhat more heavily than equivalent gains
  (loss aversion), so agents learn capital preservation matters — but *mildly*,
  to avoid the paralysis failure mode above.

### Additional shaping terms
- **Risk-adjusted penalty:** penalize outsized single-bet exposure relative to
  bankroll, even on wins. A reckless win should not be reinforced as strongly
  as a well-sized win. (Weight varies by personality — heaviest for Kyle,
  lightest for Cartman.)
- **Drawdown penalty:** meaningful penalty for large peak-to-trough bankroll
  declines, encouraging survival over volatility.
- **Overtrading penalty:** small per-trade cost applied regardless of outcome
  (this also naturally models real trading fees), discouraging spam-betting.
- **Inaction handling:** a day with no trades yields a *neutral-to-very-
  slightly-negative* reward — enough that permanent inaction isn't optimal, not
  so much that agents are forced to trade when nothing looks good. **Choosing
  not to trade must remain a legitimate strategy**, since it's a real thing a
  human does.
- **Bankroll floor:** if an agent's virtual bankroll hits zero, that's a strong
  terminal penalty. Decide and document a reset policy (recommendation: reset
  to $100 and log it as a "bankruptcy event" the dashboard displays — a
  bankruptcy count per agent is genuinely interesting data).

### Expect to iterate
Reward shaping is a known-hard problem. Agents will likely find at least one
loophole that technically maximizes reward while doing something obviously
undesirable. Build the reward weights as **easily-tunable config values, not
hardcoded constants**, because they will need adjustment after observing real
behavior.

---

## 6. What is and isn't observable

- **Observable and dashboard-surfaced:** the full explicit memory bank, every
  action taken (or deliberately not taken), bet sizes, entry/exit prices,
  positions held, P&L, all derived analytics.
- **Not built:** natural-language reasoning traces explaining *why* the agent
  chose an action. The decision function itself stays a black box; we learn
  from *what they remember* and *what they do*, not from generated
  explanations. Do not add an explanation layer.

---

## 7. Position and settlement rules

- **Opening a position** immediately deducts the staked capital from the
  agent's available balance (reflected instantly in live balance and that day's
  activity log).
- **P&L is credited on resolution, not on entry.** A position's profit or loss
  counts toward the reward of **the day it resolves**, not the day it was
  opened — because resolution day is when the learnable outcome actually
  becomes known.
- Agents may hold positions across multiple days. Multi-day open positions are
  displayed on the dashboard as unrealized/pending, clearly distinguished from
  realized results.
- **Selling before resolution is fully supported** (buy-low-sell-high). A
  voluntary exit realizes P&L immediately and counts toward that day's reward.

---

## 8. Sleep / downtime mechanic

Each agent has a **personality-driven sleep schedule** — a recurring window
each day during which it takes no actions at all, simulating a human's offline
hours. Schedules are individual, not synchronized:

- **Stan** — regular, conventional hours (a normal ~8-hour block).
- **Kyle** — the most disciplined and consistent schedule of the four; same
  window every day, rarely varies.
- **Cartman** — sleeps the longest; lazy, late riser.
- **Kenny** — erratic and shortest; irregular windows, sometimes skips sleep
  entirely to keep trading. Highest awake-time of the four.

Sleep windows should be visible on the dashboard (both current awake/asleep
status and historical sleep patterns), since "how much time awake" is a
genuine variable in comparing agent performance.

---

## 9. Dashboard requirements

Real-time, polished, visually appealing. Updates live as agents act.

### Views
- **Per-agent view** — deep dive on a single agent
- **All-four comparison view** — side-by-side across every metric

### Required tabs/sections
1. **Overview** — current bankroll, today's P&L, all-time P&L, awake/asleep
   status, open positions, win rate
2. **Activity history** — chronological feed of every action: bets placed,
   combos constructed, positions exited, and explicitly logged decisions to
   *not* act. Includes market, stake, entry price, outcome, P&L.
3. **Memory bank** — direct view into each agent's explicit memory store.
   What has it learned about which entities/patterns, and how has that changed
   over time. This is the centerpiece of the project.
4. **Analytics** — P&L over time, win rate trend, bet-size distribution,
   market-category preference breakdown (which markets does this agent
   gravitate toward), exploration-vs-exploitation ratio over time, average
   hold duration, bankruptcy count
5. **Head-to-head** — direct four-way comparison of all key metrics, so
   personality effectiveness can be evaluated

### Additional metrics worth including
- Learning curve (rolling performance over time — is it actually improving?)
- Calibration: when an agent bets at an implied X% probability, does it win
  ~X% of the time?
- Market diversity over time (does it narrow toward a niche or stay broad?)
- Longest winning/losing streaks
- Average time-to-resolution of chosen markets (do personalities prefer fast
  or slow markets?)

---

## 10. Build order and phasing

**The priority is a genuinely functional, learning system fast — not a complete
one.** Build in this order, **checking in with the user after each numbered
step below before moving to the next** — do not run through all five in one
unattended pass. Catching a wrong turn early (e.g. a memory schema that
doesn't actually support what's needed later) is much cheaper than unwinding
it after later steps are already built on top of it.

1. **Stan only, end to end.** One agent, full action space, full market
   freedom, explicit memory, live data, learning loop, minimal logging.
   Get him actually running and learning before anything else.
2. **Comprehensive data logging from day one.** Even before the dashboard
   exists, log everything the dashboard will eventually display — so when the
   dashboard is built, all historical learning data is already there and can be
   viewed retroactively. **This is explicitly required: do not defer logging
   until the dashboard is built.**
3. **Dashboard**, built against Stan's already-accumulating data.
4. **Add Kyle, Cartman, Kenny** with their personality parameterizations.
5. **Comparison/head-to-head views** across all four.

---

## 11. Operational requirements

- **Kill switch:** a dashboard control to pause/halt all agent activity
  immediately, for use if behavior goes truly off the rails. Otherwise the
  system runs unsupervised.
- **Continuous operation:** designed to run continuously on the user's local
  machine. Must handle WebSocket disconnects/reconnects gracefully (the
  existing implementation has no reconnect logic — this needs to be added, as
  a long-running system will definitely drop connections).
- **Persistence:** all agent state (bankroll, memory, model weights, open
  positions, full history) must survive restarts. A crash or reboot must not
  wipe learning progress.

---

## 12. Cost constraint — strict

**This project must cost $0 in money.** Compute time on the user's own machine
is fine and expected; paid services are not.

- Kalshi market data (REST + WebSocket): free, already integrated
- All libraries must be free/open-source (PyTorch, etc.)
- **No paid cloud GPU/compute services.** If training speed becomes a
  bottleneck, solve it by reducing scope or simplifying the model — never by
  suggesting paid infrastructure.
- No paid data providers. If a market's data isn't available free via Kalshi's
  API, the agents simply don't have it.

---

## 13. Expectation setting (important context for implementation decisions)

Live learning on daily-resolution episodes is **inherently slow** — realistically
a few dozen learning signals per week per agent. "Up and running" and "visibly
competent" are two very different milestones, and the second will take real
calendar time regardless of implementation quality.

Design accordingly: the system should be **interesting to watch from day one**
(rich logging, visible decisions, visible memory forming) even while actual
performance is still poor. Do not optimize for making agents look smart early
— optimize for making their learning process visible and legible from the very
first day.

---

## 14. Code quality requirement

The user will be reading this code to learn from it and explaining it in job
interviews. Therefore:

- Favor clarity over cleverness throughout
- Comment the non-obvious parts thoroughly — especially reward computation,
  action selection, the memory update mechanism, and anything RL-specific
- Avoid exotic techniques where a standard, well-documented approach works
- Structure the project so each major component (data ingestion, agent, memory,
  reward, dashboard) is cleanly separated and individually understandable

## 15. Deployment / runtime operation (how this actually runs day to day)

This section covers how the system runs in practice, not just how it's built —
important since the whole point is for it to run continuously in the
background while the user does other things.

### Dev vs. production terminal
- **During development/testing**: Claude Code may use its own VS Code-integrated
  shell as normal for building, running, and debugging.
- **For actual ongoing operation**: the system must run from a **separate,
  standalone terminal — CMD or PowerShell, not the VS Code integrated
  terminal.** Once the build is working, kill any VS Code-shell instance and
  launch the "real" long-running instance from its own CMD/PowerShell window
  that the user can minimize and leave open indefinitely. This is the terminal
  that stays alive as the actual running system.

### One process, not several to juggle
- The agent simulation loop and the dashboard's local web server should be
  launchable together (a single start command/script, or at minimum
  co-located and started from the same terminal session) so the user does not
  need to separately remember to start multiple things. The dashboard should
  be reachable at a local address (e.g. `localhost:PORT`) in a browser while
  that terminal window runs in the background.
- The browser tab/dashboard does NOT need to stay open for agents to keep
  learning — closing the tab does not stop the simulation. The dashboard is a
  read/view layer on top of data the agents are continuously writing; reopening
  it later just shows what happened while it was closed.

### What "running" actually depends on
- The simulation keeps running as long as the terminal process is alive and
  the computer is on/awake — VS Code itself does not need to be open at all.
- Closing the specific CMD/PowerShell window running the process WILL stop it
  (unless explicitly configured otherwise) — this is the one thing the user
  needs to avoid doing.
- Computer sleep, restart, or shutdown will stop the process. This is exactly
  why persistence (state saved to disk, see below) matters — on relaunch, the
  system should resume from saved state, not restart from zero.
- Getting a fully-unattended, survives-a-reboot background service is a nice
  future improvement but NOT required for v1. Start with "one terminal window,
  left open, don't close it" as the operating model. Don't over-engineer
  always-on service infrastructure before the core system is confirmed working.

### Persistence location and size
- All saved state (agent bankrolls, explicit memory banks, model
  weights/parameters, full activity history, logs) must be written to a
  **single, clearly-named, easy-to-find local folder** (e.g. a top-level
  `/data` or `/agent_state` folder within the project directory) — not
  scattered across system temp directories or hidden locations. The user wants
  to be able to browse this folder directly.
- **Be mindful of disk space.** The user has limited free disk space available.
  Favor efficient storage: compact log formats over verbose ones, reasonable
  log rotation/retention rather than unbounded growth, and avoid storing
  redundant copies of the same data. If any component (e.g. verbose historical
  pretraining data, or full raw WebSocket message logs) is likely to grow
  large, flag this explicitly to the user before implementing it, so a
  retention/pruning strategy can be agreed on rather than silently consuming
  disk space over time.
