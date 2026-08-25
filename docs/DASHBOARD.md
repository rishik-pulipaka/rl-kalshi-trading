# Reading the Dashboard

A guide to what every number means, which ones matter, and how to tell a bug
from an agent that is simply bad at trading yet.

The dashboard is a **read layer**. Closing the tab stops nothing; the agents
keep writing to `data/` whether or not anyone is looking. Reopening it later
shows everything that happened while it was closed.

```
python run.py --agents all --pretrain KXBTC15M,KXHIGHNY
```

Then open **http://localhost:8000**.

---

## The header strip

Always visible. This is system health, not agent performance.

| Reads | Means | Worry when |
|---|---|---|
| `stream up · 1,431,682 msgs · 0 reconnects` | The Kalshi firehose is connected and delivering | It says **stream DOWN**, or reconnects climbs steadily (a few a day is normal) |
| `104,325 markets · 101,701 tradeable` | Size of the live universe | Tradeable drops near zero — that means quotes stopped arriving |
| `168 MB · 192 GB free` | Process memory and disk headroom | Memory climbing past ~400 MB, or disk under a few GB |

**Pause all agents** is the kill switch (PRD §11). It stops agents *trading*
immediately; the stream keeps running so the universe stays warm and the
dashboard stays live. Nothing is lost — press it again to resume.

The agent buttons on the left switch which agent every tab is showing. Each is
colour-coded and keeps that colour in the head-to-head view.

---

## Overview

Eight cards, per agent.

**Equity** — bankroll plus the value of open positions. This is the real
"how is it doing" number. **Bankroll** alone is just uncommitted cash: an agent
with $12 bankroll and $80 at risk has not lost $88.

**All-time P&L** — realized only. Money actually won or lost on positions that
resolved or were sold. Open positions are *not* in here.

**Today's P&L** — realized today. Per §7, a position resolving today counts
today even if it was opened last week. That is deliberate: the day the outcome
becomes knowable is the day the lesson lands.

**Win rate** — over resolved positions only. Reads `no data yet` until
something settles, which is correct — no data is not a 0% win rate.

> **The single most important trap on this dashboard:** win rate and profit are
> nearly unrelated. An agent buying 90¢ favourites can win 85% of the time and
> lose money steadily. An agent buying 10¢ long shots can win 15% of the time
> and get rich. Always read win rate next to P&L, never alone.

**Awake / Asleep** — each agent has its own offline window (§8). Cartman sleeps
11 hours and rises late; Kenny sleeps 5 and sometimes skips entirely. Awake
hours matter for fair comparison: more time awake means more chances to trade.

**Learning** — weight updates, realized exploration rate, and mean absolute
error. One update = one resolved position. Error falling over weeks is what
learning looks like.

**Memory** — how many distinct patterns this agent has formed beliefs about.

**Bankruptcies** — how many times it hit the $1.00 floor and was reset to $100.
Cartman will get here first.

> Going broke is a lesson, not just a counter. Every position still open at the
> moment of ruin is written off as a total loss **and trained on**, carrying an
> extra `ruin_penalty` on top. How heavily an agent is marked by a blow-up is
> part of its personality: Kyle 2.00, Stan 1.00, Cartman 0.40 — shrugging it off
> is the point of Cartman.

Below the cards: **open positions**, marked to market, with unrealized P&L
clearly separated from realized (§7). `Size` is the fraction of bankroll that
position represented when opened — this is what the risk penalty reads.

---

## Activity

Two tables, and the second one is the interesting one.

**Positions** — every trade. `Q` is what the model predicted (expected return
per dollar risked) at the moment of entry. `Mode` is `exploit` (model's best
pick) or `explore` (deliberately random).

Comparing `Q` against actual P&L over many rows tells you whether the model's
predictions mean anything yet. Early on they will not.

> **Every `Q` reading exactly the same number early on is correct.** A brand-new
> agent has all-zero weights, so every market scores exactly its `optimism`
> setting — 0.150 for Stan, 0.200 for Kenny. A column of identical values is
> what "has learned nothing yet" looks like, not a stuck calculation. Watch for
> the day they start to differ from each other; that is the model beginning to
> distinguish between markets.

**Every decision** — including the ones where the agent did nothing. §9 requires
this and it is genuinely informative:

| Reason | What happened |
|---|---|
| `nothing_worth_doing` | Scored candidates, none beat its threshold. **A real decision, not a failure.** |
| `no_liquidity` | Wanted to trade but nothing was available at a fillable size |
| `no_candidates` | Nothing tradeable in the sample this tick |
| `insufficient_funds` | Wanted more than it could afford |
| `cannot_afford_one_contract` | Bankroll too small for even one contract |

`Looked at` is how many (market, side) pairs it scored. Roughly 2× the sampled
markets, since each market offers a YES and a NO.

A wall of `nothing_worth_doing` from Kyle is Kyle working correctly. A wall of
`no_liquidity` from everyone means the depth tier is not filling — check that
`books=N/N` in the terminal has both numbers equal and non-zero.

---

## Memory bank

The centerpiece (§3.3). This is what the agent has actually learned, in
readable form, and it feeds directly into its decisions — it is not a display
bolted onto a black box.

Filter buttons pick the granularity:

| Kind | Example | Reads as |
|---|---|---|
| `category+side` | `Sports (no)` | "Betting NO on sports markets" |
| `series+side` | `KXNBA (yes)` | "Betting YES on NBA markets" |
| `series+price+side` | `KXNBA @ 20-40c (yes)` | "...in the 20–40¢ band" |
| `entity+side` | `Stephen Curry (yes)` | "...on this specific subject" |
| `time to close` | `under_1h` | "Markets resolving within the hour" |
| `price band` | `0-5c (no)` | "Very cheap contracts" |

**Every key carries the side, and that matters.** Side-blind memory averages a
terrible YES with an excellent NO into something meaningless — which is exactly
the bug that once left an agent taking the losing side of a market for 400
simulated days. "Which way to bet on this" is the most basic thing a bettor
learns, and it is unlearnable without the side in the key.

Columns:

- **Seen** — total encounters
- **W/L** — resolved wins and losses (voids excluded; a cancelled market taught
  nothing)
- **Win rate** — over resolved only
- **Confidence** — a bar, `n/(n+10)`. This is what separates *"60% over 3 tries"*
  from *"60% over 200 tries"*. Kyle's caution is literally a preference over
  this number. **Ignore any row whose bar is nearly empty.**
- **Avg entry** — where it typically buys this pattern
- **Net P&L** — money made or lost on it
- **ROI** — profit per dollar staked. **The honest verdict on a pattern.**
- **Recency** — recency-weighted return. Cartman's reacts fast (α=0.5), Kyle's
  slowly (α=0.1). That difference *is* Cartman's "chases what recently worked".

**How to actually read this:** sort by *most seen*, ignore low-confidence rows,
then look for rows where win rate and ROI disagree. Those are where something
interesting is happening.

Click through to see how a belief formed over time rather than just its current
value.

---

## Analytics

**Equity over time** — one point per trading day. The learning curve. Over
weeks, ask: is the trend improving, or just noisy?

**Daily reward** — the shaped score from §5: P&L with mild loss aversion, minus
drawdown, overtrading, and inaction penalties. Green bars are good days.

> Note this is a **scoreboard**, not the training signal. The model trains on
> per-trade outcomes. Loss aversion and the exposure penalty are folded into
> *that* target so §5's risk shaping actually reaches the learner; drawdown and
> overtrading stay here, because blaming one position for portfolio-wide
> behaviour would be wrong.

**Calibration** — the sharpest single measure of whether an agent understands
anything. When it bets at an implied 30%, does it win ~30%? Dots on the dashed
diagonal mean it reads prices correctly *even if it is not profitable*. Dot size
is sample count; ignore small ones.

An agent can be well calibrated and still lose (paying the spread and fees), or
badly calibrated and win by luck. Calibration converges long before profit does,
which makes it the first real sign of competence.

**Exploration vs exploitation** — fraction of acted-on decisions that were
random. Should hover near each agent's `epsilon`: Kenny ~0.55, Kyle ~0.05.

**Win rate trend** — win rate over resolved positions, day by day. Read it
against equity, never alone: an agent buying favourites can walk this line
upward while steadily losing money.

**Market diversity over time** — the share of each day's positions that went
into that agent's single favourite category. Rising means it is narrowing
toward a niche; flat and low means it is staying broad. This is the "does it
specialize?" question from §9, and it deliberately measures *share* rather than
a category count — an agent can keep touching six categories while quietly
putting 90% of its money in one, and only the share shows that.

**Time to resolution of chosen markets** — the median days-to-resolution of the
markets entered that day, measured **at entry**. This is not the same as *Avg
hold time* below it: hold time is how long the agent kept a position, this is
how far out the market itself settles. It is worth watching closely, because it
governs how fast the agent learns anything at all.

> **Why this metric exists.** The tradeable universe has a **median
> time-to-close of 75 days** — only 5.3% of it resolves within a day. Sampled
> uniformly, agents parked most of their bankroll in markets that could not
> teach them anything for a quarter: over the first 18 hours live, all four
> agents together saw **six real settlements**, and two thirds of the
> "resolutions" they did get were their own exits rather than market outcomes.
>
> Discovery sampling is now weighted toward markets resolving sooner
> (`resolution_half_life_days` in `config/agents.yaml`), which drops the median
> of what gets drawn from 81 days to about 4. It is a **weight, not a filter**:
> markets more than a year out still get drawn ~0.8% of the time rather than
> never, and nothing in it looks at category, so market freedom is intact. Set
> it to `0` for uniform sampling. It is identical for all four agents on
> purpose — it controls how fast feedback arrives, so differing values would
> make the head-to-head comparison measure the sampler instead of the
> personalities.

**Awake hours per day** — the historical side of the sleep mechanic (§8), next
to the current status light on Overview. Normalize performance by this before
comparing agents: Kenny is awake 19h/day and Cartman 13h.

**Market preference** — which categories it gravitates toward. This is one of
the project's core research questions, and it is meaningful *because* the
candidate sampling is unfiltered — no category is favoured by the system, so any
concentration is the agent's own.

**Learned weights** — the entire model, readable. Every prediction is the sum of
weight × feature value, nothing hidden. `mem_series_roi = +1.03` means
"remembered ROI on this market type strongly raises my estimate". This table is
the whole reason for choosing a linear model over a neural net.

> **Every `mem_*` weight sitting at exactly `0.0000` in the first days is
> normal, and is the single most alarming-looking thing on the dashboard.**
> A weight can only move if its feature was non-zero in some training example.
> The memory bank is written *when a position resolves*, so the trades that have
> trained the model so far were opened back when memory was still empty — every
> one of them carried `mem_* = 0`. As positions opened *after* memory existed
> start resolving, these come alive on their own. Nothing is unwired.
>
> Two ways to confirm rather than take this on faith: the **Memory bank** tab
> should be filling up (if it is empty too, nothing has resolved yet, which
> explains both), and running with `--pretrain` seeds memory from history before
> the first live trade, which makes these weights non-zero almost immediately.

---

## Head-to-head

All four side by side, plus a table of exactly what makes them differ.

The second table is the honest part: **there are no per-agent code branches
anywhere in the project.** Same policy, same reward function, same loop. Every
difference in the top table traces to a number in the bottom one.

What to watch for:

- **Stan is the control.** If Cartman behaves like Stan, the config is wrong —
  there is no hidden code path where his greed might be lurking.
- **Kyle sitting at exactly $100.00 having placed no trades is correct
  behaviour**, not a broken agent. His threshold is 0.05 and nothing has cleared
  it.
- **Cartman should lose fastest early** — big bets, low threshold, high optimism.
- **Kenny should have the most positions and shortest holds.**
- Normalize by **awake hours** before concluding anything: Kenny is awake 19h/day
  and Cartman 13h.

---

## What "normal" looks like over time

These assume the resolution weighting is on (the default). With uniform
sampling the whole timeline stretches by roughly an order of magnitude, because
the median market takes 75 days to tell the agent anything.

| When | Expect |
|---|---|
| **Hour 1** | Trades across varied categories. Memory near-empty. Everyone losing a little. |
| **Day 1** | Dozens of positions, some settled. First memory rows with confidence bars still nearly empty. Win rate meaningless. |
| **Week 1** | Hundreds of updates. Mean error starting to fall. Calibration dots taking shape. Personalities visibly diverging. |
| **Week 4+** | Memory rows with real confidence. Calibration approaching the diagonal. Category preferences emerging. Profit still unlikely. |

§13 is blunt about this: *"up and running"* and *"visibly competent"* are very
different milestones. **If an agent looks good on day one, suspect a bug before
celebrating** — that is what leaked data looks like.

---

## Telling a bug from bad trading

| Symptom | Probably fine | Probably a bug |
|---|---|---|
| Agents losing money | Yes — expected for a long time | — |
| Kyle not trading | Yes — his threshold is high | Only if he *never* trades for days |
| Every decision `no_liquidity` | — | Yes. Check `books=N/N` are equal and non-zero |
| Win rate 0% with trades settled | Possible early with long shots | Check calibration: flat-lined means broken features |
| Memory patterns stuck at 0 | Only if nothing has settled | Yes if positions have settled |
| All `mem_*` weights exactly 0.0000 | Yes, for the first days — see Analytics | Yes if the memory bank has rows *and* trades opened since then have resolved |
| Every `Q` identical | Yes — that is the optimism prior | Yes if it persists after dozens of weight updates |
| An agent stops trading entirely | Kyle, or an agent inside its sleep window | Check open positions against its cap — a full cap plus long-dated holdings is a deadlock it cannot exit its way out of |
| `Time to resolution` climbing past a few weeks | Briefly, after an unlucky draw | Sustained means the weighting is off or set to 0 — learning will crawl |
| Equity exactly $100.00 after hours | Only Kyle | Any other agent means it never traded |
| Bankruptcies climbing fast | Cartman, plausibly | All four means sizing is broken |
| `stream DOWN` persisting | — | Yes. Check the terminal for reconnect backoff |

---

## Where the data lives

Everything is under `data/`, browsable directly:

```
data/
  activity.db              decisions, positions, daily episodes, events
  agents/<name>/
    memory.db              the memory bank
    weights.npz            the model, with feature names
    state.json             bankroll, open positions
  pretrain/                cached historical candles
```

State is checkpointed every 60 seconds and written atomically, so a crash or
reboot resumes rather than restarting (§11). Decision rows are pruned after 45
days; positions, daily summaries, and events are kept forever — they are small,
and they are the history the project is actually about.

---

## One thing the dashboard cannot tell you

There are no natural-language explanations of *why* an agent chose something,
and that is deliberate (§6). The decision function stays a black box. What you
get instead is what it **remembers** and what it **does** — the memory bank and
the activity feed — plus the full weight table, which is as close to "why" as an
honest system gets.
