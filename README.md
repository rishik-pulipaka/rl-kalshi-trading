# South Park Agents — Multi-Agent RL Trading Sandbox

Four reinforcement-learning agents — **Stan, Kyle, Cartman, Kenny** — independently
learn to trade [Kalshi](https://kalshi.com) prediction markets using **simulated
money against real live market data**.

This is an **observational experiment, not a profit tool**. The interesting question
isn't whether the agents get rich — it's watching four mechanically distinct
personalities each discover (or fail to discover) how prediction markets work,
starting from zero knowledge.

> **No real money. No real orders. Ever.**
> The Kalshi API is used exclusively as a read-only market-data source. This
> codebase contains no order-placement code, and a test enforces that it stays
> that way. See [Safety](#safety).

## The premise

Each agent starts with **$100 in virtual funds** and **zero knowledge** — it does
not begin knowing that a lower price means a bigger payout, or what any market
means. It discovers the mechanics through experience.

Agents have the **full action space** a human has: buy Yes/No on any market, sell
before resolution, build multi-leg parlays, and — as a first-class action —
**do nothing at all**.

They have **full market freedom**: no category restriction. Sports, crypto,
weather, economics, politics. *Which markets each agent gravitates toward on its
own* is one of the core things being measured.

They live **completely separate lives**: separate bankrolls, separate memory,
separate weights. No shared experience, no communication.

## The agents

| Agent | Personality | Mechanically |
|---|---|---|
| **Stan** | The balanced everyman — the control group | Moderate risk, moderate exploration, no market bias, standard sizing |
| **Kyle** | Cautious, analytical, rule-following | Low risk, variance/drawdown-penalized reward, low exploration, small sizing, prefers high-confidence memory |
| **Cartman** | Greedy, overconfident, swings big | High risk, reward weights big payouts, drawn to long shots and combos, over-repeats what recently worked |
| **Kenny** | Chaotic and unpredictable | Very high exploration, erratic sizing, shortest hold times, churns many small positions |

All four run the **same learning algorithm and the same code**. Personality is
expressed entirely through config — exploration rate, reward weights, bet-size
distribution, memory recency weighting, sleep schedule. There are no per-agent
code branches, which is what makes "the personalities are real, not cosmetic"
an actually verifiable claim.

Each agent also **sleeps** on its own schedule, taking no actions during its
offline window — Cartman sleeps longest, Kenny barely sleeps at all. Time spent
awake is a genuine variable when comparing their performance.

## How they learn

Deliberately **simple and explainable** over maximally sophisticated:

- **Linear function-approximation Q-learning** over a named feature vector.
  Every weight has a human-readable name, so the model itself is inspectable
  rather than being an opaque blob.
- **An explicit memory bank** — a persistent, queryable store of what the agent
  has learned about specific entities and patterns, separate from the model
  weights and readable by a human:

  ```
  market_pattern: "NBA player 3PM over"
  entity:         "Stephen Curry"
  encounters: 7   wins: 5
  avg_entry_price: 0.44
  net_pnl: +12.40
  ```

  This memory isn't a display gimmick bolted onto a neural net — it feeds
  directly into the decision function, and it's the primary window into how
  each agent is actually learning.

Training runs in two phases: a fast **historical pretraining** pass against
resolved markets to get past "doesn't understand the game at all," then a clean
handoff to **live learning**, where one real trading day is one episode.

## Status

**Running.** All four agents trade simulated money against the live Kalshi
firehose, with the dashboard alongside them.

Four minutes into one run, the personalities had already separated on their own:

```
Stan $95.08    Kyle $100.00    Cartman $87.09    Kenny $89.44
```

Kyle had not placed a single trade -- nothing cleared his `act_threshold`.
Cartman was bleeding fastest (big bets), Kenny second (churn), Stan in between.
Nobody was making money, which is what §13 predicts and what an honest day one
looks like.

What exists: the read-only data layer, a two-tier firehose over ~100k markets,
the trading simulation, the memory bank, the learning loop, comprehensive
logging from day one, and all five dashboard views. 251 tests.

Still open: historical pretraining (Phase A) is scaffolded but not wired in, and
bankruptcy currently costs an agent little because the terminal penalty lives in
the daily reward, which does not train the model.

## Safety

The real, funded Kalshi account is protected by four independent layers:

1. **The private key is never stored in this repo.** `.env` points at its
   existing location elsewhere on disk.
2. **`.gitignore` excludes** `.env`, `*.pem`, `*.key`, and all runtime `data/`.
3. **No order-placement code exists anywhere in this codebase.** The Kalshi
   client only reads market data, settled markets, and price history. Nothing
   can place an order because nothing knows how.
4. **A test enforces layer 3**, so it can't be reintroduced by accident later.

## Cost

**$0.** Kalshi market data is free. Everything runs locally on CPU with
open-source libraries. No paid compute, no paid data providers — by design, not
by budget.

## Setup

Requires Python 3.12.

```bash
python -m pip install -r requirements.txt
cp .env.example .env    # then fill in your Kalshi key id and .pem path
```

## Running

```bash
python run.py                 # Stan only
python run.py --agents all    # all four
```

Starts the agent simulation **and** the dashboard together from one terminal.
The dashboard is at `localhost:8000`.

Run it from a **standalone PowerShell or CMD window** you can minimize and leave
open — not a VS Code integrated terminal. The agents keep learning whether or not
the browser tab is open; the dashboard is just a view onto data they're
continuously writing. All state persists to `data/`, so a crash or reboot resumes
rather than restarting from zero.

## A note on what the code knows that the docs do not

Several things about Kalshi's API are not documented anywhere and cost real
debugging time. They are written into the modules that depend on them, with the
measurements that established them:

- `orderbook_delta` **rejects an unfiltered subscribe** while `ticker`, `trade`,
  and `market_lifecycle_v2` accept one. The two-tier stream design is forced by
  this, not chosen. (`kalshi/stream.py`)
- `seq` is a **connection-wide counter over every sequenced frame** — not per
  market, not per subscription, and `subscribed` acknowledgements consume one.
  Getting this wrong leaves every order book silently stale.
  (`kalshi/books.py`, reproducible via `tools/probe_sequence.py`)
- `/events?with_nested_markets=true` returns the real universe in 59 pages and
  ~20s; `/markets?status=open` needs 1,287 pages for the same thing plus 1.19M
  auto-generated parlays. (`kalshi/universe.py`)
- An open market's status is `"active"`, not `"open"` — `"open"` is the
  *event's* status. A `yes_bid` of `"0.0000"` is the absence of a bid, not a bid
  at zero.
- A settled market's `last_price` is the post-settlement print (0.9990 / 0.0010).
  Training on it is label leakage. (`kalshi/rest.py`)

## Design docs

[`agents-prd.md`](agents-prd.md) — the full product requirements this is built against.
