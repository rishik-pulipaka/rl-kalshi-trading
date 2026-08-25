"""One agent's decision loop: look, decide, act, learn.

This is where everything meets. The agent samples candidate markets from the
live universe, scores them, maybe acts, and later learns from what happened.

## The cycle

    tick()        consider entering something (or deliberately not)
    check_exits() reconsider what is already held
    settle()      credit resolved positions, train on each one
    close_day()   score the day, persist everything

## How "do nothing" works

There is no separate "should I trade at all?" step. Candidates are scored, and
if none beats the agent's `act_threshold` it does nothing and that non-action is
logged as a first-class event (PRD 2, 9). Kyle's threshold is above zero so he
needs a real edge; Kenny's is below zero so he will take almost anything.

## How exits work

The same question, asked again: *would I enter this position right now?* If the
score for a held market has fallen well below the entry threshold, the agent
gets out.

Two guards on that, both added after a simulation showed the naive version
destroying the agent:

  **A margin.** Using one threshold for both entry and exit means anything
  barely worth entering is instantly worth exiting. The agent buys, immediately
  sells, and pays the spread plus two fees for nothing.

  **A minimum hold.** An agent whose model has turned pessimistic will exit
  every position the moment it opens it -- including positions opened by
  exploration, which is self-defeating, because the entire point of exploring is
  to find out how the bet turns out. In a 300-day simulation this produced an
  agent that opened 355 positions, exited nearly all of them within the same
  day, learned only that round trips cost money, and stopped trading with $3.45
  left.

The minimum hold is also PRD 4's hold-time trait made mechanical: Kenny's is ten
minutes and Kyle's is six hours.

## Depth is requested for what the agent looks at, not just what it holds

The broad WebSocket tier carries prices for every market but the `ticker` frame
for any given market may not have arrived yet, and the REST sweep does not carry
sizes at all. So an agent evaluating a market usually has a price and no idea
how much is available at it.

`sim.fills` refuses to invent liquidity, which is correct -- but it means the
agent must actively pull a market's ladder into the depth tier before it can
trade it. Requesting depth only for *held* markets deadlocks: nothing can be
held until something fills, and nothing can fill without depth. So every market
an agent seriously considers gets subscribed, and it becomes tradeable a tick
later when its snapshot lands.

This is also what makes the depth tier track real attention rather than a
hand-picked list.

## Multi-leg combos, both kinds (PRD 2)

An agent can **take** a listed combo and **construct** one, and they are not the
same thing.

*Taking* one is free: Kalshi's ~1.19M auto-generated cross-category parlays are
pulled into the universe by `Universe.adopt_many` and then evaluated exactly
like any other market. A genuine all-or-nothing instrument, priced by the
exchange.

*Constructing* one produces a **basket**: several legs committed to as one
decision, sized as one stake, attributed together. Each leg still settles on its
own -- you cannot manufacture all-or-nothing payoff by buying legs separately,
and pretending otherwise would invent a payoff the exchange does not offer (see
`sim/portfolio.py`).

A basket is scored as the mean Q of its legs, which is the honest number: the
legs resolve independently, so the expected return of the bundle is the average
of theirs. Each leg trains the model on its own realized outcome.

How often an agent builds one is `combo_appetite` -- Cartman 0.35, Kyle 0.02.

## Learning happens on resolution, not on entry

Each position carries the feature vector from the moment it was opened. When it
settles, that vector and the realized return are one training example. This is
what makes the learning signal fire dozens of times a week rather than seven
(see the plan's per-trade credit assignment decision).
"""

import os
import random
import logging
import datetime as dt

from sim import money, fills, settlement
from sim.fills import YES, NO
from sim.portfolio import Portfolio
from . import features as feat
from . import sleep as sleep_module
from .memory import MemoryBank
from .policy import LinearQ
from .reward import daily_reward, trade_target

log = logging.getLogger(__name__)


class Candidate:
    """One (market, side) pair the agent could act on."""

    __slots__ = ("market", "side", "features", "price")

    def __init__(self, market, side, vector, price):
        self.market = market
        self.side = side
        self.features = vector
        self.price = price

    def __repr__(self):
        return f"<Candidate {self.side} {self.market.ticker} @ {self.price}>"


class Agent:
    """One agent. Owns its own bankroll, memory, and weights -- shared with nobody."""

    def __init__(self, personality, data_dir, store=None, seed=None):
        self.p = personality
        self.name = personality.name
        self.store = store
        self.rng = random.Random(seed if seed is not None
                                 else abs(hash(self.name)) % (2 ** 31))

        self.dir = os.path.join(data_dir, "agents", self.name)
        os.makedirs(self.dir, exist_ok=True)

        self.portfolio = Portfolio(self.name, bankroll=personality.starting_bankroll)
        self.memory = MemoryBank(os.path.join(self.dir, "memory.db"),
                                 ewma_alpha=personality.memory_ewma_alpha)
        self.policy = LinearQ(learning_rate=personality.policy.learning_rate,
                              epsilon=personality.policy.epsilon,
                              optimism=personality.policy.optimism,
                              act_threshold=personality.policy.act_threshold,
                              seed=seed)

        self.day = dt.date.today().isoformat()
        self.trades_today = 0
        self.last_decision_at = 0.0
        self.last_exit_check_at = 0.0
        self.last_settle_at = 0.0

        # Markets we want order-book depth on. The stream is told about these so
        # the depth tier populates from what agents actually care about.
        self.depth_wanted = set()

    # ---------- state ----------

    @property
    def weights_path(self):
        return os.path.join(self.dir, "weights.npz")

    @property
    def state_path(self):
        return os.path.join(self.dir, "state.json")

    def is_asleep(self, when=None):
        return sleep_module.is_asleep(self.name, self.p.sleep, when)

    def sleep_status(self, when=None):
        return sleep_module.status(self.name, self.p.sleep, when)

    # ---------- the decision ----------

    def build_candidates(self, universe, books=None, now=None):
        """Sample markets and turn them into scored-able candidates.

        Sampling rather than scoring all ~100k tradeable markets every minute:
        the point is a decision per tick, and it costs a constant amount of work.

        The sample is split between markets whose order book we already hold and
        markets we have never looked at. Purely uniform sampling was tried on
        live data and it starved the agent: with 99,722 tradeable markets and 40
        drawn per minute, a market already carrying depth is essentially never
        drawn twice, so seven of every eight decisions died with "no_liquidity"
        before anything could fill.

        The split is a working set, not a restriction. Any market can enter it
        -- the discovery half is drawn uniformly across the entire universe with
        no category filter -- so "which markets does this agent gravitate
        toward" stays a real question about the agent rather than an artifact of
        how we sampled. What the split buys is that the agent always has some
        candidates it can actually act on.
        """
        pool = universe.tradeable(now)
        if not pool:
            return []

        budget = self.p.trading.candidates_per_decision
        sampled = []

        # Half the budget on markets we can trade right now. Weighted the same
        # way as discovery below -- the working set is built up from whatever
        # the agent asked depth for in the past, so left unweighted it keeps
        # feeding back the long-dated markets it used to favour.
        if books is not None:
            ready = [m for m in pool if books.has_depth(m.ticker)]
            if ready:
                take = min(budget // 2, len(ready))
                sampled.extend(self._discover(ready, take, now))

        # The rest on discovery, drawn from the whole universe. This is what
        # keeps market freedom real and what grows the working set over time.
        remaining = min(budget - len(sampled), len(pool))
        if remaining > 0:
            sampled.extend(self._discover(pool, remaining, now))

        held = {p.ticker for p in self.portfolio.open_positions()}
        candidates = []
        seen = set()
        for market in sampled:
            if market.ticker in seen:
                continue        # the two samples can overlap
            seen.add(market.ticker)
            if market.ticker in held:
                continue        # already exposed here; exits are handled separately
            seconds = market.seconds_to_close(now)
            if seconds is not None and seconds < self.p.trading.min_seconds_to_close:
                continue        # too close to resolution to enter meaningfully

            has_depth = bool(books and books.has_depth(market.ticker))
            if not has_depth:
                # Ask for this market's ladder so a later tick can actually fill
                # it. Without this the agent deadlocks: the REST sweep gives
                # prices but not sizes, most markets have not had a `ticker`
                # frame yet, and depth was only ever requested for markets
                # already held -- which cannot happen until something fills. On
                # the first live run every single decision died with
                # "no_liquidity" for exactly this reason.
                self.depth_wanted.add(market.ticker)

            for side in (YES, NO):
                vector = feat.build(market, side, memory=self.memory,
                                    has_depth=has_depth, now=now)
                if vector is None:
                    continue
                price = feat.entry_price(market, side)
                candidates.append(Candidate(market, side, vector, price))
        return candidates

    # Oversampling factor for weighted discovery. The pool is ~100k markets and
    # this runs on every tick for every agent, so weighting the whole pool would
    # be wasteful. Drawing a uniform shortlist first and weighting only that is
    # far cheaper and keeps every market reachable, because the first stage is
    # still uniform over the entire universe.
    DISCOVERY_OVERSAMPLE = 40

    def _discover(self, pool, k, now=None):
        """Draw `k` markets, leaning toward the ones that resolve soonest.

        Why this exists: sampled uniformly, the tradeable universe has a median
        time-to-close of **75 days** and only 5.3% of it resolves within a day.
        An agent drawing uniformly therefore spends most of its bankroll on
        positions that cannot produce a learning signal for months -- measured
        over the first 18 hours live, all four agents together saw six real
        settlements, and two thirds of the "resolutions" they did get were
        their own exits rather than market outcomes. PRD 13 asks for the
        learning process to be legible from day one, and it cannot be if
        feedback arrives a quarter after the decision.

        This is deliberately a **weight, not a filter**. A market a year out is
        drawn about 50x less often than one resolving today, but it is never
        excluded, and nothing here looks at category -- so PRD 2's market
        freedom, and the "which markets does it gravitate toward" question,
        both survive intact. Set `resolution_half_life_days: 0` for the old
        uniform behaviour.
        """
        half_life = self.p.trading.resolution_half_life_days
        if not half_life or half_life <= 0:
            return self.rng.sample(pool, k)

        shortlist = self.rng.sample(
            pool, min(len(pool), max(k, k * self.DISCOVERY_OVERSAMPLE)))
        if len(shortlist) <= k:
            return shortlist

        # A-Res weighted sampling without replacement: key each item by
        # u**(1/w) and keep the largest k. Standard, and one pass.
        half_life_seconds = half_life * 86400.0
        keyed = []
        for market in shortlist:
            seconds = market.seconds_to_close(now)
            if seconds is None or seconds <= 0:
                weight = 1.0          # unknown close time: no opinion either way
            else:
                weight = 1.0 / (1.0 + seconds / half_life_seconds)
            keyed.append((self.rng.random() ** (1.0 / weight), market))
        keyed.sort(key=lambda pair: -pair[0])
        return [market for _, market in keyed[:k]]

    def tick(self, universe, books=None, now=None):
        """One decision cycle. Returns the `Decision`, acted on or not."""
        if self.is_asleep():
            return None
        if len(self.portfolio.open_positions()) >= self.p.trading.max_open_positions:
            return None

        candidates = self.build_candidates(universe, books, now)
        decision = self.policy.select(candidates)

        if decision.acted:
            # Sometimes commit to several legs at once rather than one bet.
            # Personality decides how often (PRD 2, 4).
            if (self.p.trading.combo_appetite > 0
                    and self.rng.random() < self.p.trading.combo_appetite
                    and len(candidates) >= self.p.trading.combo_legs):
                self._enter_basket(decision, candidates, books)
            else:
                self._enter(decision, books)
        if self.store:
            market = decision.candidate.market if decision.acted else None
            side = decision.candidate.side if decision.acted else None
            self.store.log_decision(self.name, decision, market, side, self.day)
        return decision

    def _enter(self, decision, books=None):
        """Size, fill, and record a position. May fill partially or not at all."""
        candidate = decision.candidate
        market = candidate.market

        stake = self.p.sizing.sample(self.rng, self.portfolio.bankroll)
        if stake <= 0:
            decision.candidate = None
            decision.skipped_reason = "stake_too_small"
            return None

        wanted = money.max_contracts_affordable(stake, candidate.price)
        if wanted <= 0:
            decision.candidate = None
            decision.skipped_reason = "cannot_afford_one_contract"
            return None

        book = books.get(market.ticker) if books else None
        levels = (fills.levels_from_book(book, candidate.side) if book
                  else fills.levels_from_quote(market, candidate.side))
        fill = fills.buy(levels, wanted, candidate.side)

        if not fill.filled or not self.portfolio.can_afford(fill.cost):
            if not fill.filled:
                # Either the size is genuinely not there, or we are looking at a
                # market whose ladder has not arrived yet. Ask for it either
                # way; the request is idempotent and costs one subscribe.
                self.depth_wanted.add(market.ticker)
            decision.candidate = None
            decision.skipped_reason = ("no_liquidity" if not fill.filled
                                       else "insufficient_funds")
            return None

        position = self.portfolio.open_position(
            market.ticker, candidate.side, fill, self.day,
            features=candidate.features, category=market.category,
            series=market.series, close_ts=market.close_ts)

        self.trades_today += 1
        # Now that we hold it, we want the full ladder for a better exit price.
        self.depth_wanted.add(market.ticker)

        if self.store:
            self.store.upsert_position(position, title=market.title,
                                       q_at_entry=decision.q,
                                       explored=decision.explored)
        log.info("%s entered %s %s x%d @ %d (q=%.3f%s)", self.name,
                 candidate.side, market.ticker, fill.contracts, fill.avg_price,
                 decision.q, ", explore" if decision.explored else "")
        return position

    def _enter_basket(self, decision, candidates, books=None):
        """Commit to several legs as one decision (PRD 2: "construct" a combo).

        The stake is split across the legs, so a basket is not a way to bet more
        -- it is a way to bet the same amount on a conjunction. Legs are the
        top-scoring distinct markets, because committing twice to the same
        market is just a bigger single bet wearing a hat.
        """
        legs = self._pick_legs(candidates, self.p.trading.combo_legs)
        if len(legs) < 2:
            return self._enter(decision, books)      # not enough to bundle

        total_stake = self.p.sizing.sample(self.rng, self.portfolio.bankroll)
        per_leg = total_stake // len(legs)
        if per_leg <= 0:
            decision.candidate = None
            decision.skipped_reason = "stake_too_small"
            return None

        basket = self.portfolio.open_basket(
            self.day, features=decision.candidate.features,
            label=f"{len(legs)}-leg")

        opened = []
        for candidate in legs:
            market = candidate.market
            wanted = money.max_contracts_affordable(per_leg, candidate.price)
            if wanted <= 0:
                continue
            book = books.get(market.ticker) if books else None
            levels = (fills.levels_from_book(book, candidate.side) if book
                      else fills.levels_from_quote(market, candidate.side))
            fill = fills.buy(levels, wanted, candidate.side)
            if not fill.filled or not self.portfolio.can_afford(fill.cost):
                self.depth_wanted.add(market.ticker)
                continue
            position = self.portfolio.open_position(
                market.ticker, candidate.side, fill, self.day,
                features=candidate.features, basket_id=basket.id,
                category=market.category, series=market.series,
                close_ts=market.close_ts)
            opened.append(position)
            self.trades_today += 1
            self.depth_wanted.add(market.ticker)
            if self.store:
                self.store.upsert_position(position, title=market.title,
                                           q_at_entry=decision.q,
                                           explored=decision.explored)

        if not opened:
            decision.candidate = None
            decision.skipped_reason = "no_liquidity"
            return None

        if len(opened) == 1:
            # Only one leg could be filled. That is a single bet, not a basket,
            # and labelling it as one would put a "1-leg basket" in the activity
            # feed and make the combo analytics lie about what the agent did.
            opened[0].basket_id = None
            self.portfolio.baskets.pop(basket.id, None)
            if self.store:
                self.store.upsert_position(opened[0])
            return opened[0]

        log.info("%s built a %d-leg basket %s (q=%.3f%s)", self.name,
                 len(opened), basket.id, decision.q,
                 ", explore" if decision.explored else "")
        if self.store:
            self.store.log_event("basket", self.name,
                                 {"id": basket.id, "legs": len(opened),
                                  "stake": total_stake}, self.day)
        return basket

    def _pick_legs(self, candidates, n):
        """The best-scoring candidates, one per market."""
        scored = sorted(candidates, key=lambda c: -self.policy.q(c.features))
        legs, seen = [], set()
        for candidate in scored:
            if candidate.market.ticker in seen:
                continue
            seen.add(candidate.market.ticker)
            legs.append(candidate)
            if len(legs) >= n:
                break
        return legs

    # ---------- exits ----------

    def check_exits(self, universe, books=None, now=None):
        """Re-ask "would I enter this now?" for everything held.

        See the module docstring for why this needs both a minimum hold and a
        threshold margin: without them the agent round-trips the spread on every
        position it opens and never finds out how any of them would have
        resolved.
        """
        import time as _time
        clock = now or _time.time()
        exit_threshold = self.p.policy.act_threshold - self.p.trading.exit_margin

        exited = []
        for position in self.portfolio.open_positions():
            if clock - position.opened_at < self.p.trading.min_hold_seconds:
                continue        # too soon to judge; let it run

            market = universe.get(position.ticker)
            if market is None or not market.is_tradeable:
                continue

            has_depth = bool(books and books.has_depth(position.ticker))
            vector = feat.build(market, position.side, memory=self.memory,
                                has_depth=has_depth, now=now)
            if vector is None:
                continue
            if self.policy.q(vector) > exit_threshold:
                continue        # still worth holding

            book = books.get(position.ticker) if books else None
            levels = (fills.exit_levels_from_book(book, position.side) if book
                      else fills.exit_levels_from_quote(market, position.side))
            fill = fills.sell(levels, position.contracts, position.side)
            if not fill.filled or fill.contracts != position.contracts:
                continue        # cannot exit cleanly; hold rather than part-exit

            self.portfolio.close_position(position, fill, self.day)
            self.trades_today += 1
            self._learn_from(position, market, "exit")
            exited.append(position)
            if self.store:
                self.store.upsert_position(position, title=market.title)
            log.info("%s exited %s @ %d (pnl %s)", self.name, position.ticker,
                     fill.avg_price, money.fmt(position.realized_pnl))
        return exited

    # ---------- settlement and learning ----------

    def settle(self, universe, rest_module, now=None):
        """Settle resolved positions and train on each one."""
        settled = settlement.settle_open_positions(
            self.portfolio, self.day, rest_module)
        for position, outcome in settled:
            market = universe.get(position.ticker)
            self._learn_from(position, market, outcome)
            if self.store:
                self.store.upsert_position(position)
        return settled

    def _learn_from(self, position, market, outcome):
        """One resolved position -> one memory update and one weight update.

        Voids teach nothing: the stake came back and no prediction was tested,
        so training on a zero return would tell the model that this kind of bet
        breaks even, which is not what happened.
        """
        if outcome == "void":
            return

        summary = settlement.outcome_for_memory(position, outcome)
        if market is not None:
            self.memory.record(market, summary, day=self.day,
                               side=position.side, price=position.entry_price)

        # The agent's OWN reward weights: Kyle's heavy exposure penalty and
        # Cartman's light one make the same trade a different lesson for each.
        target = trade_target(position, self.p.reward)
        if target is not None and position.entry_features is not None:
            self.policy.update(position.entry_features, target)

    # ---------- the daily episode ----------

    def close_day(self, day=None, price_of=None, universe=None):
        """Score the episode, log it, and persist state (PRD 3.2, 5).

        `universe` is optional and used only to look up the markets behind any
        positions abandoned to a bankruptcy, so their lesson reaches the memory
        bank as well as the weights. Without it the weights still learn.
        """
        day = day or self.day
        stats = self.portfolio.stats(price_of)
        equity = stats["equity"]
        self.portfolio.update_peak(equity)

        went_bankrupt = self.portfolio.is_bankrupt(price_of)
        reward, parts = daily_reward(
            self.p.reward,
            realized_pnl=self.portfolio.realized_today(day),
            trades=self.trades_today,
            drawdown=self.portfolio.drawdown(equity),
            largest_exposure=self.portfolio.largest_open_exposure(),
            bankroll=self.portfolio.bankroll,
            open_positions=len(self.portfolio.open_positions()),
            went_bankrupt=went_bankrupt,
            starting_bankroll=self.p.starting_bankroll)

        if went_bankrupt:
            count, abandoned = self.portfolio.reset_after_bankruptcy(
                self.p.starting_bankroll)
            # Learn from the wreckage. These are total losses carrying an extra
            # ruin penalty (see reward.trade_target); skipping them meant the
            # worst outcome in the system was the only one that taught nothing.
            for position in abandoned:
                market = universe.get(position.ticker) if universe else None
                self._learn_from(position, market, "bankrupt")
                if self.store:
                    self.store.upsert_position(position)
            log.warning("%s went bankrupt (#%d); reset to %s (%d positions "
                        "abandoned and learned from)", self.name, count,
                        money.fmt(self.p.starting_bankroll), len(abandoned))
            if self.store:
                self.store.log_event("bankruptcy", self.name,
                                     {"count": count, "day": day,
                                      "abandoned": len(abandoned)}, day)

        if self.store:
            self.store.log_daily(
                day, self.name, reward=reward, reward_parts=parts,
                realized_pnl=self.portfolio.realized_today(day),
                trades=self.trades_today, bankroll=self.portfolio.bankroll,
                equity=equity, drawdown=stats["drawdown"],
                open_positions=stats["open_positions"],
                win_rate=stats["win_rate"], bankruptcies=stats["bankruptcies"],
                awake_hours=sleep_module.awake_hours_on(self.name, self.p.sleep, day),
                exploration_rate=self.policy.exploration_rate,
                mean_abs_error=self.policy.mean_abs_error,
                updates=self.policy.updates,
                weights=self.policy.named_weights())

        self.save()
        return reward, parts

    def roll_day_if_needed(self, price_of=None, universe=None):
        """Close the episode and start a new one when the date changes."""
        today = dt.date.today().isoformat()
        if today == self.day:
            return None
        finished = self.day
        result = self.close_day(finished, price_of, universe)
        self.day = today
        self.trades_today = 0
        log.info("%s rolled over %s -> %s", self.name, finished, today)
        return result

    # ---------- persistence (PRD 11) ----------

    def save(self):
        """Everything needed to resume: weights, bankroll, and open positions.

        Memory writes through to SQLite on every update, so it needs nothing here.
        """
        import json
        self.policy.save(self.weights_path)
        state = {
            "day": self.day,
            "trades_today": self.trades_today,
            "bankroll": self.portfolio.bankroll,
            "peak_equity": self.portfolio.peak_equity,
            "bankruptcies": self.portfolio.bankruptcies,
            "trades_opened": self.portfolio.trades_opened,
            "trades_closed": self.portfolio.trades_closed,
            "total_fees": self.portfolio.total_fees,
            "realized_by_day": self.portfolio.realized_by_day,
            "positions": [p.to_row() | {"entry_features": p.entry_features}
                          for p in self.portfolio.positions.values()],
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        # Atomic replace: a crash mid-write must not leave a truncated state
        # file, which would silently reset an agent's bankroll to $100.
        os.replace(tmp, self.state_path)

    def load(self):
        """Resume from disk. Returns False if there was nothing to resume."""
        import json
        self.policy.load(self.weights_path)
        if not os.path.exists(self.state_path):
            return False
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except (ValueError, OSError) as exc:
            log.error("%s has an unreadable state file (%s); starting fresh",
                      self.name, exc)
            return False

        self.day = state.get("day", self.day)
        self.trades_today = state.get("trades_today", 0)
        self.portfolio.bankroll = state.get("bankroll", self.p.starting_bankroll)
        self.portfolio.peak_equity = state.get("peak_equity", self.portfolio.bankroll)
        self.portfolio.bankruptcies = state.get("bankruptcies", 0)
        self.portfolio.trades_opened = state.get("trades_opened", 0)
        self.portfolio.trades_closed = state.get("trades_closed", 0)
        self.portfolio.total_fees = state.get("total_fees", 0)
        self.portfolio.realized_by_day = state.get("realized_by_day", {})
        self._restore_positions(state.get("positions", []))
        log.info("%s resumed: %s, %d open positions, %d weight updates",
                 self.name, money.fmt(self.portfolio.bankroll),
                 len(self.portfolio.open_positions()), self.policy.updates)
        return True

    def _restore_positions(self, rows):
        """Rebuild Position objects without re-running an order through fills."""
        from sim.portfolio import Position

        class _Stub:
            """Just enough of a Fill for Position.__init__."""
            def __init__(self, row):
                self.contracts = row["contracts"]
                self.avg_price = row["entry_price"]
                self.gross = row["cost"] - (row["entry_fee"] or 0)
                self.fee = row["entry_fee"] or 0

        for row in rows:
            position = Position(row["agent"], row["ticker"], row["side"],
                                _Stub(row), row["opened_day"],
                                features=row.get("entry_features"),
                                basket_id=row.get("basket_id"),
                                category=row.get("category"),
                                series=row.get("series"),
                                close_ts=row.get("close_ts"))
            position.id = row["id"]
            position.opened_at = row["opened_at"]
            position.status = row["status"]
            position.exit_price = row["exit_price"]
            position.exit_proceeds = row["exit_proceeds"]
            position.exit_fee = row["exit_fee"]
            position.closed_at = row["closed_at"]
            position.closed_day = row["closed_day"]
            position.realized_pnl = row["realized_pnl"]
            position.result = row["result"]
            position.stake_fraction = row.get("stake_fraction", 0.0) or 0.0
            self.portfolio.positions[position.id] = position
            if position.is_open:
                self.depth_wanted.add(position.ticker)

    # ---------- reporting ----------

    def snapshot(self, price_of=None):
        """Everything the dashboard shows for this agent, in one call."""
        stats = self.portfolio.stats(price_of)
        stats.update(
            display_name=self.p.display_name,
            blurb=self.p.blurb,
            day=self.day,
            trades_today=self.trades_today,
            realized_today=self.portfolio.realized_today(self.day),
            sleep=self.sleep_status(),
            policy=self.policy.stats(),
            memory=self.memory.stats(),
        )
        return stats

    def close(self):
        self.save()
        self.memory.close()
