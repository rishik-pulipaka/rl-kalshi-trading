"""The whole system, from one terminal.

    python run.py                 # Stan only (PRD 10 step 1)
    python run.py --agents all    # all four
    python run.py --dry-run       # no dashboard, verbose, for a quick look

PRD 15 asks for one command that starts the agents *and* the dashboard, run from
a standalone PowerShell or CMD window you can minimize and leave open. The
browser tab is a read layer over data the agents are continuously writing --
closing it does not stop anything, and reopening it later shows what happened
while it was closed.

## Threads

Three, chosen so each is separately understandable rather than to be clever:

    stream thread   asyncio; the Kalshi firehose. Updates the universe and the
                    order books inline, and touches nothing else.
    agent thread    the main loop. Ticks each agent on its own schedule, checks
                    exits, settles, rolls the day over. All the money logic.
    dashboard       Flask, daemon. Reads SQLite; never writes agent state.

The universe is lock-protected because both the stream and the agents touch it.
Order books are written only by the stream thread and read by agents without a
lock: a torn read costs at most a slightly stale price in a simulated fill,
which is not worth putting a mutex on a path that handles thousands of messages
a second.

## Shutdown

Ctrl-C saves every agent's state before exiting. Bankroll, memory, weights, and
open positions all survive a restart (PRD 11) -- memory writes through to SQLite
continuously, and the rest is written atomically on every save.
"""

import os
import sys
import time
import signal
import asyncio
import logging
import argparse
import datetime as dt
import threading

from dotenv import load_dotenv

import runtime
from kalshi import auth, rest, stream as stream_module, universe as universe_module
from kalshi.books import BookRegistry
from sim import money
from agent.personality import load_all
from agent.loop import Agent
from store.db import Store

log = logging.getLogger("run")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# How often the background chores run, in seconds.
UNIVERSE_REFRESH = 600      # re-sweep /events; ~20s and it keeps the universe honest
SETTLE_INTERVAL = 300       # check whether held markets have resolved
HEALTH_INTERVAL = 60        # log a heartbeat and prune
PRUNE_INTERVAL = 86400
WARMUP_INTERVAL = 30        # top up the depth working set
ADOPT_INTERVAL = 120        # pull listed combo markets into the universe

# How many of Kalshi's auto-generated parlays to carry in the universe.
# They are not swept up front (there are ~1.19M), but PRD 2 wants agents able
# to TAKE a listed multi-leg combo, so a rotating sample of the ones actually
# quoting is materialized and evaluated like any other market.
COMBO_POOL = 1500

# How many markets to carry full order-book depth for. This is the working set
# agents actually trade out of: without it they can see 100k markets but fill
# almost none of them, because the exchange will not stream depth unfiltered.
#
# 3,000 was measured as comfortable -- ~3 MB of RAM and a manageable delta rate.
# It is a budget on *attention*, not on access: which markets occupy it is
# driven by what the agents look at, and every market in the universe stays
# visible and reachable.
DEPTH_WORKING_SET = 3000


class KillSwitch:
    """Pause every agent immediately (PRD 11).

    Set from the dashboard. Agents check it before acting; the stream keeps
    running so the universe stays warm and the dashboard stays live -- pausing
    is about stopping the agents from *trading*, not about going dark.
    """

    def __init__(self):
        self._paused = threading.Event()

    @property
    def paused(self):
        return self._paused.is_set()

    def pause(self):
        self._paused.set()
        log.warning("KILL SWITCH ENGAGED - all agent activity paused")

    def resume(self):
        self._paused.clear()
        log.warning("kill switch released - agents resumed")


class System:
    """Everything wired together."""

    def __init__(self, agent_names, data_dir=DATA_DIR, dashboard_port=8000,
                 pretrain_series=None):
        load_dotenv()
        self.data_dir = data_dir
        self.dashboard_port = dashboard_port
        self.pretrain_series = pretrain_series or []
        os.makedirs(data_dir, exist_ok=True)

        key_id = os.getenv("KALSHI_KEY_ID")
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if not key_id or not key_path:
            raise SystemExit(
                "KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set.\n"
                "Copy .env.example to .env and fill them in.")
        self.private_key = auth.load_private_key(key_path)
        self.key_id = key_id

        self.universe = universe_module.Universe()
        self.books = BookRegistry(on_desync=self._on_desync)
        self.store = Store(os.path.join(data_dir, "activity.db"))
        self.kill = KillSwitch()

        personalities = load_all()
        missing = [n for n in agent_names if n not in personalities]
        if missing:
            raise SystemExit(f"unknown agent(s): {', '.join(missing)}")
        self.agents = [Agent(personalities[n], data_dir, store=self.store)
                       for n in agent_names]

        self.stream = stream_module.Stream(
            key_id, self.private_key, self._on_message,
            on_reconnect=self._on_reconnect)

        self._stop = threading.Event()
        self._stream_thread = None
        self._loop = None
        self.started_at = time.time()

    # ---------- stream plumbing ----------

    def _on_message(self, message):
        """Inline on the receive loop. Must stay cheap -- thousands per second.

        EVERY frame goes to the book registry, not just order-book ones: Kalshi's
        `seq` is a connection-wide counter over all sequenced frames, so a
        registry that only sees order-book messages reads a gap on nearly every
        delta. See `kalshi/books.py`.
        """
        self.universe.on_message(message)
        self.books.on_message(message)

    def _on_desync(self, tickers):
        """A connection-wide sequence gap invalidated every book."""
        self.stream.request_resync(tickers)
        self.store.log_event("orderbook_desync", detail={"markets": len(tickers)})

    def _on_reconnect(self):
        """A new connection restarts the sequence, so every book is stale."""
        self.books.reset()
        self.store.log_event("stream_connected",
                             detail={"reconnects": self.stream.reconnects})

    def _run_stream(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self.stream.run())
        except asyncio.CancelledError:
            pass
        finally:
            self._loop.close()

    # ---------- startup ----------

    def start(self):
        print("=" * 68)
        print("  SOUTH PARK AGENTS - multi-agent RL trading sandbox")
        print("  SIMULATED MONEY ONLY. No real order is ever placed.")
        print("=" * 68)

        print("\nsweeping the market universe ...", flush=True)
        swept = self.universe.refresh()
        stats = self.universe.stats()
        print(f"  {swept['total']:,} markets in {swept['seconds']:.1f}s"
              f"  |  tradeable {stats['tradeable']:,}"
              f"  |  two-sided {stats['quoted']:,}")
        top = sorted(stats["categories"].items(), key=lambda kv: -kv[1])[:6]
        print("  " + "  ".join(f"{name} {count:,}" for name, count in top))

        for agent in self.agents:
            resumed = agent.load()
            # PRD 3.1: Phase A runs once, for an agent that has never learned
            # anything. The handoff is clean -- no historical training happens
            # again once an agent has live experience.
            if not resumed and self.pretrain_series:
                self._pretrain(agent)
            state = "resumed" if resumed else "fresh"
            print(f"\n  {agent.p.display_name:8s} {state:8s} "
                  f"{money.fmt(agent.portfolio.bankroll):>10}  "
                  f"{len(agent.portfolio.open_positions())} open  "
                  f"{agent.policy.updates} updates  "
                  f"{'asleep' if agent.is_asleep() else 'awake'}")
            print(f"           {agent.p.blurb}")

        print("\nconnecting to the Kalshi firehose ...")
        self._stream_thread = threading.Thread(
            target=self._run_stream, daemon=True, name="kalshi-stream")
        self._stream_thread.start()

        # Give the broad tier a moment to populate before the first decision, so
        # agents do not judge markets on sweep-stale quotes.
        time.sleep(5)
        print(f"  connected={self.stream.connected} "
              f"messages={self.stream.messages:,}")

        self._start_dashboard()
        print(f"\nrunning. Ctrl-C to stop.\n")

    def _pretrain(self, agent):
        """Phase A for one fresh agent (PRD 3.1)."""
        from agent import pretrain as pretrain_module
        print(f"  pretraining {agent.p.display_name} on "
              f"{', '.join(self.pretrain_series)} ...", flush=True)
        try:
            summary = pretrain_module.pretrain(
                agent, self.pretrain_series, markets_per_series=200,
                data_dir=self.data_dir, rng=agent.rng)
            print(f"    {summary['observations']} observations, "
                  f"{summary['patterns']} memory patterns, "
                  f"{summary['seconds']}s")
            self.store.log_event("pretrained", agent.name, summary)
        except Exception:
            log.exception("pretraining %s failed; going live untrained",
                          agent.name)

    def _start_dashboard(self):
        try:
            from dashboard.server import serve
        except Exception as exc:
            log.warning("dashboard unavailable (%s); agents run regardless", exc)
            return
        thread = threading.Thread(
            target=serve, args=(self,), kwargs={"port": self.dashboard_port},
            daemon=True, name="dashboard")
        thread.start()
        print(f"  dashboard -> http://localhost:{self.dashboard_port}")

    # ---------- the agent loop ----------

    def run_forever(self):
        """The main thread. Ticks agents and runs the background chores."""
        next_universe = time.time() + UNIVERSE_REFRESH
        next_settle = time.time() + 30
        next_health = time.time() + HEALTH_INTERVAL
        next_prune = time.time() + PRUNE_INTERVAL
        next_warmup = time.time() + 5
        next_adopt = time.time() + 45

        while not self._stop.is_set():
            now = time.time()

            for agent in self.agents:
                self._service(agent, now)

            if now >= next_settle:
                self._settle_all()
                next_settle = now + SETTLE_INTERVAL
            if now >= next_warmup:
                self._warm_depth()
                next_warmup = now + WARMUP_INTERVAL
            if now >= next_adopt:
                self._adopt_combos()
                next_adopt = now + ADOPT_INTERVAL
            if now >= next_universe:
                self._refresh_universe()
                next_universe = now + UNIVERSE_REFRESH
            if now >= next_health:
                self._heartbeat()
                next_health = now + HEALTH_INTERVAL
            if now >= next_prune:
                removed = self.store.prune()
                log.info("pruned %d old decision rows", removed)
                next_prune = now + PRUNE_INTERVAL

            # A short sleep keeps the loop responsive to Ctrl-C without spinning.
            self._stop.wait(1.0)

    def _service(self, agent, now):
        """One agent's turn: roll the day, maybe decide, maybe exit."""
        agent.roll_day_if_needed(price_of=self._price_of)

        if self.kill.paused or agent.is_asleep():
            return

        trading = agent.p.trading
        if now - agent.last_decision_at >= trading.decision_interval_seconds:
            agent.last_decision_at = now
            try:
                agent.tick(self.universe, self.books)
            except Exception:
                log.exception("%s failed during tick", agent.name)
            # Always sync, including after a skipped decision: a tick that found
            # nothing tradeable is usually a tick whose markets had no ladder
            # yet, and this is what fetches them.
            self._sync_depth(agent)

        if now - agent.last_exit_check_at >= trading.exit_check_interval_seconds:
            agent.last_exit_check_at = now
            try:
                agent.check_exits(self.universe, self.books)
            except Exception:
                log.exception("%s failed during exit check", agent.name)

    def _sync_depth(self, agent):
        """Pull the markets this agent cares about into the depth tier.

        This is what makes the depth tier follow real interest instead of a
        fixed list: agents subscribe the exchange to their own attention.
        """
        if agent.depth_wanted:
            self.stream.watch_depth(agent.depth_wanted)
            agent.depth_wanted.clear()

    def _settle_all(self):
        for agent in self.agents:
            try:
                settled = agent.settle(self.universe, rest)
                for position, outcome in settled:
                    log.info("%s %s %s  pnl %s", agent.name, outcome,
                             position.ticker, money.fmt(position.realized_pnl))
            except Exception:
                log.exception("%s failed during settlement", agent.name)

        # Stop paying for depth on markets nobody holds any more.
        # Trim only what exceeds the working set, and never a held market.
        held = {p.ticker for a in self.agents for p in a.portfolio.open_positions()}
        excess = len(self.stream.depth_tickers) - DEPTH_WORKING_SET
        if excess > 0:
            droppable = list(self.stream.depth_tickers - held)
            stale = set(droppable[:excess])
            self.stream.unwatch_depth(stale)
            self.books.forget(stale)

    def _warm_depth(self):
        """Keep the depth working set topped up.

        Agents pull markets they look at into depth, but on a cold start nothing
        has depth yet and nothing can fill. This seeds and maintains the set by
        subscribing liquid markets the agents have not reached, drawn from
        across the whole universe rather than from any chosen category.
        """
        current = len(self.stream.depth_tickers)
        if current >= DEPTH_WORKING_SET:
            return
        need = DEPTH_WORKING_SET - current
        candidates = [m for m in self.universe.tradeable()
                      if m.ticker not in self.stream.depth_tickers]
        if not candidates:
            return
        # Prefer markets with real open interest: they are the ones whose books
        # exist and whose fills mean something. Not a category judgement -- open
        # interest is a fact about the market, not a preference of ours.
        candidates.sort(key=lambda m: -(m.open_interest or 0.0))
        self.stream.watch_depth([m.ticker for m in candidates[:need]])

    def _adopt_combos(self):
        """Pull a sample of Kalshi's listed parlays into the universe.

        This is the "take a multi-leg combo" half of PRD 2. The ~1.19M
        auto-generated cross-category markets are never swept, but the ones that
        actually quote are remembered as they stream past, and a rotating sample
        is materialized so agents can evaluate them as ordinary markets.
        """
        try:
            existing = sum(1 for t in self.universe.tickers()
                           if rest.is_auto_combo(t))
            if existing >= COMBO_POOL:
                return
            sample = self.universe.unknown_sample(
                min(300, COMBO_POOL - existing))
            if sample:
                added = self.universe.adopt_many(sample)
                if added:
                    log.info("adopted %d listed combo markets (%d held)",
                             added, existing + added)
        except Exception:
            log.exception("combo adoption failed")

    def _refresh_universe(self):
        try:
            swept = self.universe.refresh()
            log.info("universe refreshed: %d markets in %.1fs",
                     swept["total"], swept["seconds"])
        except Exception:
            log.exception("universe refresh failed")

    def _price_of(self, position):
        """Current price of a position's side, for mark-to-market."""
        market = self.universe.get(position.ticker)
        if market is None:
            return None
        if position.side == "yes":
            return market.yes_bid if market.can_buy_no else None
        return money.no_price(market.yes_ask) if market.can_buy_yes else None

    def _heartbeat(self):
        health = self.stream.health()
        mem = runtime.rss_mb()
        books = self.books.stats()
        equities = "  ".join(
            f"{a.p.display_name} {money.fmt(a.portfolio.equity(self._price_of))}"
            for a in self.agents)
        log.info("stream=%s msgs=%s books=%d/%d rss=%.0fMB | %s",
                 "up" if health["connected"] else "DOWN",
                 f"{health['messages']:,}", books["synced"], books["books"],
                 mem or 0, equities)

    # ---------- shutdown ----------

    def stop(self):
        if self._stop.is_set():
            return
        print("\nstopping; saving agent state ...")
        self._stop.set()
        self.stream.stop()
        for agent in self.agents:
            try:
                agent.close()
                print(f"  {agent.p.display_name:8s} saved  "
                      f"{money.fmt(agent.portfolio.bankroll)}  "
                      f"{len(agent.portfolio.open_positions())} open")
            except Exception:
                log.exception("%s failed to save", agent.name)
        self.store.close()
        print("stopped. State is in", self.data_dir)

    # ---------- what the dashboard reads ----------

    def snapshot(self):
        return {
            "agents": [a.snapshot(self._price_of) for a in self.agents],
            "universe": self.universe.stats(),
            "stream": self.stream.health(),
            "books": self.books.stats(),
            "runtime": runtime.snapshot(),
            "store": self.store.stats(),
            "paused": self.kill.paused,
            "started_at": self.started_at,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agents", default="stan",
                    help="comma-separated names, or 'all' (default: stan)")
    ap.add_argument("--port", type=int,
                    default=int(os.getenv("DASHBOARD_PORT", "8000")))
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--pretrain", default="",
                    help="comma-separated series for Phase A historical "
                         "pretraining, run once per fresh agent "
                         "(e.g. KXBTC15M,KXHIGHNY)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s",
        datefmt="%H:%M:%S")
    logging.getLogger("websockets").setLevel(logging.WARNING)

    names = (list(load_all()) if args.agents == "all"
             else [n.strip() for n in args.agents.split(",") if n.strip()])

    system = System(names, data_dir=args.data_dir, dashboard_port=args.port,
                    pretrain_series=[s.strip() for s in args.pretrain.split(",")
                                     if s.strip()])

    def handle_signal(signum, frame):
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    system.start()
    try:
        system.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()


if __name__ == "__main__":
    main()
