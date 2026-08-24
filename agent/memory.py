"""The explicit memory bank -- what an agent has learned, in readable form.

PRD 3.3 requires this to be a persistent, inspectable, queryable store rather
than just neural-network weights, for two reasons: it is closer to how a human
bettor actually operates (consulting remembered specifics), and it is **the**
window into how each agent is learning. It is the centerpiece of the project.

## It is inside the decision, not beside it

The temptation with a "memory bank" feature is to make it a display artifact
bolted onto a model that ignores it. That is not what this is. The loop is:

    market -> pattern keys -> memory lookup -> features -> Q -> action
                   ^                                             |
                   +---------- outcome updates memory <----------+

`agent/features.py` reads this table to build the feature vector the policy
scores. An agent with no memory of a pattern genuinely sees different numbers
than one with fifty encounters, so what it remembers changes what it does.

## One market produces several memories, at different granularities

A market is not one pattern -- it is several, nested from broad to specific:

    category:Sports                     very general, fills up fast
    category:Sports|side:no             ...and which direction was taken
    series:KXNBA                        a market type
    series:KXNBA|side:yes               a market type and a direction
    series:KXNBA|price:20-40c|side:yes  type, price band, and direction
    entity:Stephen Curry                a specific subject
    entity:Stephen Curry|side:yes       a subject and a direction
    ttr:under_1h                        how soon it resolves

Recording all of them is deliberate. Broad keys accumulate evidence quickly and
are useful early; specific keys are the interesting ones but need many
encounters before they mean anything.

## Why almost every key has a side variant

This was found by simulation, not by design. In a synthetic world where backing
one side of a category was +62% EV and the other side was -34%, an agent with
side-blind memory learned nothing: `series:KXPOLI` averaged a terrible YES with
an excellent NO into something mediocre, so the memory reported "this market
type is roughly break-even" and the agent kept taking the losing side.

Side-blind memory cannot represent the single most basic thing a bettor learns
-- *which way* to bet on a familiar market. So the side is part of the key.

Confidence is exposed per key so the policy can tell "60% over 3 tries" from
"60% over 200 tries", and Kyle's preference for high-confidence patterns is
precisely a preference over that number.

## Recency weighting is a personality dial

Every row keeps both a lifetime average and an EWMA of return-on-risk. The EWMA's
alpha is per-agent: Cartman's is high, so what worked last week dominates what
worked last month -- his "over-repeats whatever recently worked" trait (PRD 4)
is this one number, not a special case in his code.

## Two tables

  `memory`        current beliefs. What the agent consults. One row per pattern.
  `memory_events` every observation that produced them. Append-only, small
                  (one row per settled position), and what lets the dashboard
                  show how a belief formed over time rather than just its
                  current value.
"""

import os
import time
import json
import sqlite3
import threading

# Price bands, in ten-thousandths. Chosen to be coarse near the middle and fine
# at the extremes, because that is where the interesting behaviour is: long
# shots and near-certainties are structurally different bets, and lumping
# everything under 20c together would hide exactly what Cartman is drawn to.
PRICE_BUCKETS = (
    (0, 500, "0-5c"),
    (500, 1000, "5-10c"),
    (1000, 2000, "10-20c"),
    (2000, 4000, "20-40c"),
    (4000, 6000, "40-60c"),
    (6000, 8000, "60-80c"),
    (8000, 9000, "80-90c"),
    (9000, 9500, "90-95c"),
    (9500, 10001, "95-100c"),
)

# Time-to-resolution bands, in seconds.
TTR_BUCKETS = (
    (0, 3600, "under_1h"),
    (3600, 21600, "1-6h"),
    (21600, 86400, "6-24h"),
    (86400, 604800, "1-7d"),
    (604800, 2592000, "1-4w"),
    (2592000, float("inf"), "over_1m"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    pattern_key      TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    label            TEXT,
    encounters       INTEGER NOT NULL DEFAULT 0,
    wins             INTEGER NOT NULL DEFAULT 0,
    losses           INTEGER NOT NULL DEFAULT 0,
    voids            INTEGER NOT NULL DEFAULT 0,
    total_staked     INTEGER NOT NULL DEFAULT 0,
    net_pnl          INTEGER NOT NULL DEFAULT 0,
    sum_entry_price  INTEGER NOT NULL DEFAULT 0,
    ewma_return      REAL,
    best_pnl         INTEGER,
    worst_pnl        INTEGER,
    first_seen       REAL,
    last_seen        REAL
);
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind);
CREATE INDEX IF NOT EXISTS idx_memory_seen ON memory(last_seen);

CREATE TABLE IF NOT EXISTS memory_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    day          TEXT,
    pattern_key  TEXT NOT NULL,
    ticker       TEXT,
    side         TEXT,
    entry_price  INTEGER,
    stake        INTEGER,
    pnl          INTEGER,
    outcome      TEXT,
    return_on_risk REAL
);
CREATE INDEX IF NOT EXISTS idx_events_pattern ON memory_events(pattern_key, ts);
CREATE INDEX IF NOT EXISTS idx_events_day ON memory_events(day);
"""


def price_bucket(price):
    """Price band label, e.g. 4200 -> '40-60c'."""
    if price is None:
        return None
    for low, high, label in PRICE_BUCKETS:
        if low <= price < high:
            return label
    return None


def ttr_bucket(seconds):
    """Time-to-resolution band label."""
    if seconds is None or seconds < 0:
        return None
    for low, high, label in TTR_BUCKETS:
        if low <= seconds < high:
            return label
    return None


def pattern_keys(market, side=None, price=None, now=None):
    """Every pattern a market belongs to, broad first.

    Returns `[(key, kind, label), ...]`. Keys are stable strings so the same
    market produces the same keys tomorrow.

    `entity` comes from the market's subtitle, which is the field that names the
    specific outcome within an event ("Stephen Curry", "Chiefs", "Above 72"),
    rather than from parsing the title. Title parsing would need per-category
    rules for thousands of series and would break silently on the ones it did
    not know about; the subtitle is already the distinguishing label.
    """
    out = []
    category = getattr(market, "category", None)
    series = getattr(market, "series", None)
    subtitle = getattr(market, "subtitle", None)
    suffix = f"|side:{side}" if side else ""
    tag = f" ({side})" if side else ""

    if category:
        out.append((f"category:{category}", "category", category))
        if side:
            out.append((f"category:{category}{suffix}", "category_side",
                        f"{category}{tag}"))
    if series:
        out.append((f"series:{series}", "series", series))
        if side:
            out.append((f"series:{series}{suffix}", "series_side",
                        f"{series}{tag}"))

    band = price_bucket(price if price is not None else getattr(market, "mid", None))
    if series and band:
        out.append((f"series:{series}|price:{band}{suffix}", "series_price",
                    f"{series} @ {band}{tag}"))
    if band:
        out.append((f"price:{band}{suffix}", "price", f"{band}{tag}"))

    if subtitle:
        entity = str(subtitle).strip()[:120]
        if entity:
            out.append((f"entity:{entity}", "entity", entity))
            if side:
                out.append((f"entity:{entity}{suffix}", "entity_side",
                            f"{entity}{tag}"))

    seconds = market.seconds_to_close(now) if hasattr(market, "seconds_to_close") else None
    band = ttr_bucket(seconds)
    if band:
        out.append((f"ttr:{band}", "ttr", band))

    return out


class Belief:
    """One row of the memory bank, as the rest of the code sees it."""

    __slots__ = ("pattern_key", "kind", "label", "encounters", "wins", "losses",
                 "voids", "total_staked", "net_pnl", "sum_entry_price",
                 "ewma_return", "best_pnl", "worst_pnl", "first_seen", "last_seen")

    def __init__(self, **row):
        for name in self.__slots__:
            setattr(self, name, row.get(name))

    @property
    def resolved(self):
        """Encounters that actually produced a win or a loss.

        Voids are excluded: a cancelled market taught the agent nothing.
        """
        return (self.wins or 0) + (self.losses or 0)

    @property
    def win_rate(self):
        """None, not 0.0, when nothing has resolved -- no data is not 0%."""
        return (self.wins / self.resolved) if self.resolved else None

    @property
    def avg_entry_price(self):
        return (self.sum_entry_price / self.encounters) if self.encounters else None

    @property
    def roi(self):
        """Net P&L per dollar staked. The headline "was this worth it" number."""
        return (self.net_pnl / self.total_staked) if self.total_staked else None

    def confidence(self, half=10.0):
        """0..1 measure of how much evidence backs this belief.

        A saturating curve: n/(n+half). At `half` resolved encounters it reads
        0.5 and it approaches 1 slowly after. The policy needs to distinguish
        "60% over 3 tries" from "60% over 200 tries", and Kyle's preference for
        well-established patterns is a preference over exactly this number.
        """
        n = self.resolved
        return n / (n + half) if n else 0.0

    def to_row(self):
        row = {name: getattr(self, name) for name in self.__slots__}
        row.update(win_rate=self.win_rate, roi=self.roi,
                   avg_entry_price=self.avg_entry_price,
                   resolved=self.resolved, confidence=self.confidence())
        return row

    def __repr__(self):
        rate = f"{self.win_rate:.0%}" if self.win_rate is not None else "--"
        return (f"<Belief {self.label!r} n={self.encounters} "
                f"w/l={self.wins}/{self.losses} ({rate}) pnl={self.net_pnl}>")


class MemoryBank:
    """One agent's memory. Backed by SQLite so it survives restarts (PRD 11).

    SQLite rather than a pickle or a JSON blob because the PRD wants this
    *queryable* and directly dashboard-renderable, and because a crash mid-write
    must not corrupt weeks of accumulated learning.
    """

    def __init__(self, path, ewma_alpha=0.2):
        self.path = path
        # How fast recent outcomes displace old ones. Per-agent: see the module
        # docstring -- this single number is Cartman's recency bias.
        self.ewma_alpha = ewma_alpha
        self._lock = threading.RLock()

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL lets the dashboard read while the agent writes, without blocking
        # either -- the dashboard is a read layer over live data (PRD 15).
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ---------- reading ----------

    def get(self, pattern_key):
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM memory WHERE pattern_key = ?", (pattern_key,)).fetchone()
        return Belief(**dict(row)) if row else None

    def lookup(self, keys):
        """Beliefs for many keys at once. Missing keys are simply absent."""
        keys = [k[0] if isinstance(k, tuple) else k for k in keys]
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM memory WHERE pattern_key IN ({placeholders})",
                keys).fetchall()
        return {row["pattern_key"]: Belief(**dict(row)) for row in rows}

    def recall(self, market, side=None, price=None, now=None):
        """The agent consulting its memory about one market.

        Returns `[(kind, Belief)]` for the patterns it has actually seen before,
        broad first. This is the call `agent/features.py` makes on every
        candidate.
        """
        keys = pattern_keys(market, side=side, price=price, now=now)
        found = self.lookup([k for k, _, _ in keys])
        return [(kind, found[key]) for key, kind, _ in keys if key in found]

    def top(self, limit=50, kind=None, order="net_pnl", min_encounters=1):
        """Best-remembered patterns, for the dashboard's memory view."""
        if order not in ("net_pnl", "encounters", "last_seen", "wins"):
            order = "net_pnl"
        sql = ("SELECT * FROM memory WHERE encounters >= ?"
               + (" AND kind = ?" if kind else "")
               + f" ORDER BY {order} DESC LIMIT ?")
        params = [min_encounters] + ([kind] if kind else []) + [limit]
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [Belief(**dict(row)) for row in rows]

    def history(self, pattern_key, limit=200):
        """Every observation behind one belief -- how it formed over time."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memory_events WHERE pattern_key = ? "
                "ORDER BY ts DESC LIMIT ?", (pattern_key, limit)).fetchall()
        return [dict(row) for row in rows]

    def stats(self):
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(encounters),0) e, "
                "COALESCE(SUM(net_pnl),0) p FROM memory").fetchone()
            kinds = self._db.execute(
                "SELECT kind, COUNT(*) n FROM memory GROUP BY kind").fetchall()
        return {"patterns": row["n"], "encounters": row["e"], "net_pnl": row["p"],
                "by_kind": {r["kind"]: r["n"] for r in kinds}}

    # ---------- writing ----------

    def record(self, market, outcome, day=None, side=None, price=None, now=None):
        """Fold one settled position into memory.

        `outcome` is the dict from `sim.settlement.outcome_for_memory`. Every
        pattern the market belongs to is updated, so one resolution teaches the
        agent at several granularities at once.
        """
        now = now or time.time()
        side = side or outcome.get("side")
        price = price if price is not None else outcome.get("entry_price")
        keys = pattern_keys(market, side=side, price=price, now=now)
        if not keys:
            return 0

        pnl = outcome.get("pnl", 0) or 0
        stake = outcome.get("stake", 0) or 0
        ror = outcome.get("return_on_risk", 0.0) or 0.0
        result = outcome.get("outcome")
        won = 1 if result == "win" else 0
        lost = 1 if result == "loss" else 0
        void = 1 if result == "void" else 0

        with self._lock:
            for key, kind, label in keys:
                self._upsert(key, kind, label, pnl, stake, price, ror,
                             won, lost, void, now)
                self._db.execute(
                    "INSERT INTO memory_events (ts, day, pattern_key, ticker, "
                    "side, entry_price, stake, pnl, outcome, return_on_risk) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (now, day, key, outcome.get("ticker"), side, price, stake,
                     pnl, result, ror))
            self._db.commit()
        return len(keys)

    def _upsert(self, key, kind, label, pnl, stake, price, ror,
                won, lost, void, now):
        """Merge one observation into one pattern's row.

        The EWMA is seeded with the first observation rather than starting from
        zero: starting at zero would make a pattern's first big win look
        mediocre and take several more encounters to recover from, which is a
        bias with no justification behind it.
        """
        row = self._db.execute(
            "SELECT ewma_return, best_pnl, worst_pnl FROM memory "
            "WHERE pattern_key = ?", (key,)).fetchone()

        if row is None:
            ewma = ror
            best, worst = pnl, pnl
        else:
            previous = row["ewma_return"]
            ewma = ror if previous is None else \
                (1 - self.ewma_alpha) * previous + self.ewma_alpha * ror
            best = max(row["best_pnl"] if row["best_pnl"] is not None else pnl, pnl)
            worst = min(row["worst_pnl"] if row["worst_pnl"] is not None else pnl, pnl)

        self._db.execute("""
            INSERT INTO memory (pattern_key, kind, label, encounters, wins,
                losses, voids, total_staked, net_pnl, sum_entry_price,
                ewma_return, best_pnl, worst_pnl, first_seen, last_seen)
            VALUES (?,?,?,1,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(pattern_key) DO UPDATE SET
                encounters      = encounters + 1,
                wins            = wins + excluded.wins,
                losses          = losses + excluded.losses,
                voids           = voids + excluded.voids,
                total_staked    = total_staked + excluded.total_staked,
                net_pnl         = net_pnl + excluded.net_pnl,
                sum_entry_price = sum_entry_price + excluded.sum_entry_price,
                ewma_return     = excluded.ewma_return,
                best_pnl        = excluded.best_pnl,
                worst_pnl       = excluded.worst_pnl,
                last_seen       = excluded.last_seen
        """, (key, kind, label, won, lost, void, stake, pnl, price or 0,
              ewma, best, worst, now, now))

    def close(self):
        with self._lock:
            self._db.close()
