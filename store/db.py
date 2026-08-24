"""Comprehensive activity logging, from day one.

PRD 10 step 2 is explicit that this is **not** deferred until the dashboard
exists: everything the dashboard will eventually display gets written from the
first run, so when the dashboard is built all the historical learning data is
already there and can be viewed retroactively. Building it later would mean
throwing away the most interesting period -- the first days, when the agents
know nothing.

## What is written, and what deliberately is not

Written: every decision including the decision **not** to act (PRD 9 requires
explicitly logged non-actions in the activity feed), every position open and
close, every daily episode with its reward broken down by term, and notable
events like bankruptcies and stream disconnections.

Not written: the raw firehose. At 1,113 messages/second it would cost 37.9
GB/day, and it is the least interesting data in the system -- PRD 6 says the
observable artifact is what agents *remember* and *do*. Market state is held in
memory and only the slice an agent actually acted on is persisted.

## Sizing

At one decision per minute per agent, four agents produce about 5,760 decision
rows a day -- a few MB a month. Positions and daily rows are far rarer. The one
table that could grow without bound is `decisions`, so it has retention (see
`prune`), which PRD 15 asks to be in place from the start rather than bolted on
once the disk is full.

One SQLite file for all agents, because the dashboard's head-to-head view wants
to query across them. Their *learning* stays strictly separate -- memory banks
and weights are per-agent files (PRD 2) -- but a shared activity log is just
bookkeeping, not shared experience.
"""

import os
import json
import time
import sqlite3
import threading
import datetime as dt

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    day         TEXT NOT NULL,
    agent       TEXT NOT NULL,
    acted       INTEGER NOT NULL,
    explored    INTEGER NOT NULL DEFAULT 0,
    reason      TEXT,
    ticker      TEXT,
    side        TEXT,
    q           REAL,
    best_q      REAL,
    considered  INTEGER,
    category    TEXT,
    series      TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_agent_ts ON decisions(agent, ts);
CREATE INDEX IF NOT EXISTS idx_decisions_day ON decisions(day, agent);

CREATE TABLE IF NOT EXISTS positions (
    id             TEXT PRIMARY KEY,
    agent          TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    title          TEXT,
    category       TEXT,
    series         TEXT,
    side           TEXT NOT NULL,
    contracts      INTEGER,
    entry_price    INTEGER,
    cost           INTEGER,
    entry_fee      INTEGER,
    opened_at      REAL,
    opened_day     TEXT,
    basket_id      TEXT,
    close_ts       REAL,
    status         TEXT,
    exit_price     INTEGER,
    exit_proceeds  INTEGER,
    exit_fee       INTEGER,
    closed_at      REAL,
    closed_day     TEXT,
    realized_pnl   INTEGER,
    result         TEXT,
    q_at_entry     REAL,
    explored       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_positions_agent ON positions(agent, opened_at);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(agent, status);
CREATE INDEX IF NOT EXISTS idx_positions_closed_day ON positions(closed_day, agent);

CREATE TABLE IF NOT EXISTS daily (
    day            TEXT NOT NULL,
    agent          TEXT NOT NULL,
    reward         REAL,
    reward_parts   TEXT,
    realized_pnl   INTEGER,
    trades         INTEGER,
    bankroll       INTEGER,
    equity         INTEGER,
    drawdown       REAL,
    open_positions INTEGER,
    win_rate       REAL,
    bankruptcies   INTEGER,
    awake_hours    REAL,
    exploration_rate REAL,
    mean_abs_error REAL,
    updates        INTEGER,
    weights        TEXT,
    PRIMARY KEY (day, agent)
);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    day     TEXT,
    agent   TEXT,
    kind    TEXT NOT NULL,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent, ts);
"""

# Decisions older than this are pruned. Positions, daily rows, and events are
# kept forever -- they are the actual history and they are small.
DECISION_RETENTION_DAYS = 45


def today():
    """The episode key. One real trading day is one episode (PRD 3.2)."""
    return dt.date.today().isoformat()


class Store:
    """Activity log for all agents. Thread-safe; several agent loops write."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL so the dashboard reads while agents write. PRD 15: closing the
        # browser tab must not affect the simulation, and vice versa.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ---------- writing ----------

    def log_decision(self, agent, decision, market=None, side=None, day=None):
        """Record one decision -- including the decision to do nothing.

        Non-actions are first-class here because PRD 9 wants them in the
        activity feed, and because "Stan looked at 40 markets and passed on all
        of them" is genuinely informative about what he has learned.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO decisions (ts, day, agent, acted, explored, reason, "
                "ticker, side, q, best_q, considered, category, series) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), day or today(), agent, int(decision.acted),
                 int(decision.explored), decision.skipped_reason,
                 market.ticker if market else None, side,
                 decision.q, decision.best_q, decision.considered,
                 market.category if market else None,
                 market.series if market else None))
            self._db.commit()

    def upsert_position(self, position, title=None, q_at_entry=None,
                        explored=None):
        """Write or update one position. Called on open and again on close."""
        row = position.to_row()
        with self._lock:
            self._db.execute("""
                INSERT INTO positions (id, agent, ticker, title, category, series,
                    side, contracts, entry_price, cost, entry_fee, opened_at,
                    opened_day, basket_id, close_ts, status, exit_price,
                    exit_proceeds, exit_fee, closed_at, closed_day, realized_pnl,
                    result, q_at_entry, explored)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    exit_price = excluded.exit_price,
                    exit_proceeds = excluded.exit_proceeds,
                    exit_fee = excluded.exit_fee,
                    closed_at = excluded.closed_at,
                    closed_day = excluded.closed_day,
                    realized_pnl = excluded.realized_pnl,
                    result = excluded.result
            """, (row["id"], row["agent"], row["ticker"], title, row["category"],
                  row["series"], row["side"], row["contracts"], row["entry_price"],
                  row["cost"], row["entry_fee"], row["opened_at"], row["opened_day"],
                  row["basket_id"], row["close_ts"], row["status"], row["exit_price"],
                  row["exit_proceeds"], row["exit_fee"], row["closed_at"],
                  row["closed_day"], row["realized_pnl"], row["result"],
                  q_at_entry, int(bool(explored))))
            self._db.commit()

    def log_daily(self, day, agent, **fields):
        """One episode summary. Upserted, so a mid-day crash does not lose it."""
        parts = fields.get("reward_parts")
        weights = fields.get("weights")
        with self._lock:
            self._db.execute("""
                INSERT INTO daily (day, agent, reward, reward_parts, realized_pnl,
                    trades, bankroll, equity, drawdown, open_positions, win_rate,
                    bankruptcies, awake_hours, exploration_rate, mean_abs_error,
                    updates, weights)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(day, agent) DO UPDATE SET
                    reward=excluded.reward, reward_parts=excluded.reward_parts,
                    realized_pnl=excluded.realized_pnl, trades=excluded.trades,
                    bankroll=excluded.bankroll, equity=excluded.equity,
                    drawdown=excluded.drawdown,
                    open_positions=excluded.open_positions,
                    win_rate=excluded.win_rate, bankruptcies=excluded.bankruptcies,
                    awake_hours=excluded.awake_hours,
                    exploration_rate=excluded.exploration_rate,
                    mean_abs_error=excluded.mean_abs_error,
                    updates=excluded.updates, weights=excluded.weights
            """, (day, agent, fields.get("reward"),
                  json.dumps(parts) if parts is not None else None,
                  fields.get("realized_pnl"), fields.get("trades"),
                  fields.get("bankroll"), fields.get("equity"),
                  fields.get("drawdown"), fields.get("open_positions"),
                  fields.get("win_rate"), fields.get("bankruptcies"),
                  fields.get("awake_hours"), fields.get("exploration_rate"),
                  fields.get("mean_abs_error"), fields.get("updates"),
                  json.dumps(weights) if weights is not None else None))
            self._db.commit()

    def log_event(self, kind, agent=None, detail=None, day=None):
        """Anything notable: bankruptcies, disconnects, sleep transitions, resets."""
        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts, day, agent, kind, detail) VALUES (?,?,?,?,?)",
                (time.time(), day or today(), agent, kind,
                 json.dumps(detail) if detail is not None else None))
            self._db.commit()

    # ---------- reading (the dashboard's queries) ----------

    def recent_decisions(self, agent=None, limit=100):
        sql = "SELECT * FROM decisions"
        params = []
        if agent:
            sql += " WHERE agent = ?"
            params.append(agent)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]

    def open_positions(self, agent=None):
        sql = "SELECT * FROM positions WHERE status = 'open'"
        params = []
        if agent:
            sql += " AND agent = ?"
            params.append(agent)
        sql += " ORDER BY opened_at DESC"
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]

    def recent_positions(self, agent=None, limit=100):
        sql = "SELECT * FROM positions"
        params = []
        if agent:
            sql += " WHERE agent = ?"
            params.append(agent)
        sql += " ORDER BY opened_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]

    def daily_series(self, agent=None, limit=365):
        """The learning curve: is performance actually improving over time?"""
        sql = "SELECT * FROM daily"
        params = []
        if agent:
            sql += " WHERE agent = ?"
            params.append(agent)
        sql += " ORDER BY day DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = [dict(r) for r in self._db.execute(sql, params).fetchall()]
        for row in rows:
            if row.get("reward_parts"):
                row["reward_parts"] = json.loads(row["reward_parts"])
            if row.get("weights"):
                row["weights"] = json.loads(row["weights"])
        return list(reversed(rows))

    def category_breakdown(self, agent):
        """Which markets does this agent gravitate toward? (PRD 9)"""
        with self._lock:
            rows = self._db.execute(
                "SELECT category, COUNT(*) n, "
                "COALESCE(SUM(realized_pnl), 0) pnl, "
                "SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) wins, "
                "SUM(CASE WHEN result IN ('win','loss') THEN 1 ELSE 0 END) resolved "
                "FROM positions WHERE agent = ? GROUP BY category ORDER BY n DESC",
                (agent,)).fetchall()
        return [dict(r) for r in rows]

    def bet_size_distribution(self, agent, buckets=10):
        with self._lock:
            rows = self._db.execute(
                "SELECT cost FROM positions WHERE agent = ?", (agent,)).fetchall()
        return [r["cost"] for r in rows]

    def exploration_over_time(self, agent, days=60):
        """Exploration-vs-exploitation ratio over time (PRD 9)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT day, "
                "SUM(explored) explored, COUNT(*) total "
                "FROM decisions WHERE agent = ? AND acted = 1 "
                "GROUP BY day ORDER BY day DESC LIMIT ?", (agent, days)).fetchall()
        return list(reversed([dict(r) for r in rows]))

    def recent_events(self, limit=50, agent=None):
        sql = "SELECT * FROM events"
        params = []
        if agent:
            sql += " WHERE agent = ?"
            params.append(agent)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = [dict(r) for r in self._db.execute(sql, params).fetchall()]
        for row in rows:
            if row.get("detail"):
                row["detail"] = json.loads(row["detail"])
        return rows

    def agent_names(self):
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT agent FROM positions "
                "UNION SELECT DISTINCT agent FROM decisions").fetchall()
        return sorted(r["agent"] for r in rows if r["agent"])

    # ---------- housekeeping ----------

    def prune(self, decision_days=DECISION_RETENTION_DAYS):
        """Drop decision rows past the retention window.

        Retention exists from the first run rather than being added once the
        disk is full (PRD 15). Positions, daily summaries, and events are never
        pruned -- they are the history the project is actually about, and they
        are small.
        """
        cutoff = (dt.date.today() - dt.timedelta(days=decision_days)).isoformat()
        with self._lock:
            cursor = self._db.execute("DELETE FROM decisions WHERE day < ?", (cutoff,))
            self._db.commit()
        return cursor.rowcount

    def size_on_disk(self):
        total = 0
        for suffix in ("", "-wal", "-shm"):
            path = self.path + suffix
            if os.path.exists(path):
                total += os.path.getsize(path)
        return total

    def stats(self):
        with self._lock:
            counts = {
                table: self._db.execute(f"SELECT COUNT(*) n FROM {table}").fetchone()["n"]
                for table in ("decisions", "positions", "daily", "events")
            }
        counts["bytes_on_disk"] = self.size_on_disk()
        return counts

    def close(self):
        with self._lock:
            self._db.close()
