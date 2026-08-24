"""The dashboard: a read layer over what the agents are writing.

PRD 9 wants five views -- overview, activity, memory bank, analytics, and a
four-way head-to-head -- plus a kill switch (PRD 11).

Two rules this server follows:

  **It never writes agent state.** The only write endpoint is the kill switch.
  Everything else is a read against SQLite (opened WAL, so the dashboard reads
  while agents write and neither blocks the other).

  **Closing the tab changes nothing.** The agents are writing to disk
  continuously whether or not anyone is looking; this just renders what is
  already there (PRD 15). Reopening it later shows what happened in between,
  which is why PRD 10 insists the logging exists from day one rather than
  arriving with the dashboard.

Money crosses into JSON as float dollars. Everything inside the system is
integer ten-thousandths (see `sim/money.py`); converting at this boundary keeps
the display code from having to know that.
"""

import os
import logging

from flask import Flask, jsonify, request, send_from_directory

from sim import money

log = logging.getLogger("dashboard")

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _dollars(value):
    return None if value is None else round(value / money.ONE_DOLLAR, 2)


def _position_row(row):
    """One position, with money converted for display."""
    out = dict(row)
    for field in ("cost", "entry_fee", "exit_proceeds", "exit_fee", "realized_pnl"):
        out[field] = _dollars(row.get(field))
    for field in ("entry_price", "exit_price"):
        value = row.get(field)
        out[field] = None if value is None else round(value / money.ONE_DOLLAR, 4)
    return out


def create_app(system):
    app = Flask(__name__, static_folder=None)
    store = system.store

    # ---------- pages ----------

    @app.route("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.route("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name)

    # ---------- live state ----------

    @app.route("/api/state")
    def api_state():
        """Everything the overview and head-to-head views need, in one poll."""
        snap = system.snapshot()
        for agent in snap["agents"]:
            for field in ("bankroll", "equity", "exposure", "peak_equity",
                          "all_time_realized", "realized_today", "total_fees"):
                agent[field] = _dollars(agent.get(field))
        return jsonify(snap)

    @app.route("/api/agents")
    def api_agents():
        """Static personality config -- what makes each agent different."""
        return jsonify([a.p.summary() for a in system.agents])

    # ---------- activity ----------

    @app.route("/api/activity/<agent>")
    def api_activity(agent):
        """The chronological feed, including explicitly logged non-actions.

        PRD 9 wants decisions *not* to act in the feed. "Stan looked at 40
        markets and passed on all of them" is genuinely informative about what
        he has learned, and hiding it would make an idle agent look broken.
        """
        limit = min(int(request.args.get("limit", 100)), 500)
        return jsonify({
            "decisions": store.recent_decisions(agent, limit),
            "positions": [_position_row(r)
                          for r in store.recent_positions(agent, limit)],
            "events": store.recent_events(limit=50, agent=agent),
        })

    @app.route("/api/positions/<agent>")
    def api_positions(agent):
        """Open positions, marked to market. Unrealized, clearly separated."""
        found = _agent(agent)
        rows = []
        for position in (found.portfolio.open_positions() if found else []):
            price = system._price_of(position)
            rows.append({
                "id": position.id,
                "ticker": position.ticker,
                "side": position.side,
                "contracts": position.contracts,
                "entry_price": round(position.entry_price / money.ONE_DOLLAR, 4),
                "current_price": (None if price is None
                                  else round(price / money.ONE_DOLLAR, 4)),
                "cost": _dollars(position.cost),
                "unrealized_pnl": (None if price is None
                                   else _dollars(position.unrealized_pnl(price))),
                "category": position.category,
                "series": position.series,
                "opened_at": position.opened_at,
                "opened_day": position.opened_day,
                "close_ts": position.close_ts,
                "stake_fraction": round(position.stake_fraction, 4),
            })
        return jsonify(rows)

    # ---------- the memory bank: the centerpiece ----------

    @app.route("/api/memory/<agent>")
    def api_memory(agent):
        """What this agent has learned, and how strongly it believes it."""
        found = _agent(agent)
        if not found:
            return jsonify({"error": "unknown agent"}), 404
        kind = request.args.get("kind") or None
        order = request.args.get("order", "encounters")
        limit = min(int(request.args.get("limit", 60)), 300)
        beliefs = found.memory.top(limit=limit, kind=kind, order=order)
        rows = []
        for belief in beliefs:
            row = belief.to_row()
            row["net_pnl"] = _dollars(row["net_pnl"])
            row["total_staked"] = _dollars(row["total_staked"])
            row["best_pnl"] = _dollars(row["best_pnl"])
            row["worst_pnl"] = _dollars(row["worst_pnl"])
            avg = row.get("avg_entry_price")
            row["avg_entry_price"] = (None if avg is None
                                      else round(avg / money.ONE_DOLLAR, 4))
            rows.append(row)
        return jsonify({"beliefs": rows, "stats": found.memory.stats()})

    @app.route("/api/memory/<agent>/history")
    def api_memory_history(agent):
        """How one belief formed over time, rather than just its current value."""
        found = _agent(agent)
        key = request.args.get("key")
        if not found or not key:
            return jsonify([])
        rows = []
        for row in found.memory.history(key, limit=200):
            row["pnl"] = _dollars(row["pnl"])
            row["stake"] = _dollars(row["stake"])
            rows.append(row)
        return jsonify(rows)

    # ---------- analytics ----------

    @app.route("/api/analytics/<agent>")
    def api_analytics(agent):
        found = _agent(agent)
        daily = store.daily_series(agent, limit=365)
        for row in daily:
            for field in ("realized_pnl", "bankroll", "equity"):
                row[field] = _dollars(row.get(field))

        categories = store.category_breakdown(agent)
        for row in categories:
            row["pnl"] = _dollars(row["pnl"])
            row["win_rate"] = ((row["wins"] / row["resolved"])
                               if row["resolved"] else None)

        return jsonify({
            "daily": daily,
            "categories": categories,
            "bet_sizes": [_dollars(v) for v in store.bet_size_distribution(agent)],
            "exploration": store.exploration_over_time(agent),
            "calibration": _calibration(store, agent),
            "streaks": _streaks(store, agent),
            "hold_times": _hold_times(store, agent),
            "weights": (found.policy.named_weights() if found else {}),
            "policy": (found.policy.stats() if found else {}),
        })

    # ---------- the kill switch (PRD 11) ----------

    @app.route("/api/kill", methods=["POST"])
    def api_kill():
        """Pause or resume every agent immediately.

        The one write endpoint. It stops agents from *trading*; the stream keeps
        running so the universe stays warm and the dashboard stays live.
        """
        action = (request.get_json(silent=True) or {}).get("action", "toggle")
        if action == "pause" or (action == "toggle" and not system.kill.paused):
            system.kill.pause()
            store.log_event("kill_switch", detail={"paused": True})
        else:
            system.kill.resume()
            store.log_event("kill_switch", detail={"paused": False})
        return jsonify({"paused": system.kill.paused})

    # ---------- helpers ----------

    def _agent(name):
        return next((a for a in system.agents if a.name == name), None)

    return app


def _calibration(store, agent, buckets=10):
    """When an agent bets at an implied X%, does it win about X% of the time?

    PRD 9 asks for this explicitly, and it is the sharpest single measure of
    whether an agent understands anything: an agent that is well calibrated is
    reading prices correctly even when it is not profitable.
    """
    rows = [r for r in store.recent_positions(agent, limit=5000)
            if r["result"] in ("win", "loss") and r["entry_price"]]
    out = []
    for i in range(buckets):
        low, high = i / buckets, (i + 1) / buckets
        band = [r for r in rows
                if low <= r["entry_price"] / money.ONE_DOLLAR < high]
        if not band:
            continue
        wins = sum(1 for r in band if r["result"] == "win")
        out.append({
            "implied": round((low + high) / 2, 3),
            "actual": round(wins / len(band), 3),
            "n": len(band),
        })
    return out


def _streaks(store, agent):
    """Longest winning and losing runs (PRD 9)."""
    rows = [r for r in reversed(store.recent_positions(agent, limit=5000))
            if r["result"] in ("win", "loss")]
    best = worst = current = 0
    for row in rows:
        if row["result"] == "win":
            current = current + 1 if current > 0 else 1
            best = max(best, current)
        else:
            current = current - 1 if current < 0 else -1
            worst = min(worst, current)
    return {"longest_win": best, "longest_loss": -worst, "current": current}


def _hold_times(store, agent):
    """Average time a position is held, split by how it ended.

    Kenny should be visibly faster than Kyle here; if he is not, the config is
    not doing what it claims.
    """
    rows = [r for r in store.recent_positions(agent, limit=5000)
            if r["closed_at"] and r["opened_at"]]
    if not rows:
        return {"mean_seconds": None, "n": 0}
    durations = [r["closed_at"] - r["opened_at"] for r in rows]
    exits = [r["closed_at"] - r["opened_at"] for r in rows if r["result"] == "exit"]
    return {
        "mean_seconds": round(sum(durations) / len(durations), 1),
        "mean_exit_seconds": (round(sum(exits) / len(exits), 1) if exits else None),
        "n": len(durations),
    }


def serve(system, host="127.0.0.1", port=8000):
    """Run the dashboard. Called on a daemon thread from run.py."""
    app = create_app(system)
    # Flask's dev server is right here: one local user, no traffic, and pulling
    # in a production WSGI server would be a dependency for nothing.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host=host, port=port, threaded=True, use_reloader=False)
