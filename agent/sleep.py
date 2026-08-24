"""Personality-driven sleep schedules (PRD 8).

Each agent has a recurring daily window in which it takes no action at all,
simulating a human's offline hours. The windows are individual and deliberately
not synchronized -- Cartman sleeps longest and rises late, Kenny barely sleeps
and sometimes skips it entirely.

This is not decoration. Time spent awake is a genuine confound when comparing
agents: an agent that is awake 20 hours a day gets more opportunities than one
awake 12, so the dashboard surfaces both current status and historical awake
time (PRD 8) and the head-to-head view can normalize for it.

## Schedules are per-day deterministic

Kenny's schedule varies day to day, which means "is Kenny asleep?" must not be
answered by a fresh coin flip each time it is asked -- it would flicker on every
poll and the dashboard would be nonsense. So the day's schedule is derived from
a hash of `(agent name, date)`: stable for the whole day, different tomorrow,
and identical across restarts. No state to persist.

## Windows can wrap past midnight

Cartman sleeps from 03:00 for eleven hours; Kenny might start at 23:00. A window
that crosses midnight is the normal case, not an edge case, so `is_asleep`
handles wrap directly rather than assuming start < end.
"""

import hashlib
import datetime as dt
from dataclasses import dataclass, asdict


@dataclass
class SleepSchedule:
    """When one agent is offline.

    `start_hour` may be fractional. `jitter_hours` is the maximum daily drift in
    either direction -- Kyle's is near zero (the most disciplined of the four),
    Kenny's is large.
    """

    start_hour: float = 23.0
    duration_hours: float = 8.0
    jitter_hours: float = 0.0
    skip_probability: float = 0.0

    @classmethod
    def from_dict(cls, data):
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in (data or {})}
        return cls(**known)

    def to_dict(self):
        return asdict(self)


def _day_seed(agent, day):
    """Stable pseudo-random value in [0, 1) for one agent on one date.

    A hash rather than a seeded RNG object so it is stateless: the same answer
    on every call, across restarts, without anything to persist.
    """
    raw = f"{agent}:{day}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def schedule_for(agent, schedule, day):
    """This agent's actual sleep window on `day`.

    Returns `(start_hour, end_hour, skipped)` where hours are floats in [0, 24)
    and `end_hour` may be numerically less than `start_hour` when the window
    wraps past midnight.
    """
    roll = _day_seed(agent, day)

    # Two independent draws from one seed: one for skipping, one for drift.
    if schedule.skip_probability and roll < schedule.skip_probability:
        return (schedule.start_hour, schedule.start_hour, True)

    drift_roll = _day_seed(agent, f"{day}:drift")
    drift = (drift_roll * 2.0 - 1.0) * schedule.jitter_hours

    start = (schedule.start_hour + drift) % 24.0
    end = (start + schedule.duration_hours) % 24.0
    return (start, end, False)


def is_asleep(agent, schedule, when=None):
    """True if `agent` is offline right now.

    `when` is a datetime in local time -- the same clock the user's machine runs
    on, since the whole point is to model the agent's day the way a person's day
    works.
    """
    when = when or dt.datetime.now()
    day = when.date().isoformat()
    start, end, skipped = schedule_for(agent, schedule, day)
    if skipped or schedule.duration_hours <= 0:
        return False

    hour = when.hour + when.minute / 60.0 + when.second / 3600.0

    if start <= end:
        return start <= hour < end
    # Window wraps past midnight: asleep either late tonight or early tomorrow.
    return hour >= start or hour < end


def status(agent, schedule, when=None):
    """Awake/asleep plus the window, for the dashboard."""
    when = when or dt.datetime.now()
    day = when.date().isoformat()
    start, end, skipped = schedule_for(agent, schedule, day)
    asleep = is_asleep(agent, schedule, when)
    return {
        "agent": agent,
        "asleep": asleep,
        "skipped_sleep": skipped,
        "window_start": round(start, 2),
        "window_end": round(end, 2),
        "duration_hours": 0.0 if skipped else schedule.duration_hours,
        "awake_hours": 24.0 if skipped else round(24.0 - schedule.duration_hours, 2),
    }


def awake_hours_on(agent, schedule, day):
    """How many hours this agent was awake on `day`. For fair comparisons."""
    _, _, skipped = schedule_for(agent, schedule, day)
    return 24.0 if skipped else max(0.0, 24.0 - schedule.duration_hours)
