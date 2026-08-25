"""Tests for Phase A historical pretraining (PRD 3.1).

Almost everything here is about **leakage**. A settled market's `last_price` is
the post-settlement print (0.9990 / 0.0010, confirmed live), and any historical
price that has seen the outcome produces an agent that looks brilliant in
backtest and clueless live -- with no error and no warning.

The one non-leakage test that matters as much is
`test_a_short_market_still_yields_usable_candles`: a flat one-hour lookahead
buffer silently produced ZERO training examples from 1,000 resolved markets,
because it was four times longer than those markets' entire lifetime.
"""

import pytest

from sim import money
from sim.fills import YES, NO
from agent import pretrain
from agent.pretrain import (safe_candles, candle_price, candle_minutes,
                            lookahead_buffer, HistoricalMarket)


def _candle(end_ts, bid=0.40, ask=0.44, close=0.42):
    return {"end_period_ts": end_ts,
            "yes_bid": {"close_dollars": str(bid)},
            "yes_ask": {"close_dollars": str(ask)},
            "price": {"close_dollars": str(close)}}


# ---------- the leakage guard ----------

def test_candles_after_the_cutoff_are_discarded():
    """A candle near resolution may already reflect the outcome."""
    close_ts = 1_000_000
    candles = [_candle(close_ts - 5000), _candle(close_ts - 100),
               _candle(close_ts + 50)]
    usable = safe_candles(candles, close_ts, buffer_s=1000)
    assert len(usable) == 1
    assert usable[0]["end_period_ts"] == close_ts - 5000


def test_a_candle_exactly_on_the_cutoff_is_allowed():
    close_ts = 1_000_000
    assert len(safe_candles([_candle(close_ts - 1000)], close_ts,
                            buffer_s=1000)) == 1


def test_no_candles_survive_without_a_close_time():
    """Unknown resolution time means we cannot prove anything is safe."""
    assert safe_candles([_candle(500)], None) == []


def test_a_candle_with_no_timestamp_is_dropped():
    assert safe_candles([{"yes_bid": {"close_dollars": "0.4"}}], 1000) == []


def test_the_buffer_scales_with_market_lifetime():
    """The bug that produced zero examples from a thousand markets.

    A 15-minute market cannot afford a one-hour buffer; a three-month election
    should not be judged on the last sixty seconds.
    """
    fifteen_minutes = lookahead_buffer(15 * 60)
    assert fifteen_minutes < 15 * 60
    assert fifteen_minutes >= pretrain.MIN_BUFFER_S

    quarterly = lookahead_buffer(90 * 86400)
    assert quarterly == pretrain.MAX_BUFFER_S


def test_the_buffer_survives_nonsense_durations():
    assert lookahead_buffer(0) == pretrain.MIN_BUFFER_S
    assert lookahead_buffer(None) == pretrain.MIN_BUFFER_S
    assert lookahead_buffer(-5) == pretrain.MIN_BUFFER_S


def test_a_short_market_still_yields_usable_candles():
    """The regression, end to end: a 15-minute market must produce something."""
    close_ts = 1_000_000
    duration = 15 * 60
    candles = [_candle(close_ts - duration + i * 60) for i in range(15)]
    usable = safe_candles(candles, close_ts,
                          buffer_s=lookahead_buffer(duration))
    assert usable, "a 15-minute market produced no usable candles"


def test_candle_granularity_adapts_to_duration():
    """Kalshi accepts 1, 60, or 1440 minutes; hourly candles on a 15-minute
    market give you at most one data point."""
    assert candle_minutes(15 * 60) == 1
    assert candle_minutes(3 * 86400) == 60
    assert candle_minutes(120 * 86400) == 1440


def test_pretraining_never_reads_last_price_off_a_kalshi_row():
    """Mechanically enforced, because a comment would not survive a refactor.

    What is forbidden is *reading* the field from an API response -- on a
    settled market that value is the post-settlement print, i.e. the answer.
    Declaring a slot of the same name on our own synthetic object is fine.
    """
    import io
    import os
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "agent", "pretrain.py")
    code = io.open(path, encoding="utf-8").read().split('"""', 2)[-1]

    # row.get("last_price"), row["last_price_dollars"], last_price_dollars ...
    forbidden = re.compile(r"""\.get\(\s*["']last_price"""
                           r"""|\[\s*["']last_price"""
                           r"""|last_price_dollars""")
    offenders = [line.strip() for line in code.splitlines()
                 if not line.strip().startswith("#") and forbidden.search(line)]
    assert not offenders, ("pretraining must never read last_price from a "
                           "Kalshi response: " + "; ".join(offenders))


# ---------- prices ----------

def test_price_comes_from_the_candle_mid():
    assert candle_price(_candle(1, bid=0.40, ask=0.44)) == 4200


def test_price_falls_back_to_the_close_when_a_side_is_missing():
    candle = {"end_period_ts": 1, "yes_bid": {}, "yes_ask": {},
              "price": {"close_dollars": "0.37"}}
    assert candle_price(candle) == 3700


def test_an_empty_candle_has_no_price():
    assert candle_price({"end_period_ts": 1}) is None


# ---------- the historical market stands in for a live one ----------

def _historical(price=4200, close_ts=1_000_000, observed=999_000):
    return HistoricalMarket(
        {"ticker": "KXTEST-1", "result": "yes", "_close_ts": close_ts,
         "title": "A resolved thing", "yes_sub_title": "Someone"},
        price, observed)


def test_it_exposes_everything_features_needs():
    """Duck-typed on purpose: pretraining must use the same feature code path
    as live trading, or it is training on a different problem."""
    from agent.features import build, N_FEATURES
    market = _historical()
    vector = build(market, YES, memory=None, now=market.updated_at)
    assert vector is not None and len(vector) == N_FEATURES


def test_the_synthetic_spread_is_never_free():
    """The candle gives a mid, not a book. Entry should cost slightly more than
    the mid, never less -- erring against the agent, not for it."""
    market = _historical(price=4200)
    assert market.yes_ask > 4200 > market.yes_bid
    assert market.is_quoted


def test_time_to_close_is_measured_from_the_observation():
    """Using wall-clock now would make every historical market read as expired,
    flattening a real feature to zero across the whole training set."""
    market = _historical(close_ts=1_000_000, observed=999_000)
    assert market.seconds_to_close() == pytest.approx(1000)


def test_prices_stay_inside_the_contract_range():
    for price in (1, 50, 9950, money.ONE_DOLLAR - 1):
        market = _historical(price=price)
        assert 0 < market.yes_bid < money.ONE_DOLLAR
        assert 0 < market.yes_ask < money.ONE_DOLLAR


# ---------- training ----------

class _FakeAgent:
    """Minimal stand-in with the two things `pretrain` writes to."""

    def __init__(self, personality):
        from agent.memory import MemoryBank
        from agent.policy import LinearQ
        import tempfile
        import os
        self.name = personality.name
        self.p = personality
        self.memory = MemoryBank(os.path.join(tempfile.mkdtemp(), "m.db"))
        self.policy = LinearQ(seed=1)
        self.saved = False

    def save(self):
        self.saved = True


def test_pretraining_trains_the_model_and_fills_memory(monkeypatch):
    from agent.personality import load_all
    agent = _FakeAgent(load_all()["stan"])

    def fake_load_series(series_ticker, **kwargs):
        for i in range(12):
            yield _historical(price=3000 + i * 100), "yes"

    monkeypatch.setattr(pretrain, "load_series", fake_load_series)
    summary = pretrain.pretrain(agent, ["KXTEST"], rng=__import__("random").Random(1))

    assert summary["observations"] == 12
    assert summary["wins"] + summary["losses"] == 12
    assert agent.policy.updates == 12
    assert summary["patterns"] > 0
    assert agent.saved is True
    agent.memory.close()


def test_agents_draw_different_lessons_from_identical_history(monkeypatch):
    """Kyle's heavy exposure penalty and Cartman's light one make the same
    resolved market a different training target for each."""
    from agent.personality import load_all
    import random

    def fake_load_series(series_ticker, **kwargs):
        for _ in range(30):
            yield _historical(price=2000), "yes"

    monkeypatch.setattr(pretrain, "load_series", fake_load_series)

    weights = {}
    for name in ("kyle", "cartman"):
        agent = _FakeAgent(load_all()[name])
        pretrain.pretrain(agent, ["KXTEST"], rng=random.Random(4))
        weights[name] = agent.policy.named_weights()["bias"]
        agent.memory.close()

    assert weights["kyle"] != weights["cartman"]
