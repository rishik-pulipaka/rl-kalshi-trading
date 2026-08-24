"""Tests for the live market universe.

These are all offline -- the universe is fed synthetic REST rows and synthetic
WebSocket frames, so the suite never touches the network and runs in
milliseconds. Anything that needs the real API belongs in tools/, not here.

The cases below encode two bugs that were caught during the build, both of which
produced an empty candidate pool while every summary count looked healthy. They
are the reason this file exists.
"""

import time

from kalshi import universe
from kalshi.universe import Market, Universe, to_price_key


# ---------- price conversion ----------

def test_to_price_key_converts_dollar_strings():
    assert to_price_key("0.4200") == 4200
    assert to_price_key("0.0060") == 60
    assert to_price_key("1.0000") == 10000


def test_to_price_key_handles_missing_values():
    assert to_price_key(None) is None
    assert to_price_key("") is None
    assert to_price_key("not-a-number") is None


# ---------- quote semantics ----------

def _market(bid, ask, status=universe.TRADEABLE_STATUS, close_ts=None):
    m = Market("M")
    m.yes_bid, m.yes_ask, m.status = bid, ask, status
    m.close_ts = close_ts
    return m


def test_zero_bid_is_not_a_quote():
    """Kalshi sends "0.0000" for an absent bid rather than omitting the field.

    Treating that as a quote (a plain None-check does) counts every untraded
    market as two-sided, which is how `quoted` once reported 96,350 out of
    96,350 markets.
    """
    m = _market(bid=0, ask=1200)
    assert m.can_buy_yes is True      # someone is offering
    assert m.can_buy_no is False      # nobody is bidding
    assert m.is_quoted is False       # so it is not two-sided
    assert m.is_tradeable is True     # but you can still buy YES


def test_full_price_is_not_a_quote():
    m = _market(bid=10000, ask=10000)
    assert m.can_buy_yes is False
    assert m.can_buy_no is False
    assert m.is_tradeable is False


def test_two_sided_market_is_quoted():
    m = _market(bid=4000, ask=4200)
    assert m.is_quoted is True
    assert m.is_tradeable is True
    assert m.mid == 4100
    assert m.spread == 200


def test_mid_and_spread_need_both_sides():
    m = _market(bid=None, ask=4200)
    assert m.mid is None
    assert m.spread is None


# ---------- the tradeable pool ----------

def test_tradeable_uses_kalshi_active_status_not_open():
    """Kalshi's open market status is "active"; "open" describes the *event*.

    Filtering on "open" silently yields an empty candidate pool -- every market
    fails, no error is raised, and the agent simply never trades.
    """
    u = Universe()
    active = _market(4000, 4200, status="active")
    active.ticker = "ACTIVE"
    finalized = _market(4000, 4200, status="finalized")
    finalized.ticker = "FINALIZED"
    u._markets = {"ACTIVE": active, "FINALIZED": finalized}

    pool = u.tradeable()
    assert [m.ticker for m in pool] == ["ACTIVE"]


def test_tradeable_excludes_markets_past_their_close():
    u = Universe()
    now = time.time()
    live = _market(4000, 4200, close_ts=now + 3600)
    live.ticker = "LIVE"
    expired = _market(4000, 4200, close_ts=now - 3600)
    expired.ticker = "EXPIRED"
    u._markets = {"LIVE": live, "EXPIRED": expired}

    assert [m.ticker for m in u.tradeable(now=now)] == ["LIVE"]


def test_tradeable_keeps_markets_with_no_close_time():
    u = Universe()
    m = _market(4000, 4200, close_ts=None)
    m.ticker = "NOCLOSE"
    u._markets = {"NOCLOSE": m}
    assert len(u.tradeable()) == 1


# ---------- applying the live stream ----------

def _ticker_frame(ticker="M", bid="0.4000", ask="0.4200", volume="10.00",
                  oi="5.00", price="0.4100"):
    return {"type": "ticker", "msg": {
        "market_ticker": ticker, "yes_bid_dollars": bid, "yes_ask_dollars": ask,
        "yes_bid_size_fp": "100.00", "yes_ask_size_fp": "200.00",
        "price_dollars": price, "volume_fp": volume, "open_interest_fp": oi}}


def test_ticker_frame_updates_quote():
    u = Universe()
    u._markets = {"M": Market("M")}
    u.on_message(_ticker_frame())
    m = u.get("M")
    assert (m.yes_bid, m.yes_ask) == (4000, 4200)
    assert (m.yes_bid_size, m.yes_ask_size) == (100.0, 200.0)
    assert m.volume == 10.0 and m.open_interest == 5.0
    assert m.updated_at is not None


def test_ticker_for_unmaterialized_market_is_counted_not_created():
    """The auto-combo space streams on `ticker` but is not held in memory.

    ~1.19M generated parlays would otherwise be materialized by the firehose
    within seconds, defeating the whole point of sweeping /events.
    """
    u = Universe()
    u.on_message(_ticker_frame(ticker="KXMVECROSSCATEGORY-SHARD1-XYZ"))
    assert len(u) == 0
    assert u.unknown_ticker_updates == 1


def test_trade_frame_updates_last_price():
    u = Universe()
    u._markets = {"M": Market("M")}
    u.on_message({"type": "trade", "msg": {
        "market_ticker": "M", "yes_price_dollars": "0.7300"}})
    assert u.get("M").last_price == 7300


def test_lifecycle_close_retires_a_market():
    u = Universe()
    u._markets = {"M": Market("M")}
    u.on_message({"type": "market_lifecycle_v2", "msg": {
        "market_ticker": "M", "event_type": "closed"}})
    assert u.get("M").status == "closed"
    assert u.tradeable() == []


def test_lifecycle_updates_close_time():
    u = Universe()
    u._markets = {"M": Market("M")}
    u.on_message({"type": "market_lifecycle_v2", "msg": {
        "market_ticker": "M", "event_type": "close_date_updated",
        "close_ts": 1800000000}})
    assert u.get("M").close_ts == 1800000000


def test_frames_without_a_ticker_are_ignored():
    u = Universe()
    u.on_message({"type": "ticker", "msg": {}})
    u.on_message({"type": "ticker"})
    assert len(u) == 0


# ---------- REST ingest ----------

def test_rest_row_populates_metadata_and_category():
    """Category lives on the event and is denormalized onto the market.

    The PRD's market-preference analytics are per-agent breakdowns by category,
    so losing it here would quietly remove a required dashboard view.
    """
    m = Market("M")
    universe._apply_rest_row(m, {
        "ticker": "M", "event_ticker": "E", "category": "Climate and Weather",
        "series_ticker": "KXHIGHNY", "title": "NYC high temp",
        "status": "active", "close_time": "2026-08-24T18:30:00Z",
        "yes_bid_dollars": "0.3000", "yes_ask_dollars": "0.3200"})
    assert m.category == "Climate and Weather"
    assert m.series == "KXHIGHNY"
    assert m.event_ticker == "E"
    assert m.close_ts is not None
    assert (m.yes_bid, m.yes_ask) == (3000, 3200)


def test_rest_sweep_does_not_clobber_a_live_quote():
    """A 20-second-old sweep must not overwrite a 200ms-old stream quote."""
    m = Market("M")
    m.yes_bid, m.yes_ask = 4000, 4200
    m.updated_at = time.time()          # marks it as stream-fed
    universe._apply_rest_row(m, {"ticker": "M", "yes_bid_dollars": "0.1000",
                                 "yes_ask_dollars": "0.1200"})
    assert (m.yes_bid, m.yes_ask) == (4000, 4200)


def test_iso_timestamps_parse_with_and_without_fractional_seconds():
    assert universe._parse_iso("2026-08-24T18:30:00Z") is not None
    assert universe._parse_iso("2026-08-24T00:01:03.7634Z") is not None
    assert universe._parse_iso(None) is None
    assert universe._parse_iso("nonsense") is None


# ---------- summary ----------

def test_stats_reports_tradeable_separately_from_quoted():
    u = Universe()
    two_sided = _market(4000, 4200)
    two_sided.ticker = "A"
    one_sided = _market(0, 4200)
    one_sided.ticker = "B"
    for m in (two_sided, one_sided):
        m.updated_at = time.time()
    u._markets = {"A": two_sided, "B": one_sided}

    s = u.stats()
    assert s["markets"] == 2
    assert s["quoted"] == 1        # only A is two-sided
    assert s["tradeable"] == 2     # but both can be entered
