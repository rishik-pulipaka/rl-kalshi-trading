"""Which markets get an order book, and why it is not just "the busiest ones".

Half of every agent's candidates are drawn from the depth working set, so how
that set is chosen quietly decides what the agents can actually buy. Ranking it
on open interest alone selected *against* fast-resolving markets and cancelled
out the resolution weighting in `Agent._discover` -- measured on the live
universe, the top 3000 by open interest had a median 128.8 days to close
against the universe's own 74.8.
"""

import time

import pytest

from kalshi.universe import Market
from run import depth_rank


def _market(ticker, open_interest, days_out, now):
    m = Market(ticker)
    m.open_interest = open_interest
    m.close_ts = now + days_out * 86400
    return m


def test_liquidity_still_wins_between_equally_timed_markets():
    """Open interest is a fact about the market, not a preference of ours, and
    a market nobody trades has no book to fill against."""
    now = time.time()
    busy = _market("BUSY", 50_000, 5, now)
    quiet = _market("QUIET", 10, 5, now)
    assert depth_rank(busy, 7.0, now) < depth_rank(quiet, 7.0, now)


def test_a_sooner_market_beats_an_equally_liquid_distant_one():
    now = time.time()
    soon = _market("SOON", 20_000, 2, now)
    distant = _market("DISTANT", 20_000, 400, now)
    assert depth_rank(soon, 7.0, now) < depth_rank(distant, 7.0, now)


def test_a_huge_but_distant_market_loses_to_a_modest_imminent_one():
    """The actual failure: championship and election markets are the biggest on
    the exchange and settle furthest out, so they crowded out everything the
    agents could learn from this month."""
    now = time.time()
    election = _market("ELECTION-28", 900_000, 800, now)
    tonight = _market("TONIGHT", 40_000, 1, now)
    assert depth_rank(tonight, 7.0, now) < depth_rank(election, 7.0, now)


def test_turning_the_weighting_off_restores_pure_liquidity_ranking():
    now = time.time()
    election = _market("ELECTION-28", 900_000, 800, now)
    tonight = _market("TONIGHT", 40_000, 1, now)
    assert depth_rank(election, 0, now) < depth_rank(tonight, 0, now)


def test_markets_without_a_close_time_fall_back_to_liquidity():
    now = time.time()
    unknown = _market("UNKNOWN", 30_000, 1, now)
    unknown.close_ts = None
    assert depth_rank(unknown, 7.0, now) == -30_000


def test_an_already_closed_market_does_not_produce_a_negative_discount():
    """A past close time must not flip the sign and rocket it to the top."""
    now = time.time()
    expired = _market("EXPIRED", 30_000, -5, now)
    assert depth_rank(expired, 7.0, now) == -30_000


def test_a_market_with_no_open_interest_ranks_last():
    now = time.time()
    empty = _market("EMPTY", None, 1, now)
    assert depth_rank(empty, 7.0, now) == 0.0
