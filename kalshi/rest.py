"""Read-only Kalshi REST client.

This module is the ONLY place the project talks to Kalshi over HTTP, and it can
only ever issue GET requests -- `_get` is the single network primitive and
nothing here builds a POST/PUT/DELETE. That is deliberate and load-bearing: the
account behind these credentials is real and funded, and the agents in this
project trade simulated money. `tests/test_read_only.py` enforces it.

None of these endpoints need authentication -- Kalshi's market data is public.
Auth is only required for the WebSocket (see `kalshi/auth.py`).

Three things live here:
  - `iter_open_markets()`  the tradeable universe
  - `settled_markets()`    resolved markets, for Phase A pretraining
  - `candlesticks()`       historical prices -- see the leakage warning below
"""

import time

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi auto-generates an enormous cross-category parlay space: ~1.19M of the
# ~1.29M open markets are machine-produced combinations of other markets. They
# are combinatorial products of markets we already stream, so we exclude them
# from the streamed universe and materialize them on demand instead. This is a
# tractability decision about generated duplicates, NOT a category restriction --
# every market a human would recognize as a market stays in play (PRD 2).
AUTO_COMBO_PREFIXES = ("KXMVECROSSCATEGORY", "KXMVESPORTSMULTIGAME")

PAGE_LIMIT = 1000
_RETRY_STATUS = {429, 500, 502, 503, 504}


def is_auto_combo(ticker):
    """True for Kalshi's machine-generated cross-category parlay markets."""
    series = (ticker or "").split("-")[0]
    return series.startswith(AUTO_COMBO_PREFIXES)


def _get(path, params=None, session=None, timeout=30, retries=4):
    """The one and only network call in this project's REST layer.

    Retries with exponential backoff on rate limits and transient 5xx. A long-
    running system will hit both, and a bare exception here would take down the
    universe refresh.
    """
    session = session or requests
    url = BASE + path
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in _RETRY_STATUS and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _paginate(path, params, session=None, page_cap=None):
    """Yield each page of a cursor-paginated Kalshi list endpoint.

    Kalshi signals the end by returning no cursor OR an empty batch; we stop on
    either. `page_cap` exists so callers (and tests) can bound a sweep -- the
    open-markets sweep is ~1,300 pages and takes minutes.
    """
    params = dict(params)
    params.setdefault("limit", PAGE_LIMIT)
    cursor = None
    pages = 0
    while True:
        if cursor:
            params["cursor"] = cursor
        data = _get(path, params, session=session)
        yield data
        pages += 1
        cursor = data.get("cursor")
        if not cursor or page_cap and pages >= page_cap:
            return


def iter_open_markets(session=None, include_auto_combos=False, page_cap=None):
    """Stream every open market, one dict at a time.

    Auto-generated combos are filtered out by default (see AUTO_COMBO_PREFIXES).
    This is a generator rather than a list because the unfiltered sweep is over a
    million markets -- materializing it all at once is pure waste when the caller
    only keeps ~95k of them.
    """
    for page in _paginate("/markets", {"status": "open"}, session=session,
                          page_cap=page_cap):
        for market in page.get("markets", []):
            if include_auto_combos or not is_auto_combo(market.get("ticker")):
                yield market


def settled_markets(series_ticker, session=None, page_cap=None):
    """Every settled market for one series -- the Phase A pretraining corpus."""
    out = []
    for page in _paginate("/markets",
                          {"status": "settled", "series_ticker": series_ticker},
                          session=session, page_cap=page_cap):
        batch = page.get("markets", [])
        out.extend(batch)
        if not batch:
            break
    return out


def candlesticks(series_ticker, ticker, start_ts, end_ts, period_interval=60,
                 session=None):
    """Historical OHLC for one market.

    *** Use this, never `last_price`, for any historical price. ***

    A settled market's `last_price` is the POST-settlement print -- 0.99 for the
    winning side, 0.01 for the losing one. It is the answer, not a forecast.
    Training on it is label leakage that will make an agent look brilliant in
    backtest and clueless live. This exact trap has already cost this codebase's
    author twice on previous Kalshi projects.

    `period_interval` is in minutes: 1, 60, or 1440.
    """
    path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
    data = _get(path, {"start_ts": int(start_ts), "end_ts": int(end_ts),
                       "period_interval": period_interval}, session=session)
    return data.get("candlesticks", [])


def series_of(ticker):
    """'KXBTC15M-26AUG241430-T7.99' -> 'KXBTC15M'. The candlesticks path needs it."""
    return (ticker or "").split("-")[0]
