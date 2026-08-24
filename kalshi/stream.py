"""Kalshi WebSocket firehose with hardened reconnect.

The live-data spine of the project. Connects, subscribes, and hands every message
to a callback. It knows nothing about agents, order books, or trading -- it is
purely transport.

## The two-tier subscription, and why it is not a design choice

Kalshi's WebSocket does not treat all channels alike (measured, not assumed --
see `tools/measure_firehose.py`):

  ticker               subscribes to EVERY market with no filter   OK
  trade                subscribes to EVERY market with no filter   OK
  market_lifecycle_v2  subscribes to EVERY market with no filter   OK
  orderbook_delta      REJECTS an unfiltered subscribe:
                       {"code": 2, "msg": "Params required"}

So full order-book depth for every market is not something the exchange offers.
The architecture that follows is forced by the API, not chosen:

  BROAD tier  -- ticker + trade + lifecycle, unfiltered, every market on the
                 exchange. `ticker` carries top-of-book (bid, ask, bid size, ask
                 size, last price, volume, open interest), which is enough to
                 evaluate and trade a market.
  DEPTH tier  -- orderbook_delta for an explicit, dynamically updated ticker
                 list: the markets an agent holds or is actively evaluating.

This preserves the PRD's market freedom completely. Every market on Kalshi is
visible and tradeable at all times; only *ladder depth* is rationed, and only
because the exchange rations it.

## Reconnect

Two improvements over the earlier single-market collector, both of which matter
for a system meant to run unattended for weeks:

1. **Exponential backoff with jitter.** The old loop retried on a flat 3-second
   sleep, so every retry after a blip lands in the same instant.
2. **Per-market resync instead of a full teardown.** Order-book sequence numbers
   are per-market. The old code dropped the whole connection on any single
   market's gap -- fine with one market subscribed, catastrophic across thousands
   where some market is always gapping. Here a gap re-snapshots that one market
   and the connection stays up.

Read-only: this subscribes to public market-data channels and never sends an
order command. See `kalshi/rest.py` for the same guarantee over HTTP.
"""

import json
import time
import random
import asyncio
import logging

import websockets

from . import auth

log = logging.getLogger(__name__)

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# Channels that accept an unfiltered subscribe (verified empirically).
BROAD_CHANNELS = ("ticker", "trade", "market_lifecycle_v2")
# Channel that requires an explicit market_tickers list (verified empirically).
DEPTH_CHANNEL = "orderbook_delta"

# Kalshi accepts large ticker lists; 5k was verified clean in one command.
# Batching keeps any single frame reasonable and makes partial failure cheap.
DEPTH_BATCH = 2000

BACKOFF_START = 1.0
BACKOFF_CEILING = 60.0
BACKOFF_FACTOR = 2.0

CONTROL_TYPES = frozenset({"subscribed", "unsubscribed", "ok"})


class Stream:
    """One WebSocket connection to Kalshi, kept alive indefinitely.

    `on_message(message)` is called for every non-control frame. It must be fast
    and must not raise -- it runs inline on the receive loop, so anything slow
    here is backpressure on the entire firehose.
    """

    def __init__(self, key_id, private_key, on_message,
                 broad_channels=BROAD_CHANNELS, depth_tickers=None,
                 on_reconnect=None):
        self.key_id = key_id
        self.private_key = private_key
        self.on_message = on_message
        self.on_reconnect = on_reconnect
        self.broad_channels = list(broad_channels)

        # Markets we want full order-book depth on. Mutable at runtime: agents
        # pull markets into depth coverage as they start evaluating them.
        self.depth_tickers = set(depth_tickers or ())

        self._stop = asyncio.Event()
        self._ws = None
        self._next_cmd_id = 1
        self._resync_queue = set()
        self._depth_pending = set(self.depth_tickers)
        self._depth_live = set()

        # Observability: a long-running system must be able to answer "is it
        # still actually receiving?" without attaching a debugger.
        self.connected = False
        self.messages = 0
        self.bytes_in = 0
        self.reconnects = 0
        self.errors = 0
        self.last_message_at = None
        self.connected_at = None

    # ---------- subscription plumbing ----------

    def _cmd(self, cmd, params):
        self._next_cmd_id += 1
        return {"id": self._next_cmd_id, "cmd": cmd, "params": params}

    async def _send(self, payload):
        if self._ws is None:
            return False
        try:
            await self._ws.send(json.dumps(payload))
            return True
        except Exception:
            # The connection is on its way down; reconnect re-subscribes
            # everything from scratch, so dropping this is harmless.
            return False

    async def _subscribe_broad(self):
        """Every market on the exchange, for the channels that allow it."""
        await self._send(self._cmd("subscribe", {"channels": self.broad_channels}))

    async def _subscribe_depth(self, tickers):
        """Order-book depth for named markets, in batches."""
        tickers = [t for t in tickers if t]
        for i in range(0, len(tickers), DEPTH_BATCH):
            batch = tickers[i:i + DEPTH_BATCH]
            ok = await self._send(self._cmd("subscribe", {
                "channels": [DEPTH_CHANNEL], "market_tickers": batch}))
            if not ok:
                return
            self._depth_live.update(batch)

    def watch_depth(self, tickers):
        """Ask for full order-book depth on these markets.

        Safe to call from anywhere; the subscribe happens on the receive loop.
        """
        new = {t for t in tickers if t} - self.depth_tickers
        if not new:
            return
        self.depth_tickers |= new
        self._depth_pending |= new

    def unwatch_depth(self, tickers):
        """Drop depth coverage for markets nobody cares about any more."""
        drop = {t for t in tickers if t} & self.depth_tickers
        self.depth_tickers -= drop
        self._depth_pending -= drop
        self._depth_live -= drop

    def request_resync(self, ticker):
        """Flag one market whose order book desynced; re-snapshot it alone."""
        self._resync_queue.add(ticker)

    async def _drain_pending(self):
        """Push queued depth subscriptions and resyncs onto the socket.

        Batched, because a busy exchange produces gaps in bursts and one command
        per market would flood the connection.
        """
        if self._depth_pending:
            pending = list(self._depth_pending)
            self._depth_pending.clear()
            await self._subscribe_depth(pending)
        if self._resync_queue:
            resync = list(self._resync_queue)
            self._resync_queue.clear()
            await self._subscribe_depth(resync)

    # ---------- the connection loop ----------

    async def run(self):
        """Connect, subscribe, receive; reconnect forever until `stop()`."""
        backoff = BACKOFF_START
        while not self._stop.is_set():
            try:
                headers = auth.ws_auth_headers(self.key_id, self.private_key)
                async with websockets.connect(
                    WS_URL, additional_headers=headers,
                    max_size=None,        # order-book snapshots can be large
                    ping_interval=20,     # detect a silently dead socket
                    ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self.connected_at = time.time()

                    await self._subscribe_broad()
                    # A reconnect invalidates every book we held, so re-request
                    # depth for the full watch set rather than the pending delta.
                    self._depth_live.clear()
                    self._depth_pending = set(self.depth_tickers)
                    await self._drain_pending()

                    log.info("connected; broad=%s depth=%d markets",
                             self.broad_channels, len(self.depth_tickers))
                    if self.on_reconnect:
                        self.on_reconnect()

                    backoff = BACKOFF_START
                    await self._receive_loop(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("stream disconnected (%s: %s); retry in %.1fs",
                            type(exc).__name__, exc, backoff)
            finally:
                self._ws = None
                self.connected = False

            if self._stop.is_set():
                break
            self.reconnects += 1
            # Jitter stops reconnects from synchronizing into a thundering herd
            # after a shared outage.
            await asyncio.sleep(backoff * (0.5 + random.random()))
            backoff = min(backoff * BACKOFF_FACTOR, BACKOFF_CEILING)

    async def _receive_loop(self, ws):
        last_drain = time.time()
        while not self._stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                # Silence is normal on quiet channels overnight; the ping
                # keepalive is what detects a dead socket. Use the lull to flush
                # any queued subscriptions, then keep waiting.
                await self._drain_pending()
                last_drain = time.time()
                continue

            self.messages += 1
            self.bytes_in += len(raw)
            self.last_message_at = time.time()

            message = json.loads(raw)
            mtype = message.get("type")
            if mtype in CONTROL_TYPES:
                continue
            if mtype == "error":
                self.errors += 1
                log.error("kalshi ws error: %s", message.get("msg"))
                continue

            self.on_message(message)

            if (self._depth_pending or self._resync_queue) and \
                    time.time() - last_drain > 1.0:
                await self._drain_pending()
                last_drain = time.time()

    def stop(self):
        self._stop.set()

    def health(self):
        """Snapshot for the dashboard and for logging."""
        now = time.time()
        return {
            "connected": self.connected,
            "messages": self.messages,
            "bytes_in": self.bytes_in,
            "reconnects": self.reconnects,
            "errors": self.errors,
            "depth_markets": len(self.depth_tickers),
            "uptime_s": (now - self.connected_at) if self.connected_at else 0.0,
            "silent_for_s": (now - self.last_message_at) if self.last_message_at else None,
        }
