"""WS reconnect gap-backfill tests (core/data/stream.py::_ws_stream).

Regression for the reconnect hole: on a mid-session WS reconnect (network drop
or staleness rebuild) the rebuilt ccxt.pro cache starts empty and never replays
bars that closed during the outage. ``last_closed_ts`` must persist across the
outer reconnect loop and the missed bars must be re-emitted once via the REST
``fetch_ohlcv`` backfill path — never silently dropped.

Fully offline: a fake ccxt.pro object scripts ``watch_ohlcv`` results (including
a forced disconnect) and serves the gap over ``fetch_ohlcv``.
"""
from __future__ import annotations

import asyncio
import types

from core.data.stream import _ws_stream

TF_MS = 60_000  # 1m timeframe


def _row(ts: int, price: float) -> list:
    """A ccxt OHLCV row [ts, open, high, low, close, volume]."""
    return [ts, price, price + 1.0, price - 1.0, price, 10.0]


class _FakeExchange:
    """One reconnect-generation of a fake ccxt.pro exchange, sharing script
    state with its siblings so the WS timeline survives across rebuilds."""

    def __init__(self, shared: dict):
        self._shared = shared

    async def watch_ohlcv(self, symbol, timeframe):
        shared = self._shared
        kind, *rest = shared["actions"][shared["idx"]]
        shared["idx"] += 1
        if kind == "raise":
            raise RuntimeError("ws connection dropped")
        if kind == "stop":
            shared["stop_event"].set()
            await asyncio.sleep(3600)  # never completes; the stopper wins
            return []
        return [list(r) for r in rest[0]]

    async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        shared = self._shared
        shared["backfill_since"].append(since)
        rows = shared["full"]
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return [list(r) for r in rows]

    async def close(self):
        self._shared["close_calls"] += 1


def _make_fake_ccxtpro(shared: dict):
    def factory(config):
        shared["instances"] += 1
        return _FakeExchange(shared)

    return types.SimpleNamespace(binance=factory)


def test_reconnect_backfills_bars_closed_during_outage():
    emitted: list[int] = []
    shared = {
        "idx": 0,
        "instances": 0,
        "close_calls": 0,
        "backfill_since": [],
        # full REST timeline the exchange can serve for backfill
        "full": [_row(ts, 100.0 + ts / TF_MS) for ts in
                 (0, TF_MS, 2 * TF_MS, 3 * TF_MS, 4 * TF_MS)],
        "actions": [
            # startup snapshot: 0 is history, 60000 is the forming candle
            ("rows", [_row(0, 100.0), _row(TF_MS, 101.0)]),
            # normal close: bar 60000 closes, 120000 now forming -> emit 60000
            ("rows", [_row(TF_MS, 101.0), _row(2 * TF_MS, 102.0)]),
            # connection drops; bars 120000 and 180000 close during the outage
            ("raise",),
            # reconnect snapshot: fresh cache only shows 180000 closed + 240000
            # forming; 120000/180000 must come from the REST backfill
            ("rows", [_row(3 * TF_MS, 104.0), _row(4 * TF_MS, 105.0)]),
            ("stop",),
        ],
    }

    async def main():
        stop_event = asyncio.Event()
        shared["stop_event"] = stop_event
        fake = _make_fake_ccxtpro(shared)

        async def on_closed_bar(bar):
            emitted.append(bar["ts"])

        await asyncio.wait_for(
            _ws_stream(fake, "binance", "BTC/USDT", "1m",
                       on_closed_bar, None, stop_event, lambda _m: None),
            timeout=10.0)

    asyncio.run(main())

    # bar 0 is startup history (not emitted); 240000 is the forming candle at
    # reconnect (not emitted). 60000 streams normally; 120000 and 180000 closed
    # during the outage and are recovered via the REST backfill — in order,
    # exactly once each, strictly increasing.
    assert emitted == [TF_MS, 2 * TF_MS, 3 * TF_MS]
    assert emitted == sorted(set(emitted))
    # exactly one reconnect happened and the backfill fetched from the gap start
    assert shared["instances"] == 2
    assert shared["backfill_since"] == [2 * TF_MS]  # last_closed_ts(60000) + tf
    assert shared["close_calls"] >= 1  # every exchange instance was closed


def test_no_backfill_on_clean_startup():
    """At initial startup (last_closed_ts is None) nothing is backfilled: the
    engine bootstraps history via REST itself, so fetch_ohlcv must not be hit
    and only genuinely new closed bars are emitted."""
    emitted: list[int] = []
    shared = {
        "idx": 0,
        "instances": 0,
        "close_calls": 0,
        "backfill_since": [],
        "full": [_row(ts, 100.0) for ts in (0, TF_MS, 2 * TF_MS)],
        "actions": [
            ("rows", [_row(0, 100.0), _row(TF_MS, 101.0)]),      # startup
            ("rows", [_row(TF_MS, 101.0), _row(2 * TF_MS, 102.0)]),  # emit 60000
            ("stop",),
        ],
    }

    async def main():
        stop_event = asyncio.Event()
        shared["stop_event"] = stop_event
        fake = _make_fake_ccxtpro(shared)

        async def on_closed_bar(bar):
            emitted.append(bar["ts"])

        await asyncio.wait_for(
            _ws_stream(fake, "binance", "BTC/USDT", "1m",
                       on_closed_bar, None, stop_event, lambda _m: None),
            timeout=10.0)

    asyncio.run(main())

    assert emitted == [TF_MS]                 # only the normally-closed bar
    assert shared["backfill_since"] == []     # no REST backfill on clean startup
    assert shared["instances"] == 1           # no reconnect
