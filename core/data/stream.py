"""Async closed-candle streams.

One entry point, :func:`candle_stream`, delivering CLOSED bars only (the
unclosed-candle guard from docs/research-brief.md §2.2 lives here for the
live path):

- Preferred transport: ``ccxt.pro`` ``watch_ohlcv`` (Binance). Candle close
  is detected by timestamp rollover — the forming candle is never emitted.
- Fallback (Upbit, ``poll_only=True``, or ccxt.pro unavailable): REST polling
  aligned to timeframe boundaries (+2s safety), deduped by last seen ts.

Robustness: staleness watchdog (no WS message for 3x timeframe -> rebuild the
exchange instance), exponential reconnect backoff 1->60s with jitter, and
``await exchange.close()`` in every exit path.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from typing import Awaitable, Callable, Union

from core.constants import TIMEFRAME_MINUTES

log = logging.getLogger(__name__)

BarCallback = Callable[[dict], Union[Awaitable, None]]
TickCallback = Callable[[float], Union[Awaitable, None]]
LogCallback = Callable[[str], None]

_MAX_BACKOFF_S = 60.0
_POLL_SAFETY_S = 2.0


async def _maybe_await(result) -> None:
    if inspect.isawaitable(result):
        await result


def _bar_dict(row) -> dict:
    return {"ts": int(row[0]), "o": float(row[1]), "h": float(row[2]),
            "l": float(row[3]), "c": float(row[4]), "v": float(row[5])}


async def candle_stream(exchange_id: str, symbol: str, timeframe: str,
                        on_closed_bar: BarCallback,
                        on_tick: TickCallback | None = None,
                        stop_event: asyncio.Event | None = None,
                        poll_only: bool = False,
                        on_log: LogCallback | None = None) -> None:
    """Stream closed candles until ``stop_event`` is set.

    ``on_closed_bar`` receives ``{"ts","o","h","l","c","v"}`` (ts = candle
    open time, ms UTC) exactly once per closed bar, in order. ``on_tick``
    receives the latest trade/close price on every update. Both callbacks may
    be sync or async.
    """
    if stop_event is None:
        stop_event = asyncio.Event()

    def _log(msg: str) -> None:
        log.info("[stream %s %s %s] %s", exchange_id, symbol, timeframe, msg)
        if on_log is not None:
            try:
                on_log(msg)
            except Exception:
                log.exception("on_log callback failed")

    use_ws = not poll_only and exchange_id != "upbit"
    ccxtpro = None
    if use_ws:
        try:
            import ccxt.pro as ccxtpro  # type: ignore[no-redef]
        except Exception as e:
            _log(f"ccxt.pro unavailable ({e!r}); falling back to REST polling")
            use_ws = False
    if use_ws:
        await _ws_stream(ccxtpro, exchange_id, symbol, timeframe,
                         on_closed_bar, on_tick, stop_event, _log)
    else:
        await _poll_stream(exchange_id, symbol, timeframe,
                           on_closed_bar, on_tick, stop_event, _log)


async def _backfill_gap(ex, symbol: str, timeframe: str, tf_ms: int,
                        last_closed_ts: int, newest_ts: int,
                        on_closed_bar: BarCallback, _log: LogCallback) -> int:
    """Emit bars that closed during a WS outage via a REST OHLCV fetch.

    Fetches closed candles in ``(last_closed_ts, newest_ts)`` (the forming
    candle ``newest_ts`` is excluded) in order and returns the advanced
    ``last_closed_ts``. On REST failure the gap is left for the next reconnect;
    live streaming continues regardless.
    """
    try:
        rows_bf = await ex.fetch_ohlcv(symbol, timeframe, since=last_closed_ts + tf_ms)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _log(f"gap backfill failed: {type(e).__name__}: {e}")
        return last_closed_ts
    emitted = 0
    for row in sorted(rows_bf or [], key=lambda r: r[0]):
        ts = int(row[0])
        if last_closed_ts < ts < newest_ts:
            await _maybe_await(on_closed_bar(_bar_dict(row)))
            last_closed_ts = ts
            emitted += 1
    if emitted:
        _log(f"backfilled {emitted} bar(s) missed during outage")
    return last_closed_ts


async def _ws_stream(ccxtpro, exchange_id: str, symbol: str, timeframe: str,
                     on_closed_bar: BarCallback, on_tick: TickCallback | None,
                     stop_event: asyncio.Event, _log: LogCallback) -> None:
    tf_ms = TIMEFRAME_MINUTES[timeframe] * 60_000
    stale_after_s = 3.0 * tf_ms / 1000.0
    backoff = 1.0
    # Persist across reconnects/staleness rebuilds: the rebuilt WS cache starts
    # empty and never replays bars closed during the outage, so we remember the
    # last emitted close and REST-backfill the gap after each reconnect.
    last_closed_ts: int | None = None
    while not stop_event.is_set():
        ex = getattr(ccxtpro, exchange_id)({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        first_after_connect = True
        try:
            while not stop_event.is_set():
                watch = asyncio.ensure_future(ex.watch_ohlcv(symbol, timeframe))
                stopper = asyncio.ensure_future(stop_event.wait())
                done, _ = await asyncio.wait(
                    {watch, stopper}, timeout=stale_after_s,
                    return_when=asyncio.FIRST_COMPLETED)
                if stopper in done:
                    watch.cancel()
                    await asyncio.gather(watch, return_exceptions=True)
                    return
                stopper.cancel()
                if watch not in done:  # staleness watchdog fired
                    watch.cancel()
                    await asyncio.gather(watch, return_exceptions=True)
                    _log(f"no ws data for {stale_after_s:.0f}s; rebuilding exchange instance")
                    break
                rows = watch.result()  # raises NetworkError etc. -> outer except
                backoff = 1.0
                if not rows:
                    continue
                newest_ts = int(rows[-1][0])
                if last_closed_ts is None:
                    # first snapshot at startup: everything before the forming
                    # candle is history (the engine bootstraps history itself)
                    last_closed_ts = int(rows[-2][0]) if len(rows) >= 2 else newest_ts - tf_ms
                    first_after_connect = False
                else:
                    if first_after_connect:
                        # reconnect/rebuild: bars that closed during the outage
                        # are absent from the fresh WS cache. REST-backfill the
                        # gap so no closed bar is dropped from the engine buffer.
                        last_closed_ts = await _backfill_gap(
                            ex, symbol, timeframe, tf_ms, last_closed_ts,
                            newest_ts, on_closed_bar, _log)
                        first_after_connect = False
                    for row in rows:
                        ts = int(row[0])
                        if last_closed_ts < ts < newest_ts:
                            await _maybe_await(on_closed_bar(_bar_dict(row)))
                            last_closed_ts = ts
                if on_tick is not None:
                    await _maybe_await(on_tick(float(rows[-1][4])))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log(f"ws error: {type(e).__name__}: {e}")
        finally:
            try:
                await ex.close()
            except Exception:
                log.exception("exchange close failed")
        if stop_event.is_set():
            return
        wait = min(backoff, _MAX_BACKOFF_S) + random.uniform(0.0, backoff * 0.25)
        _log(f"reconnecting in {wait:.1f}s")
        if await _wait_or_stop(stop_event, wait):
            return
        backoff = min(backoff * 2.0, _MAX_BACKOFF_S)


async def _poll_stream(exchange_id: str, symbol: str, timeframe: str,
                       on_closed_bar: BarCallback, on_tick: TickCallback | None,
                       stop_event: asyncio.Event, _log: LogCallback) -> None:
    import ccxt.async_support as accxt

    tf_ms = TIMEFRAME_MINUTES[timeframe] * 60_000
    ex = None
    last_seen: int | None = None
    backoff = 1.0
    failures = 0
    try:
        while not stop_event.is_set():
            if ex is None:
                ex = getattr(accxt, exchange_id)({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
            now_ms = int(time.time() * 1000)
            boundary = (now_ms // tf_ms) * tf_ms
            if last_seen is None:
                # the most recent already-closed bar counts as seen: only bars
                # closing after stream start are emitted
                last_seen = boundary - tf_ms
            next_poll_ms = boundary + tf_ms + int(_POLL_SAFETY_S * 1000)
            wait_s = max((next_poll_ms - now_ms) / 1000.0, 0.5)
            if await _wait_or_stop(stop_event, wait_s):
                return
            cur_boundary = (int(time.time() * 1000) // tf_ms) * tf_ms
            try:
                rows = await ex.fetch_ohlcv(symbol, timeframe, limit=3)
                backoff = 1.0
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failures += 1
                _log(f"poll error ({failures}): {type(e).__name__}: {e}")
                if failures >= 3:
                    try:
                        await ex.close()
                    except Exception:
                        pass
                    ex = None
                    _log("rebuilding REST exchange instance")
                wait = min(backoff, _MAX_BACKOFF_S) + random.uniform(0.0, backoff * 0.25)
                if await _wait_or_stop(stop_event, wait):
                    return
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
                continue
            for row in sorted(rows or [], key=lambda r: r[0]):
                ts = int(row[0])
                if last_seen < ts < cur_boundary:
                    await _maybe_await(on_closed_bar(_bar_dict(row)))
                    last_seen = ts
            if on_tick is not None and rows:
                await _maybe_await(on_tick(float(rows[-1][4])))
    finally:
        if ex is not None:
            try:
                await ex.close()
            except Exception:
                log.exception("exchange close failed")


async def _wait_or_stop(stop_event: asyncio.Event, timeout: float) -> bool:
    """Sleep up to ``timeout`` seconds; True if stop_event fired meanwhile."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False
