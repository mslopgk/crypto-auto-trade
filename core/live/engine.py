"""Live trading engine.

Consumes closed candles from :mod:`core.data.stream`, recomputes the
strategy stance on a rolling bar buffer, and reconciles the desired vs
actual position through a :class:`core.live.broker.Broker` — mirroring the
backtest execution model (signal on close, execute immediately after close,
which approximates next-open fills at live cadence).

Threading model (research brief §4.1): ``run_forever()`` is blocking and
owns its own asyncio loop — run it inside a dedicated ``threading.Thread``
(the GUI's QThread). ``stop()`` is thread-safe and idempotent. No Qt imports
here; the GUI observes the engine only through :class:`LiveCallbacks`.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.constants import periods_per_year
from core.indicators import atr as atr_indicator
from core.indicators import realized_vol
from core.live.broker import Broker, BrokerAuthError, Fill
from core.live.state import LiveState
from core.strategies.base import Strategy

log = logging.getLogger(__name__)

_DEFAULT_STOP_FRAC = 0.05      # stop-distance fallback when strategy has no stop params
_MAX_CONSECUTIVE_ERRORS = 10
_BALANCE_TOLERANCE = 0.01      # >1% base-balance mismatch -> trust broker


@dataclass
class LiveCallbacks:
    """Observer hooks fired from the engine thread (marshal to GUI yourself)."""

    on_bar: Optional[Callable] = None        # (bar_dict)
    on_tick: Optional[Callable] = None       # (last_price)
    on_order: Optional[Callable] = None      # (fill_dict)
    on_position: Optional[Callable] = None   # (position_dict | None)
    on_equity: Optional[Callable] = None     # (ts_ms, equity)
    on_log: Optional[Callable] = None        # (msg)
    on_status: Optional[Callable] = None     # (status)


class LiveEngine:
    """Single-symbol live/paper trading engine.

    Parameters beyond the public contract (all keyword, all optional):
    ``stream_fn`` — injectable candle source with the signature of
    :func:`core.data.stream.candle_stream` (tests feed synthetic bars);
    ``state_path`` — override the persisted-state location;
    ``warmup_bars`` / ``buffer_cap`` — history bootstrap and buffer sizing;
    ``bootstrap`` — set False to skip the REST history bootstrap (tests).
    """

    def __init__(self, exchange_id: str, symbol: str, timeframe: str,
                 strategy: Strategy, broker: Broker,
                 risk_limits=None, callbacks: LiveCallbacks | None = None,
                 poll_only: bool = False, *,
                 stream_fn: Callable | None = None,
                 state_path: str | Path | None = None,
                 warmup_bars: int = 1000, buffer_cap: int = 5000,
                 bootstrap: bool = True):
        from core.data.stream import candle_stream  # deferred: keeps import light

        self._exchange_id = exchange_id
        self._symbol = symbol
        self._timeframe = timeframe
        self._strategy = strategy
        self._broker = broker
        self._risk_limits = risk_limits
        self._callbacks = callbacks or LiveCallbacks()
        self._poll_only = poll_only
        self._stream_fn = stream_fn or candle_stream
        self._warmup_bars = max(int(warmup_bars), 1)
        self._bootstrap = bootstrap

        base, _, quote = symbol.partition("/")
        self._base_ccy = base
        self._quote_ccy = quote.split(":")[0] if quote else "USDT"

        self._state_path = Path(state_path) if state_path else \
            LiveState.default_path(exchange_id, symbol, timeframe)
        self._state = LiveState()
        self._buf: deque[tuple] = deque(maxlen=max(int(buffer_cap), self._warmup_bars))

        ep = strategy.engine_params()
        self._sl_pct = float(ep.get("sl_pct") or 0.0)
        self._trail_mult = float(ep.get("trail_atr_mult") or 0.0)
        self._atr_period = int(ep.get("atr_period") or 14)

        self._status = "stopped"
        self._status_lock = threading.Lock()
        self._thread_stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop: asyncio.Event | None = None
        self._risk = None
        self._errors = 0
        self._peak_price = 0.0
        self._last_price: float | None = None
        self._halt_day = None  # UTC date of an active daily halt

    # ------------------------------------------------------------------ status
    @property
    def status(self) -> str:
        """'stopped' | 'running' | 'halted_daily' | 'halted_mdd' | 'error'."""
        with self._status_lock:
            return self._status

    def _set_status(self, status: str) -> None:
        with self._status_lock:
            if self._status == status:
                return
            self._status = status
        log.info("engine status -> %s", status)
        self._fire(self._callbacks.on_status, status)

    @property
    def state(self) -> LiveState:
        return self._state

    # -------------------------------------------------------------- lifecycle
    def run_forever(self) -> None:
        """Blocking main loop; call as the target of a dedicated daemon thread."""
        if self._thread_stop.is_set():
            log.warning("run_forever called after stop(); not starting")
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._async_stop = asyncio.Event()
        try:
            loop.run_until_complete(self._main())
        except Exception:
            log.exception("live engine crashed")
            self._set_status("error")
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                log.exception("loop teardown failed")
            finally:
                loop.close()
                self._loop = None
            try:
                self._broker.close()
            except Exception:
                log.exception("broker close failed")
            self._save_state()
            if self.status != "error":
                self._set_status("stopped")

    def stop(self) -> None:
        """Request shutdown. Thread-safe, idempotent."""
        self._thread_stop.set()
        loop = self._loop
        event = self._async_stop
        if loop is not None and event is not None:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # loop already closed
                pass

    def re_arm(self) -> None:
        """Manual re-arm after a hard halt (GUI action). Thread-safe."""
        risk = self._risk
        if risk is not None and hasattr(risk, "re_arm"):
            risk.re_arm()
        self._halt_day = None
        if self.status in ("halted_daily", "halted_mdd"):
            self._set_status("running")
            self._log("risk halt manually re-armed")

    # ------------------------------------------------------------------- main
    async def _main(self) -> None:
        assert self._async_stop is not None
        self._set_status("running")
        self._log(f"starting: {self._exchange_id} {self._symbol} {self._timeframe} "
                  f"strategy={self._strategy.describe()}")
        if self._state.started_at is None:
            self._state.started_at = time.time()
        self._load_state()
        self._reconcile_position()
        if self._bootstrap:
            await self._bootstrap_history()
        watchdog = asyncio.ensure_future(self._watch_thread_stop())
        try:
            await self._stream_fn(
                self._exchange_id, self._symbol, self._timeframe,
                on_closed_bar=self._on_closed_bar,
                on_tick=self._on_tick,
                stop_event=self._async_stop,
                poll_only=self._poll_only,
                on_log=self._log,
            )
        finally:
            watchdog.cancel()
            await asyncio.gather(watchdog, return_exceptions=True)
            self._save_state()

    async def _watch_thread_stop(self) -> None:
        """Bridge the thread-safe stop flag into the asyncio stop event."""
        assert self._async_stop is not None
        while not self._async_stop.is_set():
            if self._thread_stop.is_set():
                self._async_stop.set()
                return
            await asyncio.sleep(0.2)

    # ------------------------------------------------------------ stream hooks
    async def _on_closed_bar(self, bar: dict) -> None:
        # broker/REST calls are blocking -> run the handler in a worker thread
        # so the websocket heartbeat stays responsive
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._handle_closed_bar, bar)

    def _on_tick(self, price: float) -> None:
        self._last_price = float(price)
        self._fire(self._callbacks.on_tick, float(price))

    # ------------------------------------------------------------ bar handling
    def _handle_closed_bar(self, bar: dict) -> None:
        """Full per-bar pipeline. Never raises: errors are counted and after
        ``_MAX_CONSECUTIVE_ERRORS`` in a row the engine halts with status 'error'."""
        try:
            ts_ms = int(bar["ts"])
            if self._buf and ts_ms <= self._buf[-1][0]:
                return  # duplicate / out-of-order bar
            self._buf.append((ts_ms, float(bar["o"]), float(bar["h"]),
                              float(bar["l"]), float(bar["c"]), float(bar["v"])))
            self._state.last_bar_ts = ts_ms
            price = float(bar["c"])
            self._last_price = price
            if self._state.position is not None:
                self._peak_price = max(self._peak_price, float(bar["h"]))
            self._fire(self._callbacks.on_bar, bar)

            ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            equity = self._mark_equity(price)
            self._state.record_equity(ts_ms, equity)
            self._fire(self._callbacks.on_equity, ts_ms, equity)

            self._apply_risk(ts, equity, price)
            halted = self.status in ("halted_daily", "halted_mdd")

            df = self._buffer_df()
            desired, size_frac = (0, None) if halted else self._compute_target(df)

            current = int(self._state.position["stance"]) if self._state.position else 0

            # engine-level protective stop (bar-close evaluation)
            if current != 0 and desired == current and self._protective_stop_hit(df, price):
                self._log(f"protective exit at {price:.8g}")
                self._exit_position(reason="protective")
                current = 0
                desired = 0  # no same-bar re-entry after a protective stop

            if desired != current:
                if current != 0:
                    self._exit_position(reason="signal")
                if desired == 1:
                    self._enter_long(df, price, equity, size_frac)

            self._save_state()
            self._errors = 0
        except Exception as e:
            self._errors += 1
            log.exception("bar handler error")
            self._log(f"bar handler error ({self._errors}/{_MAX_CONSECUTIVE_ERRORS}): "
                      f"{type(e).__name__}: {e}")
            if isinstance(e, BrokerAuthError) or self._errors >= _MAX_CONSECUTIVE_ERRORS:
                self._set_status("error")
                self.stop()

    def _compute_target(self, df: pd.DataFrame) -> tuple[int, float | None]:
        """Desired stance in {0, 1} (spot: shorts clipped) + optional size frac."""
        if len(df) < 2:
            return 0, None
        stance_arr = self._strategy.generate_signals(df)
        s = float(stance_arr[-1]) if len(stance_arr) else 0.0
        if np.isnan(s):
            s = 0.0
        desired = 1 if s > 0 else 0  # short stances clipped to flat on spot
        size_frac = None
        sf_arr = self._strategy.generate_size_frac(df)
        if sf_arr is not None and len(sf_arr):
            v = float(sf_arr[-1])
            size_frac = 0.0 if np.isnan(v) else float(np.clip(v, 0.0, 1.0))
        return desired, size_frac

    # ------------------------------------------------------------ risk overlay
    def _apply_risk(self, ts: pd.Timestamp, equity: float, price: float) -> None:
        if self._risk is None and self._risk_limits is not None:
            self._risk = self._make_risk(equity)
        risk = self._risk
        if risk is None:
            return
        # auto re-arm a daily halt on the next UTC day
        if self.status == "halted_daily" and self._halt_day is not None \
                and ts.date() > self._halt_day:
            risk.reset_daily(ts)
            self._halt_day = None
            self._set_status("running")
            self._log("daily halt released (new UTC day)")
        action = risk.update_equity(ts, equity)
        if action == "halt_daily" and self.status != "halted_daily":
            self._set_status("halted_daily")
            self._halt_day = ts.date()
            self._log("HARD DAILY LOSS HALT: flattening")
            self._exit_position(reason="halt_daily")
        elif action == "halt_mdd" and self.status != "halted_mdd":
            self._set_status("halted_mdd")
            self._log("HARD DRAWDOWN HALT: flattening (manual re-arm required)")
            self._exit_position(reason="halt_mdd")
        elif action == "soft_dd":
            self._log("soft drawdown brake active (risk halved)")

    def _make_risk(self, initial_equity: float):
        try:
            from core.risk import RiskManager
        except ImportError:
            self._log("core.risk unavailable; running WITHOUT risk overlay")
            return None
        return RiskManager(self._risk_limits, initial_equity)

    # ---------------------------------------------------------------- position
    def _protective_stop_hit(self, df: pd.DataFrame, price: float) -> bool:
        pos = self._state.position
        if pos is None or int(pos["stance"]) != 1:
            return False
        entry = float(pos["entry_price"])
        if self._sl_pct > 0.0 and price <= entry * (1.0 - self._sl_pct):
            return True
        if self._trail_mult > 0.0 and len(df) > self._atr_period:
            a = atr_indicator(df["high"].to_numpy(), df["low"].to_numpy(),
                              df["close"].to_numpy(), self._atr_period)
            last_atr = float(a[-1])
            if np.isfinite(last_atr) and \
                    price <= self._peak_price - self._trail_mult * last_atr:
                return True
        return False

    def _stop_distance(self, df: pd.DataFrame, price: float) -> float:
        """Price distance to the protective stop, for fixed-fractional sizing."""
        if self._sl_pct > 0.0:
            return self._sl_pct * price
        if self._trail_mult > 0.0 and len(df) > self._atr_period:
            a = atr_indicator(df["high"].to_numpy(), df["low"].to_numpy(),
                              df["close"].to_numpy(), self._atr_period)
            if np.isfinite(a[-1]):
                return self._trail_mult * float(a[-1])
        return _DEFAULT_STOP_FRAC * price

    def _enter_long(self, df: pd.DataFrame, price: float, equity: float,
                    size_frac: float | None) -> None:
        if self._risk is not None and not self._risk.can_open(0):
            self._log("entry blocked by risk manager (can_open=False)")
            return
        if self._risk is not None:
            rv = self._realized_vol(df)
            # RiskManager.size_order wants the stop distance as a FRACTION of
            # price (equity * risk_pct / stop_frac); _stop_distance returns an
            # absolute price distance, so convert before handing it over.
            stop_dist = self._stop_distance(df, price)
            stop_frac = stop_dist / price if price > 0.0 else 0.0
            budget = float(self._risk.size_order(equity, price, stop_frac, rv))
        else:
            budget = equity
        if size_frac is not None:
            budget = min(budget, size_frac * equity)
        avail = float(self._broker.get_balances().get(self._quote_ccy, 0.0))
        budget = min(budget, avail * 0.995)  # headroom for fee/slippage rounding
        if budget <= 0.0:
            self._log(f"entry skipped: zero budget (avail={avail:.8g})")
            return
        fill = self._broker.market_order(self._symbol, "buy", cost=budget)
        self._state.position = {
            "amount": fill.amount,
            "entry_price": fill.price,
            "entry_time": fill.timestamp,
            "stance": 1,
        }
        self._peak_price = fill.price
        self._state.realized_pnl -= fill.fee
        self._record_order(fill, reason="entry")
        self._fire(self._callbacks.on_position, dict(self._state.position))
        self._log(f"LONG {fill.amount:.8g} {self._symbol} @ {fill.price:.8g} "
                  f"(cost {fill.cost:.8g}, fee {fill.fee:.8g})")

    def _exit_position(self, reason: str) -> None:
        pos = self._state.position
        if pos is None:
            return
        amount = float(pos["amount"])
        held = float(self._broker.get_balances().get(self._base_ccy, amount))
        amount = min(amount, held)
        if amount <= 0.0:
            self._log(f"exit ({reason}): no base balance left; clearing position")
            self._state.position = None
            self._fire(self._callbacks.on_position, None)
            return
        fill = self._broker.market_order(self._symbol, "sell", amount=amount)
        pnl = (fill.price - float(pos["entry_price"])) * fill.amount - fill.fee
        self._state.realized_pnl += pnl
        self._state.position = None
        self._peak_price = 0.0
        self._record_order(fill, reason=reason)
        self._fire(self._callbacks.on_position, None)
        self._log(f"EXIT ({reason}) {fill.amount:.8g} {self._symbol} @ "
                  f"{fill.price:.8g} pnl {pnl:+.8g}")

    def _record_order(self, fill: Fill, reason: str) -> None:
        d = fill.to_dict()
        d["reason"] = reason
        self._state.record_fill(d)
        self._fire(self._callbacks.on_order, d)

    # -------------------------------------------------------------- valuation
    def _mark_equity(self, price: float) -> float:
        balances = self._broker.get_balances()
        quote = float(balances.get(self._quote_ccy, 0.0))
        base = float(balances.get(self._base_ccy, 0.0))
        return quote + base * price

    def _realized_vol(self, df: pd.DataFrame) -> float:
        rv = np.nan
        if len(df) > 31:
            arr = realized_vol(df["close"].to_numpy(), period=30,
                               ppy=periods_per_year(self._timeframe))
            rv = float(arr[-1])
        if not np.isfinite(rv) or rv <= 0.0:
            rv = float(getattr(self._risk_limits, "target_annual_vol", 0.20) or 0.20)
        return rv

    # ------------------------------------------------------- startup / buffer
    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            self._state = LiveState.load(self._state_path)
            self._log(f"restored state from {self._state_path.name} "
                      f"(position={'yes' if self._state.position else 'no'})")
        except Exception as e:
            log.warning("could not load state %s: %s", self._state_path, e)

    def _reconcile_position(self) -> None:
        """Persisted position vs broker balances: on >1% mismatch trust the broker."""
        pos = self._state.position
        if pos is None:
            return
        try:
            balances = self._broker.get_balances()
        except Exception as e:
            self._log(f"reconcile skipped (balance fetch failed: {e})")
            return
        want = float(pos["amount"])
        have = float(balances.get(self._base_ccy, 0.0))
        if want <= 0.0 or have < want * (1.0 - _BALANCE_TOLERANCE):
            self._log(f"WARNING: state says {want:.8g} {self._base_ccy} but broker "
                      f"holds {have:.8g}; trusting broker, clearing position")
            self._state.position = None
            self._fire(self._callbacks.on_position, None)
        else:
            self._peak_price = float(pos["entry_price"])

    async def _bootstrap_history(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._bootstrap_sync)
        except Exception as e:
            log.exception("history bootstrap failed")
            self._log(f"history bootstrap failed ({type(e).__name__}: {e}); "
                      f"starting with empty buffer")

    def _bootstrap_sync(self) -> None:
        from core.data import fetcher
        df = fetcher.load_ohlcv(self._exchange_id, self._symbol, self._timeframe,
                                refresh=True)
        df = df.tail(self._buf.maxlen or self._warmup_bars)
        if df.empty:
            self._log("bootstrap returned no history")
            return
        ts_ms = df.index.as_unit("ns").asi8 // 1_000_000
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        v = df["volume"].to_numpy(dtype=np.float64)
        last = self._buf[-1][0] if self._buf else -1
        for i in range(len(df)):
            if ts_ms[i] > last:
                self._buf.append((int(ts_ms[i]), o[i], h[i], l[i], c[i], v[i]))
        self._last_price = float(c[-1])
        self._log(f"bootstrapped {len(self._buf)} bars "
                  f"(through {df.index[-1].isoformat()})")

    def _buffer_df(self) -> pd.DataFrame:
        """Rolling buffer as the project-contract OHLCV DataFrame."""
        arr = np.asarray(self._buf, dtype=np.float64)
        idx = pd.DatetimeIndex(
            pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True),
            name="timestamp")
        return pd.DataFrame(
            {"open": arr[:, 1], "high": arr[:, 2], "low": arr[:, 3],
             "close": arr[:, 4], "volume": arr[:, 5]}, index=idx)

    # ---------------------------------------------------------------- helpers
    def _save_state(self) -> None:
        try:
            self._state.save(self._state_path)
        except Exception:
            log.exception("state save failed")

    def _fire(self, cb: Optional[Callable], *args) -> None:
        if cb is None:
            return
        try:
            cb(*args)
        except Exception:
            log.exception("callback failed")

    def _log(self, msg: str) -> None:
        log.info(msg)
        self._fire(self._callbacks.on_log, msg)
