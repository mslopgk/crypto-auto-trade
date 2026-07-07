"""Background workers bridging the core engine to the GUI event bus.

Every worker runs off the GUI thread and communicates results back through
:mod:`gui.event_bus` (or its own typed signals). Because the bus lives in the
GUI thread, emissions are delivered as queued events — safe cross-thread UI
updates (research brief §4.1). No worker touches a widget directly.

- :class:`BacktestWorker`   (QRunnable)  single backtest -> bus.backtest_done
- :class:`DataLoadWorker`   (QRunnable)  OHLCV load       -> own signals
- :class:`SearchWorker`     (QThread)    grid search      -> bus.search_*
- :class:`WalkForwardWorker`(QThread)    walk-forward     -> own signals
- :class:`LiveEngineController`          owns LiveEngine thread -> bus signals
"""
from __future__ import annotations

import logging
import threading

import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThread, Signal

from core.live.engine import LiveCallbacks, LiveEngine
from core.optimize.search import run_search
from core.optimize.walkforward import run_walkforward
from core.strategies.registry import get_strategy
from gui.event_bus import bus

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def instantiate_strategy(name: str, params: dict, timeframe: str):
    """Build a strategy instance, injecting ``timeframe`` when it needs one."""
    cls = get_strategy(name)
    merged = dict(params or {})
    if "timeframe" in cls.DEFAULTS:
        merged.setdefault("timeframe", timeframe)
    return cls(**merged)


class PriceHolder:
    """Thread-safe latest-price cell used as a PaperBroker ``price_source``.

    The live engine's callbacks (fired on the engine thread) push the newest
    bar-close / tick price here; the paper broker reads it synchronously when
    it fills an order (same thread, later in the same bar handler).
    """

    def __init__(self) -> None:
        self._price: float | None = None
        self._lock = threading.Lock()

    def set(self, price: float) -> None:
        with self._lock:
            self._price = float(price)

    def get(self, symbol: str | None = None) -> float:
        with self._lock:
            if self._price is None:
                raise RuntimeError("no price yet (waiting for first bar)")
            return self._price


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

class BacktestWorker(QRunnable):
    """Run one backtest in the global thread pool (numba releases the GIL).

    Loads OHLCV (``refresh=True`` by default, per spec), builds the registry
    strategy, runs ``run_strategy_backtest`` with the panel's cost overrides,
    and emits ``bus.backtest_done`` with ``{request_id, result, df, ...}`` or
    ``{..., error}`` on failure.
    """

    def __init__(self, request_id, exchange: str, symbol: str, timeframe: str,
                 strategy_name: str, params: dict, since: str | None,
                 until: str | None, fee: float, slippage: float,
                 capital: float, refresh: bool = True):
        super().__init__()
        self.setAutoDelete(True)
        self.request_id = request_id
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy_name = strategy_name
        self.params = dict(params or {})
        self.since = since
        self.until = until
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.capital = float(capital)
        self.refresh = bool(refresh)

    def run(self) -> None:
        b = bus()
        try:
            from core.backtest.runner import run_strategy_backtest
            from core.data.fetcher import load_ohlcv

            df = load_ohlcv(self.exchange, self.symbol, self.timeframe,
                            since=self.since, refresh=self.refresh)
            if self.until:
                until_ts = pd.Timestamp(self.until)
                if until_ts.tz is None:
                    until_ts = until_ts.tz_localize("UTC")
                df = df[df.index <= until_ts]
            if len(df) < 3:
                raise ValueError("not enough bars in the selected range")

            strategy = instantiate_strategy(self.strategy_name, self.params,
                                            self.timeframe)
            overrides = {"fee": self.fee, "slippage": self.slippage,
                         "initial_capital": self.capital}
            result = run_strategy_backtest(
                df, strategy, self.timeframe, exchange_id=self.exchange,
                symbol=self.symbol, overrides=overrides)
            b.backtest_done.emit({
                "request_id": self.request_id, "result": result, "df": df,
                "strategy": self.strategy_name, "symbol": self.symbol,
                "timeframe": self.timeframe, "error": None,
            })
        except Exception as e:  # never let a worker exception escape the pool
            log.exception("backtest worker failed")
            b.backtest_done.emit({
                "request_id": self.request_id, "result": None, "df": None,
                "strategy": self.strategy_name, "symbol": self.symbol,
                "timeframe": self.timeframe,
                "error": f"{type(e).__name__}: {e}",
            })


# ---------------------------------------------------------------------------
# data load (dashboard)
# ---------------------------------------------------------------------------

class _DataSignals(QObject):
    ready = Signal(str, str, str, object)   # exchange, symbol, timeframe, df
    error = Signal(str, str)                # request-key, message


class DataLoadWorker(QRunnable):
    """Load OHLCV off the GUI thread; deliver via ``self.signals``.

    ``setAutoDelete(False)`` — the caller keeps a reference until the queued
    signal is delivered (otherwise the Signals QObject could be collected
    before the event is processed).
    """

    def __init__(self, exchange: str, symbol: str, timeframe: str,
                 since: str | None = None, refresh: bool = True):
        super().__init__()
        self.setAutoDelete(False)
        self.signals = _DataSignals()
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.since = since
        self.refresh = bool(refresh)

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.symbol}:{self.timeframe}"

    def run(self) -> None:
        try:
            from core.data.fetcher import load_ohlcv
            df = load_ohlcv(self.exchange, self.symbol, self.timeframe,
                            since=self.since, refresh=self.refresh)
            self.signals.ready.emit(self.exchange, self.symbol, self.timeframe, df)
        except Exception as e:
            log.exception("data load failed")
            self.signals.error.emit(self.key, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class SearchWorker(QThread):
    """Run a mass parameter search; progress + result via the bus."""

    def __init__(self, spec, n_workers: int | None = None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.n_workers = n_workers

    def run(self) -> None:
        b = bus()
        try:
            def cb(done: int, total: int) -> None:
                b.search_progress.emit(int(done), int(total))

            df = run_search(self.spec, n_workers=self.n_workers, progress_cb=cb)
            b.search_done.emit({"results": df, "error": None, "spec": self.spec})
        except Exception as e:
            log.exception("search worker failed")
            b.search_done.emit({"results": None, "spec": self.spec,
                                "error": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

class WalkForwardWorker(QThread):
    """Run a rolling walk-forward validation; own progress/done signals."""

    progress = Signal(int, int)
    done = Signal(object)   # {"result": WalkForwardResult | None, "error": str|None}

    def __init__(self, spec, n_workers: int | None = None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.n_workers = n_workers

    def run(self) -> None:
        try:
            def cb(done: int, total: int) -> None:
                self.progress.emit(int(done), int(total))

            res = run_walkforward(self.spec, n_workers=self.n_workers,
                                  progress_cb=cb)
            self.done.emit({"result": res, "error": None})
        except Exception as e:
            log.exception("walk-forward worker failed")
            self.done.emit({"result": None, "error": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# live engine
# ---------------------------------------------------------------------------

class LiveEngineController:
    """Owns a :class:`LiveEngine` running on a dedicated daemon thread.

    Wires :class:`LiveCallbacks` (fired on the engine thread) to event-bus
    signals so panels observe the engine without importing the exchange layer.
    """

    def __init__(self, exchange_id: str, symbol: str, timeframe: str,
                 strategy, broker, risk_limits=None,
                 price_holder: PriceHolder | None = None,
                 engine_kwargs: dict | None = None):
        self._exchange_id = exchange_id
        self._symbol = symbol
        self._timeframe = timeframe
        self._strategy = strategy
        self._broker = broker
        self._risk_limits = risk_limits
        self._price_holder = price_holder
        self._engine_kwargs = dict(engine_kwargs or {})
        self._engine: LiveEngine | None = None
        self._thread: threading.Thread | None = None

    def _build_callbacks(self) -> LiveCallbacks:
        b = bus()
        sym = self._symbol
        holder = self._price_holder

        def on_bar(bar: dict) -> None:
            if holder is not None:
                try:
                    holder.set(bar["c"])
                except Exception:
                    pass
            b.bar_closed.emit(sym, bar)

        def on_tick(price: float) -> None:
            if holder is not None:
                holder.set(price)
            b.tick.emit(sym, float(price))

        def on_order(fill: dict) -> None:
            b.order_update.emit(fill)

        def on_position(pos) -> None:
            b.position_update.emit(pos)

        def on_equity(ts_ms, equity) -> None:
            b.equity_update.emit((int(ts_ms), float(equity)))

        def on_log(msg: str) -> None:
            b.engine_log.emit("INFO", str(msg))

        def on_status(status: str) -> None:
            b.engine_status.emit(str(status))

        return LiveCallbacks(on_bar=on_bar, on_tick=on_tick, on_order=on_order,
                             on_position=on_position, on_equity=on_equity,
                             on_log=on_log, on_status=on_status)

    def start(self) -> None:
        if self.running:
            return
        self._engine = LiveEngine(
            self._exchange_id, self._symbol, self._timeframe, self._strategy,
            self._broker, risk_limits=self._risk_limits,
            callbacks=self._build_callbacks(), **self._engine_kwargs)
        self._thread = threading.Thread(
            target=self._engine.run_forever, name=f"live-{self._symbol}",
            daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if self._engine is not None:
            self._engine.stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    def re_arm(self) -> None:
        if self._engine is not None:
            self._engine.re_arm()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        return self._engine.status if self._engine is not None else "stopped"
