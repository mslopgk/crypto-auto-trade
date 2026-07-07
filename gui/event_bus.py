"""Application-wide Qt signal bus.

A single :class:`EventBus` QObject is created in (and owned by) the GUI
thread. Worker threads emit its signals directly: because every receiver
lives in the GUI thread, Qt auto-connections deliver the emission as a
queued event, which makes cross-thread UI updates safe. Widgets subscribe
to the bus and never import the exchange layer (research brief §4.1).
"""
from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class EventBus(QObject):
    """Typed signals shared by every panel and worker."""

    #: symbol, last trade price
    tick = Signal(str, float)
    #: symbol, closed bar (pd.Series / dict / (ts, o, h, l, c, v))
    bar_closed = Signal(str, object)
    #: order / fill payload (core.live.broker.Fill or dict)
    order_update = Signal(object)
    #: positions snapshot (list[dict] / dict / DataFrame)
    position_update = Signal(object)
    #: live equity point (float / (ts, equity) / dict / pd.Series)
    equity_update = Signal(object)
    #: level ('INFO'|'WARNING'|...), message
    engine_log = Signal(str, str)
    #: 'stopped' | 'running' | 'halted_daily' | 'halted_mdd' | 'error'
    engine_status = Signal(str)
    #: dict payload {request_id, result, df, error, ...} from BacktestWorker
    backtest_done = Signal(object)
    #: done, total  (optimizer grid progress)
    search_progress = Signal(int, int)
    #: dict payload {results, error, spec} from SearchWorker
    search_done = Signal(object)


_bus: EventBus | None = None
_bus_lock = threading.Lock()


def bus() -> EventBus:
    """Singleton accessor.

    The first call must happen in the GUI thread (app.main does this
    explicitly) so the bus's thread affinity makes emissions from worker
    threads queue into the GUI event loop.
    """
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus
