"""Dashboard panel: live market chart with EMA overlays.

Exchange / symbol / timeframe selectors, a manual refresh and 30s auto-refresh,
a candlestick chart with toggleable EMA20/50/200 overlays. Data is loaded off
the GUI thread via :class:`DataLoadWorker` (never blocks the UI). When a live
engine is running the panel folds in ``bus.bar_closed`` / ``bus.tick`` updates.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from core.constants import DEFAULT_SYMBOLS
from core.indicators import ema
from gui.event_bus import bus
from gui.theme import COLORS
from gui.widgets.chart import CandlestickChart
from gui.workers import DataLoadWorker

log = logging.getLogger(__name__)

_TIMEFRAMES = ("1h", "4h", "1d")
_EMA_SPECS = (("EMA20", 20, "#42a5f5"), ("EMA50", 50, COLORS["warn"]),
              ("EMA200", 200, "#ab47bc"))
_AUTO_REFRESH_MS = 30_000


class DashboardPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._loaders: list[DataLoadWorker] = []

        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self._exchange = QComboBox()
        self._exchange.addItems(list(DEFAULT_SYMBOLS.keys()))
        self._exchange.currentTextChanged.connect(self._on_exchange_changed)
        self._symbol = QComboBox()
        self._timeframe = QComboBox()
        self._timeframe.addItems(_TIMEFRAMES)
        self._timeframe.setCurrentText("4h")

        self._refresh_btn = QPushButton("새로고침")
        self._refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        self._auto = QCheckBox("30초 자동 새로고침")
        self._auto.toggled.connect(self._on_auto_toggled)
        self._status = QLabel("")

        bar.addWidget(QLabel("거래소"))
        bar.addWidget(self._exchange)
        bar.addWidget(QLabel("심볼"))
        bar.addWidget(self._symbol)
        bar.addWidget(QLabel("타임프레임"))
        bar.addWidget(self._timeframe)
        bar.addWidget(self._refresh_btn)
        bar.addWidget(self._auto)
        bar.addStretch(1)
        bar.addWidget(self._status)
        root.addLayout(bar)

        ema_bar = QHBoxLayout()
        ema_bar.addWidget(QLabel("오버레이:"))
        self._ema_checks: dict[str, QCheckBox] = {}
        for name, _period, _color in _EMA_SPECS:
            cb = QCheckBox(name)
            cb.toggled.connect(self._refresh_overlays)
            self._ema_checks[name] = cb
            ema_bar.addWidget(cb)
        ema_bar.addStretch(1)
        self._last_price_lbl = QLabel("")
        ema_bar.addWidget(self._last_price_lbl)
        root.addLayout(ema_bar)

        self._chart = CandlestickChart()
        root.addWidget(self._chart, 1)

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(_AUTO_REFRESH_MS)
        self._auto_timer.timeout.connect(lambda: self.refresh(force=True))

        self._exchange.currentTextChanged.connect(lambda _: self.refresh())
        self._symbol.currentTextChanged.connect(lambda _: self.refresh())
        self._timeframe.currentTextChanged.connect(lambda _: self.refresh())

        bus().bar_closed.connect(self._on_bar_closed)
        bus().tick.connect(self._on_tick)

        self._on_exchange_changed(self._exchange.currentText())

    # -------------------------------------------------------------- selectors
    def _on_exchange_changed(self, exchange: str) -> None:
        self._symbol.blockSignals(True)
        self._symbol.clear()
        self._symbol.addItems(DEFAULT_SYMBOLS.get(exchange, []))
        self._symbol.blockSignals(False)
        self.refresh()

    def _on_auto_toggled(self, on: bool) -> None:
        if on:
            self._auto_timer.start()
        else:
            self._auto_timer.stop()

    # -------------------------------------------------------------- data load
    def refresh(self, force: bool = False) -> None:
        exchange = self._exchange.currentText()
        symbol = self._symbol.currentText()
        timeframe = self._timeframe.currentText()
        if not (exchange and symbol and timeframe):
            return
        self._status.setText("불러오는 중…")
        worker = DataLoadWorker(exchange, symbol, timeframe, refresh=force)
        worker.signals.ready.connect(self._on_data_ready)
        worker.signals.error.connect(self._on_data_error)
        self._loaders.append(worker)
        QThreadPool.globalInstance().start(worker)

    def _drop_loader(self, key: str) -> None:
        self._loaders = [w for w in self._loaders if w.key != key]

    def _on_data_ready(self, exchange: str, symbol: str, timeframe: str,
                       df: pd.DataFrame) -> None:
        self._drop_loader(f"{exchange}:{symbol}:{timeframe}")
        # ignore stale results if the user changed the selection meanwhile
        if (exchange != self._exchange.currentText()
                or symbol != self._symbol.currentText()
                or timeframe != self._timeframe.currentText()):
            return
        self._df = df
        self._chart.set_data(df)
        self._refresh_overlays()
        if len(df):
            self._status.setText(f"{len(df)}개 봉 · 최종 {df.index[-1]:%Y-%m-%d %H:%M}")
            self._last_price_lbl.setText(f"종가 {df['close'].iloc[-1]:,.4g}")
        else:
            self._status.setText("데이터 없음")

    def _on_data_error(self, key: str, message: str) -> None:
        self._drop_loader(key)
        self._status.setText(f"오류: {message}")
        log.warning("dashboard data load error: %s", message)

    # --------------------------------------------------------------- overlays
    def _refresh_overlays(self) -> None:
        if self._df is None or len(self._df) == 0:
            return
        close = self._df["close"].to_numpy(dtype=np.float64)
        # rebuild from scratch so unchecked overlays disappear
        self._chart.clear_overlays()
        if len(close) < 2:
            return
        for name, period, color in _EMA_SPECS:
            if self._ema_checks[name].isChecked():
                self._chart.add_line_overlay(name, ema(close, period), color=color)

    # ------------------------------------------------------------ live hooks
    def _on_bar_closed(self, symbol: str, bar) -> None:
        if symbol != self._symbol.currentText() or self._df is None:
            return
        try:
            ts = pd.Timestamp(int(bar["ts"]), unit="ms", tz="UTC")
            self._chart.append_closed_bar(ts, bar["o"], bar["h"], bar["l"],
                                          bar["c"], bar["v"])
        except Exception:
            log.debug("dashboard bar_closed ignored", exc_info=True)

    def _on_tick(self, symbol: str, price: float) -> None:
        if symbol == self._symbol.currentText():
            self._last_price_lbl.setText(f"현재가 {price:,.4g}")
