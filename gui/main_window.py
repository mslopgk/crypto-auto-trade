"""Application shell: tabbed main window with a status bar.

Hosts the five panels (dashboard / backtest / optimizer / live / logs), a
status bar showing the live-engine status and a UTC clock, and wires the
optimizer's "send to backtest" hook. ``closeEvent`` stops the live engine and
drains the thread pool so the process exits cleanly.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget
from PySide6.QtCore import QThreadPool

from gui.event_bus import bus
from gui.panels.backtest_panel import BacktestPanel
from gui.panels.dashboard import DashboardPanel
from gui.panels.live_panel import LivePanel
from gui.panels.log_panel import LogPanel
from gui.panels.optimizer_panel import OptimizerPanel

log = logging.getLogger(__name__)

_STATUS_KO = {
    "stopped": "정지", "running": "실행 중", "halted_daily": "일일손실 중단",
    "halted_mdd": "낙폭 킬스위치", "error": "오류",
}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crypto Auto-Trade — 암호화폐 자동매매")
        self.resize(1440, 900)

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self.dashboard = DashboardPanel()
        self.backtest = BacktestPanel()
        self.optimizer = OptimizerPanel()
        self.live = LivePanel()
        self.logs = LogPanel()

        self._tabs.addTab(self.dashboard, "대시보드")
        self._tabs.addTab(self.backtest, "백테스트")
        self._tabs.addTab(self.optimizer, "최적화")
        self._tabs.addTab(self.live, "실시간 매매")
        self._tabs.addTab(self.logs, "로그")

        self.optimizer.send_to_backtest.connect(self._load_into_backtest)

        self._status_lbl = QLabel("엔진: 정지")
        self._clock_lbl = QLabel("")
        self.statusBar().addWidget(self._status_lbl)
        self.statusBar().addPermanentWidget(self._clock_lbl)

        bus().engine_status.connect(self._on_engine_status)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

    def _tick_clock(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._clock_lbl.setText(now)

    def _on_engine_status(self, status: str) -> None:
        self._status_lbl.setText("엔진: " + _STATUS_KO.get(status, status))

    def _load_into_backtest(self, cfg: dict) -> None:
        self.backtest.load_config(cfg)
        self._tabs.setCurrentWidget(self.backtest)

    def closeEvent(self, event) -> None:
        try:
            self.live.shutdown()
        except Exception:
            log.exception("live shutdown failed")
        QThreadPool.globalInstance().waitForDone(3000)
        super().closeEvent(event)
