"""Log panel: colored, capped log view fed by the event bus.

A :class:`BusLogHandler` attached to the root logger forwards records to
``bus.engine_log(level, msg)``; this panel renders them into a
``QPlainTextEdit`` capped at 5000 blocks, WARN in orange and ERROR/CRITICAL
in red (research brief §4.3).
"""
from __future__ import annotations

import html
import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from gui.event_bus import bus
from gui.theme import COLORS

_LEVEL_COLOR = {
    "DEBUG": COLORS["text"],
    "INFO": COLORS["text"],
    "WARNING": COLORS["warn"],
    "ERROR": COLORS["err"],
    "CRITICAL": COLORS["err"],
}


class BusLogHandler(logging.Handler):
    """Logging handler that re-emits records on ``bus.engine_log``.

    Safe from any thread: the bus lives in the GUI thread, so emissions queue
    into its event loop.
    """

    def __init__(self, level=logging.INFO):
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            bus().engine_log.emit(record.levelname, msg)
        except Exception:  # never let logging raise
            pass


def attach_log_handler(level=logging.INFO) -> BusLogHandler:
    """Install the bus handler on the root logger (call once, from the GUI)."""
    handler = BusLogHandler(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    root.addHandler(handler)
    return handler


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self._autoscroll = QCheckBox("자동 스크롤")
        self._autoscroll.setChecked(True)
        clear_btn = QPushButton("지우기")
        clear_btn.clicked.connect(self._clear)
        controls.addWidget(self._autoscroll)
        controls.addStretch(1)
        controls.addWidget(clear_btn)
        layout.addLayout(controls)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(5000)
        self._view.setFont(QFont("Consolas", 9))
        layout.addWidget(self._view, 1)

        # queued so logging emitted from the GUI thread never re-enters here
        bus().engine_log.connect(self._append,
                                 Qt.ConnectionType.QueuedConnection)

    def _append(self, level: str, message: str) -> None:
        color = _LEVEL_COLOR.get(level, COLORS["text"])
        safe = html.escape(message)
        self._view.appendHtml(
            f'<span style="color:{color};">{safe}</span>')
        if self._autoscroll.isChecked():
            sb = self._view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _clear(self) -> None:
        self._view.clear()
