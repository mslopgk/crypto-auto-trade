"""Live trading panel: configure, start/stop, and monitor a live/paper engine.

Left: engine configuration (venue/symbol/timeframe, strategy + params, mode,
API keys with keyring persistence, paper balance, risk limits). Right: a
read-only monitor (status lamp, current stance, positions, equity sparkline,
order log) fed EXCLUSIVELY by event-bus signals — the panel never reads engine
internals across the thread boundary (research brief §4.1).
"""
from __future__ import annotations

import dataclasses
import logging
from collections import deque

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from core.constants import DEFAULT_SYMBOLS, cost_profile
from core.risk import RiskLimits
from core.strategies.registry import all_strategies
from gui.event_bus import bus
from gui.theme import COLORS
from gui.widgets.equity import EquityChart
from gui.widgets.param_form import ParamForm
from gui.workers import LiveEngineController, PriceHolder, instantiate_strategy

log = logging.getLogger(__name__)

_TIMEFRAMES = ("1h", "4h", "1d")
_KEYRING_SERVICE = "crypto-auto-trade"

_STATUS_COLOR = {
    "running": COLORS["up"],
    "stopped": COLORS["text"],
    "halted_daily": COLORS["warn"],
    "halted_mdd": COLORS["err"],
    "error": COLORS["err"],
}
_STATUS_KO = {
    "running": "실행 중", "stopped": "정지", "halted_daily": "일일손실 중단",
    "halted_mdd": "낙폭 킬스위치", "error": "오류",
}
_ORDER_COLS = ["시각", "구분", "수량", "가격", "금액", "수수료", "사유"]


class LivePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._controller: LiveEngineController | None = None
        self._price_holder: PriceHolder | None = None
        self._equity_pts: deque[tuple[int, float]] = deque(maxlen=5000)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_config())
        splitter.addWidget(self._build_monitor())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1040])

        root = QVBoxLayout(self)
        root.addWidget(splitter)

        # monitor is fed ONLY by the bus
        bus().engine_status.connect(self._on_status)
        bus().position_update.connect(self._on_position)
        bus().order_update.connect(self._on_order)
        bus().equity_update.connect(self._on_equity)

        self._on_exchange_changed(self._exchange.currentText())
        self._on_strategy_changed(self._strategy.currentText())
        self._load_keys()

    # ------------------------------------------------------------ config side
    def _build_config(self) -> QWidget:
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setMaximumWidth(420)
        box = QWidget()
        v = QVBoxLayout(box)

        cfg = QGroupBox("설정")
        form = QFormLayout(cfg)
        self._exchange = QComboBox()
        self._exchange.addItems(list(DEFAULT_SYMBOLS.keys()))
        self._exchange.currentTextChanged.connect(self._on_exchange_changed)
        form.addRow("거래소", self._exchange)
        self._symbol = QComboBox()
        form.addRow("심볼", self._symbol)
        self._timeframe = QComboBox()
        self._timeframe.addItems(_TIMEFRAMES)
        self._timeframe.setCurrentText("4h")
        form.addRow("타임프레임", self._timeframe)
        self._strategy = QComboBox()
        self._strategy.addItems(list(all_strategies().keys()))
        self._strategy.currentTextChanged.connect(self._on_strategy_changed)
        form.addRow("전략", self._strategy)
        v.addWidget(cfg)

        pbox = QGroupBox("파라미터")
        pl = QVBoxLayout(pbox)
        self._params = ParamForm()
        pl.addWidget(self._params)
        v.addWidget(pbox)

        mbox = QGroupBox("모드")
        ml = QHBoxLayout(mbox)
        self._mode_paper = QRadioButton("페이퍼 트레이딩")
        self._mode_real = QRadioButton("실거래")
        self._mode_paper.setChecked(True)
        self._mode_paper.toggled.connect(self._on_mode_changed)
        ml.addWidget(self._mode_paper)
        ml.addWidget(self._mode_real)
        v.addWidget(mbox)

        self._paper_box = QGroupBox("페이퍼 잔고")
        pf = QFormLayout(self._paper_box)
        self._paper_balance = QDoubleSpinBox()
        self._paper_balance.setRange(1.0, 1e12)
        self._paper_balance.setValue(10_000.0)
        self._paper_balance.setGroupSeparatorShown(True)
        pf.addRow("초기 잔고(견적통화)", self._paper_balance)
        v.addWidget(self._paper_box)

        self._api_box = QGroupBox("API 키")
        af = QFormLayout(self._api_box)
        self._api_key = QLineEdit()
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        af.addRow("API Key", self._api_key)
        self._api_secret = QLineEdit()
        self._api_secret.setEchoMode(QLineEdit.EchoMode.Password)
        af.addRow("Secret", self._api_secret)
        key_btns = QHBoxLayout()
        save_btn = QPushButton("키 저장")
        save_btn.clicked.connect(self._save_keys)
        load_btn = QPushButton("키 불러오기")
        load_btn.clicked.connect(self._load_keys)
        key_btns.addWidget(save_btn)
        key_btns.addWidget(load_btn)
        af.addRow(key_btns)
        self._api_box.setEnabled(False)
        v.addWidget(self._api_box)

        self._risk_box = self._build_risk_form()
        v.addWidget(self._risk_box)

        btns = QHBoxLayout()
        self._start_btn = QPushButton("시작")
        self._start_btn.clicked.connect(self._start)
        self._stop_btn = QPushButton("정지")
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setEnabled(False)
        btns.addWidget(self._start_btn)
        btns.addWidget(self._stop_btn)
        v.addLayout(btns)
        v.addStretch(1)

        outer.setWidget(box)
        return outer

    def _build_risk_form(self) -> QGroupBox:
        box = QGroupBox("리스크 한도")
        form = QFormLayout(box)
        self._risk_widgets: dict[str, QWidget] = {}
        defaults = RiskLimits()
        for f in dataclasses.fields(RiskLimits):
            val = getattr(defaults, f.name)
            if f.type == "int" or isinstance(val, int) and not isinstance(val, bool):
                w = QSpinBox()
                w.setRange(0, 1_000_000)
                w.setValue(int(val))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(0.0, 1e9)
                w.setSingleStep(0.005)
                w.setValue(float(val))
            self._risk_widgets[f.name] = w
            form.addRow(f.name, w)
        return box

    def _risk_limits(self) -> RiskLimits:
        kwargs = {}
        for name, w in self._risk_widgets.items():
            kwargs[name] = w.value()
        # coerce int field
        kwargs["max_positions"] = int(kwargs["max_positions"])
        return RiskLimits(**kwargs)

    # ------------------------------------------------------------ monitor side
    def _build_monitor(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)

        top = QHBoxLayout()
        self._lamp = QLabel("●")
        self._lamp.setStyleSheet(f"color: {COLORS['text']}; font-size: 20px;")
        self._status_lbl = QLabel("정지")
        self._stance_lbl = QLabel("현재 포지션: 없음(플랫)")
        top.addWidget(self._lamp)
        top.addWidget(self._status_lbl)
        top.addStretch(1)
        top.addWidget(self._stance_lbl)
        v.addLayout(top)

        pos_box = QGroupBox("포지션")
        pl = QVBoxLayout(pos_box)
        self._pos_table = QTableWidget(0, 4)
        self._pos_table.setHorizontalHeaderLabels(["수량", "진입가", "진입시각", "스탠스"])
        self._pos_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._pos_table.verticalHeader().setVisible(False)
        self._pos_table.setMaximumHeight(90)
        pl.addWidget(self._pos_table)
        v.addWidget(pos_box)

        eq_box = QGroupBox("자본 (라이브)")
        el = QVBoxLayout(eq_box)
        self._equity = EquityChart()
        el.addWidget(self._equity)
        v.addWidget(eq_box, 1)

        log_box = QGroupBox("주문 로그")
        ll = QVBoxLayout(log_box)
        self._order_table = QTableWidget(0, len(_ORDER_COLS))
        self._order_table.setHorizontalHeaderLabels(_ORDER_COLS)
        self._order_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._order_table.verticalHeader().setVisible(False)
        ll.addWidget(self._order_table)
        v.addWidget(log_box, 1)
        return box

    # -------------------------------------------------------------- reactions
    def _on_exchange_changed(self, exchange: str) -> None:
        self._symbol.blockSignals(True)
        self._symbol.clear()
        self._symbol.addItems(DEFAULT_SYMBOLS.get(exchange, []))
        self._symbol.blockSignals(False)
        self._load_keys()

    def _on_strategy_changed(self, name: str) -> None:
        if name:
            self._params.set_strategy(name)

    def _on_mode_changed(self, _checked: bool) -> None:
        real = self._mode_real.isChecked()
        self._api_box.setEnabled(real)
        self._paper_box.setEnabled(not real)

    # -------------------------------------------------------------- keyring
    def _key_names(self) -> tuple[str, str]:
        ex = self._exchange.currentText()
        return f"{ex}_api_key", f"{ex}_secret"

    def _save_keys(self) -> None:
        try:
            import keyring
            kname, sname = self._key_names()
            keyring.set_password(_KEYRING_SERVICE, kname, self._api_key.text())
            keyring.set_password(_KEYRING_SERVICE, sname, self._api_secret.text())
            QMessageBox.information(self, "API 키", "키를 저장했습니다.")
        except Exception as e:
            QMessageBox.warning(self, "API 키", f"저장 실패: {e}")

    def _load_keys(self) -> None:
        try:
            import keyring
            kname, sname = self._key_names()
            k = keyring.get_password(_KEYRING_SERVICE, kname)
            s = keyring.get_password(_KEYRING_SERVICE, sname)
            self._api_key.setText(k or "")
            self._api_secret.setText(s or "")
        except Exception as e:
            log.warning("keyring load failed: %s", e)

    # --------------------------------------------------------------- start/stop
    def _start(self) -> None:
        if self._controller is not None and self._controller.running:
            return
        exchange = self._exchange.currentText()
        symbol = self._symbol.currentText()
        timeframe = self._timeframe.currentText()
        if not (exchange and symbol and timeframe):
            return
        real = self._mode_real.isChecked()
        if real:
            text, ok = QInputDialog.getText(
                self, "실거래 확인",
                "실거래를 시작하려면 '실거래' 를 정확히 입력하세요:")
            if not ok or text.strip() != "실거래":
                QMessageBox.information(self, "실거래", "실거래 시작이 취소되었습니다.")
                return

        try:
            strategy = instantiate_strategy(
                self._strategy.currentText(), self._params.values(), timeframe)
        except Exception as e:
            QMessageBox.critical(self, "전략 오류", f"{type(e).__name__}: {e}")
            return

        try:
            broker, holder = self._build_broker(exchange, symbol, real)
        except Exception as e:
            QMessageBox.critical(self, "브로커 오류", f"{type(e).__name__}: {e}")
            return

        self._price_holder = holder
        self._equity_pts.clear()
        self._equity.set_equity(pd.Series(dtype="float64"))
        self._order_table.setRowCount(0)

        self._controller = LiveEngineController(
            exchange, symbol, timeframe, strategy, broker,
            risk_limits=self._risk_limits(), price_holder=holder)
        try:
            self._controller.start()
        except Exception as e:
            QMessageBox.critical(self, "엔진 오류", f"{type(e).__name__}: {e}")
            return
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._set_config_enabled(False)

    def _build_broker(self, exchange: str, symbol: str, real: bool):
        quote = symbol.split("/")[1] if "/" in symbol else "USDT"
        costs = cost_profile(exchange, symbol)
        if real:
            from core.live.broker import CcxtBroker
            key = self._api_key.text().strip()
            secret = self._api_secret.text().strip()
            if not (key and secret):
                raise ValueError("실거래에는 API Key/Secret 이 필요합니다")
            broker = CcxtBroker(exchange, key, secret, sandbox=False)
            return broker, None
        from core.live.broker import PaperBroker
        holder = PriceHolder()
        broker = PaperBroker(
            quote_balance=self._paper_balance.value(),
            fee=costs["fee"], slippage=costs["slippage"],
            price_source=holder.get, quote_currency=quote)
        return broker, holder

    def _stop(self) -> None:
        if self._controller is not None:
            self._controller.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._set_config_enabled(True)

    def _set_config_enabled(self, on: bool) -> None:
        for w in (self._exchange, self._symbol, self._timeframe,
                  self._strategy, self._params, self._mode_paper,
                  self._mode_real, self._paper_box, self._api_box,
                  self._risk_box):
            w.setEnabled(on)
        if on:
            self._on_mode_changed(True)

    def shutdown(self) -> None:
        """Called from MainWindow.closeEvent — stop the engine cleanly."""
        if self._controller is not None:
            self._controller.stop(timeout=8.0)
            self._controller = None

    # ------------------------------------------------------- bus-fed monitor
    def _on_status(self, status: str) -> None:
        color = _STATUS_COLOR.get(status, COLORS["text"])
        self._lamp.setStyleSheet(f"color: {color}; font-size: 20px;")
        self._status_lbl.setText(_STATUS_KO.get(status, status))
        if status in ("stopped", "error"):
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(status != "stopped")
            if status == "stopped":
                self._set_config_enabled(True)

    def _on_position(self, pos) -> None:
        self._pos_table.setRowCount(0)
        if not pos:
            self._stance_lbl.setText("현재 포지션: 없음(플랫)")
            return
        self._stance_lbl.setText("현재 포지션: 롱")
        entry_time = pos.get("entry_time")
        try:
            ts = pd.Timestamp(float(entry_time), unit="s", tz="UTC")
            entry_str = ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            entry_str = str(entry_time)
        vals = [f"{float(pos.get('amount', 0)):.8g}",
                f"{float(pos.get('entry_price', 0)):.8g}",
                entry_str,
                "롱" if int(pos.get("stance", 0)) == 1 else str(pos.get("stance"))]
        self._pos_table.insertRow(0)
        for c, val in enumerate(vals):
            self._pos_table.setItem(0, c, QTableWidgetItem(val))

    def _on_order(self, fill) -> None:
        if not isinstance(fill, dict):
            return
        try:
            ts = pd.Timestamp(float(fill.get("timestamp")), unit="s", tz="UTC")
            tstr = ts.strftime("%H:%M:%S")
        except Exception:
            tstr = str(fill.get("timestamp"))
        side = fill.get("side", "")
        side_ko = "매수" if side == "buy" else ("매도" if side == "sell" else side)
        vals = [tstr, side_ko,
                f"{float(fill.get('amount', 0)):.6g}",
                f"{float(fill.get('price', 0)):.6g}",
                f"{float(fill.get('cost', 0)):.2f}",
                f"{float(fill.get('fee', 0)):.4g}",
                str(fill.get("reason", ""))]
        self._order_table.insertRow(0)
        for c, val in enumerate(vals):
            item = QTableWidgetItem(val)
            if c == 1:
                from PySide6.QtGui import QBrush, QColor
                item.setForeground(QBrush(QColor(
                    COLORS["up"] if side == "buy" else COLORS["down"])))
            self._order_table.setItem(0, c, item)
        while self._order_table.rowCount() > 500:
            self._order_table.removeRow(self._order_table.rowCount() - 1)

    def _on_equity(self, point) -> None:
        try:
            ts_ms, equity = point
        except (TypeError, ValueError):
            return
        self._equity_pts.append((int(ts_ms), float(equity)))
        if len(self._equity_pts) < 2:
            return
        idx = pd.DatetimeIndex(
            pd.to_datetime([p[0] for p in self._equity_pts], unit="ms", utc=True),
            name="timestamp")
        series = pd.Series([p[1] for p in self._equity_pts], index=idx,
                           name="equity")
        self._equity.set_equity(series)
