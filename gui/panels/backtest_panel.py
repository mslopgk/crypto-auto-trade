"""Backtest panel: configure and run a single backtest, inspect the result.

Left: a configuration form (exchange/symbol/timeframe, date range, strategy +
dynamic parameter form, cost fields prefilled from the venue cost profile,
capital). Right: result tabs — metrics table, equity curve, price chart with
trade markers, and a trades table (P&L colored via ForegroundRole).
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (QComboBox, QDateEdit, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QMessageBox, QPushButton, QScrollArea, QSplitter,
                               QTableView, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)

from core.constants import DEFAULT_SYMBOLS, cost_profile
from core.strategies.registry import all_strategies
from gui.event_bus import bus
from gui.theme import COLORS
from gui.widgets.chart import CandlestickChart
from gui.widgets.equity import EquityChart
from gui.widgets.param_form import ParamForm
from gui.widgets.tables import DataFrameModel
from gui.workers import BacktestWorker

log = logging.getLogger(__name__)

_TIMEFRAMES = ("1h", "4h", "1d")

#: metric key -> Korean label, formatter
_METRIC_ROWS = [
    ("total_return", "총 수익률", "pct"),
    ("cagr", "CAGR", "pct"),
    ("sharpe", "샤프", "num"),
    ("sortino", "소르티노", "num"),
    ("calmar", "칼마", "num"),
    ("max_drawdown", "최대낙폭(MDD)", "pct"),
    ("ann_volatility", "연변동성", "pct"),
    ("profit_factor", "손익비(PF)", "num"),
    ("win_rate", "승률", "pct"),
    ("n_trades", "거래횟수", "int"),
    ("trades_per_year", "연간거래수", "num"),
    ("avg_trade_ret", "평균거래수익", "pct"),
    ("exposure", "노출비율", "pct"),
    ("final_equity", "최종자본", "money"),
]

_TRADE_COLS = ["entry_time", "exit_time", "direction", "entry_price",
               "exit_price", "units", "pnl", "ret_pct", "bars_held",
               "exit_reason"]


def _fmt_metric(value, kind: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "-"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "int":
        return f"{int(v)}"
    if kind == "money":
        return f"{v:,.2f}"
    return f"{v:.3f}"


class BacktestPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._req_counter = itertools.count(1)
        self._pending_req = None

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_config())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1060])

        root = QVBoxLayout(self)
        root.addWidget(splitter)

        bus().backtest_done.connect(self._on_backtest_done)

        self._on_exchange_changed(self._exchange.currentText())
        self._on_strategy_changed(self._strategy.currentText())

    # ------------------------------------------------------------ config form
    def _build_config(self) -> QWidget:
        box = QGroupBox("설정")
        form = QFormLayout(box)

        self._exchange = QComboBox()
        self._exchange.addItems(list(DEFAULT_SYMBOLS.keys()))
        self._exchange.currentTextChanged.connect(self._on_exchange_changed)
        form.addRow("거래소", self._exchange)

        self._symbol = QComboBox()
        self._symbol.currentTextChanged.connect(lambda _: self._update_costs())
        form.addRow("심볼", self._symbol)

        self._timeframe = QComboBox()
        self._timeframe.addItems(_TIMEFRAMES)
        self._timeframe.setCurrentText("4h")
        form.addRow("타임프레임", self._timeframe)

        self._start = QDateEdit()
        self._start.setCalendarPopup(True)
        self._start.setDisplayFormat("yyyy-MM-dd")
        self._start.setDate(QDate(2024, 1, 1))
        form.addRow("시작일", self._start)

        self._end = QDateEdit()
        self._end.setCalendarPopup(True)
        self._end.setDisplayFormat("yyyy-MM-dd")
        self._end.setDate(QDate.currentDate())
        form.addRow("종료일", self._end)

        self._strategy = QComboBox()
        self._strategy.addItems(list(all_strategies().keys()))
        self._strategy.currentTextChanged.connect(self._on_strategy_changed)
        form.addRow("전략", self._strategy)

        self._params = ParamForm()
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setWidget(self._params)
        param_scroll.setMinimumHeight(160)
        form.addRow(QLabel("파라미터"))
        form.addRow(param_scroll)

        self._fee = QDoubleSpinBox()
        self._fee.setDecimals(5)
        self._fee.setRange(0.0, 1.0)
        self._fee.setSingleStep(0.0001)
        form.addRow("수수료(편도)", self._fee)

        self._slippage = QDoubleSpinBox()
        self._slippage.setDecimals(5)
        self._slippage.setRange(0.0, 1.0)
        self._slippage.setSingleStep(0.0001)
        form.addRow("슬리피지(편도)", self._slippage)

        self._capital = QDoubleSpinBox()
        self._capital.setRange(1.0, 1e12)
        self._capital.setValue(10_000.0)
        self._capital.setGroupSeparatorShown(True)
        form.addRow("초기자본", self._capital)

        self._run_btn = QPushButton("백테스트 실행")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._run)
        form.addRow(self._run_btn)

        self._msg = QLabel("")
        self._msg.setWordWrap(True)
        form.addRow(self._msg)
        return box

    def _build_results(self) -> QWidget:
        self._tabs = QTabWidget()

        self._metrics_table = QTableWidget(0, 2)
        self._metrics_table.setHorizontalHeaderLabels(["지표", "값"])
        self._metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._metrics_table.verticalHeader().setVisible(False)
        self._tabs.addTab(self._metrics_table, "결과")

        self._equity = EquityChart()
        self._tabs.addTab(self._equity, "자본곡선")

        self._chart = CandlestickChart()
        self._tabs.addTab(self._chart, "차트")

        self._trades_model = DataFrameModel(color_columns={"pnl", "ret_pct"})
        self._trades_view = QTableView()
        self._trades_view.setModel(self._trades_model)
        self._trades_view.setSortingEnabled(False)
        self._trades_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self._tabs.addTab(self._trades_view, "거래내역")
        return self._tabs

    # --------------------------------------------------------------- reactions
    def _on_exchange_changed(self, exchange: str) -> None:
        self._symbol.blockSignals(True)
        self._symbol.clear()
        self._symbol.addItems(DEFAULT_SYMBOLS.get(exchange, []))
        self._symbol.blockSignals(False)
        self._update_costs()

    def _on_strategy_changed(self, name: str) -> None:
        if name:
            self._params.set_strategy(name)

    def _update_costs(self) -> None:
        exchange = self._exchange.currentText()
        symbol = self._symbol.currentText()
        if not (exchange and symbol):
            return
        costs = cost_profile(exchange, symbol)
        self._fee.setValue(costs["fee"])
        self._slippage.setValue(costs["slippage"])

    # -------------------------------------------------------------------- run
    def _run(self) -> None:
        strategy = self._strategy.currentText()
        symbol = self._symbol.currentText()
        if not (strategy and symbol):
            return
        req = next(self._req_counter)
        self._pending_req = req
        self._run_btn.setEnabled(False)
        self._msg.setText("실행 중… (최초 실행은 numba 컴파일로 수초 소요)")

        worker = BacktestWorker(
            request_id=req,
            exchange=self._exchange.currentText(),
            symbol=symbol,
            timeframe=self._timeframe.currentText(),
            strategy_name=strategy,
            params=self._params.values(),
            since=self._start.date().toString("yyyy-MM-dd"),
            until=self._end.date().toString("yyyy-MM-dd"),
            fee=self._fee.value(),
            slippage=self._slippage.value(),
            capital=self._capital.value(),
            refresh=True,
        )
        from PySide6.QtCore import QThreadPool
        QThreadPool.globalInstance().start(worker)

    def _on_backtest_done(self, payload: dict) -> None:
        if payload.get("request_id") != self._pending_req:
            return
        self._run_btn.setEnabled(True)
        error = payload.get("error")
        if error:
            self._msg.setText("")
            QMessageBox.critical(self, "백테스트 오류", str(error))
            return
        result = payload["result"]
        df = payload["df"]
        self._msg.setText("완료")
        self._show_metrics(result.metrics)
        self._equity.set_equity(result.equity)
        self._chart.set_data(df)
        self._chart.set_trade_markers(result.trades)
        self._show_trades(result.trades)

    # ---------------------------------------------------------------- display
    def _show_metrics(self, metrics: dict) -> None:
        self._metrics_table.setRowCount(0)
        for key, label, kind in _METRIC_ROWS:
            if key not in metrics:
                continue
            row = self._metrics_table.rowCount()
            self._metrics_table.insertRow(row)
            self._metrics_table.setItem(row, 0, QTableWidgetItem(label))
            item = QTableWidgetItem(_fmt_metric(metrics[key], kind))
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter))
            if key in ("total_return", "cagr", "avg_trade_ret"):
                try:
                    from PySide6.QtGui import QBrush, QColor
                    v = float(metrics[key])
                    item.setForeground(QBrush(QColor(
                        COLORS["up"] if v >= 0 else COLORS["down"])))
                except (TypeError, ValueError):
                    pass
            self._metrics_table.setItem(row, 1, item)

    def _show_trades(self, trades: pd.DataFrame) -> None:
        if trades is None or len(trades) == 0:
            self._trades_model.set_dataframe(pd.DataFrame(columns=_TRADE_COLS))
            return
        cols = [c for c in _TRADE_COLS if c in trades.columns]
        self._trades_model.set_dataframe(trades[cols].copy())

    # -------------------------------------------------------- external hook
    def load_config(self, cfg: dict) -> None:
        """Populate the form from an optimizer result row (double-click hook)."""
        exchange = cfg.get("exchange")
        if exchange and exchange in [self._exchange.itemText(i)
                                     for i in range(self._exchange.count())]:
            self._exchange.setCurrentText(exchange)
        symbol = cfg.get("symbol")
        if symbol:
            if symbol not in [self._symbol.itemText(i)
                              for i in range(self._symbol.count())]:
                self._symbol.addItem(symbol)
            self._symbol.setCurrentText(symbol)
        tf = cfg.get("timeframe")
        if tf:
            if tf not in _TIMEFRAMES:
                self._timeframe.addItem(tf)
            self._timeframe.setCurrentText(tf)
        strategy = cfg.get("strategy")
        if strategy:
            self._strategy.setCurrentText(strategy)
            self._params.set_strategy(strategy)
        params = cfg.get("params")
        if isinstance(params, dict):
            self._params.set_values(params)
