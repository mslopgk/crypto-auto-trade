"""Optimizer panel: mass parameter search + walk-forward validation.

Build a :class:`SearchSpec` from check-lists and spin boxes, run it off-thread
via :class:`SearchWorker` (progress on the bus), and browse the flat result
rows in a sortable table. Double-click a row to load it into the backtest tab;
the Walk-Forward button runs a rolling validation on the selected row and shows
per-fold + stitched results in a dialog. A previous ``results/*.parquet`` search
can be reloaded.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import (QDate, QSortFilterProxyModel, Qt, Signal)
from PySide6.QtWidgets import (QComboBox, QDateEdit, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QTableView, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from core.constants import DEFAULT_SYMBOLS
from core.optimize.search import SearchSpec, load_search_results
from core.optimize.walkforward import WalkForwardSpec
from core.strategies.registry import all_strategies
from gui.event_bus import bus
from gui.widgets.tables import DataFrameModel
from gui.workers import SearchWorker, WalkForwardWorker

log = logging.getLogger(__name__)

_TIMEFRAMES = ("1h", "4h", "1d")

#: result column -> display column
_RESULT_COLS = [
    ("strategy", "strategy"), ("symbol", "symbol"), ("timeframe", "tf"),
    ("sharpe", "sharpe"), ("cagr", "cagr"), ("max_drawdown", "mdd"),
    ("profit_factor", "pf"), ("n_trades", "n_trades"), ("params", "params"),
]


def _checked_items(lst: QListWidget) -> list[str]:
    out = []
    for i in range(lst.count()):
        it = lst.item(i)
        if it.checkState() == Qt.CheckState.Checked:
            out.append(it.text())
    return out


def _fill_checklist(lst: QListWidget, values, checked=()) -> None:
    lst.clear()
    checked = set(checked)
    for v in values:
        it = QListWidgetItem(v)
        it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Checked if v in checked
                         else Qt.CheckState.Unchecked)
        lst.addItem(it)


class OptimizerPanel(QWidget):
    #: emitted with a backtest-config dict when a result row is double-clicked
    send_to_backtest = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: pd.DataFrame | None = None
        self._search_worker: SearchWorker | None = None

        root = QHBoxLayout(self)
        root.addWidget(self._build_spec_form(), 0)
        root.addWidget(self._build_results(), 1)

        bus().search_progress.connect(self._on_progress)
        bus().search_done.connect(self._on_search_done)

        self._on_exchange_changed(self._exchange.currentText())

    # ------------------------------------------------------------- spec form
    def _build_spec_form(self) -> QWidget:
        box = QGroupBox("검색 설정")
        box.setMaximumWidth(360)
        v = QVBoxLayout(box)

        row = QFormLayout()
        self._exchange = QComboBox()
        self._exchange.addItems(list(DEFAULT_SYMBOLS.keys()))
        self._exchange.currentTextChanged.connect(self._on_exchange_changed)
        row.addRow("거래소", self._exchange)
        v.addLayout(row)

        v.addWidget(QLabel("심볼"))
        self._symbols = QListWidget()
        self._symbols.setMaximumHeight(110)
        v.addWidget(self._symbols)

        v.addWidget(QLabel("타임프레임"))
        self._tfs = QListWidget()
        self._tfs.setMaximumHeight(80)
        _fill_checklist(self._tfs, _TIMEFRAMES, checked=("4h", "1d"))
        v.addWidget(self._tfs)

        v.addWidget(QLabel("전략 (검색 대상)"))
        self._strategies = QListWidget()
        self._strategies.setMaximumHeight(150)
        searchable = [n for n, c in all_strategies().items()
                      if getattr(c, "SEARCHABLE", True)]
        _fill_checklist(self._strategies, searchable, checked=searchable)
        v.addWidget(self._strategies)

        params = QFormLayout()
        self._since = QDateEdit()
        self._since.setCalendarPopup(True)
        self._since.setDisplayFormat("yyyy-MM-dd")
        self._since.setDate(QDate(2020, 1, 1))
        params.addRow("시작일(since)", self._since)

        self._max_combos = QSpinBox()
        self._max_combos.setRange(1, 100_000)
        self._max_combos.setValue(200)
        params.addRow("최대 조합수/전략", self._max_combos)

        self._holdout = QSpinBox()
        self._holdout.setRange(0, 2000)
        self._holdout.setValue(180)
        params.addRow("홀드아웃(일)", self._holdout)

        self._cost_mult = QDoubleSpinBox()
        self._cost_mult.setRange(1.0, 5.0)
        self._cost_mult.setSingleStep(0.5)
        self._cost_mult.setValue(1.0)
        params.addRow("비용 배수", self._cost_mult)

        self._workers = QSpinBox()
        self._workers.setRange(0, 64)
        self._workers.setValue(0)
        self._workers.setToolTip("0 = 단일 프로세스(직렬)")
        params.addRow("워커 수", self._workers)
        v.addLayout(params)

        self._run_btn = QPushButton("검색 실행")
        self._run_btn.clicked.connect(self._run_search)
        v.addWidget(self._run_btn)

        self._load_btn = QPushButton("이전 결과 불러오기…")
        self._load_btn.clicked.connect(self._load_previous)
        v.addWidget(self._load_btn)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        v.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        v.addWidget(self._status)
        v.addStretch(1)
        return box

    def _build_results(self) -> QWidget:
        box = QGroupBox("결과")
        v = QVBoxLayout(box)

        self._model = DataFrameModel()
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(Qt.ItemDataRole.UserRole)

        self._view = QTableView()
        self._view.setModel(self._proxy)
        self._view.setSortingEnabled(True)
        self._view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self._view.doubleClicked.connect(self._on_double_click)
        v.addWidget(self._view, 1)

        btns = QHBoxLayout()
        self._wf_btn = QPushButton("Walk-Forward 검증")
        self._wf_btn.clicked.connect(self._run_walkforward)
        btns.addWidget(self._wf_btn)
        btns.addStretch(1)
        self._count_lbl = QLabel("")
        btns.addWidget(self._count_lbl)
        v.addLayout(btns)
        return box

    # -------------------------------------------------------------- reactions
    def _on_exchange_changed(self, exchange: str) -> None:
        syms = DEFAULT_SYMBOLS.get(exchange, [])
        _fill_checklist(self._symbols, syms, checked=syms[:1])

    def _build_spec(self) -> SearchSpec | None:
        symbols = _checked_items(self._symbols)
        tfs = _checked_items(self._tfs)
        strategies = _checked_items(self._strategies)
        if not symbols or not tfs or not strategies:
            QMessageBox.warning(self, "검색 설정",
                                "심볼, 타임프레임, 전략을 각각 1개 이상 선택하세요.")
            return None
        return SearchSpec(
            exchange=self._exchange.currentText(),
            symbols=symbols,
            timeframes=tfs,
            strategies=strategies,
            max_combos_per_strategy=self._max_combos.value(),
            since=self._since.date().toString("yyyy-MM-dd"),
            cost_multiplier=self._cost_mult.value(),
            holdout_days=self._holdout.value(),
        )

    def _run_search(self) -> None:
        if self._search_worker is not None and self._search_worker.isRunning():
            return
        spec = self._build_spec()
        if spec is None:
            return
        self._run_btn.setEnabled(False)
        self._status.setText("검색 시작…")
        self._progress.setValue(0)
        n = self._workers.value()
        self._search_worker = SearchWorker(spec, n_workers=(n or 0))
        self._search_worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        self._status.setText(f"진행: {done}/{total}")

    def _on_search_done(self, payload: dict) -> None:
        self._run_btn.setEnabled(True)
        error = payload.get("error")
        if error:
            self._status.setText("오류")
            QMessageBox.critical(self, "검색 오류", str(error))
            return
        self._set_results(payload["results"])
        self._status.setText("완료")

    def _load_previous(self) -> None:
        from core.constants import RESULTS_DIR
        path, _ = QFileDialog.getOpenFileName(
            self, "검색 결과 불러오기", str(RESULTS_DIR),
            "Parquet (*.parquet)")
        if not path:
            return
        try:
            df = load_search_results(path)
        except Exception as e:
            QMessageBox.critical(self, "불러오기 오류", f"{type(e).__name__}: {e}")
            return
        self._set_results(df)
        self._status.setText(f"불러옴: {len(df)}행")

    # ----------------------------------------------------------------- table
    def _set_results(self, df: pd.DataFrame) -> None:
        self._results = df.reset_index(drop=True)
        display = self._to_display(self._results)
        self._model.set_dataframe(display)
        self._count_lbl.setText(f"{len(self._results)}행")
        if "sharpe" in display.columns:
            col = list(display.columns).index("sharpe")
            self._view.sortByColumn(col, Qt.SortOrder.DescendingOrder)
        self._view.resizeColumnsToContents()

    def _to_display(self, df: pd.DataFrame) -> pd.DataFrame:
        data: dict[str, list] = {}
        for src, dst in _RESULT_COLS:
            if src == "params":
                data[dst] = [self._params_str(p) for p in df.get("params", [])]
            elif src in df.columns:
                data[dst] = list(df[src])
            else:
                data[dst] = [np.nan] * len(df)
        return pd.DataFrame(data)

    @staticmethod
    def _params_str(p) -> str:
        import json
        if isinstance(p, dict):
            return json.dumps(p, sort_keys=True)
        return str(p)

    def _selected_source_row(self) -> pd.Series | None:
        sel = self._view.selectionModel()
        if sel is None or not sel.hasSelection():
            return None
        proxy_idx = sel.selectedRows()[0] if sel.selectedRows() \
            else sel.currentIndex()
        src_idx = self._proxy.mapToSource(proxy_idx)
        row = src_idx.row()
        if self._results is None or row < 0 or row >= len(self._results):
            return None
        return self._results.iloc[row]

    def _row_to_cfg(self, row: pd.Series) -> dict:
        params = row.get("params")
        if isinstance(params, str):
            import json
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        return {
            "exchange": self._exchange.currentText(),
            "strategy": row.get("strategy"),
            "symbol": row.get("symbol"),
            "timeframe": row.get("timeframe"),
            "params": params if isinstance(params, dict) else {},
        }

    def _on_double_click(self, proxy_index) -> None:
        src = self._proxy.mapToSource(proxy_index)
        if self._results is None or src.row() >= len(self._results):
            return
        row = self._results.iloc[src.row()]
        self.send_to_backtest.emit(self._row_to_cfg(row))

    # -------------------------------------------------------- walk-forward
    def _run_walkforward(self) -> None:
        row = self._selected_source_row()
        if row is None:
            QMessageBox.information(self, "Walk-Forward",
                                   "먼저 결과 행을 선택하세요.")
            return
        spec = WalkForwardSpec(
            exchange=self._exchange.currentText(),
            symbol=str(row.get("symbol")),
            timeframe=str(row.get("timeframe")),
            strategy=str(row.get("strategy")),
            since=self._since.date().toString("yyyy-MM-dd"),
            max_combos=min(self._max_combos.value(), 150),
        )
        dlg = WalkForwardDialog(spec, n_workers=(self._workers.value() or 0),
                                parent=self)
        dlg.exec()


# ---------------------------------------------------------------------------
# walk-forward dialog
# ---------------------------------------------------------------------------

class WalkForwardDialog(QDialog):
    """Runs a WalkForwardWorker and shows per-fold + stitched results."""

    _FOLD_COLS = ["fold", "oos_start", "oos_end", "oos_sharpe", "oos_cagr",
                  "oos_mdd", "oos_return", "n_trades"]

    def __init__(self, spec: WalkForwardSpec, n_workers: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Walk-Forward · {spec.strategy} {spec.symbol} {spec.timeframe}")
        self.resize(760, 560)
        v = QVBoxLayout(self)

        self._header = QLabel("검증 실행 중…")
        self._header.setWordWrap(True)
        v.addWidget(self._header)

        self._progress = QProgressBar()
        v.addWidget(self._progress)

        v.addWidget(QLabel("폴드별 OOS 성과"))
        self._table = QTableWidget(0, len(self._FOLD_COLS))
        self._table.setHorizontalHeaderLabels(self._FOLD_COLS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        v.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        v.addWidget(self._summary)

        self._close_btn = QPushButton("닫기")
        self._close_btn.clicked.connect(self.reject)
        v.addWidget(self._close_btn)

        self._worker = WalkForwardWorker(spec, n_workers=n_workers, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)

    def _on_done(self, payload: dict) -> None:
        error = payload.get("error")
        if error:
            self._header.setText("오류")
            QMessageBox.critical(self, "Walk-Forward 오류", str(error))
            return
        res = payload["result"]
        self._header.setText(
            f"{res.spec.strategy} · {res.spec.symbol} {res.spec.timeframe} · "
            f"{len(res.folds)}개 폴드 (건너뜀 {res.n_skipped_folds})")
        self._fill_folds(res.folds)
        self._fill_summary(res)

    def _fill_folds(self, folds: list[dict]) -> None:
        self._table.setRowCount(0)
        for f in folds:
            m = f.get("oos_metrics", {})
            vals = [
                f.get("fold"),
                str(f.get("oos_start", ""))[:10],
                str(f.get("oos_end", ""))[:10],
                _num(m.get("sharpe")),
                _pct(m.get("cagr")),
                _pct(m.get("max_drawdown")),
                _pct(m.get("total_return")),
                m.get("n_trades"),
            ]
            r = self._table.rowCount()
            self._table.insertRow(r)
            for c, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter))
                self._table.setItem(r, c, item)

    def _fill_summary(self, res) -> None:
        sm = res.stitched_metrics
        self._summary.setText(
            f"<b>스티칭 OOS</b>  샤프 {_num(sm.get('sharpe'))} · "
            f"CAGR {_pct(sm.get('cagr'))} · MDD {_pct(sm.get('max_drawdown'))} · "
            f"PF {_num(sm.get('profit_factor'))} · 거래 {sm.get('n_trades')}<br>"
            f"<b>WFE</b> {_num(res.wfe)} "
            f"(&ge;0.5 통과, &ge;0.7 우수) · "
            f"수익 폴드비율 {_pct(res.pct_profitable_folds)} · "
            f"파라미터 안정성 {_pct(res.param_stability)}")


def _num(v) -> str:
    try:
        f = float(v)
        return f"{f:.3f}" if np.isfinite(f) else "-"
    except (TypeError, ValueError):
        return "-"


def _pct(v) -> str:
    try:
        f = float(v)
        return f"{f * 100:.2f}%" if np.isfinite(f) else "-"
    except (TypeError, ValueError):
        return "-"
