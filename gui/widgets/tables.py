"""Reusable table model for pandas DataFrames.

Cell-level formatting, right-aligned numbers, optional sign-colored columns
(P&L via ``ForegroundRole``, never QSS — research brief §4.3), and a numeric
``UserRole`` so a ``QSortFilterProxyModel`` with ``sortRole = UserRole`` sorts
numerically instead of lexicographically.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor

from gui.theme import COLORS


def fmt_value(val) -> str:
    """Human-readable cell text for an arbitrary DataFrame value."""
    if val is None:
        return ""
    if isinstance(val, float) or isinstance(val, np.floating):
        f = float(val)
        if not np.isfinite(f):
            return "-"
        if f == 0.0:
            return "0"
        if abs(f) >= 1000:
            return f"{f:,.2f}"
        if abs(f) >= 1:
            return f"{f:.4f}"
        return f"{f:.6g}"
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (dict, list, tuple)):
        try:
            return json.dumps(val, default=str, sort_keys=True)
        except TypeError:
            return str(val)
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d %H:%M")
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val)


class DataFrameModel(QAbstractTableModel):
    """Read-only table model over a DataFrame.

    ``color_columns`` names columns rendered green/red by sign (for P&L).
    """

    def __init__(self, df: pd.DataFrame | None = None,
                 color_columns=(), parent=None):
        super().__init__(parent)
        self._df = (df if df is not None else pd.DataFrame()).reset_index(drop=True)
        self._color_cols = set(color_columns)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._df.columns):
                return str(self._df.columns[section])
            return None
        return str(section + 1)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if row >= len(self._df) or col >= len(self._df.columns):
            return None
        colname = self._df.columns[col]
        val = self._df.iat[row, col]

        if role == Qt.ItemDataRole.DisplayRole:
            return fmt_value(val)

        if role == Qt.ItemDataRole.ForegroundRole and colname in self._color_cols:
            try:
                f = float(val)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(f):
                return None
            return QBrush(QColor(COLORS["up"] if f >= 0 else COLORS["down"]))

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(val, (int, float, np.number)):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.UserRole:  # numeric sort key
            if isinstance(val, (int, float, np.number)):
                f = float(val)
                return f if np.isfinite(f) else float("-inf")
            return fmt_value(val)

        return None
