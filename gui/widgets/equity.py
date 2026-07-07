"""Equity curve widget: equity pane + drawdown % pane, x-linked.

X axis is bar index (gap-free) with the same UTC datetime axis used by the
candlestick chart. Drawdown is peak-to-trough percent, always <= 0.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg

from gui.theme import COLORS
from gui.widgets.chart import TimeIndexAxis, _dt_to_ns

log = logging.getLogger(__name__)


class EquityChart(pg.GraphicsLayoutWidget):
    """Two linked panes: equity line (filled to initial capital) above a
    drawdown % area chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(COLORS["bg"])

        self._axis = TimeIndexAxis("bottom")
        self._eq_plot = self.addPlot(row=0, col=0)
        self._eq_plot.hideAxis("bottom")
        self._dd_plot = self.addPlot(row=1, col=0, axisItems={"bottom": self._axis})
        self._dd_plot.setXLink(self._eq_plot)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)

        for plot in (self._eq_plot, self._dd_plot):
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.getAxis("left").setWidth(72)
            plot.vb.setAutoVisible(y=True)
            plot.enableAutoRange(x=False, y=True)
        self._eq_plot.setLabel("left", "Equity")
        self._dd_plot.setLabel("left", "Drawdown %")
        self._dd_plot.setMouseEnabled(x=True, y=False)

        accent = pg.mkColor(COLORS["accent"])
        fill = pg.mkColor(COLORS["accent"])
        fill.setAlpha(40)
        self._eq_item = pg.PlotDataItem(pen=pg.mkPen(accent, width=1.5),
                                        fillLevel=0.0, brush=pg.mkBrush(fill),
                                        connect="finite")
        down = pg.mkColor(COLORS["down"])
        dfill = pg.mkColor(COLORS["down"])
        dfill.setAlpha(70)
        self._dd_item = pg.PlotDataItem(pen=pg.mkPen(down, width=1.0),
                                        fillLevel=0.0, brush=pg.mkBrush(dfill),
                                        connect="finite")
        self._eq_plot.addItem(self._eq_item)
        self._dd_plot.addItem(self._dd_item)
        # after addItem: clipToView needs the item parented to a ViewBox
        for item in (self._eq_item, self._dd_item):
            item.setDownsampling(auto=True)
            item.setClipToView(True)

        # dashed reference line at initial capital
        self._cap_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(COLORS["text"], width=1,
                         style=pg.QtCore.Qt.PenStyle.DashLine))
        self._cap_line.setVisible(False)
        self._eq_plot.addItem(self._cap_line, ignoreBounds=True)

    def set_equity(self, equity: pd.Series) -> None:
        """Plot an equity curve (index = UTC timestamps, values = equity)."""
        n = len(equity)
        if n == 0:
            self._eq_item.setData([], [])
            self._dd_item.setData([], [])
            self._cap_line.setVisible(False)
            self._axis.set_timestamps(np.empty(0, dtype=np.int64))
            return
        eq = equity.to_numpy(dtype=np.float64)
        x = np.arange(n, dtype=np.float64)
        initial = float(eq[0])

        peak = np.maximum.accumulate(eq)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (eq / peak - 1.0) * 100.0, 0.0)

        self._eq_item.setData(x, eq, connect="finite")
        self._eq_item.setFillLevel(initial)
        self._dd_item.setData(x, dd, connect="finite")
        self._cap_line.setPos(initial)
        self._cap_line.setVisible(True)

        if isinstance(equity.index, pd.DatetimeIndex):
            self._axis.set_timestamps(_dt_to_ns(equity.index))
        else:
            self._axis.set_timestamps(np.empty(0, dtype=np.int64))
        self._eq_plot.setXRange(-0.5, n - 0.5, padding=0.02)
        log.debug("set_equity: %d points", n)
