"""Candlestick chart widget (pyqtgraph) built for live trading.

Rendering model (docs/research-brief.md section 4.3):
- Closed history = one QPicture-backed GraphicsObject, regenerated only on
  ``set_data`` / ``append_closed_bar`` (O(n) once per bar close).
- The forming bar = a separate tiny GraphicsObject repainted per tick (O(1)).
- X axis is BAR INDEX (0..n-1) — no session gaps, no float-precision loss;
  a custom AxisItem maps index -> UTC datetime strings.
- Volume subpane x-linked below the price pane, row stretch 3:1.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from gui.theme import COLORS

log = logging.getLogger(__name__)

pg.setConfigOptions(antialias=False, background=COLORS["bg"], foreground=COLORS["text"])

_BODY_W = 0.7
_DEFAULT_VISIBLE = 300
#: fallback colors cycled for overlays added without an explicit color
_OVERLAY_CYCLE = (COLORS["accent"], COLORS["warn"], "#ab47bc", "#26c6da",
                  "#d4e157", "#f06292")


def _dt_to_ns(values) -> np.ndarray:
    """Datetime-like array -> int64 ns since epoch (UTC)."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    return np.asarray(idx.asi8, dtype=np.int64)


def _fmt_num(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    return f"{v:.5g}"


class TimeIndexAxis(pg.AxisItem):
    """Bottom axis mapping bar index -> formatted UTC datetime.

    Keeps candles gap-free on weekends/outages: positions are integer bar
    indices, timestamps are looked up in the stored index array.
    """

    def __init__(self, orientation: str = "bottom", **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self._ts = np.empty(0, dtype=np.int64)
        self._fmt = "%Y-%m-%d"

    def set_timestamps(self, ts_ns: np.ndarray) -> None:
        self._ts = np.asarray(ts_ns, dtype=np.int64)
        if len(self._ts) >= 2:
            step_s = float(np.median(np.diff(self._ts))) / 1e9
            self._fmt = "%Y-%m-%d" if step_s >= 86400.0 else "%m-%d %H:%M"
        self.picture = None
        self.update()

    def format_index(self, i: int) -> str:
        """Full datetime string for bar i (crosshair label)."""
        if not (0 <= i < len(self._ts)):
            return ""
        dt = datetime.fromtimestamp(self._ts[i] / 1e9, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    def tickStrings(self, values, scale, spacing):
        n = len(self._ts)
        out = []
        for v in values:
            i = int(round(v))
            if 0 <= i < n:
                dt = datetime.fromtimestamp(self._ts[i] / 1e9, tz=timezone.utc)
                out.append(dt.strftime(self._fmt))
            else:
                out.append("")
        return out


class _CandleItem(pg.GraphicsObject):
    """Static candlestick history painted into a QPicture (regen O(n) once)."""

    def __init__(self):
        super().__init__()
        self._pic = QtGui.QPicture()
        self._rect = QtCore.QRectF()
        self._h = np.empty(0)
        self._l = np.empty(0)
        self._n = 0

    def set_bars(self, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                 c: np.ndarray) -> None:
        self.prepareGeometryChange()
        self._h, self._l = h, l
        self._n = len(o)
        self._pic = QtGui.QPicture()
        if self._n:
            painter = QtGui.QPainter(self._pic)
            half = _BODY_W / 2.0
            up_mask = c >= o
            for mask, color in ((up_mask, COLORS["up"]), (~up_mask, COLORS["down"])):
                qc = pg.mkColor(color)
                painter.setPen(pg.mkPen(qc, width=1))
                painter.setBrush(pg.mkBrush(qc))
                for i in np.flatnonzero(mask):
                    x = float(i)
                    painter.drawLine(QtCore.QPointF(x, float(l[i])),
                                     QtCore.QPointF(x, float(h[i])))
                    top = float(max(o[i], c[i]))
                    bot = float(min(o[i], c[i]))
                    painter.drawRect(QtCore.QRectF(x - half, bot, _BODY_W, top - bot))
            painter.end()
            ymin = float(np.nanmin(l))
            ymax = float(np.nanmax(h))
            self._rect = QtCore.QRectF(-0.5, ymin, float(self._n), ymax - ymin)
        else:
            self._rect = QtCore.QRectF()
        self.update()
        self.informViewBoundsChanged()

    def paint(self, p, *args):
        self._pic.play(p)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._n == 0:
            return (None, None)
        if ax == 0:
            return (-0.5, self._n - 0.5)
        # y bounds over the visible x slice so autorange tracks the viewport
        i0, i1 = 0, self._n
        if orthoRange is not None:
            i0 = max(0, int(np.floor(orthoRange[0])))
            i1 = min(self._n, int(np.ceil(orthoRange[1])) + 1)
            if i0 >= i1:
                return (None, None)
        return (float(np.nanmin(self._l[i0:i1])), float(np.nanmax(self._h[i0:i1])))


class _VolumeItem(pg.GraphicsObject):
    """Static volume history bars, colored by candle direction."""

    def __init__(self):
        super().__init__()
        self._pic = QtGui.QPicture()
        self._rect = QtCore.QRectF()
        self._v = np.empty(0)
        self._n = 0

    def set_bars(self, v: np.ndarray, up_mask: np.ndarray) -> None:
        self.prepareGeometryChange()
        self._v = v
        self._n = len(v)
        self._pic = QtGui.QPicture()
        if self._n:
            painter = QtGui.QPainter(self._pic)
            half = _BODY_W / 2.0
            for mask, color in ((up_mask, COLORS["up"]), (~up_mask, COLORS["down"])):
                qc = pg.mkColor(color)
                qc.setAlpha(180)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(pg.mkBrush(qc))
                for i in np.flatnonzero(mask):
                    painter.drawRect(QtCore.QRectF(float(i) - half, 0.0,
                                                   _BODY_W, float(v[i])))
            painter.end()
            vmax = float(np.nanmax(v)) if np.isfinite(v).any() else 1.0
            self._rect = QtCore.QRectF(-0.5, 0.0, float(self._n), max(vmax, 1e-12))
        else:
            self._rect = QtCore.QRectF()
        self.update()
        self.informViewBoundsChanged()

    def paint(self, p, *args):
        self._pic.play(p)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self._rect)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._n == 0:
            return (None, None)
        if ax == 0:
            return (-0.5, self._n - 0.5)
        i0, i1 = 0, self._n
        if orthoRange is not None:
            i0 = max(0, int(np.floor(orthoRange[0])))
            i1 = min(self._n, int(np.ceil(orthoRange[1])) + 1)
            if i0 >= i1:
                return (None, None)
        return (0.0, float(np.nanmax(self._v[i0:i1])))


class _LiveCandleItem(pg.GraphicsObject):
    """Single forming bar; repainted per tick, O(1)."""

    def __init__(self):
        super().__init__()
        self._x = 0.0
        self._bar: tuple[float, float, float, float] | None = None

    def set_bar(self, x: int, o: float, h: float, l: float, c: float) -> None:
        self.prepareGeometryChange()
        self._x = float(x)
        self._bar = (o, h, l, c)
        self.update()
        self.informViewBoundsChanged()

    def clear_bar(self) -> None:
        self.prepareGeometryChange()
        self._bar = None
        self.update()
        self.informViewBoundsChanged()

    def paint(self, p, *args):
        if self._bar is None:
            return
        o, h, l, c = self._bar
        qc = pg.mkColor(COLORS["up"] if c >= o else COLORS["down"])
        p.setPen(pg.mkPen(qc, width=1))
        p.setBrush(pg.mkBrush(qc))
        p.drawLine(QtCore.QPointF(self._x, l), QtCore.QPointF(self._x, h))
        top, bot = max(o, c), min(o, c)
        p.drawRect(QtCore.QRectF(self._x - _BODY_W / 2.0, bot, _BODY_W, top - bot))

    def boundingRect(self) -> QtCore.QRectF:
        if self._bar is None:
            return QtCore.QRectF()
        _, h, l, _ = self._bar
        return QtCore.QRectF(self._x - 0.5, l, 1.0, h - l)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._bar is None:
            return (None, None)
        if ax == 0:
            return (self._x - 0.5, self._x + 0.5)
        if orthoRange is not None and not (orthoRange[0] <= self._x <= orthoRange[1]):
            return (None, None)
        _, h, l, _ = self._bar
        return (l, h)


class _LiveVolumeItem(pg.GraphicsObject):
    """Volume bar of the forming candle."""

    def __init__(self):
        super().__init__()
        self._x = 0.0
        self._v: float | None = None
        self._up = True

    def set_bar(self, x: int, v: float, up: bool) -> None:
        self.prepareGeometryChange()
        self._x = float(x)
        self._v = float(v)
        self._up = up
        self.update()
        self.informViewBoundsChanged()

    def clear_bar(self) -> None:
        self.prepareGeometryChange()
        self._v = None
        self.update()
        self.informViewBoundsChanged()

    def paint(self, p, *args):
        if self._v is None:
            return
        qc = pg.mkColor(COLORS["up"] if self._up else COLORS["down"])
        qc.setAlpha(180)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(pg.mkBrush(qc))
        p.drawRect(QtCore.QRectF(self._x - _BODY_W / 2.0, 0.0, _BODY_W, self._v))

    def boundingRect(self) -> QtCore.QRectF:
        if self._v is None:
            return QtCore.QRectF()
        return QtCore.QRectF(self._x - 0.5, 0.0, 1.0, max(self._v, 1e-12))

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._v is None:
            return (None, None)
        if ax == 0:
            return (self._x - 0.5, self._x + 0.5)
        if orthoRange is not None and not (orthoRange[0] <= self._x <= orthoRange[1]):
            return (None, None)
        return (0.0, self._v)


class CandlestickChart(pg.GraphicsLayoutWidget):
    """Price + volume candlestick chart with live-bar updates.

    Public API::

        set_data(df)                      # full redraw of closed bars
        update_live_bar(o, h, l, c, v)    # repaint forming bar, O(1)
        append_closed_bar(ts, o, h, l, c, v)
        add_line_overlay(name, values, color=None)
        clear_overlays()
        set_trade_markers(trades)         # engine trades schema
        clear_all()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(COLORS["bg"])

        # stored history (closed bars only)
        self._index = np.empty(0, dtype=np.int64)   # ns since epoch, UTC
        self._o = np.empty(0)
        self._h = np.empty(0)
        self._l = np.empty(0)
        self._c = np.empty(0)
        self._v = np.empty(0)
        self._step_ns: int = 0
        self._live: tuple[float, float, float, float, float] | None = None
        self._overlays: dict[str, pg.PlotDataItem] = {}

        self._axis = TimeIndexAxis("bottom")
        self._price = self.addPlot(row=0, col=0)
        self._price.hideAxis("bottom")
        self._volp = self.addPlot(row=1, col=0, axisItems={"bottom": self._axis})
        self._volp.setXLink(self._price)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)
        for plot in (self._price, self._volp):
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.getAxis("left").setWidth(72)
            plot.vb.setAutoVisible(y=True)
            plot.enableAutoRange(x=False, y=True)
        self._volp.setMouseEnabled(x=True, y=False)

        self._candles = _CandleItem()
        self._price.addItem(self._candles)
        self._live_candle = _LiveCandleItem()
        self._price.addItem(self._live_candle)
        self._volbars = _VolumeItem()
        self._volp.addItem(self._volbars)
        self._live_vol = _LiveVolumeItem()
        self._volp.addItem(self._live_vol)

        self._markers = pg.ScatterPlotItem(pxMode=True, size=12)
        self._markers.setZValue(10)
        self._price.addItem(self._markers)

        # crosshair
        xpen = pg.mkPen(COLORS["text"], width=1,
                        style=QtCore.Qt.PenStyle.DashLine)
        self._vline_p = pg.InfiniteLine(angle=90, movable=False, pen=xpen)
        self._hline_p = pg.InfiniteLine(angle=0, movable=False, pen=xpen)
        self._vline_v = pg.InfiniteLine(angle=90, movable=False, pen=xpen)
        for ln in (self._vline_p, self._hline_p):
            ln.setZValue(20)
            self._price.addItem(ln, ignoreBounds=True)
        self._vline_v.setZValue(20)
        self._volp.addItem(self._vline_v, ignoreBounds=True)
        self._set_crosshair_visible(False)

        # corner OHLCV label, parented to the ViewBox so it stays in pixel coords
        self._info = pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(30, 34, 45, 190),
                                 border=pg.mkPen("#2a2e39"))
        self._info.setZValue(30)
        self._info.setParentItem(self._price.vb)
        self._info.setPos(6, 4)
        self._info.setVisible(False)

        self._proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60,
                                     slot=self._on_mouse_moved)

    # ------------------------------------------------------------------ data
    def set_data(self, df: pd.DataFrame) -> None:
        """Full redraw of closed bars (index=UTC timestamps, OHLCV columns).

        Drops existing overlays and trade markers (their bar indices become
        invalid when the dataset changes).
        """
        n = len(df)
        if n:
            self._index = _dt_to_ns(df.index)
            self._o = df["open"].to_numpy(dtype=np.float64)
            self._h = df["high"].to_numpy(dtype=np.float64)
            self._l = df["low"].to_numpy(dtype=np.float64)
            self._c = df["close"].to_numpy(dtype=np.float64)
            self._v = df["volume"].to_numpy(dtype=np.float64)
            self._step_ns = int(np.median(np.diff(self._index))) if n >= 2 else 0
        else:
            self._index = np.empty(0, dtype=np.int64)
            self._o = self._h = self._l = self._c = self._v = np.empty(0)
            self._step_ns = 0
        self._live = None
        self._live_candle.clear_bar()
        self._live_vol.clear_bar()
        self.clear_overlays()
        self._markers.setData([])

        self._candles.set_bars(self._o, self._h, self._l, self._c)
        self._volbars.set_bars(self._v, self._c >= self._o)
        self._axis.set_timestamps(self._index)

        if n:
            x0 = max(-0.5, n - _DEFAULT_VISIBLE - 0.5)
            self._price.setXRange(x0, n - 0.5 + 1.5, padding=0)
        log.debug("set_data: %d bars", n)

    def update_live_bar(self, o: float, h: float, l: float, c: float,
                        v: float) -> None:
        """Repaint the forming bar only (index = len(history)). O(1)."""
        n = len(self._o)
        first = self._live is None
        at_edge = self._at_right_edge()
        self._live = (float(o), float(h), float(l), float(c), float(v))
        self._live_candle.set_bar(n, float(o), float(h), float(l), float(c))
        self._live_vol.set_bar(n, float(v), c >= o)
        if first and at_edge:
            # keep the new bar in view without changing zoom width
            (x0, x1), _ = self._price.vb.viewRange()
            if x1 < n + 0.5:
                shift = (n + 0.5) - x1
                self._price.setXRange(x0 + shift, x1 + shift, padding=0)

    def append_closed_bar(self, ts, o: float, h: float, l: float, c: float,
                          v: float) -> None:
        """Fold the live bar into history on bar close (O(n) picture regen)."""
        at_edge = self._at_right_edge()
        ts_ns = int(pd.Timestamp(ts).value)
        self._index = np.append(self._index, ts_ns)
        self._o = np.append(self._o, float(o))
        self._h = np.append(self._h, float(h))
        self._l = np.append(self._l, float(l))
        self._c = np.append(self._c, float(c))
        self._v = np.append(self._v, float(v))
        if len(self._index) >= 2:
            self._step_ns = int(np.median(np.diff(self._index[-50:])))
        self._live = None
        self._live_candle.clear_bar()
        self._live_vol.clear_bar()
        self._candles.set_bars(self._o, self._h, self._l, self._c)
        self._volbars.set_bars(self._v, self._c >= self._o)
        self._axis.set_timestamps(self._index)
        if at_edge:
            (x0, x1), _ = self._price.vb.viewRange()
            self._price.setXRange(x0 + 1.0, x1 + 1.0, padding=0)

    # -------------------------------------------------------------- overlays
    def add_line_overlay(self, name: str, values,
                         color: str | None = None) -> None:
        """Add/replace a line overlay on the price pane (len == len(df)).

        NaN warmup regions are not drawn (connect='finite').
        """
        vals = np.asarray(values, dtype=np.float64)
        x = np.arange(len(vals), dtype=np.float64)
        if name in self._overlays:
            item = self._overlays[name]
            if color is not None:
                item.setPen(pg.mkPen(color, width=1.2))
            item.setData(x, vals, connect="finite")
            return
        pen_color = color or _OVERLAY_CYCLE[len(self._overlays) % len(_OVERLAY_CYCLE)]
        item = pg.PlotDataItem(x, vals, pen=pg.mkPen(pen_color, width=1.2),
                               connect="finite")
        item.setZValue(5)
        self._price.addItem(item)
        # after addItem: clipToView needs the item parented to a ViewBox
        item.setDownsampling(auto=True)
        item.setClipToView(True)
        self._overlays[name] = item

    def clear_overlays(self) -> None:
        for item in self._overlays.values():
            self._price.removeItem(item)
        self._overlays.clear()

    # --------------------------------------------------------------- markers
    def set_trade_markers(self, trades: pd.DataFrame) -> None:
        """Plot entries (triangle up) / exits (triangle down) from the engine
        trades schema: entry_time/exit_time/direction/entry_price/exit_price."""
        if trades is None or len(trades) == 0 or len(self._index) == 0:
            self._markers.setData([])
            return
        spots: list[dict] = []
        n = len(self._index)
        entry_i = np.clip(np.searchsorted(self._index,
                                          _dt_to_ns(trades["entry_time"])), 0, n - 1)
        exit_i = np.clip(np.searchsorted(self._index,
                                         _dt_to_ns(trades["exit_time"])), 0, n - 1)
        entry_px = trades["entry_price"].to_numpy(dtype=np.float64)
        exit_px = trades["exit_price"].to_numpy(dtype=np.float64)
        up_brush = pg.mkBrush(COLORS["up"])
        down_brush = pg.mkBrush(COLORS["down"])
        edge = pg.mkPen(COLORS["bg"], width=0.5)
        for xi, px in zip(entry_i, entry_px):
            spots.append({"pos": (float(xi), float(px)), "symbol": "t1",
                          "brush": up_brush, "pen": edge, "size": 12})
        for xi, px in zip(exit_i, exit_px):
            spots.append({"pos": (float(xi), float(px)), "symbol": "t",
                          "brush": down_brush, "pen": edge, "size": 12})
        self._markers.setData(spots)

    def clear_all(self) -> None:
        """Remove all data, overlays, markers, and the live bar."""
        self.set_data(pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp")))

    # ------------------------------------------------------------- crosshair
    def _set_crosshair_visible(self, visible: bool) -> None:
        for ln in (self._vline_p, self._hline_p, self._vline_v):
            ln.setVisible(visible)

    def _at_right_edge(self) -> bool:
        """True when the user's viewport includes the most recent bar."""
        n = len(self._o)
        if n == 0:
            return True
        last_x = n if self._live is not None else n - 1
        (_, x1), _ = self._price.vb.viewRange()
        return x1 >= last_x - 1.0

    def _bar_at(self, i: int) -> tuple[float, float, float, float, float] | None:
        n = len(self._o)
        if 0 <= i < n:
            return (self._o[i], self._h[i], self._l[i], self._c[i], self._v[i])
        if i == n and self._live is not None:
            return self._live
        return None

    def _time_at(self, i: int) -> str:
        n = len(self._index)
        if 0 <= i < n:
            return self._axis.format_index(i)
        if i == n and self._live is not None and n and self._step_ns:
            dt = datetime.fromtimestamp((self._index[-1] + self._step_ns) / 1e9,
                                        tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M UTC") + " (live)"
        return ""

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        in_price = self._price.sceneBoundingRect().contains(pos)
        in_vol = self._volp.sceneBoundingRect().contains(pos)
        if not (in_price or in_vol):
            self._set_crosshair_visible(False)
            self._info.setVisible(False)
            return
        vb = self._price.vb if in_price else self._volp.vb
        pt = vb.mapSceneToView(pos)
        i = int(round(pt.x()))
        self._vline_p.setPos(float(i))
        self._vline_v.setPos(float(i))
        self._vline_p.setVisible(True)
        self._vline_v.setVisible(True)
        self._hline_p.setVisible(in_price)
        if in_price:
            self._hline_p.setPos(pt.y())
        bar = self._bar_at(i)
        if bar is None:
            self._info.setVisible(False)
            return
        o, h, l, c, v = bar
        color = COLORS["up"] if c >= o else COLORS["down"]
        self._info.setHtml(
            f'<span style="color:{COLORS["text"]}; font-size:8pt;">'
            f'{self._time_at(i)}&nbsp;&nbsp;</span>'
            f'<span style="color:{color}; font-size:8pt;">'
            f'O {_fmt_num(o)}&nbsp; H {_fmt_num(h)}&nbsp; L {_fmt_num(l)}'
            f'&nbsp; C {_fmt_num(c)}&nbsp;</span>'
            f'<span style="color:{COLORS["text"]}; font-size:8pt;">'
            f'V {_fmt_num(v)}</span>')
        self._info.setVisible(True)
