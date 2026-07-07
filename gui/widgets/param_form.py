"""Dynamic strategy-parameter form.

Builds editable widgets for a strategy's tunable parameters from its
``PARAM_SPACE`` (typed by the candidate list) plus any scalar ``DEFAULTS``
that are not in the grid. Non-scalar defaults (ensemble ``members`` tuples,
regime member dicts) and the ``timeframe`` key (owned by the panel's
timeframe combo) are skipped. Shared by the backtest and live panels.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QLineEdit, QSpinBox, QWidget)

from core.strategies.registry import get_strategy

log = logging.getLogger(__name__)

#: params managed elsewhere (timeframe comes from the panel's own combo)
_SKIP_KEYS = {"timeframe"}


def _is_bool(v) -> bool:
    return isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_scalar(v) -> bool:
    return isinstance(v, (bool, int, float, str))


class ParamForm(QWidget):
    """Form of parameter editors for a single strategy.

    Call :meth:`set_strategy` with a registry name to (re)build the fields,
    then :meth:`values` to read the current parameter dict.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        # key -> (widget, kind) where kind in {bool,int,float,str,line}
        self._widgets: dict[str, tuple[QWidget, str]] = {}
        self._name: str | None = None

    # ------------------------------------------------------------------ build
    def set_strategy(self, name: str) -> None:
        self._name = name
        while self._form.rowCount() > 0:
            self._form.removeRow(0)
        self._widgets.clear()

        try:
            cls = get_strategy(name)
        except Exception as e:  # unknown name -> empty form
            log.warning("param form: cannot resolve %s: %s", name, e)
            return

        space = dict(cls.PARAM_SPACE)
        defaults = dict(cls.DEFAULTS)
        keys: list[str] = list(space.keys())
        for k in defaults:
            if k not in space:
                keys.append(k)

        for key in keys:
            if key in _SKIP_KEYS:
                continue
            made = self._make_widget(key, space.get(key), defaults.get(key))
            if made is None:
                continue
            widget, kind = made
            self._widgets[key] = (widget, kind)
            self._form.addRow(key, widget)

    def _make_widget(self, key: str, candidates, default):
        if candidates:
            vals = list(candidates)
            if all(_is_bool(v) for v in vals):
                return self._checkbox(bool(default))
            if all(isinstance(v, str) for v in vals):
                return self._combo([str(v) for v in vals],
                                   str(default) if default is not None else None)
            if all(_is_int(v) for v in vals):
                return self._spin(candidates=vals, default=default)
            if all(_is_number(v) for v in vals):
                return self._dspin(candidates=vals, default=default)
            # mixed / unknown -> fall through to default-based inference

        if _is_bool(default):
            return self._checkbox(bool(default))
        if _is_int(default):
            return self._spin(default=default)
        if isinstance(default, float):
            return self._dspin(default=default)
        if isinstance(default, str):
            return self._line(default)
        # non-scalar (dict / tuple / None) -> not editable, skip
        if default is not None and not _is_scalar(default):
            return None
        # unknown default type with no candidates: show as text
        return self._line("" if default is None else str(default))

    # ------------------------------------------------------------ widget makers
    def _checkbox(self, checked: bool):
        w = QCheckBox()
        w.setChecked(bool(checked))
        return w, "bool"

    def _combo(self, options: list[str], current: str | None):
        w = QComboBox()
        w.addItems(options)
        if current is not None and current in options:
            w.setCurrentText(current)
        return w, "str"

    def _spin(self, candidates=None, default=None):
        w = QSpinBox()
        lo, hi = 0, 1_000_000
        if candidates:
            lo = min(0, int(min(candidates)))
            hi = max(int(max(candidates)) * 10 + 1000, hi)
        w.setRange(lo, hi)
        if default is not None:
            w.setValue(int(default))
        return w, "int"

    def _dspin(self, candidates=None, default=None):
        w = QDoubleSpinBox()
        w.setDecimals(6)
        w.setSingleStep(0.001)
        lo, hi = -1_000_000.0, 1_000_000.0
        w.setRange(lo, hi)
        if default is not None:
            w.setValue(float(default))
        return w, "float"

    def _line(self, text: str):
        w = QLineEdit(text)
        return w, "line"

    # ------------------------------------------------------------------- values
    def values(self) -> dict:
        out: dict = {}
        for key, (widget, kind) in self._widgets.items():
            if kind == "bool":
                out[key] = bool(widget.isChecked())
            elif kind == "int":
                out[key] = int(widget.value())
            elif kind == "float":
                out[key] = float(widget.value())
            elif kind == "str":
                out[key] = widget.currentText()
            else:  # line
                out[key] = widget.text()
        return out

    # ----------------------------------------------------------------- restore
    def set_values(self, params: dict) -> None:
        """Best-effort push of a params dict into the current widgets."""
        for key, value in (params or {}).items():
            entry = self._widgets.get(key)
            if entry is None:
                continue
            widget, kind = entry
            try:
                if kind == "bool":
                    widget.setChecked(bool(value))
                elif kind == "int":
                    widget.setValue(int(value))
                elif kind == "float":
                    widget.setValue(float(value))
                elif kind == "str":
                    widget.setCurrentText(str(value))
                else:
                    widget.setText(str(value))
            except (TypeError, ValueError):
                log.debug("param form: cannot set %s=%r", key, value)
