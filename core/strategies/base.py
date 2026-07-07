"""Strategy interface.

A Strategy converts an OHLCV DataFrame into a *stance* series:
+1 desired long, -1 desired short, 0 flat — decided at each bar's CLOSE.
The engine executes stance changes at the NEXT bar's open, so strategies
must never use future data when computing stance[i].

Engine-level exits (stop-loss / take-profit / ATR trailing) are expressed via
reserved param keys: ``sl_pct``, ``tp_pct``, ``trail_atr_mult`` — the optimizer
copies them into BacktestConfig automatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Reserved engine params any strategy may include in PARAM_SPACE.
ENGINE_PARAM_KEYS = ("sl_pct", "tp_pct", "trail_atr_mult", "atr_period")


class Strategy(ABC):
    """Base class. Subclasses define NAME, PARAM_SPACE, DEFAULTS, and generate_signals."""

    NAME: str = "base"
    #: Recommended timeframes for the search phase.
    TIMEFRAMES: tuple[str, ...] = ("1h", "4h", "1d")
    #: dict param -> list of candidate values (grid for the optimizer).
    PARAM_SPACE: dict[str, list] = {}
    #: dict param -> default value.
    DEFAULTS: dict = {}
    #: Whether the strategy can emit -1 (shorts). Spot search keeps long-only.
    SUPPORTS_SHORT: bool = False

    def __init__(self, **params):
        merged = dict(self.DEFAULTS)
        merged.update(params)
        self.params = merged

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Return stance array of float in {-1, 0, +1}, same length as df.

        Warmup region must be 0. stance[i] may only depend on rows <= i.
        """

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        """Optional per-bar fraction of equity (0..1) to deploy on entries.

        Used for conviction/vol scaling (e.g. multi-lookback agreement).
        Same no-lookahead rule applies. Return None to use config.size_frac.
        """
        return None

    # -- helpers -------------------------------------------------------------
    def engine_params(self) -> dict:
        """Extract reserved engine-level exit params present in self.params."""
        return {k: self.params[k] for k in ENGINE_PARAM_KEYS if k in self.params}

    def strategy_params(self) -> dict:
        return {k: v for k, v in self.params.items() if k not in ENGINE_PARAM_KEYS}

    def describe(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.NAME}({ps})"

    @classmethod
    def param_grid_size(cls) -> int:
        size = 1
        for v in cls.PARAM_SPACE.values():
            size *= len(v)
        return size


def hold_stance(entries: np.ndarray, exits: np.ndarray, allow_short: bool = False,
                short_entries: np.ndarray | None = None,
                short_exits: np.ndarray | None = None) -> np.ndarray:
    """Build a stance array from boolean entry/exit event arrays.

    Vectorized state machine: long entry sets stance 1 until an exit event;
    short arrays (optional) set -1 similarly. Simultaneous exit+entry on the
    same bar prefers the entry.
    """
    n = len(entries)
    stance = np.zeros(n)
    state = 0
    for i in range(n):
        if state == 1:
            if exits[i]:
                state = 0
        elif state == -1:
            if short_exits is not None and short_exits[i]:
                state = 0
        if state == 0:
            if entries[i]:
                state = 1
            elif allow_short and short_entries is not None and short_entries[i]:
                state = -1
        stance[i] = state
    return stance
