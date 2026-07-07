"""Ensemble strategies: weighted voting and regime switching.

Neither is grid-searched (``SEARCHABLE = False``): members and weights are
configured by the ensemble-construction stage. The optimizer must treat a
missing ``SEARCHABLE`` attribute as True (base.py predates the flag).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.regime import CRISIS, RANGE, TRANSITION, TREND, classify
from core.strategies.base import Strategy
from core.strategies.registry import get_strategy, register
from core.strategies.trend import shift1

log = logging.getLogger(__name__)

#: keys of classify() kwargs mirrored as RegimeSwitchEnsemble params
_REGIME_KEYS = ("adx_period", "adx_in", "adx_out", "chop_period",
                "ema_period", "vol_period", "crisis_vol_pct")


def _make_member(spec: dict) -> Strategy:
    """Instantiate a member from {name, params[, weight]} via the registry."""
    cls = get_strategy(spec["name"])
    return cls(**spec.get("params", {}))


@register
class VotingEnsemble(Strategy):
    """Weighted vote over member strategies' stances.

    ``members`` is a sequence of dicts {name, params, weight}. Stance is 1
    when the weighted mean of member stances (clipped long-only to [0, 1])
    reaches ``threshold``; the weighted mean itself drives conviction sizing.
    """

    NAME = "voting_ensemble"
    TIMEFRAMES = ("1h", "4h", "1d")
    SEARCHABLE = False
    PARAM_SPACE: dict[str, list] = {}
    DEFAULTS = {
        "members": (
            {"name": "donchian_multi", "params": {}, "weight": 1.0},
            {"name": "ema_cross", "params": {}, "weight": 1.0},
            {"name": "tsmom", "params": {}, "weight": 1.0},
        ),
        "threshold": 0.5,
    }

    def _combined(self, df: pd.DataFrame) -> np.ndarray:
        acc = np.zeros(len(df))
        wsum = 0.0
        for spec in self.params["members"]:
            member = _make_member(spec)
            w = float(spec.get("weight", 1.0))
            if w <= 0.0:
                continue  # floor 0: dropped, never negative-weighted
            acc += w * np.clip(member.generate_signals(df), 0.0, 1.0)
            wsum += w
        return acc / wsum if wsum > 0.0 else acc

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        combined = self._combined(df)
        return (combined >= float(self.params["threshold"])).astype(np.float64)

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        # shifted 1 bar: sizing must be decided at the same close as the stance
        return shift1(np.clip(self._combined(df), 0.0, 1.0), fill=0.0)


@register
class RegimeSwitchEnsemble(Strategy):
    """Route between a trend member and a range member by market regime.

    Per bar: TREND -> trend member's stance, RANGE -> range member's stance,
    CRISIS -> flat. TRANSITION keeps the previously selected member (classify
    already applies dwell/hysteresis, so this only bridges early warmup and
    committed TRANSITION states).
    """

    NAME = "regime_switch_ensemble"
    TIMEFRAMES = ("1h", "4h", "1d")
    SEARCHABLE = False
    PARAM_SPACE: dict[str, list] = {}
    DEFAULTS = {
        "trend_member": {"name": "donchian_single", "params": {}},
        "range_member": {"name": "bb_rsi_meanrev", "params": {}},
        "timeframe": "1d",
        "adx_period": 14, "adx_in": 25.0, "adx_out": 20.0,
        "chop_period": 14, "ema_period": 200,
        "vol_period": 30, "crisis_vol_pct": 0.90,
    }

    def _selection(self, df: pd.DataFrame) -> np.ndarray:
        """Per-bar selector: 0 flat, 1 trend member, 2 range member."""
        regime = classify(df, self.params["timeframe"],
                          **{k: self.params[k] for k in _REGIME_KEYS})
        sel = np.zeros(len(df), dtype=np.int8)
        last = 0
        for i, s in enumerate(regime):
            if s == TREND:
                last = 1
            elif s == RANGE:
                last = 2
            elif s == CRISIS:
                last = 0
            # TRANSITION: keep last selection
            sel[i] = last
        return sel

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        sel = self._selection(df)
        trend_s = np.clip(_make_member(self.params["trend_member"]).generate_signals(df), 0.0, 1.0)
        range_s = np.clip(_make_member(self.params["range_member"]).generate_signals(df), 0.0, 1.0)
        return np.where(sel == 1, trend_s, np.where(sel == 2, range_s, 0.0))

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        trend_m = _make_member(self.params["trend_member"])
        range_m = _make_member(self.params["range_member"])
        trend_f = trend_m.generate_size_frac(df)
        range_f = range_m.generate_size_frac(df)
        if trend_f is None and range_f is None:
            return None
        n = len(df)
        ones = np.ones(n)
        tf_ = ones if trend_f is None else np.nan_to_num(trend_f, nan=0.0)
        rf_ = ones if range_f is None else np.nan_to_num(range_f, nan=0.0)
        # selector shifted 1 bar to match execution timing of the stance
        sel = shift1(self._selection(df).astype(np.float64), fill=0.0)
        return np.where(sel == 1, tf_, np.where(sel == 2, rf_, 0.0))
