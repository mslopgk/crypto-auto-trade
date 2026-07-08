"""Trend-following strategies (research brief P1, P3, P4, P6).

All strategies are long/flat (spot). Day-based lookbacks are converted to
bars via the ``timeframe`` param, which the optimizer passes alongside the
grid params (bars_per_day = 1440 / TIMEFRAME_MINUTES[timeframe]).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.constants import TIMEFRAME_MINUTES, periods_per_year
from core.indicators import donchian, ema, realized_vol, rsi
from core.strategies.base import Strategy, hold_stance
from core.strategies.registry import register

log = logging.getLogger(__name__)


def bars_per_day(timeframe: str) -> int:
    """Number of bars per UTC day for a timeframe (>= 1)."""
    return max(1, 1440 // TIMEFRAME_MINUTES[timeframe])


def shift1(a: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Shift forward by one bar: out[i] = a[i-1], out[0] = fill."""
    out = np.empty(len(a), dtype=np.float64)
    out[0] = fill
    out[1:] = a[:-1]
    return out


#: Fixed multi-lookback set in DAYS (research brief P1).
MULTI_LOOKBACK_DAYS: tuple[int, ...] = (5, 10, 20, 30, 60, 90, 150, 250, 360)


@register
class DonchianMulti(Strategy):
    """P1 flagship: Donchian multi-lookback ensemble.

    Each lookback N (days) emits 1 after close breaks the *previous* N-day
    high band and reverts to 0 after close falls below the previous N/2-day
    low band. The continuous agreement is the mean of the sub-signals;
    stance goes long when agreement >= ``entry_threshold``. Position size
    scales with the agreement fraction via :meth:`generate_size_frac`.
    Lookbacks that exceed the available history are dropped.
    """

    NAME = "donchian_multi"
    TIMEFRAMES = ("4h", "1d")
    SEARCHABLE = True
    PARAM_SPACE = {
        "entry_threshold": [0.3, 0.4, 0.5],
        "trail_atr_mult": [0.0, 2.5],
    }
    DEFAULTS = {"entry_threshold": 0.4, "trail_atr_mult": 0.0, "timeframe": "1d"}

    def _agreement(self, df: pd.DataFrame) -> np.ndarray:
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        lookbacks = [d for d in MULTI_LOOKBACK_DAYS if d * bpd + 1 <= n]
        if not lookbacks:
            return np.zeros(n)
        subs = np.zeros((len(lookbacks), n))
        for j, days in enumerate(lookbacks):
            entry_bars = days * bpd
            exit_bars = max(1, (days * bpd) // 2)
            upper, _ = donchian(h, l, entry_bars)
            _, lower = donchian(h, l, exit_bars)
            # donchian includes the current bar -> compare to previous bar's band
            entries = c > shift1(upper)
            exits = c < shift1(lower)
            subs[j] = hold_stance(entries, exits)
        return subs.mean(axis=0)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        agree = self._agreement(df)
        thr = float(self.params["entry_threshold"])
        return (agree >= thr).astype(np.float64)

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        # Shift by 1: the engine reads size_frac at the fill bar i, but the
        # entry was decided at close of i-1 — sizing must use that bar too.
        frac = np.clip(self._agreement(df), 0.0, 1.0)
        return shift1(frac, fill=0.0)


@register
class DonchianSingle(Strategy):
    """Classic single-channel Donchian breakout with opposite-channel exit."""

    NAME = "donchian_single"
    TIMEFRAMES = ("4h", "1d")
    SEARCHABLE = True
    PARAM_SPACE = {
        "entry_n": [10, 20, 55],
        "exit_n": [5, 10, 20],
        "trail_atr_mult": [0.0, 2.0, 2.5, 3.0],
    }
    DEFAULTS = {"entry_n": 20, "exit_n": 10, "trail_atr_mult": 2.5, "timeframe": "1d"}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        entry_bars = int(self.params["entry_n"]) * bpd
        exit_bars = max(1, int(self.params["exit_n"]) * bpd)
        if entry_bars + 1 > n:
            return np.zeros(n)
        upper, _ = donchian(h, l, entry_bars)
        _, lower = donchian(h, l, exit_bars)
        entries = c > shift1(upper)
        exits = c < shift1(lower)
        return hold_stance(entries, exits)


@register
class EMACrossover(Strategy):
    """P4: EMA crossover state with optional long-term EMA regime filter.

    Periods are in BARS of the trading timeframe. ``regime_filter`` = 0
    disables the filter; otherwise longs require close > EMA(regime_filter).
    """

    NAME = "ema_cross"
    TIMEFRAMES = ("4h", "1d")
    SEARCHABLE = True
    PARAM_SPACE = {
        "fast": [9, 12, 21],
        "slow": [21, 50, 55],
        "regime_filter": [0, 100, 200],
        "trail_atr_mult": [0.0, 2.5, 3.0],
    }
    DEFAULTS = {"fast": 21, "slow": 50, "regime_filter": 200, "trail_atr_mult": 0.0}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        if fast >= slow:
            # degenerate grid point; emits no signals
            return np.zeros(len(c))
        long_ok = ema(c, fast) > ema(c, slow)
        rf = int(self.params["regime_filter"])
        if rf > 0:
            long_ok &= c > ema(c, rf)
        return long_ok.astype(np.float64)


@register
class TSMOM(Strategy):
    """P3: vol-scaled time-series momentum.

    Long while close > close ``lookback_days`` ago AND close > EMA(50 bars).
    Sizing = target_vol / realized_vol(30d), clipped to [0, 1].
    """

    NAME = "tsmom"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "lookback_days": [14, 21, 28, 42, 56],
        "target_vol": [0.10, 0.15, 0.20],
    }
    DEFAULTS = {"lookback_days": 21, "target_vol": 0.15, "timeframe": "1d",
                "ema_period": 50, "vol_days": 30}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        lb = int(self.params["lookback_days"]) * bpd
        if lb + 1 > n:
            return np.zeros(n)
        c_lag = np.full(n, np.nan)
        c_lag[lb:] = c[:-lb]
        long_ok = (c > c_lag) & (c > ema(c, int(self.params["ema_period"])))
        return long_ok.astype(np.float64)

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        c = df["close"].to_numpy(dtype=np.float64)
        tf = self.params["timeframe"]
        bpd = bars_per_day(tf)
        rv = realized_vol(c, int(self.params["vol_days"]) * bpd, ppy=periods_per_year(tf))
        tv = float(self.params["target_vol"])
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(np.isfinite(rv) & (rv > 0.0), np.clip(tv / rv, 0.0, 1.0), 0.0)
        # shift 1: sizing decided at the same close as the stance that fills next open
        return shift1(frac, fill=0.0)


@register
class RSIMomentum(Strategy):
    """P6: fast-RSI momentum (cross-of-50 family) with hysteresis.

    Long after RSI > enter_lvl, flat after RSI < exit_lvl.
    """

    NAME = "rsi_momentum"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "rsi_period": [3, 5, 8],
        "enter_lvl": [52.0, 55.0, 60.0],
        "exit_lvl": [40.0, 45.0, 48.0],
    }
    DEFAULTS = {"rsi_period": 5, "enter_lvl": 55.0, "exit_lvl": 45.0}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        r = rsi(c, int(self.params["rsi_period"]))
        entries = r > float(self.params["enter_lvl"])
        exits = r < float(self.params["exit_lvl"])
        return hold_stance(entries, exits)


@register
class TsmomLS(Strategy):
    """Round-3 candidate: long/short vol-scaled TSMOM (perp-sleeve simulation).

    Long leg identical to :class:`TSMOM`. Short leg mirrors it: short while
    close < close ``lookback_days`` ago AND close < EMA(ema_period); in
    ``short_mode='gated'`` the short additionally requires close < EMA of
    ``gate_days`` days (short only in confirmed downtrends). Funding credit
    earned by real perp shorts is deliberately NOT modeled — evaluation is
    conservative for the short leg (shorts receive funding ~92% of the time).
    """

    NAME = "tsmom_ls"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    SUPPORTS_SHORT = True
    PARAM_SPACE = {
        "lookback_days": [14, 21, 28, 42, 56],
        "target_vol": [0.10, 0.15, 0.20],
        "short_mode": ["gated", "symmetric"],
    }
    DEFAULTS = {"lookback_days": 21, "target_vol": 0.15, "timeframe": "1d",
                "ema_period": 50, "vol_days": 30, "short_mode": "gated",
                "gate_days": 200}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        lb = int(self.params["lookback_days"]) * bpd
        if lb + 1 > n:
            return np.zeros(n)
        c_lag = np.full(n, np.nan)
        c_lag[lb:] = c[:-lb]
        fast = ema(c, int(self.params["ema_period"]))
        long_ok = (c > c_lag) & (c > fast)
        short_ok = (c < c_lag) & (c < fast)
        if self.params["short_mode"] == "gated":
            gate = ema(c, int(self.params["gate_days"]) * bpd)
            short_ok &= np.where(np.isnan(gate), False, c < gate)
        stance = np.zeros(n)
        stance[long_ok] = 1.0
        stance[short_ok & ~long_ok] = -1.0
        return stance

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        c = df["close"].to_numpy(dtype=np.float64)
        tf = self.params["timeframe"]
        bpd = bars_per_day(tf)
        rv = realized_vol(c, int(self.params["vol_days"]) * bpd, ppy=periods_per_year(tf))
        tv = float(self.params["target_vol"])
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(np.isfinite(rv) & (rv > 0.0), np.clip(tv / rv, 0.0, 1.0), 0.0)
        return shift1(frac, fill=0.0)
