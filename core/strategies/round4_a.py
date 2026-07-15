"""Round-4 original strategies, group A.

Three long/flat (spot) originals:

- :class:`WickPressure` — a buying/selling-pressure proxy read straight off
  candle anatomy (wick asymmetry), smoothed and gated by a simple trend
  consent filter, with hysteresis exit and optional engine ATR trail.
- :class:`VoVCalmTrend` — a vol-of-vol *regime* gate on a plain trend core:
  hold the trend only while the volatility-of-volatility is in a historically
  calm percentile, vol-target sized exactly like ``trend.py`` TSMOM.
- :class:`VolumeShockPullback` — "buy the quiet after loud": detect a loud
  up-volume shock day, then buy the first quiet pullback that holds above the
  shock low, riding the continuation via the engine trailing stop.

Conventions copied from ``core/strategies/trend.py``: day-based lookbacks are
converted to bars via the ``timeframe`` param (``bars_per_day``); sizing is
``shift1``-ed by one bar so the fraction the engine reads at the fill bar was
decided at the *close* that produced the entry.

STRICT no-lookahead: ``stance[i]`` depends only on rows ``<= i``. Every rolling
statistic (mean / std / percentile rank) uses a TRAILING window ending at the
current bar -- never a whole-series or centered aggregate.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.constants import periods_per_year
from core.indicators import atr as atr_indicator, realized_vol, sma
from core.strategies.base import Strategy, hold_stance
from core.strategies.registry import register
from core.strategies.trend import bars_per_day, shift1

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# small trailing-window helpers (causal; value at i uses only rows <= i)
# --------------------------------------------------------------------------- #
def _roll_std(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing population std over the last ``w`` bars (inclusive).

    NaN until the whole trailing window is finite, so warmup / embedded NaNs
    never leak a partial estimate. Uses only rows <= i (truncation-invariant).
    """
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    out = np.full(n, np.nan)
    if n < w or w < 1:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, w)
    finite = np.isfinite(win).all(axis=1)
    s = win.std(axis=1)  # ddof=0; window length is constant so the ddof choice
    out[w - 1:] = np.where(finite, s, np.nan)  # is irrelevant to any later rank
    return out


def _roll_pct_rank(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing percentile rank: fraction of the last ``w`` finite values that
    are ``<= a[i]`` (current bar included). Uses only rows <= i.
    """
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    out = np.full(n, np.nan)
    for i in range(n):
        cur = a[i]
        if not np.isfinite(cur):
            continue
        lo = i - w + 1
        if lo < 0:
            lo = 0
        window = a[lo:i + 1]
        fin = window[np.isfinite(window)]
        if fin.size == 0:
            continue
        out[i] = float(np.mean(fin <= cur))
    return out


@register
class WickPressure(Strategy):
    """Buying-pressure proxy from candle anatomy.

    Per bar: ``lower_wick = min(open,close) - low``,
    ``upper_wick = high - max(open,close)``, ``rng = high - low`` (guarded so
    ``rng <= 0`` yields pressure 0). ``pressure = (lower_wick - upper_wick)/rng``
    lives in [-1, 1] -- positive when buyers defended the lows (long lower wick),
    negative when sellers capped the highs.

    Signal ``psum`` is the trailing MEAN of pressure over ``window`` days. Go
    long when ``psum > enter_thr`` AND ``close > SMA(trend_days)`` (trend
    consent). Stay long until ``psum < exit_thr`` (hysteresis) or the engine's
    ATR trailing stop (``trail_atr_mult``) fires. Long/flat only.

    SEARCHABLE=False (mirrors :class:`GatedRSI2`): the shared driftless
    random-walk contract fixture builds symmetric proportional wicks, which
    force ``pressure`` negative on every bar so ``psum`` can never clear the
    positive ``enter_thr`` grid -- the strategy is structurally flat there and
    cannot satisfy the generic ``param_change`` sweep. It is still included in
    the optimizer by explicit name and carries full dedicated contract tests
    (``tests/test_round4_a.py``) on asymmetric-wick and real BTC data.
    """

    NAME = "wick_pressure"
    TIMEFRAMES = ("1d", "4h")
    SEARCHABLE = False
    PARAM_SPACE = {
        "window": [10, 15, 20],
        "enter_thr": [0.05, 0.10, 0.15],
        "trend_days": [50, 100],
        "exit_thr": [0.0, -0.05],
        "trail_atr_mult": [0.0, 2.5],
    }
    DEFAULTS = {
        "window": 15, "enter_thr": 0.10, "trend_days": 50, "exit_thr": 0.0,
        "trail_atr_mult": 0.0, "timeframe": "1d",
    }

    def _pressure(self, df: pd.DataFrame) -> np.ndarray:
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        lower_wick = np.minimum(o, c) - l
        upper_wick = h - np.maximum(o, c)
        rng = h - l
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(rng > 0.0, (lower_wick - upper_wick) / rng, 0.0)
        return p

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        w = max(1, int(self.params["window"]) * bpd)
        trend_bars = max(1, int(self.params["trend_days"]) * bpd)
        if trend_bars + 1 > n:
            return np.zeros(n)
        p = self._pressure(df)
        psum = sma(p, w)
        trend_ma = sma(c, trend_bars)
        enter_thr = float(self.params["enter_thr"])
        exit_thr = float(self.params["exit_thr"])
        # NaN warmup -> comparisons are False -> no entry (stance 0)
        entries = (psum > enter_thr) & (c > trend_ma)
        entries = np.where(np.isnan(psum) | np.isnan(trend_ma), False, entries)
        exits = np.where(np.isnan(psum), False, psum < exit_thr)
        return hold_stance(entries, exits)


@register
class VoVCalmTrend(Strategy):
    """Vol-of-vol regime gate on a simple trend core.

    Per-bar Parkinson vol ``pv = sqrt((ln(H/L))^2 / (4 ln2))``; realized vol
    ``rv = SMA(pv, 10d)``; vol-of-vol ``vov = trailing STD(rv, vov_win)``. The
    regime signal is ``vov_rank`` = the trailing 365-day percentile rank of the
    current ``vov`` (fraction of the last year's ``vov`` values ``<=`` current).

    Trend core: ``close > close[lb days ago]``. Go long when the trend core is
    up AND ``vov_rank < calm_pct`` (calm regime). Exit when the trend core turns
    down OR ``vov_rank > calm_pct + 0.25`` (hysteresis). Position size is
    vol-targeted exactly like ``trend.py`` TSMOM. Long/flat only.
    """

    NAME = "vov_calm_trend"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "vov_win": [30, 45, 60],
        "lb": [21, 28, 42],
        "calm_pct": [0.3, 0.4, 0.5],
        "target_vol": [0.15, 0.20],
    }
    DEFAULTS = {
        "vov_win": 45, "lb": 28, "calm_pct": 0.4, "target_vol": 0.15,
        "timeframe": "1d", "rv_win": 10, "rank_win": 365, "vol_days": 30,
    }

    def _vov_rank(self, df: pd.DataFrame) -> np.ndarray:
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        bpd = bars_per_day(self.params["timeframe"])
        with np.errstate(divide="ignore", invalid="ignore"):
            pv = np.sqrt((np.log(h / l) ** 2) / (4.0 * np.log(2.0)))
        rv = sma(pv, max(1, int(self.params["rv_win"]) * bpd))
        vov = _roll_std(rv, max(1, int(self.params["vov_win"]) * bpd))
        return _roll_pct_rank(vov, max(1, int(self.params["rank_win"]) * bpd))

    def _trend_core(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        lb = max(1, int(self.params["lb"]) * bpd)
        c_lag = np.full(n, np.nan)
        if lb < n:
            c_lag[lb:] = c[:-lb]
        return np.where(np.isnan(c_lag), False, c > c_lag)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        trend = self._trend_core(df)
        rank = self._vov_rank(df)
        calm = float(self.params["calm_pct"])
        rank_ok = np.where(np.isnan(rank), False, rank < calm)
        rank_hot = np.where(np.isnan(rank), False, rank > calm + 0.25)
        entries = trend & rank_ok
        exits = (~trend) | rank_hot
        return hold_stance(entries, exits)

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
class VolumeShockPullback(Strategy):
    """Buy the quiet after loud.

    ``vol_z = (volume - SMA(volume,20d)) / trailing_std(volume,20d)``. A *shock*
    day has ``vol_z > shock_z`` AND ``close > open`` AND ``close > close[1 bar
    ago]`` (a loud up day). Within the next ``look_days`` bars we buy the first
    *quiet pullback*: a bar with ``vol_z < 0`` AND ``low >= shock low`` AND
    ``close < shock close`` (a calm dip that holds above the shock low).

    Position management is a forward state machine (no lookahead). Once long,
    ``confirmed`` flips true the first time ``close > shock high``; from then the
    trade is handed to the engine ATR trailing stop -- represented in stance
    space by a causal close-vs-(peak - trail_atr_mult*ATR) trail so the stance
    and engine stay in lockstep (no re-entry churn). Flat on: close below the
    shock low (downside stop; engine ``sl_pct`` adds the pessimistic intrabar
    version), the time-stop ``hold_max`` bars while still unconfirmed, or the
    confirmed trailing break. Long/flat only.
    """

    NAME = "volume_shock_pullback"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "shock_z": [2.5, 3.0],
        "look_days": [3, 5],
        "hold_max": [5, 10],
        "trail_atr_mult": [2.0, 3.0],
        "sl_pct": [0.05, 0.08],
    }
    DEFAULTS = {
        "shock_z": 2.5, "look_days": 5, "hold_max": 10,
        "trail_atr_mult": 2.5, "sl_pct": 0.05, "atr_period": 14,
        "vol_win": 20, "timeframe": "1d",
    }

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        v = df["volume"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        vol_win = max(1, int(self.params["vol_win"]) * bpd)
        look_bars = max(1, int(self.params["look_days"]) * bpd)
        hold_bars = max(1, int(self.params["hold_max"]))  # spec: hold_max in BARS
        shock_z = float(self.params["shock_z"])
        trail_mult = float(self.params["trail_atr_mult"])
        atr_arr = atr_indicator(h, l, c, int(self.params["atr_period"]))

        vmean = sma(v, vol_win)
        vstd = _roll_std(v, vol_win)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_z = np.where(np.isfinite(vstd) & (vstd > 0.0), (v - vmean) / vstd, np.nan)
        c_prev = shift1(c)
        is_shock = (
            np.isfinite(vol_z) & (vol_z > shock_z)
            & (c > o) & np.isfinite(c_prev) & (c > c_prev)
        )

        stance = np.zeros(n)
        SEARCH, ARMED, POS = 0, 1, 2
        state = SEARCH
        sh_low = sh_high = sh_close = np.nan
        armed_age = 0
        held = 0
        confirmed = False
        peak = np.nan
        pos_shlow = pos_shhigh = np.nan

        for i in range(n):
            if state == POS:
                held += 1
                if np.isnan(peak) or c[i] > peak:
                    peak = c[i]
                if c[i] > pos_shhigh:
                    confirmed = True
                exit_now = False
                if c[i] < pos_shlow:                       # downside stop
                    exit_now = True
                elif (not confirmed) and held >= hold_bars:  # time stop
                    exit_now = True
                elif confirmed and np.isfinite(atr_arr[i]) and \
                        c[i] < peak - trail_mult * atr_arr[i]:  # trailing break
                    exit_now = True
                if exit_now:
                    state = SEARCH
                    stance[i] = 0.0
                    continue
                stance[i] = 1.0
                continue

            if state == ARMED:
                armed_age += 1
                if is_shock[i]:                             # a louder shock re-arms
                    sh_low, sh_high, sh_close = l[i], h[i], c[i]
                    armed_age = 0
                    stance[i] = 0.0
                    continue
                if (np.isfinite(vol_z[i]) and vol_z[i] < 0.0
                        and l[i] >= sh_low and c[i] < sh_close):
                    state = POS                             # quiet pullback -> enter
                    held = 0
                    confirmed = False
                    peak = c[i]
                    pos_shlow, pos_shhigh = sh_low, sh_high
                    stance[i] = 1.0
                    continue
                if armed_age >= look_bars:                  # window expired
                    state = SEARCH
                stance[i] = 0.0
                continue

            # SEARCH
            if is_shock[i]:
                state = ARMED
                sh_low, sh_high, sh_close = l[i], h[i], c[i]
                armed_age = 0
            stance[i] = 0.0

        return stance
