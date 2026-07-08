"""Market regime classifier (research brief section 3.1).

States: 0=RANGE, 1=TREND, 2=CRISIS, 3=TRANSITION.

TREND entry needs 2-of-3 confluence {ADX > adx_in, CHOP < 38.2,
close > EMA(ema_period) and EMA50 slope > 0} and, once entered, holds via
hysteresis until ADX < adx_out. RANGE needs ADX < adx_out AND CHOP > 61.8.
CRISIS (realized vol above its rolling-1y 90th percentile, expanding during
warmup — never a whole-series percentile) overrides everything immediately.
Non-crisis state changes require 2 consecutive bars agreeing (dwell filter).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.constants import TIMEFRAME_MINUTES, periods_per_year
from core.indicators import adx as _adx, choppiness, ema, realized_vol, sma

log = logging.getLogger(__name__)

RANGE: int = 0
TREND: int = 1
CRISIS: int = 2
TRANSITION: int = 3

CHOP_TREND = 38.2
CHOP_RANGE = 61.8
_DAYS_PER_YEAR = 365


def classify(df: pd.DataFrame, timeframe: str, adx_period: int = 14,
             adx_in: float = 25.0, adx_out: float = 20.0,
             chop_period: int = 14, ema_period: int = 200,
             vol_period: int = 30, crisis_vol_pct: float = 0.90) -> np.ndarray:
    """Classify each bar into {RANGE, TREND, CRISIS, TRANSITION} (int8).

    Parameters
    ----------
    df : OHLCV frame (DatetimeIndex UTC, open/high/low/close/volume).
    timeframe : bar timeframe string (annualization + day->bar conversion).
    vol_period : realized-vol window in DAYS.
    crisis_vol_pct : rolling 1y percentile of realized vol that triggers CRISIS.

    No lookahead: state[i] uses rows <= i only.
    """
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    n = len(c)

    adx_v, _, _ = _adx(h, l, c, adx_period)
    ch = choppiness(h, l, c, chop_period)
    e_long = ema(c, ema_period)
    e50 = ema(c, 50)
    slope_up = np.zeros(n, dtype=bool)
    slope_up[1:] = (e50[1:] - e50[:-1]) > 0.0

    bpd = max(1, 1440 // TIMEFRAME_MINUTES[timeframe])
    vol_bars = vol_period * bpd
    rv = realized_vol(c, vol_bars, ppy=periods_per_year(timeframe))
    # rolling 1y percentile; min_periods makes it expanding during warmup
    win = _DAYS_PER_YEAR * bpd
    vol_thresh = (pd.Series(rv).rolling(win, min_periods=vol_bars)
                  .quantile(crisis_vol_pct).to_numpy())

    # NaN comparisons are False -> warmup contributes no votes
    votes = ((adx_v > adx_in).astype(np.int8)
             + (ch < CHOP_TREND).astype(np.int8)
             + ((c > e_long) & slope_up).astype(np.int8))
    trend_cond = votes >= 2
    range_cond = (adx_v < adx_out) & (ch > CHOP_RANGE)
    crisis_cond = rv > vol_thresh

    out = np.empty(n, dtype=np.int8)
    state = TRANSITION
    pend = -1
    pend_n = 0
    for i in range(n):
        if crisis_cond[i]:
            # crisis is a risk override — takes effect immediately, no dwell
            state = CRISIS
            pend, pend_n = -1, 0
            out[i] = state
            continue
        if state == TREND and adx_v[i] >= adx_out:  # NaN -> False -> no hold
            cand = TREND
        elif trend_cond[i]:
            cand = TREND
        elif range_cond[i]:
            cand = RANGE
        else:
            cand = TRANSITION
        if cand == state:
            pend, pend_n = -1, 0
        else:
            if cand == pend:
                pend_n += 1
            else:
                pend, pend_n = cand, 1
            if pend_n >= 2:  # dwell: 2 consecutive bars must agree
                state = cand
                pend, pend_n = -1, 0
        out[i] = state
    return out


def btc_risk_on(btc_daily_close, sma_period: int = 150) -> bool:
    """Portfolio risk-on gate: BTC daily close above its `sma_period`-day SMA.

    Returns False (risk-off, conservative) when history is insufficient.
    """
    c = np.asarray(btc_daily_close, dtype=np.float64)
    if len(c) < sma_period:
        return False
    s = sma(c, sma_period)
    return bool(c[-1] > s[-1])


# ---------------------------------------------------------------------------
# Round-2 candidate #5: realized-vol term-structure regime overlay.
#
# A pure-OHLCV overlay that reads the *ratio* of short-window to long-window
# realized (Parkinson) volatility. When short-run vol expands past its own
# longer-run baseline the market is entering a vol-expansion / crash leg
# (RISK_OFF); when short-run vol compresses well below baseline the tape is
# quiet/ranging (RANGING). This is intended to be judged on incremental
# left-tail (MDD/Calmar) reduction on top of the existing per-strategy vol
# targeting, NOT as a standalone alpha (research brief round2 #5).
# ---------------------------------------------------------------------------

NORMAL: int = 0
RISK_OFF: int = 1
RANGING: int = 2

# Parkinson (1980) range-estimator constant: var = 1/(4 ln2) * E[ln(H/L)^2].
_PARKINSON_C = 1.0 / (4.0 * np.log(2.0))


def parkinson_vol(high, low, period: int) -> np.ndarray:
    """Annualized Parkinson high-low volatility over a trailing ``period`` window.

    Parkinson (1980): ``var = 1/(4 ln2) * mean(ln(H/L)**2)`` over the window.
    Uses only the intra-bar high-low range (no close-to-close overnight gaps),
    which makes it a lower-variance vol estimate than close-close std.

    DAILY basis: pass DAILY high/low arrays; the result is annualized with
    ``sqrt(365)`` (the annualization is irrelevant to the short/long *ratio*
    used by :func:`rv_ratio_state`, but keeps the standalone value a real vol).

    No lookahead: ``value[i]`` uses rows ``<= i`` only (trailing window); the
    first ``period-1`` entries are NaN (warmup).
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    n = len(h)
    out = np.full(n, np.nan)
    if n < period or period < 1:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        hl2 = np.log(h / l) ** 2
    hl2 = np.where(np.isfinite(hl2), hl2, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(hl2, period)
    with np.errstate(invalid="ignore"):
        mean_hl2 = np.nanmean(w, axis=1)
    var = _PARKINSON_C * mean_hl2
    out[period - 1:] = np.sqrt(np.maximum(var, 0.0)) * np.sqrt(_DAYS_PER_YEAR)
    return out


def rv_ratio_state(df: pd.DataFrame, short_d: int = 10, long_d: int = 30,
                   risk_off: float = 1.25, ranging: float = 0.8,
                   ema_smooth: int = 3, hysteresis: float = 0.05) -> np.ndarray:
    """Realized-vol term-structure regime on a DAILY OHLCV frame.

    Returns an ``int8`` array aligned to ``df`` with values in
    {``NORMAL``=0, ``RISK_OFF``=1, ``RANGING``=2}.

    ``ratio = parkinson_vol(short_d) / parkinson_vol(long_d)``, then
    EMA-smoothed with ``span=ema_smooth`` to damp single-day spikes. A short
    window that is running hot relative to its own longer baseline
    (``ratio >= risk_off``) is RISK_OFF; a compressed short window
    (``ratio <= ranging``) is RANGING; in-between is NORMAL.

    Whipsaw control (brief §3.1): a ``hysteresis`` margin in ratio units keeps
    an extreme state latched until the ratio retraces ``hysteresis`` past the
    trigger, and a 2-day dwell requires two consecutive agreeing days before
    any state change commits — this prevents flag flip-flop and the next-open
    slippage churn it would cause.

    No lookahead: ``state[i]`` uses daily bars ``<= i`` only.

    PREVIOUS-COMPLETED-DAY semantics (important): ``state[i]`` is decided at the
    CLOSE of daily bar ``i`` and is only actionable from the NEXT day's open
    onward. A caller mapping this onto bar ``i`` itself, or onto intraday bars
    of day ``i``, MUST shift the daily state forward by one day first. See
    ``scripts/rv_overlay_check.py`` for the canonical alignment (state of day
    ``D`` governs trading on day ``D+1``).
    """
    if long_d <= short_d:
        raise ValueError("long_d must exceed short_d")
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    n = len(h)
    short_pv = parkinson_vol(h, l, short_d)
    long_pv = parkinson_vol(h, l, long_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(long_pv > 0, short_pv / long_pv, np.nan)

    # EMA-smooth only the valid (post-warmup, contiguous) region.
    sm = np.full(n, np.nan)
    valid = np.isfinite(ratio)
    if valid.any():
        sm[valid] = (pd.Series(ratio[valid])
                     .ewm(span=max(1, int(ema_smooth)), adjust=False)
                     .mean().to_numpy())

    hi_off = risk_off - hysteresis   # must retrace below this to leave RISK_OFF
    lo_off = ranging + hysteresis    # must rise above this to leave RANGING
    out = np.zeros(n, dtype=np.int8)
    state = NORMAL
    pend, pend_n = -1, 0
    for i in range(n):
        r = sm[i]
        if np.isnan(r):
            out[i] = NORMAL
            continue
        if r >= risk_off:
            cand = RISK_OFF
        elif r <= ranging:
            cand = RANGING
        elif state == RISK_OFF and r >= hi_off:
            cand = RISK_OFF            # hysteresis latch
        elif state == RANGING and r <= lo_off:
            cand = RANGING             # hysteresis latch
        else:
            cand = NORMAL
        if cand == state:
            pend, pend_n = -1, 0
        else:
            if cand == pend:
                pend_n += 1
            else:
                pend, pend_n = cand, 1
            if pend_n >= 2:            # 2-day dwell
                state = cand
                pend, pend_n = -1, 0
        out[i] = state
    return out
