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
