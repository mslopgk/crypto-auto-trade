"""Vectorized technical indicators.

All functions accept/return numpy float64 arrays (or pandas Series passthrough via
``.to_numpy()`` by the caller). NaN is used for warmup periods. No lookahead:
value at index i uses data up to and including i.
"""
from __future__ import annotations

import numpy as np


def _as_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a


def sma(x, period: int) -> np.ndarray:
    a = _as_array(x)
    out = np.full_like(a, np.nan)
    if len(a) < period:
        return out
    csum = np.cumsum(np.nan_to_num(a))
    out[period - 1:] = (csum[period - 1:] - np.concatenate(([0.0], csum[:-period]))) / period
    # propagate NaN warmup from input
    bad = np.isnan(a)
    if bad.any():
        # recompute honestly (rare path; inputs are usually clean)
        s = np.convolve(np.where(bad, 0.0, a), np.ones(period), "full")[: len(a)]
        cnt = np.convolve((~bad).astype(float), np.ones(period), "full")[: len(a)]
        with np.errstate(invalid="ignore"):
            out = np.where(cnt >= period, s / period, np.nan)
    return out


def ema(x, period: int) -> np.ndarray:
    """Standard EMA seeded with SMA of the first `period` values."""
    a = _as_array(x)
    n = len(a)
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = np.nanmean(a[:period])
    out[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = alpha * a[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def wilder_smooth(x, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA), used for RSI/ATR/ADX."""
    a = _as_array(x)
    n = len(a)
    out = np.full(n, np.nan)
    if n < period:
        return out
    seed = np.nanmean(a[:period])
    out[period - 1] = seed
    prev = seed
    alpha = 1.0 / period
    for i in range(period, n):
        prev = prev + alpha * (a[i] - prev)
        out[i] = prev
    return out


def rsi(close, period: int = 14) -> np.ndarray:
    c = _as_array(close)
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = 0.0
    loss[0] = 0.0
    avg_gain = wilder_smooth(gain[1:], period)
    avg_loss = wilder_smooth(loss[1:], period)
    ag = np.concatenate(([np.nan], avg_gain))
    al = np.concatenate(([np.nan], avg_loss))
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ag / al
        out = np.where(al == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    out[: period] = np.nan
    return out


def true_range(high, low, close) -> np.ndarray:
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    prev_c = np.concatenate(([np.nan], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    return tr


def atr(high, low, close, period: int = 14) -> np.ndarray:
    return wilder_smooth(true_range(high, low, close), period)


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    c = _as_array(close)
    macd_line = ema(c, fast) - ema(c, slow)
    # signal EMA computed on valid region only
    valid = ~np.isnan(macd_line)
    sig = np.full_like(macd_line, np.nan)
    if valid.sum() >= signal:
        sig[valid] = ema(macd_line[valid], signal)
    hist = macd_line - sig
    return macd_line, sig, hist


def bollinger(close, period: int = 20, num_std: float = 2.0):
    c = _as_array(close)
    mid = sma(c, period)
    n = len(c)
    sd = np.full(n, np.nan)
    if n >= period:
        # rolling std (population) via cumulative sums
        csum = np.cumsum(c)
        csum2 = np.cumsum(c * c)
        s = csum[period - 1:] - np.concatenate(([0.0], csum[:-period]))
        s2 = csum2[period - 1:] - np.concatenate(([0.0], csum2[:-period]))
        var = s2 / period - (s / period) ** 2
        sd[period - 1:] = np.sqrt(np.maximum(var, 0.0))
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return upper, mid, lower


def donchian(high, low, period: int = 20):
    """Highest high / lowest low over the trailing `period` bars INCLUDING current bar.

    For breakout entries compare price to the channel of the PREVIOUS bar
    (callers must shift by 1 to avoid self-referencing breakouts).
    """
    h, l = _as_array(high), _as_array(low)
    n = len(h)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if n >= period:
        upper[period - 1:] = sliding_window_view(h, period).max(axis=1)
        lower[period - 1:] = sliding_window_view(l, period).min(axis=1)
    return upper, lower


def supertrend(high, low, close, period: int = 10, multiplier: float = 3.0):
    """Returns (trend, line): trend +1 while price above line (uptrend), -1 below."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(c)
    a = atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    upper_basic = hl2 + multiplier * a
    lower_basic = hl2 - multiplier * a
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    trend = np.full(n, 0.0)
    line = np.full(n, np.nan)
    start = period  # first valid ATR at period-1
    if n <= start:
        return trend, line
    upper[start] = upper_basic[start]
    lower[start] = lower_basic[start]
    trend[start] = 1.0 if c[start] > upper_basic[start] else -1.0
    for i in range(start + 1, n):
        ub = upper_basic[i]
        lb = lower_basic[i]
        upper[i] = ub if (ub < upper[i - 1] or c[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lb if (lb > lower[i - 1] or c[i - 1] < lower[i - 1]) else lower[i - 1]
        if trend[i - 1] == 1.0:
            trend[i] = -1.0 if c[i] < lower[i] else 1.0
        else:
            trend[i] = 1.0 if c[i] > upper[i] else -1.0
        line[i] = lower[i] if trend[i] == 1.0 else upper[i]
    trend[:start + 1] = np.nan
    return trend, line


def adx(high, low, close, period: int = 14):
    """Returns (adx, plus_di, minus_di)."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(c)
    up_move = np.diff(h, prepend=h[0])
    down_move = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = minus_dm[0] = 0.0
    tr = true_range(h, l, c)
    atr_s = wilder_smooth(tr, period)
    pdm_s = wilder_smooth(plus_dm, period)
    mdm_s = wilder_smooth(minus_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * pdm_s / atr_s
        minus_di = 100.0 * mdm_s / atr_s
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isfinite(dx), dx, np.nan)
    adx_out = np.full(n, np.nan)
    valid = ~np.isnan(dx)
    if valid.sum() >= period:
        adx_out[valid] = wilder_smooth(dx[valid], period)
    return adx_out, plus_di, minus_di


def stochastic(high, low, close, k_period: int = 14, d_period: int = 3):
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(c)
    k = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if n >= k_period:
        hh = sliding_window_view(h, k_period).max(axis=1)
        ll = sliding_window_view(l, k_period).min(axis=1)
        rng = hh - ll
        with np.errstate(divide="ignore", invalid="ignore"):
            k[k_period - 1:] = np.where(rng > 0, 100.0 * (c[k_period - 1:] - ll) / rng, 50.0)
    d = sma(k, d_period)
    return k, d


def choppiness(high, low, close, period: int = 14) -> np.ndarray:
    """Choppiness index: >61.8 = ranging, <38.2 = trending."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = len(c)
    tr = true_range(h, l, c)
    out = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if n >= period:
        tr_sum = sliding_window_view(tr, period).sum(axis=1)
        hh = sliding_window_view(h, period).max(axis=1)
        ll = sliding_window_view(l, period).min(axis=1)
        rng = hh - ll
        with np.errstate(divide="ignore", invalid="ignore"):
            val = 100.0 * np.log10(tr_sum / rng) / np.log10(period)
        out[period - 1:] = np.where(rng > 0, val, np.nan)
    return out


def realized_vol(close, period: int = 30, ppy: float = 365 * 24) -> np.ndarray:
    """Annualized realized volatility of log returns over trailing `period` bars.

    ppy = periods per year for the bar timeframe.
    """
    c = _as_array(close)
    n = len(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(c), prepend=np.nan)
    out = np.full(n, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if n >= period + 1:
        w = sliding_window_view(lr[1:], period)
        out[period:] = w.std(axis=1) * np.sqrt(ppy)
    return out


def rolling_max(x, period: int) -> np.ndarray:
    a = _as_array(x)
    out = np.full(len(a), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if len(a) >= period:
        out[period - 1:] = sliding_window_view(a, period).max(axis=1)
    return out


def rolling_min(x, period: int) -> np.ndarray:
    a = _as_array(x)
    out = np.full(len(a), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    if len(a) >= period:
        out[period - 1:] = sliding_window_view(a, period).min(axis=1)
    return out


def obv(close, volume) -> np.ndarray:
    c, v = _as_array(close), _as_array(volume)
    direction = np.sign(np.diff(c, prepend=c[0]))
    direction[0] = 0.0
    return np.cumsum(direction * v)
