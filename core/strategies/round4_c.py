"""Round-4 original strategies, group C.

Two long-only-spot originals:

- :class:`FundingQualityBreakout` — Donchian-style breakouts gated by *who* is
  driving the rally. Real Binance USDM funding is used as an orthogonal
  positioning signal: enter breakouts only when the annualized-funding z-score
  is low (spot-led advance), skip them when funding is hot (leverage froth).
  A ``z_max=999`` grid point disables the filter so its marginal value is
  directly readable from the search results.
- :class:`TrendPullback` — dip-buying *with* the trend: after an established
  uptrend, buy a short run of consecutive down closes whose total depth sits in
  a "worth-the-costs but not a breakdown" band; exit on the first up-thrust
  (close above the prior bar's high), a bar time-stop, or the engine stop.

Causality: funding features are daily and computed only from prints settled at
or before each daily close (see :mod:`core.data.funding`); every state machine
here depends only on rows <= i (trailing windows / shift-by-one bands).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.indicators import donchian, ema
from core.strategies.base import Strategy, hold_stance
from core.strategies.funding import funding_z_for_df
from core.strategies.registry import register
from core.strategies.trend import bars_per_day, shift1

log = logging.getLogger(__name__)


def _shiftn(a: np.ndarray, k: int, fill: float = np.nan) -> np.ndarray:
    """Shift forward by k bars: out[i] = a[i-k], out[:k] = fill (k >= 0)."""
    out = np.full(len(a), fill, dtype=np.float64)
    if 0 <= k < len(a):
        out[k:] = a[: len(a) - k] if k > 0 else a
    return out


@register
class FundingQualityBreakout(Strategy):
    """Breakout entries filtered by the *quality* (spot vs leverage) of the move.

    Breakout day = close > the previous ``breakout_n``-day high. Enter long only
    when the funding z-score ``z < z_max`` (a spot-led advance — leverage is not
    crowded), skipping breakouts when ``z >= z_max`` (leverage froth). The z is
    the rolling ``L``-day z-score of daily-annualized funding, as-of the last
    settled print of each day (strictly causal, reused from the funding layer).
    Exit: close < the previous ``exit_n``-day low, or the engine's ATR trailing
    stop ``trail_atr_mult``.

    ``z_max=999`` is a grid point that disables the filter (every breakout taken)
    so the filter's marginal value is directly comparable in the search results.

    A funding z that is NaN (warmup / a missing settlement day) is treated as an
    *unconfirmed* rally and skipped when the filter is on. Missing funding data
    for the symbol -> all-flat stance (logged), matching the funding sleeve.
    """

    NAME = "funding_quality_breakout"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    PARAM_SPACE = {
        # 999 == filter OFF (baseline): take every breakout regardless of funding.
        "z_max": [0.5, 1.0, 1.5, 999.0],
        "L": [30, 45],
        "trail_atr_mult": [2.5, 3.0],
    }
    DEFAULTS = {
        "z_max": 1.0, "L": 30, "trail_atr_mult": 2.5,
        "K": 3, "breakout_n": 20, "exit_n": 10,
        "timeframe": "1d", "symbol": "BTC/USDT", "exchange": "binance",
    }

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        h = df["high"].to_numpy(dtype="float64")
        l = df["low"].to_numpy(dtype="float64")
        c = df["close"].to_numpy(dtype="float64")

        z = funding_z_for_df(df, self.params["symbol"],
                             int(self.params["K"]), int(self.params["L"]))
        if z is None:
            log.warning("funding_quality_breakout: no funding for %s -> flat",
                        self.params["symbol"])
            return np.zeros(n, dtype="float64")

        bpd = bars_per_day(self.params["timeframe"])
        breakout_bars = int(self.params["breakout_n"]) * bpd
        exit_bars = max(1, int(self.params["exit_n"]) * bpd)
        if breakout_bars + 1 > n:
            return np.zeros(n, dtype="float64")

        upper, _ = donchian(h, l, breakout_bars)
        _, lower = donchian(h, l, exit_bars)
        # Compare to the PREVIOUS bar's channel (shift1) — never self-reference.
        breakout = c > shift1(upper)
        exits = c < shift1(lower)

        z_max = float(self.params["z_max"])
        if z_max >= 999.0:
            filt = np.ones(n, dtype=bool)  # filter disabled — baseline
        else:
            filt = (~np.isnan(z)) & (z < z_max)

        entries = breakout & filt
        return hold_stance(entries, exits)


@register
class TrendPullback(Strategy):
    """Dip-buying WITH the trend (regime-consent is an uptrend, not a range).

    Context (established uptrend): close > EMA(``trend_days``) AND
    close > close ``mom_days`` days ago. Trigger: ``k_down`` CONSECUTIVE down
    closes ending at the current bar (close < prior close), AND a total pullback
    depth from the close ``k_down`` bars ago in ``[min_dip, max_dip]`` (deep
    enough to clear costs, shallow enough not to be a breakdown). Enter long;
    exit on the first close > the previous bar's high (momentum resumed), a
    ``hold_max``-bar time-stop, or the engine ``sl_pct``.

    Day-based params (``trend_days``, ``mom_days``) convert to bars via
    ``bars_per_day``; ``k_down`` and ``hold_max`` are counts of bars and do NOT
    convert.
    """

    NAME = "trend_pullback"
    TIMEFRAMES = ("1d", "4h")
    SEARCHABLE = True
    PARAM_SPACE = {
        "trend_days": [100, 200],
        "k_down": [3, 4, 5],
        "min_dip": [0.03],
        "max_dip": [0.12, 0.15],
        "hold_max": [5, 8],
        "sl_pct": [0.04, 0.06],
    }
    DEFAULTS = {
        "trend_days": 200, "k_down": 3, "min_dip": 0.03, "max_dip": 0.12,
        "hold_max": 5, "sl_pct": 0.04, "mom_days": 28, "timeframe": "1d",
    }

    def _entry_ok(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype="float64")
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        trend_bars = int(self.params["trend_days"]) * bpd
        mom_bars = int(self.params["mom_days"]) * bpd
        k = int(self.params["k_down"])
        min_dip = float(self.params["min_dip"])
        max_dip = float(self.params["max_dip"])

        # established-uptrend context (all trailing / lagged -> causal)
        trend_ema = ema(c, trend_bars)
        c_mom = _shiftn(c, mom_bars)
        context = (c > trend_ema) & (c > c_mom)

        # k consecutive down closes ending at the current bar
        down = np.zeros(n, dtype=bool)
        down[1:] = c[1:] < c[:-1]
        kdown = down.copy()
        for j in range(1, k):
            sh = np.zeros(n, dtype=bool)
            sh[j:] = down[: n - j]
            kdown &= sh

        # total pullback depth from the close k bars ago (peak before the run)
        c_kago = _shiftn(c, k)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (c_kago - c) / c_kago
        depth_ok = (depth >= min_dip) & (depth <= max_dip)

        return context & kdown & depth_ok

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        c = df["close"].to_numpy(dtype="float64")
        h = df["high"].to_numpy(dtype="float64")
        entry_ok = self._entry_ok(df)
        prev_high = shift1(h)
        hold_max = max(1, int(self.params["hold_max"]))

        stance = np.zeros(n, dtype="float64")
        state = 0
        held = 0
        for i in range(n):
            if state == 1:
                held += 1
                # momentum resumed (close reclaims prior bar's high) or time-stop
                if (not np.isnan(prev_high[i]) and c[i] > prev_high[i]) or held >= hold_max:
                    state = 0
            if state == 0:
                if entry_ok[i]:
                    state = 1
                    held = 0
            stance[i] = state
        return stance
