"""Intraday mean-reversion strategies (research brief P7).

Only intraday (1h/4h) mean reversion is allowed, hard-gated to non-trending
conditions (ADX < 20) — daily mean reversion was rejected by research.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.indicators import adx, atr, bollinger, rsi, sma
from core.strategies.base import Strategy
from core.strategies.registry import register
from core.strategies.trend import shift1

log = logging.getLogger(__name__)


@register
class BBRsiMeanRev(Strategy):
    """P7: Bollinger + RSI mean reversion, regime-gated.

    Entry (all at bar close):
      - previous close was below the lower Bollinger(bb_period, bb_std) band
        and the current close crossed back above it;
      - RSI(rsi_period) at the touch bar (previous bar) < ``oversold``;
      - hard gate ADX(14) < ``adx_max`` (default 20);
      - volatility circuit breaker: ATR(14) < 1.5 * SMA(ATR(14), 20).
    Exit: close >= SMA mid-band, or time stop after ``bars_max`` bars.
    Hard stop via engine ``sl_pct``.
    """

    NAME = "bb_rsi_meanrev"
    TIMEFRAMES = ("1h", "4h")
    SEARCHABLE = True
    PARAM_SPACE = {
        "rsi_period": [2, 14],
        "oversold": [10.0, 30.0],
        "bars_max": [12, 24],
        "sl_pct": [0.015, 0.02, 0.03],
    }
    DEFAULTS = {"rsi_period": 14, "oversold": 30.0, "bars_max": 12,
                "sl_pct": 0.02, "bb_period": 20, "bb_std": 2.0,
                "adx_max": 20.0, "atr_breaker_mult": 1.5}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)

        _, mid, lower = bollinger(c, int(self.params["bb_period"]),
                                  float(self.params["bb_std"]))
        r = rsi(c, int(self.params["rsi_period"]))
        adx_v, _, _ = adx(h, l, c, 14)
        atr14 = atr(h, l, c, 14)
        atr_ok = atr14 < float(self.params["atr_breaker_mult"]) * sma(atr14, 20)

        cross_up = (shift1(c) < shift1(lower)) & (c > lower)
        entry_sig = (cross_up
                     & (shift1(r) < float(self.params["oversold"]))
                     & (adx_v < float(self.params["adx_max"]))
                     & atr_ok)

        bars_max = int(self.params["bars_max"])
        stance = np.zeros(n)
        state = 0
        entry_bar = -1
        for i in range(n):
            if state == 1:
                hit_mid = not np.isnan(mid[i]) and c[i] >= mid[i]
                if hit_mid or (i - entry_bar) >= bars_max:
                    state = 0
            if state == 0 and entry_sig[i]:
                state = 1
                entry_bar = i
            stance[i] = state
        return stance
