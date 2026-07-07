"""Volatility breakout strategies (research brief P2).

LarryVB implements the systrader79/Larry Williams open-based daily
volatility breakout on 1h bars with UTC (Binance) daily sessions.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.strategies.base import Strategy
from core.strategies.registry import register
from core.strategies.trend import shift1

log = logging.getLogger(__name__)


@register
class LarryVB(Strategy):
    """P2: open-based volatility breakout, 1h bars, UTC daily reference.

    target = day_open + K * (prev_day_high - prev_day_low). Stance turns 1 on
    the first 1h close above the target and is forced 0 on the last bar of
    the UTC day (bar starting at 23:00), so the engine's next-open execution
    exits at the next day's first open — the canonical exit.

    K is either fixed (``k``) or adaptive (``k_mode='noise'``): the 30-day
    mean of the noise ratio 1 - |dayO - dayC| / (dayH - dayL) of prior days.
    Gates: optional ``ma_gate``-day SMA trend gate on day_open, and a minimum
    expected-range filter K*prev_range/day_open > 3*``min_range_frac``.
    """

    NAME = "larry_vb"
    TIMEFRAMES = ("1h",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "k": [0.4, 0.5, 0.6],
        "k_mode": ["fixed", "noise"],
        "ma_gate": [0, 5],
    }
    DEFAULTS = {"k": 0.5, "k_mode": "noise", "ma_gate": 5,
                "min_range_frac": 0.005, "noise_days": 30, "timeframe": "1h"}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params.get("timeframe", "1h") != "1h":
            raise ValueError("LarryVB works on 1h bars only")

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        idx = df.index

        # per-UTC-day aggregates (index is sorted -> codes monotonic)
        codes = pd.factorize(idx.floor("D"))[0]
        starts = np.flatnonzero(np.diff(codes, prepend=-1))
        ends = np.concatenate((starts[1:], [n])) - 1
        d_open = o[starts]
        d_high = np.maximum.reduceat(h, starts)
        d_low = np.minimum.reduceat(l, starts)
        d_close = c[ends]
        nd = len(starts)

        prev_range = shift1(d_high) - shift1(d_low)

        if str(self.params["k_mode"]) == "noise":
            rng_d = d_high - d_low
            with np.errstate(divide="ignore", invalid="ignore"):
                noise = np.where(rng_d > 0.0,
                                 1.0 - np.abs(d_open - d_close) / rng_d, np.nan)
            nd_days = int(self.params["noise_days"])
            # mean over the previous `noise_days` COMPLETED days (shift 1)
            k_day = (pd.Series(noise).rolling(nd_days, min_periods=nd_days)
                     .mean().shift(1).to_numpy())
        else:
            k_day = np.full(nd, float(self.params["k"]))

        target_d = d_open + k_day * prev_range
        with np.errstate(divide="ignore", invalid="ignore"):
            range_frac = k_day * prev_range / d_open
        ok_day = np.isfinite(target_d) & (range_frac > 3.0 * float(self.params["min_range_frac"]))

        gate = int(self.params["ma_gate"])
        if gate > 0:
            sma_prev = (pd.Series(d_close).rolling(gate, min_periods=gate)
                        .mean().shift(1).to_numpy())
            ok_day &= d_open > sma_prev

        breakout = ok_day[codes] & (c > target_d[codes])
        # hold from first in-day breakout until day end (groupwise cummax)
        held = pd.Series(breakout.astype(np.int8)).groupby(codes).cummax().to_numpy()
        # last 1h bar of the UTC day starts at 23:00 — calendar-based, causal
        day_end = idx.hour == 23
        return np.where(day_end, 0, held).astype(np.float64)
