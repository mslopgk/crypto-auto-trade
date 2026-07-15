"""Round-4 original strategies, group D (efficiency momentum, VB trail hybrid).

Two ORIGINAL designs (round-4 research):

1. ``efficiency_momentum`` — path-quality momentum. Only hold trends whose
   price path is *clean* (small retracement per unit of net travel), measured
   by a signed Kaufman-efficiency-like ratio. Vol-targeted sizing further
   scaled by trend conviction.
2. ``vb_trail_hybrid`` — the rejected ``larry_vb`` day-trade turned into a
   trend-riding swing: same canonical open-range breakout ENTRY, but instead
   of the forced day-end exit it rides an engine ATR trailing stop and exits
   on a failed-breakout signal (close back below the current day's open).

Both obey the strict no-lookahead contract: ``stance[i]`` depends on rows
``<= i`` only; warmup region stance is 0; long-only (spot).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.constants import periods_per_year
from core.indicators import realized_vol
from core.strategies.base import Strategy, hold_stance
from core.strategies.registry import register
from core.strategies.trend import bars_per_day, shift1

log = logging.getLogger(__name__)


@register
class EfficiencyMomentum(Strategy):
    """Path-quality momentum: hold only CLEAN trends.

    Over a lookback of ``lb`` days:
      * ``net``  = close / close[lb] - 1               (net travel, simple)
      * ``path`` = sum of |daily log returns| over lb  (total path length)
      * ``efficiency`` = net / path  (signed Kaufman-ER-like; guard path<=0)

    A high positive efficiency means the price marched up with little
    retracement (a clean uptrend); efficiency near 0 means choppy / directionless.

    Long when ``efficiency > eff_min`` (net positive AND path clean); exit when
    ``efficiency < eff_exit`` (hysteresis band). Sizing is TSMOM-style vol
    targeting (``target_vol`` / realized_vol) TIMES a conviction multiplier
    ``min(1, efficiency/0.5)`` clipped to [0, 1] (shift1 both, engine reads the
    fill-bar index i but the decision was made at close of i-1).
    """

    NAME = "efficiency_momentum"
    TIMEFRAMES = ("1d",)
    # Excluded from the *default* all-strategies sweep (still fully searchable
    # when named explicitly, e.g. the round-4 WFA). Rationale: a clean-trend
    # filter needs a genuine directional regime — its efficiency ratio over a
    # multi-week (lb-day) window on the short DRIFTLESS random-walk contract
    # fixture stays ~0.05, far below eff_min, so it trades zero there and would
    # only add dead weight to the generic sweep (same reasoning as gated_rsi2).
    # Full contract coverage lives in tests/test_round4_d.py on trending data.
    SEARCHABLE = False
    PARAM_SPACE = {
        "lb": [21, 28, 42],
        "eff_min": [0.25, 0.35, 0.45],
        "eff_exit": [0.0, 0.10],
        "target_vol": [0.15, 0.20],
    }
    DEFAULTS = {"lb": 28, "eff_min": 0.35, "eff_exit": 0.10,
                "target_vol": 0.15, "timeframe": "1d", "vol_days": 30}

    def _efficiency(self, df: pd.DataFrame) -> np.ndarray:
        """Signed path-efficiency ratio at each bar (0 over warmup)."""
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        lb = int(self.params["lb"]) * bpd
        eff = np.zeros(n)
        if lb + 1 > n:
            return eff
        # net travel over lb bars (simple return)
        net = np.full(n, np.nan)
        net[lb:] = c[lb:] / c[:-lb] - 1.0
        # path length = trailing sum of |log returns| over the lb window
        with np.errstate(divide="ignore", invalid="ignore"):
            abs_lr = np.abs(np.diff(np.log(c), prepend=np.log(c[0])))  # abs_lr[0]=0
        abs_lr[0] = 0.0
        csum = np.cumsum(abs_lr)
        path = np.full(n, np.nan)
        path[lb:] = csum[lb:] - csum[:-lb]
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(np.isfinite(net) & np.isfinite(path) & (path > 0.0),
                           net / path, 0.0)
        return eff

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        eff = self._efficiency(df)
        eff_min = float(self.params["eff_min"])
        eff_exit = float(self.params["eff_exit"])
        entries = eff > eff_min
        exits = eff < eff_exit
        return hold_stance(entries, exits)

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        c = df["close"].to_numpy(dtype=np.float64)
        tf = self.params["timeframe"]
        bpd = bars_per_day(tf)
        rv = realized_vol(c, int(self.params["vol_days"]) * bpd, ppy=periods_per_year(tf))
        tv = float(self.params["target_vol"])
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_frac = np.where(np.isfinite(rv) & (rv > 0.0),
                                np.clip(tv / rv, 0.0, 1.0), 0.0)
        eff = self._efficiency(df)
        conviction = np.clip(np.minimum(1.0, eff / 0.5), 0.0, 1.0)
        # shift1 both: sizing decided at the same close as the stance that fills
        # at the next open (copy the trend.py TSMOM convention exactly).
        return shift1(vol_frac, fill=0.0) * shift1(conviction, fill=0.0)


@register
class VBTrailHybrid(Strategy):
    """Fix for the rejected ``larry_vb``: swing exit instead of same-day exit.

    ENTRY is the canonical Larry Williams open-range breakout, identical to
    ``larry_vb``: on the first 1h close above ``day_open + K * prev_day_range``.
    K is fixed (``k``) or adaptive (``k_mode='noise'``: 30-day mean of the noise
    ratio 1 - |dayO - dayC| / (dayH - dayL) over prior COMPLETED days). Gates:
    optional ``ma_gate``-day SMA trend gate on day_open, and the minimum
    expected-range filter K*prev_range/day_open > 3*``min_range_frac``.

    EXIT is NOT the day boundary. The position is held as a trend-riding swing:
      * engine ATR trailing stop via ``trail_atr_mult`` (ATR on 1h bars,
        ``atr_period`` = 14) — an engine-level exit, not part of the stance; and
      * a stance exit when close < the CURRENT UTC day's open (a failed
        breakout / lost momentum), evaluated per bar.
    Re-entry is allowed once flat again (a fresh in-day breakout).

    The daily-reference machinery is reproduced from ``core.strategies.volbreakout``
    (LarryVB) without modifying that file.
    """

    NAME = "vb_trail_hybrid"
    TIMEFRAMES = ("1h",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "k": [0.4, 0.5, 0.6],
        "k_mode": ["fixed", "noise"],
        "ma_gate": [0, 5],
        "trail_atr_mult": [2.5, 3.0],
    }
    DEFAULTS = {"k": 0.5, "k_mode": "noise", "ma_gate": 5,
                "min_range_frac": 0.005, "noise_days": 30, "timeframe": "1h",
                "trail_atr_mult": 2.5, "atr_period": 14}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params.get("timeframe", "1h") != "1h":
            raise ValueError("VBTrailHybrid works on 1h bars only")

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        idx = df.index

        # --- per-UTC-day aggregates (COPIED from LarryVB; index sorted) -------
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
        # --- end COPIED machinery --------------------------------------------

        # Canonical breakout entry (same as larry_vb).
        breakout = ok_day[codes] & (c > target_d[codes])
        # Swing exit: failed breakout -> close back below the current day's open.
        # d_open[codes][i] is the open of bar i's day (its first bar), known at i.
        failed = c < d_open[codes]
        # Hold across days: enter on first breakout, exit only on `failed`.
        # (The engine ATR trailing stop provides the other exit, out of band.)
        return hold_stance(breakout, failed)
