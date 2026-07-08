"""Intraday mean-reversion strategies (research brief P7).

Only intraday (1h/4h) mean reversion is allowed, hard-gated to non-trending
conditions (ADX < 20) — daily mean reversion was rejected by research.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.indicators import adx, atr, bollinger, realized_vol, rsi, sma
from core.strategies.base import Strategy
from core.strategies.registry import register
from core.strategies.trend import bars_per_day, shift1

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


def _causal_ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Recursive (adjust=False) EMA seeded at the first value.

    Unlike ``core.indicators.ema`` (which discards the first ``span-1`` bars)
    this yields a value from bar 0, so the *daily* EMA200 gate is usable with
    far fewer than 200 daily bars of history. It is fully causal: value[i]
    depends only on arr[<= i], so shifting by one day gives a strict
    previous-completed-day gate with no lookahead.
    """
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()


@register
class GatedRSI2(Strategy):
    """Round-2 #3: regime-gated short-term (RSI2) mean reversion, long-only.

    The researchers' verdict is binding: *ungated* intraday MR loses money;
    the **dual regime gate + dip-in-confirmed-uptrend** is the entire edge, and
    every entry must clear a minimum expected reversion (``min_edge_pct``) so
    the book is not bled dry by round-trip costs.

    All gates are derived from the intraday ``df`` by aggregating to UTC daily
    bars and using the **previous completed day's** values (shift one day, then
    forward-filled onto the intraday index via the per-bar day code). No
    intraday bar ever references its own (still-forming) day's daily aggregate.

    Entry (long, decided at intraday close, executed next open):
      * **Gate A** (chop / near-fair-value): prev-day ``ADX(14) < adx_max``
        OR prev-day close within ``band_atr * daily ATR(14)`` of the prev-day
        daily EMA200.
      * **Gate B** (no volatility expansion): intraday ``shortRV / longRV < 1``
        (5-day- vs 30-day-equivalent bar windows).
      * **Context**: prev-day close > prev-day daily EMA200 (buy dips inside a
        broad uptrend only).
      * **Trigger**: ``RSI(rsi_period) < entry`` on the trading timeframe.
      * **Min edge**: SMA(``sma_exit``) sits at least ``min_edge_pct`` above the
        entry close, i.e. the mean-reversion target alone covers >~1 round trip.

    Exit (decided at close): ``RSI > exit_lvl`` OR ``close > SMA(sma_exit)`` OR
    a ``bars_max`` time stop. Hard downside via engine ``sl_pct``. No ``tp_pct``
    / trailing — the strategy owns its exits.
    """

    NAME = "gated_rsi2"
    TIMEFRAMES = ("1h", "4h")
    # Excluded from the *default* all-strategies sweep (still fully searchable
    # when named explicitly, e.g. round-2 WFA). Rationale: its long-only
    # dip-in-uptrend context needs a *daily* EMA200 plus a genuine up-regime,
    # neither of which the short (~125-day) driftless-random-walk contract
    # fixture provides, so it would trade zero there and only add dead weight to
    # the generic sweep. Dedicated coverage lives in tests/test_strategies.py.
    SEARCHABLE = False
    # band_atr fixed at 1.5 and exit_lvl trimmed to {50,60} to keep the grid at
    # 2*3*2*3*2*2 = 144 combos (<= ~200) while preserving full resolution on the
    # edge-defining params (the ADX gate, the RSI entry threshold, rsi_period).
    PARAM_SPACE = {
        "rsi_period": [2, 3],
        "entry": [3.0, 5.0, 10.0],
        "exit_lvl": [50.0, 60.0],
        "adx_max": [18.0, 20.0, 22.0],
        "bars_max": [12, 24],
        "sl_pct": [0.015, 0.025],
    }
    DEFAULTS = {
        "rsi_period": 2, "entry": 5.0, "exit_lvl": 50.0, "adx_max": 20.0,
        "bars_max": 12, "sl_pct": 0.02,
        "band_atr": 1.5, "ema_period": 200, "adx_period": 14,
        "atr_period": 14, "sma_exit": 5, "rv_short_days": 5, "rv_long_days": 30,
        "min_edge_pct": 0.005, "timeframe": "1h",
    }

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        o = df["open"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        idx = df.index

        # -- per-UTC-day aggregation (index sorted -> codes monotonic 0..nd-1) --
        codes = pd.factorize(idx.floor("D"))[0]
        starts = np.flatnonzero(np.diff(codes, prepend=-1))
        ends = np.concatenate((starts[1:], [n])) - 1
        d_high = np.maximum.reduceat(h, starts)
        d_low = np.minimum.reduceat(l, starts)
        d_close = c[ends]

        # -- daily indicators, then shift to the PREVIOUS COMPLETED day --------
        adx_d, _, _ = adx(d_high, d_low, d_close, int(self.params["adx_period"]))
        atr_d = atr(d_high, d_low, d_close, int(self.params["atr_period"]))
        ema_d = _causal_ema(d_close, int(self.params["ema_period"]))

        prev_adx = shift1(adx_d)[codes]
        prev_atr = shift1(atr_d)[codes]
        prev_ema = shift1(ema_d)[codes]
        prev_close = shift1(d_close)[codes]

        band = float(self.params["band_atr"])
        # Gate A: chop (low ADX) OR price near daily EMA200 (mean-reverting zone).
        # NaN warmup values compare False, so no position is taken until the
        # daily indicators are warm -- correct, not a bug.
        with np.errstate(invalid="ignore"):
            gate_a = (prev_adx < float(self.params["adx_max"])) | (
                np.abs(prev_close - prev_ema) < band * prev_atr)
            context = prev_close > prev_ema  # broad uptrend

        # Gate B: intraday realized-vol non-expansion (annualization cancels in
        # the ratio, so ppy is irrelevant). Causal trailing windows.
        tf = str(self.params.get("timeframe", "1h"))
        bpd = bars_per_day(tf)
        short_p = max(2, int(self.params["rv_short_days"]) * bpd)
        long_p = max(short_p + 1, int(self.params["rv_long_days"]) * bpd)
        rv_short = realized_vol(c, short_p)
        rv_long = realized_vol(c, long_p)
        with np.errstate(divide="ignore", invalid="ignore"):
            gate_b = (rv_short / rv_long) < 1.0

        # Trigger + min-edge (reversion target must clear the cost floor).
        r = rsi(c, int(self.params["rsi_period"]))
        sma_e = sma(c, int(self.params["sma_exit"]))
        with np.errstate(invalid="ignore"):
            trigger = r < float(self.params["entry"])
            min_edge = (sma_e - c) >= float(self.params["min_edge_pct"]) * c

        entry_sig = (np.nan_to_num(gate_a) & np.nan_to_num(context)
                     & np.nan_to_num(gate_b) & np.nan_to_num(trigger)
                     & np.nan_to_num(min_edge))

        exit_lvl = float(self.params["exit_lvl"])
        bars_max = int(self.params["bars_max"])
        stance = np.zeros(n)
        state = 0
        entry_bar = -1
        for i in range(n):
            if state == 1:
                rsi_exit = not np.isnan(r[i]) and r[i] > exit_lvl
                sma_exit = not np.isnan(sma_e[i]) and c[i] > sma_e[i]
                if rsi_exit or sma_exit or (i - entry_bar) >= bars_max:
                    state = 0
            if state == 0 and entry_sig[i]:
                state = 1
                entry_bar = i
            stance[i] = state
        return stance
