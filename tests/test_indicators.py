"""Indicator correctness vs independent plain-pandas / loop references.

References are computed here from first principles (pandas rolling, explicit
Wilder loops) and never import core implementation details.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import indicators as ind


# ---------------------------------------------------------------- references

def ref_rsi_wilder(close: np.ndarray, period: int) -> np.ndarray:
    """Straightforward textbook Wilder RSI loop: SMA seed then RMA recursion."""
    n = len(close)
    out = np.full(n, np.nan)
    delta = np.diff(close)
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    if n < period + 1:
        return out
    ag = gains[:period].mean()
    al = losses[:period].mean()

    def rsi_val(ag_, al_):
        return 100.0 if al_ == 0 else 100.0 - 100.0 / (1.0 + ag_ / al_)

    out[period] = rsi_val(ag, al)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gains[i - 1]) / period
        al = (al * (period - 1) + losses[i - 1]) / period
        out[i] = rsi_val(ag, al)
    return out


def ref_atr_wilder(high, low, close, period: int) -> np.ndarray:
    """Textbook Wilder ATR loop: TR then SMA-seeded RMA."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    out = np.full(n, np.nan)
    if n < period:
        return out
    a = tr[:period].mean()
    out[period - 1] = a
    for i in range(period, n):
        a = (a * (period - 1) + tr[i]) / period
        out[i] = a
    return out


def assert_warmup_then_finite(arr: np.ndarray, n: int) -> None:
    """len == input, leading NaN warmup, then finite with no NaN afterwards."""
    assert len(arr) == n
    finite = np.isfinite(arr)
    assert finite.any(), "indicator produced no finite values"
    first = int(np.argmax(finite))
    assert first > 0, "expected a NaN warmup region"
    assert not finite[:first].any()
    assert finite[first:].all()


# --------------------------------------------------------------------- tests

class TestSMA:
    def test_matches_rolling_mean(self, random_walk_df):
        c = random_walk_df["close"]
        for p in (5, 20, 50):
            ref = c.rolling(p).mean().to_numpy()
            out = ind.sma(c.to_numpy(), p)
            assert np.allclose(out, ref, equal_nan=True)

    def test_warmup(self, random_walk_df):
        out = ind.sma(random_walk_df["close"].to_numpy(), 20)
        assert_warmup_then_finite(out, len(random_walk_df))
        assert np.isnan(out[:19]).all() and np.isfinite(out[19])


class TestEMA:
    def test_converges_to_pandas_ewm(self, random_walk_df):
        # core EMA seeds with SMA, pandas ewm(adjust=False) seeds with x0;
        # the seed difference decays as (1-alpha)^k -> negligible after 5x period.
        c = random_walk_df["close"]
        for p in (10, 20):
            ref = c.ewm(span=p, adjust=False).mean().to_numpy()
            out = ind.ema(c.to_numpy(), p)
            tail = slice(5 * p, None)
            rel = np.abs(out[tail] - ref[tail]) / np.abs(ref[tail])
            assert np.nanmax(rel) < 1e-3

    def test_warmup(self, random_walk_df):
        out = ind.ema(random_walk_df["close"].to_numpy(), 20)
        assert_warmup_then_finite(out, len(random_walk_df))


class TestRSI:
    def test_bounds(self, random_walk_df):
        out = ind.rsi(random_walk_df["close"].to_numpy(), 14)
        finite = out[np.isfinite(out)]
        assert ((finite >= 0.0) & (finite <= 100.0)).all()

    def test_matches_wilder_loop(self, random_walk_df):
        c = random_walk_df["close"].to_numpy()
        for p in (5, 14):
            ref = ref_rsi_wilder(c, p)
            out = ind.rsi(c, p)
            np.testing.assert_allclose(out[p:], ref[p:], atol=1e-6)
            assert np.isnan(out[:p]).all()

    def test_warmup(self, random_walk_df):
        out = ind.rsi(random_walk_df["close"].to_numpy(), 14)
        assert_warmup_then_finite(out, len(random_walk_df))


class TestATR:
    def test_matches_wilder_loop(self, random_walk_df):
        h = random_walk_df["high"].to_numpy()
        l = random_walk_df["low"].to_numpy()
        c = random_walk_df["close"].to_numpy()
        for p in (7, 14):
            ref = ref_atr_wilder(h, l, c, p)
            out = ind.atr(h, l, c, p)
            np.testing.assert_allclose(out[p - 1:], ref[p - 1:], rtol=1e-9)
            assert np.isnan(out[:p - 1]).all()

    def test_positive_and_warmup(self, random_walk_df):
        out = ind.atr(random_walk_df["high"].to_numpy(),
                      random_walk_df["low"].to_numpy(),
                      random_walk_df["close"].to_numpy(), 14)
        assert_warmup_then_finite(out, len(random_walk_df))
        assert (out[np.isfinite(out)] > 0).all()


class TestBollinger:
    def test_mid_is_sma_and_width_is_4_population_std(self, random_walk_df):
        c = random_walk_df["close"]
        p, k = 20, 2.0
        upper, mid, lower = ind.bollinger(c.to_numpy(), p, k)
        ref_mid = c.rolling(p).mean().to_numpy()
        ref_sd = c.rolling(p).std(ddof=0).to_numpy()  # population std
        assert np.allclose(mid, ref_mid, equal_nan=True)
        # upper - lower = 2 * k * sd = 4 * sd for k=2
        assert np.allclose(upper - lower, 2.0 * k * ref_sd, equal_nan=True, atol=1e-9)
        assert np.allclose(upper, ref_mid + k * ref_sd, equal_nan=True, atol=1e-9)

    def test_warmup(self, random_walk_df):
        upper, mid, lower = ind.bollinger(random_walk_df["close"].to_numpy(), 20, 2.0)
        for arr in (upper, mid, lower):
            assert_warmup_then_finite(arr, len(random_walk_df))


class TestDonchian:
    def test_matches_rolling_max_min(self, random_walk_df):
        h = random_walk_df["high"]
        l = random_walk_df["low"]
        for p in (10, 55):
            upper, lower = ind.donchian(h.to_numpy(), l.to_numpy(), p)
            assert np.allclose(upper, h.rolling(p).max().to_numpy(), equal_nan=True)
            assert np.allclose(lower, l.rolling(p).min().to_numpy(), equal_nan=True)

    def test_warmup(self, random_walk_df):
        upper, lower = ind.donchian(random_walk_df["high"].to_numpy(),
                                    random_walk_df["low"].to_numpy(), 20)
        assert_warmup_then_finite(upper, len(random_walk_df))
        assert_warmup_then_finite(lower, len(random_walk_df))


class TestSupertrend:
    def test_flips_exactly_on_crafted_crosses(self, make_df):
        # Crafted sequence with constant TR = 2 -> ATR(3) = 2 exactly.
        # close rises +1/bar 100..110 (i=0..10), then crashes to 101 at i=11,
        # then drifts down. low = close-1, high = close+1, open = prev close
        # (crash bar keeps high = prev close so it's a valid red candle).
        closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 101, 100, 99]
        opens = [100] + closes[:-1]
        highs = [c + 1 for c in closes]
        highs[11] = 110.0  # crash bar opens at 110, so high must reach it
        lows = [c - 1 for c in closes]
        df = make_df(opens, highs, lows, closes)

        # period=3, mult=1 -> bands at hl2 +/- 2 = close +/- 2 (hl2 == close
        # pre-crash). start = 3 (internal), first visible value at i=4.
        # i=3: c=103 <= upper_basic=105 -> internal trend -1, upper band 105.
        # i=4..5: upper ratchets down/min -> stays 105; c=104,105 <= 105 -> -1.
        # i=6: c=106 > upper 105 -> FLIP UP (close crossed the line at 105).
        # i=7..10: uptrend, lower band ratchets up to close-2 (108 at i=10).
        # i=11: c=101 < ratcheted lower band 108 -> FLIP DOWN.
        trend, line = ind.supertrend(df["high"].to_numpy(), df["low"].to_numpy(),
                                     df["close"].to_numpy(), period=3, multiplier=1.0)
        assert np.isnan(trend[:4]).all()
        assert (trend[4:6] == -1.0).all()
        assert (trend[6:11] == 1.0).all()
        assert (trend[11:] == -1.0).all()
        # the flip bars crossed the *previous* bar's line
        assert df["close"].iloc[6] > line[5] and df["close"].iloc[5] <= line[5]
        assert df["close"].iloc[11] < line[10] and df["close"].iloc[10] >= line[10]

    def test_trend_line_side_invariant(self, random_walk_df):
        # in uptrend the line is the lower band below price; in downtrend the
        # upper band above price: sign(close - line) must agree with trend.
        h = random_walk_df["high"].to_numpy()
        l = random_walk_df["low"].to_numpy()
        c = random_walk_df["close"].to_numpy()
        trend, line = ind.supertrend(h, l, c, period=10, multiplier=3.0)
        valid = np.isfinite(trend) & np.isfinite(line)
        up = valid & (trend == 1.0)
        dn = valid & (trend == -1.0)
        assert (c[up] >= line[up] - 1e-9).all()
        assert (c[dn] <= line[dn] + 1e-9).all()

    def test_length_and_warmup(self, random_walk_df):
        trend, line = ind.supertrend(random_walk_df["high"].to_numpy(),
                                     random_walk_df["low"].to_numpy(),
                                     random_walk_df["close"].to_numpy(), 10, 3.0)
        assert len(trend) == len(line) == len(random_walk_df)
        assert_warmup_then_finite(line, len(random_walk_df))
        finite_trend = trend[np.isfinite(trend)]
        assert set(np.unique(finite_trend)) <= {-1.0, 1.0}
