"""Round-4 group D contract tests: efficiency_momentum, vb_trail_hybrid.

Generic zoo contracts (domain/length, warmup-zero, determinism, no-lookahead,
param-sensitivity) are already swept by tests/test_strategies.py because both
classes are SEARCHABLE and registered before that module's snapshot. These
dedicated tests pin the *design-specific* invariants that the generic sweep
cannot see: the efficiency ratio's causal construction and hysteresis, and the
VB-hybrid's swing exit (close-below-day-open) vs LarryVB's forced day-end exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.strategies.base import hold_stance
from core.strategies.round4_d import EfficiencyMomentum, VBTrailHybrid
from core.strategies.volbreakout import LarryVB
from core.strategies.registry import get_strategy

from conftest import ohlcv_from_close


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def trend_daily() -> pd.DataFrame:
    """~600 daily bars: a clean uptrend leg, a chop leg, then a clean downtrend
    leg, so both the efficiency entry and its hysteresis exit are exercised."""
    g = np.random.default_rng(3)
    n = 600
    drift = np.concatenate([
        np.full(200, 0.006),    # clean up
        g.normal(0.0, 0.001, 200),  # chop
        np.full(200, -0.006),   # clean down
    ])
    # vol chosen so realized_vol > target_vol (vol-targeting stays unsaturated)
    close = 100.0 * np.exp(np.cumsum(drift + g.normal(0.0, 0.011, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="1D", tz="UTC")
    df = ohlcv_from_close(close, g)
    df.index = idx
    return df


@pytest.fixture(scope="module")
def hourly_year() -> pd.DataFrame:
    """~500 UTC days of 1h bars with mild upward drift (arg to the daily gate)."""
    g = np.random.default_rng(5)
    n = 500 * 24
    close = 100.0 * np.exp(np.cumsum(g.normal(0.00006, 0.006, n)))
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    df = ohlcv_from_close(close, g)
    df.index = idx
    return df


# --------------------------------------------------- efficiency_momentum

def test_effmom_registered():
    assert get_strategy("efficiency_momentum") is EfficiencyMomentum
    assert EfficiencyMomentum.TIMEFRAMES == ("1d",)
    assert EfficiencyMomentum.param_grid_size() <= 150


def test_effmom_efficiency_bounds_and_sign(trend_daily):
    """Efficiency is bounded near [-1, 1] and its sign tracks net travel."""
    s = EfficiencyMomentum()
    eff = s._efficiency(trend_daily)
    lb = int(s.params["lb"])
    assert np.all(eff[:lb] == 0.0)               # warmup is exactly 0
    assert np.isfinite(eff).all()                # no NaN/inf leaks
    # signed ER can marginally exceed 1 (simple net vs log path) but stays sane
    assert eff.max() <= 1.5 and eff.min() >= -1.5
    c = trend_daily["close"].to_numpy()
    net = c[lb:] / c[:-lb] - 1.0
    # where net>0 efficiency>=0 and vice versa (path is always >=0)
    e = eff[lb:]
    assert np.all(e[net > 0] >= 0.0)
    assert np.all(e[net < 0] <= 0.0)


def test_effmom_hysteresis(trend_daily):
    """Entry uses eff_min, exit uses the lower eff_exit: a stance held between
    the two bands proves the hysteresis (not a bare threshold)."""
    s = EfficiencyMomentum(eff_min=0.45, eff_exit=0.0)
    eff = s._efficiency(trend_daily)
    st = s.generate_signals(trend_daily)
    # reconstruct the reference state machine
    ref = hold_stance(eff > 0.45, eff < 0.0)
    assert np.array_equal(st, ref)
    # there exists a bar held long while efficiency sits in the (0, 0.45] band
    held_band = (st == 1.0) & (eff <= 0.45) & (eff > 0.0)
    assert held_band.any(), "hysteresis band never exercised by fixture"


def test_effmom_conviction_scales_size(trend_daily):
    """Size = vol_frac * min(1, eff/0.5) clipped: raising target_vol raises the
    vol component, and the conviction factor zeroes size where eff<=0."""
    s = EfficiencyMomentum()
    sf = np.asarray(s.generate_size_frac(trend_daily))
    assert np.isfinite(sf).all()
    assert (sf >= 0.0).all() and (sf <= 1.0).all()
    # higher target_vol -> element-wise >= size (same conviction, larger vol_frac)
    lo = np.asarray(EfficiencyMomentum(target_vol=0.15).generate_size_frac(trend_daily))
    hi = np.asarray(EfficiencyMomentum(target_vol=0.20).generate_size_frac(trend_daily))
    assert np.all(hi + 1e-12 >= lo)
    assert np.nansum(hi - lo) > 0


def test_effmom_no_lookahead_multi_offsets(trend_daily):
    full_s = np.asarray(EfficiencyMomentum().generate_signals(trend_daily))
    full_f = np.asarray(EfficiencyMomentum().generate_size_frac(trend_daily))
    n = len(trend_daily)
    for k in (n - 1, n - 50, n - 300, n // 2):
        if k < 100:
            continue
        ts = np.asarray(EfficiencyMomentum().generate_signals(trend_daily.iloc[:k]))
        tf = np.asarray(EfficiencyMomentum().generate_size_frac(trend_daily.iloc[:k]))
        assert np.array_equal(full_s[:k], ts), f"stance lookahead at k={k}"
        assert np.allclose(full_f[:k], tf, atol=1e-12), f"size lookahead at k={k}"


# --------------------------------------------------------- vb_trail_hybrid

def test_vbhybrid_registered_and_1h_only():
    assert get_strategy("vb_trail_hybrid") is VBTrailHybrid
    assert VBTrailHybrid.TIMEFRAMES == ("1h",)
    assert VBTrailHybrid.param_grid_size() <= 150
    with pytest.raises(ValueError):
        VBTrailHybrid(timeframe="4h")


def test_vbhybrid_routes_engine_trail():
    """trail_atr_mult + atr_period are engine-level exits, not stance params."""
    ep = VBTrailHybrid(trail_atr_mult=3.0).engine_params()
    assert ep["trail_atr_mult"] == 3.0 and ep["atr_period"] == 14


def test_vbhybrid_holds_across_days(hourly_year):
    """The defining fix vs larry_vb: the swing is NOT force-flat at 23:00.
    Some long stance must survive a UTC day boundary."""
    st = VBTrailHybrid().generate_signals(hourly_year)
    assert set(np.unique(st)).issubset({0.0, 1.0})
    assert st[0] == 0.0 and st.sum() > 0
    hour = hourly_year.index.hour.to_numpy()
    # a bar at 23:00 held long AND the following 00:00 bar still long => crossed
    is_23 = hour == 23
    held_23 = (st == 1.0) & is_23
    crossed = held_23[:-1] & (st[1:] == 1.0)
    assert crossed.any(), "hybrid never carried a position across a day boundary"
    # larry_vb would be forced flat at every 23:00 bar
    lvb = LarryVB().generate_signals(hourly_year)
    assert np.all(lvb[is_23] == 0.0)


def test_vbhybrid_exit_on_close_below_dayopen(hourly_year):
    """Every long->flat transition that is NOT the very first bar must be a bar
    whose close is below its own UTC day's open (the failed-breakout signal),
    since the stance itself has no other exit (ATR trail is engine-side)."""
    df = hourly_year
    st = VBTrailHybrid().generate_signals(df)
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    codes = pd.factorize(df.index.floor("D"))[0]
    starts = np.flatnonzero(np.diff(codes, prepend=-1))
    d_open = o[starts][codes]
    # bars where we WERE long and are now flat
    exits = np.where((st[:-1] == 1.0) & (st[1:] == 0.0))[0] + 1
    assert len(exits) > 0
    # the stance exit fires exactly when close < current day's open
    assert np.all(c[exits] < d_open[exits])


def test_vbhybrid_entry_matches_larryvb_first_breakout(hourly_year):
    """Entries occur only on bars that are genuine breakouts above the same
    target LarryVB uses. Verify the first long bar of each isolated long run is
    a breakout bar (close > day_open + K*prev_range with gates)."""
    df = hourly_year
    s = VBTrailHybrid(k_mode="fixed", k=0.5, ma_gate=0)
    st = s.generate_signals(df)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    n = len(c)
    codes = pd.factorize(df.index.floor("D"))[0]
    starts = np.flatnonzero(np.diff(codes, prepend=-1))
    d_open = o[starts]
    d_high = np.maximum.reduceat(h, starts)
    d_low = np.minimum.reduceat(l, starts)
    from core.strategies.trend import shift1
    prev_range = shift1(d_high) - shift1(d_low)
    target = (d_open + 0.5 * prev_range)[codes]
    entries = np.where((st[:-1] == 0.0) & (st[1:] == 1.0))[0] + 1
    assert len(entries) > 0
    assert np.all(c[entries] > target[entries])


def test_vbhybrid_no_lookahead_straddling_day_boundaries(hourly_year):
    df = hourly_year
    full = np.asarray(VBTrailHybrid().generate_signals(df))
    n = len(df)
    for k in (n - 1, n - 24, n - 25, n - 23, n - 300, n // 2 + 7):
        if k < 300:
            continue
        trunc = np.asarray(VBTrailHybrid().generate_signals(df.iloc[:k]))
        assert len(trunc) == k
        assert np.array_equal(full[:k], trunc), f"lookahead at k={k}"
