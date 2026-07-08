"""Round-2 candidate #5: RV term-structure regime primitives (core.regime).

Locks in the two properties that matter for a signal used to gate live gross:
no lookahead (prefix stability under truncation) and previous-completed-day
causality, plus the estimator's basic correctness and whipsaw damping.
"""
from __future__ import annotations

import numpy as np

from core.regime import NORMAL, RANGING, RISK_OFF, parkinson_vol, rv_ratio_state


def test_parkinson_warmup_and_positivity(random_walk_df):
    h = random_walk_df["high"].to_numpy()
    l = random_walk_df["low"].to_numpy()
    pv = parkinson_vol(h, l, 10)
    assert np.isnan(pv[:9]).all()          # period-1 warmup is NaN
    assert np.all(pv[10:] >= 0.0)          # volatility is non-negative
    assert np.isfinite(pv[10:]).all()


def test_parkinson_scales_with_range(make_df):
    # a wide-range frame must report higher Parkinson vol than a tight one
    n = 60
    tight = make_df([100.0] * n, [100.5] * n, [99.5] * n, [100.0] * n)
    wide = make_df([100.0] * n, [110.0] * n, [90.0] * n, [100.0] * n)
    pt = parkinson_vol(tight["high"].to_numpy(), tight["low"].to_numpy(), 20)
    pw = parkinson_vol(wide["high"].to_numpy(), wide["low"].to_numpy(), 20)
    assert pw[-1] > pt[-1] * 5


def test_rv_state_values_and_no_lookahead(random_walk_df):
    st = rv_ratio_state(random_walk_df, short_d=10, long_d=30,
                        risk_off=1.2, ranging=0.8, ema_smooth=3, hysteresis=0.05)
    assert st.dtype == np.int8
    assert set(np.unique(st)).issubset({NORMAL, RISK_OFF, RANGING})
    # truncating the frame must not change any earlier state (causality)
    k = 700
    st_pre = rv_ratio_state(random_walk_df.iloc[:k], short_d=10, long_d=30,
                            risk_off=1.2, ranging=0.8, ema_smooth=3, hysteresis=0.05)
    assert np.array_equal(st[:k], st_pre)


def test_rv_state_hysteresis_reduces_flips(random_walk_df):
    common = dict(short_d=10, long_d=30, risk_off=1.2, ranging=0.8, ema_smooth=3)
    st_no = rv_ratio_state(random_walk_df, hysteresis=0.0, **common)
    st_hy = rv_ratio_state(random_walk_df, hysteresis=0.15, **common)
    flips_no = int(np.sum(st_no[1:] != st_no[:-1]))
    flips_hy = int(np.sum(st_hy[1:] != st_hy[:-1]))
    assert flips_hy <= flips_no      # hysteresis never increases flag churn


def test_rv_state_long_must_exceed_short(random_walk_df):
    import pytest
    with pytest.raises(ValueError):
        rv_ratio_state(random_walk_df, short_d=30, long_d=30)
