"""Strategy-zoo contract tests (core/strategies/*).

Contract under test (strategies/base.py):
- generate_signals returns a float array in {0, 1} (spot long/flat), same
  length as the input frame, with a leading zero warmup;
- deterministic: same input -> identical output;
- CRITICAL no-lookahead: stance computed on a truncated prefix df[:k] must
  equal the full-history stance sliced to [:k] (a strategy may never use a
  future bar to decide stance[i]);
- parameters actually matter: some param change alters the signals.

SEARCHABLE=False ensembles (voting / regime-switch) need explicit members and
are exercised in dedicated tests below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.strategies.base import ENGINE_PARAM_KEYS
from core.strategies.registry import all_strategies, get_strategy

from conftest import ohlcv_from_close

# ---------------------------------------------------------------- fixtures

N_BARS = 3000


@pytest.fixture(scope="module")
def walk_df() -> pd.DataFrame:
    """3000-bar 1h geometric random walk (drift ~0): enough history for the
    longest lookbacks and varied enough that every strategy is exercised."""
    g = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(g.normal(0.0, 0.01, N_BARS)))
    return ohlcv_from_close(close, g)


SEARCHABLE = {n: c for n, c in all_strategies().items()
              if getattr(c, "SEARCHABLE", True)}


def _instantiate(cls, **extra):
    """Build a strategy with sensible defaults; inject timeframe='1h' when the
    strategy is timeframe-aware (larry_vb *requires* 1h)."""
    params = dict(extra)
    if "timeframe" in cls.DEFAULTS:
        params.setdefault("timeframe", "1h")
    return cls(**params)


# --------------------------------------------------- generic per-strategy tests

@pytest.mark.parametrize("name", sorted(SEARCHABLE))
def test_stance_domain_and_length(name, walk_df):
    sig = _instantiate(SEARCHABLE[name]).generate_signals(walk_df)
    assert len(sig) == len(walk_df)
    assert set(np.unique(sig)).issubset({0.0, 1.0})


@pytest.mark.parametrize("name", sorted(SEARCHABLE))
def test_warmup_is_zero(name, walk_df):
    sig = _instantiate(SEARCHABLE[name]).generate_signals(walk_df)
    # indicators are NaN over their warmup -> no position can be taken on bar 0
    assert sig[0] == 0.0


@pytest.mark.parametrize("name", sorted(SEARCHABLE))
def test_deterministic(name, walk_df):
    strat = _instantiate(SEARCHABLE[name])
    a = strat.generate_signals(walk_df)
    b = strat.generate_signals(walk_df)
    assert np.array_equal(a, b)
    # a freshly constructed instance with the same params must also agree
    c = _instantiate(SEARCHABLE[name]).generate_signals(walk_df)
    assert np.array_equal(a, c)


@pytest.mark.parametrize("name", sorted(SEARCHABLE))
def test_no_lookahead_truncation(name, walk_df):
    """stance(df[:k]) == stance(df)[:k] — the defining causality property."""
    strat = _instantiate(SEARCHABLE[name])
    full = np.asarray(strat.generate_signals(walk_df))
    k = len(walk_df) - 200
    trunc = np.asarray(_instantiate(SEARCHABLE[name]).generate_signals(walk_df.iloc[:k]))
    assert len(trunc) == k
    assert np.array_equal(full[:k], trunc)


@pytest.mark.parametrize("name", sorted(SEARCHABLE))
def test_param_change_alters_signals(name, walk_df):
    cls = SEARCHABLE[name]
    base = _instantiate(cls).generate_signals(walk_df)
    changed = False
    for param, values in cls.PARAM_SPACE.items():
        if param in ENGINE_PARAM_KEYS:
            continue  # engine-exit params never touch generate_signals
        for v in values:
            if v == cls.DEFAULTS.get(param):
                continue
            sig = _instantiate(cls, **{param: v}).generate_signals(walk_df)
            if not np.array_equal(sig, base):
                changed = True
                break
        if changed:
            break
    assert changed, f"no signal-affecting param found for {name}"


# ------------------------------------------------------ voting ensemble math

def test_voting_ensemble_threshold_math(walk_df):
    VotingEnsemble = get_strategy("voting_ensemble")
    m1 = {"name": "ema_cross", "params": {"fast": 9, "slow": 21, "regime_filter": 0}}
    m2 = {"name": "ema_cross", "params": {"fast": 21, "slow": 55, "regime_filter": 0}}

    # independent member stances (long-only clipped) — the reference vote
    s1 = np.clip(get_strategy("ema_cross")(**m1["params"]).generate_signals(walk_df), 0.0, 1.0)
    s2 = np.clip(get_strategy("ema_cross")(**m2["params"]).generate_signals(walk_df), 0.0, 1.0)
    assert not np.array_equal(s1, s2)  # members genuinely disagree somewhere

    # equal weights: combined in {0, 0.5, 1}. threshold 0.5 = OR, 1.0 = AND.
    for thr in (0.5, 1.0):
        ens = VotingEnsemble(
            members=(dict(m1, weight=1.0), dict(m2, weight=1.0)), threshold=thr)
        combined = (s1 + s2) / 2.0
        expected = (combined >= thr).astype(np.float64)
        assert np.array_equal(ens.generate_signals(walk_df), expected)
    # a 0.5 threshold must be strictly looser than 1.0 on these members
    or_sig = VotingEnsemble(members=(dict(m1, weight=1.0), dict(m2, weight=1.0)),
                            threshold=0.5).generate_signals(walk_df)
    and_sig = VotingEnsemble(members=(dict(m1, weight=1.0), dict(m2, weight=1.0)),
                             threshold=1.0).generate_signals(walk_df)
    assert or_sig.sum() > and_sig.sum()

    # weighted vote (3:1): combined = (3*s1 + 1*s2) / 4
    ens_w = VotingEnsemble(
        members=(dict(m1, weight=3.0), dict(m2, weight=1.0)), threshold=0.5)
    combined_w = (3.0 * s1 + 1.0 * s2) / 4.0
    assert np.array_equal(ens_w.generate_signals(walk_df),
                          (combined_w >= 0.5).astype(np.float64))

    # zero-weight member is dropped (floor 0, never negative-weighted)
    ens_drop = VotingEnsemble(
        members=(dict(m1, weight=1.0), dict(m2, weight=0.0)), threshold=0.5)
    assert np.array_equal(ens_drop.generate_signals(walk_df),
                          (s1 >= 0.5).astype(np.float64))


# --------------------------------------------------- regime-switch ensemble

def test_regime_switch_ensemble_runs_and_valid(walk_df):
    RegimeSwitch = get_strategy("regime_switch_ensemble")
    ens = RegimeSwitch(
        trend_member={"name": "donchian_single", "params": {"timeframe": "1h"}},
        range_member={"name": "bb_rsi_meanrev", "params": {}},
        timeframe="1h",
    )
    sig = ens.generate_signals(walk_df)
    assert len(sig) == len(walk_df)
    assert set(np.unique(sig)).issubset({0.0, 1.0})
    # size-frac path must also be well-formed (no NaN, bounded to [0, 1])
    frac = ens.generate_size_frac(walk_df)
    if frac is not None:
        frac = np.asarray(frac)
        assert np.isfinite(frac).all()
        assert (frac >= 0.0).all()
