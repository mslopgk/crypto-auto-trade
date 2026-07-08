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


# ============================================================================
# Round-2 funding-regime sleeve (candidate #1). Appended, self-contained, and
# guarded: skipped entirely when the funding parquet (and, for backtest checks,
# the BTC/USDT 1d OHLCV cache) is absent. Importing the module registers both
# strategies AFTER the module-level SEARCHABLE snapshot above, so the generic
# parametrized contract tests deliberately do NOT sweep them (they carry their
# own dedicated checks here and depend on the funding data being present).
# ============================================================================

from core.backtest.runner import run_strategy_backtest  # noqa: E402
from core.data.fetcher import cache_path, load_ohlcv  # noqa: E402
from core.data.funding import (daily_funding_features, funding_path,  # noqa: E402
                               load_funding)
import core.strategies.funding as _funding_mod  # noqa: E402,F401  (registers)
from core.strategies.funding import (FundingCapitulation,  # noqa: E402
                                      TsmomFundingGated, funding_z_for_df)
from core.strategies.trend import TSMOM  # noqa: E402

_FUNDING_OK = funding_path("BTC").exists()
_BTC_1D_OK = cache_path("binance", "BTC/USDT", "1d").exists()
requires_funding = pytest.mark.skipif(not _FUNDING_OK, reason="no BTC funding parquet")
requires_btc = pytest.mark.skipif(
    not (_FUNDING_OK and _BTC_1D_OK),
    reason="need BTC funding parquet + BTC/USDT 1d OHLCV cache")


@pytest.fixture(scope="module")
def btc_daily() -> pd.DataFrame:
    return load_ohlcv("binance", "BTC/USDT", "1d", since="2020-01-01", refresh=False)


@requires_funding
def test_daily_funding_features_shape_and_causality():
    r = load_funding("BTC")
    feats = daily_funding_features(r, [1, 3, 9], [14, 30])
    for K in (1, 3, 9):
        assert f"annfund_K{K}" in feats.columns
        for L in (14, 30):
            assert f"z_K{K}_L{L}" in feats.columns
    # daily grid, monotonic, unique
    assert feats.index.is_monotonic_increasing
    assert not feats.index.has_duplicates
    # causality: features on a truncated prefix of the raw prints match the
    # full-history features on the shared days (drop the last, possibly-partial
    # day of the prefix). No future print may change a settled day's value.
    cut = int(len(r) * 0.7)
    part = daily_funding_features(r.iloc[:cut], [3], [30])
    common = part.index[:-1]
    a = feats.loc[common, "z_K3_L30"].to_numpy()
    b = part.loc[common, "z_K3_L30"].to_numpy()
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-9)


@requires_btc
def test_funding_capitulation_backtest_runs(btc_daily):
    strat = FundingCapitulation(symbol="BTC/USDT")
    stance = strat.generate_signals(btc_daily)
    assert set(np.unique(stance)).issubset({0.0, 1.0})
    assert stance[0] == 0.0
    assert stance.sum() > 0  # fires on real BTC funding history
    res = run_strategy_backtest(btc_daily, strat, "1d", symbol="BTC/USDT")
    assert res.n_trades > 0
    assert np.isfinite(res.metrics["sharpe"])
    assert np.isfinite(res.equity.to_numpy()).all()


@requires_btc
def test_tsmom_gated_stance_matches_plain_tsmom(btc_daily):
    gated = TsmomFundingGated(symbol="BTC/USDT")
    plain = TSMOM()
    # de-gross is a SIZE overlay only: the trend stance is untouched
    assert np.array_equal(gated.generate_signals(btc_daily),
                          plain.generate_signals(btc_daily))


@requires_btc
def test_tsmom_gated_degross_only_cuts_size(btc_daily):
    gated = TsmomFundingGated(symbol="BTC/USDT", size_mult=0.0)
    plain = TSMOM()
    gs = np.asarray(gated.generate_size_frac(btc_daily))
    ps = np.asarray(plain.generate_size_frac(btc_daily))
    # multiplier is in {size_mult, 1.0} with size_mult<=1 -> gated <= plain
    assert np.all(np.nan_to_num(gs) <= np.nan_to_num(ps) + 1e-12)
    assert np.nansum(np.abs(gs - ps)) > 0  # de-gross is actually engaged
    res = run_strategy_backtest(btc_daily, gated, "1d", symbol="BTC/USDT")
    assert np.isfinite(res.metrics["sharpe"])


@requires_btc
@pytest.mark.parametrize("cls", [FundingCapitulation, TsmomFundingGated])
def test_funding_no_lookahead_truncation(cls, btc_daily):
    """stance(df[:k]) == stance(df)[:k] AND size_frac(df[:k]) == size_frac(df)[:k]."""
    k = len(btc_daily) - 200
    full = cls(symbol="BTC/USDT")
    trunc = cls(symbol="BTC/USDT")
    f_st = np.asarray(full.generate_signals(btc_daily))
    t_st = np.asarray(trunc.generate_signals(btc_daily.iloc[:k]))
    assert len(t_st) == k
    assert np.array_equal(np.nan_to_num(f_st[:k]), np.nan_to_num(t_st))

    f_sf = full.generate_size_frac(btc_daily)
    t_sf = trunc.generate_size_frac(btc_daily.iloc[:k])
    if f_sf is None:
        assert t_sf is None
    else:
        f_sf = np.asarray(f_sf)
        t_sf = np.asarray(t_sf)
        assert np.allclose(np.nan_to_num(f_sf[:k]), np.nan_to_num(t_sf), atol=1e-12)


@requires_btc
def test_funding_missing_symbol_falls_back(btc_daily):
    """A symbol with no funding parquet: Capitulation flat, Gated == plain TSMOM."""
    assert not funding_path("NOFUND").exists()
    assert funding_z_for_df(btc_daily, "NOFUND/USDT", 3, 30) is None
    cap = FundingCapitulation(symbol="NOFUND/USDT")
    assert (cap.generate_signals(btc_daily) == 0.0).all()
    gated = TsmomFundingGated(symbol="NOFUND/USDT")
    plain = TSMOM()
    gs = np.asarray(gated.generate_size_frac(btc_daily))
    ps = np.asarray(plain.generate_size_frac(btc_daily))
    assert np.allclose(np.nan_to_num(gs), np.nan_to_num(ps))


@requires_btc
def test_micro_run_search_symbol_injection():
    """A tiny serial search resolves the funding strategy by registry NAME and
    injects the traded symbol into params (search.py minimal edit)."""
    from core.optimize.search import SearchSpec, run_search
    spec = SearchSpec(symbols=["BTC/USDT"], timeframes=["1d"],
                      strategies=["funding_capitulation"],
                      max_combos_per_strategy=12, since="2020-01-01")
    df = run_search(spec, n_workers=0)
    assert len(df) == 12
    assert df["error"].isna().all(), df.loc[df["error"].notna(), "error"].tolist()
    assert (df["n_trades"] > 0).any()


# ============================================================================
# Round-2 regime-gated RSI2 mean reversion (candidate #3). Appended,
# self-contained.
#
# GatedRSI2 is SEARCHABLE=False (see the class docstring): its long-only
# dip-in-uptrend context needs a *daily* EMA200 and a genuine up-regime, which
# the short driftless walk_df fixture never supplies, so it is not exercised by
# the generic parametrized contract tests above. These dedicated tests give it
# full-contract coverage on an *uptrending, multi-hundred-day* synthetic, with
# special attention to the daily-resample gates where lookahead bugs hide.
# ============================================================================

GATED_RSI2_BARS = 11000  # ~458 UTC days of 1h bars: daily ADX/ATR + EMA200 warm


@pytest.fixture(scope="module")
def uptrend_1h_df() -> pd.DataFrame:
    """1h geometric random walk with a *mild upward drift* so that the daily
    close spends time above its (lagging) daily EMA200 — the only regime in
    which GatedRSI2's long-only context gate can ever fire."""
    g = np.random.default_rng(11)
    close = 100.0 * np.exp(np.cumsum(g.normal(0.00008, 0.008, GATED_RSI2_BARS)))
    return ohlcv_from_close(close, g)


def test_gated_rsi2_registered():
    """Registry integration: resolvable by NAME and is the expected class."""
    from core.strategies.meanrev import GatedRSI2
    cls = get_strategy("gated_rsi2")
    assert cls is GatedRSI2
    assert cls.NAME == "gated_rsi2"
    assert cls.TIMEFRAMES == ("1h", "4h")
    assert cls.param_grid_size() <= 200  # stays within the ~200-combo budget


def test_gated_rsi2_contract(uptrend_1h_df):
    """Stance domain/length, zero warmup, determinism, and that it actually
    trades in the up-regime (otherwise the other assertions are vacuous)."""
    from core.strategies.meanrev import GatedRSI2
    strat = GatedRSI2(timeframe="1h")
    sig = strat.generate_signals(uptrend_1h_df)
    assert len(sig) == len(uptrend_1h_df)
    assert set(np.unique(sig)).issubset({0.0, 1.0})
    assert sig[0] == 0.0
    assert sig.sum() > 0, "expected some long exposure in the up-regime fixture"
    again = GatedRSI2(timeframe="1h").generate_signals(uptrend_1h_df)
    assert np.array_equal(sig, again)


def test_gated_rsi2_param_change_alters_signals(uptrend_1h_df):
    from core.strategies.base import ENGINE_PARAM_KEYS
    from core.strategies.meanrev import GatedRSI2
    base = GatedRSI2(timeframe="1h").generate_signals(uptrend_1h_df)
    changed = False
    for param, values in GatedRSI2.PARAM_SPACE.items():
        if param in ENGINE_PARAM_KEYS:
            continue
        for v in values:
            if v == GatedRSI2.DEFAULTS.get(param):
                continue
            sig = GatedRSI2(timeframe="1h", **{param: v}).generate_signals(uptrend_1h_df)
            if not np.array_equal(sig, base):
                changed = True
                break
        if changed:
            break
    assert changed, "no signal-affecting param found for gated_rsi2"


def test_gated_rsi2_min_edge_filter_has_bite(uptrend_1h_df):
    """The mandated min-edge-per-trade filter must actually gate entries:
    raising the floor monotonically reduces long exposure."""
    from core.strategies.meanrev import GatedRSI2
    n_lo = GatedRSI2(timeframe="1h", min_edge_pct=0.0).generate_signals(uptrend_1h_df).sum()
    n_mid = GatedRSI2(timeframe="1h", min_edge_pct=0.005).generate_signals(uptrend_1h_df).sum()
    n_hi = GatedRSI2(timeframe="1h", min_edge_pct=0.02).generate_signals(uptrend_1h_df).sum()
    assert n_lo >= n_mid >= n_hi
    assert n_lo > n_hi, "min-edge filter had no effect at all"


@pytest.mark.parametrize("tf", ["1h", "4h"])
def test_gated_rsi2_no_lookahead_with_daily_gates(uptrend_1h_df, tf):
    """The defining causality property, exercised where it is most fragile:
    the DAILY-resampled gates (ADX/ATR/EMA200 aggregated per UTC day and
    shifted to the previous completed day). Truncation is tested at several
    offsets *straddling day boundaries* so a partial trailing day cannot leak
    into the prefix's stance. stance(df[:k]) must equal stance(df)[:k]."""
    from core.strategies.meanrev import GatedRSI2
    if tf == "4h":
        df = uptrend_1h_df.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
    else:
        df = uptrend_1h_df
    full = np.asarray(GatedRSI2(timeframe=tf).generate_signals(df))
    n = len(df)
    bpd = 24 if tf == "1h" else 6
    for k in (n - 1, n - bpd, n - bpd - 1, n - bpd + 1, n - 3 * bpd, n // 2 + 5):
        if k < 300:
            continue
        trunc = np.asarray(GatedRSI2(timeframe=tf).generate_signals(df.iloc[:k]))
        assert len(trunc) == k
        assert np.array_equal(full[:k], trunc), f"lookahead at tf={tf} k={k}"
