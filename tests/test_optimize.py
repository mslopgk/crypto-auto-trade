"""Optimizer end-to-end micro tests (core/optimize/search.py + walkforward.py).

Both run against a synthetic cache-only parquet (never the network). They are
deliberately tiny (few combos, short windows) but exercise the real code paths:
task enumeration, multiprocessing fan-out, metric collection, plateau/raw pick,
OOS-with-IS-warmup evaluation and OOS equity stitching.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from core.data.fetcher import cache_path
from core.optimize.gates import select_plateau_center
from core.optimize.search import (SearchSpec, build_tasks, evaluate_task,
                                   run_search)
from core.optimize.walkforward import WalkForwardSpec, run_walkforward


def _write_synth(symbol: str, n: int, seed: int) -> None:
    path = cache_path("binance", symbol, "1h")
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    g = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(g.normal(0.0001, 0.01, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    wick = g.uniform(0.0005, 0.004, n)
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": g.lognormal(6.0, 0.5, n),
    }, index=idx).astype(np.float64)
    df.to_parquet(path)


@pytest.fixture(scope="module")
def testopt_data():
    symbol = "TESTOPT/USDT"
    _write_synth(symbol, 4000, seed=11)
    # drop any memoized load from a previous run so the fresh parquet is used
    from core.optimize.search import _load_cached
    _load_cached.cache_clear()
    yield symbol
    _load_cached.cache_clear()
    shutil.rmtree(cache_path("binance", symbol, "1h").parent, ignore_errors=True)


# --------------------------------------------------------------- build_tasks

def test_explicit_strategy_honors_requested_timeframe(testopt_data):
    # ema_cross recommends 4h/1d, but an explicit request for 1h must be honored
    spec = SearchSpec(symbols=[testopt_data], timeframes=["1h"],
                      strategies=["ema_cross"], max_combos_per_strategy=6)
    tasks = build_tasks(spec)
    assert len(tasks) == 6
    assert {t["timeframe"] for t in tasks} == {"1h"}


# ------------------------------------------------------------------ run_search

def test_run_search_micro_parallel(testopt_data, tmp_path):
    spec = SearchSpec(symbols=[testopt_data], timeframes=["1h"],
                      strategies=["ema_cross"], max_combos_per_strategy=6)
    df = run_search(spec, n_workers=2, out_path=tmp_path / "search.parquet")

    assert len(df) > 0
    assert len(df) == 6
    ok = df[df["error"].isna()]
    assert len(ok) > 0, f"all combos errored: {df['error'].tolist()}"
    sharpe = pd.to_numeric(ok["sharpe"], errors="coerce")
    assert np.isfinite(sharpe.to_numpy()).all()
    assert (ok["n_trades"] > 0).any()
    assert (tmp_path / "search.parquet").exists()


# ------------------------------------------------------------ run_walkforward

def test_run_walkforward_micro(testopt_data):
    spec = WalkForwardSpec(
        exchange="binance", symbol=testopt_data, timeframe="1h",
        strategy="ema_cross", is_days=60, oos_days=20, step_days=20,
        max_combos=6, since="2019-01-01", min_trades_is=1)
    res = run_walkforward(spec, n_workers=0)

    assert len(res.folds) >= 3
    # every retained fold carries a chosen param set and IS/OOS metric blocks
    for fold in res.folds:
        assert isinstance(fold["params"], dict)
        assert "oos_metrics" in fold and "sharpe" in fold["oos_metrics"]

    eq = res.stitched_oos_equity
    assert isinstance(eq, pd.Series)
    assert len(eq) >= 2
    assert np.isfinite(eq.to_numpy()).all()
    assert eq.index.is_monotonic_increasing
    assert np.isfinite(res.stitched_metrics["sharpe"])


# ------------------------------------------------- score-window (warmup slice)

def test_score_start_excludes_warmup_from_metrics(testopt_data):
    """A task with ``score_start`` must load the earlier bars for indicator
    warmup but grade ONLY the [score_start, end) slice — so its metric window
    (years) and trade count reflect the scored slice, not the full window."""
    base = {
        "exchange": "binance", "symbol": testopt_data, "timeframe": "1h",
        "strategy": "ema_cross",
        "params": json.dumps({"fast": 9, "slow": 21, "regime_filter": 0,
                              "trail_atr_mult": 0.0}, sort_keys=True),
        "since": "2019-01-01",
    }
    full = evaluate_task(dict(base))
    # ~120 days of warmup precede score_start; only the tail (~46 days) is scored
    scored = evaluate_task({**base, "score_start": "2021-05-01T00:00:00+00:00"})

    assert full["error"] is None, full["error"]
    assert scored["error"] is None, scored["error"]
    # the scored window is strictly shorter than the full window
    assert scored["years"] < full["years"]
    assert full["years"] > 2.0 * scored["years"]
    # warmup-period trades are excluded from the scored count
    assert scored["n_trades"] < full["n_trades"]


# --------------------------------------------------- plateau lone-peak guard

def test_plateau_center_rejects_lone_peak():
    """An isolated spike whose one-step neighbors are absent/filtered must NOT
    win over a genuinely supported plateau (research brief §2.4)."""
    # 1-param grid; fast=40 is a fragile lone spike (sharpe 5.0) with < 2 present
    # neighbors, while 15/20/25 form a supported plateau (~1.0).
    rows = []
    for fast, sharpe in [(10, 0.5), (15, 1.0), (20, 1.1), (25, 1.0), (40, 5.0)]:
        rows.append({
            "strategy": "ema_cross", "symbol": "X/USDT", "timeframe": "1d",
            "params": json.dumps({"fast": fast}), "error": None, "sharpe": sharpe,
            "n_trades": 50,
        })
    df = pd.DataFrame(rows)
    pick = select_plateau_center(df, "ema_cross", "X/USDT", "1d", metric="sharpe")
    picked_fast = json.loads(pick["params"])["fast"]

    assert picked_fast != 40, "lone spike must not win the plateau selection"
    assert picked_fast in (15, 20, 25), "should pick a supported plateau member"
    assert pick["plateau_metric"] < 5.0
