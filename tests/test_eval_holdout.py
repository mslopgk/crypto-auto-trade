"""Unit tests for scripts/eval_holdout selection/normalization helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_holdout import holdout_trades, normalize_daily  # noqa: E402


def test_holdout_trades_filters_by_entry_not_exit():
    holdout_start = pd.Timestamp("2024-01-01", tz="UTC")
    trades = pd.DataFrame({
        # entered during warmup, exits inside the holdout -> must be EXCLUDED
        "entry_time": [pd.Timestamp("2023-12-20", tz="UTC"),
                       pd.Timestamp("2024-01-05", tz="UTC"),
                       pd.Timestamp("2024-02-01", tz="UTC")],
        "exit_time": [pd.Timestamp("2024-01-03", tz="UTC"),
                      pd.Timestamp("2024-01-10", tz="UTC"),
                      pd.Timestamp("2024-02-05", tz="UTC")],
        "pnl": [999.0, 1.0, 2.0],
    })
    kept = holdout_trades(trades, holdout_start)
    assert len(kept) == 2
    # the warmup-entry trade (pnl 999) must not leak into the holdout count
    assert 999.0 not in set(kept["pnl"])
    assert (kept["entry_time"] >= holdout_start).all()


def test_normalize_daily_baseline_is_one():
    idx = pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC")
    # first daily row is NOT 1.0 (last intra-day bar after .last() resampling)
    daily = pd.DataFrame({
        "a": [1.02, 1.05, 1.10, 1.08],
        "b": [0.98, 1.01, 0.95, 1.03],
    }, index=idx)
    norm = normalize_daily(daily)
    # every column re-based to exactly 1.0 at the first daily timestamp
    assert (norm.iloc[0] == 1.0).all()
    # a weighted portfolio therefore starts at exactly 1.0
    w = pd.Series({"a": 0.5, "b": 0.5})
    port = (norm * w).sum(axis=1)
    assert abs(float(port.iloc[0]) - 1.0) < 1e-12
    # ratios preserved: a grows 1.08/1.02 over the window
    assert abs(float(norm["a"].iloc[-1]) - 1.08 / 1.02) < 1e-12
