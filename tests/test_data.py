"""Data-layer tests (core/data/fetcher.py) — offline, no network.

- fill_gaps: Upbit omits zero-trade candles; the loader reindexes to a full
  time grid, forward-fills close, sets open/high/low to that close and volume 0.
- load_ohlcv drop-last-incomplete: the final cached bar whose close time is in
  the future is the still-forming candle and must be dropped (refresh=False so
  nothing hits the network).
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from core.data.fetcher import cache_path, fill_gaps, load_ohlcv


# ------------------------------------------------------------------ fill_gaps

def test_fill_gaps_synthesizes_flat_zero_volume_bars():
    # hourly bars with 02:00 and 05:00 missing (a gapped Upbit-style frame)
    idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00",
         "2023-01-01 04:00", "2023-01-01 06:00"],
        tz="UTC", name="timestamp")
    df = pd.DataFrame({
        "open":   [10.0, 11.0, 13.0, 14.0, 16.0],
        "high":   [10.5, 11.5, 13.5, 14.5, 16.5],
        "low":    [ 9.5, 10.5, 12.5, 13.5, 15.5],
        "close":  [10.2, 11.2, 13.2, 14.2, 16.2],
        "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
    }, index=idx)

    out = fill_gaps(df, "1h")

    # grid is complete and contiguous at the 1h step
    full = pd.date_range(idx[0], idx[-1], freq="1h", tz="UTC")
    assert out.index.equals(full)
    assert out.index.name == "timestamp"
    assert len(out) == 7  # 5 real + 2 synthesized

    for gap_ts, prev_close in [("2023-01-01 02:00", 11.2), ("2023-01-01 05:00", 14.2)]:
        row = out.loc[pd.Timestamp(gap_ts, tz="UTC")]
        assert row["close"] == pytest.approx(prev_close)   # close forward-filled
        assert row["open"] == pytest.approx(prev_close)    # OHL flattened to close
        assert row["high"] == pytest.approx(prev_close)
        assert row["low"] == pytest.approx(prev_close)
        assert row["volume"] == 0.0                        # zero-trade candle

    # real bars are untouched and no synthesized bar carries volume
    assert out.loc[pd.Timestamp("2023-01-01 03:00", tz="UTC"), "volume"] == 300.0


def test_fill_gaps_noop_when_grid_already_complete():
    idx = pd.date_range("2023-01-01", periods=6, freq="1h", tz="UTC", name="timestamp")
    df = pd.DataFrame({c: np.arange(1.0, 7.0) for c in
                       ("open", "high", "low", "close", "volume")}, index=idx)
    out = fill_gaps(df, "1h")
    assert out.index.equals(idx)
    assert (out["volume"] == df["volume"]).all()


# --------------------------------------------------- load_ohlcv drop-last bar

@pytest.fixture
def testdata_parquet():
    """Write a cached-only 1h parquet whose LAST bar's timestamp is 'now'
    (its close time is one hour in the future -> forming candle)."""
    path = cache_path("binance", "TESTDATA/USDT", "1h")
    path.parent.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    idx = pd.DatetimeIndex(
        sorted(now - pd.Timedelta(hours=h) for h in (3, 2, 1, 0)),
        name="timestamp")
    df = pd.DataFrame({
        "open":   [1.0, 2.0, 3.0, 4.0],
        "high":   [1.5, 2.5, 3.5, 4.5],
        "low":    [0.5, 1.5, 2.5, 3.5],
        "close":  [1.2, 2.2, 3.2, 4.2],
        "volume": [10.0, 20.0, 30.0, 40.0],
    }, index=idx).astype(np.float64)
    df.to_parquet(path)
    yield now, path
    shutil.rmtree(path.parent, ignore_errors=True)


def test_load_ohlcv_drops_last_incomplete_bar(testdata_parquet):
    now, _ = testdata_parquet
    out = load_ohlcv("binance", "TESTDATA/USDT", "1h", refresh=False)

    assert len(out) == 3                              # the 'now' bar is dropped
    assert now not in out.index
    assert (now - pd.Timedelta(hours=1)) in out.index  # previous bar retained
    assert out.index.is_monotonic_increasing


def test_load_ohlcv_keeps_last_bar_when_drop_disabled(testdata_parquet):
    now, _ = testdata_parquet
    out = load_ohlcv("binance", "TESTDATA/USDT", "1h", refresh=False,
                     drop_last_incomplete=False)
    assert len(out) == 4
    assert now in out.index
