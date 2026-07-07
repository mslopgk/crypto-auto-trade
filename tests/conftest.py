"""Shared fixtures: rng-seeded synthetic OHLCV DataFrame builders.

All frames follow the project contract: DatetimeIndex (UTC, tz-aware, named
'timestamp'), float64 columns open/high/low/close/volume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OHLCV_COLS = ("open", "high", "low", "close", "volume")


def hourly_index(n: int, start: str = "2023-01-01") -> pd.DatetimeIndex:
    """Contiguous hourly UTC index per the project DataFrame contract."""
    return pd.date_range(start, periods=n, freq="1h", tz="UTC", name="timestamp")


def ohlcv_from_close(close: np.ndarray, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Build a contract-conforming OHLCV frame from a close path.

    open = previous close (gap-free); highs/lows extend beyond open/close by
    a small wick so high >= max(open, close) >= min(open, close) >= low.
    """
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    open_ = np.concatenate(([close[0]], close[:-1]))
    if rng is None:
        wick = np.full(n, 0.001)
        volume = np.full(n, 1000.0)
    else:
        wick = rng.uniform(0.0005, 0.004, n)
        volume = rng.lognormal(6.0, 0.5, n)
    high = np.maximum(open_, close) * (1.0 + wick)
    low = np.minimum(open_, close) * (1.0 - wick)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=hourly_index(n),
    ).astype(np.float64)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def random_walk_df() -> pd.DataFrame:
    """1000 bars of 1h geometric random walk, seeded."""
    g = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(g.normal(0.0, 0.01, 1000)))
    return ohlcv_from_close(close, g)


@pytest.fixture
def uptrend_df() -> pd.DataFrame:
    """Strong persistent uptrend with mild noise, 1000 bars 1h."""
    g = np.random.default_rng(8)
    close = 100.0 * np.exp(np.cumsum(g.normal(0.004, 0.003, 1000)))
    return ohlcv_from_close(close, g)


@pytest.fixture
def downtrend_df() -> pd.DataFrame:
    """Strong persistent downtrend with mild noise, 1000 bars 1h."""
    g = np.random.default_rng(9)
    close = 100.0 * np.exp(np.cumsum(g.normal(-0.004, 0.003, 1000)))
    return ohlcv_from_close(close, g)


@pytest.fixture
def range_df() -> pd.DataFrame:
    """Flat sideways range: sinusoid around 100 plus noise, 1000 bars 1h."""
    g = np.random.default_rng(10)
    t = np.arange(1000)
    close = 100.0 * (1.0 + 0.03 * np.sin(t / 12.0)) + g.normal(0.0, 0.3, 1000)
    return ohlcv_from_close(close, g)


@pytest.fixture
def make_df():
    """Factory for crafted tiny frames: make_df(opens, highs, lows, closes, volume=None)."""

    def _make(opens, highs, lows, closes, volume=None) -> pd.DataFrame:
        n = len(opens)
        vol = np.full(n, 1000.0) if volume is None else np.asarray(volume, dtype=np.float64)
        df = pd.DataFrame(
            {
                "open": np.asarray(opens, dtype=np.float64),
                "high": np.asarray(highs, dtype=np.float64),
                "low": np.asarray(lows, dtype=np.float64),
                "close": np.asarray(closes, dtype=np.float64),
                "volume": vol,
            },
            index=hourly_index(n),
        )
        # sanity: crafted bars must be internally consistent
        assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12).all()
        return df

    return _make
