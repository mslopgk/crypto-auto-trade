"""Perpetual-funding data layer (Binance USDM).

The parquet cache under ``data/binanceusdm_funding/{BASE}_USDT.parquet`` holds
one column ``rate`` — the per-interval *realized* funding rate (fraction, e.g.
0.0001 = 0.01% / 8h) indexed by settlement timestamp (UTC, ~8h cadence: 00:00,
08:00, 16:00). These are settled prints, so consuming them as-of a bar close is
strictly causal.

Two public helpers:

``load_funding(base)``
    Module-cached load of the 8h rate Series for a symbol base (e.g. ``"BTC"``).
    Returns ``None`` when no parquet exists (callers fall back gracefully).

``daily_funding_features(rates, K_list, L_list)``
    Resamples the 8h prints to a **daily** frame of annualized funding and its
    rolling z-scores, using only prints settled on-or-before each daily close
    (no lookahead). Annualization: ``rate * 3 * 365`` (3 settlements/day).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.constants import DATA_DIR

log = logging.getLogger(__name__)

FUNDING_DIR = DATA_DIR / "binanceusdm_funding"

# module-level cache: base -> Series | None
_CACHE: dict[str, pd.Series | None] = {}


def funding_path(symbol_base: str):
    return FUNDING_DIR / f"{symbol_base.upper()}_USDT.parquet"


def load_funding(symbol_base: str) -> pd.Series | None:
    """Load the 8h realized-funding rate Series for ``symbol_base`` (e.g. 'BTC').

    Returns a float Series named 'rate' with a UTC DatetimeIndex, or ``None``
    when no cache file exists. Results (including ``None``) are memoized.
    """
    key = symbol_base.upper()
    if key in _CACHE:
        return _CACHE[key]
    path = funding_path(key)
    if not path.exists():
        log.warning("no funding parquet for %s at %s", key, path)
        _CACHE[key] = None
        return None
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    s = df["rate"].astype("float64")
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "rate"
    _CACHE[key] = s
    return s


def daily_funding_features(rates: pd.Series, K_list, L_list) -> pd.DataFrame:
    """Daily annualized funding + rolling z-scores, causal as-of daily close.

    For each ``K`` in ``K_list`` the annualized funding is the trailing mean of
    the last ``K`` *prints* (``rate * 3 * 365``); it is then resampled to a
    contiguous daily grid taking the **last settled print of each calendar day**
    (i.e. the value known at that day's close — never the next day's 00:00
    settlement). For each ``L`` in ``L_list`` a rolling z-score over ``L`` days
    is computed on that daily series (population std; NaN when std == 0).

    Returns a DataFrame (daily UTC midnight index) with columns
    ``annfund_K{K}`` and ``z_K{K}_L{L}``. Warmup rows are NaN.
    """
    s = rates.astype("float64").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    ann_print = s * 3.0 * 365.0

    out: dict[str, pd.Series] = {}
    for K in K_list:
        K = int(K)
        roll_k = ann_print.rolling(K, min_periods=K).mean()
        # ".last()" within each calendar day == the last settled print that day,
        # which is known at that day's OHLCV close (D+1 00:00 settlement is
        # deliberately excluded from day D).
        daily = roll_k.resample("D").last()
        out[f"annfund_K{K}"] = daily
        for L in L_list:
            L = int(L)
            m = daily.rolling(L, min_periods=L).mean()
            sd = daily.rolling(L, min_periods=L).std(ddof=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (daily - m) / sd
            z = z.where(sd > 0.0)
            out[f"z_K{K}_L{L}"] = z

    feats = pd.DataFrame(out)
    feats.index.name = "timestamp"
    return feats
