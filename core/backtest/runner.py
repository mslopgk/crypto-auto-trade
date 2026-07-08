"""Glue: run a Strategy instance through the engine with venue cost profiles."""
from __future__ import annotations

import pandas as pd

from core.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from core.constants import cost_profile
from core.strategies.base import Strategy


def run_strategy_backtest(df: pd.DataFrame, strategy: Strategy, timeframe: str,
                          exchange_id: str = "binance", symbol: str = "BTC/USDT",
                          overrides: dict | None = None) -> BacktestResult:
    """Backtest `strategy` on `df` with venue-tier costs and the strategy's
    engine-level exit params. `overrides` wins over everything."""
    costs = cost_profile(exchange_id, symbol)
    kwargs: dict = dict(costs)
    kwargs.update(strategy.engine_params())
    if getattr(strategy, "SUPPORTS_SHORT", False):
        kwargs["allow_short"] = True
    if overrides:
        kwargs.update(overrides)
    config = BacktestConfig(**kwargs)
    stance = strategy.generate_signals(df)
    size_frac = strategy.generate_size_frac(df)
    return run_backtest(df, stance, config, timeframe=timeframe, size_frac_arr=size_frac)
