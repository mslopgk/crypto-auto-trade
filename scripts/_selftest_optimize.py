"""Self-test for the optimize pipeline on synthetic data (no network).

Writes a synthetic OHLCV parquet into the data cache for symbol TEST/USDT,
then runs run_search (2 workers) and run_walkforward and asserts sane output.

Run from the project root:
    python scripts/_selftest_optimize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.indicators import sma  # noqa: E402
from core.strategies.base import Strategy  # noqa: E402

SYMBOL = "TEST/USDT"
TIMEFRAME = "1h"
N_BARS = 4000


class InlineTestStrategy(Strategy):
    """Minimal SMA-crossover used when the real strategy zoo is unavailable."""

    NAME = "inline_sma_cross"
    TIMEFRAMES = ("1h",)
    PARAM_SPACE = {"fast": [5, 10], "slow": [20, 40], "trail_atr_mult": [0.0, 2.5]}
    DEFAULTS = {"fast": 10, "slow": 40, "trail_atr_mult": 0.0}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        f = sma(c, int(self.params["fast"]))
        s = sma(c, int(self.params["slow"]))
        return np.where(np.isnan(f) | np.isnan(s), 0.0,
                        np.where(f > s, 1.0, 0.0))


def write_fixture() -> None:
    from core.data.fetcher import cache_path
    rng = np.random.default_rng(7)
    t = np.arange(N_BARS)
    drift = 0.0004 * np.sin(t / 300.0)  # alternating trend regimes
    rets = drift + rng.normal(0.0, 0.006, N_BARS)
    close = 20_000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.003, N_BARS))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.003, N_BARS))
    volume = rng.uniform(10.0, 100.0, N_BARS)
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="h", tz="UTC", name="timestamp")
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": volume}, index=idx)
    path = cache_path("binance", SYMBOL, TIMEFRAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"fixture: {len(df)} bars -> {path}")


def pick_strategy() -> str:
    """Prefer the real registry; fall back to the inline strategy while the
    strategy zoo is still being written by another agent."""
    try:
        from core.strategies.registry import all_strategies
        strats = all_strategies()
        if "ema_crossover" in strats and TIMEFRAME in strats["ema_crossover"].TIMEFRAMES:
            return "ema_crossover"
        for name, cls in strats.items():
            if TIMEFRAME in cls.TIMEFRAMES and getattr(cls, "SEARCHABLE", True):
                return name
    except Exception as e:
        print(f"registry unavailable ({type(e).__name__}: {e}) -> inline strategy")
    return "scripts._selftest_optimize:InlineTestStrategy"


def main() -> int:
    from core.optimize.gates import apply_gates, select_plateau_center
    from core.optimize.search import SearchSpec, run_search
    from core.optimize.walkforward import WalkForwardSpec, run_walkforward

    write_fixture()
    strategy = pick_strategy()
    print(f"strategy under test: {strategy}")

    # -- search --
    spec = SearchSpec(exchange="binance", symbols=[SYMBOL], timeframes=[TIMEFRAME],
                      strategies=[strategy], max_combos_per_strategy=8,
                      since="2024-01-01")
    out = Path(__file__).resolve().parents[1] / "results" / "search_selftest.parquet"
    df = run_search(spec, n_workers=2, out_path=out)
    assert len(df) > 0, "search returned no rows"
    ok = df[df["error"].isna()]
    assert len(ok) > 0, f"all combos errored: {df['error'].dropna().tolist()[:3]}"
    for col in ("sharpe", "sortino", "max_drawdown", "profit_factor",
                "n_trades", "cagr"):
        assert col in ok.columns, f"missing metric column {col}"
    print(f"search OK: {len(ok)}/{len(df)} combos succeeded, "
          f"best sharpe {ok['sharpe'].max():.2f}")

    gated = apply_gates(ok, min_trades=1, min_sharpe=-99, max_mdd=1.0,
                        min_pf=0.0, min_trades_per_year=0.0, return_all=True)
    assert {"passed", "reasons", "suspicious"} <= set(gated.columns)
    center = select_plateau_center(ok, strategy, SYMBOL, TIMEFRAME)
    print(f"gates OK: {int(gated['passed'].sum())} passed relaxed gates; "
          f"plateau center sharpe {center['sharpe']:.2f} "
          f"(neighborhood-median {center['plateau_metric']:.2f})")

    # -- walk-forward --
    wspec = WalkForwardSpec(exchange="binance", symbol=SYMBOL, timeframe=TIMEFRAME,
                            strategy=strategy, is_days=60, oos_days=20, step_days=20,
                            min_trades_is=2, max_combos=8, since="2024-01-01")
    wres = run_walkforward(wspec, n_workers=2)
    assert len(wres.folds) >= 3, f"expected >=3 folds, got {len(wres.folds)}"
    eq = wres.stitched_oos_equity
    assert eq.index.is_monotonic_increasing, "stitched equity index not monotone"
    assert eq.index.is_unique, "stitched equity index has duplicates"
    assert np.isfinite(eq.to_numpy(dtype=np.float64)).all(), "non-finite equity"
    saved = wres.save(Path(__file__).resolve().parents[1] / "results" / "wfa_selftest.json")
    print(f"walk-forward OK: {len(wres.folds)} folds, "
          f"WFE {wres.wfe:.2f}, profitable {wres.pct_profitable_folds:.0%}, "
          f"stability {wres.param_stability:.0%}, saved {saved.name}")

    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
