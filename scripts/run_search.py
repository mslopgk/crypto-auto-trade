"""CLI: mass strategy parameter search over cached OHLCV data.

Example:
    python scripts/run_search.py --symbols BTC/USDT,ETH/USDT --timeframes 1d,4h ^
        --strategies donchian,ema_crossover --max-combos 200 --holdout-days 180
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.constants import DEFAULT_SYMBOLS  # noqa: E402
from core.optimize.search import SearchSpec, run_search  # noqa: E402

TOP_COLUMNS = ["strategy", "symbol", "timeframe", "params", "sharpe", "sortino",
               "calmar", "max_drawdown", "profit_factor", "cagr", "n_trades",
               "trades_per_year"]


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid/random search over strategies.")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbols", type=_csv, default=None,
                    help="comma-separated, e.g. BTC/USDT,ETH/USDT")
    ap.add_argument("--timeframes", type=_csv, default=["1d", "4h"])
    ap.add_argument("--strategies", type=_csv, default=None,
                    help="comma-separated registry names (default: all searchable)")
    ap.add_argument("--max-combos", type=int, default=200,
                    help="max param combos per strategy (seeded subsample)")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--holdout-days", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None, help="results parquet path")
    args = ap.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS.get(args.exchange, ["BTC/USDT"])
    spec = SearchSpec(
        exchange=args.exchange,
        symbols=symbols,
        timeframes=args.timeframes,
        strategies=args.strategies,
        max_combos_per_strategy=args.max_combos,
        since=args.since,
        holdout_days=args.holdout_days,
        cost_multiplier=args.cost_mult,
    )

    state = {"last_pct": -5, "t0": time.time()}

    def progress(done: int, total: int) -> None:
        pct = 100 * done // total
        if pct >= state["last_pct"] + 5:
            state["last_pct"] = pct
            elapsed = time.time() - state["t0"]
            print(f"  {pct:3d}%  ({done}/{total} combos, {elapsed:.0f}s)", flush=True)

    print(f"search: {len(symbols)} symbols x {args.timeframes} "
          f"(cost x{args.cost_mult}, holdout {args.holdout_days}d)")
    df = run_search(spec, n_workers=args.workers, out_path=args.out,
                    progress_cb=progress)

    ok = df[df["error"].isna()].copy()
    n_err = len(df) - len(ok)
    print(f"\ndone: {len(df)} combos, {n_err} errors")
    if n_err:
        top_err = df.loc[df["error"].notna(), "error"].value_counts().head(5)
        print("most common errors:")
        print(top_err.to_string())
    if ok.empty:
        print("no successful combos")
        return 1

    top = ok.sort_values("sharpe", ascending=False).head(20)
    cols = [c for c in TOP_COLUMNS if c in top.columns]
    with pd.option_context("display.max_colwidth", 60, "display.width", 220):
        print("\ntop 20 by sharpe:")
        print(top[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
