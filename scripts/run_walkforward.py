"""CLI: rolling walk-forward analysis for one strategy/symbol/timeframe.

Example:
    python scripts/run_walkforward.py --strategy donchian --symbol BTC/USDT ^
        --timeframe 1d --is-days 365 --oos-days 90 --step-days 90
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.optimize.walkforward import WalkForwardSpec, run_walkforward  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rolling walk-forward analysis.")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--strategy", required=True,
                    help="registry NAME or module.path:ClassName")
    ap.add_argument("--is-days", type=int, default=365)
    ap.add_argument("--oos-days", type=int, default=90)
    ap.add_argument("--step-days", type=int, default=90)
    ap.add_argument("--objective", default="sharpe")
    ap.add_argument("--min-trades-is", type=int, default=20)
    ap.add_argument("--max-combos", type=int, default=150)
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--holdout-days", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None, help="result json path")
    args = ap.parse_args()

    spec = WalkForwardSpec(
        exchange=args.exchange, symbol=args.symbol, timeframe=args.timeframe,
        strategy=args.strategy, is_days=args.is_days, oos_days=args.oos_days,
        step_days=args.step_days, objective=args.objective,
        min_trades_is=args.min_trades_is, max_combos=args.max_combos,
        since=args.since, holdout_days=args.holdout_days,
        cost_multiplier=args.cost_mult,
    )

    state = {"last_pct": -5, "t0": time.time()}

    def progress(done: int, total: int) -> None:
        pct = 100 * done // total
        if pct >= state["last_pct"] + 5:
            state["last_pct"] = pct
            print(f"  {pct:3d}%  ({done}/{total}, {time.time() - state['t0']:.0f}s)",
                  flush=True)

    print(f"walk-forward: {args.strategy} {args.symbol} {args.timeframe} "
          f"IS {args.is_days}d / OOS {args.oos_days}d step {args.step_days}d")
    result = run_walkforward(spec, n_workers=args.workers, progress_cb=progress)

    print(f"\nfolds: {len(result.folds)} (skipped {result.n_skipped_folds})")
    for f in result.folds:
        print(f"  fold {f['fold']:2d}  {f['oos_start'][:10]} -> {f['oos_end'][:10]}"
              f"  IS sharpe {f['is_metrics'].get('sharpe', float('nan')):6.2f}"
              f"  OOS sharpe {f['oos_metrics'].get('sharpe', float('nan')):6.2f}"
              f"  OOS ret {f['oos_metrics'].get('total_return', float('nan')):7.2%}"
              f"  params {json.dumps(f['params'], sort_keys=True)}")

    sm = result.stitched_metrics
    print("\nstitched OOS:"
          f"  sharpe {sm.get('sharpe', float('nan')):.2f}"
          f"  cagr {sm.get('cagr', float('nan')):.2%}"
          f"  mdd {sm.get('max_drawdown', float('nan')):.2%}"
          f"  pf {sm.get('profit_factor', float('nan')):.2f}"
          f"  trades {sm.get('n_trades', 0)}")
    print(f"WFE {result.wfe:.2f} | profitable folds {result.pct_profitable_folds:.0%}"
          f" | param stability {result.param_stability:.0%}")

    path = result.save(args.out)
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
