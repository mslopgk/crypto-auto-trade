"""Batch walk-forward validation over candidate (strategy, symbol, timeframe) cells.

Writes one JSON per cell to results/wfa/ plus a summary table results/wfa_summary.csv.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from core.constants import RESULTS_DIR  # noqa: E402
from core.optimize.walkforward import WalkForwardSpec, run_walkforward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("wfa_batch")

# (exchange, symbol, timeframe, strategy) — round-1 gate survivors worth validating
CELLS = [
    # Binance — TSMOM family (most robust in round 1)
    ("binance", "BTC/USDT", "1d", "tsmom"),
    ("binance", "ETH/USDT", "1d", "tsmom"),
    ("binance", "SOL/USDT", "1d", "tsmom"),
    ("binance", "ADA/USDT", "1d", "tsmom"),
    # Binance — Donchian family
    ("binance", "BTC/USDT", "1d", "donchian_multi"),
    ("binance", "BTC/USDT", "4h", "donchian_multi"),
    ("binance", "BTC/USDT", "1d", "donchian_single"),
    ("binance", "ETH/USDT", "1d", "donchian_single"),
    # Binance — EMA cross
    ("binance", "BTC/USDT", "1d", "ema_cross"),
    ("binance", "ETH/USDT", "1d", "ema_cross"),
    ("binance", "BTC/USDT", "4h", "ema_cross"),
    # Binance — RSI momentum
    ("binance", "BTC/USDT", "1d", "rsi_momentum"),
    # Upbit — KRW markets
    ("upbit", "BTC/KRW", "1d", "tsmom"),
    ("upbit", "ETH/KRW", "1d", "tsmom"),
    ("upbit", "SOL/KRW", "1d", "tsmom"),
    ("upbit", "BTC/KRW", "4h", "ema_cross"),
    ("upbit", "ETH/KRW", "4h", "ema_cross"),
    ("upbit", "BTC/KRW", "1d", "ema_cross"),
    ("upbit", "BTC/KRW", "1d", "donchian_single"),
    ("upbit", "ETH/KRW", "1d", "donchian_single"),
    ("upbit", "BTC/KRW", "1d", "rsi_momentum"),
    ("upbit", "BTC/KRW", "1h", "larry_vb"),
]

HOLDOUT_DAYS = 180  # untouched final tail, same as round-1 search


def main() -> int:
    out_dir = RESULTS_DIR / "wfa"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, (exchange, symbol, timeframe, strategy) in enumerate(CELLS, 1):
        tag = f"{exchange}_{symbol.replace('/', '_')}_{timeframe}_{strategy}"
        log.info("[%d/%d] %s", i, len(CELLS), tag)
        spec = WalkForwardSpec(
            exchange=exchange, symbol=symbol, timeframe=timeframe, strategy=strategy,
            is_days=365, oos_days=90, step_days=90,
            holdout_days=HOLDOUT_DAYS, max_combos=150,
        )
        try:
            res = run_walkforward(spec, n_workers=12)
            d = res.to_dict()
            (out_dir / f"{tag}.json").write_text(json.dumps(d), encoding="utf-8")
            sm = res.stitched_metrics
            rows.append({
                "exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                "strategy": strategy,
                "oos_sharpe": sm.get("sharpe"), "oos_cagr": sm.get("cagr"),
                "oos_mdd": sm.get("max_drawdown"), "oos_pf": sm.get("profit_factor"),
                "oos_trades": sm.get("n_trades"), "wfe": res.wfe,
                "pct_profitable_folds": res.pct_profitable_folds,
                "param_stability": res.param_stability,
                "n_folds": len(res.folds), "n_skipped": res.n_skipped_folds,
            })
            log.info("  -> OOS sharpe %.2f cagr %.1f%% mdd %.1f%% wfe %.2f folds+ %.0f%%",
                     sm.get("sharpe", 0), 100 * sm.get("cagr", 0),
                     100 * sm.get("max_drawdown", 0), res.wfe,
                     100 * res.pct_profitable_folds)
        except Exception as e:
            log.error("  FAILED %s: %s", tag, e)
            rows.append({"exchange": exchange, "symbol": symbol, "timeframe": timeframe,
                         "strategy": strategy, "error": str(e)[:200]})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "wfa_summary.csv", index=False)
    log.info("batch done in %.0fs -> results/wfa_summary.csv", time.time() - t0)
    with pd.option_context("display.width", 220):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
