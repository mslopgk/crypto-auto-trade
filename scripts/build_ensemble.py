"""Build the final deployment ensemble from walk-forward survivors.

Steps:
1. For each surviving (exchange, symbol, timeframe, strategy) cell, pick
   deployment params by plateau-center selection on the most recent 365-day
   in-sample window (ending at the holdout boundary) — the same rule each
   walk-forward fold used, so deployment params are chosen exactly the way
   the validated OOS results were.
2. Weight cells by inverse annualized volatility of their stitched WFA OOS
   equity, with a correlation penalty (brief §3.2) and a 35% cap per cell.
3. Write config/ensemble.json.

The holdout evaluation lives in scripts/eval_holdout.py and must be run ONCE.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.constants import CONFIG_DIR, RESULTS_DIR, periods_per_year  # noqa: E402
from core.optimize.gates import select_plateau_center  # noqa: E402
from core.optimize.search import SearchSpec, run_search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_ensemble")

HOLDOUT_DAYS = 180
DEPLOY_IS_DAYS = 365
WEIGHT_CAP = 0.35

# WFA survivors: OOS sharpe >= ~0.9, WFE >= 0.5, stitched OOS MDD acceptable.
# (Corrected-methodology run 2026-07-08: binance SOL 1d tsmom dropped on
# WFE 0.30; binance BTC 1d ema_cross dropped on OOS sharpe 0.64 / MDD 47%.)
SURVIVORS = [
    ("binance", "BTC/USDT", "1d", "tsmom"),
    ("binance", "ETH/USDT", "1d", "tsmom"),
    ("binance", "ADA/USDT", "1d", "tsmom"),
    ("upbit", "BTC/KRW", "1d", "tsmom"),
    ("upbit", "ETH/KRW", "1d", "tsmom"),
    ("upbit", "SOL/KRW", "1d", "tsmom"),
    ("upbit", "BTC/KRW", "4h", "ema_cross"),
    ("upbit", "ETH/KRW", "4h", "ema_cross"),
    # Round-4 additions (2026-07-10): original vol-of-vol gated trend cells.
    # WFA OOS 1.68/1.38, MDD ~15-16%, WFE 0.70/0.50; max corr vs book 0.66/0.63.
    # Binance vov cells rejected: 0.70-0.75 corr vs same-symbol tsmom and weaker.
    ("upbit", "BTC/KRW", "4h", "vov_calm_trend"),
    ("upbit", "ETH/KRW", "4h", "vov_calm_trend"),
]


def deployment_params(exchange: str, symbol: str, timeframe: str, strategy: str) -> dict:
    """Plateau-center params on the most recent DEPLOY_IS_DAYS window (ex-holdout)."""
    from core.data.fetcher import load_ohlcv
    df = load_ohlcv(exchange, symbol, timeframe, refresh=False)
    end = df.index[-1] - pd.Timedelta(days=HOLDOUT_DAYS)
    start = end - pd.Timedelta(days=DEPLOY_IS_DAYS)
    # generous warmup for EMA200-style indicators: load 400 extra days before
    # ``start`` and mark them warmup-only via ``score_start`` so they warm the
    # indicators but are NOT graded — yielding the documented most-recent
    # DEPLOY_IS_DAYS scored window, exactly as each WFA fold scores its IS
    # window (research brief §2.2). NB: pass score_start, NOT start — a hard
    # ``start`` trim would drop the warmup prefix and cold-start the backtest.
    since = (start - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    spec = SearchSpec(exchange=exchange, symbols=[symbol], timeframes=[timeframe],
                      strategies=[strategy], max_combos_per_strategy=150,
                      since=since, score_start=start.isoformat(),
                      holdout_days=HOLDOUT_DAYS)
    res = run_search(spec, n_workers=12, out_path=None)
    ok = res[res["error"].isna() & (res["n_trades"] >= 5)]
    if ok.empty:
        raise ValueError(f"no valid IS rows for {exchange} {symbol} {timeframe} {strategy}")
    row = select_plateau_center(ok, strategy, symbol, timeframe)
    return json.loads(row["params"]), {
        "is_sharpe": float(row["sharpe"]), "is_cagr": float(row["cagr"]),
        "is_mdd": float(row["max_drawdown"]), "is_trades": int(row["n_trades"]),
    }


def load_oos_equity(exchange: str, symbol: str, timeframe: str, strategy: str) -> pd.Series:
    tag = f"{exchange}_{symbol.replace('/', '_')}_{timeframe}_{strategy}"
    d = json.loads((RESULTS_DIR / "wfa" / f"{tag}.json").read_text(encoding="utf-8"))
    eq = d["stitched_oos_equity"]
    s = pd.Series(eq["equity"], index=pd.DatetimeIndex(eq["timestamp"]), name=tag)
    return s


def main() -> int:
    # -- 1. deployment params per cell --
    members = []
    for exchange, symbol, timeframe, strategy in SURVIVORS:
        tag = f"{exchange}_{symbol.replace('/', '_')}_{timeframe}_{strategy}"
        log.info("selecting deployment params: %s", tag)
        params, is_stats = deployment_params(exchange, symbol, timeframe, strategy)
        wfa = json.loads((RESULTS_DIR / "wfa" / f"{tag}.json").read_text(encoding="utf-8"))
        members.append({
            "id": tag, "exchange": exchange, "symbol": symbol,
            "timeframe": timeframe, "strategy": strategy,
            "params": params, "recent_is": is_stats,
            "wfa": {
                "oos_sharpe": wfa["stitched_metrics"]["sharpe"],
                "oos_cagr": wfa["stitched_metrics"]["cagr"],
                "oos_mdd": wfa["stitched_metrics"]["max_drawdown"],
                "wfe": wfa["wfe"],
                "pct_profitable_folds": wfa["pct_profitable_folds"],
            },
        })
        log.info("  params=%s  recent IS sharpe %.2f", params, is_stats["is_sharpe"])

    # -- 2. weights: inverse vol with correlation penalty, capped --
    curves = {m["id"]: load_oos_equity(m["exchange"], m["symbol"], m["timeframe"], m["strategy"])
              for m in members}
    rets = pd.DataFrame({k: v.pct_change() for k, v in curves.items()}).dropna(how="all")
    ann = {}
    for m in members:
        ppy = periods_per_year(m["timeframe"])
        r = rets[m["id"]].dropna()
        ann[m["id"]] = float(r.std(ddof=1) * np.sqrt(ppy)) if len(r) > 10 else np.nan
    corr = rets.corr(min_periods=30)
    raw_w = {}
    for m in members:
        iv = 1.0 / ann[m["id"]] if ann[m["id"]] and np.isfinite(ann[m["id"]]) and ann[m["id"]] > 0 else 0.0
        high_corr = int(((corr[m["id"]] > 0.7) & (corr[m["id"]].index != m["id"])).sum())
        raw_w[m["id"]] = iv / (1.0 + high_corr)
    total = sum(raw_w.values())
    weights = {k: v / total for k, v in raw_w.items()}
    # cap + renormalize (iterative)
    for _ in range(10):
        over = {k: v for k, v in weights.items() if v > WEIGHT_CAP}
        if not over:
            break
        excess = sum(v - WEIGHT_CAP for v in over.values())
        under = {k: v for k, v in weights.items() if v < WEIGHT_CAP}
        for k in over:
            weights[k] = WEIGHT_CAP
        s_under = sum(under.values())
        for k in under:
            weights[k] += excess * (under[k] / s_under)
    for m in members:
        m["weight"] = round(weights[m["id"]], 4)

    config = {
        "created": pd.Timestamp.now(tz="UTC").isoformat(),
        "methodology": "round1 grid search -> gates -> rolling WFA (365/90d, plateau-center) "
                       "-> inverse-vol weights w/ corr penalty, cap 35% "
                       "-> holdout (last 180d) evaluated once via scripts/eval_holdout.py",
        "holdout_days": HOLDOUT_DAYS,
        "risk_defaults": {"target_annual_vol": 0.20, "risk_per_trade_pct": 0.01,
                          "dd_soft_pct": 0.10, "dd_hard_pct": 0.15,
                          "daily_loss_soft_pct": 0.015, "daily_loss_hard_pct": 0.03},
        "members": members,
        "correlation_matrix": corr.round(3).to_dict(),
    }
    out = CONFIG_DIR / "ensemble.json"
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    print("\nFinal ensemble:")
    for m in members:
        print(f"  {m['weight']:5.1%}  {m['id']:40s} params={m['params']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
