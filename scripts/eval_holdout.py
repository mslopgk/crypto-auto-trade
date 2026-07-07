"""Single-shot holdout evaluation of config/ensemble.json.

Backtests every ensemble member on the untouched final holdout window with its
deployment params, combines member equity by weight, and reports portfolio
metrics. This is meant to be run ONCE — repeated runs against the same holdout
turn it into another in-sample set.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.backtest.metrics import compute_metrics  # noqa: E402
from core.backtest.runner import run_strategy_backtest  # noqa: E402
from core.constants import CONFIG_DIR, RESULTS_DIR  # noqa: E402
from core.data.fetcher import load_ohlcv  # noqa: E402
from core.strategies.registry import get_strategy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eval_holdout")

WARMUP_DAYS = 400  # indicator warmup before the holdout window


def holdout_trades(trades: pd.DataFrame, holdout_start: pd.Timestamp) -> pd.DataFrame:
    """Trades ENTERED at/after the holdout start.

    Filtering on ``entry_time`` (not ``exit_time``) avoids attributing a
    warmup-entered position's full warmup+holdout pnl to the holdout window.
    """
    return trades[trades["entry_time"] >= holdout_start]


def normalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Re-base each daily member curve to 1.0 at the first daily timestamp.

    Daily ``.last()`` resampling puts the last intra-day bar (not the
    holdout-start value of 1.0) in the first row, so a weighted portfolio built
    from these curves would not start at exactly 1.0 without this re-base.
    """
    return daily / daily.iloc[0]


def main() -> int:
    config = json.loads((CONFIG_DIR / "ensemble.json").read_text(encoding="utf-8"))
    holdout_days = int(config["holdout_days"])

    member_rows = []
    curves: dict[str, pd.Series] = {}
    for m in config["members"]:
        tag = m["id"]
        df = load_ohlcv(m["exchange"], m["symbol"], m["timeframe"], refresh=False)
        holdout_start = df.index[-1] - pd.Timedelta(days=holdout_days)
        window = df[df.index >= holdout_start - pd.Timedelta(days=WARMUP_DAYS)]

        cls = get_strategy(m["strategy"])
        params = dict(m["params"])
        if "timeframe" in cls.DEFAULTS:
            params["timeframe"] = m["timeframe"]
        strategy = cls(**params)
        result = run_strategy_backtest(window, strategy, m["timeframe"],
                                       exchange_id=m["exchange"], symbol=m["symbol"])
        eq = result.equity[result.equity.index >= holdout_start]
        eq = eq / float(eq.iloc[0])  # normalize to 1.0 at holdout start
        # count only trades ENTERED in the holdout; a warmup-entry position
        # closing inside the holdout would otherwise attribute its full
        # warmup+holdout pnl to the holdout window.
        trades = holdout_trades(result.trades, holdout_start)
        metrics = compute_metrics(eq * 10_000, trades.reset_index(drop=True),
                                  timeframe=m["timeframe"])
        curves[tag] = eq
        member_rows.append({
            "id": tag, "weight": m["weight"],
            "holdout_return": float(eq.iloc[-1] - 1.0),
            "sharpe": metrics["sharpe"], "mdd": metrics["max_drawdown"],
            "n_trades": int(metrics["n_trades"]),
            "wfa_oos_sharpe": m["wfa"]["oos_sharpe"],
        })
        log.info("%-40s ret %+6.1f%% sharpe %5.2f mdd %4.1f%% trades %d",
                 tag, 100 * (eq.iloc[-1] - 1), metrics["sharpe"],
                 100 * metrics["max_drawdown"], int(metrics["n_trades"]))

    # -- portfolio: weighted average of normalized member curves (daily grid) --
    daily = pd.DataFrame({
        k: v.resample("1D").last().ffill() for k, v in curves.items()
    }).ffill().dropna(how="any")
    # re-normalize each member on the daily grid so the portfolio baseline is
    # exactly 1.0: daily .last() resampling puts the last intra-day bar (not the
    # holdout-start value of 1.0) in the first daily row, so divide it out.
    daily = normalize_daily(daily)
    w = np.array([m["weight"] for m in config["members"]])
    w = w / w.sum()
    port = (daily * w).sum(axis=1)
    port_metrics = compute_metrics(port * 10_000, pd.DataFrame(columns=["pnl", "ret_pct", "bars_held"]),
                                   timeframe="1d")

    report = {
        "evaluated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "holdout_days": holdout_days,
        "holdout_span": [str(daily.index[0]), str(daily.index[-1])],
        "portfolio": {
            "total_return": float(port.iloc[-1] - 1.0),
            "sharpe": port_metrics["sharpe"],
            "max_drawdown": port_metrics["max_drawdown"],
            "ann_volatility": port_metrics["ann_volatility"],
        },
        "members": member_rows,
        "portfolio_equity": {"date": [str(d.date()) for d in port.index],
                             "value": [float(v) for v in port]},
    }
    out = RESULTS_DIR / "holdout_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== HOLDOUT (final untouched %dd) ===" % holdout_days)
    print(f"span: {daily.index[0].date()} -> {daily.index[-1].date()}")
    print(f"PORTFOLIO: return {100 * (port.iloc[-1] - 1):+.1f}%  "
          f"sharpe {port_metrics['sharpe']:.2f}  "
          f"mdd {100 * port_metrics['max_drawdown']:.1f}%  "
          f"ann.vol {100 * port_metrics['ann_volatility']:.1f}%")
    print("\nmembers:")
    for r in sorted(member_rows, key=lambda x: -x["holdout_return"]):
        print(f"  {r['weight']:5.1%} {r['id']:40s} ret {100 * r['holdout_return']:+6.1f}% "
              f"sharpe {r['sharpe']:5.2f} mdd {100 * r['mdd']:4.1f}% "
              f"(WFA OOS sharpe {r['wfa_oos_sharpe']:.2f})")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
