"""Rolling walk-forward analysis (research brief §2.3).

Per fold: grid-search the strategy on the in-sample (IS) window, choose params
by plateau-center (neighborhood-median objective, falling back to the raw best
when too few candidates), then evaluate those params out-of-sample (OOS). The
OOS backtest runs on IS+OOS bars so indicators are warm, but only the OOS
region is scored (equity and trades sliced to the OOS window). OOS equity
segments are stitched into one continuous curve for aggregate metrics.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core.backtest.metrics import compute_metrics
from core.constants import RESULTS_DIR
from core.optimize.gates import select_plateau_center
from core.optimize.search import (MIN_BARS, _apply_window, _load_cached,
                                  _run_combo, evaluate_task, param_grid,
                                  resolve_strategy)

log = logging.getLogger(__name__)

#: minimum surviving IS candidates for plateau-center selection (else raw best)
MIN_PLATEAU_ROWS = 5


@dataclass
class WalkForwardSpec:
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1d"
    strategy: str = ""                # registry NAME or 'module.path:ClassName'
    is_days: int = 365
    oos_days: int = 90
    step_days: int = 90
    objective: str = "sharpe"         # metric column used for IS selection
    min_trades_is: int = 20
    max_combos: int = 150
    since: str = "2019-01-01"
    holdout_days: int = 0
    cost_multiplier: float = 1.0
    initial_capital: float = 10_000.0
    seed: int = 42


@dataclass
class WalkForwardResult:
    spec: WalkForwardSpec
    folds: list[dict]                       # per fold: windows, params, is/oos metrics
    stitched_oos_equity: pd.Series
    stitched_metrics: dict
    wfe: float                              # stitched OOS CAGR / mean IS CAGR
    pct_profitable_folds: float
    param_stability: float                  # fraction of folds choosing the modal params
    n_skipped_folds: int = 0

    def to_dict(self) -> dict:
        eq = self.stitched_oos_equity
        return {
            "spec": asdict(self.spec),
            "folds": self.folds,
            "stitched_metrics": self.stitched_metrics,
            "wfe": self.wfe,
            "pct_profitable_folds": self.pct_profitable_folds,
            "param_stability": self.param_stability,
            "n_skipped_folds": self.n_skipped_folds,
            "stitched_oos_equity": {
                "timestamp": [ts.isoformat() for ts in eq.index],
                "equity": [float(v) for v in eq.to_numpy()],
            },
        }

    def save(self, path: str | Path | None = None) -> Path:
        if path is None:
            strat = self.spec.strategy.replace(":", "-").replace(".", "-") or "strategy"
            sym = self.spec.symbol.replace("/", "_")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = RESULTS_DIR / f"wfa_{strat}_{sym}_{self.spec.timeframe}_{stamp}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        log.info("walk-forward result saved to %s", path)
        return path


def _fold_windows(t0: pd.Timestamp, t_end: pd.Timestamp, spec: WalkForwardSpec
                  ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Rolling (is_start, is_end==oos_start, oos_end) triples; full OOS only."""
    windows = []
    k = 0
    while True:
        is_start = t0 + pd.Timedelta(days=k * spec.step_days)
        is_end = is_start + pd.Timedelta(days=spec.is_days)
        oos_end = is_end + pd.Timedelta(days=spec.oos_days)
        if oos_end > t_end:
            break
        windows.append((is_start, is_end, oos_end))
        k += 1
    return windows


def _base_task(spec: WalkForwardSpec, params_json: str) -> dict:
    return {
        "exchange": spec.exchange,
        "symbol": spec.symbol,
        "timeframe": spec.timeframe,
        "strategy": spec.strategy,
        "params": params_json,
        "since": spec.since,
        "initial_capital": spec.initial_capital,
        "cost_multiplier": spec.cost_multiplier,
    }


def _pick_params(is_df: pd.DataFrame, spec: WalkForwardSpec) -> pd.Series | None:
    """Plateau-center pick on IS results; raw best fallback; None if unusable."""
    valid = is_df[is_df["error"].isna()].copy()
    if valid.empty:
        return None
    valid[spec.objective] = pd.to_numeric(valid[spec.objective], errors="coerce")
    valid = valid[valid[spec.objective].notna()]
    if valid.empty:
        return None
    filtered = valid[valid["n_trades"] >= spec.min_trades_is]
    if filtered.empty:
        filtered = valid  # nothing meets min trades: pick least-bad anyway
    if len(filtered) >= MIN_PLATEAU_ROWS:
        try:
            return select_plateau_center(filtered, spec.strategy, spec.symbol,
                                         spec.timeframe, metric=spec.objective)
        except ValueError:
            pass
    return filtered.loc[filtered[spec.objective].idxmax()]


def run_walkforward(spec: WalkForwardSpec, n_workers: int | None = None,
                    progress_cb: Callable[[int, int], None] | None = None
                    ) -> WalkForwardResult:
    """Run rolling walk-forward. ``n_workers=None`` or ``0`` -> serial
    in-process; >= 2 -> ProcessPoolExecutor for the IS grids (workers reload
    OHLCV from the parquet cache)."""
    cls = resolve_strategy(spec.strategy)
    df = _load_cached(spec.exchange, spec.symbol, spec.timeframe, spec.since)
    df = _apply_window(df, None, spec.holdout_days, None, None)
    if len(df) < MIN_BARS:
        raise ValueError(f"only {len(df)} bars after holdout trim")

    windows = _fold_windows(df.index[0], df.index[-1], spec)
    if not windows:
        raise ValueError("data range too short for a single IS+OOS fold")
    combos = param_grid(cls, spec.max_combos, spec.seed)
    log.info("walk-forward: %d folds x %d combos", len(windows), len(combos))

    # -- IS grid over all folds (tag = fold index) --
    tasks: list[dict] = []
    for fold_i, (is_start, is_end, _oos_end) in enumerate(windows):
        for combo in combos:
            task = _base_task(spec, json.dumps(combo, sort_keys=True))
            task["start"] = is_start.isoformat()
            task["end"] = is_end.isoformat()
            task["tag"] = fold_i
            tasks.append(task)

    total = len(tasks) + len(windows)  # + one OOS run per fold
    done = 0
    rows: list[dict] = []
    if n_workers and n_workers >= 2:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(evaluate_task, t) for t in tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())
                done += 1
                if progress_cb:
                    progress_cb(done, total)
    else:
        for task in tasks:
            rows.append(evaluate_task(task))
            done += 1
            if progress_cb:
                progress_cb(done, total)
    is_results = pd.DataFrame(rows)
    if "error" not in is_results.columns:
        is_results["error"] = None

    # -- per fold: pick params, evaluate OOS with IS warmup --
    folds: list[dict] = []
    oos_segments: list[pd.Series] = []
    oos_trades_parts: list[pd.DataFrame] = []
    n_skipped = 0
    for fold_i, (is_start, is_end, oos_end) in enumerate(windows):
        fold_rows = is_results[is_results["tag"] == fold_i]
        pick = _pick_params(fold_rows, spec)
        done += 1
        if progress_cb:
            progress_cb(done, total)
        if pick is None:
            n_skipped += 1
            log.warning("fold %d skipped: no valid IS results", fold_i)
            continue
        params_json = pick["params"]
        is_metrics = {k: pick[k] for k in pick.index
                      if k not in ("strategy", "symbol", "timeframe", "params",
                                   "tag", "error", "reasons", "passed", "suspicious")}

        task = _base_task(spec, params_json)
        window_df = df[(df.index >= is_start) & (df.index < oos_end)]
        try:
            # full IS prepended as warmup; score only the OOS slice below
            result = _run_combo(window_df, task)
        except Exception as e:
            n_skipped += 1
            log.warning("fold %d OOS backtest failed: %s", fold_i, e)
            continue
        oos_eq = result.equity[result.equity.index >= is_end]
        oos_trades = result.trades[result.trades["exit_time"] >= is_end].reset_index(drop=True)
        if len(oos_eq) < 2:
            n_skipped += 1
            log.warning("fold %d skipped: empty OOS equity", fold_i)
            continue
        oos_metrics = compute_metrics(oos_eq, oos_trades, timeframe=spec.timeframe)

        folds.append({
            "fold": fold_i,
            "is_start": is_start.isoformat(),
            "is_end": is_end.isoformat(),
            "oos_start": is_end.isoformat(),
            "oos_end": oos_end.isoformat(),
            "params": json.loads(params_json),
            "n_is_candidates": int(fold_rows["error"].isna().sum()),
            "is_metrics": {k: _jsonable(v) for k, v in is_metrics.items()},
            "oos_metrics": {k: _jsonable(v) for k, v in oos_metrics.items()},
        })
        oos_segments.append(oos_eq)
        oos_trades_parts.append(oos_trades)

    if not folds:
        raise ValueError("all walk-forward folds failed — check data / strategy")

    # -- stitch OOS equity, rebasing each segment onto the previous end --
    base = float(spec.initial_capital)
    rebased: list[pd.Series] = []
    for seg in oos_segments:
        seg = seg / float(seg.iloc[0]) * base
        rebased.append(seg)
        base = float(seg.iloc[-1])
    stitched = pd.concat(rebased).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    all_oos_trades = pd.concat(oos_trades_parts, ignore_index=True) \
        if oos_trades_parts else pd.DataFrame(columns=["pnl", "ret_pct", "bars_held"])
    stitched_metrics = compute_metrics(stitched, all_oos_trades, timeframe=spec.timeframe)

    is_cagrs = [f["is_metrics"].get("cagr", np.nan) for f in folds]
    mean_is_cagr = float(np.nanmean(np.asarray(is_cagrs, dtype=np.float64)))
    # WFE = OOS / IS annualized return is only meaningful when IS is profitable
    # (brief §2.3: >=0.5 accept). A non-positive IS denominator is undefined:
    # a negative/negative ratio would masquerade as a "strong" WFE for a system
    # that in fact lost money both in- and out-of-sample.
    if np.isfinite(mean_is_cagr) and mean_is_cagr > 1e-9:
        wfe = float(stitched_metrics["cagr"] / mean_is_cagr)
    else:
        wfe = float("nan")

    oos_returns = [f["oos_metrics"]["total_return"] for f in folds]
    pct_profitable = float(np.mean([r > 0 for r in oos_returns]))

    param_keys = [json.dumps(f["params"], sort_keys=True) for f in folds]
    _, mode_count = Counter(param_keys).most_common(1)[0]
    param_stability = mode_count / len(folds)

    return WalkForwardResult(
        spec=spec,
        folds=folds,
        stitched_oos_equity=stitched,
        stitched_metrics={k: _jsonable(v) for k, v in stitched_metrics.items()},
        wfe=wfe,
        pct_profitable_folds=pct_profitable,
        param_stability=param_stability,
        n_skipped_folds=n_skipped,
    )


def _jsonable(v):
    """Coerce numpy scalars to plain python for JSON persistence."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
