"""Mass parameter search over (strategy, params, symbol, timeframe) combos.

Spawn-safe design: the worker (``evaluate_task``) is a module-level function
that receives a small dict of primitives, loads OHLCV from the parquet cache
only (never the network), and returns a flat result row. Strategy classes are
resolved *at call time* — either by registry NAME or by an explicit
``"module.path:ClassName"`` reference — so the module imports cleanly even
while strategy modules are still being written.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Callable

import pandas as pd

from core.constants import RESULTS_DIR, cost_profile
from core.strategies.base import Strategy

log = logging.getLogger(__name__)

#: minimum bars a window must contain to be worth backtesting
MIN_BARS = 50


@dataclass
class SearchSpec:
    """What to search. ``holdout_days`` bars at the tail of each dataset
    (after ``until`` or the data end) are excluded from the search entirely —
    they are the single-use holdout (research brief §2.3)."""
    exchange: str = "binance"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1d", "4h"])
    strategies: list[str] | None = None       # None = all searchable registry strategies
    max_combos_per_strategy: int = 200        # random subsample (seeded) if grid larger
    since: str = "2019-01-01"
    until: str | None = None
    # Optional hard-trim window bounds applied AFTER until/holdout: [start, end).
    start: str | None = None
    end: str | None = None
    # Optional score cutoff: bars before ``score_start`` are still backtested
    # (indicator warmup) but excluded from the scored metrics (research brief
    # §2.2 — "fetch warmup + slice it off before scoring"). Unlike ``start`` this
    # keeps the warmup prefix loaded rather than trimming it away.
    score_start: str | None = None
    initial_capital: float = 10_000.0
    cost_multiplier: float = 1.0              # 2.0 = stress test at double costs
    holdout_days: int = 0
    seed: int = 42


# -- strategy resolution (call-time, spawn-safe) ------------------------------

def resolve_strategy(name: str) -> type[Strategy]:
    """Resolve a strategy class by registry NAME or ``'module.path:ClassName'``."""
    if ":" in name:
        mod_name, _, cls_name = name.partition(":")
        cls = getattr(import_module(mod_name), cls_name)
        if not (isinstance(cls, type) and issubclass(cls, Strategy)):
            raise TypeError(f"{name} is not a Strategy subclass")
        return cls
    from core.strategies.registry import get_strategy
    return get_strategy(name)


def _spec_strategies(names: list[str] | None) -> dict[str, type[Strategy]]:
    """Map resolvable-name -> class. Default: all registry strategies not
    marked SEARCHABLE=False. Explicit names are always included."""
    if names:
        return {n: resolve_strategy(n) for n in names}
    from core.strategies.registry import all_strategies
    return {n: c for n, c in all_strategies().items()
            if getattr(c, "SEARCHABLE", True)}


def param_grid(cls: type[Strategy], max_combos: int, seed: int = 42) -> list[dict]:
    """Full cartesian grid of ``cls.PARAM_SPACE``; seeded random subsample of
    ``max_combos`` if the grid is larger. Empty space -> single defaults combo."""
    space = dict(cls.PARAM_SPACE)
    if not space:
        return [{}]
    keys = list(space)
    sizes = [len(space[k]) for k in keys]
    total = 1
    for s in sizes:
        total *= s
    if total <= max_combos:
        flat_indices = range(total)
    else:
        rng = random.Random(f"{seed}:{cls.NAME}")
        flat_indices = rng.sample(range(total), max_combos)
    combos: list[dict] = []
    for flat in flat_indices:
        combo, rem = {}, flat
        for k, s in zip(keys, sizes):  # mixed-radix decode, avoids materializing grid
            rem, j = divmod(rem, s)
            combo[k] = space[k][j]
        combos.append(combo)
    return combos


# -- worker (module-level for pickling under spawn) ---------------------------

@functools.lru_cache(maxsize=16)
def _load_cached(exchange: str, symbol: str, timeframe: str, since: str | None) -> pd.DataFrame:
    """Cache-only OHLCV load, memoized per worker process."""
    from core.data.fetcher import cache_path, load_ohlcv
    if not cache_path(exchange, symbol, timeframe).exists():
        raise FileNotFoundError(f"no parquet cache for {exchange} {symbol} {timeframe}")
    return load_ohlcv(exchange, symbol, timeframe, since=since, refresh=False)


def _ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def _apply_window(df: pd.DataFrame, until: str | None, holdout_days: float,
                  start: str | None, end: str | None) -> pd.DataFrame:
    """Trim to [start, end) after removing the ``until`` tail and the holdout."""
    if until:
        df = df[df.index <= _ts(until)]
    if holdout_days and len(df):
        cutoff = df.index[-1] - pd.Timedelta(days=float(holdout_days))
        df = df[df.index <= cutoff]
    if start:
        df = df[df.index >= _ts(start)]
    if end:
        df = df[df.index < _ts(end)]
    return df


def _run_combo(df: pd.DataFrame, task: dict):
    """Instantiate the strategy and backtest it with venue costs x multiplier."""
    from core.backtest.runner import run_strategy_backtest
    cls = resolve_strategy(task["strategy"])
    params = json.loads(task["params"])
    if "timeframe" in cls.DEFAULTS:
        params.setdefault("timeframe", task["timeframe"])
    strategy = cls(**params)
    costs = cost_profile(task["exchange"], task["symbol"])
    mult = float(task.get("cost_multiplier", 1.0))
    overrides = {
        "fee": costs["fee"] * mult,
        "slippage": costs["slippage"] * mult,
        "initial_capital": float(task.get("initial_capital", 10_000.0)),
    }
    return run_strategy_backtest(df, strategy, task["timeframe"],
                                 exchange_id=task["exchange"], symbol=task["symbol"],
                                 overrides=overrides)


def evaluate_task(task: dict) -> dict:
    """Evaluate one combo -> flat result row. Never raises: failures fill 'error'."""
    row: dict = {
        "strategy": task["strategy"],
        "symbol": task["symbol"],
        "timeframe": task["timeframe"],
        "params": task["params"],
    }
    if task.get("tag") is not None:
        row["tag"] = task["tag"]
    try:
        df = _load_cached(task["exchange"], task["symbol"], task["timeframe"],
                          task.get("since"))
        df = _apply_window(df, task.get("until"), task.get("holdout_days", 0),
                           task.get("start"), task.get("end"))
        if len(df) < MIN_BARS:
            raise ValueError(f"only {len(df)} bars in window")
        result = _run_combo(df, task)
        score_start = task.get("score_start")
        if score_start:
            # Warmup prefix was backtested so indicators enter the scored window
            # warm; slice it off and score only [score_start, end) — symmetric
            # with the walk-forward OOS path (research brief §2.2/§2.3).
            from core.backtest.metrics import compute_metrics
            ss = _ts(score_start)
            eq = result.equity[result.equity.index >= ss]
            trades = result.trades[result.trades["exit_time"] >= ss].reset_index(drop=True)
            if len(eq) < 2:
                raise ValueError(f"only {len(eq)} scored bars after warmup slice")
            metrics = compute_metrics(eq, trades, timeframe=task["timeframe"])
        else:
            metrics = result.metrics
        row.update(metrics)
        row["error"] = None
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


# -- driver --------------------------------------------------------------------

def build_tasks(spec: SearchSpec) -> list[dict]:
    """Enumerate all (strategy, params, symbol, timeframe) tasks for a spec."""
    classes = _spec_strategies(spec.strategies)
    # TIMEFRAMES is only a *recommended* set (see strategies/base.py). When the
    # caller names strategies explicitly, honor the requested timeframes as-is;
    # only the default (all-searchable) sweep restricts to each strategy's
    # recommended TIMEFRAMES. Without this, an explicit request like
    # strategies=['ema_cross'], timeframes=['1h'] silently yields zero combos.
    explicit = spec.strategies is not None
    tasks: list[dict] = []
    for name, cls in classes.items():
        if explicit:
            tfs = list(spec.timeframes)
        else:
            tfs = [tf for tf in spec.timeframes if tf in cls.TIMEFRAMES]
        if not tfs:
            log.info("skip %s: none of %s in its TIMEFRAMES", name, spec.timeframes)
            continue
        combos = param_grid(cls, spec.max_combos_per_strategy, spec.seed)
        for symbol in spec.symbols:
            for tf in tfs:
                for combo in combos:
                    tasks.append({
                        "exchange": spec.exchange,
                        "symbol": symbol,
                        "timeframe": tf,
                        "strategy": name,
                        "params": json.dumps(combo, sort_keys=True),
                        "since": spec.since,
                        "until": spec.until,
                        "start": spec.start,
                        "end": spec.end,
                        "score_start": spec.score_start,
                        "holdout_days": spec.holdout_days,
                        "initial_capital": spec.initial_capital,
                        "cost_multiplier": spec.cost_multiplier,
                    })
    return tasks


def run_search(spec: SearchSpec, n_workers: int | None = None,
               out_path: str | Path | None = None,
               progress_cb: Callable[[int, int], None] | None = None,
               run_id: str | None = None) -> pd.DataFrame:
    """Run the full search. Returns one row per combo (params as JSON string,
    all backtest metrics as columns, 'error' filled on failure).

    n_workers: None -> cpu_count-1 processes; 0 -> serial in-process.
    Results are saved as parquet + a sidecar ``.meta.json`` under results/.
    """
    tasks = build_tasks(spec)
    total = len(tasks)
    if total == 0:
        raise ValueError("search spec produced no combos (check strategies/timeframes)")
    log.info("search: %d combos", total)

    rows: list[dict] = []
    if n_workers == 0:
        for done, task in enumerate(tasks, 1):
            rows.append(evaluate_task(task))
            if progress_cb:
                progress_cb(done, total)
    else:
        workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(evaluate_task, t) for t in tasks]
            for done, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if progress_cb:
                    progress_cb(done, total)

    df = pd.DataFrame(rows)
    if "error" not in df.columns:
        df["error"] = None

    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    out = Path(out_path) if out_path else RESULTS_DIR / f"search_{run_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "run_id": run_id,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_combos": total,
        "n_errors": int(df["error"].notna().sum()),
        "spec": asdict(spec),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("search saved to %s (%d rows, %d errors)", out, len(df), meta["n_errors"])
    return df


def load_search_results(path: str | Path) -> pd.DataFrame:
    """Load a saved search parquet; 'params' column decoded back to dicts."""
    df = pd.read_parquet(path)
    df["params"] = [json.loads(p) if isinstance(p, str) else p for p in df["params"]]
    return df
