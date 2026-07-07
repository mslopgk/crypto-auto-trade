"""Acceptance gates and plateau (parameter-robustness) analysis.

Implements the search-phase filters from docs/research-brief.md §2.3-2.4:
hard metric gates, overfit alarms (Sharpe > 3 / PF > 4 treated as bugs),
mean-trade-vs-cost viability, and neighborhood-median plateau selection
(never pick a lone parameter peak).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: overfit alarm thresholds (research brief §2.4)
OVERFIT_SHARPE = 3.0
OVERFIT_PF = 4.0


def flag_overfit(row) -> bool:
    """True if the row trips an overfit alarm (too good to be real)."""
    sharpe = row.get("sharpe", 0.0) or 0.0
    pf = row.get("profit_factor", 0.0) or 0.0
    return bool(sharpe > OVERFIT_SHARPE or pf > OVERFIT_PF)


def mean_trade_vs_cost(row, round_trip_cost: float) -> bool:
    """True if the average trade return clears 3x the modeled round-trip cost."""
    avg = row.get("avg_trade_ret", 0.0) or 0.0
    return bool(avg > 3.0 * round_trip_cost)


def apply_gates(df: pd.DataFrame, min_trades: int = 30, min_sharpe: float = 0.8,
                max_mdd: float = 0.35, min_pf: float = 1.15,
                min_trades_per_year: float = 6.0,
                return_all: bool = False) -> pd.DataFrame:
    """Filter search results by hard gates.

    Adds columns: ``passed`` (bool), ``reasons`` (';'-joined failure reasons,
    empty when passed) and ``suspicious`` (overfit alarm). Returns only the
    passing rows unless ``return_all=True`` (useful for inspecting rejects).
    """
    out = df.copy()
    reasons: list[str] = []
    for _, row in out.iterrows():
        r: list[str] = []
        err = row.get("error")
        if not (err is None or pd.isna(err)):
            r.append("error")
        else:
            def _num(key: str) -> float:
                try:
                    v = float(row.get(key))
                except (TypeError, ValueError):
                    return float("nan")
                return v if np.isfinite(v) else float("nan")

            if not (_num("n_trades") >= min_trades):
                r.append(f"n_trades<{min_trades}")
            if not (_num("sharpe") >= min_sharpe):
                r.append(f"sharpe<{min_sharpe}")
            if not (_num("max_drawdown") <= max_mdd):
                r.append(f"mdd>{max_mdd}")
            if not (_num("profit_factor") >= min_pf):
                r.append(f"pf<{min_pf}")
            if not (_num("trades_per_year") >= min_trades_per_year):
                r.append(f"trades_per_year<{min_trades_per_year}")
        reasons.append(";".join(r))
    out["reasons"] = reasons
    out["passed"] = [not r for r in reasons]
    out["suspicious"] = [flag_overfit(row) for _, row in out.iterrows()]
    if return_all:
        return out
    return out[out["passed"]].copy()


# -- plateau analysis ----------------------------------------------------------

def _params_of(row) -> dict:
    p = row["params"]
    return json.loads(p) if isinstance(p, str) else dict(p)


def grid_from_results(df: pd.DataFrame) -> dict[str, list]:
    """Reconstruct the (sorted) grid values per parameter from result rows."""
    all_params = [_params_of(row) for _, row in df.iterrows()]
    keys: set = set()
    for p in all_params:
        keys |= set(p)
    space: dict[str, list] = {}
    for k in keys:
        vals = {p[k] for p in all_params if k in p}
        try:
            space[k] = sorted(vals)
        except TypeError:  # mixed / non-orderable values (e.g. strings)
            space[k] = sorted(vals, key=str)
    return space


def _is_one_step_neighbor(p: dict, q: dict, space: dict[str, list]) -> bool:
    """True if q differs from p by exactly one grid step in exactly one param.

    This is the *immediate* (radius-1, single-axis) neighborhood used to build
    the plateau-median score — deliberately narrower than the brief's §2.4
    ±2-step / all-axes perturbation robustness check (see plateau_score), which
    is a separate, stricter alarm.
    """
    if set(p) != set(q):
        return False
    diffs = 0
    for k, pv in p.items():
        qv = q[k]
        if pv == qv:
            continue
        grid = space.get(k, [])
        try:
            step = abs(grid.index(pv) - grid.index(qv))
        except ValueError:
            return False
        if step != 1:
            return False
        diffs += 1
        if diffs > 1:
            return False
    return diffs == 1


def plateau_score(results_df: pd.DataFrame, best_row, param_space: dict[str, list]) -> float:
    """median(one-step-neighbor sharpe) / best sharpe.

    ~1.0 means the peak sits on a plateau; << 1 (or negative) means a fragile
    lone spike. NaN when no neighbors exist or best sharpe is ~0.
    """
    best_params = _params_of(best_row)
    sub = results_df
    for col in ("strategy", "symbol", "timeframe"):
        if col in sub.columns and col in best_row:
            sub = sub[sub[col] == best_row[col]]
    neighbor_sharpes = [
        float(row["sharpe"]) for _, row in sub.iterrows()
        if np.isfinite(row.get("sharpe", np.nan))
        and _is_one_step_neighbor(best_params, _params_of(row), param_space)
    ]
    best_sharpe = float(best_row.get("sharpe", np.nan))
    if not neighbor_sharpes or not np.isfinite(best_sharpe) or abs(best_sharpe) < 1e-9:
        return float("nan")
    return float(np.median(neighbor_sharpes) / best_sharpe)


def select_plateau_center(results_df: pd.DataFrame, strategy: str, symbol: str,
                          timeframe: str, metric: str = "sharpe") -> pd.Series:
    """Pick the best row by *neighborhood-median* metric instead of the raw peak.

    Each candidate is scored by the median of {its own metric} U {metrics of
    all one-grid-step neighbors}; the highest neighborhood-median wins. The
    returned row gains a ``plateau_metric`` field with that score.

    A candidate needs at least ``MIN_NEIGHBORS`` real (present, unfiltered)
    one-step neighbors to be eligible — otherwise a fragile lone spike adjacent
    to a filtered/unsampled region (whose neighborhood collapses to {self} and
    thus scores at its raw peak) could win over a genuinely supported plateau
    (research brief §2.4: never pick a lone parameter peak). If the whole grid
    is too sparse for any candidate to clear the floor, fall back to the raw
    peak metric.
    """
    sub = results_df[(results_df["strategy"] == strategy)
                     & (results_df["symbol"] == symbol)
                     & (results_df["timeframe"] == timeframe)]
    if "error" in sub.columns:
        sub = sub[sub["error"].isna()]
    sub = sub[pd.to_numeric(sub[metric], errors="coerce").notna()]
    if sub.empty:
        raise ValueError(f"no valid rows for {strategy} {symbol} {timeframe}")
    sub = sub.reset_index(drop=True)

    space = grid_from_results(sub)
    params_list = [_params_of(row) for _, row in sub.iterrows()]
    values = sub[metric].to_numpy(dtype=np.float64)

    MIN_NEIGHBORS = 2  # research brief §2.4: never pick a lone peak
    scores = np.full(len(sub), -np.inf)
    n_present = np.zeros(len(sub), dtype=int)
    for i, p in enumerate(params_list):
        neighborhood = [values[i]]
        for j, q in enumerate(params_list):
            if j != i and _is_one_step_neighbor(p, q, space):
                neighborhood.append(values[j])
        n_present[i] = len(neighborhood) - 1
        if n_present[i] >= MIN_NEIGHBORS:
            scores[i] = float(np.median(neighborhood))
    if not np.isfinite(scores).any():
        # Sparse/tiny grid: no candidate has enough real neighbors to judge
        # robustness — fall back to the raw peak metric.
        scores = values.copy()
    best_i = int(np.argmax(scores))
    row = sub.iloc[best_i].copy()
    row["plateau_metric"] = float(scores[best_i])
    return row
