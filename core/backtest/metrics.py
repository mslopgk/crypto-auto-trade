"""Performance metrics. Crypto convention: 365-day annualization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.constants import periods_per_year, DAYS_PER_YEAR


def max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """Returns (mdd as positive fraction, longest drawdown length in bars)."""
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - equity / peak
    mdd = float(np.nanmax(dd)) if len(dd) else 0.0
    # longest time under water
    under = dd > 1e-12
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        if cur > longest:
            longest = cur
    return mdd, longest


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, timeframe: str = "1h",
                    exposure: np.ndarray | None = None) -> dict:
    eq = equity.to_numpy(dtype=np.float64)
    n = len(eq)
    ppy = periods_per_year(timeframe)
    out: dict = {}

    initial = eq[0]
    final = eq[-1]
    total_return = final / initial - 1.0

    # elapsed years from actual timestamps (robust to gaps)
    try:
        elapsed_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    except Exception:
        elapsed_days = n / (ppy / DAYS_PER_YEAR)
    years = max(elapsed_days / DAYS_PER_YEAR, 1e-9)
    cagr = (final / initial) ** (1.0 / years) - 1.0 if final > 0 else -1.0

    rets = np.diff(eq) / eq[:-1]
    rets = np.where(np.isfinite(rets), rets, 0.0)
    mu = float(np.mean(rets)) if len(rets) else 0.0
    sd = float(np.std(rets, ddof=1)) if len(rets) > 2 else 0.0
    sharpe = (mu / sd) * np.sqrt(ppy) if sd > 0 else 0.0

    downside = rets[rets < 0]
    dsd = float(np.std(downside, ddof=1)) if len(downside) > 2 else 0.0
    sortino = (mu / dsd) * np.sqrt(ppy) if dsd > 0 else (0.0 if mu <= 0 else np.inf)

    mdd, dd_bars = max_drawdown(eq)
    calmar = cagr / mdd if mdd > 1e-9 else (0.0 if cagr <= 0 else np.inf)

    ann_vol = sd * np.sqrt(ppy)

    out.update({
        "initial_capital": initial,
        "final_equity": float(final),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino) if np.isfinite(sortino) else 999.0,
        "calmar": float(calmar) if np.isfinite(calmar) else 999.0,
        "max_drawdown": float(mdd),
        "dd_bars": int(dd_bars),
        "ann_volatility": float(ann_vol),
        "years": float(years),
    })

    if exposure is not None and len(exposure):
        out["exposure"] = float(np.mean(exposure))

    nt = len(trades)
    out["n_trades"] = nt
    out["trades_per_year"] = nt / years
    if nt:
        pnl = trades["pnl"].to_numpy(dtype=np.float64)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        out["win_rate"] = float(len(wins) / nt)
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.sum()) if len(losses) else 0.0
        out["profit_factor"] = gross_win / gross_loss if gross_loss > 1e-12 else (999.0 if gross_win > 0 else 0.0)
        out["avg_trade_pnl"] = float(pnl.mean())
        out["avg_trade_ret"] = float(trades["ret_pct"].mean())
        out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        out["best_trade"] = float(pnl.max())
        out["worst_trade"] = float(pnl.min())
        out["avg_bars_held"] = float(np.mean(trades["bars_held"].to_numpy(dtype=np.float64)))
        # expectancy per trade normalized by avg loss (SQN-ish quality)
        r = trades["ret_pct"].to_numpy(dtype=np.float64)
        out["sqn"] = float(np.sqrt(nt) * r.mean() / r.std(ddof=1)) if nt > 2 and r.std(ddof=1) > 0 else 0.0
    else:
        out.update({"win_rate": 0.0, "profit_factor": 0.0, "avg_trade_pnl": 0.0,
                    "avg_trade_ret": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                    "best_trade": 0.0, "worst_trade": 0.0, "avg_bars_held": 0.0, "sqn": 0.0})
    return out
