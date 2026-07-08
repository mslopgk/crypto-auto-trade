"""Candidate #2 — Cross-sectional low-volatility tilt (long-only defensive core).

Standalone PORTFOLIO backtest + rolling walk-forward validation. The single-symbol
Strategy/engine interface does not apply here (this ranks a *cross-section* of
symbols each rebalance), so this is a self-contained script per the research note
in docs/round2-research.md (#2) and the engine note in the task brief.

WHAT IT DOES
------------
Universe = every Binance 1d parquet under data/binance/ (18 symbols; stablecoins
excluded — none present). Point-in-time: a symbol only enters the tradeable
universe after its first cached bar + ``--min-history-days`` (default 60).

Each rebalance (every R days):
  * eligible = listed long enough AND trailing ADV(30d, quote=close*volume) >=
    ``--min-adv`` (default 20e6 USD), computed point-in-time (trailing only);
  * rank eligible ASCENDING by trailing Parkinson high-low vol (window L days);
  * long the bottom-k, equal or inverse-vol weighted (k "slots"; if fewer than k
    are eligible the empty slots stay in cash — never over-concentrate);
  * optional per-name absolute-momentum cash-out: a selected name whose close is
    below its EMA(100/200) is held in cash instead (dual-momentum defensive; not
    renormalised, so the cash-out genuinely de-risks).

Execution model (strict no-lookahead, matching core/backtest/engine.py):
  * every signal (Parkinson vol, ADV, EMA, universe membership) at decision day t
    uses ONLY rows <= t;
  * a rebalance decided at the CLOSE of day t is FILLED at the OPEN of day t+1;
  * per-name per-side costs come from core.constants.cost_profile (fee+slippage);
  * positions are carried (buy-and-hold drift) between rebalances; turnover tracked.

VALIDATION
----------
Rolling IS(365d)/OOS(90d) step 90d, exactly like the project WFA. Per fold the
config is chosen by IS Sharpe over a small grid (no plateau logic needed given
the tiny grid, per spec). OOS equity segments are stitched (rebased) into one
curve; 365d annualisation throughout; holdout tail (default 180d) EXCLUDED
entirely. Also reports the post-2021-only slice separately (the low-vol anomaly's
sign flipped pre-2021 — research). Reports correlation of stitched OOS daily
returns vs (a) BTC buy&hold and (b) the TSMOM sleeve's stitched OOS equity.

Diversification bar (task): corr > 0.7 vs TSMOM OR stitched OOS Sharpe < 0.5 =>
fails the bar. Reporting a failure honestly is a valid result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.metrics import compute_metrics  # noqa: E402
from core.constants import DATA_DIR, DAYS_PER_YEAR, RESULTS_DIR, cost_profile  # noqa: E402
from core.indicators import ema as ema_indicator  # noqa: E402

_LN2_4 = 4.0 * math.log(2.0)

# Stablecoins never belong in a vol-ranked risk basket (defensive: none of the
# 18 cached USDT majors are stables, but keep the guard so a future cache is safe).
STABLES = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDD", "USDP", "GUSD"}
# Non-tradeable / non-USD-quoted cache entries to keep out of the USD cross-section:
# ETH/BTC is a BTC-quoted ratio pair (ADV in BTC, ~0.03 price scale); TEST is a
# fixture. The tradeable universe is the 18 USDT-quoted majors.
EXCLUDE_BASES = {"TEST"}
QUOTE = "USDT"

# Default search grid (docs/round2-research.md #2). 4*2*3*2*3 = 144 configs.
L_GRID = [20, 30, 45, 60]
R_GRID = [7, 14]
K_GRID = [3, 4, 5]
WEIGHT_GRID = ["equal", "invvol"]
CASHOUT_GRID = [0, 100, 200]  # 0 = no cash-out; else EMA period


@dataclass(frozen=True)
class Config:
    L: int
    R: int
    k: int
    weight: str
    cashout: int  # 0 or EMA period

    def as_dict(self) -> dict:
        return {"L": self.L, "R": self.R, "k": self.k,
                "weight": self.weight, "cashout": self.cashout}


def build_grid() -> list[Config]:
    return [Config(L, R, k, w, c)
            for L in L_GRID for R in R_GRID for k in K_GRID
            for w in WEIGHT_GRID for c in CASHOUT_GRID]


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
class Panel:
    """Point-in-time aligned market data + precomputed trailing signals.

    All 2D arrays are shape (n_sym, n_days) on a shared daily UTC calendar;
    NaN marks days a symbol was not yet listed. Every signal is a TRAILING
    statistic (value at column j uses only columns <= j) => no lookahead.
    """

    def __init__(self, exchange: str, data_dir: Path):
        self.exchange = exchange
        paths = sorted((data_dir / exchange).glob("*/1d.parquet"))
        frames: dict[str, pd.DataFrame] = {}
        for p in paths:
            sym_dir = p.parent.name              # e.g. BTC_USDT
            parts = sym_dir.split("_")
            base, quote = parts[0], parts[-1]
            if quote != QUOTE or base in STABLES or base in EXCLUDE_BASES:
                continue
            frames[sym_dir.replace("_", "/")] = pd.read_parquet(p)
        if not frames:
            raise FileNotFoundError(f"no 1d parquets under {data_dir / exchange}")
        self.symbols = sorted(frames)
        gmin = min(df.index[0] for df in frames.values())
        gmax = max(df.index[-1] for df in frames.values())
        self.dates = pd.date_range(gmin, gmax, freq="1D", tz="UTC")
        n_sym, n_days = len(self.symbols), len(self.dates)

        self.O = np.full((n_sym, n_days), np.nan)
        self.C = np.full((n_sym, n_days), np.nan)
        self.ADV = np.full((n_sym, n_days), np.nan)
        self.first_valid = np.full(n_sym, n_days, dtype=np.int64)
        self.csum = np.empty(n_sym)               # per-side fee+slippage
        self._pvol: dict[int, np.ndarray] = {}
        self._ema: dict[int, np.ndarray] = {}

        for si, sym in enumerate(self.symbols):
            df = frames[sym].reindex(self.dates)
            o = df["open"].to_numpy(float)
            c = df["close"].to_numpy(float)
            v = df["volume"].to_numpy(float)
            self.O[si] = o
            self.C[si] = c
            # ADV: quote-volume (close*base_volume) over trailing 30d, full window.
            qv = pd.Series(c * v, index=self.dates)
            self.ADV[si] = qv.rolling(30, min_periods=30).mean().to_numpy(float)
            valid = np.where(~np.isnan(c))[0]
            if len(valid):
                self.first_valid[si] = int(valid[0])
            prof = cost_profile(exchange, sym)
            self.csum[si] = prof["fee"] + prof["slippage"]

    def pvol(self, L: int) -> np.ndarray:
        """Trailing L-day Parkinson high-low volatility (daily units)."""
        if L not in self._pvol:
            out = np.full_like(self.C, np.nan)
            # recompute from raw H/L aligned to master calendar
            for si, sym in enumerate(self.symbols):
                p = DATA_DIR / self.exchange / sym.replace("/", "_") / "1d.parquet"
                df = pd.read_parquet(p).reindex(self.dates)
                h = df["high"].to_numpy(float)
                l = df["low"].to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.log(h / l) ** 2 / _LN2_4      # per-day Parkinson var
                var = pd.Series(term, index=self.dates).rolling(
                    L, min_periods=L).mean().to_numpy(float)
                out[si] = np.sqrt(var)
            self._pvol[L] = out
        return self._pvol[L]

    def ema(self, N: int) -> np.ndarray:
        if N not in self._ema:
            out = np.full_like(self.C, np.nan)
            for si in range(len(self.symbols)):
                c = self.C[si]
                fv = self.first_valid[si]
                if fv < len(c):
                    seg = c[fv:]
                    out[si, fv:] = ema_indicator(seg, N)   # SMA-seeded EMA, no lookahead
            self._ema[N] = out
        return self._ema[N]


# --------------------------------------------------------------------------- #
# Portfolio simulation
# --------------------------------------------------------------------------- #
def _target_weights(panel: Panel, pos: int, cfg: Config,
                    min_adv: float, min_history: int) -> np.ndarray:
    """Weight vector (len n_sym, sums <= 1) decided at CLOSE of day `pos`."""
    n_sym = len(panel.symbols)
    w = np.zeros(n_sym)
    pv = panel.pvol(cfg.L)[:, pos]
    adv = panel.ADV[:, pos]
    close = panel.C[:, pos]
    age = pos - panel.first_valid
    eligible = (~np.isnan(close) & ~np.isnan(pv) & ~np.isnan(adv)
                & (adv >= min_adv) & (age >= min_history))
    idx = np.where(eligible)[0]
    if idx.size == 0:
        return w
    order = idx[np.argsort(pv[idx], kind="stable")]   # ascending vol
    sel = order[: cfg.k]
    if cfg.weight == "equal":
        base = np.full(sel.size, 1.0 / cfg.k)          # empty slots -> cash
    else:  # inverse-vol, scaled by fill ratio so empty slots stay in cash
        inv = 1.0 / pv[sel]
        base = (sel.size / cfg.k) * inv / inv.sum()
    w[sel] = base
    if cfg.cashout:
        ema_row = panel.ema(cfg.cashout)[:, pos]
        below = close[sel] < ema_row[sel]              # NaN EMA -> not below (keep)
        below = np.where(np.isnan(ema_row[sel]), False, below)
        w[sel[below]] = 0.0                            # de-risk to cash, no renorm
    return w


def simulate(panel: Panel, p0: int, p1: int, cfg: Config,
             min_adv: float, min_history: int, initial_capital: float):
    """Daily portfolio sim over master-calendar positions [p0, p1] inclusive.

    Returns (equity Series over the window, annualised one-way turnover)."""
    n_sym = len(panel.symbols)
    units = np.zeros(n_sym)
    cash = float(initial_capital)
    pending: np.ndarray | None = None
    eq = np.empty(p1 - p0 + 1)
    turnover = 0.0

    for j, pos in enumerate(range(p0, p1 + 1)):
        # 1) execute the pending target at THIS day's open (t-1 close -> t open)
        if pending is not None:
            op = panel.O[:, pos]
            tradable = ~np.isnan(op)
            mtm = np.where(tradable, units * op, 0.0)
            eq_open = cash + mtm.sum()
            if eq_open > 0:
                tgt_units = np.where(tradable & (pending > 0),
                                     pending * eq_open / np.where(tradable, op, 1.0),
                                     0.0)
                # can't trade a name with no open price today; hold it
                tgt_units = np.where(tradable, tgt_units, units)
                delta = tgt_units - units
                notional = delta * np.where(tradable, op, 0.0)
                cost = np.abs(notional) * panel.csum
                cash -= notional.sum() + cost.sum()
                turnover += np.abs(notional).sum() / eq_open
                units = tgt_units
            pending = None

        # 2) decision at the close of rebalance days
        if (pos - p0) % cfg.R == 0:
            pending = _target_weights(panel, pos, cfg, min_adv, min_history)

        # 3) mark to market at close
        cl = panel.C[:, pos]
        mtm = np.where(np.isnan(cl), 0.0, units * cl)
        eq[j] = cash + mtm.sum()

    equity = pd.Series(eq, index=panel.dates[p0:p1 + 1], name="equity")
    years = max((equity.index[-1] - equity.index[0]).days / DAYS_PER_YEAR, 1e-9)
    ann_turnover = turnover / years
    return equity, ann_turnover


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def _pos_at_or_after(dates: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    return int(dates.searchsorted(ts, side="left"))


def run_walkforward(panel: Panel, grid: list[Config], args) -> dict:
    dates = panel.dates
    t0 = dates[0]
    t_end = dates[-1]
    if args.holdout_days > 0:
        t_end = t_end - pd.Timedelta(days=args.holdout_days)
    # rolling fold windows (identical convention to core/optimize/walkforward.py)
    since = pd.Timestamp(args.since, tz="UTC")
    t0 = max(t0, since)
    windows = []
    k = 0
    while True:
        is_start = t0 + pd.Timedelta(days=k * args.step_days)
        is_end = is_start + pd.Timedelta(days=args.is_days)
        oos_end = is_end + pd.Timedelta(days=args.oos_days)
        if oos_end > t_end:
            break
        windows.append((is_start, is_end, oos_end))
        k += 1
    if not windows:
        raise ValueError("data range too short for one IS+OOS fold")

    folds = []
    oos_segments: list[pd.Series] = []
    seg_turnovers: list[float] = []
    for fi, (is_start, is_end, oos_end) in enumerate(windows):
        ps, pe = _pos_at_or_after(dates, is_start), _pos_at_or_after(dates, is_end)
        oe = _pos_at_or_after(dates, oos_end)
        if pe - ps < 60 or oe - pe < 10:
            continue
        # -- IS: score every config by Sharpe, pick the best --
        best_cfg, best_sharpe = None, -np.inf
        for cfg in grid:
            eq_is, _ = simulate(panel, ps, pe - 1, cfg, args.min_adv,
                                args.min_history_days, args.initial_capital)
            if len(eq_is) < 3:
                continue
            m = compute_metrics(eq_is, _empty_trades(), timeframe="1d")
            s = m["sharpe"]
            if np.isfinite(s) and s > best_sharpe:
                best_sharpe, best_cfg = s, cfg
        if best_cfg is None:
            continue
        is_metrics = compute_metrics(
            simulate(panel, ps, pe - 1, best_cfg, args.min_adv,
                     args.min_history_days, args.initial_capital)[0],
            _empty_trades(), timeframe="1d")
        # -- OOS: evaluate chosen config, start flat at OOS open --
        eq_oos, turn_oos = simulate(panel, pe, oe - 1, best_cfg, args.min_adv,
                                    args.min_history_days, args.initial_capital)
        if len(eq_oos) < 2:
            continue
        oos_metrics = compute_metrics(eq_oos, _empty_trades(), timeframe="1d")
        oos_metrics["ann_turnover"] = turn_oos
        folds.append({
            "fold": fi,
            "is_start": is_start.isoformat(), "is_end": is_end.isoformat(),
            "oos_start": is_end.isoformat(), "oos_end": oos_end.isoformat(),
            "params": best_cfg.as_dict(),
            "is_sharpe": float(is_metrics["sharpe"]),
            "oos_sharpe": float(oos_metrics["sharpe"]),
            "oos_cagr": float(oos_metrics["cagr"]),
            "oos_return": float(oos_metrics["total_return"]),
            "oos_mdd": float(oos_metrics["max_drawdown"]),
            "oos_ann_turnover": float(turn_oos),
        })
        oos_segments.append(eq_oos)
        seg_turnovers.append(turn_oos)

    if not oos_segments:
        raise ValueError("all folds failed")

    # -- stitch (rebase each segment onto previous end; drop anchor bar) --
    base = float(args.initial_capital)
    rebased = []
    for i, seg in enumerate(oos_segments):
        seg = seg / float(seg.iloc[0]) * base
        base = float(seg.iloc[-1])
        rebased.append(seg if i == 0 else seg.iloc[1:])
    stitched = pd.concat(rebased).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]

    return {
        "windows": windows,
        "folds": folds,
        "stitched": stitched,
        "seg_turnovers": seg_turnovers,
    }


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame({"pnl": pd.Series(dtype=float),
                         "ret_pct": pd.Series(dtype=float),
                         "bars_held": pd.Series(dtype=float)})


# --------------------------------------------------------------------------- #
# Correlations & references
# --------------------------------------------------------------------------- #
def load_tsmom_oos(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    se = d.get("stitched_oos_equity")
    if not se:
        return None
    idx = pd.to_datetime(se["timestamp"], utc=True)
    return pd.Series(se["equity"], index=idx, name="tsmom")


def daily_returns(equity: pd.Series) -> pd.Series:
    r = equity.pct_change().dropna()
    r.index = r.index.normalize()
    return r


def corr_against(stitched: pd.Series, other_ret: pd.Series) -> tuple[float, int]:
    a = daily_returns(stitched)
    b = other_ret
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 3:
        return float("nan"), len(j)
    return float(j["a"].corr(j["b"])), len(j)


# --------------------------------------------------------------------------- #
def summarise(equity: pd.Series, turnover: float) -> dict:
    m = compute_metrics(equity, _empty_trades(), timeframe="1d")
    return {
        "sharpe": float(m["sharpe"]), "cagr": float(m["cagr"]),
        "max_drawdown": float(m["max_drawdown"]), "sortino": float(m["sortino"]),
        "calmar": float(m["calmar"]), "total_return": float(m["total_return"]),
        "ann_volatility": float(m["ann_volatility"]), "years": float(m["years"]),
        "ann_turnover": float(turnover),
        "start": equity.index[0].isoformat(), "end": equity.index[-1].isoformat(),
        "n_days": int(len(equity)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--min-adv", type=float, default=20e6,
                    help="min trailing 30d quote ADV in USD (point-in-time)")
    ap.add_argument("--min-history-days", type=int, default=60)
    ap.add_argument("--initial-capital", type=float, default=10_000.0)
    ap.add_argument("--is-days", type=int, default=365)
    ap.add_argument("--oos-days", type=int, default=90)
    ap.add_argument("--step-days", type=int, default=90)
    ap.add_argument("--holdout-days", type=int, default=180)
    ap.add_argument("--post2021", default="2021-01-01",
                    help="split date for the post-2021-only report")
    ap.add_argument("--tsmom-json",
                    default=str(RESULTS_DIR / "wfa" / "binance_BTC_USDT_1d_tsmom.json"))
    ap.add_argument("--btc-symbol", default="BTC/USDT")
    ap.add_argument("--out", default=str(RESULTS_DIR / "xs_lowvol_validation.json"))
    args = ap.parse_args()

    panel = Panel(args.exchange, Path(args.data_dir))
    grid = build_grid()
    print(f"universe: {len(panel.symbols)} symbols {panel.symbols}")
    print(f"calendar: {panel.dates[0].date()} -> {panel.dates[-1].date()}  "
          f"({len(panel.dates)} days) | grid: {len(grid)} configs")

    wf = run_walkforward(panel, grid, args)
    stitched = wf["stitched"]
    folds = wf["folds"]
    mean_turnover = float(np.mean(wf["seg_turnovers"])) if wf["seg_turnovers"] else 0.0

    full = summarise(stitched, mean_turnover)

    # post-2021-only slice of the stitched OOS curve
    cut = pd.Timestamp(args.post2021, tz="UTC")
    post = stitched[stitched.index >= cut]
    post_summary = summarise(post, mean_turnover) if len(post) > 2 else None

    # --- correlations ---
    btc_idx = panel.symbols.index(args.btc_symbol) if args.btc_symbol in panel.symbols else None
    corr_btc, n_btc = float("nan"), 0
    if btc_idx is not None:
        btc_close = pd.Series(panel.C[btc_idx], index=panel.dates).dropna()
        btc_ret = daily_returns(btc_close)
        corr_btc, n_btc = corr_against(stitched, btc_ret)
    tsmom = load_tsmom_oos(Path(args.tsmom_json))
    corr_tsmom, n_tsmom = (corr_against(stitched, daily_returns(tsmom))
                           if tsmom is not None else (float("nan"), 0))

    # diversification bar
    fails = []
    if np.isfinite(corr_tsmom) and corr_tsmom > 0.7:
        fails.append(f"corr vs TSMOM {corr_tsmom:.2f} > 0.70")
    if not (np.isfinite(full["sharpe"]) and full["sharpe"] >= 0.5):
        fails.append(f"stitched OOS Sharpe {full['sharpe']:.2f} < 0.50")
    verdict = "PASS" if not fails else "FAIL: " + "; ".join(fails)

    result = {
        "spec": {
            "exchange": args.exchange, "since": args.since,
            "min_adv": args.min_adv, "min_history_days": args.min_history_days,
            "is_days": args.is_days, "oos_days": args.oos_days,
            "step_days": args.step_days, "holdout_days": args.holdout_days,
            "grid": {"L": L_GRID, "R": R_GRID, "k": K_GRID,
                     "weight": WEIGHT_GRID, "cashout": CASHOUT_GRID},
            "universe": panel.symbols,
            "post2021_split": args.post2021,
        },
        "stitched_oos_full": full,
        "stitched_oos_post2021": post_summary,
        "correlations": {
            "vs_btc_buyhold": {"corr": corr_btc, "n_obs": n_btc},
            "vs_tsmom_oos": {"corr": corr_tsmom, "n_obs": n_tsmom},
        },
        "diversification_bar": {"verdict": verdict, "failures": fails},
        "n_folds": len(folds),
        "folds": folds,
        "stitched_oos_equity": {
            "timestamp": [ts.isoformat() for ts in stitched.index],
            "equity": [float(v) for v in stitched.to_numpy()],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))

    # ------- console summary -------
    print("\n=== per-fold OOS ===")
    print(f"{'fold':>4} {'oos_start':>10} {'IS_shp':>7} {'OOS_shp':>7} "
          f"{'OOS_ret':>8} {'turn':>6}  params")
    for f in folds:
        print(f"{f['fold']:>4} {f['oos_start'][:10]:>10} {f['is_sharpe']:>7.2f} "
              f"{f['oos_sharpe']:>7.2f} {f['oos_return']:>7.1%} "
              f"{f['oos_ann_turnover']:>6.1f}  {json.dumps(f['params'])}")

    def line(tag, s):
        if s is None:
            print(f"{tag}: (empty)")
            return
        print(f"{tag}: sharpe {s['sharpe']:.2f}  cagr {s['cagr']:.2%}  "
              f"mdd {s['max_drawdown']:.2%}  calmar {s['calmar']:.2f}  "
              f"ann_turnover {s['ann_turnover']:.1f}x  "
              f"[{s['start'][:10]}->{s['end'][:10]}, {s['n_days']}d]")

    print(f"\nfolds used: {len(folds)}")
    line("stitched OOS (full)   ", full)
    line("stitched OOS (>=2021) ", post_summary)
    print(f"corr vs BTC buy&hold : {corr_btc:+.3f}  (n={n_btc})")
    print(f"corr vs TSMOM OOS    : {corr_tsmom:+.3f}  (n={n_tsmom})")
    print(f"\nDIVERSIFICATION BAR  : {verdict}")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
