"""Round-2 candidate #4 — ETH/BTC relative-value tilt (long-only spot form).

Thesis: the ETH/BTC ratio mean-reverts, so a spot portfolio that tilts between
ETH and BTC as the ratio stretches away from its rolling mean can earn a spread
return that is near-zero-beta to total-crypto direction. Research reality check
(docs/round2-research.md #4): the 2026 rotation regime is shallow (BTC.D high,
ETH/BTC well below altseason levels), so rotations are infrequent and mild --
this must be sized modestly and reported honestly, mean-reversion-led.

Expression (long-only spot, no shorting):
  ratio  = Binance ETHBTC daily close.
  z      = (ratio - rolling_mean(M)) / rolling_std(M)            [trailing, no lookahead]
  tilt state (fade z, hysteresis):
    z >= +z_entry  -> overweight BTC  (w_eth = 0.30)   (ETH rich vs BTC)
    z <= -z_entry  -> overweight ETH  (w_eth = 0.70)   (ETH cheap vs BTC)
    |z| <= z_exit  -> neutral         (w_eth = 0.50)
    otherwise      -> hold the previous state (hysteresis band)
  A 2-asset ETH/BTC spot portfolio, daily, NEXT-OPEN fills (open->open returns,
  weight for a day set from the prior day's close signal), costs charged ONLY on
  allocation CHANGES (rebalance on signal change; drift is free between changes).

Validation: rolling IS/OOS identical in shape to the cross-sectional script --
IS 365d / OOS 90d / step 90d, last 180d held out (excluded), folds restricted to
the post-2021 split. Per fold: grid-search on IS (pick best IS Sharpe by
plateau-neighborhood median, raw-best fallback), evaluate OOS, stitch OOS equity.
Benchmarks on the same stitched OOS timeline: 50/50 daily-rebalanced, BTC-only.

Saves results/ethbtc_tilt_validation.json.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.backtest.metrics import compute_metrics  # noqa: E402
from core.constants import RESULTS_DIR, cost_profile  # noqa: E402
from core.data.fetcher import load_ohlcv  # noqa: E402

# tilt allocation bounds (assignment: [30/70 .. 70/30], w_eth in {0.3,0.5,0.7})
W_NEUTRAL, W_OW_ETH, W_OW_BTC = 0.50, 0.70, 0.30

GRID = {
    "M": [90, 120, 180],
    "z_entry": [1.5, 2.0, 2.5],
    "z_exit": [0.0, 0.5],
}
SINCE = "2019-01-01"
POST_2021 = pd.Timestamp("2021-01-01", tz="UTC")
HOLDOUT_DAYS = 180
IS_DAYS, OOS_DAYS, STEP_DAYS = 365, 90, 90
INITIAL_CAPITAL = 10_000.0
EMPTY_TRADES = pd.DataFrame({"pnl": [], "ret_pct": [], "bars_held": []})

# per-side cost (fee + slippage) for each spot leg; both are Binance majors.
_ce = cost_profile("binance", "ETH/USDT"); COST_E = _ce["fee"] + _ce["slippage"]
_cb = cost_profile("binance", "BTC/USDT"); COST_B = _cb["fee"] + _cb["slippage"]


def load_aligned() -> pd.DataFrame:
    """ETH/BTC ratio close + ETH/USDT & BTC/USDT opens, aligned on common days."""
    ratio = load_ohlcv("binance", "ETH/BTC", "1d", since=SINCE)
    eth = load_ohlcv("binance", "ETH/USDT", "1d", since=SINCE)
    btc = load_ohlcv("binance", "BTC/USDT", "1d", since=SINCE)
    df = pd.DataFrame({
        "ratio_close": ratio["close"],
        "eth_open": eth["open"],
        "btc_open": btc["open"],
    }).dropna()
    return df


def zscore(ratio_close: pd.Series, M: int) -> np.ndarray:
    """Trailing rolling z-score (rows <= i only; population std to match engine)."""
    rmean = ratio_close.rolling(M, min_periods=M).mean()
    rstd = ratio_close.rolling(M, min_periods=M).std(ddof=0)
    z = (ratio_close - rmean) / rstd
    return z.to_numpy()


def target_weights(z: np.ndarray, z_entry: float, z_exit: float) -> np.ndarray:
    """ETH weight decided at each day's close (fade z, hysteresis latch)."""
    n = len(z)
    w = np.full(n, W_NEUTRAL)
    state = W_NEUTRAL
    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            state = W_NEUTRAL
        elif zi >= z_entry:
            state = W_OW_BTC
        elif zi <= -z_entry:
            state = W_OW_ETH
        elif abs(zi) <= z_exit:
            state = W_NEUTRAL
        # else: hold previous state (hysteresis band z_exit < |z| < z_entry)
        w[i] = state
    return w


def simulate(dates, eth_open, btc_open, w_target, capital=INITIAL_CAPITAL):
    """Open->open 2-asset spot portfolio. Weight for day i (open[i]->open[i+1]) is
    ``w_target[i-1]`` (prior close signal, next-open fill). Cost charged only when
    the target state changes. Returns (equity Series over open times, n_rebalances,
    daily portfolio returns Series)."""
    n = len(dates)
    ro_e = eth_open[1:] / eth_open[:-1] - 1.0
    ro_b = btc_open[1:] / btc_open[:-1] - 1.0
    V = capital
    ve, vb = 0.5 * V, 0.5 * V
    prev_w = W_NEUTRAL
    eq_t = [dates[0]]
    eq_v = [V]
    n_rebal = 0
    for i in range(0, n - 1):
        w = w_target[i - 1] if i >= 1 else W_NEUTRAL   # prior close decision
        f0 = ve / V if V > 0 else 0.5
        if w != prev_w:                                # rebalance on state change
            turnover = abs(w - f0)
            V -= V * turnover * (COST_E + COST_B)
            ve, vb = w * V, (1.0 - w) * V
            prev_w = w
            n_rebal += 1
        ve *= (1.0 + ro_e[i])
        vb *= (1.0 + ro_b[i])
        V = ve + vb
        eq_t.append(dates[i + 1])
        eq_v.append(V)
    eq = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t), name="equity")
    return eq, n_rebal, eq.pct_change().dropna()


def simulate_5050(dates, eth_open, btc_open, capital=INITIAL_CAPITAL):
    """50/50 daily-rebalanced benchmark (cost charged on daily drift)."""
    n = len(dates)
    ro_e = eth_open[1:] / eth_open[:-1] - 1.0
    ro_b = btc_open[1:] / btc_open[:-1] - 1.0
    V = capital
    ve, vb = 0.5 * V, 0.5 * V
    eq_v = [V]
    for i in range(0, n - 1):
        f0 = ve / V if V > 0 else 0.5
        V -= V * abs(0.5 - f0) * (COST_E + COST_B)
        ve, vb = 0.5 * V, 0.5 * V
        ve *= (1.0 + ro_e[i]); vb *= (1.0 + ro_b[i]); V = ve + vb
        eq_v.append(V)
    return pd.Series(eq_v, index=pd.DatetimeIndex(dates), name="eq5050")


def simulate_btc(dates, btc_open, capital=INITIAL_CAPITAL):
    ro_b = btc_open[1:] / btc_open[:-1] - 1.0
    eq = capital * np.concatenate([[1.0], np.cumprod(1.0 + ro_b)])
    return pd.Series(eq, index=pd.DatetimeIndex(dates), name="eqbtc")


def metrics_of(equity: pd.Series) -> dict:
    m = compute_metrics(equity, EMPTY_TRADES, timeframe="1d")
    return {k: m[k] for k in ("cagr", "sharpe", "sortino", "calmar",
                              "max_drawdown", "total_return", "ann_volatility")}


def fold_windows(t0, t_end):
    windows = []
    k = 0
    while True:
        is_start = t0 + pd.Timedelta(days=k * STEP_DAYS)
        is_end = is_start + pd.Timedelta(days=IS_DAYS)
        oos_end = is_end + pd.Timedelta(days=OOS_DAYS)
        if oos_end > t_end:
            break
        windows.append((is_start, is_end, oos_end))
        k += 1
    return windows


def run_combo_window(df, w_full, start, end):
    """Slice [start, end) and simulate the tilt; z/weights precomputed on full
    history (warm). Returns (equity, n_rebal, daily_rets)."""
    mask = (df.index >= start) & (df.index < end)
    sub = df[mask]
    if len(sub) < 30:
        return None
    idx = np.where(mask)[0]
    w_sub = w_full[idx]
    return simulate(sub.index.to_numpy(), sub["eth_open"].to_numpy(),
                    sub["btc_open"].to_numpy(), w_sub)


def plateau_pick(is_scores: dict) -> tuple:
    """Pick the combo by neighborhood-median Sharpe over the ordered grid
    (M, z_entry, z_exit); fall back to raw best when a neighborhood is degenerate."""
    Ms, zes, zxs = GRID["M"], GRID["z_entry"], GRID["z_exit"]
    best_key, best_val = None, -np.inf
    for key, sc in is_scores.items():
        M, ze, zx = key
        mi, ei, xi = Ms.index(M), zes.index(ze), zxs.index(zx)
        neigh = []
        for dm in (-1, 0, 1):
            for de in (-1, 0, 1):
                if 0 <= mi + dm < len(Ms) and 0 <= ei + de < len(zes):
                    nk = (Ms[mi + dm], zes[ei + de], zx)
                    if nk in is_scores:
                        neigh.append(is_scores[nk])
        val = float(np.median(neigh)) if neigh else sc
        if val > best_val:
            best_val, best_key = val, key
    return best_key


def rebase_stitch(segments):
    """Rebase each OOS segment onto the previous end; drop the anchor bar of
    folds after the first (identical to core WFA stitching)."""
    base = INITIAL_CAPITAL
    out = []
    for i, seg in enumerate(segments):
        seg = seg / float(seg.iloc[0]) * base
        base = float(seg.iloc[-1])
        out.append(seg if i == 0 else seg.iloc[1:])
    s = pd.concat(out).sort_index()
    return s[~s.index.duplicated(keep="first")]


def main() -> int:
    df = load_aligned()
    print(f"data: {len(df)} aligned days  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"per-side cost ETH={COST_E:.4%} BTC={COST_B:.4%}  rebalance cost/unit turnover={COST_E+COST_B:.4%}")

    # holdout: exclude last 180 days entirely.
    holdout_cutoff = df.index[-1] - pd.Timedelta(days=HOLDOUT_DAYS)
    df_val = df[df.index <= holdout_cutoff]
    print(f"holdout excluded after {holdout_cutoff.date()} ({(df.index > holdout_cutoff).sum()} days held out)")

    # precompute z and weights for every combo on FULL (pre-holdout) history so
    # rolling means are warm at every fold's IS start.
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    z_by_M = {M: zscore(df_val["ratio_close"], M) for M in GRID["M"]}
    w_by_combo = {(c["M"], c["z_entry"], c["z_exit"]):
                  target_weights(z_by_M[c["M"]], c["z_entry"], c["z_exit"])
                  for c in combos}

    # post-2021 split: folds start no earlier than 2021-01-01.
    t0 = max(df_val.index[0], POST_2021)
    windows = fold_windows(t0, df_val.index[-1])
    if not windows:
        raise SystemExit("no folds")
    print(f"folds: {len(windows)} (IS {IS_DAYS}d / OOS {OOS_DAYS}d / step {STEP_DAYS}d, post-2021)")

    folds = []
    tilt_segs, b5050_segs, btc_segs = [], [], []
    n_rebal_total = 0
    for fi, (is_start, is_end, oos_end) in enumerate(windows):
        # -- IS grid --
        is_scores = {}
        for key, w_full in w_by_combo.items():
            res = run_combo_window(df_val, w_full, is_start, is_end)
            if res is None:
                continue
            eq, _, rets = res
            is_scores[key] = metrics_of(eq)["sharpe"]
        if not is_scores:
            continue
        pick = plateau_pick(is_scores)
        # -- OOS with picked combo --
        oos = run_combo_window(df_val, w_by_combo[pick], is_end, oos_end)
        if oos is None:
            continue
        oos_eq, oos_rebal, oos_rets = oos
        n_rebal_total += oos_rebal
        # benchmarks on identical OOS window
        mask = (df_val.index >= is_end) & (df_val.index < oos_end)
        sub = df_val[mask]
        b5050 = simulate_5050(sub.index.to_numpy(), sub["eth_open"].to_numpy(), sub["btc_open"].to_numpy())
        btc = simulate_btc(sub.index.to_numpy(), sub["btc_open"].to_numpy())

        is_eq = run_combo_window(df_val, w_by_combo[pick], is_start, is_end)[0]
        folds.append({
            "fold": fi,
            "is_start": is_start.isoformat(), "is_end": is_end.isoformat(),
            "oos_start": is_end.isoformat(), "oos_end": oos_end.isoformat(),
            "params": {"M": pick[0], "z_entry": pick[1], "z_exit": pick[2]},
            "is_sharpe": round(is_scores[pick], 4),
            "is_metrics": {k: round(v, 4) for k, v in metrics_of(is_eq).items()},
            "oos_metrics": {k: round(v, 4) for k, v in metrics_of(oos_eq).items()},
            "oos_rebalances": oos_rebal,
            "oos_5050_return": round(metrics_of(b5050)["total_return"], 4),
            "oos_btc_return": round(metrics_of(btc)["total_return"], 4),
        })
        tilt_segs.append(oos_eq)
        b5050_segs.append(b5050)
        btc_segs.append(btc)

    if not folds:
        raise SystemExit("all folds failed")

    tilt = rebase_stitch(tilt_segs)
    b5050 = rebase_stitch(b5050_segs)
    btc = rebase_stitch(btc_segs)
    tm, bm, cm = metrics_of(tilt), metrics_of(b5050), metrics_of(btc)

    # WFE (guard non-positive IS denominator, as core WFA does)
    is_cagrs = [f["is_metrics"]["cagr"] for f in folds]
    mean_is_cagr = float(np.nanmean(is_cagrs))
    wfe = float(tm["cagr"] / mean_is_cagr) if mean_is_cagr > 1e-9 else float("nan")
    pct_prof = float(np.mean([f["oos_metrics"]["total_return"] > 0 for f in folds]))
    keys = [json.dumps(f["params"], sort_keys=True) for f in folds]
    from collections import Counter
    param_stability = Counter(keys).most_common(1)[0][1] / len(folds)

    # OOS correlation of tilt vs BTC daily returns (beta check)
    tr = tilt.pct_change().dropna(); br = btc.pct_change().dropna()
    b5r = b5050.pct_change().dropna()
    common = tr.index.intersection(br.index)
    corr_btc = float(np.corrcoef(tr.loc[common], br.loc[common])[0, 1]) if len(common) > 10 else float("nan")

    # ACTIVE (spread) return = tilt - 50/50: this isolates the pure ETH-vs-BTC
    # rotation bet from the always-long market beta both share. The research
    # thesis (near-zero-beta spread alpha) lives or dies here, not in the raw
    # tilt curve which is ~100% long spot by construction.
    ca = tr.index.intersection(b5r.index)
    active = (tr.loc[ca] - b5r.loc[ca])
    active_ann_mean = float(active.mean() * 365)
    active_ann_vol = float(active.std(ddof=1) * np.sqrt(365))
    active_sharpe = active_ann_mean / active_ann_vol if active_ann_vol > 0 else 0.0
    cab = active.index.intersection(br.index)
    active_corr_btc = float(np.corrcoef(active.loc[cab], br.loc[cab])[0, 1]) if len(cab) > 10 else float("nan")

    verdict = _verdict(tm, bm, cm, wfe, pct_prof, corr_btc, active_ann_mean,
                       active_sharpe, n_rebal_total, len(folds))

    out = {
        "description": "ETH/BTC relative-value tilt (long-only spot), rolling WFA "
                       "(IS365/OOS90/step90, holdout 180d, post-2021).",
        "grid": GRID, "bounds": {"w_eth": [W_OW_BTC, W_NEUTRAL, W_OW_ETH]},
        "costs": {"eth_per_side": COST_E, "btc_per_side": COST_B},
        "n_folds": len(folds),
        "stitched_oos": {
            "tilt": {k: round(v, 4) for k, v in tm.items()},
            "bench_5050_rebal": {k: round(v, 4) for k, v in bm.items()},
            "bench_btc_only": {k: round(v, 4) for k, v in cm.items()},
        },
        "wfe": round(wfe, 4) if np.isfinite(wfe) else None,
        "pct_profitable_folds": round(pct_prof, 4),
        "param_stability": round(param_stability, 4),
        "oos_corr_tilt_vs_btc": round(corr_btc, 4) if np.isfinite(corr_btc) else None,
        "active_vs_5050": {
            "ann_mean": round(active_ann_mean, 4),
            "ann_vol": round(active_ann_vol, 4),
            "sharpe": round(active_sharpe, 4),
            "corr_vs_btc": round(active_corr_btc, 4) if np.isfinite(active_corr_btc) else None,
        },
        "total_oos_rebalances": n_rebal_total,
        "avg_rebalances_per_fold": round(n_rebal_total / len(folds), 2),
        "verdict": verdict,
        "folds": folds,
        "stitched_oos_equity": {
            "timestamp": [t.isoformat() for t in tilt.index],
            "tilt": [float(v) for v in tilt.to_numpy()],
        },
    }
    path = RESULTS_DIR / "ethbtc_tilt_validation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== ETH/BTC tilt: stitched OOS ({len(folds)} folds) ===")
    for label, m in (("tilt", tm), ("50/50 rebal", bm), ("BTC only", cm)):
        print(f"  {label:12s} cagr {m['cagr']:+.2%}  sharpe {m['sharpe']:5.2f}  "
              f"mdd {m['max_drawdown']:.2%}  calmar {m['calmar']:5.2f}  totret {m['total_return']:+.2%}")
    print(f"  WFE {wfe:.2f} | profitable folds {pct_prof:.0%} | param stability {param_stability:.0%}")
    print(f"  OOS corr(tilt, BTC) {corr_btc:.2f} | rebalances/fold {n_rebal_total/len(folds):.1f}")
    print(f"  ACTIVE (tilt-5050) ann_mean {active_ann_mean:+.2%} sharpe {active_sharpe:+.2f} "
          f"corr_vs_BTC {active_corr_btc:+.2f}")
    print("\nVERDICT:", verdict)
    print(f"saved: {path}")
    return 0


def _verdict(tm, bm, cm, wfe, pct_prof, corr_btc, active_mean, active_sharpe, n_rebal, n_folds):
    # The only fair test of the diversifier thesis is the ACTIVE return vs 50/50:
    # the raw tilt curve is ~100% long spot, so its P&L and its BTC correlation
    # are dominated by market beta, not by the ETH-vs-BTC rotation.
    if active_mean <= 0:
        head = ("REJECT: the tilt's active return vs a 50/50 hold is NEGATIVE "
                f"({active_mean:+.2%}/yr, active Sharpe {active_sharpe:+.2f}) -- the "
                "rotation bet destroyed value net of costs; you are better off just "
                "holding 50/50")
    elif active_sharpe < 0.3:
        head = (f"WEAK/REJECT: active vs 50/50 is barely positive ({active_mean:+.2%}/yr, "
                f"active Sharpe {active_sharpe:+.2f}) -- inside noise, not a deployable edge")
    else:
        head = (f"MARGINAL PASS: positive active return vs 50/50 ({active_mean:+.2%}/yr, "
                f"active Sharpe {active_sharpe:+.2f})")
    return (f"{head}. As the research warned, rotations are shallow "
            f"(~{n_rebal/n_folds:.1f} rebalances/fold) and the long-only-spot form carries full "
            f"market beta (raw OOS corr vs BTC {corr_btc:.2f}), so this is NOT the near-zero-beta "
            f"diversifier the ratio would be in long/short form. If used at all, size tiny.")


if __name__ == "__main__":
    raise SystemExit(main())
