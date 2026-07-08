"""Round-2 candidate #5 — realized-vol term-structure overlay: incremental test.

Honest question this answers: on top of the ensemble members that ALREADY carry
per-strategy vol targeting (commit 6d51c36), does gating gross exposure on the
RV term-structure RISK_OFF flag give an *incremental* left-tail (Calmar)
improvement — and does it do so on a PLATEAU of configs across most members, or
only at a lone cherry-picked point?

Method (deliberately conservative / no re-fitting of the members):
  * For each of the 8 CURRENT ensemble members, load its stitched WFA OOS equity
    curve (results/wfa/<id>.json), the exact validated OOS record.
  * Compute the daily RV term-structure state from THAT symbol's own DAILY OHLCV
    (core.regime.rv_ratio_state), applied with PREVIOUS-COMPLETED-DAY semantics
    (state of day D governs bars on day D+1 — shift +1 day, then as-of/ffill onto
    the member's equity timestamps; works for both 1d and 4h members).
  * Overlay = scale each bar's return by ``gross_mult`` while RISK_OFF, unchanged
    otherwise. Recompute sharpe / MDD / Calmar with vs without the overlay.
  * Sweep a small grid and report, per config, how many of the 8 members improve
    Calmar. Research bar: is there a PLATEAU (a config whose parameter neighbours
    also pass) that improves Calmar on >= 6 of 8 members?

Saves results/rv_overlay_check.json. Read-only w.r.t. the members — the overlay
never re-optimizes them, so any gain is purely the flag's incremental value.
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
from core.constants import RESULTS_DIR  # noqa: E402
from core.data.fetcher import load_ohlcv  # noqa: E402
from core.regime import RISK_OFF, rv_ratio_state  # noqa: E402

# The 8 current ensemble members (config/ensemble.json SURVIVORS).
MEMBERS = [
    ("binance", "BTC/USDT", "1d", "tsmom"),
    ("binance", "ETH/USDT", "1d", "tsmom"),
    ("binance", "ADA/USDT", "1d", "tsmom"),
    ("upbit", "BTC/KRW", "1d", "tsmom"),
    ("upbit", "ETH/KRW", "1d", "tsmom"),
    ("upbit", "SOL/KRW", "1d", "tsmom"),
    ("upbit", "BTC/KRW", "4h", "ema_cross"),
    ("upbit", "ETH/KRW", "4h", "ema_cross"),
]

# Small grid (research §5 grid, trimmed to keep it a *small* honest sweep).
GRID = {
    "short_d": [5, 10],
    "long_d": [30, 60],
    "risk_off": [1.15, 1.25, 1.35],
    "ema_smooth": [2, 3, 5],
    "gross_mult": [0.0, 0.5],   # RISK_OFF -> flat / half size
}
RANGING_FIXED = 0.8    # ranging threshold: irrelevant here (we only scale RISK_OFF)
HYSTERESIS = 0.05
IMPROVE_EPS = 1e-4     # min Calmar gain to count as an improvement
BAR = 6                # research bar: improve on >= 6 of 8 members
EMPTY_TRADES = pd.DataFrame({"pnl": [], "ret_pct": [], "bars_held": []})


def member_id(exchange: str, symbol: str, timeframe: str, strategy: str) -> str:
    return f"{exchange}_{symbol.replace('/', '_')}_{timeframe}_{strategy}"


def load_oos_equity(mid: str) -> pd.Series:
    d = json.loads((RESULTS_DIR / "wfa" / f"{mid}.json").read_text(encoding="utf-8"))
    eq = d["stitched_oos_equity"]
    idx = pd.DatetimeIndex(eq["timestamp"])
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return pd.Series(eq["equity"], index=idx, name=mid)


def daily_state_series(daily_df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Daily RV state, timestamped at when it becomes ACTIONABLE (next-day open).

    state[i] is decided at close of daily bar i (index i, a bar-open timestamp
    covering [i, i+1)); it is actionable from the next day's open == index + 1d.
    Indexing the series at (index + 1 day) lets a plain ffill reindex honour the
    previous-completed-day rule for any downstream timestamp.
    """
    st = rv_ratio_state(daily_df, short_d=cfg["short_d"], long_d=cfg["long_d"],
                        risk_off=cfg["risk_off"], ranging=RANGING_FIXED,
                        ema_smooth=cfg["ema_smooth"], hysteresis=HYSTERESIS)
    return pd.Series(st, index=daily_df.index + pd.Timedelta(days=1), name="state")


def overlay_returns(equity: pd.Series, state_act: pd.Series, gross_mult: float):
    """Return (baseline_equity, overlaid_equity, frac_risk_off_bars)."""
    r = equity.pct_change()
    gov = state_act.reindex(equity.index, method="ffill")
    is_risk_off = (gov.to_numpy() == RISK_OFF)
    mult = np.where(is_risk_off, gross_mult, 1.0)
    r_ov = r.to_numpy() * mult
    # rebuild curves from the same starting capital
    base0 = float(equity.iloc[0])
    r0 = np.nan_to_num(r.to_numpy(), nan=0.0)
    r_ov0 = np.nan_to_num(r_ov, nan=0.0)
    base_eq = pd.Series(base0 * np.cumprod(1.0 + r0), index=equity.index)
    ov_eq = pd.Series(base0 * np.cumprod(1.0 + r_ov0), index=equity.index)
    # frac of active (non-warmup) return bars that were scaled
    active = ~np.isnan(r.to_numpy())
    frac = float(np.mean(is_risk_off[active])) if active.any() else 0.0
    return base_eq, ov_eq, frac


def calmar_sharpe_mdd(equity: pd.Series, timeframe: str) -> dict:
    m = compute_metrics(equity, EMPTY_TRADES, timeframe=timeframe)
    return {"calmar": m["calmar"], "sharpe": m["sharpe"],
            "mdd": m["max_drawdown"], "cagr": m["cagr"]}


def main() -> int:
    # Preload member equity + daily OHLCV once.
    members = []
    for exchange, symbol, timeframe, strategy in MEMBERS:
        mid = member_id(exchange, symbol, timeframe, strategy)
        equity = load_oos_equity(mid)
        daily = load_ohlcv(exchange, symbol, "1d", refresh=False)
        members.append({"id": mid, "exchange": exchange, "symbol": symbol,
                        "timeframe": timeframe, "equity": equity, "daily": daily})
        base = calmar_sharpe_mdd(equity.pct_change().pipe(
            lambda r: pd.Series(float(equity.iloc[0]) * np.cumprod(1 + np.nan_to_num(r.to_numpy())),
                                index=equity.index)), timeframe)
        print(f"  {mid:38s} bars={len(equity):5d} baseline calmar={base['calmar']:.2f} "
              f"sharpe={base['sharpe']:.2f} mdd={base['mdd']:.2%}")

    keys = list(GRID)
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]
    configs = [c for c in configs if c["long_d"] > c["short_d"]]

    results = []
    for cfg in configs:
        per_member = []
        n_improved = 0
        for m in members:
            state_act = daily_state_series(m["daily"], cfg)
            base_eq, ov_eq, frac = overlay_returns(m["equity"], state_act, cfg["gross_mult"])
            b = calmar_sharpe_mdd(base_eq, m["timeframe"])
            o = calmar_sharpe_mdd(ov_eq, m["timeframe"])
            d_calmar = o["calmar"] - b["calmar"]
            improved = d_calmar > IMPROVE_EPS
            n_improved += int(improved)
            per_member.append({
                "id": m["id"], "frac_risk_off": round(frac, 4),
                "base_calmar": round(b["calmar"], 4), "ov_calmar": round(o["calmar"], 4),
                "d_calmar": round(d_calmar, 4),
                "base_mdd": round(b["mdd"], 4), "ov_mdd": round(o["mdd"], 4),
                "base_sharpe": round(b["sharpe"], 4), "ov_sharpe": round(o["sharpe"], 4),
                "improved": bool(improved),
            })
        results.append({"config": cfg, "n_improved": n_improved,
                        "mean_d_calmar": round(float(np.mean([p["d_calmar"] for p in per_member])), 4),
                        "mean_d_mdd": round(float(np.mean([p["ov_mdd"] - p["base_mdd"] for p in per_member])), 4),
                        "members": per_member})

    # ---- plateau detection --------------------------------------------------
    # A config is a "plateau centre" if it clears the bar (n_improved >= BAR)
    # AND a majority of its 1-step neighbours (vary risk_off OR ema_smooth by one
    # ordered grid step, other params fixed) also clear the bar. A lone peak with
    # no passing neighbours does NOT count (brief §2.3 plateau requirement).
    def cfg_key(c):
        return tuple(c[k] for k in keys)
    by_key = {cfg_key(r["config"]): r for r in results}
    ro_vals, ema_vals = GRID["risk_off"], GRID["ema_smooth"]

    def neighbours(c):
        out = []
        ri, ei = ro_vals.index(c["risk_off"]), ema_vals.index(c["ema_smooth"])
        for dj in (-1, 1):
            if 0 <= ri + dj < len(ro_vals):
                nc = dict(c); nc["risk_off"] = ro_vals[ri + dj]; out.append(nc)
            if 0 <= ei + dj < len(ema_vals):
                nc = dict(c); nc["ema_smooth"] = ema_vals[ei + dj]; out.append(nc)
        return out

    plateau_centres = []
    for r in results:
        if r["n_improved"] < BAR:
            continue
        nbrs = neighbours(r["config"])
        n_pass = sum(1 for nc in nbrs if by_key[cfg_key(nc)]["n_improved"] >= BAR)
        if nbrs and n_pass >= (len(nbrs) + 1) // 2:
            plateau_centres.append({"config": r["config"], "n_improved": r["n_improved"],
                                    "neighbours_total": len(nbrs), "neighbours_passing": n_pass,
                                    "mean_d_calmar": r["mean_d_calmar"]})

    max_improved = max(r["n_improved"] for r in results)
    n_configs_at_bar = sum(1 for r in results if r["n_improved"] >= BAR)
    plateau_exists = len(plateau_centres) > 0

    verdict = _verdict(plateau_exists, plateau_centres, max_improved,
                       n_configs_at_bar, len(configs), results)

    out = {
        "description": "Incremental RV term-structure overlay test on the 8 current "
                       "ensemble members' stitched WFA OOS equity.",
        "bar": f"improve Calmar on >= {BAR} of {len(members)} members on a config PLATEAU",
        "grid": GRID, "ranging_fixed": RANGING_FIXED, "hysteresis": HYSTERESIS,
        "n_configs": len(configs),
        "max_members_improved": max_improved,
        "n_configs_meeting_bar": n_configs_at_bar,
        "plateau_exists": plateau_exists,
        "plateau_centres": plateau_centres,
        "verdict": verdict,
        "all_configs": results,
    }
    path = RESULTS_DIR / "rv_overlay_check.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== RV overlay incremental check ===")
    print(f"configs tested          : {len(configs)}")
    print(f"max members improved    : {max_improved} / {len(members)}")
    print(f"configs meeting >= {BAR}/8 : {n_configs_at_bar}")
    print(f"plateau (>= {BAR}/8 with passing neighbours) exists: {plateau_exists}")
    # show the best few configs
    top = sorted(results, key=lambda r: (r["n_improved"], r["mean_d_calmar"]), reverse=True)[:5]
    print("\ntop configs by (members improved, mean dCalmar):")
    for r in top:
        c = r["config"]
        print(f"  short={c['short_d']} long={c['long_d']} ro={c['risk_off']} "
              f"ema={c['ema_smooth']} gm={c['gross_mult']}  "
              f"improved={r['n_improved']}/8  meanDCalmar={r['mean_d_calmar']:+.3f} "
              f"meanDMdd={r['mean_d_mdd']:+.3%}")
    print("\nVERDICT:", verdict)
    print(f"saved: {path}")
    return 0


def _verdict(plateau_exists, plateau_centres, max_improved, n_at_bar, n_cfg, results):
    if plateau_exists:
        c = plateau_centres[0]["config"]
        return (f"PASS (marginal): a plateau clears the >= {BAR}/8 bar "
                f"(centre short={c['short_d']} long={c['long_d']} ro={c['risk_off']} "
                f"ema={c['ema_smooth']} gm={c['gross_mult']}; {n_at_bar}/{n_cfg} configs meet "
                f"the bar). The overlay adds incremental left-tail reduction beyond the "
                f"members' existing vol targeting. Size it as a small tail overlay, not alpha.")
    if max_improved >= BAR:
        return (f"WEAK/REJECT: {n_at_bar}/{n_cfg} config(s) reach {BAR}/8 but form no plateau "
                f"(no passing neighbours) -- a lone peak, i.e. curve-fit, not a robust edge.")
    return (f"REJECT: no config improves Calmar on >= {BAR}/8 members (best {max_improved}/8). "
            f"On top of the members' existing vol targeting the RV term-structure flag adds "
            f"no incremental left-tail benefit -- it is largely a re-labeled version of the vol "
            f"scaling already in the book. Do not add it as a gross scaler.")


if __name__ == "__main__":
    raise SystemExit(main())
