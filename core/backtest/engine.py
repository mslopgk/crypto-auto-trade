"""Backtest engine.

Execution model (strict no-lookahead):
- ``stance[i]`` is the desired position decided at the CLOSE of bar ``i``
  (+1 long, -1 short, 0 flat). It is executed at the OPEN of bar ``i+1``.
- Every fill pays ``fee`` and suffers ``slippage`` (both fractions, per side,
  slippage always adverse).
- Intrabar stop-loss / take-profit / ATR trailing stop are evaluated against
  the bar's high/low with pessimistic assumptions:
  * gap through a stop -> filled at the open, not at the stop price;
  * SL and TP both touched within one bar -> SL is assumed to fill first.
- Equity is marked to market at every bar close.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from numba import njit

from core.constants import periods_per_year
from core.indicators import atr as atr_indicator
from core.backtest.metrics import compute_metrics

EXIT_REASONS = {0: "signal", 1: "stop_loss", 2: "take_profit", 3: "trailing", 4: "end_of_data"}


@dataclass
class BacktestConfig:
    fee: float = 0.0010          # per-side, fraction (0.001 = 0.10%)
    slippage: float = 0.0005     # per-side, fraction, always adverse
    initial_capital: float = 10_000.0
    size_frac: float = 1.0       # fraction of equity deployed per entry
    sl_pct: float = 0.0          # hard stop-loss from entry, 0 disables
    tp_pct: float = 0.0          # take-profit from entry, 0 disables
    trail_atr_mult: float = 0.0  # ATR trailing stop multiplier, 0 disables
    atr_period: int = 14
    allow_short: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict
    stance: pd.Series
    config: BacktestConfig = field(repr=False, default_factory=BacktestConfig)

    @property
    def n_trades(self) -> int:
        return len(self.trades)


@njit(cache=True)
def _simulate(open_, high, low, close, atr_arr, stance, size_frac_arr,
              fee, slip, sl_pct, tp_pct, trail_mult, allow_short, initial_capital):
    n = len(close)
    equity = np.empty(n)
    exposure = np.zeros(n)
    equity[0] = initial_capital

    # trade record buffers
    max_tr = n + 1
    t_entry_i = np.empty(max_tr, dtype=np.int64)
    t_exit_i = np.empty(max_tr, dtype=np.int64)
    t_dir = np.empty(max_tr, dtype=np.int64)
    t_entry_px = np.empty(max_tr)
    t_exit_px = np.empty(max_tr)
    t_units = np.empty(max_tr)
    t_pnl = np.empty(max_tr)
    t_reason = np.empty(max_tr, dtype=np.int64)
    n_tr = 0

    cash = initial_capital
    pos = 0            # -1, 0, +1
    units = 0.0
    entry_px = 0.0
    entry_i = -1
    entry_cost = 0.0   # total fees paid at entry (for pnl accounting)
    peak = 0.0         # highest high since entry (long trailing)
    trough = 0.0       # lowest low since entry (short trailing)
    sl_price = 0.0
    tp_price = 0.0

    for i in range(1, n):
        target = stance[i - 1]
        if np.isnan(target):
            target = 0.0
        tgt = int(target)
        if tgt < 0 and not allow_short:
            tgt = 0

        # --- 1) execute signal change at the open ---
        if pos != 0 and tgt != pos:
            # exit at open
            if pos == 1:
                fill = open_[i] * (1.0 - slip)
                proceeds = units * fill
                fee_paid = proceeds * fee
                cash += proceeds - fee_paid
                pnl = proceeds - fee_paid - units * entry_px - entry_cost
            else:
                fill = open_[i] * (1.0 + slip)
                cost = units * fill
                fee_paid = cost * fee
                cash -= cost + fee_paid
                pnl = units * entry_px - cost - fee_paid - entry_cost
            t_entry_i[n_tr] = entry_i
            t_exit_i[n_tr] = i
            t_dir[n_tr] = pos
            t_entry_px[n_tr] = entry_px
            t_exit_px[n_tr] = fill
            t_units[n_tr] = units
            t_pnl[n_tr] = pnl
            t_reason[n_tr] = 0
            n_tr += 1
            pos = 0
            units = 0.0

        if pos == 0 and tgt != 0:
            # enter at open
            eq_now = cash
            budget = eq_now * size_frac_arr[i]
            if budget > 0.0:
                if tgt == 1:
                    fill = open_[i] * (1.0 + slip)
                    # spend budget on units + fee: units*fill*(1+fee) = budget
                    units = budget / (fill * (1.0 + fee))
                    entry_cost = units * fill * fee
                    cash -= units * fill + entry_cost
                    pos = 1
                else:
                    fill = open_[i] * (1.0 - slip)
                    units = budget / fill
                    entry_cost = units * fill * fee
                    cash += units * fill - entry_cost
                    pos = -1
                entry_px = fill
                entry_i = i
                peak = fill
                trough = fill
                sl_price = 0.0
                tp_price = 0.0
                if pos == 1:
                    if sl_pct > 0.0:
                        sl_price = fill * (1.0 - sl_pct)
                    if tp_pct > 0.0:
                        tp_price = fill * (1.0 + tp_pct)
                else:
                    if sl_pct > 0.0:
                        sl_price = fill * (1.0 + sl_pct)
                    if tp_pct > 0.0:
                        tp_price = fill * (1.0 - tp_pct)

        # --- 2) intrabar stop management ---
        if pos != 0:
            exit_fill = -1.0
            reason = -1
            if pos == 1:
                # trailing stop uses highs up to current bar
                if trail_mult > 0.0 and not np.isnan(atr_arr[i]):
                    if high[i] > peak:
                        peak = high[i]
                    trail = peak - trail_mult * atr_arr[i]
                else:
                    trail = -1.0
                stop_lv = sl_price
                if trail > stop_lv:
                    stop_lv = trail
                    stop_reason = 3 if trail > sl_price else 1
                else:
                    stop_reason = 1
                if sl_price <= 0.0 and trail <= 0.0:
                    stop_lv = 0.0
                # pessimistic: stop before take-profit
                if stop_lv > 0.0 and low[i] <= stop_lv:
                    raw = stop_lv if open_[i] > stop_lv else open_[i]
                    exit_fill = raw * (1.0 - slip)
                    reason = stop_reason
                elif tp_price > 0.0 and high[i] >= tp_price:
                    raw = tp_price if open_[i] < tp_price else open_[i]
                    exit_fill = raw * (1.0 - slip)
                    reason = 2
                if exit_fill > 0.0:
                    proceeds = units * exit_fill
                    fee_paid = proceeds * fee
                    cash += proceeds - fee_paid
                    pnl = proceeds - fee_paid - units * entry_px - entry_cost
            else:
                if trail_mult > 0.0 and not np.isnan(atr_arr[i]):
                    if low[i] < trough:
                        trough = low[i]
                    trail = trough + trail_mult * atr_arr[i]
                else:
                    trail = 1e308
                stop_lv = sl_price if sl_price > 0.0 else 1e308
                if trail < stop_lv:
                    stop_lv = trail
                    stop_reason = 3
                else:
                    stop_reason = 1
                if stop_lv < 1e308 and high[i] >= stop_lv:
                    raw = stop_lv if open_[i] < stop_lv else open_[i]
                    exit_fill = raw * (1.0 + slip)
                    reason = stop_reason
                elif tp_price > 0.0 and low[i] <= tp_price:
                    raw = tp_price if open_[i] > tp_price else open_[i]
                    exit_fill = raw * (1.0 + slip)
                    reason = 2
                if exit_fill > 0.0:
                    cost = units * exit_fill
                    fee_paid = cost * fee
                    cash -= cost + fee_paid
                    pnl = units * entry_px - cost - fee_paid - entry_cost

            if exit_fill > 0.0:
                t_entry_i[n_tr] = entry_i
                t_exit_i[n_tr] = i
                t_dir[n_tr] = pos
                t_entry_px[n_tr] = entry_px
                t_exit_px[n_tr] = exit_fill
                t_units[n_tr] = units
                t_pnl[n_tr] = pnl
                t_reason[n_tr] = reason
                n_tr += 1
                pos = 0
                units = 0.0

        # --- 3) mark to market ---
        if pos == 1:
            equity[i] = cash + units * close[i]
            exposure[i] = 1.0
        elif pos == -1:
            equity[i] = cash - units * close[i]
            exposure[i] = 1.0
        else:
            equity[i] = cash

    # close any open position at the last close
    if pos != 0:
        i = n - 1
        if pos == 1:
            fill = close[i] * (1.0 - slip)
            proceeds = units * fill
            fee_paid = proceeds * fee
            cash += proceeds - fee_paid
            pnl = proceeds - fee_paid - units * entry_px - entry_cost
        else:
            fill = close[i] * (1.0 + slip)
            cost = units * fill
            fee_paid = cost * fee
            cash -= cost + fee_paid
            pnl = units * entry_px - cost - fee_paid - entry_cost
        t_entry_i[n_tr] = entry_i
        t_exit_i[n_tr] = i
        t_dir[n_tr] = pos
        t_entry_px[n_tr] = entry_px
        t_exit_px[n_tr] = fill
        t_units[n_tr] = units
        t_pnl[n_tr] = pnl
        t_reason[n_tr] = 4
        n_tr += 1
        equity[i] = cash

    return (equity, exposure,
            t_entry_i[:n_tr], t_exit_i[:n_tr], t_dir[:n_tr],
            t_entry_px[:n_tr], t_exit_px[:n_tr], t_units[:n_tr],
            t_pnl[:n_tr], t_reason[:n_tr])


def run_backtest(df: pd.DataFrame, stance, config: BacktestConfig | None = None,
                 timeframe: str = "1h", size_frac_arr: np.ndarray | None = None) -> BacktestResult:
    """Run a backtest.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex and columns open/high/low/close/volume.
    stance : array-like of {-1, 0, 1}, same length as df; decided at bar close.
    config : cost/risk configuration.
    timeframe : bar timeframe string, used for metric annualization.
    size_frac_arr : optional per-bar fraction of equity to deploy on entries
        (overrides ``config.size_frac``; used for volatility targeting).
    """
    config = config or BacktestConfig()
    n = len(df)
    if n < 3:
        raise ValueError("not enough bars")

    stance_arr = np.asarray(stance, dtype=np.float64)
    if len(stance_arr) != n:
        raise ValueError("stance length mismatch")

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    if config.trail_atr_mult > 0:
        atr_arr = atr_indicator(high, low, close, config.atr_period)
    else:
        atr_arr = np.zeros(n)

    if size_frac_arr is None:
        sf = np.full(n, float(config.size_frac))
    else:
        sf = np.nan_to_num(np.asarray(size_frac_arr, dtype=np.float64), nan=0.0)
        sf = np.clip(sf, 0.0, 1.0)

    (equity, exposure, e_i, x_i, dirs, e_px, x_px, units, pnl, reason) = _simulate(
        open_, high, low, close, atr_arr, stance_arr, sf,
        float(config.fee), float(config.slippage), float(config.sl_pct),
        float(config.tp_pct), float(config.trail_atr_mult),
        bool(config.allow_short), float(config.initial_capital),
    )

    idx = df.index
    equity_s = pd.Series(equity, index=idx, name="equity")

    notional = np.abs(units) * e_px
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_pct = np.where(notional > 0, pnl / notional, 0.0)
    # NB: fancy-indexing with an empty int array preserves the (tz-aware) index
    # dtype — never substitute a naive DatetimeIndex([]) here or empty trade
    # frames become incomparable with tz-aware timestamps under pandas 3.
    trades = pd.DataFrame({
        "entry_time": idx[e_i],
        "exit_time": idx[x_i],
        "direction": np.where(dirs > 0, "long", "short"),
        "entry_price": e_px,
        "exit_price": x_px,
        "units": units,
        "pnl": pnl,
        "ret_pct": ret_pct,
        "bars_held": x_i - e_i,
        "exit_reason": [EXIT_REASONS[int(r)] for r in reason],
    })

    metrics = compute_metrics(equity_s, trades, timeframe=timeframe, exposure=exposure)
    return BacktestResult(equity=equity_s, trades=trades, metrics=metrics,
                          stance=pd.Series(stance_arr, index=idx, name="stance"),
                          config=config)
