"""Backtest engine correctness on crafted scenarios with hand-computed fills.

Engine contract under test (core/backtest/engine.py):
- stance[i] decided at close of bar i, executed at open of bar i+1;
- long entry fill = open*(1+slip); the whole budget covers units + entry fee,
  so units = budget / (fill * (1+fee));
- long exit fill = open*(1-slip), fee deducted from proceeds;
- pessimistic intrabar: gap through stop -> fill at open; SL before TP;
- open position at end of data -> closed at the last close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import BacktestConfig, run_backtest

FEE = 0.001
SLIP = 0.0005
CAP = 10_000.0


def base_config(**kw) -> BacktestConfig:
    kw.setdefault("fee", FEE)
    kw.setdefault("slippage", SLIP)
    kw.setdefault("initial_capital", CAP)
    return BacktestConfig(**kw)


def ramp_df(make_df, n: int = 10) -> pd.DataFrame:
    """open[i] = 100+i, close = open+0.5, high = open+1, low = open-1."""
    opens = 100.0 + np.arange(n)
    return make_df(opens, opens + 1.0, opens - 1.0, opens + 0.5)


def flat_df(make_df, n: int = 10) -> pd.DataFrame:
    """Flat bars around 100: open 100, close 100.2, high 101, low 99.5."""
    return make_df([100.0] * n, [101.0] * n, [99.5] * n, [100.2] * n)


def long_units(open_px: float, budget: float = CAP) -> float:
    """Units bought at a long entry: budget = units * fill * (1+fee)."""
    fill = open_px * (1.0 + SLIP)
    return budget / (fill * (1.0 + FEE))


# ------------------------------------------------------------------ 1) entry

def test_entry_timing_and_fill(make_df):
    df = ramp_df(make_df)
    stance = np.zeros(10)
    stance[3:] = 1.0  # stance goes long at the CLOSE of bar 3

    res = run_backtest(df, stance, base_config())

    # entry must be at bar 4 open, not bar 3:
    #   fill  = open[4] * (1+slip)      = 104 * 1.0005      = 104.052
    #   units = 10000 / (fill * 1.001)  = 10000 / 104.156052 = 96.0097535...
    #   cash after entry = 10000 - units*fill - units*fill*fee = 0 exactly
    #   equity[4] = cash + units * close[4] = units * 104.5
    fill = 104.0 * (1.0 + SLIP)
    units = CAP / (fill * (1.0 + FEE))
    assert (res.equity.iloc[:4] == CAP).all()  # still flat through bar 3
    assert res.equity.iloc[4] == pytest.approx(units * 104.5, rel=1e-12)
    assert res.n_trades == 1
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == df.index[4]
    assert tr["entry_price"] == pytest.approx(fill, rel=1e-12)
    assert tr["units"] == pytest.approx(units, rel=1e-12)


# ---------------------------------------------------------- 2) exit on signal

def test_exit_on_signal(make_df):
    df = ramp_df(make_df)
    stance = np.zeros(10)
    stance[3:6] = 1.0  # long from close of bar 3, flat from close of bar 6

    res = run_backtest(df, stance, base_config())

    # entry bar 4:  fill_e = 104 * 1.0005 = 104.052
    #               units  = 10000 / (104.052 * 1.001) = 96.00975353...
    #               cash   = 0
    # exit bar 7:   fill_x = open[7] * (1-slip) = 107 * 0.9995 = 106.9465
    #               proceeds = units * 106.9465
    #               final cash = proceeds * (1 - fee) = units * 106.9465 * 0.999
    units = long_units(104.0)
    final = units * (107.0 * (1.0 - SLIP)) * (1.0 - FEE)
    assert res.n_trades == 1
    tr = res.trades.iloc[0]
    assert tr["exit_time"] == df.index[7]
    assert tr["exit_price"] == pytest.approx(107.0 * (1.0 - SLIP), rel=1e-12)
    assert tr["exit_reason"] == "signal"
    assert tr["bars_held"] == 3
    # entry consumed exactly the initial capital, so pnl = final - initial
    assert tr["pnl"] == pytest.approx(final - CAP, rel=1e-9)
    assert res.equity.iloc[7] == pytest.approx(final, rel=1e-9)
    assert res.equity.iloc[-1] == pytest.approx(final, rel=1e-9)  # flat after


# -------------------------------------------------------------- 3) stop-loss

def test_stop_loss_intrabar_fill(make_df):
    df = flat_df(make_df)
    # bar 6 dips through the stop but opens above it
    df.iloc[6] = [98.0, 98.5, 94.0, 94.5, 1000.0]
    stance = np.zeros(10)
    stance[3:6] = 1.0

    res = run_backtest(df, stance, base_config(sl_pct=0.05))

    # entry bar 4: fill = 100 * 1.0005 = 100.05 ; sl = 100.05 * 0.95 = 95.0475
    # bar 6: low 94 <= 95.0475 <= open 98 -> fill at the stop:
    #        exit = 95.0475 * (1-slip) = 95.0475 * 0.9995 = 95.00000...
    sl = 100.0 * (1.0 + SLIP) * 0.95
    tr = res.trades.iloc[0]
    assert tr["exit_time"] == df.index[6]
    assert tr["exit_reason"] == "stop_loss"
    assert tr["exit_price"] == pytest.approx(sl * (1.0 - SLIP), rel=1e-12)
    units = long_units(100.0)
    assert res.equity.iloc[6] == pytest.approx(units * sl * (1.0 - SLIP) * (1.0 - FEE), rel=1e-9)


def test_stop_loss_gap_down_fills_at_open(make_df):
    df = flat_df(make_df)
    # bar 6 gaps BELOW the stop (open 93 < sl 95.0475) -> pessimistic open fill
    df.iloc[6] = [93.0, 93.5, 92.0, 92.5, 1000.0]
    stance = np.zeros(10)
    stance[3:6] = 1.0

    res = run_backtest(df, stance, base_config(sl_pct=0.05))
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop_loss"
    # exit = open * (1-slip) = 93 * 0.9995 = 92.9535
    assert tr["exit_price"] == pytest.approx(93.0 * (1.0 - SLIP), rel=1e-12)


# ------------------------------------------------------------ 4) take-profit

def test_take_profit_intrabar_fill(make_df):
    df = flat_df(make_df)
    # bar 6 trades through the target: open 103 < tp 105.0525 <= high 106
    df.iloc[6] = [103.0, 106.0, 102.5, 105.5, 1000.0]
    stance = np.zeros(10)
    stance[3:6] = 1.0

    res = run_backtest(df, stance, base_config(tp_pct=0.05))

    # entry fill = 100.05 ; tp = 100.05 * 1.05 = 105.0525
    # exit = tp * (1-slip) = 105.0525 * 0.9995
    tp = 100.0 * (1.0 + SLIP) * 1.05
    tr = res.trades.iloc[0]
    assert tr["exit_time"] == df.index[6]
    assert tr["exit_reason"] == "take_profit"
    assert tr["exit_price"] == pytest.approx(tp * (1.0 - SLIP), rel=1e-12)


def test_take_profit_gap_up_fills_at_open(make_df):
    df = flat_df(make_df)
    # bar 6 gaps ABOVE the target (open 107 > tp 105.0525) -> fill at open
    df.iloc[6] = [107.0, 108.0, 106.5, 107.0, 1000.0]
    stance = np.zeros(10)
    stance[3:6] = 1.0

    res = run_backtest(df, stance, base_config(tp_pct=0.05))
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "take_profit"
    assert tr["exit_price"] == pytest.approx(107.0 * (1.0 - SLIP), rel=1e-12)


# ------------------------------------------------- 5) SL + TP in the same bar

def test_sl_and_tp_same_bar_sl_wins(make_df):
    df = flat_df(make_df)
    # bar 6 touches BOTH: low 94 <= sl 95.0475 and high 106 >= tp 105.0525
    df.iloc[6] = [100.0, 106.0, 94.0, 95.0, 1000.0]
    stance = np.zeros(10)
    stance[3:6] = 1.0

    res = run_backtest(df, stance, base_config(sl_pct=0.05, tp_pct=0.05))
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop_loss"  # pessimistic: SL assumed first
    sl = 100.0 * (1.0 + SLIP) * 0.95
    assert tr["exit_price"] == pytest.approx(sl * (1.0 - SLIP), rel=1e-12)


# ------------------------------------------------------------- 6) ATR trailing

def test_trailing_stop_ratchets_and_exit_price(make_df):
    # Rising-then-falling series crafted so TR = 2 on every bar -> ATR(3) = 2
    # exactly (Wilder seed = mean(2,2,2) = 2, recursion stays at 2).
    # close: +1/bar 100..110 (i=0..10), then -1/bar (109, 108, 107, 106, 105).
    # open = prev close, high = close+1, low = close-1.
    closes = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
              109, 108, 107, 106, 105]
    opens = [100] + closes[:-1]
    df = make_df(opens, [c + 1 for c in closes], [c - 1 for c in closes], closes)

    stance = np.zeros(16)
    stance[4:12] = 1.0  # entry at bar 5 open; no signal exit before bar 13

    res = run_backtest(df, stance, base_config(trail_atr_mult=2.0, atr_period=3))

    # entry bar 5: fill = open[5]*(1+slip) = 104 * 1.0005 = 104.052
    # trailing = peak_high_since_entry - 2*ATR = peak - 4, ratcheting UP only:
    #   bar 5:  peak = high[5]  = 106 -> trail 102 (low 104 safe)
    #   bar 6..10: peak rises 107..111 -> trail 103..107 (lows always +2 above)
    #   bar 11: high 110 < peak 111 -> trail STAYS 107 (never ratchets down);
    #           low 108 > 107, still in
    #   bar 12: high 109 < peak -> trail still 107; low 107 <= 107 -> EXIT
    #           open[12] = 109 > 107 -> fill at the stop:
    #           exit = 107 * (1-slip) = 107 * 0.9995 = 106.9465
    assert res.n_trades == 1
    tr = res.trades.iloc[0]
    assert tr["entry_time"] == df.index[5]
    assert tr["exit_time"] == df.index[12]
    assert tr["exit_reason"] == "trailing"
    assert tr["exit_price"] == pytest.approx(107.0 * (1.0 - SLIP), rel=1e-12)
    # 107 = 111 - 4 comes from the bar-10 peak, proving the stop kept the
    # highest level even after two lower highs (never ratcheted down).
    units = long_units(104.0)
    final = units * 107.0 * (1.0 - SLIP) * (1.0 - FEE)
    assert res.equity.iloc[12] == pytest.approx(final, rel=1e-9)
    assert res.equity.iloc[-1] == pytest.approx(final, rel=1e-9)


# ------------------------------------------------------------ 7) end of data

def test_open_position_closed_at_end_of_data(make_df):
    df = ramp_df(make_df)
    stance = np.zeros(10)
    stance[3:] = 1.0  # never signals an exit

    res = run_backtest(df, stance, base_config())

    # forced close at last close: exit = close[9]*(1-slip) = 109.5 * 0.9995
    units = long_units(104.0)
    exit_px = 109.5 * (1.0 - SLIP)
    final = units * exit_px * (1.0 - FEE)
    assert res.n_trades == 1
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "end_of_data"
    assert tr["exit_time"] == df.index[9]
    assert tr["exit_price"] == pytest.approx(exit_px, rel=1e-12)
    assert res.equity.iloc[-1] == pytest.approx(final, rel=1e-9)


# ------------------------------------------- 8) accounting identity (property)

@pytest.mark.parametrize("allow_short", [False, True])
def test_accounting_identity_random(random_walk_df, allow_short):
    g = np.random.default_rng(123 if allow_short else 321)
    n = len(random_walk_df)
    stance = g.integers(-1, 2, n).astype(np.float64)
    stance[:50] = np.nan  # NaN warmup must be treated as flat
    cfg = base_config(sl_pct=0.03, tp_pct=0.05, trail_atr_mult=2.0,
                      atr_period=14, allow_short=allow_short)

    res = run_backtest(random_walk_df, stance, cfg)

    assert res.n_trades > 10  # scenario actually exercises the engine
    final = res.equity.iloc[-1]
    # initial capital + sum of all trade PnL must equal final equity
    assert final == pytest.approx(CAP + res.trades["pnl"].sum(), rel=1e-6)
    assert np.isfinite(res.equity.to_numpy()).all()


# ------------------------------------------------------------------ 9) no-trade

def test_flat_stance_no_trades(random_walk_df):
    stance = np.zeros(len(random_walk_df))
    res = run_backtest(random_walk_df, stance, base_config())
    assert res.n_trades == 0
    assert (res.equity == CAP).all()
    assert res.metrics["n_trades"] == 0
    assert res.metrics["total_return"] == 0.0


# ------------------------------------------------------------ 10) size_frac_arr

def test_size_frac_half_scales_pnl_and_fees(make_df):
    df = ramp_df(make_df)
    stance = np.zeros(10)
    stance[3:6] = 1.0  # one round trip: entry bar 4, exit bar 7

    full = run_backtest(df, stance, base_config())
    half = run_backtest(df, stance, base_config(),
                        size_frac_arr=np.full(10, 0.5))

    tf, th = full.trades.iloc[0], half.trades.iloc[0]
    # budget halves -> units halve -> proceeds, fees and pnl all halve exactly
    assert th["units"] == pytest.approx(0.5 * tf["units"], rel=1e-12)
    assert th["pnl"] == pytest.approx(0.5 * tf["pnl"], rel=1e-9)
    assert half.equity.iloc[-1] - CAP == pytest.approx(
        0.5 * (full.equity.iloc[-1] - CAP), rel=1e-9)

    # config.size_frac must take the same path as a constant size_frac_arr
    via_cfg = run_backtest(df, stance, base_config(size_frac=0.5))
    assert via_cfg.trades.iloc[0]["pnl"] == pytest.approx(th["pnl"], rel=1e-12)


# ---------------------------------------------------------------- 11) shorts

def test_short_round_trip(make_df):
    opens = [100.0, 100.0, 100.0, 100.0, 100.0, 98.0, 96.0, 90.0, 90.0, 90.0]
    closes = [o - 0.5 for o in opens]
    highs = [o + 1.0 for o in opens]
    lows = [c - 1.0 for c in closes]
    df = make_df(opens, highs, lows, closes)
    stance = np.zeros(10)
    stance[3:6] = -1.0  # short from close of bar 3, cover from close of bar 6

    res = run_backtest(df, stance, base_config(allow_short=True))

    # short entry bar 4: fill_e = 100 * (1-slip) = 99.95
    #   units = budget / fill_e = 10000 / 99.95 = 100.0500250...
    #   entry fee = units * fill_e * fee = 10000 * 0.001 = 10
    #   cash = 10000 + 10000 - 10 = 19990
    # cover bar 7: fill_x = open[7] * (1+slip) = 90 * 1.0005 = 90.045
    #   cost = units * 90.045 ; fee_x = cost * 0.001
    #   pnl = units*99.95 - cost - fee_x - 10 = 10000 - cost*1.001 - 10
    fill_e = 100.0 * (1.0 - SLIP)
    units = CAP / fill_e
    fill_x = 90.0 * (1.0 + SLIP)
    cost = units * fill_x
    pnl = CAP - cost * (1.0 + FEE) - CAP * FEE
    assert res.n_trades == 1
    tr = res.trades.iloc[0]
    assert tr["direction"] == "short"
    assert tr["entry_time"] == df.index[4]
    assert tr["exit_time"] == df.index[7]
    assert tr["entry_price"] == pytest.approx(fill_e, rel=1e-12)
    assert tr["exit_price"] == pytest.approx(fill_x, rel=1e-12)
    assert tr["pnl"] == pytest.approx(pnl, rel=1e-9)
    assert pnl > 0  # price fell 10% -> profitable short
    assert res.equity.iloc[-1] == pytest.approx(CAP + pnl, rel=1e-9)

    # same stance with allow_short=False must never enter
    res_no = run_backtest(df, stance, base_config(allow_short=False))
    assert res_no.n_trades == 0
    assert (res_no.equity == CAP).all()
