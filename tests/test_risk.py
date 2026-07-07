"""Risk-overlay tests (core/risk.py) — brief sections 3.3 / 3.4.

- drawdown ladder: soft brake at 10% peak-to-trough, hard kill at 15%;
- daily-loss ladder: soft brake at 1.5% of day-start equity, hard lock at 3%,
  auto-cleared on the UTC-day rollover;
- size_order: MINIMUM of {per-position cap, vol-target, fixed-fractional},
  halved under a soft brake, zero under any hard halt;
- re_arm: clears the hard halt and rebases the ladder.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.risk import RiskManager, RiskLimits


def _ts(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=timezone.utc)


def _rm(equity: float = 10_000.0) -> RiskManager:
    return RiskManager(RiskLimits(), equity)


# ------------------------------------------------------------ drawdown ladder

def test_drawdown_ladder_soft_then_hard():
    rm = _rm()
    assert rm.update_equity(_ts(1, 0), 10_000.0) == "none"
    assert rm.update_equity(_ts(1, 1), 11_000.0) == "none"   # peak -> 11_000

    # 10% off the 11_000 peak (= 9_900); only 1% below the 10_000 day start, so
    # the daily ladder stays quiet and the drawdown soft brake fires cleanly.
    assert rm.update_equity(_ts(1, 2), 9_900.0) == "soft_dd"
    assert rm.status()["soft_dd"] is True
    assert rm.halted is False

    # roll to a new UTC day so the daily ladder rebases to 9_900 and cannot
    # steal the transition; then drop to 15% off peak (= 9_350) -> hard kill.
    assert rm.update_equity(_ts(2, 0), 9_900.0) == "none"
    assert rm.update_equity(_ts(2, 1), 9_350.0) == "halt_mdd"
    assert rm.halted is True
    # a further drop produces no *new* transition (halt is latched/sticky)
    assert rm.update_equity(_ts(2, 2), 9_000.0) == "none"
    assert rm.size_order(9_000.0, 100.0, 0.05, 0.30) == 0.0


def test_drawdown_hard_kill_requires_manual_rearm():
    rm = _rm()
    rm.update_equity(_ts(1, 0), 10_000.0)
    assert rm.update_equity(_ts(1, 1), 8_500.0) == "halt_mdd"  # 15% straight down
    assert rm.halted is True
    # a UTC-day rollover does NOT clear an MDD kill (unlike the daily lock)
    assert rm.update_equity(_ts(2, 0), 8_500.0) == "none"
    assert rm.halted is True

    rm.re_arm()
    assert rm.halted is False
    # peak rebased to current equity -> ladder measures fresh losses only
    assert rm.size_order(8_500.0, 100.0, 0.05, 0.30) > 0.0


# --------------------------------------------------------- daily-loss ladder

def test_daily_loss_ladder_soft_then_hard_then_rollover_reset():
    rm = _rm()
    rm.update_equity(_ts(1, 0), 10_000.0)                       # day start 10_000
    assert rm.update_equity(_ts(1, 1), 9_850.0) == "soft_daily"  # -1.5%
    assert rm.update_equity(_ts(1, 2), 9_700.0) == "halt_daily"  # -3.0%
    assert rm.halted is True
    assert rm.can_open(0) is False

    # UTC-day rollover rebases day-start equity and auto-clears the daily lock
    assert rm.update_equity(_ts(2, 0), 9_700.0) == "none"
    assert rm.halted is False
    st = rm.status()
    assert st["halted_daily"] is False
    assert st["soft_daily"] is False
    assert rm.can_open(0) is True


def test_reset_daily_rebases_without_rollover():
    rm = _rm()
    rm.update_equity(_ts(1, 0), 10_000.0)
    rm.update_equity(_ts(1, 1), 9_700.0)          # hard daily lock
    assert rm.halted is True
    rm.reset_daily(_ts(1, 2))                       # manual intraday rebase
    assert rm.status()["halted_daily"] is False


# --------------------------------------------------------------- sizing stack

def test_size_order_is_minimum_of_stack():
    rm = _rm()
    # per-position cap : 0.30 * 10_000                     = 3_000
    # vol target       : 10_000 * 0.20 / 0.40              = 5_000
    # fixed-fractional : 10_000 * 0.01 / 0.05              = 2_000  <-- binds
    assert rm.size_order(10_000.0, 100.0, 0.05, 0.40) == pytest.approx(2_000.0)

    # tighten the stop so fixed-fractional grows and the per-position cap binds
    # fixed-fractional : 10_000 * 0.01 / 0.01 = 10_000 ; vol = 5_000 ; cap 3_000
    assert rm.size_order(10_000.0, 100.0, 0.01, 0.40) == pytest.approx(3_000.0)

    # no stop and no vol estimate -> only the per-position cap applies
    assert rm.size_order(10_000.0, 100.0, 0.0, 0.0) == pytest.approx(3_000.0)


def test_size_order_halved_under_soft_brake():
    rm = _rm()
    full = rm.size_order(10_000.0, 100.0, 0.05, 0.40)
    rm.update_equity(_ts(1, 0), 10_000.0)
    rm.update_equity(_ts(1, 1), 11_000.0)
    assert rm.update_equity(_ts(1, 2), 9_900.0) == "soft_dd"   # soft brake on
    assert rm.size_order(10_000.0, 100.0, 0.05, 0.40) == pytest.approx(0.5 * full)


def test_size_order_zero_when_halted_and_can_open_false():
    rm = _rm()
    rm.update_equity(_ts(1, 0), 10_000.0)
    rm.update_equity(_ts(1, 1), 8_500.0)          # MDD kill
    assert rm.size_order(10_000.0, 100.0, 0.05, 0.40) == 0.0
    assert rm.can_open(0) is False
    rm.re_arm()
    assert rm.size_order(10_000.0, 100.0, 0.05, 0.40) == pytest.approx(2_000.0)
    assert rm.can_open(0) is True
