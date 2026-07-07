"""Persistence of risk-halt latch and trailing-stop peak across restarts.

Covers the restart-safety bugs in the live layer:
  (a) LiveState save/load round-trips the RiskManager status snapshot;
  (b) RiskManager.restore re-seeds a fresh instance so a latched MDD kill and
      the running peak survive a process restart (no rebasing to current
      equity, no silent re-arm);
  (c) the open-position dict round-trips its trailing-stop peak_price;
  (d) LiveEngine resumes in the halted_mdd status (and rebuilds the risk
      overlay from the persisted peak) instead of unconditionally 'running'.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.live.broker import PaperBroker
from core.live.engine import LiveEngine
from core.live.state import LiveState
from core.risk import RiskLimits, RiskManager
from core.strategies.base import Strategy


def _ts(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=timezone.utc)


class _Flat(Strategy):
    NAME = "persist_flat"
    TIMEFRAMES = ("1h",)
    DEFAULTS: dict = {}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))


# ------------------------------------------------------------ (a) round-trip

def test_engine_state_round_trips_risk_snapshot():
    snap = {
        "equity": 8_500.0, "peak": 10_000.0, "day": "2026-01-02",
        "day_start_equity": 9_900.0, "drawdown": 0.15, "daily_loss": 0.0,
        "soft_dd": False, "soft_daily": False,
        "halted_daily": False, "halted_mdd": True,
    }
    st = LiveState(realized_pnl=12.5, risk_snapshot=snap)
    tmp = Path(tempfile.mkdtemp(prefix="persist_snap_")) / "state.json"
    st.save(tmp)

    loaded = LiveState.load(tmp)
    assert loaded.risk_snapshot == snap
    assert loaded.risk_snapshot["halted_mdd"] is True
    assert loaded.risk_snapshot["peak"] == 10_000.0

    # a state with no risk overlay must still round-trip (None, not missing)
    st2 = LiveState()
    tmp2 = Path(tempfile.mkdtemp(prefix="persist_none_")) / "state.json"
    st2.save(tmp2)
    assert LiveState.load(tmp2).risk_snapshot is None


# -------------------------------------------------- (b) RiskManager.restore

def test_riskmanager_restore_preserves_peak_and_mdd_latch():
    rm = RiskManager(RiskLimits(), 10_000.0)
    rm.update_equity(_ts(1, 0), 10_000.0)
    rm.update_equity(_ts(1, 1), 11_000.0)          # peak -> 11_000
    assert rm.update_equity(_ts(1, 2), 9_300.0) == "halt_mdd"  # >15% off peak
    assert rm.halted is True
    snap = rm.status()
    assert snap["halted_mdd"] is True
    assert snap["peak"] == 11_000.0

    # simulate a process restart: a brand-new manager seeded with the (already
    # drawn-down) current equity would normally re-arm and reset its peak.
    fresh = RiskManager(RiskLimits(), 9_300.0)
    assert fresh.halted is False           # fresh instance starts un-halted
    fresh.restore(snap)

    assert fresh.halted is True            # latch survives the restart
    assert fresh.status()["halted_mdd"] is True
    assert fresh.status()["peak"] == 11_000.0        # true peak preserved
    assert fresh.size_order(9_300.0, 100.0, 0.05, 0.30) == 0.0
    assert fresh.can_open(0) is False

    # and it stays gated until an explicit manual re-arm
    fresh.re_arm()
    assert fresh.halted is False
    assert fresh.can_open(0) is True
    assert fresh.status()["peak"] == 9_300.0         # re_arm rebases the peak


def test_riskmanager_restore_round_trips_daily_halt_and_day():
    rm = RiskManager(RiskLimits(), 10_000.0)
    rm.update_equity(_ts(3, 0), 10_000.0)
    assert rm.update_equity(_ts(3, 1), 9_700.0) == "halt_daily"  # -3% intraday
    snap = rm.status()

    fresh = RiskManager(RiskLimits(), 9_700.0)
    fresh.restore(snap)
    assert fresh.halted is True
    fst = fresh.status()
    assert fst["halted_daily"] is True
    assert fst["day"] == "2026-01-03"
    assert fst["day_start_equity"] == 10_000.0


# ---------------------------------------------------- (c) peak_price on pos

def test_position_dict_round_trips_peak_price():
    pos = {"amount": 1.5, "entry_price": 100.0, "entry_time": 1_700_000_000_000,
           "stance": 1, "peak_price": 132.0}
    st = LiveState(position=pos)
    tmp = Path(tempfile.mkdtemp(prefix="persist_pos_")) / "state.json"
    st.save(tmp)

    loaded = LiveState.load(tmp)
    assert loaded.position is not None
    assert loaded.position["peak_price"] == 132.0
    assert loaded.position["entry_price"] == 100.0


# ------------------------------------------- (d) engine resumes halted_mdd

def test_engine_resumes_halted_mdd_across_restart():
    symbol = "BTC/USDT"
    prices = {symbol: 100.0}
    broker = PaperBroker(9_000.0, fee=0.0005, slippage=0.001,
                         price_source=lambda s: prices[s], quote_currency="USDT")

    # a state file written by a prior session that latched the MDD kill at a
    # 10_000 peak, then restarted with only 9_000 equity remaining
    snap = {
        "equity": 9_000.0, "peak": 10_000.0, "day": "2026-01-05",
        "day_start_equity": 10_000.0, "drawdown": 0.10, "daily_loss": 0.10,
        "soft_dd": False, "soft_daily": False,
        "halted_daily": False, "halted_mdd": True,
    }
    tmp = Path(tempfile.mkdtemp(prefix="persist_engine_")) / "state.json"
    LiveState(risk_snapshot=snap).save(tmp)

    engine = LiveEngine("binance", symbol, "1h", _Flat(), broker,
                        risk_limits=RiskLimits(), bootstrap=False, state_path=tmp)

    # replicate the startup sequence run inside _main (no thread/stream needed)
    engine._load_state()
    engine._restore_status()

    assert engine.status == "halted_mdd"          # NOT unconditionally 'running'

    # the lazily-rebuilt overlay must restore the true peak, not rebase to 9_000
    risk = engine._make_risk(9_000.0)
    assert risk.halted is True
    assert risk.status()["peak"] == 10_000.0
    assert risk.can_open(0) is False

    # a manual re-arm clears the latch and persists the cleared snapshot to disk
    engine._risk = risk
    engine.re_arm()
    assert engine.status == "running"
    assert engine._state.risk_snapshot["halted_mdd"] is False
