"""Self-test for the live layer (core/live/* + core/data/stream.py). No network.

Three parts, per the assignment:
  (a) PaperBroker unit checks with hand-computed fills on both sides.
  (b) LiveEngine driven by an injected fake stream feeding ~300 synthetic
      closed 1h bars through an inline SMA-cross Strategy; asserts orders
      placed, position tracked, state json written+reloaded, equity history
      grows, a strategy exception is survived (error-counter path), and
      stop() joins within 2s.
  (c) CcxtBroker offline: markets dict injected (no network) to exercise the
      amount_to_precision path and the min-notional rejection.

Run from the project root:
    python scripts/_selftest_live.py
"""
from __future__ import annotations

import asyncio
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ccxt  # noqa: E402

from core.indicators import sma  # noqa: E402
from core.live.broker import BrokerError, CcxtBroker, Fill, PaperBroker  # noqa: E402
from core.live.engine import LiveCallbacks, LiveEngine  # noqa: E402
from core.live.state import LiveState  # noqa: E402
from core.risk import RiskLimits  # noqa: E402
from core.strategies.base import Strategy  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok  {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def approx(a: float, b: float, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


# ---------------------------------------------------------------------------
# (a) PaperBroker unit checks
# ---------------------------------------------------------------------------

def test_paper_broker() -> None:
    print("[a] PaperBroker fill math")
    px = {"BTC/USDT": 100.0}
    fee, slip = 0.0005, 0.001

    # --- buy by cost ------------------------------------------------------
    b = PaperBroker(10_000.0, fee=fee, slippage=slip,
                    price_source=lambda s: px[s], quote_currency="USDT")
    f = b.market_order("BTC/USDT", "buy", cost=1_000.0)
    fill_px = 100.0 * (1.0 + slip)                       # 100.1
    units = 1_000.0 / (fill_px * (1.0 + fee))
    fee_paid = units * fill_px * fee
    check(approx(f.price, fill_px), f"buy fill price {f.price} == {fill_px}")
    check(approx(f.amount, units), f"buy units {f.amount} == {units}")
    check(approx(f.fee, fee_paid), f"buy fee {f.fee} == {fee_paid}")
    check(approx(f.cost, units * fill_px), "buy Fill.cost is notional ex-fee")
    bal = b.get_balances()
    check(approx(bal["USDT"], 10_000.0 - 1_000.0), "buy debits exactly cost from quote")
    check(approx(bal["BTC"], units), "buy credits base units")

    # --- sell by amount ---------------------------------------------------
    f2 = b.market_order("BTC/USDT", "sell", amount=units)
    sell_px = 100.0 * (1.0 - slip)                       # 99.9
    proceeds = units * sell_px
    sfee = proceeds * fee
    check(approx(f2.price, sell_px), f"sell fill price {f2.price} == {sell_px}")
    check(approx(f2.amount, units), "sell amount == held")
    check(approx(f2.fee, sfee), f"sell fee {f2.fee} == {sfee}")
    bal2 = b.get_balances()
    check(approx(bal2["USDT"], 9_000.0 + proceeds - sfee), "sell credits proceeds-fee")
    check(approx(bal2["BTC"], 0.0), "sell zeroes base")

    # --- buy by amount ----------------------------------------------------
    b2 = PaperBroker(10_000.0, fee=fee, slippage=slip, price_source=lambda s: px[s])
    f3 = b2.market_order("BTC/USDT", "buy", amount=5.0)
    exp_cost = 5.0 * fill_px * (1.0 + fee)
    check(approx(f3.amount, 5.0), "buy-by-amount credits exact base amount")
    check(approx(b2.get_balances()["USDT"], 10_000.0 - exp_cost),
          "buy-by-amount debits amount*fill_px*(1+fee)")

    # --- error paths ------------------------------------------------------
    def raises(fn, exc=Exception) -> bool:
        try:
            fn()
            return False
        except exc:
            return True

    check(raises(lambda: b2.market_order("BTC/USDT", "buy", cost=1e12), ValueError),
          "insufficient quote raises")
    check(raises(lambda: b2.market_order("BTC/USDT", "sell", amount=1e9), ValueError),
          "insufficient base raises")
    check(raises(lambda: b2.market_order("BTC/USDT", "hold", cost=1.0), ValueError),
          "invalid side raises")
    check(raises(lambda: b2.market_order("BTC/USDT", "sell"), ValueError),
          "sell without amount raises")
    check(raises(lambda: b2.market_order("BTC/USDT", "buy"), ValueError),
          "buy without cost/amount raises")
    nps = PaperBroker(100.0, price_source=None)
    check(raises(lambda: nps.get_last_price("BTC/USDT"), BrokerError),
          "no price_source raises BrokerError")


# ---------------------------------------------------------------------------
# (b) LiveEngine with injected fake stream
# ---------------------------------------------------------------------------

POISON_LEN = 150  # strategy throws exactly once, on the bar making buffer len==150


class SMACross(Strategy):
    NAME = "selftest_sma"
    TIMEFRAMES = ("1h",)
    DEFAULTS = {"fast": 10, "slow": 30}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == POISON_LEN:
            raise RuntimeError("injected strategy failure")
        c = df["close"].to_numpy(dtype=np.float64)
        f = sma(c, int(self.params["fast"]))
        s = sma(c, int(self.params["slow"]))
        return np.where(np.isnan(f) | np.isnan(s), 0.0,
                        np.where(f > s, 1.0, 0.0))


def _make_bars(n: int = 300) -> list[dict]:
    t0 = 1_700_000_000_000  # arbitrary ms epoch, aligned enough for the test
    step = 3_600_000
    bars = []
    prev_c = 100.0
    for i in range(n):
        c = 100.0 + 30.0 * math.sin(i / 15.0)  # ~94-bar cycle -> multiple crosses
        o = prev_c
        h = max(o, c) * 1.001
        low = min(o, c) * 0.999
        bars.append({"ts": t0 + i * step, "o": o, "h": h, "l": low,
                     "c": c, "v": 50.0})
        prev_c = c
    return bars


def test_live_engine() -> None:
    print("[b] LiveEngine with injected fake stream")
    bars = _make_bars(300)
    symbol = "BTC/USDT"
    prices = {symbol: bars[0]["c"]}
    broker = PaperBroker(10_000.0, fee=0.0005, slippage=0.001,
                         price_source=lambda s: prices[s], quote_currency="USDT")

    logs: list[str] = []
    orders: list[dict] = []
    positions: list = []
    equities: list = []
    cb = LiveCallbacks(
        on_log=lambda m: logs.append(m),
        on_order=lambda d: orders.append(d),
        on_position=lambda p: positions.append(p),
        on_equity=lambda ts, eq: equities.append((ts, eq)),
    )

    bars_fed = threading.Event()

    async def fake_stream(exchange_id, symbol, timeframe, *, on_closed_bar,
                          on_tick=None, stop_event=None, poll_only=False,
                          on_log=None):
        for bar in bars:
            if stop_event is not None and stop_event.is_set():
                break
            prices[symbol] = bar["c"]
            await on_closed_bar(bar)
            if on_tick is not None:
                on_tick(bar["c"])
        bars_fed.set()
        if stop_event is not None:
            await stop_event.wait()

    tmp = Path(tempfile.mkdtemp(prefix="live_selftest_")) / "state.json"
    engine = LiveEngine(
        "binance", symbol, "1h", SMACross(), broker,
        risk_limits=None, callbacks=cb, stream_fn=fake_stream,
        state_path=tmp, warmup_bars=50, buffer_cap=1000, bootstrap=False,
    )

    t = threading.Thread(target=engine.run_forever, daemon=True)
    t.start()
    if not bars_fed.wait(timeout=15.0):
        FAILURES.append("stream did not finish feeding bars within 15s")
        engine.stop(); t.join(timeout=3.0)
        return
    # small settle so the last executor bar-handler finishes its save
    time.sleep(0.2)

    state = engine.state
    check(engine.status == "running", f"engine still running after bars ({engine.status})")
    check(len(state.trade_log) >= 2, f"orders placed (trade_log={len(state.trade_log)})")
    sides = {o["side"] for o in state.trade_log}
    check("buy" in sides and "sell" in sides, f"both buy and sell occurred ({sides})")
    check(len(state.equity_history) >= 250,
          f"equity history grew (={len(state.equity_history)})")
    check(any("bar handler error" in m for m in logs),
          "strategy exception surfaced via error-counter path")
    check(engine._errors == 0, f"error counter reset after recovery (={engine._errors})")
    check(len(equities) >= 250, f"on_equity fired per bar (={len(equities)})")
    check(len(orders) == len(state.trade_log), "on_order fired for every fill")

    # state persisted to disk and reloadable
    check(tmp.exists(), "state json written to disk")
    reloaded = LiveState.load(tmp)
    check(len(reloaded.equity_history) >= 250, "reloaded equity history present")
    check(len(reloaded.trade_log) == len(state.trade_log), "reloaded trade log matches")
    check(reloaded.last_bar_ts == state.last_bar_ts, "reloaded last_bar_ts matches")

    # position tracking: net base position must equal broker's base balance
    net_base = broker.get_balances().get("BTC", 0.0)
    pos_amt = state.position["amount"] if state.position else 0.0
    check(approx(pos_amt, net_base, rel=1e-6, abs_=1e-6),
          f"tracked position {pos_amt} == broker base {net_base}")

    # stop() joins within 2s
    t0 = time.time()
    engine.stop()
    t.join(timeout=2.0)
    elapsed = time.time() - t0
    check(not t.is_alive(), "thread joined after stop()")
    check(elapsed < 2.0, f"stop() joined within 2s (took {elapsed:.2f}s)")
    check(engine.status == "stopped", f"status stopped after shutdown ({engine.status})")

    # final state saved on shutdown
    final = LiveState.load(tmp)
    check(final.started_at is not None, "started_at persisted")


# ---------------------------------------------------------------------------
# (c) CcxtBroker offline
# ---------------------------------------------------------------------------

def _inject_markets(broker: CcxtBroker) -> dict:
    market = {
        "id": "BTCUSDT", "symbol": "BTC/USDT", "base": "BTC", "quote": "USDT",
        "spot": True, "active": True, "type": "spot",
        "precision": {"amount": 0.001, "price": 0.01, "cost": None},
        "limits": {"amount": {"min": 0.0001, "max": None},
                   "price": {"min": None, "max": None},
                   "cost": {"min": 5.0, "max": None}},
        "taker": 0.001, "maker": 0.001,
    }
    ex = broker._ex
    ex.markets = {"BTC/USDT": market}
    ex.markets_by_id = {"BTCUSDT": [market]}
    ex.symbols = ["BTC/USDT"]
    broker._markets_loaded = True
    return market


def test_ccxt_broker_offline() -> None:
    print("[c] CcxtBroker offline (injected markets, no network)")
    broker = CcxtBroker("binance", "key", "secret")
    _inject_markets(broker)

    # min-notional rejection on a buy below 5 USDT
    try:
        broker.market_order("BTC/USDT", "buy", cost=1.0)
        check(False, "min-notional buy should raise")
    except ValueError as e:
        check("below min" in str(e), f"min-notional buy rejected ({e})")
    except Exception as e:  # pragma: no cover
        check(False, f"min-notional buy raised wrong type: {type(e).__name__}: {e}")

    # amount_to_precision path on a sell (stub create_order; no network)
    captured: dict = {}

    def fake_create_order(symbol, type_, side, amount, price=None, params=None):
        captured["amount"] = amount
        captured["params"] = params
        a = float(amount)
        return {"id": "test-1", "status": "closed", "filled": a, "amount": a,
                "average": 100.0, "cost": a * 100.0, "timestamp": 1_700_000_000_000,
                "fee": {"cost": a * 100.0 * 0.001, "currency": "USDT"}}

    broker._ex.create_order = fake_create_order
    fill = broker.market_order("BTC/USDT", "sell", amount=0.0016789)
    check(isinstance(fill, Fill), "sell returns a Fill")
    check(approx(fill.amount, 0.001, rel=1e-9, abs_=1e-9),
          f"amount truncated to 0.001 tick ({fill.amount})")
    check(approx(float(captured.get("amount", -1)), 0.001, abs_=1e-9),
          "precision applied before create_order")
    cid_param = broker._client_id_param()
    check(cid_param in (captured.get("params") or {}),
          f"clientOrderId param '{cid_param}' attached")

    # sell that rounds below the tick must raise (never silently place 0);
    # ccxt.amount_to_precision raises InvalidOrder for sub-tick amounts, and
    # the broker's own guard raises ValueError if a venue truncates to 0.
    try:
        broker.market_order("BTC/USDT", "sell", amount=0.0004)
        check(False, "sub-tick sell should raise")
    except (ValueError, ccxt.InvalidOrder) as e:
        check(True, f"sub-tick sell rejected ({type(e).__name__})")


def test_risk_sizing_units() -> None:
    """Regression: engine must hand size_order a stop distance as a FRACTION of
    price (0.05), not an absolute price distance (0.05*price)."""
    print("[d] risk sizing stop-distance units")
    symbol = "BTC/USDT"
    price = 100.0
    prices = {symbol: price}
    broker = PaperBroker(10_000.0, fee=0.0005, slippage=0.001,
                         price_source=lambda s: prices[s])
    tmp = Path(tempfile.mkdtemp(prefix="live_sizing_")) / "state.json"
    engine = LiveEngine("binance", symbol, "1h", SMACross(), broker,
                        risk_limits=RiskLimits(), bootstrap=False, state_path=tmp)

    captured: dict = {}

    class RecordingRisk:
        def can_open(self, n):  # noqa: ANN001
            return True

        def size_order(self, equity, price, stop_distance, rv):  # noqa: ANN001
            captured["stop_distance"] = stop_distance
            captured["equity"] = equity
            return 0.30 * equity  # arbitrary positive budget

    engine._risk = RecordingRisk()

    # a flat 60-bar buffer at price 100 (no sl_pct/trail -> _DEFAULT_STOP_FRAC)
    idx = pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC", name="timestamp")
    df = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                       "close": 100.0, "volume": 50.0}, index=idx)

    engine._enter_long(df, price, 10_000.0, None)
    sd = captured.get("stop_distance")
    check(sd is not None, "size_order was called")
    # fraction ~0.05; the pre-fix bug would pass 0.05*price = 5.0
    check(sd is not None and sd < 1.0,
          f"stop_distance passed as fraction, not price distance ({sd})")
    check(sd is not None and approx(sd, 0.05, rel=1e-9, abs_=1e-9),
          f"stop_distance == _DEFAULT_STOP_FRAC fraction ({sd})")
    engine._save_state()  # ensure no crash on teardown


def main() -> int:
    test_paper_broker()
    test_live_engine()
    test_ccxt_broker_offline()
    test_risk_sizing_units()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for m in FAILURES:
            print(f"  - {m}")
        return 1
    print("ALL LIVE SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
