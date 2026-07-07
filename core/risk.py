"""Risk overlay: vol-target sizing, per-trade sizing stack, and loss-limit ladder.

Implements docs/research-brief.md sections 3.3 (position sizing stack: compute
all terms, take the MINIMUM) and 3.4 (risk overlay: daily-loss ladder and
peak-to-trough drawdown ladder with soft brake / hard kill).

Independent of any strategy: ``RiskManager`` overrides every strategy's sizing
and drives GUI status. Pure python/numpy, no Qt. State mutations are guarded by
a ``threading.Lock`` so the asyncio worker thread and GUI thread can share one
instance.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

from core.constants import periods_per_year
from core.indicators import realized_vol

log = logging.getLogger(__name__)

# tolerance so transitions fire at exactly the threshold despite float rounding
_EPS = 1e-12

# bases treated as majors; every other base shares one correlated "alt bucket"
_MAJOR_BASES = frozenset({"BTC", "ETH"})


def vol_target_size_frac(close, timeframe: str, target_annual_vol: float = 0.20,
                         vol_period: int = 30, max_frac: float = 1.0) -> np.ndarray:
    """Per-bar volatility-targeting size fraction for backtests.

    ``frac[i] = target_annual_vol / realized_vol[i]`` where realized vol is the
    annualized (365-day based) trailing ``vol_period``-bar log-return vol,
    clipped to ``[0, max_frac]``. NaN warmup (and zero/degenerate vol) -> 0.
    No lookahead: frac[i] uses closes up to and including bar i only.

    Feed the result to ``run_backtest(size_frac_arr=...)``.
    """
    ppy = periods_per_year(timeframe)
    rv = realized_vol(close, period=vol_period, ppy=ppy)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = target_annual_vol / rv
    frac = np.where(np.isfinite(frac) & (frac > 0.0), frac, 0.0)
    return np.clip(frac, 0.0, max_frac)


@dataclass
class RiskLimits:
    """Loss-limit ladder and sizing caps (fractions, not percent)."""
    risk_per_trade_pct: float = 0.01      # fixed-fractional risk per trade
    daily_loss_soft_pct: float = 0.015    # half-size below day-start equity
    daily_loss_hard_pct: float = 0.03     # flatten + lock until next UTC day
    dd_soft_pct: float = 0.10             # peak-to-trough: halve all risk
    dd_hard_pct: float = 0.15             # kill switch, manual re-arm
    max_positions: int = 4
    max_position_frac: float = 0.30       # per-position notional cap vs equity
    target_annual_vol: float = 0.20


@dataclass
class PortfolioLimits:
    """Portfolio-level exposure caps."""
    max_alt_bucket_frac: float = 0.5      # all non-BTC/ETH notional vs equity


def check_portfolio(positions: dict[str, float], equity: float,
                    limits: PortfolioLimits) -> list[str]:
    """Check portfolio-level exposure limits.

    Parameters
    ----------
    positions : mapping symbol (e.g. ``"SOL/USDT"``) -> open notional in quote
        currency (absolute value used).
    equity : current account equity in quote currency.
    limits : portfolio caps.

    Returns a list of human-readable violation strings (empty = OK). All
    non-BTC/ETH symbols share one correlated alt bucket (brief 3.4).
    """
    violations: list[str] = []
    if equity <= 0.0:
        violations.append(f"equity non-positive: {equity:.2f}")
        return violations
    alt_notional = sum(
        abs(notional) for symbol, notional in positions.items()
        if symbol.split("/")[0].upper() not in _MAJOR_BASES
    )
    cap = limits.max_alt_bucket_frac * equity
    if alt_notional > cap + _EPS:
        violations.append(
            f"alt bucket notional {alt_notional:.2f} exceeds "
            f"{limits.max_alt_bucket_frac:.0%} of equity ({cap:.2f})"
        )
    return violations


def _utc_date(ts_utc: datetime) -> date:
    """UTC calendar date of a timestamp; naive timestamps are assumed UTC."""
    if ts_utc.tzinfo is None:
        return ts_utc.date()
    return ts_utc.astimezone(timezone.utc).date()


class RiskManager:
    """Stateful risk overlay for the live engine (and event-driven paper fills).

    Tracks day-start equity per UTC day and the running equity peak, applies
    the drawdown / daily-loss ladders (brief 3.4), and sizes orders as the
    MINIMUM of vol-target / fixed-fractional / per-position cap (brief 3.3).

    Actions returned by :meth:`update_equity` (only on state TRANSITIONS):
      - ``'soft_dd'``     drawdown >= dd_soft_pct: all sizing halved
      - ``'soft_daily'``  daily loss >= daily_loss_soft_pct: all sizing halved
      - ``'halt_daily'``  daily loss >= daily_loss_hard_pct: flatten + lock,
                          auto-clears at the next UTC day rollover
      - ``'halt_mdd'``    drawdown >= dd_hard_pct: kill switch, cleared only
                          by manual :meth:`re_arm`
      - ``'none'``        otherwise
    Thread-safe: all state access goes through one lock.
    """

    def __init__(self, limits: RiskLimits, initial_equity: float):
        if initial_equity <= 0.0:
            raise ValueError("initial_equity must be positive")
        self.limits = limits
        self._lock = threading.Lock()
        self._equity = float(initial_equity)
        self._peak = float(initial_equity)
        self._day: date | None = None
        self._day_start_equity = float(initial_equity)
        self._soft_dd = False
        self._soft_daily = False
        self._halted_daily = False
        self._halted_mdd = False

    # -- state inspection (for GUI status) -----------------------------------
    @property
    def halted(self) -> bool:
        """True while any hard halt (daily lock or MDD kill) is active."""
        with self._lock:
            return self._halted_daily or self._halted_mdd

    def status(self) -> dict:
        """Snapshot of the overlay state for dashboards/logging."""
        with self._lock:
            return {
                "equity": self._equity,
                "peak": self._peak,
                "day": self._day.isoformat() if self._day is not None else None,
                "day_start_equity": self._day_start_equity,
                "drawdown": self._drawdown(),
                "daily_loss": self._daily_loss(),
                "soft_dd": self._soft_dd,
                "soft_daily": self._soft_daily,
                "halted_daily": self._halted_daily,
                "halted_mdd": self._halted_mdd,
            }

    def restore(self, snapshot: dict) -> None:
        """Re-seed latched ladder state from a :meth:`status` snapshot.

        Used on live-engine restart so the drawdown / daily ladders resume the
        SAME peak and day-start baseline (and any latched hard halt) instead of
        rebasing to the — already drawn-down — current equity. In particular a
        latched MDD kill (``halted_mdd``) stays latched across the restart until
        a human calls :meth:`re_arm`. Missing keys leave the corresponding
        constructor-seeded value untouched.
        """
        with self._lock:
            if snapshot.get("equity") is not None:
                self._equity = float(snapshot["equity"])
            if snapshot.get("peak") is not None:
                self._peak = float(snapshot["peak"])
            if snapshot.get("day_start_equity") is not None:
                self._day_start_equity = float(snapshot["day_start_equity"])
            day = snapshot.get("day")
            if day is not None:
                self._day = date.fromisoformat(day) if isinstance(day, str) else day
            self._soft_dd = bool(snapshot.get("soft_dd", self._soft_dd))
            self._soft_daily = bool(snapshot.get("soft_daily", self._soft_daily))
            self._halted_daily = bool(snapshot.get("halted_daily", self._halted_daily))
            self._halted_mdd = bool(snapshot.get("halted_mdd", self._halted_mdd))

    # -- internals (call with lock held) --------------------------------------
    def _drawdown(self) -> float:
        return 1.0 - self._equity / self._peak if self._peak > 0.0 else 0.0

    def _daily_loss(self) -> float:
        if self._day_start_equity <= 0.0:
            return 0.0
        return 1.0 - self._equity / self._day_start_equity

    # -- lifecycle -------------------------------------------------------------
    def update_equity(self, ts_utc: datetime, equity: float) -> str:
        """Feed a marked-to-market equity observation; run the loss ladders.

        Returns the NEW action string only on transitions, else ``'none'``.
        Handles the UTC day rollover internally (resets day-start equity and
        auto-clears the daily lock).
        """
        d = _utc_date(ts_utc)
        with self._lock:
            if self._day is None:
                self._day = d
            elif d != self._day:
                # UTC day rollover: rebase daily ladder, unlock daily halt
                self._day = d
                self._day_start_equity = float(equity)
                self._halted_daily = False
                self._soft_daily = False

            self._equity = float(equity)
            if equity > self._peak:
                self._peak = float(equity)

            dd = self._drawdown()
            daily = self._daily_loss()
            lim = self.limits

            new_halt_mdd = dd >= lim.dd_hard_pct - _EPS
            new_halt_daily = daily >= lim.daily_loss_hard_pct - _EPS
            new_soft_dd = dd >= lim.dd_soft_pct - _EPS and not new_halt_mdd
            new_soft_daily = daily >= lim.daily_loss_soft_pct - _EPS and not new_halt_daily

            action = "none"
            if new_halt_mdd and not self._halted_mdd:
                action = "halt_mdd"
            elif new_halt_daily and not self._halted_daily:
                action = "halt_daily"
            elif new_soft_dd and not self._soft_dd:
                action = "soft_dd"
            elif new_soft_daily and not self._soft_daily:
                action = "soft_daily"

            # hard halts are sticky (latched); soft flags track current level
            self._halted_mdd = self._halted_mdd or new_halt_mdd
            self._halted_daily = self._halted_daily or new_halt_daily
            self._soft_dd = new_soft_dd
            self._soft_daily = new_soft_daily

            if action != "none":
                log.warning("risk action %s (dd=%.4f daily=%.4f equity=%.2f)",
                            action, dd, daily, equity)
            return action

    def reset_daily(self, ts_utc: datetime) -> None:
        """Manually rebase the daily ladder to the current equity at ``ts_utc``."""
        with self._lock:
            self._day = _utc_date(ts_utc)
            self._day_start_equity = self._equity
            self._halted_daily = False
            self._soft_daily = False

    def re_arm(self) -> None:
        """Manual re-arm after a hard halt (GUI action).

        Clears both hard halts and rebases the drawdown peak and day-start
        equity to current equity, so the ladder measures fresh losses only.
        """
        with self._lock:
            self._halted_daily = False
            self._halted_mdd = False
            self._peak = self._equity
            self._day_start_equity = self._equity
            self._soft_dd = False
            self._soft_daily = False
            log.info("risk manager re-armed at equity %.2f", self._equity)

    # -- sizing ----------------------------------------------------------------
    def size_order(self, equity: float, price: float, stop_distance: float,
                   realized_vol_annual: float) -> float:
        """Quote-currency budget for a new order (brief 3.3: min of the stack).

        Parameters
        ----------
        equity : current account equity (quote currency).
        price : intended entry price (informational; budget is quote-based —
            callers derive units as ``budget / price``).
        stop_distance : stop distance as a FRACTION of price (e.g. 0.05 for a
            5% stop). ``<= 0`` skips the fixed-fractional term.
        realized_vol_annual : current annualized realized vol (365-day based).
            ``<= 0`` skips the vol-target term (per-position cap still binds).

        Returns 0.0 while any hard halt is active; returns half the budget
        while a soft brake (drawdown or daily) is active.
        """
        with self._lock:
            if self._halted_daily or self._halted_mdd:
                return 0.0
            if equity <= 0.0:
                return 0.0
            lim = self.limits
            budget = lim.max_position_frac * equity
            if realized_vol_annual > 0.0:
                budget = min(budget, equity * lim.target_annual_vol / realized_vol_annual)
            if stop_distance > 0.0:
                budget = min(budget, equity * lim.risk_per_trade_pct / stop_distance)
            if self._soft_dd or self._soft_daily:
                budget *= 0.5
            return max(budget, 0.0)

    def can_open(self, n_open_positions: int) -> bool:
        """True if a new position may be opened (no hard halt, below max count)."""
        with self._lock:
            if self._halted_daily or self._halted_mdd:
                return False
            return n_open_positions < self.limits.max_positions
