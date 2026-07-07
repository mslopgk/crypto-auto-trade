"""Persistent live-session state.

JSON-serialized snapshot of a live engine session: open position, realized
PnL, capped equity history and trade log. Saves are atomic (tmp + os.replace)
so a crash mid-write never corrupts the previous snapshot.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from core.constants import STATE_DIR

log = logging.getLogger(__name__)

EQUITY_HISTORY_CAP = 50_000
TRADE_LOG_CAP = 10_000


def _sanitize(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "-")


@dataclass
class LiveState:
    """Snapshot of one live engine session (single exchange/symbol/timeframe).

    position: ``{"amount", "entry_price", "entry_time", "stance"}`` or None.
    equity_history: ``[[ts_ms, equity], ...]`` capped at 50k points.
    trade_log: list of Fill dicts capped at 10k entries.
    """

    position: dict | None = None
    realized_pnl: float = 0.0
    equity_history: list[list[float]] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    started_at: float | None = None   # unix seconds
    last_bar_ts: int | None = None    # candle open time, ms

    # -- mutation helpers (enforce caps) --------------------------------------
    def record_equity(self, ts_ms: int, equity: float) -> None:
        self.equity_history.append([int(ts_ms), float(equity)])
        if len(self.equity_history) > EQUITY_HISTORY_CAP:
            del self.equity_history[: len(self.equity_history) - EQUITY_HISTORY_CAP]

    def record_fill(self, fill: dict) -> None:
        self.trade_log.append(dict(fill))
        if len(self.trade_log) > TRADE_LOG_CAP:
            del self.trade_log[: len(self.trade_log) - TRADE_LOG_CAP]

    # -- (de)serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "realized_pnl": self.realized_pnl,
            "equity_history": self.equity_history,
            "trade_log": self.trade_log,
            "started_at": self.started_at,
            "last_bar_ts": self.last_bar_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LiveState":
        return cls(
            position=d.get("position"),
            realized_pnl=float(d.get("realized_pnl") or 0.0),
            equity_history=list(d.get("equity_history") or []),
            trade_log=list(d.get("trade_log") or []),
            started_at=d.get("started_at"),
            last_bar_ts=d.get("last_bar_ts"),
        )

    def save(self, path: str | Path) -> None:
        """Atomic write: dump to a temp file in the same dir, then os.replace."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "LiveState":
        """Load a snapshot; raises FileNotFoundError / json errors to the caller."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @staticmethod
    def default_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
        return STATE_DIR / f"live_{exchange_id}_{_sanitize(symbol)}_{timeframe}.json"
