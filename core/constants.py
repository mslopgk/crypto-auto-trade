"""Shared constants: paths, timeframes, default costs."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"
STATE_DIR = PROJECT_ROOT / "state"

for _d in (DATA_DIR, RESULTS_DIR, CONFIG_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# timeframe -> minutes
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "1w": 10080,
}

# Crypto trades 24/7 -> 365-day annualization
DAYS_PER_YEAR = 365.0


def periods_per_year(timeframe: str) -> float:
    return DAYS_PER_YEAR * 24.0 * 60.0 / TIMEFRAME_MINUTES[timeframe]


# Default per-side costs (fraction, not %). Conservative taker assumptions.
DEFAULT_COSTS = {
    "binance": {"fee": 0.0010, "slippage": 0.0005},
    "upbit": {"fee": 0.0005, "slippage": 0.0010},
}

DEFAULT_EXCHANGE = "binance"

DEFAULT_SYMBOLS = {
    "binance": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT",
    ],
    "upbit": ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW"],
}
