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
# Research-derived tiers (docs/research-brief.md §2.1). Acceptance gates re-test at 2x.
DEFAULT_COSTS = {
    "binance": {"fee": 0.00075, "slippage": 0.0005},   # BTC/ETH tier w/ BNB discount
    "upbit": {"fee": 0.0005, "slippage": 0.0010},      # KRW majors
}

# per-side slippage overrides by liquidity tier
SLIPPAGE_TIERS = {
    "binance": {"major": 0.0005, "alt": 0.0010},        # majors: BTC, ETH
    "upbit": {"major": 0.0010, "alt": 0.0025},
}
MAJOR_SYMBOLS = {"BTC/USDT", "ETH/USDT", "BTC/KRW", "ETH/KRW"}


def cost_profile(exchange_id: str, symbol: str) -> dict:
    """Per-side fee/slippage for a symbol; conservative tiering."""
    fee = DEFAULT_COSTS.get(exchange_id, {"fee": 0.001})["fee"]
    tiers = SLIPPAGE_TIERS.get(exchange_id, {"major": 0.0005, "alt": 0.0010})
    slip = tiers["major"] if symbol in MAJOR_SYMBOLS else tiers["alt"]
    return {"fee": fee, "slippage": slip}

DEFAULT_EXCHANGE = "binance"

DEFAULT_SYMBOLS = {
    "binance": [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT",
    ],
    "upbit": ["BTC/KRW", "ETH/KRW", "XRP/KRW", "SOL/KRW"],
}
