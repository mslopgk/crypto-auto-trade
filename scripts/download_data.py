"""Bulk historical OHLCV download.

Usage: python -m scripts.download_data [--exchange binance] [--since 2019-01-01]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import DEFAULT_SYMBOLS
from core.data.fetcher import load_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download")

PLAN = {
    "binance": {
        "symbols": DEFAULT_SYMBOLS["binance"],
        "timeframes": ["15m", "1h", "4h", "1d"],
        "since": "2019-01-01",
    },
    "upbit": {
        "symbols": DEFAULT_SYMBOLS["upbit"],
        "timeframes": ["1h", "4h", "1d"],
        "since": "2019-01-01",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default=None, help="binance | upbit | (default: all)")
    ap.add_argument("--since", default=None)
    args = ap.parse_args()

    exchanges = [args.exchange] if args.exchange else list(PLAN)
    for ex in exchanges:
        plan = PLAN[ex]
        since = args.since or plan["since"]
        for symbol in plan["symbols"]:
            for tf in plan["timeframes"]:
                try:
                    df = load_ohlcv(ex, symbol, tf, since=since, refresh=True)
                    log.info("%s %s %s: %d bars  %s -> %s", ex, symbol, tf, len(df),
                             df.index[0] if len(df) else "-", df.index[-1] if len(df) else "-")
                except Exception as e:
                    log.error("FAILED %s %s %s: %s", ex, symbol, tf, e)


if __name__ == "__main__":
    main()
