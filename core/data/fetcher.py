"""Historical OHLCV download via ccxt REST, with parquet caching.

Cache layout: data/{exchange}/{SYMBOL with / -> _}/{timeframe}.parquet
DataFrame contract everywhere in this project:
    index: DatetimeIndex (UTC, tz-aware), name 'timestamp'
    columns: open, high, low, close, volume (float64)
"""
from __future__ import annotations

import time
import logging
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from core.constants import DATA_DIR, TIMEFRAME_MINUTES

log = logging.getLogger(__name__)

_EXCHANGE_CACHE: dict[str, ccxt.Exchange] = {}

# per-request candle limits
_FETCH_LIMIT = {"binance": 1000, "upbit": 200}


def get_exchange(exchange_id: str) -> ccxt.Exchange:
    ex = _EXCHANGE_CACHE.get(exchange_id)
    if ex is None:
        klass = getattr(ccxt, exchange_id)
        ex = klass({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        _EXCHANGE_CACHE[exchange_id] = ex
    return ex


def cache_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    sym = symbol.replace("/", "_").replace(":", "-")
    return DATA_DIR / exchange_id / sym / f"{timeframe}.parquet"


def _to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.astype({c: "float64" for c in ["open", "high", "low", "close", "volume"]})
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _fetch_ohlcv_backward(exchange_id: str, symbol: str, timeframe: str,
                          since_ms: int | None, until_ms: int,
                          max_retries: int = 5) -> pd.DataFrame:
    """Backward pagination via the exchange's `to` param (Upbit style).

    Upbit returns nothing for a `since` that predates listing, so we walk
    backward from `until_ms` until the exchange runs out of candles.
    """
    ex = get_exchange(exchange_id)
    limit = _FETCH_LIMIT.get(exchange_id, 200)
    all_rows: list = []
    cursor = until_ms
    retries = 0
    while True:
        try:
            rows = ex.fetch_ohlcv(symbol, timeframe, limit=limit,
                                  params={"to": ex.iso8601(cursor)})
            retries = 0
        except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable) as e:
            retries += 1
            if retries > max_retries:
                raise
            time.sleep(min(2 ** retries, 30))
            continue
        if not rows:
            break
        all_rows = rows + all_rows
        first_ts = rows[0][0]
        if since_ms is not None and first_ts <= since_ms:
            break
        if first_ts >= cursor:  # no progress guard
            break
        cursor = first_ts
    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = _to_df(all_rows)
    if since_ms is not None:
        df = df[df.index >= pd.Timestamp(since_ms, unit="ms", tz="UTC")]
    return df


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str,
                since_ms: int | None = None, until_ms: int | None = None,
                max_retries: int = 5) -> pd.DataFrame:
    """Paginated full download from `since_ms` (or exchange listing) to `until_ms`/now."""
    ex = get_exchange(exchange_id)
    limit = _FETCH_LIMIT.get(exchange_id, 500)
    tf_ms = TIMEFRAME_MINUTES[timeframe] * 60_000
    now_ms = int(time.time() * 1000)
    until_ms = until_ms or now_ms
    cursor = since_ms if since_ms is not None else 0
    all_rows: list = []
    retries = 0
    while True:
        try:
            rows = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            retries = 0
        except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable) as e:
            retries += 1
            if retries > max_retries:
                raise
            wait = min(2 ** retries, 30)
            log.warning("fetch retry %d for %s %s %s: %s", retries, exchange_id, symbol, timeframe, e)
            time.sleep(wait)
            continue
        if not rows:
            if not all_rows and exchange_id == "upbit":
                # `since` may predate listing; Upbit returns nothing then.
                return _fetch_ohlcv_backward(exchange_id, symbol, timeframe,
                                             since_ms, until_ms, max_retries)
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:  # no progress guard
            break
        cursor = next_cursor
        if cursor > until_ms or len(rows) < limit and last_ts + tf_ms > now_ms - tf_ms:
            break
        if len(rows) < limit and exchange_id != "upbit":
            # binance returns partial page only at the head of history or live edge
            if last_ts + tf_ms > now_ms - 2 * tf_ms:
                break
    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = _to_df(all_rows)
    if until_ms:
        df = df[df.index <= pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    return df


def load_ohlcv(exchange_id: str, symbol: str, timeframe: str,
               since: str | None = None, refresh: bool = True,
               drop_last_incomplete: bool = True) -> pd.DataFrame:
    """Load from cache; incrementally fetch missing tail (and head if `since` predates cache).

    since: ISO date string like '2019-01-01' (UTC).
    """
    path = cache_path(exchange_id, symbol, timeframe)
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000) if since else None
    df = None
    if path.exists():
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    if refresh or df is None or df.empty:
        if df is not None and not df.empty:
            head_ok = since_ms is None or df.index[0] <= pd.Timestamp(since_ms, unit="ms", tz="UTC") + pd.Timedelta(days=32)
            if not head_ok:
                older = fetch_ohlcv(exchange_id, symbol, timeframe, since_ms=since_ms,
                                    until_ms=int(df.index[0].timestamp() * 1000))
                df = pd.concat([older, df]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
            tail_from = int(df.index[-1].timestamp() * 1000)  # refetch last cached bar (may have been partial)
            newer = fetch_ohlcv(exchange_id, symbol, timeframe, since_ms=tail_from)
            if not newer.empty:
                df = pd.concat([df, newer]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
        else:
            df = fetch_ohlcv(exchange_id, symbol, timeframe, since_ms=since_ms)
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
    if since_ms is not None and not df.empty:
        df = df[df.index >= pd.Timestamp(since_ms, unit="ms", tz="UTC")]
    if drop_last_incomplete and len(df) > 1:
        # last row is the currently-forming candle if its close time is in the future
        tf_min = TIMEFRAME_MINUTES[timeframe]
        last_close_time = df.index[-1] + pd.Timedelta(minutes=tf_min)
        if last_close_time > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]
    if exchange_id == "upbit" and len(df) > 1:
        df = fill_gaps(df, timeframe)
    return df


def fill_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Reindex to a full time grid. Upbit omits zero-trade candles; synthesize
    them as flat bars at the previous close with volume 0."""
    tf_min = TIMEFRAME_MINUTES[timeframe]
    full = pd.date_range(df.index[0], df.index[-1], freq=f"{tf_min}min", tz="UTC")
    if len(full) == len(df):
        return df
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df.index.name = "timestamp"
    return df


def available_cached(exchange_id: str | None = None) -> list[dict]:
    """List cached datasets: [{exchange, symbol, timeframe, rows, start, end}]."""
    out = []
    roots = [DATA_DIR / exchange_id] if exchange_id else [p for p in DATA_DIR.iterdir() if p.is_dir()]
    for root in roots:
        if not root.is_dir():
            continue
        for sym_dir in root.iterdir():
            if not sym_dir.is_dir():
                continue
            for f in sym_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(f, columns=["close"])
                    out.append({
                        "exchange": root.name,
                        "symbol": sym_dir.name.replace("_", "/"),
                        "timeframe": f.stem,
                        "rows": len(df),
                        "start": str(df.index[0]) if len(df) else "",
                        "end": str(df.index[-1]) if len(df) else "",
                    })
                except Exception as e:  # corrupted cache entry
                    log.warning("bad cache file %s: %s", f, e)
    return out
