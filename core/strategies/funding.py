"""Funding-regime sleeve (round-2 candidate #1).

Two long-only-spot strategies driven by Binance USDM *realized* funding, an
orthogonal positioning/leverage signal (not a price transform):

- :class:`FundingCapitulation` (Leg B, standalone long): buy deeply-negative
  funding once price stops making new lows; exit on funding normalization, a
  time-stop, or the engine's ATR/SL.
- :class:`TsmomFundingGated` (Leg A, overlay): the existing vol-scaled TSMOM
  *stance*, but its size is de-grossed to ``size_mult`` while funding is
  euphoric (z >= z_high), released with hysteresis (z < 0.5).

Both read the traded symbol from ``params['symbol']`` (injected by the
optimizer / runner) to pick the funding series, and fall back gracefully when
no funding parquet exists for that base (Capitulation -> flat; Gated -> plain
TSMOM sizing), logging a warning.

Causality: funding features are daily, computed only from prints settled at or
before each daily close (see :mod:`core.data.funding`), then aligned to each
bar by calendar date. All state machines below depend only on rows <= i.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.data.funding import daily_funding_features, load_funding
from core.strategies.base import Strategy
from core.strategies.registry import register
from core.strategies.trend import TSMOM, bars_per_day, shift1

log = logging.getLogger(__name__)


def _symbol_base(symbol: str) -> str:
    return str(symbol).split("/")[0].upper()


def funding_z_for_df(df: pd.DataFrame, symbol: str, K: int, L: int) -> np.ndarray | None:
    """Per-bar rolling funding z-score aligned to ``df`` by calendar date.

    Returns a float array (NaN warmup / missing days) length ``len(df)``, or
    ``None`` when no funding parquet exists for the symbol's base. The value at
    bar ``i`` is the z-score known at that bar's daily close -> strictly causal.
    """
    rates = load_funding(_symbol_base(symbol))
    if rates is None or len(rates) == 0:
        return None
    feats = daily_funding_features(rates, [int(K)], [int(L)])
    z = feats[f"z_K{int(K)}_L{int(L)}"]
    # Align daily funding to each bar by its UTC calendar day. For intraday
    # bars every bar in a day inherits that day's (already-closed) funding.
    aligned = z.reindex(df.index.normalize())
    return aligned.to_numpy(dtype="float64")


@register
class FundingCapitulation(Strategy):
    """Leg B — contrarian capitulation long, daily timeframe.

    Entry: funding z <= ``z_low`` AND price stops making new lows, confirmed by
    ``confirm_mode``:
      * ``"prev_low"``  -> close > previous bar's low;
      * ``"hl3"``       -> a 3-bar rising-low (low[i] > low[i-1] > low[i-2]).
    Exit: funding z > 0, or a ``hold_days`` time-stop; the engine's ``sl_pct``
    (and optional ``tp_pct``) provide the hard ATR/price stop. Position size is
    the engine default (full ``size_frac``) — no funding gate on the entry leg.

    Missing funding -> all-flat stance (logged).
    """

    NAME = "funding_capitulation"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "L": [14, 21, 30, 45],
        "K": [1, 3, 9],
        "z_low": [-1.5, -2.0, -2.5],
        "hold_days": [3, 5, 10],
        "sl_pct": [0.03, 0.05],
        "confirm_mode": ["prev_low", "hl3"],
    }
    DEFAULTS = {
        "L": 30, "K": 3, "z_low": -2.0, "hold_days": 5,
        "sl_pct": 0.05, "confirm_mode": "prev_low",
        "timeframe": "1d", "symbol": "BTC/USDT", "exchange": "binance",
    }

    def _confirm(self, df: pd.DataFrame) -> np.ndarray:
        low = df["low"].to_numpy(dtype="float64")
        close = df["close"].to_numpy(dtype="float64")
        mode = self.params["confirm_mode"]
        if mode == "hl3":
            prev1 = shift1(low)
            prev2 = shift1(prev1)
            return (low > prev1) & (prev1 > prev2)
        # default "prev_low": current close reclaims the prior bar's low
        return close > shift1(low)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        z = funding_z_for_df(df, self.params["symbol"],
                             int(self.params["K"]), int(self.params["L"]))
        if z is None:
            log.warning("funding_capitulation: no funding for %s -> flat",
                        self.params["symbol"])
            return np.zeros(n, dtype="float64")

        confirm = self._confirm(df)
        z_low = float(self.params["z_low"])
        bpd = bars_per_day(self.params["timeframe"])
        hold_bars = max(1, int(self.params["hold_days"]) * bpd)

        stance = np.zeros(n, dtype="float64")
        state = 0
        held = 0
        for i in range(n):
            zi = z[i]
            if state == 1:
                held += 1
                if (not np.isnan(zi) and zi > 0.0) or held >= hold_bars:
                    state = 0
            if state == 0:
                if (not np.isnan(zi)) and zi <= z_low and confirm[i]:
                    state = 1
                    held = 0
            stance[i] = state
        return stance


@register
class TsmomFundingGated(TSMOM):
    """Leg A — vol-scaled TSMOM with a funding de-gross overlay.

    Stance is identical to :class:`~core.strategies.trend.TSMOM` (trend exits
    untouched). Size = TSMOM vol-target size * de-gross multiplier: 1.0
    normally, cut to ``size_mult`` while funding z >= ``z_high``, released with
    hysteresis once z < ``release`` (0.5). Missing funding -> plain TSMOM size.

    Grid = TSMOM's (lookback fixed to {14,21,28,42}) x funding gate — 72 combos.
    """

    NAME = "tsmom_funding_gated"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    PARAM_SPACE = {
        "lookback_days": [14, 21, 28, 42],
        "target_vol": [0.10, 0.15, 0.20],
        "z_high": [1.5, 2.0, 2.5],
        "size_mult": [0.0, 0.5],
    }
    DEFAULTS = {
        **TSMOM.DEFAULTS,
        "z_high": 2.0, "size_mult": 0.5, "release": 0.5,
        "funding_L": 30, "funding_K": 3,
        "symbol": "BTC/USDT", "exchange": "binance",
    }

    def _degross_mult(self, z: np.ndarray) -> np.ndarray:
        """Hysteretic de-gross multiplier: 1.0 -> size_mult at z>=z_high,
        back to 1.0 at z<release. Causal (depends only on z[<=i])."""
        n = len(z)
        mult = np.ones(n, dtype="float64")
        z_high = float(self.params["z_high"])
        size_mult = float(self.params["size_mult"])
        release = float(self.params["release"])
        degross = False
        for i in range(n):
            zi = z[i]
            if not np.isnan(zi):
                if degross:
                    if zi < release:
                        degross = False
                elif zi >= z_high:
                    degross = True
            mult[i] = size_mult if degross else 1.0
        return mult

    def generate_size_frac(self, df: pd.DataFrame) -> np.ndarray | None:
        base = super().generate_size_frac(df)
        n = len(df)
        if base is None:
            base = np.full(n, float(self.params.get("size_frac", 1.0)))
        base = np.asarray(base, dtype="float64")

        z = funding_z_for_df(df, self.params["symbol"],
                             int(self.params["funding_K"]), int(self.params["funding_L"]))
        if z is None:
            log.warning("tsmom_funding_gated: no funding for %s -> plain TSMOM size",
                        self.params["symbol"])
            return base

        # TSMOM already shifts its size by 1 bar (decision at close i-1 -> fill
        # bar i). The de-gross decision is made on the same close, so shift the
        # multiplier identically: shift1(base_raw)*shift1(mult) == shift1(raw*mult).
        mult = self._degross_mult(z)
        return base * shift1(mult, fill=1.0)
