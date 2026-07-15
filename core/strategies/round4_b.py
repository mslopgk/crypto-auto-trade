"""Round-4 original strategies, group B (compression ladder, weekday, v-shape recovery).

All three are long/flat (spot). Day-based lookbacks are converted to bars via the
``timeframe`` param exactly as in ``core/strategies/trend.py``
(bars_per_day = 1440 / TIMEFRAME_MINUTES[timeframe]).

Strict no-lookahead: stance[i] depends only on rows <= i. Every 'median/mean of
history' uses a TRAILING window (cumulative sums / pandas rolling), never a
whole-series aggregate; self-inclusion is removed with an explicit shift where the
spec calls for it.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.indicators import ema, rolling_max, rolling_min
from core.strategies.base import Strategy, hold_stance
from core.strategies.registry import register
from core.strategies.trend import bars_per_day, shift1

log = logging.getLogger(__name__)


@register
class CompressionLadder(Strategy):
    """Multi-horizon range compression, then a directional (upward) break.

    For each horizon H in {3,5,7,14} days the current H-day range
    ``max(high,H) - min(low,H)`` is compared to the trailing MEDIAN of that same
    H-day range over the last 60 days. The median window ENDS AT THE PREVIOUS BAR
    (``shift1`` of a 60d rolling median), so the current bar's own range never
    contributes to the threshold it is tested against -- this is the documented
    resolution of the "self-inclusion subtleties" the spec warns about. A horizon
    is *compressed* when its current range is below that trailing median.

    ``score`` = number of compressed horizons (0-4), evaluated ON THE PREVIOUS BAR
    (``shift1`` of the per-bar count). Entry requires prev score >= ``min_score``
    AND an upward break (today's close above the previous 7-day high) AND a
    long-term regime gate (close > EMA(``gate_days``)). Exit is the mandatory
    engine ATR trail plus a stance exit when close falls below the previous
    14-day low. State carried with :func:`hold_stance`.
    """

    NAME = "compression_ladder"
    TIMEFRAMES = ("1d",)
    SEARCHABLE = True
    HORIZON_DAYS: tuple[int, ...] = (3, 5, 7, 14)
    MEDIAN_DAYS: int = 60
    PARAM_SPACE = {
        "min_score": [3, 4],
        "gate_days": [100, 200],
        "trail_atr_mult": [2.0, 2.5, 3.0],
    }
    DEFAULTS = {"min_score": 3, "gate_days": 200, "trail_atr_mult": 2.5,
                "timeframe": "1d"}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        med_bars = self.MEDIAN_DAYS * bpd

        # per-horizon compression flags -> integer score in [0, 4]
        score = np.zeros(n, dtype=np.float64)
        for days in self.HORIZON_DAYS:
            hb = max(1, days * bpd)
            rng = rolling_max(h, hb) - rolling_min(l, hb)   # includes current bar
            # trailing 60d median ENDING AT THE PREVIOUS BAR (self-excluding)
            med = pd.Series(rng).rolling(med_bars).median().to_numpy()
            med_prev = shift1(med)
            with np.errstate(invalid="ignore"):
                compressed = rng < med_prev                 # NaN -> False
            score += compressed.astype(np.float64)

        prev_score = shift1(score, fill=0.0)                # score of the PREVIOUS bar
        prev_7d_high = shift1(rolling_max(h, max(1, 7 * bpd)))
        prev_14d_low = shift1(rolling_min(l, max(1, 14 * bpd)))
        gate = ema(c, int(self.params["gate_days"]) * bpd)

        with np.errstate(invalid="ignore"):
            entries = ((prev_score >= float(self.params["min_score"]))
                       & (c > prev_7d_high)                 # upward break
                       & (c > gate))                        # NaN gate -> False
            exits = c < prev_14d_low
        entries = np.nan_to_num(entries, nan=False).astype(bool)
        exits = np.nan_to_num(exits, nan=False).astype(bool)
        return hold_stance(entries, exits)


@register
class AdaptiveWeekday(Strategy):
    """Self-adapting weekday seasonality with a trend-consent filter.

    For each weekday d the strategy tracks the TRAILING MEAN of per-bar log
    returns realised on weekday-d bars over the past ``lookback`` calendar days
    (a trailing cumulative-sum window; only bars <= i contribute). The stance is
    decided at close i but filled next open, so the relevant seasonality is
    TOMORROW's weekday -- computed as ``(timestamp_i + 1 day).dayofweek``, a pure
    function of bar i's own timestamp (no future bar is read).

    Long when the trailing mean return of tomorrow's weekday exceeds ``min_edge``
    AND close > EMA(200d). Per-bar exposure signal (recomputed every bar), NOT a
    hold-until-exit -- expect high turnover; avg-trade economics may fail the cost
    floor, which is an acceptable, honest result for an exposure-timing sleeve.
    """

    NAME = "adaptive_weekday"
    TIMEFRAMES = ("1d",)
    # SEARCHABLE=False (like GatedRSI2): this is a DAILY-only strategy whose
    # EMA(200d) consent gate needs 200 UTC days of history. The generic zoo
    # contract fixture is a 3000-bar 1h driftless walk (~4800 bars for a 200d
    # EMA at 1h -> gate all-NaN), so it can never exercise this strategy and its
    # params would appear inert there. Excluded from the DEFAULT search sweep;
    # still fully searchable when named explicitly, and covered on a proper daily
    # fixture + real BTC in tests/test_round4_b.py.
    SEARCHABLE = False
    EMA_DAYS: int = 200
    PARAM_SPACE = {
        "lookback": [60, 90, 180],
        "min_edge": [0.0, 0.001],
    }
    DEFAULTS = {"lookback": 90, "min_edge": 0.0, "timeframe": "1d"}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])
        idx = df.index

        # per-bar log returns; attributed to the weekday of the bar they close on
        lr = np.full(n, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr[1:] = np.log(c[1:] / c[:-1])
        valid = np.isfinite(lr)
        weekday = np.asarray(idx.dayofweek, dtype=np.int64)
        # tomorrow's weekday: pure function of THIS bar's timestamp (causal)
        tomorrow_wd = np.asarray((idx + pd.Timedelta(days=1)).dayofweek, dtype=np.int64)

        window = max(1, int(self.params["lookback"]) * bpd)
        arange = np.arange(n)
        starts = np.maximum(0, arange - window + 1)

        # trailing mean of lr per weekday via cumulative sums (window ending at i)
        trailing = np.full(n, np.nan)
        for d in range(7):
            mask = (weekday == d) & valid
            if not mask.any():
                continue
            vals = np.where(mask, lr, 0.0)
            csv = np.concatenate(([0.0], np.cumsum(vals)))
            csc = np.concatenate(([0.0], np.cumsum(mask.astype(np.float64))))
            s = csv[arange + 1] - csv[starts]
            cnt = csc[arange + 1] - csc[starts]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_d = np.where(cnt > 0, s / cnt, np.nan)
            sel = tomorrow_wd == d
            trailing[sel] = mean_d[sel]

        gate = ema(c, self.EMA_DAYS * bpd)
        with np.errstate(invalid="ignore"):
            long_ok = (trailing > float(self.params["min_edge"])) & (c > gate)
        return np.nan_to_num(long_ok, nan=False).astype(np.float64)


@register
class VShapeRecovery(Strategy):
    """Post-capitulation inflection catcher (fills the regime TSMOM sits out).

    State machine, strictly causal:

    * ``drawdown`` = close / rolling_max(close, 90d) - 1 (rolling max includes the
      current bar).
    * ARM when drawdown < -``dd_arm``.
    * While armed and flat, ENTER long on the first sign of strength -- close above
      the PREVIOUS bar's 10-day high -- provided price is still ``dd_exit_level``
      below the 90d high (i.e. not already recovered).
    * EXIT (stance) when close reaches the PREVIOUS bar's 60-day high (recovery
      complete); the mandatory engine ATR trail and hard ``sl_pct`` provide the
      other two exits. After any stance exit the machine DISARMS until the
      drawdown condition re-arms it.

    ``dd_exit_level`` is a fixed DEFAULTS constant, not a grid axis: the spec names
    it in the entry gate but assigns it no grid, so it is held at 0.10 (enter only
    while still >= 10% below the 90d high). Documented as a resolved ambiguity.
    """

    NAME = "vshape_recovery"
    TIMEFRAMES = ("1d",)
    # SEARCHABLE=False (like GatedRSI2): a DAILY capitulation strategy. Its arming
    # gate needs a 90d drawdown of >= dd_arm (25-45%), a regime the short driftless
    # 1h walk fixture never produces, so its one non-engine param (dd_arm) is inert
    # there. Excluded from the DEFAULT search sweep; still searchable when named,
    # and fully covered on a regime-varied daily fixture + real BTC in
    # tests/test_round4_b.py.
    SEARCHABLE = False
    DD_DAYS: int = 90
    HIGH_DAYS: int = 10
    RECOVER_DAYS: int = 60
    PARAM_SPACE = {
        "dd_arm": [0.25, 0.35, 0.45],
        "trail_atr_mult": [2.5, 3.0],
        "sl_pct": [0.07, 0.10],
    }
    DEFAULTS = {"dd_arm": 0.35, "trail_atr_mult": 3.0, "sl_pct": 0.10,
                "dd_exit_level": 0.10, "timeframe": "1d"}

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        n = len(c)
        bpd = bars_per_day(self.params["timeframe"])

        roll_hi_90 = rolling_max(c, max(1, self.DD_DAYS * bpd))
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = c / roll_hi_90 - 1.0
        prev_10d_high = shift1(rolling_max(c, max(1, self.HIGH_DAYS * bpd)))
        prev_60d_high = shift1(rolling_max(c, max(1, self.RECOVER_DAYS * bpd)))

        dd_arm = float(self.params["dd_arm"])
        dd_exit_level = float(self.params["dd_exit_level"])

        stance = np.zeros(n, dtype=np.float64)
        armed = False
        pos = False
        for i in range(n):
            if pos:
                # recovery complete -> stance exit + disarm (engine trail/sl also active)
                if np.isfinite(prev_60d_high[i]) and c[i] >= prev_60d_high[i]:
                    pos = False
                    armed = False
                    stance[i] = 0.0
                else:
                    stance[i] = 1.0
                continue
            # flat: (re-)arm on deep drawdown
            if not armed and np.isfinite(drawdown[i]) and drawdown[i] < -dd_arm:
                armed = True
            # armed + first strength + not yet recovered -> enter
            if (armed and np.isfinite(prev_10d_high[i]) and c[i] > prev_10d_high[i]
                    and np.isfinite(drawdown[i]) and drawdown[i] < -dd_exit_level):
                pos = True
                stance[i] = 1.0
            else:
                stance[i] = 0.0
        return stance
