# Crypto Auto-Trade — Design Document

Date: 2026-07-08
Status: Approved for implementation (user directed autonomous execution via ultracode)

## 1. Goal

GUI 기반 암호화폐 실시간 자동매매 프로그램.

- 다수의 트레이딩 전략(로직)을 체계적으로 생성하고, 과거 데이터로 백테스트하여 성능이 검증된 전략/앙상블을 선별한다.
- 선별된 전략을 GUI에서 실시간으로 구동한다 (페이퍼 트레이딩 기본, 실거래 지원).
- 범용: 거래소(Binance/Upbit), 심볼, 타임프레임, 전략을 자유롭게 조합.

## 2. Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | Installed; rich quant ecosystem |
| GUI | PySide6 + pyqtgraph | Installed; pyqtgraph is the standard for fast real-time charts in Qt |
| Exchange | ccxt 4.x (REST + WebSocket via `ccxt.pro` namespace) | One API for Binance/Upbit; free WS support |
| Compute | numpy + numba (JIT loops), pandas 3.0 | Vectorized backtests; numba for stop-loss/trailing loops |
| Storage | parquet (pyarrow) for OHLCV cache; JSON for configs/results | Fast, compact |
| Optimization | Custom grid/random search + walk-forward; optuna available | Full control over overfitting guards |

Considered alternatives:
- **backtesting.py / vectorbt / freqtrade**: rejected. freqtrade is a full framework (would fight our custom GUI); vectorbt free tier is limited and numpy-2.x compatibility churn; a custom engine (~500 lines) gives exact control over execution assumptions, which is where backtests silently lie.
- **Web GUI (FastAPI+React)**: rejected. Desktop Qt app is a better fit for a personal Windows tool, no server to run, native feel; PySide6 already installed.

## 3. Architecture

```
crypto-auto-trade/
├─ app.py                      # GUI entry point
├─ core/
│  ├─ constants.py             # timeframes, fees, paths
│  ├─ data/
│  │  ├─ fetcher.py            # ccxt REST OHLCV downloader (pagination, retry) + parquet cache
│  │  └─ stream.py             # async websocket live candles (ccxt.pro watch_ohlcv), REST-polling fallback
│  ├─ indicators.py            # vectorized: SMA/EMA/RSI/MACD/BB/ATR/Donchian/SuperTrend/ADX/Stoch/CHOP/...
│  ├─ backtest/
│  │  ├─ engine.py             # numba event loop: next-open execution, fees+slippage, intrabar SL/TP/trailing
│  │  └─ metrics.py            # CAGR, Sharpe/Sortino/Calmar (365d), MDD, PF, win rate, exposure, trades
│  ├─ strategies/
│  │  ├─ base.py               # Strategy ABC: PARAM_SPACE, generate_signals(df) -> SignalResult
│  │  ├─ registry.py           # name -> class registry for GUI/optimizer
│  │  ├─ trend.py              # EMACross, DonchianBreakout, SuperTrendStrategy, MACDTrend, TSMOM
│  │  ├─ meanrev.py            # RSI+BB mean reversion, BB squeeze
│  │  ├─ volbreakout.py        # Larry Williams K-volatility breakout (daily), noise-adaptive K
│  │  └─ ensemble.py           # VotingEnsemble, RegimeSwitchEnsemble
│  ├─ regime.py                # trend/range/high-vol classification (ADX, vol percentile, EMA slope)
│  ├─ risk.py                  # vol targeting, fixed-fractional, ATR stops, DD circuit breaker, daily loss limit
│  ├─ optimize/
│  │  ├─ search.py             # multiprocessing grid/random search across (strategy, params, symbol, tf)
│  │  └─ walkforward.py        # rolling IS/OOS windows, stitched OOS equity, plateau checks
│  └─ live/
│     ├─ broker.py             # Broker ABC; PaperBroker; CcxtBroker (Binance spot / Upbit)
│     ├─ engine.py             # LiveEngine: candle-close driven decide→size→execute loop, risk overlays
│     └─ state.py              # positions/trades/equity persistence (JSON), crash recovery
├─ gui/
│  ├─ main_window.py           # QMainWindow + tabs
│  ├─ theme.py                 # dark QSS
│  ├─ widgets/chart.py         # CandlestickItem, volume, indicator overlays, trade markers, crosshair
│  └─ panels/
│     ├─ dashboard.py          # live chart + equity + positions
│     ├─ backtest_panel.py     # strategy/params picker, run, equity curve, stats, trade table
│     ├─ optimizer_panel.py    # search config, progress, results table (sortable), WFA
│     ├─ live_panel.py         # exchange/API config, paper/live switch, start/stop, order log
│     └─ log_panel.py
├─ scripts/
│  ├─ download_data.py         # bulk historical download
│  ├─ run_search.py            # CLI mass search (writes results/*.json)
│  └─ run_walkforward.py
├─ tests/                      # pytest: engine correctness, indicator parity, no-lookahead tests
├─ data/                       # parquet cache (gitignored)
├─ results/                    # search outputs (gitignored)
└─ config/                     # app settings, final ensemble config
```

## 4. Backtest correctness rules (non-negotiable)

1. **No lookahead**: signals computed on bar `t` close → orders filled at bar `t+1` open.
2. **Costs**: per-side fee + slippage applied to every fill. Defaults: Binance spot taker 0.10% + slippage 0.05%; Upbit 0.05% + slippage 0.10%. Configurable.
3. **Intrabar stops**: SL/TP checked against bar high/low with conservative fills (gap-through fills at open, not at stop price). If both SL and TP hit in one bar → assume SL first (pessimistic).
4. **Warmup**: indicator NaN periods excluded from tradable region.
5. **Annualization**: 365 days (crypto trades 24/7), periods-per-year = 365*24*60 / timeframe-minutes.
6. **Overfitting guards**: min trade count (≥30 IS), walk-forward OOS required for deployment, parameter plateau check (neighbors within 30% of best), multi-symbol robustness.

## 5. Strategy zoo (initial families)

| Family | Strategies | Timeframes |
|---|---|---|
| Trend | EMA cross (+trend filter), Donchian breakout (Turtle), SuperTrend, MACD, TSMOM | 1h/4h/1d |
| Mean reversion | RSI(2-14) + Bollinger re-entry, BB squeeze expansion | 15m/1h/4h |
| Volatility breakout | Larry Williams K (daily open + K*range), noise-adaptive K | 1d built from 1h |
| Regime-switched | trend family in trends, mean-rev in ranges (ADX/vol classifier) | composite |
| Ensemble | equal-weight & correlation-aware voting of walk-forward survivors | composite |

Long-only for spot (Binance spot / Upbit); short support kept in engine for future perp use. Position exits: signal flip, ATR trailing stop, hard SL/TP.

## 6. Search → validate → deploy pipeline

1. Grid/random search over PARAM_SPACE × symbols (BTC, ETH, SOL, XRP…) × timeframes → multiprocessing, results to `results/`.
2. Filter: trades ≥ 30, PF > 1.1, MDD < 45%, Sharpe > 0.8 (IS).
3. Walk-forward: rolling 12mo IS / 3mo OOS, re-optimize per window; survivors need stitched-OOS Sharpe > 0.7 and OOS/IS Sharpe ratio > 0.5.
4. Ensemble: pick top-N low-correlation survivors, vol-weight; final config saved to `config/ensemble.json` and loadable in GUI/live engine.

## 7. Live engine

- Decision cadence = candle close of the strategy timeframe (websocket-driven, REST fallback).
- PaperBroker default: simulated fills at next tick with same cost model as backtest.
- CcxtBroker: market orders with `amount_to_precision`, min-notional checks, balance sync, retry w/ backoff; sandbox mode for Binance testnet.
- Risk overlays enforced pre-order: max position count, per-trade risk %, daily loss limit → auto-halt, MDD circuit breaker → flatten & halt.
- API keys: stored via Windows Credential Manager (`keyring`), never in plaintext config; trade-only permissions recommended in UI copy.
- Crash recovery: state.json restored on restart; positions reconciled against exchange balances.

## 8. GUI

Dark-themed QMainWindow, 5 tabs: Dashboard / Backtest / Optimizer / Live / Settings+Log.
- Chart: custom pyqtgraph CandlestickItem, incremental last-bar updates, indicator overlays, buy/sell markers, crosshair OHLC readout.
- Long work (backtest/search) runs in QThread workers or subprocess; progress via signals. GUI never blocks.
- Live feed: asyncio in dedicated thread (or qasync), UI updates only via Qt signals.

## 9. Testing & verification

- pytest unit tests: indicator parity vs reference, engine no-lookahead property test (shifting signals must change results appropriately), cost accounting, stop-fill logic, metrics sanity.
- Integration: full search round on real downloaded data; GUI smoke launch; paper-trade session on live websocket data.
- Adversarial multi-agent code review of engine/live modules before final.

## 10. Risks & mitigations

- **Overfitting** → walk-forward + plateau + multi-symbol + cost stress (2x fees) as standard gates.
- **pandas 3.0 API changes** → avoid deprecated idioms; numeric numpy arrays in hot paths.
- **Exchange API instability** → retry/backoff, REST fallback, engine halts safely on repeated failures.
- **User misuse of live mode** → paper default, explicit double-confirm to go live, risk limits on by default.
