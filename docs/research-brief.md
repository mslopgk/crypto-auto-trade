# DESIGN BRIEF — GUI Real-Time Crypto Auto-Trading System
Python 3.12 / PySide6 / pyqtgraph / ccxt >= 4.5.x / numpy+numba / Windows. Venues: Binance (spot + USDT-M), Upbit (KRW spot, long/flat only).

---

## 0. Conflict Resolutions (binding decisions)

| Conflict | Resolution |
|---|---|
| Slippage estimates (1-3 bps vs 5-15 bps) | Standardized cost table in Section 2.1. Use the conservative figure per tier; stress at 2x. |
| Upbit public rate limit (10/s vs 30/s) | **10 req/s per IP** (conservative, matches global-docs). Order endpoint 8 req/s per account. Backfill and live bot share one IP budget — serialize them. |
| Walk-forward anchored vs rolling | **Rolling** windows (12mo IS / 3mo OOS, step = 3mo). Rolling adapts to crypto regime shifts; anchored dilutes recent regimes. |
| ADX threshold 20 vs 25 | **Hysteresis**: enter trend regime at ADX(14) > 25, exit only below 20. Chop kill-switch (suspend trend entries) at ADX < 20. |
| Mean reversion viability | Daily MR (buy N-day lows, RSI dip-buys) is **rejected** — went flat/negative OOS post-2022. MR allowed only intraday (1h/4h) with hard ADX < 20 gate, Phase 2. Daily RSI(5) cross-of-50 is momentum, not MR — allowed. |
| Vol-breakout target formula (open vs prev-close based) | **Open-based** (canonical systrader79): `target = today_09KST_open + prev_range * K`. Same formula in backtest and live, enforced by shared code path. |
| Grid trading | **Deferred to Phase 3 / optional.** Short-vol negative-skew P&L; only regime-conditional with hard stop. Not part of the core ensemble. |
| Kelly sizing | Never full Kelly. `size = min(0.25x Kelly, fixed-fractional 1% risk, vol-target size)`. |
| qasync vs QThread-asyncio vs QtAsyncio | **Dedicated QThread owning an asyncio loop** (Option B). PySide6.QtAsyncio is forbidden (no network primitives — ccxt.pro crashes). qasync acceptable only for throwaway prototypes. |
| Annualization | **sqrt(365)** daily, sqrt(2190) 4h, sqrt(8760) hourly. 252 is banned repo-wide (lint check). |
| Drawdown limits (15% vs 20% kill) | Soft brake at 10% peak-to-trough (halve all risk), **hard kill at 15%** (flatten, cancel all, manual re-arm). |
| Vendor performance claims | All third-party numbers (SuperTrend +810%, 240x vol-breakout, Pionex 175-500%, 73-77% win rates) treated as directional only. Every strategy re-derived in our engine before any weight > 0. |

---

## 1. Strategy Families — Implementation Priority & Starting Parameters

Build order below. All parameters are **grid ranges for the optimizer**, with plateau-center selection (never single-peak). Primary timeframes: **1d and 4h**. 1h only with daily gate. 15m trend following: do not build.

### P1 — Donchian Multi-Lookback Ensemble (flagship trend, Binance + Upbit)
Strongest OOS evidence (net Sharpe ~1.58, top-20 coins, survivorship-free).
- Lookbacks: fixed set **{5, 10, 20, 30, 60, 90, 150, 250, 360} days**; each emits +1 (price > N-day high band regime) / 0 / -1; final signal = mean of the 9.
- Position size proportional to fraction of lookbacks agreeing, then vol-scaled (Section 3).
- Exit per model: opposite 10/20-day channel, or global 2.5x ATR(14-20) trailing stop.
- Classic single-N variants for comparison: N=20 entry / 10-day exit; N=55 / 20-day exit; stop 2x ATR(20).
- Upbit: long/flat only (clip signal at 0 on short side).

### P2 — Volatility Breakout, adaptive-K (flagship Upbit daily strategy)
- Entry: `target = 09:00 KST open + (prev_high - prev_low) * K`; buy on intraday cross; exit next 09:00 KST open.
- K: adaptive = **30-day mean of noise ratio**, `noise = 1 - |O-C|/(H-L)`; grid-search fixed K in **[0.3, 0.7]** as baseline comparison.
- Universe filter: 30d avg noise < 0.55. Trend gate: price > 5-day MA (or MA-score = count(price > MA{3,5,10,20})/4 scaling position).
- Sizing: target-vol, daily target 1-2%, `frac = (target_vol / 5d_avg((H-L)/O)) / n_signal_coins`.
- Trade only when `(prev_range * K)/open > 3x round-trip cost`. Daily bars only — never sub-daily.
- Validate on post-2021 data explicitly; 2017-18 headline results are bull artifacts.

### P3 — Vol-Scaled TSMOM (Binance, 1d and optionally 6h)
Vol scaling is the single biggest Sharpe lever (0.65 -> 1.83 in the 2026 arXiv study).
- Lookback: **21-28 days** (grid 14-56); holding/rebalance **5-7 days**; signal = sign of trailing return.
- Mandatory vol targeting: 10-15% annualized target / realized vol.
- Optional adaptive variant later: monthly re-optimized ROC lookback + 2.5x ATR trailing stop.

### P4 — EMA Crossover with regime gates (Binance + Upbit, 1d; 4h with daily gate)
- Pairs: **9/21** and **21/50**; grids fast 9-21, slow 21-55; trend filter EMA/SMA **100-200**.
- 21/50 + EMA(200) filter is the highest-PF single config (PF 2.14 on D1).
- Never trade 1h crossovers without the daily-direction gate.

### P5 — SuperTrend as exit/trailing engine (not standalone entry)
- ATR period **10-14**, multiplier **2.5-3.0** on 4h/1d (4-5 on volatile alts). Pair with Donchian/EMA entries.

### P6 — Daily RSI-momentum BTC (simple, diversifying)
- RSI(5) crosses above 50 = long, below 50 = flat. Grid RSI period 3-8.

### P7 (Phase 2) — Intraday BB+RSI mean reversion, regime-gated
- 1h/4h only. Bollinger(20, 2.0) + RSI(14) < 30 (aggressive: RSI(2) < 10); enter on first close back inside band; exit at 20-SMA mid-band; stop 1-1.5x ATR + time stop (e.g. 12 bars).
- **Hard gate: ADX(14) < 20** and regime = RANGE (PF 1.62 ranging vs negative trending — the gate IS the edge).
- Volatility circuit breaker: pause entries when ATR(14) > 1.5x its 20-period average. Optional FOMC/CPI event-day suppressor.

### P8 (Phase 2) — VWAP reversion intraday
- Anchor 00:00 UTC (Binance) / 09:00 KST (Upbit). Bands ±2.0 session SD; entry needs rejection wick (lower wick > 2x body); target VWAP; stop 1x ATR beyond extreme; skip when 5m ADX > 25.

### P9 (Phase 2/3) — Funding-rate module (Binance perps)
- (a) Sentiment filter for the whole portfolio: block new longs when funding > +0.05%/8h; enable contrarian long bias after multi-day funding < -0.05%/8h.
- (b) Delta-neutral carry: activate only when funding persistently > 0.03%/8h (4-leg cost ~0.30-0.40% needs 6-8 funding periods at baseline to break even). Backtest with actual historical per-interval settlements, never extrapolated snapshots.
- Ingest via GET /fapi/v1/fundingRate + premiumIndex + markPrice WS.

### Variants explicitly rejected
Raw MACD(12,26,9) signal-cross (Sharpe 0.33; only zero-line variant + ADX>20 + daily gate allowed as low-priority experiment). Daily RSI dip-buying. Buying N-day lows. Sub-hourly trend following. Grid bots without regime gate + hard stop.

---

## 2. Backtest Engine Requirements Checklist

### 2.1 Cost model (hardcoded per venue, per liquidity tier; per-side = fee + slippage)
| Venue / tier | Fee | Slippage | Per-side | Round trip |
|---|---|---|---|---|
| Binance spot BTC/ETH (BNB discount) | 0.075% | 0.05% | 0.125% | 0.25% |
| Binance spot top-50 alts | 0.075% | 0.10% | 0.175% | 0.35% |
| Binance spot beyond top-100 | 0.075% | 0.50-2.0% | — | avoid |
| Binance USDT-M futures taker | 0.045-0.05% | 0.02-0.05% | ~0.07-0.10% | ~0.15-0.20% **+ funding 0.01%/8h** |
| Upbit KRW majors | 0.05% | 0.10% | 0.15% | 0.30% |
| Upbit KRW thin alts | 0.05% | 0.20-0.50% | — | size vs live book depth |
- Upbit stop-limit fee is **0.139%**, not 0.05%. Upbit BTC/USDT-quoted markets are 0.25% — never trade them with KRW-tuned strategies.
- Perp funding modeled from historical per-interval settlements (8h standard; some symbols 4h/1h; caps changed 2025, BTCUSDT ±0.3%).
- Slippage is per-symbol config; breakout-chasing entries double slippage during vol spikes.
- **Acceptance requires profitability at 2x all costs.** Every strategy's mean trade PnL must exceed 3x modeled round-trip cost.

### 2.2 Bias prevention (engine-enforced, not per-strategy)
- [ ] Signal-on-close, execute-next-open: loop hands strategies candles [0..t], fills at open[t+1] ± slippage. No strategy code ever sees t+1.
- [ ] Automated freqtrade-style lookahead test: recompute indicators on truncated data, diff values; fail build on mismatch. Ban shift(-n), centered rolling, whole-series aggregates in indicator code.
- [ ] Warmup: discard 3-5x longest indicator period (EMA200 -> 600-1000 bars) before scoring; fetch warmup + range, slice warmup off.
- [ ] Worst-case intrabar rules (explicit constants, flippable for sensitivity): stop fills before TP when both in one candle; gap-through-stop fills at open; low-before-high for stops.
- [ ] Limit orders fill only if price trades strictly through the limit.
- [ ] Point-in-time universe: symbols enter after actual listing date; delisted symbols (LUNA, FTT) retained. Monthly universe re-selection by liquidity rank.
- [ ] Venue-native data only: Upbit strategies on Upbit candles (kimchi premium 1-5%), Binance on Binance. Upbit candles reindexed to full time grid with ffill-close/volume=0 (Upbit omits zero-trade candles).
- [ ] Unclosed-candle guard: last candle from any fetch/watch is the forming bar — indicators computed on closed bars only, in both backtest and live (shared code path).
- [ ] sqrt(365)-family annualization everywhere; 365-day CAGR/Calmar.

### 2.3 Walk-forward + validation pipeline
- Rolling WFA: **12mo IS / 3mo OOS, step 3mo, >= 8 folds** (5 absolute minimum). WFE = OOS/IS annualized return: **>= 0.5 accept**, >= 0.7 strong, < 0.3 reject.
- Plateau check: perturb every parameter ±20-30% (±2 grid steps); rank candidates by **neighborhood-median** score, not peak; reject if Sharpe drops > 50% one step away.
- Trial logging: every optimizer evaluation logged; compute **Deflated Sharpe with effective N** (cluster correlated variants); deploy only if **DSR >= 0.95** (equivalently t-stat > 3).
- Single-use holdout: final ~25% of history (mid-2024 -> mid-2026), evaluated exactly once after all selection. Peeking burns it permanently.
- Per-regime scoring, all five buckets: 2020-11→2021-11 bull, 2021-11→2022-12 bear (LUNA/FTX gap candles test stop handling), 2023 chop, 2024 ETF bull, 2025-26 compressed-vol/whipsaw. Must be profitable in >= 2 of {2022 bear, 2023 chop, 2024-25 bull}; reject single-regime edges. Budget ~30-40% IS->live decay.

### 2.4 Deployment acceptance gate (OOS/holdout, after costs — fail any two, reject)
Sharpe >= 1.0 | Sortino >= 1.5 | Calmar >= 1.0 | MDD <= 30% | PF 1.3-2.0 | >= 100 trades (prefer 200+) | profitable at 2x costs | positive in >= 2 of 3 regime buckets.
**Overfit alarms (treat as bugs):** Sharpe > 3, PF > 4, straight-line log equity, fills at candle extremes.

### 2.5 Data layer
- Binance history: bulk zips from **data.binance.vision** (no rate cost); incremental via /api/v3/klines limit=1000, forward-paged on `since`, weight budget 6000/min, watch X-MBX-USED-WEIGHT-1M.
- Upbit history: /v1/candles count=200, **backward-paged on `to`**, <= 10 req/s, parse Remaining-Req header; then gap-fill.
- Storage: Parquet (bulk) + SQLite (state), keyed (exchange, symbol, timeframe, timestamp). One shared paginated downloader with per-exchange page size/direction.

---

## 3. Risk & Ensemble Layer Spec

### 3.1 Regime detector (evaluated on higher TF than entries; e.g. 4h regime, 1h entries; 1d regime, 4h entries)
- Confluence, 2-of-3 for TREND: {ADX(14) > 25, CHOP(14) < 38.2, price > EMA200 AND normalized EMA50 slope > 0.1 ATR/bar}.
- RANGE: ADX < 20 AND CHOP > 61.8. Between = transition, no switch.
- CRISIS override -> defensive (flat or 25-50% size): realized vol > 90th percentile (1y rolling, sqrt(365)) OR ATR +30% week-over-week.
- Anti-whipsaw: hysteresis (25 in / 20 out), 2 consecutive closed candles to confirm, minimum dwell time (e.g. 6 x bar), daily switch cap per pair.
- Portfolio risk-on gate: **BTC > its 100-200d SMA** required for alt longs.
- Optional second opinion, never primary: 2-state GaussianHMM on daily log returns, walk-forward monthly refit on rolling 2-4y, states sorted by variance each refit (label-switch fix); confirm/veto only.

### 3.2 Ensemble combiner
- Each strategy trades only its assigned regime (TREND -> Donchian/EMA/TSMOM/SuperTrend-exits; RANGE -> BB+RSI/VWAP reversion; CRISIS -> flat/defensive).
- Signals as continuous forecasts scaled to a common range (Carver [-20,+20], E|f|=10) or discrete votes with >= 60% weighted agreement to act.
- Weights: inverse-vol / rolling-OOS-Sharpe, rebalanced monthly; **cap 35% per strategy, floor 0** (drop, never negative-weight).
- Correlation control on return streams (not signals), rolling 90-180d: down-weight/merge pairs corr > 0.7; target cross-family < 0.5; handcraft groups (equal across groups, then within). Diversification multiplier IDM = 1/sqrt(w'Ωw), **capped 2.5**. Assume alt/BTC strategy correlations 0.7-0.9 in stress — realized diversification is small.

### 3.3 Position sizing stack (compute all, take the MINIMUM)
1. Vol targeting: notional = equity x target_vol / realized_vol (EWMA or 20-30d, sqrt(365)); portfolio target **15-20% annualized**; adjust only on >10-20% drift; leverage cap 1x spot.
2. Fixed fractional: **0.5-1% equity risk per trade**, size = equity x risk% / stop_distance.
3. Kelly (optional): **0.25x Kelly max**, and never above the 1-2% absolute risk ceiling.
- Stops: trend entries 2-2.5x ATR(14) initial + Chandelier(22, 3.0x ATR) trailing (PF 1.61 evidence); mean reversion 1-1.5x ATR + time stop; SuperTrend(10, 3) as alternative trail.

### 3.4 Risk overlay (independent module; overrides every strategy; also drives GUI status)
- Daily loss: half-size at 1.5% of day-start equity, **flatten + lock until next UTC day at 3%**.
- Drawdown ladder: 10% -> halve all risk; **15% -> kill switch** (cancel all orders, flatten, manual re-arm from GUI). Weekly cap 5-6%, monthly 10-12% (Elder 6% variant acceptable).
- Exposure: max **4-6 concurrent positions**; per-asset cap 20-30% notional; total open risk-to-stop <= 5-6% equity; **all alts = one correlated bucket** capped at ~2-3x single-position risk.
- Non-P&L kill triggers: stale market data (watchdog), repeated API errors, order-reject storms, auth failures.
- Chop kill-switch: suspend crossover/breakout entries at ADX(14) < 20 on the trading TF.

---

## 4. Live Engine + GUI Architecture

### 4.1 Concurrency
- **One dedicated QThread owning an asyncio event loop** running all ccxt.pro tasks (asyncio.gather of watch_* loops). GUI -> engine: `asyncio.run_coroutine_threadsafe(coro, worker.loop)`. Engine -> GUI: typed Qt Signals only (cross-thread = QueuedConnection); **no widget access from worker thread, ever**.
- Singleton event-bus QObject: ticker, bar_closed(symbol, bar), order_update, position_update, log(level, msg) signals. Widgets never import the exchange layer.
- UI batching: worker writes latest values into dicts/deques; GUI QTimer at 10-20 Hz does one setData + fine-grained dataChanged per burst; bar-close events emitted immediately.
- Always `await exchange.close()` on shutdown (session leaks kill 24/7 uptime).

### 4.2 Exchange integration (ccxt >= 4.5.x, enableRateLimit=True everywhere)
- Binance live data: `ccxt.pro watch_ohlcv`; act only on candle-close (timestamp rollover / k.x==true).
- **Upbit watch_ohlcv supports only '1s'** — build 1m+ bars by local aggregation from watch_trades, or subscribe raw `candle.1m` WS channel, or poll REST; carry-forward empty bars client-side (Upbit pushes candles only on trades). WS limits: 5 conn/s, 5 msg/s, 100 msg/min per connection.
- Central order function (single code path): load_markets() once; amount_to_precision / price_to_precision (Binance LOT_SIZE/tickSize -> avoids -1013; Upbit KRW tick brackets, changed mid-2025); pre-check min notional (5 USDT Binance / 5,000 KRW Upbit); attach UUID clientOrderId (newClientOrderId / identifier); **on RequestTimeout reconcile by clientOrderId before any re-place** (double-fill prevention).
- Upbit market buys: `options['createMarketBuyOrderRequiresPrice']=False`, pass KRW cost as amount (native 'price' type); market sells = 'market' with volume.
- Error policy: NetworkError -> exp backoff 1s->60s + jitter, max ~8, then circuit-break; RateLimit/DDoS -> pause >= 60s (429->418 IP-ban escalation); AuthenticationError -> halt + alert, never retry; InvalidOrder/InsufficientFunds -> log + skip, never retry unchanged. WS staleness watchdog: no message in N x timeframe -> close() and rebuild exchange instance.
- Rate budgets: Upbit orders <= 8 req/s per account, public <= 10 req/s per IP (shared with backfill); Binance weight 6000/min, orders 50/10s.
- Paper trading: Binance = Demo Trading via set_sandbox_mode(True) with testnet keys; smoke-test fetch_balance and on 'Invalid Api-Key ID' override urls to https://demo-api.binance.com (ccxt #27266). Upbit = **no sandbox**: POST /v1/orders/test for validation + internal paper-fill engine (fill vs live book mid ± slippage, 0.05% fee); final check with real 5,000-10,000 KRW orders.
- Keys: `keyring` -> Windows Credential Manager (DPAPI); withdrawal permission OFF; IP-restrict both (Upbit whitelist is mandatory and static-IP; Binance unrestricted keys auto-delete at 90d). GUI must surface Upbit's ~1-year key expiry.

### 4.3 GUI
- QMainWindow + QTabWidget: Dashboard / Backtest / Strategies-Optimizer / Settings / Logs. Dashboard panels via native QDockWidget (or PySide6-QtAds if user-rearrangeable layouts required; avoid pyqtgraph DockArea restoreState — issue #3125).
- Chart: two custom pg.GraphicsObject items — static-history QPicture regenerated on bar close (O(n)) + single live-bar item repainted per tick (O(1)); prepareGeometryChange + informViewBoundsChanged after setData; dataBounds() over visible slice for viewport Y-autorange; **x = bar index** with custom datetime AxisItem (no gaps, no float precision loss); volume PlotItem via setXLink, row stretch 3:1; crosshair via SignalProxy(rateLimit=60) + mapSceneToView; <= ~3000 visible candles; overlays = PlotDataItem with setDownsampling(auto=True) + setClipToView(True). antialias=False.
- Tables: QAbstractTableModel per panel, cell-level dataChanged (never beginResetModel per tick); QSortFilterProxyModel filtering. Logs: QPlainTextEdit setMaximumBlockCount(5000).
- Theme: Fusion style + hand-built dark QPalette + ~100 lines QSS. TradingView palette: bg #131722, base #1e222d, text #d1d4dc, up #26a69a, down #ef5350 (P&L colors via ForegroundRole, not QSS). Library fallback: qdarkstyle 3.x or pyqtdarktheme-fork (original is archived, pins <3.12).
- Backtest execution: single run -> QThreadPool + QRunnable (numpy/numba release GIL); optimizer grid -> ProcessPoolExecutor (spawn), pass file paths not DataFrames, dataset loaded in pool initializer, progress via Manager().Queue drained by 100 ms QTimer; QProcess only for crash-isolated runs.
- Packaging: PyInstaller 6.x **--onedir --noconsole**; exactly one Qt binding in venv; run from clean root (no __init__.py in cwd); `multiprocessing.freeze_support()` first line of main; --collect-submodules pyqtgraph if needed; SSL_CERT_FILE -> bundled certifi; state under QStandardPaths.AppDataLocation; --exclude-module QtWebEngineCore.

### 4.4 Build order
1. Data layer (downloaders, Parquet store, Upbit gap-fill, integrity checks).
2. Backtest engine core with cost model + bias guards + lookahead CI test (Section 2).
3. Strategies P1-P6 + validation pipeline (WFA, plateau, DSR, holdout).
4. Risk overlay + regime detector + ensemble combiner (Section 3), re-backtested end-to-end.
5. Live engine: exchange adapters, paper-fill engine, order reconciliation, watchdogs.
6. GUI shell + chart + dashboards (can proceed in parallel with 3-5 once event bus is defined).
7. Paper trade >= 4-6 weeks on Binance demo + Upbit paper engine; compare live fills vs backtest assumptions; then minimum-size live (5,000-10,000 KRW / ~10 USDT) before scaling.
8. Phase 2: P7-P9. Phase 3 (optional): grid module, funding carry.