# TOP 5 Candidates to Implement NOW — Lead Quant Ranking

Ranked by the weighted objective: chop/bear diversification value × implementability × cost-adjusted evidence × walk-forward applicability. Fresh 2024–2026 evidence checks are folded in below and materially reshaped the ranking (they demote cross-sectional reversal and confirm the funding + low-vol picks).

---

## #1 — Funding-Regime Sleeve (euphoria de-gross + capitulation add)
**The single highest-value diversifier: orthogonal information source, fully backtestable, protects the exact regime the trend book bleeds in.**

- **Why it wins:** Its signal is *positioning/leverage*, not price — so it is not a monotone transform of TSMOM or the 4h EMA-cross. It fires maximally-protective exactly when both long-only trend sleeves are maximally long into a leveraged blowoff top, and it buys capitulation the trend systems refuse. `fetchFundingRateHistory` has multi-year history on Binance USDM → **our WFA fully applies** (this is the key discriminator vs. the OI-divergence idea, which cannot be backtested).
- **Venue/symbols:** Binance USDM funding for BTC/ETH/SOL/ADA (the TSMOM basket). Long-only-spot compatible. Upbit KRW sleeve inherits the *derived* regime flag only (no perps there) — validate Upbit separately (kimchi-premium behaves differently).
- **Rules:** At each daily close, `ann_funding = mean(last K realized prints) × 3 × 365`; `z = (ann_funding − rollmean_L)/rollstd_L`. Use only *settled* funding known at close → next-open fill.
  - *Leg A — de-gross (overlay on TSMOM longs):* if `z ≥ z_high` (or absolute > +0.05%/8h sustained ≥3 prints), cut that asset's spot weight to `size_mult`; restore only when `z < +0.5` (hysteresis). Hard-flatten to `size_mult` if additionally price > 2·ATR above 20d MA. Trend exits untouched.
  - *Leg B — capitulation add (standalone long):* if `z ≤ z_low` **AND** price stops making new lows (3-bar higher-low / close > prior-bar low), add a contrarian tranche; exit on `z > 0`, a 3–10d time-stop, or an ATR target; hard ATR stop.
- **Grid:** `L∈{14,21,30,45}d`; `K∈{1,3,9,21}` prints; `z_high∈{1.5,2.0,2.5}`; `z_low∈{−1.5,−2.0,−2.5}`; `size_mult∈{0,0.5}`; `add_weight∈{0.25,0.5}×base`; `hold∈{3,5,10}d`; `ATR_stop∈{2,3}`.
- **Timeframe:** daily decision aligned to TSMOM; funding sampled at native 8h.
- **Ensemble role:** Tail-risk reducer (Leg A) + small chop/bear contrarian long (Leg B). Its "return" is largely *avoided drawdown* at euphoric tops plus capitulation snapbacks — near-zero-to-negative correlation with the trend book by construction.
- **Watch:** funding was positive ~92% of days in 2025–26, so only extreme, hysteresis-gated thresholds are informative; a naive "block longs when funding>0" destroys the book. Look-ahead: use realized, not predicted, funding.

---

## #2 — Cross-Sectional Low-Volatility Tilt (long-only defensive core)
**Cleanest fully-backtestable long-only diversifier; ranks on vol, not trend/return, so structurally decorrelated from both sleeves.**

- **Evidence (freshly confirmed):** The 2026 "Revisiting the low-volatility anomaly" study (432 Binance coins, **Jan 2018–Nov 2025**) finds a significantly *negative* cross-sectional vol premium that **emerges as the market matured** post-2021 — reversing the 2013–2019 null. Vol ranks are highly persistent → low turnover → trivial cost drag (<1%/yr at our 25–30bps).
- **Venue/symbols:** Binance USDT, 12–20 most-liquid non-stablecoin pairs. **Point-in-time universe** (ADV>$20M screened with past-only data at each rebalance); include delisted coins (LUNA/FTT) priced to last/zero or OOS Sharpe is inflated.
- **Rules:** Daily OHLCV → trailing realized vol (Parkinson high-low preferred over close-close). Rank ascending; long bottom quartile/tercile or fixed k=3–5; inverse-vol or equal weight. Signal on close → next open, weekly/biweekly rebalance. Optional per-name absolute-momentum cash-out to USDT when its close < EMA100/200 (dual-momentum defensive, adds no basket-directional beta).
- **Grid:** `L∈{20,30,45,60}d`; selection `{tercile, quartile, k=3,4,5}`; `R∈{7,14}d`; weight `{equal, inv-vol}`; cash-out EMA `{none,100,200}`.
- **Timeframe:** daily signal, weekly/biweekly rebalance. Separate cross-sectional/rotation backtest script (acceptable per your engine note).
- **Ensemble role:** Defensive long-only core that over-weights the calmest coins and *selects away* from the high-beta names TSMOM/EMA-cross pile into — avoids whipsaw in chop, holds the least-declining in bear.
- **Watch:** **Split-sample post-2021 only** (sign flips pre-2021). Search evidence also flags that crypto anomalies "thrive after bull markets, gains insignificant in bear" — so treat this as *drawdown-cushioning*, not bear alpha (the bear alpha is #1 and #3). Measure realized OOS P&L correlation vs. TSMOM inside the fold, not just standalone Sharpe.

---

## #3 — Regime-Gated Short-Term Mean Reversion (RSI2 / lower-band fade)
**Highest structural chop-diversification; the only new sleeve besides #2 that also helps the Upbit book; drops straight into the existing single-symbol Strategy interface.**

- **Venue/symbols:** Binance BTC/ETH USDT **and Upbit BTC/ETH KRW** (extend to SOL/ADA only if cost supports), on 1h/4h.
- **Rules (long-only):** Enter only when **both** gates true — (A) daily `ADX(14)<20` OR price within ±1.5·ATR of daily EMA200; (B) `shortRV/longRV < 1.0` (no vol expansion). Directional context: daily close > EMA200 (buy dips inside a broad up-range — highest base-rate case). Trigger: `RSI(2)<5` (or close < lower BB(20,2.5–3.0)). Exit: `RSI(2)>50` OR close>SMA(5) OR ATR target OR time-stop. Close → next open.
- **Grid:** tf `{1h,4h}`; RSI period `{2,3}`, entry `{3,5,10}`, exit `{45,50,60}`; BB `(20, 2.5/3.0)`; ADX gate `{18,20,22}`; EMA200 band `{1.0,1.5,2.0}·ATR`; time-stop `{6,12,24}` bars; target `{1.0,1.5,2.5}·ATR`. Enforce a **minimum trades/window** floor.
- **Ensemble role:** Chop alpha whose active windows are ~the complement of the trend book's → low/negative return correlation by design.
- **Evidence / honesty:** This is a **validate-tier** candidate, not a proven net winner. Ungated intraday MR *loses* (documented −16.88%/6mo at 66% hit-rate); the gate is the whole edge (BB MR profit factor ~1.62 at ADX<20 vs −0.74 at ADX>30). We previously rejected *ungated* BB+RSI (too few trades) — the differentiator here is the dual regime gate + dip-in-confirmed-uptrend. **Must clear 0.3% RT with a min-edge-per-trade filter** or it dies like the earlier rejects.

---

## #4 — ETH/BTC Relative-Value Tilt (mean-reversion-led, long-only spot form)
**The lowest-beta diversifier — a ratio, so it can pay when the market is flat/falling — but sized modestly given the shallow 2026 rotation regime.**

- **Rules:** Build the ETH/BTC series from Binance ETHBTC daily candles directly. **Mean-reversion-led** (see regime note): `z=(ratio − rollmean_M)/rollstd_M`; fade `|z|>z_entry` toward `z_exit`. Optional trend base: tilt to ETH when ratio > EMA(N) and rising. **Long-only-spot expression:** a dynamic ETH:BTC allocation split (bounds 30/70 ↔ 70/30) — no shorting. Take the ETH-overweight leg only when total market cap is rising (avoids over-firing).
- **Grid:** `M∈{90,120,180}d`; `z_entry∈{1.5,2.0,2.5}`; `z_exit∈{0,0.5}`; trend `EMA N∈{30,50,100}`; tilt bounds `{30/70, 20/80}`; dominance/mktcap confirm `{on,off}`.
- **Timeframe:** daily; low turnover (rebalance on signal change). Separate 2-asset backtest script.
- **Ensemble role:** Near-zero beta to total-crypto direction; P&L driver is the ETH-vs-BTC spread, uncorrelated with absolute-price trend sleeves.
- **Regime reality (verified today):** BTC.D ~57–60%, ETH/BTC ~0.031 (well below the ~0.07 altseason threshold), Altcoin-Season Index ~46 (<75), ETF flows concentrate in BTC → **rotations are shallow and the trend mode will rarely fire**. Lean on the mean-reversion mode, size small. WFA watch: use a *rolling* mean and confirm the reversion half-life is stable across folds before trusting entries (ratio non-stationarity is the failure mode).

---

## #5 — Realized-Vol Term-Structure Regime Overlay (+ trend-quality gross scaler / cash rotation)
**Not a standalone alpha — the allocator/enabler that gates #3, cuts trend-book gross into vol-expansion crashes, and rotates freed capital to stablecoin yield. Judge on MDD/Calmar and correlation, not Sharpe.**

- **Rules (pure OHLCV):** `shortRV(5–10d)` and `longRV(30–60d)` via Parkinson/Garman-Klass; `ratio = shortRV/longRV`, EMA-smoothed with hysteresis. `ratio > 1.15–1.35` → RISK-OFF (cut book gross, keep trend book only, disable #3/#4-reversion); `ratio < 0.7–0.9` → RANGING (enable #3, scale trend down). Layer a Kaufman-ER/ADX trend-quality gross scaler with cash-to-stablecoin rotation (haircut APY toward 0 — depeg/counterparty risk).
- **Grid:** shortRV `{5,10}`; longRV `{30,60}`; estimator `{Parkinson, GK}`; risk-off `{1.15,1.25,1.35}`; ranging `{0.7,0.8,0.9}`; smooth EMA `{2,3,5}`.
- **Ensemble role:** Reshapes the exposure envelope — concentrates its effect in the tails/regimes where TSMOM and EMA-cross take their worst hits. It is the master switch that makes #3 safe to run.
- **Watch:** overlaps partially with your existing vol-targeting/risk-halt (commit 6d51c36) — validate that it adds *incremental* Calmar/left-tail reduction, not a re-labeled version of what you have. Add hysteresis to prevent regime-flag whipsaw + next-open slippage.

---

## Explicitly OVERHYPED / do NOT prioritize

- **Cross-sectional short-term/weekly reversal on a liquid majors basket** — pitched by *two* researchers, but 2024–2025 evidence I verified says the opposite for our universe: *"the largest and most tradeable coins exhibit daily momentum rather than reversal"* — reversal is an illiquid small-cap / perp-liquidation-cascade artifact. On a 4–6 name liquid basket with 100%+ weekly churn at our 0.3% RT, it very likely fails WFA. **Our regime-gated single-symbol MR (#3) is the correct vehicle for the "mean-revert in chop" thesis, not cross-sectional reversal on majors.**
- **Long-only top-k cross-sectional momentum** — the cross-sectional researcher honestly self-flags it as low-priority; it's a TSMOM sibling (high positive correlation), −2.35%/yr OOS in the Starkiller study, and the BTC-EMA gate that tames its drawdowns just re-imports our existing trend beta. **Reject as a diversifier.**
- **Delta-neutral funding carry** — the "8–20% APY / Sharpe 2.2–2.4" headlines are stale/optimistic; basis compressed 25%→<5% (93% of days below the ~5% breakeven), carry Sharpe went 6.45→4.06→negative (2025), and it requires shorting perps (not long-only-spot). Use funding as a *signal* (#1), not as carry. Defer to a conditional Phase-2 perp sleeve that only runs when 7d funding annualizes >~8–10%.
- **Market-neutral (long-short) XS momentum on perps** — the >1 long-short Sharpe is fragile; the short leg blows up on violent loser rebounds (Han/Kang/Ryu). Perp-only, deferred.
- **Perp short-side trend sleeve** — legitimately corrects the "funding is a cost" error (a short *receives* funding ~92% of the time), but literature confirms it doesn't raise Sharpe (drift + squeezes offset the credit); it only trims bear MDD. Keep as small, gated Phase-2 drawdown insurance once perps are live — not a top-5 NOW pick.
- **OI-price divergence filter** — **disqualified from NOW by validation gate (d):** Binance `openInterestHist` REST serves only ~30 days, so it cannot be walk-forward validated; paper-trade/forward-log only.
- **Squeeze / NR7 / "Monday Asia open" seasonality** — breakout/trend-correlated with the book you already have (and with the rejected Donchian / Larry-Williams breakouts); no diversification.

---

## Cross-cutting WFA requirements for all five
1. Measure realized OOS **P&L correlation** of each new sleeve vs. the live TSMOM and EMA-cross sleeves *inside the fold* — reject anything that looks like a trend sibling regardless of standalone Sharpe.
2. Demand a **parameter plateau** (same neighborhood works across folds), not a lone peak — consistent with the multiple-testing fixes in commit 6d51c36.
3. Model per-name maker/taker + slippage on next-open fills; capacity set by the weakest coin in any basket (#2/#4). Enforce min-trade-count and min-edge-per-trade floors (#3 especially).
4. Point-in-time universe + delisted coins priced in (#2) — no 2026-hindsight basket.
5. Realized-only funding, aligned to next-open fill (#1); rolling (not fixed) reversion mean (#4).

Sources: [Revisiting the low-vol anomaly (SSRN/ScienceDirect S1544612326003818)](https://www.sciencedirect.com/science/article/abs/pii/S1544612326003818); [Cryptocurrency anomalies and economic constraints (S1057521924001509)](https://www.sciencedirect.com/science/article/abs/pii/S1057521924001509); [Up or down? Short-term reversal, momentum, liquidity (S1057521921002349)](https://www.researchgate.net/publication/354881251_Up_or_down_Short-term_reversal_momentum_and_liquidity_effects_in_cryptocurrency_markets); [Emerald four-factor cross-sectional model](https://www.emerald.com/cafr/article/27/4/493/1271913/Unravelling-cross-sectional-patterns-in); [Two-tiered funding-rate structure (MDPI 14/2/346)](https://www.mdpi.com/2227-7390/14/2/346); [Market-neutral in crypto (TradingView Hub)](https://www.tv-hub.org/guide/market-neutral-strategy-crypto); [Funding-rate analysis guide 2026 (Zipmex)](https://zipmex.com/blog/how-to-analyze-funding-rates-in-crypto/); [Bitcoin dominance cycles 2017–2026 (SimianX)](https://www.simianx.ai/stories/bitcoin-dominance-cycles-when-altseason-starts-2017-2026); [Altseason signals 2026 (TechJuice)](https://www.techjuice.pk/altcoin-season-signals-crypto-2026-bitcoin-dominance-eth-btc/).