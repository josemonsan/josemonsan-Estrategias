# Cava v3 — Real-Data Backtest Findings

**Date:** 2026-09-17
**Scope:** Ran `cava_trend_strategy_v3.py` beyond its built-in synthetic demo, against real market data, to see whether any part of the engine shows a real edge. All runs used the frozen default `CavaConfigV3()` except for the specific toggles noted per test — no parameter fitting was done anywhere in this session beyond enabling/disabling entry modules.

**Bottom line:** almost nothing tested here survived a change in universe or time window — most apparent edges reversed or decayed when re-tested under a different (but equally reasonable) condition. One partial exception looked promising at first: `macd_daily_cross` isolated on a large universe held up (non-negative) across a 3-fold walk-forward. A same-day follow-up (§8) sliced that result more finely and found it's actually a choppy alternation of sharp winning/losing half-years that happens to net near flat-to-positive over 2020–2026, on trade counts (2–18 per half-year) too thin to call it a real edge rather than noise. Treat the strategy as **not validated** for live trading based on this session's work.

---

## Setup

- **Environment:** the sandbox had no Python data-science stack installed. Installed `python3-pandas`, `python3-numpy`, `python3-pip`, `python3.14-venv` via `apt`, plus `yfinance` in a scratch virtualenv (not added to this repo or its dependencies).
- **Price data:** Yahoo Finance daily OHLCV, split/dividend-adjusted (`yfinance`, `auto_adjust=True`).
- **Macro data:** FRED series `BOGMBASE` (USD monetary base, monthly) — the only macro column the engine actually reads (`macro_regime()`, column default `"monetary_base"`).
- **Date range:** 2015-01-01 through 2026-09-17 (today).
- **Scripts used:** ad hoc, written to the session scratchpad only (`run_real_data.py`, `run_walkforward.py`, `run_macd_isolated.py`, `run_macd_isolated_walkforward.py`) — **not committed to this repo**. Not reproducible from the repo alone; regenerate on request if needed.

---

## 1. Synthetic self-test (sanity check, built-in `main()`)

Before touching real data, ran the script's own synthetic demo to confirm the pipeline is unbiased:

- Self-tests: divergence causality OK, SAR monotonicity OK.
- **Control negativo** (pure random walk, 963 trades): Sharpe -0.19, expectancy ≈ 0, PSR 0.31 — correctly shows no exploitable signal in noise.
- **Control positivo** (planted 120-day drift regimes, 941 trades): Sharpe +0.08, PSR 0.64 — engine extracts *some* of the planted structure, weakly.

No bugs or obvious bias detected in the mechanics. This only validates plumbing, not the strategy.

## 2. Real data, all modules on, 13–14 large-cap tickers

Universe: `AAPL MSFT AMZN GOOGL META NVDA JPM XOM JNJ PG UNH HD DIS KO` (XOM intermittently failed to download via `yfinance` — 13 or 14 tickers depending on the run).

Default config, all six entry modules enabled, 2015–2026:

| Metric | Value |
|---|---|
| Trades | 1,519 |
| CAGR | -4.53% |
| Sharpe | -0.51 |
| Max Drawdown | -44.6% |
| PSR (SR*=0) | 0.05 |
| Win rate / payoff | 29.8% / 2.38 |

By sub-period, only 2020-11→2023-10 was positive (Sharpe +0.62); the other three quartiles were all negative, one with a -30.4% drawdown segment.

## 3. Per-module breakdown (same run as §2)

| Module | Trades | Expectancy/trade | Share of closed PnL |
|---|---|---|---|
| `trap_reversal` | 1,415 | +0.01% | **103.0%** (i.e. accounts for the entire net loss and then some) |
| `rsi_macd_weekly` | 29 | -0.23% | +2.8% |
| `macd_daily_cross` | 70 | +0.26% | -1.8% |
| `adx_ema20_pullback` | 5 | -0.01% | -4.0% |

`trap_reversal` (the BASE/CURSO-derived false-breakout module, not from the book) generates ~93% of all trades and is responsible for essentially the entire portfolio loss. The other three modules combined were roughly flat. Longs (457 trades) had positive expectancy (+0.73%); shorts (1,062 trades, mostly `trap_reversal`) were negative (-0.30%).

## 4. `trap_reversal` disabled

Same 14-ticker universe, `enable_trap_reversal=False`, all other modules unchanged:

| Metric | All modules ON | `trap_reversal` OFF |
|---|---|---|
| Trades | 1,543 | 160 |
| CAGR | -4.77% | **+1.11%** |
| Sharpe | -0.53 | **+0.25** |
| Max Drawdown | -46.9% | **-13.9%** |
| PSR (SR*=0) | 0.04 | **0.83** |

Per-module with `trap_reversal` off: `macd_daily_cross` (107 trades, +0.52%/trade) drove 70.7% of the (now positive) closed PnL; `rsi_macd_weekly` (47 trades, -0.15%/trade) was a drag; `adx_ema20_pullback` (6 trades) too small to read. Sub-period Sharpe: +0.58, +0.04, +0.79, **-0.51** (2023–2026) — the fix is not uniform across time even here.

## 5. Walk-forward validation of the `trap_reversal` toggle

Three expanding-window folds. The *only* decision made on train data: `enable_trap_reversal` True vs False, picked by train-period Sharpe, then applied blind to the following out-of-sample window.

| Fold | Train winner | OOS window | OOS Sharpe | OOS trades | OOS expectancy |
|---|---|---|---|---|---|
| 1 | OFF (train Sharpe -0.65 vs +0.33) | 2020–2021 | **+0.97** | 31 | +1.74% |
| 2 | OFF (train Sharpe -0.36 vs +0.52) | 2022–2023 | -0.11 | 19 | +0.28% |
| 3 | OFF (train Sharpe -0.41 vs +0.42) | 2024–2026 | **-0.51** | 35 | -0.72% |

Concatenated OOS (all three windows, no overlap): CAGR +0.69%, Sharpe **+0.18**, Max Drawdown -7.46%.

`trap_reversal=OFF` won every train fold (consistent, not a coin flip) — but the resulting OOS performance **degrades monotonically**: strong in 2020–21, flat in 2022–23, negative in 2024–26. Per-fold trade counts (19–35) are too thin to separate "regime change" from noise. This also isn't a fully independent test: the toggle itself was originally discovered by looking at the full 2015–2026 sample, so even fold 1's 2015–2019 training window already reflects knowledge of how the strategy performs in a 2020-shaped world.

## 6. `macd_daily_cross` isolated, larger 43-ticker universe

Motivation: §4's positive read on `macd_daily_cross` came from a hand-picked 14-name universe. Retested it alone (all other entry modules off) on a 43-ticker universe spanning tech, communications, staples, healthcare, financials, industrials, and energy, letting the engine's built-in weekly `top_n_candidates=10` ranking (by ADX) rotate which names are actually eligible — this is the "rolling universe" mechanism already built into the code (`cava_trend_strategy_v3.py:1052`).

| Metric | 14-ticker universe | 43-ticker universe |
|---|---|---|
| Trades | 107 | 175 |
| Expectancy/trade | +0.52% | **-0.17%** |
| Sharpe | (part of mixed run) | **-0.11** |
| Win rate | 39% | 30% |
| PSR (SR*=0) | — | 0.38 |
| Max Drawdown | — | -20.5% |

The edge **fully reverses** on the larger universe. Sub-period Sharpe: -0.50 (2015–17), -0.94 (2017–20), +0.34 (2020–23), +0.40 (2023–26) — negative in the first half of the sample, weakly positive since 2020. Per-ticker PnL was scattered (best: ADBE, CRM, NKE, MSFT, BA; worst: CMCSA, DIS, COST, AVGO, WFC), so this isn't one outlier ticker driving the flip.

## 7. Walk-forward validation of `macd_daily_cross` isolated, 43-ticker universe

Same fold structure as §5, applied to the isolated `macd_daily_cross` config (all other entry modules off) on the 43-ticker universe. No decision is fit here — a single fixed config — so train-window metrics are shown only for reference, not used to pick anything.

| Fold | Train Sharpe (ref. only) | OOS window | OOS Sharpe | OOS trades | OOS expectancy |
|---|---|---|---|---|---|
| 1 | -0.71 | 2020–2021 | +0.05 | 36 | +0.17% |
| 2 | -0.46 | 2022–2023 | +0.30 | 30 | -0.34% |
| 3 | -0.24 | 2024–2026 | +0.30 | 36 | +0.55% |

Concatenated OOS (all three windows, no overlap): CAGR +1.06%, Sharpe **+0.22**, Max Drawdown -8.63%.

Unlike §5's `trap_reversal` walk-forward, which decayed monotonically from Sharpe +0.97 to -0.51 across folds, this one is **flat-to-mildly-positive and non-decaying**: every OOS fold is ≥0, and the worst fold (+0.05) is still non-negative. All three train windows were negative (dragged down by the weak 2015–2020 stretch seen in §6), but none of that shows up in any post-2020 OOS window — the concatenated OOS Sharpe (+0.22) is marginally better than `trap_reversal`'s (+0.18) and, more importantly, comes without the decay pattern. Per-fold trade counts (30–36) remain small, and the OOS windows only start in 2020, so this result doesn't contradict §6's negative 2015–2020 sub-periods — it just doesn't have to face them.

---

## Overall conclusion

Across seven tests, every apparent edge was universe- or window-dependent, with one partial exception:

- `trap_reversal` looks catastrophic on the 14-ticker universe (responsible for >100% of the net loss) — a fairly robust finding in itself.
- Removing it turns the aggregate result strongly positive (Sharpe 0.25, PSR 0.83) **in-sample**, but that same decision, walk-forward validated, decays from Sharpe +0.97 to -0.51 across three sequential OOS windows.
- `macd_daily_cross` — the module with the cleanest positive read in-sample on the small universe — reverses sign entirely on a 3x larger, more sector-diverse universe (§6).
- That same isolated `macd_daily_cross` config, walk-forward validated on the large universe (§7), is the one test in this session that did **not** decay or reverse: OOS Sharpe was non-negative in all three folds (+0.05, +0.30, +0.30), concatenated Sharpe +0.22. It's the mildest positive signal found, not a strong one — three folds of ~30 trades each is still a thin sample, and it owes its positive read partly to the OOS windows starting only in 2020, after the module's weak 2015–2020 stretch.

None of this rules out that some sub-piece of Cava v3 has real edge; it means this session found at most one candidate (`macd_daily_cross`, isolated, on a broad universe, post-2020) that survived a basic robustness check — and even that one is thin evidence, not confirmation. The strongest, most consistent *negative* signal across every test remains structural: `trap_reversal` is a persistent drag and is the one component worth reconsidering or redesigning first.

## Caveats / limitations

- No costs beyond the engine's built-in 2.5bps/side and default risk sizing (0.5%/trade, 5% vol target, max 10 concurrent positions, 2x max gross leverage) were varied — this session only toggled entry-module on/off flags.
- Universes were hand-picked large/mid-cap US equities, not a point-in-time, survivorship-bias-free index membership list.
- FRED `BOGMBASE` was used as the sole macro input; `M1`/`M2` (present in the synthetic macro generator) were never wired in because the engine's default `macro_regime()` only reads `monetary_base`.
- Per-fold and per-module OOS trade counts were often in the 5–40 range — too small for tight statistical confidence; treat Sharpe/expectancy figures at that scale as noisy point estimates.
- All analysis scripts live only in this session's scratchpad and are not part of the repo.

## Pine Script port (`pine/cava_trend_strategy_v3.pine`)

Added a single-symbol Pine v6 port of the entry/exit logic in `cava_trend_strategy_v3.py`, for visual bar-by-bar trade inspection on a TradingView chart, with each indicator and entry module independently toggleable.

Deliberate differences from the Python engine:

- Single symbol, single position at a time — the Python PORTFOLIO layer (`top_n_candidates` ranking across tickers, vol targeting, gross-leverage cap) is not replicated; position size uses simple fixed risk-per-trade sizing instead.
- `v2_confluence` and the seasonality module are not ported (both OFF by default in the Python config, never enabled in this session's testing, and `v2_confluence` is legacy/deprecated per the source docstring). Everything else — `rsi_macd_weekly`, `macd_daily_cross`, `adx_ema20_pullback`, `cci_threshold`, `trap_reversal`, all gates, both stop types, all three trailing modes — is ported.
- The macro (USD monetary base) gate is optional and OFF by default; it needs a monthly external data series (e.g. `FRED:BOGMBASE` on TradingView, plan-dependent).
- Higher-timeframe (weekly/monthly) values use `request.security()` with lookahead explicitly off, mirroring the Python resample+ffill logic — only ever uses the last completed weekly/monthly bar.

## 8. Follow-up — finer time-slicing of `macd_daily_cross` isolated (same day, 2026-09-17)

Re-ran the isolated `macd_daily_cross` config (43-ticker universe, all other entry modules off, same as §6–7) with a fresh data pull, then sliced the resulting single fixed-config equity curve at finer granularity than the three ~2-year walk-forward folds, per the suggested next step above. No new fitting decision is made anywhere in this section — it's the same one candidate config from §7, just inspected at finer resolution. 174 trades over 2015–2026 (vs. 175 in §6; trivial difference from a fresh Yahoo Finance pull on a different day).

**Quarterly slices, full 2015–2026 history:** Sharpe by quarter swings wildly and has no visible trend — e.g. 2015Q2 -3.25, 2016Q2 +2.68, 2018Q3 -2.59, 2021Q4 +3.67, 2023Q3 +1.75, 2025Q4 -3.00, 2026Q1 +2.45. Several quarters have 0–2 trades, so many of these point estimates are single-trade noise, not signal.

**Half-year slices, 2020–2026** (the period the §7 walk-forward covered):

| Period | Sharpe | Trades | Expectancy/trade |
|---|---|---|---|
| 2020 H1 | -1.87 | 3 | -1.27% |
| 2020 H2 | -0.39 | 6 | -0.47% |
| 2021 H1 | -1.73 | 9 | -1.13% |
| 2021 H2 | **+1.54** | 18 | +1.27% |
| 2022 H1 | -0.55 | 9 | -0.70% |
| 2022 H2 | -0.77 | 4 | -0.34% |
| 2023 H1 | -0.40 | 5 | -0.74% |
| 2023 H2 | **+1.50** | 12 | +0.10% |
| 2024 H1 | +0.32 | 13 | +2.03% |
| 2024 H2 | -0.42 | 4 | +0.02% |
| 2025 H1 | -0.27 | 3 | +0.92% |
| 2025 H2 | -1.38 | 5 | -1.49% |
| 2026 H1 | **+1.60** | 9 | -0.05% |
| 2026 Q3 (partial) | -0.48 | 2 | -0.92% |

10 of 14 half-years are negative; the whole-period positive result rests on three standout half-years (2021 H2, 2023 H2, 2026 H1) outweighing a larger number of smaller losing half-years. This is a materially different picture from "flat-to-mildly-positive, non-decaying" — §7's three ~2-year folds happened to land on boundaries that averaged the swings into three modestly-positive numbers; they didn't reveal that the underlying signal is this choppy.

**Rolling 252-trading-day Sharpe (sampled monthly, 2020–2026)** confirms the same thing at continuous resolution rather than at fold boundaries: it swings from -1.20 (mid-2021) up to +1.42 (Sept 2022), back down to -0.63/-0.55 (late 2022/early 2023), up to ~+1.0 (late 2023–mid 2024), decaying to -0.90 (late 2025), then back up to +0.65–0.68 through 2026. Full monthly series saved in this session's scratchpad output, not reproduced in full here — the shape is: no persistent regime lasts much beyond ~9–12 months in either direction.

**Revised read:** the non-decaying 3-fold walk-forward result from §7 is not wrong, but it understates how noisy the underlying signal is. At finer resolution, `macd_daily_cross` isolated looks less like "a mild, steady edge" and more like a strategy that alternates between sharp multi-month winning and losing streaks that happen to net out close to flat/slightly positive over 2020–2026. Given trade counts of 2–18 per half-year, this is still consistent with the signal being mostly noise around zero rather than a real, exploitable edge — the finer slicing weakens rather than strengthens the case for this being the one candidate worth pursuing.

## 9. Follow-up — pattern mining on `macd_daily_cross` trades, 43-ticker universe (same day, 2026-09-17)

Goal: find an indicator-based filter that improves the module's win rate/expectancy (maximize gains, reduce losses) by attaching a snapshot of 18 other indicators (ADX, ±DI, RSI, MACD histogram level/slope, stochastic, CCI, ATR%, volume ratio, distance from EMA20/50/200, gap%, 5/20/60-day momentum, macro regime, weekday) to each of the 174 trades, measured at the signal bar (bar before `entry_date`, so no lookahead).

**Correlation scan (in-sample, all 174 trades):** every single-feature correlation with `pnl_pct` is weak — the strongest is `atr_pct` at Spearman -0.19 (all trades) / -0.20 (shorts only). Nothing clears ~0.2. This is consistent with §8: there just isn't a strong linear or rank relationship between any tested indicator and trade outcome.

**Honest out-of-sample check of the best in-sample candidate:** split trades at 2021-12-31 (train n=108, test n=66). The best train-sample feature, `atr_pct` (Spearman -0.30 in train), gave a filter ("keep only below-median ATR% trades") that looked good in train (kept: 33.3% win rate / -0.27% expectancy vs. dropped: 24.1% / -0.54%) — but **inverted completely out-of-sample**: in test, the kept (low-ATR) trades did *worse* (27.8% win rate / -0.88% expectancy) than the dropped ones (33.3% / +0.53%). This is the same trap as §8's fold-boundary illusion — an in-sample correlation on ~100-170 trades is not a reliable filter.

**One theoretically-motivated rule, tested rather than mined:** `macd_daily_cross` requires only a MACD cross plus the weekly/monthly trend gate — it never checks whether the *daily* ±DI already agrees with the trade direction (that check exists only in the separate `adx_ema20_pullback` module). Added it as an extra confirmation (long only if `pdi > mdi`, short only if `mdi > pdi`) and tested it directly (no threshold was fit, so no train/test split is needed to avoid look-ahead bias in the rule itself):

| Scope | Baseline win rate | Baseline expectancy | DI-confirmed win rate | DI-confirmed expectancy |
|---|---|---|---|---|
| All trades (n=174) | 29.9% | -0.20% | 42.2% (n=64) | -0.28% |
| Longs (n=119) | 30.3% | -0.12% | 38.6% (n=44) | -0.25% |
| Shorts (n=55) | 29.1% | -0.37% | 50.0% (n=20) | -0.33% |
| Train ≤2021 (n=108) | 28.7% | -0.41% | 45.0% (n=40) | -0.20% |
| Test >2021 (n=66) | 31.8% | +0.14% | 37.5% (n=24) | -0.41% |

The win-rate lift is real and consistent — it holds in both the train and test halves and for both trade directions (roughly +10 to +21 points). But **expectancy does not improve alongside it**, and in the test half it gets worse. This filter changes the *shape* of outcomes (more, smaller wins) without adding edge — a classic win-rate-vs-expectancy trap, not a usable profit filter.

**Exit-reason breakdown** (a mechanism-level finding, not a mined correlation, so it doesn't need the same OOS skepticism): of 174 trades, `stop_gap` exits (20 trades, 11%) lost far more than any other bucket — mean -2.18%, only 15% win rate — versus `stop` exits (126 trades, mean ~0%, 28.6% win rate) and `macd_cross` signal-based exits (28 trades, the best bucket: +0.28% mean, 46.4% win rate). Gap-through-the-stop losses (the position exits at the open, past the intended stop level — usually overnight news/earnings-shaped moves) are a disproportionate share of the module's losses.

**MAE/MFE by outcome** confirms the usual pattern rather than revealing a fixable flaw: winners have a milder average adverse excursion (-1.15%, median -0.60%) and are held ~3x longer (13 bars) than losers (-2.83% MAE, 4 bars). Losers use most of their stop budget before being cut — there's no obvious sign that a tighter stop would spare more losers than it would cut winners short.

**Overall read:** no indicator- or correlation-based entry filter tested here survives out-of-sample scrutiny at this sample size (174 trades, down to ~20-60 per sub-group) — reinforcing, not reversing, §8's downgrade of this module to "not distinguishable from noise." The one concrete, actionable lever this analysis surfaced is about **risk management around overnight gaps**, not signal filtering: `stop_gap` exits are the single worst bucket by a wide margin, which points toward smaller position sizing (or avoiding entries) around known gap-risk events (earnings, macro releases) rather than toward a smarter entry indicator. The DI-confirmation idea is worth keeping only if the goal is reducing the frequency/size of losing streaks (behavioral/psychological), not for raising expectancy — it doesn't do that.

Analysis script (`run_macd_pattern_mining.py`) and the per-trade feature CSV live only in this session's scratchpad, same convention as the earlier scripts — not committed to the repo.

## 10. Follow-up — earnings-date overlap with `stop_gap` losses (same day, 2026-09-17)

§9 flagged `stop_gap` exits (20 of 174 `macd_daily_cross` trades, 11.5%) as the module's single worst bucket by a wide margin (-2.18% mean pnl, 15% win rate) and hypothesized overnight gap risk — plausibly earnings — as the mechanism. Pulled actual earnings-report dates for all 40 tickers that appear in the trade set from Yahoo Finance (`yfinance`'s `get_earnings_dates`, which goes back to 2001-2004 depending on the ticker) and checked overlap two ways: (a) an earnings date within ±2/±5 calendar days of the `exit_date`, and (b) an earnings date falling anywhere inside the trade's actual holding window (`entry_date` → `exit_date`).

**Earnings are over-represented in `stop_gap` exits, but only explain a minority of them:**

| exit_reason | n | earnings during hold | mean pnl_pct |
|---|---|---|---|
| `macd_cross` | 28 | 10.7% | +0.28% |
| `stop` | 126 | 11.1% | 0.00% |
| `stop_gap` | 20 | **25.0%** | -2.18% |

`stop_gap` trades are ~2.3x more likely to have an earnings report land during the holding period than the other two exit types (whose ~11% rate is close to what you'd expect by chance alone, given ~4 earnings/year and typical holds of 5-13 bars). That's a real, mechanistically sensible elevation — earnings gaps are a genuine contributor.

But it's a minor contributor by damage share: of the 20 `stop_gap` trades' combined -43.6 percentage points of pnl, only 5 trades (with earnings during the hold) account for -9.3 points (21%); the other 15 trades — no earnings anywhere near them — account for -34.4 points (79%) (e.g. the WFC/CVX pair exiting 2018-03-01, right in the Feb-2018 VIX-spike gap-down, or NKE 2016-06-29, unrelated to any earnings report). **Most of the `stop_gap` damage is generic overnight/market-wide gap risk, not earnings-specific.**

**Practical implication:** a rule like "skip/flatten `macd_daily_cross` entries if an earnings report is due within the next N days" would be cheap to add and removes a real (if modest, ~1/5) slice of this module's worst-performing bucket — but it would not fix `stop_gap` as a category, because 3 out of 4 of those losses come from gaps with no scheduled earnings involved at all. Reducing position size or tightening initial risk specifically for `macd_daily_cross` (independent of earnings) would address a larger share of the problem than an earnings-calendar filter alone.

Earnings-date cache, the merged trades+earnings CSV, and the analysis script (`run_earnings_overlap.py`) live only in this session's scratchpad, same convention as the earlier scripts.

## 11. Follow-up — gap-risk position sizing rule + 3x leverage test (same day, 2026-09-17)

Implemented the risk-management lever §10 pointed at (size reduction, not an earnings-date veto) directly in `cava_trend_strategy_v3.py`, then tested it combined with raising `max_gross_leverage` from the 2.0x default to 3.0x, per request.

**Code change (committed to the repo, OFF by default — a new untested trial, not a validated recommendation):**
- New `CavaConfigV3` fields: `use_gap_risk_sizing` (bool, default `False`), `gap_risk_buffer_mult` (default 1.25 — shrinks every entry's size by dividing the risk budget by this factor, targeting the ~79% of `stop_gap` damage that isn't earnings-related per §10), `earnings_gap_lookahead_days` (default 3), `earnings_gap_size_mult` (default 0.5 — extra size cut if an earnings report falls within that lookahead of entry, targeting the ~21% that is earnings-related).
- `run_portfolio_backtest()` gained an optional `earnings_dates: dict[str, pd.DatetimeIndex] | None` parameter (backward compatible, defaults to `None`/no-op). When `use_gap_risk_sizing=True`, the risk-cash used to size every `macd_daily_cross` entry is divided by `gap_risk_buffer_mult`, and further divided by `1/earnings_gap_size_mult` if an earnings report is due within the lookahead window.
- Confirmed the change is inert when unused: re-ran the file's own synthetic self-tests (`main()`) and got byte-for-byte the same Sharpe/PSR numbers as the original §1 baseline.

**Four-way comparison, same 43-ticker universe / 2015-2026 window as §6-10 (isolated `macd_daily_cross`, using the real earnings dates pulled in §10):**

| Variant | CAGR | Vol | Sharpe | MaxDD | PSR | avg gross exposure |
|---|---|---|---|---|---|---|
| A) baseline (lev 2.0x, sizing OFF) | -0.58% | 4.40% | -0.13 | -20.5% | 0.35 | 0.097 |
| B) gap sizing ON, lev 2.0x | -0.31% | 3.56% | -0.09 | -17.3% | 0.41 | 0.085 |
| **C) gap sizing ON, lev 3.0x** | -0.21% | 4.24% | -0.05 | -18.3% | 0.46 | 0.091 |
| D) sizing OFF, lev 3.0x only | +0.06% | 5.37% | +0.01 | -20.9% | 0.55 | 0.107 |

**Important caveat that changes the read on "3x leverage":** average gross exposure across all four variants is only ~9-11% of equity — the strategy almost never comes close to even the 2.0x cap in the first place (the vol-target-to-5%-annual mechanism, not the leverage ceiling, is what's actually sizing positions here; `skipped_entries` due to hitting the leverage cap was 3/174 trades in A, and 0 in C/D). **Raising `max_gross_leverage` to 3x mostly does nothing by itself** — compare D to A: Sharpe moves from -0.13 to a barely-positive +0.01, but Vol also jumps (4.4%→5.4%) and MaxDD gets *worse* (-20.5%→-20.9%). That's leverage amplifying an already-near-zero-expectancy signal both ways, landing on the positive side this time by coincidence, not because 3x leverage added anything real.

The gap-risk sizing rule (B, C) is the part that actually helped: it cut MaxDD by 2-3 points and improved Sharpe/PSR versus baseline in every combination, because it uniformly shrinks size (lower vol, lower drawdown) without changing which trades are taken or their %-return (per-trade `pnl_pct`, and the `stop_gap` bucket's own win rate/expectancy, are identical across all four variants — sizing changes dollar risk, not trade outcomes). C (sizing + 3x leverage together) posts the best Sharpe/PSR of the four, but that's the sizing rule's improvement showing through *despite* the extra leverage headroom, not because of it.

**Recommendation:** `use_gap_risk_sizing=True` is a reasonable, low-risk default to adopt for this module — it's a real (if modest) drawdown-and-vol reduction with no downside seen here. Pushing `max_gross_leverage` to 3x specifically is **not supported** by this test as something that adds value on its own (see D) — it only avoided being harmful when paired with the offsetting size cut (C), and even that pairing is still net-negative CAGR/Sharpe. This entire module remains unvalidated per §8-9; a sizing tweak makes the losses smaller, it does not create an edge that isn't there.

## 12. Follow-up — `use_gap_risk_sizing` turned ON by default, walk-forward re-run (same day, 2026-09-17)

Flipped `CavaConfigV3.use_gap_risk_sizing` to `True` by default (was `False`/opt-in since §11). Re-ran the exact §7 three-fold expanding-window walk-forward (isolated `macd_daily_cross`, 43-ticker universe) with this new default and real earnings dates supplied, to see whether the risk reduction seen in §11's single-window test survives proper walk-forward testing rather than just a full-sample comparison.

Sanity check first: reran the file's own synthetic self-test (`main()`, which doesn't pass `earnings_dates`) — runs cleanly, numbers shift slightly from the §1 baseline (as expected, since every position is now sized smaller by default) with no errors.

**Walk-forward, side by side with §7:**

| Fold | OOS window | §7 Sharpe (sizing OFF) | §12 Sharpe (sizing ON) | §7 trades | §12 trades |
|---|---|---|---|---|---|
| 1 | 2020–2021 | +0.05 | **+0.39** | 36 | 38 |
| 2 | 2022–2023 | +0.30 | +0.26 | 30 | 30 |
| 3 | 2024–2026 | +0.30 | +0.30 | 36 | 36 |
| Concatenated | — | +0.22 | **+0.32** | 102 | 104 |

Concatenated OOS MaxDD also improved, -8.63% → -6.83%. Folds 2 and 3 land on the same trade sets as §7 (sizing doesn't change which signals fire, only position size, so per-trade %-returns match almost exactly) — fold 2's Sharpe moved slightly (0.30→0.26) purely from a lower portfolio-vol denominator interacting with the vol-targeting mechanism. Fold 1 is the one real difference: 2 extra trades got admitted (smaller positions leave more room under the leverage cap before `skipped_entries` kicks in, same mechanism seen in §11), and those 2 trades happened to be profitable, pulling fold 1's OOS Sharpe from +0.05 to +0.39.

**Read:** the walk-forward result is now uniformly positive and slightly better everywhere, not just in the single-window §11 comparison — this is a genuine (if still modest) confirmation that the sizing rule's risk reduction holds up under the same fold structure that was used to validate the module in the first place, rather than being a full-sample-only artifact. It does not change the underlying verdict from §8-9: sample sizes remain thin (30-38 trades/fold), the edge (to the extent trade selection differs at all) is not from a smarter filter, and this whole module is still not validated for live trading — it is simply smaller losses/drawdown wrapped around the same signal.

## Suggested next steps

- If pursuing `trap_reversal` further: understand *why* it loses (concentrated in a few large losers vs. broad-based, per the max 22–23-trade losing streaks seen in every run) before deciding to fix, gate more tightly, or remove it.
- §8 downgrades `macd_daily_cross` from "one candidate that survived robustness checks" to "noisy, cyclical, and not distinguishable from zero-edge at finer resolution given trade counts this small." Before pursuing it further, a real significance test (e.g. bootstrap on trade-level returns, or comparing against random entry timing with the same holding-period distribution) is needed — quarter/half-year Sharpe eyeballing is not enough at this sample size.
- §9 tried to rescue `macd_daily_cross` with indicator-based entry filters and failed to find one that survives out-of-sample; the only real lever found was risk management around overnight gap-through-stop losses (`stop_gap` exits).
- §10 checked that lever against real earnings dates: earnings only explain ~1/5 of `stop_gap`'s total damage (2.3x over-represented but a minority contributor); an earnings-avoidance filter is cheap and worth adding but won't fix the category — the bulk of `stop_gap` risk is generic overnight gaps unrelated to earnings, better addressed via position sizing/risk limits than a calendar filter.
- §11 implemented that sizing rule (`use_gap_risk_sizing`, off by default at the time) and tested it with 3x leverage: the sizing rule genuinely trims MaxDD/vol; 3x leverage alone does not help (avg gross exposure never got close to even the 2x cap, so the leverage ceiling was rarely the binding constraint) — the module is still net-negative on every variant tested.
- §12 turned `use_gap_risk_sizing` ON by default and re-ran the §7 walk-forward: OOS Sharpe improved in every fold (concatenated +0.22 → +0.32, MaxDD -8.63% → -6.83%), confirming the risk reduction holds under walk-forward, not just a full-sample comparison. Still not an edge — smaller losses on the same signal, same thin sample-size caveats as §8-9.
- Consider point-in-time index constituents (e.g. historical S&P 500 membership) instead of a hand-picked, survivorship-biased ticker list, to remove one more degree of freedom from these results.
