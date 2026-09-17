# Cava v3 — Real-Data Backtest Findings

**Date:** 2026-09-17
**Scope:** Ran `cava_trend_strategy_v3.py` beyond its built-in synthetic demo, against real market data, to see whether any part of the engine shows a real edge. All runs used the frozen default `CavaConfigV3()` except for the specific toggles noted per test — no parameter fitting was done anywhere in this session beyond enabling/disabling entry modules.

**Bottom line:** nothing tested here survived a change in universe or time window. Every apparent edge found in one test reversed or decayed when re-tested under a different (but equally reasonable) condition. Treat the strategy as **not validated** for live trading based on this session's work.

---

## Setup

- **Environment:** the sandbox had no Python data-science stack installed. Installed `python3-pandas`, `python3-numpy`, `python3-pip`, `python3.14-venv` via `apt`, plus `yfinance` in a scratch virtualenv (not added to this repo or its dependencies).
- **Price data:** Yahoo Finance daily OHLCV, split/dividend-adjusted (`yfinance`, `auto_adjust=True`).
- **Macro data:** FRED series `BOGMBASE` (USD monetary base, monthly) — the only macro column the engine actually reads (`macro_regime()`, column default `"monetary_base"`).
- **Date range:** 2015-01-01 through 2026-09-17 (today).
- **Scripts used:** ad hoc, written to the session scratchpad only (`run_real_data.py`, `run_walkforward.py`, `run_macd_isolated.py`) — **not committed to this repo**. Not reproducible from the repo alone; regenerate on request if needed.

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

---

## Overall conclusion

Across six tests, every apparent edge was universe- or window-dependent:

- `trap_reversal` looks catastrophic on the 14-ticker universe (responsible for >100% of the net loss) — a fairly robust finding in itself.
- Removing it turns the aggregate result strongly positive (Sharpe 0.25, PSR 0.83) **in-sample**, but that same decision, walk-forward validated, decays from Sharpe +0.97 to -0.51 across three sequential OOS windows.
- `macd_daily_cross` — the module with the cleanest positive read in-sample — reverses sign entirely on a 3x larger, more sector-diverse universe.

None of this rules out that some sub-piece of Cava v3 has real edge; it means this session did not find one that survives a basic robustness check. The strongest, most consistent signal across every test is structural rather than a trading edge: `trap_reversal` is a persistent drag and is the one component worth reconsidering or redesigning first.

## Caveats / limitations

- No costs beyond the engine's built-in 2.5bps/side and default risk sizing (0.5%/trade, 5% vol target, max 10 concurrent positions, 2x max gross leverage) were varied — this session only toggled entry-module on/off flags.
- Universes were hand-picked large/mid-cap US equities, not a point-in-time, survivorship-bias-free index membership list.
- FRED `BOGMBASE` was used as the sole macro input; `M1`/`M2` (present in the synthetic macro generator) were never wired in because the engine's default `macro_regime()` only reads `monetary_base`.
- Per-fold and per-module OOS trade counts were often in the 5–40 range — too small for tight statistical confidence; treat Sharpe/expectancy figures at that scale as noisy point estimates.
- All analysis scripts live only in this session's scratchpad and are not part of the repo.

## Suggested next steps

- If pursuing `trap_reversal` further: understand *why* it loses (concentrated in a few large losers vs. broad-based, per the max 22–23-trade losing streaks seen in every run) before deciding to fix, gate more tightly, or remove it.
- Re-run the `macd_daily_cross`-isolated walk-forward on the 43-ticker universe (mentioned as a next step but not yet done) to see whether its 2020-onward positive stretch is itself robust or another artifact.
- Consider point-in-time index constituents (e.g. historical S&P 500 membership) instead of a hand-picked, survivorship-biased ticker list, to remove one more degree of freedom from these results.
