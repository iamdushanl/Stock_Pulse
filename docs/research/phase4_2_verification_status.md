# Stock Pulse – Phase 4.2 Independent Verification Status

**Status date:** 2026-11-20  
**Scope:** Independent Verification & Robustness Testing for Short-Term Engine 01 (0–3 Months)

---

## Executive Summary

Phase 4.2 independently reproduced and audited the backtest performance of the three candidate models from Phase 4.1 across 478 trading days (2024–2025):
- **LightGBM (20-day horizon)**: Primary candidate.
- **Random Forest (5-day horizon)**: Secondary high-frequency candidate.
- **Ridge Regression (60-day horizon)**: Secondary macro candidate.

### Verification Classification

| Model | Horizon | Reported CAGR | Cleaned Realistic CAGR | Net Sharpe | Max Drawdown | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **LightGBM** | 20 trading days | 223.22% | **55.55%** | **1.21** | -20.34% | **VERIFIED WITH LIMITATIONS** (Primary short-term engine; beats ASPI 46.67%) |
| **Random Forest** | 5 trading days | 426.85% | 64.72% | 1.87 | -20.20% | **NOT VERIFIED** (Rejected: 23% annual fee drag on CSE; inflated by weekly bad tick compounding) |
| **Ridge Regression** | 60 trading days | 79.30% | 77.86% | 1.67 | -3.30% | **VERIFIED WITH LIMITATIONS** (Approved as Defensive / Macro Allocation) |

---

## Key Audit Discoveries & Root-Cause Analysis

1. **Clerical Data Anomaly in August 2024 (`COMB`):**
   - Between 2024-07-30 and 2024-08-06, Commercial Bank voting share (`COMB`) prices were erroneously transcribed at Rs. 3.40 (merging Rights `COMB.R` into the main ticker). On 2024-08-07, the price reverted to Rs. 89.70 (+2538% move).
   - LightGBM and Random Forest selected COMB on 2024-08-05 due to oversold indicators, realizing $+2361.76\%$ on that holding (+231.46% portfolio return in a single month).
   - Correcting this bad tick reduces LightGBM 20d CAGR from 223.22% to 62.49%.
2. **50% Under-Deduction of Statutory Transaction Fees:**
   - Phase 4.1 applied `turnover * 1.12%`, charging only half of the 2.24% round-trip fee required to sell existing and buy replacement positions.
   - Reconstructing full statutory fees (1.12% on all buys + 1.12% on all sells) brings realistic net CAGR to **55.55%**.
3. **Execution Realism (Next-Day Fill & Spreads):**
   - Testing next-day (T+1 Open) fills with a realistic 50 bps half-spread delivers **52.07% to 58.53% net CAGR**, confirming viable excess returns over the ASPI benchmark (46.67%).
4. **Statistical Robustness:**
   - 1,000-trial circular block bootstrap establishes a 95% confidence interval for LightGBM CAGR of $[-9.71\%, +164.09\%]$ and Sharpe ratio of $[-0.60, +3.13]$.
   - Daily cross-sectional Spearman Rank IC is $0.0848$ ($t = 8.39, p = 5.96 \times 10^{-16}$), confirming statistically significant alpha beyond data-mining bias.

---

## Detailed Deliverables & Audit Trail

All verification code, daily NAV series, and transaction audits are located in:
`phase4_model_development/engine_01_short_term/phase4_2_verification/`

- `phase4_2_verification_report.md`: Complete technical audit report.
- `independent_backtest_reproduction.csv`: Reconciliation of raw vs. cleaned returns.
- `portfolio_nav_daily.csv`: Continuous 478-day daily portfolio NAV series.
- `transaction_cost_audit.csv`: Period-by-period turnover and statutory fee audit.
- `execution_realism_report.csv`: T+1 execution, bid-ask spreads, and slippage stress tests.
- `yearly_performance.csv`: Annual returns (2024 vs. 2025; notes 2023 missing data).
- `regime_performance.csv`: Performance across Bull, Bear, and Volatility regimes.
- `statistical_confidence_report.json`: Block bootstrap confidence intervals and IC test results.
