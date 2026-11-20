# Stock Pulse – Phase 4.1 Research Status

**Status date:** 2026-11-20  
**Scope:** Short-term CSE trading research

## Current status

The team reports completion of Phase 4.1, comparing Ridge Regression, Random Forest, XGBoost and LightGBM for 5-, 20- and 60-trading-day forward-return targets.

Reported horizon candidates:
- 5-day: Random Forest
- 20-day: LightGBM
- 60-day: Ridge Regression

The 20-day LightGBM is the provisional primary candidate. These results have not yet been independently reproduced; they are research findings, not production performance claims.

## Known scope and limitations

- Historical feature panel: 20 S&P SL20 constituents, not the full CSE universe.
- Static November 2026 announcement fields were excluded from historical training after the Phase 3.5 audit.
- 2026 interim fundamental ratios were excluded from historical training due to point-in-time leakage.
- Medium- and long-term models remain blocked pending historical point-in-time financial statements.
- Backtest results require independent reproduction, transaction-cost review and execution realism tests.

## Next milestone: Phase 4.2

1. Reproduce portfolio NAV and CAGR from daily holdings/trades and raw price data.
2. Recalculate costs from actual portfolio weight changes.
3. Test next-day execution, spread, slippage, liquidity, suspensions and price limits.
4. Report performance by year and market regime.
5. Compare against ASPI, equal-weight and simple momentum baselines.
6. Classify each candidate as VERIFIED, VERIFIED WITH LIMITATIONS or NOT VERIFIED.

## Reproducibility checklist

Record the dataset version/hash, code commit, feature schema, target horizon, time splits and purging rules, model parameters, random seeds, portfolio construction rules, transaction-cost assumptions and output NAV series.

No raw CSE data, large Parquet files or model binaries are stored in this documentation update.
