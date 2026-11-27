"""
Stockpulse Phase 3C – Fundamental Feature Engineering, Financial Quality, Growth & Valuation Metrics
====================================================================================================
Point-in-time, restatement-aware fundamental feature engineering library for the Colombo Stock Exchange (CSE).
Modules:
  - quality_gating: FX availability audit, unit-scale shift forensics, BS exception classification,
    business-type identification (`NON_FINANCIAL_CORPORATE`, `BANK_FINANCE`, `INSURANCE`).
  - dividends_valuation_ttm: Causal historical share counts (`List 16` + `List 21` + `Phase 1B CA`),
    dividend sustainability (`VALIDATED` vs excluded `% of par` / scrip / split / bonus / rights),
    TTM 4-quarter flows, and strict PIT historical valuation ratios.
  - fundamental_metrics: Growth (w/ loss-turnaround indicator), Profitability & Capital Returns
    (Average Beginning+Ending ROE/ROA/ROIC), Financial Health (corporate vs financial institution rules),
    and Cash Flow / Earnings Quality (`FCF = OCF - abs(CapEx)`).
  - pit_engine_and_audit: Chronological per-company PIT replay engine, `build_fundamental_features_as_of`
    public API, strategy-specific feature matrices (`medium_term`, `long_term`), and quality/redundancy audit.
"""

from .quality_gating import (
    assess_fx_availability,
    build_financial_input_quality_flags,
    classify_business_types,
)
from .dividends_valuation_ttm import (
    HistoricalShareSchedule,
    build_historical_share_schedule,
    compute_ttm_from_active_facts,
    compute_dividends_as_of,
)
from .fundamental_metrics import (
    ALL_FEATURE_COLUMNS,
    MEDIUM_TERM_FEATURE_COLUMNS,
    LONG_TERM_FEATURE_COLUMNS,
    compute_fundamental_snapshot_from_active_facts,
)
from .pit_engine_and_audit import (
    build_fundamental_features_as_of,
    run_phase3c_pipeline,
)

__all__ = [
    "assess_fx_availability",
    "build_financial_input_quality_flags",
    "classify_business_types",
    "HistoricalShareSchedule",
    "build_historical_share_schedule",
    "compute_ttm_from_active_facts",
    "compute_dividends_as_of",
    "ALL_FEATURE_COLUMNS",
    "MEDIUM_TERM_FEATURE_COLUMNS",
    "LONG_TERM_FEATURE_COLUMNS",
    "compute_fundamental_snapshot_from_active_facts",
    "build_fundamental_features_as_of",
    "run_phase3c_pipeline",
]
