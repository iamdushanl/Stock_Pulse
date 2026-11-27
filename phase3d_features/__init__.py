"""
Stockpulse Phase 3D – Unified Point-in-Time Fundamental Integration, Alignment & Strategy Validation
====================================================================================================
Reusable point-in-time historical integration package combining Phases 1B, 2A, 2B, 2C, 3B, and 3C
with strict temporal causality (`AvailabilityDate <= DecisionDate`, `PriceDate <= 2025-12-31`),
multi-class identity hygiene (`N` & `X` ordinary share classes), business-type enforcement
(`NON_FINANCIAL_CORPORATE`, `BANK_FINANCE`, `INSURANCE`), and strategy-level validation.
"""

from .contract_and_mapping import (
    UPSTREAM_CORE_PATHS,
    audit_upstream_data_contracts,
    build_company_security_pit_mapping,
    compute_file_sha256,
    verify_upstream_immutability,
)
from .asof_and_valuation_engine import (
    DEFAULT_MAX_CARRY_DAYS,
    DEFAULT_STALE_DAYS,
    MAX_HISTORICAL_PRICE_DATE,
    STALENESS_THRESHOLDS_DAYS,
    build_causal_corporate_actions,
    build_company_fundamental_pit_snapshot,
    build_historical_valuation_validation,
    build_security_shares_and_dividend_index,
    retrieve_integrated_asof,
)
from .strategy_integration_and_audit import (
    build_all_audits_and_readiness,
    build_phase3d_visualizations,
    build_strategy_integrated_universes,
    run_causal_injection_suite,
    run_phase3d_pipeline,
)

__all__ = [
    "UPSTREAM_CORE_PATHS",
    "audit_upstream_data_contracts",
    "build_company_security_pit_mapping",
    "compute_file_sha256",
    "verify_upstream_immutability",
    "DEFAULT_MAX_CARRY_DAYS",
    "DEFAULT_STALE_DAYS",
    "MAX_HISTORICAL_PRICE_DATE",
    "STALENESS_THRESHOLDS_DAYS",
    "build_causal_corporate_actions",
    "build_company_fundamental_pit_snapshot",
    "build_historical_valuation_validation",
    "build_security_shares_and_dividend_index",
    "retrieve_integrated_asof",
    "build_all_audits_and_readiness",
    "build_phase3d_visualizations",
    "build_strategy_integrated_universes",
    "run_causal_injection_suite",
    "run_phase3d_pipeline",
]
