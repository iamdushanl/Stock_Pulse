"""
Stockpulse Phase 3B – Historical Financial Time-Series Reconstruction & PIT Fundamental DB
=========================================================================================
Modular pipeline combining:
  - Official CSE 2007–2026 filing discovery & resumable acquisition (`identity_and_discovery`, `historical_acquisition`)
  - Multi-year historical company identity & name-transition reconciliation (`identity_and_discovery`)
  - Enhanced Annual + Interim coordinate & OCR PDF statement extraction (`historical_parser`)
  - Restatement-aware & strict Point-in-Time (`AvailabilityDate`) database & standalone quarterly
    derivation (`timeseries_and_pit`)
  - Historical coverage matrices, quality checks, fundamental metric readiness, and strategy
    readiness evaluation (`quality_coverage_readiness`)
"""

from .identity_and_discovery import (
    discover_and_catalog_filings,
    build_historical_company_identity,
    parse_historical_quoted_names,
)
from .historical_acquisition import (
    acquire_historical_pdfs,
    select_acquisition_targets,
)
from .historical_parser import (
    extract_historical_pdfs,
    repair_and_enhance_date_parsing,
)
from .timeseries_and_pit import (
    reconstruct_historical_database,
    get_financial_facts_as_of,
    get_latest_financial_facts,
)
from .quality_coverage_readiness import (
    build_company_coverage_matrix,
    run_historical_quality_checks,
    assess_fundamental_and_strategy_readiness,
)

__all__ = [
    "discover_and_catalog_filings",
    "build_historical_company_identity",
    "parse_historical_quoted_names",
    "acquire_historical_pdfs",
    "select_acquisition_targets",
    "extract_historical_pdfs",
    "repair_and_enhance_date_parsing",
    "reconstruct_historical_database",
    "get_financial_facts_as_of",
    "get_latest_financial_facts",
    "build_company_coverage_matrix",
    "run_historical_quality_checks",
    "assess_fundamental_and_strategy_readiness",
]
