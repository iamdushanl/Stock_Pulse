"""
Stockpulse Phase 2B – Full CSE Feature Quality, Market Context & Training Readiness Audit
=======================================================================================
Audits Phase 1B and Phase 2A datasets without mutating upstream files or training models.
"""

from .integrity_and_coverage import audit_dataset_integrity, audit_coverage_and_missingness
from .liquidity_and_tradability import audit_liquidity_and_tradability
from .market_ca_redundancy import (
    validate_market_relative,
    validate_corporate_actions,
    audit_feature_redundancy,
)
from .survivorship_and_readiness import (
    audit_survivorship_and_universe,
    assess_strategy_readiness,
    generate_visualizations,
)

__all__ = [
    "audit_dataset_integrity",
    "audit_coverage_and_missingness",
    "audit_liquidity_and_tradability",
    "validate_market_relative",
    "validate_corporate_actions",
    "audit_feature_redundancy",
    "audit_survivorship_and_universe",
    "assess_strategy_readiness",
    "generate_visualizations",
]
