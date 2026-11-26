"""
Stockpulse Phase 2A – Full CSE Technical Indicator Library
==========================================================
Reproducible, causal, corporate-action-consistent technical feature
engineering across the full Colombo Stock Exchange universe (719 SecurityIDs).
"""

from .metadata_catalog import INDICATOR_METADATA, get_indicator_metadata_df
from .indicators_core import (
    compute_parabolic_sar,
    compute_technical_features_panel,
    compute_market_relative_panel,
)

__all__ = [
    "INDICATOR_METADATA",
    "get_indicator_metadata_df",
    "compute_parabolic_sar",
    "compute_technical_features_panel",
    "compute_market_relative_panel",
]
