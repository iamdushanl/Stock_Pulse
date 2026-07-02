"""
CSE EDA Utilities Package
=========================

Centralized utilities for loading Colombo Stock Exchange (CSE) data
and creating publication-quality visualizations.
"""

from .data_loader import (
    load_all_daily_prices,
    load_market_indices,
    load_market_stats,
    load_securities_list,
    load_splits,
    load_listings,
    load_gics,
    load_sector_market_cap,
    load_dividends,
    load_sector_ratios,
)

from .plot_helpers import (
    CSE_COLORS,
    SECTOR_PALETTE,
    CRASH_PERIODS,
    setup_plotting_style,
    format_large_number,
    add_crash_bands,
    plot_time_series,
    plot_dual_axis,
)

from .data_cleaning import clean_and_prepare_data
from .features import generate_technical_features
from .targets import generate_targets
from .model_trainer import prepare_ml_dataset, train_models, evaluate_models
from .recommender import generate_recommendations, multi_horizon_recommendations

__all__ = [
    # Data loaders
    "load_all_daily_prices",
    "load_market_indices",
    "load_market_stats",
    "load_securities_list",
    "load_splits",
    "load_listings",
    "load_gics",
    "load_sector_market_cap",
    "load_dividends",
    "load_sector_ratios",
    # Plot helpers
    "CSE_COLORS",
    "SECTOR_PALETTE",
    "CRASH_PERIODS",
    "setup_plotting_style",
    "format_large_number",
    "add_crash_bands",
    "plot_time_series",
    "plot_dual_axis",
    # Phase 2
    "clean_and_prepare_data",
    "generate_technical_features",
    "generate_targets",
    # Phase 3
    "prepare_ml_dataset",
    "train_models",
    "evaluate_models",
    "generate_recommendations",
    "multi_horizon_recommendations",
]
