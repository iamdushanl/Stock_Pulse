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
]
