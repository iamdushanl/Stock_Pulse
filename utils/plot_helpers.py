"""
CSE EDA Plot Helpers
====================

Publication-quality plotting utilities for Colombo Stock Exchange
market analysis. Provides consistent styling, color palettes,
and reusable chart components.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# Color Palettes
# ══════════════════════════════════════════════════════════════════════════════

CSE_COLORS = {
    "primary": "#0d1b2a",       # Dark navy
    "secondary": "#1b263b",     # Navy
    "accent1": "#415a77",       # Steel blue
    "accent2": "#778da9",       # Light steel
    "highlight": "#e0e1dd",     # Off white
    "positive": "#2dc653",      # Green (gains)
    "negative": "#ef233c",      # Red (losses)
    "neutral": "#8d99ae",       # Gray
    "gold": "#ffd60a",          # Gold
    "teal": "#00b4d8",          # Teal
    "orange": "#fb8500",        # Orange
    "purple": "#7209b7",        # Purple
    "background": "#f8f9fa",    # Light bg
    "grid": "#dee2e6",          # Grid
    "text": "#212529",          # Dark text
}

# 25 distinct colors for sector-level comparisons
SECTOR_PALETTE = [
    "#1f77b4",  # Muted blue
    "#ff7f0e",  # Safety orange
    "#2ca02c",  # Cooked asparagus green
    "#d62728",  # Brick red
    "#9467bd",  # Muted purple
    "#8c564b",  # Chestnut brown
    "#e377c2",  # Raspberry yogurt pink
    "#7f7f7f",  # Middle gray
    "#bcbd22",  # Curry yellow-green
    "#17becf",  # Blue-teal
    "#aec7e8",  # Light blue
    "#ffbb78",  # Light orange
    "#98df8a",  # Light green
    "#ff9896",  # Light red
    "#c5b0d5",  # Light purple
    "#c49c94",  # Light brown
    "#f7b6d2",  # Light pink
    "#c7c7c7",  # Light gray
    "#dbdb8d",  # Light yellow-green
    "#9edae5",  # Light blue-teal
    "#393b79",  # Dark blue
    "#637939",  # Dark green
    "#8c6d31",  # Dark gold
    "#843c39",  # Dark red
    "#7b4173",  # Dark magenta
]

# ══════════════════════════════════════════════════════════════════════════════
# Market Crash / Event Periods — Sri Lankan Market
# ══════════════════════════════════════════════════════════════════════════════

CRASH_PERIODS = [
    ("2001-06-01", "2002-06-30", "LTTE Airport Attack / Post-9/11"),
    ("2008-01-01", "2009-03-31", "Global Financial Crisis"),
    ("2011-02-01", "2012-12-31", "Post-War Bubble Correction"),
    ("2015-01-01", "2016-06-30", "Political Uncertainty"),
    ("2019-04-01", "2019-12-31", "Easter Sunday Attacks"),
    ("2020-03-01", "2020-06-30", "COVID-19 Pandemic"),
    ("2022-03-01", "2022-12-31", "Sri Lanka Economic Crisis"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Style Setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_plotting_style():
    """
    Set up publication-quality matplotlib / seaborn style.

    Configures:
    - Seaborn whitegrid theme with custom overrides
    - Font sizes for titles, labels, ticks
    - Figure DPI for crisp output
    - Grid appearance
    - Legend styling
    """
    # Use seaborn whitegrid as base
    try:
        sns.set_style("whitegrid")
    except Exception:
        pass

    custom_rc = {
        # ── Figure ──
        "figure.figsize": (14, 6),
        "figure.dpi": 120,
        "figure.facecolor": CSE_COLORS["background"],
        "figure.edgecolor": "none",

        # ── Axes ──
        "axes.facecolor": "white",
        "axes.edgecolor": CSE_COLORS["grid"],
        "axes.linewidth": 0.8,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "axes.labelpad": 8,
        "axes.labelcolor": CSE_COLORS["text"],
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.spines.top": False,
        "axes.spines.right": False,

        # ── Grid ──
        "grid.color": CSE_COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "grid.linestyle": "--",

        # ── Ticks ──
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": CSE_COLORS["text"],
        "ytick.color": CSE_COLORS["text"],
        "xtick.direction": "out",
        "ytick.direction": "out",

        # ── Legend ──
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": CSE_COLORS["grid"],
        "legend.fancybox": True,

        # ── Font ──
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter", "Segoe UI", "Helvetica Neue",
            "Arial", "DejaVu Sans", "sans-serif",
        ],
        "font.size": 11,

        # ── Lines ──
        "lines.linewidth": 1.8,
        "lines.antialiased": True,

        # ── Patches ──
        "patch.edgecolor": "none",

        # ── Savefig ──
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "savefig.facecolor": CSE_COLORS["background"],
    }

    plt.rcParams.update(custom_rc)


# ══════════════════════════════════════════════════════════════════════════════
# Number Formatting
# ══════════════════════════════════════════════════════════════════════════════

def format_large_number(num):
    """
    Format large numbers into human-readable strings.

    Parameters
    ----------
    num : int or float
        Number to format.

    Returns
    -------
    str
        Formatted string, e.g. ``'1.5B'``, ``'230M'``, ``'45K'``, ``'892'``.

    Examples
    --------
    >>> format_large_number(1_500_000_000)
    '1.5B'
    >>> format_large_number(230_000_000)
    '230.0M'
    >>> format_large_number(45_000)
    '45.0K'
    >>> format_large_number(892)
    '892'
    """
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "N/A"

    num = float(num)
    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num >= 1e12:
        return f"{sign}{abs_num / 1e12:.1f}T"
    elif abs_num >= 1e9:
        return f"{sign}{abs_num / 1e9:.1f}B"
    elif abs_num >= 1e6:
        return f"{sign}{abs_num / 1e6:.1f}M"
    elif abs_num >= 1e3:
        return f"{sign}{abs_num / 1e3:.1f}K"
    else:
        if abs_num == int(abs_num):
            return f"{sign}{int(abs_num)}"
        return f"{sign}{abs_num:.1f}"


def large_number_formatter(x, pos):
    """
    Matplotlib tick formatter that uses :func:`format_large_number`.

    Usage::

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(large_number_formatter))
    """
    return format_large_number(x)


# ══════════════════════════════════════════════════════════════════════════════
# Crash / Event Band Overlay
# ══════════════════════════════════════════════════════════════════════════════

def add_crash_bands(ax, alpha=0.15, color=None, label_crashes=False,
                    periods=None):
    """
    Add shaded vertical bands for known market crash / stress periods.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    alpha : float, default 0.15
        Transparency of the shaded bands.
    color : str or None
        Override color for all bands.  Defaults to ``CSE_COLORS['negative']``.
    label_crashes : bool, default False
        If True, add a small rotated label for each event.
    periods : list of tuple or None
        Custom periods ``(start, end, label)``.  Defaults to
        :data:`CRASH_PERIODS`.
    """
    if periods is None:
        periods = CRASH_PERIODS

    band_color = color or CSE_COLORS["negative"]

    for start, end, label in periods:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        # Only draw if the band overlaps with the current x-axis limits
        xlim = ax.get_xlim()
        try:
            ax_start = mdates.num2date(xlim[0]).replace(tzinfo=None)
            ax_end = mdates.num2date(xlim[1]).replace(tzinfo=None)
        except Exception:
            # If x-axis isn't date-based, just draw anyway
            ax.axvspan(start_dt, end_dt, alpha=alpha, color=band_color,
                       zorder=0)
            continue

        if end_dt < ax_start or start_dt > ax_end:
            continue  # Band is outside visible range

        ax.axvspan(start_dt, end_dt, alpha=alpha, color=band_color,
                   zorder=0, label=None)

        if label_crashes:
            mid = start_dt + (end_dt - start_dt) / 2
            ypos = ax.get_ylim()[1] * 0.95
            ax.text(
                mid, ypos, label,
                rotation=90, va="top", ha="center",
                fontsize=7, alpha=0.6,
                color=CSE_COLORS["text"],
            )


# ══════════════════════════════════════════════════════════════════════════════
# Reusable Plot Functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_time_series(data, x, y, title, ylabel, figsize=(14, 6),
                     color=None, add_crashes=True, xlabel="Date",
                     ax=None, alpha=1.0, linewidth=None, label=None,
                     fill_between=False, fill_alpha=0.15,
                     crash_alpha=0.12, format_y_axis=False,
                     tight_layout=True, return_fig=False):
    """
    Generic time series line plot with optional crash bands.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe.
    x : str
        Column name for the x-axis (typically ``'Date'``).
    y : str or list of str
        Column name(s) for the y-axis.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    figsize : tuple, default (14, 6)
        Figure size in inches.
    color : str or list or None
        Line color(s).  Defaults to ``CSE_COLORS['accent1']``.
    add_crashes : bool, default True
        Whether to overlay crash period bands.
    xlabel : str, default 'Date'
        X-axis label.
    ax : matplotlib.axes.Axes or None
        Existing axes to plot on.  If None, a new figure is created.
    alpha : float, default 1.0
        Line alpha.
    linewidth : float or None
        Override default line width.
    label : str or None
        Legend label for the line.
    fill_between : bool, default False
        Fill area under the line.
    fill_alpha : float, default 0.15
        Alpha for the fill-between area.
    crash_alpha : float, default 0.12
        Alpha for crash band overlay.
    format_y_axis : bool, default False
        If True, format the y-axis with :func:`format_large_number`.
    tight_layout : bool, default True
        Apply ``plt.tight_layout()``.
    return_fig : bool, default False
        If True, return ``(fig, ax)`` instead of just ``ax``.

    Returns
    -------
    ax or (fig, ax)
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    # Handle single vs. multiple y columns
    if isinstance(y, str):
        y_cols = [y]
    else:
        y_cols = list(y)

    if color is None:
        if len(y_cols) == 1:
            colors = [CSE_COLORS["accent1"]]
        else:
            colors = SECTOR_PALETTE[:len(y_cols)]
    elif isinstance(color, str):
        colors = [color] * len(y_cols)
    else:
        colors = list(color)

    lw = linewidth or plt.rcParams.get("lines.linewidth", 1.8)

    for i, col in enumerate(y_cols):
        c = colors[i % len(colors)]
        lbl = label if len(y_cols) == 1 else col
        ax.plot(
            data[x], data[col],
            color=c, alpha=alpha, linewidth=lw, label=lbl,
        )
        if fill_between:
            ax.fill_between(
                data[x], data[col],
                alpha=fill_alpha, color=c,
            )

    ax.set_title(title, fontsize=16, fontweight="bold",
                 color=CSE_COLORS["primary"], pad=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Format x-axis for dates
    if pd.api.types.is_datetime64_any_dtype(data[x]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    if format_y_axis:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(large_number_formatter)
        )

    if add_crashes:
        add_crash_bands(ax, alpha=crash_alpha)

    if len(y_cols) > 1 or label:
        ax.legend(loc="best", framealpha=0.9)

    if tight_layout and created_fig:
        fig.tight_layout()

    if return_fig:
        return fig, ax
    return ax


def plot_dual_axis(data, x, y1, y2, title, ylabel1, ylabel2,
                   figsize=(14, 6), color1=None, color2=None,
                   add_crashes=True, crash_alpha=0.12,
                   format_y1=False, format_y2=False,
                   label1=None, label2=None,
                   tight_layout=True, return_fig=False):
    """
    Plot with dual y-axes sharing the same x-axis.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe.
    x : str
        Column name for the x-axis.
    y1 : str
        Column for the left y-axis.
    y2 : str
        Column for the right y-axis.
    title : str
        Plot title.
    ylabel1 : str
        Left y-axis label.
    ylabel2 : str
        Right y-axis label.
    figsize : tuple, default (14, 6)
        Figure dimensions.
    color1 : str or None
        Color for y1.  Defaults to ``CSE_COLORS['accent1']``.
    color2 : str or None
        Color for y2.  Defaults to ``CSE_COLORS['orange']``.
    add_crashes : bool, default True
        Overlay crash bands.
    crash_alpha : float, default 0.12
        Crash band transparency.
    format_y1 : bool, default False
        Apply large-number formatting to left axis.
    format_y2 : bool, default False
        Apply large-number formatting to right axis.
    label1 : str or None
        Legend label for y1.
    label2 : str or None
        Legend label for y2.
    tight_layout : bool, default True
        Apply tight layout.
    return_fig : bool, default False
        If True, return ``(fig, ax1, ax2)``.

    Returns
    -------
    (ax1, ax2) or (fig, ax1, ax2)
    """
    c1 = color1 or CSE_COLORS["accent1"]
    c2 = color2 or CSE_COLORS["orange"]
    l1 = label1 or y1
    l2 = label2 or y2

    fig, ax1 = plt.subplots(figsize=figsize)

    # Left axis
    line1 = ax1.plot(data[x], data[y1], color=c1, linewidth=1.8, label=l1)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel(ylabel1, fontsize=12, color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    if format_y1:
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(large_number_formatter)
        )

    # Right axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(data[x], data[y2], color=c2, linewidth=1.8, label=l2)
    ax2.set_ylabel(ylabel2, fontsize=12, color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.spines["right"].set_visible(True)

    if format_y2:
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(large_number_formatter)
        )

    # Title
    ax1.set_title(title, fontsize=16, fontweight="bold",
                  color=CSE_COLORS["primary"], pad=14)

    # Date formatting
    if pd.api.types.is_datetime64_any_dtype(data[x]):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.xaxis.set_major_locator(mdates.YearLocator(5))
        ax1.xaxis.set_minor_locator(mdates.YearLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Crash bands
    if add_crashes:
        add_crash_bands(ax1, alpha=crash_alpha)

    # Combined legend
    lines = line1 + line2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    if tight_layout:
        fig.tight_layout()

    if return_fig:
        return fig, ax1, ax2
    return ax1, ax2


# ══════════════════════════════════════════════════════════════════════════════
# Additional Convenience Helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_returns_distribution(returns, title="Returns Distribution",
                              bins=100, figsize=(12, 5), ax=None,
                              color=None, return_fig=False):
    """
    Histogram with KDE overlay for return distributions.

    Parameters
    ----------
    returns : array-like
        Return values (daily, monthly, etc.).
    title : str
        Plot title.
    bins : int, default 100
        Number of histogram bins.
    figsize : tuple
        Figure size.
    ax : Axes or None
        Existing axes.
    color : str or None
        Histogram color.
    return_fig : bool
        Return ``(fig, ax)`` if True.

    Returns
    -------
    ax or (fig, ax)
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    c = color or CSE_COLORS["accent1"]
    returns_clean = pd.Series(returns).dropna()

    ax.hist(returns_clean, bins=bins, density=True, alpha=0.6,
            color=c, edgecolor="white", linewidth=0.5)

    # KDE overlay
    try:
        returns_clean.plot.kde(ax=ax, color=CSE_COLORS["primary"],
                               linewidth=2)
    except Exception:
        pass  # KDE can fail with very few data points

    # Mark mean and median
    mean_val = returns_clean.mean()
    median_val = returns_clean.median()
    ax.axvline(mean_val, color=CSE_COLORS["negative"], linestyle="--",
               linewidth=1.5, label=f"Mean: {mean_val:.4f}")
    ax.axvline(median_val, color=CSE_COLORS["positive"], linestyle="--",
               linewidth=1.5, label=f"Median: {median_val:.4f}")

    ax.set_title(title, fontsize=16, fontweight="bold",
                 color=CSE_COLORS["primary"])
    ax.set_xlabel("Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=10)

    if created_fig:
        fig.tight_layout()

    if return_fig:
        return fig, ax
    return ax


def plot_heatmap(data, title="Heatmap", figsize=(12, 8), cmap="RdYlGn",
                 fmt=".2f", annot=True, ax=None, vmin=None, vmax=None,
                 center=None, return_fig=False):
    """
    Styled correlation / value heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Matrix data for the heatmap.
    title : str
        Plot title.
    figsize : tuple
        Figure dimensions.
    cmap : str
        Matplotlib colormap name.
    fmt : str
        Annotation format string.
    annot : bool
        Whether to annotate cells.
    ax : Axes or None
        Existing axes.
    vmin, vmax, center : float or None
        Colorbar range and center.
    return_fig : bool
        Return ``(fig, ax)``.

    Returns
    -------
    ax or (fig, ax)
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    sns.heatmap(
        data, annot=annot, fmt=fmt, cmap=cmap,
        ax=ax, vmin=vmin, vmax=vmax, center=center,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title, fontsize=16, fontweight="bold",
                 color=CSE_COLORS["primary"], pad=14)

    if created_fig:
        fig.tight_layout()

    if return_fig:
        return fig, ax
    return ax


def annotate_point(ax, x, y, text, offset=(10, 15), fontsize=9,
                   color=None, arrowprops=None):
    """
    Add an annotation with an arrow to a specific point on a plot.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x, y : scalar
        Coordinates of the point to annotate.
    text : str
        Annotation text.
    offset : tuple, default (10, 15)
        Offset in points for the text placement.
    fontsize : int, default 9
        Font size.
    color : str or None
        Text color.
    arrowprops : dict or None
        Arrow style properties.
    """
    c = color or CSE_COLORS["text"]
    arrow = arrowprops or dict(
        arrowstyle="->",
        color=CSE_COLORS["neutral"],
        lw=1.2,
    )

    ax.annotate(
        text, xy=(x, y), xytext=offset,
        textcoords="offset points",
        fontsize=fontsize, color=c,
        arrowprops=arrow,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CSE_COLORS["grid"],
                  alpha=0.85),
    )


def save_figure(fig, filename, output_dir=None, formats=("png",), dpi=150):
    """
    Save a figure to disk in one or more formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    filename : str
        Base filename (without extension).
    output_dir : str or None
        Directory to save to.  Defaults to current working directory.
    formats : tuple of str
        Image formats to save (e.g. ``('png', 'svg', 'pdf')``).
    dpi : int, default 150
        Resolution.
    """
    import os
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")


# ── Auto-apply style on import ──
setup_plotting_style()
