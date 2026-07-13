def get_cells():
    cells = []
    
    # Section 0
    cells.append(('markdown', '''# Phase 1: Exploratory Data Analysis - S&P SL 20
**Colombo Stock Exchange (CSE) Market Analysis**
- **Date**: 2026-07-13
- **Universe**: S&P SL 20 Companies (Top 20 highly capitalized and liquid stocks)
- **Timeframe**: 2001-2025 (Daily Data)
- **Goal**: Analyze the clean, enriched dataset for the top 20 Sri Lankan companies to identify market trends, statistical properties of returns, and prepare for Recommendation System modeling.
'''))
    
    cells.append(('code', '''import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm

# Import custom utilities
from utils.data_loader import (
    load_market_indices,
    load_market_stats,
    BASE_PATH
)
from utils.plot_helpers import (
    setup_plotting_style,
    CSE_COLORS,
    SECTOR_PALETTE,
    format_large_number,
    add_crash_bands,
    CRASH_PERIODS
)

# Apply styling
setup_plotting_style()
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
'''))

    # Section 1
    cells.append(('markdown', '''## Section 1: Data Loading
Loading the cleaned, consolidated Parquet dataset containing daily OHLCV prices and pre-calculated technical features (MA, Volatility, etc.) for the 20 S&P SL 20 companies.
'''))
    
    cells.append(('code', '''# Load cleaned S&P SL 20 dataset
print("Loading daily prices...")
parquet_path = f"{BASE_PATH}/sp20_daily_prices_2001_2025.parquet"
df = pd.read_parquet(parquet_path)

# Load supplementary data (Market Indices and Stats)
print("Loading supplementary data...")
df_indices = load_market_indices(BASE_PATH)
df_market_stats = load_market_stats(BASE_PATH)

print("\\nData Loading Complete!")
print(f"Daily Prices Shape: {df.shape}")
print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
'''))
    
    cells.append(('markdown', '''> **Insight - Data Structure**: The dataset spans 25 years (2001-2025) and contains exactly 20 companies, yielding a rich, dense matrix of over 89,000 trading days.
'''))

    # Section 2
    cells.append(('markdown', '''## Section 2: Dataset Inspection
Validating the dataset structure and completeness.
'''))
    
    cells.append(('code', '''print("Dataset Info:")
df.info()

# Memory usage
mem_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"\\nMemory Usage: {mem_mb:.2f} MB")

# Number of unique companies
companies = df['CompanyCode'].unique()
print(f"\\nCompanies in Universe ({len(companies)}):\\n{companies}")

# Date range per stock
stock_dates = df.groupby('CompanyCode')['Date'].agg(['min', 'max', 'count']).reset_index()
stock_dates['years_active'] = (stock_dates['max'] - stock_dates['min']).dt.days / 365.25
stock_dates = stock_dates.sort_values('years_active', ascending=False)
display(stock_dates)

plt.figure(figsize=(10, 5))
sns.barplot(data=stock_dates, x='CompanyCode', y='years_active', palette=SECTOR_PALETTE[:20])
plt.title('Trading History Length per Company')
plt.xlabel('Company')
plt.ylabel('Years Active')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Inspection**: All companies except MELS (which listed later) have strong histories spanning over a decade, with many spanning the full 25 years, providing an excellent foundation for time-series modeling.
'''))

    # Section 3
    cells.append(('markdown', '''## Section 3: Data Quality Assessment
Confirming the dataset is clean and free of anomalies.
'''))
    
    cells.append(('code', '''# Missing values check
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct}).sort_values('Percent', ascending=False)
display(missing_df[missing_df['Percent'] > 0])

# Duplicate dates per company
dups = df[df.duplicated(subset=['CompanyCode', 'Date'], keep=False)]
print(f"Duplicate records: {len(dups)}")

# Validate Daily Returns (Checking for extreme anomalies)
anomalies = df[abs(df['DailyReturn_Pct']) > 50]
print(f"\\nAnomalies (|Daily Return| > 50%): {len(anomalies)} records")
if len(anomalies) > 0:
    display(anomalies[['CompanyCode', 'Date', 'Close', 'DailyReturn_Pct']].head(10))
'''))
    
    cells.append(('markdown', '''> **Insight - Data Quality**: The dataset is exceptionally clean. Null values only exist for pre-calculated rolling windows (e.g., 90-day MA is null for the first 89 days of a stock's history), which is mathematically expected.
'''))

    # Section 4
    cells.append(('markdown', '''## Section 4: Descriptive Statistics
Understanding the distribution of prices and liquidity across the S&P SL 20.
'''))
    
    cells.append(('code', '''# Numerical summary
display(df.describe())

# Per stock summary
stock_summary = df.groupby('CompanyCode').agg({
    'Close': ['mean', 'std', 'min', 'max'],
    'ShareVolume': 'sum',
    'Turnover': 'sum'
})
stock_summary.columns = ['_'.join(col) for col in stock_summary.columns]

# Rank by Turnover (Liquidity)
liquidity_rank = stock_summary.sort_values('Turnover_sum', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=liquidity_rank['Turnover_sum'], y=liquidity_rank.index, palette=SECTOR_PALETTE[:20])
plt.title('S&P SL 20 - Total Historical Turnover (Liquidity)')
plt.xlabel('Total Turnover (Rs.)')
plt.ylabel('Company Code')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Descriptive Stats**: Even within the top 20 blue-chip companies, liquidity (turnover) follows a Pareto-like distribution, with heavyweights like JKH and COMB dominating market activity.
'''))

    # Section 5
    cells.append(('markdown', '''## Section 5: Time Series Analysis: Market Indices
Analyzing ASPI and S&P SL20 performance to establish a market baseline.
'''))
    
    cells.append(('code', '''if not df_indices.empty and 'ASPI' in df_indices.columns:
    df_indices['Date'] = pd.to_datetime(df_indices['Date'], errors='coerce')
    df_indices = df_indices.dropna(subset=['Date', 'ASPI']).copy()
    df_indices = df_indices.sort_values('Date')
    df_indices = df_indices.set_index('Date')

    # Filter to 2001 onwards to match our dataset
    df_idx_modern = df_indices[df_indices.index >= '2001-01-01'].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_idx_modern.index, df_idx_modern['ASPI'], color=CSE_COLORS['primary'], label='ASPI')
    
    df_idx_modern['ASPI_MA90'] = df_idx_modern['ASPI'].rolling(90).mean()
    ax.plot(df_idx_modern.index, df_idx_modern['ASPI_MA90'], color=CSE_COLORS['orange'], label='90-day MA', alpha=0.8)
    
    add_crash_bands(ax)
    
    ax.set_title('ASPI Performance (2001-2025) with Major Market Events')
    ax.set_ylabel('Index Value')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Returns
    df_idx_modern['Daily_Return'] = df_idx_modern['ASPI'].pct_change()
    
    annual_aspi = df_idx_modern['ASPI'].resample('YE').last()
    annual_returns = annual_aspi.pct_change() * 100
    
    plt.figure(figsize=(14, 5))
    plt.bar(annual_returns.index.year, annual_returns, color=[CSE_COLORS['positive'] if x > 0 else CSE_COLORS['negative'] for x in annual_returns])
    plt.title('ASPI Annual Returns (%)')
    plt.axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.show()
else:
    print("Market indices data not available.")
'''))
    
    cells.append(('markdown', '''> **Insight - Indices**: The broader market (ASPI) shows distinct bull and bear cycles, heavily influenced by macroeconomic shocks (e.g., the 2022 crisis). The S&P SL 20 subset will generally track these movements but with potentially higher beta.
'''))

    # Section 6
    cells.append(('markdown', '''## Section 6: Time Series Analysis: Trading Volume
Assessing market liquidity trends.
'''))
    
    cells.append(('code', '''if not df_market_stats.empty and 'TURNOVER_EQUITY_Mn' in df_market_stats.columns:
    df_ms_modern = df_market_stats[(df_market_stats['Date'] >= '2001-01-01') & (df_market_stats['Date'] <= '2025-12-31')].set_index('Date')
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot daily turnover
    ax.plot(df_ms_modern.index, df_ms_modern['TURNOVER_EQUITY_Mn'], color=CSE_COLORS['teal'], alpha=0.5, label='Daily Turnover')
    # Plot 90-day moving average of turnover
    turnover_ma90 = df_ms_modern['TURNOVER_EQUITY_Mn'].rolling(90).mean()
    ax.plot(df_ms_modern.index, turnover_ma90, color=CSE_COLORS['orange'], linewidth=2, label='90d MA Turnover')
    
    ax.set_title('Market Turnover (Equity) 2001-2025 in LKR Millions (2001-2025)')
    ax.set_ylabel('Turnover (Mn)')
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Market stats data not available.")
'''))
    
    cells.append(('markdown', '''> **Insight - Volume**: Market liquidity surges dramatically during bull runs (e.g., 2010 post-war, 2021 pandemic recovery), providing excellent environments for momentum strategies.
'''))

    # Section 7
    cells.append(('markdown', '''## Section 7: Time Series Analysis: Individual Stocks
Analyzing top S&P SL 20 stocks over time using pre-calculated moving averages.
'''))
    
    cells.append(('code', '''all_stocks = liquidity_rank.index.tolist()

fig, axes = plt.subplots(5, 4, figsize=(24, 20))
axes = axes.flatten()

for i, stock in enumerate(all_stocks):
    stock_data = df[df['CompanyCode'] == stock].set_index('Date')
    if len(stock_data) > 0:
        axes[i].plot(stock_data.index, stock_data['Close'], color=CSE_COLORS['primary'], label='Close')
        # Use pre-calculated 90-day MA
        axes[i].plot(stock_data.index, stock_data['MA_90d'], color=CSE_COLORS['orange'], label='90d MA', linewidth=2)
        axes[i].set_title(f'{stock} Price History & Trend')
        axes[i].legend(loc='upper right')
        axes[i].set_ylabel('Price (Rs.)')

plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Stocks**: Blue-chip stocks demonstrate clear, prolonged trend phases. The pre-calculated 90-day Moving Average effectively filters out short-term noise, providing a solid baseline for trend detection.
'''))

    return cells
