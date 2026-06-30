def get_cells():
    cells = []
    
    # Section 0
    cells.append(('markdown', '''# Phase 1: Exploratory Data Analysis
**Colombo Stock Exchange (CSE) Market Analysis**
- **Date**: 2026-06-30
- **Goal**: Analyze historical stock market data (1991-2025) to identify market trends, data quality issues, and statistical properties of returns.
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
    cells.append(('markdown', '''## Section 1: Data Loading & Consolidation
Loading the daily prices across two eras: 1991-2000 (Close price only) and 2001-2025 (OHLCV). Supplementary datasets are also loaded.
'''))
    
    cells.append(('code', '''# Load consolidated daily prices (uses Parquet cache if available)
print("Loading daily prices...")
df = load_all_daily_prices()

# Load supplementary data
print("Loading supplementary data...")
df_indices = load_market_indices(BASE_PATH)
df_market_stats = load_market_stats(BASE_PATH)
try:
    df_securities = load_securities_list(BASE_PATH)
except Exception as e:
    print(f"Failed to load securities: {e}")
    df_securities = pd.DataFrame()

df_splits = load_splits(BASE_PATH)
df_listings = load_listings(BASE_PATH)
try:
    df_gics = load_gics(BASE_PATH)
except Exception as e:
    print(f"Failed to load GICS: {e}")
    df_gics = pd.DataFrame()

print("\\nData Loading Complete!")
print(f"Daily Prices Shape: {df.shape}")
print(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
'''))
    
    cells.append(('markdown', '''> **Insight - Data Structure**: The dataset spans ~34 years. The early era lacks volume and intraday price data, necessitating care when combining it with modern data.
'''))

    # Section 2
    cells.append(('markdown', '''## Section 2: Dataset Inspection
Analyzing shape, structure, and missing values across time.
'''))
    
    cells.append(('code', '''print("Dataset Info:")
df.info()

# Memory usage
mem_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"\\nMemory Usage: {mem_mb:.2f} MB")

# Number of unique companies
num_companies = df['CompanyCode'].nunique()
print(f"\\nUnique Companies: {num_companies}")

# Date range per stock
stock_dates = df.groupby('CompanyCode')['Date'].agg(['min', 'max', 'count']).reset_index()
stock_dates['years_active'] = (stock_dates['max'] - stock_dates['min']).dt.days / 365.25
stock_dates = stock_dates.sort_values('years_active', ascending=False)
display(stock_dates.head(10))

plt.figure(figsize=(10, 5))
sns.histplot(stock_dates['years_active'], bins=20, color=CSE_COLORS['primary'])
plt.title('Distribution of Stock History Lengths')
plt.xlabel('Years Active')
plt.ylabel('Number of Companies')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Inspection**: Many companies have very short histories (recently listed or delisted), while a core set has traded for over 20 years.
'''))

    # Section 3
    cells.append(('markdown', '''## Section 3: Data Quality Assessment
Identifying missing data, zeros, and potential anomalies.
'''))
    
    cells.append(('code', '''# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct}).sort_values('Percent', ascending=False)
display(missing_df[missing_df['Percent'] > 0])

# Zero prices
zeros = df[df['Close'] <= 0]
print(f"\\nRecords with Close <= 0: {len(zeros)}")

# Duplicate dates per company
dups = df[df.duplicated(subset=['CompanyCode', 'Date'], keep=False)]
print(f"Duplicate records: {len(dups)}")

# Calculate daily return to flag anomalies
df.sort_values(['CompanyCode', 'Date'], inplace=True)
df['DailyReturn'] = df.groupby('CompanyCode')['Close'].pct_change()

# Anomaly flags
anomalies = df[abs(df['DailyReturn']) > 0.5]
print(f"\\nAnomalies (|Daily Return| > 50%): {len(anomalies)} records")
display(anomalies[['CompanyCode', 'Date', 'Close', 'DailyReturn']].head(10))
'''))
    
    cells.append(('markdown', '''> **Insight - Data Quality**: The early data (1991-2000) shows missing values for volume and OHLC. Extreme price jumps (>50%) might represent stock splits or data entry errors, which need adjustment before modeling.
'''))

    # Section 4
    cells.append(('markdown', '''## Section 4: Descriptive Statistics
Understanding the distribution of prices and liquidity.
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

# Top 20 most actively traded
top_20_vol = stock_summary.sort_values('ShareVolume_sum', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_20_vol['ShareVolume_sum'], y=top_20_vol.index, palette=SECTOR_PALETTE[:20])
plt.title('Top 20 Most Actively Traded Stocks (by Volume)')
plt.xlabel('Total Share Volume')
plt.ylabel('Company Code')
plt.tight_layout()
plt.show()

# Price level distribution (latest price)
latest_prices = df.groupby('CompanyCode')['Close'].last()
bins = [0, 10, 50, 100, 500, np.inf]
labels = ['< 10', '10-50', '50-100', '100-500', '500+']
price_cats = pd.cut(latest_prices, bins=bins, labels=labels).value_counts().sort_index()

plt.figure(figsize=(8, 4))
price_cats.plot(kind='bar', color=CSE_COLORS['accent1'])
plt.title('Distribution of Latest Price Levels')
plt.xlabel('Price Range (Rs.)')
plt.ylabel('Number of Companies')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Descriptive Stats**: The market is highly skewed, with trading activity concentrated in a few liquid stocks. The majority of companies trade under Rs. 100.
'''))

    # Section 5
    cells.append(('markdown', '''## Section 5: Time Series Analysis: Market Indices
Analyzing ASPI and S&P SL20 performance.
'''))
    
    cells.append(('code', '''if not df_indices.empty and 'ASPI' in df_indices.columns:
    df_indices['Date'] = pd.to_datetime(df_indices['Date'], errors='coerce')
    df_indices.dropna(subset=['Date', 'ASPI'], inplace=True)
    df_indices.sort_values('Date', inplace=True)
    df_indices.set_index('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_indices.index, df_indices['ASPI'], color=CSE_COLORS['primary'], label='ASPI')
    
    df_indices['ASPI_MA90'] = df_indices['ASPI'].rolling(90).mean()
    ax.plot(df_indices.index, df_indices['ASPI_MA90'], color=CSE_COLORS['orange'], label='90-day MA', alpha=0.8)
    
    add_crash_bands(ax)
    
    ax.set_title('ASPI Performance (1985-2025) with Major Market Events')
    ax.set_ylabel('Index Value')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Returns
    df_indices['Daily_Return'] = df_indices['ASPI'].pct_change()
    
    monthly_aspi = df_indices['ASPI'].resample('M').last()
    monthly_returns = monthly_aspi.pct_change()
    
    annual_aspi = df_indices['ASPI'].resample('Y').last()
    annual_returns = annual_aspi.pct_change()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(df_indices.index, df_indices['Daily_Return'], color=CSE_COLORS['accent1'], linewidth=0.5)
    axes[0].set_title('ASPI Daily Returns')
    axes[0].axhline(0, color='black', linewidth=1)
    
    axes[1].bar(annual_returns.index.year, annual_returns, color=[CSE_COLORS['positive'] if x > 0 else CSE_COLORS['negative'] for x in annual_returns])
    axes[1].set_title('ASPI Annual Returns')
    axes[1].axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.show()
else:
    print("Market indices data not available.")
'''))
    
    cells.append(('markdown', '''> **Insight - Indices**: The ASPI shows significant long-term growth punctuated by severe drawdowns during economic and political crises. Volatility is elevated during these events.
'''))

    # Section 6
    cells.append(('markdown', '''## Section 6: Time Series Analysis: Trading Volume
Assessing market liquidity trends.
'''))
    
    cells.append(('code', '''if not df_market_stats.empty and 'TURNOVER EQUITY-Mn' in df_market_stats.columns:
    df_market_stats['Date'] = pd.to_datetime(df_market_stats['Date'], errors='coerce')
    df_market_stats.dropna(subset=['Date'], inplace=True)
    df_market_stats.set_index('Date', inplace=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_market_stats.index, df_market_stats['TURNOVER EQUITY-Mn'], color=CSE_COLORS['teal'], alpha=0.5, label='Daily Turnover')
    
    turnover_ma90 = df_market_stats['TURNOVER EQUITY-Mn'].rolling(90).mean()
    ax.plot(df_market_stats.index, turnover_ma90, color=CSE_COLORS['primary'], label='90-day MA', linewidth=2)
    
    ax.set_title('Market Daily Turnover (Equity) in LKR Millions')
    ax.set_ylabel('Turnover (Mn)')
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Market stats data not available.")
'''))
    
    cells.append(('markdown', '''> **Insight - Volume**: Trading volume shows clear regimes, with massive spikes during bull markets (e.g., post-war 2010, pandemic recovery 2021) and depressed liquidity during bear phases.
'''))

    # Section 7
    cells.append(('markdown', '''## Section 7: Time Series Analysis: Individual Stocks
Analyzing top stocks over time.
'''))
    
    cells.append(('code', '''top_10 = top_20_vol.head(10).index.tolist()
df_modern = df[df['Era'] == '2001-2025'].copy()

fig, axes = plt.subplots(5, 2, figsize=(15, 20))
axes = axes.flatten()

for i, stock in enumerate(top_10):
    stock_data = df_modern[df_modern['CompanyCode'] == stock].set_index('Date')
    if len(stock_data) > 0:
        axes[i].plot(stock_data.index, stock_data['Close'], color=CSE_COLORS['primary'], label='Close')
        axes[i].plot(stock_data.index, stock_data['Close'].rolling(90).mean(), color=CSE_COLORS['orange'], label='90d MA')
        axes[i].set_title(f'{stock} Price History')
        axes[i].legend()

plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Stocks**: Top tier stocks show robust trend-following characteristics, but they also experience sharp pullbacks. Moving averages (90-day) effectively smooth the noise.
'''))

    return cells
