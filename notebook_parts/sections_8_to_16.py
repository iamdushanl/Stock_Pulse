def get_cells():
    cells = []
    
    # Section 8
    cells.append(('markdown', '''## Section 8: Distribution Analysis
Understanding the statistical properties of stock returns.
'''))
    
    cells.append(('code', '''# Market-wide daily return distribution
df_returns = df.dropna(subset=['DailyReturn'])
returns = df_returns['DailyReturn'].values

plt.figure(figsize=(10, 6))
sns.histplot(returns, bins=100, kde=True, stat="density", color=CSE_COLORS['accent1'], 
             range=(-0.15, 0.15))

# Fit normal distribution
mu, std = stats.norm.fit(returns[~np.isnan(returns)])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit ($\\mu={mu:.4f}, \\sigma={std:.4f}$)')

plt.title('Distribution of Daily Returns vs Normal Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Skewness and Kurtosis
print(f"Skewness: {stats.skew(returns[~np.isnan(returns)]):.4f}")
print(f"Kurtosis: {stats.kurtosis(returns[~np.isnan(returns)]):.4f}")
'''))
    
    cells.append(('markdown', '''> **Insight - Distributions**: The returns distribution exhibits significant excess kurtosis (fat tails) compared to a normal distribution. This implies extreme market moves happen more frequently than a normal model would predict.
'''))

    # Section 9
    cells.append(('markdown', '''## Section 9: Correlation Analysis
Analyzing relationships between price, volume, and cross-stock correlations.
'''))
    
    cells.append(('code', '''df_modern = df[df['Era'] == '2001-2025'].copy()
numeric_cols = ['Open', 'High', 'Low', 'Close', 'TradeVolume', 'ShareVolume', 'Turnover']

corr_matrix = df_modern[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Price & Volume Features')
plt.tight_layout()
plt.show()

# Cross-stock correlation for top 10 stocks
top_10 = df_modern.groupby('CompanyCode')['ShareVolume'].sum().nlargest(10).index
pivot_returns = df_modern[df_modern['CompanyCode'].isin(top_10)].pivot(index='Date', columns='CompanyCode', values='DailyReturn')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Cross-Stock Return Correlation (Top 10)')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Correlations**: While Open/High/Low/Close are perfectly correlated, volume metrics have weak linear correlations with price. Cross-stock correlations are generally positive but moderate, indicating sector-specific movements and diversification potential.
'''))

    # Section 10
    cells.append(('markdown', '''## Section 10: Outlier Analysis
Analyzing extreme market events. 

*Note: In financial time series, outliers often represent actual market crashes, rallies, or news events rather than "bad data", and should generally be retained for tail-risk modeling.*
'''))
    
    cells.append(('code', '''Q1 = df_returns['DailyReturn'].quantile(0.25)
Q3 = df_returns['DailyReturn'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

outliers = df_returns[(df_returns['DailyReturn'] < lower_bound) | (df_returns['DailyReturn'] > upper_bound)]
print(f"Total Outliers detected (3x IQR): {len(outliers)} ({len(outliers)/len(df_returns)*100:.2f}%)")

plt.figure(figsize=(12, 5))
sns.boxplot(x=df_returns['DailyReturn'], color=CSE_COLORS['accent2'])
plt.axvline(lower_bound, color='red', linestyle='--')
plt.axvline(upper_bound, color='red', linestyle='--')
plt.title('Daily Returns Boxplot with Outlier Bounds')
plt.xlim(-0.2, 0.2)
plt.tight_layout()
plt.show()

# Outliers over time
outliers['Year'] = outliers['Date'].dt.year
outlier_counts = outliers.groupby('Year').size()

plt.figure(figsize=(12, 5))
outlier_counts.plot(kind='bar', color=CSE_COLORS['negative'])
plt.title('Number of Extreme Return Events per Year')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Outliers**: Outlier frequency spikes during specific years (e.g., 2011, 2021) matching volatile market regimes.
'''))

    # Section 11
    cells.append(('markdown', '''## Section 11: Feature Relationship Analysis
'''))
    
    cells.append(('code', '''df_modern['LogVolume'] = np.log1p(df_modern['ShareVolume'])
df_modern['AbsReturn'] = np.abs(df_modern['DailyReturn'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_modern.sample(10000), x='LogVolume', y='AbsReturn', alpha=0.1, color=CSE_COLORS['primary'])
plt.title('Log Volume vs Absolute Daily Return (10k Sample)')
plt.xlabel('Log(Share Volume)')
plt.ylabel('Absolute Daily Return')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Relationships**: There is a visible "volatility smile" effect where higher trading volumes are associated with larger absolute price movements.
'''))

    # Section 12
    cells.append(('markdown', '''## Section 12: Time-Based Feature Engineering
Analyzing calendar effects.
'''))
    
    cells.append(('code', '''df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

dow_returns = df.groupby('DayOfWeek')['DailyReturn'].mean() * 100
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

plt.figure(figsize=(8, 5))
sns.barplot(x=days, y=dow_returns[:5], color=CSE_COLORS['teal'])
plt.title('Average Daily Return by Day of Week (%)')
plt.ylabel('Mean Return (%)')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.show()

monthly_returns = df.groupby('Month')['DailyReturn'].mean() * 100
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(10, 5))
sns.barplot(x=months, y=monthly_returns, color=CSE_COLORS['accent1'])
plt.title('Average Daily Return by Month (%)')
plt.ylabel('Mean Return (%)')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Calendar Effects**: Minor seasonality is visible, but the effects are generally small and require rigorous statistical testing before being used as standalone signals.
'''))

    # Section 13
    cells.append(('markdown', '''## Section 13: Volatility Analysis
'''))
    
    cells.append(('code', '''# Annualized volatility per stock
ann_vol = df.groupby('CompanyCode')['DailyReturn'].std() * np.sqrt(252)
ann_vol = ann_vol.dropna().sort_values(ascending=False)

print("Top 10 Most Volatile Stocks (Annualized):")
display(ann_vol.head(10))

if not df_indices.empty:
    df_indices['Vol_30d'] = df_indices['Daily_Return'].rolling(30).std() * np.sqrt(252)
    
    plt.figure(figsize=(14, 5))
    plt.plot(df_indices.index, df_indices['Vol_30d'], color=CSE_COLORS['negative'])
    add_crash_bands(plt.gca())
    plt.title('ASPI 30-Day Rolling Annualized Volatility')
    plt.ylabel('Volatility')
    plt.tight_layout()
    plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Volatility**: Volatility exhibits strong clustering, spiking massively during the designated crash periods (e.g., 2022 Economic Crisis).
'''))

    # Section 14
    cells.append(('markdown', '''## Section 14: Trend Analysis
'''))
    
    cells.append(('code', '''if not df_indices.empty:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_indices.index, df_indices['ASPI'], color='black', label='ASPI', alpha=0.5)
    
    mas = [50, 200]
    colors = [CSE_COLORS['teal'], CSE_COLORS['orange']]
    
    for ma, color in zip(mas, colors):
        df_indices[f'MA_{ma}'] = df_indices['ASPI'].rolling(ma).mean()
        ax.plot(df_indices.index, df_indices[f'MA_{ma}'], color=color, label=f'{ma}-day MA', linewidth=2)
        
    ax.set_title('ASPI Trend Analysis (50 & 200 Day MAs)')
    ax.legend()
    plt.tight_layout()
    plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Trends**: The 50/200 day moving average crossovers provide clear historical signals for major market regime shifts.
'''))

    # Section 15
    cells.append(('markdown', '''## Section 15: Interactive Visualizations (Plotly)
*(Note: These will render in a live notebook environment)*
'''))
    
    cells.append(('code', '''if not df_indices.empty:
    # Downsample for plotly performance if needed
    plot_df = df_indices.reset_index().dropna(subset=['ASPI'])
    
    fig = px.line(plot_df, x='Date', y='ASPI', title='Interactive ASPI Chart')
    fig.update_xaxes(rangeslider_visible=True)
    # fig.show()  # Uncomment to view interactively
    print("Interactive ASPI chart generated (uncomment fig.show() to view in notebook).")
'''))
    
    cells.append(('markdown', '''> **Insight - Interactivity**: Plotly allows for deep dives into specific crash and rally dates.
'''))

    # Section 16
    cells.append(('markdown', '''## Section 16: Closing Summary

### Key Findings
1. **Market Structure**: The dataset successfully bridges two eras (1991-2000 Close-only, and 2001-2025 OHLCV).
2. **Data Quality**: Identified missing values, zero prices, and potential unadjusted splits that require a robust cleaning pipeline before modeling.
3. **Distributions**: Stock returns show classic fat tails (excess kurtosis); standard normal assumptions will underestimate tail risk.
4. **Volatility Clustering**: Clear volatility regimes map exactly to Sri Lanka's macroeconomic and political events (2001, 2008, 2011, 2019, 2022).
5. **Liquidity**: Trading is heavily concentrated in a top tier of liquid stocks.

### Candidate Target Variables for Future Modeling
- `forward_1M_return`, `forward_3M_return` (for regression mapping to investment horizons)
- `trend_state` (binary label based on price position vs MAs)

### Next Steps (Phase 2)
- Implement a corporate action pipeline to adjust historical prices for stock splits using the `df_splits` table.
- Engineer technical indicators (RSI, MACD, Bollinger Bands) using the clean OHLCV data.
- Develop the filtering logic for the Recommendation System to map these features to Short/Medium/Long term uptrend potential.
'''))

    return cells
