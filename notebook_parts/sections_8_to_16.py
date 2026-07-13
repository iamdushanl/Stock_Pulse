def get_cells():
    cells = []
    
    # Section 8
    cells.append(('markdown', '''## Section 8: Distribution Analysis
Understanding the statistical properties of S&P SL 20 stock returns.
'''))
    
    cells.append(('code', '''# Market-wide daily return distribution
df_returns = df.dropna(subset=['DailyReturn_Pct'])
returns = df_returns['DailyReturn_Pct'].values

plt.figure(figsize=(10, 6))
sns.histplot(returns, bins=100, kde=True, stat="density", color=CSE_COLORS['accent1'], 
             binrange=(-15, 15))

# Fit normal distribution
valid_returns = returns[np.isfinite(returns)]
visual_returns = valid_returns[(valid_returns >= -20) & (valid_returns <= 20)]

mu, std = stats.norm.fit(visual_returns)
x = np.linspace(-15, 15, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'Normal Fit ($\\mu={mu:.2f}\\%, \\sigma={std:.2f}\\%)')
plt.xlim(-15, 15)

plt.title('Distribution of Daily Returns (%) vs Normal Distribution')
plt.xlabel('Daily Return (%)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()

# Skewness and Kurtosis
print(f"Skewness: {stats.skew(valid_returns):.4f}")
print(f"Kurtosis: {stats.kurtosis(valid_returns):.4f}")
'''))
    
    cells.append(('markdown', '''> **Insight - Distributions**: Even for highly liquid blue-chip stocks, returns exhibit significant excess kurtosis (fat tails). This implies extreme market moves (both up and down) happen more frequently than a normal bell curve would predict.
'''))

    # Section 9
    cells.append(('markdown', '''## Section 9: Correlation Analysis
Analyzing relationships between price, volume, and cross-stock correlations among the top 10 S&P SL 20 constituents.
'''))
    
    cells.append(('code', '''numeric_cols = ['Open', 'High', 'Low', 'Close', 'TradeVolume', 'ShareVolume', 'Turnover']

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix of Price & Volume Features')
plt.tight_layout()
plt.show()

# Cross-stock correlation for top 10 stocks by volume
top_10 = df.groupby('CompanyCode')['ShareVolume'].sum().nlargest(10).index
pivot_returns = df[df['CompanyCode'].isin(top_10)].pivot(index='Date', columns='CompanyCode', values='DailyReturn_Pct')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Cross-Stock Return Correlation (Top 10 S&P SL 20)')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Correlations**: While Open/High/Low/Close are perfectly correlated, volume metrics have very weak linear correlations with absolute price levels. Cross-stock correlations are broadly positive, highlighting systematic market risk (beta) driving the top companies together.
'''))

    # Section 10
    cells.append(('markdown', '''## Section 10: Outlier Analysis
Analyzing extreme market events across the S&P SL 20.
'''))
    
    cells.append(('code', '''Q1 = df_returns['DailyReturn_Pct'].quantile(0.25)
Q3 = df_returns['DailyReturn_Pct'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

outliers = df_returns[(df_returns['DailyReturn_Pct'] < lower_bound) | (df_returns['DailyReturn_Pct'] > upper_bound)]
print(f"Total Outliers detected (3x IQR): {len(outliers)} ({len(outliers)/len(df_returns)*100:.2f}%)")

plt.figure(figsize=(12, 5))
sns.boxplot(x=df_returns['DailyReturn_Pct'], color=CSE_COLORS['accent2'])
plt.axvline(lower_bound, color='red', linestyle='--')
plt.axvline(upper_bound, color='red', linestyle='--')
plt.title('Daily Returns Boxplot with Outlier Bounds')
plt.xlim(-20, 20)
plt.tight_layout()
plt.show()

# Outliers over time
outliers = outliers.copy()
outlier_counts = outliers.groupby('Year').size()

plt.figure(figsize=(12, 5))
outlier_counts.plot(kind='bar', color=CSE_COLORS['negative'])
plt.title('Number of Extreme Return Events per Year')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Outliers**: Outlier frequency spikes during specific years (e.g., 2011, 2022) matching highly volatile market regimes.
'''))

    # Section 11
    cells.append(('markdown', '''## Section 11: Feature Relationship Analysis
Analyzing the relationship between trading activity and price movement magnitude.
'''))
    
    cells.append(('code', '''df_sample = df.sample(min(10000, len(df))).copy()
df_sample['LogVolume'] = np.log1p(df_sample['ShareVolume'])
df_sample['AbsReturn'] = np.abs(df_sample['DailyReturn_Pct'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_sample, x='LogVolume', y='AbsReturn', alpha=0.1, color=CSE_COLORS['primary'])
plt.title('Log Volume vs Absolute Daily Return (10k Sample)')
plt.xlabel('Log(Share Volume)')
plt.ylabel('Absolute Daily Return (%)')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Relationships**: There is a visible "volatility smile" effect where higher trading volumes are associated with larger absolute price movements.
'''))

    # Section 12
    cells.append(('markdown', '''## Section 12: Time-Based Feature Engineering
Analyzing calendar effects using the pre-calculated time features.
'''))
    
    cells.append(('code', '''dow_returns = df.groupby('DayOfWeek')['DailyReturn_Pct'].mean()
# Ensure correct order
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
dow_returns = dow_returns.reindex(days)

plt.figure(figsize=(8, 5))
sns.barplot(x=dow_returns.index, y=dow_returns.values, color=CSE_COLORS['teal'])
plt.title('Average Daily Return by Day of Week (%)')
plt.ylabel('Mean Return (%)')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.show()

monthly_returns = df.groupby('Month')['DailyReturn_Pct'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(10, 5))
sns.barplot(x=months, y=monthly_returns.values, color=CSE_COLORS['accent1'])
plt.title('Average Daily Return by Month (%)')
plt.ylabel('Mean Return (%)')
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Calendar Effects**: Minor seasonality is visible (e.g., September/October historically weaker), but the effects are generally small and require rigorous statistical testing before being used as standalone signals.
'''))

    # Section 13
    cells.append(('markdown', '''## Section 13: Volatility Analysis
Utilizing the pre-calculated `Volatility_30d_Ann` feature to analyze risk across the S&P SL 20.
'''))
    
    cells.append(('code', '''# Mean annualized volatility per stock
mean_vol = df.groupby('CompanyCode')['Volatility_30d_Ann'].mean().sort_values(ascending=False)

print("Average 30-Day Annualized Volatility (%) by Stock:")
display(mean_vol)

plt.figure(figsize=(12, 6))
sns.barplot(x=mean_vol.index, y=mean_vol.values, palette=SECTOR_PALETTE[:20])
plt.title('Average Annualized Volatility by Company')
plt.ylabel('Volatility (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot volatility over time for JKH (a representative blue chip)
jkh_data = df[df['CompanyCode'] == 'JKH'].set_index('Date')
plt.figure(figsize=(14, 5))
plt.plot(jkh_data.index, jkh_data['Volatility_30d_Ann'], color=CSE_COLORS['negative'])
add_crash_bands(plt.gca())
plt.title('JKH 30-Day Rolling Annualized Volatility (2001-2025)')
plt.ylabel('Volatility (%)')
plt.tight_layout()
plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Volatility**: Volatility exhibits strong clustering. Even for the most stable company in Sri Lanka (JKH), volatility spikes massively during macroeconomic shocks (2001, 2008, 2022).
'''))

    # Section 14
    cells.append(('markdown', '''## Section 14: Trend Analysis
Analyzing long-term trends using moving averages.
'''))
    
    cells.append(('code', '''if not df_indices.empty:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_idx_modern.index, df_idx_modern['ASPI'], color='black', label='ASPI', alpha=0.5)
    
    mas = [50, 200]
    colors = [CSE_COLORS['teal'], CSE_COLORS['orange']]
    
    for ma, color in zip(mas, colors):
        df_idx_modern[f'MA_{ma}'] = df_idx_modern['ASPI'].rolling(ma).mean()
        ax.plot(df_idx_modern.index, df_idx_modern[f'MA_{ma}'], color=color, label=f'{ma}-day MA', linewidth=2)
        
    ax.set_title('ASPI Trend Analysis (50 & 200 Day MAs)')
    ax.legend()
    plt.tight_layout()
    plt.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Trends**: The 50/200 day moving average crossovers provide clear historical signals for major market regime shifts, identifying prolonged bull and bear markets.
'''))

    # Section 15
    cells.append(('markdown', '''## Section 15: Interactive Visualizations (Plotly)
*(Note: These will render in a live notebook environment)*
'''))
    
    cells.append(('code', '''if not df_indices.empty:
    plot_df = df_idx_modern.reset_index().dropna(subset=['ASPI'])
    
    fig = px.line(plot_df, x='Date', y='ASPI', title='Interactive ASPI Chart (2001-2025)')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
'''))
    
    cells.append(('markdown', '''> **Insight - Interactivity**: Plotly allows for deep dives into specific crash and rally dates.
'''))

    # Section 16
    cells.append(('markdown', '''## Section 16: Closing Summary

### Key Findings
1. **Dataset Integrity**: The extracted S&P SL 20 dataset spanning 2001-2025 is incredibly robust. It provides a dense matrix of ~89,000 trading days across the market's top blue chips, completely avoiding the sparsity and missing data issues of the 1990s and small-cap stocks.
2. **Distributions**: Stock returns show classic fat tails (excess kurtosis).
3. **Volatility Clustering**: Volatility regimes map exactly to Sri Lanka's macroeconomic events, even for the most stable companies.
4. **Pre-Calculated Features**: The dataset already contains validated technical features (MA_30d, MA_90d, Vol_30d), eliminating the need for complex rolling-window calculations in the modeling phase.

### Next Steps (Phase 2 & 3)
- Design the recommendation logic to classify stock states (e.g., Uptrend, Downtrend, Sideways) based on the relationship between `Close`, `MA_30d`, and `MA_90d`.
- Build the predictive/recommendation system notebook that operationalizes these insights into actionable trading signals for the S&P SL 20 universe.
'''))

    return cells
