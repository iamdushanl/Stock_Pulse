import nbformat
import sys
import os

def create_notebook():
    nb = nbformat.v4.new_notebook()
    
    # Metadata
    nb.metadata.update({
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'title': 'Phase 2: Feature Engineering & Target Definition'
    })
    
    # Setup
    nb.cells.append(nbformat.v4.new_markdown_cell('# Phase 2: Feature Engineering & Target Definition\n\nConverting the raw dataset into ML-ready features and target variables.'))
    
    nb.cells.append(nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from utils.data_loader import BASE_PATH
from utils.plot_helpers import setup_plotting_style, CSE_COLORS
from utils.features import generate_technical_features
from utils.targets import generate_targets

setup_plotting_style()
pd.set_option('display.max_columns', 100)
'''))

    # Load Data
    nb.cells.append(nbformat.v4.new_markdown_cell('## 1. Data Loading\nLoading the pre-cleaned, split-adjusted S&P SL 20 dataset from Phase 1.'))
    nb.cells.append(nbformat.v4.new_code_cell('''# Load cleaned data
df_clean = pd.read_parquet(os.path.join(BASE_PATH, 'sp20_daily_prices_2001_2025.parquet'))
print(f"Cleaned data shape: {df_clean.shape}")
'''))

    # Data Cleaning (skipped because it's pre-cleaned)
    nb.cells.append(nbformat.v4.new_markdown_cell('## 2. Verify Stock Split Adjustment\nVerify that the stock split logic applied in Phase 1 worked as expected.'))
    nb.cells.append(nbformat.v4.new_code_cell('''# Sunshine Holdings (SUN.N0000) split 1 to 3 on 2021-03-31
sun_clean = df_clean[(df_clean['CompanyCode'] == 'SUN') & (df_clean['Date'].dt.year >= 2020)]

if not sun_clean.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sun_clean['Date'], sun_clean['Close'], label='Cleaned (Adjusted)', linewidth=2, color='green')
    ax.set_title('Stock Split Adjustment Verification: Sunshine Holdings (SUN)')
    ax.legend()
    plt.tight_layout()
    plt.show()
'''))

    # Feature Engineering
    nb.cells.append(nbformat.v4.new_markdown_cell('## 3. Feature Engineering\nGenerating technical indicators (RSI, MACD, Bollinger Bands, MAs) per stock.'))
    nb.cells.append(nbformat.v4.new_code_cell('''# Generate features
df_features = generate_technical_features(df_clean)

# Visualize features for a sample stock
sample_stock = df_features['CompanyCode'].value_counts().index[0]  # Get most traded stock
sample = df_features[df_features['CompanyCode'] == sample_stock].tail(252).set_index('Date') # Last year

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

# Price and Bollinger Bands
ax1.plot(sample.index, sample['Close'], color='black', label='Close')
ax1.plot(sample.index, sample['SMA_20'], color=CSE_COLORS['teal'], label='SMA 20')
ax1.fill_between(sample.index, sample['BB_Lower'], sample['BB_Upper'], color=CSE_COLORS['teal'], alpha=0.1, label='Bollinger Bands')
ax1.set_title(f'{sample_stock} - Price & Bollinger Bands')
ax1.legend()

# MACD
ax2.plot(sample.index, sample['MACD'], color='blue', label='MACD')
ax2.plot(sample.index, sample['MACD_Signal'], color='red', label='Signal')
ax2.bar(sample.index, sample['MACD_Hist'], color='gray', alpha=0.5, label='Histogram')
ax2.set_title('MACD')
ax2.legend()

# RSI
ax3.plot(sample.index, sample['RSI_14'], color='purple', label='RSI 14')
ax3.axhline(70, color='red', linestyle='--')
ax3.axhline(30, color='green', linestyle='--')
ax3.set_title('RSI (Relative Strength Index)')
ax3.set_ylim(0, 100)

plt.tight_layout()
plt.show()
'''))

    # Target Generation
    nb.cells.append(nbformat.v4.new_markdown_cell('## 4. Target Variable Definition\nGenerating Short, Medium, and Long-Term forward returns and binary classification labels.'))
    nb.cells.append(nbformat.v4.new_code_cell('''# Generate targets
df_final = generate_targets(df_features)

# Target Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_final['Target_Return_3M'].dropna(), bins=100, binrange=(-1, 2), kde=True, color=CSE_COLORS['primary'])
plt.axvline(0, color='red', linestyle='--')
plt.axvline(0.05, color='green', linestyle='--', label='5% Uptrend Threshold')
plt.title('Distribution of 3-Month Forward Returns')
plt.legend()
plt.tight_layout()
plt.show()

print("Target Class Balance (3M Uptrend > 5%):")
print(df_final['Is_Uptrend_3M'].value_counts(normalize=True) * 100)
'''))

    # Correlation Analysis
    nb.cells.append(nbformat.v4.new_markdown_cell('## 5. Feature-Target Correlation\nWhich features are most correlated with our 3-month forward returns?'))
    nb.cells.append(nbformat.v4.new_code_cell('''corr_features = ['Dist_SMA20', 'Dist_SMA50', 'Dist_SMA200', 'RSI_14', 'MACD', 
                 'BB_Width', 'NATR_14', 'ROC_10', 'ROC_21', 'Vol_63d', 'Target_Return_3M']

corr = df_final[corr_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr[['Target_Return_3M']].sort_values(by='Target_Return_3M', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-0.1, vmax=0.1)
plt.title('Feature Correlation with 3-Month Forward Return')
plt.tight_layout()
plt.show()
'''))

    # Save to Parquet
    nb.cells.append(nbformat.v4.new_markdown_cell('## 6. Save Dataset\nExporting the final ML-ready dataset for Phase 3.'))
    nb.cells.append(nbformat.v4.new_code_cell('''import os
out_path = os.path.join(BASE_PATH, 'engineered_features.parquet')
df_final.to_parquet(out_path, index=False)
print(f"Successfully saved engineered dataset to: {out_path}")
print(f"Final shape: {df_final.shape}")
'''))

    # Write notebook to file
    with open('c:/Users/HP/Documents/Stock_pulse/02_Feature_Engineering_and_Targets.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print("Notebook 02_Feature_Engineering_and_Targets.ipynb created successfully.")

if __name__ == "__main__":
    create_notebook()
