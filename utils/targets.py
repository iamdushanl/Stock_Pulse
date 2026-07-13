import pandas as pd
import numpy as np

def generate_targets(df):
    """
    Generate target variables for the recommendation system.
    Calculations are applied per CompanyCode based on forward returns.
    
    Target Horizons:
    - Short term: 21 trading days (~1 month)
    - Medium term: 63 trading days (~3 months)
    - Long term: 126 trading days (~6 months)
    """
    print("Generating target variables...")
    df_targets = df.copy()
    
    def apply_targets(group):
        group = group.copy()
        
        # Continuous Forward Returns (percentage change from current close to future close)
        # We shift by negative periods to look into the future
        group['Target_Return_1M'] = group['Close'].shift(-21) / group['Close'] - 1
        group['Target_Return_3M'] = group['Close'].shift(-63) / group['Close'] - 1
        group['Target_Return_6M'] = group['Close'].shift(-126) / group['Close'] - 1
        
        # Max Drawdown in the forward period (Risk measure)
        # Helps identify stocks that might go up in 3M but suffer a massive drop in between
        group['Forward_Min_1M'] = group['Close'].rolling(window=21).min().shift(-21)
        group['Target_MaxDrawdown_1M'] = (group['Forward_Min_1M'] - group['Close']) / group['Close']
        
        # Risk-Adjusted Returns (Return / Volatility)
        # Uses the current 63-day volatility as a risk proxy
        group['Target_RiskAdj_3M'] = group['Target_Return_3M'] / group['Vol_63d'].replace(0, np.nan)
        
        # Binary Classification Labels (Uptrend vs Not)
        # Defining an uptrend as a return > 5% in the period
        group['Is_Uptrend_1M'] = (group['Target_Return_1M'] > 0.05).astype(int)
        group['Is_Uptrend_3M'] = (group['Target_Return_3M'] > 0.05).astype(int)
        group['Is_Uptrend_6M'] = (group['Target_Return_6M'] > 0.05).astype(int)
        
        # If the forward return is NaN (end of dataset), the binary label should also be NaN, not 0
        group.loc[group['Target_Return_1M'].isna(), 'Is_Uptrend_1M'] = np.nan
        group.loc[group['Target_Return_3M'].isna(), 'Is_Uptrend_3M'] = np.nan
        group.loc[group['Target_Return_6M'].isna(), 'Is_Uptrend_6M'] = np.nan
        
        return group
        
    target_dfs = []
    for company, group in df_targets.groupby('CompanyCode'):
        processed_group = apply_targets(group)
        processed_group['CompanyCode'] = company
        target_dfs.append(processed_group)
        
    df_targets = pd.concat(target_dfs, ignore_index=True)
    
    print("Target variables generation complete.")
    return df_targets
