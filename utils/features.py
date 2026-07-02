import pandas as pd
import numpy as np

def generate_technical_features(df):
    """
    Generate technical indicators for the cleaned dataframe.
    Calculations are applied per CompanyCode.
    """
    print("Generating technical features...")
    df_feat = df.copy()
    
    # We will use pandas groupby and apply for calculating indicators per stock
    # To optimize speed, we define a function to apply to each group
    
    def apply_indicators(group, company_code):
        group = group.copy()
        
        # 1. Moving Averages
        group['SMA_20'] = group['Close'].rolling(window=20).mean()
        group['SMA_50'] = group['Close'].rolling(window=50).mean()
        group['SMA_200'] = group['Close'].rolling(window=200).mean()
        
        # Distance from Moving Averages (Trend strength)
        group['Dist_SMA20'] = (group['Close'] - group['SMA_20']) / group['SMA_20']
        group['Dist_SMA50'] = (group['Close'] - group['SMA_50']) / group['SMA_50']
        group['Dist_SMA200'] = (group['Close'] - group['SMA_200']) / group['SMA_200']
        
        # 2. RSI (Relative Strength Index) 14-day
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        group['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (Moving Average Convergence Divergence)
        ema_12 = group['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = group['Close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = ema_12 - ema_26
        group['MACD_Signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
        group['MACD_Hist'] = group['MACD'] - group['MACD_Signal']
        
        # 4. Bollinger Bands (20-day, 2 std dev)
        std_20 = group['Close'].rolling(window=20).std()
        group['BB_Upper'] = group['SMA_20'] + (2 * std_20)
        group['BB_Lower'] = group['SMA_20'] - (2 * std_20)
        group['BB_Width'] = (group['BB_Upper'] - group['BB_Lower']) / group['SMA_20']
        
        # 5. ATR (Average True Range) 14-day
        high_low = group['High'] - group['Low']
        high_close = np.abs(group['High'] - group['Close'].shift())
        low_close = np.abs(group['Low'] - group['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        group['ATR_14'] = true_range.rolling(14).mean()
        # Normalized ATR
        group['NATR_14'] = group['ATR_14'] / group['Close']
        
        # 6. Momentum / Rate of Change (ROC)
        group['ROC_10'] = group['Close'].pct_change(periods=10)
        group['ROC_21'] = group['Close'].pct_change(periods=21)
        
        # 6b. Stochastic Oscillator (14-day)
        low_14 = group['Low'].rolling(window=14).min()
        high_14 = group['High'].rolling(window=14).max()
        group['Stoch_K'] = 100 * (group['Close'] - low_14) / (high_14 - low_14)
        group['Stoch_D'] = group['Stoch_K'].rolling(window=3).mean()
        
        # 7. Volatility Features
        group['Vol_21d'] = group['DailyReturn'].rolling(21).std() * np.sqrt(252)
        group['Vol_63d'] = group['DailyReturn'].rolling(63).std() * np.sqrt(252)
        
        # 8. Volume Features
        group['Volume_SMA_20'] = group['ShareVolume'].rolling(window=20).mean()
        # Volume ratio (current volume vs 20d average)
        group['Volume_Ratio'] = group['ShareVolume'] / group['Volume_SMA_20'].replace(0, np.nan)
        
        # Explicitly restore CompanyCode
        group['CompanyCode'] = company_code
        
        return group
    
    # Apply the function per company
    # This might take a minute on a large dataset
    # We loop over groupby to avoid Pandas index exclusion behavior and preserve CompanyCode
    clean_dfs = []
    for company, group in df_feat.groupby('CompanyCode'):
        processed_group = apply_indicators(group, company)
        clean_dfs.append(processed_group)
        
    df_feat = pd.concat(clean_dfs, ignore_index=True)
    
    # 9. Time-Based Features (Global)
    df_feat['Year'] = df_feat['Date'].dt.year
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Quarter'] = df_feat['Date'].dt.quarter
    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek
    
    print("Technical features generation complete.")
    return df_feat
