import pandas as pd
import numpy as np
import warnings

def clean_and_prepare_data(df, df_splits=None, start_year=2001):
    """
    Master function to clean and prepare the raw daily prices dataframe.
    
    Args:
        df (pd.DataFrame): Raw daily prices dataframe
        df_splits (pd.DataFrame): Splits dataframe for adjustments
        start_year (int): Year to filter the dataset from (default 2001)
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for feature engineering
    """
    print(f"Starting data cleaning. Initial shape: {df.shape}")
    
    # 1. Filter to Modern Era (2001-2025)
    # The early era lacks OHLCV data which is required for technical indicators
    df_clean = df[df['Date'].dt.year >= start_year].copy()
    print(f"Filtered to >= {start_year}. Shape: {df_clean.shape}")
    
    # 2. Sort by Company and Date
    df_clean = df_clean.sort_values(['CompanyCode', 'Date']).reset_index(drop=True)
    
    # 3. Handle Zero Prices / Missing Data
    # Some stocks have days with 0 volume and 0 price. We replace 0 prices with NaN to forward-fill
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        
    # 4. Standardize Trading Calendar (Forward-fill illiquid days)
    # We want a continuous calendar for every stock between its first and last trading day
    print("Aligning trading calendars and forward-filling missing days...")
    df_clean = align_trading_calendar(df_clean)
    
    # 5. Apply Stock Splits
    if df_splits is not None and not df_splits.empty:
        print("Adjusting historical prices for stock splits...")
        df_clean = apply_split_adjustments(df_clean, df_splits)
    else:
        print("Warning: No splits data provided. Prices will remain unadjusted.")
        
    # 6. Final cleanup
    # Re-calculate daily returns after splits and fills
    df_clean['DailyReturn'] = df_clean.groupby('CompanyCode')['Close'].pct_change()
    
    print(f"Data cleaning complete. Final shape: {df_clean.shape}")
    return df_clean


def align_trading_calendar(df, max_fill_days=5):
    """
    Creates a standardized trading calendar for each stock and forward-fills 
    missing prices for up to max_fill_days.
    """
    # Get the global market trading calendar (all unique dates in the dataset)
    market_dates = pd.Series(df['Date'].unique()).sort_values().reset_index(drop=True)
    
    # Find min/max dates per stock
    stock_ranges = df.groupby('CompanyCode')['Date'].agg(['min', 'max'])
    
    clean_dfs = []
    
    # Process each stock (using a loop for safety, though pandas reindex is faster, 
    # doing it per-stock ensures we don't extend dates beyond a stock's listing life)
    for company, group in df.groupby('CompanyCode'):
        min_date = stock_ranges.loc[company, 'min']
        max_date = stock_ranges.loc[company, 'max']
        
        # Valid trading days for this specific stock's lifespan
        valid_dates = market_dates[(market_dates >= min_date) & (market_dates <= max_date)]
        
        # Set index to date for reindexing
        group = group.set_index('Date')
        
        # Reindex to the full valid trading calendar
        group = group.reindex(valid_dates)
        
        # Forward fill prices up to max_fill_days
        price_cols = ['Open', 'High', 'Low', 'Close']
        group[price_cols] = group[price_cols].ffill(limit=max_fill_days)
        
        # For volume, missing days mean 0 volume
        vol_cols = ['TradeVolume', 'ShareVolume', 'Turnover']
        group[vol_cols] = group[vol_cols].fillna(0)
        
        # Forward fill metadata
        meta_cols = ['CompanyCode', 'ShortName', 'MainType', 'SubType', 'Era']
        group[meta_cols] = group[meta_cols].ffill().bfill()
        
        # Reset index to get Date back as a column
        group = group.reset_index().rename(columns={'index': 'Date'})
        clean_dfs.append(group)
        
    # Combine back
    return pd.concat(clean_dfs, ignore_index=True)


def apply_split_adjustments(df, df_splits):
    """
    Adjusts historical prices backwards to account for stock splits.
    If a 1-to-4 split happens, all prices prior to the split date are divided by 4,
    and all volumes are multiplied by 4.
    """
    df_adjusted = df.copy()
    
    # Ensure date formats
    df_splits = df_splits.copy()
    df_splits['EFFECTIVE DATE'] = pd.to_datetime(df_splits['EFFECTIVE DATE'], errors='coerce')
    df_splits = df_splits.dropna(subset=['EFFECTIVE DATE', 'OLD PROPORTION', 'NEW PROPORTION'])
    
    # Calculate split ratio
    # e.g., 1 old share becomes 4 new shares -> ratio = 4/1 = 4.0
    # Meaning past prices should be divided by 4, past volume multiplied by 4.
    df_splits['SplitRatio'] = df_splits['NEW PROPORTION'] / df_splits['OLD PROPORTION']
    
    # Some company IDs in splits might have '.N0000' appended. Clean it.
    df_splits['CleanCompany'] = df_splits['COMPANY ID'].astype(str).str.split('.').str[0].str.strip()
    
    price_cols = ['Open', 'High', 'Low', 'Close']
    vol_cols = ['TradeVolume', 'ShareVolume']
    
    # Sort splits by effective date descending (newest first)
    # This allows chaining adjustments properly if a stock split multiple times
    df_splits = df_splits.sort_values('EFFECTIVE DATE', ascending=False)
    
    for _, row in df_splits.iterrows():
        company = row['CleanCompany']
        eff_date = row['EFFECTIVE DATE']
        ratio = row['SplitRatio']
        
        if ratio <= 0 or ratio == 1.0:
            continue
            
        # Create mask for rows that need adjustment (this company, before effective date)
        mask = (df_adjusted['CompanyCode'] == company) & (df_adjusted['Date'] < eff_date)
        
        if mask.sum() > 0:
            # Adjust prices down
            df_adjusted.loc[mask, price_cols] = df_adjusted.loc[mask, price_cols] / ratio
            
            # Adjust volume up
            df_adjusted.loc[mask, vol_cols] = df_adjusted.loc[mask, vol_cols] * ratio
            
    return df_adjusted
