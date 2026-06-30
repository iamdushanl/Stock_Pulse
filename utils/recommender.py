import pandas as pd
import numpy as np

def generate_recommendations(df_current, model, feature_cols, min_price=1.0, max_volatility=0.60, top_n=10):
    """
    Generates actionable stock recommendations based on current market data and ML model.
    
    Args:
        df_current (pd.DataFrame): Dataframe containing only the most recent date's row per stock.
        model: Trained ML model (e.g., XGBoost).
        feature_cols (list): Features the model expects.
        min_price (float): Filter out penny stocks below this price.
        max_volatility (float): Filter out hyper-volatile stocks (annualized).
        top_n (int): Number of top recommendations to return.
        
    Returns:
        pd.DataFrame: Ranked recommendations.
    """
    # 1. Base Filters (Risk Management)
    df_filtered = df_current[
        (df_current['Close'] >= min_price) & 
        (df_current['Vol_63d'] <= max_volatility)
    ].copy()
    
    # 2. Handle NaNs in current features
    # If a stock doesn't have enough history for an indicator (e.g., SMA 200), we drop it for this round
    df_filtered = df_filtered.dropna(subset=feature_cols)
    
    if len(df_filtered) == 0:
        print("Warning: No stocks passed the filters or had complete feature sets.")
        return pd.DataFrame()
        
    # 3. Predict Probability of Uptrend
    X_current = df_filtered[feature_cols]
    
    # Get probability of class 1 (Uptrend)
    probabilities = model.predict_proba(X_current)[:, 1]
    df_filtered['Uptrend_Probability'] = probabilities
    
    # 4. Rank and Format
    recommendations = df_filtered.sort_values(by='Uptrend_Probability', ascending=False).head(top_n)
    
    output_cols = [
        'CompanyCode', 'Date', 'Close', 'Uptrend_Probability', 
        'Vol_63d', 'RSI_14', 'MACD', 'Dist_SMA50'
    ]
    
    return recommendations[output_cols].reset_index(drop=True)


def multi_horizon_recommendations(df_current, models_dict, feature_cols, top_n=5):
    """
    Runs recommendations across Short (1M), Medium (3M), and Long (6M) models.
    models_dict format: {'1M': model_1m, '3M': model_3m, '6M': model_6m}
    """
    recs = {}
    for horizon, model in models_dict.items():
        print(f"\\nGenerating Top {top_n} {horizon} Recommendations...")
        r = generate_recommendations(df_current, model, feature_cols, top_n=top_n)
        recs[horizon] = r
    
    return recs
