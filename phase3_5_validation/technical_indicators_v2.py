"""
Stockpulse Phase 3.5: Versioned Technical Indicators Engine (v2)
Maintains backward compatibility with technical_indicators.py while adding:
- PSAR (Parabolic Stop and Reverse)
- PSAR_Trend (1 = Bullish, -1 = Bearish)
- PSAR_Distance_Pct ((Close - PSAR) / Close * 100)
- PSAR_Flip (1 = Trend flip on current bar, 0 = Continuation)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add phase3_feature_engineering to sys.path to import base engine
sys.path.append(str(Path("phase3_feature_engineering").resolve()))
from technical_indicators import compute_top_50_technical_indicators

def compute_parabolic_sar(df_ticker: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.20) -> pd.DataFrame:
    """
    Computes Wilder's Parabolic SAR iteratively per ticker without look-ahead bias.
    """
    high = df_ticker["High"].values
    low = df_ticker["Low"].values
    close = df_ticker["Close"].values
    n = len(df_ticker)

    psar = np.zeros(n)
    psar_trend = np.zeros(n, dtype=int) # 1 = up, -1 = down
    psar_flip = np.zeros(n, dtype=int)

    if n < 3:
        return pd.DataFrame({
            "PSAR": psar,
            "PSAR_Trend": psar_trend,
            "PSAR_Distance_Pct": np.zeros(n),
            "PSAR_Flip": psar_flip
        }, index=df_ticker.index)

    # Initialization
    # If close[1] >= close[0], start in uptrend, else downtrend
    is_uptrend = close[1] >= close[0]
    psar_trend[0] = 1 if is_uptrend else -1
    psar[0] = low[0] if is_uptrend else high[0]
    ep = high[0] if is_uptrend else low[0]
    af = af_start

    for i in range(1, n):
        prev_sar = psar[i - 1]
        prev_trend = is_uptrend

        # Current tentative SAR
        curr_sar = prev_sar + af * (ep - prev_sar)

        if is_uptrend:
            # In uptrend, SAR cannot exceed previous two lows
            if i >= 2:
                curr_sar = min(curr_sar, low[i - 1], low[i - 2])
            else:
                curr_sar = min(curr_sar, low[i - 1])

            # Check for reversal
            if low[i] < curr_sar:
                is_uptrend = False
                curr_sar = max(ep, high[i]) # Switch to EP
                ep = low[i]
                af = af_start
                psar_flip[i] = 1
            else:
                psar_flip[i] = 0
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            # In downtrend, SAR cannot be lower than previous two highs
            if i >= 2:
                curr_sar = max(curr_sar, high[i - 1], high[i - 2])
            else:
                curr_sar = max(curr_sar, high[i - 1])

            # Check for reversal
            if high[i] > curr_sar:
                is_uptrend = True
                curr_sar = min(ep, low[i]) # Switch to EP
                ep = high[i]
                af = af_start
                psar_flip[i] = 1
            else:
                psar_flip[i] = 0
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

        psar[i] = curr_sar
        psar_trend[i] = 1 if is_uptrend else -1

    psar_dist_pct = ((close - psar) / (close + 1e-9)) * 100.0

    return pd.DataFrame({
        "PSAR": psar,
        "PSAR_Trend": psar_trend,
        "PSAR_Distance_Pct": psar_dist_pct,
        "PSAR_Flip": psar_flip
    }, index=df_ticker.index)

def compute_all_indicators_v2(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Computes base 50 indicators plus 4 PSAR indicators = 54 total indicators.
    """
    df_base = compute_top_50_technical_indicators(df_input)

    # Compute PSAR per ticker
    psar_dfs = []
    for code, group in df_base.groupby("CompanyCode", sort=False):
        psar_df = compute_parabolic_sar(group)
        psar_dfs.append(psar_df)

    psar_all = pd.concat(psar_dfs, axis=0)
    for col in psar_all.columns:
        df_base[col] = psar_all[col]

    return df_base

if __name__ == "__main__":
    print("Testing technical_indicators_v2 on SP20 dataset sample...")
    df_sample = pd.read_excel("01_existing_cse_sources/SP20_CSE_Dataset_2001_2025_Adjusted.xlsx", sheet_name="Full_Dataset_2001_2025", nrows=1000)
    df_res = compute_all_indicators_v2(df_sample)
    print("PSAR columns added:", [c for c in df_res.columns if "PSAR" in c])
    print(df_res[["Date", "CompanyCode", "Close", "PSAR", "PSAR_Trend", "PSAR_Distance_Pct", "PSAR_Flip"]].head(10))
