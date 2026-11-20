"""
Stockpulse Phase 3: Top 50 Vectorized Technical Indicators Engine
Computes 50 standard technical analysis indicators across price and volume time series
without look-ahead bias.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_top_50_technical_indicators(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Computes top 50 technical indicators for a price dataframe.
    Expected columns: ['Date', 'CompanyCode', 'Open', 'High', 'Low', 'Close', 'ShareVolume', 'Turnover']
    Sorts by CompanyCode and Date before computing.
    """
    df = df_input.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["CompanyCode", "Date"]).reset_index(drop=True)

    results = []
    for code, group in df.groupby("CompanyCode", sort=False):
        g = group.copy()
        c = g["Close"]
        h = g["High"]
        l = g["Low"]
        o = g["Open"]
        v = g["ShareVolume"].fillna(0)
        t = g["Turnover"].fillna(0)

        # ── 1. Trend & Moving Averages (12) ──────────────────────────────────
        g["SMA_10"] = c.rolling(10, min_periods=5).mean()
        g["SMA_20"] = c.rolling(20, min_periods=10).mean()
        g["SMA_50"] = c.rolling(50, min_periods=25).mean()
        g["SMA_200"] = c.rolling(200, min_periods=100).mean()

        g["EMA_9"] = c.ewm(span=9, adjust=False).mean()
        g["EMA_21"] = c.ewm(span=21, adjust=False).mean()
        g["EMA_50"] = c.ewm(span=50, adjust=False).mean()
        g["EMA_200"] = c.ewm(span=200, adjust=False).mean()

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        g["MACD_Line"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD_Line"].ewm(span=9, adjust=False).mean()
        g["MACD_Hist"] = g["MACD_Line"] - g["MACD_Signal"]

        # ADX (14)
        prev_c = c.shift(1)
        tr1 = h - l
        tr2 = (h - prev_c).abs()
        tr3 = (l - prev_c).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=7).mean()

        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=g.index).rolling(14, min_periods=7).mean() / (atr14 + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=g.index).rolling(14, min_periods=7).mean() / (atr14 + 1e-9))
        dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9))
        g["ADX_14"] = dx.rolling(14, min_periods=7).mean()

        # ── 2. Momentum & Oscillators (12) ───────────────────────────────────
        # RSI 14 & 28
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain14 = gain.rolling(14, min_periods=7).mean()
        avg_loss14 = loss.rolling(14, min_periods=7).mean()
        rs14 = avg_gain14 / (avg_loss14 + 1e-9)
        g["RSI_14"] = 100 - (100 / (1 + rs14))

        avg_gain28 = gain.rolling(28, min_periods=14).mean()
        avg_loss28 = loss.rolling(28, min_periods=14).mean()
        rs28 = avg_gain28 / (avg_loss28 + 1e-9)
        g["RSI_28"] = 100 - (100 / (1 + rs28))

        # Stochastic %K & %D (14)
        low14 = l.rolling(14, min_periods=7).min()
        high14 = h.rolling(14, min_periods=7).max()
        g["Stoch_K"] = 100 * ((c - low14) / ((high14 - low14) + 1e-9))
        g["Stoch_D"] = g["Stoch_K"].rolling(3, min_periods=2).mean()

        # Williams %R (14)
        g["Williams_R_14"] = -100 * ((high14 - c) / ((high14 - low14) + 1e-9))

        # Rate of Change (ROC 10 & 21)
        g["ROC_10"] = (c - c.shift(10)) / (c.shift(10) + 1e-9) * 100
        g["ROC_21"] = (c - c.shift(21)) / (c.shift(21) + 1e-9) * 100

        # Momentum (10)
        g["MOM_10"] = c - c.shift(10)

        # Commodity Channel Index (CCI 20)
        tp = (h + l + c) / 3.0
        sma_tp = tp.rolling(20, min_periods=10).mean()
        mad_tp = tp.rolling(20, min_periods=10).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        g["CCI_20"] = (tp - sma_tp) / (0.015 * (mad_tp + 1e-9))

        # Awesome Oscillator (SMA 5 median - SMA 34 median)
        mp = (h + l) / 2.0
        g["Awesome_Oscillator"] = mp.rolling(5, min_periods=3).mean() - mp.rolling(34, min_periods=17).mean()

        # Percentage Price Oscillator (PPO)
        g["PPO"] = ((ema12 - ema26) / (ema26 + 1e-9)) * 100

        # True Strength Index (TSI 25, 13)
        diff = c.diff()
        smooth1 = diff.ewm(span=25, adjust=False).mean()
        smooth2 = smooth1.ewm(span=13, adjust=False).mean()
        abs_smooth1 = diff.abs().ewm(span=25, adjust=False).mean()
        abs_smooth2 = abs_smooth1.ewm(span=13, adjust=False).mean()
        g["TSI"] = 100 * (smooth2 / (abs_smooth2 + 1e-9))

        # ── 3. Volatility & Bands (8) ─────────────────────────────────────────
        sma20 = g["SMA_20"]
        std20 = c.rolling(20, min_periods=10).std()
        g["BB_Middle_20"] = sma20
        g["BB_Upper_20_2"] = sma20 + (2.0 * std20)
        g["BB_Lower_20_2"] = sma20 - (2.0 * std20)
        g["BB_Width_20"] = (g["BB_Upper_20_2"] - g["BB_Lower_20_2"]) / (sma20 + 1e-9)
        g["BB_PctB_20"] = (c - g["BB_Lower_20_2"]) / ((g["BB_Upper_20_2"] - g["BB_Lower_20_2"]) + 1e-9)

        # ATR 14 & NATR 14
        g["ATR_14"] = atr14
        g["NATR_14"] = (atr14 / (c + 1e-9)) * 100

        # Historical Volatility (30d annualized log returns)
        log_ret = np.log(c / (c.shift(1) + 1e-9))
        g["Hist_Volatility_30d"] = log_ret.rolling(30, min_periods=15).std() * np.sqrt(252)

        # ── 4. Volume & Liquidity (10) ───────────────────────────────────────
        # On-Balance Volume (OBV)
        obv_direction = np.sign(c.diff().fillna(0))
        g["OBV"] = (obv_direction * v).cumsum()

        # VWAP (rolling 20d)
        cum_vp = (c * v).rolling(20, min_periods=5).sum()
        cum_vol = v.rolling(20, min_periods=5).sum()
        g["VWAP"] = cum_vp / (cum_vol + 1e-9)

        # Chaikin Money Flow (CMF 20)
        clv = ((c - l) - (h - c)) / ((h - l) + 1e-9)
        vol_money = clv * v
        g["CMF_20"] = vol_money.rolling(20, min_periods=10).sum() / (v.rolling(20, min_periods=10).sum() + 1e-9)

        # Money Flow Index (MFI 14)
        raw_money = tp * v
        tp_diff = tp.diff()
        pos_flow = np.where(tp_diff > 0, raw_money, 0.0)
        neg_flow = np.where(tp_diff < 0, raw_money, 0.0)
        pos_sum = pd.Series(pos_flow, index=g.index).rolling(14, min_periods=7).sum()
        neg_sum = pd.Series(neg_flow, index=g.index).rolling(14, min_periods=7).sum()
        money_ratio = pos_sum / (neg_sum + 1e-9)
        g["MFI_14"] = 100 - (100 / (1 + money_ratio))

        # Accumulation / Distribution Line (AD Line)
        g["AD_Line"] = vol_money.cumsum()

        # Volume Price Trend (VPT)
        g["VPT"] = (v * (c.diff() / (prev_c + 1e-9))).cumsum()

        # Volume Moving Averages & RVOL
        g["Volume_SMA_20"] = v.rolling(20, min_periods=10).mean()
        g["RVOL_20"] = v / (g["Volume_SMA_20"] + 1e-9)

        # Turnover Moving Average & Amihud Illiquidity
        g["Turnover_SMA_20"] = t.rolling(20, min_periods=10).mean()
        ret_abs = (c.pct_change()).abs()
        g["Illiquidity_Amihud"] = (ret_abs / (t + 1e-9)) * 1e6

        # ── 5. Signals, Trends & Risk Indicators (8) ─────────────────────────
        # Distance from Moving Averages (%)
        g["Dist_SMA_50_Pct"] = ((c - g["SMA_50"]) / (g["SMA_50"] + 1e-9)) * 100
        g["Dist_SMA_200_Pct"] = ((c - g["SMA_200"]) / (g["SMA_200"] + 1e-9)) * 100

        # 52-Week High & Low Proximity (%)
        high_252 = h.rolling(252, min_periods=60).max()
        low_252 = l.rolling(252, min_periods=60).min()
        g["High_52W_Pct"] = ((c - high_252) / (high_252 + 1e-9)) * 100
        g["Low_52W_Pct"] = ((c - low_252) / (low_252 + 1e-9)) * 100

        # Golden Cross (SMA 50 > SMA 200)
        g["Golden_Cross"] = (g["SMA_50"] > g["SMA_200"]).astype(int)

        # RSI Overbought / Oversold
        g["RSI_Oversold"] = (g["RSI_14"] < 30).astype(int)
        g["RSI_Overbought"] = (g["RSI_14"] > 70).astype(int)

        # Volume Breakout (> 2x 20d avg volume with positive price close)
        g["Volume_Breakout"] = ((v > (2.0 * g["Volume_SMA_20"])) & (c > o)).astype(int)

        results.append(g)

    return pd.concat(results, ignore_index=True)

if __name__ == "__main__":
    print("Testing technical indicators calculation on SP20 dataset...")
    df_raw = pd.read_excel("01_existing_cse_sources/SP20_CSE_Dataset_2001_2025_Adjusted.xlsx", sheet_name="Full_Dataset_2001_2025")
    df_ti = compute_top_50_technical_indicators(df_raw)
    print(f"Calculated {len(df_ti.columns)} columns across {len(df_ti)} rows!")
    print("Sample computed columns:", [c for c in df_ti.columns if c not in df_raw.columns][:15])
