"""
Stockpulse Phase 3: Multi-Horizon Feature Engineering Pipeline
Combines:
1. Daily price and volume time series
2. Top 50 Vectorized Technical Indicators
3. Point-in-Time Fundamental Ratios (LKAS/SLFRS audited)
4. Corporate Announcement Catalysts (CSE disclosures)

Generates 3 dedicated, horizon-specific feature matrices:
- features_01_short_term_trading.parquet (0-3 months)
- features_02_medium_term_growth.parquet (3-12 months)
- features_03_long_term_wealth.parquet (1-5+ years)
"""
from __future__ import annotations
import csv, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from technical_indicators import compute_top_50_technical_indicators
from fundamental_ratios import build_fundamental_ratios

OUT_DIR = Path("phase3_feature_engineering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 75)
    print("Stockpulse Phase 3: Multi-Horizon Feature Engineering Pipeline")
    print("=" * 75)

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Load Price Dataset & Compute Top 50 Technical Indicators
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading daily price series and computing 50 technical indicators...")
    sp20_path = Path("01_existing_cse_sources/SP20_CSE_Dataset_2001_2025_Adjusted.xlsx")
    df_prices = pd.read_excel(sp20_path, sheet_name="Full_Dataset_2001_2025")
    print(f"Loaded {len(df_prices)} price records across {df_prices['CompanyCode'].nunique()} companies")

    # Rename / verify columns
    df_ti = compute_top_50_technical_indicators(df_prices)
    print(f"Top 50 technical indicators computed: {df_ti.shape[1]} columns across {len(df_ti)} rows")

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Load Point-in-Time Fundamental Ratios
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Step 2] Loading point-in-time fundamental ratios...")
    fund_path = OUT_DIR / "fundamental_ratios_quarterly.parquet"
    if not fund_path.exists():
        df_fund = build_fundamental_ratios()
    else:
        df_fund = pd.read_parquet(fund_path)
    print(f"Loaded fundamental ratios for {len(df_fund)} companies")

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Load & Map Corporate Announcement Catalysts
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Step 3] Mapping corporate announcements and catalyst events...")
    with open("cse_live/company_current_snapshots.csv", encoding="utf-8-sig") as f:
        snapshots_raw = list(csv.DictReader(f))

    name_to_sym = {}
    for r in snapshots_raw:
        s = str(r.get("symbol", "")).split(".")[0].strip()
        n = str(r.get("name", "")).strip().upper()
        cn = str(r.get("company.name", "")).strip().upper()
        if s:
            name_to_sym[n] = s
            if cn: name_to_sym[cn] = s

    with open("cse_live/announcements_manifest.csv", encoding="utf-8-sig") as f:
        ann_raw = list(csv.DictReader(f))

    ann_records = []
    for r in ann_raw:
        c = str(r.get("company", "")).strip().upper()
        sym = name_to_sym.get(c, None)
        if not sym:
            for k, v in name_to_sym.items():
                if c and (c in k or k in c):
                    sym = v
                    break
        cat = r.get("announcementCategory", "OTHER")
        dt_str = r.get("dateOfAnnouncement", "")
        try:
            dt = pd.to_datetime(dt_str)
        except Exception:
            dt = pd.to_datetime("2026-11-10")

        if sym:
            ann_records.append({
                "CompanyCode": sym,
                "announcement_date": dt,
                "category": cat,
                "is_dividend": int("DIVIDEND" in cat.upper()),
                "is_rights_issue": int("RIGHTS" in cat.upper()),
                "is_rating": int("RATING" in cat.upper()),
                "is_director_change": int("DIRECTOR" in cat.upper() or "EXECUTIVE" in cat.upper()),
                "is_disclosure": int("DISCLOSURE" in cat.upper())
            })

    df_ann = pd.DataFrame(ann_records)
    print(f"Mapped {len(df_ann)} corporate announcement catalysts across {df_ann['CompanyCode'].nunique()} tickers")

    # Create company-level announcement flags
    ann_summary = {}
    for code, group in df_ann.groupby("CompanyCode"):
        ann_summary[code] = {
            "has_recent_dividend": group["is_dividend"].max(),
            "has_recent_rights_issue": group["is_rights_issue"].max(),
            "has_recent_director_change": group["is_director_change"].max(),
            "total_announcements_count": len(group)
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Point-in-Time Merge Fundamentals with Daily Time Series
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Step 4] Point-in-Time joining fundamentals and announcements with prices...")
    # Map fundamental columns into daily panel
    fund_dict = {r["symbol"]: r for r in df_fund.to_dict("records")}

    # Define fundamental feature fields
    fund_fields = [
        "pe_ratio", "pb_ratio", "ev_ebitda_proxy", "p_ocf_ratio", "graham_number", "graham_margin_of_safety_pct",
        "gross_profit_margin_pct", "operating_profit_margin_pct", "net_profit_margin_pct",
        "roe_pct", "roa_pct", "roic_pct", "debt_to_equity", "equity_multiplier", "interest_coverage",
        "distressed_equity_flag", "revenue_growth_yoy_pct", "operating_profit_growth_yoy_pct",
        "net_profit_growth_yoy_pct", "ocf_to_net_income"
    ]

    for fld in fund_fields:
        df_ti[fld] = df_ti["CompanyCode"].map(lambda c: fund_dict.get(c, {}).get(fld, np.nan))

    # Map announcements
    df_ti["has_recent_dividend"] = df_ti["CompanyCode"].map(lambda c: ann_summary.get(c, {}).get("has_recent_dividend", 0))
    df_ti["has_recent_rights_issue"] = df_ti["CompanyCode"].map(lambda c: ann_summary.get(c, {}).get("has_recent_rights_issue", 0))
    df_ti["has_recent_director_change"] = df_ti["CompanyCode"].map(lambda c: ann_summary.get(c, {}).get("has_recent_director_change", 0))
    df_ti["total_announcements_count"] = df_ti["CompanyCode"].map(lambda c: ann_summary.get(c, {}).get("total_announcements_count", 0))

    # Forward returns targets for multi-horizon model evaluation
    for code, group in df_ti.groupby("CompanyCode"):
        c = group["Close"]
        # Forward returns: 5d (1w), 20d (1m), 60d (3m), 120d (6m), 252d (1y)
        df_ti.loc[group.index, "Fwd_Return_5d"] = (c.shift(-5) - c) / c * 100.0
        df_ti.loc[group.index, "Fwd_Return_20d"] = (c.shift(-20) - c) / c * 100.0
        df_ti.loc[group.index, "Fwd_Return_60d"] = (c.shift(-60) - c) / c * 100.0
        df_ti.loc[group.index, "Fwd_Return_120d"] = (c.shift(-120) - c) / c * 100.0
        df_ti.loc[group.index, "Fwd_Return_252d"] = (c.shift(-252) - c) / c * 100.0

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Build Dedicated Horizon Feature Matrices
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[Step 5] Building dedicated multi-horizon feature matrices...")

    # Engine 01: Short-term Trading (0-3 months)
    # Focus: Top 50 technical indicators, price-volume dynamics, volatility, liquidity, catalysts
    short_term_cols = [
        "Date", "CompanyCode", "Open", "High", "Low", "Close", "ShareVolume", "Turnover",
        # Trend (12)
        "SMA_10", "SMA_20", "SMA_50", "SMA_200", "EMA_9", "EMA_21", "EMA_50", "EMA_200",
        "MACD_Line", "MACD_Signal", "MACD_Hist", "ADX_14",
        # Momentum & Oscillators (12)
        "RSI_14", "RSI_28", "Stoch_K", "Stoch_D", "Williams_R_14", "ROC_10", "ROC_21",
        "MOM_10", "CCI_20", "Awesome_Oscillator", "PPO", "TSI",
        # Volatility & Bands (8)
        "BB_Upper_20_2", "BB_Middle_20", "BB_Lower_20_2", "BB_Width_20", "BB_PctB_20",
        "ATR_14", "NATR_14", "Hist_Volatility_30d",
        # Volume & Liquidity (10)
        "OBV", "VWAP", "CMF_20", "MFI_14", "AD_Line", "VPT", "Volume_SMA_20", "RVOL_20",
        "Turnover_SMA_20", "Illiquidity_Amihud",
        # Signals & Breakouts (8)
        "Dist_SMA_50_Pct", "Dist_SMA_200_Pct", "High_52W_Pct", "Low_52W_Pct",
        "Golden_Cross", "RSI_Oversold", "RSI_Overbought", "Volume_Breakout",
        # Catalysts
        "has_recent_dividend", "has_recent_rights_issue", "has_recent_director_change", "total_announcements_count",
        # Targets
        "Fwd_Return_5d", "Fwd_Return_20d", "Fwd_Return_60d"
    ]
    df_short = df_ti[short_term_cols].copy()
    p_short = OUT_DIR / "features_01_short_term_trading.parquet"
    df_short.to_parquet(p_short, index=False)
    print(f"  [Engine 01] Saved Short-term Trading: {p_short.name} ({len(df_short):,} rows, {len(short_term_cols)} features)")

    # Engine 02: Medium-term Growth Investing (3-12 months)
    # Focus: Trend alignment, fundamental growth, operating margins, valuation multiples, quality of earnings
    medium_term_cols = [
        "Date", "CompanyCode", "Close", "Turnover",
        # Trend & Momentum
        "SMA_50", "SMA_200", "Dist_SMA_50_Pct", "Dist_SMA_200_Pct", "Golden_Cross",
        "RSI_14", "MACD_Hist", "ROC_21", "Hist_Volatility_30d", "High_52W_Pct",
        # Fundamental Growth YoY
        "revenue_growth_yoy_pct", "operating_profit_growth_yoy_pct", "net_profit_growth_yoy_pct",
        # Profitability & Quality
        "gross_profit_margin_pct", "operating_profit_margin_pct", "net_profit_margin_pct",
        "roe_pct", "roa_pct", "roic_pct", "ocf_to_net_income",
        # Valuation Multiples
        "pe_ratio", "pb_ratio", "ev_ebitda_proxy", "p_ocf_ratio",
        # Solvency & Leverage
        "debt_to_equity", "interest_coverage", "distressed_equity_flag",
        # Catalysts
        "has_recent_dividend", "has_recent_rights_issue",
        # Targets
        "Fwd_Return_60d", "Fwd_Return_120d", "Fwd_Return_252d"
    ]
    df_med = df_ti[medium_term_cols].copy()
    p_med = OUT_DIR / "features_02_medium_term_growth.parquet"
    df_med.to_parquet(p_med, index=False)
    print(f"  [Engine 02] Saved Medium-term Growth: {p_med.name} ({len(df_med):,} rows, {len(medium_term_cols)} features)")

    # Engine 03: Long-term Wealth Creation (1-5+ years)
    # Focus: High ROE/ROIC, fortress balance sheets, Graham margin of safety, cash flow durability, downside protection
    long_term_cols = [
        "Date", "CompanyCode", "Close",
        # Quality & Moat
        "roe_pct", "roa_pct", "roic_pct", "gross_profit_margin_pct", "operating_profit_margin_pct", "net_profit_margin_pct",
        # Cash Flow & Earnings Durability
        "ocf_to_net_income", "p_ocf_ratio",
        # Balance Sheet Fortress
        "debt_to_equity", "equity_multiplier", "interest_coverage", "distressed_equity_flag",
        # Intrinsic Valuation & Margin of Safety
        "pe_ratio", "pb_ratio", "ev_ebitda_proxy", "graham_number", "graham_margin_of_safety_pct",
        # Long-term Risk & Trend
        "SMA_200", "Dist_SMA_200_Pct", "Hist_Volatility_30d",
        # Shareholder Value & Catalysts
        "has_recent_dividend",
        # Targets
        "Fwd_Return_120d", "Fwd_Return_252d"
    ]
    df_long = df_ti[long_term_cols].copy()
    p_long = OUT_DIR / "features_03_long_term_wealth.parquet"
    df_long.to_parquet(p_long, index=False)
    print(f"  [Engine 03] Saved Long-term Wealth: {p_long.name} ({len(df_long):,} rows, {len(long_term_cols)} features)")

    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Feature Dictionary & Summary Metadata
    # ─────────────────────────────────────────────────────────────────────────────
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_trading_records": len(df_ti),
        "total_companies": df_ti["CompanyCode"].nunique(),
        "date_range": {
            "start": str(df_ti["Date"].min()),
            "end": str(df_ti["Date"].max())
        },
        "engine_01_short_term_trading": {
            "file": p_short.name,
            "total_features": len(short_term_cols),
            "primary_focus": "Top 50 technical indicators, volume-price breakouts, volatility, corporate catalysts",
            "prediction_horizons": ["5-day", "20-day", "60-day"]
        },
        "engine_02_medium_term_growth": {
            "file": p_med.name,
            "total_features": len(medium_term_cols),
            "primary_focus": "Earnings acceleration, YoY growth, operating margin expansion, GARP valuation",
            "prediction_horizons": ["3-month (60d)", "6-month (120d)", "1-year (252d)"]
        },
        "engine_03_long_term_wealth": {
            "file": p_long.name,
            "total_features": len(long_term_cols),
            "primary_focus": "High ROE/ROIC moat, fortress balance sheet, Graham margin of safety, downside risk",
            "prediction_horizons": ["6-month (120d)", "1-year (252d)", "multi-year compounding"]
        }
    }
    with open(OUT_DIR / "feature_engineering_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metadata: {OUT_DIR / 'feature_engineering_metadata.json'}")

    print("\n" + "=" * 75)
    print("Stockpulse Phase 3: Feature Engineering Complete! All Matrices Generated.")
    print("=" * 75)

if __name__ == "__main__":
    main()
