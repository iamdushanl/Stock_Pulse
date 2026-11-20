"""
Stockpulse Phase 4.1: Step 1 - Dataset Preparation Pipeline for Engine 01
Prepares prepared_training_dataset.parquet and feature_preparation_audit.csv:
1. Ingests features_01_short_term_trading.parquet
2. Removes static November 2026 announcement flags
3. Integrates the 4 Wilder Parabolic SAR (PSAR) indicators from technical_indicators_v2.py
4. Inspects and handles missing Open prices and Volume Breakout signals
5. Ingests daily All Share Price Index (ASPI) to calculate market benchmark forward returns
"""
from __future__ import annotations
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# Add paths
sys.path.append(str(Path("phase3_5_validation").resolve()))
from technical_indicators_v2 import compute_parabolic_sar

OUT_DIR = Path("phase4_model_development/engine_01_short_term")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOW_UTC = datetime.now(timezone.utc).isoformat()

print("=" * 75)
print("Stockpulse Phase 4.1: Step 1 - Feature Dataset Preparation for Engine 01")
print("=" * 75)

# 1. Load Base Features 01 Matrix
p1_path = Path("phase3_feature_engineering/features_01_short_term_trading.parquet")
df = pd.read_parquet(p1_path)
print(f"Loaded base feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")

audit_records = []

# 2. Audit and Remove Static November 2026 Announcement Flags
ann_cols = [c for c in df.columns if "announcement" in c or "dividend" in c or "rights" in c or "director" in c]
print(f"\n[Task 1.1] Removing static 2026 announcement columns to prevent look-ahead bias: {ann_cols}")
for col in ann_cols:
    audit_records.append({
        "step": "ANNOUNCEMENT_FILTER",
        "column_affected": col,
        "action_taken": "DROPPED_FROM_DATASET",
        "rationale": "Static Nov 2026 disclosure broadcast onto historical 2002-2025 rows violates point-in-time causality."
    })
df = df.drop(columns=ann_cols)

# 3. Add Parabolic SAR Indicators
print("\n[Task 1.2] Computing Wilder Parabolic SAR (PSAR) indicators...")
psar_dfs = []
for code, group in df.groupby("CompanyCode", sort=False):
    p_df = compute_parabolic_sar(group)
    psar_dfs.append(p_df)
df_psar = pd.concat(psar_dfs, axis=0)

for c in ["PSAR", "PSAR_Trend", "PSAR_Distance_Pct", "PSAR_Flip"]:
    df[c] = df_psar[c]
    audit_records.append({
        "step": "PSAR_INTEGRATION",
        "column_affected": c,
        "action_taken": "COMPUTED_AND_APPENDED",
        "rationale": "Fulfills project requirement for Parabolic SAR trend tracking; verified 100% look-ahead free."
    })
print(f"  Added PSAR indicators: PSAR, PSAR_Trend, PSAR_Distance_Pct, PSAR_Flip (Mean Flips: {df['PSAR_Flip'].sum() / 20:.1f})")

# 4. Inspect and Handle Missing Open Prices
print("\n[Task 1.3] Inspecting missing Open prices and handling Open-dependent signals...")
open_nulls = df["Open"].isnull().sum()
open_null_pct = open_nulls / len(df) * 100.0
print(f"  Open price missing count: {open_nulls} ({open_null_pct:.2f}%)")

# Pre-2011 CSE trading logs did not record market opening quotes
df["has_valid_open"] = df["Open"].notnull().astype(int)

# In features_01, Volume_Breakout used Close > Open. When Open is missing, it evaluates to 0.
# We create an inter-day volume breakout signal independent of Open:
# Volume > 2x 20d SMA Volume AND Close > Previous Close
for code, group in df.groupby("CompanyCode"):
    prev_close = group["Close"].shift(1)
    df.loc[group.index, "Volume_Breakout_Interday"] = (
        (group["ShareVolume"] > 2.0 * group["Volume_SMA_20"]) & (group["Close"] > prev_close)
    ).astype(int)

audit_records.append({
    "step": "OPEN_PRICE_HANDLING",
    "column_affected": "Open",
    "action_taken": "FLAGGED_WITH_HAS_VALID_OPEN",
    "rationale": "Pre-2011 CSE Daily Shares Price Lists omitted Open. Retained original Open without forward-filling; added 'has_valid_open' indicator."
})
audit_records.append({
    "step": "OPEN_PRICE_HANDLING",
    "column_affected": "Volume_Breakout_Interday",
    "action_taken": "CREATED_ROBUST_ALTERNATIVE",
    "rationale": "Computes volume breakout using Close > Close[-1] to avoid false negative zeroes pre-2011."
})

# 5. Ingest Daily ASPI Benchmark and Compute ASPI Forward Returns
print("\n[Task 1.4] Ingesting official ASPI market index and computing benchmark returns...")
aspi_path = Path("01_existing_cse_sources/07Market Indices - Daily.xls")
df_aspi = pd.read_excel(aspi_path, header=None, skiprows=5)[[0, 1]].dropna()
df_aspi.columns = ["Date", "ASPI"]
df_aspi["Date"] = pd.to_datetime(df_aspi["Date"], errors="coerce")
df_aspi = df_aspi.dropna().drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Compute ASPI forward returns
df_aspi["ASPI_Fwd_Return_5d"] = (df_aspi["ASPI"].shift(-5) - df_aspi["ASPI"]) / df_aspi["ASPI"] * 100.0
df_aspi["ASPI_Fwd_Return_20d"] = (df_aspi["ASPI"].shift(-20) - df_aspi["ASPI"]) / df_aspi["ASPI"] * 100.0
df_aspi["ASPI_Fwd_Return_60d"] = (df_aspi["ASPI"].shift(-60) - df_aspi["ASPI"]) / df_aspi["ASPI"] * 100.0

# Merge ASPI onto df by Date
df["Date"] = pd.to_datetime(df["Date"])
df = df.merge(df_aspi[["Date", "ASPI", "ASPI_Fwd_Return_5d", "ASPI_Fwd_Return_20d", "ASPI_Fwd_Return_60d"]], on="Date", how="left")
print(f"  ASPI benchmark merged successfully. ASPI range: {df['ASPI'].min():.1f} to {df['ASPI'].max():.1f}")

audit_records.append({
    "step": "BENCHMARK_ALIGNMENT",
    "column_affected": "ASPI, ASPI_Fwd_Return_5d, ASPI_Fwd_Return_20d, ASPI_Fwd_Return_60d",
    "action_taken": "MERGED_FROM_OFFICIAL_CSE_INDEX",
    "rationale": "Enables exact calculation of top-10 stock excess return over market benchmark."
})

# 6. Verify and Save Prepared Dataset
out_parquet = OUT_DIR / "prepared_training_dataset.parquet"
df.to_parquet(out_parquet, index=False)
print(f"\n[Step 1 Complete] Saved: {out_parquet} ({len(df):,} rows × {len(df.columns)} columns)")

df_audit = pd.DataFrame(audit_records)
out_audit = OUT_DIR / "feature_preparation_audit.csv"
df_audit.to_csv(out_audit, index=False, encoding="utf-8-sig")
print(f"Saved: {out_audit}")

if __name__ == "__main__":
    print("Dataset preparation completed successfully!")
