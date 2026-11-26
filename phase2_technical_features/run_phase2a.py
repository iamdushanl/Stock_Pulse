"""
Phase 2A Runner – Full CSE Technical Indicator Engineering
===========================================================
Executes technical, price-volume, and market-relative feature generation
across all 719 SecurityIDs (1,414,977 rows) from Phase 1B outputs without
modifying Phase 1A or Phase 1B files.

Outputs written to Dataset/processed/phase2/:
  - cse_technical_features.parquet
  - cse_price_volume_features.parquet
  - cse_market_relative_features.parquet
  - technical_indicator_metadata.csv
  - feature_quality_report.csv
  - phase2_manifest.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2_technical_features.indicators_core import (
    compute_market_relative_panel,
    compute_technical_features_panel,
)
from phase2_technical_features.metadata_catalog import get_indicator_metadata_df
from phase2_technical_features.validation import (
    build_feature_quality_report,
    run_validation_suite,
)
from utils.data_loader import load_gics, load_market_indices

RAW_DIR = REPO_ROOT / "01_existing_cse_sources"
PHASE1A_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1"
PHASE1B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1b"
OUT_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2"


def run_phase2a() -> dict:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Stockpulse Phase 2A – Full CSE Technical Indicator Engineering")
    print("=" * 72)

    # 1. Load validated Phase 1B & market benchmark sources
    print("\n[1/5] Loading Phase 1B validated datasets and market/sector benchmarks...")
    df_adj = pd.read_parquet(PHASE1B_DIR / "cse_daily_prices_split_adjusted.parquet")
    df_raw = pd.read_parquet(PHASE1B_DIR / "cse_daily_prices_integrated.parquet")
    df_sec_master = pd.read_parquet(PHASE1B_DIR / "cse_security_master_enriched.parquet")
    df_sector_map = pd.read_parquet(PHASE1B_DIR / "cse_sector_mapping.parquet")
    df_calendar = pd.read_parquet(PHASE1A_DIR / "cse_market_calendar.parquet")

    assert len(df_adj) == len(df_raw), "Split-adjusted and integrated row count mismatch"
    market_indices = load_market_indices(str(RAW_DIR))
    gics_df = load_gics(str(RAW_DIR))
    print(
        f"      Loaded {len(df_adj):,} rows across {df_adj['SecurityID'].nunique()} SecurityIDs, "
        f"{df_adj['CompanyCode'].nunique()} CompanyCodes ({len(df_calendar):,} calendar days)."
    )

    # 2. Compute technical + price-volume indicators (Groups A-E) per SecurityID
    print("\n[2/5] Computing Groups A-E technical & price-volume indicators per SecurityID...")
    t1 = time.time()
    tech_df, pv_df = compute_technical_features_panel(df_adj)
    # Include untouched RawClose reference column alongside SplitAdjClose & CapitalAdjClose
    raw_ref = df_raw[["SecurityID", "Date", "Close"]].rename(columns={"Close": "RawClose"})
    pv_df = pv_df.merge(raw_ref, on=["SecurityID", "Date"], how="left")
    cols_pv = (
        ["SecurityID", "CompanyCode", "Date", "MainType", "SubType", "RawClose", "SplitAdjClose", "CapitalAdjClose"]
        + [c for c in pv_df.columns if c not in ("SecurityID", "CompanyCode", "Date", "MainType", "SubType", "RawClose", "SplitAdjClose", "CapitalAdjClose")]
    )
    pv_df = pv_df[cols_pv]
    print(f"      Completed Groups A-E in {time.time() - t1:.2f}s.")

    # 3. Compute market & sector relative indicators (Group F)
    print("\n[3/5] Computing Group F market & sector relative indicators...")
    t2 = time.time()
    mr_df = compute_market_relative_panel(df_adj, market_indices, df_sector_map, gics_df=gics_df)
    print(f"      Completed Group F in {time.time() - t2:.2f}s.")

    # 4. Write metadata & run comprehensive quality/validation suite
    print("\n[4/5] Running quality, causality, PSAR & corporate-action validation suite...")
    meta_df = get_indicator_metadata_df()
    quality_df = build_feature_quality_report(tech_df, pv_df, mr_df, meta_df)
    val_summary = run_validation_suite(
        df_adj=df_adj,
        tech_df=tech_df,
        pv_df=pv_df,
        mr_df=mr_df,
        quality_df=quality_df,
        market_indices=market_indices,
        sector_mapping=df_sector_map,
        gics_df=gics_df,
    )

    # 5. Persist Phase 2A outputs (atomic write to Dataset/processed/phase2/)
    print("\n[5/5] Writing Phase 2A Parquet, CSV, and manifest deliverables...")
    tech_df.to_parquet(OUT_DIR / "cse_technical_features.parquet", index=False)
    pv_df.to_parquet(OUT_DIR / "cse_price_volume_features.parquet", index=False)
    mr_df.to_parquet(OUT_DIR / "cse_market_relative_features.parquet", index=False)
    meta_df.to_csv(OUT_DIR / "technical_indicator_metadata.csv", index=False)
    quality_df.to_csv(OUT_DIR / "feature_quality_report.csv", index=False)

    manifest = {
        "phase": "2A",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "input_sources": {
            "split_adjusted_prices": str(PHASE1B_DIR / "cse_daily_prices_split_adjusted.parquet"),
            "integrated_raw_prices": str(PHASE1B_DIR / "cse_daily_prices_integrated.parquet"),
            "security_master_enriched": str(PHASE1B_DIR / "cse_security_master_enriched.parquet"),
            "sector_mapping": str(PHASE1B_DIR / "cse_sector_mapping.parquet"),
            "market_calendar": str(PHASE1A_DIR / "cse_market_calendar.parquet"),
            "market_indices": str(RAW_DIR / "07Market Indices - Daily.xls"),
            "gics_indices": str(RAW_DIR / "39GICS-Daily.xlsx"),
        },
        "row_count": int(len(tech_df)),
        "unique_security_ids": int(tech_df["SecurityID"].nunique()),
        "unique_company_codes": int(tech_df["CompanyCode"].nunique()),
        "listed_security_ids": int((df_sec_master["HistoricalStatus"] == "Listed").sum()),
        "delisted_security_ids": int((df_sec_master["HistoricalStatus"] == "Delisted").sum()),
        "date_range": [str(tech_df["Date"].min().date()), str(tech_df["Date"].max().date())],
        "indicator_count": int(len(meta_df)),
        "indicators_by_group": {str(k): int(v) for k, v in meta_df["group"].value_counts().items()},
        "validation_summary": val_summary,
        "output_files": [
            "cse_technical_features.parquet",
            "cse_price_volume_features.parquet",
            "cse_market_relative_features.parquet",
            "technical_indicator_metadata.csv",
            "feature_quality_report.csv",
            "phase2_manifest.json",
        ],
    }
    with open(OUT_DIR / "phase2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"\nPhase 2A COMPLETE in {manifest['elapsed_seconds']}s | "
        f"{manifest['indicator_count']} indicators | Inf={val_summary['total_inf_values']} | "
        f"BoundsOK={val_summary['all_bounds_ok']} | CausalMaxDiff={val_summary['causality_prefix_invariance_max_diff']}"
    )
    return manifest


if __name__ == "__main__":
    run_phase2a()
