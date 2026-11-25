"""
Phase 1B Runner – Full CSE Data Integration & Corporate Action Engine
=====================================================================
Executes Tasks 2, 3, 4, and 5 reproducibly without altering original raw files
or Phase 1A outputs.

Outputs written to Dataset/processed/phase1b/:
  - cse_daily_prices_integrated.parquet
  - cse_daily_prices_split_adjusted.parquet
  - cse_total_return_history.parquet
  - cse_security_master_enriched.parquet
  - cse_corporate_actions_validated.parquet
  - cse_sector_mapping.parquet
  - corporate_action_adjustment_factors.parquet
  - phase1b_manifest.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from utils.data_loader import load_cse_daily_prices, load_cse_security_master
from phase1b_integration.identity_model import build_security_identity_model
from phase1b_integration.corporate_actions_engine import (
    parse_and_validate_corporate_actions,
    compute_adjustment_factors_and_series,
)
from phase1b_integration.sector_mapping import build_historical_sector_mapping


SOURCE_DIR  = WORKSPACE_ROOT / "01_existing_cse_sources"
PHASE1A_DIR = WORKSPACE_ROOT / "Dataset" / "processed" / "phase1"
OUT_DIR     = WORKSPACE_ROOT / "Dataset" / "processed" / "phase1b"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_phase1b() -> dict:
    t0 = time.time()
    print("=" * 72)
    print("Stockpulse Phase 1B – Full CSE Data Integration & Corporate Action Engine")
    print("=" * 72)

    # 1. Load full Phase 1A validated universe via upgraded data loader
    print("\n[1/5] Loading full CSE validated price dataset & Phase 1A master...")
    prices = load_cse_daily_prices(path=PHASE1A_DIR)
    p1_master = load_cse_security_master(path=PHASE1A_DIR)
    print(f"      Loaded {len(prices):,} rows across {prices['CompanyCode'].nunique()} codes "
          f"and {prices['SecurityID'].nunique()} SecurityIDs.")

    # 2. Task 2 – Security Identity Model
    print("\n[2/5] Building Security & Issuer Identity Model...")
    identity_df, name_aliases, yearly_qty = build_security_identity_model(
        prices, p1_master, SOURCE_DIR
    )
    identity_df.to_parquet(OUT_DIR / "cse_security_master_enriched.parquet", index=False)
    print(f"      SecurityIDs: {len(identity_df):,} | Issuers established: "
          f"{identity_df['IssuerID'].dropna().nunique()} | Manual review: "
          f"{int(identity_df['ManualReviewRequired'].sum())}")

    # Build integrated raw price table with SecurityID, IssuerID, ISIN, HistoricalStatus
    id_lookup = identity_df[[
        "SecurityID", "IssuerID", "ISIN", "HistoricalStatus", "CompanyName"
    ]].copy()
    integrated = prices.merge(id_lookup, on="SecurityID", how="left")
    integrated.to_parquet(OUT_DIR / "cse_daily_prices_integrated.parquet", index=False)
    print(f"      Saved cse_daily_prices_integrated.parquet ({len(integrated):,} rows)")

    # 3. Task 3 – Corporate Action Validation & Adjustment Engine
    print("\n[3/5] Parsing & validating corporate actions (01Dividends .. 05Splits)...")
    ca_validated, max_price_date = parse_and_validate_corporate_actions(
        SOURCE_DIR, integrated, identity_df, name_aliases, yearly_qty
    )
    ca_validated.to_parquet(OUT_DIR / "cse_corporate_actions_validated.parquet", index=False)

    print("      Computing chronological adjustment factors & separated price/TR series...")
    factors_audit, split_adj, tr_hist = compute_adjustment_factors_and_series(
        integrated, ca_validated
    )
    factors_audit.to_parquet(OUT_DIR / "corporate_action_adjustment_factors.parquet", index=False)
    split_adj.to_parquet(OUT_DIR / "cse_daily_prices_split_adjusted.parquet", index=False)
    tr_hist.to_parquet(OUT_DIR / "cse_total_return_history.parquet", index=False)
    print(f"      Validated adjustment events: {len(factors_audit):,} | "
          f"SplitAdj rows: {len(split_adj):,} | TotalReturn rows: {len(tr_hist):,}")

    # 4. Task 4 – Historical Sector & Industry Mapping Audit
    print("\n[4/5] Auditing sector sources & writing historical sector mapping table...")
    sector_df, sector_audit = build_historical_sector_mapping(identity_df, SOURCE_DIR)
    sector_df.to_parquet(OUT_DIR / "cse_sector_mapping.parquet", index=False)
    print(f"      Saved cse_sector_mapping.parquet ({len(sector_df):,} rows; "
          f"manual research queue={int(sector_df['ManualResearchQueue'].sum())})")

    # 5. Build Phase 1B Manifest
    print("\n[5/5] Writing phase1b_manifest.json...")
    ca_by_type = {str(k): int(v) for k, v in ca_validated["EventType"].value_counts().items()}
    ca_status_summary = {
        str(k): int(v) for k, v in ca_validated["ValidationStatus"].value_counts().items()
    }
    validated_events_count = int(len(factors_audit))
    unresolved_events_count = int(
        ca_validated["ValidationStatus"].astype(str).str.startswith("UNRESOLVED").sum()
    )

    manifest = {
        "phase": "1B",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "phase1a_source_dir": str(PHASE1A_DIR),
        "raw_source_dir": str(SOURCE_DIR),
        "output_dir": str(OUT_DIR),
        "price_date_range": [
            str(integrated["Date"].min().date()),
            str(integrated["Date"].max().date()),
        ],
        "unique_security_ids": int(identity_df["SecurityID"].nunique()),
        "unique_company_codes": int(identity_df["CompanyCode"].nunique()),
        "issuers_established_codes": int(identity_df["IssuerID"].dropna().nunique()),
        "securities_manual_review": int(identity_df["ManualReviewRequired"].sum()),
        "listed_company_codes": int(
            identity_df.drop_duplicates("CompanyCode")["HistoricalStatus"].eq("Listed").sum()
        ),
        "delisted_company_codes": int(
            identity_df.drop_duplicates("CompanyCode")["HistoricalStatus"].eq("Delisted").sum()
        ),
        "security_classes_counts": {
            str(k): int(v) for k, v in identity_df["MainType"].value_counts().items()
        },
        "integrated_rows": int(len(integrated)),
        "split_adjusted_rows": int(len(split_adj)),
        "total_return_rows": int(len(tr_hist)),
        "corporate_actions_total_parsed": int(len(ca_validated)),
        "corporate_actions_by_type": ca_by_type,
        "corporate_actions_by_status": ca_status_summary,
        "validated_adjustment_events": validated_events_count,
        "unresolved_corporate_actions": unresolved_events_count,
        "sector_mapping_rows": int(len(sector_df)),
        "sector_audit": sector_audit,
        "output_files": [
            "cse_daily_prices_integrated.parquet",
            "cse_daily_prices_split_adjusted.parquet",
            "cse_total_return_history.parquet",
            "cse_security_master_enriched.parquet",
            "cse_corporate_actions_validated.parquet",
            "cse_sector_mapping.parquet",
            "corporate_action_adjustment_factors.parquet",
            "phase1b_manifest.json",
        ],
    }
    with open(OUT_DIR / "phase1b_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nPhase 1B pipeline COMPLETE in {manifest['elapsed_seconds']}s.")
    return manifest


if __name__ == "__main__":
    run_phase1b()
