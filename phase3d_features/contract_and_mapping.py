"""
Phase 3D Workstreams A & B – Upstream Data Contract Audit & PIT Company-to-Security Mapping
===========================================================================================
1. Audits all schemas, primary keys, date boundaries, entity counts, checksums, and null profiles
   across Phases 1A, 1B, 2, 2B, 2C, 3A, 3B, and 3C (`upstream_data_contract_report.json`).
2. Reconciles company-level identities (`CompanyID`, `IssuerID`, `CompanyCode`) cleanly against
   security-level instruments (`SecurityID`, `ISIN`, share class) without merging unrelated firms
   or duplicating fundamentals onto debt/rights/warrant classes (`company_security_pit_mapping.parquet`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("phase3d.contract_mapping")

# Canonical upstream contracts audited in Workstream A (relative to repo root)
UPSTREAM_CONTRACT_SPECS: list[dict[str, Any]] = [
    # Phase 1A
    {"phase": "1A", "relpath": "Dataset/processed/phase1/cse_security_master.parquet", "pk": ["CompanyCode"], "date_col": None},
    {"phase": "1A", "relpath": "Dataset/processed/phase1/cse_market_calendar.parquet", "pk": ["Date"], "date_col": "Date"},
    {"phase": "1A", "relpath": "Dataset/processed/phase1/phase1_manifest.json", "pk": None, "date_col": None},
    # Phase 1B
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/cse_security_master_enriched.parquet", "pk": ["SecurityID"], "date_col": None},
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/cse_corporate_actions_validated.parquet", "pk": ["EventID"], "date_col": "ExDate"},
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/corporate_action_adjustment_factors.parquet", "pk": ["EventID"], "date_col": "ExDate"},
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/cse_daily_prices_integrated.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/cse_daily_prices_split_adjusted.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/cse_total_return_history.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/cse_sector_mapping.parquet", "pk": ["SecurityID"], "date_col": None},
    {"phase": "1B", "relpath": "Dataset/processed/phase1b/phase1b_manifest.json", "pk": None, "date_col": None},
    # Phase 2A
    {"phase": "2A", "relpath": "Dataset/processed/phase2/cse_technical_features.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2A", "relpath": "Dataset/processed/phase2/cse_price_volume_features.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2A", "relpath": "Dataset/processed/phase2/cse_market_relative_features.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2A", "relpath": "Dataset/processed/phase2/technical_indicator_metadata.csv", "pk": ["indicator_name"], "date_col": None},
    {"phase": "2A", "relpath": "Dataset/processed/phase2/phase2_manifest.json", "pk": None, "date_col": None},
    # Phase 2B
    {"phase": "2B", "relpath": "Dataset/processed/phase2b/feature_coverage_report.csv", "pk": None, "date_col": None},
    {"phase": "2B", "relpath": "Dataset/processed/phase2b/liquidity_analysis.csv", "pk": ["SecurityID"], "date_col": None},
    {"phase": "2B", "relpath": "Dataset/processed/phase2b/market_relative_validation.csv", "pk": None, "date_col": None},
    {"phase": "2B", "relpath": "Dataset/processed/phase2b/security_history_coverage.csv", "pk": ["SecurityID"], "date_col": None},
    {"phase": "2B", "relpath": "Dataset/processed/phase2b/strategy_readiness_assessment.json", "pk": None, "date_col": None},
    {"phase": "2B", "relpath": "Dataset/processed/phase2b/phase2b_manifest.json", "pk": None, "date_col": None},
    # Phase 2C
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/point_in_time_quality_flags.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/short_term_research_universe.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/medium_term_research_universe.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/long_term_research_universe.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/security_lifecycle_validation.parquet", "pk": ["SecurityID"], "date_col": None},
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/beta_reliability.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/liquidity_eligibility.parquet", "pk": ["SecurityID", "Date"], "date_col": "Date"},
    {"phase": "2C", "relpath": "Dataset/processed/phase2c/phase2c_manifest.json", "pk": None, "date_col": None},
    # Phase 3A
    {"phase": "3A", "relpath": "Dataset/processed/phase3a/phase3a_manifest.json", "pk": None, "date_col": None},
    # Phase 3B
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/historical_financial_facts.parquet", "pk": ["FactID"], "date_col": "AvailabilityDate"},
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/historical_company_identity.parquet", "pk": ["CompanyID"], "date_col": None},
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/historical_annual_financials.parquet", "pk": ["FactID"], "date_col": "AvailabilityDate"},
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/historical_quarterly_financials.parquet", "pk": ["FactID"], "date_col": "AvailabilityDate"},
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/historical_restated_values.parquet", "pk": ["RestatementID"], "date_col": "RestatedAvailabilityDate"},
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/financial_pit_availability.parquet", "pk": ["CompanyID", "PeriodEnd", "PeriodType", "ConsolidationStatus"], "date_col": "FirstReportedDate"},
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/company_fundamental_readiness.parquet", "pk": ["CompanyID"], "date_col": None},
    {"phase": "3B", "relpath": "Dataset/processed/phase3b/phase3b_manifest.json", "pk": None, "date_col": None},
    # Phase 3C
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/financial_input_quality_flags.parquet", "pk": ["FactID"], "date_col": "PeriodEnd"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_growth_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_profitability_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_financial_health_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_cashflow_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_dividend_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_valuation_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_ttm_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_pit_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/medium_term_fundamental_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/long_term_fundamental_features.parquet", "pk": ["CompanyID", "DecisionDate"], "date_col": "DecisionDate"},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_feature_coverage.csv", "pk": None, "date_col": None},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_feature_metadata.csv", "pk": ["FeatureName"], "date_col": None},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/fundamental_feature_quality_report.csv", "pk": None, "date_col": None},
    {"phase": "3C", "relpath": "Dataset/processed/phase3c/phase3c_manifest.json", "pk": None, "date_col": None},
]

# Critical manifests & core upstream inputs locked against modification
UPSTREAM_CORE_PATHS: list[str] = [
    "Dataset/processed/phase1/phase1_manifest.json",
    "Dataset/processed/phase1b/phase1b_manifest.json",
    "Dataset/processed/phase1b/cse_security_master_enriched.parquet",
    "Dataset/processed/phase1b/cse_corporate_actions_validated.parquet",
    "Dataset/processed/phase1b/cse_total_return_history.parquet",
    "Dataset/processed/phase2/phase2_manifest.json",
    "Dataset/processed/phase2b/phase2b_manifest.json",
    "Dataset/processed/phase2c/phase2c_manifest.json",
    "Dataset/processed/phase2c/short_term_research_universe.parquet",
    "Dataset/processed/phase2c/medium_term_research_universe.parquet",
    "Dataset/processed/phase2c/long_term_research_universe.parquet",
    "Dataset/processed/phase3a/phase3a_manifest.json",
    "Dataset/processed/phase3b/phase3b_manifest.json",
    "Dataset/processed/phase3b/historical_financial_facts.parquet",
    "Dataset/processed/phase3b/historical_company_identity.parquet",
    "Dataset/processed/phase3c/phase3c_manifest.json",
    "Dataset/processed/phase3c/fundamental_pit_features.parquet",
]

_SHARE_CLASS_LABEL = {
    "N": "Voting_Ordinary_N",
    "X": "Non_Voting_Ordinary_X",
    "R": "Rights_R",
    "D": "Debenture_D",
    "B": "Debenture_Or_Bond_B",
    "W": "Warrant_W",
    "P": "Preference_P",
    "U": "Unit_U",
}


def compute_file_sha256(path: Path) -> str:
    """Stream-compute SHA-256 hex digest cleanly for files of any size."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_upstream_immutability(
    repo_root: Path,
    baseline_checksums: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    """Compare current upstream SHA-256 digests against captured baseline digests."""
    mismatches: dict[str, Any] = {}
    verified: dict[str, str] = {}
    for relpath, expected in baseline_checksums.items():
        p = repo_root / relpath
        if not p.exists():
            mismatches[relpath] = {"expected": expected, "actual": "MISSING"}
            continue
        actual = compute_file_sha256(p)
        verified[relpath] = actual
        if actual != expected:
            mismatches[relpath] = {"expected": expected, "actual": actual}
    return len(mismatches) == 0, {"verified": verified, "mismatches": mismatches}


def _audit_single_contract(repo_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Inspect schema, primary key uniqueness, dates, entities, nulls & checksum cleanly without copying wide tables."""
    rel = spec["relpath"]
    p = repo_root / rel
    if not p.exists():
        raise FileNotFoundError(f"Required upstream Phase file missing: {p}")

    sha = compute_file_sha256(p)
    ext = p.suffix.lower()
    entry: dict[str, Any] = {
        "phase": spec["phase"],
        "source_path": rel,
        "format": ext.lstrip("."),
        "size_bytes": int(p.stat().st_size),
        "sha256": sha,
        "primary_key": spec["pk"],
    }

    if ext == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        entry["top_keys"] = list(obj.keys())[:30]
        entry["row_count"] = None
        return entry

    if ext == ".csv":
        df = pd.read_csv(p, low_memory=False)
        entry["row_count"] = int(len(df))
        entry["column_count"] = int(len(df.columns))
        entry["schema"] = {c: str(df[c].dtype) for c in df.columns}
        pk = spec["pk"]
        if pk and all(k in df.columns for k in pk):
            entry["duplicate_pk_count"] = int(df.duplicated(subset=pk).sum())
        key_cols = [c for c in ["SecurityID", "CompanyCode", "CompanyID", "FeatureName", "indicator_name"] if c in df.columns]
        entry["null_profile_pct"] = {c: round(float(df[c].isna().mean() * 100.0), 3) for c in key_cols}
        return entry

    # Parquet inspection: schema from metadata, column subset read for fast stats
    pf = pq.ParquetFile(p)
    schema = {field.name: str(field.type) for field in pf.schema_arrow}
    all_cols = list(schema.keys())
    entry["row_count"] = int(pf.metadata.num_rows)
    entry["column_count"] = int(len(all_cols))
    entry["schema"] = schema

    pk = spec["pk"] or []
    dcol = spec["date_col"]
    ent_cols = [c for c in ["CompanyID", "CompanyCode", "SecurityID", "IssuerID"] if c in all_cols]
    status_cols = [c for c in ["MainType", "BusinessType", "InputEligibility", "ValidationStatus", "IsShortTermEligible", "IsMediumTermEligible", "IsLongTermEligible"] if c in all_cols]
    read_cols = list(dict.fromkeys([c for c in (pk + ([dcol] if dcol else []) + ent_cols + status_cols[:3]) if c in all_cols]))

    if read_cols:
        sub = pd.read_parquet(p, columns=read_cols)
        if pk and all(k in sub.columns for k in pk):
            entry["duplicate_pk_count"] = int(sub.duplicated(subset=pk).sum())
        if dcol and dcol in sub.columns:
            dt_s = pd.to_datetime(sub[dcol], errors="coerce")
            entry["date_column"] = dcol
            entry["date_min"] = None if dt_s.dropna().empty else str(dt_s.min())
            entry["date_max"] = None if dt_s.dropna().empty else str(dt_s.max())
            entry["date_null_count"] = int(dt_s.isna().sum())
        entry["entity_coverage"] = {f"unique_{c}": int(sub[c].nunique(dropna=True)) for c in ent_cols}
        entry["null_profile_pct"] = {c: round(float(sub[c].isna().mean() * 100.0), 3) for c in read_cols[:12]}
    return entry


def audit_upstream_data_contracts(
    repo_root: Path,
    output_json_path: Path | None = None,
) -> dict[str, Any]:
    """
    Workstream A: Audit schemas, primary keys, entity/date ranges, checksums, and semantic differences
    across all Phase 1A–3C datasets (`upstream_data_contract_report.json`).
    """
    contracts: list[dict[str, Any]] = []
    sha_baseline: dict[str, str] = {}
    for spec in UPSTREAM_CONTRACT_SPECS:
        rec = _audit_single_contract(repo_root, spec)
        contracts.append(rec)
        if spec["relpath"] in UPSTREAM_CORE_PATHS:
            sha_baseline[spec["relpath"]] = rec["sha256"]

    # Ensure every core immutability target is hashed cleanly even if not in contract table
    for rp in UPSTREAM_CORE_PATHS:
        if rp not in sha_baseline:
            sha_baseline[rp] = compute_file_sha256(repo_root / rp)

    schema_mismatches_and_notes = [
        {
            "topic": "Price vs Fundamental Date Horizon Boundary",
            "upstream_sources": ["Phase 2C universes (max Date = 2025-12-31)", "Phase 3C fundamental_pit_features (max DecisionDate = 2026-11-23)"],
            "semantic_difference": "Phase 2C equity quotes terminate at 2025-12-31 whereas CSE interim/annual filings extend to 2026-11-23 (and Phase 3C carried forward 2025-12-31 close into 46 post-2025 rows).",
            "phase3d_rule": "Allow pure accounting fundamentals through 2026-11-23 in company_fundamental_pit_snapshot, but force ALL price-dependent valuations to NaN after 2025-12-31 with reason POST_2025_NO_VERIFIED_MARKET_PRICE_DATA.",
        },
        {
            "topic": "Intraday AvailabilityDate Timestamps vs Midnight Trading Bars",
            "upstream_sources": ["Phase 3B historical_financial_facts (63.0% intraday HH:MM:SS)", "Phase 2C Date (00:00:00)"],
            "semantic_difference": "Daily trading bars are stored at calendar midnight (00:00:00), while CSE announcements carry exact submission times (e.g. 16:12:04 post-close).",
            "phase3d_rule": "Strict timestamp join LatestAvailabilityDate <= DecisionDate (00:00:00) prevents same-day post-open/after-hours look-ahead leakage; intraday announcements activate on the next calendar/trading date.",
        },
        {
            "topic": "Ordinary Equity Share Classes (N vs X) in Phase 2C vs Phase 1B",
            "upstream_sources": ["Phase 1B cse_security_master_enriched (394 N, 25 X)", "Phase 2C research universes (Is*Eligible only True on N)"],
            "semantic_difference": "Phase 2C set IsHistoricallyTradable=False and strategy flags False on MainType=='X' non-voting shares despite valid historical quotes & List 16/21 issued share counts.",
            "phase3d_rule": "Retain Phase 2C technical strategy flags explicitly as Is*TechnicalEligible, map both N and X equity share classes cleanly to parent CompanyID with security-specific quotes/shares/valuations, and prevent issuer double-counting via IsPrimaryShareClass.",
        },
        {
            "topic": "Corporate Action Share Count Column Semantics",
            "upstream_sources": ["Phase 1B cse_corporate_actions_validated.CumulativeSharesListed", "16 List of Quoted Securities (ISSUED QUANTITY)", "21Market Capitalisation (INDEXED QUANTITY)"],
            "semantic_difference": "Phase 1B CumulativeSharesListed stores incremental allotment counts on Rights/Scrip events, whereas List 16 records security-class total Issued Quantity and List 21 records ordinary Indexed Quantity matching official Market Cap.",
            "phase3d_rule": "Start share schedules from List 16 (class-accurate) / List 21 and propagate causal CA split/bonus factors and verified allotments only when max(CorporateActionAvailabilityDate, ExDate) <= DecisionDate.",
        },
        {
            "topic": "Sector Mapping Honesty",
            "upstream_sources": ["Phase 1B cse_sector_mapping (Confidence=0.0, MANUAL_RESEARCH_REQUIRED)", "Phase 2C point_in_time_quality_flags (IsSectorMappingAvailable=False)"],
            "semantic_difference": "No verified historical point-in-time GICS classification exists in archives.",
            "phase3d_rule": "Keep IsSectorMappingAvailable=False everywhere; track verified accounting-based BusinessType (NON_FINANCIAL_CORPORATE, BANK_FINANCE, INSURANCE) cleanly without inventing GICS sector momentum.",
        },
        {
            "topic": "Pre-2008 Percentage-of-Par Dividends",
            "upstream_sources": ["Phase 1B CashDividends (2,787 UNRESOLVED_PERCENT_OF_PAR)", "Phase 2C IsDividendResolutionClean"],
            "semantic_difference": "2,787 dividend declarations lack per-share par value resolution.",
            "phase3d_rule": "Exclude unresolved percent-of-par dividends from strict dividend yield/DPS calculations and carry IsDividendResolutionClean into Long-Term strict total-return gating.",
        },
    ]

    report = {
        "phase": "3D",
        "workstream": "A_UPSTREAM_DATA_CONTRACT_AUDIT",
        "total_datasets_audited": len(contracts),
        "all_primary_keys_unique": all((c.get("duplicate_pk_count") in (None, 0)) for c in contracts),
        "upstream_sha256_baseline": sha_baseline,
        "schema_mismatches_and_semantic_notes": schema_mismatches_and_notes,
        "contracts": contracts,
    }

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return report


def build_company_security_pit_mapping(
    repo_root: Path,
    output_parquet_path: Path | None = None,
) -> pd.DataFrame:
    """
    Workstream B: Build unambiguous PIT-aware mapping linking CompanyID / IssuerID / CompanyCode
    with SecurityID / ISIN / ShareClass across all 719 Phase 1B securities + 19 non-Phase-1B entities.
    """
    sm = pd.read_parquet(repo_root / "Dataset" / "processed" / "phase1b" / "cse_security_master_enriched.parquet")
    lc = pd.read_parquet(
        repo_root / "Dataset" / "processed" / "phase2c" / "security_lifecycle_validation.parquet",
        columns=[
            "SecurityID", "RemediatedStatus", "RemediatedListingDate", "ListingDateSource",
            "RemediatedDelistingDate", "DelistingDateSource", "LifecycleValidationStatus",
            "AnomalyTag", "FirstPriceDate", "LastPriceDate", "CompanySecurityClassCount",
        ],
    )
    id3b = pd.read_parquet(repo_root / "Dataset" / "processed" / "phase3b" / "historical_company_identity.parquet")
    # Also attach BusinessType if Phase 3C output exists
    bt_map: dict[str, tuple[str, bool]] = {}
    p3c_pit_path = repo_root / "Dataset" / "processed" / "phase3c" / "fundamental_pit_features.parquet"
    if p3c_pit_path.exists():
        bt_df = pd.read_parquet(p3c_pit_path, columns=["CompanyID", "BusinessType", "IsFinancialInstitution"]).drop_duplicates("CompanyID")
        for _, br in bt_df.iterrows():
            bt_map[str(br["CompanyID"])] = (str(br["BusinessType"]), bool(br["IsFinancialInstitution"]))

    # Map Phase 3B CompanyID & PrimarySecurityID cleanly by CompanyCode (exact match on Issuer-verified or exact CompanyCode)
    id_by_code = {str(r["CompanyCode"]).strip(): r for _, r in id3b.iterrows()}

    sec_merged = sm.merge(lc, on="SecurityID", how="left")
    rows: list[dict[str, Any]] = []
    seen_phase3b_cids: set[str] = set()

    for _, sr in sec_merged.iterrows():
        sid = str(sr["SecurityID"]).strip()
        cc = str(sr["CompanyCode"]).strip()
        mtype = str(sr["MainType"]).strip().upper()
        stype = str(sr["SubType"]).strip()
        iss_val_status = str(sr.get("IssuerValidationStatus") or "")
        lc_status = str(sr.get("LifecycleValidationStatus") or "")

        id_row = id_by_code.get(cc)
        cid = str(id_row["CompanyID"]) if id_row is not None else f"UNMAPPED_{cc}"
        if id_row is not None:
            seen_phase3b_cids.add(cid)

        p3b_status = str(id_row["MappingStatus"]) if id_row is not None else "NO_PHASE3B_IDENTITY"
        p3b_conf = float(id_row["MappingConfidence"]) if id_row is not None else 0.0
        primary_sid = str(id_row.get("PrimarySecurityID") or "") if id_row is not None else f"{cc}-N-0000"
        cname = str(id_row["CompanyName"]) if (id_row is not None and pd.notna(id_row["CompanyName"])) else (str(sr.get("CompanyName")) if pd.notna(sr.get("CompanyName")) else "")
        hname = str(id_row["HistoricalCompanyName"]) if (id_row is not None and pd.notna(id_row.get("HistoricalCompanyName"))) else cname
        nhist = str(id_row["NameHistoryJSON"]) if (id_row is not None and pd.notna(id_row.get("NameHistoryJSON"))) else "[]"
        disc_cnt = int(id_row["DiscoveredDisclosuresCount"]) if id_row is not None else 0

        is_equity = mtype in ("N", "X")
        is_primary = bool(is_equity and (sid == primary_sid or (mtype == "N" and stype == "0000" and not primary_sid)))
        # Ensure Validity Dates come strictly from remediated lifecycle or observed first price date
        v_from = pd.to_datetime(sr.get("RemediatedListingDate"), errors="coerce")
        if pd.isna(v_from):
            v_from = pd.to_datetime(sr.get("ListingDate"), errors="coerce")
        if pd.isna(v_from):
            v_from = pd.to_datetime(sr.get("FirstPriceDate"), errors="coerce")
        v_to = pd.to_datetime(sr.get("RemediatedDelistingDate"), errors="coerce")
        if pd.isna(v_to):
            v_to = pd.to_datetime(sr.get("DelistingDate"), errors="coerce")

        merger_flag = iss_val_status == "MANUAL_REVIEW_MERGER_NAME_OVERLAP"
        unverified_issuer = iss_val_status == "MANUAL_REVIEW_UNVERIFIED_ISSUER"
        unresolved_id = p3b_status in ("UNRESOLVED", "NO_PHASE3B_IDENTITY")

        exc_reasons: list[str] = []
        if not is_equity:
            exc_reasons.append(f"NON_EQUITY_SHARE_CLASS_{mtype}")
        if unverified_issuer:
            exc_reasons.append("MANUAL_REVIEW_UNVERIFIED_ISSUER")
        if unresolved_id:
            exc_reasons.append(p3b_status)
        if merger_flag:
            exc_reasons.append("MERGER_NAME_OVERLAP_ISOLATED_BY_ISSUER_AND_DATES")
        if lc_status in ("QUARANTINED", "UNRESOLVED"):
            exc_reasons.append(f"LIFECYCLE_{lc_status}")

        # Equity N & X classes with verified CompanyID (not unverified/unresolved; merger entities ARE isolated by distinct IssuerID & ListingDate/DelistingDate!)
        is_strict_elig = bool(
            is_equity
            and (id_row is not None)
            and (not unverified_issuer)
            and (p3b_status not in ("UNRESOLVED", "NO_PHASE3B_IDENTITY"))
        )

        # Effective mapping confidence combines Phase 3B mapping confidence & Phase 1B issuer confidence
        eff_conf = p3b_conf
        if unverified_issuer:
            eff_conf = min(eff_conf, 0.50)
        elif merger_flag:
            eff_conf = min(eff_conf, 0.90)

        btype, is_fin = bt_map.get(cid, ("UNKNOWN_NO_3C_FACTS", False))

        rows.append({
            "MappingRecordID": f"MAP_{sid}",
            "CompanyID": cid,
            "IssuerID": str(sr["IssuerID"]) if pd.notna(sr.get("IssuerID")) else (str(id_row["IssuerID"]) if (id_row is not None and pd.notna(id_row.get("IssuerID"))) else ""),
            "CompanyCode": cc,
            "SecurityID": sid,
            "ISIN": str(sr["ISIN"]) if pd.notna(sr.get("ISIN")) else "",
            "IssuerPrimaryISIN": str(sr.get("IssuerPrimaryISIN")) if pd.notna(sr.get("IssuerPrimaryISIN")) else "",
            "MainType": mtype,
            "SubType": stype,
            "ShareClass": _SHARE_CLASS_LABEL.get(mtype, f"Other_{mtype}"),
            "IsEquityShareClass": is_equity,
            "IsPrimaryShareClass": is_primary,
            "CompanyName": cname,
            "HistoricalCompanyName": hname,
            "SecurityName": str(sr.get("CompanyName")) if pd.notna(sr.get("CompanyName")) else cname,
            "ShortName": str(sr.get("ShortName")) if pd.notna(sr.get("ShortName")) else "",
            "NameHistoryJSON": nhist,
            "BusinessType": btype,
            "IsFinancialInstitution": is_fin,
            "MappingValidFrom": v_from,
            "MappingValidTo": v_to,
            "FirstPriceDate": pd.to_datetime(sr.get("FirstPriceDate"), errors="coerce"),
            "LastPriceDate": pd.to_datetime(sr.get("LastPriceDate"), errors="coerce"),
            "HistoricalStatus": str(sr.get("RemediatedStatus") or sr.get("HistoricalStatus") or ""),
            "MappingConfidence": round(float(eff_conf), 4),
            "IssuerValidationStatus": iss_val_status,
            "Phase3BMappingStatus": p3b_status,
            "LifecycleValidationStatus": lc_status,
            "AnomalyTag": str(sr.get("AnomalyTag") or "NONE"),
            "MergerNameOverlapFlag": merger_flag,
            "DiscoveredDisclosuresCount": disc_cnt,
            "IsPhase1BTradingSecurity": True,
            "IsStrictPITIntegrationEligible": is_strict_elig,
            "MappingExceptionReason": ";".join(exc_reasons) if exc_reasons else "VERIFIED_OK",
        })

    # Add the 19 Phase 3B entities not present in Phase 1B equity history (`CSE_LIVE_ONLY_POST_2025`, `DEBT_OR_QUOTED_ISSUER_VERIFIED`, `UNRESOLVED`)
    for _, r3 in id3b.iterrows():
        cid = str(r3["CompanyID"])
        if cid in seen_phase3b_cids:
            continue
        cc = str(r3["CompanyCode"]).strip()
        p3b_status = str(r3["MappingStatus"])
        psid = str(r3.get("PrimarySecurityID") or f"NONP1B_{cc}")
        m_conf = float(r3["MappingConfidence"])
        btype, is_fin = bt_map.get(cid, ("UNKNOWN_NO_3C_FACTS", False))

        if p3b_status == "UNRESOLVED":
            exc = "UNRESOLVED_COMPANY_IDENTITY"
        elif p3b_status == "CSE_LIVE_ONLY_POST_2025":
            exc = "POST_2025_CSE_LIVE_ONLY_NO_PHASE2C_PRICES"
        else:
            exc = "QUOTED_DEBT_OR_OTHER_NO_PHASE2C_EQUITY_SERIES"

        rows.append({
            "MappingRecordID": f"MAP_{cid}",
            "CompanyID": cid,
            "IssuerID": str(r3.get("IssuerID") or ""),
            "CompanyCode": cc,
            "SecurityID": psid,
            "ISIN": str(r3.get("ISIN") or ""),
            "IssuerPrimaryISIN": str(r3.get("ISIN") or ""),
            "MainType": ("D" if "DEBT" in p3b_status or ".D" in psid or "/BD/" in psid else "OTHER"),
            "SubType": "0000",
            "ShareClass": "Non_Phase1B_Or_Debt",
            "IsEquityShareClass": False,
            "IsPrimaryShareClass": False,
            "CompanyName": str(r3.get("CompanyName") or cc),
            "HistoricalCompanyName": str(r3.get("HistoricalCompanyName") or r3.get("CompanyName") or cc),
            "SecurityName": str(r3.get("CompanyName") or cc),
            "ShortName": cc,
            "NameHistoryJSON": str(r3.get("NameHistoryJSON") or "[]"),
            "BusinessType": btype,
            "IsFinancialInstitution": is_fin,
            "MappingValidFrom": pd.to_datetime(r3.get("MappingValidFrom"), errors="coerce"),
            "MappingValidTo": pd.to_datetime(r3.get("MappingValidTo"), errors="coerce"),
            "FirstPriceDate": pd.NaT,
            "LastPriceDate": pd.NaT,
            "HistoricalStatus": str(r3.get("HistoricalStatus") or ""),
            "MappingConfidence": round(m_conf, 4),
            "IssuerValidationStatus": "NOT_IN_PHASE1B",
            "Phase3BMappingStatus": p3b_status,
            "LifecycleValidationStatus": "NOT_IN_PHASE2C",
            "AnomalyTag": "NONE",
            "MergerNameOverlapFlag": False,
            "DiscoveredDisclosuresCount": int(r3.get("DiscoveredDisclosuresCount") or 0),
            "IsPhase1BTradingSecurity": False,
            "IsStrictPITIntegrationEligible": False,
            "MappingExceptionReason": exc,
        })

    df = pd.DataFrame(rows).sort_values(["CompanyCode", "SecurityID", "CompanyID"]).reset_index(drop=True)
    if output_parquet_path is not None:
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_parquet_path, index=False)
    return df
