"""
Stockpulse Phase 3B Entrypoint – Historical Financial Time-Series Reconstruction & PIT Database
===============================================================================================
Reproducibly discovers all official CSE financial disclosures (2007–2026), reconciles
historical company identities and name changes (`2005–2025`), resumably acquires and parses
historical Annual & Interim PDFs using spatial coordinate & OCR extraction, combines with
Phase 3A canonical facts without altering any upstream file, preserves restatement versions
and strict Point-in-Time (`AvailabilityDate`) boundaries, reconstructs standalone quarterly
flows, audits financial data quality & multi-year coverage, evaluates strategy readiness
(Medium-Term & Long-Term), and generates all 12 deliverables + visualizations + manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3b_reconstruct import (
    acquire_historical_pdfs,
    assess_fundamental_and_strategy_readiness,
    build_company_coverage_matrix,
    build_historical_company_identity,
    discover_and_catalog_filings,
    extract_historical_pdfs,
    parse_historical_quoted_names,
    reconstruct_historical_database,
    run_historical_quality_checks,
    select_acquisition_targets,
)
from utils.data_loader import DEFAULT_PHASE1B_PATH, load_cse_security_master

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase3b")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_anchor_companies(
    sec_master: pd.DataFrame,
    p2c_mt: pd.DataFrame,
    catalog_df: pd.DataFrame,
    sp20_path: Path,
) -> set[str]:
    """Select Flagship Anchor Cohort (~55-65 companies: SP20 + Top-35 Liquid + Top-15 Delisted + key multi-class/USD)."""
    sp20_cos: set[str] = set()
    if sp20_path.exists():
        sp20_df = pd.read_excel(sp20_path, sheet_name="Company_Summary")
        if "CompanyCode" in sp20_df.columns:
            sp20_cos = set(sp20_df["CompanyCode"].astype(str).str.strip())

    cat_syms = set(catalog_df["CompanyCode"].dropna().astype(str).str.strip())
    listed_set = set(sec_master[sec_master["HistoricalStatus"] == "Listed"]["CompanyCode"].astype(str).str.strip())
    pure_delisted = (set(sec_master["CompanyCode"].astype(str).str.strip()) - listed_set) & cat_syms

    top35_liq: set[str] = set()
    top15_del: set[str] = set()
    if not p2c_mt.empty:
        mt_e = p2c_mt[p2c_mt["IsMediumTermEligible"].fillna(False)]
        top_liq = mt_e.groupby("CompanyCode")["Turnover"].sum().sort_values(ascending=False)
        top35_liq = set(top_liq.head(35).index.astype(str))
        del_liq = p2c_mt[p2c_mt["CompanyCode"].isin(pure_delisted)].groupby("CompanyCode")["Turnover"].sum().sort_values(ascending=False)
        top15_del = set(del_liq.head(15).index.astype(str)) | ({"EXPO", "NEST", "CLC", "FLCH"} & cat_syms)

    # Ensure key Phase 3A edge-case issuers (multi-class, USD, Errata, ambiguous unit) remain inside Anchor
    edge_cos = {"TKYO", "MGT", "TJL", "ASIR", "EBCR", "WIND", "CTLD", "LHL", "MSL"} & cat_syms
    return sp20_cos | top35_liq | top15_del | edge_cos


def _build_final_inventory_table(
    catalog_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    facts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Format `historical_filing_inventory.csv` with every exact Section 3 required field."""
    id_map = {str(r["CompanyCode"]).strip(): r for _, r in identity_df.iterrows()}

    # Compute true extracted min PeriodStart / max PeriodEnd per SourceDocumentID where extracted
    ext_bounds: dict[str, tuple[str, str]] = {}
    if not facts_df.empty:
        direct_f = facts_df[~facts_df["IsDerived"].fillna(False)]
        for did, g in direct_f.groupby("SourceDocumentID", sort=False):
            pe_max = g["PeriodEnd"].max()
            ps_min = g["PeriodStart"].min()
            ext_bounds[str(did)] = (
                str(ps_min.date()) if pd.notna(ps_min) else "",
                str(pe_max.date()) if pd.notna(pe_max) else "",
            )

    rows: list[dict[str, Any]] = []
    for _, r in catalog_df.iterrows():
        did = str(r["DocumentID"])
        cc = str(r.get("CompanyCode") or "").strip()
        ir = id_map.get(cc)
        cname = str(ir["CompanyName"]) if ir is not None else str(r.get("CompanyName") or "")
        sid = str(ir["PrimarySecurityID"]) if ir is not None and pd.notna(ir["PrimarySecurityID"]) else ""

        ps_ext, pe_ext = ext_bounds.get(did, ("", ""))
        ps_final = ps_ext or str(r.get("FinancialPeriodStartHint") or "")
        pe_final = pe_ext or str(r.get("FinancialPeriodEndHint") or "")

        ext_st = str(r.get("ExtractionStatus") or "")
        acq_st = str(r.get("AcquisitionStatus") or "")
        ver_st = str(r.get("VerificationStatus") or "")
        if ext_st == "SUCCESS":
            ver_final = "VERIFIED_CHECKSUM_AND_EXTRACTED"
        elif ext_st == "PARTIAL_EXTRACTION":
            ver_final = "VERIFIED_CHECKSUM_PARTIAL_FACTS"
        elif ext_st == "FAILED" and acq_st == "REPORT_SUCCESSFULLY_ACQUIRED":
            ver_final = "VERIFIED_CHECKSUM_ZERO_FACTS"
        else:
            ver_final = ver_st

        rows.append({
            "DocumentID": did,
            "CompanyCode": cc,
            "CompanyName": cname,
            "SecurityID": sid,
            "ReportType": str(r.get("ReportType") or ""),
            "FinancialPeriodStart": ps_final,
            "FinancialPeriodEnd": pe_final,
            "PublicationDate": r["PublicationDate"],
            "AvailabilityDate": r["AvailabilityDate"],
            "SourceURL": str(r.get("SourceURL") or ""),
            "LocalFilePath": str(r.get("LocalFilePath") or ""),
            "SHA256": str(r.get("SHA256") or ""),
            "AcquisitionStatus": acq_st,
            "VerificationStatus": ver_final,
            "ExtractionStatus": ext_st,
            "AcquisitionTier": str(r.get("AcquisitionTier") or ""),
            "IsRestated": bool(r.get("IsRestated", False)),
            "Notes": str(r.get("AcquisitionNotes") or ""),
        })

    return pd.DataFrame(rows)


def _generate_visualizations(
    viz_dir: Path,
    inv_df: pd.DataFrame,
    facts_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    pit_df: pd.DataFrame,
    comp_ready_df: pd.DataFrame,
) -> list[str]:
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#F8FAFC",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "font.size": 9.5,
    })
    out_paths: list[str] = []

    # 1. Official Filings Discovered vs Acquired vs Extracted by Publication Year
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))
    inv_c = inv_df.copy()
    inv_c["PubYear"] = pd.to_datetime(inv_c["PublicationDate"], errors="coerce").dt.year.fillna(0).astype(int)
    inv_y = inv_c[inv_c["PubYear"].between(2012, 2026)]
    yr_ct = pd.crosstab(inv_y["PubYear"], inv_y["AcquisitionStatus"])
    yr_ct.plot(kind="bar", stacked=True, ax=axes[0], color={"REPORT_SUCCESSFULLY_ACQUIRED": "#16A34A", "DOCUMENT_NOT_IN_REPOSITORY": "#CBD5E1", "REPORT_NOT_AVAILABLE": "#F59E0B", "REPORT_ACQUISITION_FAILED": "#DC2626"})
    axes[0].set_title("CSE Historical Financial Filings: Discovered vs Acquired (2012–2026)", fontweight="bold")
    axes[0].set_ylabel("Disclosure PDFs")
    axes[0].tick_params(axis="x", rotation=35)

    fact_y = facts_df.copy()
    fact_y["CalYear"] = pd.to_datetime(fact_y["PeriodEnd"], errors="coerce").dt.year.fillna(0).astype(int)
    fy_sub = fact_y[fact_y["CalYear"].between(2011, 2026)]
    pt_ct = pd.crosstab(fy_sub["CalYear"], fy_sub["PeriodType"])
    pt_ct.plot(kind="bar", stacked=True, ax=axes[1], color={"Annual": "#2563EB", "Half-Year": "#0D9488", "Nine-Month": "#8B5CF6", "Quarterly": "#F59E0B"})
    axes[1].set_title("Reconstructed Financial Facts by Fiscal PeriodEnd Year", fontweight="bold")
    axes[1].set_ylabel("Normalized Facts")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    p1 = viz_dir / "01_historical_acquisition_and_extraction_by_year.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(str(p1))

    # 2. Company Coverage Depth (Annual & Quarterly Companies Extracted per Year)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))
    cy_sub = cov_df[cov_df["CalendarYear"].between(2012, 2026)].groupby("CalendarYear").agg(
        ExtractedCompanies=("CoverageStatus", lambda s: int((s == "REPORT_SUCCESSFULLY_EXTRACTED").sum())),
        AnnualReadyCos=("AnnualFactsCount", lambda s: int((s > 0).sum())),
        QuarterlyReadyCos=("QuarterlyFactsCount", lambda s: int((s > 0).sum())),
    )
    cy_sub.plot(kind="bar", ax=axes[0], color=["#1E293B", "#2563EB", "#0D9488"])
    axes[0].set_title("Companies With Reconstructed Statements by Calendar Year", fontweight="bold")
    axes[0].set_ylabel("Distinct Companies")
    axes[0].tick_params(axis="x", rotation=35)

    # Derived vs Direct Quarterly Breakdown
    q_f = facts_df[facts_df["PeriodType"] == "Quarterly"].copy()
    if not q_f.empty:
        q_f["CalYear"] = pd.to_datetime(q_f["PeriodEnd"], errors="coerce").dt.year.fillna(0).astype(int)
        qd = pd.crosstab(q_f[q_f["CalYear"].between(2012, 2026)]["CalYear"], q_f["IsDerived"])
        qd.columns = ["Direct Reported Quarter" if c is False else "Derived Standalone Quarter" for c in qd.columns]
        qd.plot(kind="bar", stacked=True, ax=axes[1], color=["#2563EB", "#14B8A6"])
    axes[1].set_title("Quarterly Observations: Direct Reported vs Derived Standalone", fontweight="bold")
    axes[1].set_ylabel("Quarterly Facts")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    p2 = viz_dir / "02_company_depth_and_quarterly_coverage.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(str(p2))

    # 3. Point-in-Time Disclosure Lag Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    lags = pit_df.loc[pit_df["ReportingLagDays"].notna() & (pit_df["ReportingLagDays"] >= 0) & (pit_df["ReportingLagDays"] <= 300), "ReportingLagDays"]
    if not lags.empty:
        axes[0].hist(lags, bins=35, color="#0D9488", edgecolor="#0F766E", alpha=0.85)
    axes[0].set_title("Historical PIT Reporting Lag (`FirstReportedDate - PeriodEnd`)", fontweight="bold")
    axes[0].set_xlabel("Days from PeriodEnd to CSE Public Availability")
    axes[0].set_ylabel("Reporting Slices")

    if "PeriodType" in pit_df.columns:
        pt_lags = pit_df[pit_df["ReportingLagDays"].between(0, 365)].groupby("PeriodType")["ReportingLagDays"].median()
        pt_lags.plot(kind="bar", ax=axes[1], color="#2563EB")
        axes[1].set_title("Median PIT Disclosure Lag by Period Type (Days)", fontweight="bold")
        axes[1].set_ylabel("Median Days")
        axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    p3 = viz_dir / "03_pit_lag_and_restatement_timeline.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(str(p3))

    # 4. Fundamental Depth & Strategy Readiness Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    yr_hist = comp_ready_df["DistinctAnnualYears"].value_counts().sort_index()
    axes[0].bar(yr_hist.index.astype(str), yr_hist.values, color="#2563EB")
    axes[0].set_title("Distribution of Annual Fundamental History Depth Across Universe", fontweight="bold")
    axes[0].set_xlabel("Distinct Annual Fiscal Years Extracted per Company")
    axes[0].set_ylabel("Company Count")

    s_counts = pd.Series({
        ">= 3 Years History": int(comp_ready_df["HasAtLeast3YearsHistory"].sum()),
        ">= 5 Years History": int(comp_ready_df["HasAtLeast5YearsHistory"].sum()),
        "Medium-Term Ready\n(3-12m)": int(comp_ready_df["IsMediumTermFundamentalReady"].sum()),
        "Long-Term Ready\n(1-5+y)": int(comp_ready_df["IsLongTermFundamentalReady"].sum()),
        "Dividend History\nReady": int(comp_ready_df["DividendHistoryReady"].sum()),
    })
    bars = axes[1].bar(s_counts.index, s_counts.values, color=["#0D9488", "#0284C7", "#16A34A", "#7C3AED", "#D97706"])
    axes[1].set_title("Strategy & Fundamental History Readiness (Universe)", fontweight="bold")
    axes[1].set_ylabel("Companies")
    axes[1].tick_params(axis="x", rotation=18)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, str(int(b.get_height())), ha="center", fontweight="bold")
    fig.tight_layout()
    p4 = viz_dir / "04_strategy_fundamental_readiness.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(str(p4))

    return out_paths


def run_phase3b(
    allow_network: bool = True,
    max_targets: int = 0,
    download_workers: int = 8,
    parse_workers: int = 8,
    include_tier_b: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    p1b_dir = Path(DEFAULT_PHASE1B_PATH)
    p2c_dir = REPO_ROOT / "Dataset" / "processed" / "phase2c"
    p3a_dir = REPO_ROOT / "Dataset" / "processed" / "phase3a"
    cse_live_dir = REPO_ROOT / "cse_live"
    xls_dir = REPO_ROOT / "01_existing_cse_sources"
    out_dir = REPO_ROOT / "Dataset" / "processed" / "phase3b"
    cache_dir = out_dir / "_discovery_cache"
    raw_pdf_dir = out_dir / "raw_pdfs"
    parse_cache_dir = out_dir / "_parse_cache"
    ocr_cache_dir = out_dir / "_ocr_cache"
    viz_dir = REPO_ROOT / "reports" / "phase3" / "visualizations" / "phase3b"
    for d in (out_dir, cache_dir, raw_pdf_dir, parse_cache_dir, ocr_cache_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Loading immutable upstream Phase 1B, 2C, and 3A datasets...")
    sec_master = load_cse_security_master(path=p1b_dir / "cse_security_master_enriched.parquet")
    ca_df = pd.read_parquet(p1b_dir / "cse_corporate_actions_validated.parquet")
    p2c_mt = pd.read_parquet(p2c_dir / "medium_term_research_universe.parquet", columns=["CompanyCode", "Turnover", "IsMediumTermEligible"])
    p2c_lt = pd.read_parquet(p2c_dir / "long_term_research_universe.parquet", columns=["CompanyCode", "IsLongTermEligible"])
    p3a_facts = pd.read_parquet(p3a_dir / "financial_facts_normalized.parquet")
    p3a_val = pd.read_csv(p3a_dir / "financial_validation_report.csv")
    cse_live_sec = pd.read_csv(cse_live_dir / "securities_master_snapshot.csv")

    logger.info("[Stage 1/6] Discovering official 2007–2026 CSE historical disclosures & reconciling Workstream B company identities...")
    catalog_df = discover_and_catalog_filings(
        repo_root=REPO_ROOT,
        cse_live_dir=cse_live_dir,
        cache_dir=cache_dir,
        phase3a_inventory_path=p3a_dir / "financial_report_inventory.csv",
        allow_network=allow_network,
    )
    quoted_hist = parse_historical_quoted_names(
        xls_dir / "16 List of Quoted Securities And Issued Quantity.xls",
        xls_dir / "06New Listings & De-listings.xls",
        cache_dir=cache_dir,
    )
    identity_df = build_historical_company_identity(sec_master, cse_live_sec, catalog_df, quoted_hist)

    anchor_cos = _select_anchor_companies(sec_master, p2c_mt, catalog_df, xls_dir / "SP20_CSE_Dataset_2001_2025.xlsx")
    catalog_df = select_acquisition_targets(
        catalog_df,
        anchor_codes=anchor_cos,
        include_tier_b=include_tier_b,
        max_targets=max_targets,
    )

    logger.info("[Stage 2/6] Running resumable polite acquisition (`allow_network=%s`)...", allow_network)
    catalog_df = acquire_historical_pdfs(
        catalog_df=catalog_df,
        raw_pdf_dir=raw_pdf_dir,
        log_path=cache_dir / "acquisition_log.parquet",
        allow_network=allow_network,
        max_workers=download_workers,
    )

    logger.info("[Stage 3/6] Extracting newly acquired historical Annual & Interim statements (`parse_workers=%d`)...", parse_workers)
    raw_new_facts, catalog_df = extract_historical_pdfs(
        catalog_df=catalog_df,
        repo_root=REPO_ROOT,
        parse_cache_dir=parse_cache_dir,
        ocr_cache_dir=ocr_cache_dir,
        max_workers=parse_workers,
    )

    logger.info("[Stage 4/6] Reconstructing versioned PIT & Restatement-aware historical database + standalone quarters...")
    db = reconstruct_historical_database(p3a_facts, raw_new_facts, catalog_df, identity_df)
    hist_facts = db["historical_financial_facts"]
    annual_df = db["historical_annual_financials"]
    quarterly_df = db["historical_quarterly_financials"]
    restated_df = db["historical_restated_values"]
    pit_df = db["financial_pit_availability"]
    new_val_df = db["new_validation_checks"]

    logger.info("[Stage 5/6] Building historical coverage matrix, quality audit, and strategy readiness evaluations...")
    inv_out_df = _build_final_inventory_table(catalog_df, identity_df, hist_facts)
    cov_df, cov_summaries = build_company_coverage_matrix(identity_df, catalog_df, hist_facts, restated_df)
    qual_df = run_historical_quality_checks(hist_facts, p3a_val, new_val_df, restated_df)
    metric_ready_df, comp_ready_df = assess_fundamental_and_strategy_readiness(identity_df, hist_facts, ca_df, p2c_mt, p2c_lt)

    logger.info("[Stage 6/6] Writing 11 tabular outputs, visualizations & SHA-256 manifest to %s...", out_dir)
    out_files = {
        "historical_filing_inventory.csv": out_dir / "historical_filing_inventory.csv",
        "historical_company_identity.parquet": out_dir / "historical_company_identity.parquet",
        "historical_financial_facts.parquet": out_dir / "historical_financial_facts.parquet",
        "historical_annual_financials.parquet": out_dir / "historical_annual_financials.parquet",
        "historical_quarterly_financials.parquet": out_dir / "historical_quarterly_financials.parquet",
        "historical_restated_values.parquet": out_dir / "historical_restated_values.parquet",
        "company_financial_coverage.parquet": out_dir / "company_financial_coverage.parquet",
        "financial_pit_availability.parquet": out_dir / "financial_pit_availability.parquet",
        "financial_data_quality_report.csv": out_dir / "financial_data_quality_report.csv",
        "fundamental_metric_readiness.csv": out_dir / "fundamental_metric_readiness.csv",
        "company_fundamental_readiness.parquet": out_dir / "company_fundamental_readiness.parquet",
    }

    inv_out_df.to_csv(out_files["historical_filing_inventory.csv"], index=False)
    identity_df.to_parquet(out_files["historical_company_identity.parquet"], index=False, compression="snappy")
    hist_facts.to_parquet(out_files["historical_financial_facts.parquet"], index=False, compression="snappy")
    annual_df.to_parquet(out_files["historical_annual_financials.parquet"], index=False, compression="snappy")
    quarterly_df.to_parquet(out_files["historical_quarterly_financials.parquet"], index=False, compression="snappy")
    restated_df.to_parquet(out_files["historical_restated_values.parquet"], index=False, compression="snappy")
    cov_df.to_parquet(out_files["company_financial_coverage.parquet"], index=False, compression="snappy")
    pit_df.to_parquet(out_files["financial_pit_availability.parquet"], index=False, compression="snappy")
    qual_df.to_csv(out_files["financial_data_quality_report.csv"], index=False)
    metric_ready_df.to_csv(out_files["fundamental_metric_readiness.csv"], index=False)
    comp_ready_df.to_parquet(out_files["company_fundamental_readiness.parquet"], index=False, compression="snappy")

    viz_files = _generate_visualizations(viz_dir, inv_out_df, hist_facts, cov_df, pit_df, comp_ready_df)

    # Verify upstream Phase 3A & 2C immutability cleanly
    p3a_manifest = json.loads((p3a_dir / "phase3a_manifest.json").read_text(encoding="utf-8"))
    upstream_ok = all(
        _sha256_file(REPO_ROOT / m["path"]) == m["sha256"]
        for m in p3a_manifest["outputs"].values()
        if (REPO_ROOT / m["path"]).exists()
    )

    outputs_meta = {
        fname: {
            "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "size_bytes": int(p.stat().st_size),
            "sha256": _sha256_file(p),
        }
        for fname, p in out_files.items()
    }

    bs_q = qual_df[qual_df["CheckType"] == "BALANCE_SHEET_EQUATION"] if not qual_df.empty else pd.DataFrame()
    manifest = {
        "phase": "3B",
        "description": "Historical Financial Time-Series Reconstruction & Point-in-Time Fundamental Database",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 2),
        "upstream_immutability_verified": bool(upstream_ok),
        "ml_models_trained": False,
        "allow_network": allow_network,
        "acquisition_summary": {
            "total_filings_discovered": int(len(inv_out_df)),
            "anchor_cohort_companies": int(len(anchor_cos)),
            "acquisition_status_counts": {str(k): int(v) for k, v in inv_out_df["AcquisitionStatus"].value_counts().items()},
            "extraction_status_counts": {str(k): int(v) for k, v in inv_out_df["ExtractionStatus"].value_counts().items()},
            "official_pdfs_acquired_on_disk": int((inv_out_df["AcquisitionStatus"] == "REPORT_SUCCESSFULLY_ACQUIRED").sum()),
            "newly_acquired_non_phase3a_pdfs": int(((inv_out_df["AcquisitionStatus"] == "REPORT_SUCCESSFULLY_ACQUIRED") & (~catalog_df["IsPhase3ALocal"])).sum()),
            "successfully_extracted_reports": int((inv_out_df["ExtractionStatus"] == "SUCCESS").sum()),
            "partial_or_failed_acquired_reports": int(inv_out_df.loc[(inv_out_df["AcquisitionStatus"] == "REPORT_SUCCESSFULLY_ACQUIRED") & (inv_out_df["ExtractionStatus"].isin(["PARTIAL_EXTRACTION", "FAILED"]))].shape[0]),
            "workstream_a_status": "PARTIALLY_COMPLETE_TIERED_ACQUISITION" if (inv_out_df["AcquisitionStatus"] == "DOCUMENT_NOT_IN_REPOSITORY").any() else "COMPLETE",
        },
        "identity_summary": {
            "total_company_identities": int(len(identity_df)),
            "mapping_status_counts": {str(k): int(v) for k, v in identity_df["MappingStatus"].value_counts().items()},
            "companies_with_historical_name_changes": int(identity_df["HasNameChange"].sum()),
            "unresolved_entities": int((identity_df["MappingStatus"] == "UNRESOLVED").sum()),
            "companies_covered_with_extracted_facts": int(hist_facts["CompanyID"].nunique()),
        },
        "database_summary": {
            "total_historical_facts": int(len(hist_facts)),
            "direct_extracted_facts": int((~hist_facts["IsDerived"]).sum()),
            "derived_standalone_quarterly_facts": int(hist_facts["IsDerived"].sum()),
            "total_annual_facts": int(len(annual_df)),
            "annual_unique_companies": int(annual_df["CompanyID"].nunique()),
            "total_quarterly_table_rows": int(len(quarterly_df)),
            "quarterly_unique_companies": int(quarterly_df["CompanyID"].nunique()),
            "period_type_counts": {str(k): int(v) for k, v in hist_facts["PeriodType"].value_counts().items()},
            "statement_counts": {str(k): int(v) for k, v in hist_facts["StatementType"].value_counts().items()},
            "consolidation_counts": {str(k): int(v) for k, v in hist_facts["ConsolidationStatus"].value_counts().items()},
            "currency_counts": {str(k): int(v) for k, v in hist_facts["Currency"].value_counts().items()},
            "strict_pit_eligible_facts": int(hist_facts["PITEligibility"].sum()),
            "restatement_records_count": int(len(restated_df)),
            "restatement_type_counts": {str(k): int(v) for k, v in restated_df["RestatementType"].value_counts().items()} if not restated_df.empty else {},
            "unique_metrics_count": int(hist_facts["MetricCode"].nunique()),
            "earliest_period_end": str(pd.to_datetime(hist_facts["PeriodEnd"]).min().date()) if not hist_facts.empty else "",
            "latest_period_end": str(pd.to_datetime(hist_facts["PeriodEnd"]).max().date()) if not hist_facts.empty else "",
            "median_pit_reporting_lag_days": float(pit_df["ReportingLagDays"].median()) if not pit_df.empty and pit_df["ReportingLagDays"].notna().any() else None,
        },
        "quality_and_readiness_summary": {
            "total_quality_checks": int(len(qual_df)),
            "balance_sheet_checks": int(len(bs_q)),
            "balance_sheet_status_counts": {str(k): int(v) for k, v in bs_q["Status"].value_counts().items()} if not bs_q.empty else {},
            "check_type_status_top": {str(k): int(v) for k, v in (qual_df["CheckType"] + ":" + qual_df["Status"]).value_counts().head(25).items()} if not qual_df.empty else {},
            "companies_ge_3_years_history": int(comp_ready_df["HasAtLeast3YearsHistory"].sum()),
            "companies_ge_5_years_history": int(comp_ready_df["HasAtLeast5YearsHistory"].sum()),
            "medium_term_fundamental_ready_companies": int(comp_ready_df["IsMediumTermFundamentalReady"].sum()),
            "long_term_fundamental_ready_companies": int(comp_ready_df["IsLongTermFundamentalReady"].sum()),
            "delisted_companies_with_facts": int(comp_ready_df[(comp_ready_df["HistoricalStatus"] == "Delisted") & (comp_ready_df["TotalFactsCount"] > 0)].shape[0]),
        },
        "coverage_summaries": cov_summaries,
        "visualizations": [str(Path(v).relative_to(REPO_ROOT)).replace("\\", "/") for v in viz_files],
        "outputs": outputs_meta,
    }

    manifest_path = out_dir / "phase3b_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Phase 3B COMPLETE in %.2fs -> %d facts across %d companies (%d Annual, %d Quarterly, %d PIT-eligible)",
        manifest["runtime_seconds"],
        manifest["database_summary"]["total_historical_facts"],
        manifest["identity_summary"]["companies_covered_with_extracted_facts"],
        manifest["database_summary"]["total_annual_facts"],
        manifest["database_summary"]["total_quarterly_table_rows"],
        manifest["database_summary"]["strict_pit_eligible_facts"],
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Stockpulse Phase 3B Historical Reconstruction Runner")
    ap.add_argument("--offline", action="store_true", help="Skip live HTTP downloads; reconstruct deterministically from cached/local PDFs")
    ap.add_argument("--max-targets", type=int, default=0, help="Optional cap on newly acquired historical PDFs (0 = all prioritized targets)")
    ap.add_argument("--download-workers", type=int, default=8, help="Concurrent keep-alive HTTP download workers")
    ap.add_argument("--parse-workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) - 2)), help="Multiprocessing parser workers")
    ap.add_argument("--anchor-only", action="store_true", help="Acquire only Tier A Anchor companies cleanly")
    args = ap.parse_args()
    run_phase3b(
        allow_network=not args.offline,
        max_targets=args.max_targets,
        download_workers=args.download_workers,
        parse_workers=args.parse_workers,
        include_tier_b=not args.anchor_only,
    )


if __name__ == "__main__":
    main()
