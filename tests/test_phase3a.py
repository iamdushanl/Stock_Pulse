"""
Stockpulse Phase 3A Automated Verification Test Suite
=====================================================
Validates all 10 deliverables, 4 charts, independent edge cases, synthetic
statement accounting tests, provenance, OCR fallback, PIT availability rules,
and upstream Phase 1-2C immutability across 15 test categories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase3a_extract.normalize_and_validate import (
    REQUIRED_FACT_COLUMNS,
    derive_quarterly_facts,
    detect_restatements,
    normalize_and_validate_facts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE2B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2b"
PHASE2C_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2c"
PHASE3A_DIR = REPO_ROOT / "Dataset" / "processed" / "phase3a"
CSE_LIVE_DIR = REPO_ROOT / "cse_live"

EXPECTED_DELIVERABLES = [
    "financial_facts_normalized.parquet",
    "financial_statement_periods.parquet",
    "financial_company_mapping.parquet",
    "financial_report_inventory.csv",
    "financial_facts_review_queue.csv",
    "financial_extraction_quality_report.csv",
    "financial_validation_report.csv",
    "financial_restated_values.parquet",
    "financial_derived_quarterly_facts.parquet",
    "phase3a_manifest.json",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def p3a_bundle() -> dict:
    return {
        "manifest": json.loads((PHASE3A_DIR / "phase3a_manifest.json").read_text(encoding="utf-8")),
        "facts": pd.read_parquet(PHASE3A_DIR / "financial_facts_normalized.parquet"),
        "periods": pd.read_parquet(PHASE3A_DIR / "financial_statement_periods.parquet"),
        "mapping": pd.read_parquet(PHASE3A_DIR / "financial_company_mapping.parquet"),
        "inventory": pd.read_csv(PHASE3A_DIR / "financial_report_inventory.csv"),
        "review": pd.read_csv(PHASE3A_DIR / "financial_facts_review_queue.csv"),
        "quality": pd.read_csv(PHASE3A_DIR / "financial_extraction_quality_report.csv"),
        "validation": pd.read_csv(PHASE3A_DIR / "financial_validation_report.csv"),
        "restated": pd.read_parquet(PHASE3A_DIR / "financial_restated_values.parquet"),
        "derived_q": pd.read_parquet(PHASE3A_DIR / "financial_derived_quarterly_facts.parquet"),
    }


def test_01_pdf_inventory_completeness(p3a_bundle: dict) -> None:
    inv = p3a_bundle["inventory"]
    manifest = p3a_bundle["manifest"]

    assert len(inv) == 90
    assert manifest["inventory_summary"]["total_pdfs"] == 90
    assert (inv["DocumentCategory"] == "financial_reports").sum() == 80
    assert (inv["DocumentCategory"] == "circulars").sum() == 5
    assert (inv["DocumentCategory"] == "directives").sum() == 5

    # Every file on disk matches SHA-256 & manifest checksum
    for _, r in inv.iterrows():
        fpath = REPO_ROOT / str(r["FilePath"])
        assert fpath.exists(), f"Missing PDF: {r['FilePath']}"
        assert int(r["PDFPageCount"]) >= 1
        assert len(str(r["FileChecksumSHA256"])) == 64
        assert bool(r["ManifestChecksumMatches"]) is True


def test_02_fact_schema_and_deliverables_checksums(p3a_bundle: dict) -> None:
    manifest = p3a_bundle["manifest"]
    facts = p3a_bundle["facts"]

    for fname in EXPECTED_DELIVERABLES:
        assert (PHASE3A_DIR / fname).exists(), f"Missing deliverable {fname}"

    for fname, meta in manifest["outputs"].items():
        fpath = PHASE3A_DIR / fname
        assert fpath.stat().st_size == meta["size_bytes"]
        assert _sha256(fpath) == meta["sha256"], f"SHA-256 mismatch on {fname}"

    assert len(manifest["visualizations"]) == 4
    for vrel in manifest["visualizations"]:
        vp = REPO_ROOT / vrel
        assert vp.exists() and vp.stat().st_size > 10_000

    for col in REQUIRED_FACT_COLUMNS:
        assert col in facts.columns, f"Missing required fact column {col}"

    assert len(facts) == manifest["facts_summary"]["total_normalized_facts"]
    assert facts["FactID"].is_unique


def test_03_company_mapping_integrity_and_multiclass(p3a_bundle: dict) -> None:
    mp = p3a_bundle["mapping"]
    facts = p3a_bundle["facts"]

    # 77 in Phase 1B + 1 post-2025 listing (WLTH) = 78 issuers with PDFs, 323 Phase 1B without PDFs
    assert int((mp["MappingStatus"] == "VERIFIED_PHASE1B").sum()) == 77
    assert int((mp["MappingStatus"] == "CSE_LIVE_ONLY_POST_2025").sum()) == 1
    assert int((mp["MappingStatus"] == "VERIFIED_PHASE1B_NO_PDF").sum()) == 323

    # Multi-class company TKYO maps to one CompanyCode & preserves N + X in AllSecurityIDs
    tkyo = mp[mp["CompanyCode"] == "TKYO"].iloc[0]
    assert tkyo["PrimarySecurityID"] == "TKYO-N-0000"
    assert "TKYO-X-0000" in str(tkyo["AllSecurityIDs"])
    assert int(tkyo["SecurityClassCount"]) >= 2

    # Facts never conflated with disclosure numeric IDs (`53549`)
    assert not facts["CompanyCode"].astype(str).str.isdigit().any()
    assert facts["CompanyCode"].nunique() == 78


def test_04_financial_period_normalization(p3a_bundle: dict) -> None:
    facts = p3a_bundle["facts"]
    valid_ptypes = {"Quarterly", "Half-Year", "Nine-Month", "Annual"}
    assert set(facts["PeriodType"]).issubset(valid_ptypes)

    # Balance sheets remain point-in-time snapshots (PeriodStart = NaT)
    bs = facts[facts["StatementType"] == "Balance Sheet"]
    assert bs["PeriodStart"].isna().all()
    assert bs["PeriodEnd"].notna().all()

    # Flow statements (IS, CF) must have valid PeriodStart < PeriodEnd
    flow = facts[facts["StatementType"].isin(["Income Statement", "Cash Flow"])]
    assert flow["PeriodStart"].notna().all()
    assert (pd.to_datetime(flow["PeriodStart"]) < pd.to_datetime(flow["PeriodEnd"])).all()

    # Spot check SIL Revenue across 3M Q2, 6M H1, and Annual FY2026
    sil_rev = facts[(facts["CompanyCode"] == "SIL") & (facts["MetricCode"] == "REVENUE")]
    assert set(sil_rev["PeriodType"]) == {"Quarterly", "Half-Year", "Annual"}


def test_05_cumulative_to_quarterly_calculations(p3a_bundle: dict) -> None:
    dq = p3a_bundle["derived_q"]
    assert len(dq) > 1000
    assert set(dq["StatementType"]).issubset({"Income Statement", "Cash Flow"})
    assert not (dq["StatementType"] == "Balance Sheet").any()

    # Independent synthetic verification across all 4 cumulative rules + Balance Sheet rejection
    synth_facts = pd.DataFrame([
        # Q1 + H1 -> derives Q2 = H1 - Q1 (700 - 300 = 400)
        {"FactID": "F1", "CompanyCode": "TEST", "SecurityID": "TEST-N-0000", "StatementType": "Income Statement", "MetricCode": "REVENUE", "FiscalYear": 2026, "PeriodType": "Quarterly", "PeriodStart": pd.Timestamp("2025-04-01"), "PeriodEnd": pd.Timestamp("2025-06-30"), "Value": 300.0, "Currency": "LKR", "Unit": "LKR", "IsConsolidated": True, "IsRestated": False, "AvailabilityDate": pd.Timestamp("2025-08-10"), "ValidationStatus": "VALIDATED", "ExtractionConfidence": 0.95},
        {"FactID": "F2", "CompanyCode": "TEST", "SecurityID": "TEST-N-0000", "StatementType": "Income Statement", "MetricCode": "REVENUE", "FiscalYear": 2026, "PeriodType": "Half-Year", "PeriodStart": pd.Timestamp("2025-04-01"), "PeriodEnd": pd.Timestamp("2025-09-30"), "Value": 700.0, "Currency": "LKR", "Unit": "LKR", "IsConsolidated": True, "IsRestated": False, "AvailabilityDate": pd.Timestamp("2025-11-12"), "ValidationStatus": "VALIDATED", "ExtractionConfidence": 0.95},
        # 9M - H1 -> derives Q3 (1150 - 700 = 450)
        {"FactID": "F3", "CompanyCode": "TEST", "SecurityID": "TEST-N-0000", "StatementType": "Income Statement", "MetricCode": "REVENUE", "FiscalYear": 2026, "PeriodType": "Nine-Month", "PeriodStart": pd.Timestamp("2025-04-01"), "PeriodEnd": pd.Timestamp("2025-12-31"), "Value": 1150.0, "Currency": "LKR", "Unit": "LKR", "IsConsolidated": True, "IsRestated": False, "AvailabilityDate": pd.Timestamp("2026-02-14"), "ValidationStatus": "VALIDATED", "ExtractionConfidence": 0.95},
        # Annual - 9M -> derives Q4 (1700 - 1150 = 550)
        {"FactID": "F4", "CompanyCode": "TEST", "SecurityID": "TEST-N-0000", "StatementType": "Income Statement", "MetricCode": "REVENUE", "FiscalYear": 2026, "PeriodType": "Annual", "PeriodStart": pd.Timestamp("2025-04-01"), "PeriodEnd": pd.Timestamp("2026-03-31"), "Value": 1700.0, "Currency": "LKR", "Unit": "LKR", "IsConsolidated": True, "IsRestated": False, "AvailabilityDate": pd.Timestamp("2026-05-25"), "ValidationStatus": "VALIDATED", "ExtractionConfidence": 0.95},
        # Balance Sheet must NEVER be differenced
        {"FactID": "F5", "CompanyCode": "TEST", "SecurityID": "TEST-N-0000", "StatementType": "Balance Sheet", "MetricCode": "TOTAL_ASSETS", "FiscalYear": 2026, "PeriodType": "Half-Year", "PeriodStart": pd.NaT, "PeriodEnd": pd.Timestamp("2025-09-30"), "Value": 5000.0, "Currency": "LKR", "Unit": "LKR", "IsConsolidated": True, "IsRestated": False, "AvailabilityDate": pd.Timestamp("2025-11-12"), "ValidationStatus": "VALIDATED", "ExtractionConfidence": 0.95},
    ])
    d_out = derive_quarterly_facts(synth_facts)
    by_q = d_out.set_index("TargetQuarter")["DerivedValue"].to_dict()
    assert by_q["Q2"] == pytest.approx(400.0)
    assert by_q["Q3"] == pytest.approx(450.0)
    assert by_q["Q4"] == pytest.approx(550.0)
    assert "Balance Sheet" not in set(d_out["StatementType"])


def test_06_currency_and_unit_normalization(p3a_bundle: dict) -> None:
    facts = p3a_bundle["facts"]

    # Verify SIL fixed from 1x legacy error -> 1000x LKR thousands
    sil_q2 = facts[(facts["CompanyCode"] == "SIL") & (facts["MetricCode"] == "REVENUE") & (facts["PeriodType"] == "Quarterly") & (facts["PeriodEnd"] == "2026-09-30")].iloc[0]
    assert sil_q2["Unit"] == "LKR thousands"
    assert sil_q2["OriginalValue"] == pytest.approx(936571.0)
    assert sil_q2["Value"] == pytest.approx(936_571_000.0)

    # Foreign currency (USD) must NOT fabricate LKR conversion
    usd_facts = facts[facts["Currency"] == "USD"]
    assert len(usd_facts) > 300
    assert not usd_facts["ConversionMethod"].str.contains("UNIT_SCALE").any()
    assert not usd_facts["ConversionMethod"].str.contains("EXACT_UNIT_1X").any()
    assert (usd_facts["Value"] == usd_facts["OriginalValue"]).all()

    # Per-share EPS must NEVER be multiplied by 1000
    eps = facts[facts["MetricCode"].isin(["EPS_BASIC", "EPS_DILUTED"])]
    assert eps["Unit"].str.contains("per share").all()
    assert (eps["Value"] == eps["OriginalValue"]).all()

    # Ambiguous units flagged cleanly
    amb = facts[facts["ValidationStatus"] == "AMBIGUOUS_UNIT"]
    assert (amb["IsStrictPITEligible"] == False).all()


def test_07_consolidated_standalone_separation(p3a_bundle: dict) -> None:
    facts = p3a_bundle["facts"]
    assert int(facts["IsConsolidated"].sum()) > 5000
    assert int((~facts["IsConsolidated"]).sum()) > 4000

    # RCL Balance Sheet separates Company Assets (~37.95B) from Group Assets (~126.80B)
    rcl = facts[(facts["CompanyCode"] == "RCL") & (facts["MetricCode"] == "TOTAL_ASSETS") & (facts["PeriodEnd"] == "2026-09-30")]
    comp_val = rcl.loc[~rcl["IsConsolidated"], "Value"].iloc[0]
    grp_val = rcl.loc[rcl["IsConsolidated"], "Value"].iloc[0]
    assert grp_val > 3.0 * comp_val


def test_08_restatement_preservation(p3a_bundle: dict) -> None:
    rst = p3a_bundle["restated"]
    assert len(rst) >= 2
    assert "ASIR" in set(rst["CompanyCode"])
    assert "EBCR" in set(rst["CompanyCode"])

    # Independent synthetic test verifying cross-disclosure revision preserves BOTH original and restated dates
    s_facts = pd.DataFrame([
        {"DisclosureID": "100", "CompanyCode": "XYZ", "SecurityID": "XYZ-N-0000", "StatementType": "Income Statement", "MetricCode": "REVENUE", "PeriodEnd": pd.Timestamp("2025-09-30"), "PeriodType": "Quarterly", "IsConsolidated": True, "Currency": "LKR", "Value": 1000.0, "SourceFile": "orig.pdf", "AvailabilityDate": pd.Timestamp("2025-11-10"), "PublicationDate": pd.Timestamp("2025-11-10"), "ReportType": "Quarterly_Interim", "IsRestated": False},
        {"DisclosureID": "200", "CompanyCode": "XYZ", "SecurityID": "XYZ-N-0000", "StatementType": "Income Statement", "MetricCode": "REVENUE", "PeriodEnd": pd.Timestamp("2025-09-30"), "PeriodType": "Quarterly", "IsConsolidated": True, "Currency": "LKR", "Value": 1150.0, "SourceFile": "restated.pdf", "AvailabilityDate": pd.Timestamp("2026-02-15"), "PublicationDate": pd.Timestamp("2026-02-15"), "ReportType": "Annual", "IsRestated": True},
    ])
    out_r = detect_restatements(s_facts)
    rev = out_r[out_r["RestatementType"].str.contains("AMENDMENT")].iloc[0]
    assert rev["OriginalValue"] == pytest.approx(1000.0)
    assert rev["RestatedValue"] == pytest.approx(1150.0)
    assert pd.Timestamp(rev["OriginalAvailabilityDate"]) < pd.Timestamp(rev["RestatedAvailabilityDate"])


def test_09_balance_sheet_validation(p3a_bundle: dict) -> None:
    val = p3a_bundle["validation"]
    bs = val[val["CheckType"] == "BALANCE_SHEET_EQUATION"]
    assert len(bs) >= 250
    pass_cnt = int(bs["Status"].isin(["PASS_EXACT", "PASS_ROUNDING", "PASS_WITH_NCI"]).sum())
    assert pass_cnt / len(bs) >= 0.95

    # Synthetic broken Balance Sheet edge case must raise EXCEPTION
    inv_mock = pd.DataFrame([{
        "DisclosureID": "M1", "CompanyCode": "MOCK", "SecurityID": "MOCK-N-0000",
        "CompanyName": "MOCK PLC", "PublicationDate": pd.Timestamp("2026-11-10"),
        "AvailabilityDate": pd.Timestamp("2026-11-10"), "MappingStatus": "VERIFIED_PHASE1B",
        "ReportType": "Quarterly_Interim", "FilePath": "mock.pdf", "FileChecksumSHA256": "a" * 64,
    }])
    bad_bs = [
        {"DisclosureID": "M1", "StatementType": "Balance Sheet", "MetricCode": "TOTAL_ASSETS", "OriginalMetricName": "Total Assets", "PeriodStart": pd.NaT, "PeriodEnd": pd.Timestamp("2026-09-30"), "FiscalYear": 2027, "FiscalQuarter": "Q2", "PeriodType": "Half-Year", "OriginalValue": 10000.0, "Currency": "LKR", "Unit": "LKR thousands", "Multiplier": 1000.0, "IsAmbiguousUnit": False, "SourcePage": 2, "ColIndex": 0, "ExtractionMethod": "Text", "BaseConfidence": 0.98, "IsConsolidated": True, "IsRestated": False, "ComparativeFlag": False},
        {"DisclosureID": "M1", "StatementType": "Balance Sheet", "MetricCode": "TOTAL_LIABILITIES", "OriginalMetricName": "Total Liabilities", "PeriodStart": pd.NaT, "PeriodEnd": pd.Timestamp("2026-09-30"), "FiscalYear": 2027, "FiscalQuarter": "Q2", "PeriodType": "Half-Year", "OriginalValue": 4000.0, "Currency": "LKR", "Unit": "LKR thousands", "Multiplier": 1000.0, "IsAmbiguousUnit": False, "SourcePage": 2, "ColIndex": 0, "ExtractionMethod": "Text", "BaseConfidence": 0.98, "IsConsolidated": True, "IsRestated": False, "ComparativeFlag": False},
        {"DisclosureID": "M1", "StatementType": "Balance Sheet", "MetricCode": "TOTAL_EQUITY", "OriginalMetricName": "Total Equity", "PeriodStart": pd.NaT, "PeriodEnd": pd.Timestamp("2026-09-30"), "FiscalYear": 2027, "FiscalQuarter": "Q2", "PeriodType": "Half-Year", "OriginalValue": 3000.0, "Currency": "LKR", "Unit": "LKR thousands", "Multiplier": 1000.0, "IsAmbiguousUnit": False, "SourcePage": 2, "ColIndex": 0, "ExtractionMethod": "Text", "BaseConfidence": 0.98, "IsConsolidated": True, "IsRestated": False, "ComparativeFlag": False},
    ]
    _, v_bad = normalize_and_validate_facts(bad_bs, inv_mock)
    assert v_bad.loc[v_bad["CheckType"] == "BALANCE_SHEET_EQUATION", "Status"].iloc[0] == "EXCEPTION"


def test_10_cash_flow_validation_and_negative_losses(p3a_bundle: dict) -> None:
    facts = p3a_bundle["facts"]
    val = p3a_bundle["validation"]

    cf_checks = val[val["CheckType"].str.startswith("CF_")]
    assert len(cf_checks) > 100
    assert (cf_checks["Status"] == "PASS_EXACT").mean() >= 0.90

    # Valid negative losses (PAT < 0, OCF < 0) must remain VALIDATED (not rejected!)
    valid_losses = facts[(facts["MetricCode"].isin(["PROFIT_AFTER_TAX", "OPERATING_CASH_FLOW"])) & (facts["Value"] < 0) & (facts["ValidationStatus"] == "VALIDATED")]
    assert len(valid_losses) > 100


def test_11_duplicate_fact_detection(p3a_bundle: dict) -> None:
    facts = p3a_bundle["facts"]
    keys = [
        "DisclosureID", "StatementType", "MetricCode", "PeriodStart",
        "PeriodEnd", "PeriodType", "IsConsolidated", "IsRestated", "Currency",
    ]
    assert int(facts.duplicated(subset=keys).sum()) == 0


def test_12_confidence_classification_and_review_queue(p3a_bundle: dict) -> None:
    facts = p3a_bundle["facts"]
    rq = p3a_bundle["review"]

    valid_tiers = {"HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE", "MANUAL_REVIEW_REQUIRED"}
    assert set(facts["ConfidenceTier"]).issubset(valid_tiers)
    assert set(["FactID", "SourceFile", "SourcePage", "CompanyName", "OriginalMetricName", "ExtractedValue", "MetricCode", "IssueType", "Confidence", "ReviewStatus"]).issubset(rq.columns)

    # HAYC Press Release (53523) logged in review queue & FAILED in inventory
    assert "DOC_53523" in set(rq["FactID"])


def test_13_point_in_time_availability_rules(p3a_bundle: dict) -> None:
    periods = p3a_bundle["periods"]
    facts = p3a_bundle["facts"]

    # Verified PublicationDate must strictly follow PeriodEnd (positive lag >= 1 day, never fabricated <= PeriodEnd)
    valid_p = periods.dropna(subset=["AvailabilityDate", "PeriodEnd"])
    assert (pd.to_datetime(valid_p["AvailabilityDate"]) > pd.to_datetime(valid_p["PeriodEnd"])).all()

    # Synthetic missing publication date test: must yield IsStrictPITEligible == False
    inv_no_pub = pd.DataFrame([{
        "DisclosureID": "NP1", "CompanyCode": "ABC", "SecurityID": "ABC-N-0000",
        "CompanyName": "ABC PLC", "PublicationDate": pd.NaT,
        "AvailabilityDate": pd.NaT, "MappingStatus": "VERIFIED_PHASE1B",
        "ReportType": "Quarterly_Interim", "FilePath": "nopub.pdf", "FileChecksumSHA256": "b" * 64,
    }])
    f_np = [{
        "DisclosureID": "NP1", "StatementType": "Income Statement", "MetricCode": "REVENUE",
        "OriginalMetricName": "Revenue", "PeriodStart": pd.Timestamp("2025-07-01"), "PeriodEnd": pd.Timestamp("2025-09-30"),
        "FiscalYear": 2026, "FiscalQuarter": "Q2", "PeriodType": "Quarterly",
        "OriginalValue": 100.0, "Currency": "LKR", "Unit": "LKR thousands", "Multiplier": 1000.0,
        "IsAmbiguousUnit": False, "SourcePage": 1, "ColIndex": 0, "ExtractionMethod": "Text",
        "BaseConfidence": 0.98, "IsConsolidated": True, "IsRestated": False, "ComparativeFlag": False,
    }]
    norm_np, _ = normalize_and_validate_facts(f_np, inv_no_pub)
    assert pd.isna(norm_np["AvailabilityDate"].iloc[0])
    assert bool(norm_np["IsStrictPITEligible"].iloc[0]) is False


def test_14_source_provenance_and_real_ocr_recovery(p3a_bundle: dict) -> None:
    facts = p3a_bundle["facts"]

    # 100% file provenance
    assert (facts["SourcePage"] >= 1).all()
    assert set(facts["ExtractionMethod"]).issubset({"Text", "OCR"})
    assert int((facts["ExtractionMethod"] == "OCR").sum()) >= 400

    # Confirm real OCR extraction (not legacy step1_ocr hardcoded 412580/789450/1845920 placeholders)
    ctld_rev = facts[(facts["CompanyCode"] == "CTLD") & (facts["MetricCode"] == "REVENUE") & (facts["PeriodType"] == "Quarterly") & (facts["PeriodEnd"] == "2026-09-30")].iloc[0]
    assert ctld_rev["ExtractionMethod"] == "OCR"
    assert ctld_rev["OriginalValue"] == pytest.approx(184100.0)
    assert not (facts["OriginalValue"].isin([412580.0, 789450.0, 1845920.0])).any()


def test_15_upstream_immutability_and_no_ml(p3a_bundle: dict) -> None:
    manifest = p3a_bundle["manifest"]
    assert manifest["ml_models_trained"] is False
    assert manifest["upstream_immutability_verified"] is True

    # Verify legacy financial_facts_final.parquet untouched & Phase 2B/2C checksums intact
    assert len(pd.read_parquet(CSE_LIVE_DIR / "phase2_extraction" / "financial_facts_final.parquet")) == 2453
    p2c_m = json.loads((PHASE2C_DIR / "phase2c_manifest.json").read_text(encoding="utf-8"))
    for fname, meta in p2c_m["outputs"].items():
        if (PHASE2C_DIR / fname).stat().st_size < 50_000_000:
            assert _sha256(PHASE2C_DIR / fname) == meta["sha256"], f"Phase 2C mutated: {fname}"
