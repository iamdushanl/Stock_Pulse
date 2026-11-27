"""
Phase 3C Comprehensive Independent Verification Suite (`tests/test_phase3c.py`)
=============================================================================
Covers all 20 verification domains & edge cases required by Section 16:
  1.  Financial input eligibility & `financial_input_quality_flags.parquet` completeness
  2.  USD / FX handling & cross-currency LKR blocking (no fabricated FX rates)
  3.  Multi-year unit-scale validation (excludes corroborated scale mismatches without rescaling)
  4.  Growth calculations (YoY, QoQ, 3Y/5Y CAGR exact math & span validation)
  5.  Negative earnings handling & `earnings_recovery_flag` turnaround indicator
  6.  ROE and ROA strictly using Beginning+Ending Average Balance Sheet values (never 1 snapshot)
  7.  ROIC strictly non-financial, consistent Borrowings+Equity-Cash Invested Capital, ETR/statutory tax
  8.  Financial health ratios & strict suppression of industrial Debt/Liquidity/EV on Banks/Finance
  9.  Cash flow calculations (`CapEx_Outflow = abs(CAPEX)`, `FCF = OCF - CapEx_Outflow`, missing CapEx != 0)
 10.  Dividend calculations (strict `VALIDATED` cash dividends vs excluded `2,787 UNRESOLVED_PERCENT_OF_PAR` & splits/bonus/scrip)
 11.  Historical valuation (strict PIT price, multi-class List 16/21+CA causal shares, negative P/E omitted)
 12.  TTM construction (strictly 4 sequential compatible quarters; <4 incomplete marked unavailable)
 13.  Restatement handling (chronological version evolution without backward leakage)
 14.  PIT feature generation & `build_fundamental_features_as_of` before vs after publication dates
 15.  Historical price & Phase 2C alignment (`Date <= 2025-12-31`, strategy sets `MT`/`LT` + `ST` untouched)
 16.  Missing financial data preserved (`NaN`, zero silent zero-imputation)
 17.  Consolidated vs Standalone separation (no mixing across time or TTM)
 18.  Metric provenance & `fundamental_feature_metadata.csv`
 19.  Feature coverage & redundancy/quality report (`fundamental_feature_coverage.csv`, `quality_report.csv`)
 20.  Upstream data immutability across Phase 1A–3B & `ml_models_trained == False`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase3c_features import (
    build_fundamental_features_as_of,
    build_historical_share_schedule,
    compute_fundamental_snapshot_from_active_facts,
    compute_ttm_from_active_facts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE3C_DIR = REPO_ROOT / "Dataset" / "processed" / "phase3c"
PHASE3B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase3b"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


@pytest.fixture(scope="module")
def p3c_bundle() -> dict:
    assert PHASE3C_DIR.exists(), "Run `run_phase3c.py` to generate Dataset/processed/phase3c/"
    return {
        "flags": pd.read_parquet(PHASE3C_DIR / "financial_input_quality_flags.parquet"),
        "growth": pd.read_parquet(PHASE3C_DIR / "fundamental_growth_features.parquet"),
        "prof": pd.read_parquet(PHASE3C_DIR / "fundamental_profitability_features.parquet"),
        "health": pd.read_parquet(PHASE3C_DIR / "fundamental_financial_health_features.parquet"),
        "cf": pd.read_parquet(PHASE3C_DIR / "fundamental_cashflow_features.parquet"),
        "div": pd.read_parquet(PHASE3C_DIR / "fundamental_dividend_features.parquet"),
        "val": pd.read_parquet(PHASE3C_DIR / "fundamental_valuation_features.parquet"),
        "ttm": pd.read_parquet(PHASE3C_DIR / "fundamental_ttm_features.parquet"),
        "pit": pd.read_parquet(PHASE3C_DIR / "fundamental_pit_features.parquet"),
        "mt": pd.read_parquet(PHASE3C_DIR / "medium_term_fundamental_features.parquet"),
        "lt": pd.read_parquet(PHASE3C_DIR / "long_term_fundamental_features.parquet"),
        "cov": pd.read_csv(PHASE3C_DIR / "fundamental_feature_coverage.csv"),
        "meta": pd.read_csv(PHASE3C_DIR / "fundamental_feature_metadata.csv"),
        "qrep": pd.read_csv(PHASE3C_DIR / "fundamental_feature_quality_report.csv"),
        "manifest": json.loads((PHASE3C_DIR / "phase3c_manifest.json").read_text(encoding="utf-8")),
    }


def _mk_fact(
    fid: str,
    cid: str,
    ccode: str,
    stmt: str,
    mcode: str,
    ps: str,
    pe: str,
    fy: int,
    fq: str,
    ptype: str,
    val: float,
    av: str,
    *,
    cur: str = "LKR",
    unit: str = "LKR thousands",
    cons: str = "Consolidated_Group",
    rv: int = 1,
    is_elig: bool = True,
) -> dict:
    return {
        "FactID": fid,
        "CompanyID": cid,
        "CompanyCode": ccode,
        "SecurityID": f"{ccode}-N-0000",
        "CompanyName": f"{ccode} PLC",
        "StatementType": stmt,
        "MetricCode": mcode,
        "OriginalMetricName": mcode,
        "PeriodStart": pd.Timestamp(ps),
        "PeriodEnd": pd.Timestamp(pe),
        "PeriodStart_dt": pd.Timestamp(ps),
        "PeriodEnd_dt": pd.Timestamp(pe),
        "FiscalYear": fy,
        "FiscalQuarter": fq,
        "PeriodType": ptype,
        "Value": float(val),
        "OriginalValue": float(val),
        "Currency": cur,
        "Unit": unit,
        "ConsolidationStatus": cons,
        "PublicationDate": pd.Timestamp(av),
        "AvailabilityDate": pd.Timestamp(av),
        "AvailabilityDate_dt": pd.Timestamp(av),
        "RestatementVersion": rv,
        "SourceDocumentID": f"DOC_{fid}",
        "SourceFile": "syn.pdf",
        "SourcePage": 1,
        "ExtractionMethod": "Text",
        "ExtractionConfidence": 0.99,
        "ValidationStatus": "VALIDATED",
        "PITEligibility": True,
        "IsDerived": False,
        "DerivationMethod": "",
        "ParentFactIDs": "",
        "IsRestated": rv > 1,
        "ComparativeFlag": False,
        "ReportType": "Annual_Report",
        "SourceSHA256": "a" * 64,
        "Notes": "",
        "InputEligibility": "ELIGIBLE" if is_elig else "EXCLUDED",
        "IsInputEligible": is_elig,
        "IsLKRCompatible": cur == "LKR",
    }


def test_01_financial_input_eligibility_and_deliverables(p3c_bundle: dict) -> None:
    flags = p3c_bundle["flags"]
    manifest = p3c_bundle["manifest"]
    req_flag_cols = {
        "FactID", "CompanyID", "MetricCode", "UnitValidationStatus",
        "CurrencyValidationStatus", "AccountingValidationStatus",
        "PITValidationStatus", "InputEligibility", "ExclusionReason",
    }
    assert req_flag_cols.issubset(flags.columns)
    assert len(flags) == 789805, "Flags table must cover every single Phase 3B fact"
    assert len(manifest["outputs"]) == 14
    for fname, info in manifest["outputs"].items():
        fp = REPO_ROOT / info["path"]
        assert fp.exists(), f"Missing output deliverable: {fp}"
        assert _sha256(fp) == info["sha256"]


def test_02_currency_handling_and_cross_currency_guard(p3c_bundle: dict) -> None:
    flags = p3c_bundle["flags"]
    pit = p3c_bundle["pit"]
    manifest = p3c_bundle["manifest"]
    assert manifest["fx_availability_audit"]["verified_daily_cbsl_fx_available"] is False

    usd_flags = flags[flags["Currency"] == "USD"]
    assert len(usd_flags) == 10147
    # No USD row claimed LKR compatible without verified CBSL FX
    assert not usd_flags["IsLKRCompatible"].any()

    # Any USD-basis snapshot in PIT table must never fabricate LKR market valuation multiples
    usd_pit = pit[pit["CurrencyBasis"] == "USD"]
    if not usd_pit.empty:
        assert usd_pit["pe_ratio"].isna().all()
        assert usd_pit["pb_ratio"].isna().all()
        assert usd_pit["valuation_unavailable_reason"].str.contains("CURRENCY_MISMATCH", na=False).all()


def test_03_unit_scale_and_bs_exception_gating(p3c_bundle: dict) -> None:
    flags = p3c_bundle["flags"]
    # AMBIGUOUS_UNIT facts must be strictly excluded
    amb = flags[flags["Unit"] == "AMBIGUOUS_UNIT"]
    assert (amb["InputEligibility"] == "EXCLUDED").all()

    # Unit-scale mismatches & BS internal 1000x mismatches must be excluded cleanly
    assert int((flags["UnitValidationStatus"] == "EXCLUDED_CONFIRMED_UNIT_SCALE_MISMATCH").sum()) > 0
    excluded_bs = flags[flags["AccountingValidationStatus"].astype(str).str.startswith("EXCLUDED_")]
    assert not excluded_bs.empty
    assert (excluded_bs["InputEligibility"] == "EXCLUDED").all()


def test_04_growth_calculations_exact_math(p3c_bundle: dict) -> None:
    rows = [
        _mk_fact("F1", "CID_G", "G", "Income Statement", "REVENUE", "2021-04-01", "2022-03-31", 2022, "FY", "Annual", 1000.0, "2022-05-15"),
        _mk_fact("F2", "CID_G", "G", "Income Statement", "REVENUE", "2022-04-01", "2023-03-31", 2023, "FY", "Annual", 1210.0, "2023-05-15"),
        _mk_fact("F3", "CID_G", "G", "Income Statement", "REVENUE", "2023-04-01", "2024-03-31", 2024, "FY", "Annual", 1464.1, "2024-05-15"),
        _mk_fact("F4", "CID_G", "G", "Income Statement", "REVENUE", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 1771.561, "2025-05-15"),
        _mk_fact("P1", "CID_G", "G", "Income Statement", "PROFIT_AFTER_TAX", "2023-04-01", "2024-03-31", 2024, "FY", "Annual", 100.0, "2024-05-15"),
        _mk_fact("P2", "CID_G", "G", "Income Statement", "PROFIT_AFTER_TAX", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 130.0, "2025-05-15"),
    ]
    df = pd.DataFrame(rows)
    meta = {"CompanyID": "CID_G", "CompanyCode": "G", "CompanyName": "G PLC", "BusinessType": "NON_FINANCIAL_CORPORATE", "IsFinancialInstitution": False}
    snap = compute_fundamental_snapshot_from_active_facts(df, meta, pd.Timestamp("2025-06-01"))
    assert snap["rev_growth_yoy_pct"] == pytest.approx(21.0, rel=1e-4)
    assert snap["rev_cagr_3y_pct"] == pytest.approx(21.0, rel=1e-4)
    assert snap["net_income_growth_yoy_pct"] == pytest.approx(30.0, rel=1e-4)
    assert snap["earnings_turnaround_status"] == "CONTINUING_PROFIT"


def test_05_negative_earnings_and_turnaround_guard() -> None:
    rows = [
        _mk_fact("P1", "CID_L", "L", "Income Statement", "PROFIT_AFTER_TAX", "2023-04-01", "2024-03-31", 2024, "FY", "Annual", -50.0, "2024-05-15"),
        _mk_fact("P2", "CID_L", "L", "Income Statement", "PROFIT_AFTER_TAX", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 40.0, "2025-05-15"),
        _mk_fact("E1", "CID_L", "L", "Income Statement", "EPS_BASIC", "2023-04-01", "2024-03-31", 2024, "FY", "Annual", -2.5, "2024-05-15", unit="LKR per share"),
        _mk_fact("E2", "CID_L", "L", "Income Statement", "EPS_BASIC", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 2.0, "2025-05-15", unit="LKR per share"),
    ]
    meta = {"CompanyID": "CID_L", "CompanyCode": "L", "CompanyName": "L PLC", "BusinessType": "NON_FINANCIAL_CORPORATE", "IsFinancialInstitution": False}
    snap = compute_fundamental_snapshot_from_active_facts(pd.DataFrame(rows), meta, pd.Timestamp("2025-06-01"))
    # Ordinary % growth from negative denominator MUST be NaN, paired with explicit turnaround indicator
    assert pd.isna(snap["net_income_growth_yoy_pct"])
    assert pd.isna(snap["eps_growth_yoy_pct"])
    assert snap["earnings_recovery_flag"] == 1
    assert snap["earnings_turnaround_status"] == "LOSS_TO_PROFIT"


def test_06_roe_and_roa_strict_average_balance_sheet() -> None:
    meta = {"CompanyID": "CID_R", "CompanyCode": "R", "CompanyName": "R PLC", "BusinessType": "NON_FINANCIAL_CORPORATE", "IsFinancialInstitution": False}
    # Case A: Single snapshot ONLY -> ROE and ROA must stay NaN (no single-snapshot fallacy!)
    single_snap = pd.DataFrame([
        _mk_fact("P", "CID_R", "R", "Income Statement", "PROFIT_AFTER_TAX", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 200.0, "2025-05-15"),
        _mk_fact("E", "CID_R", "R", "Balance Sheet", "TOTAL_EQUITY", "2025-03-31", "2025-03-31", 2025, "FY", "Annual", 1200.0, "2025-05-15"),
        _mk_fact("A", "CID_R", "R", "Balance Sheet", "TOTAL_ASSETS", "2025-03-31", "2025-03-31", 2025, "FY", "Annual", 2200.0, "2025-05-15"),
    ])
    s1 = compute_fundamental_snapshot_from_active_facts(single_snap, meta, pd.Timestamp("2025-06-01"))
    assert pd.isna(s1["roe_pct"])
    assert pd.isna(s1["roa_pct"])

    # Case B: Beginning (800 Eq / 1800 Assets) + Ending (1200 Eq / 2200 Assets) -> Avg Eq = 1000, Avg Assets = 2000
    dual_snap = pd.concat([
        single_snap,
        pd.DataFrame([
            _mk_fact("E0", "CID_R", "R", "Balance Sheet", "TOTAL_EQUITY", "2024-03-31", "2024-03-31", 2024, "FY", "Annual", 800.0, "2024-05-15"),
            _mk_fact("A0", "CID_R", "R", "Balance Sheet", "TOTAL_ASSETS", "2024-03-31", "2024-03-31", 2024, "FY", "Annual", 1800.0, "2024-05-15"),
        ]),
    ], ignore_index=True)
    s2 = compute_fundamental_snapshot_from_active_facts(dual_snap, meta, pd.Timestamp("2025-06-01"))
    assert s2["roe_pct"] == pytest.approx(20.0)   # 200 / 1000 * 100
    assert s2["roa_pct"] == pytest.approx(10.0)   # 200 / 2000 * 100


def test_07_roic_formula_and_tax_assumption() -> None:
    rows = [
        _mk_fact("EB", "CID_C", "C", "Income Statement", "OPERATING_PROFIT", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 300.0, "2025-05-15"),
        _mk_fact("PB", "CID_C", "C", "Income Statement", "PROFIT_BEFORE_TAX", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 250.0, "2025-05-15"),
        _mk_fact("TX", "CID_C", "C", "Income Statement", "INCOME_TAX_EXPENSE", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", -50.0, "2025-05-15"), # 20% ETR
        _mk_fact("E0", "CID_C", "C", "Balance Sheet", "TOTAL_EQUITY", "2024-03-31", "2024-03-31", 2024, "FY", "Annual", 700.0, "2024-05-15"),
        _mk_fact("D0", "CID_C", "C", "Balance Sheet", "LONG_TERM_BORROWINGS", "2024-03-31", "2024-03-31", 2024, "FY", "Annual", 300.0, "2024-05-15"),
        _mk_fact("C0", "CID_C", "C", "Balance Sheet", "CASH_AND_EQUIVALENTS", "2024-03-31", "2024-03-31", 2024, "FY", "Annual", 100.0, "2024-05-15"), # IC0 = 900
        _mk_fact("E1", "CID_C", "C", "Balance Sheet", "TOTAL_EQUITY", "2025-03-31", "2025-03-31", 2025, "FY", "Annual", 850.0, "2025-05-15"),
        _mk_fact("D1", "CID_C", "C", "Balance Sheet", "LONG_TERM_BORROWINGS", "2025-03-31", "2025-03-31", 2025, "FY", "Annual", 400.0, "2025-05-15"),
        _mk_fact("C1", "CID_C", "C", "Balance Sheet", "CASH_AND_EQUIVALENTS", "2025-03-31", "2025-03-31", 2025, "FY", "Annual", 150.0, "2025-05-15"), # IC1 = 1100
    ]
    meta = {"CompanyID": "CID_C", "CompanyCode": "C", "CompanyName": "C PLC", "BusinessType": "NON_FINANCIAL_CORPORATE", "IsFinancialInstitution": False}
    snap = compute_fundamental_snapshot_from_active_facts(pd.DataFrame(rows), meta, pd.Timestamp("2025-06-01"))
    # NOPAT = 300 * (1 - 0.20) = 240; Avg IC = (900 + 1100) / 2 = 1000 -> ROIC = 24.0%
    assert snap["roic_pct"] == pytest.approx(24.0)
    assert snap["roic_tax_assumption"] == "EFFECTIVE_TAX_RATE_ACTUAL"


def test_08_financial_health_and_bank_separation(p3c_bundle: dict) -> None:
    health = p3c_bundle["health"]
    banks = health[health["BusinessType"] == "BANK_FINANCE"]
    assert not banks.empty
    # Industrial ratios MUST be suppressed for Banks/Finance
    assert banks["current_ratio"].isna().all()
    assert banks["debt_to_equity"].isna().all()
    assert banks["net_debt_to_ebitda"].isna().all()
    # Institution-appropriate capitalization MUST be populated
    assert banks["equity_to_assets"].notna().any()

    # Non-financial corporates have liquidity & debt/equity populated
    corp = health[health["BusinessType"] == "NON_FINANCIAL_CORPORATE"]
    assert corp["current_ratio"].notna().sum() > 1000
    assert corp["debt_to_equity"].notna().sum() > 1000


def test_09_cashflow_capex_sign_and_fcf() -> None:
    meta = {"CompanyID": "CID_F", "CompanyCode": "F", "CompanyName": "F PLC", "BusinessType": "NON_FINANCIAL_CORPORATE", "IsFinancialInstitution": False}
    # Cash Flow outflow CAPEX reported negatively as -150 -> CapEx_Outflow = 150 -> FCF = 500 - 150 = 350
    neg_cx = pd.DataFrame([
        _mk_fact("O1", "CID_F", "F", "Cash Flow", "OPERATING_CASH_FLOW", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 500.0, "2025-05-15"),
        _mk_fact("X1", "CID_F", "F", "Cash Flow", "CAPEX", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", -150.0, "2025-05-15"),
    ])
    s_neg = compute_fundamental_snapshot_from_active_facts(neg_cx, meta, pd.Timestamp("2025-06-01"))
    assert s_neg["capex_outflow"] == pytest.approx(150.0)
    assert s_neg["free_cash_flow"] == pytest.approx(350.0)

    # Unsigned positive acquisition CAPEX 150 -> identical FCF = 350
    pos_cx = pd.DataFrame([
        _mk_fact("O1", "CID_F", "F", "Cash Flow", "OPERATING_CASH_FLOW", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 500.0, "2025-05-15"),
        _mk_fact("X1", "CID_F", "F", "Cash Flow", "CAPEX", "2024-04-01", "2025-03-31", 2025, "FY", "Annual", 150.0, "2025-05-15"),
    ])
    s_pos = compute_fundamental_snapshot_from_active_facts(pos_cx, meta, pd.Timestamp("2025-06-01"))
    assert s_pos["free_cash_flow"] == pytest.approx(350.0)


def test_10_dividend_calculations_and_unresolved_par_separation(p3c_bundle: dict) -> None:
    div = p3c_bundle["div"]
    assert (div["unresolved_percent_of_par_div_count"] >= 0).all()
    assert div["unresolved_percent_of_par_div_count"].max() > 0, "Legacy unresolved % of par dividend counters must be tracked"
    assert div["dividend_yield_pct"].notna().sum() > 500
    assert div["consecutive_dividend_years"].max() >= 5


def test_11_historical_valuation_and_negative_pe_guard(p3c_bundle: dict) -> None:
    val = p3c_bundle["val"]
    pit = p3c_bundle["pit"]
    # Negative or non-positive earnings on active TTM P/E basis must NEVER yield misleading P/E
    neg_ttm = pit[
        (pit["pe_earnings_basis"] == "TTM")
        & (pit["ttm_net_income"] <= 0)
        & (pit["ttm_profit_attributable"].isna() | (pit["ttm_profit_attributable"] <= 0))
    ]
    if not neg_ttm.empty:
        assert neg_ttm["pe_ratio"].isna().all()
    neg_flagged = pit[pit["valuation_unavailable_reason"] == "NEGATIVE_EARNINGS_PE_OMITTED"]
    if not neg_flagged.empty:
        assert neg_flagged["pe_ratio"].isna().all()
    assert val["market_cap_lkr"].notna().sum() > 2000
    assert val["pb_ratio"].notna().sum() > 1500
    assert val["pe_ratio"].notna().sum() > 1500


def test_12_ttm_construction_strict_four_quarters() -> None:
    # 3 quarters -> must NOT extrapolate TTM
    q3 = pd.DataFrame([
        _mk_fact("Q1", "CID_T", "T", "Income Statement", "REVENUE", "2024-04-01", "2024-06-30", 2025, "Q1", "Quarterly", 100.0, "2024-08-10"),
        _mk_fact("Q2", "CID_T", "T", "Income Statement", "REVENUE", "2024-07-01", "2024-09-30", 2025, "Q2", "Quarterly", 120.0, "2024-11-10"),
        _mk_fact("Q3", "CID_T", "T", "Income Statement", "REVENUE", "2024-10-01", "2024-12-31", 2025, "Q3", "Quarterly", 150.0, "2025-02-10"),
    ])
    t3 = compute_ttm_from_active_facts(q3)
    assert t3["is_ttm_complete"] is False
    assert pd.isna(t3["ttm_revenue"])

    # Add 4th consecutive quarter -> TTM = 100 + 120 + 150 + 230 = 600
    q4 = pd.concat([
        q3,
        pd.DataFrame([_mk_fact("Q4", "CID_T", "T", "Income Statement", "REVENUE", "2025-01-01", "2025-03-31", 2025, "Q4", "Quarterly", 230.0, "2025-05-20")]),
    ], ignore_index=True)
    t4 = compute_ttm_from_active_facts(q4)
    assert t4["is_ttm_complete"] is True
    assert t4["ttm_revenue"] == pytest.approx(600.0)
    assert pd.Timestamp(t4["ttm_availability_date"]) == pd.Timestamp("2025-05-20")


def test_13_restatement_handling_as_of() -> None:
    ident = pd.DataFrame([{"CompanyID": "CID_RS", "CompanyCode": "RS", "CompanyName": "RS PLC", "PrimarySecurityID": "RS-N-0000"}])
    bt = pd.DataFrame([{"CompanyID": "CID_RS", "CompanyCode": "RS", "BusinessType": "NON_FINANCIAL_CORPORATE", "IsFinancialInstitution": False}])
    facts = pd.DataFrame([
        _mk_fact("R_V1", "CID_RS", "RS", "Income Statement", "REVENUE", "2022-04-01", "2023-03-31", 2023, "FY", "Annual", 1000.0, "2023-05-20", rv=1),
        _mk_fact("R_V2", "CID_RS", "RS", "Income Statement", "REVENUE", "2022-04-01", "2023-03-31", 2023, "FY", "Annual", 1200.0, "2024-05-20", rv=2),
        _mk_fact("R_24", "CID_RS", "RS", "Income Statement", "REVENUE", "2023-04-01", "2024-03-31", 2024, "FY", "Annual", 1320.0, "2024-05-20", rv=1),
    ])
    flags = pd.DataFrame([
        {"FactID": f["FactID"], "InputEligibility": "ELIGIBLE", "IsInputEligible": True, "IsLKRCompatible": True}
        for _, f in facts.iterrows()
    ])
    # Check before restatement (`2024-01-15`) -> cannot see 2024-05-20 Version 2 or FY2024!
    pre = build_fundamental_features_as_of(
        "CID_RS", "2024-01-15", facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt,
        share_schedule=build_historical_share_schedule(REPO_ROOT, pd.DataFrame(columns=["EventType","ValidationStatus","ExDate","EffectiveDate","CompanyCode","SecurityID","DividendPerShare"]), pd.DataFrame(columns=["SecurityID","Date","RawClose","SplitBonusAdjClose","CumCapitalFactor"]))
    )
    assert pd.Timestamp(pre.iloc[0]["LatestPeriodEnd"]) == pd.Timestamp("2023-03-31")
    assert pd.isna(pre.iloc[0]["rev_growth_yoy_pct"])

    # Check after restatement (`2024-06-01`) -> uses Version 2 (1200) base -> (1320-1200)/1200 = +10.0%
    post = build_fundamental_features_as_of(
        "CID_RS", "2024-06-01", facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt,
        share_schedule=build_historical_share_schedule(REPO_ROOT, pd.DataFrame(columns=["EventType","ValidationStatus","ExDate","EffectiveDate","CompanyCode","SecurityID","DividendPerShare"]), pd.DataFrame(columns=["SecurityID","Date","RawClose","SplitBonusAdjClose","CumCapitalFactor"]))
    )
    assert post.iloc[0]["rev_growth_yoy_pct"] == pytest.approx(10.0)


def test_14_pit_feature_generation_before_and_after_real_disclosure(p3c_bundle: dict) -> None:
    pit = p3c_bundle["pit"]
    # Zero look-ahead violations across the entire PIT master table
    assert (pd.to_datetime(pit["LatestAvailabilityDate"]) <= pd.to_datetime(pit["DecisionDate"])).all()

    # Live API check on real CSE issuer (COMB): pre-announcement vs post-announcement isolation
    facts = pd.read_parquet(PHASE3B_DIR / "historical_financial_facts.parquet")
    ident = pd.read_parquet(PHASE3B_DIR / "historical_company_identity.parquet")
    flags = p3c_bundle["flags"]
    ca = pd.read_parquet(REPO_ROOT / "Dataset" / "processed" / "phase1b" / "cse_corporate_actions_validated.parquet")
    sched = build_historical_share_schedule(REPO_ROOT, ca)

    # COMB has multi-year filings; compare 2016-01-01 vs 2024-06-01
    b_pre = build_fundamental_features_as_of("COMB", "2016-01-01", facts_df=facts, flags_df=flags, identity_df=ident, share_schedule=sched)
    b_post = build_fundamental_features_as_of("COMB", "2024-06-01", facts_df=facts, flags_df=flags, identity_df=ident, share_schedule=sched)
    assert pd.Timestamp(b_pre.iloc[0]["LatestAvailabilityDate"]) <= pd.Timestamp("2016-01-01")
    assert pd.Timestamp(b_post.iloc[0]["LatestAvailabilityDate"]) <= pd.Timestamp("2024-06-01")
    assert pd.Timestamp(b_pre.iloc[0]["LatestPeriodEnd"]) < pd.Timestamp("2016-01-01")


def test_15_historical_price_alignment_and_strategy_sets(p3c_bundle: dict) -> None:
    val = p3c_bundle["val"]
    mt = p3c_bundle["mt"]
    lt = p3c_bundle["lt"]
    # Market prices matched must never exceed DecisionDate or 2025-12-31 historical price horizon
    has_px = val[val["market_price_date"].notna()]
    assert (pd.to_datetime(has_px["market_price_date"]) <= pd.to_datetime(has_px["DecisionDate"])).all()
    assert (pd.to_datetime(has_px["market_price_date"]) <= pd.Timestamp("2025-12-31")).all()
    # Post-2025-12-31 + 45d dates must NOT fabricate prices
    post25 = val[pd.to_datetime(val["DecisionDate"]) > pd.Timestamp("2026-02-20")]
    assert post25["market_price_lkr"].isna().all()
    assert len(mt) > 1000 and mt["CompanyID"].nunique() >= 150
    assert len(lt) > 1000 and lt["CompanyID"].nunique() >= 120


def test_16_missing_financial_data_preserved_not_zeroed(p3c_bundle: dict) -> None:
    cf = p3c_bundle["cf"]
    # When CAPEX missing or OCF missing, FCF must remain NaN (no silent 0 fill)
    no_capex = cf[cf["capex_outflow"].isna()]
    if not no_capex.empty:
        assert no_capex["free_cash_flow"].isna().all()


def test_17_consolidated_standalone_separation(p3c_bundle: dict) -> None:
    pit = p3c_bundle["pit"]
    assert set(pit["ConsolidationBasis"].unique()).issubset({"Consolidated_Group", "Standalone_Company"})


def test_18_metric_provenance_and_metadata_table(p3c_bundle: dict) -> None:
    meta = p3c_bundle["meta"]
    pit = p3c_bundle["pit"]
    assert {"FeatureName", "FeatureGroup", "FormulaDefinition", "BusinessTypeScope", "EngineeringNotesAndGuards"}.issubset(meta.columns)
    assert len(meta) >= 45
    assert (pit["InputFactIDsCount"] > 0).all()


def test_19_coverage_and_quality_redundancy_reports(p3c_bundle: dict) -> None:
    cov = p3c_bundle["cov"]
    qrep = p3c_bundle["qrep"]
    assert {"OVERALL", "NON_FINANCIAL_CORPORATE", "BANK_FINANCE", "INSURANCE", "COMPANY_SUMMARY"}.issubset(set(cov["Scope"]))
    assert "METRIC_DISTRIBUTION_AND_OUTLIERS" in set(qrep["CheckCategory"])
    assert "HIGH_SPEARMAN_CORRELATION_PAIR" in set(qrep["CheckCategory"])


def test_20_upstream_immutability_and_no_ml(p3c_bundle: dict) -> None:
    manifest = p3c_bundle["manifest"]
    assert manifest["ml_models_trained"] is False
    assert manifest["upstream_immutability_verified"] is True
    assert manifest["short_term_strategy_independence_verified"] is True
    for rel, expected_sha in manifest["upstream_sha256_verified"].items():
        fp = REPO_ROOT / rel
        assert fp.exists(), f"Missing upstream file {rel}"
        assert _sha256(fp) == expected_sha, f"Upstream mutation detected on {rel}"
