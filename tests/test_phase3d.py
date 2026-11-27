"""
Phase 3D Comprehensive Verification Suite (`tests/test_phase3d.py`)
====================================================================
Tests all 21 mandatory Section 15 validation domains:
  1. Upstream schema validation (`upstream_data_contract_report.json`)
  2. Company/security mapping (`company_security_pit_mapping.parquet`)
  3. Historical as-of joins
  4. Publication-date enforcement (`AvailabilityDate <= DecisionDate`)
  5. Restatement version selection (`SelectedRestatementVersion` & pre/post switching)
  6. Fundamental staleness (120/180/270/365d + 548d FFill cutoff + trading session age)
  7. Historical price alignment & `<= 2025-12-31` hard boundary
  8. Historical market capitalization & List 21 official reconciliation
  9. P/E calculations (exact expected value + negative/zero omission)
 10. P/B calculations (exact expected value + negative equity omitted + multi-class consistency)
 11. EV/EBITDA eligibility (`NON_FINANCIAL_CORPORATE` exact math vs `0%` Bank/Insurance)
 12. Dividend yield alignment (`VALIDATED` only vs `2,787` unresolved `% of par` excluded)
 13. Future-information injection
 14. Future-price injection
 15. Future corporate-action injection
 16. Share-class duplication prevention (`COMB-N` & `COMB-X`, `IsPrimaryShareClass`)
 17. Business-type metric restrictions
 18. Missing-value handling (`NaN` preserved, never silent 0)
 19. Strategy dataset schemas (`short_term`, `medium_term`, `long_term`)
 20. Coverage calculations, staleness sensitivity & readiness classification
 21. Upstream SHA-256 immutability across Phases 1A–3C & zero ML training
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from phase3c_features.dividends_valuation_ttm import HistoricalShareSchedule
from phase3d_features import (
    DEFAULT_MAX_CARRY_DAYS,
    MAX_HISTORICAL_PRICE_DATE,
    compute_file_sha256,
    retrieve_integrated_asof,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
P3D_DIR = REPO_ROOT / "Dataset" / "processed" / "phase3d"


@pytest.fixture(scope="module")
def p3d_bundle() -> dict:
    assert P3D_DIR.exists(), "Run `python run_phase3d.py` before running Phase 3D tests."
    return {
        "contracts": json.loads((P3D_DIR / "upstream_data_contract_report.json").read_text(encoding="utf-8")),
        "mapping": pd.read_parquet(P3D_DIR / "company_security_pit_mapping.parquet"),
        "snapshot": pd.read_parquet(P3D_DIR / "company_fundamental_pit_snapshot.parquet"),
        "cov": pd.read_csv(P3D_DIR / "integrated_feature_coverage.csv"),
        "stale": pd.read_csv(P3D_DIR / "integrated_feature_staleness.csv"),
        "val": pd.read_csv(P3D_DIR / "historical_valuation_validation.csv"),
        "bt": pd.read_csv(P3D_DIR / "business_type_feature_validation.csv"),
        "causal": json.loads((P3D_DIR / "pit_causality_test_report.json").read_text(encoding="utf-8")),
        "readiness": json.loads((P3D_DIR / "strategy_readiness_assessment.json").read_text(encoding="utf-8")),
        "manifest": json.loads((P3D_DIR / "phase3d_manifest.json").read_text(encoding="utf-8")),
    }


def _synthetic_fixture():
    ident = pd.DataFrame([{
        "CompanyID": "CID_UNIT3D", "IssuerID": "ISS_UNIT3D", "CompanyCode": "UNIT",
        "CompanyName": "Unit Verification PLC", "PrimarySecurityID": "UNIT-N-0000",
        "MappingStatus": "VERIFIED_PHASE1B", "MappingConfidence": 1.0,
    }])
    bt = pd.DataFrame([{
        "CompanyID": "CID_UNIT3D", "CompanyCode": "UNIT",
        "BusinessType": "NON_FINANCIAL_CORPORATE", "IsFinancialInstitution": False,
    }])

    def _f(fid, mc, ps, pe, ptype, val, av, rv=1, stmt="INCOME_STATEMENT"):
        return {
            "FactID": fid, "CompanyID": "CID_UNIT3D", "CompanyCode": "UNIT",
            "SecurityID": "UNIT-N-0000", "CompanyName": "Unit Verification PLC",
            "StatementType": stmt, "MetricCode": mc, "OriginalMetricName": mc,
            "PeriodStart": pd.Timestamp(ps), "PeriodEnd": pd.Timestamp(pe),
            "PeriodStart_dt": pd.Timestamp(ps), "PeriodEnd_dt": pd.Timestamp(pe),
            "FiscalYear": pd.Timestamp(pe).year, "FiscalQuarter": "FY", "PeriodType": ptype,
            "Value": float(val), "OriginalValue": float(val), "Currency": "LKR", "Unit": "LKR",
            "ConsolidationStatus": "Consolidated_Group",
            "PublicationDate": pd.Timestamp(av), "AvailabilityDate": pd.Timestamp(av),
            "AvailabilityDate_dt": pd.Timestamp(av), "RestatementVersion": rv,
            "SourceDocumentID": f"DOC_{fid}", "IsDerived": False, "ExtractionConfidence": 1.0,
            "InputEligibility": "ELIGIBLE", "IsInputEligible": True, "IsLKRCompatible": True,
        }

    # Note: 10M shares * 50 LKR = 500M MarketCap
    # Debt = 100M, Cash = 50M, NCI = 20M -> EV = 500 + 100 - 50 + 20 = 570M
    # OP = 80M, Depr = 20M -> EBITDA = 100M -> EV/EBITDA = 5.70
    # Equity = 250M, NetIncome = 100M -> P/E = 500/100 = 5.00, P/B = 500/250 = 2.00
    facts = pd.DataFrame([
        _f("U1", "REVENUE", "2023-04-01", "2024-03-31", "Annual", 1_000_000_000.0, "2024-05-31 15:00:00", stmt="Income Statement"),
        _f("U2", "OPERATING_PROFIT", "2023-04-01", "2024-03-31", "Annual", 80_000_000.0, "2024-05-31 15:00:00", stmt="Income Statement"),
        _f("U3", "DEPRECIATION", "2023-04-01", "2024-03-31", "Annual", 20_000_000.0, "2024-05-31 15:00:00", stmt="Income Statement"),
        _f("U4", "PROFIT_AFTER_TAX", "2023-04-01", "2024-03-31", "Annual", 100_000_000.0, "2024-05-31 15:00:00", stmt="Income Statement"),
        _f("U5", "TOTAL_EQUITY", "2024-03-31", "2024-03-31", "Annual", 270_000_000.0, "2024-05-31 15:00:00", stmt="Balance Sheet"),
        _f("U6", "NON_CONTROLLING_INTEREST", "2024-03-31", "2024-03-31", "Annual", 20_000_000.0, "2024-05-31 15:00:00", stmt="Balance Sheet"),
        _f("U7", "EQUITY_ATTRIBUTABLE_PARENT", "2024-03-31", "2024-03-31", "Annual", 250_000_000.0, "2024-05-31 15:00:00", stmt="Balance Sheet"),
        _f("U8", "LONG_TERM_BORROWINGS", "2024-03-31", "2024-03-31", "Annual", 100_000_000.0, "2024-05-31 15:00:00", stmt="Balance Sheet"),
        _f("U9", "CASH_AND_EQUIVALENTS", "2024-03-31", "2024-03-31", "Annual", 50_000_000.0, "2024-05-31 15:00:00", stmt="Balance Sheet"),
        _f("U10", "TOTAL_ASSETS", "2024-03-31", "2024-03-31", "Annual", 450_000_000.0, "2024-05-31 15:00:00", stmt="Balance Sheet"),
    ])
    flags = pd.DataFrame([{"FactID": f, "InputEligibility": "ELIGIBLE", "IsInputEligible": True, "IsLKRCompatible": True} for f in facts["FactID"]])
    prices = pd.DataFrame([
        {"Date": pd.Timestamp("2024-06-14"), "SecurityID": "UNIT-N-0000", "RawClose": 50.0, "Close": 50.0, "CumSplitBonusFactor": 1.0},
        {"Date": pd.Timestamp("2024-08-19"), "SecurityID": "UNIT-N-0000", "RawClose": 50.0, "Close": 50.0, "CumSplitBonusFactor": 1.0},
    ])
    ca = pd.DataFrame([
        {"EventID": "D_VAL", "SecurityID": "UNIT-N-0000", "CompanyCode": "UNIT", "EventType": "CashDividend", "AnnouncementDate": pd.Timestamp("2024-05-15"), "ExDate": pd.Timestamp("2024-06-05"), "EffectiveDate": pd.Timestamp("2024-06-05"), "DividendPerShare": 2.5, "ValidationStatus": "VALIDATED", "Ratio": np.nan, "CumulativeSharesListed": np.nan},
        {"EventID": "D_UNRES", "SecurityID": "UNIT-N-0000", "CompanyCode": "UNIT", "EventType": "CashDividend", "AnnouncementDate": pd.Timestamp("2024-05-15"), "ExDate": pd.Timestamp("2024-06-05"), "EffectiveDate": pd.Timestamp("2024-06-05"), "DividendPerShare": 50.0, "ValidationStatus": "UNRESOLVED_PERCENT_OF_PAR", "Ratio": np.nan, "CumulativeSharesListed": np.nan},
    ])
    sched = HistoricalShareSchedule(
        yearly_sec_shares={"UNIT-N-0000": [(pd.Timestamp("2023-12-31"), 10_000_000.0, "List16_2023")]},
        yearly_code_shares={"UNIT": [(pd.Timestamp("2023-12-31"), 10_000_000.0, "List16_2023")]},
        sec_classes_by_code={"UNIT": ["UNIT-N-0000"]},
        ca_events_by_sec={}, cash_divs_by_code={}, unresolved_par_by_code={}, non_cash_ca_by_code={}, price_arrays_by_sec={},
    )
    return ident, bt, facts, flags, prices, ca, sched


def test_01_upstream_schema_validation(p3d_bundle):
    rep = p3d_bundle["contracts"]
    assert rep["total_datasets_audited"] >= 45
    assert rep["all_primary_keys_unique"] is True
    assert len(rep["schema_mismatches_and_semantic_notes"]) >= 6


def test_02_company_security_mapping(p3d_bundle):
    m = p3d_bundle["mapping"]
    assert len(m) == 738  # 719 Phase 1B securities + 19 non-Phase1B entities
    assert int(m["IsPhase1BTradingSecurity"].sum()) == 719
    assert m["CompanyID"].nunique() == 419
    # Unresolved FVH & UDA never strictly integrated
    unres = m[m["Phase3BMappingStatus"] == "UNRESOLVED"]
    assert len(unres) == 2 and not bool(unres["IsStrictPITIntegrationEligible"].any())
    # Merger overlap entities (`BFN`, `ORIN`, `CLC`, `LOFC`) preserved strictly separate by IssuerID/CompanyID
    m_ov = m[(m["MergerNameOverlapFlag"]) & (m["IsEquityShareClass"])]
    assert m_ov["CompanyID"].nunique() == 4
    assert m_ov["IssuerID"].nunique() == 4


def test_03_historical_asof_joins(p3d_bundle):
    ident, bt, facts, flags, prices, ca, sched = _synthetic_fixture()
    res = retrieve_integrated_asof(
        "UNIT-N-0000", "2024-06-20",
        facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt,
        ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT,
    )
    assert res["IsFundamentalPITValid"] is True
    assert res["CompanyCode"] == "UNIT"
    assert res["LatestPeriodEnd"] == pd.Timestamp("2024-03-31")


def test_04_publication_date_enforcement(p3d_bundle):
    snap = p3d_bundle["snapshot"]
    av = pd.to_datetime(snap["FundamentalAvailabilityDate"], errors="coerce")
    dec = pd.to_datetime(snap["DecisionDate"])
    assert int((av > dec).sum()) == 0
    # Intraday time guard test: fact released at 2024-05-31 15:00:00 MUST NOT appear on 2024-05-31 00:00:00 trading bar!
    ident, bt, facts, flags, prices, ca, sched = _synthetic_fixture()
    r_same_day_midnight = retrieve_integrated_asof(
        "UNIT-N-0000", "2024-05-31 00:00:00",
        facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt,
        ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT,
    )
    assert r_same_day_midnight["IsFundamentalPITValid"] is False
    r_next_session = retrieve_integrated_asof(
        "UNIT-N-0000", "2024-06-01 00:00:00",
        facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt,
        ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT,
    )
    assert r_next_session["IsFundamentalPITValid"] is True


def test_05_restatement_version_selection(p3d_bundle):
    snap = p3d_bundle["snapshot"]
    assert int(snap["HasRestatementAtDecision"].sum()) > 0
    ident, bt, facts, flags, prices, ca, sched = _synthetic_fixture()
    f2 = facts.copy()
    rest_row = f2[f2["MetricCode"] == "PROFIT_AFTER_TAX"].iloc[0].to_dict()
    rest_row.update({
        "FactID": "U4_R2", "Value": 200_000_000.0, "OriginalValue": 200_000_000.0,
        "AvailabilityDate": pd.Timestamp("2024-08-15"), "AvailabilityDate_dt": pd.Timestamp("2024-08-15"),
        "PublicationDate": pd.Timestamp("2024-08-15"), "RestatementVersion": 2,
    })
    f2 = pd.concat([f2, pd.DataFrame([rest_row])], ignore_index=True)
    fl2 = pd.concat([flags, pd.DataFrame([{"FactID": "U4_R2", "InputEligibility": "ELIGIBLE", "IsInputEligible": True, "IsLKRCompatible": True}])], ignore_index=True)
    r_before = retrieve_integrated_asof("UNIT-N-0000", "2024-07-01", facts_df=f2, flags_df=fl2, identity_df=ident, business_type_df=bt, ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT)
    r_after = retrieve_integrated_asof("UNIT-N-0000", "2024-08-20", facts_df=f2, flags_df=fl2, identity_df=ident, business_type_df=bt, ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT)
    assert r_before["SelectedRestatementVersion"] == 1 and np.isclose(r_before["pe_ratio"], 5.0)
    assert r_after["SelectedRestatementVersion"] == 2 and np.isclose(r_after["pe_ratio"], 2.5)


def test_06_fundamental_staleness(p3d_bundle):
    snap = p3d_bundle["snapshot"]
    for t in (120, 180, 270, 365):
        assert f"IsFundamentalStale_{t}d" in snap.columns
    assert "FundamentalAgeTradingSessions" in snap.columns
    exp_rows = snap[snap["IsFundamentalExpired"]]
    assert len(exp_rows) > 0
    assert not bool(exp_rows["IsFundamentalPITValid"].any())
    # Verify hard 548d carry-forward cutoff nullifies stale ratio columns cleanly
    assert int(exp_rows["net_profit_margin_pct"].notna().sum()) == 0


def test_07_historical_price_alignment_and_post2025_boundary(p3d_bundle):
    snap = p3d_bundle["snapshot"]
    dec = pd.to_datetime(snap["DecisionDate"])
    post25 = snap[dec > MAX_HISTORICAL_PRICE_DATE]
    assert len(post25) > 1000
    # Zero price-dependent valuations allowed in 2026!
    assert int(post25["market_price_lkr"].notna().sum()) == 0
    assert int(post25["pe_ratio"].notna().sum()) == 0
    assert int(post25["pb_ratio"].notna().sum()) == 0
    assert int(post25["market_cap_lkr"].notna().sum()) == 0
    assert not bool(post25["IsValuationPriceBoundaryValid"].any())
    pre26 = snap[(dec <= MAX_HISTORICAL_PRICE_DATE) & snap["market_price_date"].notna()]
    assert int((pd.to_datetime(pre26["market_price_date"]) > pd.to_datetime(pre26["DecisionDate"])).sum()) == 0


def test_08_historical_market_cap_and_list21_validation(p3d_bundle):
    val = p3d_bundle["val"]
    mc_row = val[val["ValidationCheck"] == "MARKET_CAP_VS_CSE_LIST21_OFFICIAL_2005_2025"].iloc[0]
    assert mc_row["TotalEvaluated"] > 3000
    assert mc_row["Within5PctRate"] >= 90.0
    assert mc_row["MedianMetricOrAbsDiff"] < 0.1


def test_09_pe_calculations_and_negative_earnings_guard(p3d_bundle):
    ident, bt, facts, flags, prices, ca, sched = _synthetic_fixture()
    r = retrieve_integrated_asof("UNIT-N-0000", "2024-06-20", facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt, ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT)
    assert np.isclose(r["pe_ratio"], 5.0, rtol=1e-6)
    assert np.isclose(r["earnings_yield_pct"], 20.0, rtol=1e-6)
    # Negative earnings P/E must be NaN
    snap = p3d_bundle["snapshot"]
    assert not bool((snap["pe_ratio"] <= 0).fillna(False).any())


def test_10_pb_calculations_and_negative_equity_guard(p3d_bundle):
    ident, bt, facts, flags, prices, ca, sched = _synthetic_fixture()
    r = retrieve_integrated_asof("UNIT-N-0000", "2024-06-20", facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt, ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT)
    assert np.isclose(r["pb_ratio"], 2.0, rtol=1e-6)
    assert np.isclose(r["bvps_lkr"], 25.0, rtol=1e-6)
    snap = p3d_bundle["snapshot"]
    assert not bool((snap["pb_ratio"] <= 0).fillna(False).any())


def test_11_ev_to_ebitda_eligibility(p3d_bundle):
    ident, bt, facts, flags, prices, ca, sched = _synthetic_fixture()
    r = retrieve_integrated_asof("UNIT-N-0000", "2024-06-20", facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt, ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT)
    assert np.isclose(r["enterprise_value_lkr"], 570_000_000.0, rtol=1e-6)
    assert np.isclose(r["ev_to_ebitda"], 5.70, rtol=1e-6)
    val = p3d_bundle["val"]
    ev_row = val[val["ValidationCheck"] == "EV_TO_EBITDA_NON_FIN_PURITY"].iloc[0]
    assert int(ev_row["Within5PctCount"]) == 0  # Zero FinInst EV/EBITDA leakage


def test_12_dividend_yield_alignment(p3d_bundle):
    ident, bt, facts, flags, prices, ca, sched = _synthetic_fixture()
    r = retrieve_integrated_asof("UNIT-N-0000", "2024-06-20", facts_df=facts, flags_df=flags, identity_df=ident, business_type_df=bt, ca_df=ca, prices_df=prices, share_schedule=sched, repo_root=REPO_ROOT)
    # Must reflect ONLY 2.5 VALIDATED DPS (= 5.0% yield at 50 LKR), NOT UNRESOLVED_PERCENT_OF_PAR 50.0!
    assert np.isclose(r["ttm_cash_dps_lkr"], 2.5, rtol=1e-6)
    assert np.isclose(r["dividend_yield_pct"], 5.0, rtol=1e-6)


def test_13_future_information_injection(p3d_bundle):
    assert p3d_bundle["causal"]["checks"]["future_information_injection_test"]["passed"] is True


def test_14_future_price_injection(p3d_bundle):
    assert p3d_bundle["causal"]["checks"]["future_price_injection_and_post2025_boundary_test"]["passed"] is True


def test_15_future_corporate_action_injection(p3d_bundle):
    assert p3d_bundle["causal"]["checks"]["future_corporate_action_injection_test"]["passed"] is True


def test_16_share_class_duplication_prevention(p3d_bundle):
    m = p3d_bundle["mapping"]
    # Dual-class ordinary issuers (COMB, HNB, TKYO, etc.) must have exactly ONE primary class for issuer dedup
    eq = m[m["IsEquityShareClass"] & m["IsStrictPITIntegrationEligible"]]
    prim_per_cid = eq.groupby("CompanyID")["IsPrimaryShareClass"].sum()
    assert int((prim_per_cid > 1).sum()) == 0
    assert p3d_bundle["causal"]["checks"]["multiclass_identity_non_duplication_test"]["passed"] is True


def test_17_business_type_metric_restrictions(p3d_bundle):
    bt = p3d_bundle["bt"]
    fin_rows = bt[bt["BusinessType"].isin(["BANK_FINANCE", "INSURANCE"])]
    assert len(fin_rows) > 0
    for col in ["CurrentRatio_NonNull", "DebtToEquity_NonNull", "ROIC_NonNull", "EVToEBITDA_NonNull"]:
        assert int(fin_rows[col].sum()) == 0


def test_18_missing_value_handling(p3d_bundle):
    assert p3d_bundle["causal"]["checks"]["missing_data_null_preservation_test"]["passed"] is True
    # Check real Medium-Term integrated universe keeps missing fundamentals strictly NaN when pre-2012 or absent
    mt_sub = pd.read_parquet(
        P3D_DIR / "medium_term_integrated_research_universe.parquet",
        columns=["Date", "IsFundamentalPITValid", "roe_pct", "sec_pe_ratio"],
    )
    no_pit = mt_sub[~mt_sub["IsFundamentalPITValid"]]
    assert int(no_pit["roe_pct"].notna().sum()) == 0
    assert int(no_pit["sec_pe_ratio"].notna().sum()) == 0


def test_19_strategy_dataset_schemas(p3d_bundle):
    st_pq = pq.ParquetFile(P3D_DIR / "short_term_integrated_research_universe.parquet")
    mt_pq = pq.ParquetFile(P3D_DIR / "medium_term_integrated_research_universe.parquet")
    lt_pq = pq.ParquetFile(P3D_DIR / "long_term_integrated_research_universe.parquet")
    for pf in (st_pq, mt_pq, lt_pq):
        assert pf.metadata.num_rows == 1_414_977
        cols = set(pf.schema_arrow.names)
        assert {"CompanyID", "BusinessType", "IsEquityShareClass", "IsPrimaryShareClass", "IsFundamentalPITValid", "IsFundamentalStale", "IsSectorMappingAvailable"}.issubset(cols)
    assert "IsStrictTotalReturnClean" in set(lt_pq.schema_arrow.names)

    # Verify Short-Term eligibility is IDENTICAL to Phase 2C (fundamentals NEVER gate out short-term candidates)
    st_s = pd.read_parquet(P3D_DIR / "short_term_integrated_research_universe.parquet", columns=["IsShortTermTechnicalEligible", "IsShortTermIntegratedEligible", "IsSectorMappingAvailable"])
    assert bool((st_s["IsShortTermTechnicalEligible"] == st_s["IsShortTermIntegratedEligible"]).all())
    assert not bool(st_s["IsSectorMappingAvailable"].any())


def test_20_coverage_staleness_and_readiness_assessment(p3d_bundle):
    cov = p3d_bundle["cov"]
    stale = p3d_bundle["stale"]
    rd = p3d_bundle["readiness"]
    assert set(cov["RecordType"]).issuperset({"STRATEGY_SUMMARY", "STRATEGY_YEARLY", "COMPANY_SNAPSHOT_SUMMARY"})
    assert set(stale["ThresholdDays"]).issuperset({120, 180, 270, 365})
    allowed_statuses = {"READY_FOR_MODEL_DATASET_PREPARATION", "CONDITIONALLY_READY", "NOT_READY"}
    for s_info in rd["strategies"].values():
        assert s_info["status"] in allowed_statuses


def test_21_upstream_immutability_and_no_ml(p3d_bundle):
    manifest = p3d_bundle["manifest"]
    assert manifest["ml_models_trained"] is False
    assert manifest["rankings_or_top10_generated"] is False
    assert manifest["upstream_immutability_verified"] is True
    for relpath, expected_sha in manifest["upstream_sha256_verified"].items():
        actual_sha = compute_file_sha256(REPO_ROOT / relpath)
        assert actual_sha == expected_sha, f"Upstream modified: {relpath}"
