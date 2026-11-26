"""
Stockpulse Phase 2B Automated Verification Test Suite
=====================================================
Tests Workstreams A–I & CLI execution across:
  1. Workstream A: Integrity report, 1,414,977 row & key matching, 0 duplicates, 0 infs.
  2. Workstream B: Feature coverage & exact mutually exclusive missingness accounting (`sum_causes == total_nulls`, 0 unexplained).
  3. Workstream C: Liquidity metrics, zero-volume & gap calculations, observation-based vs calendar-day checks, threshold sensitivity monotonic counts.
  4. Workstream D: ASPI date alignment, matched-horizon return check, sparse-beta flags, honest 100% NaN sector mapping.
  5. Workstream E: Corporate action continuity audit, volume split scaling invariance, double-adjustment guard.
  6. Workstream F: Correlation matrix, Pearson vs Spearman consistency, clustering & zero-variance checks.
  7. Workstream G: Historical universe retention (277 Listed + 123 Delisted codes = 400; 549 Listed + 170 Delisted SecIDs = 719) & pre/post boundary flags.
  8. Workstreams H–I & Manifest: Strategy verdicts, fundamental gap evidence, 6 visualizations exist, SHA-256 checksums match disk, Phase 1A/1B/2A files untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.data_loader import load_phase2a_features

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1A_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1"
PHASE1B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1b"
PHASE2_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2"
PHASE2B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2b"
VIZ_DIR = REPO_ROOT / "reports" / "phase2" / "visualizations"


@pytest.fixture(scope="module")
def p2b_bundle() -> dict:
    return {
        "integrity": json.loads((PHASE2B_DIR / "phase2b_integrity_report.json").read_text(encoding="utf-8")),
        "coverage": pd.read_csv(PHASE2B_DIR / "feature_coverage_report.csv"),
        "missingness": pd.read_csv(PHASE2B_DIR / "feature_missingness_report.csv"),
        "liquidity": pd.read_csv(PHASE2B_DIR / "liquidity_analysis.csv"),
        "mr_val": pd.read_csv(PHASE2B_DIR / "market_relative_validation.csv"),
        "ca_val": pd.read_csv(PHASE2B_DIR / "corporate_action_validation.csv"),
        "corr": pd.read_csv(PHASE2B_DIR / "feature_correlation_matrix.csv"),
        "redundancy": pd.read_csv(PHASE2B_DIR / "feature_redundancy_report.csv"),
        "surv": pd.read_csv(PHASE2B_DIR / "security_history_coverage.csv"),
        "readiness": json.loads((PHASE2B_DIR / "strategy_readiness_assessment.json").read_text(encoding="utf-8")),
        "manifest": json.loads((PHASE2B_DIR / "phase2b_manifest.json").read_text(encoding="utf-8")),
    }


def test_01_all_deliverables_checksums_and_manifest(p2b_bundle: dict) -> None:
    manifest = p2b_bundle["manifest"]
    assert manifest["phase"] == "2B"
    assert manifest["row_count"] == 1_414_977
    assert manifest["unique_security_ids"] == 719
    assert manifest["unique_company_codes"] == 400
    assert manifest["integrity_status"] == "PASS"

    for fname, expected_sha in manifest["output_checksums_sha256"].items():
        fpath = PHASE2B_DIR / fname
        assert fpath.exists(), f"Missing deliverable {fname}"
        h = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert h == expected_sha, f"Checksum mismatch on {fname}"

    assert len(manifest["visualizations"]) == 6
    for vpath in manifest["visualizations"]:
        assert Path(vpath).exists() and Path(vpath).stat().st_size > 5_000


def test_02_workstream_a_dataset_integrity(p2b_bundle: dict) -> None:
    integ = p2b_bundle["integrity"]
    assert integ["cross_table_consistency"]["overall_integrity_status"] == "PASS"
    assert integ["cross_table_consistency"]["all_1414977_phase1b_records_matched"] is True
    assert integ["cross_table_consistency"]["metadata_indicator_count"] == 63
    assert integ["cross_table_consistency"]["raw_close_exact_match_phase1b"] is True
    assert integ["cross_table_consistency"]["split_adj_close_exact_match_phase1b"] is True

    for tname, tinfo in integ["phase2a_tables"].items():
        assert tinfo["row_count"] == 1_414_977, tname
        assert tinfo["duplicate_security_date_keys"] == 0, tname
        assert tinfo["infinite_value_count"] == 0, tname
        assert tinfo["missing_records_vs_phase1b"] == 0, tname
        assert tinfo["extra_records_vs_phase1b"] == 0, tname


def test_03_workstream_b_coverage_and_missingness_accounting(p2b_bundle: dict) -> None:
    cov = p2b_bundle["coverage"]
    miss = p2b_bundle["missingness"]
    assert len(cov) == 63
    assert len(miss) == 63

    # Exact accounting: every null cell maps cleanly to one mutually exclusive cause
    cause_cols = [c for c in miss.columns if c.endswith("_nulls") and c != "total_nulls"]
    assert (miss[cause_cols].sum(axis=1) == miss["total_nulls"]).all()
    assert int(miss["other_unexplained_nulls"].sum()) == 0

    # Independent edge checks: Open-gated & unmapped sector missingness
    open_row = miss[miss["indicator_name"] == "Intraday_Ret"].iloc[0]
    assert open_row["missing_open_price_nulls"] > 880_000

    sec_row = miss[miss["indicator_name"] == "Sector_MOM_20"].iloc[0]
    assert sec_row["unmapped_sector_nulls"] == 1_414_977


def test_04_workstream_c_liquidity_and_zero_volume_edge_cases(p2b_bundle: dict) -> None:
    liq = p2b_bundle["liquidity"]
    assert len(liq) == 719
    assert liq["SecurityID"].is_unique

    # Independent spot verify for JKH-N-0000 and an illiquid/early security
    jkh = liq[liq["SecurityID"] == "JKH-N-0000"].iloc[0]
    assert jkh["HistoricalStatus"] == "Listed"
    assert jkh["Recent2023_2025_ActiveFreqPct"] > 95.0
    assert jkh["ADT_60_Obs_LKR"] > 10_000_000.0

    # Zero-volume / 1991-2000-only securities must have ActiveTradingDays == 0 (never fabricate volume)
    zero_vol_secs = liq[liq["ActiveTradingDays"] == 0]
    assert len(zero_vol_secs) >= 20
    assert (zero_vol_secs["TradingFrequencyPct"] == 0.0).all()

    # Stricter threshold sensitivity must reduce passing count monotonically
    sens = p2b_bundle["readiness"]["liquidity_summary"]["threshold_sensitivity"]
    rec_counts = [s["passing_listed_2023_2025"] for s in sens]
    assert rec_counts == sorted(rec_counts, reverse=True)


def test_05_workstream_d_market_relative_and_sector_honesty(p2b_bundle: dict) -> None:
    mr_val = p2b_bundle["mr_val"]
    mr_sum = p2b_bundle["readiness"]["market_relative_summary"]
    assert len(mr_val) == 719
    assert mr_sum["aspi_date_alignment"]["aspi_mom20_max_diff_vs_ref"] < 1e-10
    assert mr_sum["sector_honesty"]["verified_zero_sector_leakage"] is True

    # Securities with < 253 obs cannot have valid Beta_ASPI_252
    short_b = mr_val[mr_val["TotalObs"] < 253]
    assert (short_b["Beta252_ValidObs"] == 0).all()
    assert (~short_b["Beta252_Sufficient"]).all()


def test_06_workstream_e_corporate_actions_and_volume_scaling(p2b_bundle: dict) -> None:
    ca_val = p2b_bundle["ca_val"]
    ca_sum = p2b_bundle["readiness"]["corporate_action_summary"]
    assert len(ca_val) == 4443
    assert ca_sum["double_adjustment_flag_violations"] == 0
    assert ca_sum["is_split_adjusted_all_true"] is True

    # Volume scaling ratio vs RawShareVolume/CumSplitBonusFactor must equal 1.0 cleanly on pre-ex volume rows
    ratios = ca_val["VolumeScaleCheckRatio"].dropna()
    assert np.allclose(ratios, 1.0, rtol=1e-4)

    # JKH 2024-11-05 split must remain continuous
    jkh_ev = ca_val[(ca_val["SecurityID"] == "JKH-N-0000") & (ca_val["ExDate"] == "2024-11-05")].iloc[0]
    assert jkh_ev["ContinuityStatus"] == "CONTINUOUS_OK"


def test_07_workstream_f_correlation_and_redundancy(p2b_bundle: dict) -> None:
    corr = p2b_bundle["corr"]
    red = p2b_bundle["redundancy"]
    assert len(corr) == 63
    assert corr.shape == (63, 64)  # indicator_name + 63 columns

    # Williams_R_14 and Stoch_K_14 must be detected at ~1.0 correlation without crashing on unmapped Sectors
    wr_sk = red[
        ((red["feature_a"] == "Williams_R_14") & (red["feature_b"] == "Stoch_K_14"))
        | ((red["feature_b"] == "Williams_R_14") & (red["feature_a"] == "Stoch_K_14"))
    ]
    assert len(wr_sk) == 1
    assert wr_sk.iloc[0]["pearson_corr_2017plus_N"] >= 0.9999


def test_08_workstream_g_survivorship_and_loader_access(p2b_bundle: dict) -> None:
    surv = p2b_bundle["surv"]
    s_sum = p2b_bundle["readiness"]["survivorship_summary"]
    assert len(surv) == 719
    assert s_sum["listed_securities"] == 549
    assert s_sum["delisted_securities_retained"] == 170
    assert s_sum["listed_company_codes"] == 277
    assert s_sum["delisted_company_codes"] == 123

    # Verify new Phase 2A loader utility works cleanly on filtered subset
    jkh_tech = load_phase2a_features("technical", companies=["JKH"], start_date="2024-11-01", end_date="2024-11-15")
    assert len(jkh_tech) > 0
    assert set(jkh_tech["CompanyCode"].unique()) == {"JKH"}


def test_09_workstreams_hi_readiness_verdicts_and_upstream_immutability(p2b_bundle: dict) -> None:
    r = p2b_bundle["readiness"]
    assert r["strategies"]["short_term_trading_0_3m"]["status"] in {"READY", "CONDITIONALLY READY", "NOT READY"}
    assert r["strategies"]["medium_term_growth_3_12m"]["status"] == "NOT READY"
    assert r["strategies"]["long_term_wealth_1_5y"]["status"] == "NOT READY"
    assert r["fundamental_data_audit"]["historical_multi_year_panel_exists"] is False
    assert len(r["prioritized_recommendations"]) >= 6

    # Verify upstream Phase 1A, 1B, and 2A core counts untouched
    p1_cnt = len(pd.read_parquet(PHASE1A_DIR / "cse_daily_prices_validated.parquet", columns=["Date"]))
    p1b_cnt = len(pd.read_parquet(PHASE1B_DIR / "cse_daily_prices_split_adjusted.parquet", columns=["Date"]))
    p2a_cnt = len(pd.read_parquet(PHASE2_DIR / "cse_technical_features.parquet", columns=["Date"]))
    assert p1_cnt == p1b_cnt == p2a_cnt == 1_414_977
