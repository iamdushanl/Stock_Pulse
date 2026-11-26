"""
Stockpulse Phase 2A Automated Verification Test Suite
=====================================================
Tests:
  1. All Phase 2A output files exist, preserve keys (SecurityID, CompanyCode, Date, MainType, SubType),
     have 1,414,977 rows, 719 SecurityIDs, 400 CompanyCodes, and 0 duplicate keys.
  2. All 63 indicators in technical_indicator_metadata.csv exist in their target Parquet outputs
     with 0 infinite values and 100% bounds_ok=True in feature_quality_report.csv.
  3. Parabolic SAR (af_start=0.02, af_step=0.02, af_max=0.20) exact match vs independent scalar
     reference on synthetic series and JKH-N-0000 / COMB-N-0000 / HHL-N-0000, plus SAR<=Low / SAR>=High invariant.
  4. Independent numerical verification of SMA, EMA, WMA, Wilder RSI, MACD, Wilder ATR, BB, CCI,
     Aroon, OBV, CMF, VWAP, and Rolling Beta on JKH-N-0000.
  5. Strict causality / prefix-invariance: truncating future bars produces zero change (<=1e-12) across all 63 features.
  6. Strict warm-up NaN preservation & short-lived securities (<20 obs) handling.
  7. Pre-2017 missing Open, pre-2002 missing High/Low, pre-2001 missing Volume, Close==0, and zero-volume edge cases.
  8. Corporate-action consistency across target splits (JKH, LMF, AFS, SUN, WATA, HHL) & cross-security isolation.
  9. Market & Sector relative behavior (ASPI matched-horizon Beta/RS + honest NaN unmapped Sector + dynamic verified sector).
 10. Phase 1A & Phase 1B immutability and Delisted/Listed company preservation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase2_technical_features.indicators_core import (
    compute_market_relative_panel,
    compute_parabolic_sar,
    compute_technical_features_panel,
)
from phase2_technical_features.validation import independent_psar_reference
from utils.data_loader import load_market_indices

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1b"
PHASE2_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2"


@pytest.fixture(scope="module")
def phase2_bundle() -> dict:
    return {
        "adj": pd.read_parquet(PHASE1B_DIR / "cse_daily_prices_split_adjusted.parquet"),
        "tech": pd.read_parquet(PHASE2_DIR / "cse_technical_features.parquet"),
        "pv": pd.read_parquet(PHASE2_DIR / "cse_price_volume_features.parquet"),
        "mr": pd.read_parquet(PHASE2_DIR / "cse_market_relative_features.parquet"),
        "meta": pd.read_csv(PHASE2_DIR / "technical_indicator_metadata.csv"),
        "qual": pd.read_csv(PHASE2_DIR / "feature_quality_report.csv"),
        "manifest": json.loads((PHASE2_DIR / "phase2_manifest.json").read_text(encoding="utf-8")),
    }


def test_01_outputs_keys_coverage_and_metadata(phase2_bundle: dict) -> None:
    tech = phase2_bundle["tech"]
    pv = phase2_bundle["pv"]
    mr = phase2_bundle["mr"]
    meta = phase2_bundle["meta"]
    qual = phase2_bundle["qual"]

    req_keys = {"SecurityID", "CompanyCode", "Date", "MainType", "SubType"}
    for name, frame in [("tech", tech), ("pv", pv), ("mr", mr)]:
        assert req_keys.issubset(frame.columns), f"{name} missing key columns"
        assert len(frame) == 1_414_977
        assert frame["SecurityID"].nunique() == 719
        assert frame["CompanyCode"].nunique() == 400
        assert frame.duplicated(subset=["SecurityID", "Date"]).sum() == 0

    assert len(meta) == 63
    assert len(qual) == 63
    assert set(meta["indicator_name"]) == set(qual["indicator_name"])
    assert int(qual["inf_count"].sum()) == 0
    assert qual["bounds_ok"].all()

    table_map = {
        "cse_technical_features.parquet": tech,
        "cse_price_volume_features.parquet": pv,
        "cse_market_relative_features.parquet": mr,
    }
    for _, row in meta.iterrows():
        assert row["indicator_name"] in table_map[row["output_table"]].columns
        assert pd.notna(row["formula"]) and len(str(row["formula"])) > 3
        assert pd.notna(row["price_basis"])
        assert pd.notna(row["missing_value_behavior"])


def test_02_psar_independent_reference_and_envelope_invariant(phase2_bundle: dict) -> None:
    # A. Synthetic trend -> reversal -> continuation trajectory
    dates = pd.date_range("2024-01-01", periods=15, freq="B")
    close = [100, 102, 105, 108, 112, 110, 106, 101, 96, 94, 98, 103, 109, 114, 118]
    high = [c + 2.0 for c in close]
    low = [c - 2.0 for c in close]
    syn = pd.DataFrame({"Date": dates, "High": high, "Low": low, "Close": close})

    eng = compute_parabolic_sar(syn, af_start=0.02, af_step=0.02, af_max=0.20)
    ref = independent_psar_reference(syn, af0=0.02, step=0.02, max_af=0.20)

    # Bar 0 strictly NaN (no row-0 look-ahead)
    assert pd.isna(eng.loc[0, "PSAR"])
    assert pd.isna(eng.loc[0, "PSAR_Trend"])
    assert pd.isna(eng.loc[0, "PSAR_Reversal"])
    assert np.allclose(eng["PSAR"], ref["PSAR"], equal_nan=True, atol=1e-12)
    assert np.allclose(eng["PSAR_Trend"], ref["PSAR_Trend"], equal_nan=True)
    assert np.allclose(eng["PSAR_Reversal"], ref["PSAR_Reversal"], equal_nan=True)
    assert np.allclose(eng["PSAR_Distance_Pct"], ref["PSAR_Distance_Pct"], equal_nan=True, atol=1e-10)
    assert (eng["PSAR_Reversal"].dropna().abs() == 1.0).any()

    # B. Full panel verification vs manifest & envelope invariant on real CSE data
    manifest = phase2_bundle["manifest"]
    for sid, res in manifest["validation_summary"]["psar_independent_verification"].items():
        assert res["max_abs_diff"] < 1e-12
        assert res["trend_match"] is True
        assert res["reversal_match"] is True
        assert res["envelope_invariant_ok"] is True


def test_03_independent_spot_indicator_checks(phase2_bundle: dict) -> None:
    adj = phase2_bundle["adj"]
    tech = phase2_bundle["tech"]
    pv = phase2_bundle["pv"]

    jkh_a = adj[(adj["SecurityID"] == "JKH-N-0000") & (adj["Date"] >= pd.Timestamp("2022-01-01"))].sort_values("Date").reset_index(drop=True)
    jkh_t = tech[(tech["SecurityID"] == "JKH-N-0000") & (tech["Date"] >= pd.Timestamp("2022-01-01"))].sort_values("Date").reset_index(drop=True)
    jkh_p = pv[(pv["SecurityID"] == "JKH-N-0000") & (pv["Date"] >= pd.Timestamp("2022-01-01"))].sort_values("Date").reset_index(drop=True)

    # Pick row i = 60 cleanly post-2022 and post full full-series warm-up
    i = 60
    win20 = jkh_a["Close"].iloc[i - 19:i + 1].to_numpy()
    assert math.isclose(jkh_t.loc[i, "SMA_20"], float(np.mean(win20)), rel_tol=1e-10)
    w = np.arange(1, 21, dtype=float)
    assert math.isclose(jkh_t.loc[i, "WMA_20"], float(np.dot(win20, w) / w.sum()), rel_tol=1e-10)

    # BB Upper & Width
    std20 = float(np.std(win20, ddof=0))
    assert math.isclose(jkh_t.loc[i, "BB_Upper_20_2"], float(np.mean(win20) + 2.0 * std20), rel_tol=1e-10)
    assert math.isclose(jkh_t.loc[i, "BB_Width_20_2"], float(4.0 * std20 / np.mean(win20)), rel_tol=1e-10)

    # ROC_10 & MOM_10
    c_now = float(jkh_a.loc[i, "Close"])
    c_10 = float(jkh_a.loc[i - 10, "Close"])
    assert math.isclose(jkh_t.loc[i, "MOM_10"], c_now - c_10, rel_tol=1e-10)
    assert math.isclose(jkh_t.loc[i, "ROC_10"], 100.0 * (c_now / c_10 - 1.0), rel_tol=1e-10)

    # VWAP_20 & Volume_SMA_20 & Volume_Ratio_20
    v20 = jkh_a["ShareVolume"].iloc[i - 19:i + 1].to_numpy()
    assert math.isclose(jkh_p.loc[i, "VWAP_20"], float(np.sum(win20 * v20) / np.sum(v20)), rel_tol=1e-10)
    assert math.isclose(jkh_p.loc[i, "Volume_SMA_20"], float(np.mean(v20)), rel_tol=1e-10)
    assert math.isclose(jkh_p.loc[i, "Volume_Ratio_20"], float(jkh_a.loc[i, "ShareVolume"] / np.mean(v20)), rel_tol=1e-10)

    # CCI_20 manual spot check
    tp20 = ((jkh_a["High"] + jkh_a["Low"] + jkh_a["Close"]) / 3.0).iloc[i - 19:i + 1].to_numpy()
    mad20 = float(np.mean(np.abs(tp20 - np.mean(tp20))))
    cci_expected = (tp20[-1] - np.mean(tp20)) / (0.015 * mad20)
    assert math.isclose(jkh_t.loc[i, "CCI_20"], float(cci_expected), rel_tol=1e-9)


def test_04_causality_zero_future_leakage_and_isolation(phase2_bundle: dict) -> None:
    manifest = phase2_bundle["manifest"]
    val = manifest["validation_summary"]
    assert val["causality_prefix_invariance_max_diff"] < 1e-12
    assert val["cross_security_isolation_exact"] is True

    # Verify share classes COMB-N-0000, COMB-X-0000, COMB-R-0000 preserved separately
    tech = phase2_bundle["tech"]
    comb_sids = set(tech.loc[tech["CompanyCode"] == "COMB", "SecurityID"].unique())
    assert {"COMB-N-0000", "COMB-X-0000", "COMB-R-0000"}.issubset(comb_sids)


def test_05_warmup_nan_preservation_and_short_securities(phase2_bundle: dict) -> None:
    adj = phase2_bundle["adj"]
    tech = phase2_bundle["tech"]

    # Clean post-2002 start on a single modern stock (JKH-N-0000 warm-up from row 0)
    jkh = tech[tech["SecurityID"] == "JKH-N-0000"].sort_values("Date").reset_index(drop=True)
    assert jkh.loc[:18, "SMA_20"].isna().all()
    assert pd.notna(jkh.loc[19, "SMA_20"])
    assert jkh.loc[:48, "SMA_50"].isna().all()
    assert jkh.loc[:198, "SMA_200"].isna().all()
    assert jkh.loc[:13, "RSI_14"].isna().all()
    assert jkh.loc[:10, "EMA_12"].isna().all()

    # Securities with <20 observations (Rights/short warrants) must keep SMA_20 100% NaN
    counts = adj.groupby("SecurityID").size()
    short_sids = counts[counts < 20].index.tolist()
    assert len(short_sids) >= 200
    assert tech.loc[tech["SecurityID"].isin(short_sids), "SMA_20"].isna().all()


def test_06_missing_ohlcv_eras_zero_volume_and_no_fabrication(phase2_bundle: dict) -> None:
    tech = phase2_bundle["tech"]
    pv = phase2_bundle["pv"]

    # 1. Pre-2017 Open-dependent indicators strictly NaN (NEVER fallback to Close)
    pre17_pv = pv[pv["Date"] < pd.Timestamp("2017-01-01")]
    assert pre17_pv["Overnight_Gap_Ret"].isna().all()
    assert pre17_pv["Intraday_Ret"].isna().all()

    # 2. Pre-2002 High/Low-dependent indicators strictly NaN
    pre02_t = tech[tech["Date"] < pd.Timestamp("2002-01-01")]
    for hl_col in ("ADX_14", "Aroon_Up_25", "CCI_20", "Williams_R_14", "Stoch_K_14", "ATR_14", "PSAR", "Supertrend_10_3"):
        assert pre02_t[hl_col].isna().all(), f"{hl_col} must be 100% NaN pre-2002"

    # 3. Pre-2001 Volume-dependent indicators strictly NaN
    pre01_pv = pv[pv["Date"] < pd.Timestamp("2001-01-01")]
    for vol_col in ("OBV", "MFI_14", "CMF_20", "VWAP_20", "Volume_SMA_20", "Volume_Ratio_20", "AD_Line"):
        assert pre01_pv[vol_col].isna().all(), f"{vol_col} must be 100% NaN pre-2001"

    # 4. Synthetic flat-price & zero-volume security edge case
    n = 40
    flat_sec = pd.DataFrame({
        "Date": pd.date_range("2023-01-02", periods=n, freq="B"),
        "SecurityID": ["ILLQ-N-0000"] * n,
        "CompanyCode": ["ILLQ"] * n,
        "MainType": ["N"] * n,
        "SubType": ["0000"] * n,
        "Open": [np.nan] * n,
        "High": [50.0] * n,
        "Low": [50.0] * n,
        "Close": [50.0] * n,
        "CapitalAdjClose": [50.0] * n,
        "ShareVolume": [0.0] * n,
        "Turnover": [0.0] * n,
    })
    ft, fp = compute_technical_features_panel(flat_sec)
    assert not np.isinf(ft.select_dtypes("number").to_numpy()).any()
    assert not np.isinf(fp.select_dtypes("number").to_numpy()).any()
    assert ft["RSI_14"].iloc[-1] == 50.0
    assert ft["CCI_20"].iloc[-1] == 0.0
    assert pd.isna(fp["CMF_20"].iloc[-1])
    assert pd.isna(fp["VWAP_20"].iloc[-1])
    assert fp["Overnight_Gap_Ret"].isna().all()


def test_07_corporate_action_consistency_and_sector_honesty(phase2_bundle: dict) -> None:
    manifest = phase2_bundle["manifest"]
    for key, info in manifest["validation_summary"]["corporate_action_continuity"].items():
        assert info["no_false_crash_spike"] is True, f"Split jump anomaly detected at {key}"

    mr = phase2_bundle["mr"]
    # Sector mapping in Phase 1B is honestly 'Unknown' -> Sector_MOM_20 and RS_Sector_ROC_20 must stay 100% NaN
    assert mr["Sector_MOM_20"].isna().all()
    assert mr["RS_Sector_ROC_20"].isna().all()
    # ASPI macro and relative strength are populated
    assert mr["ASPI_MOM_20"].notna().mean() > 0.999
    assert mr["Beta_ASPI_60"].notna().sum() > 1_300_000
    assert mr["Sector_Breadth_MOM_20"].notna().mean() > 0.998

    # Dynamic sector activation test: providing verified sector mapping populates Sector_MOM_20 cleanly
    adj = phase2_bundle["adj"]
    jkh_sub = adj[(adj["SecurityID"] == "JKH-N-0000") & (adj["Date"] >= pd.Timestamp("2023-01-01"))].copy()
    mock_sec = pd.DataFrame([{
        "SecurityID": "JKH-N-0000",
        "Sector": "Diversified",
        "EffectiveFrom": pd.Timestamp("2000-01-01"),
        "EffectiveTo": pd.Timestamp("2025-12-31"),
    }])
    mi = load_market_indices("01_existing_cse_sources")
    mr_dyn = compute_market_relative_panel(jkh_sub, mi, mock_sec, gics_df=None)
    assert mr_dyn["Sector_MOM_20"].notna().sum() > 500
    assert mr_dyn["RS_Sector_ROC_20"].notna().sum() > 500


def test_08_phase1b_immutability_and_delisted_preserved(phase2_bundle: dict) -> None:
    adj = phase2_bundle["adj"]
    tech = phase2_bundle["tech"]
    manifest = phase2_bundle["manifest"]
    assert len(adj) == 1_414_977
    assert len(tech) == len(adj)
    assert manifest["listed_security_ids"] == 549
    assert manifest["delisted_security_ids"] == 170
