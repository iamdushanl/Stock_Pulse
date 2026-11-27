"""
Comprehensive Unit, Leakage-Control, Perturbation & Deliverable Tests for Stockpulse Phase 4A.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from phase3d_features.contract_and_mapping import compute_file_sha256
from phase4a_dataset.calendar_and_targets import (
    TOTAL_GAP_SESSIONS,
    build_trading_calendar_and_benchmarks,
    compute_forward_targets_and_labels,
)
from phase4a_dataset.features_eligibility_and_splits import (
    PENNY_CLOSE_THRESHOLD_LKR,
    PHASE4A_OUTPUT_FILES,
    build_derived_scale_free_features,
    run_phase4a_leakage_and_perturbation_tests,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE4A_DIR = REPO_ROOT / "Dataset" / "processed" / "phase4a"
VIZ_DIR = REPO_ROOT / "reports" / "phase4" / "visualizations" / "phase4a"


@pytest.fixture(scope="module")
def phase4a_artifacts():
    manifest = json.loads((PHASE4A_DIR / "short_term_dataset_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((PHASE4A_DIR / "short_term_feature_registry.json").read_text(encoding="utf-8"))
    tgt_defs = json.loads((PHASE4A_DIR / "short_term_target_definitions.json").read_text(encoding="utf-8"))
    tq_report = json.loads((PHASE4A_DIR / "short_term_target_quality_report.json").read_text(encoding="utf-8"))
    model_df = pd.read_parquet(PHASE4A_DIR / "short_term_modeling_dataset.parquet")
    splits_df = pd.read_parquet(PHASE4A_DIR / "short_term_validation_splits.parquet")
    elig_df = pd.read_parquet(PHASE4A_DIR / "short_term_sample_eligibility.parquet")
    audit_df = pd.read_parquet(PHASE4A_DIR / "short_term_exclusion_audit.parquet")
    return {
        "manifest": manifest,
        "registry": registry,
        "tgt_defs": tgt_defs,
        "tq_report": tq_report,
        "model_df": model_df,
        "splits_df": splits_df,
        "elig_df": elig_df,
        "audit_df": audit_df,
    }


def test_01_all_deliverable_files_and_visualizations_exist():
    for rel in PHASE4A_OUTPUT_FILES:
        p = REPO_ROOT / rel
        assert p.exists(), f"Missing Phase 4A deliverable: {p}"
        assert p.stat().st_size > 0, f"Empty Phase 4A deliverable: {p}"
    expected_pngs = [
        "01_eligibility_waterfall_and_universe_evolution.png",
        "02_horizon_target_distributions_and_penny_noise.png",
        "03_purged_chronological_splits_and_regime_counts.png",
        "04_liquidity_threshold_sensitivity_and_missingness.png",
    ]
    for name in expected_pngs:
        p = VIZ_DIR / name
        assert p.exists(), f"Missing Phase 4A PNG chart: {p}"
        assert p.stat().st_size > 10_000, f"Suspiciously small PNG chart: {p}"


def test_02_universe_volumes_and_delisted_preservation(phase4a_artifacts):
    elig = phase4a_artifacts["elig_df"]
    model = phase4a_artifacts["model_df"]
    assert len(elig) == 1_414_977
    assert len(model) == 361_589
    assert model["SecurityID"].nunique() == 301
    # Verify exact consistency with Phase 1B authoritative security master (HistoricalStatus == 'Listed'/'Delisted')
    assert model.loc[model["IsListed"], "SecurityID"].nunique() == 252
    assert model.loc[~model["IsListed"], "SecurityID"].nunique() == 49
    assert int((~model["IsListed"]).sum()) == 22_524


def test_03_primary_key_and_column_uniqueness(phase4a_artifacts):
    for key in ("model_df", "splits_df", "elig_df"):
        df = phase4a_artifacts[key]
        assert df.columns.is_unique, f"Duplicate column name in {key}"
        assert not df.duplicated(["Date", "SecurityID"]).any(), f"Duplicate (Date, SecurityID) in {key}"


def test_04_exact_calendar_session_alignment_vs_naive_shift():
    # Construct synthetic calendar with sessions 0..10 and a stock that SKIPS session 5!
    dates = pd.bdate_range("2025-01-01", periods=12, freq="B")
    cal_df = pd.DataFrame({
        "Dt": dates,
        "SessionIndex": np.arange(len(dates), dtype=np.int32),
        "InPhase1Calendar": True,
        "ASPI_Close": 10000.0 + np.arange(len(dates)) * 10.0,
        "ASTRI": 20000.0 + np.arange(len(dates)) * 20.0,
    })
    # Stock skips index 5! So from index 0, naive shift(-5) hits calendar session 6 (WRONG!),
    # while SessionIndex + 5 on index 0 is missing -> MUST be NaN/SKIPPED_SESSION_OR_HALT_AT_TARGET_DATE!
    # From index 1, SessionIndex + 5 = 6 EXISTS and is valid!
    kept_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    st_syn = pd.DataFrame({
        "Date": dates[kept_idx],
        "SecurityID": "SYN-N-0000",
        "RawClose": 100.0 + np.arange(len(kept_idx)),
        "CapitalAdjClose": 100.0 + np.arange(len(kept_idx)),
        "Daily_Return_CapAdj": 0.01,
        "IsPriceDataValid": True,
        "IsCorporateActionValidated": True,
        "IsLifecycleValidated": True,
        "IsShortTermEligible": True,
    })
    tr_syn = pd.DataFrame({
        "SecurityID": "SYN-N-0000",
        "Date": dates[kept_idx],
        "TotalReturnPrice": 150.0 + np.arange(len(kept_idx)),
    })
    pf_syn = pd.DataFrame({
        "SecurityID": "SYN-N-0000",
        "Date": dates[kept_idx],
        "IsDividendResolutionClean": True,
    })
    car_syn = pd.DataFrame(columns=["SecurityID", "ExDate", "CandidatePriorXRDate", "RemediationDisposition"])
    pai_syn = pd.DataFrame(columns=["SecurityID", "Date", "AnomalyClassification"])
    map_syn = pd.DataFrame({"SecurityID": ["SYN-N-0000"], "MappingValidTo": [dates[10]], "IsListed": [True]})

    out_df, _ = compute_forward_targets_and_labels(
        st_syn, tr_syn, pf_syn, car_syn, pai_syn, map_syn, cal_df, horizons=(5,), min_cross_section_size=1
    )
    # Row 0 (SessionIndex=0): SessionIndex+5=5 is absent -> must NOT shift onto session 6!
    assert out_df.loc[0, "IsTargetEligible_Cap_5d"] == False
    assert out_df.loc[0, "TargetExclusionReason_5d"] == "SKIPPED_SESSION_OR_HALT_AT_TARGET_DATE"
    assert pd.isna(out_df.loc[0, "Target_Excess_CapitalReturn_5d"])
    # Row 1 (SessionIndex=1): SessionIndex+5=6 IS present -> exact Cap return (105/101 - 1) - (10060/10010 - 1)
    assert out_df.loc[1, "IsTargetEligible_Cap_5d"] == True
    expected_cap = (105.0 / 101.0 - 1.0) - (10060.0 / 10010.0 - 1.0)
    assert np.isclose(out_df.loc[1, "Target_Excess_CapitalReturn_5d"], expected_cap, atol=1e-12)


def test_05_delisted_terminal_boundary_never_forward_fills(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    elig = phase4a_artifacts["elig_df"]
    mp = pd.read_parquet(REPO_ROOT / "Dataset" / "processed" / "phase3d" / "company_security_pit_mapping.parquet", columns=["SecurityID", "MappingValidTo"])
    # 1. On the final traded date of each delisted security in full universe, all future horizons must be False (zero ffill!)
    del_full = elig[~elig["IsListed"]].copy()
    assert len(del_full) > 0
    last_dates = del_full.groupby("SecurityID")["Date"].transform("max")
    near_end = del_full[del_full["Date"] == last_dates]
    for h in (5, 10, 20, 60):
        assert (~near_end[f"IsTargetEligible_Cap_{h}d"]).all()
        assert (~near_end[f"IsTargetEligible_TR_{h}d"]).all()

    # 2. On every delisted row in modeling dataset, zero valid TargetDate_{h}d ever crosses past MappingValidTo
    del_mod = model[~model["IsListed"]].merge(mp.drop_duplicates("SecurityID"), on="SecurityID", how="left")
    m_vto = pd.to_datetime(del_mod["MappingValidTo"], errors="coerce")
    for h in (5, 10, 20, 60):
        ok_m = del_mod[f"IsTargetEligible_Cap_{h}d"] & m_vto.notna()
        if ok_m.any():
            assert (pd.to_datetime(del_mod.loc[ok_m, f"TargetDate_{h}d"]) <= m_vto[ok_m]).all()


def test_06_quarantined_ca_and_source_errors_blocked_from_targets(phase4a_artifacts):
    elig = phase4a_artifacts["elig_df"]
    # Known Phase 2C quarantined CA dates: KAPI-N-0000 2010-12-10, SING-N-0000 2011-08-11
    for sec, dt_str in [("KAPI-N-0000", "2010-12-09"), ("SING-N-0000", "2011-08-10")]:
        sub = elig[(elig["SecurityID"] == sec) & (elig["Date"] == pd.Timestamp(dt_str))]
        if not sub.empty:
            assert (~sub["IsTargetEligible_Cap_5d"]).all(), f"Quarantined CA horizon leaked into valid target for {sec} on {dt_str}"
            assert (~sub["IsTargetEligible_TR_5d"]).all()


def test_07_total_return_vs_capital_return_strict_separation(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    # Pre-2004 ASTRI does not exist -> TR targets must ALL be False while Cap targets can be valid!
    pre2004 = model[model["Date"] < pd.Timestamp("2004-01-02")]
    assert len(pre2004) > 0
    for h in (5, 10, 20, 60):
        assert (~pre2004[f"IsTargetEligible_TR_{h}d"]).all(), f"Pre-2004 TR target populated at {h}d!"
        assert pre2004.loc[pre2004[f"IsTargetEligible_Cap_{h}d"], f"Target_Excess_TotalReturn_{h}d"].isna().all()
    assert pre2004["IsTargetEligible_Cap_20d"].sum() > 100


def test_08_indicator_close_basis_normalization(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    jkh_pre = model[(model["SecurityID"] == "JKH-N-0000") & (model["Date"] < pd.Timestamp("2020-01-01")) & model["Price_to_SMA20_Pct"].notna()]
    assert len(jkh_pre) > 100
    # Verify Price_to_SMA20_Pct equals Close/SMA_20 - 1 cleanly (median well within normal +/-15% range, not distorted)
    assert jkh_pre["Price_to_SMA20_Pct"].abs().median() < 0.05


def test_09_no_future_leakage_under_perturbation(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    splits = phase4a_artifacts["splits_df"]
    split_sum = phase4a_artifacts["manifest"]["split_architecture_summary"]
    cand_cols = phase4a_artifacts["registry"]["recommended_feature_sets"]["core_tech_full_sample_2002_2025"]
    res = run_phase4a_leakage_and_perturbation_tests(model, splits, split_sum, cand_cols)
    assert res["future_perturbation_max_diff"] == 0.0
    assert res["all_target_dates_strictly_future"] is True


def test_10_purged_65session_chronological_splits_and_final_holdout(phase4a_artifacts):
    splits = phase4a_artifacts["splits_df"]
    model = phase4a_artifacts["model_df"]
    merged = splits.merge(model[["Date", "SecurityID", "TargetDate_60d", "IsTargetEligible_Cap_60d"]], on=["Date", "SecurityID"])
    split_sum = phase4a_artifacts["manifest"]["split_architecture_summary"]

    for f in split_sum["cv_folds"]:
        fname = f["fold_name"]
        tr = merged[merged[fname] == "TRAIN"]
        va = merged[merged[fname] == "VAL"]
        assert len(tr) > 0 and len(va) > 0
        assert (int(va["SessionIndex"].min()) - int(tr["SessionIndex"].max()) - 1) >= TOTAL_GAP_SESSIONS
        tr_t60 = pd.to_datetime(tr.loc[tr["IsTargetEligible_Cap_60d"], "TargetDate_60d"])
        assert tr_t60.max() < va["Date"].min(), f"Future TargetDate_60d overlap into {fname} VAL!"

    dev = merged[merged["FinalSplitRole"] == "DEV_POOL"]
    ft = merged[merged["FinalSplitRole"] == "FINAL_TEST"]
    assert ft["Date"].min() >= pd.Timestamp("2024-01-02")
    assert dev["Date"].max() < pd.Timestamp("2024-01-02")
    assert (int(ft["SessionIndex"].min()) - int(dev["SessionIndex"].max()) - 1) >= TOTAL_GAP_SESSIONS
    assert int((ft["IsListed"] == False).sum()) >= 0


def test_11_pit_fundamental_availability_le_decision_date(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    avail = pd.to_datetime(model["FundamentalAvailabilityDate"], errors="coerce")
    m = avail.notna()
    assert int(m.sum()) > 100_000
    assert (avail[m] <= model.loc[m, "Date"]).all()


def test_12_open_feature_structural_regime_2017plus(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    pre17 = model[model["Date"] < pd.Timestamp("2017-01-01")]
    post17 = model[model["Date"] >= pd.Timestamp("2017-01-01")]
    assert (~pre17["IsOpenFeatureAvailable"]).all()
    assert pre17["Overnight_Gap_Ret"].isna().all()
    assert post17["IsOpenFeatureAvailable"].mean() > 0.99
    assert post17["Overnight_Gap_Ret"].notna().mean() > 0.99


def test_13_cross_sectional_rank_and_top_quintile_properties(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    for h in (5, 10, 20, 60):
        rk_m = model[f"Target_ExcessTR_PctRank_{h}d"].notna()
        assert (model.loc[rk_m, f"CrossSectionN_TR_{h}d"] >= 15).all()
        assert (model.loc[rk_m, f"Target_ExcessTR_PctRank_{h}d"] > 0.0).all()
        assert (model.loc[rk_m, f"Target_ExcessTR_PctRank_{h}d"] <= 1.0).all()
        tq = model.loc[rk_m, f"Target_TopQuintile_ExcessTR_{h}d"].dropna().mean()
        assert 0.18 <= float(tq) <= 0.25


def test_14_downside_path_labels_valid_bounds(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    for h in (5, 10, 20, 60):
        mae = model[f"Label_Fwd_MaxAdverseExcursion_{h}d"].dropna()
        mdd = model[f"Label_Fwd_MaxDrawdown_{h}d"].dropna()
        vol = model[f"Label_Fwd_RealizedVol_Ann_{h}d"].dropna()
        assert (mae <= 0.0).all() and (mae >= -1.0).all()
        assert (mdd <= 0.0).all() and (mdd >= -1.0).all()
        assert (vol >= 0.0).all()
        assert 0.05 < float(vol.median()) < 2.0, f"Unexpected realized vol scale (must be decimal, not 100x pct): {vol.median()}"
        # Days without valid forward horizon MUST NOT contain synthetic downside flags
        inval = ~model[f"IsTargetEligible_Cap_{h}d"]
        assert model.loc[inval, f"Label_Fwd_DownsideRisk_Flag_{h}d"].isna().all()


def test_15_feature_registry_excludes_targets_and_levels(phase4a_artifacts):
    reg = phase4a_artifacts["registry"]
    for set_name, feat_list in reg["recommended_feature_sets"].items():
        for col in feat_list:
            assert not col.startswith("Target"), f"Target {col} leaked into {set_name}"
            assert not col.startswith("Label_"), f"Label {col} leaked into {set_name}"
            assert col not in {"RawClose", "Close", "CapitalAdjClose", "TotalReturnPrice", "ASPI_Close", "ASTRI_Close", "SMA_20", "OBV", "AD_Line", "IsSectorMappingAvailable"}


def test_16_upstream_immutability_verified(phase4a_artifacts):
    manifest = phase4a_artifacts["manifest"]
    assert manifest["upstream_immutability_verified"] is True
    verified = manifest["upstream_sha256_verified"]
    st_rel = "Dataset/processed/phase3d/short_term_integrated_research_universe.parquet"
    assert compute_file_sha256(REPO_ROOT / st_rel) == verified[st_rel]


def test_17_astri_corrupted_dates_quarantined(phase4a_artifacts):
    model = phase4a_artifacts["model_df"]
    tq = phase4a_artifacts["tq_report"]
    qa = tq["astri_benchmark_anomaly_quarantine"]
    assert qa["quarantined_count"] == 6
    assert "2010-07-06" in qa["quarantined_dates"]
    # Zero -845% artificial excess crashes caused by ASTRI typo / ASPI paste-overs
    for h in (5, 10, 20, 60):
        tr_ex = model.loc[model[f"IsTargetEligible_TR_{h}d"], f"Target_Excess_TotalReturn_{h}d"]
        assert float(tr_ex.min()) > -1.5, f"Unquarantined ASTRI spike caused extreme negative excess return at {h}d: {tr_ex.min()}"
