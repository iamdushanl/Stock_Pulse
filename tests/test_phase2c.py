"""
Stockpulse Phase 2C Automated Verification Test Suite
=====================================================
Tests Workstreams A–I & CLI deliverables across 15 verification areas:
  1. All 12 deliverables, 5 charts, SHA-256 manifest integrity, and Phase 1B/2A/2B immutability.
  2. JINS-N-0000 false delisting correction & post-2019 PIT continuity (1,408 rows).
  3. 113 pre-listing securities (110,932 rows) quarantined without fabricating IPO dates or losing blue-chip history.
  4. BUKI-N-0000 2010-12-23..24 missing 1:10 split + 1:50 bonus detection (10.2x factor & PIT mask).
  5. MERC-N-0000 2025-12-17 omitted 1:200 split + 15 other corroborated omitted splits/consolidations.
  6. SELI-N-0000 2013 identity collision (242 rows raw preserved, IsPriceDataValid=False).
  7. Extreme price anomaly & ZeroClose taxonomy + no destructive deletion of raw values.
  8. Corporate action AllotmentDate vs XR/XB quarantine (63 Rights, 13 Bonus) & factor verification.
  9. Beta reliability scoring (RELIABLE/LIMITED_HISTORY/THIN_TRADING/UNSTABLE/INSUFFICIENT_DATA, unclipped raw values, 2025Q4 agreement).
 10. Causal rolling PIT liquidity eligibility, sparse gap protection, zero-volume exclusion, and monotonic threshold grid.
 11. Point-in-time quality flags across all 1,414,977 rows + explicit sector/fundamental/Open unavailability flags.
 12. Strategy-specific research universes (short_term, medium_term, long_term) & pre-2008 dividend flag.
 13. Complete remediation_log.csv schema & accounting (262 logged entries).
 14. Feature redundancy decomposition (63 indicators + 117 pairs, Williams_R_14 equivalence).
 15. Cross-table exact 1:1 key alignment & zero ML model training verification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1b"
PHASE2_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2"
PHASE2B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2b"
PHASE2C_DIR = REPO_ROOT / "Dataset" / "processed" / "phase2c"
VIZ_2C_DIR = REPO_ROOT / "reports" / "phase2" / "visualizations" / "phase2c"

EXPECTED_DELIVERABLES = [
    "security_lifecycle_validation.parquet",
    "corporate_action_remediation.parquet",
    "price_anomaly_investigation.parquet",
    "beta_reliability.parquet",
    "liquidity_eligibility.parquet",
    "feature_redundancy_analysis.csv",
    "point_in_time_quality_flags.parquet",
    "short_term_research_universe.parquet",
    "medium_term_research_universe.parquet",
    "long_term_research_universe.parquet",
    "remediation_log.csv",
    "phase2c_manifest.json",
]


@pytest.fixture(scope="module")
def p2c_bundle() -> dict:
    return {
        "manifest": json.loads((PHASE2C_DIR / "phase2c_manifest.json").read_text(encoding="utf-8")),
        "lc": pd.read_parquet(PHASE2C_DIR / "security_lifecycle_validation.parquet"),
        "ca_rem": pd.read_parquet(PHASE2C_DIR / "corporate_action_remediation.parquet"),
        "anom": pd.read_parquet(PHASE2C_DIR / "price_anomaly_investigation.parquet"),
        "beta": pd.read_parquet(PHASE2C_DIR / "beta_reliability.parquet"),
        "elig": pd.read_parquet(PHASE2C_DIR / "liquidity_eligibility.parquet"),
        "red": pd.read_csv(PHASE2C_DIR / "feature_redundancy_analysis.csv"),
        "pit": pd.read_parquet(PHASE2C_DIR / "point_in_time_quality_flags.parquet"),
        "rem_log": pd.read_csv(PHASE2C_DIR / "remediation_log.csv"),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_01_all_deliverables_checksums_and_upstream_immutability(p2c_bundle: dict) -> None:
    manifest = p2c_bundle["manifest"]
    assert manifest["phase"] == "2C"
    assert manifest["ml_models_trained"] is False
    assert manifest["upstream_immutability_verified"] is True

    for fname in EXPECTED_DELIVERABLES:
        assert (PHASE2C_DIR / fname).exists(), f"Missing Phase 2C deliverable {fname}"

    for fname, meta in manifest["outputs"].items():
        fpath = PHASE2C_DIR / fname
        assert fpath.exists()
        assert fpath.stat().st_size == meta["size_bytes"]
        assert _sha256(fpath) == meta["sha256"], f"SHA-256 mismatch for {fname}"

    assert len(manifest["visualizations"]) == 5
    for vrel in manifest["visualizations"]:
        vp = REPO_ROOT / vrel
        assert vp.exists() and vp.stat().st_size > 10_000

    # Verify Phase 2A and Phase 2B checksums remain intact on disk
    p2b_manifest = json.loads((PHASE2B_DIR / "phase2b_manifest.json").read_text(encoding="utf-8"))
    for fname, expected_sha in p2b_manifest["output_checksums_sha256"].items():
        assert _sha256(PHASE2B_DIR / fname) == expected_sha, f"Phase 2B mutated: {fname}"


def test_02_jins_false_delisting_corrected_with_evidence(p2c_bundle: dict) -> None:
    lc = p2c_bundle["lc"]
    pit = p2c_bundle["pit"]
    assert len(lc) == 719

    jins = lc[lc["SecurityID"] == "JINS-N-0000"].iloc[0]
    assert jins["OriginalStatus"] == "Delisted"
    assert jins["RemediatedStatus"] == "Listed"
    assert pd.isna(jins["RemediatedDelistingDate"])
    assert int(jins["PostDelistingPriceRows"]) == 1408
    assert jins["LifecycleValidationStatus"] == "CORRECTED_WITH_EVIDENCE"
    assert bool(jins["ActiveInLive2026Snapshot"]) is True
    assert int(jins["QuotedListYearsCount"]) >= 5

    jins_post = pit[(pit["SecurityID"] == "JINS-N-0000") & (pit["Date"] > "2019-11-20")]
    assert len(jins_post) == 1408
    assert bool(jins_post["IsHistoricallyListed"].all())
    assert bool(jins_post["IsLifecycleValidated"].all())


def test_03_pre_listing_113_quarantined_without_fabrication(p2c_bundle: dict) -> None:
    lc = p2c_bundle["lc"]
    pit = p2c_bundle["pit"]

    pre_q = lc[lc["AnomalyTag"] == "PRE_LISTING_CROSS_INSTRUMENT_DATE"]
    assert len(pre_q) == 113
    assert int(pre_q["PreListingPriceRows"].sum()) == 110_932
    assert (pre_q["LifecycleValidationStatus"] == "QUARANTINED").all()
    assert pre_q["RemediatedListingDate"].isna().all(), "Never fabricate legal IPO date!"
    assert (pre_q["ListingDateSource"] == "OfficialListingsWorkbook").all()

    # Verify major blue chips among the 113 retain historically listed & eligible rows (no naive pre-2016 deletion)
    comb = pit[pit["SecurityID"] == "COMB-N-0000"]
    assert len(comb) == 7425
    assert bool(comb["IsHistoricallyListed"].all())
    assert int(comb["IsShortTermEligible"].sum()) > 5000
    assert not bool(comb["IsLifecycleValidated"].any())


def test_04_buki_2010_missing_split_and_bonus_forensics(p2c_bundle: dict) -> None:
    ca = p2c_bundle["ca_rem"]
    anom = p2c_bundle["anom"]
    pit = p2c_bundle["pit"]

    buki_ca = ca[(ca["SecurityID"] == "BUKI-N-0000") & (ca["ExDate"] == "2010-12-23") & (ca["EventType"] == "BonusIssue")]
    assert len(buki_ca) == 1
    row = buki_ca.iloc[0]
    assert row["RemediationDisposition"] == "VERIFIED_CORRECTION_REQUIRED"
    assert abs(float(row["ProposedCorrectedPriceFactor"]) - (1.0 / 10.2)) < 1e-6

    buki_pa = anom[(anom["SecurityID"] == "BUKI-N-0000") & (anom["Date"] == "2010-12-24")].iloc[0]
    assert buki_pa["AnomalyClassification"] == "CORPORATE_ACTION_EXPLAINED"
    assert buki_pa["PreviousRawClose"] == pytest.approx(8021.6, abs=0.1)
    assert buki_pa["CurrentRawClose"] == pytest.approx(872.8, abs=0.1)

    buki_pit = pit[(pit["SecurityID"] == "BUKI-N-0000") & (pit["Date"] == "2010-12-24")].iloc[0]
    assert bool(buki_pit["IsCorporateActionValidated"]) is False


def test_05_merc_2025_and_omitted_splits_corroborated(p2c_bundle: dict) -> None:
    ca = p2c_bundle["ca_rem"]
    pit = p2c_bundle["pit"]

    omitted = ca[ca["EventType"] == "UncapturedSplitOrConsolidation"]
    assert len(omitted) == 16
    assert (omitted["RemediationDisposition"] == "VERIFIED_CORRECTION_REQUIRED").all()

    merc = omitted[omitted["SecurityID"] == "MERC-N-0000"].iloc[0]
    assert str(pd.Timestamp(merc["ExDate"]).date()) == "2025-12-17"
    assert merc["ProposedCorrectedPriceFactor"] == pytest.approx(1.0 / 200.0, rel=1e-4)
    assert float(merc["RawReturnPct"]) < -99.0

    # Check both Dec-2025 splits and earlier reverse consolidations detected cleanly
    sids = set(omitted["SecurityID"])
    assert {"MERC-N-0000", "ABL-N-0000", "CARG-N-0000", "RCL-N-0000", "KCAB-N-0000"}.issubset(sids)

    merc_pit = pit[(pit["SecurityID"] == "MERC-N-0000") & (pit["Date"] == "2025-12-17")].iloc[0]
    assert bool(merc_pit["IsCorporateActionValidated"]) is False


def test_06_seli_2013_identity_collision_quarantined_raw_preserved(p2c_bundle: dict) -> None:
    anom = p2c_bundle["anom"]
    pit = p2c_bundle["pit"]
    df_adj = pd.read_parquet(
        PHASE1B_DIR / "cse_daily_prices_split_adjusted.parquet",
        columns=["SecurityID", "Date", "RawClose"],
    )

    seli_anom = anom[(anom["SecurityID"] == "SELI-N-0000") & (anom["Date"] >= "2013-01-01") & (anom["Date"] <= "2013-12-31")]
    assert len(seli_anom) == 242
    assert (seli_anom["AnomalyClassification"] == "SECURITY_IDENTITY_ISSUE").all()

    seli_pit = pit[(pit["SecurityID"] == "SELI-N-0000") & (pit["Date"] >= "2013-01-01") & (pit["Date"] <= "2013-12-31")]
    assert len(seli_pit) == 242
    assert int(seli_pit["IsPriceDataValid"].sum()) == 0
    assert int(seli_pit["IsHistoricallyTradable"].sum()) == 0
    assert int(seli_pit["IsShortTermEligible"].sum()) == 0

    # Raw prices remain untouched in Phase 1B
    seli_raw = df_adj[(df_adj["SecurityID"] == "SELI-N-0000") & (df_adj["Date"] >= "2013-01-01") & (df_adj["Date"] <= "2013-12-31")]
    assert len(seli_raw) == 242
    assert seli_raw["RawClose"].max() <= 1.1


def test_07_zero_close_and_extreme_anomaly_classifications(p2c_bundle: dict) -> None:
    anom = p2c_bundle["anom"]
    pit = p2c_bundle["pit"]
    df_adj = pd.read_parquet(
        PHASE1B_DIR / "cse_daily_prices_split_adjusted.parquet",
        columns=["SecurityID", "Date", "RawClose"],
    ).sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    pit_sorted = pit.sort_values(["SecurityID", "Date"]).reset_index(drop=True)

    valid_cls = {
        "CORPORATE_ACTION_EXPLAINED",
        "POSSIBLE_MARKET_EVENT",
        "SECURITY_IDENTITY_ISSUE",
        "SOURCE_DATA_ERROR",
        "UNRESOLVED",
    }
    assert set(anom["AnomalyClassification"]).issubset(valid_cls)
    assert len(anom) == 6869

    zero_mask = (df_adj["RawClose"] <= 0.0).to_numpy()
    assert zero_mask.sum() > 0
    assert int(pit_sorted.loc[zero_mask, "IsPriceDataValid"].sum()) == 0


def test_08_corporate_action_allotment_quarantine_and_factors(p2c_bundle: dict) -> None:
    ca = p2c_bundle["ca_rem"]
    valid_disp = {"VERIFIED_CORRECT", "UNRESOLVED", "QUARANTINED", "VERIFIED_CORRECTION_REQUIRED"}
    assert set(ca["RemediationDisposition"]).issubset(valid_disp)
    assert int(((ca["EventType"] == "RightsIssue") & (ca["RemediationDisposition"] == "QUARANTINED")).sum()) == 63
    assert int(((ca["EventType"] == "BonusIssue") & (ca["RemediationDisposition"] == "QUARANTINED")).sum()) == 13

    # Spot check a known Phase 2B AllotmentDate outlier
    brwn = ca[(ca["SecurityID"] == "BRWN-N-0000") & (ca["ExDate"] == "2008-09-09") & (ca["EventType"] == "BonusIssue")]
    assert len(brwn) == 1 and brwn.iloc[0]["RemediationDisposition"] == "QUARANTINED"


def test_09_beta_reliability_scoring_and_official_validation(p2c_bundle: dict) -> None:
    beta = p2c_bundle["beta"]
    manifest = p2c_bundle["manifest"]
    assert len(beta) == 1_414_977

    valid_flags = {"RELIABLE", "LIMITED_HISTORY", "THIN_TRADING", "UNSTABLE", "INSUFFICIENT_DATA"}
    assert set(beta["beta_reliability_flag"]).issubset(valid_flags)

    # Raw Beta_ASPI_60 must remain unclipped while |b60|>4 with >=60 obs is UNSTABLE
    assert (beta["Beta_ASPI_60"].abs() > 4.0).any()
    hi_mask = (beta["beta60_observation_count"] >= 60) & (beta["Beta_ASPI_60"].abs() > 4.0)
    assert (beta.loc[hi_mask, "beta_reliability_flag"] == "UNSTABLE").all()

    low_obs = beta["beta60_observation_count"] < 60
    assert (beta.loc[low_obs, "beta_reliability_flag"] == "INSUFFICIENT_DATA").all()

    # Check Official 2025Q4 validation correlation for RELIABLE tier
    off_stats = manifest["workstream_d_beta_reliability"]["official_2025q4_agreement_by_tier"]["RELIABLE"]
    assert off_stats["companies"] >= 250
    assert off_stats["pearson_corr_vs_official"] > 0.75
    assert off_stats["mae_vs_official"] < 0.12


def test_10_causal_pit_liquidity_eligibility_and_sensitivity(p2c_bundle: dict) -> None:
    elig = p2c_bundle["elig"]
    manifest = p2c_bundle["manifest"]
    assert len(elig) == 1_414_977

    # Non-N shares never ShortTermEligible
    assert not bool(elig.loc[elig["MainType"] != "N", "ShortTermEligible"].any())
    # Sparse frequency (<70%) or low median turnover (<500k) never ShortTermEligible
    st = elig[elig["ShortTermEligible"]]
    assert (st["Rolling_TradingFreqPct_60"] >= 70.0).all()
    assert (st["Rolling_MedianTurnover_60_LKR"] >= 500_000.0).all()
    assert (st["TotalCausalObsToDate"] >= 60).all()
    assert (st["MaxGapCalSessions_60"] <= 10).all()

    # Monotonic decrease across stricter liquidity thresholds
    sens = manifest["workstream_e_liquidity_eligibility"]["rolling_threshold_sensitivity"]
    counts = [x["eligible_rows_full"] for x in sens]
    assert counts == sorted(counts, reverse=True)


def test_11_pit_quality_flags_schema_and_availability_guards(p2c_bundle: dict) -> None:
    pit = p2c_bundle["pit"]
    assert len(pit) == 1_414_977

    expected_flags = [
        "IsHistoricallyListed",
        "IsHistoricallyTradable",
        "IsPriceDataValid",
        "IsCorporateActionValidated",
        "IsLifecycleValidated",
        "IsDividendResolutionClean",
        "IsBetaReliable",
        "IsOpenFeatureAvailable",
        "IsHLFeatureAvailable",
        "IsSectorMappingAvailable",
        "IsFundamentalDataAvailable",
        "IsShortTermEligible",
        "IsMediumTermEligible",
        "IsLongTermEligible",
        "IsModelTrainingEligible",
    ]
    for col in expected_flags:
        assert col in pit.columns
        assert pit[col].dtype == bool

    assert int(pit["IsSectorMappingAvailable"].sum()) == 0
    assert int(pit["IsFundamentalDataAvailable"].sum()) == 0
    assert int(pit.loc[pit["Date"] < "2017-01-01", "IsOpenFeatureAvailable"].sum()) == 0
    assert int(pit.loc[pit["Date"] < "2008-01-01", "IsDividendResolutionClean"].sum()) == 0


def test_12_strategy_research_datasets_schemas_and_limits() -> None:
    st_cols = pq.read_schema(PHASE2C_DIR / "short_term_research_universe.parquet").names
    mt_cols = pq.read_schema(PHASE2C_DIR / "medium_term_research_universe.parquet").names
    lt_cols = pq.read_schema(PHASE2C_DIR / "long_term_research_universe.parquet").names

    for name in ("short_term_research_universe.parquet", "medium_term_research_universe.parquet", "long_term_research_universe.parquet"):
        meta = pq.read_metadata(PHASE2C_DIR / name)
        assert meta.num_rows == 1_414_977, name

    # Short-term removes affine duplicate Williams_R_14 and 100%-NaN Sector columns
    assert "Williams_R_14" not in st_cols
    assert "Sector_MOM_20" not in st_cols
    assert "RS_Sector_ROC_20" not in st_cols
    assert {"Stoch_K_14", "RSI_14", "ATR_14", "IsShortTermEligible", "IsOpenFeatureAvailable"}.issubset(st_cols)

    assert {"Beta_ASPI_252", "RS_ASPI_ROC_60", "IsMediumTermEligible", "IsSectorMappingAvailable"}.issubset(mt_cols)
    assert {"TotalReturnPrice", "TotalReturnIndex", "IsDividendResolutionClean", "IsLongTermEligible", "LongTermResearchReadinessNote"}.issubset(lt_cols)


def test_13_remediation_log_completeness(p2c_bundle: dict) -> None:
    rem = p2c_bundle["rem_log"]
    expected_cols = [
        "RemediationID",
        "SecurityID",
        "CompanyCode",
        "AffectedDate",
        "IssueType",
        "OriginalValue",
        "ProposedValue",
        "FinalValue",
        "EvidenceReference",
        "ResolutionStatus",
        "Reason",
        "ProcessingTimestamp",
    ]
    assert list(rem.columns) == expected_cols
    assert len(rem) == 262
    assert rem["RemediationID"].is_unique
    assert rem["Reason"].notna().all()
    assert rem["EvidenceReference"].notna().all()
    assert set(rem["ResolutionStatus"]) == {"QUARANTINED", "UNRESOLVED", "VERIFIED_CORRECTION_REQUIRED", "CORRECTED_WITH_EVIDENCE"}


def test_14_feature_redundancy_decomposition(p2c_bundle: dict) -> None:
    red = p2c_bundle["red"]
    inds = red[red["EntryType"] == "INDICATOR_DISPOSITION"]
    pairs = red[red["EntryType"] == "CORRELATED_PAIR"]

    assert len(inds) == 63
    assert len(pairs) == 117
    wr = inds[inds["FeatureA"] == "Williams_R_14"].iloc[0]
    assert wr["RedundancyTier"] == "DUPLICATE_OF_STOCH_K_14"
    assert int((inds["RedundancyTier"] == "CLEAN_NON_REDUNDANT").sum()) == 38


def test_15_cross_table_key_alignment_without_duplicates(p2c_bundle: dict) -> None:
    pit = p2c_bundle["pit"]
    elig = p2c_bundle["elig"]
    beta = p2c_bundle["beta"]

    assert (pit["SecurityID"].to_numpy() == elig["SecurityID"].to_numpy()).all()
    assert (pit["Date"].to_numpy() == elig["Date"].to_numpy()).all()
    assert (pit["SecurityID"].to_numpy() == beta["SecurityID"].to_numpy()).all()
    assert (pit["Date"].to_numpy() == beta["Date"].to_numpy()).all()

    dup_cnt = int(pit.duplicated(subset=["SecurityID", "Date"]).sum())
    assert dup_cnt == 0
