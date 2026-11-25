"""
Stockpulse Phase 1B Automated Verification Test Suite
=====================================================
Covers all 14 Phase 1B Task 6 test requirements across:
  1. New full-CSE data loader & zero silent SP20 fallback
  2. Date-range, company, security-type, SecurityID, and incremental loading
  3. Share class preservation (N, X, R, W, P, D, B, U)
  4. Issuer identity resolution & merger/ambiguity manual review flags
  5. Verified target stock splits (JKH, LMF, AFS, SUN, WATA, HHL, CIC N & X)
  6. Bonus issue & scrip dividend adjustment math
  7. Rights issue TERP math & underwater non-dilutive handling
  8. Cash dividend & Total Return separation (never mixed into split-adjusted Close)
  9. Chronological backward adjustment strictness (Date < ExDate only)
 10. Double-adjustment prevention (IsSplitAdjusted guard & boundary-year dedup)
 11. Date parsing validation & unresolved event flagging (pre-2008 %, missing ExDate)
 12. Historical Listed/Delisted status preservation (400 codes = 277 Listed + 123 Delisted)
 13. Historical sector mapping audit integrity & no future GICS leakage
 14. Factor-to-source-record traceability & Phase 1A immutability
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phase1b_integration.corporate_actions_engine import (
    compute_adjustment_factors_and_series,
    parse_excel_date,
)
from utils.data_cleaning import apply_split_adjustments
from utils.data_loader import (
    load_all_daily_prices,
    load_cse_corporate_actions,
    load_cse_daily_prices,
    load_cse_market_calendar,
    load_cse_security_master,
    load_sp20_daily_prices,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1A_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1"
PHASE1B_DIR = REPO_ROOT / "Dataset" / "processed" / "phase1b"


@pytest.fixture(scope="module")
def phase1b_outputs() -> dict[str, pd.DataFrame]:
    return {
        "integrated": pd.read_parquet(PHASE1B_DIR / "cse_daily_prices_integrated.parquet"),
        "split_adj": pd.read_parquet(PHASE1B_DIR / "cse_daily_prices_split_adjusted.parquet"),
        "tr": pd.read_parquet(PHASE1B_DIR / "cse_total_return_history.parquet"),
        "master": pd.read_parquet(PHASE1B_DIR / "cse_security_master_enriched.parquet"),
        "ca": pd.read_parquet(PHASE1B_DIR / "cse_corporate_actions_validated.parquet"),
        "factors": pd.read_parquet(PHASE1B_DIR / "corporate_action_adjustment_factors.parquet"),
        "sector": pd.read_parquet(PHASE1B_DIR / "cse_sector_mapping.parquet"),
        "manifest": json.loads((PHASE1B_DIR / "phase1b_manifest.json").read_text(encoding="utf-8")),
    }


# ---------------------------------------------------------------------------
# 1 & 2. Data Loader: Full CSE Coverage, No Silent SP20 Fallback, Filters, Dtypes
# ---------------------------------------------------------------------------
def test_01_loader_full_cse_no_silent_sp20_fallback() -> None:
    df_all = load_all_daily_prices()
    assert len(df_all) == 1_414_977
    assert df_all["CompanyCode"].nunique() == 400
    assert df_all["SecurityID"].nunique() == 719
    assert not pd.api.types.is_string_dtype(df_all["Date"])
    assert pd.api.types.is_datetime64_any_dtype(df_all["Date"])
    assert pd.api.types.is_float_dtype(df_all["Close"])

    # Nonexistent path must raise explicitly, never silently fall back to SP20
    with pytest.raises(FileNotFoundError):
        load_cse_daily_prices(path=REPO_ROOT / "Dataset" / "nonexistent_dir_xyz")

    # Separate explicit SP20 comparison loader remains isolated
    assert callable(load_sp20_daily_prices)


def test_02_loader_filters_and_incremental_modes() -> None:
    jkh = load_cse_daily_prices(companies=["JKH"], start_date="2024-10-01", end_date="2024-11-30")
    assert len(jkh) > 0
    assert set(jkh["CompanyCode"].unique()) == {"JKH"}
    assert jkh["Date"].min() >= pd.Timestamp("2024-10-01")
    assert jkh["Date"].max() <= pd.Timestamp("2024-11-30")

    non_voting = load_cse_daily_prices(security_types=["X"], start_date="2025-01-01")
    assert len(non_voting) > 0
    assert set(non_voting["MainType"].unique()) == {"X"}

    yr_chunks = list(
        load_cse_daily_prices(
            companies=["COMB"],
            start_date="2024-01-01",
            end_date="2025-12-31",
            incremental_by_year=True,
        )
    )
    assert len(yr_chunks) == 2
    assert sum(len(c) for c in yr_chunks) == len(
        load_cse_daily_prices(companies=["COMB"], start_date="2024-01-01", end_date="2025-12-31")
    )

    row_chunks = list(
        load_cse_daily_prices(companies=["JKH"], start_date="2024-01-01", chunk_size=100)
    )
    assert len(row_chunks) >= 2
    assert all(len(c) <= 100 for c in row_chunks)

    master = load_cse_security_master(active_status="Delisted")
    assert len(master) == 123
    ca = load_cse_corporate_actions(action_category="splits")
    assert len(ca) > 0
    cal = load_cse_market_calendar(start_date="2024-01-01", end_date="2024-12-31")
    assert len(cal) > 200


# ---------------------------------------------------------------------------
# 3 & 4. Security Identity Model: Share Class Preservation & Issuer Resolution
# ---------------------------------------------------------------------------
def test_03_share_class_preservation(phase1b_outputs: dict) -> None:
    master = phase1b_outputs["master"]
    assert set(master["MainType"].unique()) == {"N", "X", "R", "W", "P", "D", "B", "U"}

    comb_sids = set(master.loc[master["CompanyCode"] == "COMB", "SecurityID"])
    assert {"COMB-N-0000", "COMB-X-0000", "COMB-R-0000"}.issubset(comb_sids)

    jkh_sids = set(master.loc[master["CompanyCode"] == "JKH", "SecurityID"])
    assert {"JKH-N-0000", "JKH-R-0000", "JKH-W-0022", "JKH-W-0023"}.issubset(jkh_sids)


def test_04_issuer_identity_and_merger_ambiguity_flags(phase1b_outputs: dict) -> None:
    master = phase1b_outputs["master"]
    assert master["SecurityID"].is_unique
    assert master["IssuerID"].dropna().nunique() == 350
    assert int(master["ManualReviewRequired"].sum()) == 56

    mergers = master[master["IssuerValidationStatus"] == "MANUAL_REVIEW_MERGER_NAME_OVERLAP"]
    assert {"BFN", "ORIN", "CLC", "LOFC"} == set(mergers["CompanyCode"].unique())
    assert mergers["ManualReviewRequired"].all()

    unresolved = master[master["IssuerValidationStatus"] == "MANUAL_REVIEW_UNVERIFIED_ISSUER"]
    assert len(unresolved) == 50
    assert unresolved["IssuerID"].isna().all()
    assert unresolved["ManualReviewRequired"].all()


# ---------------------------------------------------------------------------
# 5. Verified Target Stock Splits & Pre/Post Continuity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sid,ex_date,expected_pf,expected_ratio",
    [
        ("JKH-N-0000", "2024-11-05", 0.10, 10.0),
        ("JKH-N-0000", "2016-06-30", 0.875, 8.0 / 7.0),
        ("JKH-N-0000", "2015-07-03", 0.875, 8.0 / 7.0),
        ("JKH-N-0000", "2011-06-30", 0.75, 4.0 / 3.0),
        ("LMF-N-0000", "2024-01-16", 0.10, 10.0),
        ("AFS-N-0000", "2024-01-24", 0.20, 5.0),
        ("SUN-N-0000", "2025-02-18", 0.25, 4.0),
        ("SUN-N-0000", "2021-03-31", 1.0 / 3.0, 3.0),
        ("WATA-N-0000", "2025-02-28", 0.20, 5.0),
        ("HHL-N-0000", "2025-05-06", 0.20, 5.0),
        ("CIC-N-0000", "2025-10-17", 0.20, 5.0),
        ("CIC-X-0000", "2025-10-17", 0.20, 5.0),
    ],
)
def test_05_verified_target_splits_and_continuity(
    phase1b_outputs: dict, sid: str, ex_date: str, expected_pf: float, expected_ratio: float
) -> None:
    fac = phase1b_outputs["factors"]
    adj = phase1b_outputs["split_adj"]
    dt = pd.Timestamp(ex_date)

    ev = fac[(fac["SecurityID"] == sid) & (fac["ExDate"] == dt) & (fac["EventType"] == "StockSplit")]
    assert len(ev) == 1
    assert math.isclose(ev["PriceFactor"].iloc[0], expected_pf, rel_tol=1e-5)
    assert math.isclose(ev["Ratio"].iloc[0], expected_ratio, rel_tol=1e-5)

    sub = adj[adj["SecurityID"] == sid].sort_values("Date")
    pre = sub[sub["Date"] < dt].iloc[-1]
    post = sub[sub["Date"] >= dt].iloc[0]

    # RawClose untouched; adjusted pre-ex Close aligns continuously vs raw jump
    assert pre["RawClose"] / pre["Close"] > 1.0
    rel_gap_adj = abs(pre["Close"] - post["Close"]) / pre["Close"]
    rel_gap_raw = abs(pre["RawClose"] - post["RawClose"]) / pre["RawClose"]
    assert rel_gap_adj < 0.25, f"{sid} {ex_date}: pre adj={pre['Close']:.2f} vs post={post['Close']:.2f}"
    assert rel_gap_adj < rel_gap_raw * 0.25


# ---------------------------------------------------------------------------
# 6, 7, 8. Bonus/Scrip, Rights TERP, Cash Dividend & Total Return Math
# ---------------------------------------------------------------------------
def test_06_bonus_and_scrip_math(phase1b_outputs: dict) -> None:
    fac = phase1b_outputs["factors"]
    bonus_vll = fac[
        (fac["SecurityID"] == "VLL-N-0000")
        & (fac["ExDate"] == pd.Timestamp("2015-09-23"))
        & (fac["EventType"] == "BonusIssue")
    ]
    assert len(bonus_vll) == 1
    # 1 new : 2 old -> ratio=0.5, PriceFactor = 2/(2+1) = 0.666667
    assert math.isclose(bonus_vll["Ratio"].iloc[0], 0.5, rel_tol=1e-5)
    assert math.isclose(bonus_vll["PriceFactor"].iloc[0], 2.0 / 3.0, rel_tol=1e-5)
    assert bonus_vll["DividendFactor"].iloc[0] == 1.0

    scrips = fac[fac["EventType"] == "ScripDividend"]
    assert len(scrips) == 165
    expected_pf = scrips["OldProportion"] / (scrips["OldProportion"] + scrips["NewProportion"])
    assert np.allclose(scrips["PriceFactor"], expected_pf, rtol=1e-6)
    assert (scrips["DividendFactor"] == 1.0).all()


def test_07_rights_terp_and_underwater_math(phase1b_outputs: dict) -> None:
    ca = phase1b_outputs["ca"]
    fac = phase1b_outputs["factors"]

    terp_rows = fac[(fac["EventType"] == "RightsIssue") & (fac["ValidationStatus"] == "VALIDATED_TERP")]
    assert len(terp_rows) == 210
    p_cum = terp_rows["PreExClose"]
    r = terp_rows["Ratio"]
    sub_p = terp_rows["SubscriptionPrice"]
    terp = (p_cum + r * sub_p) / (1.0 + r)
    assert np.allclose(terp_rows["PriceFactor"], terp / p_cum, rtol=1e-6)
    assert (terp_rows["PriceFactor"] > 0).all() and (terp_rows["PriceFactor"] < 1.0).all()

    underwater = ca[
        (ca["EventType"] == "RightsIssue") & (ca["ValidationStatus"] == "VALIDATED_NON_DILUTIVE_UNDERWATER")
    ]
    assert len(underwater) == 49
    assert (underwater["PriceFactor"] == 1.0).all()


def test_08_cash_dividend_total_return_separation(phase1b_outputs: dict) -> None:
    fac = phase1b_outputs["factors"]
    adj = phase1b_outputs["split_adj"]
    tr = phase1b_outputs["tr"]

    divs = fac[fac["EventType"] == "CashDividend"]
    assert len(divs) == 3887
    assert (divs["PriceFactor"] == 1.0).all()
    assert (divs["VolumeFactor"] == 1.0).all()
    expected_df = (divs["PreExClose"] - divs["DividendPerShare"]) / divs["PreExClose"]
    assert np.allclose(divs["DividendFactor"], expected_df, rtol=1e-6)

    # Split-adjusted Close NEVER includes cash dividends; TotalReturnPrice separates CashDiv cleanly
    assert np.allclose(tr["SplitBonusAdjClose"].dropna(), adj["Close"].dropna(), rtol=1e-9)
    assert np.allclose(
        tr["TotalReturnPrice"].dropna(),
        (tr["CapitalAdjClose"] * tr["CumCashDividendFactor"]).dropna(),
        rtol=1e-9,
    )
    # Historical pre-dividend prices in TotalReturnPrice are lower than CapitalAdjClose (dividend yield added going forward)
    jkh_early = tr[(tr["SecurityID"] == "JKH-N-0000") & (tr["Date"] < pd.Timestamp("2020-01-01"))].iloc[0]
    assert jkh_early["CumCashDividendFactor"] < 1.0
    assert jkh_early["TotalReturnPrice"] < jkh_early["CapitalAdjClose"]


# ---------------------------------------------------------------------------
# 9 & 10. Chronological Strictness (Date < ExDate) & Double-Adjustment Guard
# ---------------------------------------------------------------------------
def test_09_chronological_ex_date_boundary_and_cumulative_product(phase1b_outputs: dict) -> None:
    adj = phase1b_outputs["split_adj"]
    jkh = adj[adj["SecurityID"] == "JKH-N-0000"].sort_values("Date")

    # On or after the latest 2024-11-05 split, CumSplitFactor == 1.0
    post_2024 = jkh[jkh["Date"] >= pd.Timestamp("2024-11-05")]
    assert np.allclose(post_2024["CumSplitFactor"], 1.0)
    assert np.allclose(post_2024["Close"], post_2024["RawClose"])

    # Between 2016-06-30 and 2024-11-05 (excluding ex-date), only the 2024 split (0.10) applies
    mid_win = jkh[(jkh["Date"] >= pd.Timestamp("2016-06-30")) & (jkh["Date"] < pd.Timestamp("2024-11-05"))]
    assert np.allclose(mid_win["CumSplitFactor"], 0.10)

    # Prior to 2011-06-30, all 4 JKH splits compound: 0.1 * 0.875 * 0.875 * 0.75
    early_jkh = jkh[jkh["Date"] < pd.Timestamp("2011-06-30")].iloc[-1]
    assert math.isclose(early_jkh["CumSplitFactor"], 0.1 * 0.875 * 0.875 * 0.75, rel_tol=1e-9)


def test_10_double_adjustment_prevention(phase1b_outputs: dict) -> None:
    ca = phase1b_outputs["ca"]
    adj = phase1b_outputs["split_adj"]

    assert adj["IsSplitAdjusted"].all()
    sample = adj[adj["SecurityID"] == "JKH-N-0000"].tail(10).copy()
    splits = ca[(ca["CompanyCode"] == "JKH") & (ca["EventType"] == "StockSplit")].copy()

    # Applying split adjustment a second time on already-adjusted prices must raise ValueError
    with pytest.raises(ValueError, match="already split-adjusted"):
        apply_split_adjustments(sample, splits)

    # Boundary-year duplicate events flagged with neutral 1.0 factors
    dups = ca[ca["ValidationStatus"] == "DUPLICATE_BOUNDARY_EVENT_SKIPPED"]
    assert len(dups) == 27
    assert (dups["PriceFactor"] == 1.0).all()
    assert (dups["DividendFactor"] == 1.0).all()

    # Synthetic multi-split + cash dividend compute test guarantees Date < ExDate boundary
    toy_prices = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "SecurityID": ["TEST-N-0000"] * 3,
        "CompanyCode": ["TEST"] * 3,
        "IssuerID": ["ISS_TEST"] * 3,
        "MainType": ["N"] * 3,
        "SubType": ["0000"] * 3,
        "Open": [100.0, 20.0, 21.0],
        "High": [102.0, 21.0, 22.0],
        "Low": [98.0, 19.0, 20.0],
        "Close": [100.0, 20.0, 21.0],
        "ShareVolume": [1000.0, 5000.0, 6000.0],
        "TradeVolume": [10.0, 20.0, 30.0],
        "Turnover": [100000.0, 100000.0, 126000.0],
        "Era": ["2021-2025"] * 3,
        "DataFormat": ["Phase1B_Test"] * 3,
        "DataQualityFlag": ["OK"] * 3,
    })
    toy_ca = pd.DataFrame({
        "EventID": ["CA_T1"],
        "SecurityID": ["TEST-N-0000"],
        "CompanyCode": ["TEST"],
        "EventType": ["StockSplit"],
        "AnnouncementDate": [pd.Timestamp("2023-12-15")],
        "ExDate": [pd.Timestamp("2024-01-02")],
        "ExDateBasis": ["EFFECTIVE_DATE"],
        "EffectiveDate": [pd.Timestamp("2024-01-02")],
        "OldProportion": [1.0],
        "NewProportion": [5.0],
        "Ratio": [5.0],
        "SubscriptionPrice": [np.nan],
        "DividendPerShare": [np.nan],
        "PreExClose": [100.0],
        "PriceFactor": [0.2],
        "VolumeFactor": [5.0],
        "DividendFactor": [1.0],
        "Source": ["test.xls"],
        "SourceSheet": ["2024"],
        "SourceRow": [5],
        "ValidationStatus": ["VALIDATED"],
        "ValidationNotes": ["unit test split"],
    })
    _, toy_adj, _ = compute_adjustment_factors_and_series(toy_prices, toy_ca)
    assert math.isclose(toy_adj.loc[0, "Close"], 20.0)
    assert math.isclose(toy_adj.loc[0, "ShareVolume"], 5000.0)
    assert math.isclose(toy_adj.loc[1, "Close"], 20.0)
    assert math.isclose(toy_adj.loc[1, "CumSplitFactor"], 1.0)


# ---------------------------------------------------------------------------
# 11. Date Validation & Incomplete Event Flagging
# ---------------------------------------------------------------------------
def test_11_date_parsing_and_unresolved_flags(phase1b_outputs: dict) -> None:
    assert parse_excel_date("2024-11-05") == pd.Timestamp("2024-11-05")
    assert parse_excel_date("06-05-2025") == pd.Timestamp("2025-05-06")  # CSE day-first convention
    assert parse_excel_date("22/05/2020") == pd.Timestamp("2020-05-22")
    assert pd.isna(parse_excel_date("-"))
    assert pd.isna(parse_excel_date("1900-01-01"))

    ca = phase1b_outputs["ca"]
    pct_divs = ca[ca["ValidationStatus"] == "UNRESOLVED_PERCENT_OF_PAR"]
    assert len(pct_divs) == 2787
    assert (pct_divs["DividendFactor"] == 1.0).all()

    missing_ex = ca[ca["ValidationStatus"] == "UNRESOLVED_MISSING_EX_DATE"]
    assert len(missing_ex) == 336
    assert (missing_ex["PriceFactor"] == 1.0).all()


# ---------------------------------------------------------------------------
# 12 & 13. Delisting Preservation & Sector Mapping Audit Integrity
# ---------------------------------------------------------------------------
def test_12_historical_listing_delisting_preserved(phase1b_outputs: dict) -> None:
    master = phase1b_outputs["master"]
    integrated = phase1b_outputs["integrated"]

    by_code = master.drop_duplicates("CompanyCode")["HistoricalStatus"].value_counts().to_dict()
    assert by_code == {"Listed": 277, "Delisted": 123}

    delisted_codes = set(master.loc[master["HistoricalStatus"] == "Delisted", "CompanyCode"])
    assert len(delisted_codes & set(integrated["CompanyCode"].unique())) == 123
    assert master.loc[master["HistoricalStatus"] == "Delisted", "DelistingDate"].notna().all()


def test_13_sector_mapping_no_future_leakage(phase1b_outputs: dict) -> None:
    sector = phase1b_outputs["sector"]
    master = phase1b_outputs["master"]

    assert len(sector) == len(master) == 719
    assert (sector["Sector"] == "Unknown").all()
    assert (sector["Industry"] == "Unknown").all()
    assert (sector["ValidationStatus"] == "MANUAL_RESEARCH_REQUIRED").all()
    assert sector["ManualResearchQueue"].all()
    assert (sector["Confidence"] == 0.0).all()
    assert sector["EffectiveFrom"].notna().all()


# ---------------------------------------------------------------------------
# 14. Traceability & Phase 1A Immutability
# ---------------------------------------------------------------------------
def test_14_factor_traceability_and_phase1a_immutability(phase1b_outputs: dict) -> None:
    fac = phase1b_outputs["factors"]
    integrated = phase1b_outputs["integrated"]
    p1_val = pd.read_parquet(PHASE1A_DIR / "cse_daily_prices_validated.parquet")

    assert len(fac) == 4443
    assert fac["EventID"].is_unique
    assert fac["Source"].notna().all()
    assert fac["SourceSheet"].notna().all()
    assert (fac["SourceRow"] > 0).all()
    assert (fac["PriceFactor"] > 0).all() and (fac["PriceFactor"] <= 1.0).all()
    assert (fac["DividendFactor"] > 0).all() and (fac["DividendFactor"] <= 1.0).all()

    # Exact row-for-row raw close preservation vs Phase 1A validated dataset
    assert len(p1_val) == 1_414_977
    assert len(integrated) == len(p1_val)
    assert np.array_equal(
        p1_val["Close"].fillna(-1.0).to_numpy(),
        integrated["Close"].fillna(-1.0).to_numpy(),
    )
