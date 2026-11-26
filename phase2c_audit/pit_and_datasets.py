"""
Phase 2C Workstreams G & H + Visualizations
=========================================
Implements:
  - Workstream G: Point-in-time historical quality & eligibility flags across
    all 1,414,977 rows (`point_in_time_quality_flags.parquet`).
  - Workstream H: Strategy-specific PIT research universes for Short-term,
    Medium-term, and Long-term horizons preserving all historical observations
    with deterministic filtering flags and explicit unavailability disclosures.
  - Phase 2C Forensic & Readiness Visualization Suite (5 publication charts).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _assert_aligned(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"Row count mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)}).")
    if not (
        (a["SecurityID"].to_numpy() == b["SecurityID"].to_numpy()).all()
        and (pd.to_datetime(a["Date"]).to_numpy() == pd.to_datetime(b["Date"]).to_numpy()).all()
    ):
        raise ValueError(f"Key alignment mismatch between {name_a} and {name_b} after sort.")


def build_pit_quality_flags(
    df_adj: pd.DataFrame,
    df_raw: pd.DataFrame,
    ca_validated: pd.DataFrame,
    lc_df: pd.DataFrame,
    ca_rem_df: pd.DataFrame,
    anom_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    elig_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Workstream G: Construct per-observation point-in-time quality, validity,
    regime availability, and strategy eligibility flags across all 1,414,977 rows.
    """
    adj = df_adj.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    raw = df_raw.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    beta = beta_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    elig = elig_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)

    _assert_aligned(adj, raw, "df_adj", "df_raw")
    _assert_aligned(adj, beta, "df_adj", "beta_df")
    _assert_aligned(adj, elig, "df_adj", "elig_df")

    n = len(adj)
    sids = adj["SecurityID"].astype(str).to_numpy()
    mtypes = adj["MainType"].astype(str).to_numpy()
    dates = pd.to_datetime(adj["Date"]).dt.normalize()
    dates_ns = dates.astype("datetime64[ns]").astype("int64").to_numpy()
    raw_open = pd.to_numeric(adj["RawOpen"], errors="coerce").to_numpy(dtype=np.float64)
    raw_high = pd.to_numeric(adj["RawHigh"], errors="coerce").to_numpy(dtype=np.float64)
    raw_low = pd.to_numeric(adj["RawLow"], errors="coerce").to_numpy(dtype=np.float64)
    raw_close = pd.to_numeric(adj["RawClose"], errors="coerce").to_numpy(dtype=np.float64)
    share_vol = pd.to_numeric(adj["ShareVolume"], errors="coerce").to_numpy(dtype=np.float64)
    data_fmt = adj["DataFormat"].astype(str).to_numpy()
    dq_flag = adj["DataQualityFlag"].astype(str).to_numpy()

    # ── 1. Lifecycle lookup per SecurityID ───────────────────────────────────
    lc_map = lc_df.set_index("SecurityID")
    rem_status_arr = pd.Series(sids).map(lc_map["RemediatedStatus"]).astype(str).to_numpy()
    lc_val_status_arr = pd.Series(sids).map(lc_map["LifecycleValidationStatus"]).astype(str).to_numpy()
    rem_del_dt = pd.Series(sids).map(lc_map["RemediatedDelistingDate"])
    rem_del_ns = pd.to_datetime(rem_del_dt).dt.normalize().astype("datetime64[ns]").astype("int64").to_numpy()
    fp_ns = pd.to_datetime(pd.Series(sids).map(lc_map["FirstPriceDate"])).dt.normalize().astype("datetime64[ns]").astype("int64").to_numpy()

    nat_i64 = np.iinfo(np.int64).min
    after_remediated_delisting = (
        (rem_status_arr == "Delisted")
        & (rem_del_ns != nat_i64)
        & (dates_ns > rem_del_ns)
    )
    is_hist_listed = (dates_ns >= fp_ns) & (~after_remediated_delisting)
    is_lifecycle_valid = np.isin(lc_val_status_arr, ["VERIFIED", "CORRECTED_WITH_EVIDENCE"])
    not_unresolved_issuer = lc_val_status_arr != "UNRESOLVED"

    # ── 2. Price data validity (SELI 2013, ZeroClose, OHLC inversions, spike errors) ──
    bad_ohlc = (
        (np.isfinite(raw_high) & np.isfinite(raw_low) & (raw_high < raw_low))
        | (np.isfinite(raw_open) & (raw_open <= 0))
        | (np.isfinite(raw_high) & (raw_high <= 0))
        | (np.isfinite(raw_low) & (raw_low <= 0))
    )
    bad_close = (~np.isfinite(raw_close)) | (raw_close <= 0.0)

    # Mark exact (SecurityID, Date) anomaly quarantines from Workstream C
    bad_anom_keys: set[tuple[str, int]] = set()
    if not anom_df.empty:
        err_sub = anom_df[anom_df["AnomalyClassification"].isin(["SECURITY_IDENTITY_ISSUE", "SOURCE_DATA_ERROR"])]
        for _, ar in err_sub.iterrows():
            d_i64 = pd.Timestamp(ar["Date"]).normalize().value
            bad_anom_keys.add((str(ar["SecurityID"]), int(d_i64)))

    row_keys = list(zip(sids.tolist(), dates_ns.tolist()))
    is_anom_quarantined = np.fromiter((k in bad_anom_keys for k in row_keys), dtype=bool, count=n)
    is_price_valid = (~bad_close) & (~bad_ohlc) & (~is_anom_quarantined)

    # Also double check against df_raw Close<=0 preservation
    raw_c_orig = pd.to_numeric(raw["Close"], errors="coerce").to_numpy(dtype=np.float64)
    is_price_valid &= (~np.isfinite(raw_c_orig) | (raw_c_orig > 0.0))

    # ── 3. Corporate Action Validity Mask around Unadjusted / Offset Windows ─
    # Mark transition bar +/- 2 trading observations around:
    #  a) VERIFIED_CORRECTION_REQUIRED (e.g. BUKI 2010-12-23..24, MERC 2025-12-17, omitted splits)
    #  b) QUARANTINED Rights/Bonus AllotmentDate offsets
    ca_mask_targets: dict[str, list[int]] = {}
    ca_bad = ca_rem_df[
        ca_rem_df["RemediationDisposition"].isin(["VERIFIED_CORRECTION_REQUIRED", "QUARANTINED"])
        & ca_rem_df["SecurityID"].notna()
        & ca_rem_df["ExDate"].notna()
    ]
    for _, cr in ca_bad.iterrows():
        ca_mask_targets.setdefault(str(cr["SecurityID"]), []).append(
            int(pd.Timestamp(cr["ExDate"]).normalize().value)
        )

    is_ca_valid = np.ones(n, dtype=bool)
    # Build contiguous slices per SecurityID in sorted order
    change_idx = np.flatnonzero(sids[1:] != sids[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [n]))
    for lo, hi in zip(starts, ends):
        sid = sids[lo]
        if sid not in ca_mask_targets:
            continue
        sec_dts = dates_ns[lo:hi]
        for target_ns in ca_mask_targets[sid]:
            pos = int(np.searchsorted(sec_dts, target_ns, side="left"))
            w_lo = max(0, pos - 2)
            w_hi = min(hi - lo, pos + 3)
            is_ca_valid[lo + w_lo : lo + w_hi] = False

    # ── 4. Dividend resolution completeness per (SecurityID, Date) for Long-Term ──
    unresolved_div = ca_validated[
        (ca_validated["EventType"] == "CashDividend")
        & ca_validated["ValidationStatus"].astype(str).str.startswith("UNRESOLVED")
        & ca_validated["SecurityID"].notna()
        & ca_validated["ExDate"].notna()
    ]
    max_unres_div_ns = (
        unresolved_div.assign(ExDate=pd.to_datetime(unresolved_div["ExDate"]).dt.normalize())
        .groupby("SecurityID")["ExDate"]
        .max()
    )
    row_max_unres_ns = pd.to_datetime(pd.Series(sids).map(max_unres_div_ns)).astype("datetime64[ns]").astype("int64").to_numpy()
    no_future_unres_div = (row_max_unres_ns == nat_i64) | (dates_ns > row_max_unres_ns)
    is_post_2008 = dates_ns >= pd.Timestamp("2008-01-01").value
    is_div_clean = is_post_2008 & no_future_unres_div

    # ── 5. Regimes & Contextual Unavailability Flags ─────────────────────────
    is_tradable = (
        is_hist_listed
        & is_price_valid
        & (mtypes == "N")
        & ((data_fmt == "close_only") | (np.isfinite(share_vol) & (share_vol > 0.0)))
    )
    is_beta_rel = np.isin(beta["beta_reliability_flag"].astype(str).to_numpy(), ["RELIABLE", "LIMITED_HISTORY"])
    is_open_avail = (dates_ns >= pd.Timestamp("2017-01-01").value) & np.isfinite(raw_open) & (raw_open > 0.0)
    is_hl_avail = np.isfinite(raw_high) & np.isfinite(raw_low) & (raw_high > 0.0) & (raw_low > 0.0) & (raw_high >= raw_low)
    is_sector_avail = np.zeros(n, dtype=bool)       # Explicitly 0% PIT sector history available
    is_fundamental_avail = np.zeros(n, dtype=bool)  # Explicitly 0% PIT fundamental data available

    st_elig = (
        elig["ShortTermEligible"].to_numpy(dtype=bool)
        & is_hist_listed
        & is_tradable
        & is_price_valid
        & is_ca_valid
        & not_unresolved_issuer
    )
    mt_elig = (
        elig["MediumTermEligible"].to_numpy(dtype=bool)
        & is_hist_listed
        & is_tradable
        & is_price_valid
        & is_ca_valid
        & not_unresolved_issuer
    )
    lt_elig = (
        elig["LongTermEligible"].to_numpy(dtype=bool)
        & is_hist_listed
        & is_tradable
        & is_price_valid
        & is_ca_valid
        & not_unresolved_issuer
    )
    model_elig = st_elig | mt_elig | lt_elig

    pit_df = pd.DataFrame({
        "SecurityID": adj["SecurityID"],
        "CompanyCode": adj["CompanyCode"],
        "Date": adj["Date"],
        "MainType": adj["MainType"],
        "SubType": adj["SubType"],
        "IsHistoricallyListed": is_hist_listed,
        "IsHistoricallyTradable": is_tradable,
        "IsPriceDataValid": is_price_valid,
        "IsCorporateActionValidated": is_ca_valid,
        "IsLifecycleValidated": is_lifecycle_valid,
        "IsDividendResolutionClean": is_div_clean,
        "IsBetaReliable": is_beta_rel,
        "IsOpenFeatureAvailable": is_open_avail,
        "IsHLFeatureAvailable": is_hl_avail,
        "IsSectorMappingAvailable": is_sector_avail,
        "IsFundamentalDataAvailable": is_fundamental_avail,
        "IsShortTermEligible": st_elig,
        "IsMediumTermEligible": mt_elig,
        "IsLongTermEligible": lt_elig,
        "IsModelTrainingEligible": model_elig,
    })

    bool_cols = [c for c in pit_df.columns if c.startswith("Is")]
    flag_counts = {c: int(pit_df[c].sum()) for c in bool_cols}
    flag_pcts = {c: round(100.0 * float(pit_df[c].mean()), 2) for c in bool_cols}

    # Specific forensic spot-checks for audit summary
    jins_mask = (sids == "JINS-N-0000") & (dates_ns > pd.Timestamp("2019-11-20").value)
    seli_2013_mask = (sids == "SELI-N-0000") & (dates_ns >= pd.Timestamp("2013-01-01").value) & (dates_ns <= pd.Timestamp("2013-12-31").value) & (raw_close <= 5.0)
    buki_dec10_mask = (sids == "BUKI-N-0000") & (dates_ns == pd.Timestamp("2010-12-24").value)
    merc_dec25_mask = (sids == "MERC-N-0000") & (dates_ns == pd.Timestamp("2025-12-17").value)

    summary = {
        "total_rows": int(n),
        "flag_true_counts": flag_counts,
        "flag_true_pcts": flag_pcts,
        "spot_checks": {
            "jins_post_2019_rows": int(jins_mask.sum()),
            "jins_post_2019_historically_listed": int(is_hist_listed[jins_mask].sum()),
            "jins_post_2019_lifecycle_valid": int(is_lifecycle_valid[jins_mask].sum()),
            "seli_2013_rows": int(seli_2013_mask.sum()),
            "seli_2013_price_valid_count": int(is_price_valid[seli_2013_mask].sum()),
            "buki_2010_12_24_ca_valid_count": int(is_ca_valid[buki_dec10_mask].sum()),
            "merc_2025_12_17_ca_valid_count": int(is_ca_valid[merc_dec25_mask].sum()),
        },
    }
    return pit_df, summary


def build_strategy_research_datasets(
    out_dir: Path,
    df_adj: pd.DataFrame,
    tech_df: pd.DataFrame,
    pv_df: pd.DataFrame,
    mr_df: pd.DataFrame,
    tr_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    elig_df: pd.DataFrame,
    pit_flags_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Workstream H: Build horizon-specific PIT research datasets without duplicating
    60-column frames or leaking post-snapshot sectors/fundamentals:
      1. short_term_research_universe.parquet (0-3 months)
      2. medium_term_research_universe.parquet (3-12 months)
      3. long_term_research_universe.parquet (1-5+ years)
    Preserves all 1,414,977 rows with explicit eligibility & regime flags.
    """
    adj = df_adj.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    tech = tech_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    pv = pv_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    mr = mr_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    tr = tr_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    beta = beta_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    elig = elig_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    pit = pit_flags_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)

    _assert_aligned(adj, tech, "df_adj", "tech_df")
    _assert_aligned(adj, pv, "df_adj", "pv_df")
    _assert_aligned(adj, mr, "df_adj", "mr_df")
    _assert_aligned(adj, tr, "df_adj", "tr_df")
    _assert_aligned(adj, beta, "df_adj", "beta_df")
    _assert_aligned(adj, elig, "df_adj", "elig_df")
    _assert_aligned(adj, pit, "df_adj", "pit_flags_df")

    base_keys: dict[str, Any] = {
        "Date": adj["Date"],
        "SecurityID": adj["SecurityID"],
        "CompanyCode": adj["CompanyCode"],
        "IssuerID": adj["IssuerID"],
        "MainType": adj["MainType"],
        "SubType": adj["SubType"],
        "Era": adj["Era"],
        "DataFormat": adj["DataFormat"],
        "DataQualityFlag": adj["DataQualityFlag"],
    }

    summaries: dict[str, Any] = {}

    # ── 1. Short-Term Research Universe (0-3 months) ─────────────────────────
    # Excludes 100%-NaN Sector_MOM_20/RS_Sector_ROC_20 & affine duplicate Williams_R_14
    st_dict = dict(base_keys)
    for c in ["RawOpen", "RawHigh", "RawLow", "RawClose", "Close", "CapitalAdjClose", "ShareVolume", "Turnover"]:
        st_dict[c] = adj[c]
    for c in [
        "Daily_Return_SplitAdj", "Daily_Return_CapAdj", "OBV", "MFI_14", "CMF_20",
        "VWAP_20", "Volume_SMA_20", "Volume_Ratio_20", "AD_Line", "Chaikin_Osc_3_10",
        "Amihud_Illiquidity_20", "Overnight_Gap_Ret", "Intraday_Ret",
    ]:
        st_dict[c] = pv[c]
    for c in [
        "SMA_20", "SMA_50", "EMA_12", "EMA_26", "WMA_20", "ADX_14", "Aroon_Up_25", "Aroon_Down_25",
        "Ichimoku_TK_Spread_Pct", "Ichimoku_Kumo_Pos_Pct", "Supertrend_10_3", "Supertrend_Dir_10_3",
        "Supertrend_Reversal", "RSI_14", "MACD_Line", "MACD_Signal", "MACD_Hist",
        "ROC_10", "ROC_20", "CCI_20", "Stoch_K_14", "Stoch_D_3", "MOM_10",
        "ATR_14", "NATR_14", "BB_Upper_20_2", "BB_Lower_20_2", "BB_Width_20_2", "BB_PctB_20_2",
        "Keltner_Pos_20", "Hist_Vol_21d", "Donchian_Pos_20", "PSAR", "PSAR_Trend",
        "PSAR_Distance_Pct", "PSAR_Reversal", "MA_Cross_Spread_20_50", "MA_Cross_Signal_20_50",
        "RSI_Div_Bullish_14", "RSI_Div_Bearish_14",
    ]:
        st_dict[c] = tech[c]
    for c in ["ASPI_Close", "ASPI_MOM_20", "RS_ASPI_ROC_20", "Beta_ASPI_60", "Sector_Breadth_MOM_20"]:
        st_dict[c] = mr[c]
    for c in [
        "Rolling_ActiveDays_20", "Rolling_ActiveDays_60", "Rolling_TradingFreqPct_60",
        "Rolling_MeanTurnover_20_LKR", "Rolling_MedianTurnover_60_LKR", "ConsecutiveNonActiveDays", "LiquidityStatus",
    ]:
        st_dict[c] = elig[c]
    for c in ["beta60_observation_count", "beta60_nonzero_return_obs", "beta_reliability_flag", "beta_validity_status"]:
        st_dict[c] = beta[c]
    for c in [
        "IsHistoricallyListed", "IsHistoricallyTradable", "IsPriceDataValid",
        "IsCorporateActionValidated", "IsLifecycleValidated", "IsBetaReliable",
        "IsOpenFeatureAvailable", "IsHLFeatureAvailable", "IsSectorMappingAvailable",
        "IsFundamentalDataAvailable", "IsShortTermEligible", "IsModelTrainingEligible",
    ]:
        st_dict[c] = pit[c]

    st_df = pd.DataFrame(st_dict)
    st_path = out_dir / "short_term_research_universe.parquet"
    st_df.to_parquet(st_path, index=False, compression="snappy")
    st_mask = st_df["IsShortTermEligible"].to_numpy(dtype=bool)
    st_2017 = st_mask & st_df["IsOpenFeatureAvailable"].to_numpy(dtype=bool)
    summaries["short_term"] = {
        "path": str(st_path.relative_to(out_dir.parents[2])).replace("\\", "/"),
        "rows": int(len(st_df)),
        "columns": int(st_df.shape[1]),
        "eligible_rows": int(st_mask.sum()),
        "eligible_rows_2017plus_ohlcv": int(st_2017.sum()),
        "eligible_securities_ever": int(st_df.loc[st_mask, "SecurityID"].nunique()),
        "excluded_columns_documented": ["Williams_R_14 (duplicate of Stoch_K_14)", "Sector_MOM_20 (100% NaN)", "RS_Sector_ROC_20 (100% NaN)"],
    }
    del st_df, st_dict
    gc.collect()

    # ── 2. Medium-Term Research Universe (3-12 months) ───────────────────────
    mt_dict = dict(base_keys)
    for c in ["RawHigh", "RawLow", "RawClose", "Close", "CapitalAdjClose", "ShareVolume", "Turnover"]:
        mt_dict[c] = adj[c]
    for c in [
        "Daily_Return_CapAdj", "OBV", "CMF_20", "Volume_Ratio_20", "AD_Line", "Amihud_Illiquidity_20",
    ]:
        mt_dict[c] = pv[c]
    for c in [
        "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "ADX_14",
        "Ichimoku_TK_Spread_Pct", "Ichimoku_Kumo_Pos_Pct", "Supertrend_10_3", "Supertrend_Dir_10_3",
        "RSI_14", "MACD_Line", "MACD_Signal", "MACD_Hist", "ROC_20", "NATR_14",
        "BB_Width_20_2", "BB_PctB_20_2", "Hist_Vol_21d", "MA_Cross_Spread_20_50", "MA_Cross_Signal_20_50",
    ]:
        mt_dict[c] = tech[c]
    for c in [
        "ASPI_Close", "ASPI_MOM_20", "ASPI_MOM_60", "RS_ASPI_Level",
        "RS_ASPI_ROC_20", "RS_ASPI_ROC_60", "Beta_ASPI_60", "Beta_ASPI_252", "Sector_Breadth_MOM_20",
    ]:
        mt_dict[c] = mr[c]
    for c in [
        "TotalCausalObsToDate", "Rolling_ActiveDays_60", "Rolling_ActiveDays_120",
        "Rolling_TradingFreqPct_60", "Rolling_MeanTurnover_60_LKR",
        "Rolling_MedianTurnover_60_LKR", "MaxGapCalSessions_60", "LiquidityStatus",
    ]:
        mt_dict[c] = elig[c]
    for c in [
        "beta60_observation_count", "beta60_nonzero_return_obs",
        "beta252_observation_count", "beta252_nonzero_return_obs",
        "beta_reliability_flag", "beta_validity_status",
    ]:
        mt_dict[c] = beta[c]
    for c in [
        "IsHistoricallyListed", "IsHistoricallyTradable", "IsPriceDataValid",
        "IsCorporateActionValidated", "IsLifecycleValidated", "IsBetaReliable",
        "IsHLFeatureAvailable", "IsSectorMappingAvailable",
        "IsFundamentalDataAvailable", "IsMediumTermEligible", "IsModelTrainingEligible",
    ]:
        mt_dict[c] = pit[c]

    mt_df = pd.DataFrame(mt_dict)
    mt_path = out_dir / "medium_term_research_universe.parquet"
    mt_df.to_parquet(mt_path, index=False, compression="snappy")
    mt_mask = mt_df["IsMediumTermEligible"].to_numpy(dtype=bool)
    summaries["medium_term"] = {
        "path": str(mt_path.relative_to(out_dir.parents[2])).replace("\\", "/"),
        "rows": int(len(mt_df)),
        "columns": int(mt_df.shape[1]),
        "eligible_rows": int(mt_mask.sum()),
        "eligible_securities_ever": int(mt_df.loc[mt_mask, "SecurityID"].nunique()),
        "pit_limitations_flagged": ["IsSectorMappingAvailable=False", "IsFundamentalDataAvailable=False"],
    }
    del mt_df, mt_dict
    gc.collect()

    # ── 3. Long-Term Research Universe (1-5+ years) ──────────────────────────
    lt_dict = dict(base_keys)
    for c in ["RawClose", "Close", "CapitalAdjClose", "ShareVolume", "Turnover"]:
        lt_dict[c] = adj[c]
    for c in [
        "TotalReturnPrice", "TotalReturnIndex", "CumCapitalFactor",
        "CumCashDividendFactor", "CumTotalReturnFactor",
    ]:
        lt_dict[c] = tr[c]
    lt_dict["Daily_Return_CapAdj"] = pv["Daily_Return_CapAdj"]
    lt_dict["Amihud_Illiquidity_20"] = pv["Amihud_Illiquidity_20"]
    for c in ["SMA_50", "SMA_200", "MA_Cross_Spread_20_50", "ROC_20", "Hist_Vol_21d", "NATR_14"]:
        lt_dict[c] = tech[c]
    for c in ["ASPI_Close", "ASPI_MOM_60", "RS_ASPI_Level", "RS_ASPI_ROC_60", "Beta_ASPI_252"]:
        lt_dict[c] = mr[c]
    for c in [
        "TotalCausalObsToDate", "Rolling_ActiveDays_120", "Rolling_TradingFreqPct_60",
        "Rolling_MedianTurnover_120_LKR", "LiquidityStatus",
    ]:
        lt_dict[c] = elig[c]
    for c in ["beta252_observation_count", "beta252_nonzero_return_obs", "beta_reliability_flag", "beta_validity_status"]:
        lt_dict[c] = beta[c]
    for c in [
        "IsHistoricallyListed", "IsHistoricallyTradable", "IsPriceDataValid",
        "IsCorporateActionValidated", "IsLifecycleValidated", "IsDividendResolutionClean",
        "IsBetaReliable", "IsSectorMappingAvailable", "IsFundamentalDataAvailable",
        "IsLongTermEligible", "IsModelTrainingEligible",
    ]:
        lt_dict[c] = pit[c]

    lt_div_clean = pit["IsDividendResolutionClean"].to_numpy(dtype=bool)
    lt_dict["LongTermResearchReadinessNote"] = np.where(
        lt_div_clean,
        "POST2008_DIV_CLEAN; REQUIRES_FUNDAMENTALS_AND_PIT_SECTOR_IN_PHASE3",
        "PRE2008_OR_UNRESOLVED_DIVIDEND_LIMITATION; REQUIRES_FUNDAMENTALS_AND_PIT_SECTOR",
    )

    lt_df = pd.DataFrame(lt_dict)
    lt_path = out_dir / "long_term_research_universe.parquet"
    lt_df.to_parquet(lt_path, index=False, compression="snappy")
    lt_mask = lt_df["IsLongTermEligible"].to_numpy(dtype=bool)
    lt_div_elig = lt_mask & lt_df["IsDividendResolutionClean"].to_numpy(dtype=bool)
    summaries["long_term"] = {
        "path": str(lt_path.relative_to(out_dir.parents[2])).replace("\\", "/"),
        "rows": int(len(lt_df)),
        "columns": int(lt_df.shape[1]),
        "eligible_rows": int(lt_mask.sum()),
        "eligible_rows_with_clean_dividends": int(lt_div_elig.sum()),
        "eligible_securities_ever": int(lt_df.loc[lt_mask, "SecurityID"].nunique()),
        "pit_limitations_flagged": [
            "IsFundamentalDataAvailable=False",
            "IsSectorMappingAvailable=False",
            "IsDividendResolutionClean flags 2,787 pre-2008 par-value & unresolved cash dividends",
        ],
    }
    del lt_df, lt_dict
    gc.collect()

    return summaries


def generate_phase2c_visualizations(
    viz_dir: Path,
    lc_df: pd.DataFrame,
    ca_rem_df: pd.DataFrame,
    anom_df: pd.DataFrame,
    df_adj: pd.DataFrame,
    beta_df: pd.DataFrame,
    elig_df: pd.DataFrame,
    pit_flags_df: pd.DataFrame,
    red_2c_df: pd.DataFrame,
    raw_dir: Path,
) -> list[str]:
    """
    Create 5 forensic and readiness charts in `reports/phase2/visualizations/phase2c/`.
    """
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#F8FAFC",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.size": 9.5,
    })
    generated: list[str] = []

    # ── Chart 1: Lifecycle & CA Remediation Outcomes ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    lc_cnts = lc_df["LifecycleValidationStatus"].value_counts()
    lc_colors = {"VERIFIED": "#16A34A", "QUARANTINED": "#D97706", "UNRESOLVED": "#DC2626", "CORRECTED_WITH_EVIDENCE": "#2563EB"}
    bars1 = axes[0].bar(lc_cnts.index, lc_cnts.values, color=[lc_colors.get(k, "#64748B") for k in lc_cnts.index])
    axes[0].set_title("Workstream A: Security Lifecycle Dispositions (719 Securities)", fontweight="bold")
    axes[0].set_ylabel("SecurityID Count")
    axes[0].tick_params(axis="x", rotation=18)
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 8, f"{int(h)}", ha="center", va="bottom", fontweight="bold")

    ca_cnts = ca_rem_df["RemediationDisposition"].value_counts()
    ca_colors = {"VERIFIED_CORRECT": "#16A34A", "UNRESOLVED": "#64748B", "QUARANTINED": "#D97706", "VERIFIED_CORRECTION_REQUIRED": "#DC2626"}
    bars2 = axes[1].bar(ca_cnts.index, ca_cnts.values, color=[ca_colors.get(k, "#64748B") for k in ca_cnts.index])
    axes[1].set_yscale("log")
    axes[1].set_ylim(1, max(ca_cnts.values) * 3)
    axes[1].set_title("Workstream B: Corporate Action Dispositions (Log Scale)", fontweight="bold")
    axes[1].set_ylabel("Event Count (Log10)")
    axes[1].tick_params(axis="x", rotation=18)
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h * 1.15, f"{int(h):,}", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    p1 = viz_dir / "01_lifecycle_and_ca_remediation.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p1))

    # ── Chart 2: Price Anomalies Breakdown & SELI-2013 Collision Timeline ────
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    ac_cnts = anom_df["AnomalyClassification"].value_counts().sort_values()
    ac_colors = {
        "CORPORATE_ACTION_EXPLAINED": "#2563EB",
        "POSSIBLE_MARKET_EVENT": "#0891B2",
        "SECURITY_IDENTITY_ISSUE": "#9333EA",
        "SOURCE_DATA_ERROR": "#DC2626",
        "UNRESOLVED": "#D97706",
    }
    bars = axes[0].barh(ac_cnts.index, ac_cnts.values, color=[ac_colors.get(k, "#64748B") for k in ac_cnts.index])
    axes[0].set_title("Workstream C: Extreme Price Anomaly Classifications", fontweight="bold")
    axes[0].set_xlabel("Flagged Anomaly Bars")
    for b in bars:
        w = b.get_width()
        axes[0].text(w + max(ac_cnts.values) * 0.015, b.get_y() + b.get_height() / 2, f"{int(w):,}", va="center", fontweight="bold")
    axes[0].set_xlim(0, max(ac_cnts.values) * 1.18)

    seli = df_adj[(df_adj["SecurityID"] == "SELI-N-0000") & (df_adj["Date"] >= "2011-01-01") & (df_adj["Date"] <= "2015-12-31")].sort_values("Date")
    if not seli.empty:
        axes[1].plot(pd.to_datetime(seli["Date"]), seli["RawClose"], color="#1E293B", linewidth=1.2, label="RawClose (LKR)")
        axes[1].axvspan(pd.Timestamp("2013-01-01"), pd.Timestamp("2013-12-31"), color="#EF4444", alpha=0.18, label="Quarantined 2013 SEMB Collision (242 rows)")
        axes[1].set_title("Forensic Proof: SELI-N-0000 2013 Source Identity Collision", fontweight="bold")
        axes[1].set_ylabel("Raw Close (LKR)")
        axes[1].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    p2 = viz_dir / "02_price_anomalies_and_seli_2013.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p2))

    # ── Chart 3: Beta Reliability Distribution & Official 2025Q4 Agreement ───
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    b_cnts = beta_df["beta_reliability_flag"].value_counts()
    b_cols = {"RELIABLE": "#16A34A", "LIMITED_HISTORY": "#3B82F6", "THIN_TRADING": "#D97706", "INSUFFICIENT_DATA": "#94A3B8", "UNSTABLE": "#DC2626"}
    bars_b = axes[0].bar(b_cnts.index, b_cnts.values / 1000.0, color=[b_cols.get(k, "#64748B") for k in b_cnts.index])
    axes[0].set_title("Workstream D: Rolling Beta Reliability Tiers (1.415M Rows)", fontweight="bold")
    axes[0].set_ylabel("Observations (Thousands)")
    axes[0].tick_params(axis="x", rotation=18)
    for b in bars_b:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 8, f"{h:.0f}k", ha="center", va="bottom", fontweight="bold")

    from phase2b_audit.market_ca_redundancy import _parse_cse_official_beta
    off = _parse_cse_official_beta(raw_dir, "2025Q4")
    if not off.empty:
        dec25 = beta_df[(beta_df["MainType"] == "N") & (beta_df["Date"] >= "2025-12-01") & (beta_df["Date"] <= "2025-12-31")]
        latest = dec25.sort_values("Date").groupby("CompanyCode").last().reset_index()
        m = off.merge(latest, on="CompanyCode", how="inner").dropna(subset=["Official_ASPI_Beta", "Beta_ASPI_60"])
        for tier in ["RELIABLE", "LIMITED_HISTORY", "THIN_TRADING", "UNSTABLE"]:
            sub = m[m["beta_reliability_flag"] == tier]
            if not sub.empty:
                axes[1].scatter(sub["Official_ASPI_Beta"], sub["Beta_ASPI_60"].clip(-4, 4), s=22, alpha=0.75, label=f"{tier} (n={len(sub)})", color=b_cols.get(tier, "#64748B"))
        axes[1].plot([-2, 3], [-2, 3], "k--", linewidth=1.0, alpha=0.6, label="45 deg parity")
        axes[1].set_xlim(-1.5, 3.0)
        axes[1].set_ylim(-2.5, 4.2)
        axes[1].set_title("CSE Official 2025Q4 vs Calc Beta_ASPI_60 by Reliability", fontweight="bold")
        axes[1].set_xlabel("Official CSE 2025Q4 ASPI Beta")
        axes[1].set_ylabel("Phase 2A Beta_ASPI_60 (clipped view +-4)")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    p3 = viz_dir / "03_beta_reliability_and_official_validation.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p3))

    # ── Chart 4: Yearly Eligible Universe by Investment Horizon (1997-2025) ─
    df_y = pd.DataFrame({
        "Year": pd.to_datetime(pit_flags_df["Date"]).dt.year,
        "SecurityID": pit_flags_df["SecurityID"],
        "MainType": pit_flags_df["MainType"],
        "ListedN": pit_flags_df["IsHistoricallyListed"] & (pit_flags_df["MainType"] == "N"),
        "ST": pit_flags_df["IsShortTermEligible"],
        "MT": pit_flags_df["IsMediumTermEligible"],
        "LT": pit_flags_df["IsLongTermEligible"],
    })
    df_y = df_y[df_y["Year"] >= 1997]
    yr_agg = pd.DataFrame({
        "Listed Ordinary N": df_y[df_y["ListedN"]].groupby("Year")["SecurityID"].nunique(),
        "Long-Term Eligible": df_y[df_y["LT"]].groupby("Year")["SecurityID"].nunique(),
        "Medium-Term Eligible": df_y[df_y["MT"]].groupby("Year")["SecurityID"].nunique(),
        "Short-Term Eligible": df_y[df_y["ST"]].groupby("Year")["SecurityID"].nunique(),
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.plot(yr_agg.index, yr_agg["Listed Ordinary N"], color="#475569", linewidth=2.0, linestyle="--", label="Historically Listed Ordinary N")
    ax.plot(yr_agg.index, yr_agg["Long-Term Eligible"], color="#16A34A", linewidth=2.0, marker="o", markersize=3.5, label="Long-Term PIT Eligible")
    ax.plot(yr_agg.index, yr_agg["Medium-Term Eligible"], color="#2563EB", linewidth=2.0, marker="s", markersize=3.5, label="Medium-Term PIT Eligible")
    ax.plot(yr_agg.index, yr_agg["Short-Term Eligible"], color="#DC2626", linewidth=2.0, marker="^", markersize=3.5, label="Short-Term PIT Eligible")
    for xline, txt in [(2001, "2001+ OHLC"), (2008, "2008+ LKR Div"), (2017, "2017+ Open")]:
        ax.axvline(xline, color="#94A3B8", linestyle=":", linewidth=1.2)
        ax.text(xline + 0.2, ax.get_ylim()[1] * 0.90, txt, fontsize=8.5, color="#334155")
    ax.set_title("Workstream E & G: Point-in-Time Eligible Ordinary Universe by Year (1997-2025)", fontweight="bold")
    ax.set_xlabel("Calendar Year")
    ax.set_ylabel("Distinct Eligible SecurityIDs")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p4 = viz_dir / "04_pit_liquidity_eligibility_over_time.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p4))

    # ── Chart 5: Feature Redundancy Causes & Indicator Readiness Disposition ─
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))
    pairs = red_2c_df[red_2c_df["EntryType"] == "CORRELATED_PAIR"]
    inds = red_2c_df[red_2c_df["EntryType"] == "INDICATOR_DISPOSITION"]
    pc = pairs["CorrelationCause"].value_counts().sort_values()
    bars_p = axes[0].barh(pc.index, pc.values, color="#2563EB")
    axes[0].set_title(f"Workstream F: Root Causes of {len(pairs)} Correlated Pairs (|r|>=0.80)", fontweight="bold")
    axes[0].set_xlabel("Pair Count")
    for b in bars_p:
        w = b.get_width()
        axes[0].text(w + 1, b.get_y() + b.get_height() / 2, f"{int(w)}", va="center", fontweight="bold")
    axes[0].set_xlim(0, max(pc.values) * 1.18)

    ic = inds["RedundancyTier"].value_counts().sort_values()
    bars_i = axes[1].barh(ic.index, ic.values, color="#0D9488")
    axes[1].set_title(f"63 Phase 2A Indicator Modeling Dispositions", fontweight="bold")
    axes[1].set_xlabel("Indicator Count")
    for b in bars_i:
        w = b.get_width()
        axes[1].text(w + 0.6, b.get_y() + b.get_height() / 2, f"{int(w)}", va="center", fontweight="bold")
    axes[1].set_xlim(0, max(ic.values) * 1.18)
    fig.tight_layout()
    p5 = viz_dir / "05_feature_redundancy_decomposition.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(p5))

    return generated
