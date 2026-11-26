"""
Phase 2C Workstreams E & F – Causal Rolling Liquidity Engine & Feature Redundancy
=================================================================================
Implements:
  - Workstream E: Strictly point-in-time historical rolling liquidity eligibility
    using trailing `[t-W+1..t]` observations paired with CSE market-calendar spans
    (no future leakage, preserves all 1,414,977 rows with explicit eligibility flags).
  - Workstream F: Explanatory decomposition of all 117 Phase 2B correlation pairs
    (`MATHEMATICAL_EQUIVALENCE`, `PRICE_LEVEL_SCALING`, `LOOKBACK_WINDOW_OVERLAP`,
    `SHARED_INPUT_DERIVATIVE`, `ECONOMIC_SIMILARITY`) + per-indicator modeling tags.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LIQUIDITY_CONFIG = {
    "short_term": {
        "allowed_main_types": ["N"],
        "min_causal_obs": 60,
        "min_rolling_freq_pct_60": 70.0,
        "min_rolling_median_turnover_60_lkr": 500_000.0,
        "min_rolling_mean_turnover_20_lkr": 250_000.0,
        "max_gap_cal_sessions_60": 10,
    },
    "medium_term": {
        "allowed_main_types": ["N", "X"],
        "min_causal_obs": 120,
        "min_rolling_freq_pct_60": 50.0,
        "min_rolling_median_turnover_60_lkr": 250_000.0,
    },
    "long_term": {
        "allowed_main_types": ["N", "X"],
        "min_causal_obs": 252,
        "min_rolling_freq_pct_60": 30.0,
        "min_rolling_median_turnover_120_lkr": 100_000.0,
    },
}


def build_point_in_time_liquidity(
    df_adj: pd.DataFrame,
    market_calendar: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Workstream E: Compute causal rolling liquidity & strategy eligibility flags at EVERY
    (SecurityID, Date) row using ONLY historical observations `[0..t]`.
    """
    cfg = config or DEFAULT_LIQUIDITY_CONFIG
    base = df_adj.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    n = len(base)

    dts = pd.to_datetime(base["Date"]).dt.normalize()
    cal_dates = np.sort(pd.to_datetime(market_calendar["Date"]).dt.normalize().unique())
    cal_idx = np.searchsorted(cal_dates, dts.to_numpy(), side="left").astype(np.int64)

    sv = pd.to_numeric(base["ShareVolume"], errors="coerce").to_numpy(dtype=np.float64)
    to = pd.to_numeric(base["Turnover"], errors="coerce").to_numpy(dtype=np.float64)
    mtypes = base["MainType"].astype(str).to_numpy()
    sids = base["SecurityID"]

    # Verified execution row: volume column exists (>0) and Turnover > 0
    active_trade = (np.isfinite(sv) & (sv > 0) & np.isfinite(to) & (to > 0)).astype(np.int32)
    valid_to = np.where(np.isfinite(to) & (to >= 0), to, np.nan)

    obs_to_date = base.groupby("SecurityID", sort=False).cumcount().to_numpy(dtype=np.int32) + 1

    # Vectorized C-speed rolling means, medians, and active sums per SecurityID
    g_act = pd.Series(active_trade).groupby(sids, sort=False)
    act_20 = g_act.rolling(20, min_periods=1).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.int32)
    act_60 = g_act.rolling(60, min_periods=1).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.int32)
    act_120 = g_act.rolling(120, min_periods=1).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.int32)

    g_to = pd.Series(valid_to).groupby(sids, sort=False)
    mean_to_20 = g_to.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True).to_numpy(dtype=np.float64)
    med_to_60 = g_to.rolling(60, min_periods=60).median().reset_index(level=0, drop=True).to_numpy(dtype=np.float64)
    med_to_120 = g_to.rolling(120, min_periods=120).median().reset_index(level=0, drop=True).to_numpy(dtype=np.float64)
    mean_to_60 = g_to.rolling(60, min_periods=60).mean().reset_index(level=0, drop=True).to_numpy(dtype=np.float64)

    # Exact causal calendar span covered by the trailing <=60 trading observations + max single gap
    freq_60 = np.zeros(n, dtype=np.float64)
    max_gap_60 = np.zeros(n, dtype=np.int32)
    consec_no_trade = np.zeros(n, dtype=np.int32)

    cal_list = cal_idx.tolist()
    act_list = active_trade.tolist()
    for _, idx in base.groupby("SecurityID", sort=False).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        k = len(pos)
        p0 = int(pos[0])
        streak = 0
        gaps: list[int] = []
        for j in range(k):
            gi = p0 + j
            if act_list[gi] == 1:
                streak = 0
            else:
                streak += 1
            consec_no_trade[gi] = streak

            g_prev = (cal_list[gi] - cal_list[gi - 1]) if j > 0 else 0
            gaps.append(g_prev)
            lo_j = max(0, j - 59)
            span_cal = max(1, cal_list[gi] - cal_list[p0 + lo_j] + 1)
            freq_60[gi] = 100.0 * act_60[gi] / float(span_cal)
            max_gap_60[gi] = max(gaps[max(0, j - 59): j + 1])

    # Apply configurable eligibility rules strictly causally
    st_c = cfg["short_term"]
    mt_c = cfg["medium_term"]
    lt_c = cfg["long_term"]

    is_n = np.isin(mtypes, st_c["allowed_main_types"])
    is_nx = np.isin(mtypes, mt_c["allowed_main_types"])
    has_vol_now = active_trade == 1

    st_elig = (
        is_n
        & (obs_to_date >= st_c["min_causal_obs"])
        & (freq_60 >= st_c["min_rolling_freq_pct_60"])
        & np.isfinite(med_to_60) & (med_to_60 >= st_c["min_rolling_median_turnover_60_lkr"])
        & np.isfinite(mean_to_20) & (mean_to_20 >= st_c["min_rolling_mean_turnover_20_lkr"])
        & (max_gap_60 <= st_c["max_gap_cal_sessions_60"])
        & has_vol_now
    )
    mt_elig = (
        is_nx
        & (obs_to_date >= mt_c["min_causal_obs"])
        & (freq_60 >= mt_c["min_rolling_freq_pct_60"])
        & np.isfinite(med_to_60) & (med_to_60 >= mt_c["min_rolling_median_turnover_60_lkr"])
    )
    lt_elig = (
        is_nx
        & (obs_to_date >= lt_c["min_causal_obs"])
        & (freq_60 >= lt_c["min_rolling_freq_pct_60"])
        & np.isfinite(med_to_120) & (med_to_120 >= lt_c["min_rolling_median_turnover_120_lkr"])
    )

    liq_status = np.where(
        ~np.isfinite(to),
        "NO_TURNOVER_DATA_PRE2001",
        np.where(
            ~is_nx,
            "NON_EQUITY_CLASS",
            np.where(
                st_elig,
                "LIQUID_PRIME_SHORT_TERM",
                np.where(
                    mt_elig,
                    "MODERATE_LIQUIDITY_MED_TERM",
                    np.where(lt_elig, "LOW_LIQUIDITY_LONG_TERM_ONLY", "ILLIQUID_OR_WARMUP"),
                ),
            ),
        ),
    )

    # Informative EligibilityReason for transparency
    reason = np.where(
        ~np.isfinite(to),
        "Pre-2001/missing Turnover column",
        np.where(
            ~is_nx,
            "Excluded ShareClass (R/W/D/B/P/U)",
            np.where(
                obs_to_date < st_c["min_causal_obs"],
                "Warm-up (<60 causal obs)",
                np.where(
                    ~is_n,
                    "Non-voting X (ineligible for default N short-term)",
                    np.where(
                        ~has_vol_now,
                        "Zero ShareVolume/Turnover on Date",
                        np.where(
                            freq_60 < st_c["min_rolling_freq_pct_60"],
                            "Low 60-obs rolling calendar frequency",
                            np.where(
                                ~(np.isfinite(med_to_60) & (med_to_60 >= st_c["min_rolling_median_turnover_60_lkr"])),
                                "Rolling 60-obs median turnover below threshold",
                                np.where(
                                    max_gap_60 > st_c["max_gap_cal_sessions_60"],
                                    "Multi-session halt/gap > 10 days inside 60d window",
                                    "ELIGIBLE_SHORT_TERM",
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    elig_df = pd.DataFrame({
        "SecurityID": base["SecurityID"],
        "CompanyCode": base["CompanyCode"],
        "Date": base["Date"],
        "MainType": base["MainType"],
        "SubType": base["SubType"],
        "TotalCausalObsToDate": obs_to_date,
        "Rolling_ActiveDays_20": act_20,
        "Rolling_ActiveDays_60": act_60,
        "Rolling_ActiveDays_120": act_120,
        "Rolling_TradingFreqPct_60": np.round(freq_60, 2),
        "Rolling_MeanTurnover_20_LKR": np.round(mean_to_20, 2),
        "Rolling_MedianTurnover_60_LKR": np.round(med_to_60, 2),
        "Rolling_MeanTurnover_60_LKR": np.round(mean_to_60, 2),
        "Rolling_MedianTurnover_120_LKR": np.round(med_to_120, 2),
        "ConsecutiveNonActiveDays": consec_no_trade,
        "MaxGapCalSessions_60": max_gap_60,
        "ShortTermEligible": st_elig,
        "MediumTermEligible": mt_elig,
        "LongTermEligible": lt_elig,
        "LiquidityStatus": liq_status,
        "EligibilityReason": reason,
    })

    # Threshold sensitivity grid across recent 2023-2025 & full history
    rec_mask = base["Date"] >= pd.Timestamp("2023-01-01")
    grid = [
        (100_000, 30.0),
        (250_000, 50.0),
        (500_000, 70.0),
        (1_000_000, 80.0),
        (2_500_000, 85.0),
    ]
    sens = []
    for to_min, fr_min in grid:
        m_full = is_n & (obs_to_date >= 60) & (freq_60 >= fr_min) & np.isfinite(med_to_60) & (med_to_60 >= to_min) & has_vol_now
        m_rec = m_full & rec_mask.to_numpy()
        sens.append({
            "min_med_turnover_60_lkr": to_min,
            "min_freq_pct_60": fr_min,
            "eligible_rows_full": int(m_full.sum()),
            "eligible_securities_ever": int(elig_df.loc[m_full, "SecurityID"].nunique()),
            "eligible_securities_2023_2025": int(elig_df.loc[m_rec, "SecurityID"].nunique()),
        })

    # Overall summary per strategy (ever eligible + 2023-2025 active)
    summary = {
        "short_term_eligible_rows": int(st_elig.sum()),
        "short_term_eligible_securities_ever": int(elig_df.loc[st_elig, "SecurityID"].nunique()),
        "short_term_eligible_securities_2023_2025": int(elig_df.loc[st_elig & rec_mask.to_numpy(), "SecurityID"].nunique()),
        "medium_term_eligible_rows": int(mt_elig.sum()),
        "medium_term_eligible_securities_ever": int(elig_df.loc[mt_elig, "SecurityID"].nunique()),
        "medium_term_eligible_securities_2023_2025": int(elig_df.loc[mt_elig & rec_mask.to_numpy(), "SecurityID"].nunique()),
        "long_term_eligible_rows": int(lt_elig.sum()),
        "long_term_eligible_securities_ever": int(elig_df.loc[lt_elig, "SecurityID"].nunique()),
        "long_term_eligible_securities_2023_2025": int(elig_df.loc[lt_elig & rec_mask.to_numpy(), "SecurityID"].nunique()),
        "config": cfg,
        "rolling_threshold_sensitivity": sens,
    }
    return elig_df, summary


def analyze_feature_redundancy(
    red_2b_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Workstream F: Decompose Phase 2B correlation pairs into structural causes and
    issue clear Phase 3 feature selection guidance without deleting Phase 2A features.
    """
    price_level_feats = {
        "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26", "WMA_20",
        "BB_Upper_20_2", "BB_Lower_20_2", "Supertrend_10_3", "PSAR", "VWAP_20",
        "ATR_14", "MOM_10", "MACD_Line", "MACD_Signal", "MACD_Hist", "RS_ASPI_Level",
    }

    # Explain each pair in red_2b_df
    pair_rows: list[dict[str, Any]] = []
    for _, r in red_2b_df.iterrows():
        fa, fb = str(r["feature_a"]), str(r["feature_b"])
        pr = float(r["pearson_corr_2017plus_N"])
        if {fa, fb} == {"Williams_R_14", "Stoch_K_14"}:
            cause = "MATHEMATICAL_EQUIVALENCE"
            action = "DROP_ONE_EXACT_AFFINE_DUPLICATE (Keep Stoch_K_14; drop Williams_R_14 = Stoch_K_14 - 100)"
        elif fa in price_level_feats and fb in price_level_feats:
            cause = "PRICE_LEVEL_SCALING"
            action = "CONVERT_OR_PRUNE_NOMINAL_LEVEL (Replace raw LKR levels with ratios to Close/SMA for cross-sectional ML)"
        elif fa.startswith(("SMA_", "EMA_", "WMA_")) and fb.startswith(("SMA_", "EMA_", "WMA_")):
            cause = "LOOKBACK_WINDOW_OVERLAP"
            action = "KEEP_REPRESENTATIVE_SPREAD (Retain MA_Cross_Spread_20_50)"
        elif fa.startswith("ROC_") or fb.startswith("ROC_"):
            cause = "SHARED_INPUT_DERIVATIVE"
            action = "MONITOR_VIF (Retain ROC_10 vs RS_ASPI_ROC_20; ASPI neutralization decouples relative strength)"
        elif {fa, fb} == {"ASPI_MOM_20", "Sector_Breadth_MOM_20"}:
            cause = "ECONOMIC_SIMILARITY"
            action = "RETAIN_BOTH_OR_SPREAD (Macro breadth vs cap-weighted ASPI momentum)"
        else:
            cause = "SHARED_INPUT_DERIVATIVE"
            action = "MONITOR_IN_FEATURE_SELECTION"

        pair_rows.append({
            "EntryType": "CORRELATED_PAIR",
            "FeatureA": fa,
            "FeatureB": fb,
            "GroupA": r["group_a"],
            "GroupB": r["group_b"],
            "PearsonCorr": pr,
            "SpearmanCorr": r["spearman_corr_2017plus_N"],
            "WithinTop10Spearman": r["within_top10_stocks_spearman"],
            "RedundancyTier": r["redundancy_tier"],
            "CorrelationCause": cause,
            "Phase3Guidance": action,
        })

    # Append 63 individual indicator readiness dispositions (so single CSV guides Phase 3 selection completely)
    ind_rows: list[dict[str, Any]] = []
    unmapped = {"Sector_MOM_20", "RS_Sector_ROC_20"}
    open_gated = {"Overnight_Gap_Ret", "Intraday_Ret"}
    for _, m in meta_df.iterrows():
        ind = str(m["indicator_name"])
        if ind in unmapped:
            cause = "UNAVAILABLE_SECTOR_UNMAPPED"
            tier = "UNAVAILABLE_100PCT_NAN"
            action = "EXCLUDE_UNTIL_SECTOR_CROSSWALK_RESOLVED"
        elif ind == "Williams_R_14":
            cause = "MATHEMATICAL_EQUIVALENCE"
            tier = "DUPLICATE_OF_STOCH_K_14"
            action = "RECOMMEND_REMOVE_FOR_LINEAR_AND_TREE_MODELS"
        elif ind in open_gated:
            cause = "SCHEMA_REGIME_GATED_2017PLUS"
            tier = "REGIME_GATED_VALID"
            action = "KEEP_ONLY_IN_2017PLUS_PIPELINE"
        elif ind in price_level_feats:
            cause = "PRICE_LEVEL_SCALING"
            tier = "LEVEL_CONFOUNDED_CROSS_SECTION"
            action = "USE_SCALE_NORMALIZED_EQUIVALENT_OR_PRUNE_NOMINAL_LEVEL"
        elif ind in {"OBV", "AD_Line", "Volume_SMA_20"}:
            cause = "CUMULATIVE_SIZE_SCALED"
            tier = "SCALE_DEPENDENT_VOLUME"
            action = "PREFER_VOLUME_RATIO_20_OR_CMF_20_FOR_RANKING"
        else:
            cause = "SCALE_FREE_PRIMARY"
            tier = "CLEAN_NON_REDUNDANT"
            action = "RECOMMEND_KEEP_IN_SHORT_TERM_MODEL_MATRIX"

        ind_rows.append({
            "EntryType": "INDICATOR_DISPOSITION",
            "FeatureA": ind,
            "FeatureB": "",
            "GroupA": m["group"],
            "GroupB": m["output_table"],
            "PearsonCorr": np.nan,
            "SpearmanCorr": np.nan,
            "WithinTop10Spearman": np.nan,
            "RedundancyTier": tier,
            "CorrelationCause": cause,
            "Phase3Guidance": action,
        })

    out_df = pd.DataFrame(ind_rows + pair_rows)
    cause_counts = pd.DataFrame(pair_rows)["CorrelationCause"].value_counts().to_dict() if pair_rows else {}
    disp_counts = pd.DataFrame(ind_rows)["RedundancyTier"].value_counts().to_dict()
    summary = {
        "total_pairs_analyzed": int(len(pair_rows)),
        "pair_cause_breakdown": {str(k): int(v) for k, v in cause_counts.items()},
        "indicator_disposition_breakdown": {str(k): int(v) for k, v in disp_counts.items()},
        "recommended_clean_scale_free_core_count": int(disp_counts.get("CLEAN_NON_REDUNDANT", 0)),
    }
    return out_df, summary
