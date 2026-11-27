"""
Phase 4A Features, Eligibility Waterfall, Purged Splits, Quality Audit & Pipeline Orchestrator
==============================================================================================

Constructs:
1. Causal scale-free derived features & `short_term_feature_registry.json` (using exact
   split-adjusted `Close` basis matching Phase 2A moving-average/level indicators).
2. Complete 1.415M-row eligibility audit (`short_term_sample_eligibility.parquet` and
   `short_term_exclusion_audit.parquet`) preserving all delisted securities cleanly.
3. Expanding chronological cross-validation splits with strict 65-session purge+embargo
   gaps (`60d max horizon + 5d embargo`) and untouched 2024-2025 `FINAL_TEST` holdout
   (`short_term_validation_splits.parquet`).
4. Liquidity/penny-tick sensitivity matrix, tail diagnostics, and leakage assertions
   (`short_term_target_definitions.json`, `short_term_target_quality_report.json`).
5. Model-ready dataset (`short_term_modeling_dataset.parquet`), SHA-256 manifest, and 4
   publication-grade diagnostic PNG visualizations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase3d_features.contract_and_mapping import (
    UPSTREAM_CORE_PATHS,
    compute_file_sha256,
    verify_upstream_immutability,
)
from phase4a_dataset.calendar_and_targets import (
    EMBARGO_SESSIONS,
    HORIZONS,
    MAX_HORIZON,
    MIN_CROSS_SECTION_SIZE,
    PURGE_SESSIONS,
    TOTAL_GAP_SESSIONS,
    build_trading_calendar_and_benchmarks,
    compute_forward_targets_and_labels,
)

logger = logging.getLogger("phase4a.pipeline")

PENNY_CLOSE_THRESHOLD_LKR = 1.00
FINAL_TEST_START_DATE = "2024-01-02"

# Expanding-window chronological CV validation windows strictly prior to FINAL_TEST
DEFAULT_CV_FOLD_SPECS: list[dict[str, str]] = [
    {"fold_name": "CV_Fold_1", "val_start": "2019-01-02", "val_end": "2019-12-31", "regime_note": "Pre-COVID subdued liquidity / consolidation"},
    {"fold_name": "CV_Fold_2", "val_start": "2020-01-02", "val_end": "2020-12-31", "regime_note": "COVID-19 closure crash & Q3-Q4 rebound"},
    {"fold_name": "CV_Fold_3", "val_start": "2021-01-04", "val_end": "2021-12-31", "regime_note": "Low-rate retail liquidity boom & peak rally"},
    {"fold_name": "CV_Fold_4", "val_start": "2022-01-03", "val_end": "2023-09-26", "regime_note": "FX crisis, rate shock & IMF stabilization recovery"},
]

PHASE4A_OUTPUT_FILES: list[str] = [
    "Dataset/processed/phase4a/short_term_modeling_dataset.parquet",
    "Dataset/processed/phase4a/short_term_feature_registry.json",
    "Dataset/processed/phase4a/short_term_target_definitions.json",
    "Dataset/processed/phase4a/short_term_sample_eligibility.parquet",
    "Dataset/processed/phase4a/short_term_exclusion_audit.parquet",
    "Dataset/processed/phase4a/short_term_validation_splits.parquet",
    "Dataset/processed/phase4a/short_term_target_quality_report.json",
    "Dataset/processed/phase4a/short_term_dataset_manifest.json",
]

# Explicit Phase 3D inputs locked against modification alongside Phase 1-3C cores
PHASE4A_LOCKED_UPSTREAM_PATHS: list[str] = UPSTREAM_CORE_PATHS + [
    "Dataset/processed/phase3d/phase3d_manifest.json",
    "Dataset/processed/phase3d/short_term_integrated_research_universe.parquet",
    "Dataset/processed/phase3d/company_security_pit_mapping.parquet",
    "Dataset/processed/phase2c/corporate_action_remediation.parquet",
    "Dataset/processed/phase2c/price_anomaly_investigation.parquet",
]

# Retained base scale-free / bounded indicators verified in Phase 2C/3D
CORE_FULL_SAMPLE_TECH_FEATURES: list[str] = [
    "Daily_Return_SplitAdj",
    "Daily_Return_CapAdj",
    "ADX_14",
    "Aroon_Up_25",
    "Aroon_Down_25",
    "Ichimoku_TK_Spread_Pct",
    "Ichimoku_Kumo_Pos_Pct",
    "Supertrend_Dir_10_3",
    "Supertrend_Reversal",
    "RSI_14",
    "ROC_10",
    "ROC_20",
    "CCI_20",
    "Stoch_K_14",
    "Stoch_D_3",
    "NATR_14",
    "BB_Width_20_2",
    "BB_PctB_20_2",
    "Keltner_Pos_20",
    "Hist_Vol_21d",
    "Donchian_Pos_20",
    "PSAR_Trend",
    "PSAR_Distance_Pct",
    "PSAR_Reversal",
    "MA_Cross_Spread_20_50",
    "MA_Cross_Signal_20_50",
    "RSI_Div_Bullish_14",
    "RSI_Div_Bearish_14",
    "MFI_14",
    "CMF_20",
    "Volume_Ratio_20",
    "Amihud_Illiquidity_20",
    "ASPI_MOM_20",
    "RS_ASPI_ROC_20",
    "Beta_ASPI_60",
    "Sector_Breadth_MOM_20",
    "Rolling_ActiveDays_20",
    "Rolling_ActiveDays_60",
    "Rolling_TradingFreqPct_60",
    "ConsecutiveNonActiveDays",
    "beta60_observation_count",
    "beta60_nonzero_return_obs",
    "IsBetaReliable",
]

OPEN_REGIME_2017PLUS_FEATURES: list[str] = [
    "Overnight_Gap_Ret",
    "Intraday_Ret",
    "IsOpenSessionRegime",
]

PIT_FUNDAMENTAL_FEATURES: list[str] = [
    "IsFinancialInstitution",
    "IsFundamentalPITValid",
    "IsFundamentalDataAvailable",
    "IsFundamentalStale_120d",
    "IsFundamentalStale_180d",
    "IsFundamentalStale_270d",
    "IsFundamentalStale_365d",
    "IsFundamentalStale",
    "IsFundamentalExpired",
    "HasRestatementAtDecision",
    "FundamentalAgeDays_Clean",
    "rev_growth_qtr_yoy_pct",
    "net_income_growth_qtr_yoy_pct",
    "earnings_recovery_flag",
    "roe_pct",
    "net_profit_margin_pct",
    "is_ttm_complete",
    "sec_pe_ratio",
    "sec_earnings_yield_pct",
    "sec_pb_ratio",
    "company_pb_ratio",
    "sec_dividend_yield_pct",
    "Log_MarketCap_LKR",
    "Turnover_to_MktCap_Ratio",
]

DERIVED_SCALE_FREE_FEATURES: list[str] = [
    "Price_to_SMA20_Pct",
    "Price_to_SMA50_Pct",
    "SMA20_to_SMA50_Pct",
    "Price_to_EMA12_Pct",
    "Price_to_EMA26_Pct",
    "EMA12_to_EMA26_Pct",
    "Price_to_WMA20_Pct",
    "Price_to_VWAP20_Pct",
    "Price_to_Supertrend_Pct",
    "BB_Upper_Dist_Pct",
    "BB_Lower_Dist_Pct",
    "MACD_Line_PctClose",
    "MACD_Signal_PctClose",
    "MACD_Hist_PctClose",
    "MOM_10_PctClose",
    "ATR_PctClose",
    "Vol_Ratio_ATR_HistVol",
    "Log_ShareVolume",
    "Log_Turnover_LKR",
    "Log_Rolling_MeanTurnover_20",
    "Log_Rolling_MedianTurnover_60",
    "Amihud_Log1p",
    "IsLowPrice_Sub1LKR",
]

EXCLUDED_PRICE_LEVEL_OR_SCALE_DEPS: dict[str, dict[str, str]] = {
    "RawOpen": {"reason": "LEVEL_CONFOUNDED_PRICE", "replacement": "Overnight_Gap_Ret, Intraday_Ret (2017+)"},
    "RawHigh": {"reason": "LEVEL_CONFOUNDED_PRICE", "replacement": "Donchian_Pos_20, Keltner_Pos_20, NATR_14"},
    "RawLow": {"reason": "LEVEL_CONFOUNDED_PRICE", "replacement": "Donchian_Pos_20, Keltner_Pos_20, NATR_14"},
    "RawClose": {"reason": "LEVEL_CONFOUNDED_PRICE_RETAINED_FOR_PENNY_FILTER", "replacement": "IsLowPrice_Sub1LKR"},
    "Close": {"reason": "LEVEL_CONFOUNDED_SPLIT_ADJ_BASIS", "replacement": "Price_to_*_Pct ratio transforms"},
    "CapitalAdjClose": {"reason": "LEVEL_CONFOUNDED_CA_BASIS_TARGET_ONLY", "replacement": "Daily_Return_CapAdj / Targets"},
    "TotalReturnPrice": {"reason": "TARGET_BASIS_ONLY_LEAKAGE_IF_USED", "replacement": "Targets only"},
    "ASPI_Close": {"reason": "NON_STATIONARY_INDEX_LEVEL", "replacement": "ASPI_MOM_20, RS_ASPI_ROC_20"},
    "ASTRI_Close": {"reason": "NON_STATIONARY_INDEX_LEVEL_TARGET_ONLY", "replacement": "Targets only"},
    "SMA_20": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "Price_to_SMA20_Pct, SMA20_to_SMA50_Pct"},
    "SMA_50": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "Price_to_SMA50_Pct, SMA20_to_SMA50_Pct"},
    "EMA_12": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "Price_to_EMA12_Pct, EMA12_to_EMA26_Pct"},
    "EMA_26": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "Price_to_EMA26_Pct, EMA12_to_EMA26_Pct"},
    "WMA_20": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "Price_to_WMA20_Pct"},
    "VWAP_20": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "Price_to_VWAP20_Pct"},
    "Supertrend_10_3": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "Price_to_Supertrend_Pct, Supertrend_Dir_10_3"},
    "BB_Upper_20_2": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "BB_Upper_Dist_Pct, BB_PctB_20_2"},
    "BB_Lower_20_2": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "BB_Lower_Dist_Pct, BB_PctB_20_2"},
    "PSAR": {"reason": "LEVEL_CONFOUNDED_CROSS_SECTION", "replacement": "PSAR_Distance_Pct, PSAR_Trend"},
    "MACD_Line": {"reason": "LEVEL_CONFOUNDED_CURRENCY_DIFF", "replacement": "MACD_Line_PctClose"},
    "MACD_Signal": {"reason": "LEVEL_CONFOUNDED_CURRENCY_DIFF", "replacement": "MACD_Signal_PctClose"},
    "MACD_Hist": {"reason": "LEVEL_CONFOUNDED_CURRENCY_DIFF", "replacement": "MACD_Hist_PctClose"},
    "MOM_10": {"reason": "LEVEL_CONFOUNDED_CURRENCY_DIFF", "replacement": "MOM_10_PctClose, ROC_10"},
    "ATR_14": {"reason": "LEVEL_CONFOUNDED_CURRENCY_DIFF", "replacement": "ATR_PctClose, NATR_14"},
    "ShareVolume": {"reason": "SCALE_DEPENDENT_SHARE_COUNT", "replacement": "Log_ShareVolume, Volume_Ratio_20"},
    "Turnover": {"reason": "SCALE_DEPENDENT_LKR", "replacement": "Log_Turnover_LKR, Turnover_to_MktCap_Ratio"},
    "Volume_SMA_20": {"reason": "SCALE_DEPENDENT_SHARE_COUNT", "replacement": "Volume_Ratio_20"},
    "OBV": {"reason": "UNBOUNDED_PATH_DEPENDENT_CUMSUM", "replacement": "CMF_20, MFI_14, Volume_Ratio_20"},
    "AD_Line": {"reason": "UNBOUNDED_PATH_DEPENDENT_CUMSUM", "replacement": "CMF_20, MFI_14"},
    "Chaikin_Osc_3_10": {"reason": "UNBOUNDED_SCALE_DEPENDENT_VOLUME_DIFF", "replacement": "CMF_20, MFI_14"},
    "Rolling_MeanTurnover_20_LKR": {"reason": "SCALE_DEPENDENT_LKR", "replacement": "Log_Rolling_MeanTurnover_20"},
    "Rolling_MedianTurnover_60_LKR": {"reason": "SCALE_DEPENDENT_LKR", "replacement": "Log_Rolling_MedianTurnover_60"},
    "sec_shares_outstanding_pit": {"reason": "SCALE_DEPENDENT_SHARE_COUNT", "replacement": "Log_MarketCap_LKR, sec_pb_ratio"},
    "sec_market_cap_lkr": {"reason": "SCALE_DEPENDENT_LKR", "replacement": "Log_MarketCap_LKR, Turnover_to_MktCap_Ratio"},
    "company_market_cap_lkr": {"reason": "SCALE_DEPENDENT_LKR", "replacement": "Log_MarketCap_LKR"},
    "ttm_revenue": {"reason": "SCALE_DEPENDENT_LKR", "replacement": "rev_growth_qtr_yoy_pct, net_profit_margin_pct"},
    "ttm_net_income": {"reason": "SCALE_DEPENDENT_LKR", "replacement": "net_income_growth_qtr_yoy_pct, roe_pct, sec_earnings_yield_pct"},
    "sec_ttm_cash_dps_lkr": {"reason": "LEVEL_CONFOUNDED_DPS", "replacement": "sec_dividend_yield_pct"},
}

EXCLUDED_CONSTANT_OR_ELIGIBILITY_FLAGS: list[str] = [
    "IsSectorMappingAvailable",
    "IsHistoricallyListed",
    "IsHistoricallyTradable",
    "IsPriceDataValid",
    "IsCorporateActionValidated",
    "IsLifecycleValidated",
    "IsOpenFeatureAvailable",
    "IsHLFeatureAvailable",
    "IsShortTermEligible",
    "IsModelTrainingEligible",
    "IsEquityShareClass",
    "IsPrimaryShareClass",
    "IsStrictPITIntegrationEligible",
    "IsCompanyMappingValidAtDate",
    "IsShortTermTechnicalEligible",
    "IsShortTermIntegratedEligible",
]

IDENTIFIER_AND_METADATA_COLS: list[str] = [
    "Date",
    "SessionIndex",
    "SecurityID",
    "CompanyCode",
    "IssuerID",
    "CompanyID",
    "MainType",
    "SubType",
    "ShareClass",
    "BusinessType",
    "Era",
    "DataFormat",
    "DataQualityFlag",
    "LiquidityStatus",
    "beta_reliability_flag",
    "beta_validity_status",
    "MappingConfidence",
    "FundamentalPeriodEnd",
    "FundamentalAvailabilityDate",
    "FundamentalAgeDays",
    "DaysSinceAvailabilityDate",
    "FundamentalAgeTradingSessions",
    "SelectedRestatementVersion",
    "ConsolidationBasis",
    "CurrencyBasis",
]


def _safe_ratio_minus_one(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Safely evaluate causal `(num / den) - 1.0`, returning NaN if denominator non-positive/non-finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = np.isfinite(num) & np.isfinite(den) & (den > 1e-8)
        return np.where(valid, (num / den) - 1.0, np.nan)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Safely evaluate causal `num / den`, returning NaN if denominator non-positive/non-finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = np.isfinite(num) & np.isfinite(den) & (den > 1e-8)
        return np.where(valid, num / den, np.nan)


def build_derived_scale_free_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly contemporaneous causal scale-free transforms from Phase 3D columns.

    CRITICAL BASIS INVARIANT (verified against JKH/CSE series):
    Phase 2A indicators (`SMA_20`, `SMA_50`, `EMA_12`, `EMA_26`, `WMA_20`, `VWAP_20`,
    `Supertrend_10_3`, `BB_Upper_20_2`, `BB_Lower_20_2`, `MACD_*`, `MOM_10`, `ATR_14`)
    are constructed on split-adjusted `Close`. Dividing by `Close` eliminates spurious
    rights/bonus corporate-action discontinuities that would appear if dividing by
    `CapitalAdjClose` or `RawClose`.
    """
    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=np.float64)
    raw_close = pd.to_numeric(df["RawClose"], errors="coerce").to_numpy(dtype=np.float64)

    out["Price_to_SMA20_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["SMA_20"], errors="coerce").to_numpy(dtype=np.float64))
    out["Price_to_SMA50_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["SMA_50"], errors="coerce").to_numpy(dtype=np.float64))
    out["SMA20_to_SMA50_Pct"] = _safe_ratio_minus_one(
        pd.to_numeric(df["SMA_20"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df["SMA_50"], errors="coerce").to_numpy(dtype=np.float64),
    )
    out["Price_to_EMA12_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["EMA_12"], errors="coerce").to_numpy(dtype=np.float64))
    out["Price_to_EMA26_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["EMA_26"], errors="coerce").to_numpy(dtype=np.float64))
    out["EMA12_to_EMA26_Pct"] = _safe_ratio_minus_one(
        pd.to_numeric(df["EMA_12"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df["EMA_26"], errors="coerce").to_numpy(dtype=np.float64),
    )
    out["Price_to_WMA20_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["WMA_20"], errors="coerce").to_numpy(dtype=np.float64))
    out["Price_to_VWAP20_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["VWAP_20"], errors="coerce").to_numpy(dtype=np.float64))
    out["Price_to_Supertrend_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["Supertrend_10_3"], errors="coerce").to_numpy(dtype=np.float64))

    out["BB_Upper_Dist_Pct"] = _safe_ratio_minus_one(pd.to_numeric(df["BB_Upper_20_2"], errors="coerce").to_numpy(dtype=np.float64), close)
    out["BB_Lower_Dist_Pct"] = _safe_ratio_minus_one(close, pd.to_numeric(df["BB_Lower_20_2"], errors="coerce").to_numpy(dtype=np.float64))

    out["MACD_Line_PctClose"] = _safe_ratio(pd.to_numeric(df["MACD_Line"], errors="coerce").to_numpy(dtype=np.float64), close)
    out["MACD_Signal_PctClose"] = _safe_ratio(pd.to_numeric(df["MACD_Signal"], errors="coerce").to_numpy(dtype=np.float64), close)
    out["MACD_Hist_PctClose"] = _safe_ratio(pd.to_numeric(df["MACD_Hist"], errors="coerce").to_numpy(dtype=np.float64), close)
    out["MOM_10_PctClose"] = _safe_ratio(pd.to_numeric(df["MOM_10"], errors="coerce").to_numpy(dtype=np.float64), close)

    atr_pct = _safe_ratio(pd.to_numeric(df["ATR_14"], errors="coerce").to_numpy(dtype=np.float64), close)
    out["ATR_PctClose"] = atr_pct
    hvol = pd.to_numeric(df["Hist_Vol_21d"], errors="coerce").to_numpy(dtype=np.float64)
    out["Vol_Ratio_ATR_HistVol"] = _safe_ratio(atr_pct, np.where(hvol > 1e-6, hvol, np.nan))

    sv = pd.to_numeric(df["ShareVolume"], errors="coerce").to_numpy(dtype=np.float64)
    tv = pd.to_numeric(df["Turnover"], errors="coerce").to_numpy(dtype=np.float64)
    mt20 = pd.to_numeric(df["Rolling_MeanTurnover_20_LKR"], errors="coerce").to_numpy(dtype=np.float64)
    mt60 = pd.to_numeric(df["Rolling_MedianTurnover_60_LKR"], errors="coerce").to_numpy(dtype=np.float64)
    amh = pd.to_numeric(df["Amihud_Illiquidity_20"], errors="coerce").to_numpy(dtype=np.float64)

    out["Log_ShareVolume"] = np.where(np.isfinite(sv) & (sv >= 0), np.log1p(np.maximum(sv, 0.0)), np.nan)
    out["Log_Turnover_LKR"] = np.where(np.isfinite(tv) & (tv >= 0), np.log1p(np.maximum(tv, 0.0)), np.nan)
    out["Log_Rolling_MeanTurnover_20"] = np.where(np.isfinite(mt20) & (mt20 >= 0), np.log1p(np.maximum(mt20, 0.0)), np.nan)
    out["Log_Rolling_MedianTurnover_60"] = np.where(np.isfinite(mt60) & (mt60 >= 0), np.log1p(np.maximum(mt60, 0.0)), np.nan)
    out["Amihud_Log1p"] = np.where(np.isfinite(amh) & (amh >= 0), np.log1p(np.maximum(amh, 0.0)), np.nan)

    # Strictly PIT-gated fundamental log-size and turnover ratio
    pit_ok = df["IsFundamentalPITValid"].fillna(False).to_numpy(dtype=bool)
    mcap = pd.to_numeric(df["sec_market_cap_lkr"], errors="coerce").to_numpy(dtype=np.float64)
    mcap_ok = pit_ok & np.isfinite(mcap) & (mcap > 0)
    out["Log_MarketCap_LKR"] = np.where(mcap_ok, np.log1p(mcap), np.nan)
    out["Turnover_to_MktCap_Ratio"] = np.where(mcap_ok & np.isfinite(tv) & (tv >= 0), tv / mcap, np.nan)

    f_age = pd.to_numeric(df["FundamentalAgeDays"], errors="coerce").to_numpy(dtype=np.float64)
    out["FundamentalAgeDays_Clean"] = np.where(pit_ok & np.isfinite(f_age) & (f_age >= 0), f_age, np.nan)

    out["IsLowPrice_Sub1LKR"] = pd.Series(
        np.where(np.isfinite(raw_close), (raw_close < PENNY_CLOSE_THRESHOLD_LKR).astype(np.int8), 0),
        index=df.index,
        dtype=np.int8,
    )
    out["IsOpenSessionRegime"] = df["IsOpenFeatureAvailable"].fillna(False).to_numpy(dtype=np.int8)
    return out


def assign_chronological_purged_splits(
    model_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    purge_sessions: int = PURGE_SESSIONS,
    embargo_sessions: int = EMBARGO_SESSIONS,
    cv_fold_specs: list[dict[str, str]] = DEFAULT_CV_FOLD_SPECS,
    final_test_start_date: str = FINAL_TEST_START_DATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign strictly expanding purged CV folds + quarantined pre-2024 gap + untouched FINAL_TEST."""
    gap_sessions = int(purge_sessions + embargo_sessions)
    cal_dts = pd.to_datetime(cal_df["Dt"]).dt.normalize()

    def _date_to_pos(date_str: str) -> int:
        ts = pd.Timestamp(date_str).normalize()
        idx = int(np.searchsorted(cal_dts.to_numpy(dtype="datetime64[ns]"), ts.to_datetime64(), side="left"))
        return min(idx, len(cal_df) - 1)

    def _date_to_end_pos(date_str: str) -> int:
        ts = pd.Timestamp(date_str).normalize()
        idx = int(np.searchsorted(cal_dts.to_numpy(dtype="datetime64[ns]"), ts.to_datetime64(), side="right")) - 1
        return max(0, idx)

    ft_start_pos = _date_to_pos(final_test_start_date)
    dev_train_end_pos = ft_start_pos - gap_sessions - 1

    pos_arr = model_df["SessionIndex"].to_numpy(dtype=np.int32)
    listed_arr = model_df["IsListed"].to_numpy(dtype=bool)
    tr20_arr = model_df["IsTargetEligible_TR_20d"].to_numpy(dtype=bool)
    cap20_arr = model_df["IsTargetEligible_Cap_20d"].to_numpy(dtype=bool)

    splits_df = pd.DataFrame(
        {
            "Date": model_df["Date"].to_numpy(),
            "SessionIndex": pos_arr,
            "SecurityID": model_df["SecurityID"].to_numpy(),
            "CompanyCode": model_df["CompanyCode"].to_numpy(),
            "IssuerID": model_df["IssuerID"].to_numpy(),
            "IsListed": listed_arr,
            "IsTargetEligible_TR_20d": tr20_arr,
            "IsTargetEligible_Cap_20d": cap20_arr,
        },
        index=model_df.index,
    )

    final_role = np.where(
        pos_arr >= ft_start_pos,
        "FINAL_TEST",
        np.where(pos_arr <= dev_train_end_pos, "DEV_POOL", "PURGE_EMBARGO_PRE_FINAL_TEST"),
    )
    splits_df["FinalSplitRole"] = pd.Series(final_role, index=model_df.index, dtype=object)

    # Also build primary mutually-exclusive chronological regime label across the sample for reporting
    regime_lbl = np.where(
        pos_arr >= ft_start_pos,
        "FINAL_TEST_2024_2025",
        np.where(
            pos_arr <= dev_train_end_pos,
            "DEV_2002_2023Q3",
            "PURGE_EMBARGO_2023Q4",
        ),
    )
    splits_df["EraSplitGroup"] = pd.Series(regime_lbl, index=model_df.index, dtype=object)

    fold_summaries: list[dict[str, Any]] = []
    for spec in cv_fold_specs:
        fname = spec["fold_name"]
        v_start_pos = _date_to_pos(spec["val_start"])
        v_end_pos = _date_to_end_pos(spec["val_end"])
        # Respect strict pre-FINAL_TEST cutoff on last CV fold automatically
        if v_end_pos > dev_train_end_pos:
            v_end_pos = dev_train_end_pos
        tr_end_pos = v_start_pos - gap_sessions - 1

        role = np.full(len(model_df), "EXCLUDED_POST_VAL", dtype=object)
        role[pos_arr <= tr_end_pos] = "TRAIN"
        role[(pos_arr > tr_end_pos) & (pos_arr < v_start_pos)] = "PURGE_EMBARGO_GAP"
        role[(pos_arr >= v_start_pos) & (pos_arr <= v_end_pos)] = "VAL"
        splits_df[fname] = pd.Series(role, index=model_df.index, dtype=object)

        m_tr = role == "TRAIN"
        m_gap = role == "PURGE_EMBARGO_GAP"
        m_val = role == "VAL"

        # Verify zero forward label leakage: max TargetDate_60d in TRAIN <= val_start_date strictly!
        val_start_dt = pd.Timestamp(cal_dts.iloc[v_start_pos])
        tr_tgt60 = pd.to_datetime(model_df.loc[m_tr & model_df["IsTargetEligible_Cap_60d"], f"TargetDate_{MAX_HORIZON}d"])
        max_tr_tgt60_dt = tr_tgt60.max() if len(tr_tgt60) else pd.NaT
        zero_overlap_verified = bool(pd.isna(max_tr_tgt60_dt) or (max_tr_tgt60_dt < val_start_dt))

        fold_summaries.append(
            {
                "fold_name": fname,
                "regime_note": spec["regime_note"],
                "train_start_date": str(pd.Timestamp(cal_dts.iloc[pos_arr[m_tr].min()]).date()) if m_tr.any() else None,
                "train_end_date": str(pd.Timestamp(cal_dts.iloc[tr_end_pos]).date()),
                "train_max_session_index": int(tr_end_pos),
                "gap_sessions": int(gap_sessions),
                "purge_sessions": int(purge_sessions),
                "embargo_sessions": int(embargo_sessions),
                "val_start_date": str(pd.Timestamp(cal_dts.iloc[v_start_pos]).date()),
                "val_min_session_index": int(v_start_pos),
                "val_end_date": str(pd.Timestamp(cal_dts.iloc[v_end_pos]).date()),
                "val_max_session_index": int(v_end_pos),
                "train_rows": int(m_tr.sum()),
                "train_tr20_valid_rows": int((m_tr & tr20_arr).sum()),
                "train_cap20_valid_rows": int((m_tr & cap20_arr).sum()),
                "train_delisted_rows": int((m_tr & (~listed_arr)).sum()),
                "train_unique_securities": int(model_df.loc[m_tr, "SecurityID"].nunique()),
                "train_delisted_securities": int(model_df.loc[m_tr & (~listed_arr), "SecurityID"].nunique()),
                "gap_rows_purged": int(m_gap.sum()),
                "val_rows": int(m_val.sum()),
                "val_tr20_valid_rows": int((m_val & tr20_arr).sum()),
                "val_cap20_valid_rows": int((m_val & cap20_arr).sum()),
                "val_delisted_rows": int((m_val & (~listed_arr)).sum()),
                "val_unique_securities": int(model_df.loc[m_val, "SecurityID"].nunique()),
                "val_delisted_securities": int(model_df.loc[m_val & (~listed_arr), "SecurityID"].nunique()),
                "max_train_target60d_date": str(max_tr_tgt60_dt.date()) if pd.notna(max_tr_tgt60_dt) else None,
                "zero_target60d_overlap_with_val": zero_overlap_verified,
            }
        )

    # Verify DEV_POOL vs FINAL_TEST zero overlap identically
    m_dev = final_role == "DEV_POOL"
    m_pre_gap = final_role == "PURGE_EMBARGO_PRE_FINAL_TEST"
    m_ft = final_role == "FINAL_TEST"
    ft_start_dt = pd.Timestamp(cal_dts.iloc[ft_start_pos])
    dev_tgt60 = pd.to_datetime(model_df.loc[m_dev & model_df["IsTargetEligible_Cap_60d"], f"TargetDate_{MAX_HORIZON}d"])
    max_dev_tgt60 = dev_tgt60.max() if len(dev_tgt60) else pd.NaT

    split_summary = {
        "purge_sessions": int(purge_sessions),
        "embargo_sessions": int(embargo_sessions),
        "total_gap_sessions": int(gap_sessions),
        "cv_folds": fold_summaries,
        "final_holdout_split": {
            "dev_pool_start_date": str(pd.Timestamp(cal_dts.iloc[pos_arr[m_dev].min()]).date()) if m_dev.any() else None,
            "dev_pool_end_date": str(pd.Timestamp(cal_dts.iloc[dev_train_end_pos]).date()),
            "dev_pool_max_session_index": int(dev_train_end_pos),
            "dev_pool_rows": int(m_dev.sum()),
            "dev_pool_tr20_valid_rows": int((m_dev & tr20_arr).sum()),
            "dev_pool_delisted_rows": int((m_dev & (~listed_arr)).sum()),
            "dev_pool_unique_securities": int(model_df.loc[m_dev, "SecurityID"].nunique()),
            "pre_final_gap_rows": int(m_pre_gap.sum()),
            "final_test_start_date": str(ft_start_dt.date()),
            "final_test_min_session_index": int(ft_start_pos),
            "final_test_end_date": str(pd.Timestamp(cal_dts.iloc[-1]).date()),
            "final_test_rows": int(m_ft.sum()),
            "final_test_tr20_valid_rows": int((m_ft & tr20_arr).sum()),
            "final_test_cap20_valid_rows": int((m_ft & cap20_arr).sum()),
            "final_test_delisted_rows": int((m_ft & (~listed_arr)).sum()),
            "final_test_unique_securities": int(model_df.loc[m_ft, "SecurityID"].nunique()),
            "max_dev_target60d_date": str(max_dev_tgt60.date()) if pd.notna(max_dev_tgt60) else None,
            "zero_dev_target60d_overlap_with_final_test": bool(pd.isna(max_dev_tgt60) or (max_dev_tgt60 < ft_start_dt)),
        },
    }
    return splits_df, split_summary


def build_eligibility_and_exclusion_audit(
    st_df: pd.DataFrame,
    tgt_all_df: pd.DataFrame,
    splits_elig_df: pd.DataFrame,
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Classify all 1,414,977 rows into a mutually-exclusive eligibility waterfall & build audit table."""
    listed_secs = set(master_df.loc[master_df["HistoricalStatus"].astype(str) == "Listed", "SecurityID"].astype(str))
    is_listed = st_df["SecurityID"].astype(str).isin(listed_secs).to_numpy(dtype=bool)

    elig_st = st_df["IsShortTermEligible"].fillna(False).to_numpy(dtype=bool)
    is_eq = st_df["IsEquityShareClass"].fillna(False).to_numpy(dtype=bool)
    is_trd = st_df["IsHistoricallyTradable"].fillna(False).to_numpy(dtype=bool)
    is_life = st_df["IsLifecycleValidated"].fillna(False).to_numpy(dtype=bool)
    is_px = st_df["IsPriceDataValid"].fillna(False).to_numpy(dtype=bool)
    is_ca = st_df["IsCorporateActionValidated"].fillna(False).to_numpy(dtype=bool)
    ca_bad_entry = tgt_all_df["_EntryBadCAPath"].to_numpy(dtype=bool)
    src_err_entry = tgt_all_df["_EntrySrcErr"].to_numpy(dtype=bool)

    consec = pd.to_numeric(st_df["ConsecutiveNonActiveDays"], errors="coerce").fillna(999).to_numpy()
    ad20 = pd.to_numeric(st_df["Rolling_ActiveDays_20"], errors="coerce").fillna(0).to_numpy()
    ad60 = pd.to_numeric(st_df["Rolling_ActiveDays_60"], errors="coerce").fillna(0).to_numpy()
    freq60 = pd.to_numeric(st_df["Rolling_TradingFreqPct_60"], errors="coerce").fillna(0.0).to_numpy()
    med_tv60 = pd.to_numeric(st_df["Rolling_MedianTurnover_60_LKR"], errors="coerce").fillna(0.0).to_numpy()
    raw_px = pd.to_numeric(st_df["RawClose"], errors="coerce").to_numpy(dtype=np.float64)
    penny_risk = np.where(np.isfinite(raw_px), raw_px < PENNY_CLOSE_THRESHOLD_LKR, False)

    # Hierarchical mutual-exclusive waterfall (assign broad->specific OR specific-first? Order carefully!)
    reason = np.full(len(st_df), "OTHER_PHASE2C_LIQUIDITY_STATUS_EXCLUDED", dtype=object)
    reason[~(is_trd & is_life)] = "NOT_HISTORICALLY_TRADABLE_OR_LIFECYCLE_INVALID"
    reason[(is_trd & is_life) & (~is_eq)] = "NON_EQUITY_OR_NON_ORDINARY_INSTRUMENT"
    reason[(is_trd & is_life & is_eq) & (~(is_px & is_ca))] = "INVALID_PRICE_OR_UNVALIDATED_CA"
    base_valid = is_trd & is_life & is_eq & is_px & is_ca
    reason[base_valid & (consec > 0)] = "CONSECUTIVE_SUSPENSION_OR_GAP_DAY"
    reason[base_valid & (consec == 0) & ((ad20 < 20) | (ad60 < 59) | (freq60 < 70.0))] = "FAILED_TRADING_FREQ_OR_ACTIVE_WINDOW"
    reason[base_valid & (consec == 0) & (ad20 >= 20) & (ad60 >= 59) & (freq60 >= 70.0) & (med_tv60 < 500_000.0)] = "FAILED_LIQUIDITY_TURNOVER_500K_THRESHOLD"
    reason[elig_st] = "ELIGIBLE_SHORT_TERM_CORE"

    split_map = pd.Series(splits_elig_df["EraSplitGroup"].to_numpy(), index=splits_elig_df.index)
    split_full = split_map.reindex(st_df.index).fillna("INELIGIBLE_FOR_SHORT_TERM_POOL").to_numpy(dtype=object)

    sample_elig = pd.DataFrame(
        {
            "Date": pd.to_datetime(st_df["Date"]).dt.normalize(),
            "SessionIndex": tgt_all_df["SessionIndex"].to_numpy(dtype=np.int32),
            "SecurityID": st_df["SecurityID"].to_numpy(),
            "CompanyCode": st_df["CompanyCode"].to_numpy(),
            "IssuerID": st_df["IssuerID"].to_numpy(),
            "MainType": st_df["MainType"].to_numpy(),
            "SubType": st_df["SubType"].to_numpy(),
            "Era": st_df["Era"].to_numpy(),
            "LiquidityStatus": st_df["LiquidityStatus"].to_numpy(),
            "IsListed": is_listed,
            "IsHistoricallyListed": st_df["IsHistoricallyListed"].fillna(False).to_numpy(dtype=bool),
            "IsHistoricallyTradable": is_trd,
            "IsLifecycleValidated": is_life,
            "IsPriceDataValid": is_px,
            "IsCorporateActionValidated": is_ca,
            "IsEntryHazardFlagged": ca_bad_entry | src_err_entry,
            "IsShortTermEligible": elig_st,
            "IsOpenFeatureAvailable": st_df["IsOpenFeatureAvailable"].fillna(False).to_numpy(dtype=bool),
            "IsPennyStockTickNoiseRisk": penny_risk.astype(bool),
            "IsTargetEligible_TR_5d": tgt_all_df["IsTargetEligible_TR_5d"].to_numpy(dtype=bool),
            "IsTargetEligible_TR_10d": tgt_all_df["IsTargetEligible_TR_10d"].to_numpy(dtype=bool),
            "IsTargetEligible_TR_20d": tgt_all_df["IsTargetEligible_TR_20d"].to_numpy(dtype=bool),
            "IsTargetEligible_TR_60d": tgt_all_df["IsTargetEligible_TR_60d"].to_numpy(dtype=bool),
            "IsTargetEligible_Cap_5d": tgt_all_df["IsTargetEligible_Cap_5d"].to_numpy(dtype=bool),
            "IsTargetEligible_Cap_10d": tgt_all_df["IsTargetEligible_Cap_10d"].to_numpy(dtype=bool),
            "IsTargetEligible_Cap_20d": tgt_all_df["IsTargetEligible_Cap_20d"].to_numpy(dtype=bool),
            "IsTargetEligible_Cap_60d": tgt_all_df["IsTargetEligible_Cap_60d"].to_numpy(dtype=bool),
            "TargetExclusionReason_20d": tgt_all_df["TargetExclusionReason_20d"].to_numpy(),
            "PrimaryExclusionReason": reason,
            "EraSplitGroup": split_full,
        },
        index=st_df.index,
    )

    # Verify exact partition against IsShortTermEligible
    assert int((sample_elig["PrimaryExclusionReason"] == "ELIGIBLE_SHORT_TERM_CORE").sum()) == int(elig_st.sum())

    audit_rows: list[dict[str, Any]] = []
    tot = len(sample_elig)
    for (r_code, l_status, listed_val), grp in sample_elig.groupby(["PrimaryExclusionReason", "LiquidityStatus", "IsListed"], dropna=False):
        audit_rows.append(
            {
                "AuditScope": "SAMPLE_ELIGIBILITY_WATERFALL",
                "ReasonCode": str(r_code),
                "LiquidityStatus": str(l_status),
                "ListingStatus": "Listed" if bool(listed_val) else "Delisted",
                "RowCount": int(len(grp)),
                "ShareOfTotalRowsPct": round(100.0 * len(grp) / tot, 4),
                "UniqueSecurityCount": int(grp["SecurityID"].nunique()),
                "MinDate": str(grp["Date"].min().date()),
                "MaxDate": str(grp["Date"].max().date()),
            }
        )

    # Also append target-horizon exclusion breakdown across IsShortTermEligible rows
    st_only = sample_elig[sample_elig["IsShortTermEligible"]]
    tot_st = len(st_only)
    for h in HORIZONS:
        col_r = f"TargetExclusionReason_{h}d"
        ser_r = tgt_all_df.loc[st_only.index, col_r]
        for (rc, listed_val), g_idx in st_only.assign(_rc=ser_r.to_numpy()).groupby(["_rc", "IsListed"]):
            audit_rows.append(
                {
                    "AuditScope": f"TARGET_HORIZON_{h}D",
                    "ReasonCode": str(rc),
                    "LiquidityStatus": "LIQUID_PRIME_SHORT_TERM",
                    "ListingStatus": "Listed" if bool(listed_val) else "Delisted",
                    "RowCount": int(len(g_idx)),
                    "ShareOfTotalRowsPct": round(100.0 * len(g_idx) / max(1, tot_st), 4),
                    "UniqueSecurityCount": int(g_idx["SecurityID"].nunique()),
                    "MinDate": str(g_idx["Date"].min().date()),
                    "MaxDate": str(g_idx["Date"].max().date()),
                }
            )

    excl_audit_df = pd.DataFrame(audit_rows).sort_values(["AuditScope", "RowCount"], ascending=[True, False]).reset_index(drop=True)

    waterfall_summary = {
        "total_universe_rows": int(tot),
        "short_term_eligible_rows": int(tot_st),
        "short_term_eligible_securities": int(st_only["SecurityID"].nunique()),
        "short_term_eligible_listed_securities": int(st_only.loc[st_only["IsListed"], "SecurityID"].nunique()),
        "short_term_eligible_listed_rows": int(st_only["IsListed"].sum()),
        "short_term_eligible_delisted_securities": int(st_only.loc[~st_only["IsListed"], "SecurityID"].nunique()),
        "short_term_eligible_delisted_rows": int((~st_only["IsListed"]).sum()),
        "penny_sub1_lkr_rows_in_eligible": int(st_only["IsPennyStockTickNoiseRisk"].sum()),
        "open_2017plus_rows_in_eligible": int(st_only["IsOpenFeatureAvailable"].sum()),
        "primary_reason_counts": sample_elig["PrimaryExclusionReason"].value_counts().to_dict(),
    }
    return sample_elig, excl_audit_df, waterfall_summary


def build_short_term_feature_registry(
    model_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    phase3d_cols: list[str],
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    """Build full Phase 4A feature registry with dev-only missingness & Spearman redundancy pruning."""
    dev_mask = (splits_df["FinalSplitRole"] == "DEV_POOL").to_numpy()
    open_dev_mask = dev_mask & (model_df["IsOpenFeatureAvailable"].fillna(False).to_numpy(dtype=bool))
    ft_mask = (splits_df["FinalSplitRole"] == "FINAL_TEST").to_numpy()

    all_candidate_features = CORE_FULL_SAMPLE_TECH_FEATURES + DERIVED_SCALE_FREE_FEATURES + OPEN_REGIME_2017PLUS_FEATURES + PIT_FUNDAMENTAL_FEATURES

    # Strictly compute pairwise Spearman redundancy ONLY on DEV_POOL (never FINAL_TEST!)
    dev_tech = model_df.loc[dev_mask, CORE_FULL_SAMPLE_TECH_FEATURES + DERIVED_SCALE_FREE_FEATURES]
    corr_matrix = dev_tech.corr(method="spearman")
    cols_c = list(corr_matrix.columns)
    corr_vals = corr_matrix.to_numpy()

    high_corr_pairs: list[dict[str, Any]] = []
    redundant_drop_set: set[str] = set()
    # Canonical preference ordering when picking representative out of collinear pair
    prefer_keep = {
        "Daily_Return_CapAdj", "ROC_10", "NATR_14", "BB_PctB_20_2",
        "Price_to_SMA20_Pct", "Log_Rolling_MedianTurnover_60", "Rolling_TradingFreqPct_60",
    }
    for i in range(len(cols_c)):
        for j in range(i + 1, len(cols_c)):
            r_val = corr_vals[i, j]
            if np.isfinite(r_val) and abs(r_val) >= 0.90:
                c1, c2 = cols_c[i], cols_c[j]
                high_corr_pairs.append({"feature_a": c1, "feature_b": c2, "spearman_r_dev": round(float(r_val), 4)})
                # Deterministic drop rule favoring preferred canonical ratios
                if c1 in prefer_keep and c2 not in prefer_keep:
                    redundant_drop_set.add(c2)
                elif c2 in prefer_keep and c1 not in prefer_keep:
                    redundant_drop_set.add(c1)
                else:
                    redundant_drop_set.add(c2)

    # Also prune highly redundant PIT flags (`IsFundamentalExpired` ~= `IsFundamentalStale`, `company_pb_ratio` ~= `sec_pb_ratio`)
    redundant_fund_drop = {"IsFundamentalExpired", "company_pb_ratio", "IsFundamentalStale_365d"}
    redundant_drop_set.update(redundant_fund_drop)

    core_pruned_full = [c for c in (CORE_FULL_SAMPLE_TECH_FEATURES + DERIVED_SCALE_FREE_FEATURES) if c not in redundant_drop_set]
    core_pruned_with_fund = core_pruned_full + [c for c in PIT_FUNDAMENTAL_FEATURES if c not in redundant_drop_set]
    post2017_pruned = core_pruned_with_fund + OPEN_REGIME_2017PLUS_FEATURES

    registry_entries: dict[str, Any] = {}
    for col in sorted(set(phase3d_cols) | set(model_df.columns)):
        if col in CORE_FULL_SAMPLE_TECH_FEATURES:
            role = "MODEL_FEATURE_CORE_FULL_SAMPLE"
            origin = "phase3d_technical_or_market_rel"
        elif col in DERIVED_SCALE_FREE_FEATURES:
            role = "MODEL_FEATURE_DERIVED_SCALE_FREE"
            origin = "phase4a_causal_close_ratio_or_log"
        elif col in OPEN_REGIME_2017PLUS_FEATURES:
            role = "MODEL_FEATURE_2017PLUS_OPEN_REGIME"
            origin = "phase3d_or_phase4a_open_regime"
        elif col in PIT_FUNDAMENTAL_FEATURES:
            role = "MODEL_FEATURE_PIT_FUNDAMENTAL"
            origin = "phase3d_pit_fundamental"
        elif col in EXCLUDED_PRICE_LEVEL_OR_SCALE_DEPS:
            role = "EXCLUDED_LEVEL_OR_SCALE_CONFOUNDED"
            origin = "phase3d_excluded"
        elif col in EXCLUDED_CONSTANT_OR_ELIGIBILITY_FLAGS:
            role = "EXCLUDED_CONSTANT_OR_ELIGIBILITY_GATE"
            origin = "phase3d_gate"
        elif col in IDENTIFIER_AND_METADATA_COLS or col.startswith("_"):
            role = "IDENTIFIER_OR_METADATA"
            origin = "phase3d_metadata"
        elif col.startswith("Target") or col.startswith("Label_") or col.startswith("IsTargetEligible") or col.startswith("CrossSectionN"):
            role = "TARGET_OR_LABEL_DO_NOT_USE_AS_FEATURE"
            origin = "phase4a_forward_target"
        elif col in {"FinalSplitRole", "EraSplitGroup", "CV_Fold_1", "CV_Fold_2", "CV_Fold_3", "CV_Fold_4", "IsListed", "IsPennyStockTickNoiseRisk", "IsCleanCorePool"}:
            role = "SPLIT_OR_REGIME_AUDIT_FLAG"
            origin = "phase4a_split_or_audit"
        else:
            role = "OTHER_AUXILIARY_EXCLUDED"
            origin = "auxiliary"

        rec: dict[str, Any] = {
            "role": role,
            "origin": origin,
            "is_model_candidate": col in all_candidate_features,
            "in_pruned_low_collinearity_set": (col in core_pruned_full) or (col in core_pruned_with_fund),
            "redundant_pruned_in_linear_set": col in redundant_drop_set,
        }
        if col in EXCLUDED_PRICE_LEVEL_OR_SCALE_DEPS:
            rec.update(EXCLUDED_PRICE_LEVEL_OR_SCALE_DEPS[col])
        if col in model_df.columns and col in all_candidate_features:
            s_dev = pd.to_numeric(model_df.loc[dev_mask, col], errors="coerce")
            s_ft = pd.to_numeric(model_df.loc[ft_mask, col], errors="coerce")
            rec["dev_missing_rate"] = round(float(s_dev.isna().mean()), 5)
            rec["final_test_missing_rate"] = round(float(s_ft.isna().mean()), 5)
            rec["dev_mean"] = round(float(s_dev.mean()), 6) if s_dev.notna().any() else None
            rec["dev_std"] = round(float(s_dev.std()), 6) if s_dev.notna().any() else None
            if col in OPEN_REGIME_2017PLUS_FEATURES:
                s_open = pd.to_numeric(model_df.loc[open_dev_mask, col], errors="coerce")
                rec["dev_2017plus_missing_rate"] = round(float(s_open.isna().mean()), 5)
        registry_entries[col] = rec

    registry = {
        "schema_version": "4A.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "correlation_audit_split_scope": "DEV_POOL_ONLY (2002-01-02 to 2023-09-26, strictly zero FINAL_TEST peeking)",
        "indicator_price_basis_verification": (
            "Verified empirically against JKH-N-0000 pre-rights/bonus history (2019-12): Phase 2A SMA_20/MOM_10/VWAP_20/BB/ATR "
            "align to split-adjusted Close (max error ~2e-12) rather than CapitalAdjClose (0.102 LKR drift). All derived "
            "ratio features therefore normalize strictly by Close."
        ),
        "feature_set_counts": {
            "total_candidate_features": len(all_candidate_features),
            "core_full_sample_technical": len(CORE_FULL_SAMPLE_TECH_FEATURES),
            "derived_scale_free": len(DERIVED_SCALE_FREE_FEATURES),
            "open_regime_2017plus": len(OPEN_REGIME_2017PLUS_FEATURES),
            "pit_fundamental": len(PIT_FUNDAMENTAL_FEATURES),
            "pruned_core_technical_set": len(core_pruned_full),
            "pruned_core_plus_pit_fundamentals_set": len(core_pruned_with_fund),
            "post_2017_full_set": len(post2017_pruned),
        },
        "recommended_feature_sets": {
            "core_tech_full_sample_2002_2025": CORE_FULL_SAMPLE_TECH_FEATURES + DERIVED_SCALE_FREE_FEATURES,
            "pruned_core_tech_low_collinearity": core_pruned_full,
            "core_plus_pit_fundamentals_pruned": core_pruned_with_fund,
            "post_2017_with_open_regime": post2017_pruned,
            "open_regime_only_features": OPEN_REGIME_2017PLUS_FEATURES,
        },
        "preprocessing_contract": {
            "imputation_policy": (
                "NEVER fit median/mean/quantile scalers on full dataset. Fit imputer + RobustScaler/winsorizer strictly "
                "inside each CV fold TRAIN slice (or DEV_POOL before FINAL_TEST). For PIT fundamentals & 2017+ Open features: "
                "(a) LightGBM/XGBoost/CatBoost tree pipelines should pass native NaNs alongside binary regime indicators; "
                "(b) Linear/Ridge/Neural pipelines should either restrict to 2017+ or fit within-fold median + missingness indicator."
            ),
            "cross_sectional_rank_optional_policy": (
                "Because all retained features are already scale-free ratios, returns, bounded oscillators, or log-scale "
                "volumes/mcap, models can operate directly on within-fold scaled features OR apply daily cross-sectional "
                "z-score/percentile transforms (purely within date t, zero cross-date leakage)."
            ),
            "sector_feature_disposition": "Excluded (`IsSectorMappingAvailable == False` across 100% of ST universe in Phase 3D).",
        },
        "dev_high_correlation_pairs_spearman_ge_0_90": high_corr_pairs,
        "columns": registry_entries,
    }
    return registry, all_candidate_features, core_pruned_full, post2017_pruned


def evaluate_sensitivity_and_target_quality(
    st_df: pd.DataFrame,
    model_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    target_summary: dict[str, Any],
    waterfall_summary: dict[str, Any],
    all_listed_secs: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run liquidity/penny threshold sensitivity matrix, tail-event audit, and target quality report."""
    # Evaluate liquidity & price sensitivity strictly on Phase 2C base-valid ST-ready universe slice (consec==0, is_eq, is_px, is_ca)
    elig_mask = st_df["IsShortTermEligible"].fillna(False).to_numpy()
    raw_px_elig = pd.to_numeric(model_df["RawClose"], errors="coerce").to_numpy(dtype=np.float64)
    med60_elig = np.round(np.expm1(pd.to_numeric(model_df["Log_Rolling_MedianTurnover_60"], errors="coerce").to_numpy(dtype=np.float64)), 2)
    freq60_elig = pd.to_numeric(model_df["Rolling_TradingFreqPct_60"], errors="coerce").to_numpy(dtype=np.float64)
    ad20_elig = pd.to_numeric(model_df["Rolling_ActiveDays_20"], errors="coerce").to_numpy(dtype=np.float64)
    tr20_ret = model_df["Target_Excess_TotalReturn_20d"].to_numpy(dtype=np.float64)
    tr20_ok = model_df["IsTargetEligible_TR_20d"].to_numpy(dtype=bool)
    ds20 = pd.to_numeric(model_df["Label_Fwd_DownsideRisk_Flag_20d"], errors="coerce").to_numpy(dtype=np.float64)
    dates_elig = pd.to_datetime(model_df["Date"]).dt.normalize()
    secs_elig = model_df["SecurityID"].to_numpy()
    listed_elig = model_df["IsListed"].to_numpy(dtype=bool)

    sensitivity_grid = [
        {"scenario": "Baseline_Phase2C (MedTurn60>=500k, Freq>=70%, AllRawClose)", "min_med60": 500_000, "min_freq60": 70.0, "min_raw_px": 0.0, "min_ad20": 20},
        {"scenario": "Baseline_ExclSub1LKR (MedTurn60>=500k, Freq>=70%, RawClose>=1.0)", "min_med60": 500_000, "min_freq60": 70.0, "min_raw_px": 1.0, "min_ad20": 20},
        {"scenario": "Strict_Excl2LKR (MedTurn60>=500k, Freq>=70%, RawClose>=2.0)", "min_med60": 500_000, "min_freq60": 70.0, "min_raw_px": 2.0, "min_ad20": 20},
        {"scenario": "HighLiq_1M_AllPx (MedTurn60>=1M, Freq>=75%, AllRawClose)", "min_med60": 1_000_000, "min_freq60": 75.0, "min_raw_px": 0.0, "min_ad20": 20},
        {"scenario": "HighLiq_1M_ExclSub1LKR (MedTurn60>=1M, Freq>=75%, RawClose>=1.0)", "min_med60": 1_000_000, "min_freq60": 75.0, "min_raw_px": 1.0, "min_ad20": 20},
        {"scenario": "Institutional_2.5M_ExclSub1LKR (MedTurn60>=2.5M, Freq>=80%, RawClose>=1.0)", "min_med60": 2_500_000, "min_freq60": 80.0, "min_raw_px": 1.0, "min_ad20": 20},
    ]

    # Also test relaxed broader thresholds from full phase3d universe (`MedTurn60>=100k` or `250k`)
    base_full_valid = (
        st_df["IsEquityShareClass"].fillna(False).to_numpy()
        & st_df["IsHistoricallyTradable"].fillna(False).to_numpy()
        & st_df["IsLifecycleValidated"].fillna(False).to_numpy()
        & st_df["IsPriceDataValid"].fillna(False).to_numpy()
        & st_df["IsCorporateActionValidated"].fillna(False).to_numpy()
        & (pd.to_numeric(st_df["ConsecutiveNonActiveDays"], errors="coerce").fillna(99).to_numpy() == 0)
        & (pd.to_numeric(st_df["Rolling_ActiveDays_20"], errors="coerce").fillna(0).to_numpy() >= 18)
        & (pd.to_numeric(st_df["Rolling_ActiveDays_60"], errors="coerce").fillna(0).to_numpy() >= 54)
    )
    med60_full = pd.to_numeric(st_df["Rolling_MedianTurnover_60_LKR"], errors="coerce").fillna(0.0).to_numpy()
    freq60_full = pd.to_numeric(st_df["Rolling_TradingFreqPct_60"], errors="coerce").fillna(0.0).to_numpy()

    sensitivity_results: list[dict[str, Any]] = []
    for name_r, min_m, min_f in [("Relaxed_100k_Freq60", 100_000, 60.0), ("Relaxed_250k_Freq65", 250_000, 65.0)]:
        m_r = base_full_valid & (med60_full >= min_m) & (freq60_full >= min_f)
        dt_cnt = pd.Series(m_r).groupby(pd.to_datetime(st_df["Date"]).dt.normalize()).sum()
        pos_cnt = dt_cnt[dt_cnt > 0]
        sensitivity_results.append(
            {
                "scenario": name_r,
                "row_count": int(m_r.sum()),
                "unique_securities": int(st_df.loc[m_r, "SecurityID"].nunique()),
                "delisted_rows": int(st_df.loc[m_r, "SecurityID"].nunique()),  # sec count vs row below
                "active_dates": int(len(pos_cnt)),
                "median_daily_cross_section": int(pos_cnt.median()) if len(pos_cnt) else 0,
                "pct_dates_ge_15": round(100.0 * float((pos_cnt >= MIN_CROSS_SECTION_SIZE).mean()), 2) if len(pos_cnt) else 0.0,
                "tr20_std": None,
                "tr20_p99_p01_spread": None,
                "downside20_rate": None,
            }
        )
    # Fix delisted_rows accurately on relaxed rows across ALL CSE securities
    listed_set = all_listed_secs if all_listed_secs is not None else set(model_df.loc[model_df["IsListed"], "SecurityID"].astype(str))
    for i, (min_m, min_f) in enumerate([(100_000, 60.0), (250_000, 65.0)]):
        m_r = base_full_valid & (med60_full >= min_m) & (freq60_full >= min_f)
        sensitivity_results[i]["delisted_rows"] = int((m_r & (~st_df["SecurityID"].astype(str).isin(listed_set).to_numpy())).sum())

    for g in sensitivity_grid:
        m = elig_mask & True  # unused helper readability
        m_g = (
            (med60_elig >= g["min_med60"])
            & (freq60_elig >= g["min_freq60"])
            & (raw_px_elig >= g["min_raw_px"])
            & (ad20_elig >= g["min_ad20"])
        )
        m_tr = m_g & tr20_ok & np.isfinite(tr20_ret)
        dt_cnt = pd.Series(m_g).groupby(dates_elig).sum()
        pos_cnt = dt_cnt[dt_cnt > 0]
        r_slice = tr20_ret[m_tr]
        ds_slice = ds20[m_g & np.isfinite(ds20)]
        sensitivity_results.append(
            {
                "scenario": g["scenario"],
                "row_count": int(m_g.sum()),
                "unique_securities": int(pd.Series(secs_elig[m_g]).nunique()),
                "delisted_rows": int((m_g & (~listed_elig)).sum()),
                "active_dates": int(len(pos_cnt)),
                "median_daily_cross_section": int(pos_cnt.median()) if len(pos_cnt) else 0,
                "pct_dates_ge_15": round(100.0 * float((pos_cnt >= MIN_CROSS_SECTION_SIZE).mean()), 2) if len(pos_cnt) else 0.0,
                "tr20_std": round(float(np.std(r_slice)), 6) if len(r_slice) else None,
                "tr20_p99_p01_spread": round(float(np.quantile(r_slice, 0.99) - np.quantile(r_slice, 0.01)), 6) if len(r_slice) else None,
                "downside20_rate": round(float(np.mean(ds_slice)), 4) if len(ds_slice) else None,
            }
        )

    # Penny-stock noise & tail decomposition on 20d Clean TR
    penny_mask = raw_px_elig < PENNY_CLOSE_THRESHOLD_LKR
    tr_ok_all = tr20_ok & np.isfinite(tr20_ret)
    penny_tr = tr20_ret[tr_ok_all & penny_mask]
    core_tr = tr20_ret[tr_ok_all & (~penny_mask)]

    # Daily cross-section percentiles across eligible sample
    daily_cs_all = pd.Series(np.ones(len(model_df), dtype=np.int32)).groupby(dates_elig).sum()
    daily_cs_tr20 = pd.Series(tr20_ok.astype(np.int32)).groupby(dates_elig).sum()
    daily_cs_tr20_pos = daily_cs_tr20[daily_cs_tr20 > 0]

    # Yearly breakdown
    yearly_stats: list[dict[str, Any]] = []
    for yr, grp in model_df.assign(_yr=dates_elig.dt.year).groupby("_yr"):
        n_d = int(grp["Date"].nunique())
        cs_yr = grp.groupby("Date")["SecurityID"].count()
        yearly_stats.append(
            {
                "year": int(yr),
                "rows": int(len(grp)),
                "unique_securities": int(grp["SecurityID"].nunique()),
                "delisted_securities": int(grp.loc[~grp["IsListed"], "SecurityID"].nunique()),
                "open_avail_pct": round(100.0 * float(grp["IsOpenFeatureAvailable"].mean()), 2),
                "penny_sub1_pct": round(100.0 * float((grp["RawClose"] < PENNY_CLOSE_THRESHOLD_LKR).mean()), 2),
                "tr20_valid_pct": round(100.0 * float(grp["IsTargetEligible_TR_20d"].mean()), 2),
                "median_daily_cs": int(cs_yr.median()),
                "days_lt_15_cs": int((cs_yr < MIN_CROSS_SECTION_SIZE).sum()),
            }
        )

    target_definitions = {
        "schema_version": "4A.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons_trading_sessions": list(HORIZONS),
        "primary_short_term_benchmark": "ASTRI (All Share Total Return Index, CSE official TRI 2004-01-02..2025-12-31)",
        "secondary_price_benchmark": "ASPI (All Share Price Index, full historical 1990..2025)",
        "targets": {
            "Target_Excess_TotalReturn_{h}d": {
                "formula": "(TotalReturnPrice[t+h] / TotalReturnPrice[t] - 1) - (ASTRI[t+h] / ASTRI[t] - 1)",
                "role": "Primary continuous ranking / regression target (dividend-inclusive excess alpha)",
                "validity_gate": "IsTargetEligible_TR_{h}d == True (exact SessionIndex + h, valid entry/exit prices, zero quarantined CA or source error in [t, t+h], 100% resolved dividend path, valid ASTRI at t and t+h)",
            },
            "Target_Excess_CapitalReturn_{h}d": {
                "formula": "(CapitalAdjClose[t+h] / CapitalAdjClose[t] - 1) - (ASPI[t+h] / ASPI[t] - 1)",
                "role": "Secondary/fallback price-only target covering 2002-2003 pre-ASTRI and unresolved-cash-dividend paths without polluting TotalReturn",
                "validity_gate": "IsTargetEligible_Cap_{h}d == True",
            },
            "Target_ExcessTR_PctRank_{h}d": {
                "formula": "Cross-sectional percentile rank within Date t among IsShortTermEligible & IsTargetEligible_TR_{h}d stocks where daily CrossSectionN_TR_{h}d >= 15",
                "role": "Primary cross-sectional ranking target immune to market-regime mean shifts and heavy outlier returns",
            },
            "Target_TopQuintile_ExcessTR_{h}d": {
                "formula": "1 if Target_ExcessTR_PctRank_{h}d >= 0.80 else 0 (requires CrossSectionN >= 15)",
                "role": "Primary classification target for top-20% outperformer selection",
            },
            "Target_Positive_ExcessTR_{h}d": {
                "formula": "1 if Target_Excess_TotalReturn_{h}d > 0.0 else 0",
                "role": "Binary benchmark-outperformance target",
            },
            "Label_Fwd_DownsideRisk_Flag_{h}d": {
                "formula": "1 if (MaxAdverseExcursion[t+1..t+h] <= -0.10 OR MaxDrawdown[t..t+h] <= -0.12) else 0",
                "role": "Strict future-path tail-risk / drawdown auxiliary label (NEVER used as a feature)",
            },
            "Label_Fwd_RealizedVol_Ann_{h}d": {
                "formula": "std(Daily_Return_CapAdj / 100.0 over (t, t+h], ddof=1) * sqrt(252)",
                "role": "Future annualized return volatility auxiliary label in decimal units (e.g. 0.34 = 34% ann. vol; NEVER used as a feature)",
            },
        },
        "critical_target_engineering_rules": [
            "Never use DataFrame.shift(-h): CSE securities skip non-traded sessions, so shifts must map exact SessionIndex + h on cse_market_calendar.",
            "Never forward-fill future delisted/halted prices beyond LastActivePriceDate or across halts.",
            "Never replace missing pre-2004 or unresolved-dividend TotalReturn targets with CapitalReturn; keep separate IsTargetEligible_TR and IsTargetEligible_Cap masks.",
            "Quarantined corporate-action events (76 QUARANTINED / 17 VERIFIED_CORRECTION_REQUIRED) and SOURCE_DATA_ERROR / SECURITY_IDENTITY_ISSUE windows are strictly masked out of target windows.",
        ],
    }

    target_quality_report = {
        "schema_version": "4A.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility_waterfall_summary": waterfall_summary,
        "horizon_target_statistics": target_summary,
        "cross_section_distribution": {
            "total_eligible_dates": int(len(daily_cs_all)),
            "eligible_cs_percentiles": {
                "min": int(daily_cs_all.min()),
                "p05": int(daily_cs_all.quantile(0.05)),
                "p10": int(daily_cs_all.quantile(0.10)),
                "p25": int(daily_cs_all.quantile(0.25)),
                "median": int(daily_cs_all.median()),
                "p75": int(daily_cs_all.quantile(0.75)),
                "p90": int(daily_cs_all.quantile(0.90)),
                "max": int(daily_cs_all.max()),
            },
            "eligible_days_below_15_securities": int((daily_cs_all < MIN_CROSS_SECTION_SIZE).sum()),
            "tr20_valid_days_ge_15_securities": int((daily_cs_tr20_pos >= MIN_CROSS_SECTION_SIZE).sum()),
        },
        "penny_stock_tick_noise_audit_20d": {
            "threshold_raw_close_lkr": PENNY_CLOSE_THRESHOLD_LKR,
            "sub1_lkr_rows": int(len(penny_tr)),
            "sub1_lkr_std": round(float(np.std(penny_tr)), 6) if len(penny_tr) else None,
            "sub1_lkr_p99": round(float(np.quantile(penny_tr, 0.99)), 6) if len(penny_tr) else None,
            "sub1_lkr_max": round(float(np.max(penny_tr)), 6) if len(penny_tr) else None,
            "core_ge1_lkr_rows": int(len(core_tr)),
            "core_ge1_lkr_std": round(float(np.std(core_tr)), 6) if len(core_tr) else None,
            "core_ge1_lkr_p01": round(float(np.quantile(core_tr, 0.01)), 6) if len(core_tr) else None,
            "core_ge1_lkr_p99": round(float(np.quantile(core_tr, 0.99)), 6) if len(core_tr) else None,
            "core_ge1_lkr_max": round(float(np.max(core_tr)), 6) if len(core_tr) else None,
            "extreme_gt_100pct_in_sub1": int((penny_tr > 1.0).sum()) if len(penny_tr) else 0,
            "extreme_gt_100pct_in_core": int((core_tr > 1.0).sum()) if len(core_tr) else 0,
        },
        "threshold_sensitivity_matrix": sensitivity_results,
        "yearly_eligible_profile": yearly_stats,
    }
    return target_definitions, target_quality_report


def run_phase4a_leakage_and_perturbation_tests(
    model_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    split_summary: dict[str, Any],
    feature_cols: list[str],
    raw_st_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Execute strict programmatic leakage, perturbation, PIT, and split-isolation tests."""
    # 1. Contemporaneous future-price perturbation test: perturb target/price cols strictly at t > cutoff on real subset
    #    and verify build_derived_scale_free_features on rows <= cutoff is bitwise identical!
    if raw_st_df is not None:
        sub = raw_st_df.head(15000).copy()
        if "SessionIndex" not in sub.columns:
            sub = sub.merge(model_df[["Date", "SecurityID", "SessionIndex"]], on=["Date", "SecurityID"], how="inner")
    else:
        # Standalone dataset verification path (when raw Phase 3D table isn't loaded, e.g., pytest):
        sub = model_df.head(5000).copy()
        rng = np.random.default_rng(42)
        c_arr = pd.to_numeric(sub["Close"], errors="coerce").fillna(50.0).to_numpy()
        for col_m in ("SMA_20", "SMA_50", "EMA_12", "EMA_26", "WMA_20", "VWAP_20", "Supertrend_10_3", "BB_Upper_20_2", "BB_Lower_20_2"):
            sub[col_m] = c_arr * (1.0 + rng.normal(0.0, 0.02, len(sub)))
        for col_d in ("MACD_Line", "MACD_Signal", "MACD_Hist", "MOM_10", "ATR_14"):
            sub[col_d] = c_arr * rng.uniform(-0.05, 0.05, len(sub))
        sub["ShareVolume"] = np.expm1(pd.to_numeric(sub["Log_ShareVolume"], errors="coerce").fillna(8.0).to_numpy())
        sub["Turnover"] = np.expm1(pd.to_numeric(sub["Log_Turnover_LKR"], errors="coerce").fillna(12.0).to_numpy())
        sub["Rolling_MeanTurnover_20_LKR"] = np.expm1(pd.to_numeric(sub["Log_Rolling_MeanTurnover_20"], errors="coerce").fillna(13.0).to_numpy())
        sub["Rolling_MedianTurnover_60_LKR"] = np.expm1(pd.to_numeric(sub["Log_Rolling_MedianTurnover_60"], errors="coerce").fillna(13.0).to_numpy())
        sub["sec_market_cap_lkr"] = np.expm1(pd.to_numeric(sub["Log_MarketCap_LKR"], errors="coerce").fillna(20.0).to_numpy())

    cutoff_pos = int(sub["SessionIndex"].median())
    pre_mask = sub["SessionIndex"] <= cutoff_pos
    fut_mask = sub["SessionIndex"] > cutoff_pos
    assert pre_mask.any() and fut_mask.any(), "Perturbation slice must have both pre and future rows"

    base_derived = build_derived_scale_free_features(sub)
    sub_pert = sub.copy()
    for pcol in ["RawClose", "Close", "CapitalAdjClose", "TotalReturnPrice", "ASPI_Close", "Turnover", "SMA_20", "ShareVolume", "sec_market_cap_lkr"]:
        if pcol in sub_pert.columns:
            sub_pert.loc[fut_mask, pcol] = sub_pert.loc[fut_mask, pcol] * 9.5 + 100.0
    pert_derived = build_derived_scale_free_features(sub_pert)

    max_feature_diff = 0.0
    for c in pert_derived.columns:
        v1 = pd.to_numeric(base_derived.loc[pre_mask, c], errors="coerce").to_numpy(dtype=np.float64)
        v2 = pd.to_numeric(pert_derived.loc[pre_mask, c], errors="coerce").to_numpy(dtype=np.float64)
        both_nan = np.isnan(v1) & np.isnan(v2)
        assert np.array_equal(np.isnan(v1), np.isnan(v2)), f"NaN mask drift under future perturbation on {c}"
        if (~both_nan).any():
            diff = float(np.max(np.abs(v1[~both_nan] - v2[~both_nan])))
            max_feature_diff = max(max_feature_diff, diff)
    assert max_feature_diff == 0.0, f"Feature leaked future values: max diff={max_feature_diff}"

    # 2. Target causal ordering check across all 4 horizons
    for h in HORIZONS:
        m = model_df[f"IsTargetEligible_Cap_{h}d"].to_numpy(dtype=bool)
        if m.any():
            assert (model_df.loc[m, f"TargetDate_{h}d"] > model_df.loc[m, "Date"]).all(), f"Non-causal TargetDate_{h}d"

    # 3. Verify PIT fundamental availability date <= Decision Date everywhere non-null
    pit_avail = pd.to_datetime(model_df["FundamentalAvailabilityDate"], errors="coerce")
    has_pit = pit_avail.notna()
    assert (pit_avail[has_pit] <= pd.to_datetime(model_df.loc[has_pit, "Date"])).all(), "PIT violation in FundamentalAvailabilityDate!"

    # 4. Verify zero overlap & 65-session separation across CV folds & FINAL_TEST
    for f in split_summary["cv_folds"]:
        assert f["zero_target60d_overlap_with_val"] is True, f"Target overlap violated in {f['fold_name']}"
        assert (f["val_min_session_index"] - f["train_max_session_index"] - 1) >= TOTAL_GAP_SESSIONS
    assert split_summary["final_holdout_split"]["zero_dev_target60d_overlap_with_final_test"] is True

    # 5. Verify feature list excludes all future targets/labels and level-confounded series
    forbidden = [c for c in feature_cols if c.startswith("Target") or c.startswith("Label_") or c in EXCLUDED_PRICE_LEVEL_OR_SCALE_DEPS]
    assert len(forbidden) == 0, f"Forbidden columns in candidate features: {forbidden}"

    return {
        "future_perturbation_max_diff": max_feature_diff,
        "perturbation_rows_tested": int(pre_mask.sum()),
        "all_target_dates_strictly_future": True,
        "pit_availability_le_decision_date_verified": True,
        "all_cv_and_final_splits_65session_purged": True,
        "zero_forbidden_columns_in_features": True,
    }


def generate_phase4a_visualizations(
    viz_dir: Path,
    sample_elig: pd.DataFrame,
    model_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    split_summary: dict[str, Any],
    tq_report: dict[str, Any],
    feature_registry: dict[str, Any],
) -> list[str]:
    """Generate 4 publication-grade PNG diagnostics for Phase 4A."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8})
    paths: list[str] = []

    # 1. Eligibility waterfall & yearly active cross-section evolution
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    rc = pd.Series(tq_report["eligibility_waterfall_summary"]["primary_reason_counts"]).sort_values(ascending=True)
    colors = ["#0f766e" if k == "ELIGIBLE_SHORT_TERM_CORE" else "#64748b" for k in rc.index]
    axes[0].barh(rc.index, rc.values, color=colors)
    for i, v in enumerate(rc.values):
        axes[0].text(v + 12000, i, f"{v:,} ({100*v/len(sample_elig):.1f}%)", va="center", fontsize=8.5)
    axes[0].set_xlim(0, rc.max() * 1.32)
    axes[0].set_title("1.415M CSE Record Eligibility Waterfall", fontweight="bold")
    axes[0].set_xlabel("Historical Daily Rows")
    axes[0].tick_params(axis="y", labelsize=8.5)

    yr_df = pd.DataFrame(tq_report["yearly_eligible_profile"])
    ax1 = axes[1]
    ax1.bar(yr_df["year"], yr_df["median_daily_cs"], color="#2563eb", alpha=0.82, label="Median Daily Cross-Section N")
    ax1.axhline(MIN_CROSS_SECTION_SIZE, color="#dc2626", linestyle="--", linewidth=1.3, label=f"Cross-Section Floor (N={MIN_CROSS_SECTION_SIZE})")
    ax1.axvline(2016.5, color="#d97706", linestyle=":", linewidth=1.5, label="Open/Intraday Regime (2017+)")
    ax1.set_title("Eligible Daily Cross-Section Median (2002–2025)", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Median Daily Eligible Securities")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p1 = viz_dir / "01_eligibility_waterfall_and_universe_evolution.png"
    fig.savefig(p1)
    plt.close(fig)
    paths.append(str(p1))

    # 2. Forward horizon distributions & coverage breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    h_stats = tq_report["horizon_target_statistics"]
    hs = [f"{h}d" for h in HORIZONS]
    tr_cnts = [h_stats[h]["valid_tr_excess_rows"] for h in hs]
    cap_only = [h_stats[h]["valid_cap_excess_rows"] - h_stats[h]["valid_tr_excess_rows"] for h in hs]
    excluded = [h_stats[h]["st_eligible_base_rows"] - h_stats[h]["valid_cap_excess_rows"] for h in hs]
    x = np.arange(len(hs))
    axes[0].bar(x, tr_cnts, label="Clean Total Return (ASTRI)", color="#0f766e")
    axes[0].bar(x, cap_only, bottom=tr_cnts, label="Cap-Only (Pre-2004 / Unresolved Div)", color="#f59e0b")
    axes[0].bar(x, excluded, bottom=np.array(tr_cnts) + np.array(cap_only), label="Excluded (EoS / Halt / CA-Quarantine)", color="#94a3b8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{h} Sessions" for h in HORIZONS])
    axes[0].set_title("Forward Target Availability Across Short-Term Horizons", fontweight="bold")
    axes[0].set_ylabel("Short-Term Eligible Rows (Total = 361,589)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)

    p01s = [h_stats[h]["tr_excess_stats"]["p01"] * 100 for h in hs]
    meds = [h_stats[h]["tr_excess_stats"]["median"] * 100 for h in hs]
    p99s = [h_stats[h]["tr_excess_stats"]["p99"] * 100 for h in hs]
    stds = [h_stats[h]["tr_excess_stats"]["std"] * 100 for h in hs]
    axes[1].plot(hs, stds, marker="o", color="#dc2626", linewidth=2, label="Std Dev (%)")
    axes[1].plot(hs, p99s, marker="^", color="#0f766e", linewidth=1.8, linestyle="--", label="99th Percentile Excess Return (%)")
    axes[1].plot(hs, p01s, marker="v", color="#2563eb", linewidth=1.8, linestyle="--", label="1st Percentile Excess Return (%)")
    axes[1].plot(hs, meds, marker="s", color="#334155", linewidth=1.5, label="Median Excess Return (%)")
    axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
    axes[1].set_title("Excess Total Return (vs ASTRI) Dispersion by Horizon", fontweight="bold")
    axes[1].set_xlabel("Forward Horizon")
    axes[1].set_ylabel("Excess Total Return (%)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    p2 = viz_dir / "02_horizon_target_distributions_and_penny_noise.png"
    fig.savefig(p2)
    plt.close(fig)
    paths.append(str(p2))

    # 3. Purged chronological CV folds & untouched FINAL_TEST timeline
    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    folds = split_summary["cv_folds"]
    y_pos = np.arange(len(folds) + 1)
    for idx, f in enumerate(folds):
        t0 = pd.Timestamp(f["train_start_date"])
        t1 = pd.Timestamp(f["train_end_date"])
        v0 = pd.Timestamp(f["val_start_date"])
        v1 = pd.Timestamp(f["val_end_date"])
        ax.barh(idx, (t1 - t0).days, left=t0, height=0.55, color="#2563eb", label="Train (Expanding)" if idx == 0 else "")
        ax.barh(idx, (v0 - t1).days, left=t1, height=0.55, color="#f97316", label=f"Purge+Embargo Gap ({TOTAL_GAP_SESSIONS} Sessions)" if idx == 0 else "")
        ax.barh(idx, (v1 - v0).days, left=v0, height=0.55, color="#16a34a", label="CV Validation" if idx == 0 else "")
        ax.text(v1 + pd.Timedelta(days=60), idx, f"Train {f['train_rows']:,} | Val {f['val_rows']:,}", va="center", fontsize=8.5)

    ft = split_summary["final_holdout_split"]
    d0 = pd.Timestamp(ft["dev_pool_start_date"])
    d1 = pd.Timestamp(ft["dev_pool_end_date"])
    f0 = pd.Timestamp(ft["final_test_start_date"])
    f1 = pd.Timestamp(ft["final_test_end_date"])
    ax.barh(len(folds), (d1 - d0).days, left=d0, height=0.55, color="#1d4ed8")
    ax.barh(len(folds), (f0 - d1).days, left=d1, height=0.55, color="#f97316")
    ax.barh(len(folds), (f1 - f0).days, left=f0, height=0.55, color="#9333ea", label="Untouched FINAL_TEST (2024–2025)")
    ax.text(f1 + pd.Timedelta(days=60), len(folds), f"Dev {ft['dev_pool_rows']:,} | Test {ft['final_test_rows']:,}", va="center", fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f["fold_name"] for f in folds] + ["Final Holdout"])
    ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp("2001-06-01"), pd.Timestamp("2030-06-01"))
    ax.set_title("Chronological Purged Expanding-Window Splits & Final Holdout Architecture", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p3 = viz_dir / "03_purged_chronological_splits_and_regime_counts.png"
    fig.savefig(p3)
    plt.close(fig)
    paths.append(str(p3))

    # 4. Liquidity threshold sensitivity & feature group missingness
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    sens_df = pd.DataFrame(tq_report["threshold_sensitivity_matrix"])
    short_labels = [s.split(" (")[0] for s in sens_df["scenario"]]
    y_s = np.arange(len(sens_df))
    axes[0].barh(y_s, sens_df["row_count"], color=["#94a3b8" if "Relaxed" in lbl else ("#0f766e" if "Baseline_Phase2C" in lbl else "#2563eb") for lbl in short_labels])
    for i, r in sens_df.iterrows():
        cs_m = r["median_daily_cross_section"]
        axes[0].text(r["row_count"] + 6000, i, f"{r['row_count']:,} rows (med CS={cs_m})", va="center", fontsize=8)
    axes[0].set_yticks(y_s)
    axes[0].set_yticklabels(short_labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, sens_df["row_count"].max() * 1.45)
    axes[0].set_title("Liquidity & Penny-Stock Filter Sensitivity (ST Pool)", fontweight="bold")
    axes[0].set_xlabel("Eligible Rows")

    # Dev vs FinalTest missingness bar summary for selected key groups
    cols_meta = feature_registry["columns"]
    sel_feats = [
        "Daily_Return_CapAdj", "RSI_14", "NATR_14", "Price_to_SMA20_Pct",
        "Beta_ASPI_60", "Log_Rolling_MedianTurnover_60",
        "Overnight_Gap_Ret", "IsFundamentalPITValid",
        "sec_pb_ratio", "roe_pct", "sec_pe_ratio", "net_income_growth_qtr_yoy_pct",
    ]
    miss_dev = [100.0 * cols_meta[c].get("dev_missing_rate", 0.0) for c in sel_feats]
    miss_ft = [100.0 * cols_meta[c].get("final_test_missing_rate", 0.0) for c in sel_feats]
    y_f = np.arange(len(sel_feats))
    axes[1].barh(y_f - 0.18, miss_dev, height=0.35, color="#3b82f6", label="DEV_POOL (2002–2023Q3)")
    axes[1].barh(y_f + 0.18, miss_ft, height=0.35, color="#10b981", label="FINAL_TEST (2024–2025)")
    axes[1].set_yticks(y_f)
    axes[1].set_yticklabels(sel_feats, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Missingness (%)")
    axes[1].set_title("Feature Regime Missingness: DEV_POOL vs FINAL_TEST", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p4 = viz_dir / "04_liquidity_threshold_sensitivity_and_missingness.png"
    fig.savefig(p4)
    plt.close(fig)
    paths.append(str(p4))

    return paths


def _input_signature(repo_root: Path) -> str:
    """Compute composite SHA-256 signature over locked upstream inputs for --incremental runs."""
    digests = {rel: compute_file_sha256(repo_root / rel) for rel in PHASE4A_LOCKED_UPSTREAM_PATHS}
    blob = json.dumps(digests, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_phase4a_pipeline(
    repo_root: Path,
    *,
    incremental: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end Phase 4A short-term modeling dataset construction & verification."""
    repo_root = Path(repo_root).resolve()
    out_dir = repo_root / "Dataset" / "processed" / "phase4a"
    viz_dir = repo_root / "reports" / "phase4" / "visualizations" / "phase4a"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "short_term_dataset_manifest.json"

    # Capture upstream baseline checksums before touching anything
    baseline_checksums = {rel: compute_file_sha256(repo_root / rel) for rel in PHASE4A_LOCKED_UPSTREAM_PATHS}
    in_sig = _input_signature(repo_root)

    if incremental and manifest_path.exists():
        all_exist = all((repo_root / rel).exists() for rel in PHASE4A_OUTPUT_FILES)
        if all_exist:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if prev.get("input_signature_sha256") == in_sig:
                logger.info("Phase 4A incremental cache HIT (input signature %s); skipping recomputation.", in_sig[:12])
                return prev

    logger.info("Loading Phase 3D short_term_integrated_research_universe & Phase 1B/2C/3D authorities...")
    st_path = repo_root / "Dataset" / "processed" / "phase3d" / "short_term_integrated_research_universe.parquet"
    st_df = pd.read_parquet(st_path)
    phase3d_cols = list(st_df.columns)

    tr_df = pd.read_parquet(
        repo_root / "Dataset" / "processed" / "phase1b" / "cse_total_return_history.parquet",
        columns=["SecurityID", "Date", "TotalReturnPrice"],
    )
    pf2c_df = pd.read_parquet(
        repo_root / "Dataset" / "processed" / "phase2c" / "point_in_time_quality_flags.parquet",
        columns=["SecurityID", "Date", "IsDividendResolutionClean"],
    )
    car_df = pd.read_parquet(repo_root / "Dataset" / "processed" / "phase2c" / "corporate_action_remediation.parquet")
    pai_df = pd.read_parquet(repo_root / "Dataset" / "processed" / "phase2c" / "price_anomaly_investigation.parquet")
    mapping_df = pd.read_parquet(repo_root / "Dataset" / "processed" / "phase3d" / "company_security_pit_mapping.parquet")
    master_df = pd.read_parquet(
        repo_root / "Dataset" / "processed" / "phase1b" / "cse_security_master_enriched.parquet",
        columns=["SecurityID", "HistoricalStatus"],
    )

    aspi_by_date = (
        st_df[["Date", "ASPI_Close"]]
        .dropna(subset=["ASPI_Close"])
        .assign(Date=lambda d: pd.to_datetime(d["Date"]).dt.normalize())
        .drop_duplicates("Date")
    )
    cal_df = build_trading_calendar_and_benchmarks(repo_root, st_df["Date"], aspi_by_date)
    logger.info("Calendar built: %d exchange sessions (%s .. %s).", len(cal_df), cal_df["Dt"].min().date(), cal_df["Dt"].max().date())

    tgt_all_df, target_summary = compute_forward_targets_and_labels(
        st_df, tr_df, pf2c_df, car_df, pai_df, mapping_df, cal_df,
        horizons=HORIZONS, min_cross_section_size=MIN_CROSS_SECTION_SIZE,
    )

    # Assemble 361,589-row modeling pool cleanly with derived features & targets
    elig_idx = st_df.index[st_df["IsShortTermEligible"].fillna(False)]
    st_elig = st_df.loc[elig_idx].copy()
    derived_elig = build_derived_scale_free_features(st_elig)

    listed_set = set(master_df.loc[master_df["HistoricalStatus"].astype(str) == "Listed", "SecurityID"].astype(str))
    is_listed_elig = pd.Series(st_elig["SecurityID"].astype(str).isin(listed_set).to_numpy(dtype=bool), index=elig_idx, dtype=bool)
    tgt_elig = tgt_all_df.loc[elig_idx].copy()

    # Select clean non-overlapping modeling columns (metadata + candidate features + targets/labels; excluding raw level-confounded indicators)
    keep_meta = [c for c in IDENTIFIER_AND_METADATA_COLS if c in st_elig.columns] + [
        "RawClose", "Close", "CapitalAdjClose", "ASPI_Close", "IsOpenFeatureAvailable", "IsShortTermEligible"
    ]
    keep_base_feats = [c for c in (CORE_FULL_SAMPLE_TECH_FEATURES + ["Overnight_Gap_Ret", "Intraday_Ret"] + [c for c in PIT_FUNDAMENTAL_FEATURES if c in st_elig.columns]) if c in st_elig.columns]
    tgt_cols_keep = [c for c in tgt_elig.columns if not c.startswith("_")]

    model_df = pd.concat(
        [
            st_elig[keep_meta],
            pd.DataFrame({"IsListed": is_listed_elig}, index=elig_idx),
            st_elig[[c for c in keep_base_feats if c not in keep_meta]],
            derived_elig,
            tgt_elig[tgt_cols_keep],
        ],
        axis=1,
    )
    model_df["Date"] = pd.to_datetime(model_df["Date"]).dt.normalize()
    model_df["IsCleanCorePool"] = (
        (~model_df["IsLowPrice_Sub1LKR"].astype(bool))
        & (~tgt_elig["_EntryBadCAPath"].to_numpy(dtype=bool))
        & (~tgt_elig["_EntrySrcErr"].to_numpy(dtype=bool))
    )
    model_df["IsPennyStockTickNoiseRisk"] = model_df["IsLowPrice_Sub1LKR"].astype(bool)

    splits_df, split_summary = assign_chronological_purged_splits(model_df, cal_df)
    model_df = pd.concat(
        [model_df, splits_df[["FinalSplitRole", "EraSplitGroup", "CV_Fold_1", "CV_Fold_2", "CV_Fold_3", "CV_Fold_4"]]],
        axis=1,
    )
    sample_elig_df, excl_audit_df, waterfall_summary = build_eligibility_and_exclusion_audit(
        st_df, tgt_all_df, splits_df, master_df
    )
    model_df = model_df.sort_values(["Date", "SecurityID"]).reset_index(drop=True)
    splits_df = splits_df.sort_values(["Date", "SecurityID"]).reset_index(drop=True)

    assert model_df.columns.is_unique, "Duplicate column names detected in model_df!"
    assert not model_df.duplicated(["Date", "SecurityID"]).any(), "Duplicate (Date, SecurityID) detected in model_df!"
    assert not splits_df.duplicated(["Date", "SecurityID"]).any(), "Duplicate (Date, SecurityID) detected in splits_df!"

    feature_registry, all_cand_features, pruned_core_features, post2017_features = build_short_term_feature_registry(
        model_df, splits_df, phase3d_cols
    )
    all_listed_master = set(master_df.loc[master_df["HistoricalStatus"].astype(str) == "Listed", "SecurityID"].astype(str))
    target_definitions, target_quality_report = evaluate_sensitivity_and_target_quality(
        st_df, model_df, splits_df, target_summary, waterfall_summary, all_listed_secs=all_listed_master
    )
    st_elig_with_sess = st_elig.assign(SessionIndex=tgt_elig["SessionIndex"].to_numpy())
    leakage_checks = run_phase4a_leakage_and_perturbation_tests(
        model_df, splits_df, split_summary, all_cand_features, raw_st_df=st_elig_with_sess
    )
    target_quality_report["leakage_and_causality_verification"] = leakage_checks
    target_quality_report["astri_benchmark_anomaly_quarantine"] = {
        "rule": "|r_ASTRI - r_ASPI| > 5% with immediate >5% next-session opposite reversal (set ASTRI=NaN, never interpolated)",
        "quarantined_count": len(cal_df.attrs.get("quarantined_astri_dates", [])),
        "quarantined_dates": cal_df.attrs.get("quarantined_astri_dates", []),
    }

    # Persist outputs deterministically
    logger.info("Writing Phase 4A parquet and JSON deliverables to %s...", out_dir)
    model_df.to_parquet(out_dir / "short_term_modeling_dataset.parquet", index=False)
    sample_elig_df.to_parquet(out_dir / "short_term_sample_eligibility.parquet", index=False)
    excl_audit_df.to_parquet(out_dir / "short_term_exclusion_audit.parquet", index=False)
    splits_df.to_parquet(out_dir / "short_term_validation_splits.parquet", index=False)

    (out_dir / "short_term_feature_registry.json").write_text(json.dumps(feature_registry, indent=2), encoding="utf-8")
    (out_dir / "short_term_target_definitions.json").write_text(json.dumps(target_definitions, indent=2), encoding="utf-8")
    (out_dir / "short_term_target_quality_report.json").write_text(json.dumps(target_quality_report, indent=2), encoding="utf-8")

    viz_paths = generate_phase4a_visualizations(
        viz_dir, sample_elig_df, model_df, splits_df, split_summary, target_quality_report, feature_registry
    )

    # Re-verify zero upstream modification after completion
    immutable_ok, immutability_audit = verify_upstream_immutability(repo_root, baseline_checksums)
    if not immutable_ok:
        raise RuntimeError(f"Upstream immutability violation detected: {immutability_audit['mismatches']}")

    output_hashes = {
        rel: compute_file_sha256(repo_root / rel)
        for rel in PHASE4A_OUTPUT_FILES
        if not rel.endswith("short_term_dataset_manifest.json")
    }

    manifest: dict[str, Any] = {
        "phase": "4A",
        "title": "Short-Term Modeling Dataset Preparation & Target Engineering",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_signature_sha256": in_sig,
        "upstream_immutability_verified": immutable_ok,
        "upstream_files_locked_count": len(baseline_checksums),
        "dataset_summary": {
            "full_universe_audit_rows": int(len(sample_elig_df)),
            "modeling_eligible_rows": int(len(model_df)),
            "modeling_dataset_columns": int(model_df.shape[1]),
            "modeling_unique_securities": int(model_df["SecurityID"].nunique()),
            "modeling_listed_securities": int(model_df.loc[model_df["IsListed"], "SecurityID"].nunique()),
            "modeling_delisted_securities": int(model_df.loc[~model_df["IsListed"], "SecurityID"].nunique()),
            "modeling_delisted_rows": int((~model_df["IsListed"]).sum()),
            "candidate_features_total": len(all_cand_features),
            "pruned_core_technical_features": len(pruned_core_features),
            "post2017_with_open_features": len(post2017_features),
            "horizons": list(HORIZONS),
            "date_min": str(model_df["Date"].min().date()),
            "date_max": str(model_df["Date"].max().date()),
        },
        "split_architecture_summary": split_summary,
        "leakage_checks": leakage_checks,
        "visualizations": viz_paths,
        "output_sha256": output_hashes,
        "upstream_sha256_verified": baseline_checksums,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Phase 4A completed cleanly: %d eligible modeling rows, %d features, 100%% upstream immutability verified.",
                len(model_df), len(all_cand_features))
    return manifest
