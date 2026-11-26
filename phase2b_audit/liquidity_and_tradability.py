"""
Phase 2B Workstream C – Liquidity & Tradability Analysis Engine
================================================================
Evaluates per-security and market-regime liquidity across all 719 CSE SecurityIDs
using actual trading observations vs market-calendar days, analyzes zero-volume
streaks, turnover distributions, extreme block trades/price gaps, and tests
configurable liquidity eligibility thresholds for Short, Medium, and Long strategies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def audit_liquidity_and_tradability(
    df_adj: pd.DataFrame,
    sec_master: pd.DataFrame,
    market_calendar: pd.DataFrame,
    pv_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute per-SecurityID liquidity profiles (`liquidity_analysis.csv`) and
    configurable strategy threshold sensitivity tables.

    Methodology Note:
      - Rolling ADT_20 / ADT_60 / ADT_120 and ADV_20 / ADV_60 / ADV_120 are calculated
        explicitly over the trailing 20 / 60 / 120 **trading observations** of each SecurityID
        in 2001–2025 (where Turnover & ShareVolume exist).
      - Active Trading Frequency (%) is measured over **CSE market-calendar trading days**
        within each security's [FirstPriceDate, LastPriceDate] lifespan:
        `(days with positive ShareVolume & Turnover) / (market calendar sessions in lifespan)`.
    """
    base = df_adj.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    ret_lookup = pv_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)["Daily_Return_SplitAdj"].to_numpy(dtype=np.float64)
    base_ret = ret_lookup

    cal_dates = np.sort(pd.to_datetime(market_calendar["Date"]).dt.normalize().unique())
    master_map = sec_master.set_index("SecurityID")

    rows: list[dict[str, Any]] = []
    # Also track regime stats by era
    era_stats_rows: list[dict[str, Any]] = []
    for era, g_era in base.groupby("Era"):
        to_e = pd.to_numeric(g_era["Turnover"], errors="coerce")
        sv_e = pd.to_numeric(g_era["ShareVolume"], errors="coerce")
        zero_or_na_vol = int((sv_e.isna() | (sv_e <= 0)).sum())
        pos_to = to_e[to_e > 0].dropna()
        era_stats_rows.append({
            "era": str(era),
            "records": int(len(g_era)),
            "unique_security_ids": int(g_era["SecurityID"].nunique()),
            "pct_missing_or_zero_volume": round(100.0 * zero_or_na_vol / max(1, len(g_era)), 2),
            "median_turnover_lkr": round(float(pos_to.median()), 2) if len(pos_to) else None,
            "p90_turnover_lkr": round(float(pos_to.quantile(0.90)), 2) if len(pos_to) else None,
        })

    # Pre-split indices per SecurityID cleanly without making full DataFrame copies
    dates_all = pd.to_datetime(base["Date"]).dt.normalize().to_numpy()
    sv_all = pd.to_numeric(base["ShareVolume"], errors="coerce").to_numpy(dtype=np.float64)
    to_all = pd.to_numeric(base["Turnover"], errors="coerce").to_numpy(dtype=np.float64)
    close_all = pd.to_numeric(base["Close"], errors="coerce").to_numpy(dtype=np.float64)
    code_all = base["CompanyCode"].astype(str).to_numpy()
    mt_all = base["MainType"].astype(str).to_numpy()

    def _max_streak(cond: np.ndarray) -> int:
        if not cond.any():
            return 0
        best = 0
        cur = 0
        for flag in cond.tolist():
            if flag:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    for sid, idx in base.groupby("SecurityID", sort=False).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        dts = dates_all[pos]
        sv = sv_all[pos]
        to = to_all[pos]
        cl = close_all[pos]
        rets = base_ret[pos]

        n_obs = len(pos)
        d_first = pd.Timestamp(dts[0])
        d_last = pd.Timestamp(dts[-1])
        # Market calendar days spanning security's price history
        cal_lo = int(np.searchsorted(cal_dates, np.datetime64(d_first), side="left"))
        cal_hi = int(np.searchsorted(cal_dates, np.datetime64(d_last), side="right"))
        cal_span_days = max(1, cal_hi - cal_lo)

        # Positive turnover / volume indicates verified active execution
        has_vol_col = np.isfinite(sv)
        active_trade_mask = np.isfinite(sv) & (sv > 0) & np.isfinite(to) & (to > 0)
        zero_vol_explicit = np.isfinite(sv) & (sv == 0)
        missing_vol_rows = int((~has_vol_col).sum())
        active_days = int(active_trade_mask.sum())
        trading_freq_pct = round(100.0 * active_days / cal_span_days, 2)

        # Multi-day calendar gaps between consecutive price observations
        if n_obs >= 2:
            gap_bars = np.diff(
                np.searchsorted(cal_dates, dts, side="left")
            )
            max_cal_gap_sessions = int(np.max(gap_bars))
            mean_cal_gap_sessions = float(np.mean(gap_bars))
        else:
            max_cal_gap_sessions = 0
            mean_cal_gap_sessions = 0.0

        max_consec_zero_or_missing_vol = _max_streak(~active_trade_mask)

        # Turnover & Volume statistics over valid 2001+ observations
        valid_to_vals = to[np.isfinite(to) & (to >= 0)]
        valid_sv_vals = sv[np.isfinite(sv) & (sv >= 0)]

        if len(valid_to_vals) > 0:
            mean_to = float(np.mean(valid_to_vals))
            med_to = float(np.median(valid_to_vals))
            std_to = float(np.std(valid_to_vals, ddof=1)) if len(valid_to_vals) > 1 else 0.0
            cv_to = float(std_to / mean_to) if mean_to > 0 else np.nan
            p95_to = float(np.percentile(valid_to_vals, 95))
            max_to = float(np.max(valid_to_vals))
            # Latest rolling observation-based ADT over trailing 20 / 60 / 120 obs
            adt_20 = float(np.mean(valid_to_vals[-20:])) if len(valid_to_vals) >= 20 else float(np.mean(valid_to_vals))
            adt_60 = float(np.mean(valid_to_vals[-60:])) if len(valid_to_vals) >= 60 else float(np.mean(valid_to_vals))
            adt_120 = float(np.mean(valid_to_vals[-120:])) if len(valid_to_vals) >= 120 else float(np.mean(valid_to_vals))
            extreme_spike_cnt = int(np.sum(valid_to_vals > max(10.0 * max(med_to, 1.0), 10_000_000.0)))
        else:
            mean_to = med_to = std_to = cv_to = p95_to = max_to = adt_20 = adt_60 = adt_120 = np.nan
            extreme_spike_cnt = 0

        if len(valid_sv_vals) > 0:
            mean_sv = float(np.mean(valid_sv_vals))
            med_sv = float(np.median(valid_sv_vals))
            adv_20 = float(np.mean(valid_sv_vals[-20:])) if len(valid_sv_vals) >= 20 else float(np.mean(valid_sv_vals))
            adv_60 = float(np.mean(valid_sv_vals[-60:])) if len(valid_sv_vals) >= 60 else float(np.mean(valid_sv_vals))
            adv_120 = float(np.mean(valid_sv_vals[-120:])) if len(valid_sv_vals) >= 120 else float(np.mean(valid_sv_vals))
        else:
            mean_sv = med_sv = adv_20 = adv_60 = adv_120 = np.nan

        # Price gaps (>25% single-bar split-adj jump)
        fin_rets = rets[np.isfinite(rets)]
        extreme_price_gap_cnt = int(np.sum(np.abs(fin_rets) > 25.0)) if len(fin_rets) else 0

        # Recent 2023-2025 active window metrics (crucial for current tradability assessment)
        rec_mask = (dts >= np.datetime64("2023-01-01")) & active_trade_mask
        if rec_mask.any():
            rec_lo = int(np.searchsorted(cal_dates, np.datetime64("2023-01-01"), side="left"))
            rec_hi = int(np.searchsorted(cal_dates, np.datetime64("2025-12-31"), side="right"))
            rec_cal = max(1, min(cal_hi, rec_hi) - max(cal_lo, rec_lo))
            rec_freq_pct = round(100.0 * int(rec_mask.sum()) / rec_cal, 2)
            rec_med_to = float(np.median(to[rec_mask]))
            rec_mean_to = float(np.mean(to[rec_mask]))
        else:
            rec_freq_pct = 0.0
            rec_med_to = np.nan
            rec_mean_to = np.nan

        hist_status = str(master_map.loc[sid, "HistoricalStatus"]) if sid in master_map.index else "Unknown"
        rows.append({
            "SecurityID": sid,
            "CompanyCode": str(code_all[pos[0]]),
            "MainType": str(mt_all[pos[0]]),
            "HistoricalStatus": hist_status,
            "FirstPriceDate": str(d_first.date()),
            "LastPriceDate": str(d_last.date()),
            "TotalObservations": n_obs,
            "MarketCalendarSpanDays": cal_span_days,
            "ActiveTradingDays": active_days,
            "ExplicitZeroVolumeDays": int(zero_vol_explicit.sum()),
            "MissingVolumeDays": missing_vol_rows,
            "TradingFrequencyPct": trading_freq_pct,
            "MaxCalGapBetweenObsDays": max_cal_gap_sessions,
            "MeanCalGapBetweenObsDays": round(mean_cal_gap_sessions, 2),
            "MaxConsecNonActiveObsStreak": max_consec_zero_or_missing_vol,
            "MeanShareVolume": round(mean_sv, 2) if np.isfinite(mean_sv) else np.nan,
            "MedianShareVolume": round(med_sv, 2) if np.isfinite(med_sv) else np.nan,
            "ADV_20_Obs": round(adv_20, 2) if np.isfinite(adv_20) else np.nan,
            "ADV_60_Obs": round(adv_60, 2) if np.isfinite(adv_60) else np.nan,
            "ADV_120_Obs": round(adv_120, 2) if np.isfinite(adv_120) else np.nan,
            "MeanTurnoverLKR": round(mean_to, 2) if np.isfinite(mean_to) else np.nan,
            "MedianTurnoverLKR": round(med_to, 2) if np.isfinite(med_to) else np.nan,
            "TurnoverVolatilityLKR": round(std_to, 2) if np.isfinite(std_to) else np.nan,
            "TurnoverCV": round(cv_to, 4) if np.isfinite(cv_to) else np.nan,
            "P95TurnoverLKR": round(p95_to, 2) if np.isfinite(p95_to) else np.nan,
            "MaxTurnoverLKR": round(max_to, 2) if np.isfinite(max_to) else np.nan,
            "ADT_20_Obs_LKR": round(adt_20, 2) if np.isfinite(adt_20) else np.nan,
            "ADT_60_Obs_LKR": round(adt_60, 2) if np.isfinite(adt_60) else np.nan,
            "ADT_120_Obs_LKR": round(adt_120, 2) if np.isfinite(adt_120) else np.nan,
            "ExtremeTurnoverSpikeDays": extreme_spike_cnt,
            "ExtremePriceGapDays_Gt25Pct": extreme_price_gap_cnt,
            "Recent2023_2025_ActiveFreqPct": rec_freq_pct,
            "Recent2023_2025_MedianTurnoverLKR": round(rec_med_to, 2) if np.isfinite(rec_med_to) else np.nan,
            "Recent2023_2025_MeanTurnoverLKR": round(rec_mean_to, 2) if np.isfinite(rec_mean_to) else np.nan,
        })

    liq_df = pd.DataFrame(rows).sort_values(["HistoricalStatus", "MeanTurnoverLKR"], ascending=[False, False]).reset_index(drop=True)

    # ── Configurable Threshold Sensitivity Analysis (Equity N+X Classes) ─────
    eq_df = liq_df[liq_df["MainType"].isin(["N", "X"])].copy()
    eq_active_2325 = eq_df[eq_df["LastPriceDate"] >= "2023-01-01"].copy()

    sensitivity_grid = [
        {"rule_name": "Broad_Floor_100k_30pct", "min_med_to_lkr": 100_000, "min_freq_pct": 30.0, "min_obs": 60, "target_strategy": "Long-term Universe Floor"},
        {"rule_name": "MedTerm_Floor_250k_50pct", "min_med_to_lkr": 250_000, "min_freq_pct": 50.0, "min_obs": 120, "target_strategy": "Medium-term Growth"},
        {"rule_name": "ShortTerm_Mod_500k_70pct", "min_med_to_lkr": 500_000, "min_freq_pct": 70.0, "min_obs": 120, "target_strategy": "Short-term Standard"},
        {"rule_name": "ShortTerm_Liquid_1M_80pct", "min_med_to_lkr": 1_000_000, "min_freq_pct": 80.0, "min_obs": 120, "target_strategy": "Short-term Institutional"},
        {"rule_name": "ShortTerm_Prime_2p5M_85pct", "min_med_to_lkr": 2_500_000, "min_freq_pct": 85.0, "min_obs": 200, "target_strategy": "Short-term High Capacity"},
        {"rule_name": "BlueChip_5M_90pct", "min_med_to_lkr": 5_000_000, "min_freq_pct": 90.0, "min_obs": 200, "target_strategy": "Top Tier Benchmark"},
    ]
    sens_results: list[dict[str, Any]] = []
    for cfg in sensitivity_grid:
        pass_hist = eq_df[
            (eq_df["TotalObservations"] >= cfg["min_obs"])
            & (eq_df["TradingFrequencyPct"] >= cfg["min_freq_pct"])
            & (eq_df["MedianTurnoverLKR"].fillna(0) >= cfg["min_med_to_lkr"])
        ]
        pass_rec = eq_active_2325[
            (eq_active_2325["TotalObservations"] >= cfg["min_obs"])
            & (eq_active_2325["Recent2023_2025_ActiveFreqPct"] >= cfg["min_freq_pct"])
            & (eq_active_2325["Recent2023_2025_MedianTurnoverLKR"].fillna(0) >= cfg["min_med_to_lkr"])
        ]
        sens_results.append({
            **cfg,
            "passing_securities_full_history": int(len(pass_hist)),
            "passing_pct_of_equity_universe": round(100.0 * len(pass_hist) / max(1, len(eq_df)), 2),
            "passing_listed_2023_2025": int(len(pass_rec)),
            "passing_pct_of_2023_2025_active": round(100.0 * len(pass_rec) / max(1, len(eq_active_2325)), 2),
        })

    liq_summary = {
        "total_securities": int(len(liq_df)),
        "equity_nx_securities": int(len(eq_df)),
        "active_equity_2023_2025_count": int(len(eq_active_2325)),
        "securities_zero_active_volume_all_time": int((liq_df["ActiveTradingDays"] == 0).sum()),
        "securities_with_freq_ge_75pct": int((eq_df["TradingFrequencyPct"] >= 75.0).sum()),
        "securities_with_freq_lt_25pct": int((eq_df["TradingFrequencyPct"] < 25.0).sum()),
        "era_liquidity_summary": era_stats_rows,
        "threshold_sensitivity": sens_results,
        "calculation_basis": "Observation-based trailing windows (20/60/120 obs); calendar-session frequency denominator (cse_market_calendar.parquet)",
    }
    return liq_df, liq_summary
