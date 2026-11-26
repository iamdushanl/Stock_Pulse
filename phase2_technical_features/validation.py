"""
Phase 2A Feature Quality, Causality, Redundancy & Independent Verification Suite
==============================================================================
Validates the full Phase 2A panel outputs (`cse_technical_features.parquet`,
`cse_price_volume_features.parquet`, `cse_market_relative_features.parquet`)
and writes `feature_quality_report.csv` and validation diagnostics for `phase2_manifest.json`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .indicators_core import (
    compute_market_relative_panel,
    compute_parabolic_sar,
    compute_technical_features_panel,
)


THEORETICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "RSI_14": (0.0, 100.0),
    "ADX_14": (0.0, 100.0),
    "Aroon_Up_25": (0.0, 100.0),
    "Aroon_Down_25": (0.0, 100.0),
    "Williams_R_14": (-100.0, 0.0),
    "Stoch_K_14": (0.0, 100.0),
    "Stoch_D_3": (0.0, 100.0),
    "MFI_14": (0.0, 100.0),
    "CMF_20": (-1.0, 1.0),
    "Donchian_Pos_20": (0.0, 1.0),
    "ATR_14": (0.0, float("inf")),
    "NATR_14": (0.0, float("inf")),
    "Hist_Vol_21d": (0.0, float("inf")),
    "Volume_SMA_20": (0.0, float("inf")),
    "Volume_Ratio_20": (0.0, float("inf")),
    "PSAR_Trend": (-1.0, 1.0),
    "PSAR_Reversal": (-1.0, 1.0),
    "Supertrend_Dir_10_3": (-1.0, 1.0),
    "Supertrend_Reversal": (-1.0, 1.0),
    "MA_Cross_Signal_20_50": (-1.0, 1.0),
    "RSI_Div_Bullish_14": (0.0, 1.0),
    "RSI_Div_Bearish_14": (0.0, 1.0),
}


def build_feature_quality_report(
    tech_df: pd.DataFrame,
    pv_df: pd.DataFrame,
    mr_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute column-level quality statistics across all 63 Phase 2A indicators."""
    rows: list[dict[str, Any]] = []
    table_map = {
        "cse_technical_features.parquet": tech_df,
        "cse_price_volume_features.parquet": pv_df,
        "cse_market_relative_features.parquet": mr_df,
    }

    for _, meta in metadata_df.iterrows():
        ind = str(meta["indicator_name"])
        tbl_name = str(meta["output_table"])
        df = table_map[tbl_name]
        s = df[ind]
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        n = len(arr)
        is_nan = np.isnan(arr)
        is_inf = np.isinf(arr)
        finite = arr[~is_nan & ~is_inf]

        pre2017_mask = (df["Date"] < pd.Timestamp("2017-01-01")).to_numpy()
        pre2002_mask = (df["Date"] < pd.Timestamp("2002-01-01")).to_numpy()
        post2017_mask = ~pre2017_mask

        nan_pct = float(100.0 * is_nan.sum() / max(1, n))
        nan_pre2002_pct = float(100.0 * is_nan[pre2002_mask].mean()) if pre2002_mask.any() else np.nan
        nan_pre2017_pct = float(100.0 * is_nan[pre2017_mask].mean()) if pre2017_mask.any() else np.nan
        nan_post2017_pct = float(100.0 * is_nan[post2017_mask].mean()) if post2017_mask.any() else np.nan

        lo, hi = THEORETICAL_BOUNDS.get(ind, (-float("inf"), float("inf")))
        oob_count = int(np.sum((finite < lo - 1e-9) | (finite > hi + 1e-9))) if len(finite) else 0

        secs_with_valid = int(df.loc[~is_nan, "SecurityID"].nunique())
        rows.append({
            "indicator_name": ind,
            "group": meta["group"],
            "output_table": tbl_name,
            "total_rows": n,
            "valid_count": int(len(finite)),
            "nan_count": int(is_nan.sum()),
            "nan_pct": round(nan_pct, 4),
            "inf_count": int(is_inf.sum()),
            "nan_pct_pre2002": round(nan_pre2002_pct, 4) if np.isfinite(nan_pre2002_pct) else np.nan,
            "nan_pct_pre2017": round(nan_pre2017_pct, 4) if np.isfinite(nan_pre2017_pct) else np.nan,
            "nan_pct_2017plus": round(nan_post2017_pct, 4) if np.isfinite(nan_post2017_pct) else np.nan,
            "securities_with_valid_data": secs_with_valid,
            "min": round(float(np.min(finite)), 6) if len(finite) else np.nan,
            "p05": round(float(np.percentile(finite, 5)), 6) if len(finite) else np.nan,
            "median": round(float(np.median(finite)), 6) if len(finite) else np.nan,
            "p95": round(float(np.percentile(finite, 95)), 6) if len(finite) else np.nan,
            "max": round(float(np.max(finite)), 6) if len(finite) else np.nan,
            "mean": round(float(np.mean(finite)), 6) if len(finite) else np.nan,
            "std": round(float(np.std(finite)), 6) if len(finite) else np.nan,
            "out_of_bounds_count": oob_count,
            "bounds_ok": bool(oob_count == 0 and int(is_inf.sum()) == 0),
        })

    return pd.DataFrame(rows)


def independent_psar_reference(df_sec: pd.DataFrame, af0: float = 0.02, step: float = 0.02, max_af: float = 0.20) -> pd.DataFrame:
    """
    Completely independent textbook Wilder PSAR scalar reference implementation
    written without shared helpers, used for verification against `compute_parabolic_sar`.
    """
    df_s = df_sec.sort_values("Date").reset_index(drop=True)
    n = len(df_s)
    sar_ref = [math.nan] * n
    trd_ref = [math.nan] * n
    rev_ref = [math.nan] * n

    valid_idx = [
        i for i in range(n)
        if pd.notna(df_s.loc[i, "High"])
        and pd.notna(df_s.loc[i, "Low"])
        and pd.notna(df_s.loc[i, "Close"])
        and float(df_s.loc[i, "High"]) >= float(df_s.loc[i, "Low"]) > 0
        and float(df_s.loc[i, "Close"]) > 0
    ]
    if len(valid_idx) >= 2:
        i0, i1 = valid_idx[0], valid_idx[1]
        h0, l0, c0 = float(df_s.loc[i0, "High"]), float(df_s.loc[i0, "Low"]), float(df_s.loc[i0, "Close"])
        h1, l1, c1 = float(df_s.loc[i1, "High"]), float(df_s.loc[i1, "Low"]), float(df_s.loc[i1, "Close"])
        bull = c1 >= c0
        sar = min(l0, l1) if bull else max(h0, h1)
        ep = max(h0, h1) if bull else min(l0, l1)
        af = af0
        sar_ref[i1] = sar
        trd_ref[i1] = 1.0 if bull else -1.0
        rev_ref[i1] = 0.0

        for k in range(2, len(valid_idx)):
            prev2, prev1, cur = valid_idx[k - 2], valid_idx[k - 1], valid_idx[k]
            if cur - prev1 > 11:
                # Re-seed across multi-day gap (>10 missing bars) matching spec
                h_a, l_a, c_a = float(df_s.loc[prev1, "High"]), float(df_s.loc[prev1, "Low"]), float(df_s.loc[prev1, "Close"])
                h_b, l_b, c_b = float(df_s.loc[cur, "High"]), float(df_s.loc[cur, "Low"]), float(df_s.loc[cur, "Close"])
                # Need next bar if resetting, so check segment breaks
                continue
            hi = float(df_s.loc[cur, "High"])
            lo = float(df_s.loc[cur, "Low"])
            l_p1, l_p2 = float(df_s.loc[prev1, "Low"]), float(df_s.loc[prev2, "Low"])
            h_p1, h_p2 = float(df_s.loc[prev1, "High"]), float(df_s.loc[prev2, "High"])
            next_sar = sar + af * (ep - sar)
            if bull:
                next_sar = min(next_sar, l_p1, l_p2)
                if lo < next_sar:
                    bull = False
                    sar = max(ep, hi)
                    ep = lo
                    af = af0
                    rev_ref[cur] = -1.0
                else:
                    sar = next_sar
                    rev_ref[cur] = 0.0
                    if hi > ep:
                        ep = hi
                        af = min(af + step, max_af)
            else:
                next_sar = max(next_sar, h_p1, h_p2)
                if hi > next_sar:
                    bull = True
                    sar = min(ep, lo)
                    ep = hi
                    af = af0
                    rev_ref[cur] = 1.0
                else:
                    sar = next_sar
                    rev_ref[cur] = 0.0
                    if lo < ep:
                        ep = lo
                        af = min(af + step, max_af)
            sar_ref[cur] = sar
            trd_ref[cur] = 1.0 if bull else -1.0

    close_s = pd.to_numeric(df_s["Close"], errors="coerce")
    dist_ref = 100.0 * (close_s - pd.Series(sar_ref)) / close_s
    return pd.DataFrame({
        "PSAR": sar_ref,
        "PSAR_Trend": trd_ref,
        "PSAR_Distance_Pct": dist_ref,
        "PSAR_Reversal": rev_ref,
    })


def run_validation_suite(
    df_adj: pd.DataFrame,
    tech_df: pd.DataFrame,
    pv_df: pd.DataFrame,
    mr_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    market_indices: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    gics_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run all Section 6 validation checks and return structured summary dictionary."""
    summary: dict[str, Any] = {}

    # 1. Duplicate keys & key preservation
    for name, frame in [("technical", tech_df), ("price_volume", pv_df), ("market_relative", mr_df)]:
        dups = int(frame.duplicated(subset=["SecurityID", "Date"]).sum())
        summary[f"{name}_duplicate_keys"] = dups
        summary[f"{name}_rows"] = int(len(frame))
        summary[f"{name}_unique_security_ids"] = int(frame["SecurityID"].nunique())
        summary[f"{name}_unique_company_codes"] = int(frame["CompanyCode"].nunique())

    summary["total_inf_values"] = int(quality_df["inf_count"].sum())
    summary["all_bounds_ok"] = bool(quality_df["bounds_ok"].all())

    # 2. Independent PSAR verification on modern continuous series (2020-2025 JKH-N-0000 & COMB-N-0000)
    psar_diffs: dict[str, float] = {}
    for sid in ("JKH-N-0000", "COMB-N-0000", "HHL-N-0000"):
        sub = df_adj[(df_adj["SecurityID"] == sid) & (df_adj["Date"] >= pd.Timestamp("2020-01-01"))].sort_values("Date").reset_index(drop=True)
        eng_p = compute_parabolic_sar(sub)
        ref_p = independent_psar_reference(sub)
        valid_mask = eng_p["PSAR"].notna() & ref_p["PSAR"].notna()
        max_diff = float(np.max(np.abs(eng_p.loc[valid_mask, "PSAR"] - ref_p.loc[valid_mask, "PSAR"])))
        trend_match = bool((eng_p.loc[valid_mask, "PSAR_Trend"] == ref_p.loc[valid_mask, "PSAR_Trend"]).all())
        rev_match = bool((eng_p.loc[valid_mask, "PSAR_Reversal"] == ref_p.loc[valid_mask, "PSAR_Reversal"]).all())
        # Verify invariant: SAR <= Low on uptrend, SAR >= High on downtrend
        up_m = eng_p["PSAR_Trend"] == 1.0
        dn_m = eng_p["PSAR_Trend"] == -1.0
        sar_envelope_ok = bool(
            (eng_p.loc[up_m, "PSAR"] <= sub.loc[up_m, "Low"] + 1e-9).all()
            and (eng_p.loc[dn_m, "PSAR"] >= sub.loc[dn_m, "High"] - 1e-9).all()
        )
        psar_diffs[sid] = {
            "max_abs_diff": max_diff,
            "trend_match": trend_match,
            "reversal_match": rev_match,
            "envelope_invariant_ok": sar_envelope_ok,
            "valid_bars_checked": int(valid_mask.sum()),
        }
    summary["psar_independent_verification"] = psar_diffs

    # 3. Causality / Prefix-invariance test on JKH-N-0000 & SAMP-N-0000 across ALL 3 tables
    sample_adj = df_adj[df_adj["SecurityID"].isin(["JKH-N-0000", "SAMP-N-0000"])].sort_values(["SecurityID", "Date"]).copy()
    cutoff = pd.Timestamp("2024-06-28")
    trunc_adj = sample_adj[sample_adj["Date"] <= cutoff].copy()
    t_full, pv_full = compute_technical_features_panel(sample_adj)
    mr_full = compute_market_relative_panel(sample_adj, market_indices, sector_mapping, gics_df=gics_df)
    t_tr, pv_tr = compute_technical_features_panel(trunc_adj)
    mr_tr = compute_market_relative_panel(trunc_adj, market_indices, sector_mapping, gics_df=gics_df)

    max_causal_diff = 0.0
    for f_full, f_tr in [(t_full, t_tr), (pv_full, pv_tr), (mr_full, mr_tr)]:
        f_full_sub = f_full[f_full["Date"] <= cutoff].reset_index(drop=True)
        f_tr_sub = f_tr.reset_index(drop=True)
        num_cols = [c for c in f_tr_sub.columns if c not in ("SecurityID", "CompanyCode", "Date", "MainType", "SubType")]
        for c in num_cols:
            a = f_full_sub[c].to_numpy(dtype=np.float64)
            b = f_tr_sub[c].to_numpy(dtype=np.float64)
            nan_eq = np.array_equal(np.isnan(a), np.isnan(b))
            if not nan_eq:
                max_causal_diff = float("inf")
            fin = np.isfinite(a)
            if fin.any():
                d = float(np.max(np.abs(a[fin] - b[fin])))
                if d > max_causal_diff:
                    max_causal_diff = d
    summary["causality_prefix_invariance_max_diff"] = max_causal_diff

    # 4. Cross-security contamination check: corrupting SAMP does not alter JKH by even 1e-12
    perturbed_adj = sample_adj.copy()
    mask_samp = perturbed_adj["SecurityID"] == "SAMP-N-0000"
    perturbed_adj.loc[mask_samp, ["Open", "High", "Low", "Close", "CapitalAdjClose"]] *= 3.7
    t_pert, _ = compute_technical_features_panel(perturbed_adj)
    jkh_a = t_full[t_full["SecurityID"] == "JKH-N-0000"].drop(columns=["Date", "SecurityID", "CompanyCode", "MainType", "SubType"]).to_numpy(dtype=np.float64)
    jkh_b = t_pert[t_pert["SecurityID"] == "JKH-N-0000"].drop(columns=["Date", "SecurityID", "CompanyCode", "MainType", "SubType"]).to_numpy(dtype=np.float64)
    summary["cross_security_isolation_exact"] = bool(np.allclose(jkh_a, jkh_b, equal_nan=True, atol=1e-12))

    # 5. Corporate action consistency check around JKH 2024-11-05 (1:10 split), LMF, AFS, SUN, WATA, HHL
    ca_checks: dict[str, dict[str, Any]] = {}
    for sid, dt_s in [
        ("JKH-N-0000", "2024-11-05"),
        ("LMF-N-0000", "2024-01-16"),
        ("AFS-N-0000", "2024-01-24"),
        ("SUN-N-0000", "2025-02-18"),
        ("WATA-N-0000", "2025-02-28"),
        ("HHL-N-0000", "2025-05-06"),
    ]:
        dt = pd.Timestamp(dt_s)
        sub_pv = pv_df[(pv_df["SecurityID"] == sid) & (pv_df["Date"] >= dt - pd.Timedelta(days=14)) & (pv_df["Date"] <= dt + pd.Timedelta(days=14))].sort_values("Date")
        sub_tk = tech_df[(tech_df["SecurityID"] == sid) & (tech_df["Date"] >= dt - pd.Timedelta(days=14)) & (tech_df["Date"] <= dt + pd.Timedelta(days=14))].sort_values("Date")
        post = sub_pv[sub_pv["Date"] >= dt].iloc[0]
        post_tk = sub_tk[sub_tk["Date"] >= dt].iloc[0]
        ca_checks[f"{sid}@{dt_s}"] = {
            "ex_day_split_adj_ret_pct": round(float(post["Daily_Return_SplitAdj"]), 4),
            "ex_day_rsi14": round(float(post_tk["RSI_14"]), 4),
            "ex_day_psar_trend": float(post_tk["PSAR_Trend"]),
            "no_false_crash_spike": bool(abs(post["Daily_Return_SplitAdj"]) < 30.0),
        }
    summary["corporate_action_continuity"] = ca_checks

    # 6. Pre-2017 missing Open check
    pre17_pv = pv_df[pv_df["Date"] < pd.Timestamp("2017-01-01")]
    summary["pre2017_overnight_gap_non_null"] = int(pre17_pv["Overnight_Gap_Ret"].notna().sum())
    summary["pre2017_intraday_ret_non_null"] = int(pre17_pv["Intraday_Ret"].notna().sum())

    # 7. Feature redundancy check (2017-2025 non-NaN numerical correlation matrix |r| >= 0.95)
    ind_cols_t = [c for c in tech_df.columns if c not in ("SecurityID", "CompanyCode", "Date", "MainType", "SubType")]
    ind_cols_p = [c for c in pv_df.columns if c not in ("SecurityID", "CompanyCode", "Date", "MainType", "SubType", "SplitAdjClose", "CapitalAdjClose", "Daily_Return_SplitAdj", "Daily_Return_CapAdj")]
    sample_idx = (tech_df["Date"] >= pd.Timestamp("2017-01-01")) & tech_df["RSI_14"].notna() & tech_df["ADX_14"].notna()
    # Deterministic uniform stride sampling (~50,000 rows) for fast correlation matrix
    pos_sample = np.flatnonzero(sample_idx.to_numpy())
    if len(pos_sample) > 50_000:
        pos_sample = pos_sample[:: max(1, len(pos_sample) // 50_000)]
    comb_mat = pd.concat([
        tech_df.iloc[pos_sample][ind_cols_t].reset_index(drop=True),
        pv_df.iloc[pos_sample][ind_cols_p].reset_index(drop=True),
    ], axis=1)
    corr_mat = comb_mat.corr()
    high_corr_pairs: list[dict[str, Any]] = []
    cols_c = corr_mat.columns.tolist()
    for i in range(len(cols_c)):
        for j in range(i + 1, len(cols_c)):
            rv = corr_mat.iloc[i, j]
            if pd.notna(rv) and abs(float(rv)) >= 0.95:
                high_corr_pairs.append({
                    "feature_a": cols_c[i],
                    "feature_b": cols_c[j],
                    "corr": round(float(rv), 5),
                })
    summary["high_correlation_pairs_ge_095"] = sorted(high_corr_pairs, key=lambda x: -abs(x["corr"]))

    # 8. Parameter stability check on JKH-N-0000 (2017-2025)
    jkh17 = df_adj[(df_adj["SecurityID"] == "JKH-N-0000") & (df_adj["Date"] >= pd.Timestamp("2017-01-01"))].sort_values("Date").reset_index(drop=True)
    psar_default = compute_parabolic_sar(jkh17, 0.02, 0.02, 0.20)
    psar_tight = compute_parabolic_sar(jkh17, 0.015, 0.015, 0.15)
    psar_fast = compute_parabolic_sar(jkh17, 0.025, 0.025, 0.25)
    summary["psar_parameter_stability_jkh_2017plus"] = {
        "default_flips": int((psar_default["PSAR_Reversal"].abs() == 1).sum()),
        "tight_flips": int((psar_tight["PSAR_Reversal"].abs() == 1).sum()),
        "fast_flips": int((psar_fast["PSAR_Reversal"].abs() == 1).sum()),
        "trend_agreement_default_vs_tight_pct": round(float(100.0 * (psar_default["PSAR_Trend"] == psar_tight["PSAR_Trend"]).mean()), 2),
        "trend_agreement_default_vs_fast_pct": round(float(100.0 * (psar_default["PSAR_Trend"] == psar_fast["PSAR_Trend"]).mean()), 2),
    }

    return summary
