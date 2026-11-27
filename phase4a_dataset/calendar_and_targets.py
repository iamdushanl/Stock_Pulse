"""
Phase 4A Workstreams 2 & 3 – CSE Trading Calendar, Benchmarks & Forward Target Engineering
========================================================================================
Implements:
  1. Canonical CSE trading-session calendar (`1990-01-12` to `2025-12-31`, `8,251` market sessions)
     with exact ASPI (price index) and official CSE ASTRI (All Share Total Return Index,
     `2004-01-02` to `2025-12-31`) aligned strictly on actual trading sessions (never fabricated).
  2. Strict exact-session forward returns across `h in (5, 10, 20, 60)` CSE sessions separating:
       - Primary Total Return targets (`TotalReturnPrice` vs `ASTRI`), ONLY when all dates along
         `[t, t+h]` satisfy `IsDividendResolutionClean`, clean CA, and finite ASTRI (zero silent
         substitution from price returns).
       - Secondary Capital-Adjusted Price Return targets (`CapitalAdjClose` vs `ASPI_Close`) matching
         price-return convention consistently across 2002–2025.
  3. Hard quarantine on target windows crossing Phase 2C `QUARANTINED` / `VERIFIED_CORRECTION_REQUIRED`
     corporate-action ExDate artifacts (preventing fake allotment-proxy returns) and
     `SOURCE_DATA_ERROR` / `SECURITY_IDENTITY_ISSUE` endpoints/paths.
  4. Forward cross-sectional percentile, binary positive-excess, top-quintile classification
     targets (requiring CrossSectionN >= 15), plus future-only volatility, MAE, and max drawdown
     downside risk labels kept strictly separate from model features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xlrd

logger = logging.getLogger("phase4a.calendar_targets")

HORIZONS: tuple[int, ...] = (5, 10, 20, 60)
MAX_HORIZON: int = max(HORIZONS)
PURGE_SESSIONS: int = MAX_HORIZON
EMBARGO_SESSIONS: int = 5
TOTAL_GAP_SESSIONS: int = PURGE_SESSIONS + EMBARGO_SESSIONS
MIN_CROSS_SECTION_SIZE: int = 15

# Path density minimum required before publishing realized volatility / MAE / MDD over (t, t+h]
_MIN_PATH_OBS = {5: 4, 10: 7, 20: 14, 60: 40}


def load_astri_benchmark(xls_path: Path) -> pd.DataFrame:
    """Load official CSE All Share Total Return Index (ASTRI) from `09Total Returns Indices - Daily.xls`."""
    if not xls_path.exists():
        logger.warning("ASTRI workbook not found at %s; ASTRI total-return benchmark unavailable.", xls_path)
        return pd.DataFrame(columns=["Dt", "ASTRI"])

    wb = xlrd.open_workbook(str(xls_path))
    if "TRI" not in wb.sheet_names():
        return pd.DataFrame(columns=["Dt", "ASTRI"])

    s = wb.sheet_by_name("TRI")
    rows: list[tuple[pd.Timestamp, float]] = []
    for r in range(5, s.nrows):
        v0 = s.cell_value(r, 0)
        v1 = s.cell_value(r, 1)
        if isinstance(v0, float) and v0 > 20000:
            dt = pd.Timestamp(xlrd.xldate_as_datetime(v0, wb.datemode).date())
            try:
                val = float(v1)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val) and val > 0:
                rows.append((dt, val))

    if not rows:
        return pd.DataFrame(columns=["Dt", "ASTRI"])
    return pd.DataFrame(rows, columns=["Dt", "ASTRI"]).drop_duplicates("Dt", keep="last").sort_values("Dt").reset_index(drop=True)


def build_trading_calendar_and_benchmarks(
    repo_root: Path,
    st_dates_series: pd.Series,
    aspi_by_date_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construct the authoritative ordered CSE trading-session calendar (`SessionIndex = 0 .. N-1`)
    joining `cse_market_calendar.parquet`, early 1990 exchange dates present in Phase 3D quotes,
    ASPI price index (`ASPI_Close`), and official CSE `ASTRI` (`2004-01-02` to `2025-12-31`).
    Never invents or forward-fills missing index closes.
    """
    cal_raw = pd.read_parquet(repo_root / "Dataset" / "processed" / "phase1" / "cse_market_calendar.parquet", columns=["Date"])
    cal_set = set(pd.to_datetime(cal_raw["Date"]).dt.normalize().dropna().unique())
    st_set = set(pd.to_datetime(st_dates_series).dt.normalize().dropna().unique())
    all_dts = sorted(cal_set | st_set)

    cal_df = pd.DataFrame({
        "Dt": pd.to_datetime(all_dts).astype("datetime64[ns]"),
        "SessionIndex": np.arange(len(all_dts), dtype=np.int32),
        "InPhase1Calendar": [d in cal_set for d in all_dts],
    })

    if aspi_by_date_df is not None:
        a_df = aspi_by_date_df.copy()
        dt_col = "Dt" if "Dt" in a_df.columns else "Date"
        a_df["Dt"] = pd.to_datetime(a_df[dt_col]).dt.normalize().astype("datetime64[ns]")
        cal_df = cal_df.merge(a_df[["Dt", "ASPI_Close"]].drop_duplicates("Dt"), on="Dt", how="left")
    else:
        cal_df["ASPI_Close"] = np.nan

    astri_df = load_astri_benchmark(repo_root / "01_existing_cse_sources" / "09Total Returns Indices - Daily.xls")
    if not astri_df.empty:
        astri_df["Dt"] = pd.to_datetime(astri_df["Dt"]).dt.normalize().astype("datetime64[ns]")
        cal_df = cal_df.merge(astri_df, on="Dt", how="left")
        # Detect & quarantine isolated single-day ASTRI source-entry glitches (e.g. ASPI paste-over or digit typo
        # that diverges >5% from ASPI_Close and immediately reverses >5% next trading session; never interpolate!)
        m_both = cal_df["ASTRI"].notna() & cal_df["ASPI_Close"].notna() & (cal_df["ASPI_Close"] > 0)
        if m_both.sum() >= 3:
            sub_idx = cal_df.index[m_both]
            a_val = cal_df.loc[sub_idx, "ASTRI"].to_numpy(dtype=np.float64)
            p_val = cal_df.loc[sub_idx, "ASPI_Close"].to_numpy(dtype=np.float64)
            r_a = np.concatenate(([0.0], (a_val[1:] / a_val[:-1]) - 1.0))
            r_p = np.concatenate(([0.0], (p_val[1:] / p_val[:-1]) - 1.0))
            diff = r_a - r_p
            next_diff = np.concatenate((diff[1:], [0.0]))
            isolated_spike = (np.abs(diff) > 0.05) & (np.abs(next_diff) > 0.05) & ((diff * next_diff) < 0.0)
            if isolated_spike.any():
                q_dates = [str(pd.Timestamp(d).date()) for d in cal_df.loc[sub_idx[isolated_spike], "Dt"]]
                cal_df.loc[sub_idx[isolated_spike], "ASTRI"] = np.nan
                cal_df.attrs["quarantined_astri_dates"] = q_dates
            else:
                cal_df.attrs["quarantined_astri_dates"] = []
        else:
            cal_df.attrs["quarantined_astri_dates"] = []
    else:
        cal_df["ASTRI"] = np.nan
        cal_df.attrs["quarantined_astri_dates"] = []

    return cal_df


def _build_quarantined_ca_mask(
    st_sec: np.ndarray,
    st_pos: np.ndarray,
    cal_df: pd.DataFrame,
    car_df: pd.DataFrame,
) -> np.ndarray:
    """
    Flag `(SecurityID, SessionIndex)` bars lying on Phase 2C `QUARANTINED` or `VERIFIED_CORRECTION_REQUIRED`
    corporate-action ExDates / prior-XR ranges so no horizon crossing them emits fake CapitalAdj return artifacts.
    """
    n = len(st_sec)
    is_bad_ca = np.zeros(n, dtype=bool)
    if car_df.empty:
        return is_bad_ca

    cal_dts = pd.to_datetime(cal_df["Dt"]).to_numpy(dtype="datetime64[ns]").view("int64")
    bad_car = car_df[car_df["RemediationDisposition"].isin(["QUARANTINED", "VERIFIED_CORRECTION_REQUIRED"])].copy()
    if bad_car.empty:
        return is_bad_ca

    # Build interval markers per SecurityID in SessionIndex space
    sec_intervals: dict[str, list[tuple[int, int]]] = {}
    for _, cr in bad_car.iterrows():
        sid = str(cr["SecurityID"]).strip()
        ex_d = pd.to_datetime(cr.get("ExDate"), errors="coerce")
        pr_d = pd.to_datetime(cr.get("CandidatePriorXRDate"), errors="coerce")
        if pd.isna(ex_d):
            continue
        d_hi = np.datetime64(ex_d.normalize().to_datetime64(), "ns").astype("int64")
        d_lo = np.datetime64(pr_d.normalize().to_datetime64(), "ns").astype("int64") if pd.notna(pr_d) else d_hi
        if d_lo > d_hi:
            d_lo, d_hi = d_hi, d_lo
        p_lo = int(np.searchsorted(cal_dts, d_lo, side="left"))
        p_hi = int(np.searchsorted(cal_dts, d_hi, side="right")) - 1
        # Expand +/- 1 trading session to block immediate pre/post ex-date mismatch jumps cleanly
        sec_intervals.setdefault(sid, []).append((max(0, p_lo - 1), p_hi + 1))

    order = np.argsort(st_sec, kind="mergesort")
    sec_s = st_sec[order]
    pos_s = st_pos[order]
    bounds = np.flatnonzero(sec_s[1:] != sec_s[:-1]) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n]))

    for s, e in zip(starts, ends):
        sid = str(sec_s[s])
        ivs = sec_intervals.get(sid)
        if not ivs:
            continue
        p_slice = pos_s[s:e]
        m_local = np.zeros(len(p_slice), dtype=bool)
        for lo, hi in ivs:
            m_local |= (p_slice >= lo) & (p_slice <= hi)
        is_bad_ca[order[s:e]] = m_local

    return is_bad_ca


def compute_forward_targets_and_labels(
    st_df: pd.DataFrame,
    tr_df: pd.DataFrame,
    pf2c_df: pd.DataFrame,
    car_df: pd.DataFrame,
    pai_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    min_cross_section_size: int = MIN_CROSS_SECTION_SIZE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute leakage-free forward targets across actual CSE trading sessions for `st_df`:
      - Uses exact `(SecurityID, SessionIndex + h)` target date match (ZERO shift/ffill bugs).
      - Strictly separates Total Return targets (`TotalReturnPrice` vs `ASTRI`, clean CA + dividends)
        from Capital-Adjusted Price Return targets (`CapitalAdjClose` vs `ASPI_Close`).
      - Quarantines `QUARANTINED` / `VERIFIED_CORRECTION_REQUIRED` CA crossings and
        `SOURCE_DATA_ERROR` / `SECURITY_IDENTITY_ISSUE` path corruptions.
      - Computes future-only volatility, Maximum Adverse Excursion (MAE), Maximum Drawdown,
        and downside risk flags over `(pos_t, pos_t + h]`.
    """
    df = st_df[["Date", "SecurityID", "CapitalAdjClose", "RawClose", "Daily_Return_CapAdj", "IsPriceDataValid", "IsCorporateActionValidated", "IsLifecycleValidated", "IsShortTermEligible"]].copy()
    df["Dt"] = pd.to_datetime(df["Date"]).dt.normalize().astype("datetime64[ns]")
    df = df.merge(cal_df[["Dt", "SessionIndex", "ASPI_Close", "ASTRI"]], on="Dt", how="left")

    # Attach Phase 1B Total Return & Phase 2C DividendResolutionClean strictly by (SecurityID, Date)
    tr_sub = tr_df[["Date", "SecurityID", "TotalReturnPrice"]].copy()
    tr_sub["Dt"] = pd.to_datetime(tr_sub["Date"]).dt.normalize().astype("datetime64[ns]")
    tr_sub = tr_sub.drop_duplicates(["SecurityID", "Dt"])
    df = df.merge(tr_sub[["Dt", "SecurityID", "TotalReturnPrice"]], on=["Dt", "SecurityID"], how="left")

    pf_sub = pf2c_df[["Date", "SecurityID", "IsDividendResolutionClean"]].copy()
    pf_sub["Dt"] = pd.to_datetime(pf_sub["Date"]).dt.normalize().astype("datetime64[ns]")
    pf_sub = pf_sub.drop_duplicates(["SecurityID", "Dt"])
    df = df.merge(pf_sub[["Dt", "SecurityID", "IsDividendResolutionClean"]], on=["Dt", "SecurityID"], how="left")
    df["IsDividendResolutionClean"] = df["IsDividendResolutionClean"].fillna(False).astype(bool)

    # Attach MappingValidTo boundary so targets NEVER leak past delisting date
    mp_sub = mapping_df[["SecurityID", "MappingValidTo"]].drop_duplicates("SecurityID")
    df = df.merge(mp_sub, on="SecurityID", how="left")
    delist_dt = pd.to_datetime(df["MappingValidTo"], errors="coerce")

    # Attach Phase 2C price anomaly source-error flags strictly by (SecurityID, Dt)
    if not pai_df.empty:
        pai_bad = pai_df[pai_df["AnomalyClassification"].isin(["SOURCE_DATA_ERROR", "SECURITY_IDENTITY_ISSUE"])].copy()
        pai_bad["Dt"] = pd.to_datetime(pai_bad["Date"]).dt.normalize().astype("datetime64[ns]")
        pai_bad["_is_src_err"] = True
        pai_bad = pai_bad[["SecurityID", "Dt", "_is_src_err"]].drop_duplicates(["SecurityID", "Dt"])
        df = df.merge(pai_bad, on=["Dt", "SecurityID"], how="left")
        df["_is_src_err"] = df["_is_src_err"].fillna(False).astype(bool)
    else:
        df["_is_src_err"] = False

    # Sort deterministically by (SecurityID, SessionIndex) and validate uniqueness
    df["_orig_idx"] = np.arange(len(df), dtype=np.int32)
    df = df.sort_values(["SecurityID", "SessionIndex"]).reset_index(drop=True)
    n = len(df)

    sec_arr = df["SecurityID"].astype(str).to_numpy()
    pos_arr = df["SessionIndex"].to_numpy(dtype=np.int32)
    cap_px = pd.to_numeric(df["CapitalAdjClose"], errors="coerce").to_numpy(dtype=np.float64)
    raw_px = pd.to_numeric(df["RawClose"], errors="coerce").to_numpy(dtype=np.float64)
    tot_px = pd.to_numeric(df["TotalReturnPrice"], errors="coerce").to_numpy(dtype=np.float64)
    ret_daily = pd.to_numeric(df["Daily_Return_CapAdj"], errors="coerce").to_numpy(dtype=np.float64)
    px_ok = df["IsPriceDataValid"].fillna(False).to_numpy(dtype=bool) & np.isfinite(cap_px) & (cap_px > 0) & np.isfinite(raw_px) & (raw_px > 0)
    ca_ok = df["IsCorporateActionValidated"].fillna(False).to_numpy(dtype=bool)
    div_ok = df["IsDividendResolutionClean"].to_numpy(dtype=bool) & np.isfinite(tot_px) & (tot_px > 0)
    src_err = df["_is_src_err"].to_numpy(dtype=bool)
    elig_arr = df["IsShortTermEligible"].fillna(False).to_numpy(dtype=bool)
    bad_ca = _build_quarantined_ca_mask(sec_arr, pos_arr, cal_df, car_df)
    delist_arr = pd.to_datetime(df["MappingValidTo"], errors="coerce").to_numpy(dtype="datetime64[ns]")

    # Index arrays by calendar date (`cal_dates[0..max_pos]`)
    cal_dts = pd.to_datetime(cal_df["Dt"]).to_numpy(dtype="datetime64[ns]")
    max_pos = len(cal_dts) - 1
    aspi_cal = pd.to_numeric(cal_df["ASPI_Close"], errors="coerce").to_numpy(dtype=np.float64)
    astri_cal = pd.to_numeric(cal_df["ASTRI"], errors="coerce").to_numpy(dtype=np.float64)

    # Pre-allocate per-horizon output dictionary on sorted array indices (re-mapped cleanly back at end)
    out: dict[str, np.ndarray] = {
        "SessionIndex": pos_arr,
        "TotalReturnPrice": tot_px,
        "IsDividendResolutionClean": div_ok,
        "ASTRI_Close": astri_cal[pos_arr],
        "_EntryBadCAPath": bad_ca,
        "_EntrySrcErr": src_err,
    }

    for h in horizons:
        out[f"TargetDate_{h}d"] = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        out[f"Target_Sec_TotalReturn_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Target_ASTRI_Return_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Target_Excess_TotalReturn_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Target_Sec_CapitalReturn_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Target_ASPI_PriceReturn_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Target_Excess_CapitalReturn_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Label_Fwd_RealizedVol_Ann_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Label_Fwd_MaxAdverseExcursion_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Label_Fwd_MaxDrawdown_{h}d"] = np.full(n, np.nan, dtype=np.float64)
        out[f"Label_Fwd_DownsideRisk_Flag_{h}d"] = np.full(n, -1, dtype=np.int8)  # -1 mapped to pd.NA below
        out[f"IsTargetEligible_TR_{h}d"] = np.zeros(n, dtype=bool)
        out[f"IsTargetEligible_Cap_{h}d"] = np.zeros(n, dtype=bool)
        out[f"TargetExclusionReason_{h}d"] = np.full(n, "", dtype=object)

    # Compute exact forward endpoints AND future-only path stats cleanly per contiguous SecurityID slice
    bounds = np.flatnonzero(sec_arr[1:] != sec_arr[:-1]) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n]))

    for s, e in zip(starts, ends):
        p_s = pos_arr[s:e]
        k = len(p_s)
        c_s = cap_px[s:e]
        t_s = tot_px[s:e]
        ret_s = ret_daily[s:e]
        pxok_s = px_ok[s:e]
        caok_s = ca_ok[s:e]
        divok_s = div_ok[s:e]
        err_s = src_err[s:e]
        elig_s = elig_arr[s:e]
        bca_s = bad_ca[s:e]
        dlim = delist_arr[s]  # Constant per SecurityID
        has_dlim = not np.isnat(dlim)

        # Prefix sums over contiguous rows for O(1) window hazard counting
        pref_not_ca = np.concatenate(([0], np.cumsum(~caok_s, dtype=np.int32)))
        pref_not_div = np.concatenate(([0], np.cumsum(~divok_s, dtype=np.int32)))
        pref_err = np.concatenate(([0], np.cumsum(err_s, dtype=np.int32)))
        pref_bca = np.concatenate(([0], np.cumsum(bca_s, dtype=np.int32)))

        for h in horizons:
            tgt_p = p_s + h
            within_sample = tgt_p <= max_pos
            safe_tp = np.where(within_sample, tgt_p, max_pos)
            tgt_dt = np.where(within_sample, cal_dts[safe_tp], np.datetime64("NaT", "ns"))
            out[f"TargetDate_{h}d"][s:e] = tgt_dt

            # Vectorized exact searchsorted for `p_s[j] == p_s[i] + h`
            j_idx = np.searchsorted(p_s, tgt_p, side="left")
            in_b = j_idx < k
            safe_j = np.where(in_b, j_idx, 0)
            exact_match = within_sample & in_b & (p_s[safe_j] == tgt_p)

            if has_dlim:
                within_delist = within_sample & (cal_dts[safe_tp] <= dlim)
            else:
                within_delist = within_sample.copy()
            endpoint_px_ok = exact_match & pxok_s[safe_j]
            entry_ok = pxok_s & caok_s

            # Prefix check across observed bars in [i, j_idx] strictly (blocks mid-horizon quarantined CAs / bad flags!)
            j_hi = safe_j + 1
            no_bad_ca = exact_match & ((pref_bca[j_hi] - pref_bca[:k]) == 0) & ((pref_not_ca[j_hi] - pref_not_ca[:k]) == 0)
            no_src_err = exact_match & ((pref_err[j_hi] - pref_err[:k]) == 0)
            all_div_clean = exact_match & ((pref_not_div[j_hi] - pref_not_div[:k]) == 0)

            aspi_e = aspi_cal[p_s]
            aspi_x = aspi_cal[safe_tp]
            aspi_ok = within_sample & np.isfinite(aspi_e) & (aspi_e > 0) & np.isfinite(aspi_x) & (aspi_x > 0)

            astri_e = astri_cal[p_s]
            astri_x = astri_cal[safe_tp]
            astri_ok = within_sample & np.isfinite(astri_e) & (astri_e > 0) & np.isfinite(astri_x) & (astri_x > 0)

            cap_valid = exact_match & within_delist & entry_ok & endpoint_px_ok & no_bad_ca & no_src_err & aspi_ok
            tr_valid = cap_valid & all_div_clean & astri_ok

            with np.errstate(divide="ignore", invalid="ignore"):
                cap_r = np.where(cap_valid, (c_s[safe_j] / c_s) - 1.0, np.nan)
                aspi_r = np.where(cap_valid, (aspi_x / aspi_e) - 1.0, np.nan)
                exc_cap = np.where(cap_valid, cap_r - aspi_r, np.nan)

                tr_r = np.where(tr_valid, (t_s[safe_j] / t_s) - 1.0, np.nan)
                astri_r = np.where(tr_valid, (astri_x / astri_e) - 1.0, np.nan)
                exc_tr = np.where(tr_valid, tr_r - astri_r, np.nan)

            out[f"Target_Sec_CapitalReturn_{h}d"][s:e] = cap_r
            out[f"Target_ASPI_PriceReturn_{h}d"][s:e] = aspi_r
            out[f"Target_Excess_CapitalReturn_{h}d"][s:e] = exc_cap
            out[f"Target_Sec_TotalReturn_{h}d"][s:e] = tr_r
            out[f"Target_ASTRI_Return_{h}d"][s:e] = astri_r
            out[f"Target_Excess_TotalReturn_{h}d"][s:e] = exc_tr
            out[f"IsTargetEligible_Cap_{h}d"][s:e] = cap_valid
            out[f"IsTargetEligible_TR_{h}d"][s:e] = tr_valid

            # Document explicit target exclusion codes cleanly (never silent!)
            reasons = np.full(k, "ELIGIBLE_TR_AND_CAP", dtype=object)
            reasons[~tr_valid & cap_valid & (~astri_ok)] = "CAP_ONLY_PRE_2004_OR_MISSING_ASTRI"
            reasons[~tr_valid & cap_valid & astri_ok & (~all_div_clean)] = "CAP_ONLY_UNRESOLVED_DIVIDEND_IN_HORIZON"
            reasons[~cap_valid] = "OTHER_TARGET_GAP"
            reasons[~aspi_ok] = "MISSING_ASPI_BENCHMARK"
            reasons[~no_src_err] = "SOURCE_DATA_ERROR_OR_IDENTITY_ISSUE_IN_HORIZON"
            reasons[~no_bad_ca] = "CA_QUARANTINED_OR_UNVALIDATED_IN_HORIZON"
            reasons[~endpoint_px_ok] = "INVALID_EXIT_PRICE"
            reasons[~entry_ok] = "INVALID_ENTRY_PRICE_OR_CA"
            reasons[within_sample & (~within_delist)] = "POST_DELISTING_TARGET_BOUNDARY"
            reasons[within_sample & within_delist & (~exact_match)] = "SKIPPED_SESSION_OR_HALT_AT_TARGET_DATE"
            reasons[~within_sample] = "END_OF_SAMPLE_2025_12_31"
            out[f"TargetExclusionReason_{h}d"][s:e] = reasons

            # Path Downside & Realized Volatility over strictly future interval (i, j_idx]
            min_obs = _MIN_PATH_OBS[h]
            vol_out = np.full(k, np.nan, dtype=np.float64)
            mae_out = np.full(k, np.nan, dtype=np.float64)
            mdd_out = np.full(k, np.nan, dtype=np.float64)
            ds_out = np.full(k, -1, dtype=np.int8)

            # Compute path downside & realized vol only for short-term eligible entry dates (saves ~75% CPU)
            valid_indices = np.flatnonzero(cap_valid & elig_s)
            for idx_i in valid_indices:
                j_end = int(safe_j[idx_i])
                cnt_obs = j_end - idx_i
                if cnt_obs < min_obs or c_s[idx_i] <= 0.0:
                    continue
                valid_fut_mask = pxok_s[idx_i + 1 : j_end + 1]
                fut_px = c_s[idx_i + 1 : j_end + 1][valid_fut_mask]
                if len(fut_px) < min_obs:
                    continue
                path_px = np.concatenate(([c_s[idx_i]], fut_px))
                mae_v = min(0.0, float(np.min(fut_px) / c_s[idx_i] - 1.0))
                run_max = np.maximum.accumulate(path_px)
                mdd_v = float(np.min(path_px / run_max - 1.0))
                rets_win = ret_s[idx_i + 1 : j_end + 1]
                rets_fin = rets_win[np.isfinite(rets_win)]
                # Daily_Return_CapAdj is stored in percentage points (e.g. 2.5 = +2.5%); scale to decimal vol (0.34 = 34% ann.)
                vol_v = float(np.std(rets_fin / 100.0, ddof=1) * np.sqrt(252.0)) if len(rets_fin) >= min_obs else np.nan
                if np.isfinite(mae_v) and np.isfinite(mdd_v):
                    vol_out[idx_i] = vol_v
                    mae_out[idx_i] = mae_v
                    mdd_out[idx_i] = mdd_v
                    ds_out[idx_i] = 1 if (mae_v <= -0.10 or mdd_v <= -0.12) else 0

            out[f"Label_Fwd_RealizedVol_Ann_{h}d"][s:e] = vol_out
            out[f"Label_Fwd_MaxAdverseExcursion_{h}d"][s:e] = mae_out
            out[f"Label_Fwd_MaxDrawdown_{h}d"][s:e] = mdd_out
            out[f"Label_Fwd_DownsideRisk_Flag_{h}d"][s:e] = ds_out

    # Restore exact original input row order cleanly before attaching cross-sectional classification candidates
    inv_order = np.empty(n, dtype=np.int32)
    inv_order[df["_orig_idx"].to_numpy()] = np.arange(n, dtype=np.int32)
    res_df = pd.DataFrame({k: v[inv_order] for k, v in out.items()}, index=st_df.index)
    res_df["Dt"] = pd.to_datetime(st_df["Date"]).dt.normalize().astype("datetime64[ns]")
    res_df["SecurityID"] = st_df["SecurityID"].to_numpy()
    res_df["_is_st_elig"] = st_df["IsShortTermEligible"].fillna(False).to_numpy(dtype=bool)

    # Compute Cross-Sectional Percentile, Binary & Top-Quintile candidates strictly on eligible cross-sections (>= min_cross_section_size)
    target_summary: dict[str, Any] = {}
    for h in horizons:
        for mode, elig_col, exc_col in [
            ("TR", f"IsTargetEligible_TR_{h}d", f"Target_Excess_TotalReturn_{h}d"),
            ("Cap", f"IsTargetEligible_Cap_{h}d", f"Target_Excess_CapitalReturn_{h}d"),
        ]:
            m_ok = res_df["_is_st_elig"] & res_df[elig_col] & res_df[exc_col].notna()
            cs_cnt = res_df.loc[m_ok].groupby("Dt")[exc_col].transform("count").reindex(res_df.index).fillna(0).astype(np.int32)
            m_cs = m_ok & (cs_cnt >= min_cross_section_size)
            res_df[f"CrossSectionN_{mode}_{h}d"] = cs_cnt

            pct_col = f"Target_Excess{mode}_PctRank_{h}d"
            pos_col = f"Target_Positive_Excess{mode}_{h}d"
            tq_col = f"Target_TopQuintile_Excess{mode}_{h}d"

            res_df[pct_col] = np.nan
            if m_cs.any():
                res_df.loc[m_cs, pct_col] = res_df.loc[m_cs].groupby("Dt")[exc_col].rank(pct=True, method="average")

            pos_arr_h = pd.Series(pd.NA, index=res_df.index, dtype="Int8")
            pos_arr_h.loc[m_ok] = (res_df.loc[m_ok, exc_col] > 0.0).astype("Int8")
            res_df[pos_col] = pos_arr_h

            tq_arr_h = pd.Series(pd.NA, index=res_df.index, dtype="Int8")
            tq_arr_h.loc[m_cs] = (res_df.loc[m_cs, pct_col] >= 0.80).astype("Int8")
            res_df[tq_col] = tq_arr_h

        ds_col = f"Label_Fwd_DownsideRisk_Flag_{h}d"
        raw_ds = res_df[ds_col].to_numpy()
        ds_ser = pd.Series(pd.NA, index=res_df.index, dtype="Int8")
        ds_ser.loc[raw_ds >= 0] = pd.Series(raw_ds[raw_ds >= 0], dtype="Int8").to_numpy()
        res_df[ds_col] = ds_ser

        # Compile statistical audit for horizon h on IsShortTermEligible rows
        st_m = res_df["_is_st_elig"]
        tr_s = res_df.loc[st_m, f"Target_Excess_TotalReturn_{h}d"].dropna()
        cap_s = res_df.loc[st_m, f"Target_Excess_CapitalReturn_{h}d"].dropna()
        pos_tr = res_df.loc[st_m, f"Target_Positive_ExcessTR_{h}d"].dropna()
        tq_tr = res_df.loc[st_m, f"Target_TopQuintile_ExcessTR_{h}d"].dropna()
        ds_s = res_df.loc[st_m, ds_col].dropna()

        target_summary[f"{h}d"] = {
            "horizon_sessions": int(h),
            "st_eligible_base_rows": int(st_m.sum()),
            "valid_tr_excess_rows": int(len(tr_s)),
            "valid_tr_cross_sectional_rows": int(res_df.loc[st_m, f"Target_ExcessTR_PctRank_{h}d"].notna().sum()),
            "valid_cap_excess_rows": int(len(cap_s)),
            "valid_cap_cross_sectional_rows": int(res_df.loc[st_m, f"Target_ExcessCap_PctRank_{h}d"].notna().sum()),
            "tr_excess_stats": {
                "mean": round(float(tr_s.mean()), 6) if len(tr_s) else None,
                "std": round(float(tr_s.std()), 6) if len(tr_s) else None,
                "p01": round(float(tr_s.quantile(0.01)), 6) if len(tr_s) else None,
                "median": round(float(tr_s.median()), 6) if len(tr_s) else None,
                "p99": round(float(tr_s.quantile(0.99)), 6) if len(tr_s) else None,
                "min": round(float(tr_s.min()), 6) if len(tr_s) else None,
                "max": round(float(tr_s.max()), 6) if len(tr_s) else None,
            },
            "cap_excess_stats": {
                "mean": round(float(cap_s.mean()), 6) if len(cap_s) else None,
                "std": round(float(cap_s.std()), 6) if len(cap_s) else None,
                "p01": round(float(cap_s.quantile(0.01)), 6) if len(cap_s) else None,
                "median": round(float(cap_s.median()), 6) if len(cap_s) else None,
                "p99": round(float(cap_s.quantile(0.99)), 6) if len(cap_s) else None,
            },
            "tr_positive_class_rate": round(float(pos_tr.mean()), 4) if len(pos_tr) else None,
            "tr_top_quintile_class_rate": round(float(tq_tr.mean()), 4) if len(tq_tr) else None,
            "downside_risk_rate": round(float(ds_s.mean()), 4) if len(ds_s) else None,
            "exclusion_breakdown_on_st_eligible": res_df.loc[st_m, f"TargetExclusionReason_{h}d"].value_counts().to_dict(),
        }

    res_df = res_df.drop(columns=["Dt", "SecurityID", "_is_st_elig"])
    return res_df, target_summary
