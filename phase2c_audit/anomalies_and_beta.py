"""
Phase 2C Workstreams C & D – Extreme Price Anomaly Forensic Engine & Beta Reliability
===================================================================================
Implements:
  - Workstream C: Systematic detector & forensic classifier for ALL extreme price
    movements (`BUKI-N-0000 2010-12-23`, `MERC-N-0000 2025-12-17`, `SELI-N-0000 2013`
    identity collision with `SEMB-N-0000`, reverse-split consolidations, zero-close
    anomalies, Rights decay, and isolated single-bar spikes). Never deletes raw data.
  - Workstream D: Point-in-time rolling Beta & benchmark reliability scoring WITHOUT
    blind clipping (`RELIABLE`, `LIMITED_HISTORY`, `THIN_TRADING`, `UNSTABLE`,
    `INSUFFICIENT_DATA`) and validation against CSE official `29 Beta Value.xls`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def investigate_price_anomalies(
    df_adj: pd.DataFrame,
    pv_df: pd.DataFrame,
    ca_rem_df: pd.DataFrame,
    quality_flags: pd.DataFrame,
    ts_utc: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """
    Workstream C: Systematically detect and classify extreme returns (|ret| >= 40%),
    Zero-Close rows, and CapitalAdj vs SplitAdj corporate action discontinuities.
    """
    base = df_adj.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    pv = pv_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)

    ret_s = pv["Daily_Return_SplitAdj"].to_numpy(dtype=np.float64)
    ret_c = pv["Daily_Return_CapAdj"].to_numpy(dtype=np.float64)
    raw_c = pd.to_numeric(base["RawClose"], errors="coerce").to_numpy(dtype=np.float64)
    adj_c = pd.to_numeric(base["Close"], errors="coerce").to_numpy(dtype=np.float64)
    cap_c = pd.to_numeric(base["CapitalAdjClose"], errors="coerce").to_numpy(dtype=np.float64)
    vol_a = pd.to_numeric(base["ShareVolume"], errors="coerce").to_numpy(dtype=np.float64)
    to_a = pd.to_numeric(base["Turnover"], errors="coerce").to_numpy(dtype=np.float64)
    dq_flag = base["DataQualityFlag"].astype(str).to_numpy()
    dates_a = pd.to_datetime(base["Date"]).dt.normalize().to_numpy()
    sids_a = base["SecurityID"].astype(str).to_numpy()
    codes_a = base["CompanyCode"].astype(str).to_numpy()
    mtypes_a = base["MainType"].astype(str).to_numpy()

    # Build quick lookup from (SecurityID, ExDate) -> corporate action remediation info
    ca_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for _, cr in ca_rem_df[ca_rem_df["SecurityID"].notna() & ca_rem_df["ExDate"].notna()].iterrows():
        key = (str(cr["SecurityID"]), str(pd.Timestamp(cr["ExDate"]).date()))
        ca_lookup.setdefault(key, []).append({
            "EventID": str(cr["EventID"]),
            "EventType": str(cr["EventType"]),
            "Disposition": str(cr["RemediationDisposition"]),
            "Notes": str(cr["RemediationNotes"]),
            "PF": float(cr["Phase1BPriceFactor"]),
        })

    # Detect SELI 2013 systematic identity collision: all SELI-N-0000 2013 rows (`0.6..1.0` vs `1125 -> 1040.1`)
    seli_2013_mask = (
        (sids_a == "SELI-N-0000")
        & (dates_a >= np.datetime64("2013-01-01"))
        & (dates_a <= np.datetime64("2013-12-31"))
        & (raw_c <= 5.0)
    )

    # Anomaly trigger condition:
    # 1) |Daily_Return_SplitAdj| >= 40% OR |Daily_Return_CapAdj| >= 40%
    # 2) RawClose <= 0 (Zero-Close)
    # 3) SELI 2013 boundary / CSV_THOUSANDS_COMMA_REPAIRED or explicit BUKI/MERC targets
    anom_mask = (
        (np.abs(ret_s) >= 40.0)
        | (np.abs(ret_c) >= 40.0)
        | (raw_c <= 0.0)
        | np.isin(np.arange(len(base)), np.flatnonzero(seli_2013_mask))
    )

    # Also build security-level index boundaries for prev/next bar lookup
    sec_bounds: dict[str, tuple[int, int]] = {}
    for sid, idx in base.groupby("SecurityID", sort=False).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        sec_bounds[str(sid)] = (int(pos[0]), int(pos[-1]))

    anom_indices = np.flatnonzero(anom_mask)
    rows: list[dict[str, Any]] = []
    rem_entries: list[dict[str, Any]] = []
    rem_idx = 1

    # Add quarantined SELI 2013 summary entry into remediation_log cleanly once (instead of 242 duplicate rows)
    if seli_2013_mask.any():
        rem_entries.append({
            "RemediationID": f"REM_PA_{rem_idx:04d}",
            "SecurityID": "SELI-N-0000",
            "CompanyCode": "SELI",
            "AffectedDate": "2013-01-02..2013-12-31",
            "IssueType": "SECURITY_IDENTITY_COLLISION_SELI_2013",
            "OriginalValue": f"242 rows Rs.0.60-1.00 (2012-12-28=1125 -> 2013-01-02=1.00 -> 2014-01-06=1040.10); SEMB-N-0000 has 0 rows in 2013",
            "ProposedValue": "Quarantine SELI-N-0000 2013 price window (`IsPriceDataValid=False`); do not remap without byte-level re-parse",
            "FinalValue": "Raw preserved; 242 rows tagged SECURITY_IDENTITY_ISSUE / QUARANTINED",
            "EvidenceReference": "32Daily Shares Price List -2011-2020/2013 Data.xls [Sheet1 rows 50341..50658]",
            "ResolutionStatus": "QUARANTINED",
            "Reason": "2013 Data.xls lacked separating block header between SELI-N-0000 (1120) and SEMB-N-0000 (0.80), causing SEMB-N-0000 rows to overwrite SELI-N-0000 in 2013.",
            "ProcessingTimestamp": ts_utc,
        })
        rem_idx += 1

    for i in anom_indices:
        sid = sids_a[i]
        code = codes_a[i]
        mtype = mtypes_a[i]
        dt_ts = pd.Timestamp(dates_a[i])
        dt_s = str(dt_ts.date())
        lo_i, hi_i = sec_bounds[sid]

        prev_raw = float(raw_c[i - 1]) if i > lo_i else np.nan
        curr_raw = float(raw_c[i])
        prev_adj = float(adj_c[i - 1]) if i > lo_i else np.nan
        curr_adj = float(adj_c[i])
        next_adj = float(adj_c[i + 1]) if i < hi_i else np.nan
        r_s = float(ret_s[i]) if np.isfinite(ret_s[i]) else np.nan
        r_c = float(ret_c[i]) if np.isfinite(ret_c[i]) else np.nan
        sv = float(vol_a[i]) if np.isfinite(vol_a[i]) else np.nan
        tv = float(to_a[i]) if np.isfinite(to_a[i]) else np.nan
        dq = dq_flag[i]

        # Check matching or nearby (+/-5 calendar days) corporate action
        ca_hits: list[dict[str, Any]] = []
        for delta in range(-5, 6):
            cand_s = str((dt_ts + pd.Timedelta(days=delta)).date())
            if (sid, cand_s) in ca_lookup:
                ca_hits.extend(ca_lookup[(sid, cand_s)])
        ca_ref = "; ".join(f"{h['EventID']}({h['EventType']}:{h['Disposition']})" for h in ca_hits[:3]) if ca_hits else ""

        # Classification & Remediation Recommendation
        if seli_2013_mask[i]:
            cls = "SECURITY_IDENTITY_ISSUE"
            ev = "Missing block header in 2013 Data.xls (`SELI` vs `SEMB-N-0000`); SELI 2012=1125 / 2014=1040 vs 2013=0.6-1.0."
            treat = "QUARANTINE_WINDOW (Set IsPriceDataValid=False in PIT flags; exclude from returns/models)."
        elif curr_raw <= 0.0 or (np.isfinite(prev_raw) and prev_raw <= 0.0):
            cls = "SOURCE_DATA_ERROR"
            ev = f"Zero/invalid Close recorded (`{dq}`), typical of early 1991-1997 close-only untraded zero placeholders."
            treat = "MASK_AS_INVALID (Set IsPriceDataValid=False; treated as NaN in return/indicator math)."
        elif sid == "BUKI-N-0000" and dt_s == "2010-12-24":
            cls = "CORPORATE_ACTION_EXPLAINED"
            ev = "Previous close 2010-12-17 Raw=8,021.60 -> 2010-12-24 Raw=872.80; 16 Quoted Securities confirms 10M->102M shares (missing 1:10 split + 1:50 bonus)."
            treat = "CORRECTION_FLAGGED_IN_CA_LAYER (Preserve raw; mask transition date in PIT flags until 1:10 split factor added)."
        elif any(h["EventType"] == "UncapturedSplitOrConsolidation" for h in ca_hits):
            cls = "CORPORATE_ACTION_EXPLAINED"
            ev = f"Verified against 16 Quoted Securities share-count jump ({ca_hits[0]['Notes']})."
            treat = "CORRECTION_FLAGGED_IN_CA_LAYER (Preserve raw; mask unadjusted jump window in PIT flags)."
        elif ca_hits:
            cls = "CORPORATE_ACTION_EXPLAINED"
            ev = f"Coincides with {ca_ref} (`SplitAdjRet={r_s:.1f}%`, `CapAdjRet={r_c:.1f}%`)."
            treat = "KEEP_WITH_CA_CAUTION (Exclude AllotmentDate artifact bars where flagged QUARANTINED)."
        elif mtype in ("R", "W"):
            cls = "POSSIBLE_MARKET_EVENT"
            ev = f"Derivatives/Rights entitlement ({mtype}) micro-tick move near listing/expiry ({prev_adj:.2f}->{curr_adj:.2f})."
            treat = "EXCLUDE_NON_EQUITY_FROM_MODELING (Retain in archive; filter MainType=='N' in trading models)."
        elif np.isfinite(next_adj) and np.isfinite(prev_adj) and prev_adj > 0 and curr_adj > 0 and next_adj > 0:
            # Check single-bar reversal bounce (decimal/typo snapback: t jumps >50% then t+1 reverts within 25% of prev_adj)
            if abs(next_adj / prev_adj - 1.0) <= 0.25 and (sv < 50 or pd.isna(sv)):
                cls = "SOURCE_DATA_ERROR"
                ev = f"Single-bar spike/revert ({prev_adj:.2f} -> {curr_adj:.2f} -> {next_adj:.2f}) on thin/missing volume ({sv})."
                treat = "QUARANTINE_SINGLE_BAR_SPIKE (Exclude from return training targets cleanly)."
            elif np.isfinite(sv) and sv >= 100 and np.isfinite(tv) and tv >= 100_000:
                cls = "POSSIBLE_MARKET_EVENT"
                ev = f"Persisted liquid move (`Vol={sv:,.0f}`, `Turnover=LKR {tv:,.0f}`); next close={next_adj:.2f}."
                treat = "PRESERVE_GENUINE_VOLATILITY."
            else:
                cls = "UNRESOLVED"
                ev = f"Extreme move on low/missing liquidity (`Vol={sv}`, `next={next_adj:.2f}`) without documented CA."
                treat = "PRESERVE_AND_FLAG_UNRESOLVED (Filtered naturally by rolling PIT liquidity gate)."
        else:
            cls = "UNRESOLVED"
            ev = f"Extreme move without corroborating corporate action (`Vol={sv}`, `Flag={dq}`)."
            treat = "PRESERVE_AND_FLAG_UNRESOLVED."

        rows.append({
            "SecurityID": sid,
            "CompanyCode": code,
            "MainType": mtype,
            "Date": dt_ts,
            "PreviousRawClose": prev_raw,
            "CurrentRawClose": curr_raw,
            "PreviousSplitAdjClose": prev_adj,
            "CurrentSplitAdjClose": curr_adj,
            "ReturnSplitAdjPct": round(r_s, 4) if np.isfinite(r_s) else np.nan,
            "ReturnCapAdjPct": round(r_c, 4) if np.isfinite(r_c) else np.nan,
            "ShareVolume": sv,
            "TurnoverLKR": tv,
            "DataQualityFlag": dq,
            "CorporateActionRef": ca_ref,
            "AnomalyClassification": cls,
            "Evidence": ev,
            "RecommendedTreatment": treat,
        })

    anom_df = pd.DataFrame(rows)
    # Log MERC and BUKI price-side summary cleanly in remediation_log if not already duplicated
    buki_rows = anom_df[(anom_df["SecurityID"] == "BUKI-N-0000") & (anom_df["Date"] == pd.Timestamp("2010-12-24"))]
    if not buki_rows.empty:
        br = buki_rows.iloc[0]
        rem_entries.append({
            "RemediationID": f"REM_PA_{rem_idx:04d}",
            "SecurityID": "BUKI-N-0000",
            "CompanyCode": "BUKI",
            "AffectedDate": "2010-12-24 (event effective 2010-12-23)",
            "IssueType": "EXTREME_DISCONTINUITY_BUKI_DEC2010",
            "OriginalValue": f"RawClose {br['PreviousRawClose']:.2f} -> {br['CurrentRawClose']:.2f} (SplitAdjRet={br['ReturnSplitAdjPct']:.2f}%)",
            "ProposedValue": "Apply combined 10.2x capital factor (1:10 missing split + 1:50 bonus) or quarantine 2010-12-23..24 jump",
            "FinalValue": "Raw price UNCHANGED; classified CORPORATE_ACTION_EXPLAINED and masked in PIT CA flag",
            "EvidenceReference": "16 Quoted Securities: 2009=10,000,000 vs 2010=102,000,000 sh + 03Bonus 2010=2,000,000 (1:50)",
            "ResolutionStatus": "VERIFIED_CORRECTION_REQUIRED",
            "Reason": "Confirmed 1-for-10 share split preceded 1:50 bonus in Dec-2010; missing from 05Sub Division 2010 tab.",
            "ProcessingTimestamp": ts_utc,
        })

    cls_counts = {str(k): int(v) for k, v in anom_df["AnomalyClassification"].value_counts().items()}
    summary = {
        "total_anomalies_investigated": int(len(anom_df)),
        "classification_counts": cls_counts,
        "seli_2013_collision_rows": int(seli_2013_mask.sum()),
        "buki_2010_12_investigation": (
            buki_rows[["Date", "PreviousRawClose", "CurrentRawClose", "ReturnSplitAdjPct", "AnomalyClassification", "Evidence"]]
            .assign(Date=lambda d: d["Date"].astype(str))
            .to_dict(orient="records")
        ),
    }
    return anom_df, rem_entries, summary


def build_beta_reliability(
    df_adj: pd.DataFrame,
    mr_df: pd.DataFrame,
    pv_df: pd.DataFrame,
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Workstream D: Compute rolling causal reliability metadata (`beta_observation_count`,
    `beta_nonzero_return_obs`, `beta_reliability_flag`, `beta_validity_status`) across
    all 1,414,977 rows WITHOUT altering/clipping raw `Beta_ASPI_60` or `Beta_ASPI_252`.
    """
    mr = mr_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    adj = df_adj.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    pv = pv_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)

    b60 = pd.to_numeric(mr["Beta_ASPI_60"], errors="coerce").to_numpy(dtype=np.float64)
    b252 = pd.to_numeric(mr["Beta_ASPI_252"], errors="coerce").to_numpy(dtype=np.float64)
    ret_cap = pd.to_numeric(pv["Daily_Return_CapAdj"], errors="coerce").to_numpy(dtype=np.float64)
    aspi = pd.to_numeric(mr["ASPI_Close"], errors="coerce").to_numpy(dtype=np.float64)
    sv = pd.to_numeric(adj["ShareVolume"], errors="coerce").to_numpy(dtype=np.float64)

    # Track non-zero price movement returns & active trading intervals over causal 60-obs & 252-obs windows
    valid_pair = np.isfinite(ret_cap) & np.isfinite(aspi)
    nonzero_move = valid_pair & (np.abs(ret_cap) > 1e-9)

    s_sid = mr["SecurityID"]
    obs_60 = pd.Series(valid_pair.astype(np.int32)).groupby(s_sid, sort=False).rolling(60, min_periods=1).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.int32)
    nz_60 = pd.Series(nonzero_move.astype(np.int32)).groupby(s_sid, sort=False).rolling(60, min_periods=1).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.int32)
    obs_252 = pd.Series(valid_pair.astype(np.int32)).groupby(s_sid, sort=False).rolling(252, min_periods=1).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.int32)
    nz_252 = pd.Series(nonzero_move.astype(np.int32)).groupby(s_sid, sort=False).rolling(252, min_periods=1).sum().reset_index(level=0, drop=True).to_numpy(dtype=np.int32)

    has_b60 = np.isfinite(b60) & (obs_60 >= 60)
    has_b252 = np.isfinite(b252) & (obs_252 >= 252)

    # Hierarchical deterministic reliability labels (`Beta_ASPI_60` focus for short-term; tracks `Beta_ASPI_252` too)
    # 1. INSUFFICIENT_DATA: !has_b60
    # 2. UNSTABLE: |Beta_ASPI_60| > 4.0 OR (has_b252 & |Beta_ASPI_252| > 4.0)
    # 3. THIN_TRADING: nz_60 < 20 (fewer than 1/3 of bars in 60d window had price movement!)
    # 4. LIMITED_HISTORY: has_b60 & ~has_b252 (valid 60d beta, <252 obs for structural beta)
    # 5. RELIABLE: has_b60 & has_b252 & nz_60 >= 20 & nz_252 >= 80 & |b60|<=4 & |b252|<=4
    flags = np.full(len(mr), "INSUFFICIENT_DATA", dtype=object)
    valid_60_only = has_b60 & (~has_b252)
    valid_both = has_b60 & has_b252

    unstable = has_b60 & ((np.abs(b60) > 4.0) | (has_b252 & (np.abs(b252) > 4.0)))
    thin = has_b60 & (~unstable) & ((nz_60 < 20) | (valid_both & (nz_252 < 80)))
    limited = valid_60_only & (~unstable) & (~thin)
    reliable = valid_both & (~unstable) & (~thin)

    flags[limited] = "LIMITED_HISTORY"
    flags[reliable] = "RELIABLE"
    flags[thin] = "THIN_TRADING"
    flags[unstable] = "UNSTABLE"

    validity_status = np.where(
        (flags == "RELIABLE") | (flags == "LIMITED_HISTORY"),
        "VALID_FOR_MODELING",
        np.where(flags == "INSUFFICIENT_DATA", "UNAVAILABLE_WARMUP", "CAUTION_USE_WINSORIZED_OR_MASK"),
    )

    out_df = pd.DataFrame({
        "SecurityID": mr["SecurityID"],
        "CompanyCode": mr["CompanyCode"],
        "Date": mr["Date"],
        "MainType": mr["MainType"],
        "SubType": mr["SubType"],
        "Beta_ASPI_60": b60,
        "Beta_ASPI_252": b252,
        "RS_ASPI_ROC_20": mr["RS_ASPI_ROC_20"].to_numpy(dtype=np.float64),
        "ASPI_MOM_20": mr["ASPI_MOM_20"].to_numpy(dtype=np.float64),
        "beta60_observation_count": obs_60,
        "beta60_nonzero_return_obs": nz_60,
        "beta252_observation_count": obs_252,
        "beta252_nonzero_return_obs": nz_252,
        "beta_validity_status": validity_status,
        "beta_reliability_flag": flags,
    })

    # Sensitivity & validation against CSE Official 2025Q4 beta across reliability tiers
    from phase2b_audit.market_ca_redundancy import _parse_cse_official_beta
    off = _parse_cse_official_beta(raw_dir, "2025Q4")
    tier_comp: dict[str, Any] = {}
    if not off.empty:
        dec25 = out_df[(out_df["MainType"] == "N") & (out_df["Date"] >= "2025-12-01") & (out_df["Date"] <= "2025-12-31")]
        latest = dec25.sort_values("Date").groupby("CompanyCode").last().reset_index()
        m = off.merge(latest, on="CompanyCode", how="inner").dropna(subset=["Official_ASPI_Beta", "Beta_ASPI_60"])
        for tier, grp_t in m.groupby("beta_reliability_flag"):
            if len(grp_t) >= 5:
                tier_comp[str(tier)] = {
                    "companies": int(len(grp_t)),
                    "pearson_corr_vs_official": round(float(grp_t["Official_ASPI_Beta"].corr(grp_t["Beta_ASPI_60"])), 4),
                    "mae_vs_official": round(float(np.mean(np.abs(grp_t["Official_ASPI_Beta"] - grp_t["Beta_ASPI_60"]))), 4),
                }

    # Sensitivity across min_nonzero thresholds inside 60d window
    min_nz_sens = []
    for min_nz in (10, 15, 20, 30, 40):
        pass_mask = has_b60 & (nz_60 >= min_nz) & (np.abs(b60) <= 4.0)
        min_nz_sens.append({
            "min_nonzero_returns_in_60d": min_nz,
            "passing_rows": int(pass_mask.sum()),
            "passing_pct_of_panel": round(100.0 * float(pass_mask.mean()), 2),
        })

    summary = {
        "row_counts_by_reliability_flag": {str(k): int(v) for k, v in out_df["beta_reliability_flag"].value_counts().items()},
        "row_counts_by_validity_status": {str(k): int(v) for k, v in out_df["beta_validity_status"].value_counts().items()},
        "beta60_valid_gt4_rows_flagged_unstable": int((flags == "UNSTABLE").sum()),
        "min_nonzero_observation_sensitivity": min_nz_sens,
        "official_2025q4_agreement_by_tier": tier_comp,
    }
    return out_df, summary
