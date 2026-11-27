"""
Phase 3C Workstreams F, G & H – Historical Share Counts, Dividends, TTM Flows & Market Snapshots
=================================================================================================
Implements strictly Point-in-Time reconstruction of:
  1. Historical Issued Share Counts (`01_existing_cse_sources/16 List of Quoted Securities`
     `2005–2025` + `21Market Capitalisation` `1999–2025`) propagated causally through verified
     Phase 1B Stock Splits, Bonus Issues, Scrip Dividends, and `CumulativeSharesListed`.
  2. Strict PIT Market Price & Capital-Adjustment lookup (`Phase 2C` / `Phase 1B total return`,
     never using post-2025 prices or future share counts).
  3. Cash Dividend metrics (`Workstream F`) strictly isolating `VALIDATED` cash dividends from
     `2,787 UNRESOLVED_PERCENT_OF_PAR` pre-2008 rows and non-cash splits/bonus/scrip/rights.
  4. Trailing-Twelve-Month (`TTM`, Workstream H) metrics from 4 compatible standalone quarters.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xlrd

logger = logging.getLogger("phase3c.div_val_ttm")

MAX_HISTORICAL_PRICE_DATE = pd.Timestamp("2025-12-31")


@dataclass
class HistoricalShareSchedule:
    """In-memory fast index of historical year-end issued shares, Phase 1B CA, and daily market prices."""
    yearly_sec_shares: dict[str, list[tuple[pd.Timestamp, float, str]]]      # SecurityID -> sorted [(date, qty, src)]
    yearly_code_shares: dict[str, list[tuple[pd.Timestamp, float, str]]]     # CompanyCode -> combined ordinary equity [(date, qty, src)]
    sec_classes_by_code: dict[str, list[str]]                                # CompanyCode -> ['COMB-N-0000', 'COMB-X-0000']
    ca_events_by_sec: dict[str, pd.DataFrame]                                # Split/Bonus/Scrip/Rights/CumShares
    cash_divs_by_code: dict[str, pd.DataFrame]                               # Validated cash dividends
    unresolved_par_by_code: dict[str, np.ndarray]                            # Sorted ExDate timestamps of 2,787 unresolved % par
    non_cash_ca_by_code: dict[str, pd.DataFrame]                             # Scrip/Bonus/Split/Rights for non-cash tracking
    price_arrays_by_sec: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # SecurityID -> (dates_ns, raw_close, sb_ratio)


def _parse_list_16_shares(xls_path: Path) -> pd.DataFrame:
    """Parse all annual sheets (2005–2025) of `16 List of Quoted Securities And Issued Quantity.xls`."""
    if not xls_path.exists():
        return pd.DataFrame(columns=["SnapshotDate", "CompanyCode", "SecurityID", "IssuedQuantity", "Source"])

    wb = xlrd.open_workbook(str(xls_path))
    rows: list[tuple[pd.Timestamp, str, str, float, str]] = []
    for sh in wb.sheet_names():
        m_yr = re.search(r"(20\d{2}|19\d{2})", str(sh))
        if not m_yr:
            continue
        yr = int(m_yr.group(1))
        snap_dt = pd.Timestamp(year=yr, month=12, day=31)
        s = wb.sheet_by_name(sh)
        hdr_r = next(
            (r for r in range(min(12, s.nrows)) if any("SECURITY" in str(v).upper() for v in s.row_values(r))),
            None,
        )
        if hdr_r is None:
            continue
        hdr = [str(v).strip().upper() for v in s.row_values(hdr_r)]
        q_col = next((i for i, h in enumerate(hdr) if "QUANTITY" in h or "ISSUED" in h), 2)
        for r in range(hdr_r + 1, s.nrows):
            v0 = re.sub(r"\s+", "", str(s.cell_value(r, 0))).upper()
            v4 = re.sub(r"\s+", "", str(s.cell_value(r, 4))).upper() if s.ncols > 4 else ""
            raw_q = str(s.cell_value(r, q_col)).replace(",", "").strip()
            try:
                qf = float(raw_q)
            except ValueError:
                continue
            if not v0 or qf <= 0:
                continue
            if "-" not in v0 and "/" not in v0 and "-" in v4:
                sid = v4 if v4.startswith(v0) else f"{v0}-N-0000"
            else:
                sid = v0
            if "-" not in sid:
                continue
            # Restrict ordinary & non-voting voting/equity share classes (exclude Debentures, Preference, Warrants, Rights)
            parts = sid.split("-")
            if len(parts) < 3 or parts[1] not in ("N", "X"):
                continue
            cc = parts[0]
            rows.append((snap_dt, cc, sid, qf, f"List16_{yr}"))

    df = pd.DataFrame(rows, columns=["SnapshotDate", "CompanyCode", "SecurityID", "IssuedQuantity", "Source"])
    if not df.empty:
        df = df.sort_values(["SecurityID", "SnapshotDate"]).drop_duplicates(["SecurityID", "SnapshotDate"], keep="last")
    return df


def _parse_list_21_shares(xls_path: Path) -> pd.DataFrame:
    """Parse annual sheets (1999–2025) of `21Market Capitalisation of Listed Companies.xls`."""
    if not xls_path.exists():
        return pd.DataFrame(columns=["SnapshotDate", "CompanyCode", "SecurityID", "IssuedQuantity", "Source"])

    wb = xlrd.open_workbook(str(xls_path))
    rows: list[tuple[pd.Timestamp, str, str, float, str]] = []
    for sh in wb.sheet_names():
        m_yr = re.search(r"(20\d{2}|19\d{2})", str(sh))
        if not m_yr:
            continue
        yr = int(m_yr.group(1))
        snap_dt = pd.Timestamp(year=yr, month=12, day=31)
        s = wb.sheet_by_name(sh)
        hdr_r = next(
            (r for r in range(min(12, s.nrows)) if any("QUANTITY" in str(v).upper() or "SHARES" in str(v).upper() for v in s.row_values(r))),
            None,
        )
        if hdr_r is None:
            continue
        hdr = [str(v).strip().upper() for v in s.row_values(hdr_r)]
        q_col = next((i for i, h in enumerate(hdr) if "QUANTITY" in h or "SHARES" in h), None)
        st_col = next((i for i, h in enumerate(hdr) if "SECURITY" in h or "TYPE" in h), None)
        if q_col is None:
            continue
        for r in range(hdr_r + 1, s.nrows):
            cc = str(s.cell_value(r, 0)).strip().upper()
            st = re.sub(r"\s+", "", str(s.cell_value(r, st_col))).upper() if st_col is not None else f"{cc}.N0000"
            raw_q = str(s.cell_value(r, q_col)).replace(",", "").strip()
            try:
                qf = float(raw_q)
            except ValueError:
                continue
            if not cc or cc in ("TOTAL", "NAN") or qf <= 0:
                continue
            m_st = re.match(r"^([A-Z0-9]+)\.([A-Z])(\d+)$", st)
            sid = f"{m_st.group(1)}-{m_st.group(2)}-{m_st.group(3)}" if m_st else f"{cc}-N-0000"
            parts = sid.split("-")
            if len(parts) < 3 or parts[1] not in ("N", "X"):
                continue
            rows.append((snap_dt, cc, sid, qf, f"List21_{yr}"))

    df = pd.DataFrame(rows, columns=["SnapshotDate", "CompanyCode", "SecurityID", "IssuedQuantity", "Source"])
    if not df.empty:
        df = df.sort_values(["SecurityID", "SnapshotDate"]).drop_duplicates(["SecurityID", "SnapshotDate"], keep="last")
    return df


def build_historical_share_schedule(
    repo_root: Path,
    ca_df: pd.DataFrame,
    tr_df: pd.DataFrame | None = None,
) -> HistoricalShareSchedule:
    """
    Construct the verified causal share-count schedule, Phase 1B dividend index, and daily
    split/bonus price lookup table (`<= 2025-12-31`). Zero post-decision leakage.
    """
    df16 = _parse_list_16_shares(repo_root / "01_existing_cse_sources" / "16 List of Quoted Securities And Issued Quantity.xls")
    df21 = _parse_list_21_shares(repo_root / "01_existing_cse_sources" / "21Market Capitalisation of Listed Companies.xls")

    # Prefer List 16 class-accurate counts over List 21 whenever both exist for same (SecurityID, SnapshotDate)
    if not df16.empty and not df21.empty:
        merged = pd.concat([df16, df21], ignore_index=True)
        merged["Priority"] = np.where(merged["Source"].str.startswith("List16"), 0, 1)
        merged = merged.sort_values(["SecurityID", "SnapshotDate", "Priority"]).drop_duplicates(
            ["SecurityID", "SnapshotDate"], keep="first"
        )
    elif not df16.empty:
        merged = df16.copy()
    else:
        merged = df21.copy()

    yearly_sec: dict[str, list[tuple[pd.Timestamp, float, str]]] = {}
    for sid, grp in merged.groupby("SecurityID", sort=False):
        yearly_sec[str(sid)] = [
            (pd.Timestamp(r["SnapshotDate"]), float(r["IssuedQuantity"]), str(r["Source"]))
            for _, r in grp.sort_values("SnapshotDate").iterrows()
        ]

    # Combined equity shares (N + X) per CompanyCode & SnapshotDate
    sec_classes: dict[str, list[str]] = {}
    for sid in merged["SecurityID"].unique():
        cc = str(sid).split("-")[0]
        sec_classes.setdefault(cc, []).append(str(sid))
    for cc in sec_classes:
        sec_classes[cc] = sorted(sec_classes[cc], key=lambda x: (0 if "-N-" in x else 1, x))

    yearly_code: dict[str, list[tuple[pd.Timestamp, float, str]]] = {}
    code_sum = merged.groupby(["CompanyCode", "SnapshotDate"]).agg(
        IssuedQuantity=("IssuedQuantity", "sum"),
        Source=("Source", "first"),
    ).reset_index()
    for cc, grp in code_sum.groupby("CompanyCode", sort=False):
        yearly_code[str(cc)] = [
            (pd.Timestamp(r["SnapshotDate"]), float(r["IssuedQuantity"]), str(r["Source"]))
            for _, r in grp.sort_values("SnapshotDate").iterrows()
        ]

    # Index corporate actions cleanly
    ca = ca_df.copy()
    ca["ExDt"] = pd.to_datetime(ca["ExDate"], errors="coerce")
    ca["EffDt"] = pd.to_datetime(ca["EffectiveDate"], errors="coerce")
    ca["EventDt"] = ca["ExDt"].fillna(ca["EffDt"])
    ca["CompanyCode_str"] = ca["CompanyCode"].astype(str).str.strip()
    ca["SecurityID_str"] = ca["SecurityID"].astype(str).str.strip()

    # Capital events that alter share count cleanly
    cap_mask = ca["EventType"].isin(["StockSplit", "BonusIssue", "ScripDividend", "RightsIssue"]) & (
        ca["ValidationStatus"].astype(str).str.startswith("VALIDATED")
    )
    ca_cap = ca.loc[cap_mask & ca["EventDt"].notna()].sort_values("EventDt")
    ca_events_by_sec = {str(k): g.reset_index(drop=True) for k, g in ca_cap.groupby("SecurityID_str", sort=False)}

    # Non-cash CA events (for separate non-cash audit counts)
    non_cash = ca.loc[ca["EventType"].isin(["ScripDividend", "BonusIssue", "StockSplit", "RightsIssue"]) & ca["EventDt"].notna()].sort_values("EventDt")
    non_cash_by_code = {str(k): g.reset_index(drop=True) for k, g in non_cash.groupby("CompanyCode_str", sort=False)}

    # Validated Cash Dividends vs 2,787 UNRESOLVED_PERCENT_OF_PAR
    cd_val = ca.loc[
        (ca["EventType"] == "CashDividend")
        & (ca["ValidationStatus"] == "VALIDATED")
        & ca["EventDt"].notna()
        & (pd.to_numeric(ca["DividendPerShare"], errors="coerce") > 0)
    ].sort_values("EventDt")

    par_unres = ca.loc[(ca["EventType"] == "CashDividend") & (ca["ValidationStatus"] == "UNRESOLVED_PERCENT_OF_PAR") & ca["EventDt"].notna()]
    unresolved_par_by_code = {
        str(k): g["EventDt"].sort_values().values.astype("datetime64[ns]")
        for k, g in par_unres.groupby("CompanyCode_str", sort=False)
    }

    # Index Phase 1B total-return / Phase 2C daily raw prices & split-bonus factor ratios
    if tr_df is None:
        tr_path = repo_root / "Dataset" / "processed" / "phase1b" / "cse_total_return_history.parquet"
        tr_df = pd.read_parquet(
            tr_path,
            columns=["SecurityID", "Date", "RawClose", "SplitBonusAdjClose", "CumCapitalFactor"],
        )
    tr_sub = tr_df.copy()
    tr_sub["Date"] = pd.to_datetime(tr_sub["Date"], errors="coerce")
    tr_sub = tr_sub[tr_sub["Date"].notna() & (pd.to_numeric(tr_sub["RawClose"], errors="coerce") > 0)].sort_values(["SecurityID", "Date"])
    rc = tr_sub["RawClose"].astype(float).values
    sb = pd.to_numeric(tr_sub["SplitBonusAdjClose"], errors="coerce").fillna(tr_sub["RawClose"]).astype(float).values
    tr_sub["_sb_ratio"] = np.where(rc > 0, sb / rc, 1.0)

    price_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid, grp in tr_sub.groupby("SecurityID", sort=False):
        price_arrays[str(sid)] = (
            grp["Date"].values.astype("datetime64[ns]"),
            grp["RawClose"].astype(float).values,
            grp["_sb_ratio"].astype(float).values,
        )

    # Pre-scale CashDividends once per event (`AdjDPS_base = DividendPerShare * sb_ex`) for O(1) runtime lookup
    tmp_sched = HistoricalShareSchedule(
        yearly_sec_shares=yearly_sec,
        yearly_code_shares=yearly_code,
        sec_classes_by_code=sec_classes,
        ca_events_by_sec=ca_events_by_sec,
        cash_divs_by_code={},
        unresolved_par_by_code=unresolved_par_by_code,
        non_cash_ca_by_code=non_cash_by_code,
        price_arrays_by_sec=price_arrays,
    )
    cash_divs_by_code: dict[str, pd.DataFrame] = {}
    for k, g in cd_val.groupby("CompanyCode_str", sort=False):
        gc = g.reset_index(drop=True).copy()
        sb_bases: list[float] = []
        for sid_v, ex_v, dps_v in zip(gc["SecurityID_str"], gc["EventDt"], gc["DividendPerShare"]):
            _, _, sb_ex = _lookup_price_at_date(tmp_sched, str(sid_v), pd.Timestamp(ex_v), max_staleness_days=60)
            sb_bases.append(float(dps_v) * (float(sb_ex) if pd.notna(sb_ex) and sb_ex > 0 else 1.0))
        gc["AdjDPS_base"] = sb_bases
        gc["DivYear"] = pd.to_datetime(gc["EventDt"]).dt.year
        cash_divs_by_code[str(k)] = gc

    tmp_sched.cash_divs_by_code = cash_divs_by_code
    return tmp_sched


def _lookup_price_at_date(
    sched: HistoricalShareSchedule,
    sec_id: str,
    decision_date: pd.Timestamp,
    max_staleness_days: int = 45,
) -> tuple[float, pd.Timestamp | pd.NaT, float]:
    """
    Causal price lookup in Phase 1B/2C daily panel (`Date <= min(DecisionDate, 2025-12-31)`).
    Never uses prices > DecisionDate or prices > 45 calendar days stale or post-2025.
    Returns `(RawClose, PriceDate, SplitBonusRatio)`.
    """
    arrs = sched.price_arrays_by_sec.get(sec_id)
    if arrs is None:
        return np.nan, pd.NaT, 1.0
    if decision_date > MAX_HISTORICAL_PRICE_DATE + pd.Timedelta(days=max_staleness_days):
        return np.nan, pd.NaT, 1.0

    dates_ns, closes, ratios = arrs
    target_ns = np.datetime64(min(decision_date, MAX_HISTORICAL_PRICE_DATE).to_datetime64(), "ns")
    pos = int(np.searchsorted(dates_ns, target_ns, side="right")) - 1
    if pos < 0:
        return np.nan, pd.NaT, 1.0
    p_dt = pd.Timestamp(dates_ns[pos])
    if (decision_date - p_dt).days > max_staleness_days:
        return np.nan, pd.NaT, 1.0
    return float(closes[pos]), p_dt, float(ratios[pos])


def _get_sec_shares_as_of(
    sched: HistoricalShareSchedule,
    sec_id: str,
    decision_date: pd.Timestamp,
) -> tuple[float, str]:
    """
    Causal share count for a single SecurityID as of `decision_date`:
    Starts from latest List 16/21 annual snapshot `SnapshotDate <= decision_date`, then propagates
    verified `StockSplit` / `BonusIssue` / `ScripDividend` / `CumulativeSharesListed` with `EventDt`
    strictly in `(SnapshotDate, decision_date]`. Never uses future share snapshots.
    """
    snaps = sched.yearly_sec_shares.get(sec_id, [])
    base_dt = pd.NaT
    base_qty = np.nan
    src = "UNAVAILABLE"
    for sdt, qty, sname in snaps:
        if sdt <= decision_date:
            base_dt = sdt
            base_qty = qty
            src = sname
        else:
            break

    ev_df = sched.ca_events_by_sec.get(sec_id)
    if ev_df is not None and not ev_df.empty:
        sub = ev_df[ev_df["EventDt"] <= decision_date]
        if pd.notna(base_dt):
            sub = sub[sub["EventDt"] > base_dt]
        for _, er in sub.iterrows():
            cum_listed = pd.to_numeric(er.get("CumulativeSharesListed"), errors="coerce")
            etype = str(er["EventType"])
            pf = pd.to_numeric(er.get("PriceFactor"), errors="coerce")
            old_p = pd.to_numeric(er.get("OldProportion"), errors="coerce")
            new_p = pd.to_numeric(er.get("NewProportion"), errors="coerce")
            if pd.isna(base_qty):
                # Only adopt CumulativeSharesListed as initial count if no prior List 16/21 base exists AND it's a split/bonus total
                continue
            if etype == "StockSplit" and pd.notna(old_p) and pd.notna(new_p) and old_p > 0 and new_p > 0:
                base_qty *= (new_p / old_p)
                src += "+SplitAdj"
            elif etype in ("BonusIssue", "ScripDividend"):
                # Note: CumulativeSharesListed in Phase 1B CA holds incremental shares allotted for scrip/bonus when < 0.5*base_qty
                if pd.notna(cum_listed) and 0 < cum_listed < 0.85 * base_qty:
                    base_qty += float(cum_listed)
                    src += "+CA_AllotAdj"
                elif pd.notna(pf) and 0 < pf <= 1.0:
                    base_qty *= (1.0 / pf)
                    src += "+BonusAdj"
            elif etype == "RightsIssue":
                if pd.notna(cum_listed) and 0 < cum_listed < 1.5 * base_qty:
                    base_qty += float(cum_listed)
                    src += "+RightsAllotAdj"
                elif pd.notna(old_p) and pd.notna(new_p) and old_p > 0 and new_p > 0:
                    base_qty *= ((old_p + new_p) / old_p)
                    src += "+RightsRatioAdj"

    return (float(base_qty) if pd.notna(base_qty) else np.nan), src


def get_market_snapshot_as_of(
    sched: HistoricalShareSchedule,
    company_code: str,
    primary_sec_id: str,
    decision_date: pd.Timestamp,
) -> dict[str, Any]:
    """
    Return exact causal equity price, multi-class total issued shares, and MarketCap (LKR) as of `decision_date`.
    Handles multi-class `-N-` + `-X-` securities explicitly so entity-level Market Cap matches consolidated Equity/PAT.
    """
    classes = sched.sec_classes_by_code.get(company_code) or ([primary_sec_id] if primary_sec_id else [])
    if not classes and primary_sec_id:
        classes = [primary_sec_id]

    p_main, p_dt_main, sb_main = _lookup_price_at_date(sched, primary_sec_id, decision_date)
    total_shares = 0.0
    total_mcap = 0.0
    have_all_shares = False
    have_any_mcap = False
    src_parts: list[str] = []
    class_used: list[str] = []

    for sid in classes:
        qty, qsrc = _get_sec_shares_as_of(sched, sid, decision_date)
        if pd.isna(qty) or qty <= 0:
            continue
        have_all_shares = True
        total_shares += qty
        src_parts.append(qsrc)
        px, _, _ = _lookup_price_at_date(sched, sid, decision_date)
        # If non-voting class -X- had zero trades in 45d while -N- traded, do NOT mix unless -X- traded
        if pd.notna(px) and px > 0:
            total_mcap += px * qty
            have_any_mcap = True
            class_used.append(sid)

    # Fallback: if CompanyCode had no `-N-` price or class list missed primary shares
    if total_shares == 0.0 and primary_sec_id:
        q0, qsrc0 = _get_sec_shares_as_of(sched, primary_sec_id, decision_date)
        if pd.notna(q0) and q0 > 0:
            total_shares = q0
            src_parts = [qsrc0]
            if pd.notna(p_main) and p_main > 0:
                total_mcap = p_main * q0
                have_any_mcap = True
                class_used = [primary_sec_id]

    # If a multi-class company's -X- class didn't trade inside 45d but -N- did, compute voting-weighted proxy ONLY if noted,
    # otherwise scale only where both traded OR single-class exact:
    if len(classes) > 1 and pd.notna(p_main) and p_main > 0 and len(class_used) == 1 and total_shares > 0:
        # Multi-class non-voting shares didn't trade inside window: avoid false jump; use -N- Close across combined ordinary pool noted cleanly
        q_n, _ = _get_sec_shares_as_of(sched, primary_sec_id, decision_date)
        if pd.notna(q_n) and q_n > 0:
            total_mcap = p_main * total_shares
            src_parts.append("MULTI_CLASS_XN_PROXY_ON_N_CLOSE")

    return {
        "market_price_lkr": float(p_main) if pd.notna(p_main) else np.nan,
        "market_price_date": p_dt_main,
        "split_bonus_ratio_at_d": float(sb_main),
        "shares_outstanding_pit": float(total_shares) if (have_all_shares and total_shares > 0) else np.nan,
        "market_cap_lkr": float(total_mcap) if (have_any_mcap and total_mcap > 0) else np.nan,
        "share_count_source": ";".join(dict.fromkeys(src_parts)) if src_parts else "UNAVAILABLE",
        "is_multiclass_combined": bool(len(classes) > 1),
    }


def compute_dividends_as_of(
    sched: HistoricalShareSchedule,
    company_code: str,
    primary_sec_id: str,
    decision_date: pd.Timestamp,
    price_lkr: float,
    sb_ratio_at_d: float,
    ann_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Workstream F: Compute PIT cash dividend metrics using verified Phase 1B `CashDividend` events (`VALIDATED`)
    and Phase 3B Annual statement dividends, while strictly isolating `2,787 UNRESOLVED_PERCENT_OF_PAR`
    rows and non-cash Scrip/Bonus/Split/Rights distributions.
    """
    res: dict[str, Any] = {
        "ttm_cash_dps_lkr": np.nan,
        "annual_cash_dps_lkr": np.nan,
        "dividend_yield_pct": np.nan,
        "price_to_dividend": np.nan,
        "dividend_payout_ratio_pct": np.nan,
        "dividend_coverage_ratio": np.nan,
        "dividend_growth_yoy_pct": np.nan,
        "dividend_cagr_3y_pct": np.nan,
        "consecutive_dividend_years": 0,
        "dividend_consistency_5y_pct": 0.0,
        "unresolved_percent_of_par_div_count": 0,
        "non_cash_ca_events_365d_count": 0,
    }

    # Count excluded unresolved % of par events up to decision_date
    par_arr = sched.unresolved_par_by_code.get(company_code)
    if par_arr is not None:
        cutoff_ns = np.datetime64(decision_date.to_datetime64(), "ns")
        res["unresolved_percent_of_par_div_count"] = int(np.searchsorted(par_arr, cutoff_ns, side="right"))

    # Count non-cash Scrip/Split/Bonus/Rights inside trailing 365 days
    nc_df = sched.non_cash_ca_by_code.get(company_code)
    if nc_df is not None and not nc_df.empty:
        win_start = decision_date - pd.Timedelta(days=365)
        res["non_cash_ca_events_365d_count"] = int(((nc_df["EventDt"] > win_start) & (nc_df["EventDt"] <= decision_date)).sum())

    # Verified cash dividends (prefer primary -N- class to prevent N+X double-counting)
    cd_df = sched.cash_divs_by_code.get(company_code)
    if cd_df is not None and not cd_df.empty:
        ev_arr = cd_df["EventDt"].to_numpy()
        d_np = decision_date.to_datetime64()
        mask = ev_arr <= d_np
        if mask.any():
            sub_idx = np.where(mask)[0]
            sids = cd_df["SecurityID_str"].to_numpy()[sub_idx]
            n_sel = sids == primary_sec_id
            if n_sel.any():
                sub_idx = sub_idx[n_sel]
            else:
                first_sid = sids[0]
                sub_idx = sub_idx[sids == first_sid]

            denom_sb = sb_ratio_at_d if (pd.notna(sb_ratio_at_d) and sb_ratio_at_d > 0) else 1.0
            adj_dps_arr = cd_df["AdjDPS_base"].to_numpy()[sub_idx] / denom_sb
            sub_ev = ev_arr[sub_idx]
            sub_yrs = cd_df["DivYear"].to_numpy()[sub_idx]

            win_np = (decision_date - pd.Timedelta(days=365)).to_datetime64()
            ttm_dps = float(adj_dps_arr[sub_ev > win_np].sum())
            res["ttm_cash_dps_lkr"] = ttm_dps

            if pd.notna(price_lkr) and price_lkr > 0:
                res["dividend_yield_pct"] = (ttm_dps / price_lkr) * 100.0
                if ttm_dps > 0:
                    res["price_to_dividend"] = price_lkr / ttm_dps

            by_yr: dict[int, float] = {}
            for y_i, v_i in zip(sub_yrs, adj_dps_arr):
                yi = int(y_i)
                by_yr[yi] = by_yr.get(yi, 0.0) + float(v_i)

            cur_yr = int(decision_date.year)
            check_end_yr = cur_yr if (decision_date.month >= 11 and cur_yr in by_yr) else (cur_yr - 1)
            start_yr = check_end_yr if by_yr.get(check_end_yr, 0.0) > 0 else (check_end_yr - 1 if by_yr.get(check_end_yr - 1, 0.0) > 0 else None)
            consec = 0
            if start_yr is not None:
                y = start_yr
                while by_yr.get(y, 0.0) > 0:
                    consec += 1
                    y -= 1
            res["consecutive_dividend_years"] = consec

            paid5 = sum(1 for k in range(1, 6) if by_yr.get(cur_yr - k, 0.0) > 0)
            res["dividend_consistency_5y_pct"] = (paid5 / 5.0) * 100.0

            if check_end_yr in by_yr and by_yr[check_end_yr] > 0:
                res["annual_cash_dps_lkr"] = float(by_yr[check_end_yr])
                if (check_end_yr - 1) in by_yr and by_yr[check_end_yr - 1] > 0:
                    res["dividend_growth_yoy_pct"] = ((by_yr[check_end_yr] - by_yr[check_end_yr - 1]) / by_yr[check_end_yr - 1]) * 100.0
                if (check_end_yr - 3) in by_yr and by_yr[check_end_yr - 3] > 0:
                    res["dividend_cagr_3y_pct"] = ((by_yr[check_end_yr] / by_yr[check_end_yr - 3]) ** (1.0 / 3.0) - 1.0) * 100.0

    # Accept either DataFrame or pre-resolved `(mvals_dict, cur)` dict tuple for zero-overhead fast path
    mvals: dict[str, float] = {}
    ann_cur = ""
    if isinstance(ann_df, tuple):
        mvals, ann_cur = ann_df
    elif ann_df is not None and not ann_df.empty:
        latest_pe = ann_df["PeriodEnd_dt"].max()
        ann_lat = ann_df[ann_df["PeriodEnd_dt"] == latest_pe]
        mvals = dict(zip(ann_lat["MetricCode"], ann_lat["Value"]))
        ann_cur = str(ann_lat["Currency"].iloc[0]) if not ann_lat.empty else ""

    if pd.isna(res["annual_cash_dps_lkr"]) and mvals:
        fs_dps = mvals.get("DIVIDEND_PER_SHARE", np.nan)
        if pd.notna(fs_dps) and ann_cur == "LKR" and float(fs_dps) >= 0:
            res["annual_cash_dps_lkr"] = float(fs_dps)

    if mvals:
        pat = mvals.get("PROFIT_AFTER_TAX", np.nan)
        eps = mvals.get("EPS_BASIC", np.nan)
        div_cash = mvals.get("DIVIDENDS_PAID", mvals.get("EQUITY_DIVIDENDS", np.nan))
        if pd.notna(div_cash) and pd.notna(pat) and pat > 0:
            div_out = abs(float(div_cash))
            res["dividend_payout_ratio_pct"] = (div_out / float(pat)) * 100.0
            if div_out > 0:
                res["dividend_coverage_ratio"] = float(pat) / div_out
        elif pd.notna(res["annual_cash_dps_lkr"]) and pd.notna(eps) and eps > 0 and ann_cur == "LKR":
            d_val = float(res["annual_cash_dps_lkr"])
            res["dividend_payout_ratio_pct"] = (d_val / float(eps)) * 100.0
            if d_val > 0:
                res["dividend_coverage_ratio"] = float(eps) / d_val

    return res


def compute_ttm_from_qtr_snaps(
    qtr_snaps_asc: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, dict[str, float], dict[str, str]]],
    cur: str,
) -> dict[str, Any]:
    """
    Fast pure-Python TTM calculator given ascending chronological standalone Quarterly tuples:
    `[(PeriodStart_dt, PeriodEnd_dt, AvailabilityDate_dt, mvals_dict, fids_dict), ...]`
    strictly within identical ConsolidationStatus and Currency.
    """
    base_res: dict[str, Any] = {
        "is_ttm_complete": False,
        "ttm_period_end": pd.NaT,
        "ttm_availability_date": pd.NaT,
        "ttm_currency": "",
        "ttm_revenue": np.nan,
        "ttm_gross_profit": np.nan,
        "ttm_operating_profit": np.nan,
        "ttm_pbt": np.nan,
        "ttm_net_income": np.nan,
        "ttm_profit_attributable": np.nan,
        "ttm_eps": np.nan,
        "ttm_depreciation": np.nan,
        "ttm_amortization": np.nan,
        "ttm_ebitda": np.nan,
        "ttm_ebitda_status": "UNAVAILABLE_NO_TTM",
        "ttm_operating_cash_flow": np.nan,
        "ttm_capex_outflow": np.nan,
        "ttm_free_cash_flow": np.nan,
        "ttm_input_fact_ids": "",
    }
    n_q = len(qtr_snaps_asc)
    if n_q < 4:
        return base_res

    chosen: list[tuple] | None = None
    for end_idx in range(n_q - 1, 2, -1):
        cand = qtr_snaps_asc[end_idx - 3 : end_idx + 1]
        ok = True
        for k in range(3):
            gap = (cand[k + 1][0] - cand[k][1]).days
            if gap < -2 or gap > 15:
                ok = False
                break
        if ok:
            total_span = (cand[3][1] - cand[0][0]).days
            if 340 <= total_span <= 385:
                chosen = cand
                break

    if chosen is None:
        return base_res

    used_ids: list[str] = []

    def _sum4(metric: str, abs_val: bool = False) -> float:
        tot = 0.0
        for _, _, _, mv, fids in chosen:
            v = mv.get(metric)
            if v is None or np.isnan(v):
                return np.nan
            tot += abs(float(v)) if abs_val else float(v)
        for _, _, _, _, fids in chosen:
            fid = fids.get(metric)
            if fid:
                used_ids.append(fid)
        return tot

    rev = _sum4("REVENUE")
    gp = _sum4("GROSS_PROFIT")
    ebit = _sum4("OPERATING_PROFIT")
    pbt = _sum4("PROFIT_BEFORE_TAX")
    pat = _sum4("PROFIT_AFTER_TAX")
    pat_attr = _sum4("PROFIT_ATTRIBUTABLE_EQUITY_HOLDERS")
    eps = _sum4("EPS_BASIC")
    depr = _sum4("DEPRECIATION", abs_val=True)
    amort = _sum4("AMORTIZATION", abs_val=True)
    ocf = _sum4("OPERATING_CASH_FLOW")
    capex_out = _sum4("CAPEX", abs_val=True)
    fcf = (ocf - capex_out) if (pd.notna(ocf) and pd.notna(capex_out)) else np.nan

    ebitda = np.nan
    ebitda_st = "UNAVAILABLE_MISSING_EBIT_OR_DEPR"
    if pd.notna(ebit) and pd.notna(depr):
        if pd.notna(amort):
            ebitda = ebit + depr + amort
            ebitda_st = "EBIT_PLUS_DEPR_PLUS_AMORT"
        else:
            ebitda = ebit + depr
            ebitda_st = "EBIT_PLUS_DEPR_NO_AMORT"
    elif pd.notna(ebit):
        rep_eb = _sum4("EBITDA")
        if pd.notna(rep_eb):
            ebitda = rep_eb
            ebitda_st = "REPORTED_EBITDA_4Q"

    max_av = max((c[2] for c in chosen if pd.notna(c[2])), default=pd.NaT)
    base_res.update({
        "is_ttm_complete": True,
        "ttm_period_end": chosen[3][1],
        "ttm_availability_date": max_av,
        "ttm_currency": cur,
        "ttm_revenue": rev,
        "ttm_gross_profit": gp,
        "ttm_operating_profit": ebit,
        "ttm_pbt": pbt,
        "ttm_net_income": pat,
        "ttm_profit_attributable": pat_attr,
        "ttm_eps": eps,
        "ttm_depreciation": depr,
        "ttm_amortization": amort,
        "ttm_ebitda": ebitda,
        "ttm_ebitda_status": ebitda_st,
        "ttm_operating_cash_flow": ocf,
        "ttm_capex_outflow": capex_out,
        "ttm_free_cash_flow": fcf,
        "ttm_input_fact_ids": ";".join(sorted(set(used_ids))),
    })
    return base_res


def compute_ttm_from_active_facts(active_flows_df: pd.DataFrame) -> dict[str, Any]:
    """
    Workstream H: Construct strictly consecutive Trailing-Twelve-Month (TTM) Income Statement and
    Cash Flow metrics from 4 sequential standalone Quarterly (`3M`) periods.
    """
    if active_flows_df.empty:
        return compute_ttm_from_qtr_snaps([], "")

    q_df = active_flows_df[
        (active_flows_df["PeriodType"] == "Quarterly")
        & active_flows_df["StatementType"].isin(["Income Statement", "Cash Flow"])
        & active_flows_df["PeriodStart_dt"].notna()
        & active_flows_df["PeriodEnd_dt"].notna()
    ]
    if q_df.empty:
        return compute_ttm_from_qtr_snaps([], "")
    dur = (q_df["PeriodEnd_dt"] - q_df["PeriodStart_dt"]).dt.days
    q_df = q_df[dur.between(60, 115)]
    if q_df.empty:
        return compute_ttm_from_qtr_snaps([], "")

    # Resolve latest version per (PeriodEnd, MetricCode) in linear python dict pass
    best_by_slice: dict[tuple[pd.Timestamp, str], tuple[tuple, pd.Timestamp, pd.Timestamp, pd.Timestamp, str, float, str]] = {}
    for ps, pe, av, mc, cur, val, fid, rv, ider, conf, did in zip(
        q_df["PeriodStart_dt"], q_df["PeriodEnd_dt"], q_df["AvailabilityDate_dt"],
        q_df["MetricCode"], q_df["Currency"], q_df["Value"], q_df["FactID"],
        q_df["RestatementVersion"], q_df["IsDerived"], q_df["ExtractionConfidence"], q_df["SourceDocumentID"],
    ):
        if pd.isna(val):
            continue
        pe_ts = pd.Timestamp(pe)
        ps_ts = pd.Timestamp(ps)
        av_ts = pd.Timestamp(av) if pd.notna(av) else pd.NaT
        rank = (int(rv or 1), av_ts.value if pd.notna(av_ts) else 0, 0 if ider else 1, float(conf or 0.0), str(did))
        key = (pe_ts, str(mc))
        prev = best_by_slice.get(key)
        if prev is None or rank > prev[0]:
            best_by_slice[key] = (rank, ps_ts, pe_ts, av_ts, str(cur), float(val), str(fid))

    by_cur_pe: dict[tuple[str, pd.Timestamp], list[Any]] = {}
    for (_, mc), (_, ps_ts, pe_ts, av_ts, cur, val, fid) in best_by_slice.items():
        ent = by_cur_pe.get((cur, pe_ts))
        if ent is None:
            ent = [ps_ts, pe_ts, av_ts, {}, {}]
            by_cur_pe[(cur, pe_ts)] = ent
        if pd.notna(av_ts) and (pd.isna(ent[2]) or av_ts > ent[2]):
            ent[2] = av_ts
        ent[3][mc] = val
        ent[4][mc] = fid

    # Evaluate dominant / latest currency cleanly
    curs = sorted({cur for (cur, _) in by_cur_pe})
    for cur in sorted(curs, key=lambda c: sum(1 for (x, _) in by_cur_pe if x == c), reverse=True):
        q_list = sorted([(ps, pe, av, mv, fm) for (c, _), (ps, pe, av, mv, fm) in by_cur_pe.items() if c == cur], key=lambda t: t[1])
        res = compute_ttm_from_qtr_snaps(q_list, cur)
        if res["is_ttm_complete"]:
            return res
    return compute_ttm_from_qtr_snaps([], "")
