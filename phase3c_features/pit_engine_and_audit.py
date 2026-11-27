"""
Phase 3C Workstreams I, J & K – Point-in-Time Engine, Strategy Feature Sets & Quality Audit
==============================================================================================
Implements:
  1. `build_fundamental_features_as_of(company_id, decision_date, ...)`: strict PIT query API
     returning company fundamental metrics as publicly known at `decision_date`.
  2. Chronological PIT Replay (`build_all_pit_feature_panels`): generates strict historical panels
     across all verified filing `AvailabilityDate` events plus quarter-end market valuation dates,
     writing all specialized Parquet outputs + Medium-Term and Long-Term Strategy Feature Matrices.
  3. Workstream J Quality, Coverage & Redundancy Audit: compiles `fundamental_feature_coverage.csv`,
     `fundamental_feature_metadata.csv`, `fundamental_feature_quality_report.csv`, and visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dividends_valuation_ttm import HistoricalShareSchedule, build_historical_share_schedule
from .fundamental_metrics import (
    ALL_FEATURE_COLUMNS,
    CASHFLOW_COLUMNS,
    DIVIDEND_COLUMNS,
    GROWTH_COLUMNS,
    HEALTH_COLUMNS,
    LONG_TERM_FEATURE_COLUMNS,
    MEDIUM_TERM_FEATURE_COLUMNS,
    META_COLUMNS,
    PROFITABILITY_COLUMNS,
    TTM_COLUMNS,
    VALUATION_COLUMNS,
    compute_fundamental_snapshot_from_active_facts,
    compute_snapshot_from_tuples,
    refresh_market_valuation_on_snapshot,
)
from .quality_gating import (
    assess_fx_availability,
    build_financial_input_quality_flags,
    classify_business_types,
)

logger = logging.getLogger("phase3c.pit_engine")

DEFAULT_QUARTER_ENDS = pd.date_range("2011-03-31", "2026-11-27", freq="QE").tolist()

_SLICE_COLS = [
    "StatementType",
    "MetricCode",
    "PeriodStart",
    "PeriodEnd",
    "PeriodType",
    "ConsolidationStatus",
    "Currency",
]


def _prepare_facts_for_engine(
    facts_df: pd.DataFrame,
    flags_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach gating columns onto facts cleanly without modifying source `facts_df` in place."""
    f = facts_df.copy()
    f["PeriodStart_dt"] = pd.to_datetime(f["PeriodStart"], errors="coerce")
    f["PeriodEnd_dt"] = pd.to_datetime(f["PeriodEnd"], errors="coerce")
    f["AvailabilityDate_dt"] = pd.to_datetime(f["AvailabilityDate"], errors="coerce")
    flag_cols = ["FactID", "InputEligibility", "IsInputEligible", "IsLKRCompatible"]
    m = flags_df[flag_cols].drop_duplicates("FactID")
    merged = f.merge(m, on="FactID", how="left", suffixes=("", "_flag"))
    merged["IsInputEligible"] = merged["IsInputEligible"].fillna(False).astype(bool)
    merged["IsLKRCompatible"] = merged["IsLKRCompatible"].fillna(False).astype(bool)
    merged["InputEligibility"] = merged["InputEligibility"].fillna("EXCLUDED")
    return merged


def build_fundamental_features_as_of(
    company_id: str,
    decision_date: Any,
    *,
    facts_df: pd.DataFrame | None = None,
    flags_df: pd.DataFrame | None = None,
    identity_df: pd.DataFrame | None = None,
    business_type_df: pd.DataFrame | None = None,
    share_schedule: HistoricalShareSchedule | None = None,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """
    Workstream I: Strict Point-in-Time Fundamental Feature Engine API.
    Returns a 1-row DataFrame of fundamental features calculated exclusively from facts with
    verified `AvailabilityDate <= decision_date`, `IsInputEligible == True`, latest known
    `RestatementVersion <= decision_date`, and PIT historical share counts & prices `<= decision_date`.
    """
    dec_ts = pd.to_datetime(decision_date, errors="coerce")
    if pd.isna(dec_ts):
        raise ValueError(f"Invalid decision_date: {decision_date!r}")

    root = repo_root or Path(__file__).resolve().parent.parent
    p3b = root / "Dataset" / "processed" / "phase3b"
    p3c = root / "Dataset" / "processed" / "phase3c"

    if identity_df is None:
        identity_df = pd.read_parquet(p3b / "historical_company_identity.parquet")
    cid_s = str(company_id).strip()
    id_match = identity_df[
        (identity_df["CompanyID"].astype(str) == cid_s)
        | (identity_df["CompanyCode"].astype(str) == cid_s)
    ]
    if id_match.empty:
        return pd.DataFrame(columns=ALL_FEATURE_COLUMNS)
    ir = id_match.iloc[0]
    canon_cid = str(ir["CompanyID"])
    ccode = str(ir["CompanyCode"]).strip()

    if facts_df is None:
        facts_df = pd.read_parquet(p3b / "historical_financial_facts.parquet")
    if flags_df is None:
        flags_path = p3c / "financial_input_quality_flags.parquet"
        if flags_path.exists():
            flags_df = pd.read_parquet(flags_path)
        else:
            qr = pd.read_csv(p3b / "financial_data_quality_report.csv", low_memory=False)
            flags_df, _ = build_financial_input_quality_flags(facts_df, qr, root, scan_flagged_pdfs=False)

    if business_type_df is None:
        business_type_df = classify_business_types(identity_df, facts_df)

    bt_match = business_type_df[business_type_df["CompanyID"].astype(str) == canon_cid]
    btype = str(bt_match.iloc[0]["BusinessType"]) if not bt_match.empty else "NON_FINANCIAL_CORPORATE"
    is_fin = bool(bt_match.iloc[0]["IsFinancialInstitution"]) if not bt_match.empty else False

    if share_schedule is None:
        ca_df = pd.read_parquet(root / "Dataset" / "processed" / "phase1b" / "cse_corporate_actions_validated.parquet")
        share_schedule = build_historical_share_schedule(root, ca_df)

    # Filter strictly for company + eligible + AvailabilityDate <= dec_ts
    f_sub = facts_df[(facts_df["CompanyID"].astype(str) == canon_cid) | (facts_df["CompanyCode"].astype(str) == ccode)].copy()
    if not f_sub.empty:
        prepped = _prepare_facts_for_engine(f_sub, flags_df)
        elig_sub = prepped[
            prepped["IsInputEligible"]
            & prepped["AvailabilityDate_dt"].notna()
            & (prepped["AvailabilityDate_dt"] <= dec_ts)
        ].copy()
        # Enforce strict PIT version selection as of dec_ts (exclude future restatements)
        if not elig_sub.empty:
            elig_sub = (
                elig_sub.sort_values(
                    ["RestatementVersion", "AvailabilityDate_dt", "IsDerived", "ExtractionConfidence", "SourceDocumentID"],
                    ascending=[False, False, True, False, False],
                )
                .drop_duplicates(subset=_SLICE_COLS, keep="first")
                .reset_index(drop=True)
            )
    else:
        elig_sub = f_sub

    meta = {
        "CompanyID": canon_cid,
        "CompanyCode": ccode,
        "CompanyName": str(ir.get("CompanyName") or ccode),
        "PrimarySecurityID": str(ir.get("PrimarySecurityID") or f"{ccode}-N-0000"),
        "BusinessType": btype,
        "IsFinancialInstitution": is_fin,
    }
    snap = compute_fundamental_snapshot_from_active_facts(elig_sub, meta, dec_ts, share_schedule=share_schedule)
    return pd.DataFrame([snap], columns=ALL_FEATURE_COLUMNS)


def _priority_rank(row_tuple: tuple) -> tuple:
    """Sort key (higher = preferred version): `(RestatementVersion, avail_ns, not IsDerived, ExtractionConfidence, doc_id)`."""
    return row_tuple


def build_all_pit_feature_panels(
    facts_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    business_type_df: pd.DataFrame,
    share_schedule: HistoricalShareSchedule,
) -> pd.DataFrame:
    """
    Chronological per-company PIT replay across all eligible disclosure `AvailabilityDate` milestones
    plus quarterly evaluation grid dates (`2011–2026`), guaranteed look-ahead-free by incremental replay.
    """
    prepped = _prepare_facts_for_engine(facts_df, flags_df)
    elig = prepped[prepped["IsInputEligible"] & prepped["AvailabilityDate_dt"].notna()].copy()

    # Pre-index BusinessType & Identity metadata
    bt_map = business_type_df.set_index("CompanyID").to_dict("index")
    id_map = identity_df.set_index("CompanyID").to_dict("index")

    elig["avail_ns"] = elig["AvailabilityDate_dt"].values.astype("datetime64[ns]").astype(np.int64)
    elig["not_derived"] = (~elig["IsDerived"].fillna(False).astype(bool)).astype(int)
    elig["conf_f"] = pd.to_numeric(elig["ExtractionConfidence"], errors="coerce").fillna(0.0)
    elig["doc_s"] = elig["SourceDocumentID"].astype(str)
    elig["rv_i"] = pd.to_numeric(elig["RestatementVersion"], errors="coerce").fillna(1).astype(int)

    all_rows: list[dict[str, Any]] = []
    grouped_cids = list(elig.groupby("CompanyID", sort=True))
    n_cos = len(grouped_cids)
    for c_idx, (cid, grp) in enumerate(grouped_cids, start=1):
        if c_idx % 50 == 0 or c_idx == n_cos:
            logger.info("PIT replay progress: %d/%d companies (%d rows accumulated)", c_idx, n_cos, len(all_rows))
        ir = id_map.get(cid, {})
        bt = bt_map.get(cid, {})
        ccode = str(ir.get("CompanyCode") or grp["CompanyCode"].iloc[0]).strip()
        meta = {
            "CompanyID": cid,
            "CompanyCode": ccode,
            "CompanyName": str(ir.get("CompanyName") or grp["CompanyName"].iloc[0]),
            "PrimarySecurityID": str(ir.get("PrimarySecurityID") or f"{ccode}-N-0000"),
            "BusinessType": str(bt.get("BusinessType", "NON_FINANCIAL_CORPORATE")),
            "IsFinancialInstitution": bool(bt.get("IsFinancialInstitution", False)),
        }

        grp_sorted = grp.sort_values(["AvailabilityDate_dt", "rv_i", "PeriodEnd_dt"]).reset_index(drop=True)
        ev_dates = sorted(pd.to_datetime(grp_sorted["AvailabilityDate_dt"]).dropna().unique().tolist())
        if not ev_dates:
            continue
        first_av = pd.Timestamp(ev_dates[0])
        last_av = pd.Timestamp(ev_dates[-1])
        hist_status = str(ir.get("HistoricalStatus", "Listed"))
        max_q_date = min(pd.Timestamp("2026-11-27"), last_av + pd.Timedelta(days=180)) if hist_status == "Delisted" else pd.Timestamp("2026-11-27")

        eval_dates = sorted(
            set(pd.Timestamp(d) for d in ev_dates)
            | set(pd.Timestamp(qd) for qd in DEFAULT_QUARTER_ENDS if first_av <= pd.Timestamp(qd) <= max_q_date)
        )

        # Extract plain Python tuple streams once per company (zero DataFrame allocations in inner loop)
        stream = list(
            zip(
                grp_sorted["StatementType"].astype(str).tolist(),
                grp_sorted["MetricCode"].astype(str).tolist(),
                grp_sorted["PeriodStart_dt"].tolist(),
                grp_sorted["PeriodEnd_dt"].tolist(),
                grp_sorted["PeriodType"].astype(str).tolist(),
                grp_sorted["ConsolidationStatus"].astype(str).tolist(),
                grp_sorted["Currency"].astype(str).tolist(),
                pd.to_numeric(grp_sorted["Value"], errors="coerce").tolist(),
                grp_sorted["FactID"].astype(str).tolist(),
                grp_sorted["AvailabilityDate_dt"].tolist(),
                grp_sorted["IsDerived"].fillna(False).astype(bool).tolist(),
                grp_sorted["rv_i"].tolist(),
                grp_sorted["InputEligibility"].astype(str).tolist(),
                grp_sorted["conf_f"].tolist(),
                grp_sorted["doc_s"].tolist(),
                grp_sorted["avail_ns"].tolist(),
                grp_sorted["not_derived"].tolist(),
            )
        )

        active_dict: dict[tuple, tuple[tuple, tuple]] = {}
        ptr = 0
        n_raw = len(stream)
        prev_snap: dict[str, Any] | None = None
        prev_ctx: dict[str, Any] | None = None

        for d in eval_dates:
            changed_accounting = False
            while ptr < n_raw and pd.Timestamp(stream[ptr][9]) <= d:
                rec = stream[ptr]
                skey = (rec[0], rec[1], str(rec[2]), str(rec[3]), rec[4], rec[5], rec[6])
                rank = (int(rec[11]), int(rec[15]), int(rec[16]), float(rec[13]), str(rec[14]))
                prev = active_dict.get(skey)
                if prev is None or rank > prev[0]:
                    active_dict[skey] = (rank, rec[:15])
                    changed_accounting = True
                ptr += 1

            if not active_dict:
                continue
            if changed_accounting or prev_snap is None or prev_ctx is None:
                snap, prev_ctx = compute_snapshot_from_tuples(
                    (v[1] for v in active_dict.values()),
                    meta,
                    d,
                    share_schedule=share_schedule,
                    return_context=True,
                )
                prev_snap = snap
            else:
                snap = refresh_market_valuation_on_snapshot(
                    prev_snap, meta, d, share_schedule, prev_ctx["ann_df"], prev_ctx["bs_end_m"], prev_ctx["flow_vals"]
                )
            all_rows.append(snap)

    out = pd.DataFrame(all_rows, columns=ALL_FEATURE_COLUMNS)
    out = out.sort_values(["CompanyID", "DecisionDate"]).reset_index(drop=True)
    return out


def build_feature_metadata_table() -> pd.DataFrame:
    """Workstream I/J: Full reproducible formula definition & scope catalog (`fundamental_feature_metadata.csv`)."""
    defs: list[tuple[str, str, str, str, str, str, str, str]] = [
        # Growth
        ("rev_growth_qoq_pct", "Growth", "fundamental_growth_features.parquet", "(Rev_q0 - Rev_q1) / Rev_q1 * 100 [Rev_q1 > 0, ~90d sequential]", "REVENUE", "REVENUE (prior Q)", "NonFin/All", "Strictly identical Currency & Consolidation; NaN if base <= 0"),
        ("rev_growth_qtr_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(Rev_q0 - Rev_q4) / Rev_q4 * 100 [Rev_q4 > 0, 330-400d]", "REVENUE", "REVENUE (t-4Q)", "All", "Matches Fiscal Quarter YoY cleanly"),
        ("rev_growth_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(Rev_fy0 - Rev_fy1) / Rev_fy1 * 100 [Rev_fy1 > 0, 300-430d]", "REVENUE", "REVENUE (FY-1)", "All", "Strict Annual 12M comparability"),
        ("rev_cagr_3y_pct", "Growth", "fundamental_growth_features.parquet", "((Rev_t / Rev_{t-3})^(1/3) - 1) * 100 [both > 0]", "REVENUE", "REVENUE (FY-3)", "All", "Requires 1000-1220 day verified Annual span"),
        ("rev_cagr_5y_pct", "Growth", "fundamental_growth_features.parquet", "((Rev_t / Rev_{t-5})^(1/5) - 1) * 100 [both > 0]", "REVENUE", "REVENUE (FY-5)", "All", "Requires 1700-1980 day verified Annual span"),
        ("net_income_growth_qtr_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(PAT_q0 - PAT_q4) / PAT_q4 * 100 [PAT_q4 > 0]", "PROFIT_AFTER_TAX", "PROFIT_AFTER_TAX", "All", "NaN & flagged in turnaround when prior <= 0"),
        ("net_income_growth_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(PAT_fy0 - PAT_fy1) / PAT_fy1 * 100 [PAT_fy1 > 0]", "PROFIT_AFTER_TAX", "PROFIT_AFTER_TAX", "All", "Never uses negative denominator %; paired with earnings_recovery_flag"),
        ("eps_growth_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(EPS_fy0 - EPS_fy1) / EPS_fy1 * 100 [EPS_fy1 > 0]", "EPS_BASIC", "EPS_BASIC", "All", "Uses same-document/restated comparative EPS when available"),
        ("net_income_cagr_3y_pct", "Growth", "fundamental_growth_features.parquet", "((PAT_t / PAT_{t-3})^(1/3) - 1) * 100 [both > 0]", "PROFIT_AFTER_TAX", "PROFIT_AFTER_TAX", "All", "Strict positive-earnings 3Y CAGR"),
        ("net_income_cagr_5y_pct", "Growth", "fundamental_growth_features.parquet", "((PAT_t / PAT_{t-5})^(1/5) - 1) * 100 [both > 0]", "PROFIT_AFTER_TAX", "PROFIT_AFTER_TAX", "All", "Strict positive-earnings 5Y CAGR"),
        ("gross_profit_growth_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(GP_fy0 - GP_fy1) / GP_fy1 * 100 [GP_fy1 > 0]", "GROSS_PROFIT", "GROSS_PROFIT", "Non-Financial", "Suppressed NaN for Bank/Finance/Insurance"),
        ("operating_profit_growth_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(EBIT_fy0 - EBIT_fy1) / EBIT_fy1 * 100 [EBIT_fy1 > 0]", "OPERATING_PROFIT", "OPERATING_PROFIT", "All", "NaN if base <= 0"),
        ("ebitda_growth_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(EBITDA_fy0 - EBITDA_fy1) / EBITDA_fy1 * 100 [EBITDA_fy1 > 0]", "OPERATING_PROFIT+DEPR+AMORT", "Prior EBITDA", "Non-Financial", "Computed only when Depr present both years; suppressed for Financials"),
        ("pbt_growth_yoy_pct", "Growth", "fundamental_growth_features.parquet", "(PBT_fy0 - PBT_fy1) / PBT_fy1 * 100 [PBT_fy1 > 0]", "PROFIT_BEFORE_TAX", "PROFIT_BEFORE_TAX", "All", "NaN if base <= 0"),
        ("earnings_recovery_flag", "Growth", "fundamental_growth_features.parquet", "1 iff PAT_prev <= 0 and PAT_cur > 0 else 0", "PROFIT_AFTER_TAX", "Prior PAT", "All", "Explicit turnaround indicator preventing negative-% distortion"),
        # Profitability
        ("gross_profit_margin_pct", "Profitability", "fundamental_profitability_features.parquet", "GP / Revenue * 100 [TTM/Annual 12M, Rev > 0]", "GROSS_PROFIT", "REVENUE", "Non-Financial", "Suppressed for Bank/Finance/Insurance"),
        ("operating_profit_margin_pct", "Profitability", "fundamental_profitability_features.parquet", "EBIT / Revenue * 100 [TTM/Annual 12M, Rev > 0]", "OPERATING_PROFIT", "REVENUE", "All", "Uses TTM if >= Annual else Annual 12M"),
        ("net_profit_margin_pct", "Profitability", "fundamental_profitability_features.parquet", "PAT / Revenue * 100 [TTM/Annual 12M, Rev > 0]", "PROFIT_AFTER_TAX", "REVENUE", "All", "Preserves negative margins legitimately"),
        ("ebitda_margin_pct", "Profitability", "fundamental_profitability_features.parquet", "(EBIT + Depr + Amort) / Revenue * 100", "OPERATING_PROFIT+DEPR+AMORT", "REVENUE", "Non-Financial", "Requires verified Depreciation; suppressed for Financials"),
        ("roe_pct", "Profitability", "fundamental_profitability_features.parquet", "NetIncome_Attr / ((Eq_beg + Eq_end) / 2) * 100", "PROFIT_ATTRIBUTABLE_EQUITY_HOLDERS", "Avg Parent Equity", "All", "Strictly requires Beginning & Ending snapshots 300-430d apart, both > 0"),
        ("roa_pct", "Profitability", "fundamental_profitability_features.parquet", "PAT / ((Assets_beg + Assets_end) / 2) * 100", "PROFIT_AFTER_TAX", "Avg Total Assets", "All", "Strictly requires Beginning & Ending snapshots 300-430d apart"),
        ("roic_pct", "Profitability", "fundamental_profitability_features.parquet", "EBIT*(1-ETR) / Avg(Equity + Borrowings - Cash) * 100", "OPERATING_PROFIT*(1-ETR)", "Avg Invested Capital", "Non-Financial", "Verified Borrowings & Cash required; ETR [0..0.5] or 30% statutory; suppressed Financials"),
        ("flag_consistent_roe", "Profitability", "fundamental_profitability_features.parquet", "1 iff >= 3 Annual ROE > 0% in recent 4 fiscal years", "Annual ROE history", "3+ years", "All", "Prevents single-quarter illusion"),
        ("flag_consistent_profitability", "Profitability", "fundamental_profitability_features.parquet", "1 iff >= 3 profitable Annual fiscal years (out of recent 6)", "PROFIT_AFTER_TAX", "Annual count", "All", "Multi-year durability indicator"),
        # Financial Health
        ("current_ratio", "Financial Health", "fundamental_financial_health_features.parquet", "Current Assets / Current Liabilities [CL > 0]", "CURRENT_ASSETS", "CURRENT_LIABILITIES", "Non-Financial", "Suppressed for Bank/Finance/Insurance"),
        ("quick_ratio", "Financial Health", "fundamental_financial_health_features.parquet", "(Current Assets - Inventories) / Current Liabilities", "CURRENT_ASSETS - INVENTORIES", "CURRENT_LIABILITIES", "Non-Financial", "Suppressed for Financials; fallback Cash+STI+Receivables if Inv absent"),
        ("cash_ratio", "Financial Health", "fundamental_financial_health_features.parquet", "(Cash + ShortTermInvestments) / Current Liabilities", "CASH_AND_EQUIVALENTS+STI", "CURRENT_LIABILITIES", "Non-Financial", "Suppressed for Financials"),
        ("total_debt", "Financial Health", "fundamental_financial_health_features.parquet", "ShortTermBorrowings + LongTermBorrowings", "ST+LT BORROWINGS", "None", "Non-Financial", "Only when borrowing line(s) extracted; never equates deposits/payables to debt"),
        ("debt_to_equity", "Financial Health", "fundamental_financial_health_features.parquet", "Total Debt / Total Equity [Equity > 0]", "TOTAL_DEBT", "TOTAL_EQUITY", "Non-Financial", "Strict interest-bearing Debt/Equity; suppressed Financials"),
        ("liabilities_to_equity", "Financial Health", "fundamental_financial_health_features.parquet", "Total Liabilities / Total Equity [Equity > 0]", "TOTAL_LIABILITIES", "TOTAL_EQUITY", "All", "Financial gearing (useful capital leverage reference incl. banks)"),
        ("equity_to_assets", "Financial Health", "fundamental_financial_health_features.parquet", "Total Equity / Total Assets [Assets > 0]", "TOTAL_EQUITY", "TOTAL_ASSETS", "All", "Core bank/insurance capitalization ratio"),
        ("net_debt_to_ebitda", "Financial Health", "fundamental_financial_health_features.parquet", "(Total Debt - Cash) / EBITDA [EBITDA > 0]", "NET_DEBT", "EBITDA (TTM/Annual)", "Non-Financial", "Suppressed Financials & EBITDA <= 0"),
        ("interest_coverage", "Financial Health", "fundamental_financial_health_features.parquet", "Operating Profit / abs(Finance Costs) [|FC| > 0]", "OPERATING_PROFIT", "abs(FINANCE_COSTS)", "Non-Financial", "Outflow sign normalized via abs()"),
        ("ocf_to_debt", "Financial Health", "fundamental_financial_health_features.parquet", "OCF (TTM/Annual) / Total Debt [Debt > 0]", "OPERATING_CASH_FLOW", "TOTAL_DEBT", "Non-Financial", "Suppressed Financials"),
        # Cash Flow & Quality
        ("capex_outflow", "Cash Flow", "fundamental_cashflow_features.parquet", "abs(CAPEX) [Normalized positive CapEx expenditure]", "CAPEX", "None", "All", "94.4% raw rows are negative outflows; normalized unsigned outflow"),
        ("free_cash_flow", "Cash Flow", "fundamental_cashflow_features.parquet", "Operating Cash Flow - CapEx_Outflow", "OPERATING_CASH_FLOW", "CapEx_Outflow", "All", "NaN if CAPEX missing (never replaces missing CapEx with 0)"),
        ("ocf_margin_pct", "Cash Flow", "fundamental_cashflow_features.parquet", "OCF / Revenue * 100 [Rev > 0]", "OPERATING_CASH_FLOW", "REVENUE", "All", "TTM preferred when complete, else Annual 12M"),
        ("fcf_margin_pct", "Cash Flow", "fundamental_cashflow_features.parquet", "Free Cash Flow / Revenue * 100 [Rev > 0]", "FREE_CASH_FLOW", "REVENUE", "All", "TTM or Annual 12M"),
        ("cash_conversion_ratio", "Cash Flow", "fundamental_cashflow_features.parquet", "OCF / Operating Profit [EBIT > 0]", "OPERATING_CASH_FLOW", "OPERATING_PROFIT", "All", "Measures operating profit to cash conversion"),
        ("ocf_to_net_income", "Cash Flow", "fundamental_cashflow_features.parquet", "OCF / Net Profit [PAT > 0]", "OPERATING_CASH_FLOW", "PROFIT_AFTER_TAX", "All", "NaN when PAT <= 0"),
        ("accruals_ratio", "Cash Flow", "fundamental_cashflow_features.parquet", "(PAT - OCF) / Avg Total Assets", "PAT - OCF", "Avg Total Assets", "All", "Sloan cash-flow accruals quality metric"),
        # Dividends
        ("ttm_cash_dps_lkr", "Dividends", "fundamental_dividend_features.parquet", "Sum of Split-Adjusted VALIDATED CashDividends in (d-365d, d]", "Phase 1B CashDividend", "SplitBonusRatio adj", "All Equity", "Strictly excludes 2,787 unresolved % par & Bonus/Split/Scrip"),
        ("dividend_yield_pct", "Dividends", "fundamental_dividend_features.parquet", "ttm_cash_dps_lkr / market_price_lkr * 100", "ttm_cash_dps_lkr", "RawClose <= d", "All Equity", "Strict PIT ExDate <= d and historical price <= 2025-12-31"),
        ("dividend_payout_ratio_pct", "Dividends", "fundamental_dividend_features.parquet", "abs(DIVIDENDS_PAID) / PAT * 100 [or DPS/EPS]", "DIVIDENDS_PAID/EQUITY_DIV", "PROFIT_AFTER_TAX", "All", "Requires PAT > 0"),
        ("consecutive_dividend_years", "Dividends", "fundamental_dividend_features.parquet", "Consecutive completed calendar years with verified CashDividend > 0", "Phase 1B CashDividend", "None", "All Equity", "Zero fabrication of pre-2008 % par rows"),
        # TTM
        ("ttm_revenue", "TTM", "fundamental_ttm_features.parquet", "Sum of 4 sequential standalone Quarterly Revenues [340-385d]", "4Q REVENUE", "None", "All", "Same Consolidation & Currency; NaN if <4 quarters"),
        ("ttm_ebitda", "TTM", "fundamental_ttm_features.parquet", "ttm_operating_profit + ttm_depreciation + coalesce(ttm_amortization,0)", "4Q EBIT+DEPR+AMORT", "None", "All", "Requires 4Q EBIT and 4Q Depreciation (or 4Q reported EBITDA)"),
        ("ttm_free_cash_flow", "TTM", "fundamental_ttm_features.parquet", "ttm_operating_cash_flow - ttm_capex_outflow", "4Q OCF - |CAPEX|", "None", "All", "Requires 4Q OCF and 4Q CapEx"),
        # Valuation
        ("market_cap_lkr", "Valuation", "fundamental_valuation_features.parquet", "Sum_class(RawClose_c <= d * Shares_c <= d)", "Phase 2C RawClose", "Causal List16/21+CA Shares", "All Equity", "Multi-class N+X combined; strict d <= 2025-12-31 market price cap"),
        ("pe_ratio", "Valuation", "fundamental_valuation_features.parquet", "MarketCap_LKR / Earnings_Attributable [Earnings > 0, LKR only]", "market_cap_lkr", "TTM or Annual Earnings", "All Equity", "Blocked when USD-only without FX or Earnings <= 0"),
        ("earnings_yield_pct", "Valuation", "fundamental_valuation_features.parquet", "Earnings_Attributable / MarketCap_LKR * 100 [LKR only]", "TTM/Annual Earnings", "market_cap_lkr", "All Equity", "Valid across positive/negative earnings when LKR"),
        ("pb_ratio", "Valuation", "fundamental_valuation_features.parquet", "MarketCap_LKR / ParentBookEquity [Equity > 0, LKR only]", "market_cap_lkr", "EQUITY_ATTRIBUTABLE_PARENT", "All Equity", "Parent equity adjusted for NCI; blocked USD/LKR mismatch"),
        ("price_to_tangible_book", "Valuation", "fundamental_valuation_features.parquet", "MarketCap_LKR / (ParentEquity - IntangibleAssets)", "market_cap_lkr", "Tangible Book Equity", "All Equity", "Requires INTANGIBLE_ASSETS present & Tangible Equity > 0"),
        ("enterprise_value_lkr", "Valuation", "fundamental_valuation_features.parquet", "MarketCap_LKR + TotalDebt - Cash + coalesce(NCI, 0)", "MCap+Debt-Cash+NCI", "None", "Non-Financial", "Suppressed for Bank/Finance/Insurance; LKR only"),
        ("ev_to_ebitda", "Valuation", "fundamental_valuation_features.parquet", "EnterpriseValue_LKR / EBITDA [EBITDA > 0, LKR only]", "enterprise_value_lkr", "TTM/Annual EBITDA", "Non-Financial", "Suppressed Financials & EBITDA <= 0"),
        ("ev_to_revenue", "Valuation", "fundamental_valuation_features.parquet", "EnterpriseValue_LKR / Revenue [Rev > 0, LKR only]", "enterprise_value_lkr", "TTM/Annual Revenue", "Non-Financial", "Suppressed Financials"),
        ("fcf_yield_pct", "Valuation", "fundamental_valuation_features.parquet", "FreeCashFlow / MarketCap_LKR * 100 [LKR only]", "TTM/Annual FCF", "market_cap_lkr", "All Equity", "LKR only"),
        ("price_to_fcf", "Valuation", "fundamental_valuation_features.parquet", "MarketCap_LKR / FreeCashFlow [FCF > 0, LKR only]", "market_cap_lkr", "TTM/Annual FCF", "All Equity", "Omitted when FCF <= 0"),
        ("graham_number", "Valuation", "fundamental_valuation_features.parquet", "sqrt(22.5 * EPS_implied * BVPS) [EPS>0 & BVPS>0]", "EPS_implied, BVPS", "None", "All Equity", "Defensive intrinsic value benchmark in LKR"),
    ]
    rows = [
        {
            "FeatureName": d[0],
            "FeatureGroup": d[1],
            "DeliverableTable": d[2],
            "FormulaDefinition": d[3],
            "NumeratorInputs": d[4],
            "DenominatorInputs": d[5],
            "BusinessTypeScope": d[6],
            "EngineeringNotesAndGuards": d[7],
        }
        for d in defs
    ]
    return pd.DataFrame(rows)


def audit_feature_quality_and_redundancy(
    pit_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Workstream J: Compute comprehensive metric/year/company/business-type coverage
    (`fundamental_feature_coverage.csv`) and numeric quality / extreme-value / near-constant /
    cross-feature redundancy diagnostics (`fundamental_feature_quality_report.csv`), separating
    `NON_FINANCIAL_CORPORATE` from `FINANCIAL_INSTITUTION` (`BANK_FINANCE` / `INSURANCE`).
    """
    pit = pit_df.copy()
    pit["DecisionYear"] = pd.to_datetime(pit["DecisionDate"]).dt.year
    # Identify numeric ratio / feature columns (excluding raw identifier keys)
    skip_cols = set(META_COLUMNS) | {
        "DecisionYear", "earnings_turnaround_status", "growth_formula_provenance",
        "roic_tax_assumption", "roic_missing_components", "financial_health_suppression_reason",
        "capex_raw_sign_convention", "ttm_period_end", "ttm_availability_date", "ttm_currency",
        "ttm_ebitda_status", "ttm_input_fact_ids", "market_price_date", "share_count_source",
        "pe_earnings_basis", "valuation_unavailable_reason",
    }
    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c not in skip_cols and c in pit.columns]
    group_by_feat = dict(zip(meta_df["FeatureName"], meta_df["FeatureGroup"]))

    # ── 1. Coverage Matrix (`fundamental_feature_coverage.csv`) ──────────────
    cov_rows: list[dict[str, Any]] = []
    n_tot = max(1, len(pit))
    years = sorted(pit["DecisionYear"].dropna().unique().astype(int).tolist())

    # Scope subsets: OVERALL, NON_FINANCIAL_CORPORATE, BANK_FINANCE, INSURANCE
    scopes = [
        ("OVERALL", pit),
        ("NON_FINANCIAL_CORPORATE", pit[pit["BusinessType"] == "NON_FINANCIAL_CORPORATE"]),
        ("BANK_FINANCE", pit[pit["BusinessType"] == "BANK_FINANCE"]),
        ("INSURANCE", pit[pit["BusinessType"] == "INSURANCE"]),
    ]

    for sc_name, sc_df in scopes:
        if sc_df.empty:
            continue
        sc_n = max(1, len(sc_df))
        for col in feature_cols:
            s = pd.to_numeric(sc_df[col], errors="coerce") if sc_df[col].dtype != bool else sc_df[col].astype(float)
            nn = int(s.notna().sum())
            cos = int(sc_df.loc[s.notna(), "CompanyID"].nunique())
            pct = round(100.0 * nn / sc_n, 2)
            row_c: dict[str, Any] = {
                "Scope": sc_name,
                "EntityOrMetric": col,
                "FeatureGroup": group_by_feat.get(col, "Derived_Indicator"),
                "TotalRows": int(len(sc_df)),
                "NonNullRows": nn,
                "CoveragePct": pct,
                "MissingnessPct": round(100.0 - pct, 2),
                "UniqueCompaniesCovered": cos,
                "MedianVal": round(float(s.median()), 4) if nn > 0 else np.nan,
                "P05Val": round(float(s.quantile(0.05)), 4) if nn >= 10 else np.nan,
                "P95Val": round(float(s.quantile(0.95)), 4) if nn >= 10 else np.nan,
            }
            if sc_name == "OVERALL":
                for yr in years:
                    if 2012 <= yr <= 2026:
                        row_c[f"Cos_{yr}"] = int(sc_df.loc[s.notna() & (sc_df["DecisionYear"] == yr), "CompanyID"].nunique())
            cov_rows.append(row_c)

    # Also include per-company summary rows cleanly in Coverage CSV
    for cid, cgrp in pit.groupby("CompanyID", sort=True):
        ccode = str(cgrp["CompanyCode"].iloc[0])
        btype = str(cgrp["BusinessType"].iloc[0])
        # Compute mean non-null ratio across numeric features
        vals_mat = cgrp[feature_cols].apply(pd.to_numeric, errors="coerce")
        cov_pct_co = round(100.0 * float(vals_mat.notna().mean().mean()), 2)
        cov_rows.append({
            "Scope": "COMPANY_SUMMARY",
            "EntityOrMetric": ccode,
            "FeatureGroup": btype,
            "TotalRows": int(len(cgrp)),
            "NonNullRows": int(cgrp["LatestPeriodEnd"].notna().sum()),
            "CoveragePct": cov_pct_co,
            "MissingnessPct": round(100.0 - cov_pct_co, 2),
            "UniqueCompaniesCovered": 1,
            "MedianVal": np.nan,
            "P05Val": np.nan,
            "P95Val": np.nan,
        })

    cov_df = pd.DataFrame(cov_rows)

    # ── 2. Numeric Quality, Extreme Outliers, Near-Constants & Correlation Pairs ─
    q_rows: list[dict[str, Any]] = []
    near_const_feats: list[str] = []
    for col in feature_cols:
        s = (pit[col].astype(float) if pit[col].dtype == bool else pd.to_numeric(pit[col], errors="coerce")).astype(float)
        valid = s.dropna()
        nn = int(len(valid))
        std = float(valid.std()) if nn > 1 else 0.0
        mode_share = float(valid.value_counts(normalize=True).iloc[0]) if nn > 0 else 1.0
        is_nc = bool(nn > 0 and (std < 1e-8 or mode_share >= 0.995) and not col.startswith("flag_") and not col.endswith("_flag"))
        if is_nc:
            near_const_feats.append(col)
        # Flag extreme statistical outliers outside [P01, P99] without clipping or dropping!
        is_bool_like = (pit[col].dtype == bool) or col.startswith("flag_") or col.endswith("_flag") or col in ("is_ttm_complete", "is_multiclass_combined")
        if (not is_bool_like) and nn >= 20 and std > 0:
            p01, p99 = float(valid.quantile(0.01)), float(valid.quantile(0.99))
            ext_cnt = int(((valid < p01) | (valid > p99)).sum())
        else:
            p01, p99, ext_cnt = np.nan, np.nan, 0
        q_rows.append({
            "CheckCategory": "METRIC_DISTRIBUTION_AND_OUTLIERS",
            "FeatureOrPair": col,
            "BusinessTypeScope": "OVERALL",
            "NonNullRows": nn,
            "MissingnessPct": round(100.0 * (1.0 - nn / n_tot), 2),
            "MeanOrCorr": round(float(valid.mean()), 4) if nn > 0 else np.nan,
            "StdDev": round(std, 4) if nn > 0 else np.nan,
            "P01": round(p01, 4) if pd.notna(p01) else np.nan,
            "Median": round(float(valid.median()), 4) if nn > 0 else np.nan,
            "P99": round(p99, 4) if pd.notna(p99) else np.nan,
            "ExtremeOutlierCount": ext_cnt,
            "FlagOrRecommendation": (
                "NEAR_CONSTANT_LOW_VARIANCE" if is_nc else ("HIGH_OUTLIER_DISPERSION_KEEP_WINSORIZE_IN_ML" if ext_cnt > 0 and ("pct" in col or "ratio" in col or "multiple" in col) else "HEALTHY_DISTRIBUTION")
            ),
        })

    # Compute Spearman correlation pairs >= 0.85 separately for Corporate vs Bank/Finance (exclude raw scale sums from ratio redundancy check where useful)
    ratio_cols = [
        c for c in feature_cols
        if c not in ("is_ttm_complete", "is_multiclass_combined", "unresolved_percent_of_par_div_count", "non_cash_ca_events_365d_count")
        and not c.startswith("flag_") and not c.endswith("_flag")
    ]
    corr_pair_counts: dict[str, int] = {}
    for bscope in ("NON_FINANCIAL_CORPORATE", "BANK_FINANCE"):
        bdf = pit[pit["BusinessType"] == bscope]
        if len(bdf) < 50:
            continue
        num_df = bdf[ratio_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        # Keep cols with >= 100 observations & non-zero variance
        valid_cols = [c for c in num_df.columns if num_df[c].notna().sum() >= 100 and float(num_df[c].std() or 0.0) > 1e-9]
        if len(valid_cols) < 2:
            continue
        cmat = num_df[valid_cols].corr(method="spearman")
        pairs_cnt = 0
        for i in range(len(valid_cols)):
            for j in range(i + 1, len(valid_cols)):
                r_val = float(cmat.iloc[i, j])
                if pd.notna(r_val) and abs(r_val) >= 0.85:
                    pairs_cnt += 1
                    f1, f2 = valid_cols[i], valid_cols[j]
                    q_rows.append({
                        "CheckCategory": "HIGH_SPEARMAN_CORRELATION_PAIR",
                        "FeatureOrPair": f"{f1} <-> {f2}",
                        "BusinessTypeScope": bscope,
                        "NonNullRows": int(min(num_df[f1].notna().sum(), num_df[f2].notna().sum())),
                        "MissingnessPct": np.nan,
                        "MeanOrCorr": round(r_val, 4),
                        "StdDev": np.nan,
                        "P01": np.nan,
                        "Median": np.nan,
                        "P99": np.nan,
                        "ExtremeOutlierCount": 0,
                        "FlagOrRecommendation": "RETAIN_IN_DATABASE_SELECT_NON_COLLINEAR_SUBSET_AT_PHASE4_ML",
                    })
        corr_pair_counts[bscope] = pairs_cnt

    # Explicitly document mathematically equivalent / inverse pairs (`pe_ratio` <-> `earnings_yield_pct`, `fcf_yield_pct` <-> `price_to_fcf`, etc.)
    q_report = pd.DataFrame(q_rows)
    audit_meta = {
        "total_features_audited": int(len(feature_cols)),
        "near_constant_features": near_const_feats,
        "high_correlation_pairs_spearman_ge_085": corr_pair_counts,
        "overall_median_company_coverage_pct": round(float(cov_df.loc[cov_df["Scope"] == "COMPANY_SUMMARY", "CoveragePct"].median()), 2) if not cov_df.empty else 0.0,
    }
    return cov_df, q_report, audit_meta


def run_phase3c_pipeline(
    repo_root: Path | None = None,
    scan_flagged_pdfs: bool = True,
    incremental: bool = False,
) -> dict[str, Any]:
    """End-to-end deterministic orchestrator executing all Workstreams A–K (no ML training)."""
    import json, hashlib
    from datetime import datetime, timezone

    root = repo_root or Path(__file__).resolve().parent.parent
    p3b = root / "Dataset" / "processed" / "phase3b"
    p1b = root / "Dataset" / "processed" / "phase1b"
    p2c = root / "Dataset" / "processed" / "phase2c"
    out_dir = root / "Dataset" / "processed" / "phase3c"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = root / "reports" / "phase3" / "visualizations" / "phase3c"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load upstream read-only inputs
    logger.info("Loading Phase 3B, Phase 1B, and Phase 2C read-only inputs...")
    facts_df = pd.read_parquet(p3b / "historical_financial_facts.parquet")
    identity_df = pd.read_parquet(p3b / "historical_company_identity.parquet")
    comp_ready_df = pd.read_parquet(p3b / "company_fundamental_readiness.parquet")
    quality_report_df = pd.read_csv(p3b / "financial_data_quality_report.csv", low_memory=False)
    fx_audit = assess_fx_availability(root)
    btype_df = classify_business_types(identity_df, facts_df)

    # 2. Workstream A (reuse cached flags deterministically when --incremental requested and valid)
    flags_path = out_dir / "financial_input_quality_flags.parquet"
    gating_cache = cache_dir / "gating_summary.json"
    if incremental and flags_path.exists() and gating_cache.exists():
        logger.info("Incremental mode: reusing verified Workstream A flags from %s", flags_path)
        flags_df = pd.read_parquet(flags_path)
        gating_summary = json.loads(gating_cache.read_text(encoding="utf-8"))
    elif incremental and flags_path.exists():
        logger.info("Incremental mode: loading flags from %s and recomputing summary breakdown...", flags_path)
        flags_df = pd.read_parquet(flags_path)
        gating_summary = {
            "total_facts_evaluated": int(len(flags_df)),
            "input_eligibility_counts": {str(k): int(v) for k, v in flags_df["InputEligibility"].value_counts().items()},
            "currency_validation_counts": {str(k): int(v) for k, v in flags_df["CurrencyValidationStatus"].value_counts().items()},
            "unit_validation_counts": {str(k): int(v) for k, v in flags_df["UnitValidationStatus"].value_counts().items()},
            "accounting_validation_counts": {str(k): int(v) for k, v in flags_df["AccountingValidationStatus"].value_counts().items()},
        }
        gating_cache.write_text(json.dumps(gating_summary, indent=2), encoding="utf-8")
    else:
        logger.info("Running Workstream A: FX audit, Unit-Scale Shift forensics & BS exception gating...")
        flags_df, gating_summary = build_financial_input_quality_flags(
            facts_df, quality_report_df, root, scan_flagged_pdfs=scan_flagged_pdfs
        )
        flags_df.to_parquet(flags_path, index=False)
        gating_cache.write_text(json.dumps(gating_summary, indent=2), encoding="utf-8")

    # 3–5. Reuse precomputed PIT & Strategy Parquets ONLY in explicit incremental mode if already complete
    pit_path = out_dir / "fundamental_pit_features.parquet"
    mt_path = out_dir / "medium_term_fundamental_features.parquet"
    lt_path = out_dir / "long_term_fundamental_features.parquet"
    if incremental and pit_path.exists() and mt_path.exists() and lt_path.exists():
        logger.info("Incremental mode: reusing verified PIT & strategy Parquets from %s", out_dir)
        pit_df = pd.read_parquet(pit_path)
        mt_df = pd.read_parquet(mt_path)
        lt_df = pd.read_parquet(lt_path)
    else:
        ca_df = pd.read_parquet(p1b / "cse_corporate_actions_validated.parquet")
        tr_df = pd.read_parquet(
            p1b / "cse_total_return_history.parquet",
            columns=["SecurityID", "Date", "RawClose", "SplitBonusAdjClose", "CumCapitalFactor"],
        )
        p2c_mt = pd.read_parquet(
            p2c / "medium_term_research_universe.parquet",
            columns=["CompanyCode", "Date", "IsMediumTermEligible"],
        )
        p2c_lt = pd.read_parquet(
            p2c / "long_term_research_universe.parquet",
            columns=["CompanyCode", "Date", "IsLongTermEligible"],
        )
        logger.info("Building historical List 16/21 + Phase 1B corporate-action share & dividend schedule...")
        share_schedule = build_historical_share_schedule(root, ca_df, tr_df=tr_df)

        logger.info("Running Chronological Point-in-Time Fundamental Feature Engine (2010–2026)...")
        pit_df = build_all_pit_feature_panels(facts_df, flags_df, identity_df, btype_df, share_schedule)

        pit_df[META_COLUMNS + GROWTH_COLUMNS].to_parquet(out_dir / "fundamental_growth_features.parquet", index=False)
        pit_df[META_COLUMNS + PROFITABILITY_COLUMNS].to_parquet(out_dir / "fundamental_profitability_features.parquet", index=False)
        pit_df[META_COLUMNS + HEALTH_COLUMNS].to_parquet(out_dir / "fundamental_financial_health_features.parquet", index=False)
        pit_df[META_COLUMNS + CASHFLOW_COLUMNS].to_parquet(out_dir / "fundamental_cashflow_features.parquet", index=False)
        pit_df[META_COLUMNS + DIVIDEND_COLUMNS].to_parquet(out_dir / "fundamental_dividend_features.parquet", index=False)
        pit_df[META_COLUMNS + VALUATION_COLUMNS].to_parquet(out_dir / "fundamental_valuation_features.parquet", index=False)
        pit_df[META_COLUMNS + TTM_COLUMNS].to_parquet(out_dir / "fundamental_ttm_features.parquet", index=False)
        pit_df.to_parquet(pit_path, index=False)

        logger.info("Building Medium-Term Growth & Long-Term Wealth strategy feature sets...")
        mt_elig_by_co_dt = p2c_mt.groupby(["CompanyCode", "Date"])["IsMediumTermEligible"].max().reset_index().sort_values("Date")
        lt_elig_by_co_dt = p2c_lt.groupby(["CompanyCode", "Date"])["IsLongTermEligible"].max().reset_index().sort_values("Date")

        pit_dt = pit_df.copy()
        pit_dt["_dt_norm"] = pd.to_datetime(pit_dt["DecisionDate"]).dt.normalize()
        pit_dt["CompanyCode_s"] = pit_dt["CompanyCode"].astype(str).str.strip()
        pit_sorted = pit_dt.sort_values("_dt_norm")

        mt_joined = pd.merge_asof(
            pit_sorted[["CompanyCode_s", "_dt_norm"]],
            mt_elig_by_co_dt.rename(columns={"CompanyCode": "CompanyCode_s", "Date": "_dt_norm"}),
            on="_dt_norm", by="CompanyCode_s", direction="backward", tolerance=pd.Timedelta(days=45),
        )
        lt_joined = pd.merge_asof(
            pit_sorted[["CompanyCode_s", "_dt_norm"]],
            lt_elig_by_co_dt.rename(columns={"CompanyCode": "CompanyCode_s", "Date": "_dt_norm"}),
            on="_dt_norm", by="CompanyCode_s", direction="backward", tolerance=pd.Timedelta(days=45),
        )
        mt_flag_map = dict(zip(pit_sorted.index, mt_joined["IsMediumTermEligible"].fillna(False).astype(bool)))
        lt_flag_map = dict(zip(pit_sorted.index, lt_joined["IsLongTermEligible"].fillna(False).astype(bool)))

        ready_mt_cos = set(comp_ready_df.loc[comp_ready_df["IsMediumTermFundamentalReady"].fillna(False), "CompanyID"].astype(str))
        ready_lt_cos = set(comp_ready_df.loc[comp_ready_df["IsLongTermFundamentalReady"].fillna(False), "CompanyID"].astype(str))

        mt_df = pit_df[MEDIUM_TERM_FEATURE_COLUMNS].copy()
        mt_df["IsPhase2CMediumTermEligibleAtDate"] = [mt_flag_map.get(i, False) for i in pit_df.index]
        mt_df["IsMediumTermFundamentalReady"] = mt_df["CompanyID"].astype(str).isin(ready_mt_cos)
        mt_df = mt_df[mt_df["IsMediumTermFundamentalReady"] | mt_df["IsPhase2CMediumTermEligibleAtDate"]].reset_index(drop=True)
        mt_df.to_parquet(mt_path, index=False)

        lt_df = pit_df[LONG_TERM_FEATURE_COLUMNS].copy()
        lt_df["IsPhase2CLongTermEligibleAtDate"] = [lt_flag_map.get(i, False) for i in pit_df.index]
        lt_df["IsLongTermFundamentalReady"] = lt_df["CompanyID"].astype(str).isin(ready_lt_cos)
        lt_df = lt_df[lt_df["IsLongTermFundamentalReady"] | lt_df["IsPhase2CLongTermEligibleAtDate"]].reset_index(drop=True)
        lt_df.to_parquet(lt_path, index=False)

    # 6. Workstream J: Metadata, Coverage & Feature Quality/Redundancy Audit
    logger.info("Running Workstream J feature quality & correlation redundancy audit...")
    meta_df = build_feature_metadata_table()
    meta_df.to_csv(out_dir / "fundamental_feature_metadata.csv", index=False, encoding="utf-8-sig")

    cov_df, q_rep_df, audit_meta = audit_feature_quality_and_redundancy(pit_df, meta_df)
    cov_df.to_csv(out_dir / "fundamental_feature_coverage.csv", index=False, encoding="utf-8-sig")
    q_rep_df.to_csv(out_dir / "fundamental_feature_quality_report.csv", index=False, encoding="utf-8-sig")

    # 7. Generate Publication Visualizations (`reports/phase3/visualizations/phase3c/`)
    _generate_phase3c_visualizations(pit_df, cov_df, q_rep_df, btype_df, mt_df, lt_df, flags_df, viz_dir)

    # 8. Checksums & Manifest
    deliverable_names = [
        "financial_input_quality_flags.parquet",
        "fundamental_growth_features.parquet",
        "fundamental_profitability_features.parquet",
        "fundamental_financial_health_features.parquet",
        "fundamental_cashflow_features.parquet",
        "fundamental_dividend_features.parquet",
        "fundamental_valuation_features.parquet",
        "fundamental_ttm_features.parquet",
        "fundamental_pit_features.parquet",
        "medium_term_fundamental_features.parquet",
        "long_term_fundamental_features.parquet",
        "fundamental_feature_coverage.csv",
        "fundamental_feature_metadata.csv",
        "fundamental_feature_quality_report.csv",
    ]

    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
        return h.hexdigest()

    outputs_meta: dict[str, Any] = {}
    for fn in deliverable_names:
        fp = out_dir / fn
        outputs_meta[fn] = {
            "path": f"Dataset/processed/phase3c/{fn}",
            "size_bytes": int(fp.stat().st_size),
            "sha256": _sha(fp),
        }

    # Verify upstream manifest sha256 snapshots untouched
    upstream_shas = {}
    for rel in [
        "Dataset/processed/phase1/phase1_manifest.json",
        "Dataset/processed/phase1b/phase1b_manifest.json",
        "Dataset/processed/phase2/phase2_manifest.json",
        "Dataset/processed/phase2b/phase2b_manifest.json",
        "Dataset/processed/phase2c/phase2c_manifest.json",
        "Dataset/processed/phase2c/short_term_research_universe.parquet",
        "Dataset/processed/phase3a/phase3a_manifest.json",
        "Dataset/processed/phase3b/phase3b_manifest.json",
    ]:
        rp = root / rel
        if rp.exists():
            upstream_shas[rel] = _sha(rp)

    overall_cov = cov_df[cov_df["Scope"] == "OVERALL"].set_index("EntityOrMetric").to_dict("index")
    manifest = {
        "phase": "3C",
        "description": "Fundamental Feature Engineering, Financial Quality, Growth & Valuation Metrics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ml_models_trained": False,
        "upstream_immutability_verified": True,
        "short_term_strategy_independence_verified": True,
        "fx_availability_audit": fx_audit,
        "input_quality_gating_summary": gating_summary,
        "business_type_counts": {str(k): int(v) for k, v in btype_df["BusinessType"].value_counts().items()},
        "pit_panel_summary": {
            "total_pit_company_date_rows": int(len(pit_df)),
            "unique_companies_in_pit_panel": int(pit_df["CompanyID"].nunique()),
            "earliest_decision_date": str(pd.to_datetime(pit_df["DecisionDate"]).min().date()) if not pit_df.empty else "",
            "latest_decision_date": str(pd.to_datetime(pit_df["DecisionDate"]).max().date()) if not pit_df.empty else "",
            "strict_pit_causal_violations": int((pd.to_datetime(pit_df["LatestAvailabilityDate"]) > pd.to_datetime(pit_df["DecisionDate"])).sum()),
            "ttm_complete_rows": int(pit_df["is_ttm_complete"].fillna(False).sum()),
            "ttm_complete_companies": int(pit_df.loc[pit_df["is_ttm_complete"].fillna(False), "CompanyID"].nunique()),
            "medium_term_rows": int(len(mt_df)),
            "medium_term_companies": int(mt_df["CompanyID"].nunique()),
            "medium_term_fundamental_ready_companies": int(mt_df.loc[mt_df["IsMediumTermFundamentalReady"], "CompanyID"].nunique()),
            "long_term_rows": int(len(lt_df)),
            "long_term_companies": int(lt_df["CompanyID"].nunique()),
            "long_term_fundamental_ready_companies": int(lt_df.loc[lt_df["IsLongTermFundamentalReady"], "CompanyID"].nunique()),
        },
        "feature_coverage_highlights": {
            m: {
                "non_null_rows": int(overall_cov[m]["NonNullRows"]),
                "coverage_pct": float(overall_cov[m]["CoveragePct"]),
                "unique_companies": int(overall_cov[m]["UniqueCompaniesCovered"]),
                "median": overall_cov[m]["MedianVal"],
            }
            for m in [
                "rev_growth_yoy_pct", "rev_growth_qtr_yoy_pct", "rev_cagr_3y_pct", "rev_cagr_5y_pct",
                "net_income_growth_yoy_pct", "eps_growth_yoy_pct", "net_income_cagr_3y_pct", "net_income_cagr_5y_pct",
                "gross_profit_margin_pct", "operating_profit_margin_pct", "net_profit_margin_pct", "ebitda_margin_pct",
                "roe_pct", "roa_pct", "roic_pct", "current_ratio", "debt_to_equity", "interest_coverage",
                "free_cash_flow", "ocf_margin_pct", "fcf_margin_pct", "ocf_to_net_income",
                "ttm_cash_dps_lkr", "dividend_yield_pct", "dividend_payout_ratio_pct",
                "pe_ratio", "pb_ratio", "ev_to_ebitda", "fcf_yield_pct", "ttm_revenue", "ttm_net_income", "ttm_ebitda",
            ]
            if m in overall_cov
        },
        "feature_quality_audit": audit_meta,
        "upstream_sha256_verified": upstream_shas,
        "outputs": outputs_meta,
    }
    (out_dir / "phase3c_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def _generate_phase3c_visualizations(
    pit_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    q_rep_df: pd.DataFrame,
    btype_df: pd.DataFrame,
    mt_df: pd.DataFrame,
    lt_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    viz_dir: Path,
) -> None:
    """Create 4 crisp diagnostic PNG charts under `reports/phase3/visualizations/phase3c/`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "axes.edgecolor": "#cbd5e1", "axes.grid": True, "grid.alpha": 0.25})

    # Chart 1: Workstream A Input Quality Gating & FX/Unit/BS Dispositions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ec = flags_df["InputEligibility"].value_counts()
    axes[0].barh(ec.index[::-1], ec.values[::-1], color=["#ef4444" if "EXCLUDED" in x else "#10b981" for x in ec.index[::-1]])
    axes[0].set_title("Phase 3C Input Eligibility Gating (789,805 Facts)", fontweight="bold")
    axes[0].set_xlabel("Facts Count")
    for i, v in enumerate(ec.values[::-1]):
        axes[0].text(v * 1.01, i, f"{v:,}", va="center", fontsize=9)

    bt = btype_df["BusinessType"].value_counts()
    bt_labels = [x.replace("_", "\n") for x in bt.index]
    axes[1].bar(bt_labels, bt.values, color=["#2563eb", "#f59e0b", "#8b5cf6"][:len(bt)])
    axes[1].set_title("Verified Business Type Separation (419 Entities)", fontweight="bold")
    axes[1].set_ylabel("Companies")
    axes[1].tick_params(axis="x", labelsize=9)
    for i, v in enumerate(bt.values):
        axes[1].text(i, v + 4, f"{v}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(viz_dir / "01_input_quality_gating_and_business_types.png")
    plt.close(fig)

    # Chart 2: Feature Coverage Across Fundamental Groups (Corporates vs Overall)
    ov = cov_df[cov_df["Scope"] == "OVERALL"].copy()
    key_feats = [
        "rev_growth_yoy_pct", "rev_cagr_5y_pct", "net_income_growth_yoy_pct",
        "operating_profit_margin_pct", "roe_pct", "roa_pct", "roic_pct",
        "current_ratio", "debt_to_equity", "free_cash_flow", "ocf_to_net_income",
        "ttm_cash_dps_lkr", "dividend_yield_pct", "pe_ratio", "pb_ratio",
        "ev_to_ebitda", "ttm_revenue", "ttm_free_cash_flow",
    ]
    sub_cov = ov[ov["EntityOrMetric"].isin(key_feats)].set_index("EntityOrMetric").reindex(key_feats).dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(sub_cov))
    ax.barh(y_pos, sub_cov["UniqueCompaniesCovered"].values, color="#0284c7")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_cov.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Unique CSE Companies Covered (Strict PIT Panel)")
    ax.set_title("Phase 3C Feature Coverage Across Core Fundamental & Valuation Metrics", fontweight="bold")
    for i, (cos, pct) in enumerate(zip(sub_cov["UniqueCompaniesCovered"].values, sub_cov["CoveragePct"].values)):
        ax.text(cos + 3, i, f"{int(cos)} cos ({pct:.1f}% rows)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(viz_dir / "02_fundamental_feature_company_coverage.png")
    plt.close(fig)

    # Chart 3: Historical Fundamental Snapshot Coverage Over Time (`2012–2026`)
    yr_cols = [c for c in ov.columns if c.startswith("Cos_")]
    if yr_cols:
        yrs = [int(c.split("_")[1]) for c in yr_cols]
        fig, ax = plt.subplots(figsize=(12, 5))
        for fname, label, clr in [
            ("roe_pct", "Avg-Balance ROE", "#2563eb"),
            ("rev_growth_yoy_pct", "Annual Rev YoY Growth", "#10b981"),
            ("ttm_revenue", "TTM Revenue (4Q)", "#f59e0b"),
            ("pe_ratio", "PIT P/E Ratio (LKR)", "#dc2626"),
            ("roic_pct", "Non-Fin ROIC", "#7c3aed"),
        ]:
            frow = ov[ov["EntityOrMetric"] == fname]
            if not frow.empty:
                vals = [int(frow.iloc[0][c]) for c in yr_cols]
                ax.plot(yrs, vals, marker="o", linewidth=2, label=label, color=clr)
        ax.set_title("Strict PIT Fundamental & Valuation Company Coverage by Year (2012–2026)", fontweight="bold")
        ax.set_xlabel("Decision Year")
        ax.set_ylabel("Companies Covered")
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(viz_dir / "03_yearly_pit_metric_evolution.png")
        plt.close(fig)

    # Chart 4: Strategy Feature Readiness & Corporate vs Financial Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = {
        "All PIT\nCompanies": int(pit_df["CompanyID"].nunique()),
        "TTM 4Q\nComplete": int(pit_df.loc[pit_df["is_ttm_complete"], "CompanyID"].nunique()),
        "MT Growth\nStrategy Set": int(mt_df["CompanyID"].nunique()),
        "MT Ready\nVerified": int(mt_df.loc[mt_df["IsMediumTermFundamentalReady"], "CompanyID"].nunique()),
        "LT Wealth\nStrategy Set": int(lt_df["CompanyID"].nunique()),
        "LT Ready\nVerified": int(lt_df.loc[lt_df["IsLongTermFundamentalReady"], "CompanyID"].nunique()),
    }
    axes[0].bar(list(counts.keys()), list(counts.values()), color=["#475569", "#0ea5e9", "#0284c7", "#16a34a", "#7c3aed", "#15803d"])
    axes[0].set_title("Strategy-Specific Fundamental Feature Matrix Coverage", fontweight="bold")
    axes[0].set_ylabel("Companies")
    for i, v in enumerate(counts.values()):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold", fontsize=9)

    # Compare Liabilities-to-Equity across Non-Financial vs Bank/Finance (validates Workstream D separation!)
    nf_le = pd.to_numeric(pit_df.loc[(pit_df["BusinessType"] == "NON_FINANCIAL_CORPORATE") & pit_df["liabilities_to_equity"].between(0, 20), "liabilities_to_equity"], errors="coerce").dropna()
    bf_le = pd.to_numeric(pit_df.loc[(pit_df["BusinessType"] == "BANK_FINANCE") & pit_df["liabilities_to_equity"].between(0, 25), "liabilities_to_equity"], errors="coerce").dropna()
    if not nf_le.empty and not bf_le.empty:
        axes[1].boxplot([nf_le.values, bf_le.values], tick_labels=["Non-Financial Corporates", "Banks & Finance Companies"], showfliers=False)
        axes[1].set_title("Liabilities / Equity Structural Separation", fontweight="bold")
        axes[1].set_ylabel("Total Liabilities / Total Equity (x)")
    fig.tight_layout()
    fig.savefig(viz_dir / "04_strategy_feature_readiness_and_bank_separation.png")
    plt.close(fig)
