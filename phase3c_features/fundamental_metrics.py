"""
Phase 3C Workstreams B, C, D, E & G – Fundamental Ratio Engine (Single-Date Snapshot Core)
===========================================================================================
Pure, stateless financial feature calculation operating strictly on PIT-resolved, quality-gated
company facts as of a given `DecisionDate`. Shared cleanly as the single source of truth between:
  - Per-company chronological PIT replay across 2010–2026
  - `build_fundamental_features_as_of(company_id, decision_date)` public query API
Implements all safeguards requested in Workstreams B–G:
  - Safe growth handling + explicit `earnings_recovery_flag` / `earnings_turnaround_status`
  - True Beginning+Ending Average Balance Sheet denominators for ROE, ROA, and ROIC (never 1-snapshot)
  - Explicitly documented ROIC Invested Capital (`Equity + Borrowings - Cash`) & Tax assumption
  - Verified Bank/Finance/Insurance separation (suppressing industrial Debt/Liquidity/EV ratios)
  - Explicitly normalized Cash Flow outflow convention (`CapEx_Outflow = abs(CAPEX)`, `FCF = OCF - CapEx_Outflow`)
  - Strict PIT LKR market price, causal share count, and compatible-currency Valuation (`P/E`, `P/B`, `EV/EBITDA`, `P/FCF`)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .dividends_valuation_ttm import (
    HistoricalShareSchedule,
    compute_dividends_as_of,
    compute_ttm_from_qtr_snaps,
    get_market_snapshot_as_of,
)

# Canonical Column Groups across Deliverables
META_COLUMNS = [
    "CompanyID", "CompanyCode", "CompanyName", "PrimarySecurityID",
    "BusinessType", "IsFinancialInstitution", "DecisionDate",
    "LatestPeriodEnd", "LatestPeriodType", "LatestAvailabilityDate",
    "ConsolidationBasis", "CurrencyBasis",
    "FundamentalDataQualityStatus", "InputFactIDsCount", "InputFactIDsSample",
]

GROWTH_COLUMNS = [
    "rev_growth_qoq_pct", "rev_growth_qtr_yoy_pct", "rev_growth_yoy_pct",
    "rev_cagr_3y_pct", "rev_cagr_5y_pct",
    "net_income_growth_qtr_yoy_pct", "net_income_growth_yoy_pct",
    "eps_growth_yoy_pct", "net_income_cagr_3y_pct", "net_income_cagr_5y_pct",
    "gross_profit_growth_yoy_pct", "operating_profit_growth_yoy_pct",
    "ebitda_growth_yoy_pct", "pbt_growth_yoy_pct",
    "earnings_recovery_flag", "earnings_turnaround_status", "growth_formula_provenance",
]

PROFITABILITY_COLUMNS = [
    "gross_profit_margin_pct", "operating_profit_margin_pct",
    "net_profit_margin_pct", "ebitda_margin_pct",
    "roe_pct", "roa_pct", "roic_pct",
    "effective_tax_rate_used", "roic_tax_assumption", "roic_missing_components",
    "flag_positive_roe", "flag_consistent_roe", "flag_improving_roe",
    "flag_positive_roic", "flag_consistent_profitability", "profitable_years_count",
]

HEALTH_COLUMNS = [
    "current_ratio", "quick_ratio", "cash_ratio",
    "total_debt", "net_debt", "debt_to_equity", "liabilities_to_equity",
    "total_debt_to_assets", "equity_to_assets", "net_debt_to_ebitda",
    "interest_coverage", "ocf_to_debt", "ebitda_to_interest",
    "debt_growth_yoy_pct", "equity_growth_yoy_pct", "asset_growth_yoy_pct",
    "current_liability_growth_yoy_pct", "cash_balance_growth_yoy_pct",
    "working_capital", "working_capital_growth_yoy_pct",
    "financial_health_suppression_reason",
]

CASHFLOW_COLUMNS = [
    "operating_cash_flow", "capex_outflow", "capex_raw_sign_convention",
    "free_cash_flow", "ocf_margin_pct", "fcf_margin_pct",
    "cash_conversion_ratio", "ocf_to_net_income", "capex_to_revenue_pct",
    "fcf_growth_yoy_pct", "fcf_turnaround_flag",
    "accruals_ratio", "persistent_cash_conversion_flag", "repeated_negative_fcf_flag",
]

DIVIDEND_COLUMNS = [
    "ttm_cash_dps_lkr", "annual_cash_dps_lkr", "dividend_yield_pct",
    "price_to_dividend", "dividend_payout_ratio_pct", "dividend_coverage_ratio",
    "dividend_growth_yoy_pct", "dividend_cagr_3y_pct",
    "consecutive_dividend_years", "dividend_consistency_5y_pct",
    "unresolved_percent_of_par_div_count", "non_cash_ca_events_365d_count",
]

TTM_COLUMNS = [
    "is_ttm_complete", "ttm_period_end", "ttm_availability_date", "ttm_currency",
    "ttm_revenue", "ttm_gross_profit", "ttm_operating_profit", "ttm_pbt",
    "ttm_net_income", "ttm_profit_attributable", "ttm_eps",
    "ttm_depreciation", "ttm_amortization", "ttm_ebitda", "ttm_ebitda_status",
    "ttm_operating_cash_flow", "ttm_capex_outflow", "ttm_free_cash_flow", "ttm_input_fact_ids",
]

VALUATION_COLUMNS = [
    "market_price_lkr", "market_price_date", "shares_outstanding_pit",
    "market_cap_lkr", "share_count_source", "is_multiclass_combined",
    "enterprise_value_lkr", "pe_ratio", "pe_earnings_basis", "earnings_yield_pct",
    "pb_ratio", "bvps_lkr", "price_to_tangible_book",
    "ev_to_ebitda", "ev_to_revenue", "price_to_fcf", "fcf_yield_pct",
    "graham_number", "graham_margin_of_safety_pct", "valuation_unavailable_reason",
]

ALL_FEATURE_COLUMNS = (
    META_COLUMNS + GROWTH_COLUMNS + PROFITABILITY_COLUMNS + HEALTH_COLUMNS
    + CASHFLOW_COLUMNS + DIVIDEND_COLUMNS + TTM_COLUMNS + VALUATION_COLUMNS
)

MEDIUM_TERM_FEATURE_COLUMNS = META_COLUMNS + [
    # Growth (Medium-Term focus)
    "rev_growth_qoq_pct", "rev_growth_qtr_yoy_pct", "rev_growth_yoy_pct",
    "net_income_growth_qtr_yoy_pct", "net_income_growth_yoy_pct", "eps_growth_yoy_pct",
    "operating_profit_growth_yoy_pct", "ebitda_growth_yoy_pct",
    "earnings_recovery_flag", "earnings_turnaround_status",
    # Profitability & ROE
    "gross_profit_margin_pct", "operating_profit_margin_pct", "net_profit_margin_pct",
    "ebitda_margin_pct", "roe_pct", "roa_pct", "flag_improving_roe",
    # Financial Health & Coverage
    "debt_to_equity", "interest_coverage", "ocf_to_debt", "current_ratio", "equity_to_assets",
    # Cash Flow Quality & TTM
    "ocf_margin_pct", "fcf_margin_pct", "cash_conversion_ratio", "ocf_to_net_income",
    "is_ttm_complete", "ttm_revenue", "ttm_operating_profit", "ttm_net_income", "ttm_ebitda",
    "ttm_operating_cash_flow", "ttm_free_cash_flow",
    # Historical Valuation
    "market_price_lkr", "shares_outstanding_pit", "market_cap_lkr",
    "pe_ratio", "pe_earnings_basis", "earnings_yield_pct", "pb_ratio",
    "ev_to_ebitda", "ev_to_revenue", "price_to_fcf", "fcf_yield_pct", "dividend_yield_pct",
    "valuation_unavailable_reason",
]

LONG_TERM_FEATURE_COLUMNS = META_COLUMNS + [
    # Multi-Year Compounding Growth
    "rev_growth_yoy_pct", "rev_cagr_3y_pct", "rev_cagr_5y_pct",
    "net_income_growth_yoy_pct", "net_income_cagr_3y_pct", "net_income_cagr_5y_pct",
    "eps_growth_yoy_pct", "earnings_recovery_flag",
    # Moat, Capital Returns & Earnings Consistency
    "gross_profit_margin_pct", "operating_profit_margin_pct", "net_profit_margin_pct",
    "roe_pct", "roa_pct", "roic_pct", "roic_tax_assumption",
    "flag_positive_roe", "flag_consistent_roe", "flag_improving_roe",
    "flag_positive_roic", "flag_consistent_profitability", "profitable_years_count",
    # Cash Flow Durability
    "operating_cash_flow", "capex_outflow", "free_cash_flow",
    "ocf_margin_pct", "fcf_margin_pct", "cash_conversion_ratio", "ocf_to_net_income",
    "capex_to_revenue_pct", "fcf_growth_yoy_pct", "accruals_ratio",
    "persistent_cash_conversion_flag", "repeated_negative_fcf_flag",
    # Balance Sheet Fortress & Sustainability
    "total_debt", "net_debt", "debt_to_equity", "liabilities_to_equity",
    "total_debt_to_assets", "equity_to_assets", "net_debt_to_ebitda",
    "interest_coverage", "ocf_to_debt", "debt_growth_yoy_pct", "equity_growth_yoy_pct", "asset_growth_yoy_pct",
    # Dividend Sustainability
    "ttm_cash_dps_lkr", "annual_cash_dps_lkr", "dividend_yield_pct",
    "dividend_payout_ratio_pct", "dividend_coverage_ratio",
    "dividend_growth_yoy_pct", "dividend_cagr_3y_pct",
    "consecutive_dividend_years", "dividend_consistency_5y_pct",
    # Long-Term Valuation & Margin of Safety
    "market_price_lkr", "shares_outstanding_pit", "market_cap_lkr", "enterprise_value_lkr",
    "pe_ratio", "earnings_yield_pct", "pb_ratio", "bvps_lkr", "price_to_tangible_book",
    "ev_to_ebitda", "fcf_yield_pct", "graham_number", "graham_margin_of_safety_pct",
    "valuation_unavailable_reason",
]


def _safe_growth_pct(cur: float, prev: float) -> float:
    """Compute exact percentage growth strictly when prior base > 0 (never misleading negative denominator)."""
    if pd.isna(cur) or pd.isna(prev) or prev <= 0.0:
        return np.nan
    return ((float(cur) - float(prev)) / float(prev)) * 100.0


def _safe_cagr_pct(cur: float, prev: float, years: int) -> float:
    """Compute annualized CAGR strictly when both endpoints > 0."""
    if pd.isna(cur) or pd.isna(prev) or prev <= 0.0 or cur <= 0.0 or years <= 0:
        return np.nan
    return ((float(cur) / float(prev)) ** (1.0 / float(years)) - 1.0) * 100.0


def _classify_turnaround(cur: float, prev: float) -> tuple[int, str]:
    """Return `(earnings_recovery_flag, turnaround_status)` guarding negative-to-positive transitions cleanly."""
    if pd.isna(cur) or pd.isna(prev):
        return 0, "INSUFFICIENT_DATA"
    if prev <= 0.0 and cur > 0.0:
        return 1, "LOSS_TO_PROFIT"
    if prev > 0.0 and cur <= 0.0:
        return 0, "PROFIT_TO_LOSS"
    if prev <= 0.0 and cur <= 0.0:
        return 0, ("LOSS_NARROWED" if cur > prev else "LOSS_WIDENED")
    return 0, "CONTINUING_PROFIT"


def _effective_tax_rate(mvals: dict[str, float]) -> tuple[float, str]:
    """Derive clean tax rate [0, 0.50] verified against PBT - PAT, or explicit statutory 30% fallback."""
    pbt = mvals.get("PROFIT_BEFORE_TAX", np.nan)
    tax = mvals.get("INCOME_TAX_EXPENSE", np.nan)
    pat = mvals.get("PROFIT_AFTER_TAX", np.nan)
    if pd.notna(pbt) and pbt > 0 and pd.notna(tax):
        # Negative tax expense is standard outflow sign; positive can also occur if unsigned tax charge
        tax_charge = -float(tax) if tax <= 0 else (float(tax) if (pd.notna(pat) and pat < pbt) else np.nan)
        if pd.notna(tax_charge) and 0.0 <= tax_charge <= 0.55 * float(pbt):
            etr = max(0.0, min(0.50, tax_charge / float(pbt)))
            return etr, "EFFECTIVE_TAX_RATE_ACTUAL"
    return 0.30, "STATUTORY_DEFAULT_30PCT"


def _compute_annual_ebitda(mvals: dict[str, float]) -> float:
    ebit = mvals.get("OPERATING_PROFIT", np.nan)
    depr = mvals.get("DEPRECIATION", np.nan)
    amort = mvals.get("AMORTIZATION", np.nan)
    if pd.notna(ebit) and pd.notna(depr):
        return float(ebit) + abs(float(depr)) + (abs(float(amort)) if pd.notna(amort) else 0.0)
    return float(mvals["EBITDA"]) if pd.notna(mvals.get("EBITDA", np.nan)) else np.nan


def _compute_invested_capital(mvals: dict[str, float]) -> tuple[float, str]:
    """
    Consistent Financing Invested Capital:
      IC = Total Equity + Total Borrowings (ST + LT) - Cash & Cash Equivalents
    Requires Equity AND at least one explicit borrowing component; never treats Total Assets as IC.
    """
    eq = mvals.get("TOTAL_EQUITY", np.nan)
    st_b = mvals.get("SHORT_TERM_BORROWINGS", np.nan)
    lt_b = mvals.get("LONG_TERM_BORROWINGS", np.nan)
    cash = mvals.get("CASH_AND_EQUIVALENTS", np.nan)
    missing = []
    if pd.isna(eq):
        missing.append("TOTAL_EQUITY")
    if pd.isna(st_b) and pd.isna(lt_b):
        missing.append("BORROWINGS")
    if pd.isna(cash):
        missing.append("CASH_AND_EQUIVALENTS")
    if missing:
        return np.nan, "MISSING:" + ",".join(missing)
    debt = (float(st_b) if pd.notna(st_b) else 0.0) + (float(lt_b) if pd.notna(lt_b) else 0.0)
    ic = float(eq) + debt - max(0.0, float(cash))
    return (ic if ic > 0 else np.nan), ("IC_<=_0" if ic <= 0 else "")


def _parent_equity(mvals: dict[str, float]) -> float:
    if pd.notna(mvals.get("EQUITY_ATTRIBUTABLE_PARENT", np.nan)):
        return float(mvals["EQUITY_ATTRIBUTABLE_PARENT"])
    te = mvals.get("TOTAL_EQUITY", np.nan)
    nci = mvals.get("NON_CONTROLLING_INTEREST", np.nan)
    if pd.notna(te) and pd.notna(nci):
        return float(te) - float(nci)
    return float(te) if pd.notna(te) else np.nan


def _find_prior_snapshot(
    rows_desc: list[tuple[pd.Timestamp, dict[str, float], str]],
    target_pe: pd.Timestamp,
    min_days: int,
    max_days: int,
) -> tuple[pd.Timestamp | pd.NaT, dict[str, float]]:
    """Find matching historical period snapshot inside `[min_days, max_days]` prior to `target_pe`."""
    best = None
    best_err = 10**9
    target_mid = (min_days + max_days) / 2.0
    for pe, mvals, _ in rows_desc:
        diff = (target_pe - pe).days
        if min_days <= diff <= max_days:
            err = abs(diff - target_mid)
            if err < best_err:
                best_err = err
                best = (pe, mvals)
    return best if best is not None else (pd.NaT, {})


def refresh_market_valuation_on_snapshot(
    prev_snap: dict[str, Any],
    meta: dict[str, Any],
    decision_date: pd.Timestamp,
    share_schedule: HistoricalShareSchedule | None,
    ann_df: pd.DataFrame,
    bs_end_m: dict[str, float],
    flow_vals: dict[str, Any],
) -> dict[str, Any]:
    """Fast O(1) market-price, share-count, dividend & valuation refresh when accounting statements haven't changed."""
    row = dict(prev_snap)
    row["DecisionDate"] = pd.Timestamp(decision_date)
    for vc in VALUATION_COLUMNS:
        row[vc] = np.nan
    row["market_price_date"] = pd.NaT
    row["share_count_source"] = ""
    row["is_multiclass_combined"] = False
    row["pe_earnings_basis"] = ""
    row["valuation_unavailable_reason"] = ""

    if share_schedule is None:
        return row
    ccode = str(meta["CompanyCode"])
    primary_sid = str(meta["PrimarySecurityID"])
    is_fin = bool(meta["IsFinancialInstitution"])
    cur_basis = str(row.get("CurrencyBasis") or "")

    mkt = get_market_snapshot_as_of(share_schedule, ccode, primary_sid, decision_date)
    row["market_price_lkr"] = mkt["market_price_lkr"]
    row["market_price_date"] = mkt["market_price_date"]
    row["shares_outstanding_pit"] = mkt["shares_outstanding_pit"]
    row["market_cap_lkr"] = mkt["market_cap_lkr"]
    row["share_count_source"] = mkt["share_count_source"]
    row["is_multiclass_combined"] = mkt["is_multiclass_combined"]

    div_info = compute_dividends_as_of(
        share_schedule, ccode, primary_sid, decision_date,
        mkt["market_price_lkr"], mkt["split_bonus_ratio_at_d"], ann_df,
    )
    row.update(div_info)

    mcap = mkt["market_cap_lkr"]
    px = mkt["market_price_lkr"]
    sh = mkt["shares_outstanding_pit"]

    if cur_basis != "LKR":
        row["valuation_unavailable_reason"] = f"CURRENCY_MISMATCH_USD_NO_CBSL_FX ({cur_basis} vs LKR MarketCap)"
        return row
    if pd.isna(mcap) or mcap <= 0:
        row["valuation_unavailable_reason"] = (
            "NO_HISTORICAL_PRICE_POST_2025" if decision_date > pd.Timestamp("2025-12-31") else "MISSING_PIT_PRICE_OR_SHARES"
        )
        return row

    pe_earn = flow_vals["pe_earn"]
    row["pe_earnings_basis"] = flow_vals["ebasis"]
    if pd.notna(pe_earn):
        row["earnings_yield_pct"] = (pe_earn / mcap) * 100.0
        if pe_earn > 0:
            row["pe_ratio"] = mcap / pe_earn
        else:
            row["valuation_unavailable_reason"] = "NEGATIVE_EARNINGS_PE_OMITTED"

    # Book Equity attributable to parent
    attr_eq = _parent_equity(bs_end_m)
    if pd.notna(attr_eq) and attr_eq > 0 and pd.notna(sh) and sh > 0:
        bvps = attr_eq / sh
        row["bvps_lkr"] = bvps
        row["pb_ratio"] = mcap / attr_eq
        if pd.notna(bs_end_m.get("INTANGIBLE_ASSETS", np.nan)):
            tang_eq = attr_eq - max(0.0, float(bs_end_m["INTANGIBLE_ASSETS"]))
            if tang_eq > 0:
                row["price_to_tangible_book"] = mcap / tang_eq

    if not is_fin:
        tot_debt = row["total_debt"]
        cash_bs = bs_end_m.get("CASH_AND_EQUIVALENTS", np.nan)
        nci_bs = bs_end_m.get("NON_CONTROLLING_INTEREST", 0.0)
        if pd.notna(tot_debt) and pd.notna(cash_bs):
            ev = mcap + float(tot_debt) - float(cash_bs) + (float(nci_bs) if pd.notna(nci_bs) else 0.0)
            row["enterprise_value_lkr"] = ev
            if pd.notna(flow_vals["flow_ebitda"]) and flow_vals["flow_ebitda"] > 0:
                row["ev_to_ebitda"] = ev / flow_vals["flow_ebitda"]
            if pd.notna(flow_vals["flow_rev"]) and flow_vals["flow_rev"] > 0:
                row["ev_to_revenue"] = ev / flow_vals["flow_rev"]

    flow_fcf = flow_vals["flow_fcf"]
    if pd.notna(flow_fcf):
        row["fcf_yield_pct"] = (flow_fcf / mcap) * 100.0
        if flow_fcf > 0:
            row["price_to_fcf"] = mcap / flow_fcf

    if pd.notna(pe_earn) and pe_earn > 0 and pd.notna(sh) and sh > 0 and pd.notna(row["bvps_lkr"]) and row["bvps_lkr"] > 0 and pd.notna(px) and px > 0:
        eps_lkr = pe_earn / sh
        g_num = float(np.sqrt(22.5 * eps_lkr * row["bvps_lkr"]))
        row["graham_number"] = g_num
        if g_num > 0:
            row["graham_margin_of_safety_pct"] = ((g_num - px) / g_num) * 100.0
    return row


def compute_fundamental_snapshot_from_active_facts(
    active_facts_df: pd.DataFrame,
    company_meta: dict[str, Any],
    decision_date: pd.Timestamp,
    share_schedule: HistoricalShareSchedule | None = None,
    return_context: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Convert an active-facts DataFrame once into typed tuples and delegate to pure-Python `compute_snapshot_from_tuples`."""
    if active_facts_df.empty:
        return compute_snapshot_from_tuples([], company_meta, decision_date, share_schedule=share_schedule, return_context=return_context)

    ps_col = active_facts_df["PeriodStart_dt"] if "PeriodStart_dt" in active_facts_df.columns else pd.to_datetime(active_facts_df["PeriodStart"], errors="coerce")
    pe_col = active_facts_df["PeriodEnd_dt"] if "PeriodEnd_dt" in active_facts_df.columns else pd.to_datetime(active_facts_df["PeriodEnd"], errors="coerce")
    av_col = active_facts_df["AvailabilityDate_dt"] if "AvailabilityDate_dt" in active_facts_df.columns else pd.to_datetime(active_facts_df["AvailabilityDate"], errors="coerce")
    ie_col = active_facts_df["InputEligibility"] if "InputEligibility" in active_facts_df.columns else pd.Series(["ELIGIBLE"] * len(active_facts_df))

    tuples = list(
        zip(
            active_facts_df["StatementType"].astype(str).tolist(),
            active_facts_df["MetricCode"].astype(str).tolist(),
            ps_col.tolist(),
            pe_col.tolist(),
            active_facts_df["PeriodType"].astype(str).tolist(),
            active_facts_df["ConsolidationStatus"].astype(str).tolist(),
            active_facts_df["Currency"].astype(str).tolist(),
            pd.to_numeric(active_facts_df["Value"], errors="coerce").tolist(),
            active_facts_df["FactID"].astype(str).tolist(),
            av_col.tolist(),
            active_facts_df["IsDerived"].fillna(False).astype(bool).tolist(),
            pd.to_numeric(active_facts_df["RestatementVersion"], errors="coerce").fillna(1).astype(int).tolist(),
            ie_col.astype(str).tolist(),
            pd.to_numeric(active_facts_df["ExtractionConfidence"], errors="coerce").fillna(0.0).tolist(),
            active_facts_df["SourceDocumentID"].astype(str).tolist(),
        )
    )
    return compute_snapshot_from_tuples(
        tuples, company_meta, decision_date, share_schedule=share_schedule, return_context=return_context
    )


def compute_snapshot_from_tuples(
    fact_tuples: Any,
    company_meta: dict[str, Any],
    decision_date: pd.Timestamp,
    share_schedule: HistoricalShareSchedule | None = None,
    return_context: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """
    Zero-DataFrame, single-source-of-truth fundamental snapshot calculator operating on 15-element tuples:
    `(StatementType, MetricCode, PeriodStart_dt, PeriodEnd_dt, PeriodType, ConsolidationStatus, Currency,
      Value, FactID, AvailabilityDate_dt, IsDerived, RestatementVersion, InputEligibility, ExtractionConfidence, SourceDocumentID)`
    """
    cid = str(company_meta["CompanyID"])
    ccode = str(company_meta["CompanyCode"])
    cname = str(company_meta.get("CompanyName", ccode))
    primary_sid = str(company_meta.get("PrimarySecurityID") or f"{ccode}-N-0000")
    btype = str(company_meta.get("BusinessType", "NON_FINANCIAL_CORPORATE"))
    is_fin = bool(company_meta.get("IsFinancialInstitution", btype in ("BANK_FINANCE", "INSURANCE")))

    row: dict[str, Any] = {col: np.nan for col in ALL_FEATURE_COLUMNS}
    row.update({
        "CompanyID": cid,
        "CompanyCode": ccode,
        "CompanyName": cname,
        "PrimarySecurityID": primary_sid,
        "BusinessType": btype,
        "IsFinancialInstitution": is_fin,
        "DecisionDate": pd.Timestamp(decision_date),
        "LatestPeriodEnd": pd.NaT,
        "LatestPeriodType": "",
        "LatestAvailabilityDate": pd.NaT,
        "ConsolidationBasis": "",
        "CurrencyBasis": "",
        "FundamentalDataQualityStatus": "NO_ELIGIBLE_FACTS",
        "InputFactIDsCount": 0,
        "InputFactIDsSample": "",
        "earnings_recovery_flag": 0,
        "earnings_turnaround_status": "INSUFFICIENT_DATA",
        "growth_formula_provenance": "YoY=(X_t-X_{t-1})/X_{t-1} [X_{t-1}>0 only]; CAGR=(X_t/X_{t-n})^(1/n)-1 [both>0]",
        "roic_tax_assumption": "",
        "roic_missing_components": "",
        "flag_positive_roe": 0,
        "flag_consistent_roe": 0,
        "flag_improving_roe": 0,
        "flag_positive_roic": 0,
        "flag_consistent_profitability": 0,
        "profitable_years_count": 0,
        "financial_health_suppression_reason": "FINANCIAL_INSTITUTION_STRUCTURE" if is_fin else "",
        "capex_raw_sign_convention": "OUTFLOW_NORMALIZED_ABS(CAPEX)",
        "fcf_turnaround_flag": 0,
        "persistent_cash_conversion_flag": 0,
        "repeated_negative_fcf_flag": 0,
        "consecutive_dividend_years": 0,
        "dividend_consistency_5y_pct": 0.0,
        "unresolved_percent_of_par_div_count": 0,
        "non_cash_ca_events_365d_count": 0,
        "is_ttm_complete": False,
        "ttm_period_end": pd.NaT,
        "ttm_availability_date": pd.NaT,
        "ttm_currency": "",
        "ttm_ebitda_status": "UNAVAILABLE_NO_TTM",
        "ttm_input_fact_ids": "",
        "market_price_date": pd.NaT,
        "share_count_source": "",
        "is_multiclass_combined": False,
        "pe_earnings_basis": "",
        "valuation_unavailable_reason": "",
    })

    empty_ctx = {"ann_df": ({}, ""), "bs_end_m": {}, "flow_vals": {"pe_earn": np.nan, "ebasis": "", "flow_ebitda": np.nan, "flow_rev": np.nan, "flow_fcf": np.nan}}
    tuples_list = list(fact_tuples)
    if not tuples_list:
        return (row, empty_ctx) if return_context else row

    # 1. Select single consistent ConsolidationBasis & Currency in O(N)
    cons_cnt: dict[str, int] = {}
    for t in tuples_list:
        c = t[5]
        cons_cnt[c] = cons_cnt.get(c, 0) + 1
    basis_cons = "Consolidated_Group" if cons_cnt.get("Consolidated_Group", 0) >= 5 else max(cons_cnt.items(), key=lambda x: x[1])[0]

    cur_cnt: dict[str, int] = {}
    for t in tuples_list:
        if t[5] == basis_cons:
            cur_cnt[t[6]] = cur_cnt.get(t[6], 0) + 1
    if not cur_cnt:
        return (row, empty_ctx) if return_context else row
    cur_basis = "LKR" if "LKR" in cur_cnt else max(cur_cnt.items(), key=lambda x: x[1])[0]

    # 2. Keep latest version per (StatementType, MetricCode, PeriodStart, PeriodEnd, PeriodType) within basis
    best_by_slice: dict[tuple, tuple[tuple, tuple]] = {}
    for t in tuples_list:
        if t[5] != basis_cons or t[6] != cur_basis or pd.isna(t[7]):
            continue
        ps_ts = pd.Timestamp(t[2]) if pd.notna(t[2]) else pd.NaT
        pe_ts = pd.Timestamp(t[3]) if pd.notna(t[3]) else pd.NaT
        av_ts = pd.Timestamp(t[9]) if pd.notna(t[9]) else pd.NaT
        skey = (t[0], t[1], str(ps_ts), str(pe_ts), t[4])
        rank = (int(t[11]), av_ts.value if pd.notna(av_ts) else 0, 0 if t[10] else 1, float(t[13]), str(t[14]))
        prev = best_by_slice.get(skey)
        if prev is None or rank > prev[0]:
            best_by_slice[skey] = (rank, (t[0], t[1], ps_ts, pe_ts, t[4], float(t[7]), t[8], av_ts, t[10], int(t[11]), t[12], float(t[13])))

    resolved = [v[1] for v in best_by_slice.values()]
    if not resolved:
        return (row, empty_ctx) if return_context else row

    latest_avail = pd.NaT
    lat_pe = pd.NaT
    lat_av = pd.NaT
    lat_ptype = ""
    has_bs_warn = False

    ann_buckets: dict[pd.Timestamp, dict[str, tuple[tuple, float, str]]] = {}
    bs_buckets: dict[pd.Timestamp, dict[str, tuple[tuple, float, str]]] = {}
    qtr_buckets: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, dict[str, tuple[tuple, float, str]]]] = {}

    for stmt, mcode, ps_ts, pe_ts, ptype, val, fid, av_ts, is_der, rv, ielig, conf in resolved:
        if pd.notna(av_ts) and (pd.isna(latest_avail) or av_ts > latest_avail):
            latest_avail = av_ts
        if pd.notna(pe_ts) and (
            pd.isna(lat_pe) or pe_ts > lat_pe or (pe_ts == lat_pe and pd.notna(av_ts) and (pd.isna(lat_av) or av_ts > lat_av))
        ):
            lat_pe, lat_av, lat_ptype = pe_ts, av_ts, ptype
        if ielig == "ELIGIBLE_WITH_BS_WARNING":
            has_bs_warn = True

        m_rank = (rv, av_ts.value if pd.notna(av_ts) else 0, 0 if is_der else 1, conf)
        if ptype == "Annual" and pd.notna(pe_ts) and stmt in ("Income Statement", "Cash Flow"):
            if pd.isna(ps_ts) or (pe_ts - ps_ts).days >= 300:
                bk = ann_buckets.setdefault(pe_ts, {})
                if mcode not in bk or m_rank > bk[mcode][0]:
                    bk[mcode] = (m_rank, val, fid)
        if stmt == "Balance Sheet" and pd.notna(pe_ts):
            bk = bs_buckets.setdefault(pe_ts, {})
            if mcode not in bk or m_rank > bk[mcode][0]:
                bk[mcode] = (m_rank, val, fid)
            # Attach BS balance sheet line items to Annual flow periods only where true Annual Flow exists
            if ptype == "Annual" and pe_ts in ann_buckets:
                bk_a = ann_buckets[pe_ts]
                if mcode not in bk_a or m_rank > bk_a[mcode][0]:
                    bk_a[mcode] = (m_rank, val, fid)
        if ptype == "Quarterly" and pd.notna(ps_ts) and pd.notna(pe_ts):
            dur_d = (pe_ts - ps_ts).days
            if 60 <= dur_d <= 115:
                qb = qtr_buckets.get(pe_ts)
                if qb is None:
                    qb = (ps_ts, pe_ts, av_ts, {})
                    qtr_buckets[pe_ts] = qb
                elif pd.notna(av_ts) and (pd.isna(qb[2]) or av_ts > qb[2]):
                    qtr_buckets[pe_ts] = (qb[0], qb[1], av_ts, qb[3])
                    qb = qtr_buckets[pe_ts]
                bk = qb[3]
                if mcode not in bk or m_rank > bk[mcode][0]:
                    bk[mcode] = (m_rank, val, fid)

    # Second pass: attach any matching BS snapshots at exact `ann_buckets` PeriodEnds regardless of row order
    for pe_ann, bk_a in ann_buckets.items():
        bs_same = bs_buckets.get(pe_ann)
        if bs_same:
            for mc, tup in bs_same.items():
                if mc not in bk_a or tup[0] > bk_a[mc][0]:
                    bk_a[mc] = tup

    used_ids: list[str] = []
    row.update({
        "LatestPeriodEnd": lat_pe,
        "LatestPeriodType": lat_ptype,
        "LatestAvailabilityDate": latest_avail,
        "ConsolidationBasis": basis_cons,
        "CurrencyBasis": cur_basis,
        "FundamentalDataQualityStatus": (
            "ELIGIBLE_WITH_BS_WARNING" if has_bs_warn
            else ("NATIVE_USD_ONLY_NO_FX" if cur_basis != "LKR" else "VERIFIED_CLEAN_LKR")
        ),
    })

    ann_snaps: list[tuple[pd.Timestamp, dict[str, float], str]] = [
        (pe, {m: v for m, (_, v, _) in bk.items()}, ";".join([fid for _, (_, _, fid) in list(bk.items())[:12]]))
        for pe, bk in sorted(ann_buckets.items(), key=lambda x: x[0], reverse=True)
    ]
    bs_snaps: list[tuple[pd.Timestamp, dict[str, float], str]] = [
        (pe, {m: v for m, (_, v, _) in bk.items()}, ";".join([fid for _, (_, _, fid) in list(bk.items())[:10]]))
        for pe, bk in sorted(bs_buckets.items(), key=lambda x: x[0], reverse=True)
    ]
    qtr_snaps: list[tuple[pd.Timestamp, dict[str, float], str]] = []
    qtr_tuples_asc: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, dict[str, float], dict[str, str]]] = []
    for pe_q, (ps_q, _, av_q, bk_q) in sorted(qtr_buckets.items(), key=lambda x: x[0]):
        mv_q = {m: v for m, (_, v, _) in bk_q.items()}
        fm_q = {m: fid for m, (_, _, fid) in bk_q.items()}
        qtr_tuples_asc.append((ps_q, pe_q, av_q, mv_q, fm_q))
        qtr_snaps.append((pe_q, mv_q, ";".join(list(fm_q.values())[:8])))
    qtr_snaps.reverse()

    ttm = compute_ttm_from_qtr_snaps(qtr_tuples_asc, cur_basis)
    row.update(ttm)
    if ttm["ttm_input_fact_ids"]:
        used_ids.extend(ttm["ttm_input_fact_ids"].split(";"))
    ann_pair = (ann_snaps[0][1], cur_basis) if ann_snaps else ({}, cur_basis)

    # ── WORKSTREAM B: Growth Features ─────────────────────────────────────────
    if qtr_snaps:
        pe_q0, m_q0, f_q0 = qtr_snaps[0]
        _, m_q1 = _find_prior_snapshot(qtr_snaps, pe_q0, 65, 115)
        _, m_q4 = _find_prior_snapshot(qtr_snaps, pe_q0, 330, 400)
        row["rev_growth_qoq_pct"] = _safe_growth_pct(m_q0.get("REVENUE", np.nan), m_q1.get("REVENUE", np.nan))
        row["rev_growth_qtr_yoy_pct"] = _safe_growth_pct(m_q0.get("REVENUE", np.nan), m_q4.get("REVENUE", np.nan))
        row["net_income_growth_qtr_yoy_pct"] = _safe_growth_pct(m_q0.get("PROFIT_AFTER_TAX", np.nan), m_q4.get("PROFIT_AFTER_TAX", np.nan))
        rec_q, status_q = _classify_turnaround(m_q0.get("PROFIT_AFTER_TAX", np.nan), m_q4.get("PROFIT_AFTER_TAX", np.nan))
        if f_q0:
            used_ids.extend(f_q0.split(";"))
    else:
        rec_q, status_q = 0, "INSUFFICIENT_DATA"

    if ann_snaps:
        pe_a0, m_a0, f_a0 = ann_snaps[0]
        used_ids.extend(f_a0.split(";"))
        _, m_a1 = _find_prior_snapshot(ann_snaps, pe_a0, 300, 430)
        _, m_a3 = _find_prior_snapshot(ann_snaps, pe_a0, 1000, 1220)
        _, m_a5 = _find_prior_snapshot(ann_snaps, pe_a0, 1700, 1980)

        row["rev_growth_yoy_pct"] = _safe_growth_pct(m_a0.get("REVENUE", np.nan), m_a1.get("REVENUE", np.nan))
        row["rev_cagr_3y_pct"] = _safe_cagr_pct(m_a0.get("REVENUE", np.nan), m_a3.get("REVENUE", np.nan), 3)
        row["rev_cagr_5y_pct"] = _safe_cagr_pct(m_a0.get("REVENUE", np.nan), m_a5.get("REVENUE", np.nan), 5)

        pat_a0, pat_a1 = m_a0.get("PROFIT_AFTER_TAX", np.nan), m_a1.get("PROFIT_AFTER_TAX", np.nan)
        row["net_income_growth_yoy_pct"] = _safe_growth_pct(pat_a0, pat_a1)
        row["eps_growth_yoy_pct"] = _safe_growth_pct(m_a0.get("EPS_BASIC", np.nan), m_a1.get("EPS_BASIC", np.nan))
        row["net_income_cagr_3y_pct"] = _safe_cagr_pct(pat_a0, m_a3.get("PROFIT_AFTER_TAX", np.nan), 3)
        row["net_income_cagr_5y_pct"] = _safe_cagr_pct(pat_a0, m_a5.get("PROFIT_AFTER_TAX", np.nan), 5)

        if not is_fin:
            row["gross_profit_growth_yoy_pct"] = _safe_growth_pct(m_a0.get("GROSS_PROFIT", np.nan), m_a1.get("GROSS_PROFIT", np.nan))
            eb_a0, eb_a1 = _compute_annual_ebitda(m_a0), _compute_annual_ebitda(m_a1)
            row["ebitda_growth_yoy_pct"] = _safe_growth_pct(eb_a0, eb_a1)

        row["operating_profit_growth_yoy_pct"] = _safe_growth_pct(m_a0.get("OPERATING_PROFIT", np.nan), m_a1.get("OPERATING_PROFIT", np.nan))
        row["pbt_growth_yoy_pct"] = _safe_growth_pct(m_a0.get("PROFIT_BEFORE_TAX", np.nan), m_a1.get("PROFIT_BEFORE_TAX", np.nan))

        rec_a, status_a = _classify_turnaround(pat_a0, pat_a1)
        if status_a != "INSUFFICIENT_DATA":
            row["earnings_recovery_flag"] = rec_a
            row["earnings_turnaround_status"] = status_a
        else:
            row["earnings_recovery_flag"] = rec_q
            row["earnings_turnaround_status"] = status_q
    else:
        pe_a0, m_a0, m_a1 = pd.NaT, {}, {}
        row["earnings_recovery_flag"] = rec_q
        row["earnings_turnaround_status"] = status_q

    # ── WORKSTREAM C: Profitability & Average-Balance Capital Returns ──────────
    # Prefer TTM flows for timely margins/returns when TTM has complete earnings & end >= Annual end; else cohesive Annual 12M
    flow_end = pd.NaT
    ttm_has_earn = bool(ttm["is_ttm_complete"]) and (pd.notna(ttm["ttm_profit_attributable"]) or pd.notna(ttm["ttm_net_income"]))
    use_ttm = ttm_has_earn and (not ann_snaps or ttm["ttm_period_end"] >= pe_a0)
    if use_ttm:
        flow_rev = ttm["ttm_revenue"]
        flow_gp = ttm["ttm_gross_profit"]
        flow_ebit = ttm["ttm_operating_profit"]
        flow_pat = ttm["ttm_net_income"]
        flow_pat_attr = ttm["ttm_profit_attributable"] if pd.notna(ttm["ttm_profit_attributable"]) else flow_pat
        flow_ebitda = ttm["ttm_ebitda"]
        flow_ocf = ttm["ttm_operating_cash_flow"]
        flow_capex = ttm["ttm_capex_outflow"]
        flow_fcf = ttm["ttm_free_cash_flow"]
        flow_end = ttm["ttm_period_end"]
        # Match tax rate & BS at TTM period end (else latest Annual)
        flow_mvals = dict(m_a0)
        flow_bs_match = next((m for pe, m, _ in bs_snaps if abs((pe - flow_end).days) <= 20), None)
        if flow_bs_match is None and bs_snaps:
            flow_end, flow_bs_match, _ = bs_snaps[0]
        if flow_bs_match:
            flow_mvals.update(flow_bs_match)
        flow_mvals["PROFIT_BEFORE_TAX"] = ttm["ttm_pbt"] if pd.notna(ttm["ttm_pbt"]) else flow_mvals.get("PROFIT_BEFORE_TAX", np.nan)
        flow_mvals["PROFIT_AFTER_TAX"] = flow_pat
    elif ann_snaps:
        flow_rev = m_a0.get("REVENUE", np.nan)
        flow_gp = m_a0.get("GROSS_PROFIT", np.nan)
        flow_ebit = m_a0.get("OPERATING_PROFIT", np.nan)
        flow_pat = m_a0.get("PROFIT_AFTER_TAX", np.nan)
        flow_pat_attr = m_a0.get("PROFIT_ATTRIBUTABLE_EQUITY_HOLDERS", flow_pat)
        flow_ebitda = _compute_annual_ebitda(m_a0)
        flow_ocf = m_a0.get("OPERATING_CASH_FLOW", np.nan)
        cx_raw = m_a0.get("CAPEX", np.nan)
        flow_capex = abs(float(cx_raw)) if pd.notna(cx_raw) else np.nan
        flow_fcf = (float(flow_ocf) - flow_capex) if (pd.notna(flow_ocf) and pd.notna(flow_capex)) else np.nan
        flow_end = pe_a0
        flow_mvals = dict(m_a0)
    else:
        flow_rev = flow_gp = flow_ebit = flow_pat = flow_pat_attr = flow_ebitda = flow_ocf = flow_capex = flow_fcf = np.nan
        flow_mvals = {}

    if pd.notna(flow_rev) and flow_rev > 0:
        if not is_fin and pd.notna(flow_gp):
            row["gross_profit_margin_pct"] = (flow_gp / flow_rev) * 100.0
        if pd.notna(flow_ebit):
            row["operating_profit_margin_pct"] = (flow_ebit / flow_rev) * 100.0
        if pd.notna(flow_pat):
            row["net_profit_margin_pct"] = (flow_pat / flow_rev) * 100.0
        if not is_fin and pd.notna(flow_ebitda):
            row["ebitda_margin_pct"] = (flow_ebitda / flow_rev) * 100.0

    # Beginning & Ending Average Balance Sheet snapshots (STRICT 300-430d span requirement!)
    bs_end_pe, bs_end_m = (bs_snaps[0][0], bs_snaps[0][1]) if bs_snaps else (pd.NaT, {})
    if pd.notna(flow_end) and bs_snaps:
        # Match BS closest to flow_end within 25 days, else use bs_end_pe cleanly ONLY if within 25d for avg return!
        matched = next(((pe, m) for pe, m, _ in bs_snaps if abs((pe - flow_end).days) <= 25), None)
        bs_ret_pe, bs_ret_end = matched if matched is not None else (pd.NaT, {})
    else:
        bs_ret_pe, bs_ret_end = pd.NaT, {}

    if pd.notna(bs_ret_pe):
        _, bs_ret_beg = _find_prior_snapshot(bs_snaps, bs_ret_pe, 300, 430)
    else:
        _, bs_ret_beg = pd.NaT, {}

    # If TTM interim quarter lacked a matched Beginning-Ending BS pair, fall back to latest verified Annual 12M dual-BS pair
    ret_pat, ret_pat_attr, ret_ebit, ret_mvals = flow_pat, flow_pat_attr, flow_ebit, flow_mvals
    if (not bs_ret_end or not bs_ret_beg) and ann_snaps and (pd.isna(flow_end) or (flow_end - pe_a0).days <= 400):
        _, a_beg_cand = _find_prior_snapshot(bs_snaps, pe_a0, 300, 430)
        a_end_cand = bs_buckets.get(pe_a0, {})
        # Extract float dict from bs_buckets if present, else m_a0
        a_end_m = {m: v for m, (_, v, _) in a_end_cand.items()} if a_end_cand else m_a0
        if a_end_m and a_beg_cand:
            bs_ret_pe, bs_ret_end, bs_ret_beg = pe_a0, a_end_m, a_beg_cand
            ret_pat = m_a0.get("PROFIT_AFTER_TAX", np.nan)
            ret_pat_attr = m_a0.get("PROFIT_ATTRIBUTABLE_EQUITY_HOLDERS", ret_pat)
            ret_ebit = m_a0.get("OPERATING_PROFIT", np.nan)
            ret_mvals = m_a0

    avg_assets = np.nan
    avg_parent_eq = np.nan
    if bs_ret_end and bs_ret_beg:
        a_beg = bs_ret_beg.get("TOTAL_ASSETS", np.nan)
        a_end = bs_ret_end.get("TOTAL_ASSETS", np.nan)
        if pd.notna(a_beg) and pd.notna(a_end) and a_beg > 0 and a_end > 0:
            avg_assets = (a_beg + a_end) / 2.0
            if pd.notna(ret_pat):
                row["roa_pct"] = (ret_pat / avg_assets) * 100.0

        eq_beg = _parent_equity(bs_ret_beg)
        eq_end = _parent_equity(bs_ret_end)
        if pd.notna(eq_beg) and pd.notna(eq_end) and eq_beg > 0 and eq_end > 0:
            avg_parent_eq = (eq_beg + eq_end) / 2.0
            if pd.notna(ret_pat_attr):
                row["roe_pct"] = (ret_pat_attr / avg_parent_eq) * 100.0

        if is_fin:
            row["roic_missing_components"] = "SUPPRESSED_FINANCIAL_INSTITUTION"
        else:
            ic_beg, miss_beg = _compute_invested_capital(bs_ret_beg)
            ic_end, miss_end = _compute_invested_capital(bs_ret_end)
            if pd.notna(ic_beg) and pd.notna(ic_end) and (ic_beg + ic_end) > 0 and pd.notna(ret_ebit):
                etr, tax_note = _effective_tax_rate(ret_mvals)
                nopat = float(ret_ebit) * (1.0 - etr)
                avg_ic = (ic_beg + ic_end) / 2.0
                row["roic_pct"] = (nopat / avg_ic) * 100.0
                row["effective_tax_rate_used"] = etr
                row["roic_tax_assumption"] = tax_note
                row["roic_missing_components"] = ""
            else:
                row["roic_missing_components"] = miss_end or miss_beg or "MISSING_EBIT_OR_BEG_END"
    elif is_fin:
        row["roic_missing_components"] = "SUPPRESSED_FINANCIAL_INSTITUTION"
    else:
        row["roic_missing_components"] = "MISSING_TWO_ANNUAL_BS_SNAPSHOTS"

    # Multi-Year Profitability & ROE Consistency Flags (NEVER from 1 quarter!)
    ann_roes: list[float] = []
    prof_yrs = 0
    for i, (pe_i, m_i, _) in enumerate(ann_snaps[:6]):
        pat_i = m_i.get("PROFIT_AFTER_TAX", np.nan)
        if pd.notna(pat_i) and pat_i > 0:
            prof_yrs += 1
        _, m_prev_i = _find_prior_snapshot(ann_snaps, pe_i, 300, 430)
        if m_prev_i:
            e1, e0 = _parent_equity(m_i), _parent_equity(m_prev_i)
            pa_i = m_i.get("PROFIT_ATTRIBUTABLE_EQUITY_HOLDERS", pat_i)
            if pd.notna(pa_i) and pd.notna(e1) and pd.notna(e0) and e1 > 0 and e0 > 0:
                ann_roes.append((pa_i / ((e1 + e0) / 2.0)) * 100.0)

    row["profitable_years_count"] = prof_yrs
    row["flag_consistent_profitability"] = int(prof_yrs >= 3)
    row["flag_positive_roe"] = int(pd.notna(row["roe_pct"]) and row["roe_pct"] > 0 and pd.notna(flow_pat_attr) and flow_pat_attr > 0)
    row["flag_consistent_roe"] = int(len(ann_roes) >= 3 and sum(1 for r in ann_roes[:4] if r > 0) >= 3)
    if len(ann_roes) >= 2 and pd.notna(row["roe_pct"]):
        trail_mean = float(np.mean(ann_roes[1:4]))
        row["flag_improving_roe"] = int(row["roe_pct"] > ann_roes[1] and row["roe_pct"] > trail_mean)
    row["flag_positive_roic"] = int(pd.notna(row["roic_pct"]) and row["roic_pct"] > 0)

    # ── WORKSTREAM D: Financial Health & Balance Sheet Trends ─────────────────
    if bs_snaps:
        _, bs_prev, _ = (pd.NaT, {}, "") if len(bs_snaps) < 2 else (
            lambda r: (r[0], r[1], "")
        )(_find_prior_snapshot(bs_snaps, bs_end_pe, 300, 430))

        assets_c = bs_end_m.get("TOTAL_ASSETS", np.nan)
        liab_c = bs_end_m.get("TOTAL_LIABILITIES", np.nan)
        eq_c = bs_end_m.get("TOTAL_EQUITY", np.nan)
        cash_c = bs_end_m.get("CASH_AND_EQUIVALENTS", np.nan)

        if pd.notna(eq_c) and pd.notna(assets_c) and assets_c > 0:
            row["equity_to_assets"] = eq_c / assets_c
        if pd.notna(liab_c) and pd.notna(eq_c) and eq_c > 0:
            row["liabilities_to_equity"] = liab_c / eq_c

        if bs_prev:
            row["equity_growth_yoy_pct"] = _safe_growth_pct(eq_c, bs_prev.get("TOTAL_EQUITY", np.nan))
            row["asset_growth_yoy_pct"] = _safe_growth_pct(assets_c, bs_prev.get("TOTAL_ASSETS", np.nan))
            row["cash_balance_growth_yoy_pct"] = _safe_growth_pct(cash_c, bs_prev.get("CASH_AND_EQUIVALENTS", np.nan))

        if not is_fin:
            ca_c = bs_end_m.get("CURRENT_ASSETS", np.nan)
            cl_c = bs_end_m.get("CURRENT_LIABILITIES", np.nan)
            inv_c = bs_end_m.get("INVENTORIES", np.nan)
            sti_c = bs_end_m.get("SHORT_TERM_INVESTMENTS", np.nan)
            tr_c = bs_end_m.get("TRADE_RECEIVABLES", np.nan)

            if pd.notna(ca_c) and pd.notna(cl_c) and cl_c > 0:
                row["current_ratio"] = ca_c / cl_c
                if pd.notna(inv_c):
                    row["quick_ratio"] = (ca_c - inv_c) / cl_c
                elif pd.notna(cash_c) or pd.notna(tr_c):
                    row["quick_ratio"] = ((cash_c if pd.notna(cash_c) else 0.0) + (sti_c if pd.notna(sti_c) else 0.0) + (tr_c if pd.notna(tr_c) else 0.0)) / cl_c
            if pd.notna(ca_c) and pd.notna(cl_c):
                row["working_capital"] = ca_c - cl_c
                if bs_prev and pd.notna(bs_prev.get("CURRENT_ASSETS", np.nan)) and pd.notna(bs_prev.get("CURRENT_LIABILITIES", np.nan)):
                    wc_prev = bs_prev["CURRENT_ASSETS"] - bs_prev["CURRENT_LIABILITIES"]
                    row["working_capital_growth_yoy_pct"] = _safe_growth_pct(row["working_capital"], wc_prev)
            if pd.notna(cash_c) and pd.notna(cl_c) and cl_c > 0:
                row["cash_ratio"] = (cash_c + (sti_c if pd.notna(sti_c) else 0.0)) / cl_c
            if bs_prev:
                row["current_liability_growth_yoy_pct"] = _safe_growth_pct(cl_c, bs_prev.get("CURRENT_LIABILITIES", np.nan))

            st_b = bs_end_m.get("SHORT_TERM_BORROWINGS", np.nan)
            lt_b = bs_end_m.get("LONG_TERM_BORROWINGS", np.nan)
            if pd.notna(st_b) or pd.notna(lt_b):
                tot_debt = (st_b if pd.notna(st_b) else 0.0) + (lt_b if pd.notna(lt_b) else 0.0)
                row["total_debt"] = tot_debt
                if pd.notna(cash_c):
                    row["net_debt"] = tot_debt - cash_c
                if pd.notna(eq_c) and eq_c > 0:
                    row["debt_to_equity"] = tot_debt / eq_c
                if pd.notna(assets_c) and assets_c > 0:
                    row["total_debt_to_assets"] = tot_debt / assets_c
                if pd.notna(row["net_debt"]) and pd.notna(flow_ebitda) and flow_ebitda > 0:
                    row["net_debt_to_ebitda"] = row["net_debt"] / flow_ebitda
                if pd.notna(flow_ocf) and tot_debt > 0:
                    row["ocf_to_debt"] = flow_ocf / tot_debt
                if bs_prev and (pd.notna(bs_prev.get("SHORT_TERM_BORROWINGS", np.nan)) or pd.notna(bs_prev.get("LONG_TERM_BORROWINGS", np.nan))):
                    prev_debt = (bs_prev.get("SHORT_TERM_BORROWINGS", 0.0) if pd.notna(bs_prev.get("SHORT_TERM_BORROWINGS", np.nan)) else 0.0) + (
                        bs_prev.get("LONG_TERM_BORROWINGS", 0.0) if pd.notna(bs_prev.get("LONG_TERM_BORROWINGS", np.nan)) else 0.0
                    )
                    row["debt_growth_yoy_pct"] = _safe_growth_pct(tot_debt, prev_debt)

            fc = flow_mvals.get("FINANCE_COSTS", np.nan)
            if pd.notna(fc) and abs(fc) > 0:
                if pd.notna(flow_ebit):
                    row["interest_coverage"] = flow_ebit / abs(fc)
                if pd.notna(flow_ebitda):
                    row["ebitda_to_interest"] = flow_ebitda / abs(fc)
    else:
        bs_end_m = {}

    # ── WORKSTREAM E: Cash Flow & Earnings Quality ────────────────────────────
    row["operating_cash_flow"] = flow_ocf
    row["capex_outflow"] = flow_capex
    row["free_cash_flow"] = flow_fcf
    if pd.notna(flow_rev) and flow_rev > 0:
        if pd.notna(flow_ocf):
            row["ocf_margin_pct"] = (flow_ocf / flow_rev) * 100.0
        if pd.notna(flow_fcf):
            row["fcf_margin_pct"] = (flow_fcf / flow_rev) * 100.0
        if pd.notna(flow_capex):
            row["capex_to_revenue_pct"] = (flow_capex / flow_rev) * 100.0
    if pd.notna(flow_ocf) and pd.notna(flow_ebit) and flow_ebit > 0:
        row["cash_conversion_ratio"] = flow_ocf / flow_ebit
    if pd.notna(flow_ocf) and pd.notna(flow_pat) and flow_pat > 0:
        row["ocf_to_net_income"] = flow_ocf / flow_pat
    if pd.notna(flow_pat) and pd.notna(flow_ocf) and pd.notna(avg_assets) and avg_assets > 0:
        row["accruals_ratio"] = (flow_pat - flow_ocf) / avg_assets

    if ann_snaps:
        prev_ocf = m_a1.get("OPERATING_CASH_FLOW", np.nan)
        prev_cx = m_a1.get("CAPEX", np.nan)
        prev_fcf = (float(prev_ocf) - abs(float(prev_cx))) if (pd.notna(prev_ocf) and pd.notna(prev_cx)) else np.nan
        cur_fcf_ann = (float(m_a0["OPERATING_CASH_FLOW"]) - abs(float(m_a0["CAPEX"]))) if (
            pd.notna(m_a0.get("OPERATING_CASH_FLOW", np.nan)) and pd.notna(m_a0.get("CAPEX", np.nan))
        ) else np.nan
        row["fcf_growth_yoy_pct"] = _safe_growth_pct(cur_fcf_ann, prev_fcf)
        if pd.notna(prev_fcf) and pd.notna(cur_fcf_ann) and prev_fcf <= 0 and cur_fcf_ann > 0:
            row["fcf_turnaround_flag"] = 1

        # Multi-year Cash Flow persistence & repeated negative FCF (require >= 3 annual periods)
        ocf_ni_list = []
        fcf_hist = []
        for _, m_i, _ in ann_snaps[:5]:
            o_i = m_i.get("OPERATING_CASH_FLOW", np.nan)
            p_i = m_i.get("PROFIT_AFTER_TAX", np.nan)
            c_i = m_i.get("CAPEX", np.nan)
            if pd.notna(o_i) and pd.notna(p_i) and p_i > 0:
                ocf_ni_list.append(o_i / p_i)
            if pd.notna(o_i) and pd.notna(c_i):
                fcf_hist.append(float(o_i) - abs(float(c_i)))
        row["persistent_cash_conversion_flag"] = int(len(ocf_ni_list) >= 3 and sum(1 for r in ocf_ni_list[:4] if r >= 0.8) >= 3)
        row["repeated_negative_fcf_flag"] = int(len(fcf_hist) >= 3 and all(x < 0 for x in fcf_hist[:3]))

    # Include Balance Sheet & latest resolved provenance FactIDs even on Half-Year/Nine-Month-only histories
    if bs_snaps and bs_snaps[0][2]:
        used_ids.extend(bs_snaps[0][2].split(";"))
    if not used_ids and resolved:
        used_ids.extend([x[6] for x in resolved[:8]])

    # Provenance sample & Valuation via shared helper (eliminates duplication)
    unique_fids = sorted(set(x for x in used_ids if x))
    row["InputFactIDsCount"] = int(len(unique_fids))
    row["InputFactIDsSample"] = ";".join(unique_fids[:25])

    ctx = {
        "ann_df": ann_pair,
        "bs_end_m": bs_end_m,
        "flow_vals": {
            "pe_earn": (flow_pat_attr if pd.notna(flow_pat_attr) else flow_pat),
            "ebasis": ("TTM" if use_ttm else ("ANNUAL_12M" if ann_snaps else "")),
            "flow_ebitda": flow_ebitda,
            "flow_rev": flow_rev,
            "flow_fcf": flow_fcf,
        },
    }
    row = refresh_market_valuation_on_snapshot(
        row, company_meta, decision_date, share_schedule, ann_pair, bs_end_m, ctx["flow_vals"]
    )
    return (row, ctx) if return_context else row
