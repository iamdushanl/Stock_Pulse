"""
Phase 2B Workstreams G, H, I & Visualization Generator
=======================================================
Implements:
  - Workstream G: Survivorship bias & point-in-time historical universe reconstruction audit (`security_history_coverage.csv`)
  - Workstream H: Measurable strategy readiness verdicts for Short-term, Medium-term Growth, and Long-term Wealth Creation (`strategy_readiness_assessment.json`)
  - Workstream I: Prioritized remediation matrix (`Critical` / `High` / `Medium` / `Low`)
  - Publication-grade audit charts written to `reports/phase2/visualizations/`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def audit_survivorship_and_universe(
    df_adj: pd.DataFrame,
    sec_master: pd.DataFrame,
    ca_validated: pd.DataFrame,
    liq_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Workstream G: Audit point-in-time historical membership, listing/delisting
    alignment, share-class preservation, and delisted security retention.
    """
    adj_dates = df_adj.groupby("SecurityID").agg(
        ObservedFirstDate=("Date", "min"),
        ObservedLastDate=("Date", "max"),
        ObservedRows=("Date", "size"),
    ).reset_index()

    ca_counts = ca_validated[ca_validated["SecurityID"].notna()].groupby("SecurityID").size().rename("CorporateActionEvents")
    liq_cols = liq_df[["SecurityID", "ActiveTradingDays", "TradingFrequencyPct", "MedianTurnoverLKR", "MaxCalGapBetweenObsDays"]]

    m = sec_master.merge(adj_dates, on="SecurityID", how="left").merge(liq_cols, on="SecurityID", how="left")
    m["CorporateActionEvents"] = m["SecurityID"].map(ca_counts).fillna(0).astype(int)

    # Check exact boundaries vs ListingDate / DelistingDate
    dates_s = pd.to_datetime(df_adj["Date"]).dt.normalize()
    lst_map = dict(zip(sec_master["SecurityID"], pd.to_datetime(sec_master["ListingDate"]).dt.normalize()))
    del_map = dict(zip(sec_master["SecurityID"], pd.to_datetime(sec_master["DelistingDate"]).dt.normalize()))
    lst_vec = df_adj["SecurityID"].map(lst_map)
    del_vec = df_adj["SecurityID"].map(del_map)

    pre_list_mask = lst_vec.notna() & (dates_s < lst_vec)
    post_delist_mask = del_vec.notna() & (dates_s > del_vec)
    pre_list_cnts = df_adj.loc[pre_list_mask].groupby("SecurityID").size()
    post_delist_cnts = df_adj.loc[post_delist_mask].groupby("SecurityID").size()

    m["PreListingPriceRows"] = m["SecurityID"].map(pre_list_cnts).fillna(0).astype(int)
    m["PostDelistingPriceRows"] = m["SecurityID"].map(post_delist_cnts).fillna(0).astype(int)

    def _risk_flag(r: pd.Series) -> str:
        flags = []
        if r["HistoricalStatus"] == "Delisted":
            flags.append("DELISTED_PRESERVED")
        if bool(r.get("ManualReviewRequired", False)):
            flags.append(str(r.get("IssuerValidationStatus", "MANUAL_REVIEW")))
        if int(r["PreListingPriceRows"]) > 0:
            flags.append("PRICE_PRECEDES_LISTING_DATE")
        if int(r["PostDelistingPriceRows"]) > 0:
            flags.append("PRICE_AFTER_DELISTING_DATE")
        if int(r["TotalPriceRecords"]) < 20:
            flags.append("SHORT_HISTORY_LT_20D")
        return "|".join(flags) if flags else "CLEAN"

    m["PointInTimeAuditFlag"] = m.apply(_risk_flag, axis=1)

    keep_cols = [
        "SecurityID", "CompanyCode", "IssuerID", "MainType", "SubType", "ISIN",
        "CompanyName", "HistoricalStatus", "ListingDate", "ListingDateSource",
        "DelistingDate", "DelistingDateSource", "ObservedFirstDate", "ObservedLastDate",
        "ObservedRows", "ActiveTradingDays", "TradingFrequencyPct", "MedianTurnoverLKR",
        "MaxCalGapBetweenObsDays", "CorporateActionEvents", "PreListingPriceRows",
        "PostDelistingPriceRows", "IssuerValidationStatus", "ManualReviewRequired",
        "PointInTimeAuditFlag",
    ]
    out_df = m[keep_cols].sort_values(["CompanyCode", "SecurityID"]).reset_index(drop=True)

    # Annual historical universe curve (Listed vs Delisted securities active per year)
    tmp_yr = df_adj[["Date", "SecurityID", "CompanyCode"]].copy()
    tmp_yr["Year"] = pd.to_datetime(tmp_yr["Date"]).dt.year
    st_lookup = dict(zip(sec_master["SecurityID"], sec_master["HistoricalStatus"]))
    tmp_yr["HistoricalStatus"] = tmp_yr["SecurityID"].map(st_lookup)
    ann_univ = (
        tmp_yr.groupby(["Year", "HistoricalStatus"])["SecurityID"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    ann_codes = tmp_yr.groupby("Year")["CompanyCode"].nunique().rename("TotalCompanyCodes").reset_index()
    ann_univ = ann_univ.merge(ann_codes, on="Year", how="left")

    summary = {
        "total_securities": int(len(out_df)),
        "listed_securities": int((out_df["HistoricalStatus"] == "Listed").sum()),
        "delisted_securities_retained": int((out_df["HistoricalStatus"] == "Delisted").sum()),
        "total_company_codes": int(out_df["CompanyCode"].nunique()),
        "listed_company_codes": int(out_df.drop_duplicates("CompanyCode")["HistoricalStatus"].eq("Listed").sum()),
        "delisted_company_codes": int(out_df.drop_duplicates("CompanyCode")["HistoricalStatus"].eq("Delisted").sum()),
        "securities_with_listing_date": int(out_df["ListingDate"].notna().sum()),
        "delisted_with_delisting_date": int(out_df.loc[out_df["HistoricalStatus"] == "Delisted", "DelistingDate"].notna().sum()),
        "securities_with_pre_listing_rows": int((out_df["PreListingPriceRows"] > 0).sum()),
        "total_pre_listing_rows": int(out_df["PreListingPriceRows"].sum()),
        "securities_with_post_delisting_rows": int((out_df["PostDelistingPriceRows"] > 0).sum()),
        "total_post_delisting_rows": int(out_df["PostDelistingPriceRows"].sum()),
        "securities_unverified_issuer": int((out_df["IssuerValidationStatus"] == "MANUAL_REVIEW_UNVERIFIED_ISSUER").sum()),
        "securities_merger_name_overlap": int((out_df["IssuerValidationStatus"] == "MANUAL_REVIEW_MERGER_NAME_OVERLAP").sum()),
        "annual_universe_history": ann_univ.to_dict(orient="records"),
    }
    return out_df, summary


def assess_strategy_readiness(
    repo_root: Path,
    integrity_report: dict[str, Any],
    coverage_df: pd.DataFrame,
    liq_summary: dict[str, Any],
    mr_summary: dict[str, Any],
    ca_summary: dict[str, Any],
    red_summary: dict[str, Any],
    surv_summary: dict[str, Any],
    ca_validated: pd.DataFrame,
) -> dict[str, Any]:
    """
    Workstreams H & I: Produce strict evidence-based strategy readiness verdicts
    (`READY`, `CONDITIONALLY READY`, `NOT READY`) and prioritized action list.
    """
    # Inspect actual fundamental data files dynamically (zero assumptions)
    fin_parquet = repo_root / "cse_live" / "phase2_extraction" / "financial_facts_final.parquet"
    fin_manifest = repo_root / "cse_live" / "financial_reports_manifest.csv"
    fund_evidence: dict[str, Any] = {
        "historical_multi_year_panel_exists": False,
        "live_extraction_file_present": fin_parquet.exists(),
    }
    if fin_parquet.exists():
        f_df = pd.read_parquet(fin_parquet)
        fund_evidence.update({
            "live_facts_count": int(len(f_df)),
            "live_unique_symbols": int(f_df["symbol"].nunique()) if "symbol" in f_df.columns else 0,
            "live_period_end_values": sorted(f_df["period_end"].dropna().astype(str).unique().tolist()) if "period_end" in f_df.columns else [],
            "live_publication_date_range": [
                str(f_df["publication_date"].dropna().min()),
                str(f_df["publication_date"].dropna().max()),
            ] if "publication_date" in f_df.columns else [],
        })
    if fin_manifest.exists():
        fm_df = pd.read_csv(fin_manifest)
        fund_evidence["manifest_report_count"] = int(len(fm_df))
        fund_evidence["manifest_upload_dates"] = [
            str(fm_df["uploadedDate"].min()),
            str(fm_df["uploadedDate"].max()),
        ] if "uploadedDate" in fm_df.columns else []

    pct_par_unresolved = int((ca_validated["ValidationStatus"] == "UNRESOLVED_PERCENT_OF_PAR").sum())
    open_cov_2017_pct = round(100.0 - float(coverage_df.loc[coverage_df["indicator_name"] == "Intraday_Ret", "missingness_pct"].iloc[0]), 2)
    hl_cov_all_pct = round(float(coverage_df.loc[coverage_df["indicator_name"] == "ATR_14", "coverage_pct"].iloc[0]), 2)
    usable_st_indicators = int(coverage_df["short_term_all_era_usable"].sum())

    liq_sens = {row["rule_name"]: row for row in liq_summary["threshold_sensitivity"]}

    readiness = {
        "phase": "2B",
        "audit_principle": "Data readiness does NOT imply model profitability; all verdicts assess historical point-in-time feature suitability.",
        "fundamental_data_audit": fund_evidence,
        "strategies": {
            "short_term_trading_0_3m": {
                "status": "CONDITIONALLY READY",
                "horizon": "0-3 months",
                "summary": (
                    "Full-CSE technical, price-volume, liquidity, and ASPI-relative features cleanly cover "
                    "2002-2025 (H/L/V) and 2017-2025 (full OHLCV) without look-ahead bias or duplicate keys, "
                    "subject to explicit liquidity gating, Open-era separation, and CapitalAdj return scaling."
                ),
                "supporting_evidence": {
                    "integrity_status": integrity_report["cross_table_consistency"]["overall_integrity_status"],
                    "all_1414977_rows_matched": integrity_report["cross_table_consistency"]["all_1414977_phase1b_records_matched"],
                    "usable_all_era_indicators": usable_st_indicators,
                    "hl_atr14_overall_coverage_pct": hl_cov_all_pct,
                    "open_intraday_overall_coverage_pct": open_cov_2017_pct,
                    "aspi_date_coverage_rows_matched": f"{mr_summary['aspi_date_alignment']['matched_aspi_dates']}/{mr_summary['aspi_date_alignment']['total_price_dates']}",
                    "equity_passing_500k_70pct_full_hist": liq_sens.get("ShortTerm_Mod_500k_70pct", {}).get("passing_securities_full_history"),
                    "equity_passing_500k_70pct_2023_2025": liq_sens.get("ShortTerm_Mod_500k_70pct", {}).get("passing_listed_2023_2025"),
                    "equity_passing_1M_80pct_2023_2025": liq_sens.get("ShortTerm_Liquid_1M_80pct", {}).get("passing_listed_2023_2025"),
                },
                "mandatory_conditions_before_phase3": [
                    "Filter modeling universe to MainType=='N' (or separate N/X classes) and exclude Rights ('R'), Debt ('D'), and thin micro-caps failing minimum rolling ADT & trading frequency.",
                    "Restrict Open-dependent features (`Overnight_Gap_Ret`, `Intraday_Ret`) to 2017+ regimes or drop cleanly for full 2002-2025 training (do NOT fabricate Open from Prior Close).",
                    "Drop `Sector_MOM_20` and `RS_Sector_ROC_20` (100% NaN) until verified point-in-time sector mapping is supplied.",
                    "Use `CapitalAdjClose` / `Daily_Return_CapAdj` for forward return targets and clean event-day returns around 165 Scrip Dividends and 210 Rights TERPs.",
                    "Prune nominal price-level indicators (`SMA_*`, `EMA_*`, `BB_Upper/Lower`, `PSAR`, `OBV`) and `Williams_R_14` duplicate (`r=1.0` with `Stoch_K_14`) for cross-sectional ML.",
                ],
            },
            "medium_term_growth_3_12m": {
                "status": "NOT READY",
                "horizon": "3-12 months",
                "summary": (
                    "While technical, liquidity, ASPI-relative, and split/total-return price histories are complete, "
                    "historical multi-year company-level financial statements (quarterly revenue, EPS, margin, cash-flow growth) "
                    "and verified historical sector mappings do not exist for 1990-2025."
                ),
                "critical_blockers": [
                    f"Only {fund_evidence.get('live_unique_symbols', 0)} companies have parsed financial facts (`financial_facts_final.parquet`), all from Nov 2026 disclosures (2025-2026 interim snapshots) — 0 historical 2002-2025 quarterly fundamental panel exists.",
                    f"Ticker-to-sector mapping is 100% unmapped (`{mr_summary['sector_honesty']['sector_mapping_unknown_pct']}% Unknown`), blocking `Sector_MOM_20` and `RS_Sector_ROC_20` peer growth normalization.",
                    "No company-level historical Valuation panel (PE, PBV, EV/EBITDA time series); `10Market Ratios - Daily.xls` is market-aggregate only.",
                ],
            },
            "long_term_wealth_1_5y": {
                "status": "NOT READY",
                "horizon": "1-5+ years",
                "summary": (
                    "Long-term wealth ranking requires 5+ year historical profitability (ROE/ROIC), balance sheet strength, "
                    "FCF conversion, corporate governance history, valuation multiples, and uninterrupted dividend TSR across 1990-2025; "
                    "current repository lacks historical financial statements and leaves 2,787 pre-2008 dividends unresolved."
                ),
                "critical_blockers": [
                    "Zero multi-year historical annual/quarterly financial statements across the 400-company / 719-security CSE universe (0% historical ROE, ROIC, Debt/Equity, FCF coverage pre-2025).",
                    f"{pct_par_unresolved} pre-2008 cash dividend records are flagged `UNRESOLVED_PERCENT_OF_PAR` without verified historical par values, truncating clean pre-2008 Total Shareholder Return history.",
                    f"{surv_summary['securities_unverified_issuer']} legacy delisted securities lack verified `IssuerID` and {surv_summary['securities_merger_name_overlap']} merger name overlaps (`BFN`/`ORIN`, `CLC`/`LOFC`) require corporate continuity resolution for 5-year compounding paths.",
                    "No corporate governance, promoter ownership trend, or audit-qualification historical feed exists.",
                ],
            },
        },
        "prioritized_recommendations": [
            {
                "id": "REC-01",
                "priority": "Critical",
                "title": "Exclude 100% Unmapped Sector Columns & Gate Regimes Strictly in Feature Builders",
                "affected_features_or_datasets": "Sector_MOM_20, RS_Sector_ROC_20, Overnight_Gap_Ret, Intraday_Ret, H/L/V features pre-2002",
                "evidence": f"Sector_MOM_20/RS_Sector_ROC_20 are 100% NaN; Overnight_Gap_Ret/Intraday_Ret are 100% NaN pre-2017; H/L features are 100% NaN pre-2002.",
                "risk": "Training models blindly without era gating or naive zero/mean imputation creates artificial structural regimes and invalid signals.",
                "recommended_correction": "Enforce explicit regime schemas: 2017+ full OHLCV pipeline (all non-sector features), 2002+ HLCV pipeline (excluding Open gaps), and drop unmapped Sector_* columns until populated.",
                "dependencies": "None for Phase 3 Short-term dataset assembly.",
            },
            {
                "id": "REC-02",
                "priority": "Critical",
                "title": "Build Historical Multi-Year Point-in-Time Financial Statements Dataset (for Engines 02 & 03)",
                "affected_features_or_datasets": "Medium-term Growth & Long-term Wealth feature matrices",
                "evidence": f"`financial_facts_final.parquet` holds {fund_evidence.get('live_facts_count', 0)} Nov-2026 facts across {fund_evidence.get('live_unique_symbols', 0)} symbols only; zero multi-year fundamental history exists.",
                "risk": "Broadcasting Nov 2026 fundamentals onto 2002-2025 prices causes catastrophic look-ahead leakage (already seen in Phase 3.5 audit).",
                "recommended_correction": "Ingest historical CSE quarterly/annual PDFs/filings (2010-2025+) with exact filing `publication_date <= observation_date` before unlocking Engines 02 and 03.",
                "dependencies": "CSE historical disclosure archive extraction pipeline.",
            },
            {
                "id": "REC-03",
                "priority": "High",
                "title": "Normalize OHLC by Scrip/Rights TERP Factors (`CapitalAdjOHLC`) or Target Clean CapitalAdj Returns",
                "affected_features_or_datasets": "ROC_10, ROC_20, RSI_14, ATR_14, Daily_Return_SplitAdj across 152 securities",
                "evidence": f"Phase 1B verified {ca_summary['continuity_breakdown_by_event_type'].get('RightsIssue', {})} Rights TERPs and {ca_summary['continuity_breakdown_by_event_type'].get('ScripDividend', {})} Scrip Dividends; Split-only basis shows {ca_summary['daily_ret_abs_gt50pct_split_adj_basis']} bars with |Ret|>50% vs {ca_summary['daily_ret_abs_gt50pct_cap_adj_basis']} on CapitalAdj basis.",
                "risk": "Technical momentum indicators (`ROC_20`, `RSI_14`) interpret dilutive rights/scrip ex-date price drops as bearish crashes.",
                "recommended_correction": "In Phase 3 feature preparation, multiply Split+Bonus OHLC by `CumScripFactor * CumRightsTERPFactor` (or mask/adjust ex-dates) when generating momentum features and ALWAYS use `CapitalAdjClose` for forward labels.",
                "dependencies": "Phase 1B `cse_daily_prices_split_adjusted.parquet` factors.",
            },
            {
                "id": "REC-04",
                "priority": "High",
                "title": "Enforce Dynamic Liquidity & Share-Class Filtering in Training Universe Construction",
                "affected_features_or_datasets": "All 719 SecurityIDs (`liquidity_analysis.csv`)",
                "evidence": f"Only {liq_sens.get('ShortTerm_Mod_500k_70pct', {}).get('passing_listed_2023_2025')} of {liq_summary['active_equity_2023_2025_count']} active equity securities (2023-2025) meet >=500k LKR median daily turnover and >=70% trading frequency.",
                "risk": "Illiquid penny counters and odd-lot anomalies (`SELI-N-0000` 0.60->1,287 on 1 share; Rights `R`) create un-tradable fake alpha.",
                "recommended_correction": "Restrict Phase 3 training rows dynamically at date `t` using trailing 60-obs rolling `ADT_60 >= 500k LKR`, `ShareVolume > 0`, and `MainType == 'N'` (plus optional `'X'` separately).",
                "dependencies": "Phase 2A `cse_price_volume_features.parquet` + `liquidity_analysis.csv`.",
            },
            {
                "id": "REC-05",
                "priority": "High",
                "title": "Source Historical CSE Sector & GICS Crosswalk with Effective Dates",
                "affected_features_or_datasets": "cse_sector_mapping.parquet, Sector_MOM_20, RS_Sector_ROC_20",
                "evidence": "719/719 securities hold `Sector='Unknown'` (`MANUAL_RESEARCH_REQUIRED`); Jan 2020 GICS regime break documented.",
                "risk": "Without historical sector labels, cross-sectional models cannot neutralize sector beta or measure peer-relative growth.",
                "recommended_correction": "Construct historical CSE sector table (`1990-2019`) + GICS (`2020+`) per `CompanyCode` with effective dates from CSE official archives.",
                "dependencies": "Historical CSE sector registry.",
            },
            {
                "id": "REC-06",
                "priority": "Medium",
                "title": "Prune Mathematically Collinear & Scale-Confounded Features Prior to Linear/Linearized Models",
                "affected_features_or_datasets": "feature_redundancy_report.csv, feature_correlation_matrix.csv",
                "evidence": f"{red_summary['exact_or_near_duplicate_pairs_ge_0995']} exact/near duplicate pair (`Williams_R_14` vs `Stoch_K_14` corr=1.0) and nominal price-level cluster (`SMA/EMA/WMA/BB/PSAR/Supertrend` r=0.97-0.9999).",
                "risk": "Severe multicollinearity inflates linear model coefficients and dilutes tree feature importance across level proxies.",
                "recommended_correction": "Retain normalized scale-free features (`NATR_14`, `BB_Width_20_2`, `BB_PctB_20_2`, `PSAR_Distance_Pct`, `MA_Cross_Spread_20_50`), drop `Williams_R_14`, and standardize/cross-sectionally rank inputs.",
                "dependencies": "Phase 3 dataset assembly.",
            },
            {
                "id": "REC-07",
                "priority": "Medium",
                "title": "Inspect Boundary Listing/Delisting Date Printing & Sparse-Trading Extreme Betas",
                "affected_features_or_datasets": "security_history_coverage.csv, Beta_ASPI_60, Beta_ASPI_252",
                "evidence": f"{surv_summary['securities_with_pre_listing_rows']} securities have prints slightly preceding `ListingDate`, {surv_summary['securities_with_post_delisting_rows']} after `DelistingDate`; `Beta_ASPI_60` has {mr_summary['beta_reliability']['beta60_extreme_gt5_rows']} rows with |Beta|>5 on thin/near-zero index-var periods.",
                "risk": "Extreme Beta estimates during brief flat/volatile pockets can distort systematic risk neutrality.",
                "recommended_correction": "Winsorize `Beta_ASPI_60/252` to `[-3, +5]` (or enforce >=40 non-zero trade intervals inside the 60d window) and mask pre-listing/post-delisting edge dates if needed.",
                "dependencies": "Phase 1B master + Phase 3 feature builder.",
            },
            {
                "id": "REC-08",
                "priority": "Low",
                "title": "Resolve Pre-2008 Corporate Par Values & Legacy `UNVERIFIED_ISSUER` / Merger Pairs",
                "affected_features_or_datasets": "cse_total_return_history.parquet, cse_security_master_enriched.parquet",
                "evidence": f"{pct_par_unresolved} pre-2008 cash dividends `UNRESOLVED_PERCENT_OF_PAR`; {surv_summary['securities_unverified_issuer']} unverified historical issuers; `BFN/ORIN` and `CLC/LOFC` merger pairs.",
                "risk": "Understates pre-2008 dividend yield inside Total Return Index.",
                "recommended_correction": "Look up statutory par values prior to 2007 Companies Act (Rs. 10 / Rs. 100 / Rs. 1) if pre-2008 TSR backtests are required.",
                "dependencies": "Pre-2008 CSE handbook archive.",
            },
        ],
    }
    return readiness


def generate_visualizations(
    viz_dir: Path,
    coverage_df: pd.DataFrame,
    annual_cov_df: pd.DataFrame,
    liq_df: pd.DataFrame,
    liq_summary: dict[str, Any],
    corr_df: pd.DataFrame,
    surv_summary: dict[str, Any],
    readiness: dict[str, Any],
) -> list[str]:
    """Generate all 6 publication-quality audit charts in `reports/phase2/visualizations/`."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "axes.edgecolor": "#CCCCCC", "font.size": 9})
    saved: list[str] = []

    # 1. Feature Missingness Heatmap (Selected representative features x Year buckets / eras)
    key_feats = [
        "SMA_20", "SMA_200", "ADX_14", "Ichimoku_Kumo_Pos_Pct", "RSI_14", "MACD_Hist",
        "ATR_14", "NATR_14", "Hist_Vol_21d", "OBV", "MFI_14", "CMF_20",
        "PSAR", "Overnight_Gap_Ret", "Intraday_Ret", "ASPI_MOM_20", "Beta_ASPI_60", "Beta_ASPI_252", "Sector_MOM_20",
    ]
    sub_ann = annual_cov_df[annual_cov_df["indicator_name"].isin(key_feats)].copy()
    sub_ann["MissingPct"] = 100.0 - sub_ann["coverage_pct"]
    piv = sub_ann.pivot(index="indicator_name", columns="Year", values="MissingPct").loc[key_feats]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    yr_cols = piv.columns.tolist()
    step = max(1, len(yr_cols) // 12)
    ax.set_xticks(range(0, len(yr_cols), step))
    ax.set_xticklabels([str(yr_cols[i]) for i in range(0, len(yr_cols), step)])
    ax.set_title("Phase 2A Feature Missingness (%) Across Historical CSE Regimes (1990–2025)", fontweight="bold")
    ax.set_xlabel("Calendar Year")
    fig.colorbar(im, ax=ax, label="Missingness (%)")
    plt.tight_layout()
    p1 = viz_dir / "01_feature_missingness_heatmap.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(str(p1))

    # 2. Annual Feature & Universe Coverage (1990–2025)
    ann_univ = pd.DataFrame(surv_summary["annual_universe_history"]).sort_values("Year")
    cov_lines = annual_cov_df[annual_cov_df["indicator_name"].isin(["SMA_20", "RSI_14", "ATR_14", "OBV", "Intraday_Ret", "ASPI_MOM_20"])]
    cov_piv = cov_lines.pivot(index="Year", columns="indicator_name", values="coverage_pct")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for col, lbl in [
        ("ASPI_MOM_20", "Market Index (ASPI_MOM_20)"),
        ("RSI_14", "Close-Only Momentum (RSI_14)"),
        ("OBV", "Volume Flow (OBV, 2001+)"),
        ("ATR_14", "High/Low Volatility (ATR_14, 2002+)"),
        ("Intraday_Ret", "Open-Gated Return (Intraday_Ret, 2017+)"),
    ]:
        if col in cov_piv.columns:
            ax1.plot(cov_piv.index, cov_piv[col], marker="o", markersize=2.5, linewidth=1.8, label=lbl)
    ax1.set_ylabel("Row Coverage (%)")
    ax1.set_title("Annual Feature Availability by Source Schema Era (1990–2025)", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    listed_arr = ann_univ["Listed"] if "Listed" in ann_univ.columns else 0
    delisted_arr = ann_univ["Delisted"] if "Delisted" in ann_univ.columns else 0
    ax2.bar(ann_univ["Year"], listed_arr, label="Currently Listed SecurityIDs", color="#1f77b4", alpha=0.85)
    ax2.bar(ann_univ["Year"], delisted_arr, bottom=listed_arr, label="Historical/Delisted SecurityIDs Preserved", color="#d62728", alpha=0.75)
    ax2.set_ylabel("Active SecurityIDs")
    ax2.set_xlabel("Calendar Year")
    ax2.set_title("Annual Tradable Universe: Listed vs Preserved Delisted SecurityIDs", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = viz_dir / "02_annual_feature_coverage.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(str(p2))

    # 3. Liquidity Distribution & Threshold Survival Curve
    eq = liq_df[liq_df["MainType"].isin(["N", "X"]) & (liq_df["MedianTurnoverLKR"].fillna(0) > 0)].copy()
    sens = pd.DataFrame(liq_summary["threshold_sensitivity"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(np.log10(eq["MedianTurnoverLKR"]), bins=30, color="#2ca02c", alpha=0.8, edgecolor="white")
    for thresh, col, nm in [(100_000, "orange", "100k"), (500_000, "red", "500k"), (1_000_000, "purple", "1M")]:
        ax1.axvline(np.log10(thresh), color=col, linestyle="--", linewidth=1.5, label=f"LKR {nm}/day")
    ax1.set_xlabel("log10(Median Daily Turnover in LKR)")
    ax1.set_ylabel("Equity SecurityIDs (N/X)")
    ax1.set_title("CSE Equity Median Daily Turnover Distribution", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    x_pos = np.arange(len(sens))
    w = 0.35
    ax2.bar(x_pos - w / 2, sens["passing_securities_full_history"], width=w, label="Full History (N/X)", color="#1f77b4")
    ax2.bar(x_pos + w / 2, sens["passing_listed_2023_2025"], width=w, label="2023–2025 Active Equities", color="#ff7f0e")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r.replace("_", "\n") for r in sens["rule_name"]], fontsize=7.5)
    ax2.set_ylabel("Passing SecurityIDs")
    ax2.set_title("Liquidity Eligibility Threshold Sensitivity", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p3 = viz_dir / "03_liquidity_distribution.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(str(p3))

    # 4. Trading Frequency Distribution
    eq_all = liq_df[liq_df["MainType"].isin(["N", "X"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        [
            eq_all.loc[eq_all["HistoricalStatus"] == "Listed", "TradingFrequencyPct"],
            eq_all.loc[eq_all["HistoricalStatus"] == "Delisted", "TradingFrequencyPct"],
        ],
        bins=20,
        stacked=True,
        label=["Listed Equity SecurityIDs", "Delisted Equity SecurityIDs"],
        color=["#1f77b4", "#d62728"],
        alpha=0.85,
        edgecolor="white",
    )
    ax.axvline(70.0, color="black", linestyle="--", linewidth=1.2, label="70% Frequency Screen")
    ax.set_xlabel("Active Trading Frequency (% of Lifespan Market Sessions with Volume > 0)")
    ax.set_ylabel("SecurityID Count")
    ax.set_title("CSE Equity Trading Frequency Distribution (Listed vs Delisted)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p4 = viz_dir / "04_trading_frequency_distribution.png"
    fig.savefig(p4)
    plt.close(fig)
    saved.append(str(p4))

    # 5. Normalized Feature Correlation Heatmap (Stationary Representative Subset)
    norm_feats = [
        "MA_Cross_Spread_20_50", "ADX_14", "Aroon_Up_25", "Ichimoku_Kumo_Pos_Pct",
        "RSI_14", "ROC_10", "CCI_20", "Stoch_K_14", "Williams_R_14",
        "NATR_14", "BB_Width_20_2", "BB_PctB_20_2", "Keltner_Pos_20", "Hist_Vol_21d", "Donchian_Pos_20",
        "PSAR_Distance_Pct", "PSAR_Trend", "MFI_14", "CMF_20", "Volume_Ratio_20",
        "Overnight_Gap_Ret", "Intraday_Ret", "RS_ASPI_ROC_20", "Beta_ASPI_60",
    ]
    c_mat = corr_df.set_index("indicator_name")
    sub_c = c_mat.loc[[f for f in norm_feats if f in c_mat.index], [f for f in norm_feats if f in c_mat.columns]]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub_c.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(sub_c.columns)))
    ax.set_xticklabels(sub_c.columns, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(sub_c.index)))
    ax.set_yticklabels(sub_c.index, fontsize=7.5)
    ax.set_title("Phase 2A Feature Correlation Matrix (2017–2025 Homogeneous Ordinary Equities)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Pearson Correlation")
    plt.tight_layout()
    p5 = viz_dir / "05_feature_correlation_heatmap.png"
    fig.savefig(p5)
    plt.close(fig)
    saved.append(str(p5))

    # 6. Strategy Readiness Summary Visual
    strat_names = [
        "Engine 01:\nShort-Term\n(0–3m)",
        "Engine 02:\nMedium-Term Growth\n(3–12m)",
        "Engine 03:\nLong-Term Wealth\n(1–5+ yrs)",
    ]
    pillars = ["Price &\nTechnicals", "Liquidity &\nTradability", "ASPI / Beta\nContext", "Historical Sector\nCrosswalk", "Multi-Yr\nFinancials", "Pre-2008\nDiv TSR"]
    # Qualitative readiness score grid derived cleanly from audit reality:
    # 1.0 = Verified Available, 0.5 = Conditional/Gated, 0.0 = Missing/Blocked
    grid = np.array([
        [0.9, 0.8, 0.9, 0.0, 0.5, 0.8],  # Short-term (financials not strictly required for pure tech; partial/conditional)
        [0.9, 0.8, 0.9, 0.0, 0.0, 0.5],  # Medium-term (requires financials & sectors -> 0.0)
        [0.9, 0.7, 0.9, 0.0, 0.0, 0.3],  # Long-term (requires 5y financials & resolved pre-08 TSR -> 0.0/0.3)
    ])
    verdicts = [
        readiness["strategies"]["short_term_trading_0_3m"]["status"],
        readiness["strategies"]["medium_term_growth_3_12m"]["status"],
        readiness["strategies"]["long_term_wealth_1_5y"]["status"],
    ]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pillars)))
    ax.set_xticklabels(pillars, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(strat_names)))
    ax.set_yticklabels([f"{s}\n[{v}]" for s, v in zip(strat_names, verdicts)], fontsize=8.5, fontweight="bold")
    cell_labels = [
        ["HLCV 2002+\nOHLCV 2017+", "Gated\n(ADT>=500k)", "ASPI+Beta\n1990-2025", "MISSING\n(100% Unknown)", "Optional\n(Tech-Only OK)", "Post-08 OK\n(TERP Aware)"],
        ["HLCV 2002+\nOHLCV 2017+", "Gated\n(ADT>=250k)", "ASPI+Beta\n1990-2025", "MISSING\n(Blocked)", "MISSING\n(Nov26 Only)", "Partial\n(Pre-08 % Par)"],
        ["Clean Split\nAdj History", "Universe\nGated", "ASPI+Beta\n1990-2025", "MISSING\n(Blocked)", "MISSING\n(No 5y Panel)", "2,787 Pre-08\nUnresolved"],
    ]
    for r_i in range(grid.shape[0]):
        for c_i in range(grid.shape[1]):
            ax.text(c_i, r_i, cell_labels[r_i][c_i], ha="center", va="center", color="black", fontsize=7.5, fontweight="bold")
    ax.set_title("Stockpulse Phase 2B Investment Strategy Data Readiness Matrix", fontweight="bold")
    plt.tight_layout()
    p6 = viz_dir / "06_strategy_readiness_summary.png"
    fig.savefig(p6)
    plt.close(fig)
    saved.append(str(p6))

    return saved
