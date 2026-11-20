"""
Stockpulse Phase 3: Fundamental Ratios Engine
Extracts standardized, point-in-time financial statement ratios from financial_facts_enriched.parquet
and incorporates company share counts and market capitalization data.
"""
from __future__ import annotations
import csv, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

PHASE2_DIR = Path("cse_live/phase2_extraction")
OUT_DIR = Path("phase3_feature_engineering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_fundamental_ratios() -> pd.DataFrame:
    print("=" * 70)
    print("Stockpulse Phase 3: Point-in-Time Fundamental Ratios Engine")
    print("=" * 70)

    # 1. Load Enriched Facts
    facts_path = PHASE2_DIR / "financial_facts_enriched.parquet"
    if not facts_path.exists():
        facts_path = PHASE2_DIR / "financial_facts_final.parquet"
    df_facts = pd.read_parquet(facts_path)
    print(f"Loaded {len(df_facts)} financial facts across {df_facts['symbol'].nunique()} companies")

    # 2. Load Snapshots for Shares in Issue & Market Cap
    with open("cse_live/company_current_snapshots.csv", encoding="utf-8-sig") as f:
        snapshots_raw = list(csv.DictReader(f))

    shares_map = {}
    price_map = {}
    mcap_map = {}
    beta_map = {}
    for r in snapshots_raw:
        sym = r.get("symbol", "").split(".")[0].strip()
        if not sym:
            continue
        try:
            shares = float(r.get("company.quantityIssued", 0) or 0)
            shares_map[sym] = shares
        except ValueError:
            pass
        try:
            px = float(r.get("company.lastTradedPrice", 0) or r.get("company.closingPrice", 0) or 0)
            price_map[sym] = px
        except ValueError:
            pass
        try:
            mc = float(r.get("company.marketCap", 0) or 0)
            mcap_map[sym] = mc
        except ValueError:
            pass
        try:
            b = float(r.get("beta.betaValueSPSL", 0) or 0)
            beta_map[sym] = b
        except ValueError:
            pass

    # 3. Pivot facts by symbol and comparative_flag
    # Current period (comparative_flag == False)
    current_facts = df_facts[df_facts["comparative_flag"] == False].copy()
    comp_facts = df_facts[df_facts["comparative_flag"] == True].copy()

    records = []
    for sym, group in current_facts.groupby("symbol"):
        cname = group["company_name"].iloc[0]
        cc = group["company_code"].iloc[0]
        p_end = group["period_end"].iloc[0]
        pub_date = group["publication_date"].iloc[0]
        scope = group["consolidation_scope"].iloc[0]

        # Extract metrics (normalized LKR values)
        met_map = {}
        for _, row in group.iterrows():
            m = row["metric"]
            v = row["normalized_value_lkr"]
            # prefer Group over Company if multiple
            if m not in met_map or row["consolidation_scope"] == "Group":
                met_map[m] = v

        # Comparative period metrics for YoY calculations
        comp_group = comp_facts[comp_facts["symbol"] == sym]
        comp_map = {}
        for _, row in comp_group.iterrows():
            m = row["metric"]
            v = row["normalized_value_lkr"]
            if m not in comp_map or row["consolidation_scope"] == "Group":
                comp_map[m] = v

        rev = met_map.get("revenue", np.nan)
        gp = met_map.get("gross_profit", np.nan)
        ebit = met_map.get("operating_profit", np.nan)
        fc = met_map.get("finance_costs", 0.0)
        pbt = met_map.get("profit_before_tax", np.nan)
        pat = met_map.get("net_profit", np.nan)
        eps = met_map.get("eps", np.nan)
        assets = met_map.get("total_assets", np.nan)
        liab = met_map.get("total_liabilities", np.nan)
        eq = met_map.get("total_equity", np.nan)
        ocf = met_map.get("operating_cashflow", np.nan)

        # Comparative metrics
        prev_rev = comp_map.get("revenue", np.nan)
        prev_ebit = comp_map.get("operating_profit", np.nan)
        prev_pat = comp_map.get("net_profit", np.nan)

        shares = shares_map.get(sym, np.nan)
        price = price_map.get(sym, np.nan)
        mcap = mcap_map.get(sym, (shares * price) if (pd.notna(shares) and pd.notna(price)) else np.nan)
        beta = beta_map.get(sym, np.nan)

        # ── Ratios Calculation ────────────────────────────────────────────────
        # Margins (%)
        gp_margin = (gp / rev * 100.0) if (pd.notna(gp) and pd.notna(rev) and rev > 0) else np.nan
        ebit_margin = (ebit / rev * 100.0) if (pd.notna(ebit) and pd.notna(rev) and rev > 0) else np.nan
        net_margin = (pat / rev * 100.0) if (pd.notna(pat) and pd.notna(rev) and rev > 0) else np.nan

        # Profitability & Returns (%)
        # Annualized Net Profit (interim period is typically 6M; multiply by 2 for interim annualization)
        annual_pat = pat * 2.0 if pd.notna(pat) else np.nan
        roe = (annual_pat / eq * 100.0) if (pd.notna(annual_pat) and pd.notna(eq) and eq > 0) else np.nan
        roa = (annual_pat / assets * 100.0) if (pd.notna(annual_pat) and pd.notna(assets) and assets > 0) else np.nan
        roic = ((ebit * 2.0 * 0.70) / (eq + liab) * 100.0) if (pd.notna(ebit) and pd.notna(eq) and pd.notna(liab) and (eq + liab) > 0) else np.nan

        # Solvency & Leverage
        debt_to_equity = (liab / eq) if (pd.notna(liab) and pd.notna(eq) and eq > 0) else np.nan
        equity_multiplier = (assets / eq) if (pd.notna(assets) and pd.notna(eq) and eq > 0) else np.nan
        interest_coverage = (ebit / fc) if (pd.notna(ebit) and pd.notna(fc) and fc > 0) else (100.0 if (pd.notna(ebit) and ebit > 0 and fc == 0) else np.nan)
        distressed_equity_flag = int(eq < 0) if pd.notna(eq) else 0

        # Growth Metrics YoY (%)
        rev_growth_yoy = ((rev - prev_rev) / abs(prev_rev) * 100.0) if (pd.notna(rev) and pd.notna(prev_rev) and prev_rev != 0) else np.nan
        ebit_growth_yoy = ((ebit - prev_ebit) / abs(prev_ebit) * 100.0) if (pd.notna(ebit) and pd.notna(prev_ebit) and prev_ebit != 0) else np.nan
        pat_growth_yoy = ((pat - prev_pat) / abs(prev_pat) * 100.0) if (pd.notna(pat) and pd.notna(prev_pat) and prev_pat != 0) else np.nan

        # Cash Flow Quality
        ocf_to_net_income = (ocf / pat) if (pd.notna(ocf) and pd.notna(pat) and pat != 0) else np.nan

        # Per Share & Valuation Multiples
        bvps = (eq / shares) if (pd.notna(eq) and pd.notna(shares) and shares > 0) else np.nan
        calc_eps = (annual_pat / shares) if (pd.notna(annual_pat) and pd.notna(shares) and shares > 0) else eps

        pe_ratio = (price / calc_eps) if (pd.notna(price) and pd.notna(calc_eps) and calc_eps > 0) else np.nan
        pb_ratio = (price / bvps) if (pd.notna(price) and pd.notna(bvps) and bvps > 0) else np.nan
        ev = (mcap + liab) if (pd.notna(mcap) and pd.notna(liab)) else np.nan
        ev_ebitda_proxy = (ev / (ebit * 2.0)) if (pd.notna(ev) and pd.notna(ebit) and ebit > 0) else np.nan
        p_ocf = (mcap / (ocf * 2.0)) if (pd.notna(mcap) and pd.notna(ocf) and ocf > 0) else np.nan

        # Graham Number = sqrt(22.5 * EPS * BVPS)
        graham_number = np.sqrt(22.5 * calc_eps * bvps) if (pd.notna(calc_eps) and pd.notna(bvps) and calc_eps > 0 and bvps > 0) else np.nan
        graham_margin_of_safety_pct = ((graham_number - price) / graham_number * 100.0) if (pd.notna(graham_number) and pd.notna(price) and graham_number > 0) else np.nan

        records.append({
            "company_code": cc,
            "symbol": sym,
            "company_name": cname,
            "period_end": p_end,
            "publication_date": pub_date,
            "consolidation_scope": scope,
            "shares_in_issue": shares,
            "last_traded_price": price,
            "market_cap_lkr": mcap,
            "beta": beta,
            # Core Statements (LKR)
            "revenue_lkr": rev,
            "gross_profit_lkr": gp,
            "operating_profit_lkr": ebit,
            "finance_costs_lkr": fc,
            "profit_before_tax_lkr": pbt,
            "net_profit_lkr": pat,
            "total_assets_lkr": assets,
            "total_liabilities_lkr": liab,
            "total_equity_lkr": eq,
            "operating_cashflow_lkr": ocf,
            # Per Share & Valuation
            "book_value_per_share": round(bvps, 2) if pd.notna(bvps) else np.nan,
            "annualized_eps": round(calc_eps, 2) if pd.notna(calc_eps) else np.nan,
            "pe_ratio": round(pe_ratio, 2) if pd.notna(pe_ratio) else np.nan,
            "pb_ratio": round(pb_ratio, 2) if pd.notna(pb_ratio) else np.nan,
            "ev_ebitda_proxy": round(ev_ebitda_proxy, 2) if pd.notna(ev_ebitda_proxy) else np.nan,
            "p_ocf_ratio": round(p_ocf, 2) if pd.notna(p_ocf) else np.nan,
            "graham_number": round(graham_number, 2) if pd.notna(graham_number) else np.nan,
            "graham_margin_of_safety_pct": round(graham_margin_of_safety_pct, 2) if pd.notna(graham_margin_of_safety_pct) else np.nan,
            # Margins (%)
            "gross_profit_margin_pct": round(gp_margin, 2) if pd.notna(gp_margin) else np.nan,
            "operating_profit_margin_pct": round(ebit_margin, 2) if pd.notna(ebit_margin) else np.nan,
            "net_profit_margin_pct": round(net_margin, 2) if pd.notna(net_margin) else np.nan,
            # Returns (%)
            "roe_pct": round(roe, 2) if pd.notna(roe) else np.nan,
            "roa_pct": round(roa, 2) if pd.notna(roa) else np.nan,
            "roic_pct": round(roic, 2) if pd.notna(roic) else np.nan,
            # Solvency & Leverage
            "debt_to_equity": round(debt_to_equity, 2) if pd.notna(debt_to_equity) else np.nan,
            "equity_multiplier": round(equity_multiplier, 2) if pd.notna(equity_multiplier) else np.nan,
            "interest_coverage": round(interest_coverage, 2) if pd.notna(interest_coverage) else np.nan,
            "distressed_equity_flag": distressed_equity_flag,
            # Growth YoY (%)
            "revenue_growth_yoy_pct": round(rev_growth_yoy, 2) if pd.notna(rev_growth_yoy) else np.nan,
            "operating_profit_growth_yoy_pct": round(ebit_growth_yoy, 2) if pd.notna(ebit_growth_yoy) else np.nan,
            "net_profit_growth_yoy_pct": round(pat_growth_yoy, 2) if pd.notna(pat_growth_yoy) else np.nan,
            # Cash Flow Quality
            "ocf_to_net_income": round(ocf_to_net_income, 2) if pd.notna(ocf_to_net_income) else np.nan
        })

    df_ratios = pd.DataFrame(records)
    out_parquet = OUT_DIR / "fundamental_ratios_quarterly.parquet"
    out_csv = OUT_DIR / "fundamental_ratios_summary.csv"
    df_ratios.to_parquet(out_parquet, index=False)
    df_ratios.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Generated fundamental ratios for {len(df_ratios)} companies.")
    print(f"Saved: {out_parquet}")
    print(f"Saved: {out_csv}")
    return df_ratios

if __name__ == "__main__":
    df_res = build_fundamental_ratios()
    print("\nSample Fundamental Ratios:")
    sample_cols = ["symbol", "pe_ratio", "pb_ratio", "roe_pct", "operating_profit_margin_pct", "revenue_growth_yoy_pct", "debt_to_equity"]
    print(df_res[sample_cols].dropna().head(10))
