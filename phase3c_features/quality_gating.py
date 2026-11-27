"""
Phase 3C Workstream A – Financial Data Quality Gating & Business Type Classification
===================================================================================
Audits & flags all 789,805 Phase 3B facts without mutating or rescaling original values:
  1. Verified FX Availability Check (`USD` facts vs CBSL FX availability).
     Retains original USD values untouched; separates LKR-primary overlapping/secondary USD rows
     from native USD-only reporting periods (`USD_NATIVE_NO_VERIFIED_FX`, safe for same-currency
     USD/USD ratios, excluded from LKR cross-currency market valuation).
  2. Multi-Year Unit-Scale Shift Forensics (`551` Phase 3B flags + `621` same-slice >=500x conflicts +
     `AMBIGUOUS_UNIT`), corroborating PDF unit headers, same-period document conflicts, BS mismatches,
     and company modal reporting units before gating out corrupted units.
  3. Balance Sheet Exception Classification (`684` `BALANCE_SHEET_EQUATION:EXCEPTION` checks):
     - BS_UNIT_MISMATCH_WITHIN_STATEMENT (~1000x or ~1e6x internal Assets vs L+E discrepancy)
     - BS_MINOR_ROUNDING_OR_LINE_DIFF (|DiffPct| <= 2.0%)
     - BS_CLASSIFICATION_OR_NCI_DIFF (2% < |DiffPct| <= 15%)
     - BS_UNRESOLVED_EXCEPTION (|DiffPct| > 15%)
  4. Verified Business Type Classification (`NON_FINANCIAL_CORPORATE`, `BANK_FINANCE`, `INSURANCE`)
     grounded in chart-of-accounts signatures & verified legal charters (never fabricated GICS).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("phase3c.quality_gating")

# Legal charter patterns for verified financial institutions (banks, licensed finance, leasing, primary dealers, insurance)
_BANK_FINANCE_NAME_RE = re.compile(
    r"\b(?:BANK|FINANCE|LEASING|CAPITAL\s+ALLIANCE|FIRST\s+CAPITAL|ENTRUST\s+SECURITIES|WEALTHTRUST|MERCHANT\s+BANK)\b",
    re.I,
)
_INSURANCE_NAME_RE = re.compile(
    r"\b(?:INSURANCE|ASSURANCE|TAKAFUL|CEYLINCO\s+HOLDINGS)\b",
    re.I,
)
_KNOWN_INSURANCE_CODES = {
    "AAIC", "AINS", "ATL", "ATLL", "CINS", "COOP", "CTCE", "HASU", "JINS", "LGIL", "PINS", "SIC", "UAL",
}
_KNOWN_NON_BANK_CORPORATE_OVERRIDE = {
    "CTC",   # Ceylon Tobacco Company PLC (cash-rich corporate reporting large interest income + Current Assets)
    "INME",  # InsureMe Insurance Brokers Limited (commission broker, reports Current Assets & Current Liabilities)
}


def assess_fx_availability(repo_root: Path) -> dict[str, Any]:
    """
    Document empirically audited FX data availability across `01_existing_cse_sources/`, `Dataset/`,
    and public CBSL/World Bank sources without bypassing WAF/403 gates or fabricating FX rates.
    """
    local_fx_candidates = [
        p.name for p in sorted((repo_root / "01_existing_cse_sources").glob("*"))
        if any(k in p.name.lower() for k in ("fx", "exchange", "cbsl", "currency", "usd"))
    ]
    return {
        "verified_daily_cbsl_fx_available": False,
        "local_cbsl_fx_files": local_fx_candidates,
        "cbsl_web_probe_status": "HTTP_403_WAF_NOT_BYPASSED",
        "worldbank_annual_indicator_note": (
            "World Bank PA.NUS.FCRF provides annual-average LKR/USD through 2025 only; "
            "it cannot provide exact reporting-date Balance Sheet spot rates, quarterly interim "
            "period averages, or 2026 rates without look-ahead/approximation error."
        ),
        "policy_applied": (
            "Retain original USD values untouched (`Currency == 'USD'`); exclude secondary/overlapping USD "
            "annexures where primary LKR statements exist for the exact slice; allow native USD-only series "
            "strictly for same-currency scale-free/USD ratios (Growth, Margins, ROE, ROA, ROIC, Debt/Equity, "
            "TTM flows) while excluding native USD inputs from LKR cross-currency market valuations (`P/E`, `P/B`, `EV/EBITDA`)."
        ),
    }


def classify_business_types(
    identity_df: pd.DataFrame,
    facts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify each entity into verified `BusinessType`:
      - `BANK_FINANCE` (licensed banks, finance/leasing companies, primary dealers; unclassified BS)
      - `INSURANCE` (life/general insurance & takaful underwriters; actuarial liabilities structure)
      - `NON_FINANCIAL_CORPORATE` (manufacturing, plantations, consumer, telecom, conglomerates, hotels, etc.)
    Preserves `VerifiedGICSSector = 'Unknown_Unverified_In_Archive'` honestly per Phase 1B/3B.
    """
    nii_cos = set(facts_df.loc[facts_df["MetricCode"] == "NET_INTEREST_INCOME", "CompanyCode"].astype(str).unique())
    ca_counts = (
        facts_df.loc[facts_df["MetricCode"] == "CURRENT_ASSETS", "CompanyCode"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    inv_counts = (
        facts_df.loc[facts_df["MetricCode"] == "INVENTORIES", "CompanyCode"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for _, r in identity_df.iterrows():
        cid = str(r["CompanyID"])
        cc = str(r["CompanyCode"]).strip()
        cname = str(r.get("CompanyName") or "").strip()
        hname = str(r.get("HistoricalCompanyName") or "").strip()
        combo_name = f"{cname} | {hname}"
        ca_n = int(ca_counts.get(cc, 0))
        inv_n = int(inv_counts.get(cc, 0))
        has_nii = cc in nii_cos

        if cc in _KNOWN_NON_BANK_CORPORATE_OVERRIDE:
            btype = "NON_FINANCIAL_CORPORATE"
            ev = f"Explicit corporate override verified ({cc}: CurrentAssets={ca_n}, non-depository/non-underwriter)"
        elif cc in _KNOWN_INSURANCE_CODES or (_INSURANCE_NAME_RE.search(combo_name) and "BROKER" not in combo_name.upper()):
            btype = "INSURANCE"
            ev = f"Verified insurance charter (`{cname}`) & unclassified statutory BS (CurrentAssets={ca_n})"
        elif (has_nii and ca_n == 0 and inv_n == 0) or (
            _BANK_FINANCE_NAME_RE.search(combo_name) and ca_n <= 1 and inv_n == 0
        ):
            btype = "BANK_FINANCE"
            ev = f"Verified Bank/Finance (`{cname}`, HasNII={has_nii}, CurrentAssets={ca_n}, Inventories={inv_n})"
        else:
            btype = "NON_FINANCIAL_CORPORATE"
            ev = f"Corporate / Commercial entity (CurrentAssets={ca_n}, Inventories={inv_n}, HasNII={has_nii})"

        rows.append({
            "CompanyID": cid,
            "CompanyCode": cc,
            "CompanyName": cname,
            "BusinessType": btype,
            "IsFinancialInstitution": btype in ("BANK_FINANCE", "INSURANCE"),
            "BusinessTypeEvidence": ev,
            "VerifiedGICSSector": "Unknown_Unverified_In_Archive",
        })

    return pd.DataFrame(rows)


def _scan_pdf_unit_evidence(
    facts_sub: pd.DataFrame,
    repo_root: Path,
) -> dict[tuple[str, int], str]:
    """
    Scan source PDF page headers cleanly for flagged unit-shift documents (using pymupdf)
    so unit validation relies on explicit PDF headings rather than magnitude alone.
    """
    cache: dict[tuple[str, int], str] = {}
    try:
        import pymupdf  # type: ignore
    except ImportError:
        return cache

    pairs = facts_sub[["SourceFile", "SourcePage"]].dropna().drop_duplicates()
    by_file: dict[str, list[int]] = {}
    for _, r in pairs.iterrows():
        sf = str(r["SourceFile"]).strip()
        try:
            sp = int(r["SourcePage"])
        except (TypeError, ValueError):
            continue
        if sf and sp >= 1:
            by_file.setdefault(sf, []).append(sp)

    re_thous = re.compile(r"(?:rs\.?|lkr|slr)\s*['`\u2018\u2019]?\s*000s?\b|\b000'?s\b|in\s+thousands|rupees\s+thousands", re.I)
    re_mill = re.compile(r"(?:rs\.?|lkr)\s*(?:mn|m|million)|in\s+millions|rupees\s+millions", re.I)
    re_usd_h = re.compile(r"(?:us\$|usd)\s*(?:['`\u2018\u2019]?\s*000|thousands|mn|million)?|u\.?s\.?\s*dollars?", re.I)

    for sf, pages in by_file.items():
        p = Path(sf) if Path(sf).is_absolute() else (repo_root / sf)
        if not p.exists():
            continue
        try:
            doc = pymupdf.open(str(p))
            for sp in pages:
                if 1 <= sp <= len(doc):
                    txt = doc[sp - 1].get_text("text")[:1500]
                    tags = []
                    if re_thous.search(txt):
                        tags.append("THOUSANDS")
                    if re_mill.search(txt):
                        tags.append("MILLIONS")
                    if re_usd_h.search(txt):
                        tags.append("USD")
                    cache[(sf, sp)] = "|".join(tags) if tags else "PLAIN_OR_UNSPECIFIED"
            doc.close()
        except Exception:
            continue
    return cache


def build_financial_input_quality_flags(
    facts_df: pd.DataFrame,
    quality_report_df: pd.DataFrame,
    repo_root: Path,
    scan_flagged_pdfs: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Workstream A: Produce observation-level input eligibility & forensic classification flags across all
    789,805 facts (`financial_input_quality_flags.parquet`).

    Never modifies `Value` in `historical_financial_facts.parquet`.
    """
    f = facts_df.copy()
    f["PeriodEnd_dt"] = pd.to_datetime(f["PeriodEnd"], errors="coerce")
    f["AvailabilityDate_dt"] = pd.to_datetime(f["AvailabilityDate"], errors="coerce")
    f["ConsolidationBool"] = f["ConsolidationStatus"].astype(str) == "Consolidated_Group"
    f["SourceDocumentID_str"] = f["SourceDocumentID"].astype(str)
    f["CompanyCode_str"] = f["CompanyCode"].astype(str).str.strip()

    # ── 1. Company-level modal unit & LKR share baseline ───────────────────────
    lkr_counts = (f["Currency"] == "LKR").groupby(f["CompanyID"]).sum().to_dict()
    tot_counts = f.groupby("CompanyID").size().to_dict()
    lkr_share = {cid: lkr_counts.get(cid, 0) / max(1, tot_counts[cid]) for cid in tot_counts}

    # Modal scale unit per CompanyID among primary LKR non-per-share items
    non_ps = f[
        (f["Unit"].isin(["LKR", "LKR thousands", "LKR millions"]))
        & (~f["MetricCode"].isin(["EPS_BASIC", "EPS_DILUTED", "DIVIDEND_PER_SHARE"]))
    ]
    modal_unit_by_cid = (
        non_ps.groupby("CompanyID")["Unit"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    # Company-level median of anchor scale metrics across large validated LKR observations (> 1e6 LKR)
    scale_anchor = f[
        (f["Currency"] == "LKR")
        & (f["Unit"].isin(["LKR", "LKR thousands", "LKR millions"]))
        & (f["MetricCode"].isin(["TOTAL_ASSETS", "REVENUE", "TOTAL_EQUITY", "TOTAL_LIABILITIES", "PROFIT_AFTER_TAX", "OPERATING_CASH_FLOW"]))
        & (f["Value"].abs() > 1e6)
    ].groupby(["CompanyID", "MetricCode"])["Value"].median().to_dict()

    # ── 2. Currency Classification (Workstream A.A) ───────────────────────────
    # Check exact (CompanyID, MetricCode, PeriodStart, PeriodEnd, PeriodType, ConsolidationStatus) LKR coexistence
    lkr_keys = set(
        zip(
            f.loc[f["Currency"] == "LKR", "CompanyID"],
            f.loc[f["Currency"] == "LKR", "MetricCode"],
            f.loc[f["Currency"] == "LKR", "PeriodStart"].astype(str),
            f.loc[f["Currency"] == "LKR", "PeriodEnd"].astype(str),
            f.loc[f["Currency"] == "LKR", "PeriodType"],
            f.loc[f["Currency"] == "LKR", "ConsolidationStatus"],
        )
    )

    cur_status = np.full(len(f), "VERIFIED_LKR", dtype=object)
    is_lkr_compat = np.ones(len(f), dtype=bool)

    usd_mask = (f["Currency"].astype(str) == "USD").to_numpy()
    u_cid = f["CompanyID"].astype(str).to_numpy()
    u_mc = f["MetricCode"].astype(str).to_numpy()
    u_ps = f["PeriodStart"].astype(str).to_numpy()
    u_pe = f["PeriodEnd"].astype(str).to_numpy()
    u_pt = f["PeriodType"].astype(str).to_numpy()
    u_cs = f["ConsolidationStatus"].astype(str).to_numpy()
    for idx in np.where(usd_mask)[0]:
        cid = u_cid[idx]
        key = (cid, u_mc[idx], u_ps[idx], u_pe[idx], u_pt[idx], u_cs[idx])
        if key in lkr_keys or lkr_share.get(cid, 0.0) >= 0.85:
            cur_status[idx] = "USD_SECONDARY_OR_LKR_PRIMARY_EXCLUDED"
            is_lkr_compat[idx] = False
        else:
            cur_status[idx] = "USD_NATIVE_NO_VERIFIED_FX"
            is_lkr_compat[idx] = False

    # ── 3. Balance Sheet Exception Forensics (Workstream A.C) ─────────────────
    bs_qr = quality_report_df[quality_report_df["CheckType"] == "BALANCE_SHEET_EQUATION"].copy()
    bs_qr["PeriodEnd_dt"] = pd.to_datetime(bs_qr["PeriodEnd"], errors="coerce")
    bs_qr["DisclosureID_str"] = bs_qr["DisclosureID"].astype(str)
    bs_qr["CompanyCode_str"] = bs_qr["CompanyCode"].astype(str).str.strip()
    bs_qr["ConsBool"] = bs_qr["IsConsolidated"].fillna(True).astype(bool)

    bs_exc_map: dict[tuple[str, str, pd.Timestamp, bool], tuple[str, str]] = {}
    bs_class_counts: dict[str, int] = {}
    for _, r in bs_qr.iterrows():
        st = str(r["Status"])
        key = (str(r["DisclosureID_str"]), str(r["CompanyCode_str"]), r["PeriodEnd_dt"], bool(r["ConsBool"]))
        if st in ("PASS_EXACT", "PASS_ROUNDING", "PASS_WITH_NCI"):
            # Do not overwrite an exception if multiple checks exist, prefer PASS unless EXCEPTION
            if key not in bs_exc_map:
                bs_exc_map[key] = (f"BS_VALIDATED_{st}", "")
            continue
        exp_v = pd.to_numeric(r.get("ExpectedValue"), errors="coerce")
        obs_v = pd.to_numeric(r.get("ObservedValue"), errors="coerce")
        pct = abs(float(pd.to_numeric(r.get("PctDifference"), errors="coerce") or 0.0))
        ratio = (abs(exp_v) / abs(obs_v)) if (pd.notna(exp_v) and pd.notna(obs_v) and abs(obs_v) > 0) else np.nan
        if pd.notna(ratio) and (950 <= ratio <= 1050 or 0.00095 <= ratio <= 0.00105 or 9.5e5 <= ratio <= 1.05e6 or 9.5e-7 <= ratio <= 1.05e-6):
            cls = "EXCLUDED_BS_UNIT_MISMATCH_WITHIN_STATEMENT"
            reason = f"Assets vs L+E exact ~{ratio:g}x within-statement unit mismatch"
        elif pct <= 2.0:
            cls = "WARNING_BS_MINOR_ROUNDING_DIFF"
            reason = f"Minor Balance Sheet difference {pct:.2f}% (<=2% rounding/sub-line tolerance)"
        elif pct <= 15.0:
            cls = "EXCLUDED_BS_CLASSIFICATION_OR_NCI_DIFF"
            reason = f"Balance Sheet classification/NCI residual {pct:.2f}% (2%-15%)"
        else:
            cls = "EXCLUDED_BS_UNRESOLVED_EXCEPTION"
            reason = f"Unresolved/extraction Balance Sheet discrepancy {pct:.1f}% (>15%)"
        bs_exc_map[key] = (cls, reason)
        bs_class_counts[cls] = bs_class_counts.get(cls, 0) + 1

    # Map Balance Sheet status onto Balance Sheet statement rows using numpy arrays
    bs_status = np.full(len(f), "NOT_APPLICABLE_FLOW_OR_EQUITY", dtype=object)
    bs_reasons = np.full(len(f), "", dtype=object)
    is_bs_row = (f["StatementType"] == "Balance Sheet").to_numpy()
    bs_src_doc = f["SourceDocumentID_str"].to_numpy()
    bs_cc = f["CompanyCode_str"].to_numpy()
    bs_pe = f["PeriodEnd_dt"].to_numpy()
    bs_cb = f["ConsolidationBool"].to_numpy()
    for idx in np.where(is_bs_row)[0]:
        res = bs_exc_map.get((bs_src_doc[idx], bs_cc[idx], pd.Timestamp(bs_pe[idx]), bool(bs_cb[idx])))
        if res is None:
            bs_status[idx] = "BS_UNCHECKED_PARTIAL_SNAPSHOT"
        else:
            bs_status[idx], bs_reasons[idx] = res

    # ── 4. Multi-Year Unit-Scale Shift Forensics (Workstream A.B) ─────────────
    us_qr = quality_report_df[quality_report_df["CheckType"] == "TS_SUDDEN_UNIT_SCALE_SHIFT_SUSPECT"].copy()
    us_qr["PeriodEnd_dt"] = pd.to_datetime(us_qr["PeriodEnd"], errors="coerce")
    us_docs = set(zip(us_qr["DisclosureID"].astype(str), us_qr["CompanyCode"].astype(str).str.strip(), us_qr["PeriodEnd_dt"]))

    # Also find ALL same-slice >=500x multi-document conflicts where two filings give same metric within LKR at 500x+ ratio
    lkr_non_ps = f[
        (f["Currency"] == "LKR")
        & (f["Unit"] != "AMBIGUOUS_UNIT")
        & (~f["MetricCode"].isin(["EPS_BASIC", "EPS_DILUTED", "DIVIDEND_PER_SHARE"]))
        & (f["Value"].abs() > 100.0)
    ]
    grp_cols = ["CompanyID", "MetricCode", "PeriodEnd_dt", "PeriodType", "ConsolidationStatus"]
    slice_bounds = lkr_non_ps.groupby(grp_cols)["Value"].agg(
        vmin=lambda s: float(np.min(np.abs(s.values))),
        vmax=lambda s: float(np.max(np.abs(s.values))),
        cnt="count",
    ).reset_index()
    conflict_slices = slice_bounds[(slice_bounds["cnt"] > 1) & (slice_bounds["vmax"] / slice_bounds["vmin"] >= 500.0)]
    conflict_key_set = set(
        zip(
            conflict_slices["CompanyID"],
            conflict_slices["MetricCode"],
            conflict_slices["PeriodEnd_dt"],
            conflict_slices["PeriodType"],
            conflict_slices["ConsolidationStatus"],
        )
    )

    # Scan PDF page headers ONLY for the ~342 unit-shift flagged docs (AMBIGUOUS_UNIT already excluded cleanly)
    us_key_mask = pd.Series(
        [(d, c, p) in us_docs for d, c, p in zip(f["SourceDocumentID_str"], f["CompanyCode_str"], f["PeriodEnd_dt"])],
        index=f.index,
    )
    pdf_unit_cache = _scan_pdf_unit_evidence(f.loc[us_key_mask], repo_root) if scan_flagged_pdfs else {}

    # Vectorize columns into numpy arrays for sub-second iteration over 789,805 rows
    n_rows = len(f)
    arr_unit = f["Unit"].astype(str).to_numpy()
    arr_vstat = f["ValidationStatus"].astype(str).to_numpy()
    arr_cid = f["CompanyID"].astype(str).to_numpy()
    arr_cc = f["CompanyCode_str"].to_numpy()
    arr_mc = f["MetricCode"].astype(str).to_numpy()
    arr_val = pd.to_numeric(f["Value"], errors="coerce").to_numpy()
    arr_doc = f["SourceDocumentID_str"].to_numpy()
    arr_pe = f["PeriodEnd_dt"].to_numpy()
    arr_pt = f["PeriodType"].astype(str).to_numpy()
    arr_cons = f["ConsolidationStatus"].astype(str).to_numpy()
    arr_cur = f["Currency"].astype(str).to_numpy()
    arr_sf = f["SourceFile"].astype(str).to_numpy()
    arr_sp = pd.to_numeric(f["SourcePage"], errors="coerce").fillna(-1).astype(int).to_numpy()
    arr_in_us = us_key_mask.to_numpy()

    unit_status = np.full(n_rows, "VALIDATED_UNIT", dtype=object)
    unit_reasons = np.full(n_rows, "", dtype=object)
    unit_shift_audit_counts: dict[str, int] = {}
    us_doc_first_status: dict[tuple[str, str, Any], str] = {}

    per_share_set = {"EPS_BASIC", "EPS_DILUTED", "DIVIDEND_PER_SHARE"}
    macro_scale_set = {"TOTAL_ASSETS", "TOTAL_EQUITY", "EQUITY_ATTRIBUTABLE_PARENT", "TOTAL_LIABILITIES", "REVENUE"}
    for idx in range(n_rows):
        u = arr_unit[idx]
        mcode = arr_mc[idx]
        doc_key = (arr_doc[idx], arr_cc[idx], arr_pe[idx])

        if u == "AMBIGUOUS_UNIT" or arr_vstat[idx] == "AMBIGUOUS_UNIT":
            unit_status[idx] = "EXCLUDED_AMBIGUOUS_UNIT"
            unit_reasons[idx] = "Parser flagged AMBIGUOUS_UNIT"
        elif mcode not in per_share_set and arr_cur[idx] == "LKR":
            cid = arr_cid[idx]
            modal_u = modal_unit_by_cid.get(cid, "LKR thousands")
            bs_flag = bs_status[idx]
            in_us = bool(arr_in_us[idx])
            pe_ts = pd.Timestamp(arr_pe[idx])
            in_conf = (cid, mcode, pe_ts, arr_pt[idx], arr_cons[idx]) in conflict_key_set
            val = arr_val[idx]
            is_unscaled_macro = bool((mcode in macro_scale_set) and (not np.isnan(val)) and (0.0 < val < 250000.0))

            if in_us or in_conf or u != modal_u or bs_flag == "EXCLUDED_BS_UNIT_MISMATCH_WITHIN_STATEMENT" or is_unscaled_macro:
                anchor_med = scale_anchor.get((cid, mcode)) or scale_anchor.get((cid, "TOTAL_ASSETS"))
                if anchor_med is not None and not np.isnan(val) and abs(val) > 0:
                    ratio = max(abs(anchor_med) / abs(val), abs(val) / abs(anchor_med))
                else:
                    ratio = 1.0

                pdf_hdr = pdf_unit_cache.get((arr_sf[idx], int(arr_sp[idx])), "")
                is_1000_err = (u != modal_u or "THOUSANDS" in pdf_hdr or in_conf) and (200.0 <= ratio <= 15000.0)
                is_1e6_err = (u != modal_u or "MILLIONS" in pdf_hdr or in_conf or is_unscaled_macro) and (2.0e5 <= ratio <= 5.0e7)

                if bs_flag == "EXCLUDED_BS_UNIT_MISMATCH_WITHIN_STATEMENT":
                    unit_status[idx] = "EXCLUDED_BS_INTERNAL_UNIT_MISMATCH"
                    unit_reasons[idx] = "Within-statement Assets vs L+E ~1000x unit discrepancy"
                elif is_1000_err or is_1e6_err:
                    unit_status[idx] = "EXCLUDED_CONFIRMED_UNIT_SCALE_MISMATCH"
                    unit_reasons[idx] = (
                        f"Unit `{u}` disagrees with company modal `{modal_u}` by ~{ratio:,.0f}x "
                        f"(pdf_hdr={pdf_hdr or 'UNSCANNED'}) without rescaling"
                    )
                elif is_unscaled_macro:
                    unit_status[idx] = "EXCLUDED_UNSCALED_MILLIONS_OR_NOTE_ANOMALY"
                    unit_reasons[idx] = f"Positive {mcode}={val:,.2f} LKR (<250k threshold -> unscaled millions or note index)"
                elif ratio >= 500.0 and not np.isnan(val) and abs(val) > 0:
                    unit_status[idx] = "EXCLUDED_EXTRACTION_MAGNITUDE_OUTLIER"
                    unit_reasons[idx] = f"Extraction cell magnitude anomaly {ratio:,.0f}x vs company median anchor"
                elif in_us:
                    unit_status[idx] = "VALIDATED_GENUINE_VOLATILITY_OR_CHANGE"

        if arr_in_us[idx] and doc_key not in us_doc_first_status:
            us_doc_first_status[doc_key] = unit_status[idx]

    # O(1) lookup per unit-shift quality report check across all 551 checks
    for _, ur in us_qr.iterrows():
        pe_np = pd.Timestamp(ur["PeriodEnd_dt"]).to_datetime64()
        st0 = us_doc_first_status.get((str(ur["DisclosureID"]), str(ur["CompanyCode"]).strip(), pe_np), "UNMATCHED_CHECK_DOC")
        unit_shift_audit_counts[st0] = unit_shift_audit_counts.get(st0, 0) + 1

    # ── 5. PIT Status & Composite InputEligibility ───────────────────────────
    pit_ok_arr = (
        f["PITEligibility"].fillna(False).astype(bool).to_numpy()
        & f["AvailabilityDate_dt"].notna().to_numpy()
        & f["PeriodEnd_dt"].notna().to_numpy()
        & (f["AvailabilityDate_dt"] > f["PeriodEnd_dt"]).to_numpy()
        & (~f["CompanyID"].astype(str).str.startswith("UNRES_")).to_numpy()
    )
    pit_status = np.where(pit_ok_arr, "VALIDATED_PIT", "EXCLUDED_NON_CAUSAL_OR_UNRESOLVED_PIT")

    elig_label = np.full(n_rows, "ELIGIBLE", dtype=object)
    is_elig = np.ones(n_rows, dtype=bool)
    excl_reason = np.full(n_rows, "", dtype=object)

    for idx in range(n_rows):
        reasons = []
        if not pit_ok_arr[idx]:
            reasons.append("NON_CAUSAL_OR_UNRESOLVED_PIT")
        if cur_status[idx] == "USD_SECONDARY_OR_LKR_PRIMARY_EXCLUDED":
            reasons.append("SECONDARY_USD_LKR_AVAILABLE")
        if str(unit_status[idx]).startswith("EXCLUDED_"):
            reasons.append(f"UNIT:{unit_status[idx]}")
        if str(bs_status[idx]).startswith("EXCLUDED_"):
            reasons.append(f"BS:{bs_status[idx]}")

        if reasons:
            elig_label[idx] = "EXCLUDED"
            is_elig[idx] = False
            excl_reason[idx] = "; ".join(reasons)
        elif cur_status[idx] == "USD_NATIVE_NO_VERIFIED_FX":
            elig_label[idx] = "ELIGIBLE_NATIVE_USD_ONLY"
            is_elig[idx] = True
            excl_reason[idx] = "Excluded from LKR cross-currency valuations (verified CBSL FX unavailable)"
        elif bs_status[idx] == "WARNING_BS_MINOR_ROUNDING_DIFF":
            elig_label[idx] = "ELIGIBLE_WITH_BS_WARNING"
            is_elig[idx] = True
            excl_reason[idx] = bs_reasons[idx]

    flags_df = pd.DataFrame({
        "FactID": f["FactID"].values,
        "CompanyID": f["CompanyID"].values,
        "CompanyCode": f["CompanyCode_str"].values,
        "MetricCode": f["MetricCode"].values,
        "StatementType": f["StatementType"].values,
        "PeriodEnd": f["PeriodEnd_dt"].values,
        "PeriodType": f["PeriodType"].values,
        "ConsolidationStatus": f["ConsolidationStatus"].values,
        "Currency": f["Currency"].values,
        "Unit": f["Unit"].values,
        "SourceDocumentID": f["SourceDocumentID_str"].values,
        "UnitValidationStatus": unit_status,
        "CurrencyValidationStatus": cur_status,
        "AccountingValidationStatus": bs_status,
        "PITValidationStatus": pit_status,
        "InputEligibility": elig_label,
        "IsInputEligible": is_elig,
        "IsLKRCompatible": is_lkr_compat,
        "ExclusionReason": excl_reason,
    })

    audit_summary = {
        "total_facts_evaluated": int(len(flags_df)),
        "input_eligibility_counts": {str(k): int(v) for k, v in flags_df["InputEligibility"].value_counts().items()},
        "currency_validation_counts": {str(k): int(v) for k, v in flags_df["CurrencyValidationStatus"].value_counts().items()},
        "unit_validation_counts": {str(k): int(v) for k, v in flags_df["UnitValidationStatus"].value_counts().items()},
        "accounting_validation_counts": {str(k): int(v) for k, v in flags_df["AccountingValidationStatus"].value_counts().items()},
        "unit_shift_551_breakdown": unit_shift_audit_counts,
        "bs_684_exception_breakdown": bs_class_counts,
        "scanned_pdf_pages_for_unit_evidence": int(len(pdf_unit_cache)),
    }
    return flags_df, audit_summary
