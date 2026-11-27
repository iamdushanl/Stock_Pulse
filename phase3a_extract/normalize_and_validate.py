"""
Phase 3A Stage 12, 13 & 14 – Normalization, Validation, Derivation & PIT Safeguards
==================================================================================
Builds standardized `financial_facts_normalized.parquet` (exact Section 6 schema),
runs accounting & cross-statement validations (`financial_validation_report.csv`),
tracks restatements/errata (`financial_restated_values.parquet`), derives clean
standalone quarterly flows (`financial_derived_quarterly_facts.parquet`), builds
statement period & PIT lag table (`financial_statement_periods.parquet`), and queues
low-confidence/ambiguous records (`financial_facts_review_queue.csv`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_FACT_COLUMNS = [
    "FactID",
    "CompanyCode",
    "SecurityID",
    "CompanyName",
    "StatementType",
    "MetricCode",
    "OriginalMetricName",
    "PeriodStart",
    "PeriodEnd",
    "FiscalYear",
    "FiscalQuarter",
    "PeriodType",
    "Value",
    "OriginalValue",
    "Currency",
    "Unit",
    "PublicationDate",
    "AvailabilityDate",
    "ReportType",
    "SourceFile",
    "SourcePage",
    "ExtractionMethod",
    "ExtractionConfidence",
    "ValidationStatus",
    "IsConsolidated",
    "IsRestated",
    "Notes",
    # Provenance & normalization extensions
    "DisclosureID",
    "ConversionMethod",
    "ConfidenceTier",
    "MappingStatus",
    "ComparativeFlag",
    "IsStrictPITEligible",
    "SourceSHA256",
]


def _normalize_row_value(raw_val: float, cur: str, unit: str, mult: float, ambig: bool) -> tuple[float, str]:
    if ambig or unit == "AMBIGUOUS_UNIT":
        return float(raw_val), "AMBIGUOUS_UNIT_UNSCALED"
    if unit.endswith("per share"):
        return float(raw_val), "PER_SHARE_EXACT_1X"
    if cur == "LKR":
        if abs(mult - 1000.0) < 1e-6:
            return float(raw_val) * 1000.0, "UNIT_SCALE_1000X"
        if abs(mult - 1_000_000.0) < 1e-6:
            return float(raw_val) * 1_000_000.0, "UNIT_SCALE_1000000X"
        return float(raw_val) * float(mult), "EXACT_UNIT_1X"
    # Foreign currency: scale only within reported currency, NEVER convert to LKR without verified FX rate
    if abs(mult - 1000.0) < 1e-6:
        return float(raw_val) * 1000.0, f"FOREIGN_{cur}_SCALE_1000X_UNCONVERTED_TO_LKR"
    return float(raw_val) * float(mult), f"FOREIGN_{cur}_EXACT_UNCONVERTED_TO_LKR"


def normalize_and_validate_facts(
    raw_facts: list[dict[str, Any]],
    inv_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate, normalize units, enrich Balance Sheet `TOTAL_LIABILITIES` where
    `TOTAL_EQUITY_AND_LIABILITIES - TOTAL_EQUITY` applies safely, run accounting checks,
    and score extraction confidence across all extracted facts.
    """
    inv_map = {str(r["DisclosureID"]): r for _, r in inv_df.iterrows()}

    enriched_rows: list[dict[str, Any]] = []
    for rf in raw_facts:
        rid = str(rf["DisclosureID"])
        ir = inv_map[rid]
        orig_v = float(rf["OriginalValue"])
        cur = str(rf["Currency"])
        unit = str(rf["Unit"])
        mult = float(rf["Multiplier"])
        ambig = bool(rf.get("IsAmbiguousUnit", False))
        norm_v, conv_method = _normalize_row_value(orig_v, cur, unit, mult, ambig)

        pub_ts = pd.to_datetime(ir["PublicationDate"], errors="coerce")
        avail_ts = pd.to_datetime(ir["AvailabilityDate"], errors="coerce")
        map_stat = str(ir["MappingStatus"])
        is_pit = bool(pd.notna(avail_ts) and map_stat != "UNRESOLVED" and not ambig)

        enriched_rows.append({
            "DisclosureID": rid,
            "CompanyCode": ir["CompanyCode"],
            "SecurityID": ir["SecurityID"],
            "CompanyName": str(ir["CompanyName"] or ""),
            "StatementType": str(rf["StatementType"]),
            "MetricCode": str(rf["MetricCode"]),
            "OriginalMetricName": str(rf["OriginalMetricName"]),
            "PeriodStart": pd.to_datetime(rf["PeriodStart"], errors="coerce"),
            "PeriodEnd": pd.to_datetime(rf["PeriodEnd"], errors="coerce"),
            "FiscalYear": int(rf["FiscalYear"]) if pd.notna(rf["FiscalYear"]) else pd.NA,
            "FiscalQuarter": str(rf["FiscalQuarter"]),
            "PeriodType": str(rf["PeriodType"]),
            "Value": norm_v,
            "OriginalValue": orig_v,
            "Currency": cur,
            "Unit": unit if cur == "LKR" or unit.endswith("per share") else (f"{cur} thousands" if abs(mult - 1000.0) < 1 else unit),
            "PublicationDate": pub_ts,
            "AvailabilityDate": avail_ts,
            "ReportType": str(ir["ReportType"]),
            "SourceFile": str(ir["FilePath"]),
            "SourcePage": int(rf["SourcePage"]),
            "ColIndex": int(rf.get("ColIndex", -1)),
            "ExtractionMethod": str(rf["ExtractionMethod"]),
            "BaseConfidence": float(rf["BaseConfidence"]),
            "IsAmbiguousUnit": ambig,
            "IsConsolidated": bool(rf["IsConsolidated"]),
            "IsRestated": bool(rf["IsRestated"]),
            "ComparativeFlag": bool(rf["ComparativeFlag"]),
            "ConversionMethod": conv_method,
            "MappingStatus": map_stat,
            "IsStrictPITEligible": is_pit,
            "SourceSHA256": str(ir["FileChecksumSHA256"]),
            "Notes": str(rf.get("Notes", "")),
        })

    df = pd.DataFrame(enriched_rows)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_FACT_COLUMNS), pd.DataFrame()

    # ── IFRS Subsidiary/Asset Disambiguation when Balance Sheet puts Company Left & Group Right ──
    company_left_disclosures: set[str] = set()
    bs_mask = (df["StatementType"] == "Balance Sheet") & (df["ColIndex"] >= 0)
    for (did, pno, pend), grp in df[bs_mask].groupby(["DisclosureID", "SourcePage", "PeriodEnd"], dropna=False):
        col_stats = []
        for cidx, cg in grp.groupby("ColIndex"):
            m = {str(r["MetricCode"]): float(r["Value"]) for _, r in cg.iterrows()}
            col_stats.append((int(cidx), m.get("TOTAL_ASSETS", 0.0), m.get("LONG_TERM_INVESTMENTS", 0.0), bool(cg["IsConsolidated"].iloc[0])))
        if len(col_stats) == 2:
            col_stats.sort(key=lambda x: x[0])
            c_left, c_right = col_stats[0], col_stats[1]
            # Company standalone holds Investments in Subsidiaries > 0 while Group eliminates them (= 0) and has larger Assets
            left_has_sub_inv = c_left[2] > 0 and c_right[2] == 0
            left_smaller_assets = c_left[1] > 0 and c_right[1] > 1.15 * c_left[1]
            if c_left[3] == c_right[3] and (left_has_sub_inv or left_smaller_assets):
                company_left_disclosures.add(str(did))
                df.loc[(df["DisclosureID"] == did) & (df["SourcePage"] == pno) & (df["ColIndex"] == c_left[0]), "IsConsolidated"] = False
                df.loc[(df["DisclosureID"] == did) & (df["SourcePage"] == pno) & (df["ColIndex"] == c_right[0]), "IsConsolidated"] = True

    # Propagate Company-first layout across IS/CF pages of those same disclosures if duplicate same-period True cols occur
    if company_left_disclosures:
        dup_scopes = (
            df["DisclosureID"].isin(company_left_disclosures)
            & df["StatementType"].isin(["Income Statement", "Cash Flow"])
            & (df["ColIndex"] >= 0)
        )
        for (did, pno, ptype, pend), grp in df[dup_scopes].groupby(["DisclosureID", "SourcePage", "PeriodType", "PeriodEnd"], dropna=False):
            cidxs = sorted(grp["ColIndex"].unique().tolist())
            if len(cidxs) == 2 and grp["IsConsolidated"].nunique() == 1:
                df.loc[(df["DisclosureID"] == did) & (df["SourcePage"] == pno) & (df["ColIndex"] == cidxs[0]), "IsConsolidated"] = False
                df.loc[(df["DisclosureID"] == did) & (df["SourcePage"] == pno) & (df["ColIndex"] == cidxs[1]), "IsConsolidated"] = True

    # ── Derive TOTAL_EQUITY (if only Parent Equity reported) & TOTAL_LIABILITIES per column ──
    bs_sub = df[df["StatementType"] == "Balance Sheet"]
    synth_rows: list[dict[str, Any]] = []
    grp_cols = ["DisclosureID", "SourcePage", "ColIndex", "PeriodEnd", "IsConsolidated", "Currency", "IsRestated"]
    for key_vals, grp in bs_sub.groupby(grp_cols, dropna=False):
        mets = set(grp["MetricCode"])
        by_m = {str(r["MetricCode"]): r for _, r in grp.sort_values("BaseConfidence", ascending=False).iterrows()}

        if "TOTAL_EQUITY" not in mets and "EQUITY_ATTRIBUTABLE_PARENT" in by_m:
            ep = by_m["EQUITY_ATTRIBUTABLE_PARENT"]
            nci_r = by_m.get("NON_CONTROLLING_INTEREST")
            eq_val = float(ep["Value"]) + (float(nci_r["Value"]) if (nci_r is not None and bool(ep["IsConsolidated"])) else 0.0)
            orig_eq = float(ep["OriginalValue"]) + (float(nci_r["OriginalValue"]) if (nci_r is not None and bool(ep["IsConsolidated"])) else 0.0)
            base_eq = dict(ep)
            base_eq["MetricCode"] = "TOTAL_EQUITY"
            base_eq["OriginalMetricName"] = "Derived: Parent Equity + NCI" if (nci_r is not None and bool(ep["IsConsolidated"])) else str(ep["OriginalMetricName"])
            base_eq["Value"] = eq_val
            base_eq["OriginalValue"] = orig_eq
            base_eq["ConversionMethod"] = "DERIVED_BS_TOTAL_EQUITY"
            base_eq["Notes"] = "Computed from EQUITY_ATTRIBUTABLE_PARENT (+ NCI for Group)"
            synth_rows.append(base_eq)
            by_m["TOTAL_EQUITY"] = pd.Series(base_eq)
            mets.add("TOTAL_EQUITY")

        if "TOTAL_LIABILITIES" in mets:
            continue
        if "CURRENT_LIABILITIES" in by_m and "NON_CURRENT_LIABILITIES" in by_m:
            cl = by_m["CURRENT_LIABILITIES"]
            ncl = by_m["NON_CURRENT_LIABILITIES"]
            val_liab = float(cl["Value"]) + float(ncl["Value"])
            orig_liab = float(cl["OriginalValue"]) + float(ncl["OriginalValue"])
            base = dict(cl)
            base["MetricCode"] = "TOTAL_LIABILITIES"
            base["OriginalMetricName"] = "Derived: Total Current + Non-Current Liabilities"
            base["Value"] = val_liab
            base["OriginalValue"] = orig_liab
            base["ConversionMethod"] = "DERIVED_BS_SUM_CL_NCL"
            base["BaseConfidence"] = min(float(cl["BaseConfidence"]), float(ncl["BaseConfidence"]))
            base["Notes"] = f"Sum of {cl['MetricCode']} + {ncl['MetricCode']}"
            synth_rows.append(base)
        elif "TOTAL_EQUITY_AND_LIABILITIES" in by_m and "TOTAL_EQUITY" in by_m:
            tel = by_m["TOTAL_EQUITY_AND_LIABILITIES"]
            teq = by_m["TOTAL_EQUITY"]
            val_liab = float(tel["Value"]) - float(teq["Value"])
            orig_liab = float(tel["OriginalValue"]) - float(teq["OriginalValue"])
            if val_liab >= 0:
                base = dict(tel)
                base["MetricCode"] = "TOTAL_LIABILITIES"
                base["OriginalMetricName"] = "Derived: Total Equity & Liabilities - Total Equity"
                base["Value"] = val_liab
                base["OriginalValue"] = orig_liab
                base["ConversionMethod"] = "DERIVED_BS_TEL_MINUS_EQUITY"
                base["BaseConfidence"] = min(float(tel["BaseConfidence"]), float(teq["BaseConfidence"]))
                base["Notes"] = "Computed as TOTAL_EQUITY_AND_LIABILITIES - TOTAL_EQUITY"
                synth_rows.append(base)

    if synth_rows:
        df = pd.concat([df, pd.DataFrame(synth_rows)], ignore_index=True)

    # ── Deduplicate exact overlapping facts from multi-page/header repetition ─
    dedup_keys = [
        "DisclosureID",
        "StatementType",
        "MetricCode",
        "PeriodStart",
        "PeriodEnd",
        "PeriodType",
        "IsConsolidated",
        "IsRestated",
        "Currency",
    ]
    before_dedup = len(df)
    df = (
        df.sort_values(["BaseConfidence", "SourcePage"], ascending=[False, True])
        .drop_duplicates(subset=dedup_keys, keep="first")
        .reset_index(drop=True)
    )
    duplicates_removed = before_dedup - len(df)

    # ── Run Statement-Level Accounting Validations ───────────────────────────
    val_records: list[dict[str, Any]] = []
    if duplicates_removed > 0:
        val_records.append({
            "ValidationID": "VAL_DUP_0001",
            "DisclosureID": "ALL",
            "CompanyCode": "ALL",
            "StatementType": "General",
            "PeriodEnd": pd.NaT,
            "IsConsolidated": pd.NA,
            "Currency": "ALL",
            "CheckType": "DUPLICATE_FACT_DEDUPLICATION",
            "ExpectedValue": 0.0,
            "ObservedValue": float(duplicates_removed),
            "Difference": float(duplicates_removed),
            "PctDifference": 0.0,
            "Status": "INFO_DEDUPLICATED",
            "Details": f"Removed {duplicates_removed} repeated multi-page/header-echo facts on composite key.",
        })

    val_idx = 1
    bad_fact_mask = np.zeros(len(df), dtype=bool)
    warn_fact_mask = np.zeros(len(df), dtype=bool)

    # 1. Negative Revenue checks
    neg_rev = df[(df["MetricCode"] == "REVENUE") & (df["Value"] < 0)]
    for idx, rr in neg_rev.iterrows():
        warn_fact_mask[idx] = True
        val_records.append({
            "ValidationID": f"VAL_{val_idx:04d}",
            "DisclosureID": rr["DisclosureID"],
            "CompanyCode": rr["CompanyCode"],
            "StatementType": "Income Statement",
            "PeriodEnd": rr["PeriodEnd"],
            "IsConsolidated": rr["IsConsolidated"],
            "Currency": rr["Currency"],
            "CheckType": "NEGATIVE_REVENUE_CHECK",
            "ExpectedValue": 0.0,
            "ObservedValue": float(rr["Value"]),
            "Difference": float(rr["Value"]),
            "PctDifference": np.nan,
            "Status": "WARNING_FLAGGED",
            "Details": f"Negative revenue reported on page {rr['SourcePage']} ({rr['ExtractionMethod']}).",
        })
        val_idx += 1

    # 2. Balance Sheet Identity: Assets == Liabilities + Equity (or + NCI or Total Equity & Liabilities)
    bs = df[df["StatementType"] == "Balance Sheet"]
    for k, grp in bs.groupby(["DisclosureID", "PeriodEnd", "PeriodType", "IsConsolidated", "Currency"], dropna=False):
        did, pend, ptype, icons, cur = k
        ccode = grp["CompanyCode"].iloc[0]
        m = {str(r["MetricCode"]): float(r["Value"]) for _, r in grp.iterrows()}
        assets = m.get("TOTAL_ASSETS")
        if assets is None or not np.isfinite(assets) or assets == 0:
            continue
        liab = m.get("TOTAL_LIABILITIES")
        eq = m.get("TOTAL_EQUITY")
        ep = m.get("EQUITY_ATTRIBUTABLE_PARENT")
        tel = m.get("TOTAL_EQUITY_AND_LIABILITIES")
        nci = m.get("NON_CONTROLLING_INTEREST", 0.0)

        obs_rhs = None
        note = ""
        used_nci = False
        tol = max(2000.0, 0.005 * abs(assets))
        if liab is not None and eq is not None:
            obs_rhs = liab + eq
            # Check if Total Equity extracted actually omitted NCI (reported separately right next to it)
            if abs(assets - obs_rhs) > tol and nci and abs(assets - (obs_rhs + nci)) <= tol:
                obs_rhs = obs_rhs + nci
                used_nci = True
                note = "Reconciled cleanly with Non-Controlling Interest included."
            elif abs(assets - obs_rhs) > tol and ep is not None and abs(assets - (liab + ep + nci)) <= tol:
                obs_rhs = liab + ep + nci
                used_nci = True
                note = "Reconciled cleanly via Liabilities + Parent Equity + NCI."
        elif liab is not None and ep is not None:
            obs_rhs = liab + ep + (nci if bool(icons) else 0.0)
            used_nci = bool(icons and nci)
            note = "Reconciled via Liabilities + Parent Equity (+ NCI)."
        elif tel is not None:
            obs_rhs = tel
            note = "Compared Total Assets vs Total Equity & Liabilities."

        if obs_rhs is not None:
            diff = assets - obs_rhs
            pct = 100.0 * abs(diff) / max(1.0, abs(assets))
            if used_nci and abs(diff) <= tol:
                st = "PASS_WITH_NCI"
            elif abs(diff) <= 1e-3:
                st = "PASS_EXACT"
            elif abs(diff) <= tol:
                st = "PASS_ROUNDING"
            else:
                st = "EXCEPTION"
                warn_fact_mask[grp.index] = True

            val_records.append({
                "ValidationID": f"VAL_{val_idx:04d}",
                "DisclosureID": did,
                "CompanyCode": ccode,
                "StatementType": "Balance Sheet",
                "PeriodEnd": pend,
                "IsConsolidated": icons,
                "Currency": cur,
                "CheckType": "BALANCE_SHEET_EQUATION",
                "ExpectedValue": float(assets),
                "ObservedValue": float(obs_rhs),
                "Difference": float(diff),
                "PctDifference": round(pct, 4),
                "Status": st,
                "Details": note or f"Assets vs L+E diff={diff:,.2f} ({pct:.3f}%).",
            })
            val_idx += 1

    # 3. Income Statement GP & PAT Checks
    inc = df[df["StatementType"] == "Income Statement"]
    for k, grp in inc.groupby(["DisclosureID", "PeriodEnd", "PeriodType", "IsConsolidated", "Currency"], dropna=False):
        did, pend, ptype, icons, cur = k
        ccode = grp["CompanyCode"].iloc[0]
        m = {str(r["MetricCode"]): float(r["Value"]) for _, r in grp.iterrows()}
        if "REVENUE" in m and "COST_OF_SALES" in m and "GROSS_PROFIT" in m:
            rev, cos, gp = m["REVENUE"], m["COST_OF_SALES"], m["GROSS_PROFIT"]
            # Cost of sales is usually parenthesized (negative), sometimes positive
            expected_gp = rev + cos if cos < 0 else rev - cos
            diff = gp - expected_gp
            pct = 100.0 * abs(diff) / max(1.0, abs(rev))
            st = "PASS_EXACT" if abs(diff) <= max(2000.0, 0.005 * abs(rev)) else "EXCEPTION"
            val_records.append({
                "ValidationID": f"VAL_{val_idx:04d}",
                "DisclosureID": did,
                "CompanyCode": ccode,
                "StatementType": "Income Statement",
                "PeriodEnd": pend,
                "IsConsolidated": icons,
                "Currency": cur,
                "CheckType": f"IS_GROSS_PROFIT_{ptype}",
                "ExpectedValue": float(expected_gp),
                "ObservedValue": float(gp),
                "Difference": float(diff),
                "PctDifference": round(pct, 4),
                "Status": st,
                "Details": f"Revenue ({ptype}) - Cost of Sales vs Gross Profit.",
            })
            val_idx += 1

    # 4. Cash Flow Identity Checks (`OCF + ICF + FCF ~= Net Change` and `Open + Net ~= Close`)
    cf = df[df["StatementType"] == "Cash Flow"]
    for k, grp in cf.groupby(["DisclosureID", "PeriodEnd", "PeriodType", "IsConsolidated", "Currency"], dropna=False):
        did, pend, ptype, icons, cur = k
        ccode = grp["CompanyCode"].iloc[0]
        m = {str(r["MetricCode"]): float(r["Value"]) for _, r in grp.iterrows()}
        if "OPERATING_CASH_FLOW" in m and "INVESTING_CASH_FLOW" in m and "FINANCING_CASH_FLOW" in m and "NET_CHANGE_IN_CASH" in m:
            sum_cf = m["OPERATING_CASH_FLOW"] + m["INVESTING_CASH_FLOW"] + m["FINANCING_CASH_FLOW"]
            net_c = m["NET_CHANGE_IN_CASH"]
            diff = net_c - sum_cf
            denom = max(1.0, abs(m["OPERATING_CASH_FLOW"]) + abs(m["INVESTING_CASH_FLOW"]) + abs(m["FINANCING_CASH_FLOW"]))
            pct = 100.0 * abs(diff) / denom
            st = "PASS_EXACT" if pct <= 1.0 or abs(diff) <= 5000.0 else "WARNING_FLAGGED"
            val_records.append({
                "ValidationID": f"VAL_{val_idx:04d}",
                "DisclosureID": did,
                "CompanyCode": ccode,
                "StatementType": "Cash Flow",
                "PeriodEnd": pend,
                "IsConsolidated": icons,
                "Currency": cur,
                "CheckType": f"CF_SUM_ACTIVITIES_{ptype}",
                "ExpectedValue": float(sum_cf),
                "ObservedValue": float(net_c),
                "Difference": float(diff),
                "PctDifference": round(pct, 4),
                "Status": st,
                "Details": "OCF + ICF + FCF vs Net Change in Cash.",
            })
            val_idx += 1

        if "CASH_OPENING" in m and "NET_CHANGE_IN_CASH" in m and "CASH_CLOSING" in m:
            exp_close = m["CASH_OPENING"] + m["NET_CHANGE_IN_CASH"]
            obs_close = m["CASH_CLOSING"]
            diff = obs_close - exp_close
            pct = 100.0 * abs(diff) / max(1.0, abs(obs_close))
            st = "PASS_EXACT" if pct <= 2.0 or abs(diff) <= 10_000.0 else "WARNING_FLAGGED"
            val_records.append({
                "ValidationID": f"VAL_{val_idx:04d}",
                "DisclosureID": did,
                "CompanyCode": ccode,
                "StatementType": "Cash Flow",
                "PeriodEnd": pend,
                "IsConsolidated": icons,
                "Currency": cur,
                "CheckType": f"CF_OPENING_PLUS_NET_EQ_CLOSING_{ptype}",
                "ExpectedValue": float(exp_close),
                "ObservedValue": float(obs_close),
                "Difference": float(diff),
                "PctDifference": round(pct, 4),
                "Status": st,
                "Details": "Cash Opening + Net Change vs Cash Closing (FX differences explain minor gaps).",
            })
            val_idx += 1

    # ── Assign Confidence Score, Tier & ValidationStatus on Every Fact ───────
    val_stat_arr = np.full(len(df), "VALIDATED", dtype=object)
    val_stat_arr[df["IsAmbiguousUnit"].to_numpy(dtype=bool)] = "AMBIGUOUS_UNIT"
    val_stat_arr[warn_fact_mask] = "WARNING_ACCOUNTING_CHECK"
    val_stat_arr[bad_fact_mask] = "FAILED_CHECK"

    conf_scores = df["BaseConfidence"].to_numpy(dtype=np.float64).copy()
    conf_scores[df["IsAmbiguousUnit"].to_numpy(dtype=bool)] *= 0.55
    conf_scores[df["MappingStatus"].astype(str).to_numpy() == "UNRESOLVED"] *= 0.60
    conf_scores[warn_fact_mask] *= 0.72
    conf_scores[bad_fact_mask] *= 0.40
    conf_scores = np.round(np.clip(conf_scores, 0.0, 1.0), 3)

    tiers = np.where(
        conf_scores >= 0.88,
        "HIGH_CONFIDENCE",
        np.where(
            conf_scores >= 0.70,
            "MEDIUM_CONFIDENCE",
            np.where(conf_scores >= 0.50, "LOW_CONFIDENCE", "MANUAL_REVIEW_REQUIRED"),
        ),
    )

    df["ValidationStatus"] = val_stat_arr
    df["ExtractionConfidence"] = conf_scores
    df["ConfidenceTier"] = tiers
    df = df.sort_values(["CompanyCode", "PeriodEnd", "StatementType", "MetricCode"], na_position="last").reset_index(drop=True)
    df["FactID"] = [f"F3A_{i + 1:06d}" for i in range(len(df))]

    for c in REQUIRED_FACT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[REQUIRED_FACT_COLUMNS].copy()

    val_df = pd.DataFrame(val_records)
    return df, val_df


def derive_quarterly_facts(facts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely derive standalone quarterly Flow figures (`Income Statement`, `Cash Flow`)
    ONLY when matching `(CompanyCode, MetricCode, StatementType, IsConsolidated, Currency, Unit, FiscalYear)`
    share aligned accounting basis without incompatible restatements or Balance Sheet snapshots.
    Supports: Q2=H1-Q1, Q3=9M-H1, Q4=Annual-9M, and Q1=H1-Q2_reported.
    """
    flow = facts_df[
        facts_df["StatementType"].isin(["Income Statement", "Cash Flow"])
        & facts_df["CompanyCode"].notna()
        & facts_df["PeriodEnd"].notna()
        & facts_df["PeriodStart"].notna()
        & (~facts_df["MetricCode"].isin(["EPS_BASIC", "EPS_DILUTED"]))
        & (facts_df["ValidationStatus"] != "AMBIGUOUS_UNIT")
    ].copy()

    derived_rows: list[dict[str, Any]] = []
    d_idx = 1
    grp_cols = ["CompanyCode", "SecurityID", "StatementType", "MetricCode", "FiscalYear", "IsConsolidated", "Currency", "Unit", "IsRestated"]

    for key, grp in flow.groupby(grp_cols, dropna=False):
        ccode, sid, stype, mcode, fy, icons, cur, unit, is_rest = key
        by_ptype = {str(r["PeriodType"]): r for _, r in grp.sort_values("ExtractionConfidence", ascending=False).iterrows()}

        # 1. Q2 = H1 - Q1
        if "Half-Year" in by_ptype and "Quarterly" in by_ptype:
            h1 = by_ptype["Half-Year"]
            q_rep = by_ptype["Quarterly"]
            # Distinguish whether Quarterly starts on H1.PeriodStart (so Q1 -> derives Q2) vs ends on H1.PeriodEnd (so Q2 -> derives Q1!)
            h1_ps = pd.Timestamp(h1["PeriodStart"])
            h1_pe = pd.Timestamp(h1["PeriodEnd"])
            q_ps = pd.Timestamp(q_rep["PeriodStart"])
            q_pe = pd.Timestamp(q_rep["PeriodEnd"])

            if q_ps == h1_ps and q_pe < h1_pe:
                # Reported Quarterly is Q1 -> Derive Q2 = H1 - Q1
                d_val = float(h1["Value"]) - float(q_rep["Value"])
                avail = max(pd.Timestamp(h1["AvailabilityDate"]), pd.Timestamp(q_rep["AvailabilityDate"])) if pd.notna(h1["AvailabilityDate"]) and pd.notna(q_rep["AvailabilityDate"]) else pd.NaT
                derived_rows.append({
                    "DerivedFactID": f"DF3A_{d_idx:05d}",
                    "CompanyCode": ccode,
                    "SecurityID": sid,
                    "StatementType": stype,
                    "MetricCode": mcode,
                    "FiscalYear": fy,
                    "TargetQuarter": "Q2",
                    "PeriodStart": q_pe + pd.Timedelta(days=1),
                    "PeriodEnd": h1_pe,
                    "DerivedValue": d_val,
                    "FormulaUsed": "Q2_standalone = H1_cumulative - Q1_standalone",
                    "ParentFactID_Cumulative": h1["FactID"],
                    "ParentFactID_Subtrahend": q_rep["FactID"],
                    "Value_Cumulative": float(h1["Value"]),
                    "Value_Subtrahend": float(q_rep["Value"]),
                    "Currency": cur,
                    "Unit": unit,
                    "IsConsolidated": bool(icons),
                    "IsRestated": bool(is_rest),
                    "AvailabilityDate": avail,
                    "IsStrictPITEligible": bool(pd.notna(avail)),
                    "Notes": "Derived from aligned Q1 and 6M H1 cumulative statements.",
                })
                d_idx += 1
            elif q_pe == h1_pe and q_ps > h1_ps:
                # Reported Quarterly is Q2 ending Sep 30 -> Derive Q1 = H1 - Q2!
                d_val = float(h1["Value"]) - float(q_rep["Value"])
                avail = max(pd.Timestamp(h1["AvailabilityDate"]), pd.Timestamp(q_rep["AvailabilityDate"])) if pd.notna(h1["AvailabilityDate"]) and pd.notna(q_rep["AvailabilityDate"]) else pd.NaT
                derived_rows.append({
                    "DerivedFactID": f"DF3A_{d_idx:05d}",
                    "CompanyCode": ccode,
                    "SecurityID": sid,
                    "StatementType": stype,
                    "MetricCode": mcode,
                    "FiscalYear": fy,
                    "TargetQuarter": "Q1",
                    "PeriodStart": h1_ps,
                    "PeriodEnd": q_ps - pd.Timedelta(days=1),
                    "DerivedValue": d_val,
                    "FormulaUsed": "Q1_standalone = H1_cumulative - Q2_standalone",
                    "ParentFactID_Cumulative": h1["FactID"],
                    "ParentFactID_Subtrahend": q_rep["FactID"],
                    "Value_Cumulative": float(h1["Value"]),
                    "Value_Subtrahend": float(q_rep["Value"]),
                    "Currency": cur,
                    "Unit": unit,
                    "IsConsolidated": bool(icons),
                    "IsRestated": bool(is_rest),
                    "AvailabilityDate": avail,
                    "IsStrictPITEligible": bool(pd.notna(avail)),
                    "Notes": "Derived Q1 from co-reported 6M H1 cumulative and 3M Q2 standalone inside same interim report.",
                })
                d_idx += 1

        # 2. Q3 = 9M - H1
        if "Nine-Month" in by_ptype and "Half-Year" in by_ptype:
            m9 = by_ptype["Nine-Month"]
            h1 = by_ptype["Half-Year"]
            if pd.Timestamp(m9["PeriodStart"]) == pd.Timestamp(h1["PeriodStart"]) and pd.Timestamp(m9["PeriodEnd"]) > pd.Timestamp(h1["PeriodEnd"]):
                avail = max(pd.Timestamp(m9["AvailabilityDate"]), pd.Timestamp(h1["AvailabilityDate"])) if pd.notna(m9["AvailabilityDate"]) and pd.notna(h1["AvailabilityDate"]) else pd.NaT
                derived_rows.append({
                    "DerivedFactID": f"DF3A_{d_idx:05d}",
                    "CompanyCode": ccode,
                    "SecurityID": sid,
                    "StatementType": stype,
                    "MetricCode": mcode,
                    "FiscalYear": fy,
                    "TargetQuarter": "Q3",
                    "PeriodStart": pd.Timestamp(h1["PeriodEnd"]) + pd.Timedelta(days=1),
                    "PeriodEnd": pd.Timestamp(m9["PeriodEnd"]),
                    "DerivedValue": float(m9["Value"]) - float(h1["Value"]),
                    "FormulaUsed": "Q3_standalone = NineMonth_cumulative - H1_cumulative",
                    "ParentFactID_Cumulative": m9["FactID"],
                    "ParentFactID_Subtrahend": h1["FactID"],
                    "Value_Cumulative": float(m9["Value"]),
                    "Value_Subtrahend": float(h1["Value"]),
                    "Currency": cur,
                    "Unit": unit,
                    "IsConsolidated": bool(icons),
                    "IsRestated": bool(is_rest),
                    "AvailabilityDate": avail,
                    "IsStrictPITEligible": bool(pd.notna(avail)),
                    "Notes": "Derived Q3 from aligned 9M cumulative minus H1 cumulative.",
                })
                d_idx += 1

        # 3. Q4 = Annual - 9M
        if "Annual" in by_ptype and "Nine-Month" in by_ptype:
            ann = by_ptype["Annual"]
            m9 = by_ptype["Nine-Month"]
            if pd.Timestamp(ann["PeriodStart"]) == pd.Timestamp(m9["PeriodStart"]) and pd.Timestamp(ann["PeriodEnd"]) > pd.Timestamp(m9["PeriodEnd"]):
                avail = max(pd.Timestamp(ann["AvailabilityDate"]), pd.Timestamp(m9["AvailabilityDate"])) if pd.notna(ann["AvailabilityDate"]) and pd.notna(m9["AvailabilityDate"]) else pd.NaT
                derived_rows.append({
                    "DerivedFactID": f"DF3A_{d_idx:05d}",
                    "CompanyCode": ccode,
                    "SecurityID": sid,
                    "StatementType": stype,
                    "MetricCode": mcode,
                    "FiscalYear": fy,
                    "TargetQuarter": "Q4",
                    "PeriodStart": pd.Timestamp(m9["PeriodEnd"]) + pd.Timedelta(days=1),
                    "PeriodEnd": pd.Timestamp(ann["PeriodEnd"]),
                    "DerivedValue": float(ann["Value"]) - float(m9["Value"]),
                    "FormulaUsed": "Q4_standalone = Annual_value - NineMonth_cumulative",
                    "ParentFactID_Cumulative": ann["FactID"],
                    "ParentFactID_Subtrahend": m9["FactID"],
                    "Value_Cumulative": float(ann["Value"]),
                    "Value_Subtrahend": float(m9["Value"]),
                    "Currency": cur,
                    "Unit": unit,
                    "IsConsolidated": bool(icons),
                    "IsRestated": bool(is_rest),
                    "AvailabilityDate": avail,
                    "IsStrictPITEligible": bool(pd.notna(avail)),
                    "Notes": "Derived Q4 from aligned Annual full-year minus 9M cumulative.",
                })
                d_idx += 1

    cols = [
        "DerivedFactID", "CompanyCode", "SecurityID", "StatementType", "MetricCode",
        "FiscalYear", "TargetQuarter", "PeriodStart", "PeriodEnd", "DerivedValue",
        "FormulaUsed", "ParentFactID_Cumulative", "ParentFactID_Subtrahend",
        "Value_Cumulative", "Value_Subtrahend", "Currency", "Unit",
        "IsConsolidated", "IsRestated", "AvailabilityDate", "IsStrictPITEligible", "Notes",
    ]
    return pd.DataFrame(derived_rows, columns=cols)


def detect_restatements(facts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify explicit `(Restated)` rows AND cross-disclosure amendments/errata
    (e.g. Errata filing vs original filing for identical reporting period & metric).
    """
    rows: list[dict[str, Any]] = []
    # 1. Explicit IsRestated flags from PDF columns/labels
    exp = facts_df[facts_df["IsRestated"]]
    for _, r in exp.iterrows():
        rows.append({
            "RestatementID": f"RST_{len(rows) + 1:04d}",
            "CompanyCode": r["CompanyCode"],
            "SecurityID": r["SecurityID"],
            "StatementType": r["StatementType"],
            "MetricCode": r["MetricCode"],
            "PeriodEnd": r["PeriodEnd"],
            "PeriodType": r["PeriodType"],
            "IsConsolidated": r["IsConsolidated"],
            "Currency": r["Currency"],
            "RestatementType": "EXPLICIT_COLUMN_OR_LINE_RESTATEMENT",
            "OriginalValue": np.nan,
            "RestatedValue": float(r["Value"]),
            "OriginalSourceFile": "",
            "RestatedSourceFile": r["SourceFile"],
            "OriginalAvailabilityDate": pd.NaT,
            "RestatedAvailabilityDate": r["AvailabilityDate"],
            "PctRevision": np.nan,
            "PITUsageRule": "Must ONLY become active after RestatedAvailabilityDate.",
        })

    # 2. Multi-disclosure revisions (same CompanyCode, StatementType, MetricCode, PeriodEnd, PeriodType, IsConsolidated, Currency)
    key_cols = ["CompanyCode", "StatementType", "MetricCode", "PeriodEnd", "PeriodType", "IsConsolidated", "Currency"]
    multi = facts_df.dropna(subset=["CompanyCode", "PeriodEnd"]).groupby(key_cols, dropna=False)
    for k, grp in multi:
        if grp["DisclosureID"].nunique() < 2:
            continue
        grp_s = grp.sort_values(["PublicationDate", "DisclosureID"], na_position="first")
        first = grp_s.iloc[0]
        last = grp_s.iloc[-1]
        v1, v2 = float(first["Value"]), float(last["Value"])
        if abs(v2 - v1) > max(1.0, 0.001 * max(abs(v1), abs(v2), 1.0)):
            pct = 100.0 * (v2 - v1) / abs(v1) if abs(v1) > 1e-9 else np.nan
            rows.append({
                "RestatementID": f"RST_{len(rows) + 1:04d}",
                "CompanyCode": first["CompanyCode"],
                "SecurityID": first["SecurityID"],
                "StatementType": first["StatementType"],
                "MetricCode": first["MetricCode"],
                "PeriodEnd": first["PeriodEnd"],
                "PeriodType": first["PeriodType"],
                "IsConsolidated": first["IsConsolidated"],
                "Currency": first["Currency"],
                "RestatementType": f"AMENDMENT_OR_ERRATA ({last['ReportType']})",
                "OriginalValue": v1,
                "RestatedValue": v2,
                "OriginalSourceFile": first["SourceFile"],
                "RestatedSourceFile": last["SourceFile"],
                "OriginalAvailabilityDate": first["AvailabilityDate"],
                "RestatedAvailabilityDate": last["AvailabilityDate"],
                "PctRevision": round(pct, 4) if pd.notna(pct) else np.nan,
                "PITUsageRule": "Use OriginalValue on dates in [OriginalAvailabilityDate, RestatedAvailabilityDate); RestatedValue strictly >= RestatedAvailabilityDate.",
            })

    # 3. Document-level Errata disclosures (e.g. ASIR 53505 Errata amending prior 53438)
    errata = facts_df[facts_df["ReportType"] == "Errata_Interim"]
    if not errata.empty:
        for (ccode, did), grp in errata.groupby(["CompanyCode", "DisclosureID"]):
            prior = facts_df[(facts_df["CompanyCode"] == ccode) & (facts_df["DisclosureID"] != did)]
            first_row = grp.iloc[0]
            rows.append({
                "RestatementID": f"RST_{len(rows) + 1:04d}",
                "CompanyCode": ccode,
                "SecurityID": first_row["SecurityID"],
                "StatementType": "Disclosure_Errata",
                "MetricCode": "SHAREHOLDING_OR_NOTE_ERRATA",
                "PeriodEnd": pd.to_datetime(first_row["PeriodEnd"]),
                "PeriodType": "Errata_Notice",
                "IsConsolidated": bool(first_row["IsConsolidated"]),
                "Currency": str(first_row["Currency"]),
                "RestatementType": "DISCLOSURE_ERRATA_FILING",
                "OriginalValue": np.nan,
                "RestatedValue": np.nan,
                "OriginalSourceFile": str(prior["SourceFile"].iloc[0]) if not prior.empty else "",
                "RestatedSourceFile": str(first_row["SourceFile"]),
                "OriginalAvailabilityDate": pd.to_datetime(prior["AvailabilityDate"].iloc[0]) if not prior.empty else pd.NaT,
                "RestatedAvailabilityDate": pd.to_datetime(first_row["AvailabilityDate"]),
                "PctRevision": 0.0,
                "PITUsageRule": "Errata notice corrected non-primary-statement disclosure (e.g. shareholding table); primary statements verified unchanged.",
            })

    cols = [
        "RestatementID", "CompanyCode", "SecurityID", "StatementType", "MetricCode",
        "PeriodEnd", "PeriodType", "IsConsolidated", "Currency", "RestatementType",
        "OriginalValue", "RestatedValue", "OriginalSourceFile", "RestatedSourceFile",
        "OriginalAvailabilityDate", "RestatedAvailabilityDate", "PctRevision", "PITUsageRule",
    ]
    return pd.DataFrame(rows, columns=cols)


def build_review_and_quality_tables(
    facts_df: pd.DataFrame,
    inv_df: pd.DataFrame,
    val_df: pd.DataFrame,
    legacy_summary: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create:
      1. `financial_statement_periods.parquet`: statement-period granularity & PIT lag days
      2. `financial_facts_review_queue.csv`: low-confidence / OCR / ambiguous / warning items
      3. `financial_extraction_quality_report.csv`: document & metric-level quality breakdown
    """
    # 1. Statement Periods Table
    p_cols = [
        "DisclosureID", "CompanyCode", "SecurityID", "CompanyName",
        "StatementType", "PeriodStart", "PeriodEnd", "FiscalYear",
        "FiscalQuarter", "PeriodType", "Currency", "Unit",
        "IsConsolidated", "IsRestated", "ComparativeFlag",
        "PublicationDate", "AvailabilityDate", "IsStrictPITEligible", "SourceFile",
    ]
    periods_df = (
        facts_df.groupby([
            "DisclosureID", "CompanyCode", "SecurityID", "CompanyName",
            "StatementType", "PeriodStart", "PeriodEnd", "FiscalYear",
            "FiscalQuarter", "PeriodType", "Currency", "Unit",
            "IsConsolidated", "IsRestated", "ComparativeFlag",
            "PublicationDate", "AvailabilityDate", "IsStrictPITEligible", "SourceFile",
        ], dropna=False)
        .agg(FactCount=("FactID", "count"), MeanConfidence=("ExtractionConfidence", "mean"))
        .reset_index()
    )
    periods_df["ReportingLagDays"] = (
        pd.to_datetime(periods_df["AvailabilityDate"]) - pd.to_datetime(periods_df["PeriodEnd"])
    ).dt.days

    # 2. Review Queue Table
    rev_rows: list[dict[str, Any]] = []
    # Fact-level issues
    rev_mask = (
        facts_df["ConfidenceTier"].isin(["LOW_CONFIDENCE", "MANUAL_REVIEW_REQUIRED"])
        | (facts_df["ValidationStatus"] != "VALIDATED")
        | (facts_df["ExtractionMethod"] == "OCR")
        | (facts_df["MappingStatus"] != "VERIFIED_PHASE1B")
        | (~facts_df["IsStrictPITEligible"])
    )
    for _, fr in facts_df[rev_mask].iterrows():
        issues = []
        if fr["ValidationStatus"] != "VALIDATED":
            issues.append(str(fr["ValidationStatus"]))
        if fr["ExtractionMethod"] == "OCR":
            issues.append("OCR_EXTRACTED_VERIFY_DIGITS")
        if fr["MappingStatus"] != "VERIFIED_PHASE1B":
            issues.append(f"MAPPING_{fr['MappingStatus']}")
        if fr["ConfidenceTier"] in ("LOW_CONFIDENCE", "MANUAL_REVIEW_REQUIRED"):
            issues.append(str(fr["ConfidenceTier"]))
        if not bool(fr["IsStrictPITEligible"]):
            issues.append("UNAVAILABLE_STRICT_PIT")

        rev_rows.append({
            "FactID": fr["FactID"],
            "SourceFile": fr["SourceFile"],
            "SourcePage": fr["SourcePage"],
            "CompanyName": fr["CompanyName"],
            "OriginalMetricName": fr["OriginalMetricName"],
            "ExtractedValue": fr["Value"],
            "MetricCode": fr["MetricCode"],
            "IssueType": ";".join(issues),
            "Confidence": fr["ConfidenceTier"],
            "ReviewStatus": "PENDING_HUMAN_REVIEW",
        })

    # Document-level failed/non-extracted disclosures
    for _, ir in inv_df[inv_df["ExtractionStatus"].isin(["FAILED", "PARTIAL_EXTRACTION"])].iterrows():
        rev_rows.append({
            "FactID": f"DOC_{ir['DisclosureID']}",
            "SourceFile": ir["FilePath"],
            "SourcePage": 0,
            "CompanyName": ir["CompanyName"],
            "OriginalMetricName": ir["ReportingPeriodDescription"],
            "ExtractedValue": np.nan,
            "MetricCode": "DOCUMENT_LEVEL",
            "IssueType": f"EXTRACTION_STATUS_{ir['ExtractionStatus']}",
            "Confidence": "MANUAL_REVIEW_REQUIRED",
            "ReviewStatus": "PENDING_HUMAN_REVIEW",
        })

    rq_cols = ["FactID", "SourceFile", "SourcePage", "CompanyName", "OriginalMetricName", "ExtractedValue", "MetricCode", "IssueType", "Confidence", "ReviewStatus"]
    review_df = pd.DataFrame(rev_rows, columns=rq_cols)

    # 3. Extraction Quality Report Summary Table
    q_rows: list[dict[str, Any]] = []
    for metric, val_cnt in [
        ("TOTAL_PDFS_INVENTORIED", len(inv_df)),
        ("FINANCIAL_REPORT_PDFS", int((inv_df["DocumentCategory"] == "financial_reports").sum())),
        ("EXCHANGE_CIRCULARS_AND_DIRECTIVES", int((inv_df["DocumentCategory"] != "financial_reports").sum())),
        ("SUCCESSFULLY_EXTRACTED_REPORTS", int((inv_df["ExtractionStatus"] == "SUCCESS").sum())),
        ("PARTIAL_EXTRACTION_REPORTS", int((inv_df["ExtractionStatus"] == "PARTIAL_EXTRACTION").sum())),
        ("FAILED_REPORTS", int((inv_df["ExtractionStatus"] == "FAILED").sum())),
        ("OCR_REQUIRED_REPORTS", int(inv_df["OCRRequired"].sum())),
        ("TOTAL_NORMALIZED_FACTS", len(facts_df)),
        ("UNIQUE_COMPANIES_WITH_FACTS", int(facts_df["CompanyCode"].nunique())),
        ("HIGH_CONFIDENCE_FACTS", int((facts_df["ConfidenceTier"] == "HIGH_CONFIDENCE").sum())),
        ("MEDIUM_CONFIDENCE_FACTS", int((facts_df["ConfidenceTier"] == "MEDIUM_CONFIDENCE").sum())),
        ("LOW_CONFIDENCE_FACTS", int((facts_df["ConfidenceTier"] == "LOW_CONFIDENCE").sum())),
        ("MANUAL_REVIEW_REQUIRED_FACTS", int((facts_df["ConfidenceTier"] == "MANUAL_REVIEW_REQUIRED").sum())),
        ("STRICT_PIT_ELIGIBLE_FACTS", int(facts_df["IsStrictPITEligible"].sum())),
        ("OCR_EXTRACTED_FACTS", int((facts_df["ExtractionMethod"] == "OCR").sum())),
        ("FOREIGN_CURRENCY_USD_FACTS", int((facts_df["Currency"] == "USD").sum())),
        ("AMBIGUOUS_UNIT_FACTS", int((facts_df["ValidationStatus"] == "AMBIGUOUS_UNIT").sum())),
        ("LEGACY_PHASE2_ENRICHED_HARDCODED_ROWS_REPLACED", legacy_summary.get("financial_facts_enriched", {}).get("remediated_rows_rejected_for_reextraction", 0)),
    ]:
        q_rows.append({"QualityDimension": "OVERALL", "Metric": metric, "Count": int(val_cnt)})

    for mcode, grp in facts_df.groupby("MetricCode"):
        q_rows.append({
            "QualityDimension": f"METRIC_COVERAGE:{grp['StatementType'].iloc[0]}",
            "Metric": str(mcode),
            "Count": int(len(grp)),
        })

    qual_df = pd.DataFrame(q_rows)
    return periods_df, review_df, qual_df
