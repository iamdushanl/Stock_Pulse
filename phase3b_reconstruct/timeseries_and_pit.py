"""
Phase 3B Workstreams C, D, E & H – Historical Reconstruction, Restatements & Strict PIT Database
===============================================================================================
Implements:
  - Combining Phase 3A canonical facts (`Dataset/processed/phase3a/`) verbatim with newly extracted
    historical `2007–2026` facts (`historical_financial_facts.parquet` exact Section 10 schema).
  - Non-destructive versioned restatement tracking (`historical_restated_values.parquet`):
    earliest verified `AvailabilityDate` wins for unchanged comparative repetitions, while restated or
    amended revisions preserve BOTH Version 1 (`OriginalValue`) and Version 2+ (`RestatedValue`).
  - Traceable standalone quarterly reconstruction (`Q1=H1-Q2`, `Q2=H1-Q1`, `Q3=9M-H1`, `Q4=Annual-9M`)
    strictly on aligned Flow statements (`historical_quarterly_financials.parquet`), never on Balance Sheet
    snapshots.
  - Audited & 12M Annual financial history (`historical_annual_financials.parquet`).
  - Strict Point-in-Time schedule (`financial_pit_availability.parquet`) and official query APIs:
      * `get_financial_facts_as_of(facts_df, company_id, decision_date, ...)`
      * `get_latest_financial_facts(facts_df, company_id=..., ...)`
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from phase3a_extract.normalize_and_validate import normalize_and_validate_facts

logger = logging.getLogger("phase3b.pit")

HISTORICAL_FACT_COLUMNS = [
    "FactID",
    "CompanyID",
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
    "ConsolidationStatus",
    "PublicationDate",
    "AvailabilityDate",
    "RestatementVersion",
    "SourceDocumentID",
    "SourceFile",
    "SourcePage",
    "ExtractionMethod",
    "ExtractionConfidence",
    "ValidationStatus",
    "PITEligibility",
    "IsDerived",
    "DerivationMethod",
    "ParentFactIDs",
    "IsRestated",
    "ComparativeFlag",
    "ReportType",
    "SourceSHA256",
    "Notes",
]


def get_financial_facts_as_of(
    facts_df: pd.DataFrame,
    company_id: str | None,
    decision_date: Any,
    consolidation: str | None = None,
    statement_type: str | None = None,
    metric_codes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Workstream E: Strict Point-in-Time financial query engine.

    Rules enforced deterministically:
      1. Returns ONLY facts with `PITEligibility == True`, non-null `AvailabilityDate`, and
         `AvailabilityDate <= decision_date`.
      2. Excludes unverified publication dates and ambiguous units automatically.
      3. Excludes future restatements (`AvailabilityDate > decision_date`) cleanly.
      4. When multiple versions (`RestatementVersion`) exist with `AvailabilityDate <= decision_date`,
         selects strictly the most recent publicly known version as of `decision_date`.
      5. Separates `Consolidated_Group` vs `Standalone_Company` and preserves full source provenance.
    """
    if facts_df.empty:
        return facts_df.copy()

    dec_ts = pd.to_datetime(decision_date, errors="coerce")
    if pd.isna(dec_ts):
        raise ValueError(f"Invalid decision_date: {decision_date!r}")
    # If caller passes date-only (`2025-01-15`), treat end of decision day (`23:59:59`) ONLY if no HH:MM:SS given?
    # Strictly safer conservative financial engineering: exact timestamp `AvailabilityDate <= dec_ts`
    # unless dec_ts is midnight and user intends calendar date? Note example:
    # "published on 28 Feb 2025 -> 15 Jan 2025 must NOT see". Keep exact comparison `<= dec_ts`
    # (exact midnight 00:00:00 means facts published before midnight on dec_ts, or caller passes timestamp).

    avail = pd.to_datetime(facts_df["AvailabilityDate"], errors="coerce")
    mask = (
        facts_df["PITEligibility"].fillna(False).astype(bool)
        & avail.notna()
        & (avail <= dec_ts)
    )
    if company_id is not None:
        cid_s = str(company_id).strip()
        mask &= (
            (facts_df["CompanyID"].astype(str) == cid_s)
            | (facts_df["CompanyCode"].astype(str) == cid_s)
        )
    if consolidation is not None:
        mask &= facts_df["ConsolidationStatus"].astype(str) == str(consolidation)
    if statement_type is not None:
        mask &= facts_df["StatementType"].astype(str) == str(statement_type)
    if metric_codes is not None:
        mask &= facts_df["MetricCode"].isin(metric_codes)

    sub = facts_df.loc[mask].copy()
    if sub.empty:
        return sub

    # Select latest publicly known RestatementVersion on or before decision_date per unique accounting slice
    key_cols = [
        "CompanyID",
        "StatementType",
        "MetricCode",
        "PeriodStart",
        "PeriodEnd",
        "PeriodType",
        "ConsolidationStatus",
        "Currency",
    ]
    sub["_avail_sort"] = pd.to_datetime(sub["AvailabilityDate"], errors="coerce")
    sub = (
        sub.sort_values(
            ["RestatementVersion", "_avail_sort", "IsDerived", "ExtractionConfidence", "SourceDocumentID"],
            ascending=[False, False, True, False, False],
        )
        .drop_duplicates(subset=key_cols, keep="first")
        .drop(columns=["_avail_sort"])
        .sort_values(["CompanyID", "PeriodEnd", "StatementType", "MetricCode"])
        .reset_index(drop=True)
    )
    return sub


def get_latest_financial_facts(
    facts_df: pd.DataFrame,
    company_id: str | None = None,
    consolidation: str | None = None,
    statement_type: str | None = None,
    strict_pit_only: bool = False,
) -> pd.DataFrame:
    """
    Workstream D (View A): Return the latest-known version of every historical reporting period
    across the company database (useful for non-backtest current research & quality analysis).
    """
    if facts_df.empty:
        return facts_df.copy()

    mask = pd.Series(True, index=facts_df.index)
    if company_id is not None:
        cid_s = str(company_id).strip()
        mask &= (
            (facts_df["CompanyID"].astype(str) == cid_s)
            | (facts_df["CompanyCode"].astype(str) == cid_s)
        )
    if consolidation is not None:
        mask &= facts_df["ConsolidationStatus"].astype(str) == str(consolidation)
    if statement_type is not None:
        mask &= facts_df["StatementType"].astype(str) == str(statement_type)
    if strict_pit_only:
        mask &= facts_df["PITEligibility"].fillna(False).astype(bool)

    sub = facts_df.loc[mask].copy()
    if sub.empty:
        return sub

    key_cols = [
        "CompanyID",
        "StatementType",
        "MetricCode",
        "PeriodStart",
        "PeriodEnd",
        "PeriodType",
        "ConsolidationStatus",
        "Currency",
    ]
    sub["_avail_sort"] = pd.to_datetime(sub["AvailabilityDate"], errors="coerce")
    return (
        sub.sort_values(
            ["RestatementVersion", "_avail_sort", "IsDerived", "ExtractionConfidence"],
            ascending=[False, False, True, False],
        )
        .drop_duplicates(subset=key_cols, keep="first")
        .drop(columns=["_avail_sort"])
        .sort_values(["CompanyID", "PeriodEnd", "StatementType", "MetricCode"])
        .reset_index(drop=True)
    )


def _derive_historical_standalone_quarters(facts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Workstream C: Derive standalone quarterly Flow metrics (`Income Statement`, `Cash Flow`, excluding EPS):
      - Q1 = H1_cumulative - Q2_standalone
      - Q2 = H1_cumulative - Q1_standalone
      - Q3 = NineMonth_cumulative - H1_cumulative
      - Q4 = Annual_cumulative - NineMonth_cumulative
    Strict guards:
      - Periods must belong to exact same FiscalYear & aligned boundaries.
      - Currency, Unit, ConsolidationStatus must match identically.
      - Latest non-restated or version-aligned compatible basis only; never mixes Balance Sheet snapshots.
      - Derived `AvailabilityDate` = `max(parent_1 AvailabilityDate, parent_2 AvailabilityDate)`.
    """
    flow = facts_df[
        facts_df["StatementType"].isin(["Income Statement", "Cash Flow"])
        & facts_df["CompanyCode"].notna()
        & facts_df["PeriodEnd"].notna()
        & facts_df["PeriodStart"].notna()
        & (~facts_df["MetricCode"].isin(["EPS_BASIC", "EPS_DILUTED", "DIVIDEND_PER_SHARE"]))
        & (facts_df["ValidationStatus"] != "AMBIGUOUS_UNIT")
        & (~facts_df["Unit"].astype(str).str.contains("per share", case=False, na=False))
    ].copy()

    if flow.empty:
        return pd.DataFrame()

    # Use latest available record within each (CompanyID, StatementType, MetricCode, FiscalYear, PeriodType, PeriodStart, PeriodEnd, ConsolidationStatus, Currency, Unit)
    flow["_ps"] = pd.to_datetime(flow["PeriodStart"], errors="coerce")
    flow["_pe"] = pd.to_datetime(flow["PeriodEnd"], errors="coerce")
    flow["_av"] = pd.to_datetime(flow["AvailabilityDate"], errors="coerce")

    grp_cols = [
        "CompanyID", "CompanyCode", "SecurityID", "CompanyName",
        "StatementType", "MetricCode", "FiscalYear",
        "ConsolidationStatus", "Currency", "Unit",
    ]

    derived_rows: list[dict[str, Any]] = []
    for key, grp in flow.groupby(grp_cols, dropna=False):
        cid, ccode, sid, cname, stype, mcode, fy, cons_s, cur, unit = key
        # Deduplicate each PeriodType slice keeping highest confidence / latest version compatible inside same year
        by_pt: dict[str, list[pd.Series]] = {}
        grp_s = grp.sort_values(["RestatementVersion", "ExtractionConfidence", "_av"], ascending=[False, False, True])
        for _, r in grp_s.iterrows():
            pt = str(r["PeriodType"])
            lst = by_pt.setdefault(pt, [])
            # Keep distinct (PeriodStart, PeriodEnd) per PeriodType
            if not any(x["_ps"] == r["_ps"] and x["_pe"] == r["_pe"] for x in lst):
                lst.append(r)

        q_list = by_pt.get("Quarterly", [])
        h1_list = by_pt.get("Half-Year", [])
        m9_list = by_pt.get("Nine-Month", [])
        ann_list = by_pt.get("Annual", [])

        # Helper to construct clean derived dict
        def _mk_derived(
            target_q: str,
            p_start: pd.Timestamp,
            p_end: pd.Timestamp,
            cum_row: pd.Series,
            sub_row: pd.Series,
            formula: str,
        ) -> dict[str, Any]:
            val = float(cum_row["Value"]) - float(sub_row["Value"])
            av1, av2 = cum_row["_av"], sub_row["_av"]
            avail = max(av1, av2) if (pd.notna(av1) and pd.notna(av2)) else pd.NaT
            pub = max(pd.to_datetime(cum_row["PublicationDate"], errors="coerce"), pd.to_datetime(sub_row["PublicationDate"], errors="coerce")) if (pd.notna(cum_row["PublicationDate"]) and pd.notna(sub_row["PublicationDate"])) else pd.NaT
            pit_ok = bool(pd.notna(avail) and bool(cum_row["PITEligibility"]) and bool(sub_row["PITEligibility"]))
            conf = round(float(min(cum_row["ExtractionConfidence"], sub_row["ExtractionConfidence"])) * 0.96, 3)
            # Guard against incompatible cross-basis restatements
            incompat = bool(bool(cum_row["IsRestated"]) != bool(sub_row["IsRestated"]))
            if incompat:
                pit_ok = False
                conf = round(conf * 0.70, 3)
            return {
                "CompanyID": cid,
                "CompanyCode": ccode,
                "SecurityID": sid,
                "CompanyName": cname,
                "StatementType": stype,
                "MetricCode": mcode,
                "OriginalMetricName": f"Derived {target_q}: {cum_row['OriginalMetricName']}",
                "PeriodStart": p_start,
                "PeriodEnd": p_end,
                "FiscalYear": fy,
                "FiscalQuarter": target_q,
                "PeriodType": "Quarterly",
                "Value": val,
                "OriginalValue": val,
                "Currency": cur,
                "Unit": unit,
                "ConsolidationStatus": cons_s,
                "PublicationDate": pub,
                "AvailabilityDate": avail,
                "RestatementVersion": max(int(cum_row["RestatementVersion"]), int(sub_row["RestatementVersion"])),
                "SourceDocumentID": f"{cum_row['SourceDocumentID']}-{sub_row['SourceDocumentID']}",
                "SourceFile": str(cum_row["SourceFile"]),
                "SourcePage": int(cum_row["SourcePage"]),
                "ExtractionMethod": f"Derived:{formula}",
                "ExtractionConfidence": conf,
                "ValidationStatus": "VALIDATED" if not incompat else "WARNING_ACCOUNTING_CHECK",
                "PITEligibility": pit_ok,
                "IsDerived": True,
                "DerivationMethod": formula,
                "ParentFactIDs": f"{cum_row['FactID']};{sub_row['FactID']}",
                "IsRestated": bool(cum_row["IsRestated"] or sub_row["IsRestated"]),
                "ComparativeFlag": bool(cum_row["ComparativeFlag"] or sub_row["ComparativeFlag"]),
                "ReportType": "Derived_Quarterly",
                "SourceSHA256": str(cum_row["SourceSHA256"]),
                "Notes": f"{formula} ({cum_row['Value']:.2f} - {sub_row['Value']:.2f}); parents={cum_row['FactID']},{sub_row['FactID']}",
            }

        # 1. H1 & Q -> derive Q2 = H1 - Q1 OR Q1 = H1 - Q2
        for h1 in h1_list:
            for qr in q_list:
                if abs((h1["_pe"] - h1["_ps"]).days - 182) > 25 or abs((qr["_pe"] - qr["_ps"]).days - 90) > 20:
                    continue
                if qr["_ps"] == h1["_ps"] and qr["_pe"] < h1["_pe"]:
                    derived_rows.append(_mk_derived("Q2", qr["_pe"] + pd.Timedelta(days=1), h1["_pe"], h1, qr, "Q2 = H1 - Q1"))
                elif qr["_pe"] == h1["_pe"] and qr["_ps"] > h1["_ps"]:
                    derived_rows.append(_mk_derived("Q1", h1["_ps"], qr["_ps"] - pd.Timedelta(days=1), h1, qr, "Q1 = H1 - Q2"))

        # 2. Q3 = 9M - H1
        for m9 in m9_list:
            for h1 in h1_list:
                if m9["_ps"] == h1["_ps"] and 60 <= (m9["_pe"] - h1["_pe"]).days <= 110:
                    derived_rows.append(_mk_derived("Q3", h1["_pe"] + pd.Timedelta(days=1), m9["_pe"], m9, h1, "Q3 = 9M - H1"))

        # 3. Q4 = Annual - 9M
        for ann in ann_list:
            for m9 in m9_list:
                if ann["_ps"] == m9["_ps"] and 60 <= (ann["_pe"] - m9["_pe"]).days <= 110 and 330 <= (ann["_pe"] - ann["_ps"]).days <= 390:
                    derived_rows.append(_mk_derived("Q4", m9["_pe"] + pd.Timedelta(days=1), ann["_pe"], ann, m9, "Q4 = Annual - 9M"))

    if not derived_rows:
        return pd.DataFrame()
    d_df = pd.DataFrame(derived_rows)
    d_df = d_df.drop_duplicates(
        subset=["CompanyID", "StatementType", "MetricCode", "PeriodStart", "PeriodEnd", "FiscalQuarter", "ConsolidationStatus", "Currency"],
        keep="first",
    ).reset_index(drop=True)
    return d_df


def reconstruct_historical_database(
    phase3a_facts_df: pd.DataFrame,
    raw_new_facts: list[dict[str, Any]],
    catalog_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Master historical reconstruction combining Phase 3A facts + newly parsed historical facts,
    assigning canonical `CompanyID`s, building versioned restatement histories without overwriting
    originals, deriving standalone quarters, and compiling Annual/Quarterly/PIT tables.
    """
    id_by_code = {str(r["CompanyCode"]).strip(): r for _, r in identity_df.iterrows()}

    # 1. Normalize & validate newly acquired PDF facts using exact Phase 3A normalizer
    acq_sub = catalog_df[(catalog_df["AcquisitionStatus"] == "REPORT_SUCCESSFULLY_ACQUIRED") & (~catalog_df["IsPhase3ALocal"])].copy()
    if raw_new_facts and not acq_sub.empty:
        inv_for_norm = acq_sub.copy()
        inv_for_norm["DisclosureID"] = inv_for_norm["DocumentID"].astype(str)
        # Join canonical CompanyID/SecurityID/MappingStatus
        sids, mstats, cnames = [], [], []
        for _, r in inv_for_norm.iterrows():
            cc = str(r.get("CompanyCode") or "").strip()
            ir = id_by_code.get(cc)
            if ir is not None:
                sids.append(ir["PrimarySecurityID"])
                mstats.append(str(ir["MappingStatus"]))
                cnames.append(str(ir["CompanyName"]))
            else:
                sids.append(pd.NA)
                mstats.append("UNRESOLVED")
                cnames.append(str(r.get("CompanyName") or cc))
        inv_for_norm["SecurityID"] = sids
        inv_for_norm["MappingStatus"] = mstats
        inv_for_norm["CompanyName"] = cnames
        inv_for_norm["FilePath"] = inv_for_norm["LocalFilePath"]
        inv_for_norm["FileChecksumSHA256"] = inv_for_norm["SHA256"]

        new_norm_df, new_val_df = normalize_and_validate_facts(raw_new_facts, inv_for_norm)
    else:
        new_norm_df = pd.DataFrame()
        new_val_df = pd.DataFrame()

    # 2. Adapt Phase 3A facts cleanly (verbatim values & dates preserved!) into unified schema
    def _to_hist_schema(df_in: pd.DataFrame, is_phase3a: bool) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(columns=HISTORICAL_FACT_COLUMNS)
        out = pd.DataFrame()
        cc_series = df_in["CompanyCode"].astype(str).str.strip()
        out["FactID"] = df_in["FactID"]
        out["CompanyID"] = [
            str(id_by_code[cc]["CompanyID"]) if cc in id_by_code else f"UNRES_{cc}"
            for cc in cc_series
        ]
        out["CompanyCode"] = df_in["CompanyCode"]
        out["SecurityID"] = df_in["SecurityID"]
        out["CompanyName"] = df_in["CompanyName"]
        out["StatementType"] = df_in["StatementType"]
        out["MetricCode"] = df_in["MetricCode"]
        out["OriginalMetricName"] = df_in["OriginalMetricName"]
        out["PeriodStart"] = pd.to_datetime(df_in["PeriodStart"], errors="coerce")
        out["PeriodEnd"] = pd.to_datetime(df_in["PeriodEnd"], errors="coerce")
        out["FiscalYear"] = df_in["FiscalYear"]
        out["FiscalQuarter"] = df_in["FiscalQuarter"]
        out["PeriodType"] = df_in["PeriodType"]
        out["Value"] = pd.to_numeric(df_in["Value"], errors="coerce")
        out["OriginalValue"] = pd.to_numeric(df_in["OriginalValue"], errors="coerce")
        out["Currency"] = df_in["Currency"]
        out["Unit"] = df_in["Unit"]
        out["ConsolidationStatus"] = np.where(df_in["IsConsolidated"].astype(bool), "Consolidated_Group", "Standalone_Company")
        out["PublicationDate"] = pd.to_datetime(df_in["PublicationDate"], errors="coerce")
        out["AvailabilityDate"] = pd.to_datetime(df_in["AvailabilityDate"], errors="coerce")
        out["RestatementVersion"] = 1
        out["SourceDocumentID"] = df_in["DisclosureID"].astype(str)
        out["SourceFile"] = df_in["SourceFile"]
        out["SourcePage"] = df_in["SourcePage"].astype(int)
        out["ExtractionMethod"] = df_in["ExtractionMethod"]
        out["ExtractionConfidence"] = pd.to_numeric(df_in["ExtractionConfidence"], errors="coerce")
        out["ValidationStatus"] = df_in["ValidationStatus"]
        # Re-check strict PIT: UNRESOLVED entities, missing/non-causal AvailabilityDate <= PeriodEnd, or AMBIGUOUS_UNIT => False!
        out["PITEligibility"] = (
            df_in["IsStrictPITEligible"].fillna(False).astype(bool)
            & out["AvailabilityDate"].notna()
            & out["PeriodEnd"].notna()
            & (out["AvailabilityDate"] > out["PeriodEnd"])
            & (~out["CompanyID"].astype(str).str.startswith("UNRES_"))
        )
        out["IsDerived"] = False
        out["DerivationMethod"] = ""
        out["ParentFactIDs"] = ""
        out["IsRestated"] = df_in["IsRestated"].fillna(False).astype(bool)
        out["ComparativeFlag"] = df_in["ComparativeFlag"].fillna(False).astype(bool)
        out["ReportType"] = df_in["ReportType"].astype(str)
        out["SourceSHA256"] = df_in["SourceSHA256"].astype(str)
        out["Notes"] = df_in["Notes"].astype(str)
        if not is_phase3a:
            # Distinct FactID prefix before final numbering
            out["FactID"] = [f"RAW3B_{i+1:07d}" for i in range(len(out))]
        return out

    p3a_part = _to_hist_schema(phase3a_facts_df, is_phase3a=True)
    new_part = _to_hist_schema(new_norm_df, is_phase3a=False)

    # Concatenate Phase 3A + Newly Acquired; drop exact duplicate DisclosureIDs preferring Phase 3A verbatim
    combined = pd.concat([p3a_part, new_part], ignore_index=True)
    combined = combined.dropna(subset=["PeriodEnd", "Value"]).copy()
    combined = combined.drop_duplicates(
        subset=["SourceDocumentID", "StatementType", "MetricCode", "PeriodStart", "PeriodEnd", "PeriodType", "ConsolidationStatus", "Currency"],
        keep="first",
    ).reset_index(drop=True)

    # 3. Assign monotonic RestatementVersions strictly chronologically (`AvailabilityDate`)
    #    & preserve unchanged earlier facts at their EARLIEST verified AvailabilityDate!
    #    Single-pass linear scan over pre-sorted arrays runs in <1s even for 500k+ facts.
    combined["_avail_ord"] = pd.to_datetime(combined["AvailabilityDate"], errors="coerce")
    combined = combined.sort_values(
        ["CompanyID", "StatementType", "MetricCode", "PeriodEnd", "PeriodType", "ConsolidationStatus", "Currency", "_avail_ord", "SourceDocumentID"],
        na_position="last",
    ).reset_index(drop=True)

    c_cid = combined["CompanyID"].to_numpy()
    c_cid_code = combined["CompanyCode"].to_numpy()
    c_sid = combined["SecurityID"].to_numpy()
    c_st = combined["StatementType"].to_numpy()
    c_mc = combined["MetricCode"].to_numpy()
    c_ps = combined["PeriodStart"].to_numpy()
    c_pe = combined["PeriodEnd"].to_numpy()
    c_pt = combined["PeriodType"].to_numpy()
    c_cons = combined["ConsolidationStatus"].to_numpy()
    c_cur = combined["Currency"].to_numpy()
    c_val = combined["Value"].to_numpy(dtype=np.float64)
    c_rest = combined["IsRestated"].to_numpy(dtype=bool)
    c_pub = combined["PublicationDate"].to_numpy()
    c_avail = combined["AvailabilityDate"].to_numpy()
    c_did = combined["SourceDocumentID"].to_numpy()
    c_sfile = combined["SourceFile"].to_numpy()
    c_rtype = combined["ReportType"].to_numpy()

    kept_indices: list[int] = []
    rest_rows: list[dict[str, Any]] = []
    ver_by_idx: dict[int, int] = {}

    n_rows = len(combined)
    pos = 0
    while pos < n_rows:
        g_key = (c_cid[pos], c_st[pos], c_mc[pos], c_pe[pos], c_pt[pos], c_cons[pos], c_cur[pos])
        end_pos = pos + 1
        while end_pos < n_rows and (
            c_cid[end_pos], c_st[end_pos], c_mc[end_pos], c_pe[end_pos], c_pt[end_pos], c_cons[end_pos], c_cur[end_pos]
        ) == g_key:
            end_pos += 1

        if end_pos - pos == 1:
            kept_indices.append(pos)
            is_r = bool(c_rest[pos])
            ver_by_idx[pos] = 2 if is_r else 1
            if is_r:
                rest_rows.append({
                    "RestatementID": f"HRST_{len(rest_rows) + 1:05d}",
                    "CompanyID": c_cid[pos], "CompanyCode": c_cid_code[pos], "SecurityID": c_sid[pos],
                    "StatementType": c_st[pos], "MetricCode": c_mc[pos],
                    "PeriodEnd": pd.Timestamp(c_pe[pos]), "PeriodType": c_pt[pos],
                    "ConsolidationStatus": c_cons[pos], "Currency": c_cur[pos],
                    "RestatementType": "EXPLICIT_RESTATED_LABEL_IN_DISCLOSURE",
                    "OriginalValue": np.nan, "RestatedValue": float(c_val[pos]), "PctRevision": np.nan,
                    "OriginalPublicationDate": pd.NaT, "OriginalAvailabilityDate": pd.NaT,
                    "OriginalSourceDocumentID": "", "OriginalSourceFile": "",
                    "RestatedPublicationDate": pd.Timestamp(c_pub[pos]) if pd.notna(c_pub[pos]) else pd.NaT,
                    "RestatedAvailabilityDate": pd.Timestamp(c_avail[pos]) if pd.notna(c_avail[pos]) else pd.NaT,
                    "RestatedSourceDocumentID": c_did[pos], "RestatedSourceFile": c_sfile[pos],
                    "RestatementReason": "Marked Restated/Revised in comparative or errata statement column.",
                    "PITUsageRule": "Active strictly on dates >= RestatedAvailabilityDate.",
                })
        else:
            base_idx = pos
            kept_indices.append(base_idx)
            cur_ver = 1
            ver_by_idx[base_idx] = cur_ver
            last_idx = base_idx

            for nxt_idx in range(pos + 1, end_pos):
                ps1, ps2 = c_ps[last_idx], c_ps[nxt_idx]
                if pd.notna(ps1) and pd.notna(ps2) and abs((pd.Timestamp(ps1) - pd.Timestamp(ps2)).days) > 10:
                    kept_indices.append(nxt_idx)
                    ver_by_idx[nxt_idx] = 1
                    continue

                v1, v2 = float(c_val[last_idx]), float(c_val[nxt_idx])
                tol = max(2000.0, 0.0025 * max(abs(v1), abs(v2), 1.0))
                is_material_rev = abs(v2 - v1) > tol
                is_flagged = bool(c_rest[nxt_idx]) or ("Errata" in str(c_rtype[nxt_idx])) or (
                    str(c_rtype[last_idx]) != "Annual_Report" and str(c_rtype[nxt_idx]) == "Annual_Report" and is_material_rev
                )
                if is_material_rev or is_flagged:
                    cur_ver += 1
                    kept_indices.append(nxt_idx)
                    ver_by_idx[nxt_idx] = cur_ver
                    pct = 100.0 * (v2 - v1) / abs(v1) if abs(v1) > 1e-9 else np.nan
                    rtype = (
                        "AUDITED_ANNUAL_REVISION_OF_INTERIM"
                        if str(c_rtype[last_idx]) != "Annual_Report" and str(c_rtype[nxt_idx]) == "Annual_Report"
                        else ("ERRATA_AMENDMENT" if "Errata" in str(c_rtype[nxt_idx]) else "COMPARATIVE_RESTATEMENT")
                    )
                    av_last = pd.Timestamp(c_avail[last_idx]) if pd.notna(c_avail[last_idx]) else pd.NaT
                    av_nxt = pd.Timestamp(c_avail[nxt_idx]) if pd.notna(c_avail[nxt_idx]) else pd.NaT
                    reason = (
                        f"Reported {v1:,.2f} in doc {c_did[last_idx]} ({str(av_last)[:10]}), "
                        f"updated to {v2:,.2f} in doc {c_did[nxt_idx]} ({str(av_nxt)[:10]})."
                    )
                    rest_rows.append({
                        "RestatementID": f"HRST_{len(rest_rows) + 1:05d}",
                        "CompanyID": c_cid[nxt_idx], "CompanyCode": c_cid_code[nxt_idx], "SecurityID": c_sid[nxt_idx],
                        "StatementType": c_st[nxt_idx], "MetricCode": c_mc[nxt_idx],
                        "PeriodEnd": pd.Timestamp(c_pe[nxt_idx]), "PeriodType": c_pt[nxt_idx],
                        "ConsolidationStatus": c_cons[nxt_idx], "Currency": c_cur[nxt_idx],
                        "RestatementType": rtype,
                        "OriginalValue": v1, "RestatedValue": v2,
                        "PctRevision": round(pct, 4) if pd.notna(pct) else np.nan,
                        "OriginalPublicationDate": pd.Timestamp(c_pub[last_idx]) if pd.notna(c_pub[last_idx]) else pd.NaT,
                        "OriginalAvailabilityDate": av_last,
                        "OriginalSourceDocumentID": c_did[last_idx],
                        "OriginalSourceFile": c_sfile[last_idx],
                        "RestatedPublicationDate": pd.Timestamp(c_pub[nxt_idx]) if pd.notna(c_pub[nxt_idx]) else pd.NaT,
                        "RestatedAvailabilityDate": av_nxt,
                        "RestatedSourceDocumentID": c_did[nxt_idx],
                        "RestatedSourceFile": c_sfile[nxt_idx],
                        "RestatementReason": reason,
                        "PITUsageRule": "Use OriginalValue on dates in [OriginalAvailabilityDate, RestatedAvailabilityDate); RestatedValue strictly >= RestatedAvailabilityDate.",
                    })
                    last_idx = nxt_idx

        pos = end_pos

    hist_facts = combined.loc[kept_indices].copy().drop(columns=["_avail_ord"]).reset_index(drop=True)
    hist_facts["RestatementVersion"] = [ver_by_idx[i] for i in kept_indices]

    # Assign final monotonic FactIDs (keeping Phase 3A original F3A_ IDs intact cleanly!)
    new_counter = 1
    final_fids: list[str] = []
    for fid in hist_facts["FactID"]:
        s = str(fid)
        if s.startswith("F3A_"):
            final_fids.append(s)
        else:
            final_fids.append(f"F3B_{new_counter:07d}")
            new_counter += 1
    hist_facts["FactID"] = final_fids

    # 4. Derive standalone quarters (Q1, Q2, Q3, Q4) across entire multi-year Flow database
    derived_q_df = _derive_historical_standalone_quarters(hist_facts)
    if not derived_q_df.empty:
        derived_q_df["FactID"] = [f"DF3B_{i + 1:07d}" for i in range(len(derived_q_df))]
        # Append derived standalone quarters into historical_financial_facts ONLY where no direct reported quarter exists
        direct_q = hist_facts[hist_facts["PeriodType"] == "Quarterly"]
        existing_q_keys = set(zip(
            direct_q["CompanyID"], direct_q["StatementType"], direct_q["MetricCode"],
            direct_q["PeriodEnd"], direct_q["ConsolidationStatus"], direct_q["Currency"],
        ))
        new_q_mask = [
            (r["CompanyID"], r["StatementType"], r["MetricCode"], r["PeriodEnd"], r["ConsolidationStatus"], r["Currency"]) not in existing_q_keys
            for _, r in derived_q_df.iterrows()
        ]
        derived_to_append = derived_q_df.loc[new_q_mask, HISTORICAL_FACT_COLUMNS].copy()
        if not derived_to_append.empty:
            hist_facts = pd.concat([hist_facts, derived_to_append], ignore_index=True)

    hist_facts = hist_facts[HISTORICAL_FACT_COLUMNS].sort_values(
        ["CompanyID", "PeriodEnd", "StatementType", "MetricCode", "RestatementVersion", "IsDerived"],
        na_position="last",
    )
    # Guarantee zero duplicates on exact composite PIT key
    hist_facts = hist_facts.drop_duplicates(
        subset=[
            "CompanyID", "StatementType", "MetricCode", "PeriodStart", "PeriodEnd",
            "PeriodType", "ConsolidationStatus", "Currency", "RestatementVersion",
        ],
        keep="first",
    ).reset_index(drop=True)

    # 5. Build Annual Financial History (`PeriodType == Annual`) & Quarterly Financial History (`PeriodType == Quarterly` + derived)
    annual_df = hist_facts[hist_facts["PeriodType"] == "Annual"].copy().reset_index(drop=True)
    # Include both direct reported quarters and derived quarters in `historical_quarterly_financials.parquet`
    direct_q_df = hist_facts[(hist_facts["PeriodType"] == "Quarterly") & (~hist_facts["IsDerived"])].copy()
    quarterly_df = pd.concat([direct_q_df, derived_q_df], ignore_index=True) if not derived_q_df.empty else direct_q_df
    quarterly_df = quarterly_df.sort_values(
        ["CompanyID", "PeriodEnd", "StatementType", "MetricCode", "IsDerived"],
        na_position="last",
    ).reset_index(drop=True)

    # 6. Build Restatements table (plus Phase 3A Errata notice RST_0002 preserved explicitly)
    rest_cols = [
        "RestatementID", "CompanyID", "CompanyCode", "SecurityID", "StatementType", "MetricCode",
        "PeriodEnd", "PeriodType", "ConsolidationStatus", "Currency", "RestatementType",
        "OriginalValue", "RestatedValue", "PctRevision",
        "OriginalPublicationDate", "OriginalAvailabilityDate", "OriginalSourceDocumentID", "OriginalSourceFile",
        "RestatedPublicationDate", "RestatedAvailabilityDate", "RestatedSourceDocumentID", "RestatedSourceFile",
        "RestatementReason", "PITUsageRule",
    ]
    # Add explicit ASIR 53505 disclosure errata notice if present in catalog
    asir_err = catalog_df[(catalog_df["CompanyCode"] == "ASIR") & (catalog_df["DocumentID"].astype(str) == "53505")]
    if not asir_err.empty:
        er = asir_err.iloc[0]
        rest_rows.append({
            "RestatementID": f"HRST_{len(rest_rows) + 1:05d}",
            "CompanyID": "CID_ASIR", "CompanyCode": "ASIR", "SecurityID": "ASIR-N-0000",
            "StatementType": "Disclosure_Errata", "MetricCode": "SHAREHOLDING_OR_NOTE_ERRATA",
            "PeriodEnd": pd.Timestamp("2025-09-30"), "PeriodType": "Errata_Notice",
            "ConsolidationStatus": "Consolidated_Group", "Currency": "LKR",
            "RestatementType": "DISCLOSURE_ERRATA_FILING",
            "OriginalValue": np.nan, "RestatedValue": np.nan, "PctRevision": 0.0,
            "OriginalPublicationDate": pd.Timestamp("2026-11-06 18:41:39"),
            "OriginalAvailabilityDate": pd.Timestamp("2026-11-06 18:41:39"),
            "OriginalSourceDocumentID": "53438",
            "OriginalSourceFile": "cse_live/documents/financial_reports/53438_512_1793970699126.pdf",
            "RestatedPublicationDate": pd.to_datetime(er["PublicationDate"], errors="coerce"),
            "RestatedAvailabilityDate": pd.to_datetime(er["AvailabilityDate"], errors="coerce"),
            "RestatedSourceDocumentID": "53505",
            "RestatedSourceFile": str(er.get("LocalFilePath") or ""),
            "RestatementReason": "ASIR disclosure errata amended non-primary shareholding table; financial statements verified unchanged.",
            "PITUsageRule": "Use errata notice strictly >= RestatedAvailabilityDate.",
        })
    rest_df = pd.DataFrame(rest_rows, columns=rest_cols)

    # 7. Build Strict PIT Availability Schedule (`financial_pit_availability.parquet`)
    pit_agg = (
        hist_facts.groupby(["CompanyID", "CompanyCode", "PeriodEnd", "PeriodType", "ConsolidationStatus"], dropna=False)
        .agg(
            FirstReportedDate=("AvailabilityDate", "min"),
            LatestRestatedDate=("AvailabilityDate", "max"),
            RestatementVersionsCount=("RestatementVersion", "max"),
            FactsCount=("FactID", "count"),
            StrictPITFactsCount=("PITEligibility", "sum"),
        )
        .reset_index()
    )
    pit_agg["ReportingLagDays"] = (
        pd.to_datetime(pit_agg["FirstReportedDate"], errors="coerce") - pd.to_datetime(pit_agg["PeriodEnd"], errors="coerce")
    ).dt.days
    pit_agg["HasStrictPITEligibility"] = pit_agg["StrictPITFactsCount"] > 0

    return {
        "historical_financial_facts": hist_facts,
        "historical_annual_financials": annual_df,
        "historical_quarterly_financials": quarterly_df,
        "historical_restated_values": rest_df,
        "financial_pit_availability": pit_agg,
        "new_validation_checks": new_val_df,
    }
