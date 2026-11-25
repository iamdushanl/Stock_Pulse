"""
Phase 1B Task 2 – CSE Security & Issuer Identity Model
=====================================================
Constructs a complete, non-collapsing security-level identity table covering
all observed (CompanyCode, MainType, SubType) series in the reconstructed
Phase 1A historical universe without merging separate share classes or inventing
missing issuer links.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def normalize_company_name(name: object) -> str:
    """Normalize a company name for deterministic cross-workbook matching."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).upper().strip()
    if s in ("", "NAN", "NONE"):
        return ""
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"-\s*(ORDINARY\s+)?(NON\s*-?\s*VOTING|VOTING|NV)\b.*", " ", s)
    s = s.replace("&", " AND ").replace("'S", " S ").replace("'", " ")
    # Join single-letter initial runs: "C T LANDS" -> "CT LANDS", "L V L" -> "LVL"
    s = re.sub(r"\b([A-Z])\s+(?=[A-Z]\b)", r"\1", s)
    s = re.sub(
        r"\b(PLC|PVT|PRIVATE|LIMITED|LTD|CO\.?|COMPANY|THE)\b",
        " ",
        s,
    )
    return re.sub(r"[^A-Z0-9]+", " ", s).strip()


def _parse_sec_id(raw_sec: str) -> Optional[tuple[str, str, str]]:
    """Parse 'JKH-N-0000' or 'JKH.N0000' into (CompanyCode, MainType, SubType)."""
    if not raw_sec:
        return None
    s = str(raw_sec).strip().upper()
    m = re.match(r"^([A-Z0-9]+)-([A-Z])-(\d{1,4})$", s)
    if m:
        code, mtype, stype = m.group(1), m.group(2), m.group(3)
        stype = "0000" if stype in ("0", "00", "000", "0000") else stype.zfill(4)
        return code, mtype, stype
    m2 = re.match(r"^([A-Z0-9]+)\.([A-Z])(\d{1,4})$", s)
    if m2:
        code, mtype, stype = m2.group(1), m2.group(2), m2.group(3)
        stype = "0000" if stype in ("0", "00", "000", "0000") else stype.zfill(4)
        return code, mtype, stype
    return None


def extract_quoted_securities_history(source_dir: Path) -> tuple[dict[str, dict], dict[tuple[int, str], float], dict[str, set[str]]]:
    """
    Scan all 2005–2025 sheets of ``16 List of Quoted Securities And Issued Quantity.xls``
    at exact SecurityID granularity and extract historical name aliases and issued quantities.
    """
    quoted_path = source_dir / "16 List of Quoted Securities And Issued Quantity.xls"
    sec_meta: dict[str, dict] = {}
    yearly_qty: dict[tuple[int, str], float] = {}
    name_aliases: dict[str, set[str]] = {}

    if not quoted_path.exists():
        return sec_meta, yearly_qty, name_aliases

    xl = pd.ExcelFile(quoted_path)
    year_sheets = sorted(
        [s for s in xl.sheet_names if re.match(r"^\s*\d{4}\s*$", s)],
        key=lambda s: int(s.strip()),
    )

    for sh in year_sheets:
        yr = int(sh.strip())
        d = xl.parse(sh, header=None, nrows=12)
        hrow = next(
            (
                i for i in range(len(d))
                if "SECURITY" in " ".join(str(v).upper() for v in d.iloc[i].values if pd.notna(v))
                and "NAME" in " ".join(str(v).upper() for v in d.iloc[i].values if pd.notna(v))
            ),
            None,
        )
        if hrow is None:
            continue

        dq = xl.parse(sh, header=hrow)
        cols = [str(c).strip().lower() for c in dq.columns]
        dq.columns = cols

        sec_c   = next((c for c in cols if "security" in c and not c.endswith((".1", "_1"))), None)
        name_c  = next((c for c in cols if c == "name" or ("name" in c and "company" not in c and not c.endswith((".1", "_1")))), None)
        qty_c   = next((c for c in cols if "issued" in c or "quantity" in c), None)
        sec2_c  = next((c for c in cols if c.startswith("security") and c.endswith((".1", "_1"))), None)
        name2_c = next((c for c in cols if "company name" in c), None)
        isin_c  = next((c for c in cols if "isin" in c), None)

        if sec_c and name_c:
            for _, r in dq.iterrows():
                parsed = _parse_sec_id(r[sec_c])
                if not parsed:
                    continue
                code, mtype, stype = parsed
                sec_id = f"{code}-{mtype}-{stype}"
                nm = str(r[name_c]).strip() if pd.notna(r[name_c]) else ""
                qty = pd.to_numeric(r[qty_c], errors="coerce") if qty_c else np.nan
                entry = sec_meta.setdefault(sec_id, {"CompanyCode": code})
                if nm and nm.lower() != "nan":
                    entry["CompanyName"] = nm
                    nk = normalize_company_name(nm)
                    if nk:
                        name_aliases.setdefault(nk, set()).add(code)
                if pd.notna(qty) and qty > 0:
                    entry["LatestIssuedQuantity"] = float(qty)
                    yearly_qty[(yr, sec_id)] = float(qty)
                if not sec2_c and isin_c and pd.notna(r[isin_c]):
                    isin_val = str(r[isin_c]).strip()
                    if isin_val and isin_val.lower() != "nan":
                        entry["ISIN"] = isin_val

        if sec2_c and isin_c:
            for _, r in dq.iterrows():
                parsed2 = _parse_sec_id(r[sec2_c])
                if not parsed2:
                    continue
                code2, mtype2, stype2 = parsed2
                sec_id2 = f"{code2}-{mtype2}-{stype2}"
                entry2 = sec_meta.setdefault(sec_id2, {"CompanyCode": code2})
                isin2 = str(r[isin_c]).strip() if pd.notna(r[isin_c]) else ""
                if isin2 and isin2.lower() != "nan":
                    entry2["ISIN"] = isin2
                if name2_c and pd.notna(r[name2_c]):
                    nm2 = str(r[name2_c]).strip()
                    if nm2 and nm2.lower() != "nan":
                        entry2.setdefault("CompanyName", nm2)
                        nk2 = normalize_company_name(nm2)
                        if nk2:
                            name_aliases.setdefault(nk2, set()).add(code2)

    return sec_meta, yearly_qty, name_aliases


def build_security_identity_model(
    prices_df: pd.DataFrame,
    phase1_master: pd.DataFrame,
    source_dir: Path,
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[tuple[int, str], float]]:
    """
    Build the Security Identity table with one row per distinct trading
    (CompanyCode, MainType, SubType) series — never collapsing share classes.
    """
    sec_meta, yearly_qty, name_aliases = extract_quoted_securities_history(source_dir)

    # Enrich historical alias dictionary from Phase 1 master + 38Foreign Holding
    for _, r in phase1_master.iterrows():
        c = str(r["CompanyCode"]).strip().upper()
        for col in ("OfficialCompanyName", "ShortName"):
            nk = normalize_company_name(r.get(col))
            if nk and len(nk) >= 3:
                name_aliases.setdefault(nk, set()).add(c)

    fh_path = source_dir / "38Foreign Holding-Annual.xlsx"
    if fh_path.exists():
        xl_fh = pd.ExcelFile(fh_path)
        for sh in [s for s in xl_fh.sheet_names if re.match(r"^\s*\d{4}\s*$", s)]:
            try:
                d = xl_fh.parse(sh, header=None, nrows=10)
                hrow = next(
                    (i for i in range(len(d)) if "COMPANY" in " ".join(str(v).upper() for v in d.iloc[i].values if pd.notna(v))),
                    None,
                )
                if hrow is None:
                    continue
                dfh = xl_fh.parse(sh, header=hrow)
                for _, r in dfh.iterrows():
                    code = str(r.iloc[0]).strip().upper() if pd.notna(r.iloc[0]) else ""
                    nm   = str(r.iloc[3]).strip() if dfh.shape[1] > 3 and pd.notna(r.iloc[3]) else ""
                    if re.match(r"^[A-Z0-9]{2,8}$", code) and nm and nm.lower() != "nan":
                        nk = normalize_company_name(nm)
                        if nk and len(nk) >= 3:
                            name_aliases.setdefault(nk, set()).add(code)
            except Exception:
                continue

    # Aggregate observed trading series strictly at (CompanyCode, MainType, SubType)
    df_p = prices_df.sort_values(["CompanyCode", "MainType", "SubType", "Date"])
    series_agg = (
        df_p.groupby(["CompanyCode", "MainType", "SubType"], as_index=False)
        .agg(
            FirstPriceDate=("Date", "min"),
            LastPriceDate=("Date", "max"),
            TradingDaysCount=("Date", "nunique"),
            TotalPriceRecords=("Date", "count"),
            ShortName=("ShortName", lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan),
        )
    )
    series_agg["SecurityID"] = (
        series_agg["CompanyCode"].astype(str).str.upper()
        + "-" + series_agg["MainType"].astype(str).str.upper()
        + "-" + series_agg["SubType"].astype(str)
    )

    p1_idx = phase1_master.set_index("CompanyCode")

    # Detect CompanyCodes sharing identical normalized OfficialCompanyName (corporate mergers)
    p1_norm_names = phase1_master["OfficialCompanyName"].apply(normalize_company_name)
    dup_name_mask = (p1_norm_names != "") & p1_norm_names.duplicated(keep=False)
    ambig_name_codes = set(phase1_master.loc[dup_name_mask, "CompanyCode"].astype(str).str.upper())

    rows = []
    for _, r in series_agg.iterrows():
        code   = str(r["CompanyCode"]).upper()
        mtype  = str(r["MainType"]).upper()
        stype  = str(r["SubType"])
        sec_id = r["SecurityID"]

        p1_row = p1_idx.loc[code] if code in p1_idx.index else None
        q_row  = sec_meta.get(sec_id, {})
        # Fallback within issuer for N-class ISIN prefix if specific sub-class wasn't in 16 Quoted
        comp_isin = p1_row["ISINCode"] if (p1_row is not None and pd.notna(p1_row["ISINCode"])) else np.nan
        sec_isin  = q_row.get("ISIN", np.nan)
        comp_name = (
            q_row.get("CompanyName")
            or (p1_row["OfficialCompanyName"] if p1_row is not None and pd.notna(p1_row["OfficialCompanyName"]) else np.nan)
        )
        short_nm = r["ShortName"] if pd.notna(r["ShortName"]) else (
            p1_row["ShortName"] if p1_row is not None else np.nan
        )

        # Listing & delisting dates: official overrides estimate only where present; keep price bounds too
        list_off   = p1_row["ListingDateOfficial"] if p1_row is not None else pd.NaT
        delist_off = p1_row["DelistingDateOfficial"] if p1_row is not None else pd.NaT
        list_est   = p1_row["ListingDateEstimate"] if p1_row is not None else r["FirstPriceDate"]
        delist_est = p1_row["DelistingDateEstimate"] if p1_row is not None else r["LastPriceDate"]

        hist_status = (
            str(p1_row["HistoricalActiveStatus"])
            if p1_row is not None
            else ("Delisted" if r["LastPriceDate"].year < 2024 else "Listed")
        )

        # IssuerID assignment rule (strict, no silent guessing)
        isin_for_prefix = sec_isin if pd.notna(sec_isin) else comp_isin
        has_isin = pd.notna(isin_for_prefix) and bool(re.match(r"^LK\d{4}", str(isin_for_prefix)))
        has_name = pd.notna(comp_name) and str(comp_name).strip() != ""

        if code in ambig_name_codes:
            # Separate identities preserved; flagged for manual corporate merger review
            issuer_id = f"ISS_{code}"
            val_status = "MANUAL_REVIEW_MERGER_NAME_OVERLAP"
            manual_review = True
        elif has_isin:
            issuer_id = f"ISS_{code}"
            val_status = "VERIFIED_ISIN"
            manual_review = False
        elif has_name:
            issuer_id = f"ISS_{code}"
            val_status = "VERIFIED_OFFICIAL_NAME"
            manual_review = False
        else:
            issuer_id = None
            val_status = "MANUAL_REVIEW_UNVERIFIED_ISSUER"
            manual_review = True

        rows.append({
            "SecurityID": sec_id,
            "CompanyCode": code,
            "IssuerID": issuer_id,
            "MainType": mtype,
            "SubType": stype,
            "ISIN": sec_isin if pd.notna(sec_isin) else np.nan,
            "IssuerPrimaryISIN": comp_isin,
            "CompanyName": comp_name,
            "ShortName": short_nm,
            "ListingDate": list_off if pd.notna(list_off) else list_est,
            "ListingDateSource": "OfficialListingsWorkbook" if pd.notna(list_off) else "FirstObservedPriceDate",
            "DelistingDate": delist_off if pd.notna(delist_off) else (delist_est if hist_status == "Delisted" else pd.NaT),
            "DelistingDateSource": (
                "OfficialDelistingsWorkbook" if pd.notna(delist_off)
                else ("LastObservedPriceDate" if hist_status == "Delisted" else np.nan)
            ),
            "FirstPriceDate": r["FirstPriceDate"],
            "LastPriceDate": r["LastPriceDate"],
            "TradingDaysCount": int(r["TradingDaysCount"]),
            "TotalPriceRecords": int(r["TotalPriceRecords"]),
            "LatestIssuedQuantity": q_row.get("LatestIssuedQuantity", np.nan),
            "HistoricalStatus": hist_status,
            "IssuerValidationStatus": val_status,
            "ManualReviewRequired": bool(manual_review),
            "SourceFlag": "Phase1A+QuotedList+ForeignHoldings",
        })

    identity_df = pd.DataFrame(rows).sort_values(["CompanyCode", "MainType", "SubType"]).reset_index(drop=True)
    return identity_df, name_aliases, yearly_qty
