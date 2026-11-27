"""
Phase 3B Historical PDF Statement Parser & Multi-Core Extractor
===============================================================
Wraps Phase 3A's coordinate-aware spatial parser (`phase3a_extract`) cleanly with:
  1. Multi-year date fixes (2007–2027, US MM/DD/YYYY `06/30/2023`, space-separated
     `30 06 2016`, range headers `"01.01.2018 - 30.09.2018"` selecting closing date).
  2. Annual Report primary-statement isolation (rejecting `Five Year Summary`,
     `Ten Year Summary`, `Financial Highlights`, `MD&A`, and `Notes` pages while
     handling `Annual Report 2018/2019 -> 2019-03-31` & full-year Equity horizons).
  3. Additional historical metrics (`DIVIDEND_PER_SHARE`, `EBITDA`) cleanly matched.
  4. Multiprocessing (`ProcessPoolExecutor`) + persistent per-file parse cache (`_parse_cache`)
     + bounded WinOCR fallback for scanned historical Interims (`_ocr_cache`).
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf

from phase3a_extract import statement_parser as sp3a
from phase3a_extract.pdf_reader_and_ocr import PageData, WordToken, extract_pdf_pages_tokens

logger = logging.getLogger("phase3b.parser")

ANNUAL_SUMMARY_REJECT_RE = re.compile(
    r"\b(?:five[- ]?year\s+summary|ten[- ]?year\s+summary|decade\s+at\s+a\s+glance|"
    r"financial\s+highlights|performance\s+highlights|key\s+value\s+drivers|key\s+performance\s+indicators|"
    r"management\s+discussion|value\s+added\s+statement|quarterly\s+analysis|investor\s+information|"
    r"shareholder\s+information|twenty\s+major\s+shareholders|corporate\s+information|"
    r"operating\s+segments|segmental\s+(?:analysis|information)|discontinued\s+operations?|"
    r"life\s+insurance\s+fund|supplementary\s+information|table\s+of\s+contents|independent\s+auditors?)\b",
    re.I,
)
NOTES_HEADER_RE = re.compile(
    r"(?:^\s*(?:notes\s+to\s+the\s+(?:consolidated\s+|interim\s+)?financial\s+statements?|"
    r"significant\s+accounting\s+policies|material\s+accounting\s+policies|explanatory\s+notes)|"
    r"^\s*\d{1,2}\.\d+(?:\.[a-z0-9]+)?\s+[A-Z])",
    re.I | re.M,
)

# Register additional safe Phase 3B rules cleanly inside historical parsing without altering Phase 3A outputs on disk
EXTRA_METRIC_RULES: list[tuple[str, str, re.Pattern[str], re.Pattern[str] | None]] = [
    ("DIVIDEND_PER_SHARE", "Income Statement", re.compile(r"^(?:dividend[s]?\s+per\s+share|gross\s+dividend\s+per\s+share|interim\s+dividend\s+per\s+share|final\s+dividend\s+per\s+share)$", re.I), None),
    ("EBITDA", "Income Statement", re.compile(r"^(?:ebitda|earnings?\s+before\s+interest,?\s*tax(?:es)?,?\s*depreciation.*)$", re.I), None),
]


def _enhanced_detect_currency_unit(text: str, doc_hint: str) -> tuple[str, str, float, bool]:
    """Enhanced unit detector supporting singular/plural `Rs. million`, `Rs. ' 000`, `Rs.‘000s`, etc."""
    tl = text.lower()
    has_lkr_explicit = bool(re.search(
        r"(?:\blkr\b|all\s+(?:values|amounts)\s+(?:are\s+)?in\s+(?:lkr|rs|sri\s+lankan?\s+rupees)|"
        r"the\s+above\s+figures\s+are\s+in\s+(?:lkr|rs)|in\s+rs\.?\s*[-'‘’′`.,]?\s*000s?\b|"
        r"lkr\.?\s*[-'‘’′`.,]?\s*[0o]{3}s?\b|rs\.?\s*[-'‘’′`.,]?\s*[s5]?\s*[0o]{3}s?\b)",
        tl,
    ))
    if not has_lkr_explicit:
        if re.search(r"(?:united\s+states\s+dollars|us\$|\busd\b)\s*(?:\(?[0o]{3}s?|thousands)", tl):
            return "USD", "USD thousands", 1000.0, False
        if re.search(r"(?:united\s+states\s+dollars|\busd\b|\bus\$\b)", tl):
            return "USD", "USD", 1.0, False

    if re.search(r"(?:rupees\s+(?:in\s+)?millions?|rs\.?\s*mn\b|rs\.?\s*millions?\b|lkr\s*(?:in\s+)?millions?|\(rs\.?\s*m\)|\(in\s+rs\.?\s*millions?\))", tl):
        return "LKR", "LKR millions", 1_000_000.0, False
    if re.search(r"(?:sri\s+lankan?\s+rupees\s+thousands?|rupees\s+(?:in\s+)?thousands?|rs\.?\s*[-'‘’′`.,]?\s*[s5]?\s*[0o]{3}s?\b|lkr\.?\s*[-'‘’′`.,]?\s*[0o]{3}s?\b|\bin\s+['‘’′`.,]?(?:thousands|[0o]{3}s?)\b|all\s+(?:amounts|values)\s+are\s+(?:in\s+)?(?:sri\s+lankan?\s+rupees\s+thousands?|lkr\.?\s*[-'‘’′`.,]?\s*[0o]{3}s?))", tl):
        return "LKR", "LKR thousands", 1000.0, False
    if re.search(r"(?:\brs\.?(?!\s*[-'‘’′`.,]?\s*[s5]?\s*[0o]{3}s?)|\blkr\b|sri\s+lankan?\s+rupees)", tl):
        return "LKR", "LKR", 1.0, False

    if doc_hint == "USD":
        return "USD", "AMBIGUOUS_UNIT", 1.0, True
    return "LKR", "AMBIGUOUS_UNIT", 1.0, True


def _enhanced_build_column_headers(
    rows: list[list[WordToken]],
    hdr_end_idx: int,
    cols: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Separate standalone `%` variance tokens from adjacent year tokens (`2025 %`) in column headers."""
    col_headers: list[dict[str, list[str]]] = [{"narrow": [], "span": []} for _ in cols]
    for r in rows[:hdr_end_idx]:
        if not r:
            continue
        phrases: list[dict[str, Any]] = []
        cur = [r[0]]
        for t in r[1:]:
            t_cur = cur[-1].text.strip()
            t_nxt = t.text.strip()
            is_split_tok = (
                (re.match(r"^20\d{2}$", t_cur) and re.match(r"^20\d{2}$", t_nxt))
                or t_cur == "%"
                or t_nxt == "%"
            )
            if (t.x0 - cur[-1].x1) <= 10.0 and not is_split_tok:
                cur.append(t)
            else:
                phrases.append({"x0": cur[0].x0, "x1": cur[-1].x1, "xc": (cur[0].x0 + cur[-1].x1) * 0.5, "text": " ".join(w.text for w in cur)})
                cur = [t]
        phrases.append({"x0": cur[0].x0, "x1": cur[-1].x1, "xc": (cur[0].x0 + cur[-1].x1) * 0.5, "text": " ".join(w.text for w in cur)})

        for ph in phrases:
            p_txt = ph["text"].strip()
            if re.match(r"^(?:20\d{2}|%)$", p_txt):
                best_j = min(range(len(cols)), key=lambda j: abs(ph["xc"] - cols[j]["xc"]))
                if abs(ph["xc"] - cols[best_j]["xc"]) <= 45.0:
                    col_headers[best_j]["narrow"].append(p_txt)
                continue
            overlaps = [j for j, c in enumerate(cols) if min(ph["x1"], c["x1"] + 6.0) - max(ph["x0"], c["x0"] - 6.0) > 2.0]
            if len(overlaps) == 1:
                col_headers[overlaps[0]]["narrow"].append(p_txt)
            elif len(overlaps) > 1:
                for j in overlaps:
                    col_headers[j]["span"].append(p_txt)
            else:
                best_d, best_j = min((abs(ph["xc"] - c["xc"]), j) for j, c in enumerate(cols))
                if best_d <= 35.0:
                    col_headers[best_j]["narrow"].append(p_txt)

    return [{"narrow": " ".join(h["narrow"]), "span": " ".join(h["span"]), "full": " ".join(h["narrow"] + h["span"])} for h in col_headers]


def repair_and_enhance_date_parsing(txt: str) -> pd.Timestamp | None:
    """
    Parse CSE historical date expressions across 2010–2027 including:
      - Range headers (`01.01.2018 - 30.09.2018`, `1st April 2019 to 31st March 2020`) -> max date!
      - US `MM/DD/YYYY` (`06/30/2023`, `09/30/2022`)
      - Two-digit years (`11..27`, `September '16`, `31-Mar-18`)
    Strict delimiter rules forbid space-only dd mm yy matches that hit registration numbers (`30 05 09`).
    """
    t = str(txt or "").strip()
    if not t:
        return None
    cands: list[pd.Timestamp] = []

    # 1. ISO YYYY-MM-DD
    for m in re.finditer(r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b", t):
        yi, mi, di = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2010 <= yi <= 2027 and 1 <= mi <= 12 and 1 <= di <= 31:
            try:
                cands.append(pd.Timestamp(year=yi, month=mi, day=di))
            except ValueError:
                pass

    # 2. dd Month yyyy / dd-Mon-yy
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s./-]+([A-Za-z]+)[\s./-]+'?(\d{2,4})\b", t):
        mo = sp3a.MONTH_MAP.get(m.group(2).lower())
        ys = m.group(3)
        yr_raw = int(ys)
        yr = yr_raw if len(ys) == 4 else (2000 + yr_raw if 10 <= yr_raw <= 27 else -1)
        if mo and 2010 <= yr <= 2027:
            try:
                cands.append(pd.Timestamp(year=yr, month=mo, day=int(m.group(1))))
            except ValueError:
                pass

    # 3. Numeric dd/mm/yyyy, dd.mm.yy, US mm/dd/yyyy (require explicit ., / or - delimiter; no space-only match!)
    for m in re.finditer(r"\b(\d{1,2})[./-]+(\d{1,2})[./-]+(\d{2,4})\b", t):
        a, b = int(m.group(1)), int(m.group(2))
        ys = m.group(3)
        yr_raw = int(ys)
        yr = yr_raw if len(ys) == 4 else (2000 + yr_raw if 11 <= yr_raw <= 27 else -1)
        if 2010 <= yr <= 2027:
            di, mo_i = (b, a) if (1 <= a <= 12 < b <= 31) else (a, b)
            if 1 <= mo_i <= 12 and 1 <= di <= 31:
                try:
                    cands.append(pd.Timestamp(year=yr, month=mo_i, day=di))
                except ValueError:
                    pass

    # 4. Month 'yy (`September '25`)
    for m in re.finditer(r"\b([A-Za-z]+)\s*'(\d{2})\b", t):
        mo = sp3a.MONTH_MAP.get(m.group(1).lower())
        yr_raw = int(m.group(2))
        if mo and 10 <= yr_raw <= 27:
            yr = 2000 + yr_raw
            d_end = 31 if mo in (1, 3, 5, 7, 8, 10, 12) else (30 if mo != 2 else 28)
            cands.append(pd.Timestamp(year=yr, month=mo, day=d_end))

    if cands:
        return max(cands)
    return None


def _install_enhanced_parser_hooks() -> None:
    """Safely enhance Phase 3A's date, column header, unit & metric matchers in memory when running Phase 3B extraction."""
    sp3a._parse_date_string = repair_and_enhance_date_parsing
    sp3a._detect_currency_unit_from_text = _enhanced_detect_currency_unit
    sp3a._build_column_headers = _enhanced_build_column_headers
    existing_codes = {r[0] for r in sp3a.METRIC_RULES}
    for rule in EXTRA_METRIC_RULES:
        if rule[0] not in existing_codes:
            sp3a.METRIC_RULES.append(rule)
            existing_codes.add(rule[0])


def _is_primary_statement_candidate(raw_text: str, page_count: int, page_idx: int) -> bool:
    """
    Fast heuristic filter for multi-page Annual Reports (`page_count > 28`) to skip glossy narratives,
    Five-Year Summaries, Highlights, Discontinued Ops, Segment Notes, and Fund-only statements.
    """
    if page_count <= 28:
        return True
    low = raw_text.lower()
    head_low = low[:1200]
    if ANNUAL_SUMMARY_REJECT_RE.search(head_low):
        return False
    if NOTES_HEADER_RE.search(head_low[:600]):
        return False
    has_primary_title = bool(
        sp3a.STMT_TITLE_IS.search(head_low)
        or sp3a.STMT_TITLE_BS.search(head_low)
        or sp3a.STMT_TITLE_CF.search(head_low)
        or sp3a.STMT_TITLE_EQ.search(head_low)
    )
    if has_primary_title:
        return True
    # Balance Sheet signature on pages where header text spans horizontally/vertically
    if ("total assets" in low and ("total equity" in low or "total liabilities" in low)) and ("notes to the" not in head_low):
        return True
    if ("profit before tax" in low or "gross profit" in low) and ("revenue" in low or "turnover" in low) and ("notes to the" not in head_low):
        return True
    if ("operating activities" in low and "investing activities" in low) and ("notes to the" not in head_low):
        return True
    return False


def _extract_single_historical_pdf(task: dict[str, Any]) -> dict[str, Any]:
    """Worker function (process-safe) extracting one historical PDF cleanly with cache hit support."""
    _install_enhanced_parser_hooks()
    did = str(task["DocumentID"])
    pdf_path = Path(task["AbsFilePath"])
    sha = str(task["SHA256"])
    cache_file = Path(task["CacheDir"]) / f"{did}_{sha[:16]}.json"
    ocr_cache_dir = Path(task["OCRCacheDir"])

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            return cached
        except Exception:
            pass

    if not pdf_path.exists():
        return {"DocumentID": did, "Status": "FAILED", "Facts": [], "Note": "Missing file on disk"}

    try:
        doc = pymupdf.open(pdf_path)
        n_pages = len(doc)
        pages: list[PageData] = []
        total_chars = 0
        need_ocr_flag = False
        garb_re = re.compile(r"(?:\b20(?:1\d|2\d)[sSlIo]\b|\(\d+[,.\d]*[sIlOo\]]|[\d,]+[sIlOo]\d)")
        notes_begun = False

        for i in range(n_pages):
            pg = doc[i]
            raw = pg.get_text("text") or ""
            stripped = raw.strip()
            total_chars += len(stripped)
            if n_pages > 28 and len(pages) >= 2 and re.search(
                r"\b(?:notes\s+to\s+the\s+(?:consolidated\s+)?financial\s+statements|1\.\s+corporate\s+information|significant\s+accounting\s+policies)\b",
                raw[:900],
                re.I,
            ):
                notes_begun = True
            if notes_begun and n_pages > 28:
                break
            if not _is_primary_statement_candidate(raw, n_pages, i):
                continue
            # Check OCR fallback only on candidate financial pages of <= 26-page reports
            if (stripped.count("(cid:") >= 3 or len(garb_re.findall(stripped)) >= 3 or len(stripped) < 120) and n_pages <= 26:
                need_ocr_flag = True
            words = pg.get_text("words") or []
            toks = [
                WordToken(float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4]).strip(), "Text", 0.98)
                for w in words
                if str(w[4]).strip()
            ]
            pages.append(PageData(i + 1, float(pg.rect.width), float(pg.rect.height), toks, "Text", False, "", 0.98 if toks else 0.0, raw))
        doc.close()

        # Run WinOCR cleanly on scanned / corrupted Interims (`n_pages <= 26` or empty vector text)
        if need_ocr_flag and n_pages <= 26:
            try:
                pages, _ = extract_pdf_pages_tokens(
                    pdf_path=pdf_path,
                    file_sha256=sha,
                    ocr_cache_dir=ocr_cache_dir,
                    max_ocr_pages=10,
                    dpi=220,
                )
            except Exception:
                pass

        # Ensure report_end_hint falls back to PublicationDate-year rather than 2026-09-30!
        inv_mock = dict(task["InvRow"])
        pe_hint = str(inv_mock.get("FinancialPeriodEndHint") or "").strip()
        pub_ts = pd.to_datetime(inv_mock.get("PublicationDate"), errors="coerce")
        if not pe_hint and pd.notna(pub_ts):
            fye_m = int(inv_mock.get("FYEMonthHint") or 3)
            guess_yr = pub_ts.year if pub_ts.month >= fye_m else pub_ts.year - 1
            guess_d = 31 if fye_m in (3, 12) else 30
            pe_hint = f"{guess_yr}-{fye_m:02d}-{guess_d:02d}"
            inv_mock["FinancialPeriodEndHint"] = pe_hint

        if "CurrencyHint" not in inv_mock or not inv_mock["CurrencyHint"]:
            inv_mock["CurrencyHint"] = "LKR"

        raw_facts, info = sp3a.parse_financial_pdf(pages, inv_mock, {"fye_month_hint": int(inv_mock.get("FYEMonthHint") or 3)})

        # Filter residual unlabelled % variance / Note columns (`med_abs < 2500` while peer column on same page has `med >= 100,000`)
        if raw_facts:
            from statistics import median as _med
            page_col_vals: dict[tuple[str, int], dict[int, list[float]]] = {}
            for rf in raw_facts:
                if rf["MetricCode"] in ("EPS_BASIC", "EPS_DILUTED", "DIVIDEND_PER_SHARE") or rf.get("ColIndex", -1) < 0:
                    continue
                page_col_vals.setdefault((rf["StatementType"], int(rf["SourcePage"])), {}).setdefault(int(rf["ColIndex"]), []).append(abs(float(rf["OriginalValue"])))
            bad_page_cols: set[tuple[str, int, int]] = set()
            for (st_p, pg_p), col_map in page_col_vals.items():
                if len(col_map) >= 3:
                    col_meds = {cidx: _med(vs) for cidx, vs in col_map.items() if vs}
                    max_m = max(col_meds.values(), default=0.0)
                    if max_m >= 100_000.0:
                        for cidx, mval in col_meds.items():
                            if mval < 2500.0 and (max_m / max(1.0, mval)) >= 2500.0:
                                bad_page_cols.add((st_p, pg_p, cidx))
            if bad_page_cols:
                raw_facts = [
                    rf for rf in raw_facts
                    if (rf["StatementType"], int(rf["SourcePage"]), int(rf.get("ColIndex", -1))) not in bad_page_cols
                ]

        # Post-fix Annual report PeriodType/PeriodStart across all statements, per-share DPS unit & PeriodEnd sanity bounds
        is_annual_doc = str(inv_mock.get("ReportType", "")) == "Annual_Report"
        filtered_facts: list[dict[str, Any]] = []
        for f in raw_facts:
            pe_raw = f.get("PeriodEnd")
            if pd.notna(pe_raw):
                pe_ts = pd.Timestamp(pe_raw)
                # Guard against spurious non-reporting dates (`2009-05-30`) & future-dated PeriodEnd >= PublicationDate
                if pe_ts.year < 2010 or pe_ts.year > 2027:
                    continue
                if pd.notna(pub_ts) and (pe_ts >= pub_ts or (pub_ts - pe_ts).days > 1700):
                    continue
            f["DisclosureID"] = did
            if is_annual_doc:
                f["PeriodType"] = "Annual"
                if pd.notna(f.get("PeriodEnd")):
                    pe_ts = pd.Timestamp(f["PeriodEnd"])
                    f["FiscalQuarter"] = f"FY{f.get('FiscalYear') or pe_ts.year}"
                    if f["StatementType"] in ("Income Statement", "Cash Flow"):
                        f["PeriodStart"] = (pe_ts + pd.Timedelta(days=1)) - pd.DateOffset(months=12)
            if f["MetricCode"] == "DIVIDEND_PER_SHARE":
                f["Unit"] = f"{f['Currency']} per share"
                f["Multiplier"] = 1.0
                f["IsAmbiguousUnit"] = False
            # Serialize timestamps safely for JSON cache
            for tk in ("PeriodStart", "PeriodEnd"):
                if pd.notna(f.get(tk)):
                    f[tk] = str(pd.Timestamp(f[tk]).date())
                else:
                    f[tk] = None
            filtered_facts.append(f)
        raw_facts = filtered_facts

        stmts = info.get("statements_found", [])
        if len(raw_facts) >= 10 and len(stmts) >= 2:
            st = "SUCCESS"
        elif len(raw_facts) > 0:
            st = "PARTIAL_EXTRACTION"
        else:
            st = "FAILED"

        payload = {
            "DocumentID": did,
            "Status": st,
            "StatementsFound": stmts,
            "Facts": raw_facts,
            "Note": f"pages_scanned={len(pages)}/{n_pages}, stmts={stmts}",
        }
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    except Exception as exc:
        return {"DocumentID": did, "Status": "FAILED", "Facts": [], "Note": f"Parser exception: {str(exc)[:180]}"}


def extract_historical_pdfs(
    catalog_df: pd.DataFrame,
    repo_root: Path,
    parse_cache_dir: Path,
    ocr_cache_dir: Path,
    max_workers: int = 8,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Run cached multiprocessing spatial statement extraction across all newly acquired
    non-Phase-3A historical PDFs in `catalog_df`. Phase 3A documents remain untouched
    and are loaded verbatim later in `timeseries_and_pit.py`.
    """
    parse_cache_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache_dir.mkdir(parents=True, exist_ok=True)

    # Select acquired PDFs NOT already handled by Phase 3A
    acq_mask = (catalog_df["AcquisitionStatus"] == "REPORT_SUCCESSFULLY_ACQUIRED") & (~catalog_df["IsPhase3ALocal"])
    to_parse = catalog_df[acq_mask].copy()
    logger.info("Parsing %d newly acquired historical PDFs across %d CPU workers...", len(to_parse), max_workers)

    tasks: list[dict[str, Any]] = []
    for _, r in to_parse.iterrows():
        lp = str(r.get("LocalFilePath") or "").strip()
        if not lp:
            continue
        inv_row = {
            "DisclosureID": str(r["DocumentID"]),
            "CompanyCode": str(r.get("CompanyCode") or ""),
            "CompanyName": str(r.get("CompanyName") or ""),
            "ReportType": str(r.get("ReportType") or ""),
            "ReportingPeriodDescription": str(r.get("FileText") or ""),
            "FinancialPeriodEndHint": str(r.get("FinancialPeriodEndHint") or ""),
            "FYEMonthHint": int(r.get("FYEMonthHint") or 3),
            "PublicationDate": str(r["PublicationDate"]) if pd.notna(r.get("PublicationDate")) else "",
            "AvailabilityDate": str(r["AvailabilityDate"]) if pd.notna(r.get("AvailabilityDate")) else "",
            "CurrencyHint": "LKR",
        }
        tasks.append({
            "DocumentID": str(r["DocumentID"]),
            "AbsFilePath": str(repo_root / lp),
            "SHA256": str(r.get("SHA256") or ""),
            "CacheDir": str(parse_cache_dir),
            "OCRCacheDir": str(ocr_cache_dir),
            "InvRow": inv_row,
        })

    raw_facts_out: list[dict[str, Any]] = []
    status_by_did: dict[str, str] = {}
    notes_by_did: dict[str, str] = {}

    if tasks:
        # Single-process fallback if small batch (e.g. unit test < 12 files) avoids spawn overhead
        if len(tasks) < 12 or max_workers <= 1:
            for tk in tasks:
                res = _extract_single_historical_pdf(tk)
                status_by_did[res["DocumentID"]] = res["Status"]
                notes_by_did[res["DocumentID"]] = res.get("Note", "")
                raw_facts_out.extend(res["Facts"])
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                for idx, res in enumerate(ex.map(_extract_single_historical_pdf, tasks, chunksize=16), 1):
                    status_by_did[res["DocumentID"]] = res["Status"]
                    notes_by_did[res["DocumentID"]] = res.get("Note", "")
                    raw_facts_out.extend(res["Facts"])
                    if idx % 1000 == 0 or idx == len(tasks):
                        logger.info("  Parsed %d/%d PDFs -> cumulative newly extracted raw facts: %d", idx, len(tasks), len(raw_facts_out))

    # Restore Timestamp types fast via date.fromisoformat cache (100x faster than scalar pd.to_datetime)
    from datetime import date as _dt_date
    _ts_cache: dict[str, pd.Timestamp] = {}
    def _fast_ts(s: Any) -> Any:
        if not s:
            return pd.NaT
        st = str(s)
        hit = _ts_cache.get(st)
        if hit is not None:
            return hit
        try:
            ts = pd.Timestamp(_dt_date.fromisoformat(st[:10]))
            _ts_cache[st] = ts
            return ts
        except Exception:
            return pd.NaT

    for f in raw_facts_out:
        f["PeriodStart"] = _fast_ts(f.get("PeriodStart"))
        f["PeriodEnd"] = _fast_ts(f.get("PeriodEnd"))

    out_cat = catalog_df.copy()
    ext_statuses: list[str] = []
    for _, r in out_cat.iterrows():
        did = str(r["DocumentID"])
        if bool(r.get("IsPhase3ALocal", False)):
            # Inherited from Phase 3A verified pipeline (53523 Press Release = FAILED, 79 others = SUCCESS)
            ext_statuses.append("FAILED" if did == "53523" else "SUCCESS")
        elif did in status_by_did:
            ext_statuses.append(status_by_did[did])
        elif r["AcquisitionStatus"] == "REPORT_SUCCESSFULLY_ACQUIRED":
            ext_statuses.append("PENDING")
        else:
            ext_statuses.append("NOT_ACQUIRED")

    out_cat["ExtractionStatus"] = ext_statuses
    return raw_facts_out, out_cat
