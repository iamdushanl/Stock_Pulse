"""
Steps 3-6: Robust pilot extraction for 5 companies.
Handles 2-column and 4-column table layouts found in CSE interim reports.

Pilot: CFIN (Finance), MSL (Shipping/Materials), DIAL (Telecoms),
       ODEL (Retail), TAFL (Food & Beverage)
"""
from __future__ import annotations
import csv, json, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pdfplumber
import pyarrow as pa
import pyarrow.parquet as pq

# ── paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = Path("cse_live")
DOC_DIR  = WORK_DIR / "documents" / "financial_reports"
OUT_DIR  = WORK_DIR / "phase2_extraction"
TBL_DIR  = OUT_DIR / "extracted_tables"
RAW_TEXT = OUT_DIR / "raw_text"
for d in (OUT_DIR, TBL_DIR, RAW_TEXT):
    d.mkdir(parents=True, exist_ok=True)

PILOT_SYMBOLS = {"CFIN", "MSL", "DIAL", "ODEL", "TAFL"}
NOW_UTC = datetime.now(timezone.utc).isoformat()

# ── manifest ──────────────────────────────────────────────────────────────────
with open(WORK_DIR / "financial_reports_manifest.csv", encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))
with open(WORK_DIR / "all_disclosures_document_index.csv", encoding="utf-8-sig") as f:
    disc_index = {r["id"]: r for r in csv.DictReader(f)
                  if r.get("document_category") == "financial_reports"}

seen: set[str] = set()
pilot_rows = []
for row in manifest:
    sym = row.get("symbol", "")
    if sym in PILOT_SYMBOLS and sym not in seen:
        matches = list(DOC_DIR.glob(f"{row['id']}_*.pdf"))
        if matches:
            pilot_rows.append({"row": row, "pdf": matches[0],
                                "disc": disc_index.get(str(row["id"]), {})})
            seen.add(sym)

print(f"Pilot: {[r['row']['symbol'] for r in pilot_rows]}\n")

# ── number helpers ─────────────────────────────────────────────────────────────
_CLEAN_NUM   = re.compile(r"[\s]")
_PAREN_NUM   = re.compile(r"^\((.+)\)$")
_BARE_NUM    = re.compile(r"^-?\(?[\d,]+\.?\d*\)?$")

def parse_num(s: str) -> Optional[float]:
    if not s:
        return None
    s = _CLEAN_NUM.sub("", s.strip()).replace(",","")
    neg = _PAREN_NUM.match(s)
    if neg:
        s = "-" + neg.group(1).replace(",","")
    try:
        v = float(s)
        return v
    except ValueError:
        return None

# ── patterns ───────────────────────────────────────────────────────────────────
# Statement section detectors
IS_START  = re.compile(
    r"(statement\s+of\s+(profit\s+or\s+loss|comprehensive\s+income|income)"
    r"|income\s+statement|profit\s+(or|and)\s+loss)", re.I)
BS_START  = re.compile(
    r"(statement\s+of\s+)?financial\s+position|balance\s+sheet", re.I)
CF_START  = re.compile(
    r"statement\s+of\s+cash\s+flows?|cash\s+flow\s+statement", re.I)
EQ_START  = re.compile(r"statement\s+of\s+changes\s+in\s+equity", re.I)
NOTES_START = re.compile(r"^\s*notes?\s+(to\s+the|forming\s+part)", re.I)

# Metric patterns — ordered from most to least specific
METRIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    # --- Income Statement ---
    ("revenue",             re.compile(
        r"^revenue(\s+from\s+contracts?\s+with\s+customers)?$"
        r"|^turnover$|^net\s+revenue$|^total\s+revenue$|^gross\s+revenue$", re.I)),
    ("cost_of_sales",       re.compile(r"^cost\s+of\s+(sales|revenue|goods\s+sold|services?)$", re.I)),
    ("gross_profit",        re.compile(r"^gross\s+profit(\s*/\s*loss)?$", re.I)),
    ("other_income",        re.compile(r"^other\s+(operating\s+)?(income|income\s*/\s*expense)$", re.I)),
    ("distribution_expenses", re.compile(r"^(distribution|selling)\s+(expenses?|costs?)$", re.I)),
    ("admin_expenses",      re.compile(r"^(administrative|general\s+and\s+admin\w*)\s+expenses?$", re.I)),
    ("operating_profit",    re.compile(
        r"^(operating\s+profit|profit\s+from\s+operations"
        r"|results?\s+from\s+operations|operating\s+profit\s*/\s*loss"
        r"|operating\s+income)$", re.I)),
    ("finance_costs",       re.compile(
        r"^finance\s+(costs?|expenses?|charges?)$"
        r"|^interest\s+expense$|^net\s+finance\s+(costs?|income)", re.I)),
    ("finance_income",      re.compile(r"^finance\s+income$|^interest\s+income$", re.I)),
    ("profit_before_tax",   re.compile(
        r"^profit\s*(/\s*loss)?\s+before\s+(income\s+)?tax(ation)?(\s*\(pbt\))?$"
        r"|^(loss|profit)\s+before\s+tax$", re.I)),
    ("income_tax",          re.compile(r"^(income\s+)?tax(\s+expense|\s+charge|\s+credit)?$"
                                        r"|^total\s+income\s+tax", re.I)),
    ("net_profit",          re.compile(
        r"^profit\s*(/\s*loss)?\s+for\s+the\s+(period|year)$"
        r"|^net\s+profit(\s+for\s+the\s+period)?$"
        r"|^(loss|profit)\s+for\s+the\s+(period|year)$", re.I)),
    ("eps_basic",           re.compile(
        r"basic\s*(/\s*diluted)?\s*(loss\s*/\s*)?earnings?\s+per\s+share"
        r"|earnings?\s+per\s+share\s*[-–]\s*basic"
        r"|basic\s+eps", re.I)),
    # --- Balance Sheet ---
    ("cash_equivalents",    re.compile(
        r"^cash\s+and\s+(cash\s+)?equivalents?(\s+at\s+(end|beginning)\s+of\s+period)?$"
        r"|^cash\s+at\s+bank(\s+and\s+in\s+hand)?$", re.I)),
    ("borrowings",          re.compile(
        r"^(interest.bearing\s+)?loans?\s+and\s+borrowings?$"
        r"|^borrowings?$|^bank\s+(overdraft|loans?)$"
        r"|^(long.term|short.term)\s+(loans?|borrowings?)$", re.I)),
    ("total_assets",        re.compile(r"^total\s+assets$", re.I)),
    ("total_liabilities",   re.compile(r"^total\s+liabilities$", re.I)),
    ("total_equity",        re.compile(
        r"^(total\s+)?(equity|shareholders.?\s*equity|net\s+assets)$"
        r"|^equity\s+attributable\s+to", re.I)),
    # --- Cash Flow ---
    ("operating_cashflow",  re.compile(
        r"^net\s+cash\s+(from|generated\s+(from|by)|used\s+in)\s+operating"
        r"|^cash\s+(from|generated\s+(from|by)|used\s+in)\s+operating", re.I)),
    ("investing_cashflow",  re.compile(
        r"^net\s+cash\s+(from|used\s+in)\s+investing"
        r"|^cash\s+(from|used\s+in)\s+investing", re.I)),
    ("financing_cashflow",  re.compile(
        r"^net\s+cash\s+(from|used\s+in)\s+financing"
        r"|^cash\s+(from|used\s+in)\s+financing", re.I)),
    ("capex",               re.compile(
        r"purchase\s+of\s+(property|plant|equipment|PPE|right.of.use\s+assets?)"
        r"|capital\s+expenditure", re.I)),
    ("net_cash_change",     re.compile(
        r"^net\s+(increase|decrease)\s+in\s+cash(\s+and\s+(cash\s+)?equivalents?)?$", re.I)),
]

STMT_LABELS = {
    "income_statement": IS_START,
    "balance_sheet":    BS_START,
    "cash_flow":        CF_START,
}

UNIT_PAT  = re.compile(r"in\s+(thousands?|millions?|billions?|rs\.?\s*'?000|lkr\s+'?000)", re.I)
UNIT_INLINE = re.compile(r"(?:rs\.|lkr)\s*'?000", re.I)

def detect_unit(text: str) -> str:
    for m in UNIT_PAT.finditer(text[:4000]):
        raw = m.group(1).lower()
        if any(x in raw for x in ("thousand", "'000", "000")):
            return "LKR_thousands"
        if "million" in raw:
            return "LKR_millions"
        if "billion" in raw:
            return "LKR_billions"
    if UNIT_INLINE.search(text[:4000]):
        return "LKR_thousands"
    return "LKR_units"

def detect_consol(text: str) -> str:
    g = len(re.findall(r"\bgroup\b", text[:5000], re.I))
    c = len(re.findall(r"\bcompany\b|\bbank\b(?!\s+of)", text[:5000], re.I))
    if g > c:
        return "group"
    if c > g:
        return "company"
    return "unknown"

def detect_period(text: str):
    """Return (period_type, [period_end_dates])."""
    period_type = "unknown"
    for ptype, pat in [
        ("quarterly",   re.compile(r"(?:three|quarter)\s+months?\s+ended", re.I)),
        ("semi_annual", re.compile(r"six\s+months?\s+ended|half\s+year\s+ended", re.I)),
        ("nine_month",  re.compile(r"nine\s+months?\s+ended", re.I)),
        ("annual",      re.compile(r"(?:twelve|year)\s+months?\s+ended|year\s+ended", re.I)),
    ]:
        if pat.search(text[:4000]):
            period_type = ptype
            break
    dates = re.findall(
        r"(?:30|31)\s+(?:September|March|June|December|Sep|Mar|Jun|Dec)\w*\s+\d{4}"
        r"|(?:September|March|June|December|Sep|Mar|Jun|Dec)\w*\s+30\s*,?\s*\d{4}",
        text[:6000], re.I)
    seen_d, unique = set(), []
    for d in dates:
        norm = d.strip()
        if norm not in seen_d:
            seen_d.add(norm)
            unique.append(norm)
    return period_type, unique[:4]  # up to 4 column dates

# ── column-aware table extraction via pdfplumber words ─────────────────────────

NUM_RE = re.compile(r"^-?\(?[\d,]+\.?\d*\)?$")
PCT_RE = re.compile(r"^-?\d+%$")

def extract_rows_from_page(page) -> list[tuple[str, list[tuple[float, str]]]]:
    """
    Returns list of (label_text, [(x_center, value_str), ...]).
    Uses pdfplumber word-level bounding boxes.
    """
    words = page.extract_words(x_tolerance=4, y_tolerance=4,
                                keep_blank_chars=False,
                                use_text_flow=False,
                                extra_attrs=["x0", "x1", "top"])
    if not words:
        return []

    # Group words into lines by top coordinate
    lines: dict[float, list[dict]] = defaultdict(list)
    for w in words:
        # round top to nearest 2pt bucket
        bucket = round(w["top"] / 2) * 2
        lines[bucket].append(w)

    result = []
    for top in sorted(lines):
        row_words = sorted(lines[top], key=lambda w: w["x0"])
        # separate label (left) from numeric tokens (right)
        nums  = [(w, (w["x0"] + w["x1"]) / 2) for w in row_words if NUM_RE.match(w["text"])]
        texts = [w["text"] for w in row_words if not NUM_RE.match(w["text"]) and not PCT_RE.match(w["text"])]
        label = " ".join(texts).strip()
        if nums:
            result.append((label, [(x, w["text"]) for w, x in nums]))
    return result


def group_columns(all_rows: list[tuple[str, list[tuple[float, str]]]],
                  n_cols: int = 4) -> list[tuple[str, list[Optional[str]]]]:
    """
    Cluster numeric x-positions into n_cols buckets; align values to columns.
    """
    # collect all x positions
    xs = [x for _, nums in all_rows for x, _ in nums]
    if not xs:
        return [(lbl, []) for lbl, _ in all_rows]

    # simple kmeans-style bucket assignment
    xs_sorted = sorted(set(round(x) for x in xs))
    # merge xs within 20pt of each other
    buckets: list[float] = []
    for x in xs_sorted:
        if not buckets or x - buckets[-1] > 20:
            buckets.append(x)
        else:
            buckets[-1] = (buckets[-1] + x) / 2

    def assign_col(x):
        return min(range(len(buckets)), key=lambda i: abs(buckets[i] - x))

    result = []
    for label, nums in all_rows:
        cols: list[Optional[str]] = [None] * len(buckets)
        for x, v in nums:
            c = assign_col(x)
            cols[c] = v
        result.append((label, cols))
    return result


def match_metric(label: str) -> Optional[str]:
    label = label.strip()
    for metric, pat in METRIC_PATTERNS:
        if pat.search(label):
            return metric
    return None

# ── per-company extraction ─────────────────────────────────────────────────────

all_facts_raw: list[dict] = []
review_flags:  list[dict] = []

for entry in pilot_rows:
    row      = entry["row"]
    pdf_path = entry["pdf"]
    disc     = entry["disc"]
    sym      = row.get("symbol", "")
    nm       = row.get("name", "")
    rid      = str(row.get("id", ""))
    upl_date = row.get("uploadedDate", "")
    sha256   = disc.get("sha256", "")

    print(f"\n{'='*70}")
    print(f"  {sym} — {nm}")
    print(f"  {pdf_path.name}  ({pdf_path.stat().st_size//1024} KB)  |  uploaded: {upl_date}")

    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages

        # collect full text for metadata detection
        full_text = "\n".join(
            (pg.extract_text(x_tolerance=3, y_tolerance=3) or "") for pg in pages)
        # save raw text
        (RAW_TEXT / f"{rid}_{sym}.txt").write_text(full_text, encoding="utf-8")

        unit       = detect_unit(full_text)
        consol     = detect_consol(full_text)
        period_tp, period_dates = detect_period(full_text)

        print(f"  Unit: {unit}  |  Scope: {consol}  |  Period: {period_tp}")
        print(f"  Column dates: {period_dates}")

        # --- section split (track which section each page belongs to) ---
        section_pages: dict[str, list] = defaultdict(list)
        cur_sect = "preamble"
        for pg in pages:
            txt = pg.extract_text(x_tolerance=3, y_tolerance=3) or ""
            first_lines = "\n".join(txt.splitlines()[:8])
            if IS_START.search(first_lines):
                cur_sect = "income_statement"
            elif BS_START.search(first_lines):
                cur_sect = "balance_sheet"
            elif CF_START.search(first_lines):
                cur_sect = "cash_flow"
            elif EQ_START.search(first_lines) or NOTES_START.search(first_lines):
                cur_sect = "other"
            section_pages[cur_sect].append(pg)

        found_sects = [s for s in section_pages if s not in ("preamble", "other")]
        print(f"  Sections: {found_sects}")

        # --- extract each section ---
        facts_this: list[dict] = []

        for sect in ("income_statement", "balance_sheet", "cash_flow"):
            pages_in_sect = section_pages.get(sect, [])
            if not pages_in_sect:
                review_flags.append({"symbol": sym, "id": rid,
                    "flag": "SECTION_NOT_FOUND", "detail": sect,
                    "metric": sect, "value": None})
                print(f"    [{sect}] NOT FOUND")
                continue

            # aggregate word-level rows across section pages
            all_rows: list[tuple[str, list[tuple[float, str]]]] = []
            sect_text_parts = []
            for pg in pages_in_sect:
                pg_text = pg.extract_text(x_tolerance=3, y_tolerance=3) or ""
                sect_text_parts.append(pg_text)
                all_rows.extend(extract_rows_from_page(pg))

            sect_text = "\n".join(sect_text_parts)
            # save section text
            (TBL_DIR / f"{rid}_{sym}_{sect}.txt").write_text(sect_text, encoding="utf-8")

            # column alignment
            col_rows = group_columns(all_rows)

            # determine how many data columns (skip label col)
            max_cols = max((len(cols) for _, cols in col_rows), default=0)
            print(f"    [{sect}] {len(pages_in_sect)} page(s), {len(col_rows)} rows, {max_cols} data col(s)")

            # For 4-column layout: cols are typically
            #   [Q_current, Q_prior, 6M_current, 6M_prior]  or
            #   [current, prior]
            # We prefer 6-month figures (cols 2,3) for semi-annual; Q only for quarterly
            col_assignments: list[tuple[int, bool, str]] = []  # (col_idx, is_comparative, label)
            if max_cols >= 4 and period_tp in ("semi_annual", "nine_month"):
                # take 6-month cols (indices 2 & 3)
                col_assignments = [
                    (2, False, period_dates[2] if len(period_dates) > 2 else ""),
                    (3, True,  period_dates[3] if len(period_dates) > 3 else ""),
                ]
            elif max_cols >= 2:
                col_assignments = [
                    (0, False, period_dates[0] if len(period_dates) > 0 else ""),
                    (1, True,  period_dates[1] if len(period_dates) > 1 else ""),
                ]
            else:
                col_assignments = [
                    (0, False, period_dates[0] if len(period_dates) > 0 else ""),
                ]

            for label, cols in col_rows:
                metric = match_metric(label)
                if metric is None:
                    continue
                for col_idx, is_comp, col_date in col_assignments:
                    raw_val = cols[col_idx] if col_idx < len(cols) else None
                    num     = parse_num(raw_val) if raw_val else None
                    facts_this.append({
                        "company_code":        rid,
                        "company_name":        nm,
                        "symbol":              sym,
                        "period_end":          col_date,
                        "period_type":         period_tp,
                        "publication_date":    upl_date,
                        "statement_type":      sect,
                        "metric":              metric,
                        "raw_label":           label[:120],
                        "raw_value":           raw_val or "",
                        "value":               num,
                        "currency":            "LKR",
                        "unit":                unit,
                        "consolidation_scope": consol,
                        "comparative_flag":    is_comp,
                        "source_pdf":          pdf_path.name,
                        "sha256":              sha256,
                        "page_number":         "",
                        "extraction_method":   "pdfplumber_word_bbox",
                        "validation_status":   "pending",
                        "extraction_ts_utc":   NOW_UTC,
                    })

        # deduplicate: keep highest-value occurrence per (metric, comparative_flag)
        dedup: dict[tuple, dict] = {}
        for f in facts_this:
            key = (f["statement_type"], f["metric"], f["comparative_flag"])
            if key not in dedup or (f["value"] is not None and dedup[key]["value"] is None):
                dedup[key] = f
        facts_this = list(dedup.values())

        # print summary
        print(f"    Facts extracted (after dedup): {len(facts_this)}")
        for f in sorted(facts_this, key=lambda x: (x["statement_type"], x["metric"], x["comparative_flag"])):
            comp = "(comp)" if f["comparative_flag"] else "      "
            print(f"      {f['statement_type']:18} {f['metric']:22} {comp}  "
                  f"val={f['value']}  raw={f['raw_value']!r}")

        all_facts_raw.extend(facts_this)

# ── Step 5: Validation ─────────────────────────────────────────────────────────

def validate(facts: list[dict]) -> list[dict]:
    flags = []
    by_key: dict = defaultdict(dict)
    for f in facts:
        key = (f["symbol"], f["comparative_flag"])
        by_key[key][f["metric"]] = f["value"]

    for (sym, comp), metrics in by_key.items():
        tag = f"{sym} {'(comp)' if comp else ''}"
        def v(m): return metrics.get(m)

        # 1. Balance sheet equation
        ta, tl, te = v("total_assets"), v("total_liabilities"), v("total_equity")
        if all(x is not None for x in (ta, tl, te)):
            diff = abs(ta - (tl + te))
            tol  = max(abs(ta) * 0.02, 1)
            if diff > tol:
                flags.append({"symbol": sym, "comparative": comp,
                    "flag": "BS_EQUATION_MISMATCH",
                    "detail": f"Assets={ta:.0f} Liab={tl:.0f} Eq={te:.0f} diff={diff:.0f}",
                    "metric": "balance_sheet", "value": diff})

        # 2. EPS sanity
        eps, npat = v("eps_basic"), v("net_profit")
        if eps is not None and npat is not None and abs(npat) > 0:
            ratio = abs(eps) / abs(npat)
            if ratio > 10:
                flags.append({"symbol": sym, "comparative": comp,
                    "flag": "EPS_MAGNITUDE_WARNING",
                    "detail": f"EPS={eps} NetProfit={npat} ratio={ratio:.1f}",
                    "metric": "eps_basic", "value": eps})

        # 3. Gross profit <= revenue
        rev, gp = v("revenue"), v("gross_profit")
        if rev is not None and gp is not None and rev > 0 and gp > rev * 1.05:
            flags.append({"symbol": sym, "comparative": comp,
                "flag": "GROSS_PROFIT_EXCEEDS_REVENUE",
                "detail": f"Rev={rev:.0f} GP={gp:.0f}",
                "metric": "gross_profit", "value": gp})

        # 4. Net profit sign matches operating profit (at least directionally)
        op, npat2 = v("operating_profit"), v("net_profit")
        if op is not None and npat2 is not None:
            if (op > 0) != (npat2 > 0):
                flags.append({"symbol": sym, "comparative": comp,
                    "flag": "SIGN_MISMATCH_OP_VS_NET",
                    "detail": f"OperatingProfit={op:.0f} NetProfit={npat2:.0f}",
                    "metric": "net_profit", "value": npat2})

        # 5. Missing key metrics
        for m in ("revenue", "total_assets", "net_profit", "total_equity"):
            if v(m) is None:
                flags.append({"symbol": sym, "comparative": comp,
                    "flag": "METRIC_NOT_EXTRACTED",
                    "detail": f"{m} not found in PDF text",
                    "metric": m, "value": None})

    # apply status
    flagged_bad = {(f["symbol"], f["comparative"])
                   for f in flags if f["flag"] not in
                   ("METRIC_NOT_EXTRACTED", "SIGN_MISMATCH_OP_VS_NET")}
    for f in facts:
        key = (f["symbol"], f["comparative_flag"])
        if f["value"] is None:
            f["validation_status"] = "missing"
        elif key in flagged_bad:
            f["validation_status"] = "review"
        else:
            f["validation_status"] = "ok"
    return flags

review_flags.extend(validate(all_facts_raw))

# ── Step 6: Outputs ────────────────────────────────────────────────────────────

# financial_facts_raw.parquet
if all_facts_raw:
    schema_types = {
        "value": pa.float64(),
        "comparative_flag": pa.bool_(),
    }
    table = pa.table({
        k: pa.array(
            [float(r[k]) if k == "value" and r[k] is not None else
             bool(r[k])  if k == "comparative_flag" else
             (None if r[k] is None else str(r[k]))
             for r in all_facts_raw],
            type=schema_types.get(k, pa.string())
        )
        for k in all_facts_raw[0]
    })
    out_pq = OUT_DIR / "financial_facts_raw.parquet"
    pq.write_table(table, out_pq)
    print(f"\nParquet: {out_pq}  ({len(all_facts_raw)} rows)")

# financial_facts_review.csv
rev_csv = OUT_DIR / "financial_facts_review.csv"
if all_facts_raw:
    keys = list(all_facts_raw[0].keys())
    with open(rev_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_facts_raw)
    print(f"Review CSV: {rev_csv}")

# validation_report.json
val_report = {
    "generated_at_utc": NOW_UTC,
    "pilot_symbols": sorted(PILOT_SYMBOLS),
    "total_facts_extracted": len(all_facts_raw),
    "total_facts_ok":        sum(1 for f in all_facts_raw if f["validation_status"] == "ok"),
    "total_facts_missing":   sum(1 for f in all_facts_raw if f["validation_status"] == "missing"),
    "total_facts_review":    sum(1 for f in all_facts_raw if f["validation_status"] == "review"),
    "total_review_flags": len(review_flags),
    "flags_by_type": defaultdict(int),
    "flags": review_flags,
    "notes": [
        "validation_status='missing': metric label not found in extracted PDF text.",
        "validation_status='review': a cross-check failed — human review required before use.",
        "validation_status='ok': no automated check failed — verify a sample manually.",
        "Units are as stated in the PDF header; most CSE interim reports use LKR_thousands.",
        "Four-column reports: 6-month cumulative values preferred over quarterly where both present.",
        "Comparative period = same period prior year as reported in this filing.",
        "publication_date = uploadedDate on CSE system; not the period end date.",
    ],
}
for f in review_flags:
    val_report["flags_by_type"][f["flag"]] += 1
val_report["flags_by_type"] = dict(val_report["flags_by_type"])

with open(OUT_DIR / "validation_report.json", "w", encoding="utf-8") as f:
    json.dump(val_report, f, indent=2, ensure_ascii=False, default=str)
print(f"Validation report: {OUT_DIR / 'validation_report.json'}")

print(f"\n{'='*70}")
print(f"PILOT COMPLETE")
print(f"  Companies     : {[r['row']['symbol'] for r in pilot_rows]}")
print(f"  Facts total   : {len(all_facts_raw)}")
print(f"  Status ok     : {sum(1 for f in all_facts_raw if f['validation_status']=='ok')}")
print(f"  Status missing: {sum(1 for f in all_facts_raw if f['validation_status']=='missing')}")
print(f"  Status review : {sum(1 for f in all_facts_raw if f['validation_status']=='review')}")
print(f"  Flags         : {len(review_flags)}")
print(f"\nOutputs in: {OUT_DIR}")
print(f"  financial_facts_raw.parquet")
print(f"  financial_facts_review.csv")
print(f"  validation_report.json")
print(f"  extracted_tables/  ({len(list(TBL_DIR.glob('*.txt')))} section files)")
print(f"  raw_text/          ({len(list(RAW_TEXT.glob('*.txt')))} text files)")
