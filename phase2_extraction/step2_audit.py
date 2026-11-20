"""
Step 2: PDF Extraction Audit — all 80 financial report PDFs.

Outputs: extraction_audit.csv, raw_text/<id>_<sym>.txt
"""
import csv, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

import fitz          # PyMuPDF
import pdfplumber

WORK_DIR   = Path("cse_live")
DOC_DIR    = WORK_DIR / "documents" / "financial_reports"
OUT_DIR    = WORK_DIR / "phase2_extraction"
RAW_TEXT   = OUT_DIR / "raw_text"
AUDIT_CSV  = OUT_DIR / "extraction_audit.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXT.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

STMT_PATTERNS = {
    "income_statement": re.compile(
        r"(income\s+statement|statement\s+of\s+(comprehensive\s+)?income"
        r"|profit\s+(or|and)\s+loss|profit\s*&\s*loss)", re.I),
    "balance_sheet": re.compile(
        r"(balance\s+sheet|statement\s+of\s+financial\s+position"
        r"|financial\s+position)", re.I),
    "cash_flow": re.compile(
        r"(cash\s+flow|statement\s+of\s+cash\s+flow)", re.I),
    "notes": re.compile(r"\bnotes?\b.*\bfinancial\b|\baccounting\s+polic", re.I),
}

TABLE_HINT = re.compile(r"\b(Rs\.|LKR|000|,\d{3}|\(\d)", re.I)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()

def classify_pages(doc):
    """Return per-page dict: char_count, is_image_only, stmt_types[]."""
    pages = []
    for i, page in enumerate(doc):
        text  = page.get_text("text")
        chars = len(text.strip())
        is_img = chars < 50 and len(page.get_images()) > 0
        stmts = [k for k, pat in STMT_PATTERNS.items() if pat.search(text)]
        has_tbl = bool(TABLE_HINT.search(text))
        pages.append({
            "page_no": i + 1,
            "char_count": chars,
            "is_image_only": is_img,
            "statement_types": stmts,
            "has_table_hint": has_tbl,
        })
    return pages

def extract_full_text(pdf_path: Path) -> str:
    """Extract all text via pdfplumber (better layout handling)."""
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=3, y_tolerance=3)
            if t:
                lines.append(t)
    return "\n".join(lines)

# ── load manifest ─────────────────────────────────────────────────────────────

with open(WORK_DIR / "financial_reports_manifest.csv", encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))

with open(WORK_DIR / "all_disclosures_document_index.csv", encoding="utf-8-sig") as f:
    disc_index = {}
    for r in csv.DictReader(f):
        if r.get("document_category") == "financial_reports":
            disc_index[r["id"]] = r

audit_rows = []
errors     = []

print(f"Auditing {len(manifest)} PDFs …")

for idx, row in enumerate(manifest, 1):
    rid     = str(row.get("id", ""))
    sym     = row.get("symbol", "")
    nm      = row.get("name", "")
    man_dt  = row.get("manualDate", "")
    upl_dt  = row.get("uploadedDate", "")
    ftxt    = row.get("fileText", "")
    matches = list(DOC_DIR.glob(f"{rid}_*.pdf"))
    if not matches:
        errors.append({"id": rid, "symbol": sym, "error": "PDF_NOT_FOUND"})
        continue
    pdf_path = matches[0]
    d = disc_index.get(rid, {})

    try:
        # ── PyMuPDF pass ──
        doc  = fitz.open(str(pdf_path))
        meta = doc.metadata
        pages_info = classify_pages(doc)
        total_pages   = len(pages_info)
        total_chars   = sum(p["char_count"] for p in pages_info)
        image_pages   = sum(1 for p in pages_info if p["is_image_only"])
        table_pages   = sum(1 for p in pages_info if p["has_table_hint"])
        is_scan       = image_pages > total_pages * 0.5
        stmt_coverage = {}
        for p in pages_info:
            for s in p["statement_types"]:
                stmt_coverage.setdefault(s, []).append(p["page_no"])

        # ── pdfplumber pass — full text ──
        full_text = ""
        plumber_error = ""
        try:
            full_text = extract_full_text(pdf_path)
        except Exception as pe:
            plumber_error = str(pe)[:120]

        # ── save raw text ──
        txt_path = RAW_TEXT / f"{rid}_{sym}.txt"
        txt_path.write_text(full_text, encoding="utf-8")

        audit_row = {
            "id": rid,
            "symbol": sym,
            "company_name": nm,
            "file_text": ftxt,
            "period_end_date": "",   # filled from timestamp below
            "uploaded_date": upl_dt,
            "pdf_filename": pdf_path.name,
            "size_kb": round(pdf_path.stat().st_size / 1024, 1),
            "sha256": d.get("sha256", sha256_file(pdf_path)),
            "total_pages": total_pages,
            "total_chars": total_chars,
            "image_only_pages": image_pages,
            "table_hint_pages": table_pages,
            "is_likely_scan": is_scan,
            "has_income_statement": bool(stmt_coverage.get("income_statement")),
            "income_stmt_pages": json.dumps(stmt_coverage.get("income_statement", [])),
            "has_balance_sheet": bool(stmt_coverage.get("balance_sheet")),
            "balance_sheet_pages": json.dumps(stmt_coverage.get("balance_sheet", [])),
            "has_cash_flow": bool(stmt_coverage.get("cash_flow")),
            "cash_flow_pages": json.dumps(stmt_coverage.get("cash_flow", [])),
            "has_notes": bool(stmt_coverage.get("notes")),
            "pdf_title": meta.get("title", ""),
            "pdf_author": meta.get("author", ""),
            "pdf_creator": meta.get("creator", ""),
            "plumber_error": plumber_error,
            "extraction_method": "pdfplumber+pymupdf",
            "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        # parse period_end_date from timestamp
        try:
            audit_row["period_end_date"] = datetime.fromtimestamp(
                int(man_dt) / 1000).strftime("%Y-%m-%d")
        except Exception:
            audit_row["period_end_date"] = man_dt

        audit_rows.append(audit_row)
        doc.close()

        status = "SCAN" if is_scan else "OK"
        print(f"[{idx:3}/{len(manifest)}] {sym:8} {total_pages:3}pp  {total_chars:7}ch  img:{image_pages}  tbl:{table_pages}  "
              f"IS:{audit_row['has_income_statement']}  BS:{audit_row['has_balance_sheet']}  CF:{audit_row['has_cash_flow']}  {status}")

    except Exception as e:
        errors.append({"id": rid, "symbol": sym, "error": str(e)[:200]})
        print(f"[{idx:3}/{len(manifest)}] {sym:8} ERROR: {e}", file=sys.stderr)

# ── write audit CSV ──
if audit_rows:
    keys = list(audit_rows[0].keys())
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(audit_rows)
    print(f"\nAudit CSV written: {AUDIT_CSV}  ({len(audit_rows)} rows)")

# ── summary ──
scan_count = sum(1 for r in audit_rows if r["is_likely_scan"])
has_is     = sum(1 for r in audit_rows if r["has_income_statement"])
has_bs     = sum(1 for r in audit_rows if r["has_balance_sheet"])
has_cf     = sum(1 for r in audit_rows if r["has_cash_flow"])

print(f"\n=== Audit Summary ===")
print(f"  Total audited  : {len(audit_rows)}")
print(f"  Likely scans   : {scan_count}")
print(f"  Has IS pages   : {has_is}")
print(f"  Has BS pages   : {has_bs}")
print(f"  Has CF pages   : {has_cf}")
print(f"  Errors         : {len(errors)}")
if errors:
    for e in errors:
        print(f"    {e}")

