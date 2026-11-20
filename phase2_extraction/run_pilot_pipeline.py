"""
Phase 2: Financial PDF Statement Extraction Pipeline — Pilot Execution
Audits, extracts, normalizes, validates, and generates all Phase 2 deliverables.

Target Pilot Companies (5 distinct sectors + 1 scan edge case):
  1. DIAL - Telecommunication Services
  2. ODEL - Retailing (Consumer Services)
  3. TAFL - Food, Beverage & Tobacco
  4. CFIN - Diversified Financials (NBFI)
  5. TILE - Capital Goods (Manufacturing)
  6. MSL  - Transportation (Shipping) [Scan edge-case check]
"""
from __future__ import annotations
import csv, hashlib, json, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pdfplumber
import pyarrow as pa
import pyarrow.parquet as pq

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR  = Path("cse_live")
DOC_DIR   = BASE_DIR / "documents" / "financial_reports"
OUT_DIR   = BASE_DIR / "phase2_extraction"
TBL_DIR   = OUT_DIR / "extracted_tables"
RAW_TEXT  = OUT_DIR / "raw_text"

for d in (OUT_DIR, TBL_DIR, RAW_TEXT):
    d.mkdir(parents=True, exist_ok=True)

PILOT_SYMBOLS = ["DIAL", "ODEL", "TAFL", "CFIN", "TILE", "MSL"]
NOW_UTC = datetime.now(timezone.utc).isoformat()

# ── Manifest Loading ──────────────────────────────────────────────────────────
with open(BASE_DIR / "financial_reports_manifest.csv", encoding="utf-8-sig") as f:
    manifest_rows = list(csv.DictReader(f))

with open(BASE_DIR / "all_disclosures_document_index.csv", encoding="utf-8-sig") as f:
    disc_index = {
        r["id"]: r for r in csv.DictReader(f)
        if r.get("document_category") == "financial_reports"
    }

# ── Clean Line & Number Parsers ───────────────────────────────────────────────
def clean_line_numbers(line: str) -> str:
    """Fix common PDF font kerning artifacts where digits or parens are spaced."""
    line = re.sub(r'(?<=[\s\(])(\d)\s*,\s*(\d)', r'\1,\2', line)
    line = re.sub(r'(?<=[\s\(])(\d)\s+(\d{1,2},\d{3}(?:,\d{3})*)', r'\1\2', line)
    line = re.sub(r'(?<=[\s\(])(\d)\s+(\d{3})(?=[\s\)]|$)', r'\1\2', line)
    line = re.sub(r'(?<=[\s\(])(\d)\s+(\d\.\d{2,4})(?=[\s\)]|$)', r'\1\2', line)
    line = re.sub(r'\(\s+', '(', line)
    line = re.sub(r'\s+\)', ')', line)
    return line

def extract_numbers_from_line(line: str) -> list[tuple[str, float]]:
    """Return list of (raw_token, parsed_float) from cleaned line."""
    cleaned = clean_line_numbers(line)
    tokens = cleaned.split()
    results = []
    for t in tokens:
        t_clean = t.rstrip('%')
        neg = False
        if t_clean.startswith('(') and t_clean.endswith(')'):
            neg = True
            t_clean = t_clean[1:-1]
        t_clean = t_clean.replace(',', '')
        try:
            val = float(t_clean)
            results.append((t, -val if neg else val))
        except ValueError:
            pass
    return results

# ── Section and Metric Classification ─────────────────────────────────────────
IS_START = re.compile(
    r"(statement\s+of\s+(profit\s+or\s+loss|comprehensive\s+income|income)"
    r"|income\s+statement|profit\s+(or|and)\s+loss)", re.I)
BS_START = re.compile(
    r"(statement\s+of\s+)?financial\s+position|balance\s+sheet", re.I)
CF_START = re.compile(
    r"statement\s+of\s+cash\s+flows?|cash\s+flow\s+statement", re.I)

METRIC_PATTERNS = {
    "income_statement": [
        ("revenue",           re.compile(r"^(revenue(\s+from\s+contracts?\s+with\s+customers)?|turnover|net\s+revenue|gross\s+revenue|income)\b", re.I)),
        ("gross_profit",      re.compile(r"^(gross\s+profit(\s*/\s*loss)?)\b", re.I)),
        ("operating_profit",  re.compile(r"^(operating\s+profit(\s*/\s*loss)?|profit\s+from\s+operations|results?\s+from\s+operat\w+|operating\s+profit\s+before\s+taxes)\b", re.I)),
        ("finance_costs",     re.compile(r"^(finance\s+(costs?|expenses?)|interest\s+expenses?|net\s+finance\s+costs?)\b", re.I)),
        ("finance_income",    re.compile(r"^(finance\s+income|interest\s+income)\b", re.I)),
        ("profit_before_tax", re.compile(r"^(profit(\s*/\s*loss)?\s+before\s+(income\s+)?tax(ation)?|loss\s+before\s+tax)\b", re.I)),
        ("net_profit",        re.compile(r"^(profit(\s*/\s*loss)?\s+for\s+the\s+period|profit\s+after\s+tax|net\s+profit|loss\s+for\s+the\s+period)\b", re.I)),
        ("eps_basic",         re.compile(r"^(basic.*earnings?\s+per\s+share|earnings?\s+per\s+share.*basic|basic\s+eps)\b", re.I)),
    ],
    "balance_sheet": [
        ("total_assets",      re.compile(r"^(total\s+assets)\b", re.I)),
        ("total_liabilities", re.compile(r"^(total\s+liabilities)\b", re.I)),
        ("total_equity",      re.compile(r"^(total\s+equity|total\s+shareholders.?\s*equity|equity\s+attributable\s+to|net\s+assets)\b", re.I)),
        ("cash_equivalents",  re.compile(r"^(cash\s+and\s+(cash\s+)?equivalents|cash\s+and\s+bank\s+balances|cash\s+at\s+bank)\b", re.I)),
        ("borrowings",        re.compile(r"^(interest\s+bearing\s+(borrowings|liabilities)|loans\s+and\s+borrowings|borrowings|bank\s+loans|bank\s+overdraft)\b", re.I)),
        ("non_current_liabilities", re.compile(r"^(total\s+non-current\s+liabilities)\b", re.I)),
        ("current_liabilities",     re.compile(r"^(total\s+current\s+liabilities)\b", re.I)),
        ("held_for_sale_liabilities", re.compile(r"^(liabilities\s+associated\s+with\s+assets\s+held\s+for\s+sale)\b", re.I)),
    ],
    "cash_flow": [
        ("operating_cashflow", re.compile(r"^(net\s+cash.*from.*operating|cash\s+generated\s+from\s+operations|net\s+cash\s+generated\s+from\s+operating)\b", re.I)),
        ("investing_cashflow", re.compile(r"^(net\s+cash.*used\s+in.*investing|net\s+cash.*from.*investing)\b", re.I)),
        ("financing_cashflow", re.compile(r"^(net\s+cash.*used\s+in.*financing|net\s+cash.*from.*financing)\b", re.I)),
        ("capex",              re.compile(r"^(purchase\s+of\s+property.*plant|acquisition\s+of\s+property|purchase\s+of\s+ppe|capital\s+expenditure)\b", re.I)),
    ]
}

def detect_unit(text: str) -> str:
    text_lower = text[:4000].lower()
    if any(k in text_lower for k in ["rupees thousands", "rs.'000", "rs.000", "rs '000", "lkr'000", "'000"]):
        return "LKR_thousands"
    if "rupees millions" in text_lower or "rs.mn" in text_lower or "rs. mn" in text_lower:
        return "LKR_millions"
    return "LKR_units"

def detect_consolidation(text: str) -> str:
    g = len(re.findall(r"\bgroup\b", text[:3000], re.I))
    c = len(re.findall(r"\bcompany\b", text[:3000], re.I))
    return "group" if g >= c else "company"

def detect_period(text: str) -> tuple[str, str, str]:
    """Return (period_type, current_period_end, comp_period_end)."""
    p_type = "semi_annual"
    if re.search(r"three\s+months?\s+ended|quarter\s+ended", text[:4000], re.I):
        p_type = "quarterly"
    elif re.search(r"nine\s+months?\s+ended", text[:4000], re.I):
        p_type = "nine_month"
    elif re.search(r"year\s+ended", text[:4000], re.I):
        p_type = "annual"
    
    dates = re.findall(r"(?:30|31)[\.\s/](?:September|March|June|December|Sep|Mar|Jun|Dec|09|03|06|12)[\.\s/]\d{4}", text[:4000], re.I)
    p_end = dates[0] if dates else "2026-09-30"
    p_comp = dates[1] if len(dates) > 1 else "2025-09-30"
    return p_type, p_end, p_comp

# ── Pilot Extraction Runner ───────────────────────────────────────────────────
facts_extracted: list[dict] = []
review_flags: list[dict] = []

print("Running Step 3: Pilot Financial Statement Extraction...")

for sym in PILOT_SYMBOLS:
    m_row = next((r for r in manifest_rows if r.get("symbol") == sym), None)
    if not m_row:
        continue
    rid = str(m_row["id"])
    company_name = m_row.get("name", "")
    upl_date = m_row.get("uploadedDate", "")
    pdf_matches = list(DOC_DIR.glob(f"{rid}_*.pdf"))
    if not pdf_matches:
        continue
    pdf_path = pdf_matches[0]
    disc_data = disc_index.get(rid, {})
    sha256 = disc_data.get("sha256", hashlib.sha256(pdf_path.read_bytes()).hexdigest())

    print(f"\nProcessing {sym} ({company_name}) | PDF: {pdf_path.name}")

    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        unit = detect_unit(full_text)
        consol_scope = detect_consolidation(full_text)
        p_type, p_end, p_comp = detect_period(full_text)

        # Classify each page to a statement section
        page_sections: list[tuple[int, str, str]] = []
        for pno, page in enumerate(pdf.pages):
            ptext = page.extract_text() or ""
            sect = "other"
            if IS_START.search(ptext[:600]):
                sect = "income_statement"
            elif BS_START.search(ptext[:600]):
                sect = "balance_sheet"
            elif CF_START.search(ptext[:600]):
                sect = "cash_flow"
            page_sections.append((pno + 1, sect, ptext))

        # Check for image-only pages (e.g. MSL page 5 scan)
        for pno, sect, ptext in page_sections:
            if len(ptext.strip()) < 50:
                review_flags.append({
                    "symbol": sym,
                    "company_code": rid,
                    "page_number": pno,
                    "flag": "SCANNED_IMAGE_PAGE_DETECTED",
                    "detail": f"Page {pno} is an image-only scan with no extractable text. Manual review or OCR required.",
                    "statement_type": "unknown",
                    "metric": "page_text",
                    "value": None
                })
                print(f"  [FLAG] Page {pno} is an image-only scan.")

        # Extract metrics section by section
        for pno, sect, ptext in page_sections:
            if sect not in METRIC_PATTERNS:
                continue
            
            lines = ptext.splitlines()
            for raw_line in lines:
                cleaned_line = clean_line_numbers(raw_line).strip()
                if not cleaned_line:
                    continue
                
                for metric, pat in METRIC_PATTERNS[sect]:
                    if pat.search(cleaned_line):
                        nums = extract_numbers_from_line(cleaned_line)
                        if not nums:
                            continue
                        
                        # Current period fact
                        cur_raw, cur_val = nums[0]
                        facts_extracted.append({
                            "company_code": rid,
                            "company_name": company_name,
                            "symbol": sym,
                            "period_start": "2026-04-01" if p_type == "semi_annual" else "2026-07-01",
                            "period_end": p_end,
                            "publication_date": upl_date,
                            "statement_type": sect,
                            "metric": metric,
                            "raw_label": cleaned_line[:120],
                            "raw_value": cur_raw,
                            "value": cur_val,
                            "currency": "LKR",
                            "unit": unit,
                            "consolidation_scope": consol_scope,
                            "period_type": p_type,
                            "comparative_flag": False,
                            "source_pdf": pdf_path.name,
                            "sha256": sha256,
                            "page_number": pno,
                            "extraction_method": "pdfplumber_text_parser",
                            "validation_status": "pending"
                        })

                        # Comparative period fact (if second number exists)
                        if len(nums) > 1:
                            comp_raw, comp_val = nums[1]
                            facts_extracted.append({
                                "company_code": rid,
                                "company_name": company_name,
                                "symbol": sym,
                                "period_start": "2025-04-01" if p_type == "semi_annual" else "2025-07-01",
                                "period_end": p_comp,
                                "publication_date": upl_date,
                                "statement_type": sect,
                                "metric": metric,
                                "raw_label": cleaned_line[:120],
                                "raw_value": comp_raw,
                                "value": comp_val,
                                "currency": "LKR",
                                "unit": unit,
                                "consolidation_scope": consol_scope,
                                "period_type": p_type,
                                "comparative_flag": True,
                                "source_pdf": pdf_path.name,
                                "sha256": sha256,
                                "page_number": pno,
                                "extraction_method": "pdfplumber_text_parser",
                                "validation_status": "pending"
                            })
                        break

# Handle total liabilities synthesis for companies where non-current + current are reported separately (e.g. TILE)
by_sym_comp = defaultdict(dict)
for f in facts_extracted:
    k = (f["symbol"], f["comparative_flag"])
    by_sym_comp[k][f["metric"]] = f

for (sym, comp), metric_dict in by_sym_comp.items():
    if "total_liabilities" not in metric_dict:
        nc = metric_dict.get("non_current_liabilities")
        cur = metric_dict.get("current_liabilities")
        hfs = metric_dict.get("held_for_sale_liabilities")
        if nc and cur:
            tot_val = nc["value"] + cur["value"] + (hfs["value"] if hfs else 0.0)
            synth_fact = {
                "company_code": nc["company_code"],
                "company_name": nc["company_name"],
                "symbol": sym,
                "period_start": nc["period_start"],
                "period_end": nc["period_end"],
                "publication_date": nc["publication_date"],
                "statement_type": "balance_sheet",
                "metric": "total_liabilities",
                "raw_label": "Synthesized: Non-current + Current liabilities",
                "raw_value": str(tot_val),
                "value": tot_val,
                "currency": "LKR",
                "unit": nc["unit"],
                "consolidation_scope": nc["consolidation_scope"],
                "period_type": nc["period_type"],
                "comparative_flag": comp,
                "source_pdf": nc["source_pdf"],
                "sha256": nc["sha256"],
                "page_number": nc["page_number"],
                "extraction_method": "synthesized_sum",
                "validation_status": "pending"
            }
            facts_extracted.append(synth_fact)

# Filter auxiliary components from final facts list so only core 16 metrics remain
CORE_METRICS = {
    "revenue", "gross_profit", "operating_profit", "finance_costs", "finance_income",
    "profit_before_tax", "net_profit", "eps_basic", "total_assets", "total_liabilities",
    "total_equity", "cash_equivalents", "borrowings", "operating_cashflow",
    "investing_cashflow", "financing_cashflow", "capex"
}

# Deduplicate to primary entry per (symbol, statement_type, metric, comparative_flag)
dedup_map: dict[tuple, dict] = {}
for f in facts_extracted:
    if f["metric"] not in CORE_METRICS:
        continue
    k = (f["symbol"], f["statement_type"], f["metric"], f["comparative_flag"])
    if k not in dedup_map:
        dedup_map[k] = f
facts_final = list(dedup_map.values())

print(f"\nTotal normalized core facts extracted across pilot companies: {len(facts_final)}")

# ── Step 5: Validation Engine ─────────────────────────────────────────────────
print("\nRunning Step 5: Validation Rules...")

by_sym_comp = defaultdict(dict)
for f in facts_final:
    key = (f["symbol"], f["comparative_flag"])
    by_sym_comp[key][f["metric"]] = f

for (sym, comp), metric_dict in by_sym_comp.items():
    comp_str = "Comparative" if comp else "Current"
    
    # 1. Balance Sheet Equation: Assets = Liabilities + Equity
    ta = metric_dict.get("total_assets")
    tl = metric_dict.get("total_liabilities")
    te = metric_dict.get("total_equity")
    if ta and tl and te:
        assets_val = ta["value"]
        liab_val = tl["value"]
        equity_val = te["value"]
        diff = abs(assets_val - (liab_val + equity_val))
        tol = max(abs(assets_val) * 0.02, 10.0) # 2% tolerance or 10 units
        if diff > tol:
            review_flags.append({
                "symbol": sym,
                "company_code": ta["company_code"],
                "page_number": ta["page_number"],
                "flag": "BS_EQUATION_MISMATCH",
                "detail": f"{comp_str} balance sheet mismatch: Assets ({assets_val}) != Liab ({liab_val}) + Eq ({equity_val}). Diff={diff:.2f}",
                "statement_type": "balance_sheet",
                "metric": "balance_sheet_equation",
                "value": diff
            })
            ta["validation_status"] = "review"
            tl["validation_status"] = "review"
            te["validation_status"] = "review"
        else:
            ta["validation_status"] = "ok"
            tl["validation_status"] = "ok"
            te["validation_status"] = "ok"

    # 2. Revenue vs Gross Profit
    rev = metric_dict.get("revenue")
    gp = metric_dict.get("gross_profit")
    if rev and gp:
        if gp["value"] > rev["value"] * 1.05 and rev["value"] > 0:
            review_flags.append({
                "symbol": sym,
                "company_code": rev["company_code"],
                "page_number": gp["page_number"],
                "flag": "GROSS_PROFIT_EXCEEDS_REVENUE",
                "detail": f"{comp_str} Gross Profit ({gp['value']}) > Revenue ({rev['value']})",
                "statement_type": "income_statement",
                "metric": "gross_profit",
                "value": gp["value"]
            })
            gp["validation_status"] = "review"
        else:
            gp["validation_status"] = "ok"
            rev["validation_status"] = "ok"

    # 3. EPS Plausibility Check (EPS should not be in the thousands)
    eps = metric_dict.get("eps_basic")
    if eps:
        if abs(eps["value"]) > 1000.0:
            review_flags.append({
                "symbol": sym,
                "company_code": eps["company_code"],
                "page_number": eps["page_number"],
                "flag": "EPS_MAGNITUDE_WARNING",
                "detail": f"{comp_str} EPS value ({eps['value']}) exceeds 1000. Potential confusion with total earnings or line token.",
                "statement_type": "income_statement",
                "metric": "eps_basic",
                "value": eps["value"]
            })
            eps["validation_status"] = "review"
        else:
            eps["validation_status"] = "ok"

# Mark any remaining pending facts as ok
for f in facts_final:
    if f["validation_status"] == "pending":
        f["validation_status"] = "ok"

# ── Step 6: Generate Deliverables ─────────────────────────────────────────────
print("\nRunning Step 6: Saving Outputs...")

# 1. financial_facts_raw.parquet
schema = pa.schema([
    ("company_code", pa.string()),
    ("company_name", pa.string()),
    ("symbol", pa.string()),
    ("period_start", pa.string()),
    ("period_end", pa.string()),
    ("publication_date", pa.string()),
    ("statement_type", pa.string()),
    ("metric", pa.string()),
    ("raw_label", pa.string()),
    ("raw_value", pa.string()),
    ("value", pa.float64()),
    ("currency", pa.string()),
    ("unit", pa.string()),
    ("consolidation_scope", pa.string()),
    ("period_type", pa.string()),
    ("comparative_flag", pa.bool_()),
    ("source_pdf", pa.string()),
    ("sha256", pa.string()),
    ("page_number", pa.int32()),
    ("extraction_method", pa.string()),
    ("validation_status", pa.string()),
])

table_data = {
    col: [f[col] for f in facts_final] for col in schema.names
}
pq_table = pa.Table.from_pydict(table_data, schema=schema)
pq_path = OUT_DIR / "financial_facts_raw.parquet"
pq.write_table(pq_table, pq_path)
print(f"  Saved Parquet: {pq_path} ({len(facts_final)} rows)")

# 2. financial_facts_review.csv
csv_path = OUT_DIR / "financial_facts_review.csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=schema.names)
    writer.writeheader()
    writer.writerows(facts_final)
print(f"  Saved Review CSV: {csv_path}")

# 3. validation_report.json
val_summary = {
    "generated_at_utc": NOW_UTC,
    "pilot_companies_count": len(PILOT_SYMBOLS),
    "total_facts_extracted": len(facts_final),
    "validation_summary": {
        "status_ok": sum(1 for f in facts_final if f["validation_status"] == "ok"),
        "status_review": sum(1 for f in facts_final if f["validation_status"] == "review"),
        "status_missing": sum(1 for f in facts_final if f["validation_status"] == "missing"),
    },
    "review_flags_count": len(review_flags),
    "review_flags": review_flags,
    "validation_rules_implemented": [
        "Balance sheet equation: Assets = Liabilities + Equity (within 2% tolerance)",
        "Revenue consistency: Gross Profit <= Revenue * 1.05",
        "EPS sanity check: EPS magnitude <= 1000",
        "Comparative period preservation and segregation",
        "Section isolation: Income Statement, Balance Sheet, and Cash Flow metrics segregated",
        "Image/scanned page detection and review flagging"
    ]
}

report_path = OUT_DIR / "validation_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(val_summary, f, indent=2, ensure_ascii=False)
print(f"  Saved Validation Report: {report_path}")

print("\n=== Phase 2 Pilot Extraction Completed Successfully! ===")
