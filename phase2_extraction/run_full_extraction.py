"""
Stockpulse Phase 2: Full-Scale Financial Statement Extraction Pipeline (All 80 PDFs)
Extracts, normalizes, validates, and generates all required deliverables.
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path("cse_live")
DOC_DIR   = BASE_DIR / "documents" / "financial_reports"
OUT_DIR   = BASE_DIR / "phase2_extraction"
TBL_DIR   = OUT_DIR / "extracted_tables"
RAW_TEXT  = OUT_DIR / "raw_text"

for d in (OUT_DIR, TBL_DIR, RAW_TEXT):
    d.mkdir(parents=True, exist_ok=True)

NOW_UTC = datetime.now(timezone.utc).isoformat()

# ── Manifest Loading ──────────────────────────────────────────────────────────
with open(BASE_DIR / "financial_reports_manifest.csv", encoding="utf-8-sig") as f:
    manifest_rows = list(csv.DictReader(f))

with open(BASE_DIR / "all_disclosures_document_index.csv", encoding="utf-8-sig") as f:
    disc_index = {
        r["id"]: r for r in csv.DictReader(f)
        if r.get("document_category") == "financial_reports"
    }

financial_sector_keywords = {"bank", "financ", "insur", "leasing", "capital", "invest"}

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
IS_TITLE = re.compile(
    r"\b(statements?\s+of\s+(profit\s*(?:/|\s+and\s+|\s+or\s+)\s*(?:other\s+)?(?:comprehensive\s+income|loss)?|comprehensive\s+income|income)"
    r"|income\s+statements?|statement\s+of\s+operations|statement\s+of\s+profit\s*/\s*\(loss\)"
    r"|condensed\s+consolidated\s+statement\s+of\s+profit\s+or\s+loss)\b",
    re.I
)
BS_TITLE = re.compile(
    r"\b(statements?\s+of\s+financial\s+position|balance\s+sheets?|condensed\s+consolidated\s+statement\s+of\s+financial\s+position)\b",
    re.I
)
CF_TITLE = re.compile(
    r"\b(statements?\s+of\s+cash\s+flows?|cash\s+flow\s+statements?|condensed\s+consolidated\s+statement\s+of\s+cash\s+flows?)\b",
    re.I
)

def classify_page(lines: list[str]) -> str:
    # 1. Search title in first 15 lines (CF -> BS -> IS)
    for l in lines[:15]:
        l_clean = l.strip()
        if CF_TITLE.search(l_clean):
            return "cash_flow"
        if BS_TITLE.search(l_clean):
            return "balance_sheet"
        if IS_TITLE.search(l_clean):
            return "income_statement"

    # 2. Signature heuristic fallback for non-standard headers
    has_total_assets = any(re.search(r"\btotal\s+assets\b", l, re.I) for l in lines)
    has_equity_liab = any(re.search(r"\b(equity\s+and\s+liabilities|total\s+equity|total\s+liabilities|total\s+non-current\s+assets)\b", l, re.I) for l in lines)
    if (has_total_assets and has_equity_liab) or (has_equity_liab and any(re.search(r"\bcurrent\s+liabilities\b", l, re.I) for l in lines)):
        return "balance_sheet"

    has_rev = any(re.search(r"^\s*(revenue|turnover|net\s+interest\s+income|interest\s+income)\b", l, re.I) for l in lines)
    has_pbt = any(re.search(r"\bprofit\s+before\s+(income\s+)?tax\b", l, re.I) for l in lines)
    has_eps = any(re.search(r"\bearnings?\s+per\s+share\b", l, re.I) for l in lines)
    if (has_rev and has_pbt) or (has_pbt and has_eps):
        return "income_statement"

    has_ocf = any(re.search(r"\bcash\s+(generated\s+from|flows?\s+from)\s+operating\b", l, re.I) for l in lines)
    if has_ocf:
        return "cash_flow"

    return "other"

METRIC_PATTERNS = {
    "income_statement": [
        ("revenue",           re.compile(r"^(revenue(\s+from\s+contracts?\s+with\s+customers)?|turnover|net\s+revenue|gross\s+revenue|income|total\s+revenue)\b", re.I)),
        ("interest_income",   re.compile(r"^(interest\s+income|finance\s+income)\b", re.I)),
        ("gross_profit",      re.compile(r"^(gross\s+profit(\s*/\s*loss)?)\b", re.I)),
        ("operating_profit",  re.compile(r"^(operating\s+profit(\s*/\s*loss)?|profit\s+from\s+operations|results?\s+from\s+operating\s+activities|operating\s+profit\s+before\s+taxes|profit\s+from\s+operating\s+activities)\b", re.I)),
        ("finance_costs",     re.compile(r"^(finance\s+(costs?|expenses?)|interest\s+expenses?|net\s+finance\s+(costs?|expenses?))\b", re.I)),
        ("profit_before_tax", re.compile(r"^(profit(\s*/\s*loss)?\s+before\s+(income\s+)?tax(ation)?|loss\s+before\s+tax)\b", re.I)),
        ("tax_expense",       re.compile(r"^(income\s+tax\s+(expense|credit|charge)|taxation|income\s+tax|total\s+income\s+tax)\b", re.I)),
        ("net_profit",        re.compile(r"^(profit(\s*/\s*loss)?\s+for\s+the\s+period|profit\s+after\s+tax|net\s+profit|loss\s+for\s+the\s+period)\b", re.I)),
        ("eps_basic",         re.compile(r"^(basic\s*(and\s*diluted\s*)?(loss\s*/\s*)?earnings?\s+per\s+share|earnings?\s+per\s+share\s*[-–]\s*basic|basic\s+eps)\b", re.I)),
        ("eps_diluted",       re.compile(r"^(diluted\s*(loss\s*/\s*)?earnings?\s+per\s+share|earnings?\s+per\s+share\s*[-–]\s*diluted|diluted\s+eps)\b", re.I)),
    ],
    "balance_sheet": [
        ("cash_equivalents",  re.compile(r"^(cash\s+and\s+(cash\s+)?equivalents|cash\s+and\s+bank\s+balances|cash\s+at\s+bank(\s+and\s+in\s+hand)?)\b", re.I)),
        ("current_assets",    re.compile(r"^(total\s+current\s+assets|current\s+assets)\b", re.I)),
        ("total_assets",      re.compile(r"^(total\s+assets)\b", re.I)),
        ("current_liabilities", re.compile(r"^(total\s+current\s+liabilities|current\s+liabilities)\b", re.I)),
        ("non_current_liabilities", re.compile(r"^(total\s+non-current\s+liabilities|non-current\s+liabilities)\b", re.I)),
        ("total_liabilities", re.compile(r"^(total\s+liabilities)\b", re.I)),
        ("total_equity",      re.compile(r"^(total\s+equity|total\s+shareholders.?\s*equity|equity\s+attributable\s+to|net\s+assets)\b", re.I)),
        ("borrowings",        re.compile(r"^(interest\s+bearing\s+(borrowings|liabilities)|loans\s+and\s+borrowings|borrowings|bank\s+loans|bank\s+overdraft)\b", re.I)),
        ("retained_earnings", re.compile(r"^(retained\s+earnings|accumulated\s+profit(\s*/\s*losses)?|retained\s+profit)\b", re.I)),
    ],
    "cash_flow": [
        ("operating_cashflow", re.compile(r"^(net\s+cash.*from.*operating|cash\s+generated\s+from\s+operations|net\s+cash\s+generated\s+from\s+operating)\b", re.I)),
        ("investing_cashflow", re.compile(r"^(net\s+cash.*used\s+in.*investing|net\s+cash.*from.*investing)\b", re.I)),
        ("financing_cashflow", re.compile(r"^(net\s+cash.*used\s+in.*financing|net\s+cash.*from.*financing)\b", re.I)),
        ("capex",              re.compile(r"^(purchase\s+of\s+property.*plant|acquisition\s+of\s+property|purchase\s+of\s+ppe|capital\s+expenditure)\b", re.I)),
        ("dividends_paid",     re.compile(r"^(dividends?\s+paid|payment\s+of\s+dividends?|equity\s+dividends?\s+paid)\b", re.I)),
    ]
}

def detect_unit(text: str) -> tuple[str, float]:
    text_lower = text[:4000].lower()
    if any(k in text_lower for k in ["rupees thousands", "rs.'000", "rs.000", "rs '000", "lkr'000", "'000", "rs. 000"]):
        return "LKR_thousands", 1000.0
    if "rupees millions" in text_lower or "rs.mn" in text_lower or "rs. mn" in text_lower:
        return "LKR_millions", 1000000.0
    return "LKR_units", 1.0

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

# ── Extraction Loop ───────────────────────────────────────────────────────────
all_facts: list[dict] = []
scanned_pages: list[dict] = []
failed_documents: list[dict] = []
review_flags: list[dict] = []
risk_observations: list[dict] = []
audit_rows: list[dict] = []
duplicates_detected: int = 0

print(f"Starting Refined Full-Scale Extraction across {len(manifest_rows)} Financial Report PDFs...")

for idx, m_row in enumerate(manifest_rows, 1):
    rid = str(m_row["id"])
    sym = m_row.get("symbol", "")
    company_name = m_row.get("name", "")
    upl_date = m_row.get("uploadedDate", "")
    man_date = m_row.get("manualDate", "")
    
    pdf_matches = list(DOC_DIR.glob(f"{rid}_*.pdf"))
    if not pdf_matches:
        failed_documents.append({
            "company_code": rid,
            "symbol": sym,
            "company_name": company_name,
            "source_pdf": "MISSING",
            "failure_reason": "PDF file not found on disk",
            "timestamp_utc": NOW_UTC
        })
        continue

    pdf_path = pdf_matches[0]
    disc_data = disc_index.get(rid, {})
    sha256 = disc_data.get("sha256", hashlib.sha256(pdf_path.read_bytes()).hexdigest())

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            full_text_parts = []
            page_sections = []
            img_pages_count = 0
            
            for pno, page in enumerate(pdf.pages, 1):
                ptext = page.extract_text() or ""
                full_text_parts.append(ptext)
                
                # Check for image/scanned page
                if len(ptext.strip()) < 50:
                    img_pages_count += 1
                    scanned_pages.append({
                        "company_code": rid,
                        "symbol": sym,
                        "company_name": company_name,
                        "source_pdf": pdf_path.name,
                        "page_number": pno,
                        "total_pages": total_pages,
                        "char_count": len(ptext.strip()),
                        "requires_ocr": True,
                        "notes": "Scanned or vector image page with < 50 characters",
                        "timestamp_utc": NOW_UTC
                    })
                
                # Section classification using refined logic
                lines = [l.strip() for l in ptext.splitlines() if l.strip()]
                sect = classify_page(lines)
                page_sections.append((pno, sect, ptext))
            
            full_text = "\n".join(full_text_parts)
            unit_str, unit_mult = detect_unit(full_text)
            consol_scope = detect_consolidation(full_text)
            p_type, p_end, p_comp = detect_period(full_text)

            has_is = any(s == "income_statement" for _, s, _ in page_sections)
            has_bs = any(s == "balance_sheet" for _, s, _ in page_sections)
            has_cf = any(s == "cash_flow" for _, s, _ in page_sections)

            # Record audit row
            audit_rows.append({
                "id": rid,
                "symbol": sym,
                "company_name": company_name,
                "pdf_filename": pdf_path.name,
                "total_pages": total_pages,
                "total_chars": len(full_text),
                "image_only_pages": img_pages_count,
                "is_likely_scan": img_pages_count >= total_pages - 1,
                "has_income_statement": has_is,
                "income_stmt_pages": ";".join(str(p) for p, s, _ in page_sections if s == "income_statement"),
                "has_balance_sheet": has_bs,
                "balance_sheet_pages": ";".join(str(p) for p, s, _ in page_sections if s == "balance_sheet"),
                "has_cash_flow": has_cf,
                "cash_flow_pages": ";".join(str(p) for p, s, _ in page_sections if s == "cash_flow"),
                "sha256": sha256,
                "audit_timestamp_utc": NOW_UTC
            })

            # Check if entire document is a scan
            is_full_scan = (img_pages_count >= total_pages - 1) and (len(full_text.strip()) < 200)
            if is_full_scan:
                failed_documents.append({
                    "company_code": rid,
                    "symbol": sym,
                    "company_name": company_name,
                    "source_pdf": pdf_path.name,
                    "failure_reason": f"Entire PDF is scanned image ({img_pages_count}/{total_pages} pages scanned)",
                    "timestamp_utc": NOW_UTC
                })
                print(f"[{idx:2}/{len(manifest_rows)}] {sym:6} SKIPPED: Entire document is scanned image.")
                continue

            # Extract metrics from section pages
            doc_facts: list[dict] = []
            for pno, sect, ptext in page_sections:
                if sect not in METRIC_PATTERNS:
                    continue
                
                for raw_line in ptext.splitlines():
                    cleaned_line = clean_line_numbers(raw_line).strip()
                    if not cleaned_line:
                        continue
                    
                    for metric, pat in METRIC_PATTERNS[sect]:
                        if pat.search(cleaned_line):
                            nums = extract_numbers_from_line(cleaned_line)
                            if not nums:
                                continue
                            
                            mult = 1.0 if metric in ("eps_basic", "eps_diluted") else unit_mult
                            
                            # Current period
                            cur_raw, cur_val = nums[0]
                            doc_facts.append({
                                "company_code": rid,
                                "company_name": company_name,
                                "symbol": sym,
                                "period_start": "2026-04-01" if p_type == "semi_annual" else ("2026-01-01" if p_type == "nine_month" else "2026-07-01"),
                                "period_end": p_end,
                                "publication_date": upl_date,
                                "statement_type": sect,
                                "metric": metric,
                                "raw_label": cleaned_line[:120],
                                "raw_value": cur_raw,
                                "value": cur_val,
                                "currency": "LKR",
                                "unit": unit_str,
                                "multiplier": mult,
                                "normalized_value_lkr": cur_val * mult,
                                "consolidation_scope": consol_scope,
                                "period_type": p_type,
                                "comparative_flag": False,
                                "source_pdf": pdf_path.name,
                                "sha256": sha256,
                                "page_number": pno,
                                "extraction_method": "pdfplumber_text_parser",
                                "validation_status": "ok"
                            })

                            # Comparative period
                            if len(nums) > 1:
                                comp_raw, comp_val = nums[1]
                                doc_facts.append({
                                    "company_code": rid,
                                    "company_name": company_name,
                                    "symbol": sym,
                                    "period_start": "2025-04-01" if p_type == "semi_annual" else ("2025-01-01" if p_type == "nine_month" else "2025-07-01"),
                                    "period_end": p_comp,
                                    "publication_date": upl_date,
                                    "statement_type": sect,
                                    "metric": metric,
                                    "raw_label": cleaned_line[:120],
                                    "raw_value": comp_raw,
                                    "value": comp_val,
                                    "currency": "LKR",
                                    "unit": unit_str,
                                    "multiplier": mult,
                                    "normalized_value_lkr": comp_val * mult,
                                    "consolidation_scope": consol_scope,
                                    "period_type": p_type,
                                    "comparative_flag": True,
                                    "source_pdf": pdf_path.name,
                                    "sha256": sha256,
                                    "page_number": pno,
                                    "extraction_method": "pdfplumber_text_parser",
                                    "validation_status": "ok"
                                })
                            break

            # Handle total liabilities synthesis if current + non-current are separated
            by_comp = defaultdict(dict)
            for f in doc_facts:
                by_comp[f["comparative_flag"]][f["metric"]] = f
            
            for comp_flag, metric_dict in by_comp.items():
                if "total_liabilities" not in metric_dict:
                    nc = metric_dict.get("non_current_liabilities")
                    cur = metric_dict.get("current_liabilities")
                    if nc and cur:
                        tot_val = nc["value"] + cur["value"]
                        mult = nc["multiplier"]
                        doc_facts.append({
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
                            "multiplier": mult,
                            "normalized_value_lkr": tot_val * mult,
                            "consolidation_scope": nc["consolidation_scope"],
                            "period_type": nc["period_type"],
                            "comparative_flag": comp_flag,
                            "source_pdf": nc["source_pdf"],
                            "sha256": nc["sha256"],
                            "page_number": nc["page_number"],
                            "extraction_method": "synthesized_sum",
                            "validation_status": "ok"
                        })

            # Filter core metrics and deduplicate
            CORE_METRICS = {
                "revenue", "interest_income", "gross_profit", "operating_profit", "finance_costs",
                "profit_before_tax", "tax_expense", "net_profit", "eps_basic", "eps_diluted",
                "cash_equivalents", "current_assets", "total_assets", "current_liabilities",
                "total_liabilities", "total_equity", "borrowings", "retained_earnings",
                "operating_cashflow", "investing_cashflow", "financing_cashflow", "capex", "dividends_paid"
            }
            
            seen_keys: set[tuple] = set()
            doc_facts_final: list[dict] = []
            for f in doc_facts:
                if f["metric"] not in CORE_METRICS:
                    continue
                k = (f["statement_type"], f["metric"], f["comparative_flag"])
                if k in seen_keys:
                    duplicates_detected += 1
                    continue
                seen_keys.add(k)
                doc_facts_final.append(f)

            all_facts.extend(doc_facts_final)

            if len(doc_facts_final) == 0:
                failed_documents.append({
                    "company_code": rid,
                    "symbol": sym,
                    "company_name": company_name,
                    "source_pdf": pdf_path.name,
                    "failure_reason": f"No core financial facts extracted (IS={has_is}, BS={has_bs}, CF={has_cf})",
                    "timestamp_utc": NOW_UTC
                })

            print(f"[{idx:2}/{len(manifest_rows)}] {sym:6} OK: {len(doc_facts_final):2} facts extracted ({unit_str}, {consol_scope}, {p_type})")

    except Exception as e:
        failed_documents.append({
            "company_code": rid,
            "symbol": sym,
            "company_name": company_name,
            "source_pdf": pdf_path.name,
            "failure_reason": f"Extraction error: {str(e)[:150]}",
            "timestamp_utc": NOW_UTC
        })
        print(f"[{idx:2}/{len(manifest_rows)}] {sym:6} ERROR: {e}")

# ── Validation & Quality Controls ─────────────────────────────────────────────
print("\nRunning Validation and Quality Control Rules...")

by_sym_comp = defaultdict(dict)
for f in all_facts:
    key = (f["symbol"], f["comparative_flag"])
    by_sym_comp[key][f["metric"]] = f

is_financial_company = lambda name, sym: any(k in name.lower() for k in financial_sector_keywords) or sym in ("CFIN", "COMB", "HNB", "SAMP", "SEYB", "NTB", "NDB", "PABC", "DFCC", "AAIC", "CINS", "PMB", "WLTH", "VFIN", "CFI")

for (sym, comp), metric_dict in by_sym_comp.items():
    comp_str = "Comparative" if comp else "Current"
    company_name = next(iter(metric_dict.values()))["company_name"]
    is_fin = is_financial_company(company_name, sym)

    # 1. Balance Sheet Equation: Assets = Liabilities + Equity
    ta = metric_dict.get("total_assets")
    tl = metric_dict.get("total_liabilities")
    te = metric_dict.get("total_equity")
    if ta and tl and te:
        assets_val = ta["value"]
        liab_val = tl["value"]
        equity_val = te["value"]
        diff = abs(assets_val - (liab_val + equity_val))
        tol = max(abs(assets_val) * 0.03, 20.0) # 3% tolerance for rounding/formatting
        if diff > tol:
            review_flags.append({
                "symbol": sym,
                "company_code": ta["company_code"],
                "flag": "BS_EQUATION_MISMATCH",
                "detail": f"{comp_str} balance sheet mismatch: Assets ({assets_val:,.0f}) != Liab ({liab_val:,.0f}) + Eq ({equity_val:,.0f}). Diff={diff:,.0f}",
                "metric": "balance_sheet_equation",
                "validation_status": "review"
            })
            ta["validation_status"] = "review"
            tl["validation_status"] = "review"
            te["validation_status"] = "review"

    # 2. Sector-Aware Gross Profit Check
    if not is_fin:
        rev = metric_dict.get("revenue")
        gp = metric_dict.get("gross_profit")
        if rev and gp and rev["value"] > 0:
            if gp["value"] > rev["value"] * 1.05:
                review_flags.append({
                    "symbol": sym,
                    "company_code": rev["company_code"],
                    "flag": "GROSS_PROFIT_EXCEEDS_REVENUE",
                    "detail": f"{comp_str} Gross Profit ({gp['value']:,.0f}) > Revenue ({rev['value']:,.0f})",
                    "metric": "gross_profit",
                    "validation_status": "review"
                })
                gp["validation_status"] = "review"

    # 3. EPS Sanity Check (EPS should not be in thousands/millions)
    for eps_metric in ("eps_basic", "eps_diluted"):
        eps = metric_dict.get(eps_metric)
        if eps and abs(eps["value"]) > 1000.0:
            review_flags.append({
                "symbol": sym,
                "company_code": eps["company_code"],
                "flag": "EPS_MAGNITUDE_WARNING",
                "detail": f"{comp_str} {eps_metric} ({eps['value']:,.2f}) > 1000. Potential total earnings or note number.",
                "metric": eps_metric,
                "validation_status": "review"
            })
            eps["validation_status"] = "review"

    # 4. Financial Risk Observations (Logged separately, NOT extraction failures)
    if te and te["value"] < 0:
        risk_observations.append({
            "symbol": sym,
            "company_name": company_name,
            "period": comp_str,
            "observation": "FINANCIAL_RISK_NEGATIVE_EQUITY",
            "metric": "total_equity",
            "value": te["value"],
            "detail": f"Company has negative total equity (accumulated losses exceed capital): {te['value']:,.0f}"
        })

    ocf = metric_dict.get("operating_cashflow")
    if ocf and ocf["value"] < 0:
        risk_observations.append({
            "symbol": sym,
            "company_name": company_name,
            "period": comp_str,
            "observation": "FINANCIAL_RISK_NEGATIVE_OPERATING_CASH_FLOW",
            "metric": "operating_cashflow",
            "value": ocf["value"],
            "detail": f"Negative cash flow generated from operations: {ocf['value']:,.0f}"
        })

# ── Save All Deliverables ─────────────────────────────────────────────────────
print("\nSaving Deliverables...")

# Schema definition
base_fields = [
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
    ("multiplier", pa.float64()),
    ("normalized_value_lkr", pa.float64()),
    ("consolidation_scope", pa.string()),
    ("period_type", pa.string()),
    ("comparative_flag", pa.bool_()),
    ("source_pdf", pa.string()),
    ("sha256", pa.string()),
    ("page_number", pa.int32()),
    ("extraction_method", pa.string()),
    ("validation_status", pa.string()),
]

schema = pa.schema(base_fields)

# 1. financial_facts_raw.parquet
raw_cols = [f[0] for f in base_fields if f[0] not in ("multiplier", "normalized_value_lkr")]
raw_schema = pa.schema([f for f in base_fields if f[0] not in ("multiplier", "normalized_value_lkr")])
raw_data = {col: [f[col] for f in all_facts] for col in raw_cols}
pq.write_table(pa.Table.from_pydict(raw_data, schema=raw_schema), OUT_DIR / "financial_facts_raw.parquet")
print(f"  [1/7] Saved: financial_facts_raw.parquet ({len(all_facts)} rows)")

# 2. financial_facts_normalized.parquet
norm_data = {col: [f[col] for f in all_facts] for col in schema.names}
pq.write_table(pa.Table.from_pydict(norm_data, schema=schema), OUT_DIR / "financial_facts_normalized.parquet")
print(f"  [2/7] Saved: financial_facts_normalized.parquet ({len(all_facts)} rows)")

# 3. financial_facts_review.csv
with open(OUT_DIR / "financial_facts_review.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=schema.names)
    w.writeheader()
    w.writerows(all_facts)
print(f"  [3/7] Saved: financial_facts_review.csv ({len(all_facts)} rows)")

# 4. extraction_audit.csv
if audit_rows:
    with open(OUT_DIR / "extraction_audit.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    print(f"  [4/7] Saved: extraction_audit.csv ({len(audit_rows)} rows)")

# 5. failed_documents.csv
with open(OUT_DIR / "failed_documents.csv", "w", newline="", encoding="utf-8-sig") as f:
    fields = ["company_code", "symbol", "company_name", "source_pdf", "failure_reason", "timestamp_utc"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(failed_documents)
print(f"  [5/7] Saved: failed_documents.csv ({len(failed_documents)} documents)")

# 6. scanned_pages_review.csv
with open(OUT_DIR / "scanned_pages_review.csv", "w", newline="", encoding="utf-8-sig") as f:
    fields = ["company_code", "symbol", "company_name", "source_pdf", "page_number", "total_pages", "char_count", "requires_ocr", "notes", "timestamp_utc"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(scanned_pages)
print(f"  [6/7] Saved: scanned_pages_review.csv ({len(scanned_pages)} pages)")

# 7. validation_report.json
unique_companies = sorted(list(set(f["symbol"] for f in all_facts)))
status_counts = defaultdict(int)
for f in all_facts:
    status_counts[f["validation_status"]] += 1

val_summary = {
    "generated_at_utc": NOW_UTC,
    "total_pdfs_in_scope": len(manifest_rows),
    "total_pdfs_processed_successfully": len(manifest_rows) - len(failed_documents),
    "total_pdfs_failed_or_scanned": len(failed_documents),
    "unique_companies_covered": len(unique_companies),
    "total_facts_extracted": len(all_facts),
    "duplicate_facts_detected_and_deduplicated": duplicates_detected,
    "validation_status_breakdown": dict(status_counts),
    "review_flags_count": len(review_flags),
    "review_flags": review_flags,
    "financial_risk_observations_count": len(risk_observations),
    "financial_risk_observations": risk_observations,
    "scanned_pages_count": len(scanned_pages),
    "failed_documents_summary": failed_documents,
}

with open(OUT_DIR / "validation_report.json", "w", encoding="utf-8") as f:
    json.dump(val_summary, f, indent=2, ensure_ascii=False)
print(f"  [7/7] Saved: validation_report.json")

print("\n=== Phase 2 Full-Scale Financial Statement Extraction Complete! ===")
