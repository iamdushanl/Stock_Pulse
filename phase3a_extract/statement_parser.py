"""
Phase 3A Stage 6, 7, 9, 10 & 11 – Spatial Multi-Column Statement Parser
======================================================================
Parses CSE financial reports (Annual & Interim, Vector & OCR) into structured
raw financial facts with exact per-column:
  - StatementType (`Income Statement`, `Balance Sheet`, `Cash Flow`, `Equity`)
  - IsConsolidated (`Group` vs `Company`)
  - PeriodType (`Quarterly`, `Half-Year`, `Nine-Month`, `Annual`)
  - PeriodStart, PeriodEnd, FiscalYear, FiscalQuarter
  - Currency (`LKR`, `USD`, etc.) and Unit (`LKR thousands`, `LKR millions`, `LKR`, `USD`, etc.)
  - IsRestated and ComparativeFlag
Handles both left-label tables (SIL, CFIN, DIAL, TJL) and center-label T-account
tables (BERU, MARA, CIT, CFI, ASHO), filtering % change and Note columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .pdf_reader_and_ocr import PageData, WordToken, _group_tokens_into_rows

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Explicit ordered (MetricCode, StatementType, include_regex, exclude_regex)
METRIC_RULES: list[tuple[str, str, re.Pattern[str], re.Pattern[str] | None]] = [
    # ── Income Statement ──────────────────────────────────────────────────────
    ("REVENUE", "Income Statement", re.compile(r"^(?:total\s+)?revenue(?:\s+from\s+contracts?\s+with\s+customers)?$|^turnover$|^gross\s+written\s+premium$|^net\s+earned\s+premium$|^income$", re.I), re.compile(r"(?:other|interest|finance|comprehensive|before|tax|property|cost)", re.I)),
    ("INTEREST_INCOME", "Income Statement", re.compile(r"^(?:interest\s+income|gross\s+interest\s+income)$", re.I), None),
    ("NET_INTEREST_INCOME", "Income Statement", re.compile(r"^net\s+interest\s+income$", re.I), None),
    ("COST_OF_SALES", "Income Statement", re.compile(r"^(?:cost\s+of\s+sales|direct\s+costs?|cost\s+of\s+services?|direct\s+operating\s+costs?|interest\s+expenses?)$", re.I), None),
    ("GROSS_PROFIT", "Income Statement", re.compile(r"^gross\s+profit(?:\s*/\s*\(?loss\)?)?$", re.I), None),
    ("OTHER_INCOME", "Income Statement", re.compile(r"^(?:other\s+(?:operating\s+)?income(?:\s*/\s*\(?expenses?\)?)?|other\s+operating\s+gains?)$", re.I), None),
    ("SELLING_DISTRIBUTION_EXPENSE", "Income Statement", re.compile(r"^(?:selling\s*(?:&|and)?\s*distribution|distribution\s+(?:costs?|expenses?)|marketing\s*(?:&|and)?\s*distribution)", re.I), None),
    ("ADMIN_EXPENSE", "Income Statement", re.compile(r"^(?:administrative\s+(?:expenses?|costs?)|administration\s+(?:expenses?|costs?)|general\s*(?:&|and)?\s*admin)", re.I), None),
    ("OPERATING_PROFIT", "Income Statement", re.compile(r"(?:^operating\s+profit(?:\s*/\s*\(?loss\)?)?|^profit\s*(?:/\s*\(?loss\)?)?\s+from\s+operations?|^results?\s+from\s+operating\s+activities|^operating\s+\(?loss\)?\s*/\s*profit)", re.I), re.compile(r"(?:before\s+working\s+capital|cash\s+flow)", re.I)),
    ("FINANCE_INCOME", "Income Statement", re.compile(r"^finance\s+income$", re.I), None),
    ("FINANCE_COSTS", "Income Statement", re.compile(r"^(?:finance\s+(?:costs?|expenses?)|net\s+finance\s+(?:costs?|expenses?))$", re.I), None),
    ("PROFIT_BEFORE_TAX", "Income Statement", re.compile(r"(?:\bprofit\s*(?:/\s*\(?loss\)?)?\s+before\s+(?:income\s+)?tax(?:ation)?|\bloss\s*(?:/\s*profit)?\s+before\s+tax)", re.I), re.compile(r"(?:adjustments|changes\s+in\s+working)", re.I)),
    ("INCOME_TAX_EXPENSE", "Income Statement", re.compile(r"^(?:income\s+tax\s*(?:\(?expense\)?|credit|/\s*\(?credit\)?|charge)?|taxation|tax\s+expense)$", re.I), re.compile(r"(?:deferred\s+tax\s+(?:asset|liab)|paid|recoverable)", re.I)),
    ("PROFIT_AFTER_TAX", "Income Statement", re.compile(r"^(?:\(?loss\)?\s*/\s*)?(?:profit|loss)\s*(?:/\s*\(?loss\)?)?\s+(?:for\s+the\s+(?:period|year)|after\s+tax(?:ation)?)$|^net\s+profit(?:\s+for\s+the\s+period)?$", re.I), re.compile(r"(?:attributable|other\s+comprehensive)", re.I)),
    ("PROFIT_ATTRIBUTABLE_EQUITY_HOLDERS", "Income Statement", re.compile(r"(?:equity\s+holders\s+of\s+the\s+(?:parent|company)|owners\s+of\s+the\s+(?:parent|company))", re.I), re.compile(r"(?:comprehensive|non[- ]?controlling)", re.I)),
    ("EPS_BASIC", "Income Statement", re.compile(r"(?:basic(?:\s+and\s+diluted)?\s+(?:\(?loss\)?\s*/\s*)?earnings?\s+per\s+share|^earnings?\s+per\s+share\s*(?:\(|-|–|\bbasic)|\bbasic\s+eps\b|^earnings?\s+per\s+(?:ordinary\s+)?share$)", re.I), re.compile(r"diluted\s+earnings?\s+per\s+share(?!\s*$)", re.I)),
    ("EPS_DILUTED", "Income Statement", re.compile(r"(?:diluted\s+(?:\(?loss\)?\s*/\s*)?earnings?\s+per\s+share|^earnings?\s+per\s+share\s*[-–]\s*diluted|\bdiluted\s+eps\b)", re.I), None),

    # ── Balance Sheet ─────────────────────────────────────────────────────────
    ("CASH_AND_EQUIVALENTS", "Balance Sheet", re.compile(r"^(?:cash\s+and\s+(?:cash\s+)?equivalents|cash\s+in\s+hand\s+and\s+at\s+bank|cash\s+and\s+bank\s+balances|cash\s+and\s+short[- ]?term\s+funds?)$", re.I), re.compile(r"(?:beginning|end\s+of|statement\s+of)", re.I)),
    ("SHORT_TERM_INVESTMENTS", "Balance Sheet", re.compile(r"^(?:short[- ]?term\s+investments?|other\s+current\s+financial\s+assets|financial\s+assets\s+at\s+(?:fair\s+value\s+through\s+profit|fvtpl)|short[- ]?term\s+deposits)$", re.I), None),
    ("TRADE_RECEIVABLES", "Balance Sheet", re.compile(r"^(?:trade\s*(?:&|and)?\s*other\s+receivables?|trade\s+receivables?|loans?\s*(?:&|and)?\s*receivables?)", re.I), re.compile(r"(?:impairment|increase|decrease)", re.I)),
    ("INVENTORIES", "Balance Sheet", re.compile(r"^inventories(?:\s+and\s+work\s+in\s+progress)?$", re.I), re.compile(r"(?:impairment|increase|decrease)", re.I)),
    ("CURRENT_ASSETS", "Balance Sheet", re.compile(r"^total\s+current\s+assets$", re.I), None),
    ("PPE", "Balance Sheet", re.compile(r"^(?:property,?\s*plant\s*(?:&|and)?\s*equipment)$", re.I), re.compile(r"(?:depreciation|purchase|acquisition|disposal)", re.I)),
    ("INTANGIBLE_ASSETS", "Balance Sheet", re.compile(r"^(?:intangible\s+assets(?:\s+and\s+goodwill)?|goodwill\s+and\s+intangible\s+assets)$", re.I), re.compile(r"(?:amorti|purchase)", re.I)),
    ("LONG_TERM_INVESTMENTS", "Balance Sheet", re.compile(r"^(?:investments?\s+in\s+(?:subsidiaries|associates|joint\s+ventures)|other\s+non[- ]?current\s+financial\s+assets|long[- ]?term\s+investments?)", re.I), None),
    ("NON_CURRENT_ASSETS", "Balance Sheet", re.compile(r"^total\s+non[- ]?current\s+assets$", re.I), None),
    ("TOTAL_ASSETS", "Balance Sheet", re.compile(r"^total\s+assets$", re.I), None),
    ("SHORT_TERM_BORROWINGS", "Balance Sheet", re.compile(r"^(?:short[- ]?term\s+(?:borrowings|loans)|bank\s+overdrafts?|current\s+interest[- ]?bearing\s+(?:loans\s*(?:&|and)?\s*)?borrowings)$", re.I), None),
    ("TRADE_PAYABLES", "Balance Sheet", re.compile(r"^(?:trade\s*(?:&|and)?\s*other\s+payables?|trade\s+payables?)$", re.I), re.compile(r"(?:increase|decrease)", re.I)),
    ("CURRENT_LIABILITIES", "Balance Sheet", re.compile(r"^total\s+current\s+liabilities$", re.I), None),
    ("LONG_TERM_BORROWINGS", "Balance Sheet", re.compile(r"^(?:long[- ]?term\s+(?:borrowings|loans)|non[- ]?current\s+interest[- ]?bearing\s+(?:loans\s*(?:&|and)?\s*)?borrowings|interest[- ]?bearing\s+(?:loans\s*(?:&|and)?\s*)?borrowings)$", re.I), None),
    ("DEFERRED_TAX_LIABILITIES", "Balance Sheet", re.compile(r"^deferred\s+tax\s+liabilities$", re.I), None),
    ("NON_CURRENT_LIABILITIES", "Balance Sheet", re.compile(r"^total\s+non[- ]?current\s+liabilities$", re.I), None),
    ("TOTAL_LIABILITIES", "Balance Sheet", re.compile(r"^total\s+liabilities$", re.I), re.compile(r"equity", re.I)),
    ("SHARE_CAPITAL", "Balance Sheet", re.compile(r"^(?:stated\s+capital|share\s+capital)$", re.I), None),
    ("RETAINED_EARNINGS", "Balance Sheet", re.compile(r"^(?:retained\s+earnings|accumulated\s+(?:profit|losses))$", re.I), None),
    ("NON_CONTROLLING_INTEREST", "Balance Sheet", re.compile(r"^non[- ]?controlling\s+interests?$", re.I), None),
    ("EQUITY_ATTRIBUTABLE_PARENT", "Balance Sheet", re.compile(r"^(?:total\s+)?equity\s+attributable\s+to\s+(?:owners|equity\s+holders|shareholders)(?:\s+of\s+the\s+(?:parent|company))?$", re.I), re.compile(r"liabilities", re.I)),
    ("TOTAL_EQUITY", "Balance Sheet", re.compile(r"^(?:total\s+equity|total\s+shareholders.?\s*equity)$", re.I), re.compile(r"(?:liabilities|attributable)", re.I)),
    ("TOTAL_EQUITY_AND_LIABILITIES", "Balance Sheet", re.compile(r"^total\s+equity\s*(?:&|and)\s*liabilities$", re.I), None),

    # ── Cash Flow Statement ───────────────────────────────────────────────────
    ("OPERATING_CASH_FLOW", "Cash Flow", re.compile(r"(?:net\s+cash\s*(?:flows?\s*)?(?:generated\s+from|from|\(?used\s+in\)?)\s*(?:/\s*\(?used\s+in\)?\s*)?operating\s+activities|cash\s+flows?\s+from\s+operating\s+activities)", re.I), re.compile(r"before\s+working\s+capital", re.I)),
    ("INVESTING_CASH_FLOW", "Cash Flow", re.compile(r"(?:net\s+cash\s*(?:flows?\s*)?(?:generated\s+from|from|\(?used\s+in\)?)\s*(?:/\s*\(?used\s+in\)?\s*)?investing\s+activities|cash\s+flows?\s+from\s+investing\s+activities)", re.I), None),
    ("FINANCING_CASH_FLOW", "Cash Flow", re.compile(r"(?:net\s+cash\s*(?:flows?\s*)?(?:generated\s+from|from|\(?used\s+in\)?)\s*(?:/\s*\(?used\s+in\)?\s*)?financing\s+activities|cash\s+flows?\s+from\s+financing\s+activities)", re.I), None),
    ("CAPEX", "Cash Flow", re.compile(r"(?:(?:acquisition|purchase|addition)s?\s+of\s+property,?\s*plant\s*(?:&|and)?\s*equipment|capital\s+expenditure)", re.I), None),
    ("DEPRECIATION", "Cash Flow", re.compile(r"^depreciation(?:\s+of\s+property.*|\s*&\s*amorti[sz]ation)?$", re.I), re.compile(r"right[- ]?of[- ]?use", re.I)),
    ("AMORTIZATION", "Cash Flow", re.compile(r"^amorti[sz]ation(?:\s+of\s+intangible\s+assets)?$", re.I), None),
    ("INTEREST_PAID", "Cash Flow", re.compile(r"^(?:interest\s+paid|finance\s+costs?\s+paid)$", re.I), None),
    ("TAX_PAID", "Cash Flow", re.compile(r"^(?:(?:income\s+)?tax(?:es|ation)?\s+paid)$", re.I), None),
    ("DIVIDENDS_PAID", "Cash Flow", re.compile(r"^(?:dividends?\s+paid(?:\s+to\s+.*)?|payment\s+of\s+dividends?)$", re.I), None),
    ("NET_CHANGE_IN_CASH", "Cash Flow", re.compile(r"(?:net\s+(?:increase|decrease)\s*(?:/\s*\(?(?:decrease|increase)\)?)?\s+in\s+cash\s*(?:&|and)\s*(?:cash\s+)?equivalents)", re.I), None),
    ("CASH_OPENING", "Cash Flow", re.compile(r"(?:cash\s*(?:&|and)\s*(?:cash\s+)?equivalents\s+at\s+(?:the\s+)?(?:beginning|start)\s+of)", re.I), None),
    ("CASH_CLOSING", "Cash Flow", re.compile(r"(?:cash\s*(?:&|and)\s*(?:cash\s+)?equivalents\s+at\s+(?:the\s+)?end\s+of)", re.I), None),
]

STMT_TITLE_IS = re.compile(r"\b(?:statements?\s+of\s+(?:profit|comprehensive\s+income|income)|income\s+statements?)\b", re.I)
STMT_TITLE_BS = re.compile(r"\b(?:statements?\s+of\s+financial\s+position|balance\s+sheets?)\b", re.I)
STMT_TITLE_CF = re.compile(r"\b(?:statements?\s+of\s+cash\s+flows?|cash\s+flow\s+statements?)\b", re.I)
STMT_TITLE_EQ = re.compile(r"\b(?:statements?\s+of\s+changes?\s+in\s+equity)\b", re.I)
SKIP_PAGE_RE = re.compile(r"\b(?:notes\s+to\s+the\s+(?:interim\s+)?financial\s+statements?|explanatory\s+notes|corporate\s+information|twenty\s+major\s+shareholders|directors.?\s+shareholding|segmental\s+information|investor\s+information)\b", re.I)


def _merge_adjacent_row_tokens(row: list[WordToken], max_gap: float = 4.5) -> list[WordToken]:
    """Merge fractured numeric fragments inside one horizontal row (e.g. `( 1,234 )`)."""
    if not row:
        return []
    merged: list[WordToken] = [row[0]]
    num_frag = re.compile(r"^[(\-+\d,.\)%]+$")
    for tok in row[1:]:
        prev = merged[-1]
        gap = tok.x0 - prev.x1
        # Merge only tiny gaps when joining broken number glyphs/brackets
        if gap <= max_gap and num_frag.match(prev.text) and num_frag.match(tok.text):
            # Never merge two full standalone multi-digit numbers separated by normal spacing
            if re.match(r"^\(?[\d,]+\)?$", prev.text) and re.match(r"^\(?[\d,]{3,}\)?$", tok.text) and gap > 2.2:
                merged.append(tok)
            else:
                merged[-1] = WordToken(
                    x0=prev.x0,
                    y0=min(prev.y0, tok.y0),
                    x1=tok.x1,
                    y1=max(prev.y1, tok.y1),
                    text=prev.text + tok.text,
                    method=prev.method,
                    conf=min(prev.conf, tok.conf),
                )
        else:
            merged.append(tok)
    return merged


def _parse_numeric_token(txt: str) -> tuple[bool, float | None, bool, bool, bool]:
    """
    Parse numeric financial token -> (is_numeric_cell, parsed_value, is_percent, is_dash_zero, has_restated_star).
    """
    s = txt.strip()
    if s in ("-", "–", "—", "Nil", "NIL", "nil"):
        return True, 0.0, False, True, False

    has_star = False
    while s.endswith(("*", "^", "†")):
        has_star = True
        s = s[:-1].strip()

    is_pct = s.endswith("%")
    if is_pct:
        s_num = s[:-1].strip()
    else:
        s_num = s

    neg = False
    if s_num.startswith("(") and s_num.endswith(")"):
        neg = True
        s_num = s_num[1:-1].strip()
    s_num = s_num.replace(",", "").replace(" ", "")

    if not re.match(r"^[+-]?\d+(?:\.\d+)?$", s_num):
        return False, None, False, False, has_star

    val = float(s_num)
    if neg:
        val = -val
    return True, val, is_pct, False, has_star


def _classify_statement_page(page: PageData, raw_text: str) -> list[str]:
    """Determine which primary financial statement(s) appear on `page`."""
    rows = _group_tokens_into_rows(page.tokens, y_tol=4.5)
    top_text = "\n".join(" ".join(t.text for t in r) for r in rows[:16])
    full_low = raw_text.lower()

    # Reject pure notes / shareholder / investor info narrative pages
    if rows and SKIP_PAGE_RE.search(" ".join(t.text for r in rows[:6] for t in r)):
        if not (STMT_TITLE_IS.search(top_text) or STMT_TITLE_BS.search(top_text) or STMT_TITLE_CF.search(top_text) or STMT_TITLE_EQ.search(top_text)):
            return []

    found: list[str] = []
    if STMT_TITLE_IS.search(top_text):
        found.append("Income Statement")
    if STMT_TITLE_BS.search(top_text):
        found.append("Balance Sheet")
    if STMT_TITLE_CF.search(top_text):
        found.append("Cash Flow")
    if STMT_TITLE_EQ.search(top_text):
        found.append("Equity")

    if found:
        return found

    # Signature fallback for scanned/OCR or subtle headers
    has_assets = bool(re.search(r"\btotal\s+assets\b", full_low))
    has_eq_liab = bool(re.search(r"\btotal\s+(?:equity|liabilities|equity\s*(?:&|and)\s*liabilities)\b", full_low))
    if has_assets and has_eq_liab:
        return ["Balance Sheet"]
    if re.search(r"\b(?:profit\s+before\s+tax|gross\s+profit)\b", full_low) and re.search(r"\b(?:revenue|turnover|interest\s+income)\b", full_low):
        return ["Income Statement"]
    if re.search(r"\boperating\s+activities\b", full_low) and re.search(r"\binvesting\s+activities\b", full_low):
        return ["Cash Flow"]
    if re.search(r"\bbalance\s+as?\s+at\b", full_low) and re.search(r"\bstated\s+capital\b", full_low) and re.search(r"\bretained\s+earnings\b", full_low):
        return ["Equity"]
    return []


def _parse_date_string(txt: str) -> pd.Timestamp | None:
    """Parse 4-digit and 2-digit CSE date expressions (`30th September 2026`, `30-Sep-26`, `31.12.25`, `September '25`)."""
    t = txt.strip()
    # 0. ISO YYYY-MM-DD
    m0 = re.search(r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b", t)
    if m0:
        y_i, mo_i, d_i = int(m0.group(1)), int(m0.group(2)), int(m0.group(3))
        if 1 <= mo_i <= 12 and 1 <= d_i <= 31 and 2010 <= y_i <= 2027:
            try:
                return pd.Timestamp(year=y_i, month=mo_i, day=d_i)
            except ValueError:
                pass
    # 1. dd Month yyyy or dd-Mon-yy (restrict 2-digit years strictly to 15..27 so day '30'/'29' never looks like year 2030)
    m1 = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?[\s./-]+([A-Za-z]+)[\s./-]+'?(\d{2,4})\b", t)
    if m1:
        mo = MONTH_MAP.get(m1.group(2).lower())
        yr_str = m1.group(3)
        yr_raw = int(yr_str)
        if len(yr_str) == 4:
            yr = yr_raw
        elif 15 <= yr_raw <= 27:
            yr = 2000 + yr_raw
        else:
            yr = -1
        if mo and 2010 <= yr <= 2027:
            try:
                return pd.Timestamp(year=yr, month=mo, day=int(m1.group(1)))
            except ValueError:
                pass
    # 2. dd.mm.yyyy or dd/mm/yyyy or dd.mm.yy or dd-mm-yy
    m2 = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b", t)
    if m2:
        d_i, mo_i = int(m2.group(1)), int(m2.group(2))
        yr_str = m2.group(3)
        yr_raw = int(yr_str)
        if len(yr_str) == 4:
            yr = yr_raw
        elif 15 <= yr_raw <= 27:
            yr = 2000 + yr_raw
        else:
            yr = -1
        if 1 <= mo_i <= 12 and 1 <= d_i <= 31 and 2010 <= yr <= 2027:
            try:
                return pd.Timestamp(year=yr, month=mo_i, day=d_i)
            except ValueError:
                pass
    # 3. Month 'yy (e.g. `September '25`)
    m3 = re.search(r"\b([A-Za-z]+)\s*'(\d{2})\b", t)
    if m3:
        mo = MONTH_MAP.get(m3.group(1).lower())
        yr_raw = int(m3.group(2))
        if mo and 15 <= yr_raw <= 27:
            yr = 2000 + yr_raw
            d_end = 31 if mo in (1, 3, 5, 7, 8, 10, 12) else (30 if mo != 2 else 28)
            return pd.Timestamp(year=yr, month=mo, day=d_end)
    return None


def _detect_currency_unit_from_text(text: str, doc_hint: str) -> tuple[str, str, float, bool]:
    """Detect `(Currency, Unit, Multiplier, IsAmbiguous)` from header/footer text."""
    tl = text.lower()
    # Explicit Sri Lanka Rupee phrases or standalone LKR column headers take precedence over prose USD mentions
    has_lkr_explicit = bool(re.search(
        r"(?:\blkr\b|all\s+(?:values|amounts)\s+(?:are\s+)?in\s+(?:lkr|rs|sri\s+lankan?\s+rupees)|the\s+above\s+figures\s+are\s+in\s+(?:lkr|rs)|in\s+rs\.?\s*['’`.,]?000s?\b|lkr\.?\s*['’`.,]?\s*[0o]{3}s?\b|rs\.?\s*[-'’`.,]?\s*[s5]?[0o]{3}s?\b)",
        tl,
    ))
    if not has_lkr_explicit:
        if re.search(r"(?:united\s+states\s+dollars|us\$|\busd\b)\s*(?:\(?[0o]{3}s?|thousands)", tl):
            return "USD", "USD thousands", 1000.0, False
        if re.search(r"(?:united\s+states\s+dollars|\busd\b|\bus\$\b)", tl):
            return "USD", "USD", 1.0, False

    if re.search(r"(?:rupees\s+in\s+millions|rs\.?\s*mn|lkr\s*(?:in\s+)?millions|\(rs\.?\s*m\)|rupees\s+millions)", tl):
        return "LKR", "LKR millions", 1_000_000.0, False
    if re.search(r"(?:sri\s+lankan?\s+rupees\s+thousands|rupees\s+(?:in\s+)?thousands|rs\.?\s*[-'’`.,]?\s*[s5]?[0o]{3}s?\b|lkr\.?\s*[-'’`.,]?\s*[0o]{3}s?\b|\bin\s+['’`.,]?(?:thousands|[0o]{3}s?)\b|all\s+(?:amounts|values)\s+are\s+(?:in\s+)?(?:sri\s+lankan?\s+rupees\s+thousands|lkr\.?['’`.,]?[0o]{3}s?))", tl):
        return "LKR", "LKR thousands", 1000.0, False
    if re.search(r"(?:\brs\.?(?!\s*[-'’`.,]?[s5]?[0o]{3}s?)|\blkr\b|sri\s+lankan?\s+rupees)", tl):
        return "LKR", "LKR", 1.0, False

    if doc_hint == "USD":
        return "USD", "AMBIGUOUS_UNIT", 1.0, True
    return "LKR", "AMBIGUOUS_UNIT", 1.0, True


def _infer_fye_month(rows_header_text: str, doc_meta: dict[str, Any]) -> int:
    """Detect whether issuer follows March (3), December (12), or June (6) FYE."""
    tl = rows_header_text.lower()
    if any(k in tl for k in ("31st march", "31 march", "31.03.", "31-mar-", "1st april", "01 april", "six months ended 30", "6 months ended 30")):
        return 3
    if any(k in tl for k in ("31st december", "31 december", "31.12.", "31-dec-", "1st january", "01 january", "nine months ended 30", "9 months ended 30", "9 month to")):
        return 12
    if any(k in tl for k in ("30th june", "30 june", "30.06.", "30-jun-")):
        return 6
    return int(doc_meta.get("fye_month_hint", 3))


def _resolve_period_fields(
    hdr_dict: dict[str, str],
    global_banner: str,
    stmt_type: str,
    fye_month: int,
    report_end_hint: pd.Timestamp,
) -> dict[str, Any]:
    """
    Compute exact `(PeriodStart, PeriodEnd, FiscalYear, FiscalQuarter, PeriodType, ComparativeFlag, IsRestated)`
    prioritizing per-column narrow headers first, then spanning column headers, then global title banner.
    """
    narrow = hdr_dict.get("narrow", "")
    span = hdr_dict.get("span", "")
    col_header = hdr_dict.get("full", f"{narrow} {span}".strip())
    nhl = narrow.lower()
    chl = col_header.lower()
    gbl = global_banner.lower()

    is_restated = bool("restated" in chl or "re-stated" in chl or "revised" in chl)

    # Extract explicit year: narrow column header strictly precedes span! (supports '25, '26, 2025, 2026)
    yr_narrow = re.findall(r"\b(201\d|202\d)\b", narrow)
    yr_span = re.findall(r"\b(201\d|202\d)\b", span)
    dt_explicit = _parse_date_string(narrow) or _parse_date_string(span)
    col_year = dt_explicit.year if dt_explicit is not None else (int(yr_narrow[-1]) if yr_narrow else (int(yr_span[-1]) if yr_span else None))

    if dt_explicit is None and not col_year and global_banner:
        for m in re.finditer(r"(\d{1,2}(?:st|nd|rd|th)?[\s./-]+[A-Za-z]+[\s./-]+'?\d{2,4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", global_banner):
            d_cand = _parse_date_string(m.group(1))
            if d_cand is not None:
                dt_explicit = d_cand
                break

    # Duration detection: strictly per-column first, only global banner if column header had zero duration clues
    def _dur_from(s: str) -> tuple[str, int] | None:
        if re.search(r"\b(?:for\s+the\s+year|year\s+ended|twelve\s+months|12\s+months|full\s+year)\b", s) or (
            stmt_type == "Balance Sheet" and (bool(re.search(r"\baudited\b", s)) or (dt_explicit is not None and dt_explicit.month == fye_month and "30" not in s))
        ):
            return "Annual", 12
        if re.search(r"\b(nine\s+months|9\s+months|09\s+months)\b", s):
            return "Nine-Month", 9
        if re.search(r"\b(six\s+months|6\s+months|06\s+months|half\s+year)\b", s):
            return "Half-Year", 6
        if re.search(r"\b(three\s+months|3\s+months|03\s+months|quarter\s+ended)\b", s):
            return "Quarterly", 3
        return None

    dur = _dur_from(nhl) or _dur_from(chl)
    if dur is None and stmt_type != "Balance Sheet" and dt_explicit is not None and dt_explicit.month == fye_month and ("year" in chl or bool(re.search(r"\baudited\b", chl))):
        dur = ("Annual", 12)
    if dur is None:
        dur = _dur_from(gbl)

    if stmt_type == "Balance Sheet":
        months = 0
        if bool(re.search(r"\baudited\b", chl)) or (dt_explicit is not None and dt_explicit.month == fye_month):
            p_type = "Annual"
        else:
            p_type = "Nine-Month" if fye_month == 12 else "Half-Year"
    elif dur is not None:
        p_type, months = dur
    else:
        p_type = "Nine-Month" if fye_month == 12 else "Half-Year"
        months = 9 if fye_month == 12 else 6

    # Resolve exact PeriodEnd date
    if dt_explicit is not None and stmt_type == "Balance Sheet":
        p_end = dt_explicit
        if col_year and p_end.year != col_year:
            p_end = pd.Timestamp(year=col_year, month=p_end.month, day=p_end.day)
    elif dt_explicit is not None and p_type == "Annual":
        p_end = dt_explicit
        if col_year and p_end.year != col_year:
            p_end = pd.Timestamp(year=col_year, month=p_end.month, day=p_end.day)
    else:
        base_end = report_end_hint if pd.notna(report_end_hint) else (dt_explicit or pd.Timestamp("2026-09-30"))
        if p_type == "Annual":
            end_m = fye_month
            end_d = 31 if end_m in (3, 12) else 30
            yr = col_year or base_end.year
            p_end = pd.Timestamp(year=yr, month=end_m, day=end_d)
        else:
            yr = col_year or base_end.year
            p_end = pd.Timestamp(year=yr, month=base_end.month, day=base_end.day)

    # Guard against comparative Dec-FYE year-end landing after interim report_end_hint (e.g. 2026-12-31 for Sep-2026 filing)
    if p_end > (report_end_hint + pd.Timedelta(days=31)):
        p_end = pd.Timestamp(year=report_end_hint.year - (1 if p_type == "Annual" else 0), month=p_end.month, day=min(p_end.day, 28 if p_end.month == 2 else 31))
        if stmt_type != "Balance Sheet":
            p_start = (p_end + pd.Timedelta(days=1)) - pd.DateOffset(months=months)

    if stmt_type == "Balance Sheet":
        p_start = pd.NaT
    else:
        p_start = (p_end + pd.Timedelta(days=1)) - pd.DateOffset(months=months)

    # Fiscal Year & Fiscal Quarter aligned to FYE month
    if fye_month == 3:
        fy = p_end.year if p_end.month <= 3 else p_end.year + 1
        # Apr-Jun=Q1, Jul-Sep=Q2, Oct-Dec=Q3, Jan-Mar=Q4
        q_idx = ((p_end.month - 4) % 12) // 3 + 1
    elif fye_month == 6:
        fy = p_end.year if p_end.month <= 6 else p_end.year + 1
        q_idx = ((p_end.month - 7) % 12) // 3 + 1
    else:
        fy = p_end.year
        q_idx = (p_end.month - 1) // 3 + 1

    fq = f"FY{fy}" if p_type == "Annual" else f"Q{q_idx}"
    cur_ref_year = report_end_hint.year if pd.notna(report_end_hint) else 2026
    comp_flag = bool(p_end.year < cur_ref_year or (p_type == "Annual" and p_end < pd.Timestamp(year=cur_ref_year, month=9, day=1)))

    return {
        "PeriodStart": p_start,
        "PeriodEnd": p_end,
        "FiscalYear": int(fy),
        "FiscalQuarter": fq,
        "PeriodType": p_type,
        "PeriodMonths": 0 if stmt_type == "Balance Sheet" else months,
        "ComparativeFlag": comp_flag,
        "IsRestated": is_restated,
    }


def _match_metric(label: str, stmt_type: str) -> str | None:
    clean = re.sub(r"[^A-Za-z0-9\s/&(),.\-]", " ", label).strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean or len(clean) < 3:
        return None
    for code, st, inc_re, exc_re in METRIC_RULES:
        if st != stmt_type:
            continue
        if exc_re is not None and exc_re.search(clean):
            continue
        if inc_re.search(clean):
            return code
    return None


def _cluster_numeric_columns(rows: list[list[WordToken]], start_row_idx: int) -> list[dict[str, Any]]:
    """Cluster X1 right edges of numeric tokens across statement body rows into columns."""
    pts: list[tuple[float, float, float, bool, float]] = []  # (xc, x0, x1, is_pct, abs_val)
    for r in rows[start_row_idx:]:
        m_row = _merge_adjacent_row_tokens(r)
        # Check if row contains at least one alphabetic label word (prevents random page footers)
        alpha_words = [t for t in m_row if re.search(r"[A-Za-z]{2,}", t.text) and t.text.lower() not in ("nil",)]
        if not alpha_words:
            continue
        for t in m_row:
            is_n, val, is_pct, is_dash, _ = _parse_numeric_token(t.text)
            if is_n and val is not None:
                pts.append(((t.x0 + t.x1) * 0.5, t.x0, t.x1, is_pct, 0.0 if is_dash else float(val)))

    if len(pts) < 4:
        return []

    # Sort by right edge x1 (financial numbers in tables align cleanly by right edge)
    pts.sort(key=lambda z: z[2])
    clusters: list[list[tuple[float, float, float, bool, float]]] = [[pts[0]]]
    for item in pts[1:]:
        cur_x1 = np.median([z[2] for z in clusters[-1]])
        if abs(item[2] - cur_x1) <= 22.0:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    cols = []
    for cl in clusters:
        if len(cl) < 3:
            continue
        x0_min = float(np.percentile([z[1] for z in cl], 10))
        x1_max = float(np.percentile([z[2] for z in cl], 95))
        pct_ratio = float(sum(1 for z in cl if z[3])) / float(len(cl))
        abs_vals = [abs(z[4]) for z in cl]
        cols.append({
            "x0": x0_min - 8.0,
            "x1": x1_max + 8.0,
            "xc": float(np.median([z[0] for z in cl])),
            "x1_med": float(np.median([z[2] for z in cl])),
            "pct_body_ratio": pct_ratio,
            "med_abs": float(np.median(abs_vals)) if abs_vals else 0.0,
            "count": len(cl),
        })
    return cols


def _build_column_headers(
    rows: list[list[WordToken]],
    hdr_end_idx: int,
    cols: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Associate both narrow per-column headers and multi-column spanning headers
    (`Group`, `Company`, `Three months ended ...`) strictly by horizontal geometry.
    """
    col_headers = [{"narrow": [], "span": []} for _ in cols]
    for r in rows[:hdr_end_idx]:
        if not r:
            continue
        phrases: list[dict[str, Any]] = []
        cur = [r[0]]
        for t in r[1:]:
            is_yr_pair = bool(re.match(r"^20\d{2}$", cur[-1].text.strip()) and re.match(r"^20\d{2}$", t.text.strip()))
            if (t.x0 - cur[-1].x1) <= 10.0 and not is_yr_pair:
                cur.append(t)
            else:
                phrases.append({
                    "x0": cur[0].x0,
                    "x1": cur[-1].x1,
                    "xc": (cur[0].x0 + cur[-1].x1) * 0.5,
                    "text": " ".join(w.text for w in cur),
                })
                cur = [t]
        phrases.append({
            "x0": cur[0].x0,
            "x1": cur[-1].x1,
            "xc": (cur[0].x0 + cur[-1].x1) * 0.5,
            "text": " ".join(w.text for w in cur),
        })

        for ph in phrases:
            # Standalone year tokens (e.g. `2026`, `2025`) go ONLY to single closest column!
            if re.match(r"^20\d{2}$", ph["text"].strip()):
                best_j = min(range(len(cols)), key=lambda j: abs(ph["xc"] - cols[j]["xc"]))
                if abs(ph["xc"] - cols[best_j]["xc"]) <= 45.0:
                    col_headers[best_j]["narrow"].append(ph["text"])
                continue

            # Find overlapping numeric columns
            overlaps = [
                j for j, c in enumerate(cols)
                if min(ph["x1"], c["x1"] + 6.0) - max(ph["x0"], c["x0"] - 6.0) > 2.0
            ]
            if len(overlaps) == 1:
                col_headers[overlaps[0]]["narrow"].append(ph["text"])
            elif len(overlaps) > 1:
                for j in overlaps:
                    col_headers[j]["span"].append(ph["text"])
            else:
                dists = [(abs(ph["xc"] - c["xc"]), j) for j, c in enumerate(cols)]
                best_d, best_j = min(dists)
                if best_d <= 35.0:
                    col_headers[best_j]["narrow"].append(ph["text"])

    return [
        {
            "narrow": " ".join(h["narrow"]),
            "span": " ".join(h["span"]),
            "full": " ".join(h["narrow"] + h["span"]),
        }
        for h in col_headers
    ]


def _extract_statement_of_changes_in_equity(
    page: PageData,
    doc_meta: dict[str, Any],
    report_end_hint: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Specialized block parser for Statement of Changes in Equity pages."""
    rows = _group_tokens_into_rows(page.tokens, y_tol=4.5)
    top_banner = " ".join(" ".join(t.text for t in r) for r in rows[:14])
    cur, unit_lbl, mult, ambig_u = _detect_currency_unit_from_text("\n".join(" ".join(t.text for t in r) for r in rows[:20]), doc_meta.get("currency_hint", "LKR"))
    is_cons = "company" not in top_banner.lower() or "group" in top_banner.lower()

    # Find Balance rows and numeric columns
    hdr_end = min(len(rows), 10)
    for idx, r in enumerate(rows[:20]):
        line_str = " ".join(t.text for t in r)
        if re.search(r"balance\s+(?:as\s+)?at", line_str, re.I):
            hdr_end = idx
            break

    cols = _cluster_numeric_columns(rows, hdr_end)
    if not cols:
        return []

    hdrs = _build_column_headers(rows, hdr_end, cols)
    # Identify rightmost Total Equity column + Stated Capital + Retained Earnings columns
    total_idx = len(cols) - 1
    stated_idx = None
    re_idx = None
    for j, h in enumerate(hdrs):
        hl = h["full"].lower()
        if "stated capital" in hl or "share capital" in hl:
            stated_idx = j
        if "retained" in hl:
            re_idx = j
        if "total" in hl and "equity" in hl:
            total_idx = j

    fye_month = _infer_fye_month(page.raw_text, doc_meta)
    cur_p_start: pd.Timestamp | None = None
    facts: list[dict[str, Any]] = []

    for r in rows[hdr_end:]:
        m_row = _merge_adjacent_row_tokens(r)
        # Reconstruct row label left of first column
        lbl_toks = [t for t in m_row if not _parse_numeric_token(t.text)[0] or t.x1 < cols[0]["x0"]]
        label = " ".join(t.text for t in lbl_toks).strip()
        if not label:
            continue

        # Map values across detected columns
        col_vals: dict[int, float] = {}
        for t in m_row:
            is_n, val, is_pct, _, _ = _parse_numeric_token(t.text)
            if not is_n or val is None or is_pct:
                continue
            xc = (t.x0 + t.x1) * 0.5
            for j, c in enumerate(cols):
                if c["x0"] <= xc <= c["x1"]:
                    col_vals[j] = val
                    break

        tot_val = col_vals.get(total_idx)
        if tot_val is None and col_vals:
            tot_val = col_vals[max(col_vals.keys())]

        dt_bal = _parse_date_string(label) if re.search(r"balance\s+(?:as\s+)?at", label, re.I) else None
        ll = label.lower()

        metric = None
        if dt_bal is not None:
            if cur_p_start is None or dt_bal.month in (1, 4, 7) and dt_bal.day == 1:
                cur_p_start = dt_bal
                metric = "EQUITY_OPENING"
                p_end = dt_bal
            else:
                metric = "EQUITY_CLOSING"
                p_end = dt_bal
        elif "other comprehensive income" in ll or re.search(r"\boci\b", ll):
            metric = "EQUITY_OCI"
            p_end = report_end_hint
        elif "dividend" in ll:
            metric = "EQUITY_DIVIDENDS"
            p_end = report_end_hint
        elif re.search(r"(?:issue\s+of\s+shares|share\s+issue|capital\s+contribution)", ll):
            metric = "EQUITY_SHARE_CAPITAL_CHANGE"
            p_end = report_end_hint
            if stated_idx is not None and stated_idx in col_vals:
                tot_val = col_vals[stated_idx]

        if metric and tot_val is not None and pd.notna(p_end):
            if metric in ("EQUITY_OPENING", "EQUITY_CLOSING"):
                p_start = pd.NaT
                p_type = "Annual" if p_end.month == fye_month else ("Nine-Month" if fye_month == 12 else "Half-Year")
            else:
                p_start = pd.Timestamp(cur_p_start) if cur_p_start is not None else ((p_end + pd.Timedelta(days=1)) - pd.DateOffset(months=(9 if fye_month == 12 else 6)))
                p_type = "Nine-Month" if fye_month == 12 else "Half-Year"

            fy = p_end.year if (fye_month == 12 or p_end.month <= fye_month) else p_end.year + 1
            facts.append({
                "StatementType": "Equity",
                "MetricCode": metric,
                "OriginalMetricName": label[:200],
                "PeriodStart": p_start,
                "PeriodEnd": p_end,
                "FiscalYear": int(fy),
                "FiscalQuarter": f"Q{((p_end.month - 1) // 3) + 1}",
                "PeriodType": p_type,
                "OriginalValue": float(tot_val),
                "Currency": cur,
                "Unit": unit_lbl,
                "Multiplier": float(mult),
                "IsAmbiguousUnit": ambig_u,
                "SourcePage": page.page_num,
                "ExtractionMethod": page.method,
                "BaseConfidence": page.text_quality_score,
                "IsConsolidated": bool(is_cons),
                "IsRestated": bool("restated" in ll or "restated" in top_banner.lower()),
                "ComparativeFlag": bool(p_end.year < report_end_hint.year),
                "Notes": f"Equity col={hdrs[total_idx]['full'][:60]}",
            })

        # Track closing reset when second period block starts after Sep/Mar closing
        if metric == "EQUITY_CLOSING" and dt_bal is not None:
            cur_p_start = None

    return facts


def parse_financial_pdf(
    pdf_pages: list[PageData],
    inv_row: dict[str, Any],
    doc_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parse all financial statement pages inside one PDF into structured fact dicts.
    """
    report_end_hint = pd.to_datetime(inv_row.get("FinancialPeriodEndHint"), errors="coerce")
    if pd.isna(report_end_hint):
        report_end_hint = pd.Timestamp("2026-09-30")

    doc_text = "\n".join(p.raw_text[:2000] for p in pdf_pages[:8])
    fye_month = _infer_fye_month(doc_text, doc_meta)
    doc_meta = {**doc_meta, "fye_month_hint": fye_month, "currency_hint": inv_row.get("CurrencyHint", "LKR")}

    all_facts: list[dict[str, Any]] = []
    stmts_found: set[str] = set()
    page_logs: list[dict[str, Any]] = []
    last_known_unit_by_cur: dict[str, tuple[str, str, float]] = {}

    for page in pdf_pages:
        stmt_types = _classify_statement_page(page, page.raw_text)
        if not stmt_types:
            continue

        # Statement of Changes in Equity uses block equity-column extraction
        if stmt_types == ["Equity"]:
            eq_facts = _extract_statement_of_changes_in_equity(page, doc_meta, report_end_hint)
            if eq_facts:
                stmts_found.add("Equity")
                all_facts.extend(eq_facts)
                page_logs.append({"page": page.page_num, "statement": "Equity", "facts": len(eq_facts), "method": page.method})
            continue

        stmt = stmt_types[0]
        rows = _group_tokens_into_rows(page.tokens, y_tol=4.5)
        if len(rows) < 6:
            continue

        # Locate body start row (first row matching a known line-item regex!)
        body_start = None
        for idx, r in enumerate(rows):
            line_s = " ".join(t.text for t in r)
            # Check clean text excluding pure numbers
            lbl_cand = re.sub(r"\(?[\d,]+(?:\.\d+)?\)?%?", " ", line_s).strip()
            if _match_metric(lbl_cand, stmt) is not None or re.search(r"^(?:revenue\s*$|equity\s+and\s+liabilities\s*$|assets\s*$|non[- ]?current\s+assets\s*$|cash\s+flows?\s+from\s+operating)", lbl_cand, re.I):
                body_start = max(1, idx)
                break
        if body_start is None:
            continue

        cols = _cluster_numeric_columns(rows, body_start)
        if not cols:
            continue

        hdrs = _build_column_headers(rows, body_start, cols)
        # Title banner lines + bottom table footer lines (HHL/UML/WIND place units in footer!)
        title_banner = "\n".join(" ".join(t.text for t in r) for r in rows[:min(body_start, 5)])
        page_banner = "\n".join(" ".join(t.text for t in r) for r in rows[:min(body_start + 1, 20)])
        footer_banner = "\n".join(" ".join(t.text for t in r) for r in rows[-8:])
        has_restated_footnote = bool(re.search(r"\brestated\b", footer_banner, re.I))
        unit_context = f"{page_banner}\n{footer_banner}"
        max_col_med = max((c["med_abs"] for c in cols), default=0.0)

        # Identify which column indices are valid data columns (skip Variance / % Change / Note columns)
        col_metas: list[dict[str, Any] | None] = []
        valid_group_col_seen = False
        valid_comp_col_seen = False
        for j, (c, h) in enumerate(zip(cols, hdrs)):
            hl = h["full"].lower()
            nhl = h["narrow"].lower()
            is_tiny_var = (len(cols) >= 3 and j > 0 and max_col_med >= 5000.0 and c["med_abs"] < 350.0)
            is_var_col = (
                c["pct_body_ratio"] >= 0.45
                or bool(re.search(r"\b(?:variance|change|\+/\(-\)|increase|decrease|note[s]?)\b", nhl))
                or nhl.strip() == "%"
                or bool(re.search(r"^%\s*", nhl.strip()) and "rs" not in nhl and "usd" not in nhl)
                or is_tiny_var
            )
            if is_var_col:
                col_metas.append(None)
                continue

            # Determine Consolidation scope per column (`Group` vs `Company`)
            if re.search(r"\b(?:company|bank\s+only|separate)\b", hl) and not re.search(r"\bgroup\b", nhl):
                is_cons = False
                valid_comp_col_seen = True
            elif re.search(r"\b(?:group|consolidated)\b", hl):
                is_cons = True
                valid_group_col_seen = True
            elif "company" in title_banner.lower() and "group" not in title_banner.lower():
                is_cons = False
            elif "group" in title_banner.lower() and "company" not in title_banner.lower():
                is_cons = True
            elif "company" in page_banner.lower() and "group" not in page_banner.lower():
                is_cons = False
            elif "group" in page_banner.lower() and "company" not in page_banner.lower():
                is_cons = True
            else:
                # Dual Group | Company side-by-side when headers merged across banner
                is_cons = True if (j < len(cols) / 2 or not valid_group_col_seen) else False

            cur, unit_lbl, mult, ambig_u = _detect_currency_unit_from_text(
                f"{h['narrow']} {h['span']} {unit_context}", doc_meta["currency_hint"]
            )
            # Also check last_known_unit on same PDF if WIND/other LKR statements omitted footer on one page
            if cur == "USD" and ambig_u and doc_meta.get("currency_hint") == "LKR":
                cur = "LKR"
            if not ambig_u and mult >= 1000.0:
                last_known_unit_by_cur[cur] = (cur, unit_lbl, mult)
            elif not ambig_u and cur == "LKR" and mult == 1.0 and cur not in last_known_unit_by_cur:
                last_known_unit_by_cur[cur] = (cur, unit_lbl, mult)
            elif (ambig_u or (page.method == "OCR" and stmt == "Balance Sheet" and mult == 1.0)) and cur in last_known_unit_by_cur:
                cur, unit_lbl, mult = last_known_unit_by_cur[cur]
                ambig_u = False
            p_info = _resolve_period_fields(
                hdr_dict=h,
                global_banner=title_banner,
                stmt_type=stmt,
                fye_month=fye_month,
                report_end_hint=report_end_hint,
            )
            col_metas.append({
                "col_idx": j,
                "x0": c["x0"],
                "x1": c["x1"],
                "is_consolidated": is_cons,
                "currency": cur,
                "unit": unit_lbl,
                "multiplier": mult,
                "ambig_u": ambig_u,
                **p_info,
                "header_dbg": h["full"][:80],
            })

        # Extract line items row by row
        page_fact_count = 0
        # For Cash Flow, resolve ambiguous Depreciation/Amortization/etc. and track statement rows
        for r in rows[body_start:]:
            m_row = _merge_adjacent_row_tokens(r)
            # Extract numeric tokens by matching X-band AND non-numeric tokens for metric label
            row_col_vals: dict[int, tuple[float, WordToken]] = {}
            label_parts: list[str] = []

            for t in m_row:
                xc = (t.x0 + t.x1) * 0.5
                in_col_j = None
                for j, cm in enumerate(col_metas):
                    c = cols[j]
                    if c["x0"] <= xc <= c["x1"]:
                        in_col_j = j
                        break

                is_n, val, is_pct, _, has_star = _parse_numeric_token(t.text)
                if in_col_j is not None and is_n and val is not None:
                    if col_metas[in_col_j] is not None and not is_pct:
                        row_col_vals[in_col_j] = (val, t, has_star)
                else:
                    # Skip standalone percentage numbers or solitary Note numbers right next to first col
                    if not is_pct:
                        label_parts.append(t.text)

            label = " ".join(label_parts).strip()
            # Clean trailing/stray punctuation or note indices
            label_clean = re.sub(r"\s+", " ", label).strip(" :-.")
            metric = _match_metric(label_clean, stmt)
            if metric is None or not row_col_vals:
                continue

            for j, (raw_val, tok, has_star) in row_col_vals.items():
                cm = col_metas[j]
                if cm is None:
                    continue

                is_restated_fact = bool(cm["IsRestated"] or (has_star and has_restated_footnote))
                # Per-share items NEVER multiply by '000 or Millions!
                base_conf = float(min(page.text_quality_score, tok.conf))
                notes_extra = f"Header={cm['header_dbg']}"
                if has_star and has_restated_footnote:
                    notes_extra += " | FOOTNOTE_STAR_RESTATED"
                final_raw_val = float(raw_val)
                if metric in ("EPS_BASIC", "EPS_DILUTED"):
                    use_unit = f"{cm['currency']} per share"
                    use_mult = 1.0
                    use_ambig = False
                else:
                    use_unit = cm["unit"]
                    use_mult = cm["multiplier"]
                    use_ambig = cm["ambig_u"]
                    # Clean OCR thousands-separator period confusion (`4,727.670` or `4.581,539` in integer LKR/'000 tables)
                    raw_tok = tok.text.strip().rstrip("*^†")
                    tok_inner = raw_tok[1:-1].strip() if (raw_tok.startswith("(") and raw_tok.endswith(")")) else raw_tok
                    if (
                        page.method == "OCR"
                        and use_unit in ("LKR thousands", "LKR", "USD", "USD thousands")
                        and re.match(r"^(?:\d{1,3}[.,])+\d{3}$", tok_inner)
                    ):
                        repaired_abs = float(re.sub(r"[.,]", "", tok_inner))
                        final_raw_val = -repaired_abs if raw_tok.startswith("(") and raw_tok.endswith(")") else repaired_abs
                        base_conf = min(base_conf, 0.78)
                        notes_extra += f" | OCR_SEPARATOR_REPAIRED(raw={raw_tok})"
                    elif page.method == "OCR" and round(abs(final_raw_val) % 1.0, 4) > 1e-4:
                        base_conf = min(base_conf, 0.62)
                        notes_extra += f" | OCR_DECIMAL_ARTIFACT_SUSPECT(raw={raw_tok})"

                all_facts.append({
                    "StatementType": stmt,
                    "MetricCode": metric,
                    "OriginalMetricName": label[:200],
                    "PeriodStart": cm["PeriodStart"],
                    "PeriodEnd": cm["PeriodEnd"],
                    "FiscalYear": cm["FiscalYear"],
                    "FiscalQuarter": cm["FiscalQuarter"],
                    "PeriodType": cm["PeriodType"],
                    "OriginalValue": final_raw_val,
                    "Currency": cm["currency"],
                    "Unit": use_unit,
                    "Multiplier": float(use_mult),
                    "IsAmbiguousUnit": use_ambig,
                    "SourcePage": page.page_num,
                    "ColIndex": int(cm["col_idx"]),
                    "ExtractionMethod": page.method,
                    "BaseConfidence": base_conf,
                    "IsConsolidated": bool(cm["is_consolidated"]),
                    "IsRestated": is_restated_fact,
                    "ComparativeFlag": bool(cm["ComparativeFlag"]),
                    "Notes": notes_extra,
                })
                page_fact_count += 1

        if page_fact_count > 0:
            stmts_found.add(stmt)
            page_logs.append({"page": page.page_num, "statement": stmt, "facts": page_fact_count, "method": page.method})

    return all_facts, {"statements_found": sorted(stmts_found), "page_logs": page_logs}
