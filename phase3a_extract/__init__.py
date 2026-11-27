"""
Stockpulse Phase 3A – Historical Financial Statement Extraction & Normalization
=============================================================================
Modular pipeline for inventorying official CSE PDFs, mapping issuer & share-class
identities against Phase 1B, extracting vector + Windows Media OCR tables with
spatial column detection, normalizing periods/units/consolidation/restatements,
and validating point-in-time historical availability.
"""

from .inventory_and_mapping import (
    build_company_mapping,
    build_pdf_inventory,
    summarize_legacy_extraction,
)
from .pdf_reader_and_ocr import extract_pdf_pages_tokens
from .statement_parser import parse_financial_pdf
from .normalize_and_validate import (
    build_review_and_quality_tables,
    derive_quarterly_facts,
    detect_restatements,
    normalize_and_validate_facts,
)

__all__ = [
    "build_company_mapping",
    "build_pdf_inventory",
    "summarize_legacy_extraction",
    "extract_pdf_pages_tokens",
    "parse_financial_pdf",
    "normalize_and_validate_facts",
    "derive_quarterly_facts",
    "detect_restatements",
    "build_review_and_quality_tables",
]
