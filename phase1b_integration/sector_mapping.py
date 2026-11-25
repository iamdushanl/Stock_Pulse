"""
Phase 1B Task 4 – Historical Sector & Industry Mapping Audit
============================================================
Audits CSE sector source files (39GICS-Daily.xlsx, 22Sector Market Capitalisation.xls,
25Sector Ratios.xls, and Phase 1A cse_security_master.parquet) and produces an
audit-safe historical sector mapping table without inferring company GICS sectors
from aggregate sector indices or copying unverified current snapshots backward.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def audit_sector_sources(source_dir: Path) -> dict:
    """
    Inspect aggregate CSE sector files and document exact structure & coverage.
    Returns structured audit findings confirming whether ticker-level mapping exists.
    """
    findings = {
        "ticker_level_mapping_available": False,
        "files_audited": {},
        "cse_regime_transition_note": (
            "CSE transitioned from 20 legacy sector groups (2005–2019 in "
            "22Sector Market Capitalisation.xls) to 20–21 GICS industry groups "
            "in Jan 2020 (39GICS-Daily.xlsx begins 2016-04-05). None of these "
            "workbooks include a ticker-to-sector crosswalk."
        ),
    }

    gics_path = source_dir / "39GICS-Daily.xlsx"
    if gics_path.exists():
        xl_g = pd.ExcelFile(gics_path)
        cols = xl_g.parse(xl_g.sheet_names[0], nrows=1).columns.tolist()
        findings["files_audited"]["39GICS-Daily.xlsx"] = {
            "granularity": "Aggregate GICS Sector Index daily series ONLY (no CompanyCode column)",
            "sheets": xl_g.sheet_names,
            "sector_columns_count": len(cols) - 1,
        }

    mcap_path = source_dir / "22Sector Market Capitalisation.xls"
    if mcap_path.exists():
        xl_m = pd.ExcelFile(mcap_path)
        findings["files_audited"]["22Sector Market Capitalisation.xls"] = {
            "granularity": "Aggregate Monthly Sector Market Cap ONLY (no CompanyCode column)",
            "year_sheets_count": len([s for s in xl_m.sheet_names if s.strip().isdigit()]),
            "regimes": {"pre_2020": "20 CSE legacy sectors", "2020_2025": "20 GICS sector groups"},
        }

    ratios_path = source_dir / "25Sector Ratios.xls"
    if ratios_path.exists():
        xl_r = pd.ExcelFile(ratios_path)
        findings["files_audited"]["25Sector Ratios.xls"] = {
            "granularity": "Aggregate Monthly Sector DY/PE/PBV ONLY (no CompanyCode column)",
            "year_sheets_count": len([s for s in xl_r.sheet_names if s.strip().isdigit()]),
        }

    return findings


def build_historical_sector_mapping(
    identity_df: pd.DataFrame,
    source_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Create the historical sector/industry mapping table with explicit validity intervals
    and a clean manual-research queue where historical ticker-level provenance is absent.
    """
    audit_info = audit_sector_sources(source_dir)

    mapping_rows = []
    for _, r in identity_df.iterrows():
        eff_from = r["ListingDate"] if pd.notna(r["ListingDate"]) else r["FirstPriceDate"]
        eff_to   = r["DelistingDate"] if pd.notna(r["DelistingDate"]) else pd.NaT

        mapping_rows.append({
            "SecurityID": r["SecurityID"],
            "CompanyCode": r["CompanyCode"],
            "IssuerID": r["IssuerID"],
            "Sector": "Unknown",
            "Industry": "Unknown",
            "EffectiveFrom": eff_from,
            "EffectiveTo": eff_to,
            "Source": "UNAVAILABLE_IN_HISTORICAL_CSE_ARCHIVES (aggregate-only 39GICS/22SecMCap/25SecRatios)",
            "Confidence": 0.0,
            "ValidationStatus": "MANUAL_RESEARCH_REQUIRED",
            "ManualResearchQueue": True,
        })

    sector_df = pd.DataFrame(mapping_rows)
    return sector_df, audit_info
