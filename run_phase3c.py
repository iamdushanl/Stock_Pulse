"""
Stockpulse Phase 3C CLI Runner (`run_phase3c.py`)
=================================================
Reproducible, deterministic entrypoint implementing:
  Phase 3C – Fundamental Feature Engineering, Financial Quality, Growth & Valuation Metrics
Strictly preserves all Phase 1A–3B datasets and verifies upstream SHA-256 immutability.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from phase3c_features import run_phase3c_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Stockpulse Phase 3C Fundamental Feature Engineering Runner")
    parser.add_argument(
        "--skip-pdf-unit-scan",
        action="store_true",
        help="Skip scanning ~340 PDF page headers for unit-shift corroboration (faster test dry-run).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse verified cached Workstream A flags and PIT Parquets if already present.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    repo_root = Path(__file__).resolve().parent
    t0 = time.time()
    manifest = run_phase3c_pipeline(
        repo_root=repo_root,
        scan_flagged_pdfs=not args.skip_pdf_unit_scan,
        incremental=args.incremental,
    )
    elapsed = time.time() - t0
    manifest["runtime_seconds"] = round(elapsed, 2)

    mpath = repo_root / "Dataset" / "processed" / "phase3c" / "phase3c_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    logging.getLogger("run_phase3c").info(
        "Phase 3C COMPLETE in %.2fs | PIT rows=%d across %d companies | 14 deliverables + manifest generated.",
        elapsed,
        manifest["pit_panel_summary"]["total_pit_company_date_rows"],
        manifest["pit_panel_summary"]["unique_companies_in_pit_panel"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
