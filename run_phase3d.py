"""
Stockpulse Phase 3D Runner – Unified Point-in-Time Fundamental Integration & Strategy Validation
================================================================================================
Usage:
  python run_phase3d.py
  python run_phase3d.py --incremental
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from phase3d_features.asof_and_valuation_engine import DEFAULT_MAX_CARRY_DAYS, DEFAULT_STALE_DAYS
from phase3d_features.strategy_integration_and_audit import run_phase3d_pipeline


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute Stockpulse Phase 3D unified PIT integration pipeline.")
    parser.add_argument("--incremental", action="store_true", help="Skip full rebuild if upstream SHA-256 and deliverables are already valid.")
    parser.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS, help="Default freshness threshold in days (default: 270).")
    parser.add_argument("--max-carry-days", type=int, default=DEFAULT_MAX_CARRY_DAYS, help="Maximum forward-fill horizon beyond PeriodEnd (default: 548).")
    args = parser.parse_args()

    _configure_logging()
    repo_root = Path(__file__).resolve().parent
    manifest = run_phase3d_pipeline(
        repo_root,
        incremental=args.incremental,
        stale_threshold_days=args.stale_days,
        max_carry_days=args.max_carry_days,
    )
    print(json.dumps({
        "phase": manifest["phase"],
        "runtime_seconds": manifest["runtime_seconds"],
        "upstream_immutability_verified": manifest["upstream_immutability_verified"],
        "causal_verification_passed": manifest["causal_verification_passed"],
        "strategy_readiness_decisions": manifest["strategy_readiness_decisions"],
        "deliverables_count": len(manifest["expected_deliverables"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
