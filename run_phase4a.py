"""
CLI Entry Point for Stockpulse Phase 4A: Short-Term Modeling Dataset Preparation & Target Engineering.

Usage:
    python run_phase4a.py
    python run_phase4a.py --incremental
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from phase4a_dataset.features_eligibility_and_splits import run_phase4a_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stockpulse Phase 4A short-term modeling dataset pipeline.")
    parser.add_argument("--incremental", action="store_true", help="Skip recomputation when locked upstream SHA-256 signature is unchanged.")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root directory.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    manifest = run_phase4a_pipeline(Path(args.repo_root), incremental=args.incremental)
    print(json.dumps(manifest["dataset_summary"], indent=2))


if __name__ == "__main__":
    main()
