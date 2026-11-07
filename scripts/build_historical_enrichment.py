#!/usr/bin/env python3
"""Export CSE auxiliary historical files already supported by utils.data_loader.

Outputs a separate enrichment dataset and leaves the original source files intact.
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from utils import data_loader

EXPORTS = {
    "market_indices": data_loader.load_market_indices,
    "market_statistics": data_loader.load_market_stats,
    "stock_splits": data_loader.load_splits,
    "listing_delisting_history": data_loader.load_listings,
    "sector_market_cap": data_loader.load_sector_market_cap,
    "dividends": data_loader.load_dividends,
    "sector_valuation_ratios": data_loader.load_sector_ratios,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Folder containing the existing CSE source XLS/XLSX files")
    parser.add_argument("--output", default="Dataset/enrichment/cse_historical")
    parser.add_argument("--securities-year", default="2025")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_folder": str(source), "created_at_utc": datetime.now(timezone.utc).isoformat(), "exports": {}}
    jobs = dict(EXPORTS)
    jobs["quoted_securities"] = lambda base_path=None: data_loader.load_securities_list(base_path=base_path, year=args.securities_year, equity_only=True)
    for name, loader in jobs.items():
        try:
            frame = loader(base_path=str(source))
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"Expected DataFrame, got {type(frame).__name__}")
            parquet_path = output / f"{name}.parquet"
            csv_path = output / f"{name}.csv"
            frame.to_parquet(parquet_path, index=False)
            frame.to_csv(csv_path, index=False)
            manifest["exports"][name] = {
                "rows": int(len(frame)), "columns": list(frame.columns),
                "parquet": str(parquet_path), "csv": str(csv_path),
                "date_min": str(frame["Date"].min()) if "Date" in frame and len(frame) else None,
                "date_max": str(frame["Date"].max()) if "Date" in frame and len(frame) else None,
                "status": "ok",
            }
            logging.info("%s: %s rows", name, f"{len(frame):,}")
        except Exception as exc:
            logging.exception("Could not export %s", name)
            manifest["exports"][name] = {"status": "error", "error": str(exc)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logging.info("Manifest written to %s", output / "manifest.json")

if __name__ == "__main__":
    main()