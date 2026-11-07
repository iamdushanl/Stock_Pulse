#!/usr/bin/env python3
"""Collect a reproducible CSE public-data snapshot for Stockpulse.

This collector stores raw API responses first, then writes normalized CSV
indexes. It does not modify the existing price-history dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

BASE = "https://www.cse.lk"
CDN = "https://cdn.cse.lk/"
USER_AGENT = "StockpulseAcademicResearch/1.0 (CSE public data; respectful low-rate client)"
TIMEOUT = 45
PAUSE_SECONDS = 0.4

GET_ENDPOINTS = ("smd/categories",)
POST_ENDPOINTS = (
    "allSecurityCode", "cntSecurity", "listAllSectors", "marketStatus",
    "marketSummery", "aspiData", "snpData", "allSectors", "dailyMarketSummery",
    "tradeSummary", "todaySharePrice", "topGainers", "topLooses",
    "mostActiveTrades", "getNewListingsRelatedNoticesAnnouncements",
    "getBuyInBoardAnnouncements", "approvedAnnouncement",
    "getFinancialAnnouncement", "circularAnnouncement", "directiveAnnouncement",
    "getNonComplianceAnnouncements", "detailedTrades",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def find_list(payload):
    """Find the first list nested in a CSE response without assuming one schema."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
        for value in payload.values():
            found = find_list(value)
            if found is not None:
                return found
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Keep heterogeneous API fields without losing fields that only occur later.
    columns = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            })


class CSEClient:
    def __init__(self, pause: float = PAUSE_SECONDS):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Origin": BASE,
            "Referer": BASE + "/",
        })
        self.pause = pause

    def request_json(self, endpoint: str, method: str = "POST", params=None, data=None):
        url = f"{BASE}/api/{endpoint}"
        last_error = None
        for attempt in range(4):
            try:
                response = self.session.request(
                    method, url, params=params, data=data, timeout=TIMEOUT
                )
                response.raise_for_status()
                payload = response.json()
                time.sleep(self.pause)
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                logging.warning("%s %s failed (attempt %s/4): %s", method, endpoint, attempt + 1, exc)
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"CSE endpoint failed: {endpoint}: {last_error}")

    def download(self, url: str, path: Path) -> tuple[str, int]:
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with self.session.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            with path.open("wb") as out:
                for chunk in response.iter_content(1024 * 128):
                    if chunk:
                        out.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        time.sleep(self.pause)
        return digest.hexdigest(), size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="Dataset/enrichment/cse_public",
                        help="Output directory (kept separate from original Dataset files)")
    parser.add_argument("--download-pdfs", action="store_true",
                        help="Download all PDFs listed in the financial-report and circular feeds")
    parser.add_argument("--company-details", action="store_true",
                        help="Fetch per-security company summary and profile (slower; one request per security)")
    parser.add_argument("--limit-companies", type=int, default=0,
                        help="Optional test limit; 0 means all active equity symbols")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS,
                        help="Delay between requests in seconds (default 0.4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(args.output)
    raw = root / "raw"
    root.mkdir(parents=True, exist_ok=True)
    client = CSEClient(args.pause)
    run_meta = {"source": BASE, "started_at_utc": utc_now(), "collector": "Stockpulse CSE public-data collector"}

    # Master universe and sector taxonomy.
    universe = client.request_json("allSecurityCode", "POST")
    write_json(raw / "allSecurityCode.json", universe)
    securities = find_list(universe) or []
    sectors = client.request_json("listAllSectors", "POST")
    write_json(raw / "listAllSectors.json", sectors)
    internal_ids = client.request_json("cntSecurity", "POST")
    write_json(raw / "cntSecurity.json", internal_ids)

    # Category dictionary is a GET endpoint in the current public-site API.
    try:
        categories = client.request_json("smd/categories", "GET")
        write_json(raw / "smd_categories.json", categories)
    except Exception as exc:
        logging.warning("Could not collect announcement categories: %s", exc)
        categories = {"_collection_error": str(exc)}

    # Public market state, indices, sector metrics and aggregate market activity.
    market = {}
    for endpoint in POST_ENDPOINTS:
        try:
            payload = client.request_json(endpoint, "POST")
            market[endpoint] = payload
            write_json(raw / f"{safe_name(endpoint)}.json", payload)
        except Exception as exc:
            logging.exception("Skipping endpoint %s: %s", endpoint, exc)
            market[endpoint] = {"_collection_error": str(exc), "_collected_at_utc": utc_now()}

    # Per-company current profile/summary (not historical point-in-time fundamentals).
    company_rows = []
    company_errors = []
    active = [
        item for item in securities
        if isinstance(item, dict)
        and str(item.get("active", "1")) in ("1", "True", "true")
        and item.get("symbol")
        and ".N" in str(item.get("symbol"))
    ]
    if args.limit_companies:
        active = active[:args.limit_companies]

    if args.company_details:
        for item in active:
            symbol = str(item["symbol"])
            safe_symbol = safe_name(symbol)
            for endpoint in ("companyInfoSummery",):
                try:
                    payload = client.request_json(endpoint, "POST", data={"symbol": symbol})
                    write_json(raw / "companies" / safe_symbol / f"{endpoint}.json", payload)
                    if endpoint == "companyInfoSummery" and isinstance(payload, dict):
                        info = payload.get("reqSymbolInfo") or {}
                        beta = payload.get("reqSymbolBetaInfo") or {}
                        company_rows.append({
                            **item, **info, **{f"beta_{k}": v for k, v in beta.items()},
                            "source_url": f"{BASE}/company-profile?symbol={symbol}",
                            "retrieved_at_utc": utc_now(),
                        })
                except Exception as exc:
                    company_errors.append({"symbol": symbol, "endpoint": endpoint, "error": str(exc)})
                    logging.warning("Company request failed: %s %s: %s", symbol, endpoint, exc)

    write_csv(root / "securities_master.csv", [
        {**row, "source_url": f"{BASE}/"} for row in securities if isinstance(row, dict)
    ])
    write_csv(root / "sectors_master.csv", [
        {**row, "retrieved_at_utc": utc_now()} for row in (find_list(sectors) or []) if isinstance(row, dict)
    ])
    write_csv(root / "company_current_snapshot.csv", company_rows)
    write_csv(root / "collection_errors.csv", company_errors)

    # Flatten the current disclosure and report feeds into auditable manifests.
    document_rows = []
    report_rows = []
    circular_rows = []
    for endpoint, destination in (
        ("approvedAnnouncement", document_rows),
        ("getFinancialAnnouncement", report_rows),
        ("circularAnnouncement", circular_rows),
    ):
        payload = market.get(endpoint, {})
        for row in (find_list(payload) or []):
            if not isinstance(row, dict):
                continue
            record = {**row, "source_endpoint": endpoint, "retrieved_at_utc": utc_now()}
            path_value = row.get("path")
            if path_value:
                record["document_url"] = urljoin(CDN, str(path_value).lstrip("/"))
            destination.append(record)

    write_csv(root / "corporate_announcements.csv", document_rows)
    write_csv(root / "financial_reports_manifest.csv", report_rows)
    write_csv(root / "cse_circulars_manifest.csv", circular_rows)

    # Optional archival download. Preserve original report metadata and hash each file.
    if args.download_pdfs:
        for record in report_rows + circular_rows:
            url = record.get("document_url")
            if not url:
                continue
            parsed = urlparse(str(url))
            if parsed.netloc not in ("cdn.cse.lk", "www.cse.lk"):
                logging.warning("Refusing unexpected document host: %s", url)
                continue
            filename = safe_name(Path(parsed.path).name or "document.pdf")
            doc_id = safe_name(str(record.get("id") or record.get("announcementId") or filename))
            folder = "financial_reports" if record in report_rows else "circulars"
            target = root / "documents" / folder / f"{doc_id}_{filename}"
            if target.exists():
                continue
            try:
                sha256, byte_count = client.download(str(url), target)
                record["local_file"] = str(target)
                record["sha256"] = sha256
                record["bytes"] = byte_count
            except Exception as exc:
                record["download_error"] = str(exc)
                logging.warning("PDF download failed %s: %s", url, exc)
        write_csv(root / "financial_reports_manifest.csv", report_rows)
        write_csv(root / "cse_circulars_manifest.csv", circular_rows)

    run_meta.update({
        "finished_at_utc": utc_now(),
        "security_records": len(securities),
        "active_equity_symbols_considered": len(active),
        "company_details_requested": bool(args.company_details),
        "financial_report_records": len(report_rows),
        "corporate_announcement_records": len(document_rows),
        "circular_records": len(circular_rows),
        "pdf_downloads_enabled": bool(args.download_pdfs),
        "note": "CSE feeds are public API snapshots. Company summary fields are current values, not historical point-in-time fundamentals.",
    })
    write_json(root / "collection_manifest.json", run_meta)
    logging.info("CSE collection complete: %s", root.resolve())
    logging.info("Manifest: %s", root / "collection_manifest.json")


if __name__ == "__main__":
    main()
