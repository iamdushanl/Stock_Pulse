# CSE data enrichment pipeline

This pipeline collects a separate snapshot of public CSE information. It does not alter the original daily-price ZIP archives or consolidated price cache.

## Why a separate dataset?

Stockpulse needs more than daily OHLCV data to support its medium-term growth and long-term wealth-creation engines. The existing utils/data_loader.py already supports additional local CSE workbooks: dividends, splits, listing/de-listing history, market indices/statistics, GICS sector data, sector market capitalization and sector ratios. Export these historical sources first with scripts/build_historical_enrichment.py. The online collector adds a separate current CSE public-site snapshot. Together they create auditable inputs for:
- security and sector reference data;
- current company profile/summary snapshots;
- market/sector summaries and index feeds;
- CSE announcement, financial-report and circular manifests;
- optional archived PDF copies of reports/circulars.

## Important limitations

1. CSE public website endpoints are not a documented, stable public API. Endpoint names and response shapes can change. The collector stores raw JSON responses and records failures rather than silently treating missing responses as zero.
2. The current company profile/summary feed is a current snapshot, not a historical point-in-time fundamentals database. It must not be backfilled into historical training rows as if those values were known in the past.
3. The report manifest identifies documents; financial-statement facts still need extraction, normalization, period-end dates, publication dates, units, and restatement/version handling.
4. Announcements and reports must be timestamped by publication/available date. A document about a prior financial period is only usable by a model after it became public.
5. Check CSE terms, robots/access rules, copyright and permitted academic/research use before bulk collection or redistribution. Do not bypass authentication, access controls, rate limits, or anti-bot protections.
6. The first run is a schema-discovery run. Review the raw JSON and collection_manifest.json before relying on normalized output.

## Export existing historical CSE workbooks

The repository loader expects these auxiliary source files alongside the daily-price ZIPs:
- 07Market Indices - Daily.xls
- 11Market Statistics-Daily.xls
- 16 List of Quoted Securities And Issued Quantity.xls
- 05Sub Division (Share Splits).xls
- 06New Listings & De-listings.xls
- 39GICS-Daily.xlsx
- 22Sector Market Capitalisation.xls
- 01Dividends.xls
- 25Sector Ratios.xls

Run this from the repository root, replacing the source path with the actual local Dataset folder:

```bash
python scripts/build_historical_enrichment.py --source "/path/to/Dataset/2025 Q4"
```

This writes Parquet and CSV copies under Dataset/enrichment/cse_historical/ plus a manifest. Missing or malformed workbooks are recorded as errors.

## Collect current CSE website snapshots

## Setup

Python 3.10+:

```bash
pip install requests
```

## Run

From the repository root:

```bash
python scripts/collect_cse_data.py
python scripts/collect_cse_data.py --company-details
python scripts/collect_cse_data.py --company-details --download-pdfs
```

For a small schema test:

```bash
python scripts/collect_cse_data.py --company-details --limit-companies 5 --pause 1.0
```

Default output:

```
Dataset/enrichment/cse_public/
├── raw/                         # Original API JSON responses
├── documents/                   # Optional downloaded source PDFs
├── securities_master.csv
├── sectors_master.csv
├── company_current_snapshot.csv
├── corporate_announcements.csv
├── financial_reports_manifest.csv
├── cse_circulars_manifest.csv
├── collection_errors.csv
└── collection_manifest.json
```

All outputs are ignored by Git because the repository excludes Dataset/ and data files.

## Recommended next pipeline stage

After a successful collection, build a separate report parser and normalized financial fact table with at least:
- company_code
- fiscal_period_start, fiscal_period_end
- published_at / available_at
- statement_type (income, balance sheet, cash flow)
- metric, value, currency, unit
- document_url, document_sha256, page_or_note
- is_restated, version, extraction_method, validation_status

Do not merge financial facts into daily prices until publication-date-aware as-of joins and quality checks are implemented.