# Stockpulse – CSE Enrichment Dataset Pack

## Purpose
A separate data package for the Stockpulse CSE recommendation project. It preserves the 36 original CSE source files extracted from the supplied `Dataset.zip` and adds a collector for public CSE website feeds plus normalized schema templates.

## Contents
- `01_existing_cse_sources/`: original CSE market-data workbooks and daily-price ZIP archives copied byte-for-byte from the supplied archive. These are source inputs, not newly downloaded data.
- `02_collect_cse_live.py`: Python collector for CSE public website API feeds and linked CSE-hosted documents.
- `03_schemas/`: normalized schemas for financial facts, ownership/governance, corporate events, 2026 market data and valuation snapshots.
- `source_manifest.json`: source filenames, sizes and SHA-256 hashes.

## Critical collection status
The supplied source archive contains historical CSE data through 2025. It does NOT contain company-level five-year financial statement facts for all listed companies, and it does not contain 2026 daily market data.

I attempted to reach `www.cse.lk` from the execution environment, but DNS resolution failed. Therefore, no live CSE records or report PDFs are represented as successfully downloaded in this pack. The live collector is provided for execution from the user's machine/Antigravity where CSE connectivity is available. Do not mistake the API documentation or schemas for collected observations.

## Install and run
Python 3.10+:
```bash
pip install requests
```
From the extracted pack directory:
```bash
python 02_collect_cse_live.py --output ./cse_live
```
For current company snapshots (one request per listed ordinary equity symbol):
```bash
python 02_collect_cse_live.py --output ./cse_live --company-details --pause 1.0
```
To also download CSE-hosted report and disclosure documents:
```bash
python 02_collect_cse_live.py --output ./cse_live --company-details --download-documents --pause 1.0
```
Start without `--download-documents`, inspect `collection_manifest.json` and raw JSON responses, then run document download only after checking endpoint payloads and permitted use. Endpoints are undocumented/reverse-engineered, may change, and may return only the current feed window. The collector preserves raw responses and records endpoint errors.

## Financial statement extraction
CSE's financial-announcement feed is a document index, not a normalized financial database. Annual reports and interim statements must be downloaded, parsed and validated. Populate `03_schemas/financial_facts.csv` only after extracting the actual facts from report PDFs. Store both fiscal period and publication/availability date. Historical model features must be joined using publication/availability date, never fiscal period alone.

Recommended facts:
- Income statement: revenue, gross profit, operating profit, finance cost, tax, net profit, EPS.
- Balance sheet: cash, current assets/liabilities, total assets, total liabilities, equity, borrowings.
- Cash flow: operating cash flow, capital expenditure, investing/financing cash flow.
- Derived: ROE, ROA, ROIC, margins, net debt, interest coverage, free cash flow, FCF conversion.
- Governance/ownership: directors, management changes, major shareholders, public holding, foreign ownership and governance disclosures.

## Data engineering and quality rules
1. Never overwrite the original CSE source files.
2. Keep raw API JSON and original PDF files, plus retrieval timestamps and SHA-256 hashes.
3. Distinguish `period_end` from `publication_datetime` and `available_from_date`.
4. Track restatements and document versions.
5. Do not infer zero from missing values.
6. Preserve delisted securities and historical security identifiers; do not build a survivorship-biased universe.
7. Apply split/dividend adjustments consistently; distinguish price return from total shareholder return.
8. Verify units, currency, consolidated vs company-only statements, and fiscal-year lengths.
9. Check CSE terms, copyright, and permitted academic/research use. Do not bypass access controls or rate limits.

## Source notes
- CSE Data Library describes historical whole-market and listed-company data including dividends, scrip dividends, bonus issues, rights issues, splits, listing/delisting, indices, total-return indices, market ratios and market statistics: https://cdn.cse.lk/docs/DL_Form_E_20180827.pdf
- CSE website: https://www.cse.lk/
- The public website API endpoint descriptions used by the collector are based on unofficial reverse-engineered references, not a stable official API contract:
  https://github.com/adhishagc/cse-api
  https://gh0sth4cker.github.io/Colombo-Stock-Exchange-CSE-API-Documentation/
