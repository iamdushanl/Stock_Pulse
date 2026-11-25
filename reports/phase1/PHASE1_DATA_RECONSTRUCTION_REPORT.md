# Stockpulse Phase 1A – Full CSE Historical Data Reconstruction Report

> **Generated:** 2026-11-25 15:33:26  
> **Workspace:** `c:\Users\HP\Downloads\Stockpulse_CSE_Enrichment_Pack`  
> **Status:** COMPLETE — zero original source files modified

---

## Executive Summary

| Question | Verified Result |
|---|---|
| Actual historical coverage | **1990-01-12 to 2025-12-31** |
| Unique CompanyCodes | **400** (277 active/listed, 123 delisted) |
| Total raw records loaded (pre-dedup) | **1,420,392** |
| Total validated records (deduplicated) | **1,414,977** |
| 2023 data in raw sources | **YES** — **62,501** records across **276** companies and **242** trading dates |
| 2021–2022 coverage | **YES** — 2021: **63,171** rows (280 cos); 2022: **61,093** rows (283 cos) |
| Open price start year | **2017** (2017–2025 have real CSE `OPEN PRICE (Rs.)`; 1991–2016 have no Open column, zero fabricated rows) |
| Corporate actions loaded | **10,055** records across 7 action tables (`01`–`06`) |
| `HIGH < LOW` remaining violations | **0** (all 1,042 earlier violations traced to unquoted comma shifts in `2023 Data.csv` and deterministically repaired) |
| Phase 1B readiness | **READY** (with split adjustment & GICS sector enrichment queued for Phase 1B) |

---

## 1. Historical Coverage by Year

| Year | Trading Days | Companies | Records | Open Coverage | Zero Close | Odd-Lot `<100` Outside H/L | Vol `>=100` Outside H/L | `High < Low` |
|---|---|---|---|---|---|---|---|---|
| 1990 | 11 | 13 | 13 | 0.0% | 0 | 0 | 0 | 0 |
| 1991 | 120 | 163 | 4,246 | 0.0% | 67 | 0 | 0 | 0 |
| 1992 | 242 | 175 | 12,582 | 0.0% | 522 | 0 | 0 | 0 |
| 1993 | 241 | 186 | 18,945 | 0.0% | 601 | 0 | 0 | 0 |
| 1994 | 234 | 202 | 23,508 | 0.0% | 699 | 0 | 0 | 0 |
| 1995 | 240 | 215 | 19,558 | 0.0% | 945 | 0 | 0 | 0 |
| 1996 | 242 | 213 | 16,447 | 0.0% | 836 | 0 | 0 | 0 |
| 1997 | 241 | 222 | 21,820 | 0.0% | 304 | 0 | 0 | 0 |
| 1998 | 240 | 221 | 21,508 | 0.0% | 0 | 0 | 0 | 0 |
| 1999 | 241 | 221 | 21,354 | 0.0% | 0 | 0 | 0 | 0 |
| 2000 | 241 | 220 | 19,254 | 0.0% | 0 | 0 | 0 | 0 |
| 2001 | 240 | 218 | 18,579 | 0.0% | 0 | 0 | 0 | 0 |
| 2002 | 238 | 226 | 26,218 | 0.0% | 0 | 0 | 0 | 0 |
| 2003 | 240 | 234 | 29,295 | 0.0% | 0 | 0 | 0 | 0 |
| 2004 | 239 | 239 | 32,751 | 0.0% | 0 | 0 | 0 | 0 |
| 2005 | 238 | 238 | 38,443 | 0.0% | 0 | 0 | 0 | 0 |
| 2006 | 241 | 227 | 37,800 | 0.0% | 0 | 0 | 4 | 0 |
| 2007 | 241 | 229 | 37,181 | 0.0% | 0 | 0 | 0 | 0 |
| 2008 | 238 | 228 | 39,007 | 0.0% | 0 | 0 | 0 | 0 |
| 2009 | 240 | 228 | 40,653 | 0.0% | 0 | 0 | 0 | 0 |
| 2010 | 238 | 238 | 48,529 | 0.0% | 0 | 0 | 0 | 0 |
| 2011 | 239 | 262 | 57,597 | 0.0% | 0 | 0 | 0 | 0 |
| 2012 | 242 | 281 | 61,220 | 0.0% | 0 | 6 | 1 | 0 |
| 2013 | 242 | 279 | 61,084 | 0.0% | 0 | 0 | 0 | 0 |
| 2014 | 241 | 285 | 62,710 | 0.0% | 0 | 0 | 0 | 0 |
| 2015 | 239 | 279 | 58,144 | 0.0% | 0 | 3,383 | 1 | 0 |
| 2016 | 240 | 280 | 54,849 | 0.0% | 0 | 5,688 | 0 | 0 |
| 2017 | 241 | 281 | 56,252 | 100.0% | 0 | 5,769 | 5 | 0 |
| 2018 | 240 | 281 | 53,921 | 100.0% | 0 | 6,263 | 13 | 0 |
| 2019 | 241 | 277 | 52,946 | 100.0% | 0 | 6,800 | 14 | 0 |
| 2020 | 209 | 267 | 50,925 | 100.0% | 0 | 4,110 | 3 | 0 |
| 2021 | 240 | 280 | 63,171 | 100.0% | 0 | 3,892 | 2 | 0 |
| 2022 | 231 | 283 | 61,093 | 100.0% | 0 | 4,285 | 1 | 0 |
| 2023 | 242 | 276 | 62,501 | 100.0% | 0 | 5,059 | 2 | 0 |
| 2024 | 240 | 275 | 63,332 | 100.0% | 0 | 4,627 | 4 | 0 |
| 2025 | 238 | 276 | 67,541 | 100.0% | 0 | 2,524 | 10 | 0 |

---

## 2. Unique Securities & Security Master

- **Total unique `CompanyCode`s reconstructed:** **400**
- **Listed / Active (2024–2025):** **277**
- **Historical / Delisted (`DelistingDateOfficial` or last traded pre-2024):** **123** (100% preserved — 123/123 present in validated prices)
- **Official Company Names & ISINs enriched:** **350** companies matched from `16 List of Quoted Securities`, `38Foreign Holding`, and `SP20`; **346** matched with ISIN codes.
- **Note on Sector/Industry:** `39GICS-Daily.xlsx`, `22Sector Market Capitalisation.xls`, and `23Sector Trading Statistics` contain sector-index aggregates rather than a company-level ticker->GICS lookup; `Sector` is left honestly as `"Unknown"` for Phase 1A rather than extrapolated from live snapshots.

---

## 3. Root Causes Resolved

1. **Missing 2023 & Sparse 2021–2022 in Phase 3 Feature Matrices:**
   - **2023 (`62,501` rows)** exists in `40Daily Shares Price List -2021-2025/2023 Data.csv` and **2021–2022 (`63,171` / `61,093` rows)** exist in 4 quarterly tabs each.
   - Phase 3 missed them for two reasons: `phase3_feature_engineering/build_multi_horizon_features.py` loaded only `SP20_CSE_Dataset_2001_2025_Adjusted.xlsx` (20 companies, 2023 empty), and `utils/data_loader.py` hardcoded non-existent `DEFAULT_BASE_PATH = r"c:\Users\HP\Documents\Stock_pulse\Dataset\2025 Q4"` plus assumed 12-column flat layout across 2001–2015 files.
2. **2017–2020 Open Prices Recovered:**
   - Raw inspection showed `2016 Data.xls` lacks an Open column (`flat_hlc`), whereas `2017`–`2020 Data.xls` all carry `OPEN PRICE (Rs.)` (98.1%–99.5% populated). Recovering those restores >211k genuine Open observations.
3. **`2023 Data.csv` Unquoted Comma Shifts (`CSV_THOUSANDS_COMMA_REPAIRED`):**
   - High-priced stocks (`>= 1,000` LKR: `CINS`, `CTEA`, `NEH`, `SLND`, `SOY`, `BREW`, `HARI`, `NEST`, `PHAR`, `CTC`, `GOOD`, `SWAD`, `SHAL`, `INDO`) had thousands-separated numbers written unquoted in `2023 Data.csv` (`1,100.00` -> `1.00, 100.00`), shifting columns right and causing all 1,042 earlier `HIGH < LOW` errors.
   - All shifted rows were deterministically reconstructed via adjacent-pair parsing validated against `VWAP = Turnover / ShareVolume` and `[Low, High]` bounds, and logged in `data_quality_flags.parquet` (`auto_corrected=True`).
4. **SQL\*Plus Footer Junk (`1995`–`1997`) & 2001 Security ID Normalization:**
   - Filtered SQL\*Plus `"19701 rows selected."` footers that produced bogus `"ROWS"` records at `1970-01-01`.
   - Parsed `2001 data.xls` 10-char `Security ID` (`ABAN-N-000`, `HNB-D-0014`) into `(CompanyCode, MainType, SubType)` and excluded duplicate right-hand scratch columns (Cols 10–12).
   - Extracted `MainType`, `SubType`, and `ShortName` across all `2002–2015` grouped blocks so non-ordinary share classes (`X`, `R`, `W`, `B`, `D`, `P`, `U`) are preserved during deduplication.

---

## 4. Corporate Action & Anomaly Validation

- **JKH (Sept–Nov 2024):** Confirmed **1-for-10 share split** effective `2024-11-05` in `05Sub Division (Share Splits).xls` (`212.50` -> `21.20` LKR). Raw unadjusted prices preserved cleanly.
- **COMB (Aug 2024):** Confirmed multiple share classes trading simultaneously (`N`, `X`, `R` SubTypes `0000`/`0001`) during COMB's rights issue — legitimate multi-class rows, not duplicates.
- **CSE 100-Share Odd-Lot Rule (`52,406` rows):** Over 98% of `Close > High` / `Close < Low` rows from 2015 onward occur on `ShareVolume < 100`, reflecting CSE microstructure where odd lots print High/Low/Open while Official Close stays at the prior board-lot close. Only **60** rows deviate with `ShareVolume >= 100`.

---

## 5. Open Price Availability Schedule

| Era | Open Available | Source Format & Notes |
|---|---|---|
| 1990–2000 | No | `close_only` (Date, Code, Close only) |
| 2001 | No | `flat_cloprc` (Close + ShareVolume only) |
| 2002–2015 | No | `grouped_hlc` (High, Low, Close, Trades, Shares, Turnover) |
| 2016 | No | `flat_hlc` (Flat tabular, no Open column in source XLS) |
| 2017–2025 | Yes | `flat_ohlcv` (Full OHLCV; 2017–2020 ~98–99.5% Open, 2021–2025 ~100%) |

---

## 6. Output Deliverables (`Dataset/processed/phase1/`)

| File | Rows | Description |
|---|---|---|
| `cse_daily_prices_raw.parquet` | 1,420,392 | Pre-deduplication parsed historical prices |
| `cse_daily_prices_validated.parquet` | 1,414,977 | Deduplicated, class-preserved, quality-flagged universe |
| `cse_daily_prices_close_only.parquet` | 197,873 | 1990–2002 `close_only` + `flat_cloprc` subset |
| `cse_security_master.parquet` | 400 | Enriched universe (`CompanyCode`, names, ISIN, status, dates) |
| `cse_corporate_actions.parquet` | 10,055 | Combined `01`–`06` events (Dividends, Scrips, Bonus, Rights, Splits, Listings, Delistings) |
| `cse_market_calendar.parquet` | 8,240 | CSE trading dates (`1991–2025`) |
| `data_quality_flags.parquet` | 9,297 | Full anomaly & auto-repair audit log |
| `annual_company_coverage.csv` | 36 | Annual date/company/record coverage summary |
| `annual_data_quality_summary.csv` | 36 | Annual quality metrics & OHLC breakdown |
| `phase1_manifest.json` | 1 | Reproducible build & quality manifest |

---

## 7. Phase 1B Readiness

**STATUS: READY FOR PHASE 1B**

Next Phase 1B priorities:
1. Point `utils/data_loader.py` (or a new loader module) to `cse_daily_prices_validated.parquet` instead of the broken hardcoded path and 12-col assumption.
2. Apply verified split/rights/bonus adjustment factors from `cse_corporate_actions.parquet` (starting with JKH, LMF, AFS, SUN, WATA, HHL) to populate split-adjusted closes without overwriting raw prices.
3. Enrich company-level GICS/sector classifications cleanly using point-in-time historical metadata.

*Phase 1A – Audit-safe reconstruction complete. Original source files in `01_existing_cse_sources/` remain untouched.*
