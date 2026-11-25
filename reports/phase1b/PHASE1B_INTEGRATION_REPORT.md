# Stockpulse Phase 1B – Full CSE Data Integration and Corporate Action Engine Report

**Status:** COMPLETE (`25/25` automated verification checks passed; raw sources and Phase 1A datasets unmodified)  
**Universe:** Full Colombo Stock Exchange (CSE) (`1990-01-12` to `2025-12-31`)  
**Scope:** Data Loader Integration, Security & Issuer Identity Model, Traceable Corporate Action Engine, Historical Sector Audit, and Output Generation  
**Pipeline Entrypoint:** [run_phase1b.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase1b_integration/run_phase1b.py) (`29.91s` full execution)

---

## 1. Data Loader & Downstream Utility Integration

The legacy `utils/` pipeline (previously pointing to broken local directories or SP20-only files) was upgraded cleanly across six modules without altering historical data values:

- **[data_loader.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/utils/data_loader.py):**
  - Replaced hardcoded paths with repository-relative defaults (`DEFAULT_RAW_SOURCE_PATH = 01_existing_cse_sources`, `DEFAULT_PHASE1_PATH = Dataset/processed/phase1`, `DEFAULT_PHASE1B_PATH = Dataset/processed/phase1b`).
  - Added full-CSE Parquet loaders preserving exact schema dtypes (`datetime64`, `float64`, nullable `Int64`, categorical/string identifiers):
    - `load_cse_daily_prices(companies=None, start_date=None, end_date=None, security_types=None, security_ids=None, path=None, columns=None, incremental_by_year=False, chunk_size=None)`
    - `load_cse_security_master(path=None, active_status=None, companies=None)`
    - `load_cse_corporate_actions(path=None, action_category=None, companies=None, start_date=None, end_date=None)`
    - `load_cse_market_calendar(path=None, start_date=None, end_date=None)`
  - Updated `load_all_daily_prices()` to load the full Phase 1A CSE dataset (`1,414,977` rows across `400` `CompanyCode`s and `719` `SecurityID`s).
  - **Removed silent SP20 fallback:** Invalid paths explicitly raise `FileNotFoundError`. The legacy SP20 dataset is isolated in `load_sp20_daily_prices()` exclusively for explicit comparison benchmarks.
- **[data_cleaning.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/utils/data_cleaning.py):** Keyed deduplication, calendar alignment, and return computation on `SecurityID` (never collapsing share classes); added an `IsSplitAdjusted` guard + per-event dedup key in `apply_split_adjustments()` that raises `ValueError` if called on already split-adjusted data.
- **[features.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/utils/features.py), [targets.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/utils/targets.py), [recommender.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/utils/recommender.py), and [__init__.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/utils/__init__.py):** Updated grouping keys to preserve `SecurityID`, `MainType`, and `SubType` when present and exported all five new loader APIs.

---

## 2. Security & Issuer Identity Model

Implemented in [identity_model.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase1b_integration/identity_model.py) and materialized in `Dataset/processed/phase1b/cse_security_master_enriched.parquet` (`719` unique `SecurityID` records):

| Metric | Count | Details |
| :--- | ---: | :--- |
| Unique `SecurityID`s | **719** | All share classes preserved distinctly |
| Unique `CompanyCode`s | **400** | Matches Phase 1A master `1:1` |
| Unique Issuers (`IssuerID`) Established | **350** | Backed by ISIN (`657` securities) or official company name (`6`) |
| `VERIFIED_ISIN` | **657** | Issuer identity anchored via CSE quoted securities ISIN |
| `VERIFIED_OFFICIAL_NAME` | **6** | Verified via historical CSE quoted list name match |
| `MANUAL_REVIEW_MERGER_NAME_OVERLAP` | **6** | Distinct ISIN issuers sharing a post-merger official name (`BFN`/`ORIN` `ORIENT FINANCE PLC`; `CLC`/`LOFC` `LOLC FINANCE PLC`) |
| `MANUAL_REVIEW_UNVERIFIED_ISSUER` | **50** | Historical/delisted codes without official name/ISIN metadata (`IssuerID=None`) |
| Total `ManualReviewRequired = True` | **56** | Explicitly flagged instead of heuristically merged |

---

## 3. Share Classes Preserved

Every security class (`MainType`) is retained as an independent tradable `SecurityID` (`<CompanyCode>-<MainType>-<SubType>`):

| `MainType` | Description | Security Count | Example Preserved IDs |
| :---: | :--- | ---: | :--- |
| `N` | Ordinary Voting Shares | **394** | `JKH-N-0000`, `COMB-N-0000` |
| `R` | Rights Entitlements | **176** | `COMB-R-0000`, `JKH-R-0000` |
| `D` | Debentures / Debt | **58** | Corporate debt issues |
| `B` | Closed-End / Unit / Bond Instruments | **35** | Fixed income / trust issues |
| `X` | Non-Voting Ordinary Shares | **25** | `COMB-X-0000`, `CIC-X-0000` |
| `W` | Share Warrants | **23** | `JKH-W-0022`, `JKH-W-0023` |
| `P` | Preference Shares | **4** | Preference share issues |
| `U` | Units | **4** | Listed unit securities |

Multi-class corporate actions (e.g., `CIC` `(N & X)` splits and `01Dividends.xls` class-specific rows) resolve to their exact target `SecurityID` without cross-class leakage.

---

## 4. Historical Listing & Delisting Preservation

- **Listed Company Codes:** `277` (`HistoricalStatus = "Listed"`)
- **Delisted / Historical Company Codes:** `123` (`HistoricalStatus = "Delisted"`)
- **Survivorship Bias Elimination:** All `123` delisted company codes retain their complete historical price observations in `cse_daily_prices_integrated.parquet`, `cse_daily_prices_split_adjusted.parquet`, and `cse_total_return_history.parquet`, paired with verified `DelistingDate` and `FirstPriceDate`/`LastPriceDate` bounds.

---

## 5. Corporate Action Engine Verification Results

Implemented in [corporate_actions_engine.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase1b_integration/corporate_actions_engine.py) reading all five raw CSE Excel workbooks (`01Dividends.xls` through `05Sub Division (Share Splits).xls`):

- **Total Parsed Records (`cse_corporate_actions_validated.parquet`):** `8,754`
- **Validated Adjustment Events Applied (`corporate_action_adjustment_factors.parquet`):** `4,443`
- **Validated Non-Dilutive Underwater Rights (`PriceFactor = 1.0`):** `49`
- **Cross-Sheet Boundary Duplicates Neutralized (`DUPLICATE_BOUNDARY_EVENT_SKIPPED`):** `27` (preventing double adjustment where December events repeat across adjacent annual tabs)
- **Factor Health:** `0` NaN or `<= 0` `PriceFactor` or `DividendFactor` entries; `0` `inf` prices across `1,414,977` output rows.

### Breakdown by Event Type & Validation Status

| Event Type | Validated (`PriceFactor`/`DivFactor`) | Underwater (`1.0`) | Boundary Dup Skipped | `%` of Par Unresolved | Missing ExDate | Unresolved Security | Invalid / Ambiguous Ratio / Other | Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CashDividend` | **3,887** | 0 | 16 | 2,787 | 0 | 7 | 270 *(188 missing amt, 81 no pre-ex close, 1 div>=price)* | **6,967** |
| `RightsIssue` | **210** *(TERP)* | 49 | 2 | 0 | 135 | 338 | 135 *(98 invalid ratio, 22 no sub price, 12 no pre-ex close, 3 dir)* | **869** |
| `BonusIssue` | **43** | 0 | 1 | 0 | 180 | 224 | 137 *(invalid ratio)* | **585** |
| `ScripDividend` | **165** | 0 | 7 | 0 | 21 | 1 | 0 | **194** |
| `StockSplit` | **138** | 0 | 1 | 0 | 0 | 0 | 0 | **139** |
| **Total** | **4,443** | **49** | **27** | **2,787** | **336** | **570** | **542** | **8,754** |

---

## 6. Target Corporate Actions & Empirical Price Continuity Check

All target stock splits and representative corporate actions were verified directly against `cse_daily_prices_integrated.parquet` (raw) and `cse_daily_prices_split_adjusted.parquet` (adjusted):

| `SecurityID` | Event Type | `ExDate` | Ratio (`New:Old`) | `PriceFactor` | Pre-Ex Raw Close | Pre-Ex Split-Adj Close | Post-Ex Raw/Adj Close | Status |
| :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: | :--- |
| `JKH-N-0000` | StockSplit | `2024-11-05` | `10:1` | `0.10000` | 212.50 (`2024-10-28`) | **21.25** | **21.20** (`2024-11-06`) | Verified |
| `JKH-N-0000` | StockSplit | `2016-06-30` | `8:7` | `0.87500` | 155.50 (`2016-06-24`) | **13.61** | **13.50** (`2016-06-30`) | Verified |
| `JKH-N-0000` | StockSplit | `2015-07-03` | `8:7` | `0.87500` | 203.90 (`2015-06-26`) | **15.61** | **15.95** (`2015-07-03`) | Verified |
| `JKH-N-0000` | StockSplit | `2011-06-30` | `4:3` | `0.75000` | 282.30 (`2011-06-24`) | **16.21** | **15.39** (`2011-06-30`) | Verified |
| `LMF-N-0000` | StockSplit | `2024-01-16` | `10:1` | `0.10000` | 223.75 (`2024-01-05`) | **22.38** | **21.00** (`2024-01-17`) | Verified |
| `AFS-N-0000` | StockSplit | `2024-01-24` | `5:1` | `0.20000` | 36.00 (`2024-01-16`) | **7.20** | **6.00** (`2024-01-26`) | Verified |
| `SUN-N-0000` | StockSplit | `2025-02-18` | `4:1` | `0.25000` | 89.80 (`2025-02-10`) | **22.45** | **22.50** (`2025-02-19`) | Verified |
| `SUN-N-0000` | StockSplit | `2021-03-31` | `3:1` | `0.33333` | 80.60 (`2021-03-25`) | **6.72** | **6.62** (`2021-03-31`) | Verified |
| `WATA-N-0000` | StockSplit | `2025-02-28` | `5:1` | `0.20000` | 139.00 (`2025-02-20`) | **27.80** | **26.90** (`2025-03-03`) | Verified |
| `HHL-N-0000` | StockSplit | `2025-05-06` | `5:1` | `0.20000` | 121.75 (`2025-04-28`) | **24.35** | **24.50** (`2025-05-07`) | Verified |
| `CIC-N-0000` | StockSplit | `2025-10-17` | `5:1` | `0.20000` | 170.00 (`2025-10-13`) | **34.00** | **34.20** (`2025-10-22`) | Verified |
| `CIC-X-0000` | StockSplit | `2025-10-17` | `5:1` | `0.20000` | 128.25 (`2025-10-13`) | **25.65** | **26.00** (`2025-10-22`) | Verified |

- **Bonus Example (`VLL-N-0000`, `2015-09-23`):** `1 new : 2 old` (`Ratio = 0.5`), `PriceFactor = 2 / 3 = 0.666667`, `DividendFactor = 1.0`.
- **Scrip Dividend Example (`WAPO-N-0000`, `2024-09-20`):** `Ratio = 0.072187`, `PriceFactor = 0.932673`, `DividendFactor = 1.0`.
- **Rights TERP Example (`VLL-N-0000`, `2010-10-28` & `WAPO-N-0000`, `2011-02-18`):**
  - `VLL-N-0000` (`P_cum = 7.60`, `S = 4.00`, `r = 0.142857`): `TERP = 7.15`, `PriceFactor = 0.940789`.
  - `WAPO-N-0000` (`P_cum = 8950.00`, `S = 20.00`, `r = 37.5`): `TERP = 251.95`, `PriceFactor = 0.028151`.
- **Cash Dividend Example (`YORK-N-0000`, `2013-07-29`):** `DividendPerShare = Rs. 0.40`, `PreExClose = 14.50`, `PriceFactor = 1.0`, `DividendFactor = (14.50 - 0.40) / 14.50 = 0.972414`.

---

## 7. Separation of Raw, Split-Adjusted, Capital-Adjusted, and Total Return Series

Adjustment layers are strictly isolated across separate datasets and columns (`1,414,977` rows each):

1. **`cse_daily_prices_integrated.parquet` (Raw Integrated):** Exact `1:1` preservation of Phase 1A validated prices (`Raw Close` exact match across all `1,414,977` rows, including the single source-missing NaN `Close`), enriched with `SecurityID`, `IssuerID`, `ISIN`, `HistoricalStatus`, and `CompanyName`.
2. **`cse_daily_prices_split_adjusted.parquet` (Capital Structure Series):**
   - `RawOpen`, `RawHigh`, `RawLow`, `RawClose`, `RawShareVolume` preserved untouched.
   - `Open`, `High`, `Low`, `Close`, `ShareVolume` adjusted strictly backwards (`Date < ExDate`) by `CumSplitBonusFactor = CumSplitFactor * CumBonusFactor`.
   - `CapitalAdjClose` adjusts additionally by `CumScripFactor * CumRightsTERPFactor`.
   - Cash dividends are **excluded** from this file; `IsSplitAdjusted = True` blocks accidental re-adjustment.
3. **`cse_total_return_history.parquet` (Total Return Series):**
   - Exposes `RawClose`, `SplitBonusAdjClose`, `CapitalAdjClose`, `CumCapitalFactor`, `CumCashDividendFactor`, `CumTotalReturnFactor`, `TotalReturnPrice = Close * CumCapitalFactor * CumCashDividendFactor`, and `TotalReturnIndex` (normalized to `100.0` at the first positive observation per `SecurityID`).
   - Example (`JKH-N-0000`): Starts `1991-07-31` at `RawClose = 315.25`, `SplitBonusAdjClose = 18.10`, `CapitalAdjClose = 14.40`, `TotalReturnPrice = 9.60` (`TotalReturnIndex = 100.00`), reaching `2025-12-31` at `Close = 21.70` (`TotalReturnIndex = 226.00`).

---

## 8. Unresolved Cases & Refusal to Fabricate

Consistent with Phase 1B strict audit rules, incomplete historical records were explicitly flagged (`PriceFactor = 1.0`, `DividendFactor = 1.0`) rather than guessed:

- **`UNRESOLVED_PERCENT_OF_PAR` (`2,787` Cash Dividends):** Pre-2008 `01Dividends.xls` entries quote percentages of par (without verified Rs. par values per security-date); we refused to assume `Rs. 10` par.
- **`UNRESOLVED_SECURITY_MATCH` (`570` events):** Early short/ambiguous company names or unmapped historical entities (`338` Rights, `224` Bonus, `7` Dividends, `1` Scrip).
- **`UNRESOLVED_MISSING_EX_DATE` (`336` events):** Pre-2000 Bonus (`180`), Rights (`135`), and Scrip (`21`) records lacking Allotment/Ex-dates.
- **`UNRESOLVED_INVALID_RATIO` (`235` events) & `UNRESOLVED_MISSING_SUBSCRIPTION_PRICE` (`22`):** Unparseable historical proportions or omitted rights issue considerations.
- **`MANUAL_REVIEW_MERGER_NAME_OVERLAP` / `UNVERIFIED_ISSUER` (`56` securities):** Kept distinct by `CompanyCode` and `SecurityID`.

---

## 9. Sector Mapping Audit & Regime Findings

Audited in [sector_mapping.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase1b_integration/sector_mapping.py) across `39GICS-Daily.xlsx`, `22Sector Market Capitalisation.xls`, and `25Sector Ratios.xls`:

- **Verified Finding:** All three CSE sector workbooks contain **aggregate sector-level indices, market capitalization, PER, PBV, and DY series only** — none include a `CompanyCode`, `SecurityID`, or ticker-to-sector mapping table.
- **Regime Transition Documented:** CSE transitioned from `20` legacy sector groups (`2005–2019` in `22Sector Market Capitalisation.xls`) to `20–21` GICS industry groups (`Jan 2020`, with `39GICS-Daily.xlsx` backfilled from `2016-04-05`).
- **Leakage Prevention Outcome:** Rather than backfilling current Nov 2026 snapshots into `1990–2025` history, all `719` rows in `cse_sector_mapping.parquet` are set to `Sector = "Unknown"`, `Industry = "Unknown"`, `Confidence = 0.0`, `ValidationStatus = "MANUAL_RESEARCH_REQUIRED"`, and `ManualResearchQueue = True`.

---

## 10. Automated Verification Summary

Executed `pytest tests/test_phase1b.py -v` after the final code edit ([test_phase1b.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/tests/test_phase1b.py)):

- **Result:** **25 / 25 PASSED (`100%`)** in `96.73s`
- Verified coverage: full-CSE loader without SP20 fallback (`FileNotFoundError` on invalid paths), date/company/type/incremental filters, all 8 share classes (`N/X/R/W/P/D/B/U`), `350` issuers & `56` manual review flags, all target split continuity checks (`JKH`, `LMF`, `AFS`, `SUN`, `WATA`, `HHL`, `CIC N & X`), Bonus/Scrip/Rights TERP & Total Return separation math, strict `Date < ExDate` application, double-adjustment guard, CSE day-first date parsing, `123` delisted companies retained, sector no-leakage audit, and `1:1` Phase 1A `Close` immutability across `1,414,977` rows.

---

## 11. Deliverables Inventory & Next-Phase Readiness

All eight Phase 1B artifacts were generated in `Dataset/processed/phase1b/`:
- `cse_daily_prices_integrated.parquet` (`1,414,977` rows)
- `cse_daily_prices_split_adjusted.parquet` (`1,414,977` rows)
- `cse_total_return_history.parquet` (`1,414,977` rows)
- `cse_security_master_enriched.parquet` (`719` rows)
- `cse_corporate_actions_validated.parquet` (`8,754` rows)
- `corporate_action_adjustment_factors.parquet` (`4,443` active events)
- `cse_sector_mapping.parquet` (`719` rows)
- [phase1b_manifest.json](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase1b/phase1b_manifest.json)

### Readiness Recommendations Before Phase 2+ Feature Engineering / Modeling
1. **Resolve Historical Sector Crosswalk:** Populate historical pre-2020 CSE sector mappings and post-2020 GICS mappings with explicit effective dates for the `719` queued securities before computing sector-relative features.
2. **Par Value Lookup for Pre-2008 `%` Dividends:** If pre-2008 Total Return continuity is required, source verified historical par values for the `2,787` flagged `%`-of-par dividend events instead of assuming Rs. 10 par.
3. **Manual Review Queue (`56` securities):** Confirm historical corporate merger continuity links (`BFN`/`ORIN`, `CLC`/`LOFC`) and issuer links for the `50` delisted legacy codes.
4. *(No ML models were trained during Phase 1B; Phase 1A and raw source archives remain strictly untouched.)*
