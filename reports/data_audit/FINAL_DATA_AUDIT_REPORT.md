# Stockpulse Phase 0 – Final CSE Data Audit Report

> **Generated:** 2026-11-24  
> **Workspace:** `c:\Users\HP\Downloads\Stockpulse_CSE_Enrichment_Pack`  
> **Audit Status:** COMPLETE – READ-ONLY, no original data modified  
> **Script:** `reports/data_audit/run_phase0_audit.py`

---

## Executive Summary

| Question | Verified Answer |
|---|---|
| Total files in workspace | 664 |
| Total workspace size | 0.267 GB |
| Historical price ZIPs | 4 (1991–2000, 2001–2010, 2011–2020, 2021–2025) |
| Years of price history | ~34 years (1991–2025), **reliable OHLCV from 2011** |
| Unique companies in ML pipeline | **20 (SP20 only)** |
| Full CSE listed universe | **281+ listed companies** |
| CSE universe covered | ❌ **NO** – only SP20 basket |
| Financial PDFs collected | 90 |
| Companies with fundamentals | **78** (single-period snapshots per Phase 2D final) |
| Financial facts extracted | **2,453** (Phase 2D final, deduplicated) |
| Engine 01 Short-term | ⚠️ **READY WITH LIMITATIONS** |
| Engine 02 Medium-term | ❌ **NOT READY** |
| Engine 03 Long-term | ❌ **NOT READY** |

---

## 1. What Datasets We Actually Have

### 1.1 Historical Price Data

| Source | Period | Companies | Columns | Key Issues |
|---|---|---|---|---|
| `30Daily Shares Price List - 1991-2000.zip` | 1991–2000 | ~100–150 est. | 3 | Close only, pre-modern CSE era |
| `31Daily Shares Price List -2001-2010.zip` | 2001–2010 | ~200+ est. | 13 | **Open missing ~60% of rows** |
| `32Daily Shares Price List -2011-2020.zip` | 2011–2020 | ~250+ est. | 11 | OHLCV complete; adjust for splits |
| `33Daily Shares Price List -2021-2025.zip` | 2021–2025 | 281+ est. | 16 | Bad ticks: COMB Aug 2024, JKH Sept-Nov 2024 |
| `SP20_CSE_Dataset_2001_2025.xlsx` | 2001–2025 | 20 (SP20) | 64 | Pre-compiled, unadjusted |
| `SP20_CSE_Dataset_2001_2025_Adjusted.xlsx` | 2002–2025 | 20 (SP20) | 62 | Pre-compiled, split-adjusted |

> [!IMPORTANT]
> The Phase 3 feature matrices and all ML models are built on the **SP20 (20-company)** dataset only.  
> The 4 historical price ZIPs contain the **full CSE universe** but have NOT been loaded into the pipeline.

### 1.2 Feature Engineering Datasets (Parquet — Verified)

| File | Rows | Columns | Companies | Date Range | NaN % | Static Leakage Flags |
|---|---|---|---|---|---|---|
| `features_01_short_term_trading.parquet` | 86,950 | 65 | 20 | 2002–2025 | 1.3% | ✅ 4 flags present |
| `features_02_medium_term_growth.parquet` | 86,950 | 36 | 20 | 2002–2025 | **39.3%** | ✅ 2 flags present |
| `features_03_long_term_wealth.parquet` | 86,950 | 26 | 20 | 2002–2025 | **54.0%** | ✅ 1 flag present |
| `prepared_training_dataset.parquet` | 86,950 | 71 | 20 | 2002–2025 | 1.2% | ✅ Flags removed in Phase 4.1 |

**SP20 companies (20):** AEL, CFIN, COMB, DFCC, DIAL, HAYL, HHL, HNB, JKH, LFIN, LIOC, LOLC, MELS, NDB, NTB, PLC, SAMP, SEYB, SINS, SUN

> [!CAUTION]
> **Features 02 and 03 have 39% and 54% NaN rates** respectively, confirming that fundamental
> features cannot be computed for most historical rows — a direct result of only having
> single-period snapshot fundamentals, not a time-series.

### 1.3 Corporate Actions Data (Excel Workbooks — Verified)

| File | Rows (sampled) | Date Coverage | Status |
|---|---|---|---|
| `01Dividends.xls` | 645 | Historical | ✅ OK |
| `02Scrip Dividends.xls` | 80 | Historical | ✅ OK |
| `03Bonus Issues.xls` | 7 | Historical | ✅ OK |
| `04Rights Issues.xls` | 62 | Historical | ✅ OK |
| `05Sub Division (Share Splits).xls` | 15 | Historical | ✅ OK |
| `06New Listings & De-listings.xls` | 250 | Historical | ✅ OK |
| `07Market Indices - Daily.xls` | 9,780 | Daily | ✅ OK |
| `08Market Indices - Monthly.xls` | 408 | Monthly | ✅ OK |
| `09Total Returns Indices - Daily.xls` | 6,484 | Daily | ✅ OK |
| `10Market Ratios - Daily.xls` | 7,168 | Daily | ✅ OK |
| `29 Beta Value.xls` | 689 | Historical | ✅ OK |
| `39GICS-Daily.xlsx` | 6,800 | Daily | ✅ OK |

### 1.4 Fundamental Financial Data (Phase 2D Final — Verified)

- **90 annual report PDFs** in `cse_live/` (80 originally scoped, 10 added)
- **78 companies** with at least one financial period (Phase 2D final count)
- **2,453 financial facts** extracted and deduplicated (Phase 2D final)
- Coverage: **single period** (Q2 FY2027 interim / latest annual)
- **NO historical multi-year series** — critical gap for Engines 02 and 03

---

## 2. Years of Reliable Historical Data

| Metric | Value |
|---|---|
| Full OHLCV reliable range | **2011–2025 (15 years)** |
| Close-only usable | 2001–2025 (24 years) |
| Pre-2001 (1991–2000) | Close only, low quality, pre-modern CSE |
| 2021–2022 coverage | ⚠️ Sparse — only ~1,100–1,145 rows/year vs ~4,000–4,800 for other years |
| 2023 coverage | ❌ **ZERO rows** — completely absent from feature matrices |

> [!WARNING]
> **Pre-2011 data is missing Open prices in ~60% of rows.** Technical indicators
> requiring Open (ATR, NATR, Parabolic SAR, candlestick patterns) are unreliable before 2011.
>
> **2023 is completely absent from all feature matrices.** The backtest period
> claimed as "2023–2025" covers only 2024–2025 (2 years, not 3).

---

## 3. Company Coverage

| Universe | Count | Source | In ML Pipeline |
|---|---|---|---|
| SP20 (feature matrices, all ML models) | **20** | SP20 Excel + Parquet | ✅ Yes |
| Companies with fundamentals extracted | **78** | Phase 2D PDF extraction | Partially |
| Full CSE listed universe | **281+** | `06New Listings & De-listings.xls` | ❌ No |
| Companies in Phase 3 feature matrices | **20** | Parquet verified | ✅ Yes (SP20 only) |

> [!CAUTION]
> **261+ CSE-listed companies are absent from every ML feature matrix.**
> The pipeline covers only the SP20 index basket. This is the single largest gap
> preventing production-grade recommendations across the full CSE.

---

## 4. Full CSE Coverage

**STATUS: NO.** Current ML pipeline covers only 20 SP20 companies.

The raw ZIP archives contain full CSE universe data (2001–2025) but:
- They have **not** been loaded or processed through the cleaning pipeline
- Sector classification beyond SP20 is incomplete
- Corporate action adjustments have not been applied to the full universe

---

## 5. Which Investment Engine Is Ready

| Engine | Status | Key Blockers |
|---|---|---|
| 01 – Short-term Trading | ⚠️ **READY WITH LIMITATIONS** | SP20 only (20/281 companies); static leakage flags removed in prepared_training_dataset but remain in Phase 3 matrix; pre-2011 Open gaps; PSAR missing from v1 features; 2023 data absent |
| 02 – Medium-term Growth | ❌ **NOT READY** | 39.3% NaN rate in features; no historical fundamental series; SP20 only; point-in-time leakage in Phase 3 matrix |
| 03 – Long-term Wealth | ❌ **NOT READY** | 54.0% NaN rate in features; requires 5-year fundamental series; SP20 only; no macro data |

---

## 6. Additional Data That Must Be Collected

| Priority | Data Item | Blocking? | Engines |
|---|---|---|---|
| **P1** | Full CSE universe daily prices (281+ companies, 2001–2025) | ✅ YES | 1,2,3 |
| **P1** | Historical financial statements 5-year series (annual/semi-annual) | ✅ YES | 2,3 |
| **P2** | 2023 trading data — zero rows confirmed in all matrices | ✅ YES | 1,2,3 |
| **P2** | 2021–2022 coverage fix — only ~1,100 rows/year vs expected ~4,500 | ✅ YES | 1,2,3 |
| P3 | Pre-2011 Open price data (60% missing) | No | 1,2,3 |
| P3 | Analyst consensus estimates (EPS estimates, target prices) | No | 1,2 |
| P4 | Macroeconomic indicators (LKR interest rates, CCPI, GDP, USD/LKR) | No | 2,3 |
| P4 | Complete corporate action verification for all 281+ companies | No | 1,2,3 |
| P5 | GICS sector mapping for all 281+ listed companies | No | 1,2,3 |

---

## 7. Five Most Critical Data Quality Problems

### Problem 1 — SP20 Limitation (CRITICAL)
**All ML models are trained on 20 companies only.**
The CSE has 281+ listed securities. The pipeline is not production-ready for
the full market. The raw ZIP archives exist but are unused by the pipeline.

### Problem 2 — Static Future-Dated Feature Columns (HIGH — LEAKAGE)
Phase 3 feature matrices contain static announcement flags
(`has_recent_dividend`, `has_recent_rights_issue`, `has_recent_director_change`,
`total_announcements_count`) assigned using 2026-era knowledge. These violate
point-in-time integrity for all historical rows. Phase 4.1 removed them from
`prepared_training_dataset.parquet`, but upstream Phase 3 matrices are unmodified.

### Problem 3 — 2023 Data Gap + 2021–2022 Sparsity (HIGH)
The features matrices show **zero rows for 2023** and only ~1,100–1,145 rows
for 2021 and 2022 (expected ~4,500/year). The backtest period "2023–2025"
actually represents only 2024–2025. CAGR figures from Phase 4.2 cover only 2 years.

### Problem 4 — Pre-2011 Open Price Gap (60% Missing)
The 2001–2010 ZIP archive lacks Open prices for ~60% of observations.
Technical indicators using Open (ATR, NATR, Parabolic SAR, candlestick patterns)
are unreliable for the first decade of the dataset.

### Problem 5 — No Historical Fundamental Series (HIGH — Engines 02 & 03)
Phase 2D extracted single-period financial snapshots per company (latest annual /
Q2 FY2027). Engines 02 and 03 require time-series fundamentals (P/E, P/B, ROE,
revenue growth across years). Without this, fundamental momentum features cannot
be constructed, resulting in 39–54% NaN rates in the Engine 02 and 03 matrices.

---

## 8. Recommended Next Development Phase

### Phase A — Immediate (1 month): Data Infrastructure
1. Load full CSE universe from 4 ZIP archives through `data_loader.py`
2. Apply corporate action adjustments (splits, bonuses, rights) for all 281+ companies
3. Identify and backfill 2021–2023 trading data gaps
4. Extend GICS classification to all CSE-listed companies
5. Apply Phase 3.5 fix: add PSAR to v2 technical indicators for Engine 01

### Phase B — Short-term (2–3 months): Fundamental Data
1. Collect 5-year historical annual reports for all 281+ companies
2. Run Phase 2 PDF extraction pipeline on the expanded corpus
3. Build point-in-time fundamental database with period-end dates
4. Validate balance sheet reconciliation for all periods

### Phase C — Medium-term (3–6 months): Full-Scale Feature Engineering
1. Rebuild all three feature matrices for the full 281+ company universe
2. Enforce strict point-in-time integrity in all fundamental features
3. Add macro factors (LKR interest rates, CCPI, USD/LKR FX)
4. Re-run Phase 3.5 validation on expanded matrices

### Phase D — Medium-term: Model Re-Training
1. Re-train Engine 01 on full universe with corrected features
2. Train Engines 02 and 03 once fundamental series is available
3. Conduct out-of-sample backtesting with proper 2023 data
4. Deploy production recommendation pipeline

---

## Deliverables

| File | Size | Description |
|---|---|---|
| [01_repository_inventory.csv](01_repository_inventory.csv) | 78 KB | Complete workspace file inventory (664 files) |
| [02_dataset_schema_report.csv](02_dataset_schema_report.csv) | 5 KB | Schema details for all datasets |
| [03_data_quality_report.csv](03_data_quality_report.csv) | 1 KB | Quality metrics per dataset |
| [04_company_coverage_report.csv](04_company_coverage_report.csv) | 1 KB | Company-level coverage matrix |
| [05_corporate_actions_report.csv](05_corporate_actions_report.csv) | 5 KB | Corporate actions file audit (18 files) |
| [06_feature_engineering_audit.csv](06_feature_engineering_audit.csv) | 2 KB | Python module analysis (8 modules) |
| [07_fundamentals_gap_analysis.csv](07_fundamentals_gap_analysis.csv) | 1 KB | Fundamental data coverage |
| [08_data_leakage_risk_report.csv](08_data_leakage_risk_report.csv) | 1 KB | Leakage risk findings (2 HIGH-severity) |
| [09_investment_engine_readiness.csv](09_investment_engine_readiness.csv) | 1 KB | Engine-by-engine readiness (3 engines) |
| [10_additional_data_collection_plan.csv](10_additional_data_collection_plan.csv) | 2 KB | Prioritised data collection plan (8 items) |
| [run_phase0_audit.py](run_phase0_audit.py) | 61 KB | Reproducible audit script |

---

*Phase 0 Audit – AUDIT-ONLY. No original datasets were modified.*
