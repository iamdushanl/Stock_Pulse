# Stockpulse Phase 4A – Short-Term Modeling Dataset Preparation & Target Engineering Report

**Status:** Complete & Verified  
**Pipeline Entry Point:** [run_phase4a.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/run_phase4a.py)  
**Source Modules:** [calendar_and_targets.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase4a_dataset/calendar_and_targets.py), [features_eligibility_and_splits.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase4a_dataset/features_eligibility_and_splits.py)  
**Test Suite:** [test_phase4a.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/tests/test_phase4a.py) (`17/17` passed; `142/142` across Phases 1B–4A)  
**Scope:** Strictly dataset preparation, target engineering, eligibility auditing, purged chronological splits, and leakage controls (`0` predictive ML models trained; `0` investment recommendations generated).

---

## 1. Executive Summary

Phase 4A transforms the validated Phase 3D short-term integrated research universe ([short_term_integrated_research_universe.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase3d/short_term_integrated_research_universe.parquet), `1,414,977` historical daily records × `143` columns across `1990-01-12` to `2025-12-31`) into a leakage-controlled, model-ready short-term dataset ([short_term_modeling_dataset.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_modeling_dataset.parquet)):

- **Model-Ready Short-Term Pool:** **`361,589` rows × `224` columns** (`2002-04-04` to `2025-12-31`) covering **`301` unique securities** across `5,659` trading dates — preserving **`252` Listed securities (`339,065` rows)** and **`49` Delisted securities (`22,524` rows)** per authoritative Phase 1B `HistoricalStatus`.
- **Full-Universe Eligibility Waterfall:** All `1,414,977` records classified into mutually-exclusive inclusion/exclusion states in [short_term_sample_eligibility.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_sample_eligibility.parquet) and [short_term_exclusion_audit.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_exclusion_audit.parquet).
- **Exact Calendar-Session Forward Targets (`5d`, `10d`, `20d`, `60d`):** Computed strictly via `(SecurityID, SessionIndex + h)` matching against the `8,251`-session CSE exchange calendar (never `DataFrame.shift(-h)` and never forward-filled across halts or past `MappingValidTo`), strictly separating dividend-inclusive Total Return vs official **ASTRI** benchmark from Capital-Adjusted Price Return vs **ASPI**.
- **Critical Source-Data Glitches & Basis Safeguards Resolved:**
  1. Quarantined **6 corrupted single-day ASTRI benchmark entries** in `09Total Returns Indices - Daily.xls` (five ASPI paste-overs + `2010-07-06` `40,366.00` typo), eliminating spurious `-845%` excess return crashes.
  2. Normalized Phase 2A moving-average/level indicators strictly by split-adjusted `Close` (`~2e-12` precision) rather than `CapitalAdjClose` (avoiding `0.10 LKR` rights/bonus jumps).
  3. Scaled `Label_Fwd_RealizedVol_Ann_{h}d` cleanly into decimal annualized units (`Daily_Return_CapAdj / 100.0`) matching decimal return/drawdown targets.
  4. Segregated sub-rupee penny-tick dynamics (`RawClose < 1.00 LKR`, `1,349` rows) via `IsPennyStockTickNoiseRisk` / `IsLowPrice_Sub1LKR` without deleting historical observations.
- **Zero-Leakage Purged Chronological Splits:** Four expanding CV folds (`CV_Fold_1`..`CV_Fold_4`) plus held-out **`FINAL_TEST` (`2024-01-02` to `2025-12-31`, `63,674` rows, `233` securities)**, each enforced with a strict **`65`-session gap** (`60d` max target purge + `5d` embargo) and empirically verified `zero_target60d_overlap`.
- **Upstream Immutability Verified:** All `22` locked upstream files across Phases 1A–3D verified byte-for-byte via SHA-256 checksums (`upstream_immutability_verified: true`), and live future-price perturbation checks measured **`0.0` maximum feature drift** across `7,505` pre-cutoff test rows.

---

## 2. Deliverables & Integrity Manifest

| Deliverable File | Size | SHA-256 Prefix | Role |
|---|---|---|---|
| [short_term_modeling_dataset.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_modeling_dataset.parquet) | `203.639 MB` | `d2f379b901e6...` | `361,589 × 224` ST model-ready pool (features, targets, path labels, split masks) |
| [short_term_sample_eligibility.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_sample_eligibility.parquet) | `6.125 MB` | `3c085d5cd14d...` | Full `1,414,977`-row inclusion/exclusion status & horizon readiness |
| [short_term_validation_splits.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_validation_splits.parquet) | `0.505 MB` | `715c0091b87c...` | Expanding `CV_Fold_1..4` & `FINAL_TEST` assignments (`65`-session purged) |
| [short_term_exclusion_audit.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_exclusion_audit.parquet) | `0.009 MB` | `80424b22b1f2...` | `99`-row aggregated waterfall & target exclusion breakdown by Listed/Delisted |
| [short_term_feature_registry.json](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_feature_registry.json) | `0.098 MB` | `0f328ed30d7e...` | Full column dispositions (`93` candidates, `47` pruned core, Spearman dev pairs) |
| [short_term_target_definitions.json](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_target_definitions.json) | `0.003 MB` | `f8d0aa1107ca...` | Formal target/label mathematical specs & hazard gating policy |
| [short_term_target_quality_report.json](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_target_quality_report.json) | `0.016 MB` | `f46bb4138773...` | Horizon distributions, sensitivity matrix, ASTRI quarantine, yearly stats |
| [short_term_dataset_manifest.json](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_dataset_manifest.json) | `0.011 MB` | Master Index | Run metadata, SHA-256 output digests & 22-file upstream lock verification |

---

## 3. Sample Eligibility Waterfall & Survivorship Retention

![Eligibility Waterfall and Universe Evolution](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/reports/phase4/visualizations/phase4a/01_eligibility_waterfall_and_universe_evolution.png)

Across all `1,414,977` Phase 3D historical observations:

| Waterfall Stage / Primary Reason | Historical Rows | Share (%) | Listed Rows | Delisted Rows |
|---|---:|---:|---:|---:|
| `FAILED_LIQUIDITY_TURNOVER_500K_THRESHOLD` | `520,523` | `36.79%` | `466,100` | `54,423` |
| **`ELIGIBLE_SHORT_TERM_CORE`** | **`361,589`** | **`25.55%`** | **`339,065`** | **`22,524`** |
| `NOT_HISTORICALLY_TRADABLE_OR_LIFECYCLE_INVALID` | `217,202` | `15.35%` | `163,380` | `53,822` |
| `FAILED_TRADING_FREQ_OR_ACTIVE_WINDOW` | `172,957` | `12.22%` | `142,297` | `30,660` |
| `CONSECUTIVE_SUSPENSION_OR_GAP_DAY` | `141,615` | `10.01%` | `118,497` | `23,118` |
| `OTHER_PHASE2C_LIQUIDITY_STATUS_EXCLUDED` | `722` | `0.05%` | `640` | `82` |
| `INVALID_PRICE_OR_UNVALIDATED_CA` | `369` | `0.03%` | `314` | `55` |
| **Total CSE Historical Universe** | **`1,414,977`** | **`100.00%`** | **`1,230,293`** | **`184,684`** |

**Security-Level Master Reconciliation (`JINS-N-0000`):** Authoritative Phase 1B `cse_security_master_enriched.parquet` classifies `JINS-N-0000` (`3,150` eligible rows) as `Delisted` (`DelistingDate = 2019-11-20` alongside continuous quotes through `2025-12-31`), yielding `252 Listed / 49 Delisted (22,524 rows)` versus `253 Listed / 48 Delisted (19,374 rows)` in Phase 3D `company_security_pit_mapping.parquet`. Target generation strictly enforces `MappingValidTo` (`2019-11-20` for `JINS-N-0000`) so zero delisted post-boundary prices enter targets.

**Cross-Section Depth (`5,659` Eligible Dates):** Minimum `3`, `p05 = 12`, `p10 = 17`, `p25 = 27`, **`median = 49`**, `p75 = 94`, `p90 = 134`, `max = 212` (peaking at `232` distinct securities across `2025`). Exactly `426` thin early/subdued sessions fall below `MIN_CROSS_SECTION_SIZE = 15`, where cross-sectional percentile/top-quintile targets are masked to `NaN`/`NA`.

---

## 4. Target Engineering, Hazard Controls & Quality Audit

![Horizon Target Distributions and Coverage](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/reports/phase4/visualizations/phase4a/02_horizon_target_distributions_and_penny_noise.png)

### 4.1 Calendar Geometry & Return Separation
1. **Exact Session Alignment:** Consecutive CSE rows frequently skip untraded sessions (`diff_pos > 1` on `180k+` rows). Forward targets lookup exact `(SecurityID, SessionIndex + h)` on the `8,251`-session calendar (`8,240` Phase 1 sessions + `11` early `1990` quote dates). Skipped target dates or halts emit `SKIPPED_SESSION_OR_HALT_AT_TARGET_DATE` (`NaN`), never shifting onto `t+h+k`.
2. **Strict `TR` vs `Cap` Separation:**
   - **Primary (`Target_Excess_TotalReturn_{h}d`):** `(TotalReturnPrice[t+h]/TotalReturnPrice[t] - 1) - (ASTRI[t+h]/ASTRI[t] - 1)` requires `ASTRI` present at both endpoints, 100% clean dividend resolution (`IsDividendResolutionClean == True` across all observed path bars, active `2008–2025`), and zero quarantined CA / source-error intersections (`IsTargetEligible_TR_{h}d`).
   - **Secondary (`Target_Excess_CapitalReturn_{h}d`):** `(CapitalAdjClose[t+h]/CapitalAdjClose[t] - 1) - (ASPI_Close[t+h]/ASPI_Close[t] - 1)` retains `2002–2007` and unresolved-cash-dividend paths without polluting `TotalReturn` (`IsTargetEligible_Cap_{h}d`).

### 4.2 Benchmark & Corporate-Action Hazard Quarantine
- **Quarantined ASTRI Glitches (`6` sessions):** Our cross-benchmark divergence check (`|r_ASTRI - r_ASPI| > 5%` with immediate `> 5%` opposite next-day reversal) isolated 6 corrupted entries in `09Total Returns Indices - Daily.xls`: **`2007-11-14`, `2010-05-21`, `2010-06-04`, `2010-07-06`, `2010-09-28`, `2015-05-20`**. Without quarantine, `2010-06-04` (`4,354.20`) -> `2010-07-06` (`40,366.00`) caused synthetic `-845%` excess return outliers. Nullifying these 6 entries (`NaN`, zero interpolation) moved 20d TR excess `min` from `-8.4502` to `-0.9745`.
- **Corporate Action & Source Error Blocking:** Phase 2C `QUARANTINED` (`76`) / `VERIFIED_CORRECTION_REQUIRED` (`17`) allotment-proxy windows (`KAPI-N-0000 2010-12-10`, `SING-N-0000 2011-08-11`) and `SOURCE_DATA_ERROR` (`4,018`) / `SECURITY_IDENTITY_ISSUE` (`242`) are blocked across entire `[t, t+h]` intervals (`635` rows excluded at `20d`).

### 4.3 Target Summary Statistics (`IsShortTermEligible = 361,589`)

| Horizon | Valid TR Rows | Valid TR CS≥15 | Valid Cap Rows | TR Mean | TR Std | TR Median | TR p01 / p99 | Downside Risk Rate | Ann. Vol Median |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **`5d`** | `309,393` | `307,565` | `355,059` | `+0.00125` | `0.0818` | `-0.00455` | `-0.1249 / +0.2251` | `5.42%` | `0.2873` |
| **`10d`** | `307,984` | `306,113` | `353,564` | `+0.00221` | `0.1116` | `-0.00702` | `-0.1720 / +0.3260` | `13.64%` | `0.3191` |
| **`20d`** | **`305,410`** | **`303,537`** | **`350,711`** | **`+0.00426`** | **`0.1556`** | **`-0.01112`** | **`-0.2405 / +0.4948`** | **`28.87%`** | **`0.3419`** |
| **`60d`** | `294,608` | `292,572` | `339,451` | `+0.01350` | `0.3059` | `-0.02271` | `-0.4145 / +1.0218` | `65.19%` | `0.3728` |

On the primary `20d` horizon across `2008–2025` (`326,310` rows), TR eligibility reaches **`93.60%`** (`Positive` excess class rate `43.57%`, `TopQuintile` rate `20.81%`, `MaxDrawdown_20d` median `-7.56%`, p10 `-18.42%`).

### 4.4 Sub-Rupee Tick Noise & Threshold Sensitivity

![Threshold Sensitivity and Missingness](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/reports/phase4/visualizations/phase4a/04_liquidity_threshold_sensitivity_and_missingness.png)

- **Penny Tick Disparity (`RawClose < 1.00 LKR`):** `1,294` TR-20d rows below Rs. 1.00 exhibit `std = 1.0908` (`max +16.966`, e.g., `BIL-N-0000`, `PCH-N-0000`), whereas the `304,116` `Core >= 1.00 LKR` rows achieve `std = 0.1386` (`p01=-0.2399`, `p99=+0.4924`, `max=+4.2908`). Both pools are retained in the modeling file with `IsPennyStockTickNoiseRisk` / `IsCleanCorePool` toggles.
- **Liquidity Trade-Off:** Phase 2C's baseline (`MedTurn60 >= 500k LKR`, `Freq >= 70%`) sustains median daily cross-section `N = 49` (`92.47%` dates `>= 15`), while `HighLiq >= 1M` contracts median `N` to `29` (`76.04%` dates `>= 15`) and `Institutional >= 2.5M` drops median `N` to `12` (`42.85%` dates `>= 15`), confirming `500k LKR` + optional `RawClose >= 1.00 LKR` (`360,240` rows) as optimal for Phase 5A.

---

## 5. Feature Engineering, Registry & Preprocessing Contract

- **`93` Candidate Features** documented in [short_term_feature_registry.json](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_feature_registry.json):
  * `43` full-sample technical & market-relative features from Phase 2A/2B/2C/3D.
  * `23` Phase 4A causal scale-free transforms normalizing Rs moving averages/oscillators by split-adjusted **`Close`** (`Price_to_SMA20_Pct`, `SMA20_to_SMA50_Pct`, `Price_to_VWAP20_Pct`, `MACD_Hist_PctClose`, `ATR_PctClose`, `Vol_Ratio_ATR_HistVol`, etc.) and `log1p` volume/turnover/market-cap transforms.
  * `24` PIT fundamental ratios & staleness/financial-institution flags (`FundamentalAvailabilityDate <= Date` verified across `100%` of populated observations; missingness declines from `~58–77%` in `DEV_POOL` to `~13–59%` in `FINAL_TEST`).
  * `3` structural `2017+` session-open features (`Overnight_Gap_Ret`, `Intraday_Ret`, `IsOpenSessionRegime`), absent pre-2017 (`0%`) and `99.93–100%` present from 2017 onward.
- **`38` Level-Confounded & Path-Dependent Features Excluded:** Raw prices, raw MAs, `OBV`, `AD_Line`, `Chaikin_Osc_3_10`, unscaled `ShareVolume`/`Turnover`/`ttm_revenue`, plus constant `IsSectorMappingAvailable == False`.
- **Linear/Collinear Pruning (Strictly on `DEV_POOL`):** Audited `44` pairs with Spearman `|r| >= 0.90` strictly inside `DEV_POOL` (`<= 2023-09-25`), yielding a **`47`-feature pruned core technical set** (`68` with PIT fundamentals, `71` post-2017). All imputers/scalers MUST be fit strictly inside training folds.

---

## 6. Purged Chronological Validation & Untouched Holdout Architecture

![Purged Chronological Splits](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/reports/phase4/visualizations/phase4a/03_purged_chronological_splits_and_regime_counts.png)

Every expanding split enforces a strict **`65`-session gap** (`60` max horizon purge + `5` embargo sessions) in [short_term_validation_splits.parquet](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase4a/short_term_validation_splits.parquet):

| Split / Fold | Train Window | Train Rows (Delisted Secs) | Gap Sessions (Rows Purged) | Eval / Val Window | Eval Rows (TR20 Valid) | Max Train `TargetDate_60d` | Zero Overlap? |
|---|---|---:|---:|---|---:|---:|:---:|
| **`CV_Fold_1`** | `2002-04-04`..`2018-09-21` | `198,612` (`48`) | `65` (`1,194`) | `2019-01-02`..`2019-12-31` | `5,624` (`5,530`) | `2018-12-21` | Verified |
| **`CV_Fold_2`** | `2002-04-04`..`2019-09-25` | `202,891` (`48`) | `65` (`2,539`) | `2020-01-02`..`2020-12-31` | `13,644` (`13,452`) | `2019-12-23` | Verified |
| **`CV_Fold_3`** | `2002-04-04`..`2020-09-25` | `212,002` (`48`) | `65` (`7,072`) | `2021-01-04`..`2021-12-31` | `27,803` (`27,259`) | `2020-12-22` | Verified |
| **`CV_Fold_4`** | `2002-04-04`..`2021-09-27` | `237,908` (`49`) | `65` (`8,969`) | `2022-01-03`..`2023-09-25` | `44,868` (`44,319`) | `2021-12-24` | Verified |
| **`FINAL_TEST`** | `2002-04-04`..`2023-09-25` | `291,745` (`49`) | `65` (`6,170`) | **`2024-01-02`..`2025-12-31`** | **`63,674` (`59,123`)** | `2023-12-20` | **Verified** |

---

## 7. Verification, Leakage Controls & Phase 5A Handoff

- **Automated Verification (`142/142` Passed):** `17` dedicated tests in [test_phase4a.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/tests/test_phase4a.py) plus all `125` regression tests across Phases 1B–3D passed (`0` failures, `0` skipped) after the final pipeline run, along with instant `--incremental` cache verification (`513ecd4fd8f6`).
- **Live Future Perturbation:** Perturbing post-cutoff prices/turnovers/mkt-cap produced **`0.0` difference** (`7,505` pre-cutoff test rows).
- **Phase 5A Modeling Readiness:**
  * Use `CV_Fold_1..4` exclusively for hyperparameter tuning, feature selection, and calibration; evaluate final out-of-sample generalization once on **`FINAL_TEST` (`2024–2025`)**.
  * Prefer **`Target_ExcessTR_PctRank_20d`** / **`Target_Excess_TotalReturn_20d`** (with `Target_TopQuintile_ExcessTR_20d` & `Label_Fwd_DownsideRisk_Flag_20d`) as primary supervised targets, fitting all scalers/imputers inside training folds and testing sensitivity with `IsCleanCorePool == True` (`RawClose >= 1.00 LKR`).
