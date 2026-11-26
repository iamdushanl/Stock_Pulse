# Stockpulse Phase 2A – Full CSE Technical Indicator Engineering Report

**Status:** COMPLETE (`8/8` pytest verification checks passed in `7.95s`; Phase 1A and Phase 1B datasets unmodified)  
**Universe:** Full Colombo Stock Exchange (CSE) (`1990-01-12` to `2025-12-31`)  
**Coverage:** `1,414,977` rows | `719` `SecurityID`s (`549` Listed, `170` Delisted) | `400` `CompanyCode`s  
**Pipeline Entrypoint:** [run_phase2a.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase2_technical_features/run_phase2a.py) (`51.35s` end-to-end execution)

> [!IMPORTANT]
> No machine learning models were trained and no recommendation rankings were built in Phase 2A. Phase 1A (`Dataset/processed/phase1/`) and Phase 1B (`Dataset/processed/phase1b/`) files remain strictly untouched.

---

## 1. Executive Summary & Deliverables Inventory

Phase 2A built a modular, strictly causal, vectorized + stateful technical feature engineering library in [phase2_technical_features/](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase2_technical_features/__init__.py) covering **63 documented indicators** across **6 groups (A–F)**, computed independently per `SecurityID` while preserving `(SecurityID, CompanyCode, Date, MainType, SubType)` across every share class (`N, X, R, W, P, D, B, U`).

| Deliverable (`Dataset/processed/phase2/`) | Rows | Columns | Purpose |
| :--- | ---: | ---: | :--- |
| `cse_technical_features.parquet` | 1,414,977 | 47 | Groups A (Trend), B (Momentum), C (Volatility), E (Trend Reversal & PSAR) + 5 key cols |
| `cse_price_volume_features.parquet` | 1,414,977 | 20 | Group D (Volume, Liquidity, Open-dependent gaps) + `RawClose`, `SplitAdjClose`, `CapitalAdjClose`, daily returns |
| `cse_market_relative_features.parquet` | 1,414,977 | 16 | Group F (ASPI momentum, matched-horizon RS & Beta, Sector breadth, honest NaN unmapped sector slots) |
| [technical_indicator_metadata.csv](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase2/technical_indicator_metadata.csv) | 63 | 11 | Complete documentation: formula, params, inputs, price basis, warm-up, missingness, limitations |
| [feature_quality_report.csv](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase2/feature_quality_report.csv) | 63 | 21 | Per-indicator NaN %, Inf count, era sparsity, distribution quantiles, and bounds validation |
| [phase2_manifest.json](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/Dataset/processed/phase2/phase2_manifest.json) | 1 | — | Structured execution, PSAR verification, causality, correlation, and corporate-action audit log |

---

## 2. Indicator Inventory & Implementation Status

All **63** indicators are implemented in [indicators_core.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase2_technical_features/indicators_core.py) and cataloged in [metadata_catalog.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase2_technical_features/metadata_catalog.py):

| Group | Output Table | Count | Indicators Implemented (All Validated) |
| :--- | :--- | ---: | :--- |
| **A. Trend** | `cse_technical_features` | **12** | `SMA_20`, `SMA_50`, `SMA_200`, `EMA_12`, `EMA_26`, `WMA_20`, `ADX_14`, `Aroon_Up_25`, `Aroon_Down_25`, `Ichimoku_TK_Spread_Pct`, `Ichimoku_Kumo_Pos_Pct`, `Supertrend_10_3` |
| **B. Momentum** | `cse_technical_features` | **11** | `RSI_14`, `MACD_Line`, `MACD_Signal`, `MACD_Hist`, `ROC_10`, `ROC_20`, `CCI_20`, `Williams_R_14`, `Stoch_K_14`, `Stoch_D_3`, `MOM_10` |
| **C. Volatility** | `cse_technical_features` | **9** | `ATR_14`, `NATR_14`, `BB_Upper_20_2`, `BB_Lower_20_2`, `BB_Width_20_2`, `BB_PctB_20_2`, `Keltner_Pos_20`, `Hist_Vol_21d`, `Donchian_Pos_20` |
| **D. Volume & Price/Open** | `cse_price_volume_features` | **11** | `OBV`, `MFI_14`, `CMF_20`, `VWAP_20`, `Volume_SMA_20`, `Volume_Ratio_20`, `AD_Line`, `Chaikin_Osc_3_10`, `Amihud_Illiquidity_20`, `Overnight_Gap_Ret`, `Intraday_Ret` |
| **E. Trend Reversal** | `cse_technical_features` | **10** | `PSAR`, `PSAR_Trend`, `PSAR_Distance_Pct`, `PSAR_Reversal`, `Supertrend_Dir_10_3`, `Supertrend_Reversal`, `MA_Cross_Spread_20_50`, `MA_Cross_Signal_20_50`, `RSI_Div_Bullish_14`, `RSI_Div_Bearish_14` |
| **F. Market-Relative** | `cse_market_relative_features` | **10** | `ASPI_MOM_20`, `ASPI_MOM_60`, `RS_ASPI_Level`, `RS_ASPI_ROC_20`, `RS_ASPI_ROC_60`, `Beta_ASPI_60`, `Beta_ASPI_252`, `Sector_MOM_20`, `RS_Sector_ROC_20`, `Sector_Breadth_MOM_20` |

### Key Algorithmic Safeguards Built In
- **Price Basis Integrity:** Groups A–E operate on Phase 1B `cse_daily_prices_split_adjusted.parquet` (`Open`, `High`, `Low`, `Close`, `ShareVolume`, `Turnover`) where OHLC and ShareVolume remain dimensionally consistent; Group F return/Beta indicators use `CapitalAdjClose` (adjusted for Splits, Bonus Issues, Scrip Dividends, and Rights TERP) paired with matched-horizon `[Date_{t-1} -> Date_t]` ASPI returns so non-synchronous thin trading does not bias Beta downward.
- **Strictly Causal Ichimoku & RSI Divergence:** Ichimoku `Senkou Span A/B` values formed at `t-26` are shifted **forward (+26)** to sit over bar `t` (`Ichimoku_Kumo_Pos_Pct`); `Chikou Span` (`Close[t+26]`) is excluded to prevent future leakage. `RSI_Div_Bullish_14` / `RSI_Div_Bearish_14` compare the trailing 5-bar extreme strictly against the preceding 20-bar window `[t-25..t-5]`.

---

## 3. Feature Quality & Automated Validation Results

All validation checks executed inside [validation.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/phase2_technical_features/validation.py) and [test_phase2a.py](file:///C:/Users/HP/Downloads/Stockpulse_CSE_Enrichment_Pack/tests/test_phase2a.py) (**8 / 8 passed in 7.95s**):

| Check Category | Observed Result |
| :--- | :--- |
| **Duplicate `(SecurityID, Date)` keys** | **0** across all 3 Parquet outputs (`1,414,977` rows each) |
| **Infinite values (`inf` / `-inf`)** | **0** across all **63** indicators |
| **Theoretical domain bounds (`bounds_ok`)** | **63 / 63 True** (`RSI_14`, `ADX_14`, `Aroon`, `Stoch`, `MFI_14` in `[0,100]`; `Williams_R_14` in `[-100,0]`; `CMF_20` in `[-1,1]`; `Donchian_Pos_20` in `[0,1]`) |
| **Causality / Prefix invariance** | Max difference across all 63 indicators when truncating at `2024-06-28`: **`1.42e-14`** (0 future leakage) |
| **Cross-security contamination** | **`True`** (`JKH-N-0000` invariant within `1e-12` when `SAMP-N-0000` is scaled `3.7x`; `COMB-N/X/R` isolated) |
| **Corporate action continuity** | Ex-date split returns match continuous series (`JKH-N-0000@2024-11-05`: `-0.24%`; `LMF`: `-6.15%`; `AFS`: `-16.67%`; `SUN`: `+0.22%`; `WATA`: `-3.24%`; `HHL`: `+0.62%`; zero `-90%` split crash spikes) |
| **Pre-2017 `Open` fabrication check** | Non-null count before `2017-01-01`: **`0`** for both `Overnight_Gap_Ret` and `Intraday_Ret` |

### Feature Redundancy Findings (`|corr| >= 0.95`, 2017–2025 active window)
- `Williams_R_14` and `Stoch_K_14` correlate at **`+1.00000`** because `Williams_R_14 = Stoch_K_14 - 100`. Retain only one in linear/tree models.
- Price-level indicators (`SMA_20/50/200`, `EMA_12/26`, `WMA_20`, `BB_Upper_20_2`, `BB_Lower_20_2`, `Supertrend_10_3`, `PSAR`, `VWAP_20`, `RawClose`) show **`0.96–0.9999`** cross-sectional correlation solely from shared LKR price scale; normalized dimensionless ratios (`PSAR_Distance_Pct`, `BB_PctB_20_2`, `BB_Width_20_2`, `MA_Cross_Spread_20_50`) eliminate nominal price confounding.

---

## 4. Missing-Value Analysis & Honest Era Coverage

Missingness reflects authentic CSE market structure, strict warm-up windows (`232` securities have `< 20` trading days and `22` have `1` trading day), and refusal to fabricate missing quotes or historical sectors:

| Feature / Era Subset | NaN % Overall | NaN % Pre-2002 | NaN % Pre-2017 | NaN % 2017+ | Root Cause |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **`Overnight_Gap_Ret` / `Intraday_Ret`** | 62.43% | 100.00% | 100.00% | **0.01%–0.03%** | Raw CSE sources carry `OPEN PRICE` only from 2017 onward (+56 NaN Open source rows). Never substituted with prior Close. |
| **`ADX_14`, `ATR_14`, `CCI_20`, `PSAR`, `Supertrend`, `Aroon`, `Stoch`, `MFI_14`, `CMF_20`, `AD_Line`** | 13.99%–16.74% | **100.00%** | 22.4%–25.9% | **0.02%–1.56%** | `1990–2001` (`close_only`/`flat_cloprc`) lacks High & Low; `2002+` has H/L (plus 180 isolated NaN H/L rows). |
| **`OBV`, `VWAP_20`, `Volume_SMA_20`, `Volume_Ratio_20`, `Amihud_Illiquidity_20`** | 12.70%–14.73% | 90.6%–100.0% | 20.3%–23.4% | **0.02%–0.29%** | `1990–2000` lacks `ShareVolume` (`179,237` rows) and `Turnover` (`197,873` rows). Zero volume stays finite/NaN cleanly from `2001+`. |
| **`SMA_20`, `WMA_20`, `BB_*`, `ROC_20`** | 1.32%–3.06% | 6.2%–18.8% | 1.9%–4.7% | **0.28%–0.29%** | Strict warm-up windows (`20` bars) + `3,974` pre-1998 `ZERO_CLOSE` rows masked cleanly from return/divisor math. |
| **`SMA_200`, `Beta_ASPI_252`** | 10.95%–16.05% | 53.2%–58.6% | 16.4%–24.3% | **1.87%–2.33%** | 200–252 trading day warm-up excludes short-lived Rights (`R`) and recent IPOs. |
| **`ASPI_MOM_20`, `ASPI_MOM_60`, `Sector_Breadth_MOM_20`** | **0.01%** | 0.00% | 0.02% | **0.00%** | `07Market Indices` + `39GICS-Daily.xlsx` complete across `1990–2025` (194 early rows only on missing index cross-dates). |
| **`Sector_MOM_20`, `RS_Sector_ROC_20`** | **100.00%** | 100.00% | 100.00% | **100.00%** | `cse_sector_mapping.parquet` legitimately holds `Sector="Unknown"` for all 719 securities (`MANUAL_RESEARCH_REQUIRED`) to prevent Nov-2026 leakage. |

---

## 5. Parabolic SAR Verification (`af_start=0.02, af_step=0.02, af_max=0.20`)

The Phase 2A implementation fixes a row-0 look-ahead flaw present in `phase3_5_validation/technical_indicators_v2.py` (where bar `0` peeked at `close[1] >= close[0]`):

- **Causal Fix Verified:** Bar `0` of every contiguous H/L sequence is **strictly `NaN`**, initializing causally at bar `1` once `[0, 1]` closes are observed. Zero future bars are ever inspected.
- **Independent Reference Comparison (`independent_psar_reference`):** Tested on both synthetic trend-reversal paths and `2020–2025` live CSE series:
  - `JKH-N-0000` (`1,379` bars): `max_abs_diff = 0.0`, `trend_match = True`, `reversal_match = True`, `envelope_invariant_ok = True`
  - `COMB-N-0000` (`1,395` bars): `max_abs_diff = 0.0`, `trend_match = True`, `reversal_match = True`, `envelope_invariant_ok = True`
  - `HHL-N-0000` (`1,394` bars): `max_abs_diff = 0.0`, `trend_match = True`, `reversal_match = True`, `envelope_invariant_ok = True`
- **Envelope Invariant:** Verified `PSAR <= Low` on every bullish bar and `PSAR >= High` on every bearish bar.
- **Parameter Stability (`JKH-N-0000`, `2017–2025`):** Default `(0.02, 0.02, 0.20)` yields `151` flips vs `123` (`(0.015, 0.015, 0.15)`, `85.31%` regime agreement) and `185` (`(0.025, 0.025, 0.25)`, `86.73%` regime agreement).

---

## 6. Known Limitations & Microstructure Caveats

1. **Missing Historical Sector Crosswalk:** `Sector_MOM_20` and `RS_Sector_ROC_20` remain `NaN` until curated point-in-time historical sector classifications replace `Unknown` in `cse_sector_mapping.parquet`. Macro sector rotation is captured safely in `Sector_Breadth_MOM_20`.
2. **Pre-2017 Open & Pre-2002 High/Low Regimes:** Any ML pipeline training across pre-2017 years must exclude `Overnight_Gap_Ret` / `Intraday_Ret`; any pipeline training across pre-2002 years must exclude High/Low and Volume features or switch cleanly to Close/ASPI features.
3. **Unadjusted Scrip & Rights Ex-Dates in Split-Only OHLC:** Groups A–E operate on Split+Bonus adjusted OHLC (`Close`) for exact ShareVolume consistency, while `152` securities have additional Scrip/Rights events reflected in `CapitalAdjClose` (used by Group F). Note extreme tail prints exist in genuine historical raw data (e.g. `SELI-N-0000` shifting `0.60 -> 1,287.0` on `1`-share odd-lot prints in 2013–2014, and decaying `R` Rights entitlements such as `LCEY-R-0000`).
4. **CSE Odd-Lot Close Outside H/L Rule:** Because board-lot Close can differ slightly from odd-lot intra-day High/Low (`2015–2025`), `Stoch_K_14`, `Williams_R_14`, `Donchian_Pos_20`, and `CMF_20` clamp range fractions safely to their theoretical oscillator domains.

---

## 7. Recommended Features for Short-Term Trading (Engine 01: `0–3` Months)

Before training Engine 01 models in downstream phases, enforce two hygiene rules: filter Tradable Universe (`MainType == "N"` or explicitly weight classes, plus liquidity floor using `Amihud_Illiquidity_20` / `Volume_SMA_20`) and use scale-free normalized indicators (dropping price-level `SMA_*`/`EMA_*`/`BB_Upper`/`BB_Lower`/`PSAR`/`OBV` raw levels and collinear `Williams_R_14` duplicate of `Stoch_K_14`):

1. **Trend & Regime Strength (Normalized):** `MA_Cross_Spread_20_50`, `ADX_14`, `Aroon_Up_25`, `Aroon_Down_25`, `Ichimoku_TK_Spread_Pct`, `Ichimoku_Kumo_Pos_Pct`, `PSAR_Trend`, `PSAR_Distance_Pct`, `PSAR_Reversal`, `Supertrend_Dir_10_3`
2. **Momentum & Short-Term Mean-Reversion:** `RSI_14`, `MACD_Hist` (prefer scaling by `Close` at feature prep), `ROC_10`, `ROC_20`, `CCI_20`, `Stoch_K_14`, `Stoch_D_3`, `RSI_Div_Bullish_14`, `RSI_Div_Bearish_14`
3. **Volatility & Breakout Squeeze:** `NATR_14`, `BB_Width_20_2`, `BB_PctB_20_2`, `Keltner_Pos_20`, `Hist_Vol_21d`, `Donchian_Pos_20`
4. **Volume Conviction & Microstructure:** `MFI_14`, `CMF_20`, `Volume_Ratio_20`, `Amihud_Illiquidity_20`, `(Close - VWAP_20) / VWAP_20` *(plus `Overnight_Gap_Ret` and `Intraday_Ret` exclusively for 2017+ models)*
5. **Market-Relative Context:** `ASPI_MOM_20`, `RS_ASPI_ROC_20`, `RS_ASPI_ROC_60`, `Beta_ASPI_60`, `Sector_Breadth_MOM_20`
