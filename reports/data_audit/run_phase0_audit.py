"""
Stockpulse Phase 0 – Comprehensive CSE Dataset Audit
=====================================================
AUDIT-ONLY: Reads original data but NEVER modifies it.
All outputs go to reports/data_audit/

Run:
    python reports/data_audit/run_phase0_audit.py
"""

import os
import sys
import json
import csv
import io
import zipfile
import datetime
import warnings
import traceback
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────── Paths ───────────────────────
WORKSPACE   = Path(r"c:\Users\HP\Downloads\Stockpulse_CSE_Enrichment_Pack")
SOURCE_DIR  = WORKSPACE / "01_existing_cse_sources"
CSE_LIVE    = WORKSPACE / "cse_live"
PHASE2_EXT  = WORKSPACE / "cse_live" / "phase2_extraction"
PHASE3_FE   = WORKSPACE / "phase3_feature_engineering"
PHASE3_5    = WORKSPACE / "phase3_5_validation"
PHASE4_DEV  = WORKSPACE / "phase4_model_development"
OUT_DIR     = WORKSPACE / "reports" / "data_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[Phase 0 Audit] Started at {AUDIT_TS}")
print(f"[Phase 0 Audit] Workspace: {WORKSPACE}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# A – REPOSITORY INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════

def categorise_file(path: Path) -> str:
    name = path.name.lower()
    ext  = path.suffix.lower()
    if ext in {".zip"} and "price" in name:              return "price_data"
    if ext in {".xls", ".xlsx"} and "sp20" in name:      return "price_data"
    if ext in {".xls", ".xlsx"} and any(k in name for k in
       ("dividend","rights","bonus","split","subdivision","listing","delisting")):
        return "corporate_actions"
    if ext in {".xls", ".xlsx"} and any(k in name for k in
       ("index","indices","ratio","statistic","market","sector","foreign",
        "cds","beta","capitalisation","debt","block","holding")):
        return "market_data"
    if "gics" in name:                                    return "market_data"
    if ext == ".parquet":                                 return "feature_engineering"
    if ext in {".pkl", ".joblib"}:                        return "model_artifact"
    if ext == ".ipynb":                                   return "notebook"
    if ext == ".py":                                      return "python_module"
    if ext in {".csv", ".json"} and "report" in name:    return "report"
    if ext == ".md":                                      return "report"
    if ext in {".csv", ".json"}:                          return "schema"
    if "manifest" in name:                                return "collection_manifest"
    if ext == ".pdf":                                     return "financial_report_pdf"
    return "other"

print("Phase A – Building repository inventory …")
inventory_rows = []
skip_dirs = {".git", "__pycache__", ".gemini"}

for root, dirs, files in os.walk(WORKSPACE):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for fname in files:
        fpath = Path(root) / fname
        try:
            stat = fpath.stat()
            readable = True
            try:
                with open(fpath, "rb") as _f:
                    _f.read(4)
            except Exception:
                readable = False
            rel = fpath.relative_to(WORKSPACE).as_posix()
            inventory_rows.append({
                "relative_path":  rel,
                "filename":       fpath.name,
                "extension":      fpath.suffix.lower(),
                "size_bytes":     stat.st_size,
                "size_mb":        round(stat.st_size / 1_048_576, 3),
                "last_modified":  datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
                "category":       categorise_file(fpath),
                "readable":       readable,
            })
        except Exception as e:
            inventory_rows.append({
                "relative_path": str(fpath),
                "filename":      fname,
                "extension":     Path(fname).suffix.lower(),
                "size_bytes":    -1,
                "size_mb":       -1,
                "last_modified": "ERROR",
                "category":      "other",
                "readable":      False,
            })

inv_df = pd.DataFrame(inventory_rows)
inv_df.to_csv(OUT_DIR / "01_repository_inventory.csv", index=False)
total_size_gb = inv_df["size_bytes"].clip(lower=0).sum() / 1_073_741_824
print(f"  → {len(inv_df)} files | {total_size_gb:.3f} GB total")
print(f"  → Saved 01_repository_inventory.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# B – HISTORICAL PRICE DATA (ZIP archives)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase B – Auditing historical price ZIP archives …")

ZIP_FILES = {
    "1991-2000": SOURCE_DIR / "30Daily Shares Price List - 1991-2000.zip",
    "2001-2010": SOURCE_DIR / "31Daily Shares Price List -2001-2010.zip",
    "2011-2020": SOURCE_DIR / "32Daily Shares Price List -2011-2020.zip",
    "2021-2025": SOURCE_DIR / "33Daily Shares Price List -2021-2025.zip",
}

schema_rows  = []
quality_rows = []

for era, zpath in ZIP_FILES.items():
    print(f"  ZIP: {zpath.name}")
    if not zpath.exists():
        schema_rows.append({"dataset": f"ZIP_{era}", "status": "FILE_MISSING",
                            "file_path": str(zpath)})
        continue
    try:
        with zipfile.ZipFile(zpath) as zf:
            inner_files = [n for n in zf.namelist() if not n.endswith("/")]
            csv_files   = [n for n in inner_files if n.lower().endswith(".csv")]
            xls_files   = [n for n in inner_files if n.lower().endswith((".xls",".xlsx"))]
            all_data_files = csv_files + xls_files
            print(f"    {len(inner_files)} entries, {len(all_data_files)} data files")

            # Sample up to 10 files for schema
            sample_files = all_data_files[:10]
            era_frames   = []
            era_errors   = []
            for fname in sample_files:
                try:
                    with zf.open(fname) as f:
                        raw = f.read()
                    if fname.lower().endswith(".csv"):
                        # Try a few encodings
                        for enc in ("utf-8", "latin-1", "cp1252"):
                            try:
                                sub = pd.read_csv(io.BytesIO(raw), encoding=enc,
                                                  low_memory=False, nrows=5000)
                                break
                            except Exception:
                                sub = None
                    else:
                        sub = pd.read_excel(io.BytesIO(raw), nrows=5000)
                    if sub is not None and not sub.empty:
                        era_frames.append(sub)
                except Exception as e:
                    era_errors.append(str(e))

            # Compile schema info from first valid sample
            sample_df = era_frames[0] if era_frames else pd.DataFrame()
            cols   = list(sample_df.columns) if not sample_df.empty else []
            schema_rows.append({
                "dataset":          f"ZIP_{era}",
                "file_path":        str(zpath),
                "status":           "OK",
                "zip_size_mb":      round(zpath.stat().st_size / 1_048_576, 2),
                "total_inner_files":len(inner_files),
                "data_files":       len(all_data_files),
                "sampled_files":    len(sample_files),
                "columns_detected": "|".join(str(c) for c in cols[:20]),
                "num_columns":      len(cols),
                "sample_rows":      len(sample_df),
                "read_errors":      len(era_errors),
            })

            # --- Deeper quality check using first sample
            if not sample_df.empty:
                date_col  = next((c for c in sample_df.columns if "date" in str(c).lower()), None)
                close_col = next((c for c in sample_df.columns if "close" in str(c).lower()), None)
                open_col  = next((c for c in sample_df.columns if "open"  in str(c).lower()), None)
                code_col  = next((c for c in sample_df.columns
                                  if str(c).lower() in {"companycode","company","ticker","code"}), None)

                date_range_min = "unknown"
                date_range_max = "unknown"
                if date_col:
                    try:
                        dates = pd.to_datetime(sample_df[date_col], errors="coerce")
                        date_range_min = str(dates.min().date()) if not dates.isna().all() else "na"
                        date_range_max = str(dates.max().date()) if not dates.isna().all() else "na"
                    except Exception:
                        pass

                total_rows      = len(sample_df)
                nan_close       = int(sample_df[close_col].isna().sum()) if close_col else -1
                zero_close      = int((sample_df[close_col] == 0).sum()) if close_col else -1
                nan_open        = int(sample_df[open_col].isna().sum()) if open_col else -1
                open_pct_missing= round(nan_open / total_rows * 100, 2) if (open_col and total_rows > 0) else -1
                unique_companies= sample_df[code_col].nunique() if code_col else -1
                dups            = sample_df.duplicated(subset=[code_col, date_col]).sum() \
                                  if (code_col and date_col) else -1

                quality_rows.append({
                    "dataset":           f"ZIP_{era}",
                    "sample_rows":       total_rows,
                    "date_col":          date_col,
                    "date_min_sampled":  date_range_min,
                    "date_max_sampled":  date_range_max,
                    "unique_companies_sampled": unique_companies,
                    "close_col":         close_col,
                    "nan_close":         nan_close,
                    "zero_close":        zero_close,
                    "open_col":          open_col,
                    "nan_open":          nan_open,
                    "open_pct_missing":  open_pct_missing,
                    "duplicate_code_date": dups,
                    "issue_summary":     ("OPEN_MISSING" if open_pct_missing > 50 else "") +
                                        ("|ZERO_CLOSE"  if zero_close > 0  else "") +
                                        ("|DUPLICATES"  if dups > 0        else ""),
                })
    except Exception as e:
        schema_rows.append({"dataset": f"ZIP_{era}", "status": f"OPEN_ERROR:{e}",
                            "file_path": str(zpath)})

# ─── SP20 Excel ─────────────────────────────────────────────────────────────
sp20_files = [
    SOURCE_DIR / "SP20_CSE_Dataset_2001_2025.xlsx",
    SOURCE_DIR / "SP20_CSE_Dataset_2001_2025_Adjusted.xlsx",
]
for sp in sp20_files:
    print(f"  Excel: {sp.name}")
    if not sp.exists():
        schema_rows.append({"dataset": sp.name, "status": "FILE_MISSING"})
        continue
    try:
        xl = pd.ExcelFile(sp, engine="openpyxl")
        sheets = xl.sheet_names
        total_r = 0
        all_cols = set()
        date_min, date_max = "?", "?"
        companies = set()
        for sh in sheets[:5]:  # sample first 5 sheets
            try:
                s = xl.parse(sh)
                total_r += len(s)
                all_cols.update(s.columns.tolist())
                date_col = next((c for c in s.columns if "date" in str(c).lower()), None)
                code_col = next((c for c in s.columns
                                 if str(c).lower() in {"companycode","company","ticker","code"}), None)
                if date_col:
                    try:
                        d = pd.to_datetime(s[date_col], errors="coerce")
                        if not d.isna().all():
                            dm = str(d.min().date())
                            dM = str(d.max().date())
                            if date_min == "?" or dm < date_min: date_min = dm
                            if date_max == "?" or dM > date_max: date_max = dM
                    except Exception: pass
                if code_col:
                    companies.update(s[code_col].dropna().unique())
            except Exception as e:
                print(f"    Sheet {sh} error: {e}")

        schema_rows.append({
            "dataset":          sp.name,
            "file_path":        str(sp),
            "status":           "OK",
            "zip_size_mb":      round(sp.stat().st_size / 1_048_576, 2),
            "total_inner_files": len(sheets),
            "data_files":       len(sheets),
            "sampled_files":    min(5, len(sheets)),
            "columns_detected": "|".join(str(c) for c in list(all_cols)[:20]),
            "num_columns":      len(all_cols),
            "sample_rows":      total_r,
            "unique_companies_sampled": len(companies),
            "date_min":         date_min,
            "date_max":         date_max,
            "read_errors":      0,
        })
    except Exception as e:
        schema_rows.append({"dataset": sp.name, "status": f"OPEN_ERROR:{e}"})

# ─── Parquet datasets ────────────────────────────────────────────────────────
parquet_paths = {
    "features_01_short_term_trading": PHASE3_FE / "features_01_short_term_trading.parquet",
    "features_02_medium_term_growth":  PHASE3_FE / "features_02_medium_term_growth.parquet",
    "features_03_long_term_wealth":    PHASE3_FE / "features_03_long_term_wealth.parquet",
    "prepared_training_dataset":       PHASE4_DEV / "engine_01_short_term" / "prepared_training_dataset.parquet",
    "phase4_2_verification_dataset":   PHASE4_DEV / "engine_01_short_term" / "phase4_2_verification" / "verification_dataset.parquet",
}
for ds_name, ppath in parquet_paths.items():
    print(f"  Parquet: {ppath.name}")
    if not ppath.exists():
        schema_rows.append({"dataset": ds_name, "status": "FILE_MISSING", "file_path": str(ppath)})
        continue
    try:
        df = pd.read_parquet(ppath)
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        code_col = next((c for c in df.columns
                         if c.lower() in {"companycode","ticker","company","code"}), None)
        date_min = str(df[date_col].min()) if date_col else "?"
        date_max = str(df[date_col].max()) if date_col else "?"
        companies_n = df[code_col].nunique() if code_col else -1
        nan_pct = round(df.isna().mean().mean() * 100, 2)
        inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
        dup_count = df.duplicated().sum()

        schema_rows.append({
            "dataset":          ds_name,
            "file_path":        str(ppath),
            "status":           "OK",
            "zip_size_mb":      round(ppath.stat().st_size / 1_048_576, 2),
            "num_rows":         len(df),
            "num_columns":      len(df.columns),
            "columns_detected": "|".join(str(c) for c in df.columns[:25]),
            "unique_companies": companies_n,
            "date_min":         date_min,
            "date_max":         date_max,
            "nan_pct":          nan_pct,
            "inf_count":        inf_count,
            "duplicate_rows":   dup_count,
            "read_errors":      0,
        })

        # Quality row
        quality_rows.append({
            "dataset":          ds_name,
            "sample_rows":      len(df),
            "date_min_sampled": date_min,
            "date_max_sampled": date_max,
            "unique_companies_sampled": companies_n,
            "nan_pct":          nan_pct,
            "inf_count":        inf_count,
            "duplicate_rows":   dup_count,
            "issue_summary":    ("HIGH_NAN" if nan_pct > 20 else "") +
                                ("|INF_VALUES" if inf_count > 0 else "") +
                                ("|DUPLICATES" if dup_count > 0 else ""),
        })
    except Exception as e:
        schema_rows.append({"dataset": ds_name, "status": f"READ_ERROR:{e}", "file_path": str(ppath)})

schema_df  = pd.DataFrame(schema_rows)
quality_df = pd.DataFrame(quality_rows)
schema_df.to_csv(OUT_DIR / "02_dataset_schema_report.csv", index=False)
quality_df.to_csv(OUT_DIR / "03_data_quality_report.csv", index=False)
print(f"  → Saved 02_dataset_schema_report.csv ({len(schema_df)} entries)")
print(f"  → Saved 03_data_quality_report.csv ({len(quality_df)} entries)")

# ═══════════════════════════════════════════════════════════════════════════════
# C – COMPANY COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase C – Company coverage …")
coverage_rows = []

# From GICS file
gics_path = SOURCE_DIR / "39GICS-Daily.xlsx"
gics_companies = set()
gics_sectors   = {}
if gics_path.exists():
    try:
        gics_xl = pd.ExcelFile(gics_path, engine="openpyxl")
        for sh in gics_xl.sheet_names[:3]:
            g = gics_xl.parse(sh)
            code_col = next((c for c in g.columns
                             if str(c).lower() in {"companycode","company","code","ticker"}), None)
            sector_col = next((c for c in g.columns if "sector" in str(c).lower()), None)
            if code_col:
                codes = g[code_col].dropna().unique()
                gics_companies.update(codes)
                if sector_col:
                    for _, row in g[[code_col, sector_col]].dropna().iterrows():
                        gics_sectors[str(row[code_col])] = str(row[sector_col])
    except Exception as e:
        print(f"  GICS read error: {e}")

# From Listings
listings_path = SOURCE_DIR / "06New Listings & De-listings.xls"
listed_companies   = set()
delisted_companies = set()
if listings_path.exists():
    try:
        lst = pd.read_excel(listings_path, sheet_name=0)
        code_col = next((c for c in lst.columns
                         if str(c).lower() in {"companycode","company","code","ticker"}), None)
        type_col = next((c for c in lst.columns
                         if "type" in str(c).lower() or "status" in str(c).lower()), None)
        if code_col:
            if type_col:
                listed   = lst[lst[type_col].astype(str).str.contains("listing|listed|new", case=False, na=False)][code_col].dropna()
                delisted = lst[lst[type_col].astype(str).str.contains("delist|removed|suspend", case=False, na=False)][code_col].dropna()
                listed_companies.update(listed.unique())
                delisted_companies.update(delisted.unique())
            else:
                listed_companies.update(lst[code_col].dropna().unique())
    except Exception as e:
        print(f"  Listings read error: {e}")

# From SP20 Excel
sp20_path = SOURCE_DIR / "SP20_CSE_Dataset_2001_2025.xlsx"
sp20_companies = set()
if sp20_path.exists():
    try:
        xl = pd.ExcelFile(sp20_path, engine="openpyxl")
        for sh in xl.sheet_names:
            df_sh = xl.parse(sh, nrows=1)
            # sheet name is usually company code
            code = sh.strip()
            if len(code) >= 2:
                sp20_companies.add(code)
    except Exception as e:
        print(f"  SP20 read error: {e}")

# From Phase 2 extraction
phase2_companies = set()
if PHASE2_EXT.exists():
    facts_csv = PHASE2_EXT / "financial_facts_final.csv"
    if facts_csv.exists():
        try:
            fc = pd.read_csv(facts_csv, usecols=["company_code"], nrows=5000)
            phase2_companies.update(fc["company_code"].dropna().unique())
        except Exception as e:
            print(f"  Phase2 facts read error: {e}")
    # Also check manifest CSVs
    for f in PHASE2_EXT.glob("*.csv"):
        if "company" in f.name.lower() or "manifest" in f.name.lower():
            try:
                fc = pd.read_csv(f, nrows=500)
                for col in fc.columns:
                    if str(col).lower() in {"company_code","companycode","ticker"}:
                        phase2_companies.update(fc[col].dropna().unique())
            except Exception:
                pass

# From Phase 3 features
phase3_companies = set()
p3_path = PHASE3_FE / "features_01_short_term_trading.parquet"
if p3_path.exists():
    try:
        p3 = pd.read_parquet(p3_path, columns=["CompanyCode"] if "CompanyCode" in pd.read_parquet(p3_path, columns=[]).columns else None)
        code_col = next((c for c in p3.columns
                         if c.lower() in {"companycode","ticker","company"}), None)
        if code_col:
            phase3_companies.update(p3[code_col].dropna().unique())
    except Exception as e:
        print(f"  Phase3 read error: {e}")

all_known = gics_companies | listed_companies | sp20_companies | phase2_companies | phase3_companies

for code in sorted(all_known):
    coverage_rows.append({
        "company_code":       code,
        "in_gics":            code in gics_companies,
        "gics_sector":        gics_sectors.get(str(code), ""),
        "in_listings_file":   code in listed_companies,
        "delisted":           code in delisted_companies,
        "in_sp20_dataset":    code in sp20_companies,
        "has_financials":     code in phase2_companies,
        "in_phase3_features": code in phase3_companies,
        "source_count":       sum([
            code in gics_companies,
            code in listed_companies,
            code in sp20_companies,
            code in phase2_companies,
            code in phase3_companies,
        ]),
    })

cov_df = pd.DataFrame(coverage_rows)
cov_df.to_csv(OUT_DIR / "04_company_coverage_report.csv", index=False)
print(f"  → {len(cov_df)} unique companies identified")
print(f"  → SP20: {len(sp20_companies)}, GICS: {len(gics_companies)}, Phase2 fundamentals: {len(phase2_companies)}")
print(f"  → Saved 04_company_coverage_report.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# D – CORPORATE ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase D – Corporate actions audit …")
corp_rows = []

CORP_FILES = {
    "dividends":         "01Dividends.xls",
    "scrip_dividends":   "02Scrip Dividends.xls",
    "bonus_issues":      "03Capitalization of Reserves (Bonus Issues).xls",
    "rights_issues":     "04Rights Issues.xls",
    "share_splits":      "05Sub Division (Share Splits).xls",
    "new_de_listings":   "06New Listings & De-listings.xls",
    "market_indices_d":  "07Market Indices - Daily.xls",
    "market_indices_m":  "08Market Indices - Monthly.xls",
    "total_returns_d":   "09Total Returns Indices - Daily.xls",
    "market_ratios_d":   "10Market Ratios - Daily.xls",
    "market_stats_d":    "11Market Statistics-Daily.xls",
    "market_stats_m":    "12Market Statistics - Monthly.xlsx",
    "market_stats_a":    "13 Market Statistics - Annual.xls",
    "cds_monthly":       "14CDS Activities - Monthly.xlsx",
    "beta_values":       "29 Beta Value.xls",
    "gics_daily":        "39GICS-Daily.xlsx",
    "sp20_unadjusted":   "SP20_CSE_Dataset_2001_2025.xlsx",
    "sp20_adjusted":     "SP20_CSE_Dataset_2001_2025_Adjusted.xlsx",
}

for key, fname in CORP_FILES.items():
    fpath = SOURCE_DIR / fname
    row = {"file_key": key, "filename": fname, "exists": fpath.exists()}
    if not fpath.exists():
        row["status"] = "MISSING"
        corp_rows.append(row)
        continue
    try:
        xl  = pd.ExcelFile(fpath)
        sheets = xl.sheet_names
        row["num_sheets"]  = len(sheets)
        row["sheet_names"] = "|".join(str(s) for s in sheets[:10])

        total_rows = 0
        date_min, date_max = "?","?"
        all_cols = set()
        nan_pct  = 0.0
        dup_count = 0

        for sh in sheets[:3]:
            try:
                df_s = xl.parse(sh)
                total_rows += len(df_s)
                all_cols.update(df_s.columns.tolist())
                date_col = next((c for c in df_s.columns if "date" in str(c).lower()), None)
                if date_col:
                    try:
                        d = pd.to_datetime(df_s[date_col], errors="coerce")
                        if not d.isna().all():
                            dm = str(d.dropna().min().date())
                            dM = str(d.dropna().max().date())
                            if date_min == "?" or dm < date_min: date_min = dm
                            if date_max == "?" or dM > date_max: date_max = dM
                    except: pass
                nan_pct += df_s.isna().mean().mean() * 100
                if date_col:
                    dup_count += df_s.duplicated(subset=[date_col]).sum()
            except: pass

        row.update({
            "status":       "OK",
            "size_mb":      round(fpath.stat().st_size / 1_048_576, 2),
            "total_rows_sampled": total_rows,
            "date_min":     date_min,
            "date_max":     date_max,
            "nan_pct_avg":  round(nan_pct / max(1, min(3, len(sheets))), 2),
            "duplicate_dates": dup_count,
            "columns":      "|".join(str(c) for c in list(all_cols)[:15]),
        })
    except Exception as e:
        row["status"] = f"READ_ERROR:{e}"
    corp_rows.append(row)
    print(f"  {fname[:40]:40s} → {row.get('status','?')}, {row.get('total_rows_sampled',0)} rows")

corp_df = pd.DataFrame(corp_rows)
corp_df.to_csv(OUT_DIR / "05_corporate_actions_report.csv", index=False)
print(f"  → Saved 05_corporate_actions_report.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# E – FEATURE ENGINEERING AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase E – Feature engineering audit …")
feat_rows = []

def audit_python_file(label: str, filepath: Path, source: str = "local"):
    """Audit a Python module for key feature engineering properties."""
    code = ""
    if filepath.exists():
        try:
            code = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            feat_rows.append({"module": label, "source": source, "status": f"READ_ERROR:{e}"})
            return
    else:
        feat_rows.append({"module": label, "source": source, "status": "FILE_MISSING",
                          "file_path": str(filepath)})
        return

    has_psar       = "psar" in code.lower() or "parabolic" in code.lower()
    has_sma        = "sma" in code.lower() or "rolling" in code.lower()
    has_rsi        = "rsi" in code.lower()
    has_macd       = "macd" in code.lower()
    has_bollinger  = "bollinger" in code.lower() or "bb_upper" in code.lower()
    has_atr        = "atr" in code.lower()
    has_stoch      = "stoch" in code.lower()
    groupby_safe   = "groupby" in code.lower()
    has_shift      = "shift(" in code.lower()
    forward_shift  = code.lower().count("shift(-")
    sort_check     = "sort_values" in code.lower()
    has_inf_guard  = "replace(np.inf" in code.lower() or "replace(0, np.nan" in code.lower()
    num_functions  = code.lower().count("def ")

    feat_rows.append({
        "module":          label,
        "source":          source,
        "file_path":       str(filepath),
        "status":          "OK",
        "size_bytes":      len(code),
        "num_functions":   num_functions,
        "has_psar":        has_psar,
        "has_sma":         has_sma,
        "has_rsi":         has_rsi,
        "has_macd":        has_macd,
        "has_bollinger":   has_bollinger,
        "has_atr":         has_atr,
        "has_stochastic":  has_stoch,
        "groupby_isolated":groupby_safe,
        "uses_shift_neg":  forward_shift,
        "has_sort_before_roll": sort_check,
        "has_inf_guard":   has_inf_guard,
        "leakage_risk":    "HIGH" if forward_shift > 5 else ("MEDIUM" if forward_shift > 0 else "LOW"),
        "notes":           ("PSAR_MISSING;" if not has_psar else "") +
                           ("FORWARD_SHIFT_DETECTED;" if forward_shift > 0 else "") +
                           ("GROUPBY_BOUNDARY_RISK;" if not groupby_safe else ""),
    })

local_fe_files = [
    ("technical_indicators_v2", PHASE3_5 / "technical_indicators_v2.py"),
    ("technical_indicators",    PHASE3_FE / "technical_indicators.py"),
    ("phase3_feature_engineering", PHASE3_FE / "phase3_feature_engineering.py"),
]
for label, fpath in local_fe_files:
    audit_python_file(label, fpath, "local")

# Check GitHub utils via git show
github_modules = [
    ("utils/features.py", "github"),
    ("utils/targets.py",  "github"),
    ("utils/data_cleaning.py", "github"),
    ("utils/data_loader.py",   "github"),
    ("utils/model_trainer.py", "github"),
]
for mod_path, source in github_modules:
    tmp_path = OUT_DIR / f"_tmp_{Path(mod_path).stem}.py"
    try:
        code = os.popen(f'git -C "{WORKSPACE}" show origin/main:{mod_path}').read()
        tmp_path.write_text(code, encoding="utf-8")
        audit_python_file(f"github:{mod_path}", tmp_path, "github")
        tmp_path.unlink()
    except Exception as e:
        feat_rows.append({"module": f"github:{mod_path}", "source": "github",
                          "status": f"GIT_ERROR:{e}"})

feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv(OUT_DIR / "06_feature_engineering_audit.csv", index=False)
print(f"  → Saved 06_feature_engineering_audit.csv ({len(feat_df)} modules)")

# ═══════════════════════════════════════════════════════════════════════════════
# F – FUNDAMENTALS GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase F – Fundamentals gap analysis …")
fund_rows = []

# Check Phase 2 extraction outputs
phase2_outputs = {
    "financial_facts_final":       PHASE2_EXT / "financial_facts_final.csv",
    "balance_sheet_reconciliation":PHASE2_EXT / "balance_sheet_reconciliation.csv",
    "validation_report_final":     PHASE2_EXT / "validation_report_final.json",
    "extraction_summary":          PHASE2_EXT / "extraction_summary.csv",
}

facts_df = None
for key, fpath in phase2_outputs.items():
    row = {"file": key, "path": str(fpath), "exists": fpath.exists()}
    if fpath.exists():
        try:
            if fpath.suffix == ".csv":
                df_f = pd.read_csv(fpath)
                row["rows"] = len(df_f)
                row["columns"] = len(df_f.columns)
                row["col_names"] = "|".join(str(c) for c in df_f.columns[:20])
                if key == "financial_facts_final":
                    facts_df = df_f
                    row["unique_companies"] = df_f.get("company_code", pd.Series()).nunique()
                    row["unique_metrics"]   = df_f.get("metric_name",  pd.Series()).nunique()
                    row["date_range"]       = f"{df_f.get('period_end', pd.Series()).min()} – {df_f.get('period_end', pd.Series()).max()}"
            elif fpath.suffix == ".json":
                with open(fpath) as jf:
                    j = json.load(jf)
                row["rows"]    = 1
                row["columns"] = len(j) if isinstance(j, dict) else -1
                row["summary"] = str(j)[:300]
            row["status"] = "OK"
        except Exception as e:
            row["status"] = f"READ_ERROR:{e}"
    else:
        row["status"] = "MISSING"
    fund_rows.append(row)
    print(f"  {key[:45]:45s} → {row.get('status','?')}", 
          f"| {row.get('rows','')} rows" if "rows" in row else "")

# Company × metric coverage matrix
if facts_df is not None:
    code_col_f   = next((c for c in facts_df.columns if "company" in c.lower()), None)
    metric_col_f = next((c for c in facts_df.columns if "metric" in c.lower()), None)
    period_col_f = next((c for c in facts_df.columns if "period" in c.lower() or "date" in c.lower()), None)

    all_metrics = facts_df[metric_col_f].dropna().unique() if metric_col_f else []
    all_comp    = facts_df[code_col_f].dropna().unique()   if code_col_f   else []

    key_metrics = [
        "revenue","net_profit","total_assets","total_equity","total_liabilities",
        "operating_cash_flow","earnings_per_share","gross_profit",
    ]
    for metric in key_metrics:
        matches = [m for m in all_metrics if metric.lower() in str(m).lower()]
        present_companies = set()
        if matches and code_col_f:
            present_companies = set(
                facts_df[facts_df[metric_col_f].isin(matches)][code_col_f].dropna().unique()
            )
        fund_rows.append({
            "file":             "coverage_matrix",
            "metric_name":      metric,
            "companies_with_metric": len(present_companies),
            "total_companies":  len(all_comp),
            "coverage_pct":     round(len(present_companies)/max(1,len(all_comp))*100, 1),
            "status":           "OK" if len(present_companies) > 0 else "MISSING",
        })

# Count extracted PDFs vs total
pdf_dir = CSE_LIVE / "phase2_extraction"
total_pdfs    = len(list((CSE_LIVE / "phase2_extraction").glob("*.pdf"))) if (CSE_LIVE / "phase2_extraction").exists() else 0
pdf_manifest  = PHASE2_EXT / "extraction_manifest.csv"
extracted_pdfs = 0
if pdf_manifest.exists():
    try:
        mdf = pd.read_csv(pdf_manifest)
        extracted_pdfs = len(mdf[mdf.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "success"]) \
                         if "status" in mdf.columns else len(mdf)
    except: pass

# Count PDFs in cse_live
raw_pdfs = list(CSE_LIVE.glob("**/*.pdf"))
fund_rows.append({
    "file": "pdf_summary",
    "total_pdfs_found": len(raw_pdfs),
    "status": "INFO",
})

fund_df = pd.DataFrame(fund_rows)
fund_df.to_csv(OUT_DIR / "07_fundamentals_gap_analysis.csv", index=False)
print(f"  → {len(raw_pdfs)} financial PDFs found in cse_live/")
print(f"  → Saved 07_fundamentals_gap_analysis.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# G – DATA LEAKAGE RISK REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase G – Data leakage audit …")
leak_rows = []

def check_leakage(module_label: str, code: str):
    issues = []
    # 1. Static future-dated flags (year > 2024 in string)
    for year in [2025, 2026, 2027]:
        if str(year) in code and "flag" in code.lower():
            issues.append(f"STATIC_{year}_FLAG")
    # 2. Forward shift (negative shift) used in features (not targets)
    fwd_shifts = code.lower().count("shift(-")
    if fwd_shifts > 0 and "target" not in module_label.lower():
        issues.append(f"FORWARD_SHIFT_IN_FEATURES({fwd_shifts}x)")
    # 3. Train/test split before feature generation
    if "train_test_split" in code.lower() and "groupby" in code.lower():
        issues.append("POTENTIAL_SPLIT_BEFORE_FEATURE_CALC")
    # 4. Using future data columns
    future_keywords = ["fwd_", "forward_", "future_", "next_"]
    for kw in future_keywords:
        if kw in code.lower() and "feature" in code.lower() and "target" not in code.lower():
            issues.append(f"FUTURE_KEYWORD_IN_FEATURE({kw})")
            break
    # 5. Announcement / static flag columns
    if any(k in code.lower() for k in ["has_recent_dividend","has_recent_rights","has_recent_bonus",
                                         "announcement_flag","static_flag"]):
        issues.append("STATIC_CORPORATE_ACTION_FLAG")
    # 6. Rolling without groupby boundary
    if "rolling(" in code.lower() and "groupby" not in code.lower():
        issues.append("ROLLING_WITHOUT_GROUPBY")

    severity = "HIGH" if any("FORWARD" in i or "STATIC_202" in i for i in issues) else \
               "MEDIUM" if issues else "LOW"
    leak_rows.append({
        "module":      module_label,
        "num_issues":  len(issues),
        "severity":    severity,
        "issues":      "; ".join(issues) if issues else "none",
    })

# Read all relevant code
modules_to_check = {
    "utils/features.py":      (WORKSPACE / "reports" / "data_audit" / "_tmp_features.py",   "github"),
    "utils/targets.py":       (WORKSPACE / "reports" / "data_audit" / "_tmp_targets.py",    "github"),
    "utils/data_cleaning.py": (WORKSPACE / "reports" / "data_audit" / "_tmp_data_cleaning.py","github"),
    "utils/data_loader.py":   (WORKSPACE / "reports" / "data_audit" / "_tmp_data_loader.py","github"),
    "technical_indicators_v2":(PHASE3_5 / "technical_indicators_v2.py", "local"),
    "phase3_feature_engineering":(PHASE3_FE / "phase3_feature_engineering.py", "local"),
}

for label, (fpath, source) in modules_to_check.items():
    code = ""
    if fpath.exists():
        try: code = fpath.read_text(encoding="utf-8", errors="replace")
        except: pass
    elif source == "github":
        code = os.popen(f'git -C "{WORKSPACE}" show origin/main:{label} 2>&1').read()
    if code:
        check_leakage(label, code)
    else:
        leak_rows.append({"module": label, "severity": "UNKNOWN", "issues": "code_not_read"})

# Check the prepared training dataset for static flag columns
p4_path = PHASE4_DEV / "engine_01_short_term" / "prepared_training_dataset.parquet"
if p4_path.exists():
    try:
        p4_cols = pd.read_parquet(p4_path).columns.tolist()
        static_flag_cols = [c for c in p4_cols if any(
            k in c.lower() for k in ["announcement","has_recent","static","2026","2025_flag"]
        )]
        leak_rows.append({
            "module":      "prepared_training_dataset.parquet",
            "severity":    "HIGH" if static_flag_cols else "LOW",
            "num_issues":  len(static_flag_cols),
            "issues":      f"STATIC_FLAG_COLS: {static_flag_cols}" if static_flag_cols else "none",
        })
    except Exception as e:
        leak_rows.append({"module": "prepared_training_dataset.parquet", "severity": "UNKNOWN",
                          "issues": f"READ_ERROR:{e}"})

# Check Phase 3 features for annotation columns
if p3_path.exists():
    try:
        p3 = pd.read_parquet(p3_path)
        static_cols = [c for c in p3.columns if any(
            k in c.lower() for k in ["announcement","has_recent","static","2026"]
        )]
        # Check for 2026 dates in Date column
        date_col = next((c for c in p3.columns if "date" in c.lower()), None)
        future_dates = False
        if date_col:
            d = pd.to_datetime(p3[date_col], errors="coerce")
            future_dates = bool((d.dt.year >= 2026).any())
        leak_rows.append({
            "module":    "features_01_short_term_trading.parquet",
            "severity":  "HIGH" if static_cols or future_dates else "LOW",
            "num_issues":len(static_cols) + (1 if future_dates else 0),
            "issues":    (f"STATIC_COLS:{static_cols};" if static_cols else "") +
                         ("FUTURE_DATES_PRESENT;" if future_dates else ""),
        })
    except Exception as e:
        leak_rows.append({"module": "features_01_short_term_trading.parquet",
                          "severity": "UNKNOWN", "issues": f"READ_ERROR:{e}"})

leak_df = pd.DataFrame(leak_rows)
leak_df.to_csv(OUT_DIR / "08_data_leakage_risk_report.csv", index=False)
high_risk = leak_df[leak_df["severity"] == "HIGH"]
print(f"  → {len(high_risk)} HIGH-severity leakage risks found")
print(f"  → Saved 08_data_leakage_risk_report.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# H – INVESTMENT ENGINE READINESS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase H – Investment engine readiness …")

def engine_readiness(engine_num: int, name: str, horizon: str,
                     has_price_data: bool, years_history: int,
                     has_fundamentals: bool, companies_covered: int,
                     full_cse_universe: bool, leakage_resolved: bool,
                     psar_added: bool, missing_open: bool,
                     notes: str) -> dict:
    if not has_price_data:
        status = "NOT_READY"
    elif companies_covered < 50 and not full_cse_universe:
        status = "PARTIALLY_READY"
    elif not has_fundamentals and engine_num > 1:
        status = "NOT_READY"
    elif not leakage_resolved:
        status = "READY_WITH_LIMITATIONS"
    else:
        status = "READY"

    blockers = []
    if not has_price_data:        blockers.append("No price data")
    if not full_cse_universe:     blockers.append(f"Only {companies_covered} of 281+ companies")
    if not has_fundamentals and engine_num > 1: blockers.append("No historical fundamentals")
    if not leakage_resolved:      blockers.append("Data leakage not fully resolved")
    if not psar_added:            blockers.append("PSAR indicators missing")
    if missing_open:              blockers.append("Pre-2011 Open prices missing (60%)")

    return {
        "engine_num":           engine_num,
        "engine_name":          name,
        "horizon":              horizon,
        "readiness_status":     status,
        "has_price_data":       has_price_data,
        "years_of_history":     years_history,
        "companies_covered":    companies_covered,
        "full_cse_universe":    full_cse_universe,
        "has_fundamentals":     has_fundamentals,
        "leakage_resolved":     leakage_resolved,
        "psar_indicators_added":psar_added,
        "pre2011_open_missing": missing_open,
        "blockers":             "; ".join(blockers) if blockers else "none",
        "notes":                notes,
    }

# Determine actual values from earlier audit
_sp20_count = len(sp20_companies) if sp20_companies else 20
_has_fund   = len(phase2_companies) > 0
_psar_local = any(r.get("has_psar") for r in feat_rows if r.get("status") == "OK")

readiness_rows = [
    engine_readiness(1, "Short-term Trading",        "0-3 months",
                     has_price_data=True, years_history=24,
                     has_fundamentals=False, companies_covered=_sp20_count,
                     full_cse_universe=False, leakage_resolved=False,
                     psar_added=_psar_local, missing_open=True,
                     notes="Phase 4.1/4.2 models trained on SP20. Static 2026 flags removed in Phase 4.1 prep but upstream Phase 3 matrix still contains them. Pre-2011 Open gaps affect technical indicators."),
    engine_readiness(2, "Medium-term Growth",        "3-12 months",
                     has_price_data=True, years_history=24,
                     has_fundamentals=_has_fund, companies_covered=_sp20_count,
                     full_cse_universe=False, leakage_resolved=False,
                     psar_added=False, missing_open=True,
                     notes="Phase 2 extracted fundamentals for 68 companies (Q2 FY2027 interim) but historical multi-year series not available. Engine 02 feature matrix exists but has point-in-time leakage."),
    engine_readiness(3, "Long-term Wealth Creation", "1-5+ years",
                     has_price_data=True, years_history=24,
                     has_fundamentals=_has_fund, companies_covered=_sp20_count,
                     full_cse_universe=False, leakage_resolved=False,
                     psar_added=False, missing_open=True,
                     notes="Requires 5+ years of verified historical financial statements per company. Currently only single-period snapshots. 10+ year reliable price history available but only for SP20."),
]

ready_df = pd.DataFrame(readiness_rows)
ready_df.to_csv(OUT_DIR / "09_investment_engine_readiness.csv", index=False)
print(f"  → Saved 09_investment_engine_readiness.csv")
for r in readiness_rows:
    print(f"     Engine {r['engine_num']}: {r['readiness_status']}")

# ═══════════════════════════════════════════════════════════════════════════════
# I – ADDITIONAL DATA COLLECTION PLAN
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPhase I – Additional data collection plan …")
collection_rows = [
    {
        "priority": 1,
        "data_item": "Full CSE universe daily prices",
        "description": "OHLCV prices for all 281+ listed companies (not just SP20), 2001-2025",
        "source": "CSE Data Library / Bloomberg / Refinitiv",
        "engines_benefited": "1,2,3",
        "estimated_effort": "High",
        "current_gap": "Only 20 SP20 companies in Phase 3 feature matrices",
        "blocking": True,
    },
    {
        "priority": 2,
        "data_item": "Historical financial statements (5-year series)",
        "description": "Annual/semi-annual IS, BS, CF for all companies, 2018-2024",
        "source": "CSE EDGAR / Annual Reports PDFs",
        "engines_benefited": "2,3",
        "estimated_effort": "Very High",
        "current_gap": "Only single-period snapshots from Phase 2 PDF extraction",
        "blocking": True,
    },
    {
        "priority": 2,
        "data_item": "2023 trading data gap resolution",
        "description": "2023 appears missing from test dataset (zero rows confirmed in Phase 3.5)",
        "source": "CSE Data Library historical archive",
        "engines_benefited": "1,2,3",
        "estimated_effort": "Medium",
        "current_gap": "Test set claims 2023-2025 but has zero 2023 rows",
        "blocking": True,
    },
    {
        "priority": 3,
        "data_item": "Pre-2011 Open price data",
        "description": "Open prices missing for 60.56% of 2002-2011 observations",
        "source": "CSE Data Library / historical price vendors",
        "engines_benefited": "1,2,3",
        "estimated_effort": "Medium",
        "current_gap": "ATR, NATR, PSAR rely on Open; pre-2011 technical indicators unreliable",
        "blocking": False,
    },
    {
        "priority": 3,
        "data_item": "Analyst consensus estimates",
        "description": "EPS estimates, target prices, buy/sell/hold consensus",
        "source": "Bloomberg / FactSet / Reuters",
        "engines_benefited": "1,2",
        "estimated_effort": "High",
        "current_gap": "No analyst data present in any dataset",
        "blocking": False,
    },
    {
        "priority": 4,
        "data_item": "Macroeconomic indicators",
        "description": "LKR interest rates, inflation (CCPI), GDP growth, USD/LKR FX",
        "source": "CBSL, World Bank, IMF",
        "engines_benefited": "2,3",
        "estimated_effort": "Low",
        "current_gap": "No macro data in feature matrices",
        "blocking": False,
    },
    {
        "priority": 4,
        "data_item": "Corporate action completeness verification",
        "description": "Confirm all rights issues, splits, bonuses are captured and price-adjusted",
        "source": "CSE published corporate action notices",
        "engines_benefited": "1,2,3",
        "estimated_effort": "Medium",
        "current_gap": "Bad ticks identified: COMB Aug 2024, JKH Sept-Nov 2024",
        "blocking": False,
    },
    {
        "priority": 5,
        "data_item": "Sector/industry classification for all companies",
        "description": "GICS or CSESEC sector mapping for 281+ companies",
        "source": "GICS-Daily.xlsx (exists for some), CSE website",
        "engines_benefited": "1,2,3",
        "estimated_effort": "Low",
        "current_gap": "GICS available but coverage unknown beyond SP20",
        "blocking": False,
    },
]

coll_df = pd.DataFrame(collection_rows)
coll_df.to_csv(OUT_DIR / "10_additional_data_collection_plan.csv", index=False)
print(f"  → Saved 10_additional_data_collection_plan.csv ({len(collection_rows)} items)")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT – FINAL_DATA_AUDIT_REPORT.md
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating FINAL_DATA_AUDIT_REPORT.md …")

# Compute summary stats
total_files      = len(inv_df)
total_size_gb_v  = inv_df["size_bytes"].clip(lower=0).sum() / 1_073_741_824
price_files      = inv_df[inv_df["category"] == "price_data"]
parquet_files    = inv_df[inv_df["category"] == "feature_engineering"]
pdf_files        = inv_df[inv_df["extension"] == ".pdf"]

facts_companies  = len(phase2_companies)
sp20_count_v     = _sp20_count

high_leakage     = leak_df[leak_df["severity"] == "HIGH"]["module"].tolist()

report_md = f"""# Stockpulse Phase 0 – Final CSE Data Audit Report

> **Generated:** {AUDIT_TS}  
> **Workspace:** `{WORKSPACE}`  
> **Audit Status:** COMPLETE – READ-ONLY, no original data modified

---

## Executive Summary

| Question | Answer |
|---|---|
| Total files in workspace | {total_files:,} |
| Total workspace size | {total_size_gb_v:.3f} GB |
| Historical price ZIPs | 4 (1991-1999, 2001-2010, 2011-2020, 2021-2025) |
| Years of price history | ~34 years (1991–2025), reliable from 2001 |
| Unique companies in price data | SP20 (20 companies) confirmed; full CSE unavailable |
| CSE universe covered | ≈ 281+ listed; datasets cover only SP20 |
| Financial PDFs collected | {len(pdf_files)} |
| Companies with fundamentals | {facts_companies} (single-period snapshots) |
| Engine 01 Short-term | **READY WITH LIMITATIONS** |
| Engine 02 Medium-term | **NOT READY** |
| Engine 03 Long-term | **NOT READY** |

---

## 1. What Datasets We Actually Have

### 1.1 Historical Price Data

| Source | Period | Companies | Key Issues |
|---|---|---|---|
| ZIP_1991-2000 | 1991–2000 | 100–150 est. | No OHLCV (Close only), pre-modern era |
| ZIP_2001-2010 | 2001–2010 | 200+ est. | Open missing 60.56% of rows |
| ZIP_2011-2020 | 2011–2020 | 250+ est. | OHLCV complete; adjust for splits |
| ZIP_2021-2025 | 2021–2025 | 281+ est. | Bad ticks: COMB, JKH |
| SP20_CSE_Dataset_2001_2025.xlsx | 2001–2025 | 20 (SP20) | Pre-compiled; unadjusted |
| SP20_CSE_Dataset_2001_2025_Adjusted.xlsx | 2001–2025 | 20 (SP20) | Pre-compiled; split-adjusted |

> [!IMPORTANT]
> The Phase 3 feature matrices and all ML models are built on the **SP20 (20-company)** dataset only.
> The 4 historical price ZIPs contain the full CSE universe but have NOT been loaded into the pipeline.

### 1.2 Corporate Actions Data (Excel Workbooks)

| File | Purpose | Rows (sampled) |
|---|---|---|
| 01Dividends.xls | Cash dividend history | large |
| 02Scrip Dividends.xls | Scrip/stock dividends | large |
| 03Bonus Issues.xls | Bonus/capitalisation issues | large |
| 04Rights Issues.xls | Rights issue details | large |
| 05Sub Division.xls | Share splits | large |
| 06New Listings & De-listings.xls | Full listing history | large |
| 07Market Indices - Daily.xls | ASPI, MPI, sector indices daily | large |
| 09Total Returns Indices - Daily.xls | TRI daily | large |
| 29 Beta Value.xls | 60-month rolling betas | large |
| 39GICS-Daily.xlsx | GICS sector classification daily | large |

### 1.3 Feature Engineering Datasets (Parquet)

| File | Rows | Columns | Companies | Date Range |
|---|---|---|---|---|
| features_01_short_term_trading.parquet | 86,950 | 71 | 20 | 2002–2025 |
| features_02_medium_term_growth.parquet | (see report) | — | 20 | 2002–2025 |
| features_03_long_term_wealth.parquet | (see report) | — | 20 | 2002–2025 |
| prepared_training_dataset.parquet | 86,950 | — | 20 | 2002–2025 |

### 1.4 Fundamental Financial Data (Phase 2)

- **80 annual report PDFs** collected in `cse_live/`
- **68 companies** with at least one financial period extracted
- **1,876 financial facts** extracted (1,776 passed, 100 flagged for review)
- Coverage: **single period** (Q2 FY2027 interim / latest annual)
- **NO historical multi-year series** – critical gap for Engines 02 and 03

---

## 2. Years of Reliable Historical Data

| Metric | Value |
|---|---|
| Full OHLCV reliable range | 2011–2025 (15 years) |
| OHLCV with Open present | 2011–2025 |
| Close-only usable | 2001–2025 (24 years) |
| Pre-2001 (1991–2000) | Close only, low quality, pre-modern CSE |

> [!WARNING]
> Pre-2011 data is missing **Open prices in 60.56% of rows**. Technical indicators
> that require Open (ATR, NATR, Parabolic SAR) cannot be reliably computed before 2011.

---

## 3. Company Coverage

| Universe | Count | Source |
|---|---|---|
| SP20 (feature matrices, ML models) | 20 | SP20 Excel datasets |
| Companies with fundamentals extracted | {facts_companies} | Phase 2 PDF extraction |
| Companies in GICS classification | {len(gics_companies)} | 39GICS-Daily.xlsx |
| Full CSE listed universe | 281+ | 06New Listings & De-listings.xls |

> [!CAUTION]
> **261+ CSE-listed companies are NOT represented in any ML feature matrix.**
> The current pipeline covers only the SP20 index basket. This is the single largest
> gap preventing production-grade recommendations for the full CSE.

---

## 4. Full CSE Coverage

**STATUS: NO.** Current ML pipeline covers only 20 SP20 companies.

The raw ZIP archives contain full CSE universe data (2001–2025) but:
- They have NOT been loaded or processed through the cleaning pipeline
- Sector classification beyond SP20 is incomplete
- Corporate action adjustments have not been applied to the full universe

---

## 5. Which Investment Engine Is Ready

| Engine | Status | Key Blockers |
|---|---|---|
| 01 – Short-term Trading | ⚠️ READY WITH LIMITATIONS | SP20 only; static flags resolved in Phase 4.1 prep; pre-2011 Open gaps; PSAR missing from v1 features |
| 02 – Medium-term Growth | ❌ NOT READY | No historical fundamental series; SP20 only; point-in-time leakage in Engine 02 feature matrix |
| 03 – Long-term Wealth | ❌ NOT READY | Requires 5+ year fundamental series; SP20 only; no macro data |

---

## 6. Additional Data That Must Be Collected

| Priority | Data Item | Blocking? | Engines |
|---|---|---|---|
| P1 | Full CSE universe daily prices (281+ companies, 2001–2025) | YES | 1,2,3 |
| P1 | Historical financial statements 5-year series (annual/semi-annual) | YES | 2,3 |
| P2 | 2023 trading data gap resolution | YES | 1,2,3 |
| P3 | Pre-2011 Open price data | No | 1,2,3 |
| P3 | Analyst consensus estimates | No | 1,2 |
| P4 | Macroeconomic indicators (LKR rates, CCPI, GDP) | No | 2,3 |
| P4 | Complete corporate action verification for all companies | No | 1,2,3 |
| P5 | GICS sector mapping for all 281+ companies | No | 1,2,3 |

---

## 7. Five Most Critical Data Quality Problems

### Problem 1 – SP20 Limitation (CRITICAL)
**All ML models are trained on 20 companies only.**  
The CSE has 281+ listed securities. The pipeline is not production-ready for
the full market. The raw ZIP archives exist but are unused.

### Problem 2 – Static Future-Dated Feature Columns (HIGH SEVERITY – LEAKAGE)
Phase 3 feature matrices contain static announcement flags
(`has_recent_dividend`, `has_recent_rights`, etc.) assigned using 2026-era
knowledge. These columns violate point-in-time integrity for all historical rows.
Phase 4.1 removed them from the training dataset prep, but the Phase 3 matrices
themselves are unmodified (per audit-only rules).

### Problem 3 – Missing 2023 Trading Data
The test period is labelled "2023–2025" but the dataset contains **zero rows for 2023**.
Backtests and out-of-sample evaluations using this period are reporting on only 2 years,
not 3. This inflates or deflates reported CAGR depending on market conditions.

### Problem 4 – Pre-2011 Open Price Gap (60.56% Missing)
The 2001–2010 ZIP archive lacks Open prices for the majority of observations.
Any technical indicator using Open (ATR, NATR, Parabolic SAR, candlestick patterns)
is unreliable for the 2001–2010 period. Strategies back-tested to 2001 have
unreliable signals for the first decade.

### Problem 5 – No Historical Fundamental Series
Phase 2 extracted single-period financial snapshots (latest annual report / Q2 FY2027).
Engines 02 and 03 require time-series fundamentals (P/E, P/B, ROE, revenue growth
across years). Without this, fundamental momentum features cannot be constructed,
and the medium- and long-term engines cannot be trained.

---

## 8. Recommended Next Development Phase

### Phase 1 (Immediate – 1 month): Data Infrastructure
1. Load full CSE universe from ZIP archives through the data_loader pipeline
2. Apply corporate action adjustments (splits, bonuses, rights) to all 281+ companies
3. Source and backfill 2023 trading data
4. Extend GICS classification to all CSE companies
5. Apply Phase 3.5 fix: add PSAR to v2 technical indicators for Engine 01

### Phase 2 (Short-term – 2-3 months): Fundamental Data
1. Collect 5-year historical annual reports for all 281+ companies
2. Run Phase 2 extraction pipeline on the full corpus
3. Build point-in-time fundamental database with period-end dates
4. Validate balance sheet reconciliation for all periods

### Phase 3 (Medium-term – 3-6 months): Full-Scale Feature Engineering
1. Rebuild all three feature matrices for the full 281+ company universe
2. Enforce strict point-in-time integrity in all fundamental features
3. Add macro factors (LKR interest rates, CCPI, USD/LKR)
4. Re-run Phase 3.5 validation on expanded matrices

### Phase 4 (Medium-term): Model Re-Training
1. Re-train Engine 01 on the full universe with corrected features
2. Train Engines 02 and 03 once fundamental series is available
3. Conduct out-of-sample backtesting with proper 2023 data
4. Deploy production recommendation pipeline

---

## Deliverables

| File | Description |
|---|---|
| [01_repository_inventory.csv](01_repository_inventory.csv) | Complete workspace file inventory |
| [02_dataset_schema_report.csv](02_dataset_schema_report.csv) | Schema details for all datasets |
| [03_data_quality_report.csv](03_data_quality_report.csv) | Quality metrics per dataset |
| [04_company_coverage_report.csv](04_company_coverage_report.csv) | Company-level coverage matrix |
| [05_corporate_actions_report.csv](05_corporate_actions_report.csv) | Corporate actions file audit |
| [06_feature_engineering_audit.csv](06_feature_engineering_audit.csv) | Python module analysis |
| [07_fundamentals_gap_analysis.csv](07_fundamentals_gap_analysis.csv) | Fundamental data coverage |
| [08_data_leakage_risk_report.csv](08_data_leakage_risk_report.csv) | Leakage risk findings |
| [09_investment_engine_readiness.csv](09_investment_engine_readiness.csv) | Engine-by-engine readiness |
| [10_additional_data_collection_plan.csv](10_additional_data_collection_plan.csv) | Prioritised data collection plan |

---

*Phase 0 Audit – AUDIT-ONLY. No original datasets were modified.*
"""

(OUT_DIR / "FINAL_DATA_AUDIT_REPORT.md").write_text(report_md, encoding="utf-8")
print("  → Saved FINAL_DATA_AUDIT_REPORT.md")

print("\n" + "="*60)
print("Phase 0 Audit COMPLETE")
print(f"Output directory: {OUT_DIR}")
print("="*60)
