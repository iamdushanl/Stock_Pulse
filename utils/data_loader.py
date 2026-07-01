"""
CSE EDA Data Loader
===================

Comprehensive data loading utilities for the Colombo Stock Exchange (CSE)
historical dataset.  Handles:

- Daily share prices from ZIP archives (1991 – 2025)
- Market indices (ASPI, MPI, sector indices)
- Market statistics (turnover, trades, market cap)
- Securities list, stock splits, new listings / de-listings
- GICS sector indices, sector market cap, dividends, sector ratios

All loaders include robust error handling — individual year/sheet failures
are logged as warnings and never crash the full load.
"""

from __future__ import annotations

import io
import logging
import os
import warnings
import zipfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback: tqdm-less progress — just iterate normally
    def tqdm(iterable, **kwargs):
        return iterable

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_BASE_PATH = r"c:\Users\HP\Documents\Stock_pulse\Dataset"
BASE_PATH = DEFAULT_BASE_PATH  # Convenience alias used by notebooks
CACHE_FILENAME = "consolidated_daily_prices.parquet"

# ZIP archive definitions — (zip_name, internal_subfolder)
ZIP_CONFIGS = [
    (
        "30Daily Shares Price List - 1991-2000.zip",
        "32Daily Shares Price List - 1991-2000",
    ),
    (
        "31Daily Shares Price List -2001-2010.zip",
        "31Daily Shares Price List -2001-2010",
    ),
    (
        "32Daily Shares Price List -2011-2020.zip",
        "32Daily Shares Price List -2011-2020",
    ),
    (
        "33Daily Shares Price List -2021-2025.zip",
        "40Daily Shares Price List -2021-2025",
    ),
]

# Unified output columns for load_all_daily_prices()
UNIFIED_COLUMNS = [
    "Date", "CompanyCode", "ShortName", "MainType", "SubType",
    "Open", "High", "Low", "Close",
    "TradeVolume", "ShareVolume", "Turnover", "Era",
]


# ══════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_base(base_path: Optional[str] = None) -> Path:
    """Resolve and validate the dataset base directory."""
    p = Path(base_path) if base_path else Path(DEFAULT_BASE_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Dataset directory not found: {p}")
    return p


def _read_excel_or_csv(file_bytes: bytes, filename: str,
                       **kwargs) -> pd.DataFrame:
    """
    Read a file (xls, xlsx, or csv) from raw bytes.

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename (used to determine format).
    **kwargs
        Forwarded to ``pd.read_excel`` or ``pd.read_csv``.
    """
    lower = filename.strip().lower()
    buf = io.BytesIO(file_bytes)

    if lower.endswith(".csv"):
        return pd.read_csv(buf, **kwargs)
    elif lower.endswith(".xlsx"):
        return pd.read_excel(buf, engine="openpyxl", **kwargs)
    else:
        # .xls — use xlrd
        return pd.read_excel(buf, engine="xlrd", **kwargs)


def _detect_era(filename: str) -> str:
    """Determine the era from a ZIP filename."""
    if "1991-2000" in filename:
        return "1991-2000"
    return "2001-2025"


def _extract_year_from_filename(filename: str) -> Optional[str]:
    """Try to extract a 4-digit year from a filename."""
    import re
    m = re.search(r"(\d{4})", filename)
    return m.group(1) if m else None


# ══════════════════════════════════════════════════════════════════════════════
# Early Era Parser (1991–2000)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_early_era_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse an early-era (1991–2000) daily price file.

    These files have 3 columns read with ``header=None``:
    - Col 0: Date (datetime or NaT for header/separator rows)
    - Col 1: Stock code string (e.g. 'SAMP', 'ABAN')
    - Col 2: Close price (float)

    Quirks handled:
    - Row 0 is a header row (NaT, 'SECURITY_DA', 'PRICE')
    - Separator rows with '----------' strings
    - Repeated headers mid-file (rows containing 'SECURITY_DA', 'COMP', etc.)
    - Zero prices (kept, but could be flagged downstream)
    """
    try:
        df = _read_excel_or_csv(
            file_bytes, filename,
            header=None,
        )
    except Exception as e:
        logger.warning("Failed to read early-era file %s: %s", filename, e)
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # Assign working column names
    if df.shape[1] >= 3:
        df.columns = list(range(df.shape[1]))
    else:
        logger.warning("Early-era file %s has %d columns, expected ≥3",
                       filename, df.shape[1])
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # ── Filter out junk rows ──
    # Remove rows where column 1 contains header/separator strings
    junk_patterns = [
        "SECURITY_DA", "COMP", "CLOSE_PRICE", "PRICE",
        "----------", "==========",
    ]
    mask = pd.Series(True, index=df.index)
    for pat in junk_patterns:
        col1_str = df[1].astype(str).str.strip()
        mask &= ~col1_str.str.contains(pat, case=False, na=False)

    # Also remove rows where col 0 is a string (separator lines)
    col0_str = df[0].astype(str).str.strip()
    mask &= ~col0_str.str.contains(r"^-+$", na=False, regex=True)

    df = df.loc[mask].copy()

    # ── Parse dates ──
    df["Date"] = pd.to_datetime(df[0], errors="coerce")
    df = df.dropna(subset=["Date"])

    # ── Parse close price ──
    df["Close"] = pd.to_numeric(df[2], errors="coerce")

    # ── Build unified output ──
    result = pd.DataFrame({
        "Date": df["Date"],
        "CompanyCode": df[1].astype(str).str.strip(),
        "ShortName": np.nan,
        "MainType": np.nan,
        "SubType": np.nan,
        "Open": np.nan,
        "High": np.nan,
        "Low": np.nan,
        "Close": df["Close"],
        "TradeVolume": np.nan,
        "ShareVolume": np.nan,
        "Turnover": np.nan,
        "Era": "1991-2000",
    })

    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Modern Era Parser (2001–2025)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_modern_era_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse a modern-era (2001–2025) daily price file.

    These files have 12 columns.  Rows 0–3 are title / empty rows;
    row 4 is the header row.

    Columns:
    0  COMPANY ID        (str)
    1  MAIN TYPE         (str)
    2  SUB TYPE          (str)
    3  SHORT NAME        (str)
    4  TRADING DATE      (datetime)
    5  PRICE HIGH (Rs.)  (float)
    6  PRICE LOW (Rs.)   (float)
    7  CLOSE PRICE (Rs.) (float)
    8  OPEN PRICE (Rs.)  (float)
    9  TRADE VOLUME (No.)(int/float)
    10 SHARE VOLUME (No.)(int/float)
    11 TURNOVER (Rs.)    (float)
    """
    try:
        df = _read_excel_or_csv(
            file_bytes, filename,
            header=None,
            skiprows=5,  # Skip rows 0-4 (title + header)
        )
    except Exception as e:
        logger.warning("Failed to read modern-era file %s: %s", filename, e)
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    if df.shape[1] < 12:
        logger.warning(
            "Modern-era file %s has %d columns, expected 12. "
            "Attempting best-effort parse.",
            filename, df.shape[1],
        )
        # Pad with NaN columns if fewer than 12
        for i in range(df.shape[1], 12):
            df[i] = np.nan

    # Assign numeric column indices
    df.columns = list(range(df.shape[1]))

    # ── Filter out junk rows ──
    # Remove rows where col 0 is NaN or contains header-like strings
    col0_str = df[0].astype(str).str.strip()
    junk_mask = (
        col0_str.str.contains("COMPANY", case=False, na=False)
        | col0_str.str.contains("^-+$", na=False, regex=True)
        | col0_str.str.contains("^=+$", na=False, regex=True)
        | col0_str.isin(["nan", "", "None"])
    )
    df = df.loc[~junk_mask].copy()

    # ── Parse trading date ──
    df["Date"] = pd.to_datetime(df[4], errors="coerce")

    # Drop rows without a valid date
    df = df.dropna(subset=["Date"])

    # ── Numeric conversions ──
    numeric_cols = {5: "High", 6: "Low", 7: "Close", 8: "Open",
                    9: "TradeVolume", 10: "ShareVolume", 11: "Turnover"}
    for col_idx, col_name in numeric_cols.items():
        df[col_name] = pd.to_numeric(df[col_idx], errors="coerce")

    # ── Build unified output ──
    result = pd.DataFrame({
        "Date": df["Date"],
        "CompanyCode": df[0].astype(str).str.strip(),
        "ShortName": df[3].astype(str).str.strip() if 3 in df.columns else np.nan,
        "MainType": df[1].astype(str).str.strip() if 1 in df.columns else np.nan,
        "SubType": df[2].astype(str).str.strip() if 2 in df.columns else np.nan,
        "Open": df["Open"],
        "High": df["High"],
        "Low": df["Low"],
        "Close": df["Close"],
        "TradeVolume": df["TradeVolume"],
        "ShareVolume": df["ShareVolume"],
        "Turnover": df["Turnover"],
        "Era": "2001-2025",
    })

    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main Daily Prices Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_all_daily_prices(
    base_path: Optional[str] = None,
    force_reload: bool = False,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all daily share prices from CSE ZIP archives (1991–2025).

    Checks for a Parquet cache first.  If the cache exists and
    ``force_reload`` is False, the cached DataFrame is returned
    immediately.  Otherwise, all ZIPs are parsed and the result is
    cached to Parquet.

    Parameters
    ----------
    base_path : str or None
        Path to the ``Dataset`` directory.  Defaults to
        ``DEFAULT_BASE_PATH``.
    force_reload : bool, default False
        If True, ignore the cache and re-parse everything.
    cache_path : str or None
        Override the default Parquet cache path.

    Returns
    -------
    pd.DataFrame
        Unified daily prices with columns defined by
        :data:`UNIFIED_COLUMNS`.
    """
    base = _resolve_base(base_path)
    parquet_path = Path(cache_path) if cache_path else base / CACHE_FILENAME

    # ── Try cache ──
    if not force_reload and parquet_path.exists():
        logger.info("Loading cached daily prices from %s", parquet_path)
        print(f"📦 Loading cached data from {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
        print(f"   ✅ Loaded {len(df):,} rows, "
              f"{df['CompanyCode'].nunique():,} companies")
        return df

    # ── Parse all ZIPs ──
    print("🔄 Parsing daily price files from ZIP archives...")
    all_frames: list[pd.DataFrame] = []

    for zip_name, subfolder in tqdm(ZIP_CONFIGS, desc="ZIP archives"):
        zip_path = base / zip_name
        if not zip_path.exists():
            logger.warning("ZIP not found: %s", zip_path)
            continue

        era = _detect_era(zip_name)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # List data files inside the ZIP (filter to xls/xlsx/csv)
                data_files = [
                    name for name in zf.namelist()
                    if (
                        name.lower().endswith((".xls", ".xlsx", ".csv"))
                        and not name.startswith("__MACOSX")
                        and not os.path.basename(name).startswith("~$")
                    )
                ]

                for fname in tqdm(data_files,
                                  desc=f"  {zip_name[:40]}",
                                  leave=False):
                    try:
                        raw = zf.read(fname)
                        basename = os.path.basename(fname)

                        if era == "1991-2000":
                            df = _parse_early_era_file(raw, basename)
                        else:
                            df = _parse_modern_era_file(raw, basename)

                        if not df.empty:
                            all_frames.append(df)
                            logger.info(
                                "  Parsed %s: %d rows", basename, len(df),
                            )

                    except Exception as e:
                        logger.warning(
                            "  ⚠ Failed to parse %s: %s", fname, e,
                        )
                        continue

        except zipfile.BadZipFile:
            logger.error("Bad ZIP file: %s", zip_path)
            continue
        except Exception as e:
            logger.error("Error processing ZIP %s: %s", zip_path, e)
            continue

    if not all_frames:
        logger.error("No data loaded from any ZIP archive!")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # ── Concatenate ──
    print("🔗 Concatenating all price data...")
    combined = pd.concat(all_frames, ignore_index=True)

    # ── Clean-up ──
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date", "CompanyCode"])
    combined = combined.sort_values(["CompanyCode", "Date"]).reset_index(
        drop=True
    )

    # Ensure correct column order
    combined = combined[UNIFIED_COLUMNS]

    # ── Save cache ──
    try:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"💾 Cached to {parquet_path.name}")
    except Exception as e:
        logger.warning("Failed to write cache: %s", e)

    print(f"✅ Loaded {len(combined):,} rows, "
          f"{combined['CompanyCode'].nunique():,} companies, "
          f"{combined['Date'].min().date()} – {combined['Date'].max().date()}")

    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Market Indices Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_market_indices(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load market indices from ``07Market Indices - Daily.xls``, sheet
    ``'Index'``.

    Returns a DataFrame with columns:
    ``Date, ASPI, MPI, SP_SL20, <sector index names>...``

    Notes
    -----
    - Rows 0–2 are title rows, rows 3–4 contain the multi-level header.
    - Row 3 has sector index names, row 4 has ASPI/MPI labels.
    - Data starts from 1985-01-02.
    """
    base = _resolve_base(base_path)
    filepath = base / "07Market Indices - Daily.xls"

    if not filepath.exists():
        raise FileNotFoundError(f"Market indices file not found: {filepath}")

    try:
        # Read with multi-level header at rows 3 and 4
        df_raw = pd.read_excel(
            filepath,
            sheet_name="Index",
            header=None,
            engine="xlrd",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read market indices: {e}") from e

    # ── Build headers from rows 3 and 4 ──
    row3 = df_raw.iloc[3].fillna("").astype(str).str.strip()
    row4 = df_raw.iloc[4].fillna("").astype(str).str.strip()

    headers = []
    for i in range(len(row3)):
        r3 = row3.iloc[i] if i < len(row3) else ""
        r4 = row4.iloc[i] if i < len(row4) else ""

        if i == 0:
            headers.append("Date")
        elif r4 and r4 not in ("", "nan"):
            # Row 4 has labels for ASPI, MPI
            label = r4.strip()
            if "all share" in label.lower():
                headers.append("ASPI")
            elif "milanka" in label.lower():
                headers.append("MPI")
            else:
                headers.append(label)
        elif r3 and r3 not in ("", "nan"):
            # Row 3 has sector names and SP_SL20
            label = r3.strip()
            if "s&p" in label.lower() and "20" in label:
                headers.append("SP_SL20")
            else:
                headers.append(label)
        else:
            headers.append(f"Unknown_{i}")

    # ── Extract data rows (skip first 5 rows: 0–4) ──
    df_data = df_raw.iloc[5:].copy()
    df_data.columns = headers[:df_data.shape[1]]
    df_data = df_data.reset_index(drop=True)

    # ── Parse date column ──
    df_data["Date"] = pd.to_datetime(df_data["Date"], errors="coerce")
    df_data = df_data.dropna(subset=["Date"])

    # ── Convert numeric columns ──
    for col in df_data.columns:
        if col != "Date":
            df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

    df_data = df_data.sort_values("Date").reset_index(drop=True)

    logger.info(
        "Loaded market indices: %d rows, %d columns, %s – %s",
        len(df_data), df_data.shape[1],
        df_data["Date"].min().date(), df_data["Date"].max().date(),
    )

    return df_data


# ══════════════════════════════════════════════════════════════════════════════
# Market Statistics Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_market_stats(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load market statistics from ``11Market Statistics-Daily.xls``,
    sheet ``'stats'``.

    Returns a DataFrame with columns:
    ``Date, ASPI, MPI, SP_SL20, TURNOVER_EQUITY_Mn,
    TURNOVER_CORP_DEBT_000, TURNOVER_GOVN_DEBT_000,
    TURNOVER_FUNDS_000, SHARES_TRADED_EQUITY_000,
    TRADES_EQUITY, MARKET_CAP_Mn``

    Notes
    -----
    - Skip first 5 rows; headers at rows 3–4.
    - Data starts from 1994-01-03.
    """
    base = _resolve_base(base_path)
    filepath = base / "11Market Statistics-Daily.xls"

    if not filepath.exists():
        raise FileNotFoundError(f"Market stats file not found: {filepath}")

    try:
        df_raw = pd.read_excel(
            filepath,
            sheet_name="stats",
            header=None,
            engine="xlrd",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read market statistics: {e}") from e

    # ── Build headers from rows 3–4 ──
    row3 = df_raw.iloc[3].fillna("").astype(str).str.strip()
    row4 = df_raw.iloc[4].fillna("").astype(str).str.strip()

    # Combine rows 3 and 4 intelligently
    headers = []
    for i in range(len(row3)):
        r3 = row3.iloc[i] if i < len(row3) else ""
        r4 = row4.iloc[i] if i < len(row4) else ""

        if i == 0:
            headers.append("Date")
        else:
            # Combine both rows for a descriptive header
            combined = " ".join([r3, r4]).strip()
            if not combined or combined == "nan nan" or combined == "nan":
                headers.append(f"Unknown_{i}")
            else:
                headers.append(combined)

    # ── Extract data (skip rows 0–4) ──
    df_data = df_raw.iloc[5:].copy()
    df_data.columns = headers[:df_data.shape[1]]
    df_data = df_data.reset_index(drop=True)

    # ── Clean up column names ──
    rename_map = {}
    for col in df_data.columns:
        clean = col
        if col == "Date":
            continue

        lower = col.lower()
        if "all share" in lower:
            clean = "ASPI"
        elif "milanka" in lower:
            clean = "MPI"
        elif "s&p" in lower or "sp" in lower:
            clean = "SP_SL20"
        elif "turnover" in lower and "equity" in lower:
            clean = "TURNOVER_EQUITY_Mn"
        elif "turnover" in lower and "corp" in lower:
            clean = "TURNOVER_CORP_DEBT_000"
        elif "turnover" in lower and "gov" in lower:
            clean = "TURNOVER_GOVN_DEBT_000"
        elif "turnover" in lower and "fund" in lower:
            clean = "TURNOVER_FUNDS_000"
        elif "shares traded" in lower or "share" in lower and "traded" in lower:
            clean = "SHARES_TRADED_EQUITY_000"
        elif "trades" in lower and "equity" in lower:
            clean = "TRADES_EQUITY"
        elif "market cap" in lower:
            clean = "MARKET_CAP_Mn"
        else:
            # Sanitize: replace spaces and special chars
            clean = (
                col.replace(" ", "_")
                .replace("'", "")
                .replace(".", "")
                .replace(",", "")
                .replace("-", "_")
            )

        rename_map[col] = clean

    df_data = df_data.rename(columns=rename_map)

    # ── Parse date and numeric columns ──
    df_data["Date"] = pd.to_datetime(df_data["Date"], errors="coerce")
    df_data = df_data.dropna(subset=["Date"])

    for col in df_data.columns:
        if col != "Date":
            df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

    df_data = df_data.sort_values("Date").reset_index(drop=True)

    logger.info(
        "Loaded market stats: %d rows, %s – %s",
        len(df_data),
        df_data["Date"].min().date(), df_data["Date"].max().date(),
    )

    return df_data


# ══════════════════════════════════════════════════════════════════════════════
# Securities List Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_securities_list(
    base_path: Optional[str] = None,
    year: str = "2025",
    equity_only: bool = True,
) -> pd.DataFrame:
    """
    Load the list of quoted securities from
    ``16 List of Quoted Securities And Issued Quantity.xls``.

    Parameters
    ----------
    base_path : str or None
        Dataset directory path.
    year : str, default '2025'
        Sheet name (each sheet corresponds to a year).
    equity_only : bool, default True
        If True, filter to N-type (equity) securities only,
        excluding bonds (BD) and warrants (W).

    Returns
    -------
    pd.DataFrame
        Columns: ``SecurityID, Name, IssuedQuantity, ISINCode``
    """
    base = _resolve_base(base_path)
    filepath = base / "16 List of Quoted Securities And Issued Quantity.xls"

    if not filepath.exists():
        raise FileNotFoundError(f"Securities list not found: {filepath}")

    try:
        df = pd.read_excel(
            filepath,
            sheet_name=str(year),
            header=None,
            skiprows=4,  # Rows 0–3 are titles; row 3 is the header
            engine="xlrd",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to read securities list (sheet {year}): {e}"
        ) from e

    # ── Assign column names ──
    expected_cols = ["SecurityID", "Name", "IssuedQuantity", "ISINCode"]
    if df.shape[1] >= 4:
        df.columns = expected_cols + [f"Extra_{i}" for i in range(df.shape[1] - 4)]
    else:
        df.columns = expected_cols[:df.shape[1]]

    # ── Clean ──
    df = df.dropna(subset=["SecurityID"])
    df["SecurityID"] = df["SecurityID"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip() if "Name" in df.columns else np.nan
    df["IssuedQuantity"] = pd.to_numeric(
        df.get("IssuedQuantity"), errors="coerce"
    )
    if "ISINCode" in df.columns:
        df["ISINCode"] = df["ISINCode"].astype(str).str.strip()

    # ── Filter to equity (N-type) securities ──
    if equity_only:
        # SecurityID format: 'AAF-N-0000' → middle part is the type
        mask = df["SecurityID"].str.contains(r"-N-", case=False, na=False)
        df = df.loc[mask].copy()

    df = df.reset_index(drop=True)
    logger.info("Loaded %d securities for year %s", len(df), year)

    return df[expected_cols[:df.shape[1]]]


# ══════════════════════════════════════════════════════════════════════════════
# Stock Splits Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_splits(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load stock split (sub-division) data from
    ``05Sub Division (Share Splits).xls``.

    Returns
    -------
    pd.DataFrame
        Columns: ``DateAnnounced, CompanyID, CompanyName,
        OldProportion, NewProportion, EffectiveDate, SheetYear``
    """
    base = _resolve_base(base_path)
    filepath = base / "05Sub Division (Share Splits).xls"

    if not filepath.exists():
        raise FileNotFoundError(f"Splits file not found: {filepath}")

    try:
        xls = pd.ExcelFile(filepath, engine="xlrd")
    except Exception as e:
        raise RuntimeError(f"Failed to open splits file: {e}") from e

    all_frames = []

    for sheet in tqdm(xls.sheet_names, desc="Loading splits"):
        try:
            df = pd.read_excel(
                xls, sheet_name=sheet,
                header=None,
                skiprows=3,  # Skip rows 0–2; row 2 is header
            )

            if df.empty or df.shape[1] < 7:
                continue

            df.columns = [
                "No", "DateAnnounced", "CompanyID", "CompanyName",
                "OldProportion", "NewProportion", "EffectiveDate",
            ] + [f"Extra_{i}" for i in range(max(0, df.shape[1] - 7))]

            # Drop the sequence number column
            df = df.drop(columns=["No"], errors="ignore")

            # Drop rows that are all NaN or contain header text
            df = df.dropna(how="all")
            df = df[
                ~df["CompanyID"].astype(str).str.contains(
                    "COMPANY", case=False, na=False
                )
            ]

            # Parse dates
            df["DateAnnounced"] = pd.to_datetime(
                df["DateAnnounced"], errors="coerce"
            )
            df["EffectiveDate"] = pd.to_datetime(
                df["EffectiveDate"], errors="coerce"
            )

            # Parse proportions
            df["OldProportion"] = pd.to_numeric(
                df["OldProportion"], errors="coerce"
            )
            df["NewProportion"] = pd.to_numeric(
                df["NewProportion"], errors="coerce"
            )

            df["CompanyID"] = df["CompanyID"].astype(str).str.strip()
            df["CompanyName"] = df["CompanyName"].astype(str).str.strip()
            df["SheetYear"] = sheet

            keep_cols = [
                "DateAnnounced", "CompanyID", "CompanyName",
                "OldProportion", "NewProportion", "EffectiveDate",
                "SheetYear",
            ]
            all_frames.append(df[keep_cols])

        except Exception as e:
            logger.warning("Failed to parse splits sheet '%s': %s", sheet, e)
            continue

    if not all_frames:
        logger.warning("No splits data loaded")
        return pd.DataFrame(columns=[
            "DateAnnounced", "CompanyID", "CompanyName",
            "OldProportion", "NewProportion", "EffectiveDate", "SheetYear",
        ])

    result = pd.concat(all_frames, ignore_index=True)
    result = result.dropna(subset=["CompanyID"])
    result = result.sort_values("EffectiveDate").reset_index(drop=True)

    logger.info("Loaded %d stock splits", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# New Listings / De-listings Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_listings(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load new listings and de-listings from
    ``06New Listings & De-listings.xls``.

    This file has a complex structure with multiple sections per sheet
    (IPO-Equity, Introduction-Equity, De-listings, etc.).

    Returns
    -------
    pd.DataFrame
        Columns: ``SheetYear, Section, CompanyName, ListingDate,
        ExtraInfo``

    Notes
    -----
    Due to the complex, inconsistent structure across sheets, this
    loader performs best-effort extraction.  Some sections may have
    additional columns that are captured in ``ExtraInfo``.
    """
    base = _resolve_base(base_path)
    filepath = base / "06New Listings & De-listings.xls"

    if not filepath.exists():
        raise FileNotFoundError(f"Listings file not found: {filepath}")

    try:
        xls = pd.ExcelFile(filepath, engine="xlrd")
    except Exception as e:
        raise RuntimeError(f"Failed to open listings file: {e}") from e

    all_rows = []

    for sheet in tqdm(xls.sheet_names, desc="Loading listings"):
        try:
            df = pd.read_excel(
                xls, sheet_name=sheet,
                header=None,
                engine="xlrd",
            )

            if df.empty:
                continue

            current_section = "Unknown"

            for idx, row in df.iterrows():
                row_vals = [
                    str(v).strip() if pd.notna(v) else ""
                    for v in row.values
                ]
                row_text = " ".join(row_vals).strip()

                # Detect section headers
                if any(keyword in row_text.upper() for keyword in [
                    "IPO", "INTRODUCTION", "DE-LISTING", "DELISTING",
                    "DE LISTING", "NEW LISTING",
                ]):
                    current_section = row_text.strip()
                    continue

                # Skip empty or header-like rows
                if not row_text or row_text.startswith("No."):
                    continue
                if all(c in "-= " for c in row_text):
                    continue

                # Try to find a date in the row
                listing_date = None
                company_name = ""
                extra_parts = []

                for val in row.values:
                    if pd.isna(val):
                        continue
                    # Try to parse as date
                    if listing_date is None:
                        try:
                            candidate = pd.to_datetime(val, errors="coerce")
                            if pd.notna(candidate):
                                listing_date = candidate
                                continue
                        except Exception:
                            pass
                    # Try as company name
                    val_str = str(val).strip()
                    if val_str and val_str not in ("nan", ""):
                        if not company_name and not val_str.isdigit():
                            company_name = val_str
                        else:
                            extra_parts.append(val_str)

                if company_name:
                    all_rows.append({
                        "SheetYear": sheet,
                        "Section": current_section,
                        "CompanyName": company_name,
                        "ListingDate": listing_date,
                        "ExtraInfo": "; ".join(extra_parts) if extra_parts else np.nan,
                    })

        except Exception as e:
            logger.warning("Failed to parse listings sheet '%s': %s", sheet, e)
            continue

    if not all_rows:
        logger.warning("No listings data loaded")
        return pd.DataFrame(
            columns=["SheetYear", "Section", "CompanyName",
                      "ListingDate", "ExtraInfo"]
        )

    result = pd.DataFrame(all_rows)
    result = result.sort_values(
        ["ListingDate", "SheetYear"], na_position="last"
    ).reset_index(drop=True)

    logger.info("Loaded %d listing/de-listing records", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GICS Sector Indices Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_gics(
    base_path: Optional[str] = None,
    sheet_name: str = "Sectors Index",
) -> pd.DataFrame:
    """
    Load GICS sector data from ``39GICS-Daily.xlsx``.

    Parameters
    ----------
    base_path : str or None
        Dataset directory.
    sheet_name : str, default 'Sectors Index'
        Sheet to load.  Options: ``'Sectors Index'``, ``'PER'``,
        ``'PBV'``, ``'DY'``, ``'Market Cap'``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Date, <sector1>, <sector2>, ...``
        (22 columns total for 'Sectors Index': Date + 21 sectors)

    Notes
    -----
    Data available from 2016-04-05 onward.
    """
    base = _resolve_base(base_path)
    filepath = base / "39GICS-Daily.xlsx"

    if not filepath.exists():
        raise FileNotFoundError(f"GICS file not found: {filepath}")

    try:
        df = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=0,
            engine="openpyxl",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to read GICS sheet '{sheet_name}': {e}"
        ) from e

    # ── Rename first column to 'Date' ──
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Date"})

    # ── Parse date ──
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # ── Convert sector columns to numeric ──
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(
        "Loaded GICS '%s': %d rows, %d sectors, %s – %s",
        sheet_name, len(df), df.shape[1] - 1,
        df["Date"].min().date(), df["Date"].max().date(),
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Sector Market Capitalisation Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_sector_market_cap(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load sector market capitalisation from
    ``22Sector Market Capitalisation.xls``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Year, Sector, January, February, ..., December``
        (melted to long format: ``Year, Sector, Month, MarketCap``)
    """
    base = _resolve_base(base_path)
    filepath = base / "22Sector Market Capitalisation.xls"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Sector market cap file not found: {filepath}"
        )

    try:
        xls = pd.ExcelFile(filepath, engine="xlrd")
    except Exception as e:
        raise RuntimeError(
            f"Failed to open sector market cap file: {e}"
        ) from e

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    all_frames = []

    for sheet in tqdm(xls.sheet_names, desc="Loading sector market cap"):
        try:
            df = pd.read_excel(
                xls, sheet_name=sheet,
                header=None,
                skiprows=3,  # Rows 0–2 are titles; row 2 is header
                engine="xlrd",
            )

            if df.empty or df.shape[1] < 3:
                continue

            # Expect columns: index/NaN, SECTOR, Jan, Feb, ..., Dec
            # Assign column names based on expected structure
            n_cols = df.shape[1]
            if n_cols >= 14:
                col_names = ["_idx", "Sector"] + months + [
                    f"Extra_{i}" for i in range(n_cols - 14)
                ]
            elif n_cols >= 2:
                col_names = ["_idx", "Sector"] + months[:n_cols - 2]
            else:
                continue

            df.columns = col_names[:n_cols]

            # Drop index column
            df = df.drop(columns=["_idx"], errors="ignore")

            # Remove rows without a sector name
            df = df.dropna(subset=["Sector"])
            df["Sector"] = df["Sector"].astype(str).str.strip()
            df = df[
                ~df["Sector"].str.contains(
                    "SECTOR|TOTAL|^$|^nan$",
                    case=False, na=False, regex=True,
                )
            ]

            if df.empty:
                continue

            # Melt to long format
            month_cols = [c for c in df.columns if c in months]
            df_long = df.melt(
                id_vars=["Sector"],
                value_vars=month_cols,
                var_name="Month",
                value_name="MarketCap",
            )
            df_long["Year"] = sheet
            df_long["MarketCap"] = pd.to_numeric(
                df_long["MarketCap"], errors="coerce"
            )

            all_frames.append(df_long)

        except Exception as e:
            logger.warning(
                "Failed to parse sector market cap sheet '%s': %s", sheet, e,
            )
            continue

    if not all_frames:
        logger.warning("No sector market cap data loaded")
        return pd.DataFrame(
            columns=["Year", "Sector", "Month", "MarketCap"]
        )

    result = pd.concat(all_frames, ignore_index=True)
    result = result[["Year", "Sector", "Month", "MarketCap"]]

    logger.info("Loaded %d sector market cap records", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Dividends Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_dividends(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load dividend data from ``01Dividends.xls``.

    Returns
    -------
    pd.DataFrame
        Columns: ``DateAnnounced, SecurityID, ShortName,
        DividendRate, Remarks, ExDividendDate,
        PaymentDate, CumPrice, ExPrice, SheetYear``
    """
    base = _resolve_base(base_path)
    filepath = base / "01Dividends.xls"

    if not filepath.exists():
        raise FileNotFoundError(f"Dividends file not found: {filepath}")

    try:
        xls = pd.ExcelFile(filepath, engine="xlrd")
    except Exception as e:
        raise RuntimeError(f"Failed to open dividends file: {e}") from e

    all_frames = []

    for sheet in tqdm(xls.sheet_names, desc="Loading dividends"):
        try:
            df = pd.read_excel(
                xls, sheet_name=sheet,
                header=None,
                skiprows=4,  # Rows 0–3 are titles; row 3 is header
                engine="xlrd",
            )

            if df.empty:
                continue

            # Expected columns (9):
            expected = [
                "DateAnnounced", "SecurityID", "ShortName",
                "DividendRate", "Remarks", "ExDividendDate",
                "PaymentDate", "CumPrice", "ExPrice",
            ]

            if df.shape[1] >= 9:
                df.columns = expected + [
                    f"Extra_{i}" for i in range(df.shape[1] - 9)
                ]
            else:
                df.columns = expected[:df.shape[1]]

            # Clean up
            df = df.dropna(how="all")

            # Remove header-like rows
            if "SecurityID" in df.columns:
                df = df[
                    ~df["SecurityID"].astype(str).str.contains(
                        "SECURITY|DATE|ANNOUNCEMENT",
                        case=False, na=False,
                    )
                ]

            # Parse dates
            for date_col in ["DateAnnounced", "ExDividendDate", "PaymentDate"]:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(
                        df[date_col], errors="coerce"
                    )

            # Parse numeric columns
            for num_col in ["DividendRate", "CumPrice", "ExPrice"]:
                if num_col in df.columns:
                    df[num_col] = pd.to_numeric(
                        df[num_col], errors="coerce"
                    )

            # Clean strings
            for str_col in ["SecurityID", "ShortName", "Remarks"]:
                if str_col in df.columns:
                    df[str_col] = df[str_col].astype(str).str.strip()

            df["SheetYear"] = sheet

            keep_cols = [c for c in expected + ["SheetYear"] if c in df.columns]
            all_frames.append(df[keep_cols])

        except Exception as e:
            logger.warning(
                "Failed to parse dividends sheet '%s': %s", sheet, e,
            )
            continue

    if not all_frames:
        logger.warning("No dividend data loaded")
        return pd.DataFrame(columns=[
            "DateAnnounced", "SecurityID", "ShortName",
            "DividendRate", "Remarks", "ExDividendDate",
            "PaymentDate", "CumPrice", "ExPrice", "SheetYear",
        ])

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values("DateAnnounced", na_position="last")
    result = result.reset_index(drop=True)

    logger.info("Loaded %d dividend records", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sector Ratios Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_sector_ratios(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load sector ratios from ``25Sector Ratios.xls``.

    This file has a complex pivot format with sectors as rows and
    months with sub-columns (Dividend Yield, P/E Ratio, Price to
    Book Value).

    Returns
    -------
    pd.DataFrame
        Columns: ``Year, Sector, Month, DividendYield, PERatio,
        PriceToBookValue``
    """
    base = _resolve_base(base_path)
    filepath = base / "25Sector Ratios.xls"

    if not filepath.exists():
        raise FileNotFoundError(f"Sector ratios file not found: {filepath}")

    try:
        xls = pd.ExcelFile(filepath, engine="xlrd")
    except Exception as e:
        raise RuntimeError(f"Failed to open sector ratios file: {e}") from e

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    all_rows = []

    for sheet in tqdm(xls.sheet_names, desc="Loading sector ratios"):
        try:
            df = pd.read_excel(
                xls, sheet_name=sheet,
                header=None,
                engine="xlrd",
            )

            if df.empty:
                continue

            # The structure is complex: each month has 3 sub-columns
            # (DY, PE, PBV), preceded by sector name column.
            # Try to detect sectors and extract values.

            # Find the row with month names (typically row 3 or 4)
            month_row_idx = None
            for i in range(min(10, len(df))):
                row_vals = df.iloc[i].astype(str).str.strip()
                month_count = sum(
                    1 for v in row_vals
                    if v.lower() in [m.lower() for m in months]
                )
                if month_count >= 3:
                    month_row_idx = i
                    break

            if month_row_idx is None:
                logger.debug(
                    "Could not find month header in sector ratios sheet '%s'",
                    sheet,
                )
                continue

            # Find the sub-column labels row (DY, PE, PBV) — usually the next row
            sub_row_idx = month_row_idx + 1
            if sub_row_idx >= len(df):
                continue

            month_headers = df.iloc[month_row_idx].astype(str).str.strip()
            sub_headers = df.iloc[sub_row_idx].astype(str).str.strip()

            # Map column index to (month, metric)
            col_map = {}
            current_month = None
            for col_idx in range(df.shape[1]):
                mh = month_headers.iloc[col_idx] if col_idx < len(month_headers) else ""
                sh = sub_headers.iloc[col_idx] if col_idx < len(sub_headers) else ""

                # Check if this is a month column
                if mh.lower() in [m.lower() for m in months]:
                    current_month = mh.title()

                if current_month and sh:
                    sh_lower = sh.lower()
                    if "dividend" in sh_lower or "dy" in sh_lower:
                        col_map[col_idx] = (current_month, "DividendYield")
                    elif "p/e" in sh_lower or "pe" in sh_lower or "price" in sh_lower and "earn" in sh_lower:
                        col_map[col_idx] = (current_month, "PERatio")
                    elif "book" in sh_lower or "pbv" in sh_lower:
                        col_map[col_idx] = (current_month, "PriceToBookValue")

            if not col_map:
                continue

            # Data starts after the sub-header row
            data_start = sub_row_idx + 1
            sector_col = None

            # Find the sector column (usually col 0 or col 1)
            for ci in range(min(3, df.shape[1])):
                test_vals = df.iloc[data_start:, ci].dropna().astype(str)
                if len(test_vals) > 0 and any(
                    len(v) > 3 and not v.replace(".", "").isdigit()
                    for v in test_vals.head(10)
                ):
                    sector_col = ci
                    break

            if sector_col is None:
                sector_col = 0

            # Extract data rows
            for row_idx in range(data_start, len(df)):
                sector = df.iloc[row_idx, sector_col]
                if pd.isna(sector):
                    continue
                sector = str(sector).strip()
                if not sector or sector.lower() in ("nan", "total", ""):
                    continue
                if "---" in sector or "===" in sector:
                    continue

                for col_idx, (month, metric) in col_map.items():
                    val = df.iloc[row_idx, col_idx]
                    val_num = pd.to_numeric(val, errors="coerce")

                    # Find or create the row for this sector+month
                    row_key = (sheet, sector, month)
                    matching = [
                        r for r in all_rows
                        if (r["Year"], r["Sector"], r["Month"]) == row_key
                    ]
                    if matching:
                        matching[0][metric] = val_num
                    else:
                        new_row = {
                            "Year": sheet,
                            "Sector": sector,
                            "Month": month,
                            "DividendYield": np.nan,
                            "PERatio": np.nan,
                            "PriceToBookValue": np.nan,
                        }
                        new_row[metric] = val_num
                        all_rows.append(new_row)

        except Exception as e:
            logger.warning(
                "Failed to parse sector ratios sheet '%s': %s", sheet, e,
            )
            continue

    if not all_rows:
        logger.warning("No sector ratio data loaded")
        return pd.DataFrame(columns=[
            "Year", "Sector", "Month",
            "DividendYield", "PERatio", "PriceToBookValue",
        ])

    result = pd.DataFrame(all_rows)
    result = result[
        ["Year", "Sector", "Month",
         "DividendYield", "PERatio", "PriceToBookValue"]
    ]

    logger.info("Loaded %d sector ratio records", len(result))
    return result
