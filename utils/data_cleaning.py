import pandas as pd
import numpy as np

def _dedup_key_cols(df: pd.DataFrame) -> list[str]:
    """Use full security-class key when available so N/X/R/W classes are never collapsed."""
    if "SecurityID" in df.columns:
        return ["SecurityID"]
    if {"CompanyCode", "MainType", "SubType"}.issubset(df.columns):
        return ["CompanyCode", "MainType", "SubType"]
    return ["CompanyCode"]


def clean_and_prepare_data(df, df_splits=None, start_year=2001):
    """
    Clean and prepare the raw daily prices dataframe while preserving distinct
    security classes (N, X, R, W, P, D, B, U) and historical delisted securities.
    """
    df_clean = df[df["Date"].dt.year >= start_year].copy()
    key_cols = _dedup_key_cols(df_clean)

    # Deduplicate only within the SAME security class + date
    df_clean = df_clean.sort_values(by="ShareVolume", ascending=False, na_position="last")
    df_clean = df_clean.drop_duplicates(subset=key_cols + ["Date"], keep="first")
    df_clean = df_clean.sort_values(key_cols + ["Date"]).reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close"):
        df_clean[col] = df_clean[col].replace(0, np.nan)

    df_clean = align_trading_calendar(df_clean)

    if df_splits is not None and not df_splits.empty:
        df_clean = apply_split_adjustments(df_clean, df_splits)

    df_clean["DailyReturn_Pct"] = df_clean.groupby(key_cols)["Close"].pct_change() * 100
    return df_clean


def align_trading_calendar(df, max_fill_days=5):
    """Create a per-security trading calendar without merging separate share classes."""
    market_dates = pd.Series(df["Date"].unique()).sort_values().reset_index(drop=True)
    key_cols = _dedup_key_cols(df)
    clean_dfs = []

    for _, group in df.groupby(key_cols, sort=False):
        min_date = group["Date"].min()
        max_date = group["Date"].max()
        valid_dates = market_dates[(market_dates >= min_date) & (market_dates <= max_date)]

        group = group.set_index("Date").reindex(valid_dates)
        price_cols = ["Open", "High", "Low", "Close"]
        group[price_cols] = group[price_cols].ffill(limit=max_fill_days)

        vol_cols = [c for c in ("TradeVolume", "ShareVolume", "Turnover") if c in group.columns]
        group[vol_cols] = group[vol_cols].fillna(0)

        meta_cols = [
            c for c in ("CompanyCode", "SecurityID", "ShortName", "MainType", "SubType", "Era")
            if c in group.columns
        ]
        group[meta_cols] = group[meta_cols].ffill().bfill()
        group = group.reset_index().rename(columns={"index": "Date"})
        clean_dfs.append(group)

    return pd.concat(clean_dfs, ignore_index=True)


def apply_split_adjustments(df, df_splits):
    """
    Adjust historical prices backward for stock splits while preventing double-adjustment.
    Supports both SecurityID-level and CompanyCode-level split records.
    """
    if "IsSplitAdjusted" in df.columns and df["IsSplitAdjusted"].fillna(False).any():
        raise ValueError(
            "Input DataFrame is already split-adjusted (IsSplitAdjusted=True); "
            "refusing to apply double split adjustment."
        )

    df_adjusted = df.copy()
    splits = df_splits.copy()
    splits.columns = [str(c).upper() for c in splits.columns]
    splits = splits.rename(columns={
        "COMPANYID": "COMPANY ID",
        "COMPANYCODE": "COMPANY ID",
        "EFFECTIVEDATE": "EFFECTIVE DATE",
        "EXDATE": "EFFECTIVE DATE",
        "OLDPROPORTION": "OLD PROPORTION",
        "NEWPROPORTION": "NEW PROPORTION",
    })

    if "EFFECTIVE DATE" not in splits.columns:
        return df_adjusted

    splits["EFFECTIVE DATE"] = pd.to_datetime(splits["EFFECTIVE DATE"], errors="coerce")
    splits = splits.dropna(subset=["EFFECTIVE DATE", "OLD PROPORTION", "NEW PROPORTION"])
    splits["SplitRatio"] = (
        pd.to_numeric(splits["NEW PROPORTION"], errors="coerce")
        / pd.to_numeric(splits["OLD PROPORTION"], errors="coerce")
    )
    splits["CleanCompany"] = splits["COMPANY ID"].astype(str).str.split(".").str[0].str.strip().str.upper()

    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in df_adjusted.columns]
    vol_cols   = [c for c in ("TradeVolume", "ShareVolume") if c in df_adjusted.columns]

    splits = splits.sort_values("EFFECTIVE DATE", ascending=False)
    applied_keys = set()

    for _, row in splits.iterrows():
        eff_date = row["EFFECTIVE DATE"]
        ratio = row["SplitRatio"]
        if pd.isna(ratio) or ratio <= 0 or ratio == 1.0:
            continue

        sec_id = str(row.get("SECURITYID", "")).strip().upper()
        company = row["CleanCompany"]
        event_key = (sec_id or company, eff_date.strftime("%Y-%m-%d"), round(float(ratio), 6))
        if event_key in applied_keys:
            continue
        applied_keys.add(event_key)

        if sec_id and "SecurityID" in df_adjusted.columns:
            mask = (df_adjusted["SecurityID"].str.upper() == sec_id) & (df_adjusted["Date"] < eff_date)
        else:
            mask = (df_adjusted["CompanyCode"].str.upper() == company) & (df_adjusted["Date"] < eff_date)

        if mask.any():
            df_adjusted.loc[mask, price_cols] = df_adjusted.loc[mask, price_cols] / ratio
            df_adjusted.loc[mask, vol_cols]   = df_adjusted.loc[mask, vol_cols] * ratio

    df_adjusted["IsSplitAdjusted"] = True
    return df_adjusted
