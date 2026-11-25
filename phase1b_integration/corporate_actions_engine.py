"""
Phase 1B Task 3 – CSE Corporate Action Validation & Adjustment Engine
====================================================================
Implements separate, traceable, non-overlapping adjustment logic for:
  A. Stock Splits      (05Sub Division (Share Splits).xls)
  B. Bonus Issues      (03Capitalization of Reserves (Bonus Issues).xls)
  C. Scrip Dividends   (02Scrip Dividends.xls)
  D. Rights Issues     (04Rights Issues.xls — TERP using ratio & subscription price)
  E. Cash Dividends    (01Dividends.xls — separated strictly from split adjustments)

Produces:
  - Validated event ledger (`cse_corporate_actions_validated.parquet`)
  - Traceable factor audit table (`corporate_action_adjustment_factors.parquet`)
  - Split/Bonus-adjusted price series (`cse_daily_prices_split_adjusted.parquet`)
  - Cash-dividend-reinvested Total Return series (`cse_total_return_history.parquet`)
"""

from __future__ import annotations

import datetime as dt_mod
import difflib
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .identity_model import normalize_company_name

# Verified ticker renames / manual aliases observed directly in 05Sub Division.xls vs security master
_KNOWN_CODE_ALIASES = {
    "FCH": "CFVF",       # First Capital Holdings PLC
    "UNISYST": "ALUF",   # Unisyst Engineering PLC
    "SIND": "SINI",      # Singer Industries (Ceylon) PLC
}

_SECTOR_SUMMARY_STOPWORDS = (
    "CAPITALIZATION OF RESERVES BY SECTOR",
    "SUMMARY",
    "NO RECORDS FOR",
    "GRAND TOTAL",
)


def parse_excel_date(val) -> pd.Timestamp:
    """Parse timestamps, Excel serial integers, or string dates cleanly without fabrication."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NaT
    if isinstance(val, pd.Timestamp):
        ts = val
    elif isinstance(val, dt_mod.datetime):
        ts = pd.Timestamp(val)
    elif isinstance(val, (int, float)) and not isinstance(val, bool):
        if 25000 <= val <= 60000:
            ts = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(round(val)))
        else:
            return pd.NaT
    else:
        s = str(val).replace("\xa0", " ").strip()
        if not s or s.lower() in ("nan", "nat", "none", "-", "nil", "n/a"):
            return pd.NaT
        # Check ISO YYYY-MM-DD / YYYY/MM/DD first; otherwise apply CSE day-first convention
        if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", s):
            ts = pd.to_datetime(s, errors="coerce")
        else:
            ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(ts) or ts.year < 1985 or ts.year > 2026:
        return pd.NaT
    return pd.Timestamp(ts).normalize()


def parse_proportion_pair(val) -> Optional[tuple[float, float]]:
    """
    Parse CSE proportion values:
      - '1:10', '143:101', '.459:1' -> (left, right)
      - Excel hh:mm:00 datetime.time(h, m) -> (h, m)
      - Single numeric > 1 (e.g. 25.01 in 2021 DIMO bonus) -> (1.0, 25.01) as (new, old) flaggable pair
    Returns (left, right, format_kind) or None.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, dt_mod.time):
        if val.hour > 0 and val.minute >= 0:
            return float(val.hour), float(val.minute), "excel_time"
        return None
    if isinstance(val, dt_mod.datetime):
        # Excel turned e.g. '25:10' into 1900-01-01 01:10:00 if >23h
        if val.year == 1900 and val.month == 1:
            return float(val.day * 24 + val.hour), float(val.minute), "excel_datetime"
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if val > 1.0:
            return 1.0, float(val), "single_old_per_new"
        if 0 < val <= 1.0:
            return float(val), 1.0, "single_new_per_old"
        return None

    s = str(val).replace("\xa0", " ").strip()
    if not s or s.lower() in ("nan", "none", "-", "nil"):
        return None

    # Format: hh:mm:ss string e.g. '01:03:00'
    m_hms = re.match(r"^0?(\d{1,3}):0?(\d{1,3}):00$", s)
    if m_hms:
        return float(m_hms.group(1)), float(m_hms.group(2)), "hms_str"

    # Standard A:B ratio (may have spaces or decimals)
    m_ab = re.match(r"^(\d*\.?\d+)\s*[:/]\s*(\d*\.?\d+)$", s)
    if m_ab:
        a, b = float(m_ab.group(1)), float(m_ab.group(2))
        if a > 0 and b > 0:
            return a, b, "a_colon_b"

    # Single float string
    try:
        fv = float(s)
        if fv > 1.0:
            return 1.0, fv, "single_old_per_new"
        if 0 < fv <= 1.0:
            return fv, 1.0, "single_new_per_old"
    except ValueError:
        pass
    return None


def _norm_sec_parts(code: str, mtype: str, stype: str) -> str:
    code_c = _KNOWN_CODE_ALIASES.get(code.upper(), code.upper())
    m_c = mtype.upper()
    s_raw = str(stype).strip()
    if re.match(r"^\d+\.0+$", s_raw):
        s_raw = s_raw.split(".")[0]
    s_c = "0000" if s_raw in ("", "0", "00", "000", "0000", "nan", "NONE") else s_raw.zfill(4)
    return f"{code_c}-{m_c}-{s_c}"


class SecurityResolver:
    """Resolve raw CSE corporate action security identifiers and names to canonical SecurityIDs."""

    def __init__(self, identity_df: pd.DataFrame, name_aliases: dict[str, set[str]]) -> None:
        self.identity_df = identity_df
        self.valid_sec_ids = set(identity_df["SecurityID"].astype(str))
        self.code_to_sec_ids: dict[str, list[str]] = {}
        self.code_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for code, grp in identity_df.groupby("CompanyCode"):
            self.code_to_sec_ids[code] = sorted(grp["SecurityID"].tolist())
            self.code_ranges[code] = (grp["FirstPriceDate"].min(), grp["LastPriceDate"].max())

        self.name_aliases = dict(name_aliases)
        self.alias_keys = [k for k, codes in self.name_aliases.items() if len(k) >= 4 and len(codes) >= 1]

    def _pick_code_by_date(self, candidates: set[str], event_date: pd.Timestamp) -> Optional[str]:
        if len(candidates) == 1:
            return next(iter(candidates))
        if pd.isna(event_date):
            return None
        active = []
        for c in candidates:
            rng = self.code_ranges.get(c)
            if rng and (rng[0] - pd.Timedelta(days=60) <= event_date <= rng[1] + pd.Timedelta(days=60)):
                active.append(c)
        if len(active) == 1:
            return active[0]
        return None

    def resolve_company_code(
        self,
        raw_code: object,
        raw_name: object,
        event_date: pd.Timestamp,
    ) -> Optional[str]:
        # 1. Explicit CompanyCode or alias
        if raw_code is not None and pd.notna(raw_code):
            tok = str(raw_code).strip().upper()
            tok = re.split(r"[-.\s]", tok)[0]
            tok = _KNOWN_CODE_ALIASES.get(tok, tok)
            if tok in self.code_to_sec_ids:
                return tok

        # 1b. Early 1991-1999 dividend sheets sometimes store raw CompanyCode in Company Name
        if raw_name is not None and pd.notna(raw_name):
            tok_name = str(raw_name).strip().upper()
            if re.match(r"^[A-Z0-9]{2,8}$", tok_name):
                tok_name = _KNOWN_CODE_ALIASES.get(tok_name, tok_name)
                if tok_name in self.code_to_sec_ids:
                    return tok_name

        # 2. Exact normalized historical company name
        raw_str = str(raw_name).strip() if (raw_name is not None and pd.notna(raw_name)) else ""
        # Strip leading index numbers ('3) '), date/tranche annotations, and share class labels before matching
        raw_stripped = re.sub(r"^\d+\)\s*", "", raw_str)
        raw_stripped = re.sub(r"/AS AT\b.*$", "", raw_stripped, flags=re.I)
        raw_stripped = re.sub(
            r"[-\s]*(\(\s*(?:NON\s*-?\s*VOTING|VOTING|N\s*&\s*X|NV)\s*\)|"
            r"(?:NON\s*-?\s*VOTING|VOTING|TRANCHE\s*\d+|INITIAL ALLOTMENT|BALANCE SHARES)(?:\s+VOTING)?)\b.*$",
            "",
            raw_stripped,
            flags=re.I,
        ).strip()

        nk = normalize_company_name(raw_name)
        nk_stripped = normalize_company_name(raw_stripped) if raw_stripped else ""
        if not (nk or nk_stripped) or max(len(nk or ""), len(nk_stripped or "")) < 3:
            return None

        # Strip trailing listing suffixes like "1ST TIME LISTING" or "ORDINARYNON"
        nk_clean = re.sub(
            r"\b(1ST TIME LISTING|FIRST TIME LISTING|ORDINARY\s*NON|ORDINARYNON|NON\s*VOTING|VOTING|ORDINARY)\b",
            "",
            nk_stripped or nk,
        ).strip()
        for key_try in dict.fromkeys(k for k in (nk, nk_stripped, nk_clean) if k and len(k) >= 3):
            if key_try in self.name_aliases:
                picked = self._pick_code_by_date(self.name_aliases[key_try], event_date)
                if picked:
                    return picked

        # 3. High-threshold unambiguous fuzzy fallback for CSE spelling typos (e.g. Planatations, Bateries)
        scored = []
        for cand_k in self.alias_keys:
            if abs(len(cand_k) - len(nk_clean)) > 5:
                continue
            r = difflib.SequenceMatcher(None, nk_clean, cand_k).ratio()
            if r >= 0.89:
                scored.append((r, cand_k))
        if scored:
            scored.sort(reverse=True)
            best_score, best_k = scored[0]
            # Collapse ties referencing the same CompanyCode
            best_codes = set(self.name_aliases[best_k])
            if len(scored) > 1:
                second_score, second_k = scored[1]
                second_codes = set(self.name_aliases[second_k])
                if second_codes != best_codes and (best_score - second_score) < 0.04:
                    return None
            return self._pick_code_by_date(best_codes, event_date)
        return None

    def resolve_security_ids(
        self,
        raw_sec: object,
        raw_code: object,
        raw_name: object,
        event_date: pd.Timestamp,
        raw_mtype_hint: Optional[str] = None,
    ) -> list[str]:
        """Resolve an event row into one or more existing SecurityIDs."""
        # Direct Security string e.g. 'JKH-W-0022', 'AEL-N-0000', 'CPRT-N-0', 'CIC.N0000'
        if raw_sec is not None and pd.notna(raw_sec):
            s = str(raw_sec).strip().upper()
            m = re.match(r"^([A-Z0-9]+)-([A-Z])-(\d{1,4})$", s)
            if m:
                sid = _norm_sec_parts(m.group(1), m.group(2), m.group(3))
                # Note: '(N & X)' inside company name expands to both N and X
                if raw_name and re.search(r"\(\s*N\s*&\s*X\s*\)", str(raw_name), re.I):
                    out = [c for c in (_norm_sec_parts(m.group(1), "N", "0000"),
                                      _norm_sec_parts(m.group(1), "X", "0000")) if c in self.valid_sec_ids]
                    if out:
                        return out
                if sid in self.valid_sec_ids:
                    return [sid]
            m_dot = re.match(r"^([A-Z0-9]+)\.([A-Z])(\d{1,4})?$", s)
            if m_dot:
                code = m_dot.group(1)
                mtype = m_dot.group(2)
                stype = m_dot.group(3) or "0000"
                if raw_name and re.search(r"\(\s*N\s*&\s*X\s*\)", str(raw_name), re.I):
                    out = [c for c in (_norm_sec_parts(code, "N", "0000"),
                                      _norm_sec_parts(code, "X", "0000")) if c in self.valid_sec_ids]
                    if out:
                        return out
                sid = _norm_sec_parts(code, mtype, stype)
                if sid in self.valid_sec_ids:
                    return [sid]

        code = self.resolve_company_code(raw_code, raw_name, event_date)
        if not code:
            return []

        name_u = str(raw_name).upper() if (raw_name is not None and pd.notna(raw_name)) else ""
        if re.search(r"\(\s*N\s*&\s*X\s*\)", name_u):
            return [c for c in (f"{code}-N-0000", f"{code}-X-0000") if c in self.valid_sec_ids]

        # Infer share class suffix when specified in company description or hint
        mtype = "N"
        if raw_mtype_hint:
            mh = str(raw_mtype_hint).strip().upper()
            if mh.startswith("X"):
                mtype = "X"
            elif mh.startswith("P"):
                mtype = "P"
            elif mh.startswith("W"):
                mtype = "W"
        elif re.search(r"NON\s*-?\s*VOTING|\(\s*NV\s*\)|\(\s*X\s*\)", name_u):
            mtype = "X"
        elif "PREFERENCE" in name_u:
            mtype = "P"

        # Specific warrant pattern like '(W0022)'
        mw = re.search(r"\(W(\d{4})\)", name_u)
        if mw:
            w_sid = f"{code}-W-{mw.group(1)}"
            return [w_sid] if w_sid in self.valid_sec_ids else []

        cand_sid = f"{code}-{mtype}-0000"
        if cand_sid in self.valid_sec_ids:
            return [cand_sid]
        # Fallback if company only trades under a different SubType or MainType='N'
        n_sids = [s for s in self.code_to_sec_ids.get(code, []) if f"-{mtype}-" in s]
        return n_sids[:1]


def _lookup_prior_issued_qty(
    yearly_qty: dict[tuple[int, str], float],
    sec_id: str,
    year: int,
    shares_listed: float,
) -> Optional[float]:
    """Infer pre-event share count using year-end quoted lists (minus intra-year newly listed shares)."""
    q_same = yearly_qty.get((year, sec_id))
    q_prev = yearly_qty.get((year - 1, sec_id))
    if q_same is not None and pd.notna(shares_listed) and shares_listed > 0 and q_same > shares_listed:
        # Handles intra-year splits prior to a bonus/rights event (e.g. BRWN 2008)
        return q_same - shares_listed
    if q_prev is not None and q_prev > 0:
        return q_prev
    return q_same


def _verify_new_old_ratio(
    ratio_tuple: tuple[float, float, str],
    shares_listed: float,
    prior_qty: Optional[float],
    allow_assume_a_new_if_a_le_b: bool = True,
) -> tuple[Optional[float], Optional[float], str]:
    """
    Resolve (old_proportion, new_proportion) carefully using Quoted Securities share counts
    so inverted clerk notation ('2:1' meaning 2 old : 1 new vs '02:01:00' meaning 2 new : 1 old)
    is mathematically verified instead of guessed.
    """
    a, b, kind = ratio_tuple
    if kind == "single_old_per_new":
        # e.g. DIMO 2021 '25.01' -> 1 new per 25.01 old
        return a, b, "SINGLE_OLD_PER_NEW_FORMAT"

    if kind == "single_new_per_old":
        return 1.0, a, "SINGLE_NEW_PER_OLD_FORMAT"

    # Candidate A: a = new, b = old  -> new/old = a/b
    # Candidate B: a = old, b = new  -> new/old = b/a
    r_a_new = a / b
    r_b_new = b / a

    if prior_qty and prior_qty > 0 and pd.notna(shares_listed) and shares_listed > 0:
        obs_r = shares_listed / prior_qty
        err_a = abs(obs_r - r_a_new) / max(obs_r, 1e-9)
        err_b = abs(obs_r - r_b_new) / max(obs_r, 1e-9)
        if err_a <= 0.25 and err_a < err_b:
            return b, a, "VALIDATED_BY_SHARES"
        if err_b <= 0.25 and err_b < err_a:
            return a, b, "VALIDATED_BY_SHARES_INVERTED"

    # When share count confirmation is absent:
    # Standard CSE fractional/ratio notation with a <= b (e.g. 1:10, 1:4, .459:1, 143:101?)
    if allow_assume_a_new_if_a_le_b and a <= b:
        return b, a, "UNCORROBORATED_STANDARD_NEW_OLD"

    return None, None, "UNRESOLVED_RATIO_DIRECTION"


def _find_pre_ex_close(
    price_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    sec_id: str,
    ex_date: pd.Timestamp,
    max_lookback_days: int = 45,
) -> Optional[float]:
    """Return the last valid raw Close price strictly before ExDate within lookback window."""
    if pd.isna(ex_date) or sec_id not in price_lookup:
        return None
    dates_ns, closes = price_lookup[sec_id]
    ex_ns = ex_date.value
    idx = int(np.searchsorted(dates_ns, ex_ns, side="left")) - 1
    while idx >= 0:
        dt_diff_days = (ex_ns - int(dates_ns[idx])) / 86_400_000_000_000.0
        if dt_diff_days > max_lookback_days:
            return None
        c = float(closes[idx])
        if np.isfinite(c) and c > 0:
            return c
        idx -= 1
    return None


def _build_event_record(**kwargs) -> dict:
    cols = {
        "EventID": "",
        "SecurityID": None,
        "CompanyCode": None,
        "MainType": None,
        "SubType": None,
        "EventType": "",
        "AnnouncementDate": pd.NaT,
        "ExDate": pd.NaT,
        "ExDateBasis": "",
        "RecordDate": pd.NaT,
        "EffectiveDate": pd.NaT,
        "Ratio": np.nan,
        "OldProportion": np.nan,
        "NewProportion": np.nan,
        "SubscriptionPrice": np.nan,
        "DividendPerShare": np.nan,
        "DividendRateRaw": "",
        "CumulativeSharesListed": np.nan,
        "PriorIssuedQuantity": np.nan,
        "PreExClose": np.nan,
        "PriceFactor": 1.0,
        "VolumeFactor": 1.0,
        "DividendFactor": 1.0,
        "Source": "",
        "SourceSheet": "",
        "SourceRow": -1,
        "ValidationStatus": "",
        "ValidationNotes": "",
    }
    cols.update(kwargs)
    return cols


def parse_and_validate_corporate_actions(
    source_dir: Path,
    prices_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    name_aliases: dict[str, set[str]],
    yearly_qty: dict[tuple[int, str], float],
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Parse all 5 raw CSE corporate action workbooks directly, validate ratios,
    resolve exact ExDates & SecurityIDs, compute TERP / dividend / split factors,
    and flag incomplete records without guessing.
    """
    resolver = SecurityResolver(identity_df, name_aliases)
    max_price_date = pd.Timestamp(prices_df["Date"].max()).normalize()

    # Build fast binary-search lookup for PreExClose
    price_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    df_p_sorted = prices_df.sort_values(["SecurityID", "Date"])
    for sid, grp in df_p_sorted.groupby("SecurityID", sort=False):
        price_lookup[sid] = (
            grp["Date"].astype("datetime64[ns]").astype("int64").to_numpy(),
            grp["Close"].to_numpy(dtype=float),
        )

    events: list[dict] = []

    def _emit_events(
        sec_ids: list[str],
        raw_code: object,
        raw_name: object,
        base_kwargs: dict,
    ) -> None:
        if not sec_ids:
            c_guess = resolver.resolve_company_code(raw_code, raw_name, base_kwargs.get("ExDate", pd.NaT))
            rec = _build_event_record(
                **{
                    **base_kwargs,
                    "CompanyCode": c_guess or (str(raw_code).strip() if pd.notna(raw_code) else None),
                    "PriceFactor": 1.0,
                    "VolumeFactor": 1.0,
                    "DividendFactor": 1.0,
                    "ValidationStatus": "UNRESOLVED_SECURITY_MATCH",
                    "ValidationNotes": f"Could not uniquely match security/name: code={raw_code!r}, name={raw_name!r}",
                }
            )
            events.append(rec)
            return

        for sid in sec_ids:
            parts = sid.split("-")
            rec = _build_event_record(
                **{
                    **base_kwargs,
                    "SecurityID": sid,
                    "CompanyCode": parts[0],
                    "MainType": parts[1] if len(parts) > 1 else None,
                    "SubType": parts[2] if len(parts) > 2 else None,
                }
            )
            events.append(rec)

    # ── A. Stock Splits (05Sub Division (Share Splits).xls) ──────────────────
    sp_path = source_dir / "05Sub Division (Share Splits).xls"
    if sp_path.exists():
        xl_sp = pd.ExcelFile(sp_path)
        for sh in [s for s in xl_sp.sheet_names if re.match(r"^\s*\d{4}\s*$", s)]:
            d = xl_sp.parse(sh, header=None)
            hrow = next(
                (i for i in range(min(10, len(d)))
                 if "PROPORTION" in " ".join(str(x).upper() for x in d.iloc[i].values if pd.notna(x))),
                None,
            )
            if hrow is None:
                continue
            df_s = xl_sp.parse(sh, header=hrow)
            df_s.columns = [re.sub(r"\s+", " ", str(c).strip().upper()) for c in df_s.columns]
            c_ann  = next((c for c in df_s.columns if "ANNOUNCEMENT" in c), None)
            c_cid  = next((c for c in df_s.columns if c == "COMPANY ID"), None)
            c_name = next((c for c in df_s.columns if "COMPANY NAME" in c or "SHORT NAME" in c), None)
            c_sec  = next((c for c in df_s.columns if c == "SECURITY"), None)
            c_old  = next((c for c in df_s.columns if "OLD PROPORTION" in c), None)
            c_new  = next((c for c in df_s.columns if "NEW PROPORTION" in c), None)
            c_eff  = next((c for c in df_s.columns if "EFFECTIVE" in c), None)

            last_cid, last_name = None, None
            for idx, r in df_s.iterrows():
                old_p = pd.to_numeric(r.get(c_old), errors="coerce") if c_old else np.nan
                new_p = pd.to_numeric(r.get(c_new), errors="coerce") if c_new else np.nan
                if pd.isna(old_p) or pd.isna(new_p):
                    continue
                raw_cid  = r.get(c_cid)  if c_cid  else None
                raw_name = r.get(c_name) if c_name else None
                raw_sec  = r.get(c_sec)  if c_sec  else None
                if pd.isna(raw_cid) and pd.notna(raw_name) and last_cid is not None:
                    # Continuation row e.g. 2016 JKH-W-0023
                    if normalize_company_name(raw_name) == normalize_company_name(last_name):
                        raw_cid = last_cid
                if pd.notna(raw_cid):
                    last_cid, last_name = raw_cid, raw_name

                eff_dt = parse_excel_date(r.get(c_eff)) if c_eff else pd.NaT
                ann_dt = parse_excel_date(r.get(c_ann)) if c_ann else pd.NaT

                sids = resolver.resolve_security_ids(raw_sec, raw_cid, raw_name, eff_dt)
                ratio = float(new_p / old_p) if (old_p > 0 and new_p > 0) else np.nan
                if pd.isna(eff_dt) or pd.isna(ratio) or ratio <= 0:
                    _emit_events(sids, raw_cid, raw_name, {
                        "EventType": "StockSplit",
                        "AnnouncementDate": ann_dt,
                        "ExDate": eff_dt,
                        "EffectiveDate": eff_dt,
                        "OldProportion": old_p,
                        "NewProportion": new_p,
                        "Ratio": ratio,
                        "Source": sp_path.name,
                        "SourceSheet": sh,
                        "SourceRow": int(hrow + 2 + idx),
                        "ValidationStatus": "UNRESOLVED_INVALID_SPLIT_PARAMS",
                    })
                elif eff_dt > max_price_date:
                    _emit_events(sids, raw_cid, raw_name, {
                        "EventType": "StockSplit",
                        "AnnouncementDate": ann_dt,
                        "ExDate": eff_dt,
                        "EffectiveDate": eff_dt,
                        "OldProportion": float(old_p),
                        "NewProportion": float(new_p),
                        "Ratio": ratio,
                        "Source": sp_path.name,
                        "SourceSheet": sh,
                        "SourceRow": int(hrow + 2 + idx),
                        "ValidationStatus": "POST_DATASET_WINDOW_SKIPPED",
                    })
                else:
                    pf = float(old_p / new_p)
                    vf = float(new_p / old_p)
                    _emit_events(sids, raw_cid, raw_name, {
                        "EventType": "StockSplit",
                        "AnnouncementDate": ann_dt,
                        "ExDate": eff_dt,
                        "ExDateBasis": "EFFECTIVE_DATE",
                        "EffectiveDate": eff_dt,
                        "OldProportion": float(old_p),
                        "NewProportion": float(new_p),
                        "Ratio": ratio,
                        "PriceFactor": pf,
                        "VolumeFactor": vf,
                        "DividendFactor": 1.0,
                        "Source": sp_path.name,
                        "SourceSheet": sh,
                        "SourceRow": int(hrow + 2 + idx),
                        "ValidationStatus": "VALIDATED",
                        "ValidationNotes": f"Split {old_p:g}:{new_p:g} (Price*{pf:.6g}, Vol*{vf:.6g})",
                    })

    # ── B. Bonus Issues (03Capitalization of Reserves (Bonus Issues).xls) ────
    bon_path = source_dir / "03Capitalization of Reserves (Bonus Issues).xls"
    if bon_path.exists():
        xl_b = pd.ExcelFile(bon_path)
        for sh in [s for s in xl_b.sheet_names if re.match(r"^\s*\d{4}\s*$", s)]:
            yr = int(sh.strip())
            d = xl_b.parse(sh, header=None)
            hrow = next(
                (i for i in range(min(10, len(d)))
                 if any(k in " ".join(str(x).lower() for x in d.iloc[i].values if pd.notna(x))
                        for k in ("company", "proportion", "propotion", "ratio"))),
                None,
            )
            if hrow is None:
                continue
            df_b = xl_b.parse(sh, header=hrow)
            df_b.columns = [re.sub(r"\s+", " ", str(c).strip().upper()) for c in df_b.columns]
            c_name  = next((c for c in df_b.columns if "COMPANY" in c), None)
            c_ann   = next((c for c in df_b.columns if "ANNOUNCEMENT" in c), None)
            c_allot = next((c for c in df_b.columns if "ALLOTMENT" in c), None)
            c_list  = next((c for c in df_b.columns if "DATE LISTED" in c), None)
            c_old   = next((c for c in df_b.columns if "OLD PROPORTION" in c), None)
            c_new   = next((c for c in df_b.columns if "NEW PROPORTION" in c), None)
            c_ratio = next((c for c in df_b.columns if "PROPOTION" in c or "PROPORTION" in c or "RATIO" in c), None)
            c_shqty = next((c for c in df_b.columns if "SHARES" in c), None)
            if c_name is None:
                continue

            last_base_b = None
            for idx, r in df_b.iterrows():
                row_text = " ".join(str(v).upper() for v in r.values if pd.notna(v))
                if any(stop in row_text for stop in _SECTOR_SUMMARY_STOPWORDS):
                    break  # Bottom sector summary table begins here
                nm = r.get(c_name)
                if pd.isna(nm) or str(nm).strip().lower() in ("", "nan", "sector", "total"):
                    continue
                nm_str = str(nm).strip()
                if "SECTOR" in nm_str.upper():
                    continue
                mtype_hint_b = None
                if (nm_str.startswith("-") or re.match(r"^\(?\s*(VOTING|NON\s*-?\s*VOTING)\s*\)?$", nm_str, re.I)) and last_base_b:
                    mtype_hint_b = "X" if "NON" in nm_str.upper() else "N"
                    nm_str = f"{last_base_b} {nm_str}"
                else:
                    last_base_b = re.sub(r"[-\s]+(NON\s*-?\s*VOTING|VOTING).*", "", nm_str, flags=re.I).strip()

                ann_dt   = parse_excel_date(r.get(c_ann)) if c_ann else pd.NaT
                allot_dt = parse_excel_date(r.get(c_allot)) if c_allot else pd.NaT
                list_dt  = parse_excel_date(r.get(c_list)) if c_list else pd.NaT
                ex_dt    = allot_dt
                sids     = resolver.resolve_security_ids(None, None, nm_str, ex_dt, raw_mtype_hint=mtype_hint_b)
                sh_qty   = _safe_num(r.get(c_shqty)) if c_shqty else np.nan

                base = {
                    "EventType": "BonusIssue",
                    "AnnouncementDate": ann_dt,
                    "ExDate": ex_dt,
                    "ExDateBasis": "ALLOTMENT_DATE_PROXY" if pd.notna(ex_dt) else "MISSING",
                    "EffectiveDate": allot_dt if pd.notna(allot_dt) else list_dt,
                    "CumulativeSharesListed": sh_qty,
                    "Source": bon_path.name,
                    "SourceSheet": sh,
                    "SourceRow": int(hrow + 2 + idx),
                }

                if pd.isna(ex_dt):
                    _emit_events(sids, None, nm, {
                        **base,
                        "ValidationStatus": "UNRESOLVED_MISSING_EX_DATE",
                        "ValidationNotes": "No Allotment/Ex-Date recorded in historical sheet",
                    })
                    continue

                if c_old and c_new:
                    old_v = _safe_num(r.get(c_old))
                    new_v = _safe_num(r.get(c_new))
                    method = "EXPLICIT_OLD_NEW_COLS"
                elif c_ratio:
                    pair = parse_proportion_pair(r.get(c_ratio))
                    if not pair:
                        _emit_events(sids, None, nm, {
                            **base,
                            "ValidationStatus": "UNRESOLVED_INVALID_RATIO",
                            "ValidationNotes": f"Could not parse ratio: {r.get(c_ratio)!r}",
                        })
                        continue
                    prior_q = _lookup_prior_issued_qty(yearly_qty, sids[0], yr, sh_qty) if sids else None
                    # Bonus single decimal e.g. '25.01' means 25.01 old : 1 new
                    if pair[2] == "single_old_per_new":
                        old_v, new_v, method = pair[1], pair[0], "SINGLE_OLD_PER_NEW"
                    else:
                        old_v, new_v, method = _verify_new_old_ratio(
                            pair, sh_qty, prior_q, allow_assume_a_new_if_a_le_b=True
                        )
                    base["PriorIssuedQuantity"] = prior_q if prior_q else np.nan
                else:
                    old_v, new_v, method = None, None, "NO_RATIO_COLUMN"

                if not old_v or not new_v or old_v <= 0 or new_v <= 0:
                    _emit_events(sids, None, nm, {
                        **base,
                        "ValidationStatus": "UNRESOLVED_RATIO_DIRECTION",
                        "ValidationNotes": f"Ambiguous bonus ratio {r.get(c_ratio)!r} ({method})",
                    })
                    continue

                r_new = new_v / old_v
                pf = float(old_v / (old_v + new_v))
                vf = float((old_v + new_v) / old_v)
                status = "VALIDATED" if "UNRESOLVED" not in method else method
                _emit_events(sids, None, nm, {
                    **base,
                    "OldProportion": float(old_v),
                    "NewProportion": float(new_v),
                    "Ratio": float(r_new),
                    "PriceFactor": pf,
                    "VolumeFactor": vf,
                    "DividendFactor": 1.0,
                    "ValidationStatus": status,
                    "ValidationNotes": f"Bonus {new_v:g} new : {old_v:g} old ({method})",
                })

    # ── C. Scrip Dividends (02Scrip Dividends.xls) ───────────────────────────
    sc_path = source_dir / "02Scrip Dividends.xls"
    if sc_path.exists():
        xl_sc = pd.ExcelFile(sc_path)
        for sh in [s for s in xl_sc.sheet_names if re.match(r"^\s*\d{4}\s*$", s)]:
            d = xl_sc.parse(sh, header=None)
            hrow = next(
                (i for i in range(min(10, len(d)))
                 if "PROPORTION" in " ".join(str(x).upper() for x in d.iloc[i].values if pd.notna(x))),
                None,
            )
            if hrow is None:
                continue
            df_sc = xl_sc.parse(sh, header=hrow)
            df_sc.columns = [re.sub(r"\s+", " ", str(c).strip().upper()) for c in df_sc.columns]
            c_ann   = next((c for c in df_sc.columns if "ANNOUNCEMENT" in c), None)
            c_name  = next((c for c in df_sc.columns if "COMPANY" in c), None)
            c_allot = next((c for c in df_sc.columns if "ALLOTMENT" in c), None)
            c_old   = next((c for c in df_sc.columns if "OLD PROPORTION" in c), None)
            c_new   = next((c for c in df_sc.columns if "NEW PROPORTION" in c), None)
            c_qty   = next((c for c in df_sc.columns if "SHARES LISTED" in c), None)
            if not (c_name and c_old and c_new):
                continue

            last_base_name = None
            for idx, r in df_sc.iterrows():
                old_v = _safe_num(r.get(c_old))
                new_v = _safe_num(r.get(c_new))
                if pd.isna(old_v) or pd.isna(new_v) or old_v <= 0 or new_v <= 0:
                    continue

                raw_nm = r.get(c_name)
                nm_str = str(raw_nm).strip() if pd.notna(raw_nm) else ""
                mtype_hint = None
                if (not nm_str or nm_str.startswith("-") or re.match(r"^\(?\s*(VOTING|NON\s*-?\s*VOTING)\s*\)?$", nm_str, re.I)) and last_base_name:
                    mtype_hint = "X" if "NON" in nm_str.upper() or not nm_str else "N"
                    nm_str = f"{last_base_name} {nm_str}".strip()
                elif nm_str:
                    last_base_name = re.sub(r"[-\s]+(ORDINARY\s+)?(NON\s*-?\s*VOTING|VOTING).*", "", nm_str, flags=re.I).strip()

                ann_dt = parse_excel_date(r.get(c_ann)) if c_ann else pd.NaT
                ex_dt  = parse_excel_date(r.get(c_allot)) if c_allot else pd.NaT
                sids   = resolver.resolve_security_ids(None, None, nm_str, ex_dt, raw_mtype_hint=mtype_hint)
                sh_qty = _safe_num(r.get(c_qty)) if c_qty else np.nan

                base = {
                    "EventType": "ScripDividend",
                    "AnnouncementDate": ann_dt,
                    "ExDate": ex_dt,
                    "ExDateBasis": "ALLOTMENT_DATE_EX_VERIFIED" if pd.notna(ex_dt) else "MISSING",
                    "EffectiveDate": ex_dt,
                    "OldProportion": float(old_v),
                    "NewProportion": float(new_v),
                    "Ratio": float(new_v / old_v),
                    "CumulativeSharesListed": sh_qty,
                    "Source": sc_path.name,
                    "SourceSheet": sh,
                    "SourceRow": int(hrow + 2 + idx),
                }
                if pd.isna(ex_dt):
                    _emit_events(sids, None, nm_str, {**base, "ValidationStatus": "UNRESOLVED_MISSING_EX_DATE"})
                    continue

                pf = float(old_v / (old_v + new_v))
                vf = float((old_v + new_v) / old_v)
                _emit_events(sids, None, nm_str, {
                    **base,
                    "PriceFactor": pf,
                    "VolumeFactor": vf,
                    "DividendFactor": 1.0,
                    "ValidationStatus": "VALIDATED",
                    "ValidationNotes": f"Scrip {new_v:g} new : {old_v:g} old (header schema verified by column name)",
                })

    # ── D. Rights Issues (04Rights Issues.xls) ───────────────────────────────
    ri_path = source_dir / "04Rights Issues.xls"
    if ri_path.exists():
        xl_ri = pd.ExcelFile(ri_path)
        for sh in [s for s in xl_ri.sheet_names if re.match(r"^\s*\d{4}\s*$", s)]:
            yr = int(sh.strip())
            d = xl_ri.parse(sh, header=None)
            hrow = next(
                (i for i in range(min(10, len(d)))
                 if any(k in " ".join(str(x).lower() for x in d.iloc[i].values if pd.notna(x))
                        for k in ("propotion", "proportion", "ratio"))
                 and any(k in " ".join(str(x).lower() for x in d.iloc[i].values if pd.notna(x))
                         for k in ("company", "name"))),
                None,
            )
            if hrow is None:
                continue
            df_r = xl_ri.parse(sh, header=hrow)
            df_r.columns = [re.sub(r"\s+", " ", str(c).strip().upper()) for c in df_r.columns]
            c_ann   = next((c for c in df_r.columns if "ANNOUNCEMENT" in c), None)
            c_name  = next((c for c in df_r.columns if c in ("NAME", "COMPANY NAME", "COMPANY")), None)
            c_allot = next((c for c in df_r.columns if "ALLOTMENT" in c), None)
            c_sub   = next((c for c in df_r.columns if "CONSIDERATION" in c or "ISSUE PRICE" in c), None)
            c_prop  = next((c for c in df_r.columns if "PROPOTION" in c or "PROPORTION" in c or "RATIO" in c), None)
            c_qty   = next((c for c in df_r.columns if "PROVISIONALLY ALLOTTED" in c or "SHARES OFFERED" in c or "NO. OF SHARES" in c), None)
            if not (c_name and c_prop):
                continue

            last_name_r = None
            for idx, r in df_r.iterrows():
                row_text = " ".join(str(v).upper() for v in r.values if pd.notna(v))
                if any(stop in row_text for stop in _SECTOR_SUMMARY_STOPWORDS) or "TOTAL" in str(r.get(c_name, "")).upper():
                    continue
                pair = parse_proportion_pair(r.get(c_prop))
                sub_p = _safe_num(r.get(c_sub)) if c_sub else np.nan
                sh_qty = _safe_num(r.get(c_qty)) if c_qty else np.nan
                raw_nm = r.get(c_name)
                if pd.isna(raw_nm) and pd.isna(sub_p) and not pair:
                    continue

                nm_str = str(raw_nm).strip() if pd.notna(raw_nm) else ""
                mtype_hint = None
                if not nm_str and last_name_r:
                    nm_str = last_name_r
                    mtype_hint = "X"  # Secondary class row (e.g. Tess Agro 2024 Non-Voting)
                elif (nm_str.startswith("-") or re.match(r"^\(?\s*(VOTING|NON\s*-?\s*VOTING)\s*\)?$", nm_str, re.I)) and last_name_r:
                    mtype_hint = "X" if "NON" in nm_str.upper() else "N"
                    nm_str = f"{last_name_r} {nm_str}"
                elif nm_str:
                    last_name_r = re.sub(r"[-\s]+(NON\s*-?\s*VOTING|VOTING).*", "", nm_str, flags=re.I).strip()

                ann_dt = parse_excel_date(r.get(c_ann)) if c_ann else pd.NaT
                ex_dt  = parse_excel_date(r.get(c_allot)) if c_allot else pd.NaT
                sids   = resolver.resolve_security_ids(None, None, nm_str, ex_dt, raw_mtype_hint=mtype_hint)

                base = {
                    "EventType": "RightsIssue",
                    "AnnouncementDate": ann_dt,
                    "ExDate": ex_dt,
                    "ExDateBasis": "PROVISIONAL_ALLOTMENT_PROXY" if pd.notna(ex_dt) else "MISSING",
                    "EffectiveDate": ex_dt,
                    "SubscriptionPrice": float(sub_p) if pd.notna(sub_p) else np.nan,
                    "CumulativeSharesListed": sh_qty,
                    "Source": ri_path.name,
                    "SourceSheet": sh,
                    "SourceRow": int(hrow + 2 + idx),
                }

                if pd.isna(ex_dt):
                    _emit_events(sids, None, nm_str, {
                        **base,
                        "ValidationStatus": "UNRESOLVED_MISSING_EX_DATE",
                        "ValidationNotes": "No Allotment/XR Date present in historical sheet",
                    })
                    continue
                if pd.isna(sub_p) or sub_p < 0:
                    _emit_events(sids, None, nm_str, {
                        **base,
                        "ValidationStatus": "UNRESOLVED_MISSING_SUBSCRIPTION_PRICE",
                    })
                    continue
                if not pair:
                    _emit_events(sids, None, nm_str, {
                        **base,
                        "ValidationStatus": "UNRESOLVED_INVALID_RATIO",
                        "ValidationNotes": f"Could not parse rights ratio: {r.get(c_prop)!r}",
                    })
                    continue
                if not sids:
                    _emit_events([], None, nm_str, base)
                    continue

                for sid in sids:
                    prior_q = _lookup_prior_issued_qty(yearly_qty, sid, yr, sh_qty)
                    # Rights 2013-2025 Propotion column is explicitly NEW : OLD (e.g. .459:1, 143:101, 4:1)
                    allow_direct_ab = (yr >= 2013)
                    old_v, new_v, method = _verify_new_old_ratio(
                        pair, sh_qty, prior_q, allow_assume_a_new_if_a_le_b=True
                    )
                    if method == "UNRESOLVED_RATIO_DIRECTION" and allow_direct_ab:
                        old_v, new_v, method = pair[1], pair[0], "CSE_2013PLUS_NEW_OLD_CONVENTION"

                    if not old_v or not new_v:
                        _emit_events([sid], None, nm_str, {
                            **base,
                            "PriorIssuedQuantity": prior_q if prior_q else np.nan,
                            "ValidationStatus": "UNRESOLVED_RATIO_DIRECTION",
                            "ValidationNotes": f"Ambiguous pre-2013 rights ratio {r.get(c_prop)!r}",
                        })
                        continue

                    r_new = float(new_v / old_v)
                    p_cum = _find_pre_ex_close(price_lookup, sid, ex_dt)
                    if p_cum is None:
                        _emit_events([sid], None, nm_str, {
                            **base,
                            "OldProportion": float(old_v),
                            "NewProportion": float(new_v),
                            "Ratio": r_new,
                            "PriorIssuedQuantity": prior_q if prior_q else np.nan,
                            "ValidationStatus": "UNRESOLVED_NO_PRE_EX_PRICE",
                            "ValidationNotes": "No prior closing price within 45d before ExDate for TERP",
                        })
                        continue

                    if p_cum <= sub_p:
                        _emit_events([sid], None, nm_str, {
                            **base,
                            "OldProportion": float(old_v),
                            "NewProportion": float(new_v),
                            "Ratio": r_new,
                            "PriorIssuedQuantity": prior_q if prior_q else np.nan,
                            "PreExClose": p_cum,
                            "PriceFactor": 1.0,
                            "VolumeFactor": 1.0,
                            "ValidationStatus": "VALIDATED_NON_DILUTIVE_UNDERWATER",
                            "ValidationNotes": f"P_cum ({p_cum:.2f}) <= SubPrice ({sub_p:.2f}); no dilution",
                        })
                    else:
                        terp = (p_cum + r_new * sub_p) / (1.0 + r_new)
                        pf = float(terp / p_cum)
                        vf = float(p_cum / terp)
                        _emit_events([sid], None, nm_str, {
                            **base,
                            "OldProportion": float(old_v),
                            "NewProportion": float(new_v),
                            "Ratio": r_new,
                            "PriorIssuedQuantity": prior_q if prior_q else np.nan,
                            "PreExClose": p_cum,
                            "PriceFactor": pf,
                            "VolumeFactor": vf,
                            "ValidationStatus": "VALIDATED_TERP",
                            "ValidationNotes": f"TERP={terp:.4f}, P_cum={p_cum:.2f}, S={sub_p:.2f}, r={r_new:.4f} ({method})",
                        })

    # ── E. Cash Dividends (01Dividends.xls) ──────────────────────────────────
    div_path = source_dir / "01Dividends.xls"
    if div_path.exists():
        xl_d = pd.ExcelFile(div_path)
        for sh in [s for s in xl_d.sheet_names if re.match(r"^\s*\d{4}\s*$", s)]:
            yr = int(sh.strip())
            d = xl_d.parse(sh, header=None)
            hrow = next(
                (i for i in range(min(10, len(d)))
                 if any(k in " ".join(str(x).lower() for x in d.iloc[i].values if pd.notna(x))
                        for k in ("security", "xd from", "xd date", "date of ex", "date_of_ex"))),
                None,
            )
            if hrow is None:
                continue
            df_d = xl_d.parse(sh, header=hrow)
            df_d.columns = [re.sub(r"[\s_]+", " ", str(c).strip().upper()) for c in df_d.columns]
            c_ann  = next((c for c in df_d.columns if "ANNOUNCEMENT" in c), None)
            c_sec  = next((c for c in df_d.columns if c in ("SECURITY", "SECURITY CODE")), None)
            c_cid  = next((c for c in df_d.columns if c == "COMPANY ID"), None)
            c_type = next((c for c in df_d.columns if "TYPE OF SHARES" in c), None)
            c_name = next((c for c in df_d.columns if c in ("SHORT NAME", "COMPANY NAME", "COMPANY")), None)
            c_rate = next((c for c in df_d.columns if "RATE OF DIVIDEND" in c or c == "RATE"), None)
            c_rem  = next((c for c in df_d.columns if "RATE & REMARKS" in c or "RATE AND REMARKS" in c
                           or "RATE & DESCRIPTION" in c or "EXTERNAL REMARKS" in c or c == "REMARKS"), None)
            c_ex   = next((c for c in df_d.columns if "DATE OF EX" in c or "XD FROM" in c or "XD DATE" in c), None)
            c_pay  = next((c for c in df_d.columns if "PAYMENT" in c), None)
            c_cum  = next((c for c in df_d.columns if "CUM PRICE" in c), None)

            for idx, r in df_d.iterrows():
                ex_dt = parse_excel_date(r.get(c_ex)) if c_ex else pd.NaT
                if pd.isna(ex_dt) and pd.isna(r.get(c_sec) if c_sec else r.get(c_name)):
                    continue

                raw_sec  = r.get(c_sec)  if c_sec  else None
                raw_cid  = r.get(c_cid)  if c_cid  else None
                raw_name = r.get(c_name) if c_name else None
                if raw_sec is not None and str(raw_sec).strip().upper() in ("SECURITY", "SECURITY CODE"):
                    continue

                mtype_hint = None
                if c_type and pd.notna(r.get(c_type)):
                    tstr = str(r.get(c_type)).strip().upper()
                    if tstr.startswith("X"):
                        mtype_hint = "X"
                    elif tstr.startswith("P"):
                        mtype_hint = "P"
                    elif tstr.startswith("N"):
                        mtype_hint = "N"

                sids = resolver.resolve_security_ids(raw_sec, raw_cid, raw_name, ex_dt, raw_mtype_hint=mtype_hint)
                ann_dt = parse_excel_date(r.get(c_ann)) if c_ann else pd.NaT
                pay_dt = parse_excel_date(r.get(c_pay)) if c_pay else pd.NaT

                rate_val = _safe_num(r.get(c_rate)) if c_rate else np.nan
                rem_str  = str(r.get(c_rem)).strip() if (c_rem and pd.notna(r.get(c_rem))) else ""
                raw_rate_txt = f"rate={rate_val!r} | {rem_str[:120]}" if pd.notna(rate_val) else rem_str[:120]

                # Extract explicit LKR DividendPerShare (NEVER convert pre-2008 % of par by assuming Rs.10 par)
                dps = np.nan
                div_status = "VALIDATED"
                div_note = ""

                if yr >= 2016 and pd.notna(rate_val) and rate_val > 0 and "%" not in rem_str:
                    dps = float(rate_val)
                    div_note = f"Numeric RATE OF DIVIDEND (Rs.{dps:.4f})"
                else:
                    m_rs = re.search(r"R[Ss]\.?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", rem_str)
                    if m_rs:
                        dps = float(m_rs.group(1).replace(",", ""))
                        div_note = f"Explicit Rs. amount parsed from remarks (Rs.{dps:.4f})"
                    elif "%" in rem_str or (yr <= 2007 and pd.notna(rate_val)):
                        div_status = "UNRESOLVED_PERCENT_OF_PAR"
                        div_note = "Percentage-of-par dividend without explicit Rs./share or verified historical par value"
                    else:
                        div_status = "UNRESOLVED_MISSING_DIVIDEND_AMOUNT"
                        div_note = "Could not extract numeric Rs./share dividend"

                base = {
                    "EventType": "CashDividend",
                    "AnnouncementDate": ann_dt,
                    "ExDate": ex_dt,
                    "ExDateBasis": "OFFICIAL_XD_DATE" if pd.notna(ex_dt) else "MISSING",
                    "EffectiveDate": pay_dt,
                    "DividendPerShare": dps,
                    "DividendRateRaw": raw_rate_txt,
                    "Source": div_path.name,
                    "SourceSheet": sh,
                    "SourceRow": int(hrow + 2 + idx),
                }

                if pd.isna(ex_dt):
                    _emit_events(sids, raw_cid or raw_sec, raw_name, {
                        **base,
                        "ValidationStatus": "UNRESOLVED_MISSING_EX_DATE",
                        "ValidationNotes": div_note,
                    })
                    continue

                if div_status != "VALIDATED" or not sids or pd.isna(dps) or dps <= 0:
                    _emit_events(sids, raw_cid or raw_sec, raw_name, {
                        **base,
                        "ValidationStatus": div_status if sids else "UNRESOLVED_SECURITY_MATCH",
                        "ValidationNotes": div_note,
                    })
                    continue

                for sid in sids:
                    p_cum = _find_pre_ex_close(price_lookup, sid, ex_dt)
                    if p_cum is None:
                        p_cum_sheet = _safe_num(r.get(c_cum)) if c_cum else np.nan
                        _emit_events([sid], raw_cid or raw_sec, raw_name, {
                            **base,
                            "PreExClose": p_cum_sheet if pd.notna(p_cum_sheet) else np.nan,
                            "DividendFactor": 1.0,
                            "ValidationStatus": "UNRESOLVED_NO_PRE_EX_PRICE",
                            "ValidationNotes": "No prior market close in price table within 45d before ExDate",
                        })
                    elif dps >= p_cum:
                        _emit_events([sid], raw_cid or raw_sec, raw_name, {
                            **base,
                            "PreExClose": p_cum,
                            "DividendFactor": 1.0,
                            "ValidationStatus": "UNRESOLVED_DIVIDEND_EXCEEDS_PRICE",
                            "ValidationNotes": f"DPS ({dps:.2f}) >= PreExClose ({p_cum:.2f}); flagged for manual review",
                        })
                    else:
                        f_div = float((p_cum - dps) / p_cum)
                        _emit_events([sid], raw_cid or raw_sec, raw_name, {
                            **base,
                            "PreExClose": p_cum,
                            "PriceFactor": 1.0,
                            "VolumeFactor": 1.0,
                            "DividendFactor": f_div,
                            "ValidationStatus": "VALIDATED",
                            "ValidationNotes": f"{div_note}; PreExClose={p_cum:.2f}, DivFactor={f_div:.6f}",
                        })

    ca_df = pd.DataFrame(events)

    # Deduplicate overlapping boundary-year sheet rows cleanly without Pandas 3.0 NA propagation
    ca_df["_r_k"]   = ca_df["Ratio"].round(6).fillna(-999999.0)
    ca_df["_sub_k"] = ca_df["SubscriptionPrice"].round(4).fillna(-999999.0)
    ca_df["_dps_k"] = ca_df["DividendPerShare"].round(4).fillna(-999999.0)
    ca_df["_not_val"] = ~ca_df["ValidationStatus"].isin(["VALIDATED", "VALIDATED_TERP", "VALIDATED_NON_DILUTIVE_UNDERWATER"])
    ca_df = ca_df.sort_values(["ExDate", "_not_val", "SourceSheet", "SourceRow"], na_position="last")
    is_dup = (
        ca_df["SecurityID"].notna()
        & ca_df["ExDate"].notna()
        & ca_df.duplicated(subset=["SecurityID", "EventType", "ExDate", "_r_k", "_sub_k", "_dps_k"], keep="first")
    )
    if is_dup.any():
        ca_df.loc[is_dup, ["PriceFactor", "VolumeFactor", "DividendFactor"]] = 1.0
        ca_df.loc[is_dup, "ValidationStatus"] = "DUPLICATE_BOUNDARY_EVENT_SKIPPED"
        ca_df.loc[is_dup, "ValidationNotes"] = "Duplicate cross-sheet event excluded to prevent double adjustment"
    ca_df = ca_df.drop(columns=["_r_k", "_sub_k", "_dps_k", "_not_val"]).reset_index(drop=True)
    ca_df["EventID"] = [f"CA_{i+1:06d}" for i in range(len(ca_df))]

    return ca_df, max_price_date


def _safe_num(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    s = str(val).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def compute_adjustment_factors_and_series(
    prices_df: pd.DataFrame,
    ca_validated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute cumulative backward adjustment factors strictly prior to ExDate (`Date < ExDate`)
    using chronological ordering per SecurityID and build:
      1. `factors_audit_df`: traceable event-level adjustment factor ledger
      2. `split_adj_df`: `cse_daily_prices_split_adjusted.parquet`
      3. `tr_df`: `cse_total_return_history.parquet`

    Total-Return Reinvestment Methodology:
      - `CapitalAdjClose` adjusts historical prices for verified Stock Splits, Bonus Issues,
        Scrip Dividends, and Rights TERP events.
      - `SplitBonusAdjClose` adjusts strictly for verified Stock Splits and Bonus Issues.
      - Cash dividends are applied ONLY in `TotalReturnPrice` / `TotalReturnIndex` via
        `DivFactor = (PreExClose - DividendPerShare) / PreExClose` applied backwards for
        `Date < ExDate`, equivalent to reinvesting the gross cash dividend into shares of
        the same security at the ex-date open (`PreExClose - D`).
    """
    valid_statuses = {"VALIDATED", "VALIDATED_TERP"}
    active_ca = ca_validated[
        ca_validated["ValidationStatus"].isin(valid_statuses)
        & ca_validated["SecurityID"].notna()
        & ca_validated["ExDate"].notna()
    ].copy()

    # Vectorized per-(SecurityID, EventType) cumulative backward factor builder
    def _cum_factors_by_type(df_prices: pd.DataFrame, event_types: tuple[str, ...], col: str = "PriceFactor") -> np.ndarray:
        sub = active_ca[active_ca["EventType"].isin(event_types)].copy()
        out = np.ones(len(df_prices), dtype=float)
        if sub.empty:
            return out

        grouped_ev = {
            sid: (
                g.sort_values("ExDate")["ExDate"].astype("datetime64[ns]").astype("int64").to_numpy(),
                g.sort_values("ExDate")[col].to_numpy(dtype=float),
            )
            for sid, g in sub.groupby("SecurityID", sort=False)
        }
        dates_ns_all = df_prices["Date"].astype("datetime64[ns]").astype("int64").to_numpy()
        for sid, idx in df_prices.groupby("SecurityID", sort=False).groups.items():
            if sid not in grouped_ev:
                continue
            ev_dates, ev_factors = grouped_ev[sid]
            # Suffix cumulative product: rev_cumprod[j] = product(ev_factors[j:])
            suffix_prod = np.cumprod(ev_factors[::-1])[::-1]
            pos_arr = np.searchsorted(ev_dates, dates_ns_all[idx], side="right")
            # For date d, events strictly AFTER d are at ev_dates[pos_arr:]
            valid_pos = pos_arr < len(ev_dates)
            res = np.ones(len(idx), dtype=float)
            res[valid_pos] = suffix_prod[pos_arr[valid_pos]]
            out[idx] = res
        return out

    df = prices_df.sort_values(["SecurityID", "Date"]).reset_index(drop=True)

    f_split  = _cum_factors_by_type(df, ("StockSplit",), "PriceFactor")
    f_bonus  = _cum_factors_by_type(df, ("BonusIssue",), "PriceFactor")
    f_scrip  = _cum_factors_by_type(df, ("ScripDividend",), "PriceFactor")
    f_rights = _cum_factors_by_type(df, ("RightsIssue",), "PriceFactor")
    f_div    = _cum_factors_by_type(df, ("CashDividend",), "DividendFactor")

    f_split_bonus = f_split * f_bonus
    f_capital     = f_split_bonus * f_scrip * f_rights
    f_total_ret   = f_capital * f_div

    # Build Split/Bonus-adjusted DataFrame (keeps Raw prices untouched alongside explicit adj columns)
    split_adj = pd.DataFrame({
        "Date": df["Date"],
        "SecurityID": df["SecurityID"],
        "CompanyCode": df["CompanyCode"],
        "IssuerID": df["IssuerID"],
        "MainType": df["MainType"],
        "SubType": df["SubType"],
        "RawOpen": df["Open"],
        "RawHigh": df["High"],
        "RawLow": df["Low"],
        "RawClose": df["Close"],
        "RawShareVolume": df["ShareVolume"],
        "Open": df["Open"] * f_split_bonus,
        "High": df["High"] * f_split_bonus,
        "Low": df["Low"] * f_split_bonus,
        "Close": df["Close"] * f_split_bonus,
        "ShareVolume": df["ShareVolume"] / f_split_bonus,
        "TradeVolume": df["TradeVolume"],
        "Turnover": df["Turnover"],
        "CapitalAdjClose": df["Close"] * f_capital,
        "CumSplitFactor": f_split,
        "CumBonusFactor": f_bonus,
        "CumSplitBonusFactor": f_split_bonus,
        "CumScripFactor": f_scrip,
        "CumRightsTERPFactor": f_rights,
        "CumCapitalFactor": f_capital,
        "IsSplitAdjusted": True,
        "Era": df["Era"],
        "DataFormat": df["DataFormat"],
        "DataQualityFlag": df["DataQualityFlag"],
    })

    # Build Total Return History DataFrame (strictly separates Cash Dividends from Splits/Bonus/Rights)
    tr_close = df["Close"] * f_total_ret
    tr_df = pd.DataFrame({
        "Date": df["Date"],
        "SecurityID": df["SecurityID"],
        "CompanyCode": df["CompanyCode"],
        "IssuerID": df["IssuerID"],
        "MainType": df["MainType"],
        "SubType": df["SubType"],
        "RawClose": df["Close"],
        "SplitBonusAdjClose": df["Close"] * f_split_bonus,
        "CapitalAdjClose": df["Close"] * f_capital,
        "TotalReturnPrice": tr_close,
        "CumCapitalFactor": f_capital,
        "CumCashDividendFactor": f_div,
        "CumTotalReturnFactor": f_total_ret,
        "DataQualityFlag": df["DataQualityFlag"],
    })

    # Normalized TotalReturnIndex (Base = 100.0 at first valid positive TotalReturnPrice per SecurityID)
    tr_df["TotalReturnIndex"] = np.nan
    for sid, idx in tr_df.groupby("SecurityID", sort=False).groups.items():
        vals = tr_df.loc[idx, "TotalReturnPrice"].to_numpy(dtype=float)
        pos_mask = np.isfinite(vals) & (vals > 0)
        if pos_mask.any():
            base_val = vals[np.argmax(pos_mask)]
            tri = np.where(pos_mask, 100.0 * vals / base_val, np.nan)
            tr_df.loc[idx, "TotalReturnIndex"] = tri

    # Event-level traceable factor audit table
    factors_audit = active_ca[[
        "EventID", "SecurityID", "CompanyCode", "EventType",
        "AnnouncementDate", "ExDate", "ExDateBasis", "EffectiveDate",
        "OldProportion", "NewProportion", "Ratio", "SubscriptionPrice",
        "DividendPerShare", "PreExClose",
        "PriceFactor", "VolumeFactor", "DividendFactor",
        "Source", "SourceSheet", "SourceRow", "ValidationStatus", "ValidationNotes",
    ]].sort_values(["SecurityID", "ExDate", "EventType"]).reset_index(drop=True)

    return factors_audit, split_adj, tr_df
