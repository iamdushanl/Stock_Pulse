"""
Phase 2A Core Technical Indicator & Parabolic SAR Computation Engine
==================================================================
Computes 63 strictly causal technical, price-volume, and market-relative
indicators independently per SecurityID across the full CSE universe.

Design invariants:
  1. Zero future leakage: bar t depends ONLY on bars [0..t] of the same SecurityID
     (and market/sector indices on Date <= Date_t).
  2. Strict warm-up NaN preservation (never shrinks windows below documented minima).
  3. Zero fabrication of missing OHLC/Volume values (pre-2017 Open -> NaN;
     pre-2002 High/Low -> NaN; pre-2001 Volume -> NaN; Close<=0 -> NaN for math).
  4. Corporate-action consistency via Phase 1B split/bonus-adjusted OHLCV and
     CapitalAdjClose (splits + bonus + scrip + rights TERP) for relative returns.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Fast Vectorized / Contiguous Array Primitives
# ---------------------------------------------------------------------------
def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    valid = np.isfinite(arr)
    safe = np.where(valid, arr, 0.0)
    cs[1:] = np.cumsum(safe)
    vc = np.empty(n + 1, dtype=np.int64)
    vc[0] = 0
    vc[1:] = np.cumsum(valid)
    win_valid = (vc[window:] - vc[:-window]) == window
    sums = cs[window:] - cs[:-window]
    res = out[window - 1:]
    res[win_valid] = sums[win_valid] / float(window)
    return out


def _rolling_std(arr: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    wins = sliding_window_view(arr, window)
    finite_rows = np.all(np.isfinite(wins), axis=1)
    if finite_rows.any():
        out[window - 1:][finite_rows] = np.std(wins[finite_rows], axis=1, ddof=ddof)
    return out


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    wins = sliding_window_view(arr, window)
    finite_rows = np.all(np.isfinite(wins), axis=1)
    if finite_rows.any():
        out[window - 1:][finite_rows] = np.max(wins[finite_rows], axis=1)
    return out


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    wins = sliding_window_view(arr, window)
    finite_rows = np.all(np.isfinite(wins), axis=1)
    if finite_rows.any():
        out[window - 1:][finite_rows] = np.min(wins[finite_rows], axis=1)
    return out


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    valid = np.isfinite(arr)
    safe = np.where(valid, arr, 0.0)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    cs[1:] = np.cumsum(safe)
    vc = np.empty(n + 1, dtype=np.int64)
    vc[0] = 0
    vc[1:] = np.cumsum(valid)
    win_valid = (vc[window:] - vc[:-window]) == window
    sums = cs[window:] - cs[:-window]
    res = out[window - 1:]
    res[win_valid] = sums[win_valid]
    return out


def _rolling_wma(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    w = np.arange(1, window + 1, dtype=np.float64)
    w /= w.sum()
    wins = sliding_window_view(arr, window)
    finite_rows = np.all(np.isfinite(wins), axis=1)
    if finite_rows.any():
        out[window - 1:][finite_rows] = wins[finite_rows] @ w
    return out


def _rolling_cci(tp: np.ndarray, window: int = 20) -> np.ndarray:
    n = len(tp)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    wins = sliding_window_view(tp, window)
    finite_rows = np.all(np.isfinite(wins), axis=1)
    if not finite_rows.any():
        return out
    w_valid = wins[finite_rows]
    sma = w_valid.mean(axis=1)
    mad = np.abs(w_valid - sma[:, None]).mean(axis=1)
    cur = w_valid[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(mad > 1e-12, (cur - sma) / (0.015 * mad), 0.0)
    out[window - 1:][finite_rows] = cci
    return out


def _rolling_aroon(high: np.ndarray, low: np.ndarray, lookback: int = 25) -> tuple[np.ndarray, np.ndarray]:
    n = len(high)
    up = np.full(n, np.nan, dtype=np.float64)
    down = np.full(n, np.nan, dtype=np.float64)
    win_len = lookback + 1  # [t-lookback .. t] inclusive -> 26 bars
    if n < win_len:
        return up, down
    w_h = sliding_window_view(high, win_len)
    w_l = sliding_window_view(low, win_len)
    f_h = np.all(np.isfinite(w_h), axis=1)
    f_l = np.all(np.isfinite(w_l), axis=1)
    if f_h.any():
        # Reverse slice picks most recent extreme on ties
        bars_since_high = np.argmax(w_h[f_h][:, ::-1], axis=1)
        up[lookback:][f_h] = 100.0 * (lookback - bars_since_high) / float(lookback)
    if f_l.any():
        bars_since_low = np.argmin(w_l[f_l][:, ::-1], axis=1)
        down[lookback:][f_l] = 100.0 * (lookback - bars_since_low) / float(lookback)
    return up, down


def _ema_causal(arr: np.ndarray, span: int) -> np.ndarray:
    """
    Strictly causal EMA seeded by SMA of the first `span` consecutive valid bars.
    Outputs NaN on missing bars; re-seeds cleanly after gaps > 10 consecutive NaNs.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < span:
        return out
    alpha = 2.0 / (span + 1.0)
    vals = arr.tolist()
    buf: list[float] = []
    ema_val: Optional[float] = None
    nan_streak = 0
    for i, x in enumerate(vals):
        if x is None or not math.isfinite(x):
            nan_streak += 1
            if nan_streak > 10:
                ema_val = None
                buf.clear()
            continue
        nan_streak = 0
        if ema_val is None:
            buf.append(x)
            if len(buf) == span:
                ema_val = sum(buf) / float(span)
                out[i] = ema_val
        else:
            ema_val = alpha * x + (1.0 - alpha) * ema_val
            out[i] = ema_val
    return out


def _wilder_rma_causal(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's RMA (alpha = 1 / period), seeded strictly by SMA of the first
    `period` consecutive valid observations. Outputs NaN on warm-up or missing inputs.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    inv_p = 1.0 / float(period)
    one_minus = 1.0 - inv_p
    vals = arr.tolist()
    buf: list[float] = []
    rma_val: Optional[float] = None
    nan_streak = 0
    for i, x in enumerate(vals):
        if x is None or not math.isfinite(x):
            nan_streak += 1
            if nan_streak > 10:
                rma_val = None
                buf.clear()
            continue
        nan_streak = 0
        if rma_val is None:
            buf.append(x)
            if len(buf) == period:
                rma_val = sum(buf) / float(period)
                out[i] = rma_val
        else:
            rma_val = x * inv_p + rma_val * one_minus
            out[i] = rma_val
    return out


# ---------------------------------------------------------------------------
# Public Parabolic SAR Function (Strictly Causal, Tested vs Reference)
# ---------------------------------------------------------------------------
def compute_parabolic_sar(
    df_sec: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.DataFrame:
    """
    Compute Wilder's Parabolic SAR strictly causally for a single chronological
    SecurityID price series.

    Parameters
    ----------
    df_sec : DataFrame sorted by Date containing High, Low, Close.
    af_start : Initial acceleration factor (0.02).
    af_step : Acceleration increment per new extreme point (0.02).
    af_max : Maximum acceleration factor (0.20).

    Notes
    -----
    - Bar 0 of any contiguous valid (High, Low, Close>0) sequence returns NaN
      (eliminating row-0 look-ahead bias where Close[1] vs Close[0] is not yet known).
    - Bar 1 initializes causally from [0, 1] using trend = +1 if Close[1] >= Close[0] else -1.
    - Isolated missing H/L bars leave PSAR=NaN on those rows; gaps > 10 bars reset state
      (so pre-2002 missing H/L remains 100% NaN and 2002+ initializes cleanly).
    """
    h = pd.to_numeric(df_sec["High"], errors="coerce").to_numpy(dtype=np.float64)
    l = pd.to_numeric(df_sec["Low"], errors="coerce").to_numpy(dtype=np.float64)
    c = pd.to_numeric(df_sec["Close"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(c)

    psar = np.full(n, np.nan, dtype=np.float64)
    trend = np.full(n, np.nan, dtype=np.float64)
    rev = np.full(n, np.nan, dtype=np.float64)

    valid = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    hl = h.tolist()
    ll = l.tolist()
    cl = c.tolist()
    vl = valid.tolist()

    # Track contiguous valid bars via small history ring so isolated <=5 bar holes stay NaN safely
    seg_v_idx: list[int] = []
    nan_streak = 0
    is_up = True
    cur_sar = 0.0
    ep = 0.0
    af = af_start
    prev_l1 = 0.0
    prev_l2 = 0.0
    prev_h1 = 0.0
    prev_h2 = 0.0

    for i in range(n):
        if not vl[i]:
            nan_streak += 1
            if nan_streak > 10:
                seg_v_idx.clear()
            continue
        nan_streak = 0
        hi, lo, ci = hl[i], ll[i], cl[i]
        seg_pos = len(seg_v_idx)
        seg_v_idx.append(i)

        if seg_pos == 0:
            # Bar 0 of valid segment: warm-up NaN (zero look-ahead)
            prev_l1 = lo
            prev_h1 = hi
            prev_c0 = ci
            continue

        if seg_pos == 1:
            # Bar 1: causally compare bar 1 vs bar 0
            is_up = ci >= prev_c0
            cur_sar = min(prev_l1, lo) if is_up else max(prev_h1, hi)
            ep = max(prev_h1, hi) if is_up else min(prev_l1, lo)
            af = af_start
            psar[i] = cur_sar
            trend[i] = 1.0 if is_up else -1.0
            rev[i] = 0.0
            prev_l2, prev_l1 = prev_l1, lo
            prev_h2, prev_h1 = prev_h1, hi
            continue

        # Bar >= 2: standard Wilder recurrence
        cand = cur_sar + af * (ep - cur_sar)
        if is_up:
            cand = min(cand, prev_l1, prev_l2)
            if lo < cand:
                is_up = False
                cur_sar = max(ep, hi)
                ep = lo
                af = af_start
                rev[i] = -1.0
            else:
                cur_sar = cand
                rev[i] = 0.0
                if hi > ep:
                    ep = hi
                    af = min(af + af_step, af_max)
        else:
            cand = max(cand, prev_h1, prev_h2)
            if hi > cand:
                is_up = True
                cur_sar = min(ep, lo)
                ep = hi
                af = af_start
                rev[i] = 1.0
            else:
                cur_sar = cand
                rev[i] = 0.0
                if lo < ep:
                    ep = lo
                    af = min(af + af_step, af_max)

        psar[i] = cur_sar
        trend[i] = 1.0 if is_up else -1.0
        prev_l2, prev_l1 = prev_l1, lo
        prev_h2, prev_h1 = prev_h1, hi

    with np.errstate(divide="ignore", invalid="ignore"):
        dist_pct = np.where(np.isfinite(psar) & valid & (c > 0), 100.0 * (c - psar) / c, np.nan)

    return pd.DataFrame(
        {
            "PSAR": psar,
            "PSAR_Trend": trend,
            "PSAR_Distance_Pct": dist_pct,
            "PSAR_Reversal": rev,
        },
        index=df_sec.index,
    )


def _compute_supertrend(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 10, mult: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute causal Supertrend level, direction (+1/-1), and reversal (+1/-1/0)."""
    n = len(c)
    st_val = np.full(n, np.nan, dtype=np.float64)
    st_dir = np.full(n, np.nan, dtype=np.float64)
    st_rev = np.full(n, np.nan, dtype=np.float64)

    valid_hlc = np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    prev_c = np.empty(n, dtype=np.float64)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    with np.errstate(invalid="ignore"):
        tr = np.where(
            valid_hlc & np.isfinite(prev_c) & (prev_c > 0),
            np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c))),
            np.nan,
        )
    atr = _wilder_rma_causal(tr, period)
    hl2 = 0.5 * (h + l)
    basic_up = hl2 + mult * atr
    basic_dn = hl2 - mult * atr

    bu = basic_up.tolist()
    bd = basic_dn.tolist()
    cl = c.tolist()
    final_up = np.nan
    final_dn = np.nan
    dir_prev = 0
    nan_streak = 0

    for i in range(n):
        bui, bdi, ci = bu[i], bd[i], cl[i]
        if bui is None or bdi is None or not (math.isfinite(bui) and math.isfinite(bdi) and math.isfinite(ci)):
            nan_streak += 1
            if nan_streak > 10:
                final_up = np.nan
                final_dn = np.nan
                dir_prev = 0
            continue
        nan_streak = 0

        if dir_prev == 0:
            final_up = bui
            final_dn = bdi
            dir_cur = 1 if ci >= bdi else -1
            dir_prev = dir_cur
            st_dir[i] = float(dir_cur)
            st_val[i] = final_dn if dir_cur == 1 else final_up
            # No reversal defined on initial bar
            continue

        prev_c_val = cl[i - 1] if (i > 0 and cl[i - 1] is not None and math.isfinite(cl[i - 1])) else ci
        final_up = bui if (bui < final_up or prev_c_val > final_up) else final_up
        final_dn = bdi if (bdi > final_dn or prev_c_val < final_dn) else final_dn

        if dir_prev == 1 and ci < final_dn:
            dir_cur = -1
            st_rev[i] = -1.0
        elif dir_prev == -1 and ci > final_up:
            dir_cur = 1
            st_rev[i] = 1.0
        else:
            dir_cur = dir_prev
            st_rev[i] = 0.0

        dir_prev = dir_cur
        st_dir[i] = float(dir_cur)
        st_val[i] = final_dn if dir_cur == 1 else final_up

    return st_val, st_dir, st_rev


# ---------------------------------------------------------------------------
# Per-Security Core Computation (Technical + Price-Volume)
# ---------------------------------------------------------------------------
def _compute_single_security_features(df_sec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Table 1 (Technical) and Table 2 (Price-Volume) features for 1 SecurityID."""
    n = len(df_sec)
    # Extract clean positive OHLCV arrays without modifying source dataframe
    raw_c = pd.to_numeric(df_sec["Close"], errors="coerce").to_numpy(dtype=np.float64)
    c = np.where(np.isfinite(raw_c) & (raw_c > 0), raw_c, np.nan)

    raw_h = pd.to_numeric(df_sec["High"], errors="coerce").to_numpy(dtype=np.float64)
    raw_l = pd.to_numeric(df_sec["Low"], errors="coerce").to_numpy(dtype=np.float64)
    # Require H >= L and strictly finite positive prices
    valid_hl = np.isfinite(raw_h) & np.isfinite(raw_l) & np.isfinite(c) & (raw_h > 0) & (raw_l > 0) & (raw_h >= raw_l)
    h = np.where(valid_hl, raw_h, np.nan)
    l = np.where(valid_hl, raw_l, np.nan)

    raw_o = pd.to_numeric(df_sec["Open"], errors="coerce").to_numpy(dtype=np.float64)
    o = np.where(np.isfinite(raw_o) & (raw_o > 0), raw_o, np.nan)

    raw_v = pd.to_numeric(df_sec["ShareVolume"], errors="coerce").to_numpy(dtype=np.float64)
    v = np.where(np.isfinite(raw_v) & (raw_v >= 0), raw_v, np.nan)

    raw_to = pd.to_numeric(df_sec["Turnover"], errors="coerce").to_numpy(dtype=np.float64)
    to = np.where(np.isfinite(raw_to) & (raw_to >= 0), raw_to, np.nan)

    prev_c = np.empty(n, dtype=np.float64)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    # ── A. Trend ─────────────────────────────────────────────────────────────
    sma_20 = _rolling_mean(c, 20)
    sma_50 = _rolling_mean(c, 50)
    sma_200 = _rolling_mean(c, 200)
    ema_12 = _ema_causal(c, 12)
    ema_26 = _ema_causal(c, 26)
    ema_20 = _ema_causal(c, 20)
    wma_20 = _rolling_wma(c, 20)

    # Wilder True Range & ADX(14)
    with np.errstate(invalid="ignore"):
        tr = np.where(
            valid_hl & np.isfinite(prev_c),
            np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c))),
            np.nan,
        )
    atr_14 = _wilder_rma_causal(tr, 14)
    atr_10 = _wilder_rma_causal(tr, 10)

    prev_h = np.empty(n, dtype=np.float64)
    prev_l = np.empty(n, dtype=np.float64)
    prev_h[0] = np.nan
    prev_l[0] = np.nan
    prev_h[1:] = h[:-1]
    prev_l[1:] = l[:-1]

    valid_hl_pair = valid_hl & np.isfinite(prev_h) & np.isfinite(prev_l) & np.isfinite(prev_c)
    with np.errstate(invalid="ignore"):
        up_m = h - prev_h
        dn_m = prev_l - l
        plus_dm = np.where(valid_hl_pair, np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), np.nan)
        minus_dm = np.where(valid_hl_pair, np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), np.nan)

    sm_dm_plus = _wilder_rma_causal(plus_dm, 14)
    sm_dm_minus = _wilder_rma_causal(minus_dm, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(np.isfinite(atr_14) & (atr_14 > 0), 100.0 * sm_dm_plus / atr_14, np.nan)
        mdi = np.where(np.isfinite(atr_14) & (atr_14 > 0), 100.0 * sm_dm_minus / atr_14, np.nan)
        di_sum = pdi + mdi
        dx = np.where(np.isfinite(di_sum) & (di_sum > 0), 100.0 * np.abs(pdi - mdi) / di_sum, np.where(np.isfinite(di_sum), 0.0, np.nan))
    adx_14 = _wilder_rma_causal(dx, 14)

    aroon_up_25, aroon_dn_25 = _rolling_aroon(h, l, lookback=25)

    # Causal Ichimoku (no Chikou future shift; Senkou spans shifted +26 FORWARD onto active bar t)
    tenkan_9 = 0.5 * (_rolling_max(h, 9) + _rolling_min(l, 9))
    kijun_26 = 0.5 * (_rolling_max(h, 26) + _rolling_min(l, 26))
    span_a_raw = 0.5 * (tenkan_9 + kijun_26)
    span_b_raw = 0.5 * (_rolling_max(h, 52) + _rolling_min(l, 52))
    senkou_a = np.full(n, np.nan, dtype=np.float64)
    senkou_b = np.full(n, np.nan, dtype=np.float64)
    if n > 26:
        senkou_a[26:] = span_a_raw[:-26]
        senkou_b[26:] = span_b_raw[:-26]

    with np.errstate(divide="ignore", invalid="ignore"):
        ichi_tk_pct = np.where(np.isfinite(tenkan_9) & np.isfinite(kijun_26) & np.isfinite(c), 100.0 * (tenkan_9 - kijun_26) / c, np.nan)
        k_top = np.maximum(senkou_a, senkou_b)
        k_bot = np.minimum(senkou_a, senkou_b)
        ichi_kumo_pos = np.where(
            np.isfinite(k_top) & np.isfinite(k_bot) & np.isfinite(c),
            np.where(c > k_top, 100.0 * (c - k_top) / c, np.where(c < k_bot, 100.0 * (c - k_bot) / c, 0.0)),
            np.nan,
        )

    supertrend_val, supertrend_dir, supertrend_rev = _compute_supertrend(h, l, c, period=10, mult=3.0)

    # ── B. Momentum ──────────────────────────────────────────────────────────
    diff_c = np.full(n, np.nan, dtype=np.float64)
    diff_c[1:] = c[1:] - c[:-1]
    gains = np.where(np.isfinite(diff_c), np.maximum(diff_c, 0.0), np.nan)
    losses = np.where(np.isfinite(diff_c), np.maximum(-diff_c, 0.0), np.nan)
    avg_g14 = _wilder_rma_causal(gains, 14)
    avg_l14 = _wilder_rma_causal(losses, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi_14 = np.where(
            np.isfinite(avg_g14) & np.isfinite(avg_l14),
            np.where(
                avg_l14 > 0,
                100.0 - 100.0 / (1.0 + avg_g14 / avg_l14),
                np.where(avg_g14 > 0, 100.0, 50.0),
            ),
            np.nan,
        )

    macd_line = ema_12 - ema_26
    macd_signal = _ema_causal(macd_line, 9)
    macd_hist = macd_line - macd_signal

    roc_10 = np.full(n, np.nan, dtype=np.float64)
    roc_20 = np.full(n, np.nan, dtype=np.float64)
    mom_10 = np.full(n, np.nan, dtype=np.float64)
    if n > 10:
        with np.errstate(divide="ignore", invalid="ignore"):
            roc_10[10:] = 100.0 * (c[10:] / c[:-10] - 1.0)
            mom_10[10:] = c[10:] - c[:-10]
    if n > 20:
        with np.errstate(divide="ignore", invalid="ignore"):
            roc_20[20:] = 100.0 * (c[20:] / c[:-20] - 1.0)

    tp = (h + l + c) / 3.0
    cci_20 = _rolling_cci(tp, 20)

    low_14 = _rolling_min(l, 14)
    high_14 = _rolling_max(h, 14)
    hl_rng14 = high_14 - low_14
    with np.errstate(divide="ignore", invalid="ignore"):
        # Clamp strictly to oscillator domain when odd-lot prints leave Close marginally outside H/L range
        williams_r_14 = np.where(
            np.isfinite(hl_rng14) & (hl_rng14 > 0) & np.isfinite(c),
            np.clip(-100.0 * (high_14 - c) / hl_rng14, -100.0, 0.0),
            np.nan,
        )
        stoch_k_14 = np.where(
            np.isfinite(hl_rng14) & (hl_rng14 > 0) & np.isfinite(c),
            np.clip(100.0 * (c - low_14) / hl_rng14, 0.0, 100.0),
            np.nan,
        )
    stoch_d_3 = _rolling_mean(stoch_k_14, 3)

    # ── C. Volatility ────────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        natr_14 = np.where(np.isfinite(atr_14) & np.isfinite(c), 100.0 * atr_14 / c, np.nan)
    std_20 = _rolling_std(c, 20, ddof=0)
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_width = np.where(np.isfinite(bb_upper) & (sma_20 > 0), (bb_upper - bb_lower) / sma_20, np.nan)
        bb_span = bb_upper - bb_lower
        bb_pctb = np.where(np.isfinite(bb_span) & (bb_span > 0), (c - bb_lower) / bb_span, np.nan)
        keltner_pos = np.where(
            np.isfinite(ema_20) & np.isfinite(atr_10) & (atr_10 > 0) & np.isfinite(c),
            (c - (ema_20 - 2.0 * atr_10)) / (4.0 * atr_10),
            np.nan,
        )

    log_ret = np.full(n, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(c[1:] / c[:-1])
    hist_vol_21d = _rolling_std(log_ret, 21, ddof=1) * math.sqrt(252.0)

    high_20 = _rolling_max(h, 20)
    low_20 = _rolling_min(l, 20)
    hl_rng20 = high_20 - low_20
    with np.errstate(divide="ignore", invalid="ignore"):
        donchian_pos = np.where(
            np.isfinite(hl_rng20) & (hl_rng20 > 0) & np.isfinite(c),
            np.clip((c - low_20) / hl_rng20, 0.0, 1.0),
            np.nan,
        )

    # ── E. Trend Reversal & PSAR ─────────────────────────────────────────────
    psar_df = compute_parabolic_sar(df_sec, af_start=0.02, af_step=0.02, af_max=0.20)

    with np.errstate(divide="ignore", invalid="ignore"):
        ma_cross_spread = np.where(np.isfinite(sma_20) & np.isfinite(sma_50) & (sma_50 > 0), 100.0 * (sma_20 - sma_50) / sma_50, np.nan)

    ma_cross_sig = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        spread_now = sma_20[1:] - sma_50[1:]
        spread_prev = sma_20[:-1] - sma_50[:-1]
        both_valid = np.isfinite(spread_now) & np.isfinite(spread_prev)
        sig_slice = np.where(
            both_valid,
            np.where(
                (spread_prev <= 0) & (spread_now > 0),
                1.0,
                np.where((spread_prev >= 0) & (spread_now < 0), -1.0, 0.0),
            ),
            np.nan,
        )
        ma_cross_sig[1:] = sig_slice

    # 100% Causal RSI Divergence (25-bar lookback split into recent 5 bars vs prior 20 bars)
    rsi_bull_div = np.full(n, np.nan, dtype=np.float64)
    rsi_bear_div = np.full(n, np.nan, dtype=np.float64)
    if n >= 26:
        c_wins = sliding_window_view(c, 26)
        r_wins = sliding_window_view(rsi_14, 26)
        fin = np.all(np.isfinite(c_wins), axis=1) & np.all(np.isfinite(r_wins), axis=1)
        if fin.any():
            cw = c_wins[fin]
            rw = r_wins[fin]
            prior_c, rec_c, cur_c = cw[:, :21], cw[:, 21:], cw[:, -1]
            prior_r, cur_r = rw[:, :21], rw[:, -1]
            p_min_c = prior_c.min(axis=1)
            r_min_c = rec_c.min(axis=1)
            p_min_r = prior_r.min(axis=1)
            p_max_c = prior_c.max(axis=1)
            r_max_c = rec_c.max(axis=1)
            p_max_r = prior_r.max(axis=1)

            bull = (cur_c <= r_min_c) & (cur_c < p_min_c) & (cur_r > p_min_r + 3.0) & (p_min_r < 45.0)
            bear = (cur_c >= r_max_c) & (cur_c > p_max_c) & (cur_r < p_max_r - 3.0) & (p_max_r > 55.0)
            rsi_bull_div[25:][fin] = bull.astype(np.float64)
            rsi_bear_div[25:][fin] = bear.astype(np.float64)

    # ── D. Volume & Open-Aware Microstructure ────────────────────────────────
    # OBV (strictly NaN pre-2001 / where ShareVolume is NaN)
    obv = np.full(n, np.nan, dtype=np.float64)
    obv_acc = 0.0
    for i in range(1, n):
        ci, pci, vi = c[i], c[i - 1], v[i]
        if math.isfinite(ci) and math.isfinite(pci) and math.isfinite(vi):
            if ci > pci:
                obv_acc += vi
            elif ci < pci:
                obv_acc -= vi
            obv[i] = obv_acc

    # MFI(14)
    raw_mf = tp * v
    tp_prev = np.empty(n, dtype=np.float64)
    tp_prev[0] = np.nan
    tp_prev[1:] = tp[:-1]
    valid_mfi_bar = np.isfinite(tp) & np.isfinite(tp_prev) & np.isfinite(v)
    pos_mf = np.where(valid_mfi_bar, np.where(tp > tp_prev, raw_mf, 0.0), np.nan)
    neg_mf = np.where(valid_mfi_bar, np.where(tp < tp_prev, raw_mf, 0.0), np.nan)
    sum_pos_mf = _rolling_sum(pos_mf, 14)
    sum_neg_mf = _rolling_sum(neg_mf, 14)
    tot_mf = sum_pos_mf + sum_neg_mf
    with np.errstate(divide="ignore", invalid="ignore"):
        mfi_14 = np.where(np.isfinite(tot_mf) & (tot_mf > 0), 100.0 * sum_pos_mf / tot_mf, np.nan)

    # CLV & CMF(20) & AD Line & Chaikin Oscillator
    with np.errstate(divide="ignore", invalid="ignore"):
        clv = np.where(
            valid_hl & np.isfinite(c),
            np.where((h - l) > 0, np.clip(((c - l) - (h - c)) / (h - l), -1.0, 1.0), 0.0),
            np.nan,
        )
    mfv = np.where(np.isfinite(clv) & np.isfinite(v), clv * v, np.nan)
    v_for_cmf = np.where(np.isfinite(clv) & np.isfinite(v), v, np.nan)
    sum_mfv_20 = _rolling_sum(mfv, 20)
    sum_v_20_cmf = _rolling_sum(v_for_cmf, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmf_20 = np.where(np.isfinite(sum_mfv_20) & (sum_v_20_cmf > 0), sum_mfv_20 / sum_v_20_cmf, np.nan)

    ad_line = np.full(n, np.nan, dtype=np.float64)
    ad_acc = 0.0
    for i in range(n):
        mfvi = mfv[i]
        if math.isfinite(mfvi):
            ad_acc += mfvi
            ad_line[i] = ad_acc
    chaikin_osc = _ema_causal(ad_line, 3) - _ema_causal(ad_line, 10)

    # VWAP(20), Volume SMA(20), Volume Ratio(20)
    cv = np.where(np.isfinite(c) & np.isfinite(v), c * v, np.nan)
    v_valid_c = np.where(np.isfinite(c) & np.isfinite(v), v, np.nan)
    sum_cv_20 = _rolling_sum(cv, 20)
    sum_v_20 = _rolling_sum(v_valid_c, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_20 = np.where(np.isfinite(sum_cv_20) & (sum_v_20 > 0), sum_cv_20 / sum_v_20, np.nan)
    vol_sma_20 = _rolling_mean(v, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio_20 = np.where(np.isfinite(v) & np.isfinite(vol_sma_20) & (vol_sma_20 > 0), v / vol_sma_20, np.nan)

    # Returns & Amihud Illiquidity & Open-dependent indicators (strict NaN when Open missing!)
    ret_1d_split = np.full(n, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_1d_split[1:] = 100.0 * (c[1:] / c[:-1] - 1.0)
        amihud_daily = np.where(
            np.isfinite(ret_1d_split) & np.isfinite(to) & (to > 0),
            (np.abs(ret_1d_split) / 100.0) / (to / 1e6),
            np.nan,
        )
    amihud_20 = _rolling_mean(amihud_daily, 20)

    cap_c_raw = pd.to_numeric(df_sec["CapitalAdjClose"], errors="coerce").to_numpy(dtype=np.float64)
    cap_c = np.where(np.isfinite(cap_c_raw) & (cap_c_raw > 0), cap_c_raw, np.nan)
    ret_1d_cap = np.full(n, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_1d_cap[1:] = 100.0 * (cap_c[1:] / cap_c[:-1] - 1.0)
        overnight_gap = np.where(np.isfinite(o) & np.isfinite(prev_c), 100.0 * (o - prev_c) / prev_c, np.nan)
        intraday_ret = np.where(np.isfinite(o) & np.isfinite(c), 100.0 * (c - o) / o, np.nan)

    tech_df = pd.DataFrame(
        {
            "SMA_20": sma_20,
            "SMA_50": sma_50,
            "SMA_200": sma_200,
            "EMA_12": ema_12,
            "EMA_26": ema_26,
            "WMA_20": wma_20,
            "ADX_14": adx_14,
            "Aroon_Up_25": aroon_up_25,
            "Aroon_Down_25": aroon_dn_25,
            "Ichimoku_TK_Spread_Pct": ichi_tk_pct,
            "Ichimoku_Kumo_Pos_Pct": ichi_kumo_pos,
            "Supertrend_10_3": supertrend_val,
            "RSI_14": rsi_14,
            "MACD_Line": macd_line,
            "MACD_Signal": macd_signal,
            "MACD_Hist": macd_hist,
            "ROC_10": roc_10,
            "ROC_20": roc_20,
            "CCI_20": cci_20,
            "Williams_R_14": williams_r_14,
            "Stoch_K_14": stoch_k_14,
            "Stoch_D_3": stoch_d_3,
            "MOM_10": mom_10,
            "ATR_14": atr_14,
            "NATR_14": natr_14,
            "BB_Upper_20_2": bb_upper,
            "BB_Lower_20_2": bb_lower,
            "BB_Width_20_2": bb_width,
            "BB_PctB_20_2": bb_pctb,
            "Keltner_Pos_20": keltner_pos,
            "Hist_Vol_21d": hist_vol_21d,
            "Donchian_Pos_20": donchian_pos,
            "PSAR": psar_df["PSAR"].to_numpy(),
            "PSAR_Trend": psar_df["PSAR_Trend"].to_numpy(),
            "PSAR_Distance_Pct": psar_df["PSAR_Distance_Pct"].to_numpy(),
            "PSAR_Reversal": psar_df["PSAR_Reversal"].to_numpy(),
            "Supertrend_Dir_10_3": supertrend_dir,
            "Supertrend_Reversal": supertrend_rev,
            "MA_Cross_Spread_20_50": ma_cross_spread,
            "MA_Cross_Signal_20_50": ma_cross_sig,
            "RSI_Div_Bullish_14": rsi_bull_div,
            "RSI_Div_Bearish_14": rsi_bear_div,
        },
        index=df_sec.index,
    )

    pv_df = pd.DataFrame(
        {
            "SplitAdjClose": df_sec["Close"].to_numpy(dtype=np.float64),
            "CapitalAdjClose": df_sec["CapitalAdjClose"].to_numpy(dtype=np.float64),
            "Daily_Return_SplitAdj": ret_1d_split,
            "Daily_Return_CapAdj": ret_1d_cap,
            "OBV": obv,
            "MFI_14": mfi_14,
            "CMF_20": cmf_20,
            "VWAP_20": vwap_20,
            "Volume_SMA_20": vol_sma_20,
            "Volume_Ratio_20": vol_ratio_20,
            "AD_Line": ad_line,
            "Chaikin_Osc_3_10": chaikin_osc,
            "Amihud_Illiquidity_20": amihud_20,
            "Overnight_Gap_Ret": overnight_gap,
            "Intraday_Ret": intraday_ret,
        },
        index=df_sec.index,
    )
    return tech_df, pv_df


# ---------------------------------------------------------------------------
# Full Panel Builders
# ---------------------------------------------------------------------------
def compute_technical_features_panel(df_adj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute `cse_technical_features` and `cse_price_volume_features` across
    all SecurityIDs independently, maintaining SecurityID, CompanyCode, Date,
    MainType, SubType.
    """
    req = ["Date", "SecurityID", "CompanyCode", "MainType", "SubType", "Open", "High", "Low", "Close", "ShareVolume", "Turnover", "CapitalAdjClose"]
    missing_cols = [c for c in req if c not in df_adj.columns]
    if missing_cols:
        raise KeyError(f"Input split-adjusted dataframe missing required columns: {missing_cols}")

    work = df_adj.sort_values(["SecurityID", "Date"]).reset_index(drop=True)
    key_df = work[["SecurityID", "CompanyCode", "Date", "MainType", "SubType"]].copy()

    tech_parts: list[pd.DataFrame] = []
    pv_parts: list[pd.DataFrame] = []

    for _, grp in work.groupby("SecurityID", sort=False):
        t_part, p_part = _compute_single_security_features(grp)
        tech_parts.append(t_part)
        pv_parts.append(p_part)

    tech_body = pd.concat(tech_parts, axis=0)
    pv_body = pd.concat(pv_parts, axis=0)

    tech_full = pd.concat([key_df, tech_body], axis=1)
    pv_full = pd.concat([key_df, pv_body], axis=1)
    return tech_full, pv_full


def _prepare_market_benchmark_frames(
    market_indices: pd.DataFrame,
    gics_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build market-calendar-ordered ASPI momentum + cross-sector breadth series."""
    mi = market_indices.copy()
    mi["Date"] = pd.to_datetime(mi["Date"]).dt.normalize()
    mi = mi.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    aspi = pd.to_numeric(mi["ASPI"], errors="coerce")
    aspi_mom20 = 100.0 * (aspi / aspi.shift(20) - 1.0)
    aspi_mom60 = 100.0 * (aspi / aspi.shift(60) - 1.0)

    # Sector breadth momentum: legacy sectors pre-2020; GICS sectors from 2020-01-01 onward
    exclude_cols = {"Date", "ASPI", "MPI", "SP_SL20"}
    legacy_cols = [c for c in mi.columns if c not in exclude_cols and not str(c).startswith("Unknown_")]
    legacy_lvl = mi[legacy_cols].apply(pd.to_numeric, errors="coerce")
    legacy_lvl = legacy_lvl.where(legacy_lvl > 0)
    legacy_mom = 100.0 * (legacy_lvl / legacy_lvl.shift(20) - 1.0)
    legacy_breadth = legacy_mom.mean(axis=1, skipna=True)

    breadth_map = pd.DataFrame({
        "Date": mi["Date"],
        "ASPI_Close": aspi,
        "ASPI_MOM_20": aspi_mom20,
        "ASPI_MOM_60": aspi_mom60,
        "Legacy_Breadth_20": legacy_breadth,
    })

    if gics_df is not None and not gics_df.empty:
        gd = gics_df.copy()
        gd["Date"] = pd.to_datetime(gd["Date"]).dt.normalize()
        gd = gd.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        g_cols = [c for c in gd.columns if c != "Date"]
        g_lvl = gd[g_cols].apply(pd.to_numeric, errors="coerce")
        g_lvl = g_lvl.where(g_lvl > 0)
        g_mom = 100.0 * (g_lvl / g_lvl.shift(20) - 1.0)
        g_df = pd.DataFrame({"Date": gd["Date"], "GICS_Breadth_20": g_mom.mean(axis=1, skipna=True)})
        breadth_map = breadth_map.merge(g_df, on="Date", how="left")
    else:
        breadth_map["GICS_Breadth_20"] = np.nan

    # Regime-consistent switch at Jan 1, 2020 (with graceful fallback if a GICS print is missing)
    breadth_map["Sector_Breadth_MOM_20"] = np.where(
        breadth_map["Date"] >= pd.Timestamp("2020-01-01"),
        breadth_map["GICS_Breadth_20"].combine_first(breadth_map["Legacy_Breadth_20"]),
        breadth_map["Legacy_Breadth_20"],
    )

    sector_wide = mi[["Date"] + legacy_cols].copy()
    if gics_df is not None and not gics_df.empty:
        gd_norm = gics_df.copy()
        gd_norm["Date"] = pd.to_datetime(gd_norm["Date"]).dt.normalize()
        gd_norm = gd_norm.sort_values("Date").drop_duplicates("Date")
        # Keep non-colliding GICS sector columns prefixed clearly if needed
        rename_g = {c: f"GICS::{c}" for c in gd_norm.columns if c != "Date"}
        sector_wide = sector_wide.merge(gd_norm.rename(columns=rename_g), on="Date", how="outer").sort_values("Date")

    return breadth_map[["Date", "ASPI_Close", "ASPI_MOM_20", "ASPI_MOM_60", "Sector_Breadth_MOM_20"]], sector_wide


def compute_market_relative_panel(
    df_adj: pd.DataFrame,
    market_indices: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    gics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute Group F (Market & Sector Relative Indicators) independently per SecurityID:
      - ASPI_MOM_20, ASPI_MOM_60, Sector_Breadth_MOM_20 (aligned by Date)
      - RS_ASPI_Level, RS_ASPI_ROC_20, RS_ASPI_ROC_60 (matched-horizon CapitalAdjClose vs ASPI)
      - Beta_ASPI_60, Beta_ASPI_252 (matched-horizon OLS Beta avoiding thin-trading bias)
      - Sector_MOM_20, RS_Sector_ROC_20 (point-in-time verified sector lookup; strictly NaN when Sector='Unknown')
    """
    work = df_adj[["SecurityID", "CompanyCode", "Date", "MainType", "SubType", "CapitalAdjClose"]].copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.normalize()
    work = work.sort_values(["SecurityID", "Date"]).reset_index(drop=True)

    mkt_daily, sector_wide = _prepare_market_benchmark_frames(market_indices, gics_df=gics_df)
    work = work.merge(mkt_daily, on="Date", how="left")

    # Precompute point-in-time verified sector assignments ONLY where Sector != "Unknown"
    valid_sec_rows = sector_mapping[
        sector_mapping["Sector"].notna()
        & (sector_mapping["Sector"].astype(str).str.upper() != "UNKNOWN")
    ].copy()
    verified_sec_map: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    if not valid_sec_rows.empty:
        sector_mom_table = sector_wide.set_index("Date")
        sector_mom20_lookup = 100.0 * (sector_mom_table / sector_mom_table.shift(20) - 1.0)
        for sid, grp in valid_sec_rows.groupby("SecurityID"):
            spans = []
            for _, rr in grp.iterrows():
                ef = pd.Timestamp(rr["EffectiveFrom"]) if pd.notna(rr["EffectiveFrom"]) else pd.Timestamp.min
                et = pd.Timestamp(rr["EffectiveTo"]) if pd.notna(rr["EffectiveTo"]) else pd.Timestamp.max
                spans.append((ef, et, str(rr["Sector"])))
            verified_sec_map[sid] = spans
    else:
        sector_mom_table = None
        sector_mom20_lookup = None

    rs_level_all = np.full(len(work), np.nan, dtype=np.float64)
    rs_roc20_all = np.full(len(work), np.nan, dtype=np.float64)
    rs_roc60_all = np.full(len(work), np.nan, dtype=np.float64)
    beta60_all = np.full(len(work), np.nan, dtype=np.float64)
    beta252_all = np.full(len(work), np.nan, dtype=np.float64)
    sec_mom20_all = np.full(len(work), np.nan, dtype=np.float64)
    rs_sec20_all = np.full(len(work), np.nan, dtype=np.float64)

    cap_all = pd.to_numeric(work["CapitalAdjClose"], errors="coerce").to_numpy(dtype=np.float64)
    aspi_all = pd.to_numeric(work["ASPI_Close"], errors="coerce").to_numpy(dtype=np.float64)
    dates_all = work["Date"].to_numpy()

    def _rolling_beta(r_s: np.ndarray, r_m: np.ndarray, w: int) -> np.ndarray:
        m = len(r_s)
        out = np.full(m, np.nan, dtype=np.float64)
        if m < w:
            return out
        rs_w = sliding_window_view(r_s, w)
        rm_w = sliding_window_view(r_m, w)
        fin = np.all(np.isfinite(rs_w), axis=1) & np.all(np.isfinite(rm_w), axis=1)
        if not fin.any():
            return out
        xs = rs_w[fin]
        xm = rm_w[fin]
        xm_d = xm - xm.mean(axis=1, keepdims=True)
        xs_d = xs - xs.mean(axis=1, keepdims=True)
        var_m = np.sum(xm_d * xm_d, axis=1)
        cov = np.sum(xs_d * xm_d, axis=1)
        beta = np.where(var_m > 1e-12, cov / var_m, np.nan)
        out[w - 1:][fin] = beta
        return out

    for sid, idx in work.groupby("SecurityID", sort=False).groups.items():
        pos = np.asarray(idx, dtype=np.int64)
        p_raw = cap_all[pos]
        m_raw = aspi_all[pos]
        p = np.where(np.isfinite(p_raw) & (p_raw > 0), p_raw, np.nan)
        m_idx = np.where(np.isfinite(m_raw) & (m_raw > 0), m_raw, np.nan)
        k = len(p)

        with np.errstate(divide="ignore", invalid="ignore"):
            rs_level_all[pos] = np.where(np.isfinite(p) & np.isfinite(m_idx), p / m_idx, np.nan)
            if k > 20:
                r_s20 = p[20:] / p[:-20]
                r_m20 = m_idx[20:] / m_idx[:-20]
                rs_roc20_all[pos[20:]] = 100.0 * (r_s20 / r_m20 - 1.0)
            if k > 60:
                r_s60 = p[60:] / p[:-60]
                r_m60 = m_idx[60:] / m_idx[:-60]
                rs_roc60_all[pos[60:]] = 100.0 * (r_s60 / r_m60 - 1.0)

        # Matched-horizon interval returns (prev trade -> current trade) eliminate thin-trading bias
        r_s1 = np.full(k, np.nan, dtype=np.float64)
        r_m1 = np.full(k, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_s1[1:] = p[1:] / p[:-1] - 1.0
            r_m1[1:] = m_idx[1:] / m_idx[:-1] - 1.0

        # Beta_60 / Beta_252 strictly preserve warm-up NaNs for first W intervals (0..59 / 0..251)
        b60 = _rolling_beta(r_s1, r_m1, 60)
        b252 = _rolling_beta(r_s1, r_m1, 252)
        beta60_all[pos] = b60
        beta252_all[pos] = b252

        if sid in verified_sec_map and sector_mom_table is not None:
            dt_slice = pd.to_datetime(dates_all[pos])
            for ef, et, sec_name in verified_sec_map[sid]:
                if sec_name not in sector_mom_table.columns:
                    col_match = f"GICS::{sec_name}" if f"GICS::{sec_name}" in sector_mom_table.columns else None
                else:
                    col_match = sec_name
                if not col_match:
                    continue
                mask_dt = (dt_slice >= ef) & (dt_slice <= et)
                if not mask_dt.any():
                    continue
                matched_dts = dt_slice[mask_dt]
                sec_lvl = sector_mom_table[col_match].reindex(matched_dts).to_numpy(dtype=np.float64)
                sec_m20 = sector_mom20_lookup[col_match].reindex(matched_dts).to_numpy(dtype=np.float64)
                sub_p = p[mask_dt]
                rs_s = np.full(len(sub_p), np.nan, dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    if len(sub_p) > 20:
                        rs_s[20:] = 100.0 * ((sub_p[20:] / sub_p[:-20]) / (sec_lvl[20:] / sec_lvl[:-20]) - 1.0)
                sec_mom20_all[pos[mask_dt]] = sec_m20
                rs_sec20_all[pos[mask_dt]] = rs_s

    work["RS_ASPI_Level"] = rs_level_all
    work["RS_ASPI_ROC_20"] = rs_roc20_all
    work["RS_ASPI_ROC_60"] = rs_roc60_all
    work["Beta_ASPI_60"] = beta60_all
    work["Beta_ASPI_252"] = beta252_all
    work["Sector_MOM_20"] = sec_mom20_all
    work["RS_Sector_ROC_20"] = rs_sec20_all

    cols_out = [
        "SecurityID",
        "CompanyCode",
        "Date",
        "MainType",
        "SubType",
        "ASPI_Close",
        "ASPI_MOM_20",
        "ASPI_MOM_60",
        "RS_ASPI_Level",
        "RS_ASPI_ROC_20",
        "RS_ASPI_ROC_60",
        "Beta_ASPI_60",
        "Beta_ASPI_252",
        "Sector_MOM_20",
        "RS_Sector_ROC_20",
        "Sector_Breadth_MOM_20",
    ]
    return work[cols_out]
