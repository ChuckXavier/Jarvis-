"""
JARVIS V3 - Adaptive Signal Ensemble (full universe, live IC weighting)
========================================================================
ARCHITECTURAL NOTE (what changed vs V2):
(1) Signals are now computed as vectorized (date x ticker) matrices straight
    from the price matrix — no per-ticker feature DataFrames. At 600 names the
    V2 per-ticker loop would have built ~180MB of frames; this stays under
    ~15MB and fits the 512MB container.
(2) The IC-based weight adaptation that lived orphaned in backtest/adapter.py
    is now LIVE: every run computes trailing 21d/63d Spearman ICs per signal,
    EMA-adapts the weights toward IC-implied targets (floor 5%, ceiling 40%),
    and persists them in the existing `signal_weights` table. Restarts load
    the last persisted weights — no more static hardcoded 0.35/0.25/0.20/0.20.
(3) The math lives in PURE functions (compute_signal_matrices, combine_scores,
    compute_signal_ics, adapt_weights) with zero I/O, so backtest/walkforward.py
    runs the exact same code as production — research and live cannot drift.
(4) HONEST SUBSTITUTION: the spec asked for a quality/value signal (earnings
    yield, ROE, leverage). This pipeline has no fundamentals feed, and faking
    one from sporadic yfinance lookups would be hallucinated data. Signal 5 is
    therefore the low-volatility defensive anomaly — a documented, price-based
    stand-in. Adding a real fundamentals feed is future work, not pretend work.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    MOMENTUM_LOOKBACK, MOMENTUM_SKIP, TREND_FAST_LOOKBACK, TREND_SLOW_LOOKBACK,
    MIN_HISTORY_DAYS,
    IC_LOOKBACK_PRIMARY, IC_LOOKBACK_SECONDARY, IC_BLEND,
    WEIGHT_FLOOR, WEIGHT_CEILING, WEIGHT_EMA_HALFLIFE_DAYS, IC_TILT_SCALE,
)

SIGNAL_NAMES = ["xs_momentum", "mean_reversion", "trend_follow",
                "vol_adj_momentum", "low_vol"]

DEFAULT_WEIGHTS = {name: 1.0 / len(SIGNAL_NAMES) for name in SIGNAL_NAMES}


# ══════════════════════════════════════════════════════════════════════════════
# PURE MATH — shared verbatim by live pipeline and walk-forward backtest
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Z-score each row across tickers; winsorize at ±clip. NaNs propagate."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip)


def compute_signal_matrices(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    All five raw signal matrices from a (date x ticker) adj-close matrix.
    Every value at row t uses data <= t only (no look-ahead by construction).
    Tickers with < MIN_HISTORY_DAYS observations are NaN-ed out entirely.
    """
    if prices is None or prices.empty:
        return {name: pd.DataFrame() for name in SIGNAL_NAMES}

    rets = prices.pct_change()

    # Coverage mask: insufficient-history names get NaN everywhere (spec req).
    enough = prices.notna().cumsum() >= MIN_HISTORY_DAYS

    # 1) Cross-sectional momentum: 126d return ending MOMENTUM_SKIP days ago.
    mom = prices.shift(MOMENTUM_SKIP) / prices.shift(MOMENTUM_SKIP + MOMENTUM_LOOKBACK) - 1.0

    # 2) Short-term mean reversion: fade 5d move + Bollinger position.
    r5 = prices.pct_change(5)
    ma20 = prices.rolling(20).mean()
    sd20 = prices.rolling(20).std()
    bb = (prices - ma20) / sd20.replace(0, np.nan)      # +2 = stretched high
    meanrev = -(0.5 * cross_sectional_zscore(r5) + 0.5 * cross_sectional_zscore(bb))

    # 3) Trend following: price vs 50-SMA blended with 50/200 cross.
    sma_f = prices.rolling(TREND_FAST_LOOKBACK).mean()
    sma_s = prices.rolling(TREND_SLOW_LOOKBACK).mean()
    trend = 0.6 * (prices / sma_f - 1.0) + 0.4 * (sma_f / sma_s - 1.0)

    # 4) Volatility-adjusted momentum: 63d return / 63d realized vol.
    vol63 = rets.rolling(63).std() * np.sqrt(252)
    volmom = prices.pct_change(63) / vol63.replace(0, np.nan)

    # 5) Low-volatility defensive anomaly (fundamentals stand-in; see header).
    vol126 = rets.rolling(126).std() * np.sqrt(252)
    lowvol_raw = -vol126

    raw = {
        "xs_momentum": cross_sectional_zscore(mom),
        "mean_reversion": meanrev.clip(-3, 3),
        "trend_follow": cross_sectional_zscore(trend),
        "vol_adj_momentum": cross_sectional_zscore(volmom),
        "low_vol": cross_sectional_zscore(lowvol_raw),
    }
    return {name: df.where(enough) for name, df in raw.items()}


def combine_scores(signals: dict[str, pd.DataFrame], weights: dict[str, float]
                   ) -> pd.DataFrame:
    """
    Weighted composite, NaN-aware: each cell is the weight-renormalized sum of
    the signals available for that ticker/date. A ticker needs >= 3 live
    signals to receive a composite (else NaN — too thin to rank honestly).
    """
    num = None
    den = None
    count = None
    for name, df in signals.items():
        if df is None or df.empty:
            continue
        w = float(weights.get(name, 0.0))
        avail = df.notna()
        contrib = df.fillna(0.0) * w
        num = contrib if num is None else num.add(contrib, fill_value=0.0)
        den = avail * w if den is None else den.add(avail * w, fill_value=0.0)
        count = avail.astype(int) if count is None else count.add(avail.astype(int), fill_value=0)
    if num is None:
        return pd.DataFrame()
    composite = num / den.replace(0, np.nan)
    return composite.where(count >= 3)


def compute_signal_ics(signals: dict[str, pd.DataFrame], prices: pd.DataFrame,
                       lookback_days: int, asof_idx: int = -1) -> dict[str, float]:
    """
    Trailing Information Coefficient per signal: Spearman rank correlation
    between the signal cross-section `lookback_days` before `asof_idx` and the
    realized returns over that window. Pure; used identically in backtest.
    """
    from scipy.stats import spearmanr
    out: dict[str, float] = {}
    n = len(prices)
    asof = asof_idx if asof_idx >= 0 else n + asof_idx
    t0 = asof - lookback_days
    if t0 < 1:
        return {name: 0.0 for name in signals}
    fwd = prices.iloc[asof] / prices.iloc[t0] - 1.0
    for name, df in signals.items():
        ic = 0.0
        try:
            if df is not None and not df.empty and len(df) > t0:
                row = df.iloc[t0]
                valid = row.dropna().index.intersection(fwd.dropna().index)
                if len(valid) >= 10:
                    ic_val, _ = spearmanr(row[valid].values, fwd[valid].values)
                    ic = float(ic_val) if np.isfinite(ic_val) else 0.0
        except Exception:
            ic = 0.0
        out[name] = ic
    return out


def adapt_weights(current: dict[str, float], composite_ic: dict[str, float]
                  ) -> dict[str, float]:
    """
    Daily EMA toward IC-implied targets. Positive IC pulls a signal above the
    equal-weight base; negative IC decays it toward (never below) the floor.
    Floor/ceiling then renormalize. Pure function.
    """
    alpha = 2.0 / (WEIGHT_EMA_HALFLIFE_DAYS + 1.0)
    base = 1.0 / len(current)
    new = {}
    for name, w in current.items():
        ic = composite_ic.get(name, 0.0)
        target = base + max(ic, 0.0) * IC_TILT_SCALE
        nw = alpha * target + (1 - alpha) * w
        new[name] = float(np.clip(nw, WEIGHT_FLOOR, WEIGHT_CEILING))
    total = sum(new.values())
    if total > 0:
        new = {k: v / total for k, v in new.items()}
    return new


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT PERSISTENCE (reuses the V2 `signal_weights` schema)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_weight_table():
    from sqlalchemy import text
    from data.db import engine
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signal_weights (
                    date DATE, signal_name TEXT, weight REAL,
                    ic REAL, ic_3m REAL, ic_6m REAL,
                    PRIMARY KEY (date, signal_name)
                )
            """))
    except Exception as e:
        logger.warning(f"ensemble: weight table ensure failed: {e}")


def load_weights() -> dict[str, float]:
    """Latest persisted weights; equal-weight defaults on a cold database."""
    from sqlalchemy import text
    from data.db import engine
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT signal_name, weight FROM signal_weights "
                "WHERE date = (SELECT MAX(date) FROM signal_weights)"
            )).fetchall()
        loaded = {r[0]: float(r[1]) for r in rows if r[0] in SIGNAL_NAMES}
        if len(loaded) == len(SIGNAL_NAMES):
            total = sum(loaded.values())
            if total > 0:
                return {k: v / total for k, v in loaded.items()}
    except Exception as e:
        logger.warning(f"ensemble: weight load failed ({e}); using defaults")
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict[str, float], ic_21: dict[str, float],
                 ic_63: dict[str, float]) -> None:
    from sqlalchemy import text
    from data.db import engine
    try:
        _ensure_weight_table()
        today = datetime.now().date()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO signal_weights (date, signal_name, weight, ic, ic_3m, ic_6m)
                VALUES (:date, :name, :weight, :ic, :ic3, :ic6)
                ON CONFLICT (date, signal_name) DO UPDATE SET
                    weight = EXCLUDED.weight, ic = EXCLUDED.ic,
                    ic_3m = EXCLUDED.ic_3m, ic_6m = EXCLUDED.ic_6m
            """), [
                {"date": today, "name": n, "weight": weights[n],
                 "ic": ic_21.get(n, 0.0), "ic3": ic_63.get(n, 0.0), "ic6": 0.0}
                for n in SIGNAL_NAMES
            ])
    except Exception as e:
        logger.error(f"ensemble: weight save failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def compute_ensemble(prices: pd.DataFrame, weights: dict[str, float] | None = None,
                     persist: bool = True) -> dict:
    """
    Full live computation: signals -> trailing ICs -> adapted weights ->
    composite. Returns latest scores plus the audit trail. `persist=False`
    makes it side-effect free (used in tests).
    """
    if prices is None or prices.empty:
        logger.error("ensemble: no price data")
        return {"alpha_scores": pd.DataFrame(), "latest_scores": pd.Series(dtype=float),
                "weights_used": DEFAULT_WEIGHTS.copy(), "ics": {}, "signals": {}}

    logger.info("=" * 55)
    logger.info(f"ENSEMBLE — {prices.shape[1]} tickers, {prices.shape[0]} days")

    signals = compute_signal_matrices(prices)
    for name, df in signals.items():
        n_live = int(df.iloc[-1].notna().sum()) if not df.empty else 0
        logger.info(f"  {name:18s}: {n_live} tickers scored")

    if weights is None:
        weights = load_weights() if persist else DEFAULT_WEIGHTS.copy()

    ic_21 = compute_signal_ics(signals, prices, IC_LOOKBACK_PRIMARY)
    ic_63 = compute_signal_ics(signals, prices, IC_LOOKBACK_SECONDARY)
    composite_ic = {n: IC_BLEND * ic_21[n] + (1 - IC_BLEND) * ic_63[n]
                    for n in SIGNAL_NAMES}

    new_weights = adapt_weights(weights, composite_ic)
    for n in SIGNAL_NAMES:
        logger.info(f"  {n:18s}: IC21={ic_21[n]:+.3f} IC63={ic_63[n]:+.3f} "
                    f"w {weights.get(n, 0):.3f} -> {new_weights[n]:.3f}")
    if persist:
        save_weights(new_weights, ic_21, ic_63)

    alpha = combine_scores(signals, new_weights)
    latest = (alpha.iloc[-1].dropna().sort_values(ascending=False)
              if not alpha.empty else pd.Series(dtype=float))
    logger.info(f"  composite: {len(latest)} tickers ranked "
                f"(top: {list(latest.head(3).index)}, "
                f"bottom: {list(latest.tail(3).index)})")
    logger.info("=" * 55)

    return {
        # keep only the trailing slice of the composite (memory hygiene)
        "alpha_scores": alpha.tail(90) if not alpha.empty else alpha,
        "latest_scores": latest,
        "weights_used": new_weights,
        "ics": {"ic_21": ic_21, "ic_63": ic_63, "composite": composite_ic},
        "signals": {n: (df.iloc[-1] if not df.empty else pd.Series(dtype=float))
                    for n, df in signals.items()},
    }
