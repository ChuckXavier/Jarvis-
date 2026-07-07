"""
JARVIS V3 - Long/Short Portfolio Optimizer
===========================================
ARCHITECTURAL NOTE (what changed vs V2):
The 60/40 two-engine optimizer (15 longs / 10 shorts / 5 leveraged ETFs) with
its in-memory `_state` dict is replaced by a diversified long/short builder:
~40 longs and ~20 shorts, inverse-volatility weighted, with hard caps of 5%
per name, 20% long / 10% short per sector, and a 2% cash buffer. ALL regime
logic is gone from this file — posture (gross/net exposure) comes from
risk/regime.py's persisted machine, so this module is stateless and the
state-amnesia bug cannot recur here. The selection/weighting math lives in
the pure `build_targets()` so backtest/walkforward.py constructs portfolios
with the exact production code. Leveraged/inverse ETFs are excluded from the
candidate set unless settings.ALLOW_LEVERAGE is True (Option-A default: off).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    MAX_SINGLE_POSITION_PCT, MIN_POSITION_PCT, MIN_CASH_RESERVE_PCT,
    TARGET_LONG_POSITIONS, TARGET_SHORT_POSITIONS,
    MAX_SECTOR_NET_LONG, MAX_SECTOR_GROSS_SHORT,
    MIN_PRICE, MIN_DOLLAR_VOLUME, MIN_HISTORY_DAYS,
    INVERSE_VOL_FLOOR, INVERSE_VOL_LOOKBACK, ALLOW_LEVERAGE,
)

# Asset classes that never enter the candidate set when unlevered; we express
# bearish views by shorting weak names directly, not by buying inverse funds.
_GATED_CLASSES_UNLEVERED = {"leveraged", "inverse", "volatility"}
_NEVER_SHORT_CLASSES = {"fixed_income", "alternative"}  # no shorting BIL/DBMF


# ══════════════════════════════════════════════════════════════════════════════
# PURE CORE — shared with the walk-forward backtest
# ══════════════════════════════════════════════════════════════════════════════

def _inverse_vol_weights(tickers: list[str], vol: pd.Series, leg_gross: float,
                         max_pos: float) -> dict[str, float]:
    """1/vol weights normalized to leg_gross, per-name cap, dust dropped."""
    if not tickers or leg_gross <= 0:
        return {}
    v = vol.reindex(tickers).astype(float)
    v = v.clip(lower=INVERSE_VOL_FLOOR).fillna(v.median() if v.notna().any() else 0.20)
    inv = 1.0 / v
    w = (inv / inv.sum()) * leg_gross

    # Iteratively enforce the single-position cap, redistributing the excess.
    for _ in range(4):
        over = w[w > max_pos]
        if over.empty:
            break
        excess = float((over - max_pos).sum())
        w[w > max_pos] = max_pos
        under = w[w < max_pos]
        if under.empty or excess <= 0:
            break
        w[under.index] += excess * (under / under.sum())
    w = w.clip(upper=max_pos)

    # Drop dust, renormalize back to leg_gross without breaching the cap.
    w = w[w >= MIN_POSITION_PCT]
    if w.empty:
        return {}
    scale = min(leg_gross / w.sum(), max_pos / w.max())
    w = w * scale
    return {t: float(x) for t, x in w.items()}


def _apply_sector_cap(weights: dict[str, float], sector_map: dict[str, str],
                      cap: float) -> dict[str, float]:
    """Scale down any sector whose summed |weight| exceeds `cap` (one side)."""
    if not weights:
        return weights
    s = pd.Series(weights, dtype=float)
    sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in s.index})
    out = s.copy()
    for sec, grp in s.groupby(sectors):
        tot = float(grp.abs().sum())
        if tot > cap:
            out[grp.index] = grp * (cap / tot)
    return {t: float(w) for t, w in out.items() if abs(w) >= MIN_POSITION_PCT}


def build_targets(scores: pd.Series, vol: pd.Series, sector_map: dict[str, str],
                  gross: float, net: float,
                  long_count: int = TARGET_LONG_POSITIONS,
                  short_count: int = TARGET_SHORT_POSITIONS,
                  max_pos: float = MAX_SINGLE_POSITION_PCT,
                  cash_reserve: float = MIN_CASH_RESERVE_PCT,
                  no_short: set[str] | None = None) -> dict[str, float]:
    """
    PURE portfolio constructor. scores: eligible-only composite cross-section.
    vol: annualized realized vol per ticker. Returns {ticker: signed weight}.

    Leg sizing from the regime ladder:  long = (gross+net)/2, short = (gross-net)/2,
    with the long leg additionally capped by the cash reserve.
    """
    s = scores.dropna().sort_values(ascending=False)
    if s.empty:
        return {}

    long_gross = max(0.0, (gross + net) / 2.0)
    short_gross = max(0.0, (gross - net) / 2.0)
    long_gross = min(long_gross, 1.0 - cash_reserve)

    longs = list(s.head(long_count).index)
    short_pool = s.tail(short_count * 2)  # extra room after no_short filter
    if no_short:
        short_pool = short_pool[~short_pool.index.isin(no_short)]
    shorts = list(short_pool.tail(short_count).index)
    shorts = [t for t in shorts if t not in set(longs)]

    target: dict[str, float] = {}
    target.update(_inverse_vol_weights(longs, vol, long_gross, max_pos))
    long_w = _apply_sector_cap(target, sector_map, MAX_SECTOR_NET_LONG)

    short_w_raw = _inverse_vol_weights(shorts, vol, short_gross, max_pos)
    short_w = _apply_sector_cap(short_w_raw, sector_map, MAX_SECTOR_GROSS_SHORT)

    out = dict(long_w)
    for t, w in short_w.items():
        out[t] = -w
    return out


# ══════════════════════════════════════════════════════════════════════════════
# LIVE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _eligibility(latest_scores: pd.Series, prices: pd.DataFrame,
                 volume: pd.DataFrame | None) -> tuple[pd.Series, pd.Series, set]:
    """
    Filter the scored cross-section to tradeable names and compute sizing vol.
    Returns (eligible_scores, vol_series, no_short_set).
    """
    from config.universe import get_asset_class_map
    asset_class = get_asset_class_map()

    last_px = prices.iloc[-1]
    history = prices.notna().sum()
    rets = prices.pct_change()
    vol = rets.rolling(INVERSE_VOL_LOOKBACK).std().iloc[-1] * np.sqrt(252)

    eligible = latest_scores.dropna()
    eligible = eligible[eligible.index.isin(prices.columns)]
    eligible = eligible[last_px.reindex(eligible.index) >= MIN_PRICE]
    eligible = eligible[history.reindex(eligible.index) >= MIN_HISTORY_DAYS]

    if volume is not None and not volume.empty:
        dollar_vol = (prices * volume).rolling(21).median().iloc[-1]
        liquid = dollar_vol.reindex(eligible.index) >= MIN_DOLLAR_VOLUME
        # ETFs whose volume we trust by construction stay in even if the
        # median calc is NaN on a sparse day.
        from config.universe import get_all_tickers
        etfs = set(get_all_tickers())
        keep = liquid.fillna(False) | eligible.index.to_series().isin(etfs)
        dropped = int((~keep).sum())
        if dropped:
            logger.info(f"  liquidity filter dropped {dropped} names "
                        f"(<${MIN_DOLLAR_VOLUME/1e6:.0f}M median ADV)")
        eligible = eligible[keep]
    else:
        logger.warning("  no volume matrix supplied — liquidity filter skipped")

    gated = _GATED_CLASSES_UNLEVERED if not ALLOW_LEVERAGE else {"volatility"}
    eligible = eligible[[asset_class.get(t, "equity") not in gated
                         for t in eligible.index]]

    no_short = {t for t in eligible.index
                if asset_class.get(t, "equity") in _NEVER_SHORT_CLASSES}
    return eligible, vol, no_short


def optimize_portfolio(latest_scores: pd.Series, prices: pd.DataFrame,
                       portfolio_value: float, regime_info: dict,
                       volume: pd.DataFrame | None = None,
                       sector_map: dict[str, str] | None = None) -> dict:
    """
    Build the target book for today. `regime_info` comes from
    risk.regime.evaluate_and_persist() and supplies target_gross / target_net.
    """
    regime = regime_info.get("regime", "CAUTIOUS")
    gross = float(regime_info.get("target_gross", 0.6))
    net = float(regime_info.get("target_net", 0.3))

    if sector_map is None:
        from config.universe import get_sector_map
        sector_map = get_sector_map()

    logger.info("=" * 55)
    logger.info(f"OPTIMIZER — regime {regime}: gross {gross:.0%}, net {net:+.0%}"
                f"{'' if ALLOW_LEVERAGE else '  [unlevered]'}")

    eligible, vol, no_short = _eligibility(latest_scores, prices, volume)
    logger.info(f"  eligible candidates: {len(eligible)} "
                f"(from {latest_scores.notna().sum()} scored)")

    if len(eligible) < 10:
        logger.error("  too few eligible names — emitting empty target "
                     "(scheduler will hold current book)")
        return {"target_weights": {}, "mode": regime, "expected_positions": 0,
                "cash_pct": 1.0, "gross": 0.0, "net": 0.0}

    target = build_targets(eligible, vol, sector_map, gross, net,
                           no_short=no_short)

    tl = sum(w for w in target.values() if w > 0)
    ts = sum(-w for w in target.values() if w < 0)
    nl = sum(1 for w in target.values() if w > 0)
    ns = sum(1 for w in target.values() if w < 0)
    cash = max(0.0, 1.0 - tl)

    logger.info(f"  book: {nl} longs ({tl:.1%}) / {ns} shorts ({ts:.1%}) "
                f"| gross {tl+ts:.1%} net {tl-ts:+.1%} | cash {cash:.1%}")
    top = sorted(target.items(), key=lambda x: -x[1])[:5]
    bot = sorted(target.items(), key=lambda x: x[1])[:5]
    logger.info(f"  largest longs:  {[(t, round(w, 3)) for t, w in top]}")
    if ns:
        logger.info(f"  largest shorts: {[(t, round(w, 3)) for t, w in bot]}")
    logger.info("=" * 55)

    return {
        "target_weights": target,
        "mode": regime,
        "expected_positions": nl + ns,
        "cash_pct": cash,
        "gross": tl + ts,
        "net": tl - ts,
        "position_sizes": {t: portfolio_value * w for t, w in target.items()},
    }
