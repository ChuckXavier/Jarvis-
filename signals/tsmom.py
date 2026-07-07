"""
JARVIS V3 - ENGINE 2: Multi-Asset Time-Series Momentum Sleeve
==============================================================
WHAT THIS IS: a second, independent return engine. The equity book (Engine 1)
bets on WHICH stocks beat WHICH — a cross-sectional bet. This sleeve bets on
whether each ASSET CLASS is trending up or down versus its own history — a
time-series bet (Moskowitz-Ooi-Pedersen 2012), expressed long OR short across
~24 liquid ETFs spanning global equities, bonds, commodities, currencies, and
REITs. The two bet types are structurally different, which is the point:
orthogonal engines raise combined Sharpe in a way no amount of tuning one
engine can.

HONEST CONSTRAINTS, IN WRITING:
- Currencies/commodities are ETF wrappers (Alpaca has no spot FX or futures).
  USO and DBA carry futures roll decay — the trend signal must overcome it,
  and the backtest prices already include it, so the test is fair.
- Unlevered (gross <= 100%), this sleeve realizes roughly 6-10% volatility.
  Managed-futures funds lever this 2-4x; we do not, until evidence does.
- DBMF (a managed-futures ETF we hold in crisis ladders) is EXCLUDED here:
  it IS this strategy — holding it inside the sleeve would double-count.
- This module is PURE (no DB, no broker). It ships into the live book only
  after backtest/sleeve_lab.py produces evidence, and only as its own single
  change after the current paper-trading baseline is established.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Sleeve universe: ticker -> asset class. Existing JARVIS ETFs plus a small,
# deliberately conservative set of additions (all among the largest, oldest,
# most liquid ETFs in their category — nothing exotic).
SLEEVE_UNIVERSE = {
    # global equities
    "SPY": "equity", "QQQ": "equity", "IWM": "equity", "EFA": "equity",
    "EEM": "equity", "EWJ": "equity", "FXI": "equity", "VGK": "equity",
    # bonds & credit
    "TLT": "bond", "IEF": "bond", "LQD": "bond", "HYG": "bond",
    "TIP": "bond", "EMB": "bond",
    # commodities (USO/DBA: futures-based, roll decay is in the price series)
    "GLD": "commodity", "SLV": "commodity", "DBC": "commodity",
    "USO": "commodity", "DBA": "commodity", "GDX": "commodity",
    # currencies (ETF wrappers)
    "UUP": "currency", "FXE": "currency", "FXY": "currency",
    # real assets
    "VNQ": "reit",
}

SLEEVE_HORIZONS = (63, 126, 252)   # 3m / 6m / 12m trend blend
SLEEVE_VOL_TARGET = 0.08           # portfolio-level ann. vol aim (unlevered)
SLEEVE_MAX_ASSET = 0.12            # per-ETF cap, absolute weight
SLEEVE_MAX_CLASS = 0.35            # per-asset-class cap, sum of |w|
SLEEVE_GROSS_CAP = 1.00            # unlevered, always
SLEEVE_VOL_FLOOR = 0.04            # sizing floor so bonds can't blow up caps


def tsmom_scores(prices: pd.DataFrame,
                 horizons: tuple = SLEEVE_HORIZONS) -> pd.DataFrame:
    """
    Trend score per asset in [-1, +1]: the average SIGN of its own return
    over each horizon. Sign (not magnitude) is deliberate — it is the robust
    form of the anomaly and immune to outliers. Point-in-time by construction
    (only pct_change over past windows). Assets with less history than the
    longest horizon score NaN.
    """
    score = None
    for h in horizons:
        s = np.sign(prices.pct_change(h))
        score = s if score is None else score + s
    score = score / len(horizons)
    enough = prices.notna().rolling(max(horizons) + 1).count() > max(horizons)
    return score.where(enough)


def build_sleeve_targets(scores_row: pd.Series, vol_row: pd.Series,
                         asset_class: dict[str, str] | None = None,
                         vol_target: float = SLEEVE_VOL_TARGET,
                         max_asset: float = SLEEVE_MAX_ASSET,
                         max_class: float = SLEEVE_MAX_CLASS,
                         gross_cap: float = SLEEVE_GROSS_CAP) -> dict[str, float]:
    """
    PURE sleeve constructor. Each active asset gets an equal share of the
    portfolio risk budget, scaled by its trend score and inverse volatility;
    then per-asset, per-class, and gross caps clamp the book. Correlations
    are deliberately ignored in sizing (robustness over elegance); realized
    vol therefore lands below target, which errs on the safe side.
    """
    asset_class = asset_class or SLEEVE_UNIVERSE
    s = scores_row.dropna()
    s = s[s != 0]
    if s.empty:
        return {}
    v = vol_row.reindex(s.index).clip(lower=SLEEVE_VOL_FLOOR)
    v = v.fillna(v.median() if v.notna().any() else 0.15)
    n = len(s)
    w = s * (vol_target / (v * np.sqrt(n)))
    w = w.clip(-max_asset, max_asset)

    # per-class cap on summed |w|
    cls = pd.Series({t: asset_class.get(t, "other") for t in w.index})
    for c, grp in w.groupby(cls):
        tot = float(grp.abs().sum())
        if tot > max_class:
            w[grp.index] = grp * (max_class / tot)

    gross = float(w.abs().sum())
    if gross > gross_cap:
        w = w * (gross_cap / gross)
    return {t: float(x) for t, x in w.items() if abs(x) >= 0.005}
