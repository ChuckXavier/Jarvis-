"""
JARVIS V2 - Signal Ensemble: Combining All 4 Signals
=======================================================
Inspired by: Medallion Fund's principle of "many small edges combined"

HOW THIS WORKS (for non-coders):
- Each of the 4 signals gives its own opinion on every ETF
- Signal 1 (Momentum) might say: "Buy QQQ, sell TLT"
- Signal 2 (Trend) might say: "Buy SPY, sell EEM"
- Signal 3 (Mean Reversion) might say: "Buy EEM (oversold!), sell QQQ"
- Signal 4 (Regime) might say: "We're in CALM mode, favor stocks over bonds"

- The Ensemble COMBINES all 4 opinions into ONE final score per ETF
- If most signals agree → strong conviction → bigger position
- If signals disagree → low conviction → smaller or no position

THE ADAPTIVE WEIGHTING:
- Initially, all 4 signals get equal weight (25% each)
- Every month, we check which signals performed best recently
- Signals that were accurate get MORE weight next month
- Signals that were wrong get LESS weight
- This is how Jarvis "learns" over time without overfitting
"""

import pandas as pd
import numpy as np
from loguru import logger

from data.db import get_all_prices
from features.engine import compute_all_features
from signals.cross_momentum import compute_cross_sectional_momentum
from signals.trend_follow import compute_trend_following
from signals.mean_revert import compute_mean_reversion
from signals.vol_regime import compute_regime_signal


# Default signal weights (equal weight to start)
DEFAULT_WEIGHTS = {
    "cross_momentum": 0.25,
    "trend_follow": 0.25,
    "mean_reversion": 0.25,
    "vol_regime": 0.25,
}


def compute_ensemble(prices: pd.DataFrame = None, weights: dict = None) -> dict:
    """
    Run ALL 4 signals and combine them into a final alpha score per ETF.

    Parameters:
        prices: DataFrame with date index, ticker columns. If None, fetches from DB.
        weights: dict of signal weights. If None, uses equal weights.

    Returns:
        dict with:
            "alpha_scores": DataFrame (date × ticker) — the final combined score
            "latest_scores": Series — most recent day's scores (for trading)
            "signal_details": dict of individual signal DataFrames
            "weights_used": dict of weights applied
            "regime": str — current detected regime
    """
    if prices is None:
        prices = get_all_prices()

    if prices.empty:
        logger.error("No price data available for ensemble computation!")
        return _empty_result()

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    logger.info("=" * 50)
    logger.info("COMPUTING ENSEMBLE ALPHA SCORES")
    logger.info(f"Weights: {weights}")
    logger.info("=" * 50)

    # ── Step 1: Compute Features ──
    logger.info("Step 1: Computing features...")
    features = compute_all_features(prices)
    price_features = features["price_features"]
    macro_features = features["macro_features"]

    # ── Step 2: Run Signal 1 — Cross-Sectional Momentum ──
    logger.info("Step 2: Running Signal 1 (Cross-Sectional Momentum)...")
    try:
        sig_momentum = compute_cross_sectional_momentum(prices)
        logger.info(f"  Signal 1: {sig_momentum.notna().any().sum()} ETFs scored")
    except Exception as e:
        logger.error(f"  Signal 1 FAILED: {e}")
        sig_momentum = pd.DataFrame()

    # ── Step 3: Run Signal 2 — Trend Following ──
    logger.info("Step 3: Running Signal 2 (Trend Following)...")
    try:
        sig_trend = compute_trend_following(prices)
        logger.info(f"  Signal 2: {sig_trend.notna().any().sum()} ETFs scored")
    except Exception as e:
        logger.error(f"  Signal 2 FAILED: {e}")
        sig_trend = pd.DataFrame()

    # ── Step 4: Run Signal 3 — Mean Reversion ──
    logger.info("Step 4: Running Signal 3 (Mean Reversion)...")
    try:
        sig_meanrev = compute_mean_reversion(price_features, macro_features, prices)
        logger.info(f"  Signal 3: {sig_meanrev.notna().any().sum()} ETFs scored")
    except Exception as e:
        logger.error(f"  Signal 3 FAILED: {e}")
        sig_meanrev = pd.DataFrame()

    # ── Step 5: Run Signal 4 — Volatility Regime ──
    logger.info("Step 5: Running Signal 4 (Volatility Regime)...")
    try:
        regime_result = compute_regime_signal(macro_features, prices)
        sig_regime = regime_result["etf_signals"]
        current_regime = regime_result["current_regime"]
        logger.info(f"  Signal 4: Regime = {current_regime}, {sig_regime.notna().any().sum()} ETFs scored")
    except Exception as e:
        logger.error(f"  Signal 4 FAILED: {e}")
        sig_regime = pd.DataFrame()
        current_regime = "UNKNOWN"

    # ── Step 6: Align all signals to the same date range ──
    logger.info("Step 6: Aligning and combining signals...")
    signals = {
        "cross_momentum": sig_momentum,
        "trend_follow": sig_trend,
        "mean_reversion": sig_meanrev,
        "vol_regime": sig_regime,
    }

    alpha_scores = combine_signals(signals, weights, prices)

    # ── Step 7: Get latest scores ──
    if not alpha_scores.empty:
        latest = alpha_scores.iloc[-1].sort_values(ascending=False)
        logger.info(f"\n  FINAL ALPHA SCORES (latest day):")
        for ticker, score in latest.items():
            if pd.notna(score) and score != 0:
                direction = "BUY " if score > 0 else "SELL"
                logger.info(f"    {direction} {ticker:5s}: {score:+.3f}")
    else:
        latest = pd.Series(dtype=float)

    # Count how many signals are active
    active_signals = sum(1 for s in signals.values() if not s.empty)
    logger.info(f"\nEnsemble complete: {active_signals}/4 signals active, regime={current_regime}")

    return {
        "alpha_scores": alpha_scores,
        "latest_scores": latest,
        "signal_details": signals,
        "weights_used": weights,
        "regime": current_regime,
        "features": features,
    }


def combine_signals(signals: dict, weights: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Combine individual signal DataFrames into a single alpha score.

    The formula:
    Alpha(ETF) = w1 × Momentum(ETF) + w2 × Trend(ETF)
               + w3 × MeanRev(ETF) + w4 × Regime(ETF)

    Where w1+w2+w3+w4 = 1.0 (they must sum to 100%)
    """
    # Find the common date range across all available signals
    all_dates = set()
    active_signals = {}

    for name, df in signals.items():
        if not df.empty:
            all_dates.update(df.index)
            active_signals[name] = df

    if not active_signals:
        logger.warning("No active signals to combine!")
        return pd.DataFrame()

    # Use the intersection of dates where ALL active signals have data
    common_dates = None
    for name, df in active_signals.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(df.index)

    if not common_dates:
        logger.warning("No overlapping dates across signals!")
        return pd.DataFrame()

    common_dates = sorted(common_dates)
    tickers = prices.columns

    # Build the combined score
    combined = pd.DataFrame(0.0, index=common_dates, columns=tickers)

    # Normalize weights for active signals only
    active_weight_sum = sum(weights.get(name, 0) for name in active_signals)
    if active_weight_sum <= 0:
        active_weight_sum = 1.0

    for name, df in active_signals.items():
        raw_weight = weights.get(name, 0.25)
        normalized_weight = raw_weight / active_weight_sum

        # Align signal to common dates and tickers
        aligned = df.reindex(index=common_dates, columns=tickers).fillna(0.0)
        combined += aligned * normalized_weight

    return combined


def update_signal_weights(
    signals: dict,
    realized_returns: pd.DataFrame,
    current_weights: dict,
    halflife_months: int = 6
) -> dict:
    """
    Update signal weights based on their recent performance.
    This is the "learning" part of Jarvis — run monthly.

    For each signal, compute the Information Coefficient (IC):
    IC = rank correlation between signal's predictions and actual returns

    Signals with higher IC get more weight next month.

    Parameters:
        signals: dict of signal DataFrames from the past month
        realized_returns: actual ETF returns over the past month
        current_weights: current signal weights
        halflife_months: how quickly to adapt (6 = moderate pace)

    Returns:
        Updated weights dict
    """
    logger.info("Updating signal weights based on realized performance...")

    ics = {}

    for name, signal_df in signals.items():
        if signal_df.empty:
            ics[name] = 0.0
            continue

        try:
            # Get the signal scores from the start of the evaluation period
            signal_scores = signal_df.iloc[-21]  # Score from ~1 month ago

            # Get the realized returns over that month
            if realized_returns.empty:
                ics[name] = 0.0
                continue

            returns = realized_returns.iloc[-1]

            # Compute rank correlation (Spearman)
            # This measures: "Did the signal correctly rank ETFs by future return?"
            valid_tickers = signal_scores.dropna().index.intersection(returns.dropna().index)
            if len(valid_tickers) < 5:
                ics[name] = 0.0
                continue

            from scipy.stats import spearmanr
            ic, p_value = spearmanr(
                signal_scores[valid_tickers],
                returns[valid_tickers]
            )

            ics[name] = ic if not np.isnan(ic) else 0.0
            logger.info(f"  {name}: IC = {ic:.4f} (p={p_value:.4f})")

        except Exception as e:
            logger.error(f"  {name}: IC computation failed - {e}")
            ics[name] = 0.0

    # Update weights using exponential moving average
    # New weight ∝ EMA of IC (higher IC → higher weight)
    new_weights = {}
    min_weight = 0.10  # No signal gets less than 10%
    max_weight = 0.40  # No signal gets more than 40%

    for name in current_weights:
        old_w = current_weights[name]
        ic = max(ics.get(name, 0.0), 0.0)  # Only use positive IC

        # EMA: alpha = 2 / (halflife + 1)
        alpha = 2.0 / (halflife_months + 1)
        new_w = alpha * (0.25 + ic) + (1 - alpha) * old_w  # Blend toward IC-adjusted weight

        new_weights[name] = np.clip(new_w, min_weight, max_weight)

    # Normalize so weights sum to 1.0
    total = sum(new_weights.values())
    if total > 0:
        new_weights = {k: v / total for k, v in new_weights.items()}

    logger.info(f"  Updated weights: {new_weights}")

    return new_weights


def get_top_bottom_etfs(latest_scores: pd.Series, top_n: int = 5) -> dict:
    """
    Convenience function: get the top N buy candidates and bottom N sell candidates.
    """
    valid = latest_scores.dropna().sort_values(ascending=False)

    return {
        "top_buy": valid.head(top_n),
        "top_sell": valid.tail(top_n),
    }


def _empty_result() -> dict:
    """Return empty result when ensemble can't run."""
    return {
        "alpha_scores": pd.DataFrame(),
        "latest_scores": pd.Series(dtype=float),
        "signal_details": {},
        "weights_used": DEFAULT_WEIGHTS,
        "regime": "UNKNOWN",
        "features": {},
    }
