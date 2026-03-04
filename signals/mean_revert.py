"""
JARVIS V2 - Signal 3: Mean Reversion / Value
===============================================
Inspired by: Bridgewater Associates / AQR Value Strategies

HOW THIS WORKS (for non-coders):
- When an ETF drops too far too fast, it often bounces back
- Think of a rubber band: stretch it too far and it snaps back
- This signal looks for ETFs that are "oversold" — beaten down below
  where they should be based on their normal range
- It then gives them a POSITIVE score (buy the dip)

THE SAFETY FILTER (very important!):
- Sometimes prices drop for a GOOD reason (company going bankrupt,
  sector in structural decline)
- To avoid "catching a falling knife," we ONLY trigger this signal
  when the VIX is below 35 (market is NOT in full crisis mode)
- During a crisis (VIX > 35), even oversold ETFs can keep dropping

INDICATORS USED:
- RSI: Is the ETF oversold (RSI < 30) or overbought (RSI > 70)?
- Bollinger Bands: Is the price near the bottom of its normal range?
- Drawdown: How far has it fallen from its peak?
"""

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import RSI_OVERSOLD, RSI_OVERBOUGHT, VIX_CRISIS_THRESHOLD


def compute_mean_reversion(
    price_features: dict,
    macro_features: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute mean reversion scores for all ETFs.

    Parameters:
        price_features: dict of {ticker: DataFrame} from feature engine
        macro_features: DataFrame with VIX and other macro indicators
        prices: DataFrame with date index, ticker columns, adj_close values

    Returns:
        DataFrame with date index, ticker columns, Z-score values
        Positive = oversold (buy candidate — price is "too low")
        Negative = overbought (sell/trim candidate — price is "too high")
        Zero = fair value (no mean reversion opportunity)
    """
    if not price_features:
        logger.warning("Mean reversion: No price features provided")
        return pd.DataFrame()

    logger.info("Computing mean reversion signal")

    # Get VIX data for the crisis filter
    vix_series = None
    if not macro_features.empty and "vix" in macro_features.columns:
        vix_series = macro_features["vix"]

    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ticker, features in price_features.items():
        try:
            if features.empty:
                continue

            # ── Component 1: RSI Score ──
            # RSI < 30 → oversold (positive score)
            # RSI > 70 → overbought (negative score)
            # RSI 30-70 → neutral (zero)
            rsi = features.get("rsi_14")
            if rsi is None:
                continue

            rsi_score = pd.Series(0.0, index=features.index)
            rsi_score = rsi_score.where(rsi >= RSI_OVERSOLD, (RSI_OVERSOLD - rsi) / RSI_OVERSOLD)
            rsi_score = rsi_score.where(rsi <= RSI_OVERBOUGHT, -(rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT))

            # ── Component 2: Bollinger Band Score ──
            # BB Position < 0.2 → near lower band (oversold, positive score)
            # BB Position > 0.8 → near upper band (overbought, negative score)
            bb = features.get("bb_position")
            if bb is not None:
                bb_score = 0.5 - bb  # Centers at 0: positive when below mid-band
                bb_score = bb_score.clip(-1.0, 1.0)
            else:
                bb_score = pd.Series(0.0, index=features.index)

            # ── Component 3: Drawdown Score ──
            # Deeper drawdown → stronger buy signal (to a point)
            # Drawdown of -10% → moderate score
            # Drawdown of -20% → strong score
            dd = features.get("drawdown")
            if dd is not None:
                # Invert: larger drawdown (more negative) → more positive score
                dd_score = -dd * 2  # Scale so -10% drawdown → 0.2 score
                dd_score = dd_score.clip(0.0, 1.5)  # Cap the benefit
            else:
                dd_score = pd.Series(0.0, index=features.index)

            # ── Combine Components ──
            # Equal weight: RSI, Bollinger, Drawdown
            raw_signal = (rsi_score + bb_score + dd_score) / 3.0

            # ── Apply Crisis Filter ──
            # If VIX > 35, DISABLE the buy signal (don't catch falling knives)
            # But keep the SELL signal active (overbought in crisis = extra dangerous)
            if vix_series is not None:
                # Reindex VIX to match the feature dates
                vix_aligned = vix_series.reindex(features.index, method="ffill")
                crisis_mask = vix_aligned > VIX_CRISIS_THRESHOLD

                # During crisis: zero out positive (buy) signals, keep negative (sell)
                raw_signal = raw_signal.where(
                    ~crisis_mask | (raw_signal < 0),
                    0.0
                )

            signals[ticker] = raw_signal

        except Exception as e:
            logger.error(f"Mean reversion error for {ticker}: {e}")
            continue

    # Standardize across ETFs (Z-score on each day)
    signals = signals.apply(lambda row: standardize_row(row), axis=1)

    # Drop early rows
    signals = signals.iloc[252:]  # Need 1 year of history

    logger.info(f"Mean reversion signal computed for {signals.notna().any().sum()} ETFs")

    return signals


def standardize_row(row: pd.Series) -> pd.Series:
    """Standardize a row to zero mean, unit variance (Z-scores)."""
    valid = row.dropna()
    if len(valid) < 3 or valid.std() == 0:
        return pd.Series(0.0, index=row.index)
    return (row - valid.mean()) / valid.std()


def get_latest_mean_reversion_scores(
    price_features: dict,
    macro_features: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.Series:
    """
    Get the most recent mean reversion scores for today.

    Returns:
        Series with ticker as index, mean reversion Z-score as values
        Positive = oversold (buy opportunity)
        Negative = overbought (sell/trim)
    """
    signals = compute_mean_reversion(price_features, macro_features, prices)
    if signals.empty:
        return pd.Series(dtype=float)

    return signals.iloc[-1].sort_values(ascending=False)
