"""
JARVIS V2 - Signal 1: Cross-Sectional Momentum
=================================================
Inspired by: AQR Capital / Jegadeesh-Titman (1993)

HOW THIS WORKS (for non-coders):
- Look at all 25 ETFs and rank them by performance over the last 6 months
- The WINNERS (top 20%) get a POSITIVE score → buy signal
- The LOSERS (bottom 20%) get a NEGATIVE score → avoid/sell signal
- The middle 60% get a neutral score → no strong opinion

WHY IT WORKS:
- This is one of the most documented anomalies in finance
- Assets that have been going up tend to KEEP going up (for a while)
- Assets that have been going down tend to KEEP going down (for a while)
- This "momentum effect" has been found in every market, every country,
  going back over 200 years of data

THE SKIP-MONTH TRICK:
- We skip the most recent month's return in the calculation
- Why? The last month shows "short-term reversal" not momentum
- Academic research proves this improves the signal significantly
"""

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import MOMENTUM_LOOKBACK, MOMENTUM_SKIP


def compute_cross_sectional_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional momentum scores for all ETFs.

    Parameters:
        prices: DataFrame with date index, ticker columns, adj_close values

    Returns:
        DataFrame with date index, ticker columns, Z-score values
        Positive Z-score = strong momentum (buy candidate)
        Negative Z-score = weak momentum (avoid/sell candidate)
    """
    if prices.empty:
        logger.warning("Cross-momentum: No price data provided")
        return pd.DataFrame()

    logger.info(f"Computing cross-sectional momentum (lookback={MOMENTUM_LOOKBACK}, skip={MOMENTUM_SKIP})")

    # Step 1: Calculate 6-month return for each ETF
    # pct_change(126) = (price_today / price_126_days_ago) - 1
    total_return = prices.pct_change(MOMENTUM_LOOKBACK)

    # Step 2: Calculate the most recent 1-month return (to subtract)
    recent_return = prices.pct_change(MOMENTUM_SKIP)

    # Step 3: Momentum = 6-month return MINUS last month's return
    # This is the "12-1 month momentum" trick used by every serious quant fund
    momentum = total_return - recent_return

    # Step 4: Convert to Z-scores (standardize across ETFs on each day)
    # Z-score tells us "how many standard deviations above/below average"
    # SPY has Z-score of +1.5 means SPY's momentum is 1.5 std above the average ETF
    z_scores = momentum.apply(lambda row: z_score_row(row), axis=1)

    # Step 5: Drop rows where we don't have enough history
    z_scores = z_scores.iloc[MOMENTUM_LOOKBACK + 10:]  # Safety buffer

    logger.info(f"Cross-momentum signal computed for {len(z_scores.columns)} ETFs, "
                f"{len(z_scores)} trading days")

    return z_scores


def z_score_row(row: pd.Series) -> pd.Series:
    """
    Convert a single row of raw values into Z-scores.
    Z-score = (value - mean) / standard_deviation

    Handles edge cases:
    - If std is zero (all ETFs have same return), returns all zeros
    - NaN values stay NaN
    """
    valid = row.dropna()
    if len(valid) < 3 or valid.std() == 0:
        return pd.Series(0.0, index=row.index)

    mean = valid.mean()
    std = valid.std()
    return (row - mean) / std


def get_momentum_quintiles(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each ETF into quintiles (5 groups) based on momentum.

    Returns DataFrame where values are:
        5 = Top quintile (strongest momentum - BUY)
        4 = Above average
        3 = Average
        2 = Below average
        1 = Bottom quintile (weakest momentum - SELL/AVOID)

    Useful for understanding the signal's recommendations.
    """
    total_return = prices.pct_change(MOMENTUM_LOOKBACK)
    recent_return = prices.pct_change(MOMENTUM_SKIP)
    momentum = total_return - recent_return

    # Rank into 5 groups
    quintiles = momentum.apply(
        lambda row: pd.qcut(row.dropna(), q=5, labels=[1, 2, 3, 4, 5]).reindex(row.index),
        axis=1
    )

    return quintiles


def get_latest_momentum_scores(prices: pd.DataFrame) -> pd.Series:
    """
    Get the most recent momentum Z-scores for today's trading decision.

    Returns:
        Series with ticker as index, Z-score as values
        Example: SPY: 1.2, QQQ: 0.8, TLT: -1.5, ...
    """
    z_scores = compute_cross_sectional_momentum(prices)
    if z_scores.empty:
        return pd.Series(dtype=float)

    latest = z_scores.iloc[-1]
    return latest.sort_values(ascending=False)
