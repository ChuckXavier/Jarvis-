"""
JARVIS V2 - Signal 2: Time-Series Trend Following
====================================================
Inspired by: AHL / Man Group / Winton Capital

HOW THIS WORKS (for non-coders):
- Signal 1 compared ETFs to EACH OTHER ("who's winning?")
- This signal compares each ETF to ITS OWN HISTORY ("is it trending up or down?")
- If SPY has been going up for 12 months → positive score (stay long)
- If SPY has been going down for 12 months → negative score (get out)

WHY IT WORKS:
- Trends exist because information diffuses slowly through markets
- When good news comes out, not everyone reacts at the same time
- This creates a gradual price move (trend) that you can ride
- This strategy has been profitable in EVERY asset class tested,
  going back to the 1800s

THE DUAL-TIMEFRAME TRICK:
- We use TWO timeframes: fast (1 month) and slow (12 months)
- Fast catches recent trend changes quickly
- Slow captures big, secular trends
- Combining both gives a more stable signal
"""

import pandas as pd
import numpy as np
from loguru import logger

from config.settings import TREND_FAST_LOOKBACK, TREND_SLOW_LOOKBACK


def compute_trend_following(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-series trend following scores for all ETFs.

    Parameters:
        prices: DataFrame with date index, ticker columns, adj_close values

    Returns:
        DataFrame with date index, ticker columns, signal values
        Positive = uptrend (go long)
        Negative = downtrend (go to cash or hedge)
        Near zero = no clear trend
    """
    if prices.empty:
        logger.warning("Trend following: No price data provided")
        return pd.DataFrame()

    logger.info(f"Computing trend following (fast={TREND_FAST_LOOKBACK}d, slow={TREND_SLOW_LOOKBACK}d)")

    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ticker in prices.columns:
        try:
            series = prices[ticker].dropna()
            if len(series) < TREND_SLOW_LOOKBACK + 10:
                continue

            # ── Fast Signal (1-month trend) ──
            # Positive if price is above its 1-month-ago level
            fast_return = series.pct_change(TREND_FAST_LOOKBACK)
            # Normalize: divide by volatility so all ETFs are on same scale
            fast_vol = series.pct_change().rolling(TREND_FAST_LOOKBACK).std()
            fast_signal = fast_return / (fast_vol * np.sqrt(TREND_FAST_LOOKBACK))

            # ── Slow Signal (12-month trend) ──
            # Positive if price is above its 12-month-ago level
            slow_return = series.pct_change(TREND_SLOW_LOOKBACK)
            slow_vol = series.pct_change().rolling(TREND_SLOW_LOOKBACK).std()
            slow_signal = slow_return / (slow_vol * np.sqrt(TREND_SLOW_LOOKBACK))

            # ── Combined Signal ──
            # 40% fast + 60% slow (slow gets more weight for stability)
            combined = 0.4 * fast_signal + 0.6 * slow_signal

            # ── Clip extreme values ──
            # Cap at +/- 3 to prevent any single ETF from dominating
            combined = combined.clip(-3.0, 3.0)

            signals[ticker] = combined

        except Exception as e:
            logger.error(f"Trend signal error for {ticker}: {e}")
            continue

    # Drop early rows where we don't have enough history
    signals = signals.iloc[TREND_SLOW_LOOKBACK + 10:]

    logger.info(f"Trend following signal computed for {signals.notna().any().sum()} ETFs, "
                f"{len(signals)} trading days")

    return signals


def compute_trend_strength(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple trend strength indicator for each ETF.
    Uses the price position relative to key moving averages.

    Returns values from -1 (strong downtrend) to +1 (strong uptrend):
    - Price above both SMA50 and SMA200 = strong uptrend (+1.0)
    - Price above SMA200 but below SMA50 = weakening uptrend (+0.5)
    - Price below SMA200 but above SMA50 = early recovery (-0.5)
    - Price below both = strong downtrend (-1.0)
    """
    strength = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for ticker in prices.columns:
        try:
            series = prices[ticker].dropna()
            if len(series) < 200:
                continue

            sma50 = series.rolling(50).mean()
            sma200 = series.rolling(200).mean()

            # Score based on price position relative to MAs
            above_50 = (series > sma50).astype(float)
            above_200 = (series > sma200).astype(float)

            # Combined: ranges from -1 to +1
            score = (above_50 + above_200) - 1.0

            strength[ticker] = score

        except Exception:
            continue

    return strength


def get_latest_trend_scores(prices: pd.DataFrame) -> pd.Series:
    """
    Get the most recent trend scores for today's trading decision.

    Returns:
        Series with ticker as index, trend score as values
        Positive = uptrend, Negative = downtrend
    """
    signals = compute_trend_following(prices)
    if signals.empty:
        return pd.Series(dtype=float)

    latest = signals.iloc[-1]
    return latest.sort_values(ascending=False)
