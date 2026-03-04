"""
JARVIS V2 - Feature Engine
=============================
Computes all trading features from raw price data. Features are the
"ingredients" that the signal models use to make trading decisions.

HOW THIS WORKS (for non-coders):
- Raw price data is just: open, high, low, close, volume per day
- Features TRANSFORM that raw data into useful numbers, like:
    - "How much has SPY gone up in the last 6 months?" (momentum)
    - "Is QQQ's price way below its normal range?" (mean reversion)
    - "Is volume unusually high today?" (activity detection)
- Think of features as the "vital signs" of each ETF
- The signal models then read these vital signs to make decisions
"""

import pandas as pd
import numpy as np
from loguru import logger

from data.db import get_all_prices, get_macro
from config.universe import get_all_tickers


def compute_all_features(prices: pd.DataFrame = None) -> dict:
    """
    Compute ALL features for ALL ETFs.

    Parameters:
        prices: DataFrame with date index, ticker columns, adj_close values.
                If None, fetches from database automatically.

    Returns:
        A dict with:
            "price_features": DataFrame (date × ticker × feature columns)
            "cross_sectional": DataFrame (date × ticker × rank columns)
            "macro_features": DataFrame (date × macro indicator columns)

    This is the MASTER function that the signal engine calls every day.
    """
    if prices is None:
        prices = get_all_prices()

    if prices.empty:
        logger.error("No price data available for feature computation!")
        return {"price_features": pd.DataFrame(), "cross_sectional": pd.DataFrame(), "macro_features": pd.DataFrame()}

    logger.info(f"Computing features for {len(prices.columns)} ETFs, {len(prices)} trading days")

    # Step 1: Per-ETF price features
    price_features = compute_price_features(prices)

    # Step 2: Cross-sectional features (comparing ETFs to each other)
    cross_sectional = compute_cross_sectional_features(prices, price_features)

    # Step 3: Macro/regime features
    macro_features = compute_macro_features()

    logger.info(f"Features computed: {len(price_features)} ETFs with price features, "
                f"{len(cross_sectional)} ETFs with cross-sectional features")

    return {
        "price_features": price_features,
        "cross_sectional": cross_sectional,
        "macro_features": macro_features,
    }


# ============================================================
# PER-ETF PRICE FEATURES
# ============================================================

def compute_price_features(prices: pd.DataFrame) -> dict:
    """
    Compute price-based features for each individual ETF.

    Returns a dict where:
        key = ticker symbol (e.g., "SPY")
        value = DataFrame with date index and feature columns
    """
    all_features = {}

    for ticker in prices.columns:
        try:
            series = prices[ticker].dropna()
            if len(series) < 252:  # Need at least 1 year
                logger.warning(f"{ticker}: Insufficient data ({len(series)} days), skipping")
                continue

            features = pd.DataFrame(index=series.index)

            # ── RETURNS (different lookback periods) ──
            # These measure "how much has the price moved over X days?"
            features["return_1d"] = series.pct_change(1)
            features["return_5d"] = series.pct_change(5)
            features["return_21d"] = series.pct_change(21)
            features["return_63d"] = series.pct_change(63)
            features["return_126d"] = series.pct_change(126)
            features["return_252d"] = series.pct_change(252)

            # ── VOLATILITY ──
            # How wildly is the price swinging? Higher = more risky
            daily_returns = series.pct_change()
            features["volatility_21d"] = daily_returns.rolling(21).std() * np.sqrt(252)  # Annualized
            features["volatility_63d"] = daily_returns.rolling(63).std() * np.sqrt(252)

            # Vol Ratio: is recent volatility higher or lower than normal?
            # > 1.0 means volatility is INCREASING (danger signal)
            # < 1.0 means volatility is DECREASING (calm signal)
            features["vol_ratio"] = features["volatility_21d"] / features["volatility_63d"]

            # ── RSI (Relative Strength Index) ──
            # Measures if ETF is "overbought" (>70) or "oversold" (<30)
            features["rsi_14"] = compute_rsi(series, period=14)

            # ── MACD (Moving Average Convergence Divergence) ──
            # Measures trend strength. Positive = uptrend, Negative = downtrend
            ema12 = series.ewm(span=12, adjust=False).mean()
            ema26 = series.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            features["macd"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = macd_line - signal_line

            # ── BOLLINGER BANDS ──
            # Creates a "channel" around the price. When price is near
            # the bottom band, it might be oversold (buy signal).
            sma20 = series.rolling(20).mean()
            std20 = series.rolling(20).std()
            bb_upper = sma20 + (2 * std20)
            bb_lower = sma20 - (2 * std20)
            # BB Position: 0 = at lower band, 1 = at upper band
            features["bb_position"] = (series - bb_lower) / (bb_upper - bb_lower)

            # ── ATR (Average True Range) ──
            # Measures how much the price moves on a typical day
            # Used for position sizing (bigger ATR = smaller position)
            features["atr_14"] = compute_atr(prices, ticker, period=14)

            # ── VOLUME RATIO ──
            # Is today's volume higher or lower than the 21-day average?
            # High volume + price move = strong signal
            # Low volume + price move = weak/unreliable signal
            features["volume_ratio"] = compute_volume_ratio(prices, ticker)

            # ── DRAWDOWN ──
            # How far is the current price from its 1-year peak?
            # -0.10 means the ETF is 10% below its peak
            rolling_max = series.rolling(252).max()
            features["drawdown"] = (series / rolling_max) - 1

            # ── SMA CROSS ──
            # Golden Cross: 50-day SMA above 200-day SMA (bullish)
            # Death Cross: 50-day SMA below 200-day SMA (bearish)
            sma50 = series.rolling(50).mean()
            sma200 = series.rolling(200).mean()
            features["sma_cross"] = (sma50 / sma200) - 1

            # ── MOMENTUM SCORE (12-1 month) ──
            # The "classic" momentum signal: 12-month return minus last month
            # Skipping the most recent month avoids short-term reversal
            features["momentum_12_1"] = series.pct_change(252) - series.pct_change(21)

            all_features[ticker] = features

        except Exception as e:
            logger.error(f"{ticker}: Feature computation error - {e}")
            continue

    logger.info(f"Price features computed for {len(all_features)} ETFs")
    return all_features


# ============================================================
# CROSS-SECTIONAL FEATURES
# ============================================================

def compute_cross_sectional_features(prices: pd.DataFrame, price_features: dict) -> pd.DataFrame:
    """
    Compute features that compare ETFs TO EACH OTHER on each day.

    For example: "SPY's 6-month return ranks in the 80th percentile
    compared to all other ETFs" → Momentum_Rank = 0.80

    These rankings are what the cross-sectional momentum signal uses.
    """
    # Build a matrix of 126-day returns for all ETFs
    returns_126d = prices.pct_change(126)
    returns_21d = prices.pct_change(21)
    returns_252d = prices.pct_change(252)

    # Rank each ETF against all others on each day (0 = worst, 1 = best)
    momentum_rank = returns_126d.rank(axis=1, pct=True)
    short_term_rank = returns_21d.rank(axis=1, pct=True)
    annual_rank = returns_252d.rank(axis=1, pct=True)

    # Volatility ranking
    vol_data = {}
    for ticker in prices.columns:
        if ticker in price_features:
            vol_data[ticker] = price_features[ticker].get("volatility_21d", pd.Series(dtype=float))
    vol_df = pd.DataFrame(vol_data)
    volatility_rank = vol_df.rank(axis=1, pct=True)

    # Volume ranking
    volume_data = {}
    for ticker in prices.columns:
        if ticker in price_features:
            volume_data[ticker] = price_features[ticker].get("volume_ratio", pd.Series(dtype=float))
    volume_df = pd.DataFrame(volume_data)
    volume_rank = volume_df.rank(axis=1, pct=True)

    # Correlation to SPY (how much does each ETF move with the market?)
    spy_returns = prices["SPY"].pct_change() if "SPY" in prices.columns else None
    corr_to_spy = {}
    if spy_returns is not None:
        for ticker in prices.columns:
            if ticker != "SPY":
                etf_returns = prices[ticker].pct_change()
                corr_to_spy[ticker] = etf_returns.rolling(63).corr(spy_returns)
            else:
                corr_to_spy[ticker] = pd.Series(1.0, index=prices.index)

    result = {
        "momentum_rank": momentum_rank,
        "short_term_rank": short_term_rank,
        "annual_rank": annual_rank,
        "volatility_rank": volatility_rank,
        "volume_rank": volume_rank,
        "correlation_to_spy": pd.DataFrame(corr_to_spy),
    }

    return result


# ============================================================
# MACRO / REGIME FEATURES
# ============================================================

def compute_macro_features() -> pd.DataFrame:
    """
    Compute macro-level features from FRED data.
    These describe the OVERALL market environment, not individual ETFs.

    Key features:
    - Yield curve slope (positive = normal, negative/inverted = recession warning)
    - Credit spread level (high = fear, low = complacency)
    - VIX level and change (fear gauge)
    """
    features = pd.DataFrame()

    try:
        # Yield Curve: 10Y - 2Y spread
        yield_curve = get_macro("T10Y2Y")
        if not yield_curve.empty:
            features["yield_curve"] = yield_curve["value"]
            features["yield_curve_change_21d"] = features["yield_curve"].diff(21)

        # Credit Spread (High Yield)
        credit = get_macro("BAMLH0A0HYM2")
        if not credit.empty:
            features["credit_spread"] = credit["value"]
            features["credit_spread_change_21d"] = features["credit_spread"].diff(21)

        # VIX
        vix = get_macro("VIXCLS")
        if vix.empty:
            vix = get_macro("VIX_YAHOO")
        if not vix.empty:
            features["vix"] = vix["value"]
            features["vix_change_5d"] = features["vix"].diff(5)
            features["vix_sma_21d"] = features["vix"].rolling(21).mean()
            # VIX term structure proxy: current vs 21-day average
            features["vix_term_structure"] = features["vix"] / features["vix_sma_21d"]

        # Unemployment Rate
        unemployment = get_macro("UNRATE")
        if not unemployment.empty:
            features["unemployment"] = unemployment["value"]

        # Initial Jobless Claims
        claims = get_macro("ICSA")
        if not claims.empty:
            features["jobless_claims"] = claims["value"]
            features["claims_sma_4w"] = features["jobless_claims"].rolling(4).mean()

        # Forward-fill macro data (it's released less frequently than daily)
        features = features.ffill()

        logger.info(f"Macro features computed: {len(features.columns)} indicators, {len(features)} observations")

    except Exception as e:
        logger.error(f"Macro feature computation error: {e}")

    return features


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index).

    RSI ranges from 0 to 100:
    - Above 70 = "overbought" (price may be too high, could drop)
    - Below 30 = "oversold" (price may be too low, could bounce)
    - Around 50 = neutral
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use exponential smoothing after initial window
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_atr(all_prices: pd.DataFrame, ticker: str, period: int = 14) -> pd.Series:
    """
    Compute ATR (Average True Range) — a measure of daily price volatility.
    Uses the full OHLC data, not just adjusted close.
    """
    try:
        from data.db import get_prices
        df = get_prices(ticker)
        if df.empty or "high" not in df.columns:
            return pd.Series(dtype=float)

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        # Reindex to match the prices DataFrame
        atr.index = pd.to_datetime(atr.index)
        return atr

    except Exception:
        return pd.Series(dtype=float)


def compute_volume_ratio(all_prices: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Compute volume ratio: today's volume / 21-day average volume.
    > 1.5 means unusually high volume (something is happening)
    < 0.5 means unusually low volume (quiet day)
    """
    try:
        from data.db import get_prices
        df = get_prices(ticker)
        if df.empty or "volume" not in df.columns:
            return pd.Series(dtype=float)

        volume = df["volume"].astype(float)
        avg_volume = volume.rolling(21).mean()
        ratio = volume / avg_volume

        ratio.index = pd.to_datetime(ratio.index)
        return ratio

    except Exception:
        return pd.Series(dtype=float)


def get_latest_features() -> dict:
    """
    Compute features and return only the MOST RECENT day's values.
    This is what the daily trading pipeline calls.
    """
    result = compute_all_features()

    latest = {}
    for ticker, df in result["price_features"].items():
        if not df.empty:
            latest[ticker] = df.iloc[-1].to_dict()

    return latest
