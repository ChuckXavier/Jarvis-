"""
JARVIS V2 - Portfolio Optimizer
=================================
Converts alpha scores into actual dollar-weighted positions.

HOW THIS WORKS (for non-coders):
- The ensemble gives each ETF a score (e.g., SLV: +1.82, UUP: -0.49)
- This optimizer decides: "OK, how many DOLLARS should go into each ETF?"
- It uses two layers:
    1. CORE (60%): Risk Parity — spread risk equally across asset classes
    2. SATELLITE (35%): Alpha Tilt — put more money where signals are strongest
    3. CASH (5%): Always keep some cash as a safety buffer

WHY RISK PARITY FOR THE CORE?
- Bridgewater's All-Weather Fund uses this: don't bet everything on stocks
- Stocks are 3-4x more volatile than bonds, so "50/50 stocks/bonds"
  actually means 80%+ of your risk is in stocks
- Risk parity EQUALIZES the risk contribution of each asset class
- Result: smoother returns, smaller drawdowns, better sleep at night
"""

import pandas as pd
import numpy as np
from loguru import logger
from scipy.optimize import minimize

from config.settings import (
    CORE_ALLOCATION_PCT, SATELLITE_ALLOCATION_PCT, CASH_RESERVE_PCT,
    KELLY_MULTIPLIER, MAX_SINGLE_POSITION_PCT, MIN_POSITION_PCT,
)
from config.universe import RISK_PARITY_GROUPS, get_asset_class_map


def optimize_portfolio(
    alpha_scores: pd.Series,
    prices: pd.DataFrame,
    portfolio_value: float,
) -> dict:
    """
    Build the target portfolio weights.

    Parameters:
        alpha_scores: Series {ticker: score} — from the ensemble
        prices: DataFrame of recent prices (for volatility/correlation)
        portfolio_value: total portfolio value in dollars

    Returns:
        dict with:
            "target_weights": dict {ticker: weight} (weights sum to ~0.95)
            "core_weights": dict — the risk parity component
            "satellite_weights": dict — the alpha-driven component
            "position_sizes": dict {ticker: dollar_amount}
            "expected_positions": int
    """
    logger.info("Optimizing portfolio...")

    # Step 1: Build the CORE portfolio (60%) — Risk Parity
    core_weights = compute_risk_parity(prices)

    # Step 2: Build the SATELLITE portfolio (35%) — Alpha Tilt
    satellite_weights = compute_alpha_tilt(alpha_scores, prices)

    # Step 3: Combine Core + Satellite + Cash
    target_weights = combine_portfolios(core_weights, satellite_weights)

    # Step 4: Apply Kelly sizing (conservative: 25% Kelly)
    target_weights = apply_kelly_sizing(target_weights, alpha_scores)

    # Step 5: Convert to dollar amounts
    position_sizes = {}
    for ticker, weight in target_weights.items():
        dollar_amount = portfolio_value * weight
        if abs(dollar_amount) >= 10:  # Minimum $10 position
            position_sizes[ticker] = round(dollar_amount, 2)

    # Summary
    n_positions = sum(1 for w in target_weights.values() if abs(w) > MIN_POSITION_PCT)
    total_invested = sum(w for w in target_weights.values() if w > 0)
    cash_pct = 1.0 - total_invested

    logger.info(f"Portfolio optimized: {n_positions} positions, "
                f"{total_invested:.1%} invested, {cash_pct:.1%} cash")

    return {
        "target_weights": target_weights,
        "core_weights": core_weights,
        "satellite_weights": satellite_weights,
        "position_sizes": position_sizes,
        "expected_positions": n_positions,
        "cash_pct": cash_pct,
    }


def compute_risk_parity(prices: pd.DataFrame) -> dict:
    """
    Compute risk parity weights across asset class groups.

    The idea: allocate so each asset CLASS contributes EQUAL risk.
    If stocks are 3x more volatile than bonds, stocks get 1/3 the weight.
    """
    logger.info("  Computing risk parity core...")

    # Calculate volatility for each group
    group_vols = {}
    group_representatives = {}

    for group_name, tickers in RISK_PARITY_GROUPS.items():
        available = [t for t in tickers if t in prices.columns]
        if not available:
            continue

        # Use average volatility of ETFs in this group
        returns = prices[available].pct_change().dropna()
        if returns.empty or len(returns) < 63:
            continue

        # Annualized volatility of an equal-weight portfolio of this group
        group_return = returns.mean(axis=1)
        vol = group_return.tail(63).std() * np.sqrt(252)

        if vol > 0:
            group_vols[group_name] = vol
            group_representatives[group_name] = available

    if not group_vols:
        logger.warning("Cannot compute risk parity — using equal weight")
        return _equal_weight_fallback(prices)

    # Inverse volatility weighting
    inv_vols = {g: 1.0 / v for g, v in group_vols.items()}
    total_inv = sum(inv_vols.values())

    group_weights = {g: (iv / total_inv) * CORE_ALLOCATION_PCT for g, iv in inv_vols.items()}

    # Distribute group weight equally among ETFs in each group
    etf_weights = {}
    for group_name, group_weight in group_weights.items():
        tickers = group_representatives[group_name]
        per_etf = group_weight / len(tickers)
        for ticker in tickers:
            etf_weights[ticker] = per_etf

    logger.info(f"  Risk parity: {len(group_weights)} groups, {len(etf_weights)} ETFs")
    for group, weight in group_weights.items():
        logger.info(f"    {group}: {weight:.1%} (vol: {group_vols[group]:.1%})")

    return etf_weights


def compute_alpha_tilt(alpha_scores: pd.Series, prices: pd.DataFrame) -> dict:
    """
    Convert alpha scores into position weights for the satellite portfolio.

    Positive alpha → overweight (buy more than neutral)
    Negative alpha → underweight (buy less or avoid)
    Zero alpha → no position

    The weights are proportional to the STRENGTH of the signal.
    """
    logger.info("  Computing alpha satellite tilt...")

    if alpha_scores.empty:
        logger.warning("No alpha scores — satellite will be empty")
        return {}

    # Only take positions where we have conviction
    # Threshold: score must be at least 0.1 in absolute value
    valid = alpha_scores[alpha_scores.abs() > 0.1].copy()

    if valid.empty:
        return {}

    # For the satellite, we only take LONG positions (no shorting ETFs)
    # Negative scores → 0 weight (just don't hold it)
    long_scores = valid.clip(lower=0)

    if long_scores.sum() <= 0:
        return {}

    # Normalize to sum to SATELLITE_ALLOCATION_PCT
    weights = (long_scores / long_scores.sum()) * SATELLITE_ALLOCATION_PCT

    # Cap any single position
    weights = weights.clip(upper=MAX_SINGLE_POSITION_PCT)

    # Re-normalize after capping
    if weights.sum() > 0:
        weights = (weights / weights.sum()) * SATELLITE_ALLOCATION_PCT

    result = weights.to_dict()
    logger.info(f"  Alpha tilt: {sum(1 for w in result.values() if w > 0.01)} positions")

    return result


def combine_portfolios(core: dict, satellite: dict) -> dict:
    """
    Merge core (risk parity) and satellite (alpha) into one target portfolio.
    """
    combined = {}

    # Start with core weights
    for ticker, weight in core.items():
        combined[ticker] = weight

    # Add satellite weights
    for ticker, weight in satellite.items():
        combined[ticker] = combined.get(ticker, 0) + weight

    # Cap any position that exceeds the max
    for ticker in combined:
        if combined[ticker] > MAX_SINGLE_POSITION_PCT:
            combined[ticker] = MAX_SINGLE_POSITION_PCT

    # Ensure total doesn't exceed 1 - cash reserve
    max_invested = 1.0 - CASH_RESERVE_PCT
    total = sum(w for w in combined.values() if w > 0)

    if total > max_invested:
        scale = max_invested / total
        combined = {k: v * scale for k, v in combined.items()}

    return combined


def apply_kelly_sizing(weights: dict, alpha_scores: pd.Series) -> dict:
    """
    Apply fractional Kelly criterion to size positions.

    Full Kelly = optimal mathematical sizing for maximum growth
    Quarter Kelly (0.25x) = conservative version that:
    - Reduces drawdowns significantly
    - Sacrifices some return for much better risk-adjusted performance
    - Accounts for the fact that our edge estimates are imprecise

    In practice: if the signal says "bet big on SLV," Kelly says
    "OK but only 25% as big as the pure math suggests."
    """
    adjusted = {}

    for ticker, weight in weights.items():
        if weight <= 0:
            adjusted[ticker] = 0.0
            continue

        score = alpha_scores.get(ticker, 0)

        if score > 0:
            # Higher confidence (score) → closer to full weight
            # Lower confidence → reduce toward zero
            confidence = min(abs(score), 2.0) / 2.0  # 0 to 1 scale
            kelly_factor = KELLY_MULTIPLIER + (1 - KELLY_MULTIPLIER) * confidence
            adjusted[ticker] = weight * kelly_factor
        else:
            # Negative score but positive core weight → reduce but don't eliminate
            adjusted[ticker] = weight * KELLY_MULTIPLIER

    return adjusted


def _equal_weight_fallback(prices: pd.DataFrame) -> dict:
    """Fallback: equal weight across all available ETFs."""
    n = len(prices.columns)
    if n == 0:
        return {}
    weight = CORE_ALLOCATION_PCT / n
    return {ticker: weight for ticker in prices.columns}
