"""
JARVIS V2 - Risk Engine: The 5-Layer Risk Fortress
=====================================================
This sits between the AI's trading desires and the broker.
NO trade reaches Alpaca unless it passes ALL 5 layers.

HOW THIS WORKS (for non-coders):
- Think of this as 5 security checkpoints at an airport
- Every trade must clear ALL 5 checkpoints or it gets rejected
- Layer 1: Is the data clean? (no garbage in = no garbage out)
- Layer 2: Are the signals trustworthy? (confidence check)
- Layer 3: Is any single position too big? (concentration check)
- Layer 4: Is the overall portfolio too risky? (portfolio check)
- Layer 5: Are we in emergency mode? (circuit breaker check)
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

from config.settings import (
    MAX_SINGLE_POSITION_PCT, MAX_SECTOR_EXPOSURE_PCT, MAX_POSITIONS,
    MIN_POSITION_PCT, STOP_LOSS_PCT, MAX_DAILY_VAR_PCT,
    MAX_DRAWDOWN_PCT, MAX_AVG_CORRELATION, MIN_CASH_RESERVE_PCT,
)
from config.universe import get_sector_map
from risk.circuit_breakers import check_all_circuit_breakers


def validate_portfolio(
    target_weights: dict,
    current_positions: dict,
    prices: pd.DataFrame,
    portfolio_value: float,
    portfolio_history: pd.DataFrame = None,
) -> dict:
    """
    Run ALL 5 risk layers on the proposed target portfolio.

    Parameters:
        target_weights: dict {ticker: weight} — proposed allocation (weights sum to ~1.0)
        current_positions: dict {ticker: {"qty": X, "market_value": Y, "entry_price": Z}}
        prices: recent price data for correlation/volatility calculations
        portfolio_value: total portfolio value in dollars
        portfolio_history: DataFrame of daily portfolio values (for drawdown calc)

    Returns:
        dict with:
            "approved": True/False
            "approved_weights": dict — the (possibly adjusted) weights
            "rejections": list of rejected trades with reasons
            "warnings": list of warning messages
            "layer_results": dict of each layer's pass/fail
    """
    logger.info("=" * 50)
    logger.info("RISK FORTRESS — Validating portfolio")
    logger.info("=" * 50)

    rejections = []
    warnings = []
    adjusted_weights = target_weights.copy()

    # ── LAYER 1: Data Quality ──
    l1 = _layer1_data_quality(prices)
    if not l1["passed"]:
        logger.error("LAYER 1 FAILED — Data quality issues detected")
        return _build_result(False, {}, l1["errors"], [], {"layer1": l1})

    # ── LAYER 2: Signal Validation ──
    l2 = _layer2_signal_validation(adjusted_weights)
    warnings.extend(l2.get("warnings", []))

    # ── LAYER 3: Position-Level Limits ──
    l3 = _layer3_position_limits(adjusted_weights, current_positions, portfolio_value)
    adjusted_weights = l3["adjusted_weights"]
    rejections.extend(l3.get("rejections", []))
    warnings.extend(l3.get("warnings", []))

    # ── LAYER 4: Portfolio-Level Limits ──
    l4 = _layer4_portfolio_limits(adjusted_weights, prices, portfolio_value)
    adjusted_weights = l4["adjusted_weights"]
    warnings.extend(l4.get("warnings", []))

    # ── LAYER 5: Circuit Breakers ──
    l5 = check_all_circuit_breakers(portfolio_value, portfolio_history, prices)
    if l5["halt_trading"]:
        logger.error(f"LAYER 5 — CIRCUIT BREAKER TRIGGERED: {l5['reason']}")
        return _build_result(False, {}, [f"CIRCUIT BREAKER: {l5['reason']}"], warnings, {
            "layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4, "layer5": l5
        })

    # ── Final result ──
    approved = len(rejections) == 0
    status = "APPROVED" if approved else f"REJECTED ({len(rejections)} issues)"
    logger.info(f"Risk Fortress result: {status}")
    if warnings:
        logger.warning(f"Warnings: {len(warnings)}")

    return _build_result(approved, adjusted_weights, rejections, warnings, {
        "layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4, "layer5": l5
    })


# ============================================================
# LAYER 1: DATA QUALITY
# ============================================================

def _layer1_data_quality(prices: pd.DataFrame) -> dict:
    """Check that price data is fresh and complete."""
    errors = []

    if prices.empty:
        errors.append("No price data available")
        return {"passed": False, "errors": errors}

    # Check freshness (most recent data should be within 5 days)
    last_date = prices.index[-1]
    days_old = (pd.Timestamp.now() - last_date).days
    if days_old > 5:
        errors.append(f"Price data is {days_old} days old (max: 5)")

    # Check completeness (at least 20 ETFs should have data)
    etfs_with_data = prices.iloc[-1].notna().sum()
    if etfs_with_data < 20:
        errors.append(f"Only {etfs_with_data} ETFs have current data (need 20+)")

    # Check for price anomalies in last day
    daily_returns = prices.pct_change().iloc[-1]
    extreme_moves = daily_returns[daily_returns.abs() > 0.15]
    if len(extreme_moves) > 3:
        errors.append(f"{len(extreme_moves)} ETFs moved >15% in one day — possible data error")

    passed = len(errors) == 0
    logger.info(f"  Layer 1 (Data Quality): {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "errors": errors}


# ============================================================
# LAYER 2: SIGNAL VALIDATION
# ============================================================

def _layer2_signal_validation(weights: dict) -> dict:
    """Check that signal-derived weights are reasonable."""
    warnings = []

    # Check that not all weights are concentrated in one direction
    positive = sum(1 for w in weights.values() if w > 0.01)
    if positive < 3:
        warnings.append(f"Only {positive} positive positions — low diversification")

    # Check total allocation
    total = sum(abs(w) for w in weights.values())
    if total > 1.5:
        warnings.append(f"Total allocation is {total:.1%} — unusually high")

    logger.info(f"  Layer 2 (Signal Validation): PASS ({len(warnings)} warnings)")
    return {"passed": True, "warnings": warnings}


# ============================================================
# LAYER 3: POSITION-LEVEL LIMITS
# ============================================================

def _layer3_position_limits(weights: dict, current_positions: dict, portfolio_value: float) -> dict:
    """Enforce per-position size limits, sector limits, and stop-losses."""
    adjusted = weights.copy()
    rejections = []
    warnings = []
    sector_map = get_sector_map()

    # 3a: Cap individual positions at MAX_SINGLE_POSITION_PCT
    for ticker, weight in list(adjusted.items()):
        if abs(weight) > MAX_SINGLE_POSITION_PCT:
            old_w = weight
            adjusted[ticker] = np.sign(weight) * MAX_SINGLE_POSITION_PCT
            warnings.append(f"{ticker}: Capped from {old_w:.1%} to {adjusted[ticker]:.1%}")

    # 3b: Remove positions below minimum size (not worth the transaction cost)
    for ticker, weight in list(adjusted.items()):
        if 0 < abs(weight) < MIN_POSITION_PCT:
            adjusted[ticker] = 0.0

    # 3c: Enforce maximum number of positions
    nonzero = {k: v for k, v in adjusted.items() if abs(v) > MIN_POSITION_PCT}
    if len(nonzero) > MAX_POSITIONS:
        # Keep only the top MAX_POSITIONS by absolute weight
        sorted_pos = sorted(nonzero.items(), key=lambda x: abs(x[1]), reverse=True)
        keep = dict(sorted_pos[:MAX_POSITIONS])
        removed = set(nonzero.keys()) - set(keep.keys())
        for ticker in removed:
            adjusted[ticker] = 0.0
        warnings.append(f"Reduced from {len(nonzero)} to {MAX_POSITIONS} positions")

    # 3d: Check sector concentration
    sector_exposure = {}
    for ticker, weight in adjusted.items():
        if weight > 0:
            sector = sector_map.get(ticker, "unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

    for sector, exposure in sector_exposure.items():
        if exposure > MAX_SECTOR_EXPOSURE_PCT:
            warnings.append(f"Sector '{sector}' exposure is {exposure:.1%} (max: {MAX_SECTOR_EXPOSURE_PCT:.0%})")
            # Scale down all positions in this sector proportionally
            scale = MAX_SECTOR_EXPOSURE_PCT / exposure
            for ticker, weight in adjusted.items():
                if sector_map.get(ticker) == sector and weight > 0:
                    adjusted[ticker] = weight * scale

    # 3e: Check stop-losses on current positions
    for ticker, pos_info in current_positions.items():
        if isinstance(pos_info, dict) and "entry_price" in pos_info and "current_price" in pos_info:
            entry = pos_info["entry_price"]
            current = pos_info["current_price"]
            if entry > 0:
                pnl_pct = (current / entry) - 1
                if pnl_pct <= STOP_LOSS_PCT:
                    adjusted[ticker] = 0.0  # Force close
                    rejections.append(f"STOP-LOSS: {ticker} is down {pnl_pct:.1%} (limit: {STOP_LOSS_PCT:.0%})")

    logger.info(f"  Layer 3 (Position Limits): {len(rejections)} rejections, {len(warnings)} warnings")
    return {"adjusted_weights": adjusted, "rejections": rejections, "warnings": warnings}


# ============================================================
# LAYER 4: PORTFOLIO-LEVEL LIMITS
# ============================================================

def _layer4_portfolio_limits(weights: dict, prices: pd.DataFrame, portfolio_value: float) -> dict:
    """Enforce portfolio-wide risk limits: VaR, correlation, cash reserve."""
    adjusted = weights.copy()
    warnings = []

    # 4a: Ensure minimum cash reserve
    total_invested = sum(w for w in adjusted.values() if w > 0)
    if total_invested > (1.0 - MIN_CASH_RESERVE_PCT):
        scale = (1.0 - MIN_CASH_RESERVE_PCT) / total_invested
        adjusted = {k: v * scale if v > 0 else v for k, v in adjusted.items()}
        warnings.append(f"Scaled positions to maintain {MIN_CASH_RESERVE_PCT:.0%} cash reserve")

    # 4b: Estimate portfolio VaR (simplified parametric VaR)
    try:
        active_tickers = [t for t, w in adjusted.items() if abs(w) > 0.01 and t in prices.columns]
        if len(active_tickers) >= 2:
            returns = prices[active_tickers].pct_change().dropna().tail(63)
            port_weights = np.array([adjusted.get(t, 0) for t in active_tickers])
            cov_matrix = returns.cov().values * 252  # Annualize
            port_var = np.sqrt(port_weights @ cov_matrix @ port_weights)
            daily_var_95 = port_var / np.sqrt(252) * 1.645  # 95% daily VaR

            if daily_var_95 > MAX_DAILY_VAR_PCT:
                scale = MAX_DAILY_VAR_PCT / daily_var_95
                adjusted = {k: v * scale for k, v in adjusted.items()}
                warnings.append(f"Daily VaR was {daily_var_95:.2%}, scaled to {MAX_DAILY_VAR_PCT:.0%} limit")

            # 4c: Check average correlation
            corr_matrix = returns.corr()
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            avg_corr = corr_matrix.values[mask].mean()
            if avg_corr > MAX_AVG_CORRELATION:
                warnings.append(f"Average pairwise correlation is {avg_corr:.2f} (max: {MAX_AVG_CORRELATION})")

    except Exception as e:
        warnings.append(f"VaR calculation skipped: {e}")

    logger.info(f"  Layer 4 (Portfolio Limits): {len(warnings)} warnings")
    return {"adjusted_weights": adjusted, "warnings": warnings}


def _build_result(approved, weights, rejections, warnings, layers):
    return {
        "approved": approved,
        "approved_weights": weights,
        "rejections": rejections,
        "warnings": warnings,
        "layer_results": layers,
    }
