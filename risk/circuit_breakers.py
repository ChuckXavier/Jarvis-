"""
JARVIS V2 - Circuit Breakers (Risk Layer 5)
==============================================
Emergency stops that HALT all trading when things go badly wrong.

HOW THIS WORKS (for non-coders):
- These are like the emergency brakes on a train
- If the portfolio loses too much too fast, Jarvis STOPS trading
- No human override possible — the math decides
- This is what prevents a small loss from becoming a catastrophe

THE 4 CIRCUIT BREAKERS:
1. Daily Loss > 3%    → HALT all new trades for the day
2. Weekly Loss > 5%   → Reduce all positions by 50%
3. Max Drawdown > 15% → Liquidate to 50% cash immediately
4. VIX > 35           → Reduce all positions by 50% (market in panic)
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta

from config.settings import (
    CIRCUIT_DAILY_LOSS_PCT, CIRCUIT_WEEKLY_LOSS_PCT,
    CIRCUIT_MAX_DRAWDOWN_PCT, CIRCUIT_VIX_THRESHOLD,
)


def check_all_circuit_breakers(
    portfolio_value: float,
    portfolio_history: pd.DataFrame = None,
    prices: pd.DataFrame = None,
) -> dict:
    """
    Check ALL circuit breakers. If ANY triggers, trading is halted or reduced.

    Parameters:
        portfolio_value: current total portfolio value
        portfolio_history: DataFrame with 'date' index and 'total_value' column
        prices: recent price data (used to check VIX)

    Returns:
        dict with:
            "halt_trading": True/False — should we stop ALL trading?
            "reduce_positions": True/False — should we cut positions by 50%?
            "reduction_factor": float — multiply all positions by this (1.0 = no change, 0.5 = halve)
            "reason": str — why the circuit breaker triggered
            "breakers_triggered": list of which breakers fired
    """
    breakers_triggered = []
    halt = False
    reduce = False
    reduction_factor = 1.0
    reasons = []

    # ── Breaker 1: Daily Loss ──
    daily = _check_daily_loss(portfolio_value, portfolio_history)
    if daily["triggered"]:
        breakers_triggered.append("DAILY_LOSS")
        halt = True
        reasons.append(daily["reason"])

    # ── Breaker 2: Weekly Loss ──
    weekly = _check_weekly_loss(portfolio_value, portfolio_history)
    if weekly["triggered"]:
        breakers_triggered.append("WEEKLY_LOSS")
        reduce = True
        reduction_factor = min(reduction_factor, 0.5)
        reasons.append(weekly["reason"])

    # ── Breaker 3: Max Drawdown ──
    drawdown = _check_max_drawdown(portfolio_value, portfolio_history)
    if drawdown["triggered"]:
        breakers_triggered.append("MAX_DRAWDOWN")
        halt = True
        reduction_factor = min(reduction_factor, 0.5)
        reasons.append(drawdown["reason"])

    # ── Breaker 4: VIX Spike ──
    vix = _check_vix_level(prices)
    if vix["triggered"]:
        breakers_triggered.append("VIX_SPIKE")
        reduce = True
        reduction_factor = min(reduction_factor, 0.5)
        reasons.append(vix["reason"])

    # Log results
    if breakers_triggered:
        logger.warning(f"  CIRCUIT BREAKERS TRIGGERED: {breakers_triggered}")
        for r in reasons:
            logger.warning(f"    → {r}")
    else:
        logger.info("  Layer 5 (Circuit Breakers): All clear")

    return {
        "halt_trading": halt,
        "reduce_positions": reduce,
        "reduction_factor": reduction_factor,
        "reason": "; ".join(reasons) if reasons else "All clear",
        "breakers_triggered": breakers_triggered,
    }


def _check_daily_loss(portfolio_value: float, history: pd.DataFrame) -> dict:
    """Trigger if portfolio is down more than 3% today."""
    if history is None or history.empty or len(history) < 2:
        return {"triggered": False, "reason": ""}

    try:
        yesterday_value = history["total_value"].iloc[-2]
        if yesterday_value <= 0:
            return {"triggered": False, "reason": ""}

        daily_return = (portfolio_value / yesterday_value) - 1

        if daily_return <= CIRCUIT_DAILY_LOSS_PCT:
            return {
                "triggered": True,
                "reason": f"Daily loss is {daily_return:.2%} (limit: {CIRCUIT_DAILY_LOSS_PCT:.0%}) — HALT ALL TRADING"
            }
    except Exception:
        pass

    return {"triggered": False, "reason": ""}


def _check_weekly_loss(portfolio_value: float, history: pd.DataFrame) -> dict:
    """Trigger if portfolio is down more than 5% this week."""
    if history is None or history.empty or len(history) < 5:
        return {"triggered": False, "reason": ""}

    try:
        # Value from 5 trading days ago
        week_ago_value = history["total_value"].iloc[-6] if len(history) >= 6 else history["total_value"].iloc[0]
        if week_ago_value <= 0:
            return {"triggered": False, "reason": ""}

        weekly_return = (portfolio_value / week_ago_value) - 1

        if weekly_return <= CIRCUIT_WEEKLY_LOSS_PCT:
            return {
                "triggered": True,
                "reason": f"Weekly loss is {weekly_return:.2%} (limit: {CIRCUIT_WEEKLY_LOSS_PCT:.0%}) — REDUCE POSITIONS 50%"
            }
    except Exception:
        pass

    return {"triggered": False, "reason": ""}


def _check_max_drawdown(portfolio_value: float, history: pd.DataFrame) -> dict:
    """Trigger if portfolio drawdown exceeds 15% from peak."""
    if history is None or history.empty:
        return {"triggered": False, "reason": ""}

    try:
        peak = history["total_value"].max()
        if peak <= 0:
            return {"triggered": False, "reason": ""}

        drawdown = (portfolio_value / peak) - 1

        if drawdown <= CIRCUIT_MAX_DRAWDOWN_PCT:
            return {
                "triggered": True,
                "reason": f"Drawdown is {drawdown:.2%} from peak ${peak:,.0f} (limit: {CIRCUIT_MAX_DRAWDOWN_PCT:.0%}) — LIQUIDATE TO 50% CASH"
            }
    except Exception:
        pass

    return {"triggered": False, "reason": ""}


def _check_vix_level(prices: pd.DataFrame) -> dict:
    """Trigger if VIX is above crisis threshold (35)."""
    if prices is None or prices.empty:
        return {"triggered": False, "reason": ""}

    try:
        from data.db import get_macro
        vix_data = get_macro("VIXCLS")
        if vix_data.empty:
            vix_data = get_macro("VIX_YAHOO")

        if vix_data.empty:
            return {"triggered": False, "reason": ""}

        current_vix = vix_data["value"].iloc[-1]

        if current_vix >= CIRCUIT_VIX_THRESHOLD:
            return {
                "triggered": True,
                "reason": f"VIX is {current_vix:.1f} (threshold: {CIRCUIT_VIX_THRESHOLD}) — REDUCE POSITIONS 50%"
            }
    except Exception:
        pass

    return {"triggered": False, "reason": ""}
