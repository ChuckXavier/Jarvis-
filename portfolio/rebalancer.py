"""
JARVIS V2 - Portfolio Rebalancer
==================================
Compares WHERE WE ARE to WHERE WE WANT TO BE and generates the trades
needed to get there.

HOW THIS WORKS (for non-coders):
- The optimizer says: "You should have 8% in SPY, 5% in GLD, 3% in TLT..."
- The rebalancer checks: "You currently have 6% in SPY, 7% in GLD, 0% in TLT..."
- It calculates the difference and creates orders:
    • BUY more SPY (need 2% more)
    • SELL some GLD (have 2% too much)
    • BUY TLT (need 3% from scratch)

DRIFT THRESHOLD:
- We don't rebalance tiny differences (waste of money on commissions)
- Only rebalance if a position has drifted more than 20% from target
- Example: Target is 5%, current is 5.5% → drift is 10% → skip (too small)
- Example: Target is 5%, current is 7% → drift is 40% → rebalance (big enough)

FIX 4: Position sizing now correctly uses total portfolio value for ALL
positions, and properly handles selling non-target positions first to
free up capital for new buys.
"""

import math
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

from config.settings import REBALANCE_DRIFT_THRESHOLD, MAX_DAILY_TRADES


def generate_rebalance_orders(
    target_weights: dict,
    current_positions: dict,
    portfolio_value: float,
    current_prices: dict,
) -> list:
    """
    Generate the list of orders needed to rebalance from current to target.

    Parameters:
        target_weights: dict {ticker: weight} — where we want to be
        current_positions: dict {ticker: {"qty": X, "market_value": Y}}
        portfolio_value: total portfolio value
        current_prices: dict {ticker: current_price}

    Returns:
        list of order dicts, each with:
            ticker, side (buy/sell), quantity, estimated_value, reason
        Orders are sorted: SELLS first, then BUYS (free up cash before spending)
    """
    logger.info("Generating rebalance orders...")

    orders = []

    # Calculate current weights from positions
    current_weights = {}
    for ticker, pos in current_positions.items():
        if isinstance(pos, dict):
            mv = pos.get("market_value", 0)
        else:
            mv = float(pos)
        current_weights[ticker] = mv / portfolio_value if portfolio_value > 0 else 0

    # Find all tickers (union of current + target)
    all_tickers = set(target_weights.keys()) | set(current_weights.keys())

    for ticker in all_tickers:
        target_w = target_weights.get(ticker, 0.0)
        current_w = current_weights.get(ticker, 0.0)

        # Calculate drift
        diff_w = target_w - current_w

        # Check if drift exceeds threshold
        if target_w > 0:
            drift_pct = abs(diff_w) / target_w
        elif target_w < 0:
            # Short position — drift relative to absolute target
            drift_pct = abs(diff_w) / abs(target_w)
        elif current_w > 0 and target_w == 0:
            drift_pct = 1.0  # Need to close entire position
        elif current_w < 0 and target_w == 0:
            drift_pct = 1.0  # Need to close short position
        else:
            continue  # Both zero, skip

        if drift_pct < REBALANCE_DRIFT_THRESHOLD and abs(diff_w) < 0.01:
            continue  # Drift too small, not worth trading

        price = current_prices.get(ticker, 0)

        # ══════════════════════════════════════════════════════════
        # FIX 2 (complement): If no price from quotes, try to get
        # it from the position data itself (Alpaca provides it).
        # ══════════════════════════════════════════════════════════
        if price <= 0 and ticker in current_positions:
            pos_info = current_positions[ticker]
            if isinstance(pos_info, dict):
                price = pos_info.get("current_price", 0)

        if price <= 0:
            logger.warning(f"  {ticker}: No price available, skipping")
            continue

        # Handle short positions (negative target weights)
        if target_w < 0:
            # ══════════════════════════════════════════════════════
            # FIX 1 (complement): Round short shares to whole numbers
            # at the rebalancer level too, so the order quantity is
            # already correct before reaching the execution engine.
            # ══════════════════════════════════════════════════════
            target_shares = (portfolio_value * abs(target_w)) / price
            current_short_shares = abs(current_weights.get(ticker, 0)) * portfolio_value / price if current_w < 0 else 0

            if ticker not in current_positions and current_w >= 0:
                # Open new short position — round to whole shares
                whole_shares = math.floor(target_shares)
                if whole_shares < 1:
                    logger.info(f"  {ticker}: Short target {target_shares:.2f} shares rounds to 0, skipping")
                    continue
                orders.append({
                    "ticker": ticker,
                    "side": "sell",
                    "quantity": whole_shares,
                    "estimated_value": round(whole_shares * price, 2),
                    "current_weight": round(current_w, 4),
                    "target_weight": round(target_w, 4),
                    "drift_pct": round(drift_pct, 4),
                    "reason": f"Short: {current_w:.1%} → {target_w:.1%}",
                })
            elif current_w > 0:
                # Close long and open short
                # First: sell the long position (can be fractional)
                close_shares = (portfolio_value * current_w) / price
                # Then: open short (must be whole shares)
                short_shares = math.floor(target_shares)
                total_sell = close_shares + short_shares
                if total_sell < 0.001:
                    continue
                orders.append({
                    "ticker": ticker,
                    "side": "sell",
                    "quantity": round(total_sell, 4),
                    "estimated_value": round(total_sell * price, 2),
                    "current_weight": round(current_w, 4),
                    "target_weight": round(target_w, 4),
                    "drift_pct": round(drift_pct, 4),
                    "reason": f"Reverse to short: {current_w:.1%} → {target_w:.1%}",
                })
            else:
                # Adjust existing short
                diff_shares = abs(target_shares - current_short_shares)
                if diff_shares > 0.5:  # At least 1 share difference for shorts
                    side = "sell" if target_shares > current_short_shares else "buy"
                    whole_diff = math.floor(diff_shares)
                    if whole_diff < 1:
                        continue
                    orders.append({
                        "ticker": ticker,
                        "side": side,
                        "quantity": whole_diff,
                        "estimated_value": round(whole_diff * price, 2),
                        "current_weight": round(current_w, 4),
                        "target_weight": round(target_w, 4),
                        "drift_pct": round(drift_pct, 4),
                        "reason": f"Adjust short: {current_w:.1%} → {target_w:.1%}",
                    })
            continue

        # ══════════════════════════════════════════════════════════
        # FIX 4: Standard long position logic — uses portfolio_value
        # for ALL sizing (not just available cash). This ensures
        # stock positions are sized at ~$2,400 (2.4% of $100K)
        # instead of ~$500 (limited by buying power).
        # ══════════════════════════════════════════════════════════
        target_value = portfolio_value * target_w
        current_value = portfolio_value * current_w
        trade_value = target_value - current_value

        # Calculate shares (Alpaca supports fractional shares for longs)
        shares = abs(trade_value) / price
        if shares < 0.001:  # Minimum trade size
            continue

        # FIX 4: For sells, use actual held qty if selling entire position
        # This prevents "insufficient qty" errors from rounding mismatches
        side = "buy" if trade_value > 0 else "sell"
        if side == "sell" and target_w == 0 and ticker in current_positions:
            pos_info = current_positions[ticker]
            if isinstance(pos_info, dict):
                actual_qty = pos_info.get("qty", 0)
                if actual_qty > 0:
                    shares = actual_qty  # Sell exactly what we hold

        orders.append({
            "ticker": ticker,
            "side": side,
            "quantity": round(shares, 4),
            "estimated_value": round(abs(trade_value), 2),
            "current_weight": round(current_w, 4),
            "target_weight": round(target_w, 4),
            "drift_pct": round(drift_pct, 4),
            "reason": f"Rebalance: {current_w:.1%} → {target_w:.1%}",
        })

    # Sort: SELLS first (to free up cash), then BUYS
    sells = sorted([o for o in orders if o["side"] == "sell"],
                   key=lambda x: x["estimated_value"], reverse=True)
    buys = sorted([o for o in orders if o["side"] == "buy"],
                  key=lambda x: x["estimated_value"], reverse=True)

    sorted_orders = sells + buys

    # Enforce daily trade limit
    if len(sorted_orders) > MAX_DAILY_TRADES:
        logger.warning(f"Too many orders ({len(sorted_orders)}), limiting to {MAX_DAILY_TRADES}")
        sorted_orders = sorted_orders[:MAX_DAILY_TRADES]

    # Log summary
    n_buys = sum(1 for o in sorted_orders if o["side"] == "buy")
    n_sells = sum(1 for o in sorted_orders if o["side"] == "sell")
    total_buy = sum(o["estimated_value"] for o in sorted_orders if o["side"] == "buy")
    total_sell = sum(o["estimated_value"] for o in sorted_orders if o["side"] == "sell")

    logger.info(f"Rebalance orders: {n_sells} sells (${total_sell:,.0f}), "
                f"{n_buys} buys (${total_buy:,.0f})")

    return sorted_orders


def calculate_turnover(current_weights: dict, target_weights: dict) -> float:
    """
    Calculate portfolio turnover — how much trading is needed.
    Turnover of 0.10 means 10% of the portfolio needs to be traded.
    Lower is better (less trading cost).
    """
    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    total_change = sum(
        abs(target_weights.get(t, 0) - current_weights.get(t, 0))
        for t in all_tickers
    )
    return total_change / 2  # Divide by 2 because buys and sells offset
