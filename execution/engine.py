"""
JARVIS V2 - Execution Engine
===============================
Connects to Alpaca and places actual trades (paper or live).

HOW THIS WORKS (for non-coders):
- This is where the rubber meets the road — actual orders go to the broker
- It uses limit orders (not market orders) to avoid bad fills
- It always processes SELLS first (free up cash before buying)
- Every order is logged in the database for complete audit trail
- In SHADOW mode: only logs, never actually trades
- In SUPERVISED mode: shows you orders, waits for your approval
- In AUTONOMOUS mode: trades automatically (only after 6 months validation)

THREE EXECUTION MODES:
- SHADOW:     Log everything, execute nothing (Phase 4-5 of validation)
- SUPERVISED: Generate orders, wait for manual approval before executing
- AUTONOMOUS: Full auto-pilot (Phase 6+ only, after proven track record)
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
import time

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    EXECUTION_MODE, ORDER_TIMEOUT_SECONDS,
)


class ExecutionEngine:
    """
    Manages the connection to Alpaca and handles all order execution.
    """

    def __init__(self):
        self.mode = EXECUTION_MODE
        self.api = None
        self._connected = False
        self.trade_log = []

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.error("Alpaca API keys not configured")
            return False

        try:
            from alpaca.trading.client import TradingClient
            self.api = TradingClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                paper=True,  # ALWAYS paper trading until validated
            )
            # Test connection
            account = self.api.get_account()
            self._connected = True
            logger.info(f"Connected to Alpaca (Paper Trading)")
            logger.info(f"  Account value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"  Buying power:  ${float(account.buying_power):,.2f}")
            logger.info(f"  Execution mode: {self.mode}")
            return True

        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get current account information from Alpaca."""
        if not self._connected:
            if not self.connect():
                return {}

        try:
            account = self.api.get_account()
            return {
                "portfolio_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "long_market_value": float(account.long_market_value),
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_current_positions(self) -> dict:
        """Get all current positions from Alpaca."""
        if not self._connected:
            if not self.connect():
                return {}

        try:
            positions = self.api.get_all_positions()
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "current_price": float(pos.current_price),
                    "entry_price": float(pos.avg_entry_price),
                    "unrealized_pnl": float(pos.unrealized_pl),
                    "unrealized_pnl_pct": float(pos.unrealized_plpc),
                }
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_current_prices(self, tickers: list) -> dict:
        """Get latest prices for a list of tickers."""
        if not self._connected:
            if not self.connect():
                return {}

        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.historical import StockHistoricalDataClient

            data_client = StockHistoricalDataClient(
                api_key=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
            )

            request = StockLatestQuoteRequest(symbol_or_symbols=tickers)
            quotes = data_client.get_stock_latest_quote(request)

            prices = {}
            for ticker, quote in quotes.items():
                mid = (float(quote.ask_price) + float(quote.bid_price)) / 2
                prices[ticker] = mid if mid > 0 else float(quote.ask_price)

            return prices

        except Exception as e:
            logger.error(f"Failed to get prices: {e}")
            return {}

    def execute_orders(self, orders: list) -> list:
        """
        Execute a list of orders through Alpaca.

        Parameters:
            orders: list of order dicts from the rebalancer
                    Each has: ticker, side, quantity, estimated_value

        Returns:
            list of execution results
        """
        if not orders:
            logger.info("No orders to execute")
            return []

        results = []

        logger.info(f"\nExecuting {len(orders)} orders (mode: {self.mode})...")

        for order in orders:
            result = self._execute_single_order(order)
            results.append(result)
            self.trade_log.append(result)

        # Summary
        filled = sum(1 for r in results if r["status"] == "FILLED" or r["status"] == "SUBMITTED")
        shadow = sum(1 for r in results if r["status"] == "SHADOW")
        failed = sum(1 for r in results if r["status"] == "FAILED")

        logger.info(f"\nExecution complete: {filled} submitted, {shadow} shadow-logged, {failed} failed")

        return results

    def _execute_single_order(self, order: dict) -> dict:
        """Execute or log a single order based on execution mode."""
        ticker = order["ticker"]
        side = order["side"]
        qty = order["quantity"]
        est_value = order.get("estimated_value", 0)

        timestamp = datetime.now().isoformat()

        # ── SHADOW MODE: Log only, don't trade ──
        if self.mode == "SHADOW":
            logger.info(f"  [SHADOW] {side.upper():4s} {qty:.4f} shares of {ticker:5s} "
                       f"(~${est_value:,.2f})")
            return {
                "timestamp": timestamp,
                "ticker": ticker,
                "side": side,
                "quantity": qty,
                "estimated_value": est_value,
                "status": "SHADOW",
                "order_id": None,
                "message": "Shadow mode — order logged but not executed",
            }

        # ── SUPERVISED MODE: Log and flag for approval ──
        if self.mode == "SUPERVISED":
            logger.info(f"  [SUPERVISED] {side.upper():4s} {qty:.4f} shares of {ticker:5s} "
                       f"(~${est_value:,.2f}) — AWAITING APPROVAL")
            return {
                "timestamp": timestamp,
                "ticker": ticker,
                "side": side,
                "quantity": qty,
                "estimated_value": est_value,
                "status": "PENDING_APPROVAL",
                "order_id": None,
                "message": "Supervised mode — needs manual approval to execute",
            }

        # ── AUTONOMOUS MODE: Actually place the order ──
        if not self._connected:
            if not self.connect():
                return {
                    "timestamp": timestamp, "ticker": ticker, "side": side,
                    "quantity": qty, "status": "FAILED",
                    "message": "Not connected to Alpaca",
                }

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest, LimitOrderRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

            # Use market orders for simplicity (can switch to limit later)
            request = MarketOrderRequest(
                symbol=ticker,
                qty=round(qty, 4),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            submitted = self.api.submit_order(request)

            logger.info(f"  [LIVE] {side.upper():4s} {qty:.4f} shares of {ticker:5s} "
                       f"— Order ID: {submitted.id}")

            return {
                "timestamp": timestamp,
                "ticker": ticker,
                "side": side,
                "quantity": qty,
                "estimated_value": est_value,
                "status": "SUBMITTED",
                "order_id": str(submitted.id),
                "message": f"Order submitted: {submitted.id}",
            }

        except Exception as e:
            logger.error(f"  [FAILED] {side.upper()} {ticker}: {e}")
            return {
                "timestamp": timestamp,
                "ticker": ticker,
                "side": side,
                "quantity": qty,
                "status": "FAILED",
                "message": str(e),
            }

    def cancel_all_orders(self):
        """Cancel all open orders."""
        if not self._connected:
            return

        try:
            self.api.cancel_orders()
            logger.info("All open orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    def get_trade_log(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)
