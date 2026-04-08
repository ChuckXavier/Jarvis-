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

import math
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
import time

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    EXECUTION_MODE, ORDER_TIMEOUT_SECONDS,
)


# ── Asset fractionability cache (refreshed once per session) ──
_fractionable_cache = {}


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

    def _check_fractionable(self, ticker: str) -> bool:
        """
        FIX 3: Check if an asset supports fractional trading on Alpaca.
        Results are cached for the session to avoid repeated API calls.
        """
        global _fractionable_cache

        if ticker in _fractionable_cache:
            return _fractionable_cache[ticker]

        try:
            asset = self.api.get_asset(ticker)
            is_frac = getattr(asset, 'fractionable', False)
            _fractionable_cache[ticker] = is_frac
            return is_frac
        except Exception as e:
            logger.warning(f"  Cannot check fractionability for {ticker}: {e}")
            _fractionable_cache[ticker] = False
            return False

    def close_position(self, ticker: str) -> dict:
        """
        FIX 5 (helper): Close an entire position via Alpaca API.
        Used for SAFETY mode liquidation of stranded positions.
        """
        if not self._connected:
            if not self.connect():
                return {"status": "FAILED", "message": "Not connected"}

        timestamp = datetime.now().isoformat()

        if self.mode == "SHADOW":
            logger.info(f"  [SHADOW] CLOSE entire position: {ticker}")
            return {
                "timestamp": timestamp, "ticker": ticker, "side": "sell",
                "status": "SHADOW", "message": "Shadow mode — close logged only",
            }

        try:
            self.api.close_position(ticker)
            logger.info(f"  [LIVE] Closed entire position: {ticker}")
            return {
                "timestamp": timestamp, "ticker": ticker, "side": "sell",
                "status": "SUBMITTED", "message": f"Position {ticker} closed via API",
            }
        except Exception as e:
            logger.error(f"  [FAILED] Close {ticker}: {e}")
            return {
                "timestamp": timestamp, "ticker": ticker, "side": "sell",
                "status": "FAILED", "message": str(e),
            }

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
        skipped = sum(1 for r in results if r["status"] == "SKIPPED")

        logger.info(f"\nExecution complete: {filled} submitted, {shadow} shadow-logged, "
                     f"{failed} failed, {skipped} skipped")

        return results

    def _execute_single_order(self, order: dict) -> dict:
        """Execute or log a single order based on execution mode."""
        ticker = order["ticker"]
        side = order["side"]
        qty = order["quantity"]
        est_value = order.get("estimated_value", 0)
        target_weight = order.get("target_weight", 0)

        timestamp = datetime.now().isoformat()

        # ══════════════════════════════════════════════════════
        # FIX 1: Round short sells to whole shares
        # Alpaca rejects fractional short-sell orders.
        # ══════════════════════════════════════════════════════
        is_short_sell = (side == "sell" and target_weight < 0)
        if is_short_sell:
            qty = math.floor(qty)
            if qty < 1:
                logger.warning(f"  [SKIP] Short {ticker}: rounds to 0 whole shares, skipping")
                return {
                    "timestamp": timestamp, "ticker": ticker, "side": side,
                    "quantity": 0, "estimated_value": est_value,
                    "status": "SKIPPED",
                    "message": "Short order rounds to 0 whole shares",
                }

        # ══════════════════════════════════════════════════════
        # FIX 3: Check fractionability for non-short orders
        # Some small-cap stocks don't support fractional shares.
        # ══════════════════════════════════════════════════════
        if not is_short_sell and self._connected and self.mode == "AUTONOMOUS":
            if not self._check_fractionable(ticker):
                old_qty = qty
                qty = math.floor(qty)
                if qty < 1:
                    logger.warning(f"  [SKIP] {side.upper()} {ticker}: not fractionable, "
                                   f"rounds to 0 shares, skipping")
                    return {
                        "timestamp": timestamp, "ticker": ticker, "side": side,
                        "quantity": 0, "estimated_value": est_value,
                        "status": "SKIPPED",
                        "message": f"Asset not fractionable, {old_qty:.4f} rounds to 0",
                    }
                if old_qty != qty:
                    logger.info(f"  {ticker}: not fractionable, rounded {old_qty:.4f} → {qty}")

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

            # FIX 1+3: Use whole shares for shorts and non-fractionable assets
            if is_short_sell:
                # Already rounded to whole shares above
                request = MarketOrderRequest(
                    symbol=ticker,
                    qty=int(qty),
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )
            else:
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
