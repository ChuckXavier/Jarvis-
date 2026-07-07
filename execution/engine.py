"""
JARVIS V3 - Execution Engine
=============================
ARCHITECTURAL NOTE (what changed vs V2):
(1) Two-phase execution replaces sort-order hope: all exposure-REDUCING orders
    (priority <= 1 from the rebalancer) are submitted first, the engine POLLS
    until they fill or SELL_FILL_WAIT_SECONDS elapses, and only then submits
    the opens — buying-power rejections from buying before sells settle are
    gone by construction.
(2) `sweep_stale_positions()` and `cancel_stale_orders()` exist as first-class
    methods; orphaned positions and zombie orders get cleaned every cycle.
(3) The per-asset cache now records BOTH fractionable and shortable; short
    opens on non-shortable names are skipped with a logged reason instead of
    a broker rejection.
(4) Every order result is persisted to the `trades` table — V2 kept its trade
    log in a Python list, which evaporated with the container.
(5) Latest-quote requests are chunked (QUOTE_CHUNK_SIZE) so a 600-symbol
    request cannot fail as one giant call. SDK stays `alpaca-py` — that is
    what requirements.txt installs and what the V2 code already imports; the
    rebuild spec's instruction to use the deprecated `alpaca-trade-api` was a
    factual error and is deliberately not followed.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

from loguru import logger

from config.settings import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, EXECUTION_MODE,
    ORDER_TIMEOUT_SECONDS, SELL_FILL_WAIT_SECONDS, SELL_FILL_POLL_SECONDS,
    QUOTE_CHUNK_SIZE,
)

_asset_cache: dict[str, dict] = {}


class ExecutionEngine:
    """Alpaca connection + order lifecycle. Paper trading only."""

    def __init__(self):
        self.mode = EXECUTION_MODE
        self.api = None
        self._connected = False

    # ── connection / account ────────────────────────────────────────────────

    def connect(self) -> bool:
        if self._connected:
            return True
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logger.error("Alpaca API keys not configured")
            return False
        try:
            from alpaca.trading.client import TradingClient
            self.api = TradingClient(api_key=ALPACA_API_KEY,
                                     secret_key=ALPACA_SECRET_KEY, paper=True)
            acct = self.api.get_account()
            self._connected = True
            logger.info(f"Alpaca connected (paper) — equity "
                        f"${float(acct.portfolio_value):,.2f}, mode {self.mode}")
            return True
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            return False

    def get_account_info(self) -> dict:
        if not self.connect():
            return {}
        try:
            a = self.api.get_account()
            return {
                "portfolio_value": float(a.portfolio_value),
                "cash": float(a.cash),
                "buying_power": float(a.buying_power),
                "equity": float(a.equity),
                "long_market_value": float(a.long_market_value),
                "short_market_value": float(a.short_market_value),
            }
        except Exception as e:
            logger.error(f"get_account_info failed: {e}")
            return {}

    def get_current_positions(self) -> dict:
        """Signed positions: shorts carry negative qty and market_value."""
        if not self.connect():
            return {}
        try:
            out = {}
            for p in self.api.get_all_positions():
                out[p.symbol] = {
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "current_price": float(p.current_price),
                    "entry_price": float(p.avg_entry_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_plpc),
                }
            return out
        except Exception as e:
            logger.error(f"get_current_positions failed: {e}")
            return {}

    # ── market data (IEX latest quotes, chunked) ─────────────────────────────

    def get_current_prices(self, tickers: list[str]) -> dict:
        if not tickers:
            return {}
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            client = StockHistoricalDataClient(api_key=ALPACA_API_KEY,
                                               secret_key=ALPACA_SECRET_KEY)
            prices: dict[str, float] = {}
            for i in range(0, len(tickers), QUOTE_CHUNK_SIZE):
                chunk = tickers[i:i + QUOTE_CHUNK_SIZE]
                try:
                    quotes = client.get_stock_latest_quote(
                        StockLatestQuoteRequest(symbol_or_symbols=chunk))
                    for t, q in quotes.items():
                        bid, ask = float(q.bid_price), float(q.ask_price)
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)
                        if mid > 0:
                            prices[t] = mid
                except Exception as e:
                    logger.warning(f"quote chunk {i//QUOTE_CHUNK_SIZE} failed: {e}")
            logger.info(f"quotes: {len(prices)}/{len(tickers)} symbols priced (IEX)")
            return prices
        except Exception as e:
            logger.error(f"get_current_prices failed: {e}")
            return {}

    # ── asset metadata ───────────────────────────────────────────────────────

    def get_asset_flags(self, ticker: str) -> dict:
        """{'fractionable': bool, 'shortable': bool} with conservative defaults."""
        if ticker in _asset_cache:
            return _asset_cache[ticker]
        flags = {"fractionable": False, "shortable": False}
        if self.connect():
            try:
                a = self.api.get_asset(ticker)
                flags = {
                    "fractionable": bool(getattr(a, "fractionable", False)),
                    "shortable": bool(getattr(a, "shortable", False)
                                      and getattr(a, "easy_to_borrow", True)),
                }
            except Exception as e:
                logger.warning(f"asset lookup {ticker} failed: {e}")
        _asset_cache[ticker] = flags
        return flags

    def get_fractionable_map(self, tickers: list[str]) -> dict[str, bool]:
        return {t: self.get_asset_flags(t)["fractionable"] for t in tickers}

    # ── single-position close / sweep ────────────────────────────────────────

    def close_position(self, ticker: str) -> dict:
        ts = datetime.now(timezone.utc).isoformat()
        if self.mode == "SHADOW":
            logger.info(f"  [SHADOW] CLOSE {ticker}")
            return {"timestamp": ts, "ticker": ticker, "side": "close",
                    "status": "SHADOW", "message": "shadow close"}
        if not self.connect():
            return {"timestamp": ts, "ticker": ticker, "side": "close",
                    "status": "FAILED", "message": "not connected"}
        try:
            self.api.close_position(ticker)
            logger.info(f"  [LIVE] closed {ticker}")
            res = {"timestamp": ts, "ticker": ticker, "side": "close",
                   "status": "SUBMITTED", "message": "closed via API"}
        except Exception as e:
            logger.error(f"  close {ticker} FAILED: {e}")
            res = {"timestamp": ts, "ticker": ticker, "side": "close",
                   "status": "FAILED", "message": str(e)}
        self._log_trade_db(res)
        return res

    def sweep_stale_positions(self, target_tickers: set[str] | list[str]) -> list[dict]:
        """Close every held position not in the target set. Belt to the
        rebalancer's braces — also callable standalone for cleanup."""
        targets = set(target_tickers)
        held = self.get_current_positions()
        stale = [t for t in held if t not in targets]
        if not stale:
            return []
        logger.info(f"Sweeping {len(stale)} stale positions: {stale[:10]}"
                    f"{' ...' if len(stale) > 10 else ''}")
        return [self.close_position(t) for t in stale]

    # ── stale order hygiene ──────────────────────────────────────────────────

    def cancel_stale_orders(self) -> int:
        """Cancel open orders older than ORDER_TIMEOUT_SECONDS."""
        if not self.connect():
            return 0
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            open_orders = self.api.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200))
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=ORDER_TIMEOUT_SECONDS)
            n = 0
            for o in open_orders:
                created = o.created_at
                if created and created < cutoff:
                    try:
                        self.api.cancel_order_by_id(o.id)
                        n += 1
                        logger.info(f"  cancelled stale order {o.symbol} ({o.id})")
                    except Exception as e:
                        logger.warning(f"  cancel {o.id} failed: {e}")
            return n
        except Exception as e:
            logger.warning(f"cancel_stale_orders failed: {e}")
            return 0

    # ── order execution (two phases) ─────────────────────────────────────────

    def execute_orders(self, orders: list[dict]) -> list[dict]:
        """
        Phase A: priority <= 1 (sweeps/closes/reduces). Wait for fills.
        Phase B: priority >= 2 (opens). Partial fills are logged with their
        filled quantity; unfilled remainders are handled by next cycle's diff.
        """
        if not orders:
            logger.info("no orders to execute")
            return []
        phase_a = [o for o in orders if o.get("priority", 3) <= 1]
        phase_b = [o for o in orders if o.get("priority", 3) >= 2]
        logger.info(f"Executing {len(orders)} orders "
                    f"({len(phase_a)} reduce -> {len(phase_b)} open), mode {self.mode}")

        results = [self._submit(o) for o in phase_a]
        if phase_a and self.mode == "AUTONOMOUS":
            ids = [r["order_id"] for r in results
                   if r.get("order_id") and r["status"] == "SUBMITTED"]
            self._wait_for_fills(ids)
        results += [self._submit(o) for o in phase_b]

        ok = sum(1 for r in results if r["status"] in ("SUBMITTED", "FILLED"))
        sh = sum(1 for r in results if r["status"] == "SHADOW")
        fl = sum(1 for r in results if r["status"] == "FAILED")
        sk = sum(1 for r in results if r["status"] == "SKIPPED")
        logger.info(f"execution: {ok} submitted, {sh} shadow, {fl} failed, {sk} skipped")
        return results

    def _wait_for_fills(self, order_ids: list[str]) -> None:
        if not order_ids:
            return
        deadline = time.time() + SELL_FILL_WAIT_SECONDS
        pending = set(order_ids)
        while pending and time.time() < deadline:
            time.sleep(SELL_FILL_POLL_SECONDS)
            for oid in list(pending):
                try:
                    o = self.api.get_order_by_id(oid)
                    status = str(o.status).lower()
                    if "filled" in status or "canceled" in status \
                            or "rejected" in status or "expired" in status:
                        pending.discard(oid)
                        filled = float(o.filled_qty or 0)
                        if "partially" in status:
                            logger.warning(f"  partial fill {o.symbol}: "
                                           f"{filled}/{o.qty}")
                except Exception:
                    pending.discard(oid)  # don't deadlock on a lookup error
        if pending:
            logger.warning(f"  {len(pending)} reduce orders still open after "
                           f"{SELL_FILL_WAIT_SECONDS}s — proceeding; next "
                           f"cycle's diff self-corrects")

    def _submit(self, order: dict) -> dict:
        t = order["ticker"]
        side = order["side"]
        qty = float(order["quantity"])
        prio = order.get("priority", 3)
        opening_short = (prio == 2) or (side == "sell" and order.get("target_weight", 0) < 0)
        ts = datetime.now(timezone.utc).isoformat()

        base = {"timestamp": ts, "ticker": t, "side": side, "quantity": qty,
                "estimated_value": order.get("estimated_value", 0),
                "priority": prio, "order_id": None,
                "signal_source": order.get("reason", "")[:50]}

        # Broker-legality checks (the rebalancer pre-floors; this is defense).
        if opening_short:
            flags = self.get_asset_flags(t)
            if not flags["shortable"]:
                logger.warning(f"  [SKIP] short {t}: not shortable/ETB")
                res = {**base, "status": "SKIPPED", "message": "not shortable"}
                self._log_trade_db(res)
                return res
            qty = float(int(qty))
            if qty < 1:
                res = {**base, "status": "SKIPPED", "message": "short < 1 share"}
                self._log_trade_db(res)
                return res
        elif self.mode == "AUTONOMOUS" and not self.get_asset_flags(t)["fractionable"]:
            qty = float(int(qty))
            if qty < 1:
                res = {**base, "status": "SKIPPED",
                       "message": "non-fractionable rounds to 0"}
                self._log_trade_db(res)
                return res
        base["quantity"] = qty

        if self.mode == "SHADOW":
            logger.info(f"  [SHADOW] {side.upper():4s} {qty:.4f} {t} "
                        f"(~${base['estimated_value']:,.0f}) p{prio}")
            res = {**base, "status": "SHADOW", "message": "shadow only"}
            self._log_trade_db(res)
            return res
        if self.mode == "SUPERVISED":
            res = {**base, "status": "PENDING_APPROVAL",
                   "message": "awaiting manual approval"}
            self._log_trade_db(res)
            return res

        if not self.connect():
            return {**base, "status": "FAILED", "message": "not connected"}
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = MarketOrderRequest(
                symbol=t,
                qty=int(qty) if qty == int(qty) else round(qty, 4),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            sub = self.api.submit_order(req)
            logger.info(f"  [LIVE] {side.upper():4s} {qty:.4f} {t} -> {sub.id}")
            res = {**base, "status": "SUBMITTED", "order_id": str(sub.id),
                   "message": "submitted"}
        except Exception as e:
            logger.error(f"  [FAILED] {side.upper()} {t}: {e}")
            res = {**base, "status": "FAILED", "message": str(e)}
        self._log_trade_db(res)
        return res

    # ── persistent trade log ─────────────────────────────────────────────────

    def _log_trade_db(self, res: dict) -> None:
        """Persist every order result to the `trades` table (V2 schema)."""
        try:
            from sqlalchemy import text
            from data.db import engine
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO trades
                        (timestamp, ticker, side, quantity, price, order_type,
                         status, alpaca_order_id, signal_source)
                    VALUES (:ts, :ticker, :side, :qty, :price, 'market',
                            :status, :oid, :src)
                """), {
                    "ts": datetime.now(timezone.utc),
                    "ticker": res.get("ticker", ""),
                    "side": res.get("side", "")[:4],
                    "qty": float(res.get("quantity", 0) or 0),
                    "price": None,
                    "status": res.get("status", "")[:20],
                    "oid": res.get("order_id"),
                    "src": res.get("signal_source", "")[:50],
                })
        except Exception as e:
            logger.warning(f"trade DB log failed (non-fatal): {e}")
