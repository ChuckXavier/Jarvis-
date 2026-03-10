"""
JARVIS V6 - The Volatility-Aware Momentum Engine
===================================================
Three pillars, zero prediction, directly measurable signals.

Pillar 1: Momentum Core (V4 enhanced) — top 4 ETFs by 3/6/12M momentum
Pillar 2: Volatility Filter — 2x leverage ONLY when VIX > realized vol
Pillar 3: 200-Day SMA Switch — absolute override to safety when QQQ < SMA

Key V5 fixes:
- No 3x ETFs (max 2x)
- No inverse ETFs
- Max 2 trades/week, 5% drift threshold
- Monthly rotation (not weekly)
- 3% SMA buffer zone
- 3-day entry / 5-day exit confirmation for volatility filter
- -25% circuit breaker
"""

import pandas as pd
import numpy as np
from loguru import logger

# ════════════════════════════════════════════════════════════
# ETF UNIVERSE
# ════════════════════════════════════════════════════════════

TIER1_MOMENTUM = ["QQQ", "SPY", "SMH", "SOXX", "VGT", "XLK", "XLV", "XLE", "GLD", "TLT", "IEF", "DBMF"]
TIER2_LEVERAGE = {"QQQ": "QLD", "SPY": "SSO", "SMH": "USD", "SOXX": "USD", "VGT": "ROM", "XLK": "ROM"}
SAFETY_ALLOCATION = {"TLT": 0.30, "IEF": 0.30, "GLD": 0.20, "BIL": 0.10, "DBMF": 0.05, "SHY": 0.05}

DEFAULT_CONFIG = {
    "initial_capital": 100000,
    "rebalance_day": 4,        # Friday (0=Mon)
    "momentum_weights": [0.40, 0.35, 0.25],  # 3M, 6M, 12M
    "top_n": 4,
    "continuity_bonus": 0.02,
    "max_sector_pct": 0.60,
    "drift_threshold": 0.05,
    "max_trades_per_week": 2,
    "sma_buffer": 0.03,
    "sma_confirm_days": 2,
    "vol_entry_confirm": 3,
    "vol_exit_confirm": 5,
    "min_leverage_hold_days": 10,
    "max_leverage_hold_days": 60,
    "circuit_breaker_pct": -0.25,
    "circuit_breaker_cooldown": 30,
    "transaction_cost_bps": 5,
    "slippage_bps": 3,
    "annual_cost_cap_pct": 0.01,
}


class BacktestV6:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.portfolio_history = []
        self.trade_log = []
        self.state_log = []

    def run(self, prices, vix_series=None):
        if prices.empty:
            return {}

        # Determine start: need 252 days for SMA + warmup
        valid_start = prices.index[0] + pd.Timedelta(days=800)
        start = max(valid_start, pd.Timestamp("2013-01-01"))
        if start > prices.index[-1]:
            start = prices.index[252]
        end = prices.index[-1]

        logger.info("=" * 65)
        logger.info("BACKTESTING JARVIS V6 — VOLATILITY-AWARE MOMENTUM ENGINE")
        logger.info(f"Period: {start.date()} → {end.date()}")
        logger.info(f"Pillars: Momentum Core | Volatility Filter | 200-Day SMA Switch")
        logger.info(f"Max leverage: 2x | Max trades/week: {self.config['max_trades_per_week']}")
        logger.info("=" * 65)

        self._simulate(prices, vix_series, start, end)
        return {
            "portfolio_history": pd.DataFrame(self.portfolio_history),
            "trade_log": self.trade_log,
            "state_log": self.state_log,
            "metrics": self._compute_metrics(),
        }

    def _simulate(self, prices, vix, start, end):
        cfg = self.config
        capital = cfg["initial_capital"]
        cash = capital
        positions = {}  # {ticker: {"qty": float, "entry_date": date, "leveraged": bool}}
        cost_bps = cfg["transaction_cost_bps"] + cfg["slippage_bps"]

        # State tracking
        mode = "ACTIVE"  # ACTIVE or SAFETY
        leverage_on = False
        leverage_level = 1.0
        vol_filter_streak = 0   # Consecutive days VIX > RV
        vol_exit_streak = 0     # Consecutive days RV >= VIX
        sma_above_streak = 0
        sma_below_streak = 0
        circuit_breaker_active = False
        circuit_breaker_date = None
        last_momentum_date = None
        current_top4 = []
        trades_this_week = 0
        week_start = None
        annual_costs = 0
        year_start_value = capital
        leverage_entry_date = None
        peak_value = capital

        bt_dates = prices.loc[start:end].index

        for date in bt_dates:
            # ── Mark to market ──
            pv = cash
            for t, pos in positions.items():
                if t in prices.columns and date in prices.index:
                    px = prices.loc[date, t]
                    if pd.notna(px):
                        pv += pos["qty"] * px

            self.portfolio_history.append({
                "date": date, "total_value": pv, "cash": cash,
                "mode": mode, "leverage": leverage_level,
                "num_positions": len(positions),
            })

            peak_value = max(peak_value, pv)

            # Reset weekly trade counter
            if week_start is None or (date - week_start).days >= 7:
                week_start = date
                trades_this_week = 0

            # Reset annual cost tracking
            if date.month == 1 and date.day <= 5:
                annual_costs = 0
                year_start_value = pv

            # ── CIRCUIT BREAKER CHECK ──
            drawdown = (pv / peak_value) - 1 if peak_value > 0 else 0
            if drawdown <= cfg["circuit_breaker_pct"] and not circuit_breaker_active:
                circuit_breaker_active = True
                circuit_breaker_date = date
                logger.info(f"  🚨 CIRCUIT BREAKER at {date.date()}: DD={drawdown:.1%}")
                # Force to safety
                mode = "SAFETY"
                leverage_on = False
                leverage_level = 1.0
                # Close all positions
                cash = self._close_all(positions, prices, date, cost_bps, pv)
                positions = {}
                # Enter safety allocation
                cash = self._enter_safety(positions, prices, date, cost_bps, pv, cash)
                continue

            if circuit_breaker_active:
                days_since_cb = (date - circuit_breaker_date).days
                if days_since_cb >= cfg["circuit_breaker_cooldown"]:
                    # Check if conditions allow re-entry
                    if self._check_sma(prices, date) and self._check_vol_filter(prices, vix, date) > 0:
                        circuit_breaker_active = False
                        logger.info(f"  ✅ Circuit breaker released at {date.date()}")
                    else:
                        continue  # Stay in safety
                else:
                    continue  # Cooldown not complete

            # ════════════════════════════════════════════════
            # PILLAR 3: 200-DAY SMA MASTER SWITCH (daily)
            # ════════════════════════════════════════════════
            sma_status = self._check_sma_with_buffer(prices, date)

            if sma_status == "BELOW":
                sma_below_streak += 1
                sma_above_streak = 0
                if sma_below_streak >= cfg["sma_confirm_days"] and mode == "ACTIVE":
                    # Switch to SAFETY
                    mode = "SAFETY"
                    leverage_on = False
                    leverage_level = 1.0
                    logger.info(f"  📉 SAFETY MODE at {date.date()}")
                    cash = self._close_all(positions, prices, date, cost_bps, pv)
                    positions = {}
                    cash = self._enter_safety(positions, prices, date, cost_bps, pv, cash)
                    trades_this_week += len(SAFETY_ALLOCATION)

            elif sma_status == "ABOVE":
                sma_above_streak += 1
                sma_below_streak = 0
                if sma_above_streak >= cfg["sma_confirm_days"] and mode == "SAFETY":
                    # Switch to ACTIVE
                    mode = "ACTIVE"
                    logger.info(f"  📈 ACTIVE MODE at {date.date()}")
                    cash = self._close_all(positions, prices, date, cost_bps, pv)
                    positions = {}
                    last_momentum_date = None  # Force momentum recalc

            elif sma_status == "BUFFER":
                # In buffer zone: maintain current mode, no leverage
                leverage_on = False
                leverage_level = 1.0

            if mode == "SAFETY":
                continue  # Nothing else to do in safety mode

            # ════════════════════════════════════════════════
            # PILLAR 1: MOMENTUM ROTATION (monthly)
            # ════════════════════════════════════════════════
            is_month_start = (last_momentum_date is None or
                             date.month != last_momentum_date.month)

            if is_month_start and mode == "ACTIVE":
                last_momentum_date = date
                new_top4 = self._compute_momentum(prices, date, current_top4)

                if new_top4 != current_top4:
                    # Check cost budget
                    if annual_costs / max(pv, 1) < cfg["annual_cost_cap_pct"]:
                        # Execute rotation
                        target = {t: 0.25 for t in new_top4}

                        # Apply leverage overlay if active
                        if leverage_on:
                            target = self._apply_leverage(target, leverage_level)

                        cost = self._rebalance(positions, target, prices, date, pv, cash, cost_bps)
                        cash -= cost
                        annual_costs += cost
                        current_top4 = new_top4

            # ════════════════════════════════════════════════
            # PILLAR 2: VOLATILITY FILTER (daily check)
            # ════════════════════════════════════════════════
            if mode == "ACTIVE" and vix is not None:
                vol_signal = self._check_vol_filter(prices, vix, date)

                if vol_signal > 0:
                    vol_filter_streak += 1
                    vol_exit_streak = 0
                else:
                    vol_exit_streak += 1
                    vol_filter_streak = 0

                # Entry: 3 consecutive days with VIX > RV
                if (vol_filter_streak >= cfg["vol_entry_confirm"] and
                    not leverage_on and mode == "ACTIVE" and
                    sma_status != "BUFFER"):

                    # Determine leverage level
                    iv_rv_spread = vol_signal
                    if iv_rv_spread >= 0.05:
                        leverage_level = 2.0
                    else:
                        leverage_level = 1.5

                    leverage_on = True
                    leverage_entry_date = date

                    # Apply leverage to current positions
                    if current_top4 and trades_this_week < cfg["max_trades_per_week"]:
                        target = {t: 0.25 for t in current_top4}
                        target = self._apply_leverage(target, leverage_level)
                        cost = self._rebalance(positions, target, prices, date, pv, cash, cost_bps)
                        cash -= cost
                        annual_costs += cost
                        trades_this_week += 1

                # Exit: 5 consecutive days with RV >= VIX
                if (vol_exit_streak >= cfg["vol_exit_confirm"] and leverage_on):
                    # Check minimum hold period
                    if leverage_entry_date and (date - leverage_entry_date).days >= cfg["min_leverage_hold_days"]:
                        leverage_on = False
                        leverage_level = 1.0

                        # Remove leverage from positions
                        if current_top4 and trades_this_week < cfg["max_trades_per_week"]:
                            target = {t: 0.25 for t in current_top4}
                            cost = self._rebalance(positions, target, prices, date, pv, cash, cost_bps)
                            cash -= cost
                            annual_costs += cost
                            trades_this_week += 1

                # Max leverage duration
                if (leverage_on and leverage_entry_date and
                    (date - leverage_entry_date).days >= cfg["max_leverage_hold_days"]):
                    leverage_on = False
                    leverage_level = 1.0

            # Log state
            self.state_log.append({
                "date": date, "mode": mode, "leverage": leverage_level,
                "leverage_on": leverage_on, "vol_streak": vol_filter_streak,
                "sma_status": sma_status,
            })

    def _check_sma(self, prices, date):
        """Simple SMA check without buffer."""
        if "QQQ" not in prices.columns:
            return True
        qqq = prices["QQQ"].loc[:date].dropna()
        if len(qqq) < 200:
            return True
        return qqq.iloc[-1] > qqq.rolling(200).mean().iloc[-1]

    def _check_sma_with_buffer(self, prices, date):
        """SMA check with 3% buffer zone."""
        if "QQQ" not in prices.columns:
            return "ABOVE"
        qqq = prices["QQQ"].loc[:date].dropna()
        if len(qqq) < 200:
            return "ABOVE"

        sma = qqq.rolling(200).mean().iloc[-1]
        price = qqq.iloc[-1]
        buffer = self.config["sma_buffer"]

        if price > sma * (1 + buffer):
            return "ABOVE"
        elif price < sma * (1 - buffer):
            return "BELOW"
        else:
            return "BUFFER"

    def _check_vol_filter(self, prices, vix, date):
        """
        Returns: spread (IV - RV) if VIX > RV, else 0 or negative.
        Positive = favorable for leverage.
        """
        if vix is None or vix.empty or "SPY" not in prices.columns:
            return 0

        # Implied vol: 20-day average of VIX
        vix_before = vix.loc[:date].dropna()
        if len(vix_before) < 20:
            return 0
        iv = vix_before.tail(20).mean() / 100  # VIX is in %, convert to decimal

        # Realized vol: 10-day annualized std of SPY returns
        spy = prices["SPY"].loc[:date].dropna()
        if len(spy) < 11:
            return 0
        rv = spy.pct_change().tail(10).std() * np.sqrt(252)

        spread = iv - rv
        return spread if spread > 0 else -abs(spread)

    def _compute_momentum(self, prices, date, current_holdings):
        """Compute momentum scores and return top 4 ETFs."""
        w3, w6, w12 = self.config["momentum_weights"]
        scores = {}

        for ticker in TIER1_MOMENTUM:
            if ticker not in prices.columns:
                continue
            s = prices[ticker].loc[:date].dropna()

            # Need at least 252 days
            if len(s) < 252:
                continue

            r3m = (s.iloc[-1] / s.iloc[-63]) - 1 if len(s) >= 63 else 0
            r6m = (s.iloc[-1] / s.iloc[-126]) - 1 if len(s) >= 126 else 0
            r12m = (s.iloc[-1] / s.iloc[-252]) - 1 if len(s) >= 252 else 0

            score = w3 * r3m + w6 * r6m + w12 * r12m

            # Continuity bonus
            if ticker in current_holdings:
                score += self.config["continuity_bonus"]

            scores[ticker] = score

        # Rank and select top 4
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Sector concentration check
        top4 = []
        sector_counts = {}
        tech_tickers = {"QQQ", "SMH", "SOXX", "VGT", "XLK"}
        max_sector = int(self.config["top_n"] * self.config["max_sector_pct"])

        for ticker, score in ranked:
            if len(top4) >= self.config["top_n"]:
                break
            is_tech = ticker in tech_tickers
            tech_count = sector_counts.get("tech", 0)
            if is_tech and tech_count >= max_sector:
                continue  # Skip, would exceed sector limit
            top4.append(ticker)
            if is_tech:
                sector_counts["tech"] = tech_count + 1

        return top4

    def _apply_leverage(self, target, leverage_level):
        """Replace top 2 positions with their 2x equivalents."""
        sorted_positions = sorted(target.items(), key=lambda x: x[1], reverse=True)
        new_target = {}

        for i, (ticker, weight) in enumerate(sorted_positions):
            if i < 2 and ticker in TIER2_LEVERAGE and leverage_level >= 1.5:
                lev_ticker = TIER2_LEVERAGE[ticker]
                new_target[lev_ticker] = weight
            else:
                new_target[ticker] = weight

        return new_target

    def _rebalance(self, positions, target, prices, date, pv, cash, cost_bps):
        """Execute rebalance trades. Returns total cost."""
        total_cost = 0

        # Current weights
        current_w = {}
        for t, pos in positions.items():
            if t in prices.columns:
                px = prices.loc[date, t]
                if pd.notna(px) and pv > 0:
                    current_w[t] = (pos["qty"] * px) / pv

        # Determine trades needed
        all_tickers = set(target.keys()) | set(current_w.keys())
        trades = []

        for t in all_tickers:
            tw = target.get(t, 0)
            cw = current_w.get(t, 0)
            diff = tw - cw

            if abs(diff) < self.config["drift_threshold"]:
                continue

            px = prices.loc[date, t] if t in prices.columns else None
            if px is None or pd.isna(px) or px <= 0:
                continue

            trade_val = pv * diff
            shares = trade_val / px
            cost = abs(trade_val) * (cost_bps / 10000)

            if t not in positions:
                positions[t] = {"qty": 0, "entry_date": date, "leveraged": t in TIER2_LEVERAGE.values()}
            positions[t]["qty"] += shares
            positions[t]["entry_date"] = date

            total_cost += cost
            trades.append({
                "date": date, "ticker": t,
                "side": "BUY" if shares > 0 else "SELL",
                "shares": abs(shares), "value": abs(trade_val), "cost": cost,
            })

            if abs(positions[t]["qty"]) < 0.001:
                del positions[t]

        self.trade_log.extend(trades)
        return total_cost

    def _close_all(self, positions, prices, date, cost_bps, pv):
        """Close all positions, return cash."""
        cash = 0
        for t in list(positions.keys()):
            if t in prices.columns:
                px = prices.loc[date, t]
                if pd.notna(px):
                    val = positions[t]["qty"] * px
                    cost = abs(val) * (cost_bps / 10000)
                    cash += val - cost
                    self.trade_log.append({
                        "date": date, "ticker": t, "side": "SELL",
                        "shares": abs(positions[t]["qty"]), "value": abs(val), "cost": cost,
                    })
        positions.clear()
        return cash

    def _enter_safety(self, positions, prices, date, cost_bps, pv, cash):
        """Enter safety allocation."""
        for t, w in SAFETY_ALLOCATION.items():
            if t in prices.columns:
                px = prices.loc[date, t]
                if pd.notna(px) and px > 0:
                    trade_val = pv * w
                    shares = trade_val / px
                    cost = abs(trade_val) * (cost_bps / 10000)
                    positions[t] = {"qty": shares, "entry_date": date, "leveraged": False}
                    cash -= trade_val + cost
        return cash

    def _compute_metrics(self):
        if not self.portfolio_history:
            return {}
        df = pd.DataFrame(self.portfolio_history).set_index("date")
        v = df["total_value"]
        if len(v) < 2:
            return {}

        dr = v.pct_change().dropna()
        tr = (v.iloc[-1] / v.iloc[0]) - 1
        yrs = (v.index[-1] - v.index[0]).days / 365.25
        ar = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
        vol = dr.std() * np.sqrt(252)
        sharpe = (ar - 0.04) / vol if vol > 0 else 0
        ds = dr[dr < 0]
        dsv = ds.std() * np.sqrt(252) if len(ds) > 0 else vol
        sortino = (ar - 0.04) / dsv if dsv > 0 else 0
        cm = v.cummax()
        mdd = ((v / cm) - 1).min()
        calmar = abs(ar / mdd) if mdd != 0 else 0
        mo = v.resample("ME").last().pct_change().dropna()
        wr = (mo > 0).mean()
        by = v.resample("YE").last().pct_change().dropna()

        # Time in each mode
        if self.state_log:
            active_pct = sum(1 for s in self.state_log if s["mode"] == "ACTIVE") / len(self.state_log)
            lev_pct = sum(1 for s in self.state_log if s["leverage_on"]) / len(self.state_log)
        else:
            active_pct, lev_pct = 0, 0

        return {
            "total_return": tr, "annualized_return": ar,
            "annualized_volatility": vol, "sharpe_ratio": sharpe,
            "sortino_ratio": sortino, "max_drawdown": mdd,
            "calmar_ratio": calmar, "win_rate_monthly": wr,
            "profit_factor": abs(mo[mo>0].mean() / mo[mo<0].mean()) if (mo<0).any() and (mo>0).any() else 0,
            "best_month": mo.max() if not mo.empty else 0,
            "worst_month": mo.min() if not mo.empty else 0,
            "best_year": by.max() if not by.empty else 0,
            "worst_year": by.min() if not by.empty else 0,
            "total_trades": len(self.trade_log),
            "total_costs": sum(t.get("cost", 0) for t in self.trade_log),
            "years": yrs, "start_value": v.iloc[0], "end_value": v.iloc[-1],
            "pct_active_mode": active_pct,
            "pct_leveraged": lev_pct,
        }
