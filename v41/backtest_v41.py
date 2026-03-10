"""
JARVIS V4.1 - Zero Cash Drag Momentum Rotator
================================================
V4 averaged ~25-35% cash, costing 3.5-5% per year in opportunity cost.
V4.1 eliminates ALL cash drag with 4 research-backed changes:

FIX 1: NEVER HOLD CASH
  Cash → BIL (0-3 month T-bills, currently ~4.5% yield)
  Every dollar earns the risk-free rate at minimum.
  Research: Sharpe (1966) — the risk-free rate is the FLOOR, never 0%.

FIX 2: SPY AS DEFAULT POSITION
  When momentum top-N doesn't fill 100%, remainder → SPY.
  Research: Fama-French (1993) — market beta is a compensated risk factor.
  Being in the market beats being in cash >70% of rolling 1-year periods.

FIX 3: FULL POSITION SIZING (NO FRACTIONAL KELLY)
  V4: Half-Kelly in CALM (50% of target), Quarter in TRANSITION (25%)
  V4.1: Full target weight always. Risk managed by POSITION CAPS, not by
  sizing down. Max single position 15%. Max sector 40%.
  Research: Thorp (2006) — half-Kelly reduces return by 25% while only
  reducing variance by 25%. Full Kelly with hard caps is more efficient.

FIX 4: 100% INVESTED AT ALL TIMES
  V4: 5-10% cash reserve "for rebalancing"
  V4.1: 0% target cash. The SMA switch handles crash protection.
  Keeping cash for rebalancing is unnecessary — sells generate cash
  for buys within the same rebalance cycle.
  Research: Ibbotson (2010) — time in market beats timing the market.

EXPECTED IMPROVEMENT:
  V4 cash drag: ~25-35% cash × ~10% market return = 2.5-3.5% lost/year
  V4.1 removes this entirely. Expected return: V4's 6.9% + 3-4% = ~10-11%
"""

import pandas as pd
import numpy as np
from loguru import logger

# V4.1 UNIVERSE
# Momentum candidates: same as V4
MOMENTUM_POOL = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VTV", "VUG",
    "XLK", "XLF", "XLV", "XLE", "XLI",
]

# Default position: when momentum doesn't fill 100%
DEFAULT_POSITION = "SPY"

# Safety allocation (when SMA switch is OFF): 100% invested, no cash
SAFETY_ALLOCATION = {
    "TLT": 0.30,   # Long bonds (rally during crashes)
    "IEF": 0.25,   # Intermediate bonds
    "GLD": 0.25,   # Gold (safe haven)
    "BIL": 0.20,   # T-bills (risk-free rate, NOT cash)
}

# Sector mapping for concentration limits
SECTOR_MAP = {
    "SPY": "broad", "QQQ": "tech", "IWM": "small_cap", "EFA": "intl",
    "EEM": "em", "VTV": "value", "VUG": "growth", "XLK": "tech",
    "XLF": "financial", "XLV": "healthcare", "XLE": "energy", "XLI": "industrial",
    "TLT": "bonds", "IEF": "bonds", "GLD": "commodities", "BIL": "cash_equiv",
    "SHY": "cash_equiv",
}

DEFAULT_CONFIG = {
    "initial_capital": 100000,
    "top_n": 7,                     # Hold top 7 momentum ETFs
    "momentum_weights": [0.40, 0.35, 0.25],  # 3M, 6M, 12M
    "momentum_skip": 21,            # Skip last month (reversal effect)
    "continuity_bonus": 0.02,       # +2% score for current holdings
    "max_single_position": 0.15,    # 15% max per ETF (V4 was 20%)
    "max_sector_pct": 0.40,         # 40% max per sector
    "rebalance_frequency": 21,      # Monthly
    "drift_threshold": 0.03,        # 3% drift before trading (tighter for full invest)
    "sma_lookback": 200,            # 200-day SMA for crash protection
    "sma_buffer": 0.02,             # 2% buffer to prevent whipsaw
    "sma_confirm_days": 3,          # 3-day confirmation
    "transaction_cost_bps": 5,
    "slippage_bps": 3,
}


class BacktestV41:
    """V4.1: Zero Cash Drag Momentum Rotator"""

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.portfolio_history = []
        self.trade_log = []
        self.state_log = []

    def run(self, prices, vix=None):
        if prices.empty:
            return {}

        # Need 252+ days for momentum + SMA warmup
        if "QQQ" in prices.columns:
            qqq = prices["QQQ"].dropna()
            if len(qqq) > 400:
                start = qqq.index[400]
            else:
                start = prices.index[min(400, len(prices)-1)]
        else:
            start = prices.index[min(400, len(prices)-1)]
        end = prices.index[-1]

        logger.info("=" * 65)
        logger.info("BACKTESTING JARVIS V4.1 — ZERO CASH DRAG")
        logger.info(f"Period: {start.date()} → {end.date()}")
        logger.info(f"Top {self.config['top_n']} momentum ETFs, 100% invested always")
        logger.info(f"Cash position: 0% (BIL for risk-free, SPY as default)")
        logger.info("=" * 65)

        self._simulate(prices, start, end)
        return {
            "portfolio_history": pd.DataFrame(self.portfolio_history),
            "trade_log": self.trade_log,
            "state_log": self.state_log,
            "metrics": self._compute_metrics(),
        }

    def _simulate(self, prices, start, end):
        cfg = self.config
        cash = float(cfg["initial_capital"])
        positions = {}  # {ticker: qty}
        cost_bps = cfg["transaction_cost_bps"] + cfg["slippage_bps"]

        mode = "ACTIVE"  # ACTIVE or SAFETY
        sma_above_streak = 0
        sma_below_streak = 0
        last_rebalance = None
        current_holdings = []
        days_since_rebal = cfg["rebalance_frequency"]

        for date in prices.loc[start:end].index:
            # ── Mark to market ──
            pv = cash
            for t, qty in positions.items():
                if t in prices.columns:
                    px = float(prices.loc[date, t])
                    if pd.notna(px):
                        pv += qty * px

            # Track actual cash percentage
            cash_pct = cash / pv if pv > 0 else 0

            self.portfolio_history.append({
                "date": date, "total_value": pv, "cash": cash,
                "cash_pct": cash_pct, "mode": mode,
                "num_positions": len(positions),
            })

            # ══════════════════════════════════════════
            # 200-DAY SMA SWITCH (crash protection)
            # ══════════════════════════════════════════
            sma_status = self._sma_status(prices, date)

            if sma_status == "BELOW":
                sma_below_streak += 1
                sma_above_streak = 0
                if sma_below_streak >= cfg["sma_confirm_days"] and mode == "ACTIVE":
                    mode = "SAFETY"
                    logger.info(f"  📉 SAFETY at {date.date()}: QQQ below 200-SMA")
                    # Sell everything, buy safety allocation
                    cash = self._sell_all(positions, prices, date, cost_bps)
                    self._buy_allocation(positions, SAFETY_ALLOCATION, prices, date, pv, cash, cost_bps)
                    cash = self._remaining_cash(positions, prices, date, pv)
                    current_holdings = list(SAFETY_ALLOCATION.keys())

            elif sma_status == "ABOVE":
                sma_above_streak += 1
                sma_below_streak = 0
                if sma_above_streak >= cfg["sma_confirm_days"] and mode == "SAFETY":
                    mode = "ACTIVE"
                    logger.info(f"  📈 ACTIVE at {date.date()}: QQQ above 200-SMA")
                    cash = self._sell_all(positions, prices, date, cost_bps)
                    days_since_rebal = cfg["rebalance_frequency"]  # Force rebalance

            if mode == "SAFETY":
                continue

            # ══════════════════════════════════════════
            # MOMENTUM ROTATION (monthly)
            # ══════════════════════════════════════════
            days_since_rebal += 1
            if days_since_rebal < cfg["rebalance_frequency"]:
                continue

            days_since_rebal = 0
            last_rebalance = date

            # Score all ETFs
            scores = self._compute_momentum(prices, date, current_holdings)
            if not scores:
                continue

            # Select top N with sector constraints
            top_n = self._select_top_n(scores)

            if not top_n:
                continue

            # ══════════════════════════════════════════
            # V4.1 KEY: BUILD 100% INVESTED ALLOCATION
            # ══════════════════════════════════════════
            target = self._build_fully_invested_allocation(top_n, scores)

            # Execute rebalance
            self._rebalance(positions, target, prices, date, pv, cash, cost_bps)
            cash = self._remaining_cash(positions, prices, date, pv)
            current_holdings = list(target.keys())

            self.state_log.append({
                "date": date, "mode": mode,
                "holdings": current_holdings,
                "cash_pct": cash / pv if pv > 0 else 0,
            })

    def _build_fully_invested_allocation(self, top_n, scores):
        """
        V4.1 CORE: Build allocation that sums to exactly 100%.
        No cash. If momentum picks don't fill 100%, remainder → SPY.
        """
        cfg = self.config
        max_pos = cfg["max_single_position"]
        max_sector = cfg["max_sector_pct"]

        # Score-weighted allocation (higher score → bigger position)
        total_score = sum(max(scores.get(t, 0), 0.01) for t in top_n)
        target = {}

        for t in top_n:
            raw_weight = max(scores.get(t, 0), 0.01) / total_score
            # Cap at max_single_position
            target[t] = min(raw_weight, max_pos)

        # Enforce sector limits
        sector_totals = {}
        for t, w in target.items():
            sec = SECTOR_MAP.get(t, "other")
            sector_totals[sec] = sector_totals.get(sec, 0) + w

        for sec, total in sector_totals.items():
            if total > max_sector:
                scale = max_sector / total
                for t in target:
                    if SECTOR_MAP.get(t, "other") == sec:
                        target[t] *= scale

        # Renormalize to sum to 1.0
        allocated = sum(target.values())

        if allocated < 0.99:
            # V4.1 KEY: Remainder goes to SPY (default position), NOT cash
            remainder = 1.0 - allocated
            if DEFAULT_POSITION in target:
                target[DEFAULT_POSITION] += remainder
                target[DEFAULT_POSITION] = min(target[DEFAULT_POSITION], max_pos)
                # If SPY is already at cap, distribute to BIL
                if target[DEFAULT_POSITION] >= max_pos:
                    leftover = 1.0 - sum(target.values())
                    if leftover > 0.01:
                        target["BIL"] = target.get("BIL", 0) + leftover
            else:
                target[DEFAULT_POSITION] = remainder

        # Final normalization
        total = sum(target.values())
        if total > 1.001:
            target = {t: w / total for t, w in target.items()}

        # Remove tiny positions
        target = {t: w for t, w in target.items() if w > 0.02}

        # Re-check sum and top up if needed
        total = sum(target.values())
        if total < 0.98:
            target["BIL"] = target.get("BIL", 0) + (1.0 - total)

        return target

    def _compute_momentum(self, prices, date, current_holdings):
        """Weighted momentum score: 40% 3M + 35% 6M + 25% 12M, skip last month."""
        w3, w6, w12 = self.config["momentum_weights"]
        skip = self.config["momentum_skip"]
        scores = {}

        for ticker in MOMENTUM_POOL:
            if ticker not in prices.columns:
                continue
            s = prices[ticker].loc[:date].dropna()
            if len(s) < 252 + skip:
                continue

            # Skip last month (momentum reversal effect — Jegadeesh & Titman 1993)
            s_skipped = s.iloc[:-skip] if skip > 0 else s

            r3m = float((s_skipped.iloc[-1] / s_skipped.iloc[-63]) - 1) if len(s_skipped) >= 63 else 0
            r6m = float((s_skipped.iloc[-1] / s_skipped.iloc[-126]) - 1) if len(s_skipped) >= 126 else 0
            r12m = float((s_skipped.iloc[-1] / s_skipped.iloc[-252]) - 1) if len(s_skipped) >= 252 else 0

            score = w3 * r3m + w6 * r6m + w12 * r12m

            # Continuity bonus (reduces turnover — Novy-Marx 2012)
            if ticker in current_holdings:
                score += self.config["continuity_bonus"]

            scores[ticker] = score

        return scores

    def _select_top_n(self, scores):
        """Select top N with sector constraints."""
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = []
        sector_counts = {}
        max_sector_slots = max(2, int(self.config["top_n"] * 0.5))

        for ticker, score in ranked:
            if len(selected) >= self.config["top_n"]:
                break
            sec = SECTOR_MAP.get(ticker, "other")
            if sector_counts.get(sec, 0) >= max_sector_slots:
                continue
            selected.append(ticker)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        return selected

    def _sma_status(self, prices, date):
        """200-day SMA with buffer zone."""
        if "QQQ" not in prices.columns:
            return "ABOVE"
        q = prices["QQQ"].loc[:date].dropna()
        if len(q) < self.config["sma_lookback"]:
            return "ABOVE"
        sma = float(q.rolling(self.config["sma_lookback"]).mean().iloc[-1])
        price = float(q.iloc[-1])
        buf = self.config["sma_buffer"]
        if price > sma * (1 + buf):
            return "ABOVE"
        elif price < sma * (1 - buf):
            return "BELOW"
        return "BUFFER"

    def _rebalance(self, positions, target, prices, date, pv, cash, cost_bps):
        """Sell-first rebalance with cash constraint."""
        target_dollars = {t: pv * w for t, w in target.items()}

        # SELLS first
        for t in list(positions.keys()):
            if t not in target:
                if t in prices.columns:
                    px = float(prices.loc[date, t])
                    if pd.notna(px) and positions[t] != 0:
                        val = positions[t] * px
                        cost = abs(val) * (cost_bps / 10000)
                        cash += val - cost
                        self.trade_log.append({"date": date, "ticker": t, "side": "SELL",
                                               "value": abs(val), "cost": cost})
                        del positions[t]
            else:
                # Check if overweight
                if t in prices.columns:
                    px = float(prices.loc[date, t])
                    if pd.notna(px):
                        current_val = positions.get(t, 0) * px if t in positions else 0
                        target_val = target_dollars.get(t, 0)
                        if current_val > target_val * (1 + self.config["drift_threshold"]):
                            sell_val = current_val - target_val
                            sell_shares = sell_val / px
                            cost = sell_val * (cost_bps / 10000)
                            positions[t] -= sell_shares
                            cash += sell_val - cost
                            self.trade_log.append({"date": date, "ticker": t, "side": "SELL",
                                                   "value": sell_val, "cost": cost})

        # BUYS
        for t, target_val in sorted(target_dollars.items(), key=lambda x: x[1], reverse=True):
            if t in prices.columns:
                px = float(prices.loc[date, t])
                if pd.notna(px) and px > 0:
                    current_val = positions.get(t, 0) * px if t in positions else 0
                    needed = target_val - current_val

                    if needed < pv * self.config["drift_threshold"]:
                        continue

                    buy_val = min(needed, cash * 0.995)
                    if buy_val < 50:
                        continue

                    shares = buy_val / px
                    cost = buy_val * (cost_bps / 10000)

                    if t not in positions:
                        positions[t] = 0
                    positions[t] += shares
                    cash -= buy_val + cost
                    self.trade_log.append({"date": date, "ticker": t, "side": "BUY",
                                           "value": buy_val, "cost": cost})

        return cash

    def _sell_all(self, positions, prices, date, cost_bps):
        cash = 0
        for t in list(positions.keys()):
            if t in prices.columns:
                px = float(prices.loc[date, t])
                if pd.notna(px):
                    val = positions[t] * px
                    cost = abs(val) * (cost_bps / 10000)
                    cash += val - cost
                    self.trade_log.append({"date": date, "ticker": t, "side": "SELL",
                                           "value": abs(val), "cost": cost})
        positions.clear()
        return cash

    def _buy_allocation(self, positions, alloc, prices, date, pv, cash, cost_bps):
        for t, w in alloc.items():
            if t in prices.columns:
                px = float(prices.loc[date, t])
                if pd.notna(px) and px > 0:
                    buy_val = min(pv * w, cash * 0.98)
                    if buy_val < 50:
                        continue
                    shares = buy_val / px
                    cost = buy_val * (cost_bps / 10000)
                    positions[t] = shares
                    cash -= buy_val + cost

    def _remaining_cash(self, positions, prices, date, pv):
        invested = 0
        for t, qty in positions.items():
            if t in prices.columns:
                px = float(prices.loc[date, t])
                if pd.notna(px):
                    invested += qty * px
        return pv - invested

    def _compute_metrics(self):
        if not self.portfolio_history:
            return {}
        df = pd.DataFrame(self.portfolio_history).set_index("date")
        v = df["total_value"]
        if len(v) < 2:
            return {}

        dr = v.pct_change().dropna()
        tr = float(v.iloc[-1] / v.iloc[0]) - 1
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
        pf = abs(mo[mo > 0].mean() / mo[mo < 0].mean()) if (mo < 0).any() and (mo > 0).any() else 0

        # Average cash percentage
        avg_cash = df["cash_pct"].mean() if "cash_pct" in df.columns else 0

        # Time in modes
        active_pct = (df["mode"] == "ACTIVE").mean() if "mode" in df.columns else 1.0

        return {
            "total_return": tr, "annualized_return": ar,
            "annualized_volatility": vol, "sharpe_ratio": sharpe,
            "sortino_ratio": sortino, "max_drawdown": mdd,
            "calmar_ratio": calmar, "win_rate_monthly": wr,
            "profit_factor": pf,
            "best_month": mo.max() if not mo.empty else 0,
            "worst_month": mo.min() if not mo.empty else 0,
            "best_year": by.max() if not by.empty else 0,
            "worst_year": by.min() if not by.empty else 0,
            "total_trades": len(self.trade_log),
            "total_costs": sum(t.get("cost", 0) for t in self.trade_log),
            "years": yrs, "start_value": float(v.iloc[0]), "end_value": float(v.iloc[-1]),
            "avg_cash_pct": avg_cash,
            "pct_active_mode": active_pct,
        }

