"""
JARVIS V4.2 — Momentum Core + Simple Conditional Leverage
============================================================
V4.1 achieved 7.7% / 0.28 Sharpe over 25 years. Target: 15% / 0.5 Sharpe.

The gap is 7.3%/year. The ONLY way to close it with ETFs is leverage.
But V5 and V6 proved that COMPLEX leverage kills returns.

V4.2 uses the SIMPLEST possible leverage implementation:
- Same V4.1 momentum engine picks the best ETFs
- When conditions are safe: hold the 2x VERSION of the top picks
- When conditions are dangerous: hold the 1x version
- "Safe" = QQQ above 200-SMA (the single most validated filter in finance)

That's it. No regime prediction. No volatility filter. No 5-regime
framework. No inverse ETFs. Just: "Is QQQ above its 200-day average?
Yes → hold QLD instead of QQQ. No → hold QQQ."

WHY THIS SHOULD WORK (research basis):
1. QQQ 200-SMA strategy: 791% vs 428% buy-and-hold over 24 years
   (Financial Wisdom TV, Aug 2025)
2. Applied to 2x ETFs: the SMA filter removes the worst drawdown
   periods where leveraged decay destroys value
3. TQQQ/TMF with bi-monthly rebalance: 23.83% CAGR, 0.95 Sharpe
   over 15 years including COVID and 2022 (SSRN academic paper)
4. Jarvis V4 already proved momentum rotation works (0.42 Calmar)
5. V4.2 = V4's momentum + 2x when SMA says safe = best of both

WHAT V4.2 CHANGES FROM V4.1:
- In ACTIVE mode (QQQ > 200-SMA): top 2 momentum picks are replaced
  with their 2x equivalent (QQQ→QLD, SPY→SSO, XLK→ROM, SMH→USD)
- In SAFETY mode (QQQ < 200-SMA): 100% unleveraged defensive (same as V4.1)
- Position sizes are 25% each of actual portfolio value (no synthetic leverage)
- Max 2x ETFs only (no 3x). No inverse ETFs. No shorting.

WHAT COULD GO WRONG:
- The 200-SMA filter is late by definition (~2-3 weeks). During a flash
  crash, 2x positions lose twice as much before the filter triggers.
  Mitigation: -25% circuit breaker forces immediate de-leverage.
- Volatility decay in choppy markets erodes 2x ETF value.
  Mitigation: momentum rotation naturally rotates OUT of decaying ETFs.
- V4.2 will have higher drawdowns than V4.1 (~35-45% vs ~30%).
  This is the explicit tradeoff for targeting 15% returns.
"""

import pandas as pd
import numpy as np
from loguru import logger

# ════════════════════════════════════════════════════════
# UNIVERSE
# ════════════════════════════════════════════════════════

MOMENTUM_POOL = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "VTV", "VUG",
    "XLK", "XLF", "XLV", "XLE", "XLI",
]

# 2x leveraged equivalents (only for ETFs that HAVE a 2x version)
LEVERAGE_MAP = {
    "QQQ": "QLD",   # 2x Nasdaq-100
    "SPY": "SSO",   # 2x S&P 500
    "XLK": "ROM",   # 2x Technology
    "SMH": "USD",   # 2x Semiconductors
    "XLF": "UYG",   # 2x Financials
    "XLE": "DIG",   # 2x Energy
}

# ════════════════════════════════════════════════════════
# V4.2 CRISIS ALPHA: Momentum rotation among defensive assets
# Instead of static bonds/gold, apply momentum scoring to
# crisis assets and ride the winners.
#
# RESEARCH BASIS:
# - Managed Futures: Hurst, Ooi & Pedersen (2017) — profitable
#   in every decade since 1880, strongest during equity crises
# - Gold: Baur & Lucey (2010) — negative equity correlation in stress
# - Gold Miners: 2-3x gold amplification (GDX +180% Mar-Aug 2020)
# - Momentum in defensives: Asness (2014) — momentum works across
#   ALL asset classes including bonds, commodities, currencies
# ════════════════════════════════════════════════════════

# Crisis Alpha Pool: scored by momentum, top 4 held
CRISIS_ALPHA_POOL = [
    "GLD",    # Gold — the OG crisis asset
    "GDX",    # Gold miners — 2-3x gold amplification
    "SLV",    # Silver — industrial + precious metal hybrid
    "TLT",    # Long bonds — rally during rate cuts
    "IEF",    # Intermediate bonds — less volatile than TLT
    "DBMF",   # Managed futures — trend following replicator (AQR/Man AHL style)
    "CTA",    # Managed futures — Simplify's trend following
    "BIL",    # T-bills — risk-free floor (~4.5%)
    "DBC",    # Broad commodities — inflation hedge
    "XLU",    # Utilities — defensive equity (bond proxy)
    "XLP",    # Consumer staples — recession-resistant
    "XLV",    # Healthcare — defensive equity
]

# Fallback static allocation (if not enough momentum data for crisis pool)
SAFETY_FALLBACK = {
    "GLD": 0.30,
    "TLT": 0.25,
    "DBMF": 0.25,
    "BIL": 0.20,
}

CRISIS_TOP_N = 4  # Hold top 4 crisis alpha assets by momentum

SECTOR_MAP = {
    "SPY": "broad", "QQQ": "tech", "IWM": "small", "EFA": "intl",
    "EEM": "em", "VTV": "value", "VUG": "growth", "XLK": "tech",
    "XLF": "financial", "XLV": "healthcare", "XLE": "energy", "XLI": "industrial",
}

DEFAULT_CONFIG = {
    "initial_capital": 100000,
    "top_n": 5,                      # Top 5 momentum picks
    "n_leveraged": 3,                # Top 3 get 2x treatment
    "momentum_weights": [0.40, 0.35, 0.25],
    "momentum_skip": 21,
    "continuity_bonus": 0.02,
    "max_single_position": 0.25,     # 25% max per position
    "max_sector_pct": 0.45,
    "rebalance_frequency": 21,       # Monthly
    "drift_threshold": 0.03,
    "sma_lookback": 200,
    "sma_buffer": 0.02,
    "sma_confirm_days": 3,
    "circuit_breaker_pct": -0.25,
    "circuit_breaker_cooldown": 30,
    "cost_bps": 8,                   # 5 transaction + 3 slippage
}


class BacktestV42:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.portfolio_history = []
        self.trade_log = []
        self.state_log = []
        self._safety_days = 0

    def run(self, prices, vix=None):
        if prices.empty:
            return {}

        # Start after enough data for 200-SMA + 12-month momentum
        start_idx = 500
        if "QQQ" in prices.columns:
            qqq = prices["QQQ"].dropna()
            if len(qqq) > start_idx:
                start = qqq.index[start_idx]
            else:
                start = prices.index[min(start_idx, len(prices)-1)]
        else:
            start = prices.index[min(start_idx, len(prices)-1)]
        end = prices.index[-1]

        logger.info("=" * 65)
        logger.info("BACKTESTING JARVIS V4.2 — MOMENTUM + SIMPLE LEVERAGE")
        logger.info(f"Period: {start.date()} → {end.date()}")
        logger.info(f"Top {self.config['top_n']} momentum, top {self.config['n_leveraged']} get 2x")
        logger.info(f"Leverage rule: QQQ > 200-SMA → 2x | Below → 1x safety")
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
        positions = {}
        cost_bps = cfg["cost_bps"]

        mode = "ACTIVE"
        sma_up = 0
        sma_dn = 0
        last_rebalance = None
        current_holdings = []
        days_since = cfg["rebalance_frequency"]
        peak = cash
        cb_active = False
        cb_date = None

        for date in prices.loc[start:end].index:
            # Mark to market
            pv = cash
            for t, qty in positions.items():
                if t in prices.columns:
                    px = float(prices.loc[date, t])
                    if pd.notna(px):
                        pv += qty * px

            cash_pct = cash / pv if pv > 0 else 0
            peak = max(peak, pv)

            # Count how much is in leveraged vs unleveraged
            lev_val = sum(
                qty * float(prices.loc[date, t]) for t, qty in positions.items()
                if t in LEVERAGE_MAP.values() and t in prices.columns and pd.notna(prices.loc[date, t])
            )
            lev_pct = lev_val / pv if pv > 0 else 0

            self.portfolio_history.append({
                "date": date, "total_value": pv, "cash": cash,
                "cash_pct": cash_pct, "mode": mode,
                "leverage_pct": lev_pct,
                "num_positions": len(positions),
            })

            # Circuit breaker
            dd = (pv / peak) - 1 if peak > 0 else 0
            if dd <= cfg["circuit_breaker_pct"] and not cb_active:
                cb_active = True
                cb_date = date
                mode = "SAFETY"
                logger.info(f"  🚨 CIRCUIT BREAKER at {date.date()}: DD={dd:.1%}")
                cash = self._sell_all(positions, prices, date, cost_bps)
                crisis_target = self._compute_crisis_alpha(prices, date, current_holdings)
                cash = self._buy_allocation(positions, crisis_target, prices, date, pv, cash, cost_bps)
                cash = max(cash, 0)
                peak = pv  # Reset peak after circuit breaker reallocation
                continue

            if cb_active:
                if cb_date and (date - cb_date).days >= cfg["circuit_breaker_cooldown"]:
                    if self._sma_above(prices, date):
                        cb_active = False
                        mode = "ACTIVE"
                        cash = self._sell_all(positions, prices, date, cost_bps)
                        pv = cash  # Update pv after selling
                        peak = pv  # Reset peak after circuit breaker recovery
                        days_since = cfg["rebalance_frequency"]
                    else:
                        continue
                else:
                    continue

            # ── 200-DAY SMA SWITCH ──
            sma_st = self._sma_status(prices, date)

            if sma_st == "BELOW":
                sma_dn += 1
                sma_up = 0
                if sma_dn >= cfg["sma_confirm_days"] and mode == "ACTIVE":
                    mode = "SAFETY"
                    safety_rebal_days = 0  # Track rebalancing within safety
                    logger.info(f"  📉 SAFETY at {date.date()}")
                    cash = self._sell_all(positions, prices, date, cost_bps)
                    crisis_target = self._compute_crisis_alpha(prices, date, current_holdings)
                    cash = self._buy_allocation(positions, crisis_target, prices, date, pv, cash, cost_bps)
                    cash = max(cash, 0)
                    peak = pv  # Reset peak after SAFETY entry
                    current_holdings = list(crisis_target.keys())
            elif sma_st == "ABOVE":
                sma_up += 1
                sma_dn = 0
                if sma_up >= cfg["sma_confirm_days"] and mode == "SAFETY":
                    mode = "ACTIVE"
                    logger.info(f"  📈 ACTIVE at {date.date()}")
                    cash = self._sell_all(positions, prices, date, cost_bps)
                    pv = cash  # Update pv after selling
                    peak = pv  # Reset peak on mode transition to ACTIVE
                    days_since = cfg["rebalance_frequency"]

            if mode == "SAFETY":
                # V4.2 CRISIS ALPHA: Monthly rebalance within safety mode
                # Rotate among crisis assets by momentum (not static hold)
                safety_rebal_days = getattr(self, '_safety_days', 0) + 1
                self._safety_days = safety_rebal_days
                if safety_rebal_days >= cfg["rebalance_frequency"]:
                    self._safety_days = 0
                    crisis_target = self._compute_crisis_alpha(prices, date, current_holdings)
                    cash = self._rebalance(positions, crisis_target, prices, date, pv, cash, cost_bps)
                    current_holdings = list(crisis_target.keys())
                    self.state_log.append({
                        "date": date, "mode": "SAFETY",
                        "holdings": current_holdings, "leveraged": [],
                    })
                continue

            # ── MOMENTUM ROTATION (monthly) ──
            days_since += 1
            if days_since < cfg["rebalance_frequency"]:
                continue
            days_since = 0

            scores = self._compute_momentum(prices, date, current_holdings)
            if not scores:
                continue

            top_n = self._select_top_n(scores)
            if not top_n:
                continue

            # ════════════════════════════════════════
            # V4.2 KEY: TOP N_LEVERAGED GET 2x
            # ════════════════════════════════════════
            target = self._build_leveraged_allocation(top_n, scores)

            # Execute
            cash = self._rebalance(positions, target, prices, date, pv, cash, cost_bps)
            current_holdings = list(target.keys())

            self.state_log.append({
                "date": date, "mode": mode,
                "holdings": current_holdings,
                "leveraged": [t for t in current_holdings if t in LEVERAGE_MAP.values()],
            })

    def _build_leveraged_allocation(self, top_n, scores):
        """
        V4.2: Top N_LEVERAGED picks → 2x ETF equivalent.
        Remaining picks → 1x ETF.
        All positions sized equally (1/top_n each).

        Example with top_n=5, n_leveraged=3:
          #1 QQQ → QLD (2x), 20%
          #2 SMH → USD (2x), 20%
          #3 XLK → ROM (2x), 20%
          #4 XLV → XLV (1x), 20%  (no 2x version)
          #5 EFA → EFA (1x), 20%  (no 2x version)

        Effective exposure: 3×40% + 2×20% = 160% (1.6x portfolio leverage)
        But actual capital deployed: 100% (no borrowing)
        """
        n_lev = self.config["n_leveraged"]
        max_pos = self.config["max_single_position"]
        equal_weight = 1.0 / len(top_n)

        target = {}
        for i, ticker in enumerate(top_n):
            weight = min(equal_weight, max_pos)

            if i < n_lev and ticker in LEVERAGE_MAP:
                # This pick gets the 2x version
                lev_ticker = LEVERAGE_MAP[ticker]
                target[lev_ticker] = weight
            else:
                # This pick stays unleveraged
                target[ticker] = weight

        # Fill any remainder with SPY (zero cash policy)
        allocated = sum(target.values())
        if allocated < 0.98:
            target["SPY"] = target.get("SPY", 0) + (1.0 - allocated)

        # Normalize if over 1.0
        total = sum(target.values())
        if total > 1.001:
            target = {t: w / total for t, w in target.items()}

        return target

    def _compute_momentum(self, prices, date, current):
        w3, w6, w12 = self.config["momentum_weights"]
        skip = self.config["momentum_skip"]
        scores = {}

        for t in MOMENTUM_POOL:
            if t not in prices.columns:
                continue
            s = prices[t].loc[:date].dropna()
            if len(s) < 252 + skip:
                continue

            s_sk = s.iloc[:-skip] if skip > 0 else s
            r3 = float((s_sk.iloc[-1] / s_sk.iloc[-63]) - 1) if len(s_sk) >= 63 else 0
            r6 = float((s_sk.iloc[-1] / s_sk.iloc[-126]) - 1) if len(s_sk) >= 126 else 0
            r12 = float((s_sk.iloc[-1] / s_sk.iloc[-252]) - 1) if len(s_sk) >= 252 else 0

            sc = w3 * r3 + w6 * r6 + w12 * r12
            if t in current:
                sc += self.config["continuity_bonus"]
            scores[t] = sc

        return scores

    def _select_top_n(self, scores):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = []
        sec_counts = {}
        max_sec = max(2, int(self.config["top_n"] * 0.5))

        for t, _ in ranked:
            if len(selected) >= self.config["top_n"]:
                break
            sec = SECTOR_MAP.get(t, "other")
            if sec_counts.get(sec, 0) >= max_sec:
                continue
            selected.append(t)
            sec_counts[sec] = sec_counts.get(sec, 0) + 1

        return selected

    def _compute_crisis_alpha(self, prices, date, current_holdings):
        """
        V4.2 CRISIS ALPHA ENGINE
        Instead of static bonds/gold, apply momentum scoring to crisis assets.
        Pick the top CRISIS_TOP_N by 1M/3M/6M momentum and ride the winners.

        Research basis:
        - Asness, Moskowitz, Pedersen (2013): "Value and Momentum Everywhere"
          Momentum works in bonds, commodities, currencies — not just equities.
        - Hurst, Ooi, Pedersen (2017): Managed futures profit during equity crises
          because they trend-follow across ALL asset classes.
        - During 2008: Gold +5%, Managed futures +18%, TLT +34%
        - During 2020: Gold +25%, TLT +18% (then reversed)
        - During 2022: DBMF +8%, DBC +21% — ONLY winners in an all-asset crash

        The key insight: the BEST defensive asset changes every crisis.
        Static allocation picks an average. Momentum picks the winner.
        """
        scores = {}

        for ticker in CRISIS_ALPHA_POOL:
            if ticker not in prices.columns:
                continue
            s = prices[ticker].loc[:date].dropna()
            if len(s) < 63:
                continue

            # Shorter lookbacks for crisis assets (they move faster)
            r1m = float((s.iloc[-1] / s.iloc[-21]) - 1) if len(s) >= 21 else 0
            r3m = float((s.iloc[-1] / s.iloc[-63]) - 1) if len(s) >= 63 else 0
            r6m = float((s.iloc[-1] / s.iloc[-126]) - 1) if len(s) >= 126 else 0

            # Crisis momentum: heavier weight on shorter timeframes
            # (crises develop fast, 6-month lookback is too slow)
            score = 0.50 * r1m + 0.30 * r3m + 0.20 * r6m

            # Continuity bonus
            if ticker in current_holdings:
                score += 0.01

            # BIL gets a floor score — it's the risk-free fallback
            # Never negative, always available
            if ticker == "BIL":
                score = max(score, 0.005)

            scores[ticker] = score

        if not scores:
            # Fallback: only use assets with valid prices at this date
            fallback = {}
            for t, w in SAFETY_FALLBACK.items():
                if t in prices.columns and pd.notna(prices.loc[date, t]) if t in prices.columns else False:
                    fallback[t] = w
            if not fallback:
                # Last resort: use any available equity ETFs
                for t in ["XLP", "XLU", "XLV", "SPY"]:
                    if t in prices.columns and pd.notna(float(prices.loc[date, t])):
                        fallback[t] = 1.0 / 4
            if fallback:
                total = sum(fallback.values())
                return {t: w / total for t, w in fallback.items()}
            return {"SPY": 1.0}  # Absolute last resort

        # Select top CRISIS_TOP_N — only assets with valid current prices
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = []
        for ticker, score in ranked:
            if len(top) >= CRISIS_TOP_N:
                break
            # Verify the asset has a valid price today
            if ticker in prices.columns:
                px = prices.loc[date, ticker]
                if pd.notna(px) and float(px) > 0:
                    top.append((ticker, score))

        if not top:
            return {"SPY": 1.0}  # Absolute last resort

        # Score-weighted allocation (not equal weight — ride the winner harder)
        total_score = sum(max(s, 0.001) for _, s in top)
        target = {}
        for ticker, score in top:
            weight = max(score, 0.001) / total_score
            # Cap any single defensive position at 35%
            target[ticker] = min(weight, 0.35)

        # Normalize to 100% — redistribute among available assets, not BIL
        total = sum(target.values())
        if total < 0.95:
            # Distribute remainder proportionally among existing holdings
            remainder = 1.0 - total
            for t in target:
                target[t] += remainder * (target[t] / total)
        if sum(target.values()) > 1.001:
            total = sum(target.values())
            target = {t: w / total for t, w in target.items()}

        return target

    def _sma_above(self, prices, date):
        if "QQQ" not in prices.columns:
            return True
        q = prices["QQQ"].loc[:date].dropna()
        if len(q) < self.config["sma_lookback"]:
            return True
        return float(q.iloc[-1]) > float(q.rolling(self.config["sma_lookback"]).mean().iloc[-1])

    def _sma_status(self, prices, date):
        if "QQQ" not in prices.columns:
            return "ABOVE"
        q = prices["QQQ"].loc[:date].dropna()
        if len(q) < self.config["sma_lookback"]:
            return "ABOVE"
        sma = float(q.rolling(self.config["sma_lookback"]).mean().iloc[-1])
        p = float(q.iloc[-1])
        b = self.config["sma_buffer"]
        if p > sma * (1 + b):
            return "ABOVE"
        elif p < sma * (1 - b):
            return "BELOW"
        return "BUFFER"

    def _rebalance(self, positions, target, prices, date, pv, cash, cost_bps):
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
                        self.trade_log.append({"date":date,"ticker":t,"side":"SELL","value":abs(val),"cost":cost})
                        del positions[t]
            else:
                if t in prices.columns:
                    px = float(prices.loc[date, t])
                    if pd.notna(px) and t in positions:
                        cur = positions[t] * px
                        tgt = target_dollars.get(t, 0)
                        if cur > tgt * (1 + self.config["drift_threshold"]):
                            sell = cur - tgt
                            shares = sell / px
                            cost = sell * (cost_bps / 10000)
                            positions[t] -= shares
                            cash += sell - cost
                            self.trade_log.append({"date":date,"ticker":t,"side":"SELL","value":sell,"cost":cost})

        # BUYS
        for t, tgt_val in sorted(target_dollars.items(), key=lambda x: x[1], reverse=True):
            if t not in prices.columns:
                continue
            px = float(prices.loc[date, t])
            if pd.isna(px) or px <= 0:
                continue

            cur = positions.get(t, 0) * px if t in positions else 0
            needed = tgt_val - cur

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
            self.trade_log.append({"date":date,"ticker":t,"side":"BUY","value":buy_val,"cost":cost})

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
                    self.trade_log.append({"date":date,"ticker":t,"side":"SELL","value":abs(val),"cost":cost})
        positions.clear()
        return cash

    def _buy_allocation(self, positions, alloc, prices, date, pv, cash, cost_bps):
        for t, w in alloc.items():
            if t in prices.columns:
                px = float(prices.loc[date, t])
                if pd.notna(px) and px > 0:
                    buy_val = min(pv * w, cash * 0.95)
                    if buy_val < 50:
                        continue
                    shares = buy_val / px
                    cost = buy_val * (cost_bps / 10000)
                    positions[t] = shares
                    cash -= buy_val + cost
        return cash

    def _compute_metrics(self):
        if not self.portfolio_history:
            return {}
        df = pd.DataFrame(self.portfolio_history).set_index("date")
        v = df["total_value"]
        if len(v) < 2:
            return {}

        dr = v.pct_change().dropna()
        tr = float((v.iloc[-1] / v.iloc[0]) - 1)
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

        avg_cash = df["cash_pct"].mean() if "cash_pct" in df.columns else 0
        avg_lev = df["leverage_pct"].mean() if "leverage_pct" in df.columns else 0
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
            "avg_leverage_pct": avg_lev,
            "pct_active_mode": active_pct,
        }
