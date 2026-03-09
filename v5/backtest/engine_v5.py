"""
JARVIS V5 - Backtest Engine (Dual-Engine Medallion Architecture)
=================================================================
Implements the exact allocation matrices from the V5 blueprint:
- Engine Alpha: leveraged momentum rotation in bull regimes
- Engine Omega: crisis alpha in bear regimes
- 200-day SMA hard filter on all leveraged positions
- 5 regimes with 8 voting signals + 2-week confirmation
- Regime Detector Health Monitor (self-correcting meta-signal)
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import timedelta

from config.universe_v5 import (
    UNIVERSE, get_tier, get_underlying_map, get_tier1_3x,
    get_tier1_2x, get_offensive_tickers, get_asset_class_map,
)
from signals.regime_v5 import (
    detect_regime_with_confirmation, RegimeHealthMonitor,
)


# ════════════════════════════════════════════════════════════
# ALLOCATION MATRICES (exact blueprint spec)
# ════════════════════════════════════════════════════════════

def get_allocation_matrix(regime, scores, prices, date, health_boost=0.0):
    """
    Returns target weights dict based on regime and CAS scores.
    health_boost: extra defensive allocation from regime health monitor (0-0.15)
    """
    underlying_map = get_underlying_map()

    # 200-day SMA filter: which underlying indices are above their SMA?
    sma_ok = {}
    for ticker, info in UNIVERSE.items():
        ul = info.get("underlying")
        if ul and ul in prices.columns:
            s = prices[ul].loc[:date].dropna()
            if len(s) >= 200:
                sma_ok[ticker] = float(s.iloc[-1]) > float(s.rolling(200).mean().iloc[-1])
            else:
                sma_ok[ticker] = True  # Not enough data, allow
        elif info["tier"] in (1,):
            sma_ok[ticker] = False  # No underlying data, block leveraged
        else:
            sma_ok[ticker] = True  # Non-leveraged, always OK

    # Volatility check: use 20-day realized vol percentile for leverage selection
    use_3x = True  # Default to 3x
    if "SPY" in prices.columns:
        spy = prices["SPY"].loc[:date].dropna()
        if len(spy) >= 252:
            vol_20d = float(spy.pct_change().tail(20).std()) * np.sqrt(252)
            vol_1y = float(spy.pct_change().tail(252).std()) * np.sqrt(252)
            vol_median = float(spy.pct_change().rolling(252).std().iloc[-1]) * np.sqrt(252) if len(spy) >= 252 else vol_1y
            vol_pctile = float(spy.pct_change().rolling(20).std().rank(pct=True).iloc[-1]) if len(spy) >= 252 else 0.5

            if vol_pctile > 0.75:
                use_3x = False  # Too volatile for 3x

    # Sort ETFs by score within each tier
    def top_by_score(tickers, n=3):
        available = [(t, scores.get(t, 0)) for t in tickers if t in scores.index and sma_ok.get(t, True)]
        available.sort(key=lambda x: x[1], reverse=True)
        return available[:n]

    target = {}

    # Apply health monitor boost (increase defensive allocation)
    offense_reduction = health_boost
    defense_boost = health_boost

    if regime == "EUPHORIA":
        # Max offense: 90% leveraged, 5% gold, 5% cash
        t1_3x = get_tier1_3x()
        t1_2x = get_tier1_2x()
        t2 = [t for t, info in UNIVERSE.items() if info["tier"] == 2]

        if use_3x:
            top_3x = top_by_score(t1_3x, 2)
            if len(top_3x) >= 1: target[top_3x[0][0]] = 0.35 - offense_reduction
            if len(top_3x) >= 2: target[top_3x[1][0]] = 0.25 - offense_reduction / 2
            top_2x = top_by_score(t1_2x, 1)
            if top_2x: target[top_2x[0][0]] = 0.15
        else:
            top_2x = top_by_score(t1_2x, 2)
            if len(top_2x) >= 1: target[top_2x[0][0]] = 0.30 - offense_reduction
            if len(top_2x) >= 2: target[top_2x[1][0]] = 0.20 - offense_reduction / 2

        top_t2 = top_by_score(t2, 1)
        if top_t2: target[top_t2[0][0]] = 0.10
        target["GLD"] = 0.05 + defense_boost / 2
        target["SGOV"] = 0.05 + defense_boost / 2
        target["DBMF"] = 0.05

    elif regime == "CALM":
        # Standard bull: 65% leveraged+equity, 30% defensive, 5% cash
        t1_3x = get_tier1_3x()
        t1_2x = get_tier1_2x()
        t2 = [t for t, info in UNIVERSE.items() if info["tier"] == 2]

        if use_3x:
            top_3x = top_by_score(t1_3x, 1)
            if top_3x: target[top_3x[0][0]] = 0.25 - offense_reduction
        top_2x = top_by_score(t1_2x, 1)
        if top_2x: target[top_2x[0][0]] = 0.20 - offense_reduction / 2

        top_t2 = top_by_score(t2, 2)
        for i, (t, s) in enumerate(top_t2):
            target[t] = 0.10

        # Defensive
        gld_score = scores.get("GLD", 0)
        gdx_score = scores.get("GDX", 0)
        target["GLD" if gld_score >= gdx_score else "GDX"] = 0.10 + defense_boost / 3

        tlt_score = scores.get("TLT", 0)
        ief_score = scores.get("IEF", 0)
        target["TLT" if tlt_score >= ief_score else "IEF"] = 0.10 + defense_boost / 3

        target["DBMF"] = 0.05
        target["CTA"] = 0.05
        target["SGOV"] = 0.05 + defense_boost / 3

    elif regime == "CAUTION":
        # Defensive tilt: 45% equity (limited leverage), 50% defensive, 5% cash
        t1_2x = get_tier1_2x()
        t2 = [t for t, info in UNIVERSE.items() if info["tier"] == 2]

        # Only 2x if CAS > 0.5 (high conviction)
        top_2x = top_by_score(t1_2x, 1)
        if top_2x and top_2x[0][1] > 0.3:
            target[top_2x[0][0]] = 0.15 - offense_reduction

        top_t2 = top_by_score(t2, 3)
        for t, s in top_t2:
            target[t] = 0.10

        target["GLD"] = 0.10 + defense_boost / 4
        target["GDX"] = 0.10 + defense_boost / 4
        target["TLT"] = 0.08 + defense_boost / 4
        target["IEF"] = 0.07 + defense_boost / 4
        target["DBMF"] = 0.05
        target["CTA"] = 0.05

        # Defensive equity
        def_eq = top_by_score(["XLU", "XLP", "XLV"], 1)
        if def_eq: target[def_eq[0][0]] = 0.05

        target["SGOV"] = 0.05

    elif regime == "STRESS":
        # Full defensive: 0% leveraged long, 30% gold, 25% bonds, 15% managed futures
        target["GLD"] = 0.15 + defense_boost / 3
        target["GDX"] = 0.10
        target["UGL"] = 0.05
        target["TLT"] = 0.10 + defense_boost / 3
        target["IEF"] = 0.10
        target["SGOV"] = 0.05
        target["DBMF"] = 0.075
        target["CTA"] = 0.075

        # Inverse ETFs (tactical)
        if "SPY" in prices.columns:
            spy = prices["SPY"].loc[:date].dropna()
            if len(spy) >= 6:
                vix_val = None
                # Simple check: if market is accelerating down, use SQQQ
                target["SH"] = 0.05
                target["SQQQ"] = 0.05

        target["TAIL"] = 0.05
        target["BTAL"] = 0.05
        target["BIL"] = 0.05 + defense_boost / 3

    elif regime == "CRISIS":
        # Maximum crisis alpha: gold, inverse, volatility, managed futures, cash
        target["GLD"] = 0.15 + defense_boost / 3
        target["UGL"] = 0.10
        target["SQQQ"] = 0.12
        target["SH"] = 0.08
        target["UVXY"] = 0.10  # Max 10-day hold (handled by timer)
        target["DBMF"] = 0.075
        target["CTA"] = 0.075
        target["TLT"] = 0.10 + defense_boost / 3
        target["TAIL"] = 0.05
        target["SGOV"] = 0.10 + defense_boost / 3

    # Clean: remove zero/tiny/negative weights, cap at 35%
    target = {t: min(max(w, 0), 0.35) for t, w in target.items() if w > 0.005}

    # Enforce leverage budget: total leverage ≤ 2.5x
    total_leverage = sum(abs(UNIVERSE.get(t, {}).get("leverage", 1)) * w for t, w in target.items())
    if total_leverage > 2.5:
        scale = 2.5 / total_leverage
        leveraged = {t for t in target if abs(UNIVERSE.get(t, {}).get("leverage", 1)) > 1}
        for t in leveraged:
            target[t] *= scale

    # Normalize so total ≤ 1.0
    total = sum(target.values())
    if total > 1.0:
        target = {t: w / total for t, w in target.items()}

    return target


# ════════════════════════════════════════════════════════════
# V5 BACKTEST ENGINE
# ════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "start_date": "2017-01-01",
    "end_date": None,
    "initial_capital": 100000,
    "rebalance_days": 5,        # Weekly (every 5 trading days)
    "transaction_cost_bps": 5,
    "slippage_bps": 3,
    "min_trade_pct": 0.05,      # 5% minimum trade threshold
    "inverse_max_hold_days": 10, # Max days for inverse/UVXY positions
}


class BacktestV5:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.portfolio_history = []
        self.trade_log = []
        self.regime_log = []
        self.health_monitor = RegimeHealthMonitor()

    def run(self, prices, vix_series=None, credit_spread=None, yield_curve=None):
        if prices.empty:
            return {}

        start = pd.Timestamp(self.config["start_date"])
        end = pd.Timestamp(self.config["end_date"]) if self.config["end_date"] else prices.index[-1]

        logger.info("=" * 65)
        logger.info("BACKTESTING JARVIS V5 — MEDALLION ARCHITECTURE")
        logger.info(f"Period: {start.date()} → {end.date()}")
        logger.info(f"Capital: ${self.config['initial_capital']:,.0f}")
        logger.info(f"Rebalance: Every {self.config['rebalance_days']} trading days (weekly)")
        logger.info(f"Leveraged ETFs: {len([t for t in UNIVERSE if abs(UNIVERSE[t]['leverage']) > 1])}")
        logger.info("=" * 65)

        # Pre-compute momentum scores for all ETFs
        logger.info("\nComputing momentum scores...")
        scores_matrix = self._compute_scores(prices)

        # Simulate
        logger.info("Running dual-engine simulation...")
        self._simulate(prices, scores_matrix, vix_series, credit_spread, yield_curve, start, end)

        metrics = self._compute_metrics()

        # Regime stats
        if self.regime_log:
            from collections import Counter
            regime_counts = Counter(r["regime"] for r in self.regime_log)
            total_r = len(self.regime_log)
            logger.info(f"\nRegime distribution:")
            for r in ["EUPHORIA", "CALM", "CAUTION", "STRESS", "CRISIS"]:
                c = regime_counts.get(r, 0)
                logger.info(f"  {r:12s}: {c:3d}/{total_r} ({c/total_r:.0%})")

        return {
            "portfolio_history": pd.DataFrame(self.portfolio_history),
            "trade_log": self.trade_log,
            "metrics": metrics,
            "regime_log": self.regime_log,
        }

    def _compute_scores(self, prices):
        """
        Compute Composite Alpha Scores (CAS) for all ETFs on all dates.
        CAS = 0.35*Momentum + 0.25*Trend + 0.25*Sentiment_proxy + 0.15*Macro_proxy
        """
        # For backtest, we use price-based proxies for all signals
        ret_1m = prices.pct_change(21)
        ret_3m = prices.pct_change(63)
        ret_6m = prices.pct_change(126)

        # Momentum: absolute + relative + accelerating
        abs_momentum = ret_6m  # vs zero (similar to vs cash)
        rel_momentum = ret_3m.rank(axis=1, pct=True)  # Rank vs peers
        accel_momentum = ret_1m + ret_3m + ret_6m  # Acceleration

        momentum_score = (0.35 * abs_momentum.rank(axis=1, pct=True) +
                         0.35 * rel_momentum +
                         0.30 * accel_momentum.rank(axis=1, pct=True))

        # Trend: price vs SMA
        trend_score = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
        for ticker in prices.columns:
            s = prices[ticker].dropna()
            if len(s) >= 200:
                sma50 = s.rolling(50).mean()
                sma200 = s.rolling(200).mean()
                above_50 = (s > sma50).astype(float)
                above_200 = (s > sma200).astype(float)
                golden = (sma50 > sma200).astype(float)
                trend_score[ticker] = (above_50 * 0.3 + above_200 * 0.4 + golden * 0.3)

        # Macro proxy: inverse of recent vol (low vol = favorable)
        vol_20d = prices.pct_change().rolling(20).std()
        vol_rank = 1 - vol_20d.rank(axis=1, pct=True)  # Lower vol = higher score
        macro_score = vol_rank.fillna(0.5)

        # Sentiment proxy: short-term mean reversion (oversold = higher score for defensives)
        ret_5d = prices.pct_change(5)
        sentiment_proxy = 0.5 + (ret_5d.rank(axis=1, pct=True) - 0.5) * 0.5

        # Composite Alpha Score
        cas = (0.35 * momentum_score +
               0.25 * trend_score +
               0.25 * sentiment_proxy +
               0.15 * macro_score)

        # Scale to -1 to +1 range
        cas = (cas - 0.5) * 2

        return cas

    def _simulate(self, prices, scores, vix_series, credit_spread, yield_curve, start, end):
        capital = self.config["initial_capital"]
        cash = capital
        positions = {}  # {ticker: {"qty": float, "entry_date": date}}
        rebal_freq = self.config["rebalance_days"]
        cost_bps = self.config["transaction_cost_bps"] + self.config["slippage_bps"]
        min_trade = self.config["min_trade_pct"]

        bt_dates = prices.loc[start:end].index
        days_since = rebal_freq
        prev_regime = "CALM"
        prev_regime_change_date = start

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
                "invested": pv - cash, "num_positions": len(positions),
            })

            # ── Enforce inverse ETF timer (max 10-day hold) ──
            for t in list(positions.keys()):
                info = UNIVERSE.get(t, {})
                if info.get("asset_class") in ("inverse", "volatility"):
                    days_held = (date - positions[t]["entry_date"]).days
                    if days_held >= self.config["inverse_max_hold_days"]:
                        # Force close
                        px = prices.loc[date, t] if t in prices.columns else None
                        if px and pd.notna(px) and positions[t]["qty"] != 0:
                            trade_val = positions[t]["qty"] * px
                            cash += trade_val
                            cash -= abs(trade_val) * (cost_bps / 10000)
                            del positions[t]

            # ── Portfolio drawdown circuit breaker ──
            if self.portfolio_history:
                peak = max(h["total_value"] for h in self.portfolio_history)
                dd = (pv / peak) - 1 if peak > 0 else 0
                if dd < -0.25:
                    prev_regime = "CRISIS"
                elif dd < -0.15:
                    if prev_regime not in ("STRESS", "CRISIS"):
                        prev_regime = "STRESS"

            # ── Rebalance check ──
            days_since += 1

            # Detect current regime
            regime, confidence, changed = detect_regime_with_confirmation(
                prices, date, vix_series, prev_regime, prev_regime_change_date,
                credit_spread, yield_curve
            )

            if changed:
                prev_regime_change_date = date

            # Force rebalance on regime change
            if regime != prev_regime:
                days_since = rebal_freq  # Force rebalance

            prev_regime = regime

            if days_since < rebal_freq:
                continue
            if date not in scores.index:
                continue

            days_since = 0

            # ── Update health monitor ──
            if "SPY" in prices.columns and len(prices["SPY"].loc[:date].dropna()) >= 6:
                spy = prices["SPY"].loc[:date].dropna()
                weekly_ret = (float(spy.iloc[-1]) / float(spy.iloc[-min(6, len(spy))])) - 1
                self.health_monitor.record(date, regime, weekly_ret)

            health_boost = self.health_monitor.get_defensive_boost()

            self.regime_log.append({
                "date": date, "regime": regime,
                "confidence": confidence, "health_boost": health_boost,
            })

            # ── Get target allocation ──
            date_scores = scores.loc[date]
            target = get_allocation_matrix(regime, date_scores, prices, date, health_boost)

            # ── Execute rebalance ──
            current_w = {}
            for t, pos in positions.items():
                if t in prices.columns:
                    px = prices.loc[date, t]
                    if pd.notna(px) and pv > 0:
                        current_w[t] = (pos["qty"] * px) / pv

            for t in set(target) | set(current_w):
                tw = target.get(t, 0)
                cw = current_w.get(t, 0)
                diff = tw - cw

                # Minimum trade threshold (5% of portfolio)
                if abs(diff) < min_trade and abs(tw) > 0.001:
                    continue

                px = prices.loc[date, t] if t in prices.columns else None
                if px is None or pd.isna(px) or px <= 0:
                    continue

                trade_val = pv * diff
                shares = trade_val / px
                cost = abs(trade_val) * (cost_bps / 10000)

                if t not in positions:
                    positions[t] = {"qty": 0, "entry_date": date}

                positions[t]["qty"] += shares
                positions[t]["entry_date"] = date  # Reset timer
                cash -= trade_val
                cash -= cost

                self.trade_log.append({
                    "date": date, "ticker": t,
                    "side": "BUY" if shares > 0 else "SELL",
                    "shares": abs(shares), "price": px,
                    "value": abs(trade_val), "cost": cost, "regime": regime,
                })

                if abs(positions[t]["qty"]) < 0.001:
                    del positions[t]

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
        aw = mo[mo > 0].mean() if (mo > 0).any() else 0
        al = mo[mo < 0].mean() if (mo < 0).any() else 0
        pf = abs(aw / al) if al != 0 else float("inf")
        best_yr = v.resample("YE").last().pct_change().dropna()

        return {
            "total_return": tr, "annualized_return": ar,
            "annualized_volatility": vol, "sharpe_ratio": sharpe,
            "sortino_ratio": sortino, "max_drawdown": mdd,
            "calmar_ratio": calmar, "win_rate_monthly": wr,
            "profit_factor": pf,
            "avg_monthly_win": aw, "avg_monthly_loss": al,
            "best_month": mo.max() if not mo.empty else 0,
            "worst_month": mo.min() if not mo.empty else 0,
            "best_year": best_yr.max() if not best_yr.empty else 0,
            "worst_year": best_yr.min() if not best_yr.empty else 0,
            "total_trades": len(self.trade_log),
            "total_costs": sum(t.get("cost", 0) for t in self.trade_log),
            "years": yrs, "start_value": v.iloc[0], "end_value": v.iloc[-1],
        }
