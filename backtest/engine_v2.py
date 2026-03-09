"""
JARVIS V2 - Backtest Engine V2 (Regime-Aware)
================================================
Same as V1 but uses the regime-aware portfolio optimizer.
This should dramatically improve backtest results because:
- In CALM regime (~70% of the time): 50% satellite, Half-Kelly, equity tilt
- In CRISIS regime: 60% core, Eighth-Kelly, bond tilt
- Trend filter boosts equities when SPY > 200-day SMA
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta

from data.db import get_all_prices
from features.engine import compute_all_features
from signals.cross_momentum import compute_cross_sectional_momentum
from signals.trend_follow import compute_trend_following
from signals.mean_revert import compute_mean_reversion
from signals.vol_regime import compute_regime_signal
from signals.ensemble import combine_signals, DEFAULT_WEIGHTS
from portfolio.optimizer import optimize_portfolio, REGIME_PARAMS


DEFAULT_CONFIG = {
    "start_date": "2017-01-01",
    "end_date": None,
    "initial_capital": 100000,
    "rebalance_frequency": 21,
    "transaction_cost_bps": 5,
    "slippage_bps": 3,
}


class BacktestV2:
    """Regime-aware backtester."""

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.portfolio_history = []
        self.trade_log = []
        self.regime_history = []

    def run(self, prices: pd.DataFrame = None, signal_weights: dict = None) -> dict:
        if prices is None:
            prices = get_all_prices()
        if prices.empty:
            return {}
        if signal_weights is None:
            signal_weights = DEFAULT_WEIGHTS.copy()

        start = pd.Timestamp(self.config["start_date"])
        end = pd.Timestamp(self.config["end_date"]) if self.config["end_date"] else prices.index[-1]

        logger.info("=" * 60)
        logger.info("BACKTESTING JARVIS V2 (REGIME-AWARE)")
        logger.info(f"Period: {start.date()} → {end.date()}")
        logger.info(f"Capital: ${self.config['initial_capital']:,.0f}")
        logger.info("=" * 60)

        # Pre-compute signals
        logger.info("\nPre-computing signals...")
        features = compute_all_features(prices)
        price_features = features["price_features"]
        macro_features = features["macro_features"]

        try:
            sig_momentum = compute_cross_sectional_momentum(prices)
        except Exception:
            sig_momentum = pd.DataFrame()

        try:
            sig_trend = compute_trend_following(prices)
        except Exception:
            sig_trend = pd.DataFrame()

        try:
            sig_meanrev = compute_mean_reversion(price_features, macro_features, prices)
        except Exception:
            sig_meanrev = pd.DataFrame()

        try:
            regime_result = compute_regime_signal(macro_features, prices)
            sig_regime = regime_result["etf_signals"]
            regime_probs = regime_result["regime_probabilities"]
            regime_history = regime_result["regime_history"]
        except Exception:
            sig_regime = pd.DataFrame()
            regime_probs = pd.DataFrame()
            regime_history = pd.Series(dtype=int)

        signals = {
            "cross_momentum": sig_momentum,
            "trend_follow": sig_trend,
            "mean_reversion": sig_meanrev,
            "vol_regime": sig_regime,
        }

        alpha_scores = combine_signals(signals, signal_weights, prices)
        if alpha_scores.empty:
            return {}

        # Map regime states
        regime_names = {0: "CALM", 1: "TRANSITION", 2: "CRISIS"}

        # Simulate
        logger.info("\nRunning regime-aware simulation...")
        self._simulate(prices, alpha_scores, regime_history, regime_names, start, end)

        metrics = self._compute_metrics()

        # Count regime periods
        calm_days = sum(1 for r in self.regime_history if r.get("regime") == "CALM")
        trans_days = sum(1 for r in self.regime_history if r.get("regime") == "TRANSITION")
        crisis_days = sum(1 for r in self.regime_history if r.get("regime") == "CRISIS")
        total_rebal = len(self.regime_history) or 1

        logger.info(f"\nRegime breakdown: CALM {calm_days/total_rebal:.0%} | "
                     f"TRANSITION {trans_days/total_rebal:.0%} | CRISIS {crisis_days/total_rebal:.0%}")

        return {
            "portfolio_history": pd.DataFrame(self.portfolio_history),
            "trade_log": self.trade_log,
            "metrics": metrics,
            "regime_history": self.regime_history,
        }

    def _simulate(self, prices, alpha_scores, regime_history, regime_names, start, end):
        capital = self.config["initial_capital"]
        cash = capital
        positions = {}
        rebal_freq = self.config["rebalance_frequency"]
        cost_bps = self.config["transaction_cost_bps"] + self.config["slippage_bps"]

        bt_dates = prices.loc[start:end].index
        days_since_rebal = rebal_freq

        for i, date in enumerate(bt_dates):
            # Mark-to-market
            portfolio_value = cash
            for ticker, pos in positions.items():
                if ticker in prices.columns and date in prices.index:
                    px = prices.loc[date, ticker]
                    if pd.notna(px):
                        portfolio_value += pos["qty"] * px

            self.portfolio_history.append({
                "date": date,
                "total_value": portfolio_value,
                "cash": cash,
                "invested": portfolio_value - cash,
                "num_positions": len(positions),
            })

            days_since_rebal += 1
            if days_since_rebal < rebal_freq:
                continue
            if date not in alpha_scores.index:
                continue

            days_since_rebal = 0

            # Determine current regime
            if not regime_history.empty and date in regime_history.index:
                regime_code = regime_history.loc[date]
                regime = regime_names.get(regime_code, "CALM")
            else:
                regime = "CALM"

            self.regime_history.append({"date": date, "regime": regime})

            # Get alpha scores for this date
            scores = alpha_scores.loc[date].dropna().sort_values(ascending=False)
            if scores.empty:
                continue

            # Use regime-aware optimizer to get target weights
            params = REGIME_PARAMS.get(regime, REGIME_PARAMS["CALM"])
            target_weights = self._optimize_for_date(scores, prices, date, params)

            # Current weights
            current_weights = {}
            for ticker, pos in positions.items():
                if ticker in prices.columns:
                    px = prices.loc[date, ticker]
                    if pd.notna(px) and portfolio_value > 0:
                        current_weights[ticker] = (pos["qty"] * px) / portfolio_value

            # Execute rebalance
            all_tickers = set(target_weights.keys()) | set(current_weights.keys())
            for ticker in all_tickers:
                target_w = target_weights.get(ticker, 0)
                current_w = current_weights.get(ticker, 0)
                diff_w = target_w - current_w

                if abs(diff_w) < 0.005:
                    continue

                trade_value = portfolio_value * diff_w
                px = prices.loc[date, ticker] if ticker in prices.columns else None
                if px is None or pd.isna(px) or px <= 0:
                    continue

                shares = trade_value / px
                cost = abs(trade_value) * (cost_bps / 10000)

                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "entry_price": px}

                positions[ticker]["qty"] += shares
                cash -= trade_value
                cash -= cost

                self.trade_log.append({
                    "date": date, "ticker": ticker,
                    "side": "BUY" if shares > 0 else "SELL",
                    "shares": abs(shares), "price": px,
                    "value": abs(trade_value), "cost": cost,
                    "regime": regime,
                })

                if abs(positions[ticker]["qty"]) < 0.001:
                    del positions[ticker]

    def _optimize_for_date(self, scores, prices, date, params):
        """Simplified portfolio optimizer for backtest speed."""
        satellite_pct = params["satellite_pct"]
        core_pct = params["core_pct"]
        cash_pct = params["cash_pct"]
        kelly = params["kelly_multiplier"]
        max_pos = params["max_single_position"]
        equity_tilt = params["equity_tilt"]

        # Satellite: top alpha scores
        positive = scores[scores > 0]
        if not positive.empty:
            sat_weights = (positive / positive.sum()) * satellite_pct
            sat_weights = sat_weights.clip(upper=max_pos)
            if sat_weights.sum() > satellite_pct:
                sat_weights = sat_weights / sat_weights.sum() * satellite_pct
        else:
            sat_weights = pd.Series(dtype=float)

        # Core: simple inverse-vol across available groups
        from config.universe import get_asset_class_map
        asset_map = get_asset_class_map()

        # Quick risk parity: split core among asset classes
        ac_buckets = {}
        for ticker in prices.columns:
            ac = asset_map.get(ticker, "equity")
            if ac not in ac_buckets:
                ac_buckets[ac] = []
            ac_buckets[ac].append(ticker)

        core_weights = {}
        ac_weight = core_pct / max(len(ac_buckets), 1)

        for ac, tickers in ac_buckets.items():
            # Apply equity tilt
            if "equity" in ac:
                w = ac_weight + equity_tilt
            elif "fixed" in ac or "income" in ac:
                w = ac_weight - equity_tilt
            else:
                w = ac_weight

            per_etf = max(w, 0) / max(len(tickers), 1)
            for t in tickers:
                core_weights[t] = per_etf

        # Combine
        target = {}
        for t, w in core_weights.items():
            target[t] = w

        for t, w in sat_weights.items():
            target[t] = target.get(t, 0) + w

        # Apply Kelly
        for t in target:
            score = scores.get(t, 0)
            if score > 0:
                confidence = min(abs(score), 2.0) / 2.0
                factor = kelly + (1 - kelly) * confidence * 0.5
            else:
                factor = kelly * 0.5
            target[t] = target[t] * factor

        # Trend filter: SPY above 200-day SMA
        if "SPY" in prices.columns and date in prices.index:
            spy_prices = prices["SPY"].loc[:date].dropna()
            if len(spy_prices) >= 200:
                sma200 = spy_prices.rolling(200).mean().iloc[-1]
                spy_above = spy_prices.iloc[-1] > sma200

                if spy_above:
                    for t in target:
                        ac = asset_map.get(t, "equity")
                        if ac == "equity" and target[t] > 0:
                            target[t] *= 1.15
                        elif "fixed" in ac and target[t] > 0:
                            target[t] *= 0.85
                else:
                    for t in target:
                        ac = asset_map.get(t, "equity")
                        if ac == "equity" and target[t] > 0:
                            target[t] *= 0.80
                        elif "fixed" in ac and target[t] > 0:
                            target[t] *= 1.20

        # Normalize to investable
        max_invested = 1.0 - cash_pct
        total = sum(w for w in target.values() if w > 0)
        if total > max_invested:
            scale = max_invested / total
            target = {k: v * scale for k, v in target.items()}

        # Cap individual positions
        for t in target:
            target[t] = min(target[t], max_pos)

        # Remove tiny positions
        target = {t: w for t, w in target.items() if w > 0.003}

        return target

    def _compute_metrics(self) -> dict:
        if not self.portfolio_history:
            return {}

        df = pd.DataFrame(self.portfolio_history).set_index("date")
        values = df["total_value"]
        if len(values) < 2:
            return {}

        daily_returns = values.pct_change().dropna()
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        years = (values.index[-1] - values.index[0]).days / 365.25
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0

        downside = daily_returns[daily_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = (ann_return - 0.04) / downside_vol if downside_vol > 0 else 0

        cummax = values.cummax()
        max_dd = ((values / cummax) - 1).min()
        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0

        monthly = values.resample("ME").last().pct_change().dropna()
        win_rate = (monthly > 0).mean()
        avg_win = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
        avg_loss = monthly[monthly < 0].mean() if (monthly < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        total_trades = len(self.trade_log)
        total_costs = sum(t.get("cost", 0) for t in self.trade_log)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_monthly_win": avg_win,
            "avg_monthly_loss": avg_loss,
            "best_month": monthly.max() if not monthly.empty else 0,
            "worst_month": monthly.min() if not monthly.empty else 0,
            "total_trades": total_trades,
            "total_costs": total_costs,
            "years": years,
            "start_value": values.iloc[0],
            "end_value": values.iloc[-1],
        }
