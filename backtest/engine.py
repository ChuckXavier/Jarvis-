"""
JARVIS V2 - Backtesting Engine
=================================
Replays the ENTIRE signal pipeline against historical data to answer:
"If Jarvis had been running for the last 10 years, what would it have earned?"

HOW THIS WORKS (for non-coders):
- We "pretend" today is January 2017, and run Jarvis's signals
- Jarvis picks its portfolio for that month (e.g., buy SPY, GLD, sell TLT...)
- We then check what ACTUALLY happened in February 2017
- Repeat: March 2017, April 2017... all the way to today
- At the end, we can see: total return, Sharpe ratio, max drawdown, win rate

WHY THIS MATTERS:
- Paper trading only gives you a few months of data
- Backtesting gives you 8+ YEARS across bull markets, bear markets,
  COVID crash, 2022 rate hikes, and everything in between
- If Jarvis can't make money in a backtest, it definitely can't make money live

IMPORTANT LIMITATIONS:
- Backtests are OPTIMISTIC — real trading has slippage, fees, and timing issues
- We account for this by adding transaction cost estimates
- A good rule: expect live performance to be 60-70% of backtest performance
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


# Default backtest parameters
DEFAULT_CONFIG = {
    "start_date": "2017-01-01",
    "end_date": None,           # None = use latest available
    "initial_capital": 100000,
    "rebalance_frequency": 21,  # Trading days (monthly)
    "max_positions": 15,
    "max_single_weight": 0.10,
    "cash_reserve": 0.05,
    "transaction_cost_bps": 5,  # 5 basis points per trade (conservative)
    "slippage_bps": 3,          # 3 bps slippage estimate
}


class Backtest:
    """
    Runs a historical simulation of the Jarvis trading system.
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.prices = None
        self.signals = {}
        self.weights_history = []
        self.portfolio_history = []
        self.trade_log = []
        self.monthly_returns = []

    def run(self, prices: pd.DataFrame = None, signal_weights: dict = None) -> dict:
        """
        Run the full backtest.

        Returns a dict with:
            "portfolio_history": DataFrame of daily portfolio values
            "monthly_returns": Series of monthly returns
            "trade_log": list of all trades executed
            "metrics": dict of performance metrics (Sharpe, drawdown, etc.)
            "weights_history": list of monthly weight snapshots
        """
        if prices is None:
            prices = get_all_prices()

        if prices.empty:
            logger.error("No price data for backtest")
            return {}

        self.prices = prices

        if signal_weights is None:
            signal_weights = DEFAULT_WEIGHTS.copy()

        start = pd.Timestamp(self.config["start_date"])
        end = pd.Timestamp(self.config["end_date"]) if self.config["end_date"] else prices.index[-1]

        # Filter prices to backtest window (but keep prior data for feature computation)
        min_history = 252  # Need 1 year prior for features
        history_start_idx = max(0, prices.index.get_indexer([start], method="nearest")[0] - min_history)

        logger.info("=" * 60)
        logger.info("BACKTESTING JARVIS V2")
        logger.info(f"Period: {start.date()} → {end.date()}")
        logger.info(f"Capital: ${self.config['initial_capital']:,.0f}")
        logger.info(f"Rebalance: Every {self.config['rebalance_frequency']} trading days")
        logger.info(f"Transaction costs: {self.config['transaction_cost_bps']} bps")
        logger.info("=" * 60)

        # ── Pre-compute ALL signals for the full period ──
        logger.info("\nStep 1: Pre-computing signals for full history...")
        features = compute_all_features(prices)
        price_features = features["price_features"]
        macro_features = features["macro_features"]

        try:
            sig_momentum = compute_cross_sectional_momentum(prices)
            logger.info(f"  Signal 1 (Momentum): {len(sig_momentum)} days")
        except Exception as e:
            logger.error(f"  Signal 1 failed: {e}")
            sig_momentum = pd.DataFrame()

        try:
            sig_trend = compute_trend_following(prices)
            logger.info(f"  Signal 2 (Trend): {len(sig_trend)} days")
        except Exception as e:
            logger.error(f"  Signal 2 failed: {e}")
            sig_trend = pd.DataFrame()

        try:
            sig_meanrev = compute_mean_reversion(price_features, macro_features, prices)
            logger.info(f"  Signal 3 (Mean Reversion): {len(sig_meanrev)} days")
        except Exception as e:
            logger.error(f"  Signal 3 failed: {e}")
            sig_meanrev = pd.DataFrame()

        try:
            regime_result = compute_regime_signal(macro_features, prices)
            sig_regime = regime_result["etf_signals"]
            logger.info(f"  Signal 4 (Regime): {len(sig_regime)} days")
        except Exception as e:
            logger.error(f"  Signal 4 failed: {e}")
            sig_regime = pd.DataFrame()

        self.signals = {
            "cross_momentum": sig_momentum,
            "trend_follow": sig_trend,
            "mean_reversion": sig_meanrev,
            "vol_regime": sig_regime,
        }

        # ── Combine into ensemble alpha scores ──
        logger.info("\nStep 2: Computing ensemble alpha scores...")
        alpha_scores = combine_signals(self.signals, signal_weights, prices)

        if alpha_scores.empty:
            logger.error("No alpha scores generated — cannot backtest")
            return {}

        # ── Run the simulation ──
        logger.info(f"\nStep 3: Running simulation...")
        self._simulate(prices, alpha_scores, start, end)

        # ── Compute metrics ──
        logger.info("\nStep 4: Computing performance metrics...")
        metrics = self._compute_metrics()

        return {
            "portfolio_history": pd.DataFrame(self.portfolio_history),
            "monthly_returns": pd.Series(self.monthly_returns),
            "trade_log": self.trade_log,
            "metrics": metrics,
            "weights_history": self.weights_history,
        }

    def _simulate(self, prices, alpha_scores, start, end):
        """Run the day-by-day simulation."""
        capital = self.config["initial_capital"]
        cash = capital
        positions = {}  # {ticker: {"qty": X, "entry_price": Y}}
        rebal_freq = self.config["rebalance_frequency"]
        cost_bps = self.config["transaction_cost_bps"] + self.config["slippage_bps"]

        # Get trading days in our window
        bt_dates = prices.loc[start:end].index
        days_since_rebal = rebal_freq  # Force rebalance on first day

        for i, date in enumerate(bt_dates):
            # ── Daily mark-to-market ──
            portfolio_value = cash
            for ticker, pos in positions.items():
                if ticker in prices.columns and date in prices.index:
                    current_price = prices.loc[date, ticker]
                    if pd.notna(current_price):
                        portfolio_value += pos["qty"] * current_price

            self.portfolio_history.append({
                "date": date,
                "total_value": portfolio_value,
                "cash": cash,
                "invested": portfolio_value - cash,
                "num_positions": len(positions),
            })

            # ── Rebalance check ──
            days_since_rebal += 1
            if days_since_rebal < rebal_freq:
                continue

            # Skip if we don't have alpha scores for this date
            if date not in alpha_scores.index:
                continue

            days_since_rebal = 0

            # ── Get target weights ──
            scores = alpha_scores.loc[date].dropna().sort_values(ascending=False)
            if scores.empty:
                continue

            target_weights = self._scores_to_weights(scores)

            # ── Execute rebalance ──
            # First: compute current weights
            current_weights = {}
            for ticker, pos in positions.items():
                if ticker in prices.columns:
                    px = prices.loc[date, ticker]
                    if pd.notna(px) and portfolio_value > 0:
                        current_weights[ticker] = (pos["qty"] * px) / portfolio_value

            # Determine trades needed
            all_tickers = set(target_weights.keys()) | set(current_weights.keys())

            for ticker in all_tickers:
                target_w = target_weights.get(ticker, 0)
                current_w = current_weights.get(ticker, 0)
                diff_w = target_w - current_w

                if abs(diff_w) < 0.01:  # Skip tiny changes
                    continue

                trade_value = portfolio_value * diff_w
                px = prices.loc[date, ticker] if ticker in prices.columns else None

                if px is None or pd.isna(px) or px <= 0:
                    continue

                shares = trade_value / px
                cost = abs(trade_value) * (cost_bps / 10000)

                # Execute
                if ticker not in positions:
                    positions[ticker] = {"qty": 0, "entry_price": px}

                positions[ticker]["qty"] += shares
                cash -= trade_value
                cash -= cost  # Transaction costs

                self.trade_log.append({
                    "date": date,
                    "ticker": ticker,
                    "side": "BUY" if shares > 0 else "SELL",
                    "shares": abs(shares),
                    "price": px,
                    "value": abs(trade_value),
                    "cost": cost,
                })

                # Remove zero/near-zero positions
                if abs(positions[ticker]["qty"]) < 0.001:
                    del positions[ticker]

            # Record monthly snapshot
            self.weights_history.append({
                "date": date,
                "weights": target_weights.copy(),
                "portfolio_value": portfolio_value,
            })

    def _scores_to_weights(self, scores: pd.Series) -> dict:
        """Convert alpha scores to portfolio weights."""
        max_pos = self.config["max_positions"]
        max_weight = self.config["max_single_weight"]
        cash_reserve = self.config["cash_reserve"]
        investable = 1.0 - cash_reserve

        # Take only positive scores (long-only)
        positive = scores[scores > 0].head(max_pos)

        if positive.empty:
            return {}

        # Weight proportional to score
        raw_weights = positive / positive.sum() * investable

        # Cap individual weights
        capped = raw_weights.clip(upper=max_weight)

        # Re-normalize
        if capped.sum() > investable:
            capped = capped / capped.sum() * investable

        return capped.to_dict()

    def _compute_metrics(self) -> dict:
        """Compute comprehensive performance metrics."""
        if not self.portfolio_history:
            return {}

        df = pd.DataFrame(self.portfolio_history).set_index("date")
        values = df["total_value"]

        if len(values) < 2:
            return {}

        # Daily returns
        daily_returns = values.pct_change().dropna()

        # Total return
        total_return = (values.iloc[-1] / values.iloc[0]) - 1

        # Annualized return
        years = (values.index[-1] - values.index[0]).days / 365.25
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        ann_vol = daily_returns.std() * np.sqrt(252)

        # Sharpe Ratio (assuming 4% risk-free rate)
        risk_free = 0.04
        sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0

        # Sortino Ratio (only downside volatility)
        downside = daily_returns[daily_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = (ann_return - risk_free) / downside_vol if downside_vol > 0 else 0

        # Max Drawdown
        cummax = values.cummax()
        drawdowns = (values / cummax) - 1
        max_dd = drawdowns.min()

        # Calmar Ratio (ann return / max drawdown)
        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0

        # Win Rate (% of months with positive return)
        monthly = values.resample("ME").last().pct_change().dropna()
        win_rate = (monthly > 0).mean()
        avg_win = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
        avg_loss = monthly[monthly < 0].mean() if (monthly < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Best/Worst months
        best_month = monthly.max()
        worst_month = monthly.min()

        # Total trades and costs
        total_trades = len(self.trade_log)
        total_costs = sum(t.get("cost", 0) for t in self.trade_log)

        # Monthly returns for output
        self.monthly_returns = monthly.tolist()

        metrics = {
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
            "best_month": best_month,
            "worst_month": worst_month,
            "total_trades": total_trades,
            "total_costs": total_costs,
            "years": years,
            "start_value": values.iloc[0],
            "end_value": values.iloc[-1],
        }

        return metrics


def run_quick_backtest(years: int = 5) -> dict:
    """Convenience function: run a quick backtest over N years."""
    start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    bt = Backtest({"start_date": start})
    return bt.run()
