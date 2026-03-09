"""
JARVIS V2 - Performance Tracker
==================================
Tracks live (or paper) trading performance and computes key metrics
that tell you whether Jarvis is actually making money.

HOW THIS WORKS (for non-coders):
- Every day after the market closes, this module:
  1. Records the portfolio value
  2. Computes running Sharpe ratio, drawdown, win rate
  3. Compares Jarvis to benchmarks (SPY, 60/40 portfolio)
  4. Flags warnings if performance is degrading
  5. Stores everything in the database for historical analysis

THE KEY METRICS TO WATCH:
- Sharpe Ratio > 0.5 → system has an edge (good)
- Sharpe Ratio > 1.0 → strong performance (very good)
- Max Drawdown < -10% → acceptable pain (OK)
- Max Drawdown > -15% → circuit breaker territory (bad)
- Win Rate > 55% → more winning months than losing (good)
- Alpha vs SPY > 0% → beating the market (the whole point)
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from sqlalchemy import text

from data.db import engine as db_engine


class PerformanceTracker:
    """
    Tracks and computes live trading performance metrics.
    """

    def __init__(self):
        self._ensure_tables()

    def _ensure_tables(self):
        """Create performance tracking tables if they don't exist."""
        try:
            with db_engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_daily (
                        date DATE PRIMARY KEY,
                        portfolio_value REAL,
                        cash REAL,
                        invested REAL,
                        num_positions INTEGER,
                        daily_return REAL,
                        cumulative_return REAL,
                        drawdown REAL,
                        spy_value REAL,
                        spy_daily_return REAL,
                        spy_cumulative_return REAL,
                        rolling_sharpe_90d REAL,
                        rolling_vol_21d REAL,
                        regime TEXT,
                        signals_active INTEGER
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_monthly (
                        month TEXT PRIMARY KEY,
                        start_value REAL,
                        end_value REAL,
                        monthly_return REAL,
                        spy_monthly_return REAL,
                        alpha REAL,
                        sharpe_ytd REAL,
                        max_drawdown_ytd REAL,
                        win_rate_ytd REAL,
                        num_trades INTEGER,
                        best_etf TEXT,
                        worst_etf TEXT
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS signal_performance (
                        month TEXT,
                        signal_name TEXT,
                        ic REAL,
                        weight REAL,
                        contribution REAL,
                        PRIMARY KEY (month, signal_name)
                    )
                """))
            logger.info("Performance tracking tables ready")
        except Exception as e:
            logger.error(f"Failed to create performance tables: {e}")

    def record_daily(
        self,
        portfolio_value: float,
        cash: float,
        num_positions: int,
        regime: str = "UNKNOWN",
        signals_active: int = 4,
    ):
        """
        Record today's portfolio snapshot and compute daily metrics.
        Call this at the end of each trading day.
        """
        today = datetime.now().date()

        try:
            # Get yesterday's value
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT portfolio_value, cumulative_return FROM performance_daily "
                    "ORDER BY date DESC LIMIT 1"
                ))
                prev = result.fetchone()

            if prev:
                prev_value = prev[0]
                prev_cum = prev[1]
                daily_return = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0
            else:
                daily_return = 0
                prev_cum = 0

            # Cumulative return from inception
            initial = self._get_initial_value()
            if initial and initial > 0:
                cumulative_return = (portfolio_value / initial) - 1
            else:
                cumulative_return = 0

            # Drawdown from peak
            peak = self._get_peak_value()
            drawdown = (portfolio_value / peak) - 1 if peak > 0 else 0

            # SPY benchmark
            spy_daily, spy_cum = self._get_spy_returns(today)

            # Rolling Sharpe (90-day)
            rolling_sharpe = self._compute_rolling_sharpe(90)

            # Rolling volatility (21-day)
            rolling_vol = self._compute_rolling_vol(21)

            # Save to database
            invested = portfolio_value - cash
            with db_engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO performance_daily
                    (date, portfolio_value, cash, invested, num_positions,
                     daily_return, cumulative_return, drawdown,
                     spy_value, spy_daily_return, spy_cumulative_return,
                     rolling_sharpe_90d, rolling_vol_21d, regime, signals_active)
                    VALUES (:date, :pv, :cash, :inv, :npos,
                            :dr, :cr, :dd,
                            :spy_v, :spy_dr, :spy_cr,
                            :sharpe, :vol, :regime, :sig)
                    ON CONFLICT (date) DO UPDATE SET
                        portfolio_value = EXCLUDED.portfolio_value,
                        cash = EXCLUDED.cash,
                        invested = EXCLUDED.invested,
                        num_positions = EXCLUDED.num_positions,
                        daily_return = EXCLUDED.daily_return,
                        cumulative_return = EXCLUDED.cumulative_return,
                        drawdown = EXCLUDED.drawdown,
                        rolling_sharpe_90d = EXCLUDED.rolling_sharpe_90d,
                        rolling_vol_21d = EXCLUDED.rolling_vol_21d,
                        regime = EXCLUDED.regime,
                        signals_active = EXCLUDED.signals_active
                """), {
                    "date": today, "pv": portfolio_value, "cash": cash,
                    "inv": invested, "npos": num_positions,
                    "dr": daily_return, "cr": cumulative_return, "dd": drawdown,
                    "spy_v": None, "spy_dr": spy_daily, "spy_cr": spy_cum,
                    "sharpe": rolling_sharpe, "vol": rolling_vol,
                    "regime": regime, "sig": signals_active,
                })

            logger.info(f"Performance recorded: ${portfolio_value:,.2f} | "
                       f"Day: {daily_return:+.2%} | Cum: {cumulative_return:+.2%} | "
                       f"DD: {drawdown:.2%} | Sharpe(90d): {rolling_sharpe:.2f}")

        except Exception as e:
            logger.error(f"Failed to record daily performance: {e}")

    def record_monthly_summary(self):
        """
        Compute and store monthly summary. Call at end of each month.
        """
        try:
            month_str = datetime.now().strftime("%Y-%m")

            with db_engine.begin() as conn:
                # Get this month's daily records
                result = conn.execute(text(
                    "SELECT date, portfolio_value, daily_return, spy_daily_return "
                    "FROM performance_daily "
                    "WHERE TO_CHAR(date, 'YYYY-MM') = :month "
                    "ORDER BY date"
                ), {"month": month_str})
                rows = result.fetchall()

            if not rows:
                return

            start_value = rows[0][1]
            end_value = rows[-1][1]
            monthly_return = (end_value / start_value) - 1 if start_value > 0 else 0

            spy_returns = [r[3] for r in rows if r[3] is not None]
            spy_monthly = sum(spy_returns) if spy_returns else 0

            alpha = monthly_return - spy_monthly

            # YTD metrics
            ytd_metrics = self.get_ytd_metrics()

            with db_engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO performance_monthly
                    (month, start_value, end_value, monthly_return,
                     spy_monthly_return, alpha, sharpe_ytd, max_drawdown_ytd,
                     win_rate_ytd, num_trades, best_etf, worst_etf)
                    VALUES (:month, :sv, :ev, :mr, :spy_mr, :alpha,
                            :sharpe, :mdd, :wr, :trades, :best, :worst)
                    ON CONFLICT (month) DO UPDATE SET
                        end_value = EXCLUDED.end_value,
                        monthly_return = EXCLUDED.monthly_return,
                        alpha = EXCLUDED.alpha,
                        sharpe_ytd = EXCLUDED.sharpe_ytd,
                        max_drawdown_ytd = EXCLUDED.max_drawdown_ytd
                """), {
                    "month": month_str, "sv": start_value, "ev": end_value,
                    "mr": monthly_return, "spy_mr": spy_monthly, "alpha": alpha,
                    "sharpe": ytd_metrics.get("sharpe", 0),
                    "mdd": ytd_metrics.get("max_drawdown", 0),
                    "wr": ytd_metrics.get("win_rate", 0),
                    "trades": 0, "best": None, "worst": None,
                })

            logger.info(f"Monthly summary ({month_str}): Return={monthly_return:+.2%} | "
                       f"SPY={spy_monthly:+.2%} | Alpha={alpha:+.2%}")

        except Exception as e:
            logger.error(f"Failed to record monthly summary: {e}")

    def get_current_metrics(self) -> dict:
        """Get the latest performance metrics for dashboard display."""
        try:
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT * FROM performance_daily ORDER BY date DESC LIMIT 1"
                ))
                latest = result.fetchone()

            if not latest:
                return {"status": "No performance data yet"}

            return {
                "date": str(latest[0]),
                "portfolio_value": latest[1],
                "cash": latest[2],
                "invested": latest[3],
                "num_positions": latest[4],
                "daily_return": latest[5],
                "cumulative_return": latest[6],
                "drawdown": latest[7],
                "rolling_sharpe_90d": latest[11],
                "rolling_vol_21d": latest[12],
                "regime": latest[13],
            }
        except Exception:
            return {}

    def get_ytd_metrics(self) -> dict:
        """Compute year-to-date performance metrics."""
        try:
            year_start = f"{datetime.now().year}-01-01"
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT portfolio_value, daily_return FROM performance_daily "
                    "WHERE date >= :start ORDER BY date"
                ), {"start": year_start})
                rows = result.fetchall()

            if not rows:
                return {}

            values = [r[0] for r in rows]
            returns = [r[1] for r in rows if r[1] is not None]

            ytd_return = (values[-1] / values[0]) - 1 if values[0] > 0 else 0

            if returns:
                vol = np.std(returns) * np.sqrt(252)
                sharpe = (np.mean(returns) * 252 - 0.04) / vol if vol > 0 else 0
            else:
                vol, sharpe = 0, 0

            peak = max(values)
            max_dd = min((v / peak) - 1 for v in values) if peak > 0 else 0

            # Monthly win rate
            monthly_perf = pd.DataFrame({"date": [r[0] for r in rows], "value": values})

            return {
                "ytd_return": ytd_return,
                "sharpe": sharpe,
                "volatility": vol,
                "max_drawdown": max_dd,
                "win_rate": 0,
                "trading_days": len(values),
            }
        except Exception:
            return {}

    def get_performance_history(self, days: int = 90) -> pd.DataFrame:
        """Get recent performance history as a DataFrame."""
        try:
            with db_engine.begin() as conn:
                df = pd.read_sql(text(
                    "SELECT * FROM performance_daily "
                    "ORDER BY date DESC LIMIT :n"
                ), conn, params={"n": days})
            return df.sort_values("date")
        except Exception:
            return pd.DataFrame()

    def check_health_warnings(self) -> list:
        """Check for performance degradation and return warnings."""
        warnings = []
        metrics = self.get_current_metrics()

        if not metrics:
            return ["No performance data available"]

        sharpe = metrics.get("rolling_sharpe_90d", 0)
        dd = metrics.get("drawdown", 0)
        cum_return = metrics.get("cumulative_return", 0)

        if sharpe is not None and sharpe < 0:
            warnings.append(f"⚠️ Sharpe ratio is NEGATIVE ({sharpe:.2f}) — system may not have an edge")
        elif sharpe is not None and sharpe < 0.3:
            warnings.append(f"⚠️ Sharpe ratio is low ({sharpe:.2f}) — weak risk-adjusted returns")

        if dd is not None and dd < -0.10:
            warnings.append(f"🔴 Drawdown is {dd:.1%} — approaching circuit breaker level")
        if dd is not None and dd < -0.15:
            warnings.append(f"🚨 CIRCUIT BREAKER: Drawdown exceeds -15%!")

        if cum_return is not None and cum_return < -0.05:
            warnings.append(f"⚠️ Cumulative return is {cum_return:.1%} — system is losing money")

        return warnings

    def _get_initial_value(self) -> float:
        try:
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT portfolio_value FROM performance_daily ORDER BY date ASC LIMIT 1"
                ))
                row = result.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def _get_peak_value(self) -> float:
        try:
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT MAX(portfolio_value) FROM performance_daily"
                ))
                row = result.fetchone()
            return row[0] if row and row[0] else 0
        except Exception:
            return 0

    def _get_spy_returns(self, date):
        try:
            from data.db import get_prices
            spy = get_prices("SPY")
            if spy.empty:
                return 0, 0
            spy_close = spy["adj_close"]
            if len(spy_close) >= 2:
                daily = (spy_close.iloc[-1] / spy_close.iloc[-2]) - 1
                # Cumulative from 252 days ago
                if len(spy_close) >= 252:
                    cum = (spy_close.iloc[-1] / spy_close.iloc[-252]) - 1
                else:
                    cum = (spy_close.iloc[-1] / spy_close.iloc[0]) - 1
                return daily, cum
        except Exception:
            pass
        return 0, 0

    def _compute_rolling_sharpe(self, window: int) -> float:
        try:
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT daily_return FROM performance_daily "
                    "ORDER BY date DESC LIMIT :n"
                ), {"n": window})
                rows = result.fetchall()

            if len(rows) < 20:
                return 0

            returns = [r[0] for r in rows if r[0] is not None]
            if not returns:
                return 0

            mean = np.mean(returns) * 252
            vol = np.std(returns) * np.sqrt(252)
            return (mean - 0.04) / vol if vol > 0 else 0
        except Exception:
            return 0

    def _compute_rolling_vol(self, window: int) -> float:
        try:
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT daily_return FROM performance_daily "
                    "ORDER BY date DESC LIMIT :n"
                ), {"n": window})
                rows = result.fetchall()

            returns = [r[0] for r in rows if r[0] is not None]
            if len(returns) < 5:
                return 0

            return float(np.std(returns) * np.sqrt(252))
        except Exception:
            return 0
