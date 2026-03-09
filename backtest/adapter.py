"""
JARVIS V2 - Signal Weight Adapter
====================================
Makes Jarvis LEARN which signals work best over time.

HOW THIS WORKS (for non-coders):
- Every month, we check: "Which signals were RIGHT last month?"
- We compute the "Information Coefficient" (IC) for each signal
- IC measures: did the signal's ranking of ETFs match actual returns?
- IC of +0.10 = good predictive power, +0.20 = excellent, 0 = useless
- Signals with higher IC get MORE weight next month
- Signals with low/negative IC get LESS weight

THE SAFETY RAILS:
- No signal can go below 5% weight (never fully abandon a signal)
- No signal can go above 40% weight (never bet everything on one signal)
- Changes happen slowly (exponential moving average with 6-month halflife)
- We store the FULL history of IC scores so you can audit what happened

WHY THIS WORKS:
- Markets change — momentum works great in bull markets, fails in crashes
- Mean reversion works great after crashes, fails during trends
- By adapting weights, Jarvis automatically emphasizes the right signal
  for the current environment WITHOUT you having to intervene
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
from scipy.stats import spearmanr
from sqlalchemy import text

from data.db import engine as db_engine


class SignalWeightAdapter:
    """
    Adapts signal weights monthly based on realized performance.
    """

    def __init__(self):
        self.min_weight = 0.05   # 5% floor
        self.max_weight = 0.40   # 40% ceiling
        self.halflife = 6        # 6-month adaptation speed
        self._ensure_tables()

    def _ensure_tables(self):
        """Create weight tracking tables."""
        try:
            with db_engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS signal_weights (
                        date DATE,
                        signal_name TEXT,
                        weight REAL,
                        ic REAL,
                        ic_3m REAL,
                        ic_6m REAL,
                        PRIMARY KEY (date, signal_name)
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS weight_adaptation_log (
                        date DATE,
                        old_weights TEXT,
                        new_weights TEXT,
                        ic_scores TEXT,
                        reason TEXT,
                        PRIMARY KEY (date)
                    )
                """))
        except Exception as e:
            logger.warning(f"Could not create weight tables: {e}")

    def compute_signal_ics(
        self,
        signals: dict,
        prices: pd.DataFrame,
        lookback_days: int = 21,
    ) -> dict:
        """
        Compute the Information Coefficient (IC) for each signal over
        the most recent lookback period.

        IC = Spearman rank correlation between:
            - Signal's prediction (score at start of period)
            - Actual realized return (over the period)

        Higher IC = more accurate signal.
        """
        if prices.empty:
            return {}

        # Realized returns over the lookback period
        if len(prices) < lookback_days + 5:
            return {}

        # Signal scores from the start of the evaluation period
        eval_start_idx = -(lookback_days + 1)
        realized_returns = prices.iloc[-1] / prices.iloc[eval_start_idx] - 1

        ics = {}

        for signal_name, signal_df in signals.items():
            if signal_df.empty:
                ics[signal_name] = {"ic": 0.0, "p_value": 1.0, "n_etfs": 0}
                continue

            try:
                # Get signal scores from the start of eval period
                eval_date = prices.index[eval_start_idx]

                # Find closest available signal date
                available_dates = signal_df.index[signal_df.index <= eval_date]
                if available_dates.empty:
                    ics[signal_name] = {"ic": 0.0, "p_value": 1.0, "n_etfs": 0}
                    continue

                signal_date = available_dates[-1]
                signal_scores = signal_df.loc[signal_date]

                # Align tickers
                valid = signal_scores.dropna().index.intersection(realized_returns.dropna().index)
                if len(valid) < 5:
                    ics[signal_name] = {"ic": 0.0, "p_value": 1.0, "n_etfs": len(valid)}
                    continue

                ic, p_value = spearmanr(
                    signal_scores[valid].values,
                    realized_returns[valid].values
                )

                if np.isnan(ic):
                    ic = 0.0

                ics[signal_name] = {
                    "ic": float(ic),
                    "p_value": float(p_value),
                    "n_etfs": len(valid),
                }

                logger.info(f"  {signal_name}: IC={ic:+.4f} (p={p_value:.3f}, n={len(valid)})")

            except Exception as e:
                logger.error(f"  {signal_name}: IC computation failed - {e}")
                ics[signal_name] = {"ic": 0.0, "p_value": 1.0, "n_etfs": 0}

        return ics

    def adapt_weights(
        self,
        current_weights: dict,
        signals: dict,
        prices: pd.DataFrame,
    ) -> dict:
        """
        Update signal weights based on recent IC performance.

        This is the CORE learning function. Call monthly.

        Parameters:
            current_weights: dict {signal_name: weight}
            signals: dict {signal_name: DataFrame} from ensemble
            prices: price DataFrame

        Returns:
            Updated weights dict
        """
        logger.info("=" * 50)
        logger.info("SIGNAL WEIGHT ADAPTATION")
        logger.info("=" * 50)

        # Compute IC over multiple lookback periods
        logger.info("\nComputing 1-month IC:")
        ic_1m = self.compute_signal_ics(signals, prices, lookback_days=21)

        logger.info("\nComputing 3-month IC:")
        ic_3m = self.compute_signal_ics(signals, prices, lookback_days=63)

        logger.info("\nComputing 6-month IC:")
        ic_6m = self.compute_signal_ics(signals, prices, lookback_days=126)

        # Composite IC: weighted average (recent matters more)
        # 50% weight on 1-month, 30% on 3-month, 20% on 6-month
        composite_ic = {}
        for name in current_weights:
            ic1 = ic_1m.get(name, {}).get("ic", 0)
            ic3 = ic_3m.get(name, {}).get("ic", 0)
            ic6 = ic_6m.get(name, {}).get("ic", 0)
            composite_ic[name] = 0.5 * ic1 + 0.3 * ic3 + 0.2 * ic6

        logger.info(f"\nComposite IC scores:")
        for name, ic in sorted(composite_ic.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * max(0, int(ic * 50)) if ic > 0 else "░" * max(0, int(abs(ic) * 50))
            logger.info(f"  {name:20s}: {ic:+.4f}  {bar}")

        # Adapt weights using EMA
        alpha = 2.0 / (self.halflife + 1)
        base_weight = 1.0 / len(current_weights)

        new_weights = {}
        for name, old_w in current_weights.items():
            ic = composite_ic.get(name, 0)

            # Only use positive IC for weight increase
            # Negative IC means the signal is WRONG — reduce but don't eliminate
            ic_adjustment = max(ic, 0) * 0.5  # Scale IC contribution

            target = base_weight + ic_adjustment
            new_w = alpha * target + (1 - alpha) * old_w

            # Apply floor and ceiling
            new_weights[name] = np.clip(new_w, self.min_weight, self.max_weight)

        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        # Log the changes
        logger.info(f"\nWeight changes:")
        for name in current_weights:
            old = current_weights[name]
            new = new_weights[name]
            change = new - old
            arrow = "↑" if change > 0.005 else "↓" if change < -0.005 else "→"
            logger.info(f"  {name:20s}: {old:.3f} {arrow} {new:.3f} ({change:+.3f})")

        # Save to database
        self._save_weights(new_weights, ic_1m, ic_3m, ic_6m, current_weights)

        return new_weights

    def get_current_weights(self) -> dict:
        """Load the most recent signal weights from the database."""
        try:
            with db_engine.begin() as conn:
                result = conn.execute(text(
                    "SELECT signal_name, weight FROM signal_weights "
                    "WHERE date = (SELECT MAX(date) FROM signal_weights)"
                ))
                rows = result.fetchall()

            if rows:
                return {r[0]: r[1] for r in rows}
        except Exception:
            pass

        # Return defaults if nothing saved
        from signals.ensemble import DEFAULT_WEIGHTS
        return DEFAULT_WEIGHTS.copy()

    def get_weight_history(self) -> pd.DataFrame:
        """Get the full history of weight changes."""
        try:
            with db_engine.begin() as conn:
                df = pd.read_sql(text(
                    "SELECT * FROM signal_weights ORDER BY date, signal_name"
                ), conn)
            return df
        except Exception:
            return pd.DataFrame()

    def get_ic_history(self) -> pd.DataFrame:
        """Get IC scores over time for each signal."""
        try:
            with db_engine.begin() as conn:
                df = pd.read_sql(text(
                    "SELECT date, signal_name, ic, ic_3m, ic_6m "
                    "FROM signal_weights ORDER BY date"
                ), conn)
            return df
        except Exception:
            return pd.DataFrame()

    def _save_weights(self, weights, ic_1m, ic_3m, ic_6m, old_weights):
        """Save weights and IC scores to database."""
        today = datetime.now().date()
        try:
            with db_engine.begin() as conn:
                for name, weight in weights.items():
                    conn.execute(text("""
                        INSERT INTO signal_weights (date, signal_name, weight, ic, ic_3m, ic_6m)
                        VALUES (:date, :name, :weight, :ic, :ic3, :ic6)
                        ON CONFLICT (date, signal_name) DO UPDATE SET
                            weight = EXCLUDED.weight, ic = EXCLUDED.ic,
                            ic_3m = EXCLUDED.ic_3m, ic_6m = EXCLUDED.ic_6m
                    """), {
                        "date": today, "name": name, "weight": weight,
                        "ic": ic_1m.get(name, {}).get("ic", 0),
                        "ic3": ic_3m.get(name, {}).get("ic", 0),
                        "ic6": ic_6m.get(name, {}).get("ic", 0),
                    })

                # Save adaptation log
                import json
                conn.execute(text("""
                    INSERT INTO weight_adaptation_log (date, old_weights, new_weights, ic_scores, reason)
                    VALUES (:date, :old, :new, :ics, :reason)
                    ON CONFLICT (date) DO UPDATE SET
                        new_weights = EXCLUDED.new_weights,
                        ic_scores = EXCLUDED.ic_scores
                """), {
                    "date": today,
                    "old": json.dumps({k: round(v, 4) for k, v in old_weights.items()}),
                    "new": json.dumps({k: round(v, 4) for k, v in weights.items()}),
                    "ics": json.dumps({k: round(v.get("ic", 0), 4) for k, v in ic_1m.items()}),
                    "reason": "Monthly IC-based adaptation",
                })

            logger.info("Weights saved to database")
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
