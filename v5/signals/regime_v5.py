"""
JARVIS V5 - Regime Detection Module
======================================
5-regime classification with 8 voting signals and 2-week confirmation.
Includes Regime Detector Health Monitor (meta-signal from Manus addendum).

REGIMES:
  EUPHORIA:   VIX < 13, strong trends, max leverage
  CALM:       VIX 13-18, normal bull market
  CAUTION:    VIX 18-25, uncertainty rising
  STRESS:     VIX 25-35, bear market conditions
  CRISIS:     VIX > 35, crash/panic mode
"""

import pandas as pd
import numpy as np
from loguru import logger


def detect_regime_for_date(prices, date, vix_series, credit_spread=None, yield_curve=None):
    """
    Multi-signal regime detection with 8 voting signals.
    Returns: regime string and confidence score.
    """
    votes = {}

    # ── Signal 1: VIX Level (weight: 25%) ──
    vix = _get_value_at_date(vix_series, date)
    if vix is not None:
        if vix < 13:
            votes["vix_level"] = ("EUPHORIA", 0.25)
        elif vix < 18:
            votes["vix_level"] = ("CALM", 0.25)
        elif vix < 25:
            votes["vix_level"] = ("CAUTION", 0.25)
        elif vix < 35:
            votes["vix_level"] = ("STRESS", 0.25)
        else:
            votes["vix_level"] = ("CRISIS", 0.25)

    # ── Signal 2: SPY vs 200-day SMA (weight: 20%) ──
    if "SPY" in prices.columns:
        spy = prices["SPY"].loc[:date].dropna()
        if len(spy) >= 200:
            sma200 = float(spy.rolling(200).mean().iloc[-1])
            sma50 = float(spy.rolling(50).mean().iloc[-1])
            price = float(spy.iloc[-1])

            if price > sma200 and sma50 > sma200:  # Golden cross + above 200
                votes["spy_sma"] = ("EUPHORIA" if price > sma200 * 1.10 else "CALM", 0.20)
            elif price > sma200:  # Above 200 but 50 below 200
                votes["spy_sma"] = ("CAUTION", 0.20)
            elif price > sma200 * 0.95:  # Just below 200-SMA
                votes["spy_sma"] = ("CAUTION", 0.20)
            else:
                votes["spy_sma"] = ("STRESS", 0.20)

    # ── Signal 3: VIX Trend (weight: 10%) ──
    if vix_series is not None and not vix_series.empty:
        vb = vix_series.loc[:date].dropna()
        if len(vb) >= 21:
            vix_21d_avg = float(vb.tail(21).mean())
            vix_now = float(vb.iloc[-1])
            if vix_now < vix_21d_avg * 0.85:
                votes["vix_trend"] = ("CALM", 0.10)
            elif vix_now > vix_21d_avg * 1.30:
                votes["vix_trend"] = ("STRESS", 0.10)
            else:
                votes["vix_trend"] = ("CAUTION" if vix_now > vix_21d_avg else "CALM", 0.10)

    # ── Signal 4: Market Breadth (weight: 10%) ──
    if len(prices.columns) >= 5:
        equity_tickers = [t for t in ["SPY", "QQQ", "SMH", "XLK", "VGT"] if t in prices.columns]
        if equity_tickers:
            above_sma = 0
            for t in equity_tickers:
                s = prices[t].loc[:date].dropna()
                if len(s) >= 200:
                    if float(s.iloc[-1]) > float(s.rolling(200).mean().iloc[-1]):
                        above_sma += 1
            breadth = above_sma / len(equity_tickers)
            if breadth >= 0.8:
                votes["breadth"] = ("CALM", 0.10)
            elif breadth >= 0.5:
                votes["breadth"] = ("CAUTION", 0.10)
            else:
                votes["breadth"] = ("STRESS", 0.10)

    # ── Signal 5: SPY Drawdown from 52-week high (weight: 10%) ──
    if "SPY" in prices.columns:
        spy = prices["SPY"].loc[:date].dropna()
        if len(spy) >= 252:
            peak = float(spy.tail(252).max())
            dd = (float(spy.iloc[-1]) / peak) - 1
            if dd > -0.05:
                votes["drawdown"] = ("CALM", 0.10)
            elif dd > -0.10:
                votes["drawdown"] = ("CAUTION", 0.10)
            elif dd > -0.20:
                votes["drawdown"] = ("STRESS", 0.10)
            else:
                votes["drawdown"] = ("CRISIS", 0.10)

    # ── Signal 6: Credit Spread (weight: 10%) ──
    if credit_spread is not None:
        cs = _get_value_at_date(credit_spread, date)
        if cs is not None:
            if cs < 3.5:
                votes["credit"] = ("CALM", 0.10)
            elif cs < 5.0:
                votes["credit"] = ("CAUTION", 0.10)
            elif cs < 7.0:
                votes["credit"] = ("STRESS", 0.10)
            else:
                votes["credit"] = ("CRISIS", 0.10)

    # ── Signal 7: Yield Curve (weight: 5%) ──
    if yield_curve is not None:
        yc = _get_value_at_date(yield_curve, date)
        if yc is not None:
            if yc > 1.0:
                votes["yield_curve"] = ("CALM", 0.05)
            elif yc > 0:
                votes["yield_curve"] = ("CAUTION", 0.05)
            else:
                votes["yield_curve"] = ("STRESS", 0.05)

    # ── Signal 8: VIX Spike (weight: 10%) — Emergency override ──
    if vix_series is not None and not vix_series.empty:
        vb = vix_series.loc[:date].dropna()
        if len(vb) >= 6:
            vix_5d_change = (float(vb.iloc[-1]) / float(vb.iloc[-6])) - 1
            if vix_5d_change > 0.50:
                votes["vix_spike"] = ("CRISIS", 0.10)
            elif vix_5d_change > 0.30:
                votes["vix_spike"] = ("STRESS", 0.10)
            else:
                votes["vix_spike"] = ("CALM", 0.10)

    # ── Weighted majority vote ──
    regime_scores = {"EUPHORIA": 0, "CALM": 0, "CAUTION": 0, "STRESS": 0, "CRISIS": 0}
    for signal, (regime, weight) in votes.items():
        regime_scores[regime] += weight

    winner = max(regime_scores, key=regime_scores.get)
    confidence = regime_scores[winner] / max(sum(regime_scores.values()), 0.01)

    # ── Emergency overrides (bypass confirmation) ──
    if "SPY" in prices.columns:
        spy = prices["SPY"].loc[:date].dropna()
        if len(spy) >= 6:
            weekly_return = (float(spy.iloc[-1]) / float(spy.iloc[-6])) - 1
            if weekly_return < -0.05:
                winner = max(winner, "STRESS", key=_regime_severity)
            if weekly_return < -0.10:
                winner = "CRISIS"

    if vix is not None and vix > 40:
        winner = "CRISIS"

    return winner, confidence


def detect_regime_with_confirmation(
    prices, date, vix_series, prev_regime, prev_regime_date,
    credit_spread=None, yield_curve=None, confirmation_weeks=2
):
    """
    Regime detection with 2-week confirmation to prevent whipsaw.
    A new regime must persist for 2 consecutive weeks before switching.
    Emergency overrides bypass confirmation.
    """
    new_regime, confidence = detect_regime_for_date(
        prices, date, vix_series, credit_spread, yield_curve
    )

    # Emergency overrides bypass confirmation
    if new_regime == "CRISIS" and _regime_severity(prev_regime) < _regime_severity("CRISIS"):
        return new_regime, confidence, True

    # If regime hasn't changed, keep it
    if new_regime == prev_regime:
        return new_regime, confidence, False

    # If escalating (getting worse), require 1 week confirmation
    if _regime_severity(new_regime) > _regime_severity(prev_regime):
        if prev_regime_date is not None:
            days_since = (date - prev_regime_date).days
            if days_since >= 5:  # ~1 week for escalation
                return new_regime, confidence, True
        return prev_regime, confidence, False

    # If de-escalating (getting better), require 2 weeks confirmation
    if prev_regime_date is not None:
        days_since = (date - prev_regime_date).days
        if days_since >= 10:  # ~2 weeks for de-escalation
            return new_regime, confidence, True

    return prev_regime, confidence, False


def _regime_severity(regime):
    """Numeric severity for comparison."""
    return {"EUPHORIA": 0, "CALM": 1, "CAUTION": 2, "STRESS": 3, "CRISIS": 4}.get(regime, 1)


def _get_value_at_date(series, date):
    """Get the most recent value at or before a date."""
    if series is None or series.empty:
        return None
    try:
        before = series.loc[:date].dropna()
        return float(before.iloc[-1]) if not before.empty else None
    except Exception:
        return None


class RegimeHealthMonitor:
    """
    Meta-signal from Manus addendum: tracks whether the regime detector
    is actually predicting correctly. If accuracy drops, increases
    defensive allocation automatically.
    """
    def __init__(self, lookback_weeks=8, failure_threshold=3):
        self.lookback_weeks = lookback_weeks
        self.failure_threshold = failure_threshold
        self.predictions = []  # list of (date, predicted_regime, actual_market_outcome)

    def record(self, date, regime, weekly_spy_return):
        """Record a prediction and its outcome."""
        # A prediction is "wrong" if:
        # - Regime was CALM/EUPHORIA but market dropped > 3% that week
        # - Regime was STRESS/CRISIS but market rose > 3% that week
        wrong = False
        if regime in ("EUPHORIA", "CALM") and weekly_spy_return < -0.03:
            wrong = True
        if regime in ("STRESS", "CRISIS") and weekly_spy_return > 0.03:
            wrong = True

        self.predictions.append({"date": date, "regime": regime, "wrong": wrong})

        # Keep only recent predictions
        cutoff = len(self.predictions) - self.lookback_weeks
        if cutoff > 0:
            self.predictions = self.predictions[cutoff:]

    def get_defensive_boost(self):
        """Returns extra defensive allocation (0 to 0.15) if detector is failing."""
        recent_failures = sum(1 for p in self.predictions if p["wrong"])
        if recent_failures >= self.failure_threshold:
            return 0.15  # Add 15 percentage points to defensive allocation
        return 0.0
