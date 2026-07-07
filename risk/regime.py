"""
JARVIS V3 - Persistent Multi-Signal Regime Engine
==================================================
NEW FILE. Replaces the in-memory regime/mode logic that previously lived in
`portfolio/optimizer.py` (the module-level `_state` dict) and the
`_previous_mode` global in `scheduler.py`.

WHAT CHANGED AND WHY
--------------------
The old design held the regime ("ACTIVE"/"SAFETY"), the SMA confirmation
counters, the drawdown peak, and the circuit-breaker flag in ordinary Python
variables inside the Railway container. Railway recycles that container on
every deploy, crash, and idle timeout. Each restart wiped the state:

  * the confirmation counters reset to 0, so they could never accumulate the
    N consecutive days required to switch regimes; and
  * the regime itself reset, so the machine's behaviour depended on *when* the
    container happened to restart relative to a moving-average crossing.

That is the mechanical root cause of the documented failure. The machine
latched into a defensive posture during the spring drawdown and could not
climb back out, because the "3 consecutive days above the 200-SMA" counter
never survived long enough to fire.

THE FIX
-------
All regime state now lives in PostgreSQL. Container restarts are invisible:
the machine reads its counters and current state from the database on every
evaluation and writes them back. The decision itself (`decide_regime`) is a
pure function with no I/O, so it can be unit-tested deterministically — see
the simulated-sequence test that ships alongside this file.

DESIGN NOTES (deliberate, and conservative by default)
------------------------------------------------------
  * Three regimes (ACTIVE / CAUTIOUS / CRISIS), not a binary ON/OFF. A binary
    switch is what produced the violent whipsaw this system suffered from; a
    middle state lets the book scale exposure down before it scales it off.
  * Switches require 2-of-3 independent signals to agree (trend, volatility,
    credit), held for a confirmation window. That is intentionally harder to
    trip than a lone 200-SMA cross.
  * A recovery fast-path re-risks immediately when BOTH the trend and the
    volatility signal flip clean — the exact scenario the old system failed on
    (SPY back above its 200-SMA with VIX subsiding) — without waiting out the
    full confirmation window.
  * `EXPOSURE_BY_REGIME` maps each regime to a target gross/net exposure. The
    defaults keep gross <= 100% and use NO leverage. CRISIS still deploys ~50%
    of capital and may tilt modestly net-short, so the book is never parked at
    ~47% in T-bills again — but enabling >100% gross or 2x ETFs is gated behind
    ALLOW_LEVERAGE and should not be switched on before a walk-forward backtest
    earns it. This module decides *posture*; the optimizer consumes these
    targets to build the actual book.

PUBLIC API
----------
  evaluate_and_persist(prices=None, portfolio_value=None) -> dict
      Run the machine for today, write state to the DB, log any transition.
      Call this once per pipeline run (replaces the optimizer's mode logic).
  get_current_regime() -> dict
      Read the latest persisted regime. Safe to call from the API/health
      endpoint. Defaults to CAUTIOUS if the table is empty; never raises.
  should_switch_regime(prices=None) -> dict
      Dry run: report what evaluate_and_persist *would* do, without writing.
  get_target_exposure(regime) -> dict
      Gross/net exposure targets for a regime, honouring ALLOW_LEVERAGE.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Regimes, most defensive last.
ACTIVE = "ACTIVE"
CAUTIOUS = "CAUTIOUS"
CRISIS = "CRISIS"
VALID_REGIMES = (ACTIVE, CAUTIOUS, CRISIS)
DEFAULT_REGIME = CAUTIOUS  # where a cold machine starts: deployed but not aggressive

# ── Signal thresholds ────────────────────────────────────────────────────────
# Trend: SPY vs its 200-day SMA, with a symmetric dead-band so a price hovering
# on the line does not flip the vote every day.
SMA_LOOKBACK = 200
SMA_BUFFER = 0.02  # +/-2% dead-band around the SMA

# Volatility: spot VIX plus a term-structure proxy (spot / 21d average). We do
# not have VIX-futures term structure in this pipeline, so spot-over-its-own-
# average stands in for "front end stressed relative to recent normal".
VIX_CALM = 20.0
VIX_STRESS = 28.0
TERM_CALM = 1.05
TERM_STRESS = 1.15

# Credit: ICE BofA US High Yield OAS (FRED BAMLH0A0HYM2), expressed in percent.
# Level for the regime, plus a fast-widening trigger over ~10 trading days.
CREDIT_CALM = 3.5
CREDIT_STRESS = 5.5
CREDIT_SPIKE_10D = 0.75  # +75bps in two weeks is a stress signal on its own
CREDIT_CHANGE_WINDOW = 10

# ── Confirmation windows (in evaluations ~= trading days) ─────────────────────
CONFIRM_RISK_ON = 2   # consecutive 2-of-3 risk-on days to confirm ACTIVE
CONFIRM_RISK_OFF = 2  # consecutive 2-of-3 risk-off days to confirm CRISIS
MIN_SIGNALS_TO_SWITCH = 2  # never change regime on fewer than 2 live signals

# ── Exposure targets per regime ───────────────────────────────────────────────
# gross = |long| + |short|; net = long - short (as fractions of equity).
# Defaults are unlevered. The optimizer reads these to size the book.
# Single source of truth is config.settings.ALLOW_LEVERAGE; the local default
# keeps this module importable standalone (tests, notebooks).
try:
    from config.settings import ALLOW_LEVERAGE
except Exception:
    ALLOW_LEVERAGE = False  # gate for >100% gross / 2x ETFs; off until validated

EXPOSURE_BY_REGIME = {
    ACTIVE:   {"gross": 1.00, "net": 0.80},   # fully invested, light short hedge
    CAUTIOUS: {"gross": 0.60, "net": 0.30},   # de-risked, still deployed
    CRISIS:   {"gross": 0.50, "net": 0.00},   # defensive, hedged — never net
                                              # short (lab winner 2026-07-07:
                                              # net-short crisis stance cost
                                              # Sharpe AND deepened drawdown)
}
# Used only when ALLOW_LEVERAGE is True.
LEVERED_EXPOSURE_BY_REGIME = {
    ACTIVE:   {"gross": 1.30, "net": 1.00},
    CAUTIOUS: {"gross": 0.80, "net": 0.40},
    CRISIS:   {"gross": 0.60, "net": -0.20},
}


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ══════════════════════════════════════════════════════════════════════════════
# Imports are lazy so this module loads (and its pure logic stays testable) even
# if the database is momentarily unreachable — which matters for the health
# endpoint and for unit tests.

def _get_engine():
    from data.db import engine
    return engine


def _ensure_tables() -> None:
    """Create the regime tables if they do not exist. Idempotent."""
    from sqlalchemy import text
    try:
        with _get_engine().begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS regime_state (
                    date          DATE PRIMARY KEY,
                    regime        TEXT,
                    confidence    REAL,
                    signals_used  TEXT,
                    spy_vs_sma    REAL,
                    vix           REAL,
                    credit_spread REAL,
                    target_gross  REAL,
                    target_net    REAL,
                    updated_at    TIMESTAMP
                )
            """))
            # Single-row persistent state machine (id is always 1). This is the
            # part that fixes the bug: the confirmation counters live here, not
            # in container memory.
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS regime_machine (
                    id            INTEGER PRIMARY KEY,
                    regime        TEXT,
                    bullish_count INTEGER,
                    bearish_count INTEGER,
                    peak_value    REAL,
                    updated_at    TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS regime_transitions (
                    id          BIGSERIAL PRIMARY KEY,
                    timestamp   TIMESTAMP,
                    from_regime TEXT,
                    to_regime   TEXT,
                    confidence  REAL,
                    reason      TEXT
                )
            """))
    except Exception as e:
        logger.warning(f"regime: could not ensure tables: {e}")


def _load_machine() -> dict:
    """Load the persistent machine row, creating a default if none exists."""
    from sqlalchemy import text
    default = {
        "regime": DEFAULT_REGIME,
        "bullish_count": 0,
        "bearish_count": 0,
        "peak_value": 0.0,
    }
    try:
        with _get_engine().begin() as conn:
            row = conn.execute(text(
                "SELECT regime, bullish_count, bearish_count, peak_value "
                "FROM regime_machine WHERE id = 1"
            )).fetchone()
        if row is None:
            return default
        return {
            "regime": row[0] or DEFAULT_REGIME,
            "bullish_count": int(row[1] or 0),
            "bearish_count": int(row[2] or 0),
            "peak_value": float(row[3] or 0.0),
        }
    except Exception as e:
        logger.warning(f"regime: could not load machine state, using default: {e}")
        return default


def _save_machine(state: dict) -> None:
    from sqlalchemy import text
    try:
        with _get_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO regime_machine (id, regime, bullish_count, bearish_count, peak_value, updated_at)
                VALUES (1, :regime, :bull, :bear, :peak, :ts)
                ON CONFLICT (id) DO UPDATE SET
                    regime = EXCLUDED.regime,
                    bullish_count = EXCLUDED.bullish_count,
                    bearish_count = EXCLUDED.bearish_count,
                    peak_value = EXCLUDED.peak_value,
                    updated_at = EXCLUDED.updated_at
            """), {
                "regime": state["regime"],
                "bull": int(state["bullish_count"]),
                "bear": int(state["bearish_count"]),
                "peak": float(state["peak_value"]),
                "ts": datetime.now(timezone.utc),
            })
    except Exception as e:
        logger.error(f"regime: FAILED to persist machine state: {e}")


def _save_state_row(votes: dict, decision: dict, exposure: dict) -> None:
    from sqlalchemy import text
    try:
        with _get_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO regime_state
                    (date, regime, confidence, signals_used, spy_vs_sma, vix,
                     credit_spread, target_gross, target_net, updated_at)
                VALUES
                    (CURRENT_DATE, :regime, :conf, :signals, :sma, :vix,
                     :credit, :gross, :net, :ts)
                ON CONFLICT (date) DO UPDATE SET
                    regime = EXCLUDED.regime,
                    confidence = EXCLUDED.confidence,
                    signals_used = EXCLUDED.signals_used,
                    spy_vs_sma = EXCLUDED.spy_vs_sma,
                    vix = EXCLUDED.vix,
                    credit_spread = EXCLUDED.credit_spread,
                    target_gross = EXCLUDED.target_gross,
                    target_net = EXCLUDED.target_net,
                    updated_at = EXCLUDED.updated_at
            """), {
                "regime": decision["regime"],
                "conf": decision["confidence"],
                "signals": json.dumps(decision["signals_used"]),
                "sma": votes["trend"]["value"],
                "vix": votes["vol"]["value"],
                "credit": votes["credit"]["value"],
                "gross": exposure["gross"],
                "net": exposure["net"],
                "ts": datetime.now(timezone.utc),
            })
    except Exception as e:
        logger.warning(f"regime: could not write regime_state row: {e}")


def _log_transition(from_regime: str, to_regime: str, confidence: float, reason: str) -> None:
    from sqlalchemy import text
    try:
        with _get_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO regime_transitions (timestamp, from_regime, to_regime, confidence, reason)
                VALUES (:ts, :frm, :to, :conf, :reason)
            """), {
                "ts": datetime.now(timezone.utc),
                "frm": from_regime, "to": to_regime,
                "conf": confidence, "reason": reason,
            })
    except Exception as e:
        logger.warning(f"regime: could not log transition: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
# Each signal returns {"vote": -1|0|+1, "value": float|nan, "detail": str}.
# vote = +1 risk-on, -1 risk-off, 0 neutral. value is the raw indicator for
# logging/audit. A signal that cannot be computed returns vote 0 and nan value
# and is treated as "unavailable" (it does not count toward MIN_SIGNALS_TO_SWITCH).

def _trend_vote(prices: pd.DataFrame) -> dict:
    if prices is None or "SPY" not in prices.columns:
        return {"vote": 0, "value": float("nan"), "detail": "no SPY", "available": False}
    spy = prices["SPY"].dropna()
    if len(spy) < SMA_LOOKBACK:
        return {"vote": 0, "value": float("nan"), "detail": "insufficient SPY history", "available": False}
    sma = spy.rolling(SMA_LOOKBACK).mean().iloc[-1]
    px = spy.iloc[-1]
    if sma <= 0 or np.isnan(sma):
        return {"vote": 0, "value": float("nan"), "detail": "bad SMA", "available": False}
    ratio = px / sma - 1.0
    if ratio > SMA_BUFFER:
        vote = 1
    elif ratio < -SMA_BUFFER:
        vote = -1
    else:
        vote = 0
    return {"vote": vote, "value": float(ratio),
            "detail": f"SPY {ratio:+.1%} vs 200-SMA", "available": True}


def _vol_vote(vix_series: Optional[pd.Series]) -> dict:
    if vix_series is None or len(vix_series.dropna()) < 21:
        return {"vote": 0, "value": float("nan"), "detail": "no VIX", "available": False}
    s = vix_series.dropna()
    vix = float(s.iloc[-1])
    vix_sma = float(s.rolling(21).mean().iloc[-1])
    term = vix / vix_sma if vix_sma > 0 else 1.0
    if vix >= VIX_STRESS or term >= TERM_STRESS:
        vote = -1
    elif vix <= VIX_CALM and term <= TERM_CALM:
        vote = 1
    else:
        vote = 0
    return {"vote": vote, "value": vix,
            "detail": f"VIX {vix:.1f}, term {term:.2f}", "available": True}


def _credit_vote(credit_series: Optional[pd.Series]) -> dict:
    if credit_series is None or len(credit_series.dropna()) < CREDIT_CHANGE_WINDOW + 1:
        return {"vote": 0, "value": float("nan"), "detail": "no credit", "available": False}
    s = credit_series.dropna()
    oas = float(s.iloc[-1])
    change = float(oas - s.iloc[-(CREDIT_CHANGE_WINDOW + 1)])
    if oas >= CREDIT_STRESS or change >= CREDIT_SPIKE_10D:
        vote = -1
    elif oas <= CREDIT_CALM and change <= 0:
        vote = 1
    else:
        vote = 0
    return {"vote": vote, "value": oas,
            "detail": f"HY OAS {oas:.2f}%, {change:+.2f} / {CREDIT_CHANGE_WINDOW}d",
            "available": True}


def _compute_votes(prices: Optional[pd.DataFrame] = None) -> dict:
    """Gather the three regime signals from the database/price matrix."""
    if prices is None:
        from data.db import get_all_prices
        prices = get_all_prices()

    # Macro series from FRED (with the existing Yahoo VIX fallback).
    vix_series = None
    credit_series = None
    try:
        from data.db import get_macro
        vix_df = get_macro("VIXCLS")
        if vix_df is None or vix_df.empty:
            vix_df = get_macro("VIX_YAHOO")
        if vix_df is not None and not vix_df.empty:
            vix_series = vix_df["value"]
        credit_df = get_macro("BAMLH0A0HYM2")
        if credit_df is not None and not credit_df.empty:
            credit_series = credit_df["value"]
    except Exception as e:
        logger.warning(f"regime: macro fetch failed ({e}); proceeding on trend only")

    return {
        "trend": _trend_vote(prices),
        "vol": _vol_vote(vix_series),
        "credit": _credit_vote(credit_series),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PURE DECISION LOGIC  (no I/O — unit-testable)
# ══════════════════════════════════════════════════════════════════════════════

def decide_regime(votes: dict, machine: dict) -> dict:
    """
    Given today's signal votes and the persisted machine counters, decide the
    new regime and the updated counters. Pure function: identical inputs always
    give identical outputs.

    votes:   {"trend": {...}, "vol": {...}, "credit": {...}} from _compute_votes
    machine: {"regime", "bullish_count", "bearish_count", ...}

    Returns: {"regime", "bullish_count", "bearish_count", "confidence",
              "signals_used", "reason", "recovery"}
    """
    current = machine.get("regime", DEFAULT_REGIME)
    if current not in VALID_REGIMES:
        current = DEFAULT_REGIME

    trend = votes["trend"]["vote"]
    vol = votes["vol"]["vote"]
    credit = votes["credit"]["vote"]

    available = sum(1 for k in ("trend", "vol", "credit") if votes[k].get("available"))
    risk_on = sum(1 for v in (trend, vol, credit) if v == 1)
    risk_off = sum(1 for v in (trend, vol, credit) if v == -1)

    # Persisted hysteresis counters: increment on a confirmed 2-of-3 day, else reset.
    bull = machine.get("bullish_count", 0) + 1 if risk_on >= 2 else 0
    bear = machine.get("bearish_count", 0) + 1 if risk_off >= 2 else 0

    # Recovery fast-path: the exact failure case from the spring drawdown —
    # trend back positive AND volatility benign — re-risks without waiting.
    recovery = (trend == 1 and vol == 1)

    if available < MIN_SIGNALS_TO_SWITCH:
        target = current
        reason = f"hold: only {available} live signal(s)"
    elif recovery:
        target = ACTIVE
        reason = "recovery fast-path: trend up + volatility benign"
    elif bear >= CONFIRM_RISK_OFF:
        target = CRISIS
        reason = f"risk-off confirmed {bear}d (2-of-3)"
    elif bull >= CONFIRM_RISK_ON:
        target = ACTIVE
        reason = f"risk-on confirmed {bull}d (2-of-3)"
    elif risk_off >= 2 and current == ACTIVE:
        target = CAUTIOUS  # step down one notch immediately on first 2-of-3 risk-off
        reason = "step down: 2-of-3 risk-off (unconfirmed)"
    elif risk_on >= 2 and current == CRISIS:
        target = CAUTIOUS  # step up one notch out of crisis
        reason = "step up: 2-of-3 risk-on out of crisis"
    else:
        target = current
        reason = f"hold: {risk_on} on / {risk_off} off, unconfirmed"

    confidence = round(max(risk_on, risk_off) / max(available, 1), 3) if available else 0.0
    signals_used = {
        k: {"vote": votes[k]["vote"], "value": votes[k]["value"], "detail": votes[k]["detail"]}
        for k in ("trend", "vol", "credit")
    }

    return {
        "regime": target,
        "bullish_count": bull,
        "bearish_count": bear,
        "confidence": confidence,
        "signals_used": signals_used,
        "reason": reason,
        "recovery": recovery,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXPOSURE TARGETS
# ══════════════════════════════════════════════════════════════════════════════

def get_target_exposure(regime: str) -> dict:
    """Return {'gross', 'net'} exposure targets for a regime."""
    table = LEVERED_EXPOSURE_BY_REGIME if ALLOW_LEVERAGE else EXPOSURE_BY_REGIME
    return dict(table.get(regime, EXPOSURE_BY_REGIME[DEFAULT_REGIME]))


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_persist(prices: Optional[pd.DataFrame] = None,
                         portfolio_value: Optional[float] = None) -> dict:
    """
    Run the regime machine for today and persist the result. Call once per
    pipeline run. Returns the full decision dict plus exposure targets.
    """
    _ensure_tables()
    machine = _load_machine()
    votes = _compute_votes(prices)
    decision = decide_regime(votes, machine)
    exposure = get_target_exposure(decision["regime"])

    # Track the drawdown peak in persistent state too (available to the optimizer
    # for an optional circuit breaker — also previously lost on restart).
    if portfolio_value is not None and portfolio_value > 0:
        machine["peak_value"] = max(machine.get("peak_value", 0.0), float(portfolio_value))

    previous = machine["regime"]
    new_machine = {
        "regime": decision["regime"],
        "bullish_count": decision["bullish_count"],
        "bearish_count": decision["bearish_count"],
        "peak_value": machine.get("peak_value", 0.0),
    }
    _save_machine(new_machine)
    _save_state_row(votes, decision, exposure)

    logger.info("=" * 55)
    logger.info("REGIME ENGINE")
    logger.info(f"  Previous: {previous}  ->  Now: {decision['regime']}  "
                f"(confidence {decision['confidence']:.0%})")
    for k in ("trend", "vol", "credit"):
        v = votes[k]
        flag = "+" if v["vote"] == 1 else ("-" if v["vote"] == -1 else "0")
        logger.info(f"    [{flag}] {k:6s}: {v['detail']}")
    logger.info(f"  Counters: bull={decision['bullish_count']} bear={decision['bearish_count']}")
    logger.info(f"  Reason: {decision['reason']}")
    logger.info(f"  Target exposure: gross {exposure['gross']:.0%}, net {exposure['net']:+.0%}"
                f"{'  (LEVERAGE ON)' if ALLOW_LEVERAGE else ''}")
    logger.info("=" * 55)

    if decision["regime"] != previous:
        logger.warning(f"  REGIME TRANSITION: {previous} -> {decision['regime']} "
                       f"({decision['reason']})")
        _log_transition(previous, decision["regime"], decision["confidence"], decision["reason"])

    return {
        "regime": decision["regime"],
        "previous_regime": previous,
        "confidence": decision["confidence"],
        "reason": decision["reason"],
        "recovery": decision["recovery"],
        "bullish_count": decision["bullish_count"],
        "bearish_count": decision["bearish_count"],
        "signals_used": decision["signals_used"],
        "target_gross": exposure["gross"],
        "target_net": exposure["net"],
        "switched": decision["regime"] != previous,
    }


def get_current_regime() -> dict:
    """
    Read the most recent persisted regime. Safe for the health/API endpoint:
    never raises, and returns a sensible default if nothing is stored yet.
    """
    from sqlalchemy import text
    try:
        with _get_engine().begin() as conn:
            row = conn.execute(text("""
                SELECT date, regime, confidence, target_gross, target_net, updated_at
                FROM regime_state
                WHERE date = (SELECT MAX(date) FROM regime_state)
            """)).fetchone()
        if row is not None:
            return {
                "date": str(row[0]),
                "regime": row[1],
                "confidence": float(row[2]) if row[2] is not None else None,
                "target_gross": float(row[3]) if row[3] is not None else None,
                "target_net": float(row[4]) if row[4] is not None else None,
                "updated_at": str(row[5]) if row[5] is not None else None,
            }
    except Exception as e:
        logger.warning(f"regime: get_current_regime fell back to default: {e}")

    exposure = get_target_exposure(DEFAULT_REGIME)
    return {
        "date": None, "regime": DEFAULT_REGIME, "confidence": 0.0,
        "target_gross": exposure["gross"], "target_net": exposure["net"],
        "updated_at": None,
    }


def should_switch_regime(prices: Optional[pd.DataFrame] = None) -> dict:
    """
    Dry run. Compute what evaluate_and_persist would decide right now, without
    writing to the database. Useful for the dashboard and for testing.
    """
    machine = _load_machine()
    votes = _compute_votes(prices)
    decision = decide_regime(votes, machine)
    return {
        "current_regime": machine["regime"],
        "would_be_regime": decision["regime"],
        "switch": decision["regime"] != machine["regime"],
        "confidence": decision["confidence"],
        "reason": decision["reason"],
        "signals_used": decision["signals_used"],
    }


if __name__ == "__main__":
    # Smoke test against the live database if one is configured.
    try:
        result = evaluate_and_persist()
        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:  # pragma: no cover
        print(f"Live evaluation unavailable in this environment: {exc}")
