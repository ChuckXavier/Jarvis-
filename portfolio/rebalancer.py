"""
JARVIS V3 - Portfolio Rebalancer
=================================
ARCHITECTURAL NOTE (what changed vs V2):
(1) Positions are handled as SIGNED quantities throughout (Alpaca reports
    short qty/market_value as negative), so the long/short bookkeeping that
    was special-cased and partially wrong in V2 is now one code path.
(2) Side crossings (long -> short or back) are split into an explicit CLOSE
    of the old position plus an OPEN of the new one — a single net order
    through zero is exactly what brokers reject.
(3) Every held ticker absent from the targets gets a priority-0 SWEEP close;
    stranded positions (the V2 stuck-stock bug) cannot survive a cycle.
(4) Orders carry a priority (0 sweep/close, 1 reduce, 2 open-short, 3 open/
    add-long) that the execution engine uses to run all exposure-reducing
    orders, wait for fills, then run the opens — the institutional
    sell-first/buy-second sequencing, now structural rather than a sort hint.
(5) Shorts and non-fractionable assets are floored to whole shares HERE, so
    the engine receives quantities that are already broker-legal.
"""

from __future__ import annotations

import math

from loguru import logger

from config.settings import (
    REBALANCE_DRIFT_THRESHOLD, MIN_TRADE_WEIGHT, MAX_DAILY_TRADES,
    MIN_POSITION_PCT,
)

# Order priorities — phase boundary is <=1 vs >=2 in the execution engine.
PRIO_SWEEP = 0    # close a position entirely (incl. stale sweeps)
PRIO_REDUCE = 1   # shrink an existing position
PRIO_SHORT = 2    # open / add to a short
PRIO_LONG = 3     # open / add to a long


def _signed_qty(pos) -> float:
    if isinstance(pos, dict):
        return float(pos.get("qty", 0.0))
    return float(pos)


def _order(ticker, side, qty, price, prio, reason, target_w=0.0, current_w=0.0):
    return {
        "ticker": ticker,
        "side": side,                       # 'buy' | 'sell'
        "quantity": qty,                    # always positive
        "estimated_value": round(abs(qty) * price, 2),
        "price_ref": round(price, 4),
        "priority": prio,
        "target_weight": round(target_w, 4),
        "current_weight": round(current_w, 4),
        "reason": reason,
    }


def generate_rebalance_orders(target_weights: dict, current_positions: dict,
                              portfolio_value: float, current_prices: dict,
                              fractionable: dict | None = None) -> list[dict]:
    """
    Diff current signed positions against target signed weights and emit a
    priority-ordered order list.

    target_weights:    {ticker: signed weight}   (negative = short)
    current_positions: {ticker: {"qty": signed, "market_value": signed,
                                  "current_price": float}}
    fractionable:      {ticker: bool} from the execution engine's asset cache;
                       unknown tickers default to fractionable for longs.
    """
    if portfolio_value <= 0:
        logger.error("rebalancer: non-positive portfolio value")
        return []
    fractionable = fractionable or {}
    orders: list[dict] = []

    held = set(current_positions.keys())
    targeted = {t for t, w in target_weights.items() if abs(w) >= MIN_POSITION_PCT}

    def px(t):
        p = float(current_prices.get(t, 0) or 0)
        if p <= 0 and t in current_positions and isinstance(current_positions[t], dict):
            p = float(current_positions[t].get("current_price", 0) or 0)
        return p

    # ── 1) SWEEP: held but not targeted -> close entirely ──────────────────
    for t in sorted(held - targeted):
        q = _signed_qty(current_positions[t])
        if abs(q) < 1e-9:
            continue
        p = px(t)
        if p <= 0:
            logger.warning(f"  sweep {t}: no price — engine will use "
                           f"close_position API instead")
        side = "sell" if q > 0 else "buy"
        cw = (q * p / portfolio_value) if p > 0 else 0.0
        orders.append(_order(t, side, abs(q), max(p, 0.01), PRIO_SWEEP,
                             "sweep: not in targets", 0.0, cw))

    # ── 2) Targeted names: diff signed desired vs signed held ──────────────
    for t in sorted(targeted):
        w = float(target_weights[t])
        p = px(t)
        if p <= 0:
            logger.warning(f"  {t}: no price available — skipping this cycle")
            continue
        held_q = _signed_qty(current_positions.get(t, 0.0))
        cw = held_q * p / portfolio_value

        desired_q = portfolio_value * w / p
        # Whole shares for anything short-side or known non-fractionable.
        needs_whole = (w < 0) or (held_q < 0) or (fractionable.get(t, True) is False)
        if needs_whole:
            desired_q = math.copysign(math.floor(abs(desired_q)), desired_q)
            if desired_q == 0:
                if abs(held_q) > 1e-9:
                    side = "sell" if held_q > 0 else "buy"
                    orders.append(_order(t, side, abs(held_q), p, PRIO_SWEEP,
                                         "target rounds to 0 shares — close",
                                         w, cw))
                else:
                    logger.info(f"  {t}: target {w:+.2%} rounds to 0 whole "
                                f"shares — skipped")
                continue

        # Side crossing: close old, open new — two orders, never one net.
        if held_q * desired_q < 0:
            close_side = "sell" if held_q > 0 else "buy"
            orders.append(_order(t, close_side, abs(held_q), p, PRIO_SWEEP,
                                 f"cross-side close ({cw:+.1%} -> {w:+.1%})",
                                 w, cw))
            open_side = "buy" if desired_q > 0 else "sell"
            prio = PRIO_LONG if desired_q > 0 else PRIO_SHORT
            orders.append(_order(t, open_side, abs(desired_q), p, prio,
                                 f"cross-side open ({cw:+.1%} -> {w:+.1%})",
                                 w, cw))
            continue

        delta_q = desired_q - held_q
        delta_w = delta_q * p / portfolio_value

        # Drift gate: trade only if the move is material in both relative and
        # absolute terms (or it crosses sides, handled above).
        rel_gate = REBALANCE_DRIFT_THRESHOLD * abs(w)
        if abs(delta_w) < max(MIN_TRADE_WEIGHT, rel_gate):
            continue

        if needs_whole:
            delta_q = math.copysign(math.floor(abs(delta_q)), delta_q)
            if abs(delta_q) < 1:
                continue
        elif abs(delta_q) < 1e-3:
            continue

        increasing = abs(desired_q) > abs(held_q)
        if increasing:
            side = "buy" if desired_q > 0 else "sell"
            prio = PRIO_LONG if desired_q > 0 else PRIO_SHORT
            verb = "add"
        else:
            side = "sell" if held_q > 0 else "buy"
            prio = PRIO_REDUCE
            verb = "reduce"
            # Reducing to dust? Close it cleanly with the exact held qty.
            if abs(desired_q) * p / portfolio_value < MIN_POSITION_PCT:
                orders.append(_order(t, side, abs(held_q), p, PRIO_SWEEP,
                                     f"reduce-to-dust close ({cw:+.1%})",
                                     0.0, cw))
                continue
        orders.append(_order(t, side, abs(delta_q), p, prio,
                             f"{verb}: {cw:+.1%} -> {w:+.1%}", w, cw))

    # ── 3) Order + cap ──────────────────────────────────────────────────────
    orders.sort(key=lambda o: (o["priority"], -o["estimated_value"]))
    if len(orders) > MAX_DAILY_TRADES:
        protected = [o for o in orders if o["priority"] <= PRIO_REDUCE]
        opens = [o for o in orders if o["priority"] >= PRIO_SHORT]
        room = max(0, MAX_DAILY_TRADES - len(protected))
        dropped = len(opens) - room
        orders = protected + opens[:room]
        logger.warning(f"  trade cap: kept all {len(protected)} closes/reduces, "
                       f"dropped {dropped} lowest-value opens")

    n_close = sum(1 for o in orders if o["priority"] <= PRIO_REDUCE)
    n_open = len(orders) - n_close
    v_close = sum(o["estimated_value"] for o in orders if o["priority"] <= PRIO_REDUCE)
    v_open = sum(o["estimated_value"] for o in orders if o["priority"] >= PRIO_SHORT)
    logger.info(f"Rebalance orders: {n_close} closes/reduces (${v_close:,.0f}) "
                f"-> {n_open} opens (${v_open:,.0f})")
    return orders


def calculate_turnover(current_weights: dict, target_weights: dict) -> float:
    """One-sided turnover: sum |Δw| / 2."""
    all_t = set(current_weights) | set(target_weights)
    return sum(abs(target_weights.get(t, 0.0) - current_weights.get(t, 0.0))
               for t in all_t) / 2.0
