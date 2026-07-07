"""
JARVIS V3 - Daily Scheduler (master pipeline)
==============================================
ARCHITECTURAL NOTE (what changed vs V2):
(1) The module-level `_previous_mode` global is GONE. Regime state lives in
    PostgreSQL via risk/regime.py — the state-amnesia bug that froze V2 in
    SAFETY mode for 10 weeks cannot recur, because a container restart now
    resumes the regime machine exactly where it left off.
(2) The pipeline covers the FULL universe (stocks + ETFs) end to end: data
    update, signal computation, optimization, pricing, and rebalancing all
    operate on the same ticker set, so positions can no longer be stranded.
(3) New steps: weekly universe refresh (Mondays), per-position -8% stop-loss
    overlay, stale-order cancellation, and a `pipeline_runs` audit row per
    run that the /health endpoint reads.
(4) Risk-fortress semantics are explicit: an explicit REJECTION aborts
    trading (snapshot still saved); an exception inside the fortress logs
    loudly and proceeds with the optimizer's own hard caps, because the V3
    optimizer already enforces position/sector/gross limits internally.
(5) Cron uses timezone="America/New_York" — V2's hour=15 UTC silently ran at
    11:00 AM ET half the year (DST bug).
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

os.makedirs("logs", exist_ok=True)
logger.add("logs/scheduler.log", rotation="10 MB", retention="30 days", level="INFO")


# ============================================================
# PIPELINE RUN AUDIT (read by /health)
# ============================================================

def _ensure_runs_table():
    from sqlalchemy import text
    from data.db import engine
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id SERIAL PRIMARY KEY,
                run_at TIMESTAMP,
                status TEXT,
                regime TEXT,
                portfolio_value REAL,
                num_positions INTEGER,
                num_orders INTEGER,
                duration_seconds REAL,
                detail TEXT
            )
        """))


def _record_run(status, regime, pv, n_pos, n_orders, duration, detail=""):
    try:
        from sqlalchemy import text
        from data.db import engine
        _ensure_runs_table()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pipeline_runs
                    (run_at, status, regime, portfolio_value, num_positions,
                     num_orders, duration_seconds, detail)
                VALUES (:ts, :status, :regime, :pv, :np, :no, :dur, :detail)
            """), {"ts": datetime.now(timezone.utc), "status": status[:60],
                   "regime": regime[:20], "pv": float(pv or 0),
                   "np": int(n_pos or 0), "no": int(n_orders or 0),
                   "dur": float(duration or 0), "detail": str(detail)[:500]})
    except Exception as e:
        logger.warning(f"pipeline_runs record failed (non-fatal): {e}")


def get_last_run() -> dict:
    """Latest pipeline run for the health endpoint. Never raises."""
    try:
        from sqlalchemy import text
        from data.db import engine
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT run_at, status, regime, portfolio_value, num_positions,"
                " num_orders, duration_seconds FROM pipeline_runs "
                "ORDER BY id DESC LIMIT 1")).fetchone()
        if row:
            return {"run_at": str(row[0]), "status": row[1], "regime": row[2],
                    "portfolio_value": row[3], "num_positions": row[4],
                    "num_orders": row[5], "duration_seconds": row[6]}
    except Exception as e:
        logger.warning(f"get_last_run failed: {e}")
    return {"run_at": None, "status": "no runs recorded"}


# ============================================================
# THE DAILY PIPELINE
# ============================================================

def run_daily_pipeline():
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"JARVIS V3 — DAILY PIPELINE — {start_time:%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    results = {}
    regime_name = "UNKNOWN"
    pv, n_pos, n_orders = 0.0, 0, 0

    def finish(status, detail=""):
        elapsed = (datetime.now() - start_time).total_seconds()
        results["status"] = status
        _record_run(status, regime_name, pv, n_pos, n_orders, elapsed, detail)
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE {status} — {elapsed:.0f}s")
        for k, v in results.items():
            if k not in ("alpha_scores",):
                logger.info(f"  {k}: {v}")
        logger.info(f"{'='*60}\n")
        return results

    try:
        # ── STEP 1: Update market data (full universe, batched) ──
        logger.info("\n--- STEP 1: Updating market data ---")
        from data.ingest import run_daily_update
        upd = run_daily_update()
        results["data_update"] = (f"{len(upd['prices'].get('success', []))} ok / "
                                  f"{len(upd['prices'].get('failed', []))} failed")

        # ── STEP 1b: Weekly universe refresh (Mondays) ──
        if datetime.now().weekday() == 0:
            logger.info("\n--- STEP 1b: Weekly universe refresh ---")
            try:
                from config.universe import refresh_universe
                ref = refresh_universe()
                results["universe"] = (f"{ref['count']} names "
                                       f"({'refreshed' if ref['refreshed'] else 'cached'})")
            except Exception as e:
                logger.warning(f"universe refresh failed (using cache): {e}")
                results["universe"] = f"refresh failed: {e}"

        # ── STEP 2: Data quality gate ──
        logger.info("\n--- STEP 2: Data quality check ---")
        from data.quality import run_all_quality_checks
        quality = run_all_quality_checks()
        results["data_quality"] = "PASSED" if quality["passed"] else "FAILED"
        if not quality["passed"]:
            logger.error("Data quality FAILED — aborting (no trades on bad data)")
            return finish("ABORTED — data quality failure")

        # ── STEP 3: Load price + volume matrices ──
        logger.info("\n--- STEP 3: Loading price matrices ---")
        from config.universe import get_full_universe, get_sector_map
        from data.ingest import get_prices_for_universe
        from config.settings import PRICE_LOOKBACK_DAYS
        universe = get_full_universe()
        prices, volume = get_prices_for_universe(universe, PRICE_LOOKBACK_DAYS,
                                                 with_volume=True)
        if prices.empty or "SPY" not in prices.columns:
            logger.error("Price matrix empty or missing SPY — aborting")
            return finish("ABORTED — no price data")
        results["matrix"] = f"{prices.shape[1]} tickers x {prices.shape[0]} days"
        logger.info(f"  {results['matrix']}")

        # ── STEP 4: Connect to Alpaca ──
        logger.info("\n--- STEP 4: Connecting to Alpaca ---")
        from execution.engine import ExecutionEngine
        executor = ExecutionEngine()
        if not executor.connect():
            return finish("ABORTED — Alpaca connection failure")
        account = executor.get_account_info()
        pv = account.get("portfolio_value", 0)
        current_positions = executor.get_current_positions()
        n_pos = len(current_positions)
        results["portfolio_value"] = pv
        results["num_positions"] = n_pos
        logger.info(f"  Portfolio ${pv:,.2f}, {n_pos} positions")

        # ── STEP 5: Regime (persisted machine — the V2 fix) ──
        logger.info("\n--- STEP 5: Regime evaluation ---")
        from risk.regime import evaluate_and_persist
        regime_info = evaluate_and_persist(prices, portfolio_value=pv)
        regime_name = regime_info["regime"]
        results["regime"] = (f"{regime_name} (gross {regime_info['target_gross']:.0%}, "
                             f"net {regime_info['target_net']:+.0%})")

        # ── STEP 6: Signal ensemble (adaptive IC weights, persisted) ──
        logger.info("\n--- STEP 6: Signal ensemble ---")
        from signals.ensemble import compute_ensemble
        ens = compute_ensemble(prices)
        alpha_scores = ens["latest_scores"]
        results["signals"] = {k: round(v, 3) for k, v in ens["weights_used"].items()}
        if alpha_scores.empty:
            logger.error("Empty alpha scores — holding current book")
            return finish("ABORTED — no alpha scores")

        # ── STEP 7: Optimize (regime exposure + inverse-vol + caps) ──
        logger.info("\n--- STEP 7: Portfolio optimization ---")
        from portfolio.optimizer import optimize_portfolio
        sector_map = get_sector_map()
        opt = optimize_portfolio(alpha_scores, prices, pv, regime_info,
                                 volume=volume, sector_map=sector_map)
        target_weights = opt["target_weights"]
        results["target_positions"] = opt["expected_positions"]
        results["gross_net"] = f"{opt['gross']:.0%} / {opt['net']:+.0%}"
        if not target_weights:
            logger.warning("Empty target book — holding current positions")
            return finish("COMPLETE — empty target, book held")

        # ── STEP 8: Stop-loss overlay (-8% hard per-position stop) ──
        from config.settings import STOP_LOSS_PCT
        stopped = []
        for t, p in current_positions.items():
            if isinstance(p, dict) and p.get("unrealized_pnl_pct", 0) <= STOP_LOSS_PCT:
                target_weights[t] = 0.0   # force a sweep close this cycle
                stopped.append(t)
        if stopped:
            logger.warning(f"  STOP-LOSS triggered: {stopped} "
                           f"(<= {STOP_LOSS_PCT:.0%}) — forcing closes")
            results["stops"] = stopped
        target_weights = {t: w for t, w in target_weights.items() if abs(w) > 0}

        # ── STEP 9: Risk fortress validation ──
        logger.info("\n--- STEP 9: Risk validation ---")
        approved_weights = target_weights
        try:
            from risk.risk_engine import validate_portfolio
            risk_check = validate_portfolio(
                target_weights=target_weights,
                current_positions=current_positions,
                prices=prices,
                portfolio_value=pv,
            )
            results["risk_approved"] = risk_check["approved"]
            if not risk_check["approved"]:
                logger.error(f"Risk REJECTED: {risk_check.get('rejections')}")
                return finish("REJECTED by risk fortress",
                              str(risk_check.get("rejections"))[:400])
            approved_weights = risk_check.get("approved_weights") or target_weights
        except Exception as e:
            # The fortress predates the long/short book; if it crashes (rather
            # than rejects) we proceed on the optimizer's own enforced caps
            # and log loudly. A rejection always halts; a crash never trades
            # MORE than the optimizer built.
            logger.error(f"risk_engine raised ({e}) — proceeding with "
                         f"optimizer-capped weights")
            results["risk_approved"] = f"engine error: {e}"

        # ── STEP 10: Live prices for targets + held + ETF universe ──
        logger.info("\n--- STEP 10: Fetching live prices ---")
        from config.universe import get_all_tickers
        needed = list(set(get_all_tickers()) | set(approved_weights)
                      | set(current_positions))
        current_prices = executor.get_current_prices(needed)
        for t, p in current_positions.items():     # position-data fallback
            if t not in current_prices and isinstance(p, dict):
                if p.get("current_price", 0) > 0:
                    current_prices[t] = p["current_price"]
        last_close = prices.iloc[-1]               # final fallback: last close
        for t in needed:
            if t not in current_prices and t in last_close.index:
                v = last_close[t]
                if pd.notna(v) and v > 0:
                    current_prices[t] = float(v)
        logger.info(f"  priced {len(current_prices)}/{len(needed)} tickers")

        # ── STEP 11: Generate + execute orders (two-phase) ──
        logger.info("\n--- STEP 11: Rebalancing ---")
        from portfolio.rebalancer import generate_rebalance_orders
        frac_map = executor.get_fractionable_map(list(approved_weights))
        orders = generate_rebalance_orders(
            target_weights=approved_weights,
            current_positions=current_positions,
            portfolio_value=pv,
            current_prices=current_prices,
            fractionable=frac_map,
        )
        n_orders = len(orders)
        results["num_orders"] = n_orders

        executor.cancel_stale_orders()
        if orders:
            exec_results = executor.execute_orders(orders)
            results["orders_executed"] = sum(
                1 for r in exec_results if r["status"] in ("SUBMITTED", "FILLED"))
        else:
            logger.info("Portfolio on target — no trades needed")

        # ── STEP 12: Snapshot ──
        logger.info("\n--- STEP 12: Snapshot ---")
        _save_snapshot(pv, account, current_positions)
        return finish("COMPLETE")

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return finish(f"ERROR", str(e))


def _save_snapshot(portfolio_value, account, positions):
    """Daily snapshot — same schema as V2 so the dashboard keeps working."""
    try:
        from sqlalchemy import text
        from data.db import engine
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO portfolio_snapshots
                    (date, total_value, cash, invested, num_positions)
                VALUES (CURRENT_DATE, :value, :cash, :invested, :num_pos)
                ON CONFLICT (date) DO UPDATE SET
                    total_value = EXCLUDED.total_value,
                    cash = EXCLUDED.cash,
                    invested = EXCLUDED.invested,
                    num_positions = EXCLUDED.num_positions
            """), {"value": portfolio_value, "cash": account.get("cash", 0),
                   "invested": account.get("long_market_value", 0),
                   "num_pos": len(positions)})
        logger.info("  snapshot saved")
    except Exception as e:
        logger.warning(f"  snapshot failed: {e}")


# ============================================================
# SCHEDULER
# ============================================================

def run_scheduler():
    """
    10:00 AM America/New_York every weekday. The timezone-aware trigger fixes
    V2's DST bug (hour=15 UTC ran at 11 AM ET in winter). misfire_grace lets
    a run that was due during a restart still fire within the hour.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    try:
        _ensure_runs_table()
    except Exception as e:
        logger.warning(f"runs table ensure failed: {e}")

    sched = BlockingScheduler()
    sched.add_job(
        run_daily_pipeline,
        CronTrigger(day_of_week="mon-fri", hour=10, minute=0,
                    timezone="America/New_York"),
        id="daily_pipeline",
        name="Jarvis V3 Daily Pipeline",
        misfire_grace_time=3600,
        coalesce=True,
    )
    logger.info("Scheduler started — pipeline at 10:00 AM ET weekdays "
                "(DST-aware)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    run_daily_pipeline()
