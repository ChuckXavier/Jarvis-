"""
JARVIS V2 - Daily Scheduler
==============================
Runs the complete trading pipeline automatically every day.

HOW THIS WORKS (for non-coders):
- This file keeps Jarvis alive 24/7 on Railway
- Every trading day, it runs the full cycle:
    6:00 AM ET  → Refresh data (new prices from yesterday)
    6:15 AM ET  → Compute features (16+ indicators per ETF)
    6:30 AM ET  → Run all 4 signals → generate alpha scores
    6:45 AM ET  → Optimize portfolio (risk parity + alpha tilt)
    7:00 AM ET  → Risk validation (5-layer fortress)
    10:00 AM ET → Execute trades (sells first, then buys)
    4:00 PM ET  → Reconciliation (verify all orders filled)
    4:30 PM ET  → Log daily P&L and portfolio snapshot
"""

import time
import sys
import os
from datetime import datetime
from loguru import logger
import pandas as pd

# Set up logging
os.makedirs("logs", exist_ok=True)
logger.add("logs/scheduler.log", rotation="10 MB", retention="30 days", level="INFO")


def run_daily_pipeline():
    """
    Run the complete daily trading pipeline.
    This is the MAIN function that produces trading decisions.
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"JARVIS V2 — DAILY PIPELINE — {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    results = {}

    try:
        # ── Step 1: Update Data ──
        logger.info("\n--- STEP 1: Updating market data ---")
        from data.ingest import run_daily_update, download_fred_data, download_vix_from_yahoo

        etf_result = run_daily_update()
        download_fred_data(years=1)
        download_vix_from_yahoo(years=1)
        results["data_update"] = f"{len(etf_result.get('success', []))} ETFs updated"

        # ── Step 1b: Update Stock Universe (60/40 Combined) ──
        logger.info("\n--- STEP 1b: Updating stock universe ---")
        try:
            from data.stock_universe import update_stock_universe
            stock_result = update_stock_universe()
            results["stock_update"] = f"{stock_result.get('success', 0)} stocks updated"
        except Exception as e:
            logger.warning(f"Stock universe update failed: {e}")
            results["stock_update"] = f"FAILED: {e}"

        # ── Step 2: Data Quality Check ──
        logger.info("\n--- STEP 2: Data quality check ---")
        from data.quality import run_all_quality_checks

        quality = run_all_quality_checks()
        results["data_quality"] = "PASSED" if quality["passed"] else "FAILED"

        if not quality["passed"]:
            logger.error("Data quality check FAILED — aborting pipeline")
            results["status"] = "ABORTED — data quality failure"
            return results

        # ── Step 3: Compute Alpha Scores ──
        logger.info("\n--- STEP 3: Computing alpha scores ---")
        from data.db import get_all_prices
        from signals.ensemble import compute_ensemble

        prices = get_all_prices()
        ensemble = compute_ensemble(prices)
        alpha_scores = ensemble["latest_scores"]
        regime = ensemble["regime"]

        results["regime"] = regime
        results["alpha_scores"] = alpha_scores.to_dict() if not alpha_scores.empty else {}

        # ── Step 4: Connect to Alpaca ──
        logger.info("\n--- STEP 4: Connecting to Alpaca ---")
        from execution.engine import ExecutionEngine

        executor = ExecutionEngine()
        if not executor.connect():
            logger.error("Cannot connect to Alpaca — aborting")
            results["status"] = "ABORTED — Alpaca connection failure"
            return results

        account = executor.get_account_info()
        portfolio_value = account.get("portfolio_value", 0)
        current_positions = executor.get_current_positions()

        results["portfolio_value"] = portfolio_value
        results["num_positions"] = len(current_positions)

        logger.info(f"  Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"  Current positions: {len(current_positions)}")

        # ── Step 5: Optimize Portfolio ──
        logger.info("\n--- STEP 5: Optimizing portfolio ---")

        # Load stock prices and combine with ETF prices (60/40 Combined)
        try:
            from data.stock_universe import get_stock_prices
            stock_prices = get_stock_prices()
            if not stock_prices.empty:
                prices = pd.concat([prices, stock_prices], axis=1)
                prices = prices.loc[:, ~prices.columns.duplicated()]
                logger.info(f"  Combined: {len(prices.columns)} instruments")
        except Exception as e:
            logger.warning(f"  Stock prices not available: {e}")

        from portfolio.optimizer import optimize_portfolio

        optimization = optimize_portfolio(alpha_scores, prices, portfolio_value)
        target_weights = optimization["target_weights"]

        results["target_positions"] = optimization["expected_positions"]
        results["cash_pct"] = optimization["cash_pct"]

        # ── Step 6: Risk Validation ──
        logger.info("\n--- STEP 6: Risk validation ---")
        from risk.risk_engine import validate_portfolio

        risk_check = validate_portfolio(
            target_weights=target_weights,
            current_positions=current_positions,
            prices=prices,
            portfolio_value=portfolio_value,
        )

        results["risk_approved"] = risk_check["approved"]
        results["risk_warnings"] = len(risk_check["warnings"])

        if not risk_check["approved"]:
            logger.warning(f"Risk check REJECTED: {risk_check['rejections']}")
            results["status"] = "REJECTED by risk fortress"
            return results

        approved_weights = risk_check["approved_weights"]

        # ── Step 7: Generate Orders ──
        logger.info("\n--- STEP 7: Generating orders ---")
        from portfolio.rebalancer import generate_rebalance_orders
        from config.universe import get_all_tickers

        # Get all tickers that need pricing: ETF universe + any stock targets
        all_needed_tickers = list(set(get_all_tickers()) | set(approved_weights.keys()))
        current_prices = executor.get_current_prices(all_needed_tickers)

        orders = generate_rebalance_orders(
            target_weights=approved_weights,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            current_prices=current_prices,
        )

        results["num_orders"] = len(orders)

        if not orders:
            logger.info("No rebalancing needed — portfolio is on target")
            results["status"] = "COMPLETE — no trades needed"
            return results

        # ── Step 8: Execute Orders ──
        logger.info("\n--- STEP 8: Executing orders ---")
        execution_results = executor.execute_orders(orders)

        filled = sum(1 for r in execution_results if r["status"] in ("FILLED", "SUBMITTED"))
        shadow = sum(1 for r in execution_results if r["status"] == "SHADOW")

        results["orders_executed"] = filled
        results["orders_shadow"] = shadow
        results["status"] = "COMPLETE"

        # ── Step 9: Save Portfolio Snapshot ──
        logger.info("\n--- STEP 9: Saving portfolio snapshot ---")
        _save_snapshot(portfolio_value, account, current_positions)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        results["status"] = f"ERROR: {str(e)}"

    # ── Summary ──
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY PIPELINE COMPLETE — {elapsed:.0f} seconds")
    for key, val in results.items():
        if key != "alpha_scores":
            logger.info(f"  {key}: {val}")
    logger.info(f"{'='*60}\n")

    return results


def _save_snapshot(portfolio_value, account, positions):
    """Save daily portfolio snapshot to database."""
    try:
        from data.db import engine
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO portfolio_snapshots (date, total_value, cash, invested, num_positions)
                VALUES (CURRENT_DATE, :value, :cash, :invested, :num_pos)
                ON CONFLICT (date) DO UPDATE SET
                    total_value = EXCLUDED.total_value,
                    cash = EXCLUDED.cash,
                    invested = EXCLUDED.invested,
                    num_positions = EXCLUDED.num_positions
            """), {
                "value": portfolio_value,
                "cash": account.get("cash", 0),
                "invested": account.get("long_market_value", 0),
                "num_pos": len(positions),
            })
        logger.info("  Portfolio snapshot saved")
    except Exception as e:
        logger.warning(f"  Failed to save snapshot: {e}")


def run_scheduler():
    """
    Keep Jarvis alive and run the pipeline on schedule.
    Uses simple sleep-based scheduling (reliable on Railway).
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler()

    # Run the full pipeline at 10:00 AM ET (15:00 UTC) every weekday
    scheduler.add_job(
        run_daily_pipeline,
        CronTrigger(
            day_of_week="mon-fri",
            hour=15,  # 10 AM ET = 15:00 UTC
            minute=0,
            timezone="UTC",
        ),
        id="daily_pipeline",
        name="Jarvis Daily Pipeline",
    )

    logger.info("Scheduler started — Jarvis will run at 10:00 AM ET every weekday")
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    # If run directly, execute the pipeline immediately
    run_daily_pipeline()
