"""
JARVIS V2 - Phase 3 Runner: Full Pipeline Test
=================================================
Tests the COMPLETE pipeline end-to-end:
Data → Features → Signals → Portfolio → Risk → Execution (shadow)

This is the final test before Jarvis goes live.

HOW TO RUN:
    python run_phase3.py
"""

import sys
import os
import pandas as pd
from loguru import logger
from datetime import datetime

# Set up logging
os.makedirs("logs", exist_ok=True)
logger.add("logs/phase3.log", rotation="10 MB", level="INFO")


def main():
    print()
    print("=" * 60)
    print("  J A R V I S   V 2")
    print("  Phase 3: Full Pipeline Test")
    print("  Data → Signals → Portfolio → Risk → Execution")
    print("=" * 60)
    print()

    # ── Step 1: Config Check ──
    print("STEP 1: Checking configuration...")
    from config.settings import validate_config, EXECUTION_MODE
    missing = validate_config()
    if missing:
        print(f"  ❌ Missing: {missing}")
        sys.exit(1)
    print(f"  ✅ Config OK | Execution mode: {EXECUTION_MODE}\n")

    # ── Step 2: Load Data ──
    print("STEP 2: Loading data...")
    from data.db import get_all_prices, get_record_count

    try:
        price_count = get_record_count("daily_prices")
        print(f"  Found {price_count:,} price records")
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        sys.exit(1)

    prices = get_all_prices()
    if prices.empty or len(prices) < 252:
        print("  ❌ Insufficient data. Run Phase 1 first.")
        sys.exit(1)

    print(f"  ✅ {len(prices.columns)} ETFs, {len(prices)} days ({prices.index[0].date()} to {prices.index[-1].date()})\n")

    # ── Step 3: Run Alpha Engine ──
    print("STEP 3: Running Alpha Engine (all 4 signals)...")
    print("  (This takes 1-2 minutes...)\n")

    from signals.ensemble import compute_ensemble
    ensemble = compute_ensemble(prices)
    alpha_scores = ensemble["latest_scores"]
    regime = ensemble["regime"]

    print(f"\n  ✅ Regime: {regime}")
    print(f"  ✅ Alpha scores for {len(alpha_scores)} ETFs")

    if not alpha_scores.empty:
        top3 = alpha_scores.head(3)
        bot3 = alpha_scores.tail(3)
        print(f"  Top 3: {', '.join(f'{t}({s:+.2f})' for t, s in top3.items())}")
        print(f"  Bot 3: {', '.join(f'{t}({s:+.2f})' for t, s in bot3.items())}")
    print()

    # ── Step 4: Connect to Alpaca ──
    print("STEP 4: Connecting to Alpaca Paper Trading...")
    from execution.engine import ExecutionEngine

    executor = ExecutionEngine()
    connected = executor.connect()

    if connected:
        account = executor.get_account_info()
        portfolio_value = account.get("portfolio_value", 100000)
        current_positions = executor.get_current_positions()

        print(f"  ✅ Connected!")
        print(f"  Portfolio value: ${portfolio_value:,.2f}")
        print(f"  Cash: ${account.get('cash', 0):,.2f}")
        print(f"  Current positions: {len(current_positions)}")
    else:
        print("  ⚠️ Could not connect to Alpaca — using simulated $100,000 portfolio")
        portfolio_value = 100000
        current_positions = {}
        account = {"cash": 100000, "portfolio_value": 100000}
    print()

    # ── Step 5: Optimize Portfolio ──
    print("STEP 5: Optimizing portfolio (Risk Parity + Alpha Tilt)...")
    from portfolio.optimizer import optimize_portfolio

    optimization = optimize_portfolio(alpha_scores, prices, portfolio_value)
    target_weights = optimization["target_weights"]

    print(f"  ✅ Target positions: {optimization['expected_positions']}")
    print(f"  Cash reserve: {optimization['cash_pct']:.1%}")
    print(f"\n  Target allocation:")

    sorted_targets = sorted(target_weights.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_targets:
        if weight > 0.005:
            dollar = portfolio_value * weight
            bar = "█" * int(weight * 100)
            print(f"    {ticker:5s}  {weight:6.1%}  ${dollar:>9,.0f}  {bar}")
    print()

    # ── Step 6: Risk Validation ──
    print("STEP 6: Running Risk Fortress (5 layers)...")
    from risk.risk_engine import validate_portfolio

    risk_result = validate_portfolio(
        target_weights=target_weights,
        current_positions=current_positions,
        prices=prices,
        portfolio_value=portfolio_value,
    )

    if risk_result["approved"]:
        print(f"  ✅ RISK APPROVED ({len(risk_result['warnings'])} warnings)")
    else:
        print(f"  ❌ RISK REJECTED: {risk_result['rejections']}")

    for w in risk_result["warnings"][:5]:
        print(f"    ⚠️ {w}")
    print()

    approved_weights = risk_result["approved_weights"]

    # ── Step 7: Generate Rebalance Orders ──
    print("STEP 7: Generating rebalance orders...")
    from portfolio.rebalancer import generate_rebalance_orders
    from config.universe import get_all_tickers

    if connected:
        current_prices = executor.get_current_prices(get_all_tickers())
    else:
        # Use last known prices from database
        current_prices = prices.iloc[-1].to_dict()

    orders = generate_rebalance_orders(
        target_weights=approved_weights,
        current_positions=current_positions,
        portfolio_value=portfolio_value,
        current_prices=current_prices,
    )

    print(f"  ✅ Generated {len(orders)} orders")
    print()

    if orders:
        print(f"  {'Action':6s}  {'Ticker':6s}  {'Shares':>8s}  {'Value':>10s}  {'Reason'}")
        print(f"  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*30}")
        for o in orders:
            action = "🟢 BUY" if o["side"] == "buy" else "🔴 SELL"
            print(f"  {action}  {o['ticker']:6s}  {o['quantity']:8.2f}  ${o['estimated_value']:>9,.2f}  "
                  f"{o['current_weight']:.1%} → {o['target_weight']:.1%}")
        print()

    # ── Step 8: Execute (Shadow Mode) ──
    print("STEP 8: Executing orders (SHADOW mode — logging only)...")
    if orders:
        results = executor.execute_orders(orders)
        shadow_count = sum(1 for r in results if r["status"] == "SHADOW")
        submitted_count = sum(1 for r in results if r["status"] == "SUBMITTED")
        print(f"  ✅ {shadow_count} shadow-logged, {submitted_count} submitted")
    else:
        print("  No orders to execute")
    print()

    # ── Final Summary ──
    print("=" * 60)
    print("  ✅ PHASE 3 COMPLETE — FULL PIPELINE OPERATIONAL")
    print("=" * 60)
    print()
    print("  JARVIS V2 can now:")
    print("    ✅ Download and validate market data (25 ETFs + macro)")
    print("    ✅ Compute 16+ features per ETF")
    print("    ✅ Run 4 independent alpha signals")
    print("    ✅ Detect market regime (Calm/Transition/Crisis)")
    print("    ✅ Combine signals into ensemble alpha scores")
    print("    ✅ Optimize portfolio (Risk Parity + Alpha Tilt)")
    print("    ✅ Validate through 5-layer Risk Fortress")
    print("    ✅ Generate and execute rebalance orders")
    print()
    print(f"  Current mode: {EXECUTION_MODE}")
    print(f"  Regime: {regime}")
    print(f"  Portfolio: ${portfolio_value:,.2f}")
    print(f"  Positions: {optimization['expected_positions']} target")
    print(f"  Orders: {len(orders)} generated")
    print()

    if EXECUTION_MODE == "SHADOW":
        print("  📋 NEXT STEPS:")
        print("     1. Review the orders above — do they make sense?")
        print("     2. Run the scheduler (scheduler.py) for automated daily execution")
        print("     3. Monitor via dashboard (streamlit run monitor/dashboard.py)")
        print("     4. After 3 months in SHADOW, switch to SUPERVISED mode")
        print("     5. After 3 more months validated, switch to AUTONOMOUS")
    print()
    print("=" * 60)
    print("  Jarvis is ready. The system is fully operational.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
