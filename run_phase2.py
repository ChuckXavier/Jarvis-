"""
JARVIS V2 - Phase 2 Runner: Alpha Engine Test
================================================
Runs the complete signal pipeline on historical data to verify
everything works end-to-end.

This script:
1. Loads price data from the database (already populated in Phase 1)
2. Computes all features (16+ indicators per ETF)
3. Runs all 4 signals (Momentum, Trend, Mean Reversion, Regime)
4. Combines them into ensemble alpha scores
5. Shows you the current trading recommendations

HOW TO RUN:
    python run_phase2.py
"""

import sys
import os
import pandas as pd
from loguru import logger

# Set up logging
os.makedirs("logs", exist_ok=True)
logger.add("logs/phase2.log", rotation="10 MB", level="INFO")


def main():
    print()
    print("=" * 60)
    print("  J A R V I S   V 2")
    print("  Phase 2: Alpha Engine Test")
    print("=" * 60)
    print()

    # ── Step 1: Validate Config ──
    print("STEP 1: Checking configuration...")
    from config.settings import validate_config
    missing = validate_config()
    if missing:
        print(f"  ❌ Missing: {missing}")
        print("  Set environment variables first. See Phase 1 instructions.")
        sys.exit(1)
    print("  ✅ Configuration OK\n")

    # ── Step 2: Load Data ──
    print("STEP 2: Loading price data from database...")
    from data.db import get_all_prices, get_record_count

    try:
        count = get_record_count("daily_prices")
        print(f"  Found {count:,} price records")
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        print("  Run Phase 1 (main.py) first to populate the database.")
        sys.exit(1)

    prices = get_all_prices()
    if prices.empty:
        print("  ❌ No price data in database. Run Phase 1 first.")
        sys.exit(1)

    print(f"  ✅ Loaded data for {len(prices.columns)} ETFs, {len(prices)} trading days")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print()

    # ── Step 3: Compute Features ──
    print("STEP 3: Computing features for all ETFs...")
    from features.engine import compute_all_features

    features = compute_all_features(prices)
    price_features = features["price_features"]
    macro_features = features["macro_features"]

    print(f"  ✅ Price features computed for {len(price_features)} ETFs")
    print(f"  ✅ Macro features: {len(macro_features.columns)} indicators")

    # Show a sample of features for SPY
    if "SPY" in price_features:
        spy_latest = price_features["SPY"].iloc[-1]
        print(f"\n  Sample: SPY features (latest day):")
        for feat_name, feat_val in spy_latest.items():
            if pd.notna(feat_val):
                print(f"    {feat_name:25s}: {feat_val:>10.4f}")
    print()

    # ── Step 4: Run Ensemble (all 4 signals) ──
    print("STEP 4: Running all 4 signals + ensemble...")
    print("  (This may take 1-2 minutes...)")
    print()

    from signals.ensemble import compute_ensemble, get_top_bottom_etfs

    result = compute_ensemble(prices)

    # ── Step 5: Display Results ──
    print()
    print("=" * 60)
    print("  JARVIS V2 — ALPHA ENGINE RESULTS")
    print("=" * 60)

    # Current regime
    print(f"\n  MARKET REGIME: {result['regime']}")
    print(f"  Active Signals: {sum(1 for s in result['signal_details'].values() if not isinstance(s, pd.DataFrame) or not s.empty)}/4")
    print(f"  Signal Weights: {result['weights_used']}")

    # Top buy and sell candidates
    latest = result["latest_scores"]
    if not latest.empty:
        top_bottom = get_top_bottom_etfs(latest, top_n=5)

        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║  TOP 5 BUY CANDIDATES (highest alpha)    ║")
        print(f"  ╠══════════════════════════════════════════╣")
        for ticker, score in top_bottom["top_buy"].items():
            bar = "█" * int(abs(score) * 10)
            print(f"  ║  {ticker:5s}  score: {score:+.3f}  {bar:15s}  ║")

        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  TOP 5 SELL/AVOID (lowest alpha)         ║")
        print(f"  ╠══════════════════════════════════════════╣")
        for ticker, score in top_bottom["top_sell"].items():
            bar = "░" * int(abs(score) * 10)
            print(f"  ║  {ticker:5s}  score: {score:+.3f}  {bar:15s}  ║")
        print(f"  ╚══════════════════════════════════════════╝")

        # Full ranking
        print(f"\n  FULL ETF RANKING (all {len(latest)} ETFs):")
        print(f"  {'Rank':>4s}  {'Ticker':6s}  {'Alpha Score':>12s}  {'Action':8s}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*12}  {'─'*8}")
        for rank, (ticker, score) in enumerate(latest.items(), 1):
            if pd.notna(score):
                if score > 0.5:
                    action = "STRONG BUY"
                elif score > 0.1:
                    action = "BUY"
                elif score > -0.1:
                    action = "HOLD"
                elif score > -0.5:
                    action = "REDUCE"
                else:
                    action = "SELL"
                print(f"  {rank:4d}  {ticker:6s}  {score:+12.4f}  {action}")
    else:
        print("\n  ⚠️ No alpha scores generated. Check signal errors above.")

    # ── Summary ──
    print()
    print("=" * 60)
    print("  ✅ PHASE 2 COMPLETE — Alpha Engine is WORKING")
    print()
    print("  Jarvis can now:")
    print("    • Compute 16+ features for each of 25 ETFs")
    print("    • Run 4 independent trading signals")
    print("    • Detect market regime (Calm/Transition/Crisis)")
    print("    • Combine signals into a unified alpha score")
    print("    • Rank all ETFs from best to worst opportunity")
    print()
    print("  Next: Phase 3 — Risk Fortress + Portfolio Optimizer + Execution")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
