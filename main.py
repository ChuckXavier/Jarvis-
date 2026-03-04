"""
JARVIS V2 - Main Entry Point
===============================
This is the "front door" of Jarvis. When you run this file, it:
1. Checks your configuration (API keys, database connection)
2. Creates database tables if they don't exist
3. Runs the data ingestion (downloads 10 years of ETF data)
4. Runs data quality checks
5. Prints a summary of what's in the database

HOW TO RUN (for non-coders):
- Railway runs this automatically when you deploy
- Or you can run it manually: python main.py
"""

import sys
import os
from loguru import logger

# Set up logging
os.makedirs("logs", exist_ok=True)
logger.add("logs/jarvis.log", rotation="10 MB", retention="30 days", level="INFO")


def main():
    """
    Main function — the complete Phase 1 pipeline.
    """
    print()
    print("=" * 60)
    print("  J A R V I S   V 2")
    print("  Autonomous ETF Alpha Engine")
    print("  Phase 1: Data Foundation")
    print("=" * 60)
    print()

    # ── Step 1: Validate Configuration ──
    print("STEP 1: Checking configuration...")
    from config.settings import validate_config

    missing = validate_config()
    if missing:
        print(f"\n  ❌ Missing environment variables: {missing}")
        print("  Set these in Railway dashboard → Variables tab")
        print("  Required variables:")
        print("    DATABASE_URL      → Auto-set by Railway PostgreSQL")
        print("    ALPACA_API_KEY    → From alpaca.markets dashboard")
        print("    ALPACA_SECRET_KEY → From alpaca.markets dashboard")
        print("    FRED_API_KEY      → From fred.stlouisfed.org")
        print()
        print("  Jarvis cannot start without these. Exiting.")
        sys.exit(1)

    print("  ✅ All configuration variables found!\n")

    # ── Step 2: Create Database Tables ──
    print("STEP 2: Setting up database tables...")
    from data.db import create_all_tables, get_data_summary, get_record_count

    try:
        create_all_tables()
        print("  ✅ Database tables ready!\n")
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        print("  Check your DATABASE_URL in Railway variables.")
        sys.exit(1)

    # ── Step 3: Check if we already have data ──
    print("STEP 3: Checking existing data...")
    try:
        count = get_record_count("daily_prices")
        print(f"  Found {count:,} existing price records in database.")
    except Exception:
        count = 0
        print("  No existing data found (fresh database).")

    # ── Step 4: Run Data Ingestion ──
    if count < 1000:
        print("\nSTEP 4: Running FULL data ingestion (10 years)...")
        print("  This will take 2-5 minutes. Downloading from Yahoo Finance + FRED...")
        print()
        from data.ingest import run_full_ingestion
        result = run_full_ingestion()
    else:
        print("\nSTEP 4: Running daily data UPDATE (incremental)...")
        from data.ingest import run_daily_update
        result = run_daily_update()
        # Also update FRED data
        from data.ingest import download_fred_data, download_vix_from_yahoo
        download_fred_data(years=1)
        download_vix_from_yahoo(years=1)

    # ── Step 5: Data Quality Check ──
    print("\nSTEP 5: Running data quality checks...")
    from data.quality import run_all_quality_checks, print_quality_report

    report = run_all_quality_checks()
    print_quality_report(report)

    # ── Step 6: Print Summary ──
    print("\nSTEP 6: Database Summary")
    print("-" * 60)
    try:
        summary = get_data_summary()
        if not summary.empty:
            print(f"\n  Total ETFs with data: {len(summary)}")
            print(f"  Total price records:  {get_record_count('daily_prices'):,}")
            try:
                macro_count = get_record_count("macro_data")
                print(f"  Total macro records:  {macro_count:,}")
            except Exception:
                print("  Total macro records:  0")
            print(f"\n  Date range: {summary['first_date'].min()} to {summary['last_date'].max()}")
            print(f"\n  Per-ETF breakdown:")
            for _, row in summary.iterrows():
                print(f"    {row['ticker']:6s}  {row['num_days']:5d} days  ({row['first_date']} to {row['last_date']})")
        else:
            print("  No data in database yet.")
    except Exception as e:
        print(f"  Could not read summary: {e}")

    # ── Done ──
    print()
    print("=" * 60)
    if report["passed"]:
        print("  ✅ PHASE 1 COMPLETE — Data Foundation is READY")
        print("  Jarvis has 10 years of clean ETF data.")
        print("  Next: Build the Feature Engine (Phase 2)")
    else:
        print("  ⚠️  PHASE 1 COMPLETE WITH WARNINGS")
        print("  Review the errors above before proceeding.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
