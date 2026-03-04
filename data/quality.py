"""
JARVIS V2 - Data Quality Checks
==================================
Validates that the data in the database is clean, complete, and ready
for the signal engine.

HOW THIS WORKS (for non-coders):
- This is like a "health check" for your data
- It looks for problems: missing days, weird price spikes, stale data
- If data fails quality checks, Jarvis won't trade that day (safety first)
- Think of it as the "preflight checklist" before every trading day
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from config.universe import get_all_tickers
from data.db import get_all_prices, get_data_summary, get_prices


def run_all_quality_checks() -> dict:
    """
    Run every quality check and return a comprehensive report.

    Returns:
        {
            "passed": True/False (overall),
            "checks": {
                "completeness": {...},
                "freshness": {...},
                "outliers": {...},
                "volume": {...},
            },
            "warnings": [list of warning messages],
            "errors": [list of error messages],
        }
    """
    logger.info("Running data quality checks...")

    warnings = []
    errors = []

    # Check 1: Completeness
    completeness = check_completeness()
    warnings.extend(completeness.get("warnings", []))
    errors.extend(completeness.get("errors", []))

    # Check 2: Freshness
    freshness = check_freshness()
    warnings.extend(freshness.get("warnings", []))
    errors.extend(freshness.get("errors", []))

    # Check 3: Outliers / Price Spikes
    outliers = check_outliers()
    warnings.extend(outliers.get("warnings", []))
    errors.extend(outliers.get("errors", []))

    # Check 4: Volume anomalies
    volume = check_volume()
    warnings.extend(volume.get("warnings", []))

    # Overall pass/fail
    passed = len(errors) == 0

    report = {
        "passed": passed,
        "checks": {
            "completeness": completeness,
            "freshness": freshness,
            "outliers": outliers,
            "volume": volume,
        },
        "warnings": warnings,
        "errors": errors,
    }

    if passed:
        logger.info(f"Data quality: PASSED ({len(warnings)} warnings)")
    else:
        logger.error(f"Data quality: FAILED ({len(errors)} errors, {len(warnings)} warnings)")

    return report


def check_completeness() -> dict:
    """
    Check that all 25 ETFs have data, and none have large gaps.
    """
    summary = get_data_summary()
    all_tickers = set(get_all_tickers())
    warnings = []
    errors = []

    if summary.empty:
        errors.append("DATABASE IS EMPTY - no price data found. Run data ingestion first.")
        return {"passed": False, "errors": errors, "warnings": warnings}

    present_tickers = set(summary["ticker"].values)
    missing = all_tickers - present_tickers

    if len(missing) > 2:
        errors.append(f"More than 2 ETFs missing from database: {missing}")
    elif missing:
        warnings.append(f"ETFs missing from database: {missing}")

    # Check for minimum history length (need at least 252 trading days = ~1 year)
    for _, row in summary.iterrows():
        if row["num_days"] < 252:
            warnings.append(f"{row['ticker']}: Only {row['num_days']} days of data (need 252+)")

    passed = len(errors) == 0
    return {"passed": passed, "errors": errors, "warnings": warnings, "summary": summary.to_dict("records") if not summary.empty else []}


def check_freshness(max_stale_days: int = 5) -> dict:
    """
    Check that data is recent (not more than a few days old).
    Accounts for weekends and holidays.
    """
    summary = get_data_summary()
    warnings = []
    errors = []

    if summary.empty:
        errors.append("No data to check freshness")
        return {"passed": False, "errors": errors, "warnings": warnings}

    today = datetime.now().date()

    stale_tickers = []
    for _, row in summary.iterrows():
        last_date = pd.to_datetime(row["last_date"]).date()
        days_old = (today - last_date).days

        if days_old > max_stale_days:
            stale_tickers.append(f"{row['ticker']} (last: {last_date}, {days_old} days old)")

    if len(stale_tickers) > 5:
        errors.append(f"More than 5 ETFs have stale data: {stale_tickers[:5]}...")
    elif stale_tickers:
        warnings.append(f"Stale data for: {stale_tickers}")

    passed = len(errors) == 0
    return {"passed": passed, "errors": errors, "warnings": warnings}


def check_outliers(max_daily_move_pct: float = 0.20) -> dict:
    """
    Check for suspicious price spikes (> 20% daily move).
    These could indicate data errors, stock splits not adjusted, etc.
    """
    warnings = []
    errors = []

    prices = get_all_prices()
    if prices.empty:
        errors.append("No price data for outlier check")
        return {"passed": False, "errors": errors, "warnings": warnings}

    # Calculate daily returns
    returns = prices.pct_change()

    # Find extreme moves
    outlier_count = 0
    for ticker in returns.columns:
        extreme = returns[ticker].abs() > max_daily_move_pct
        n_extreme = extreme.sum()
        if n_extreme > 0:
            outlier_count += n_extreme
            worst_date = returns[ticker].abs().idxmax()
            worst_return = returns[ticker].loc[worst_date]
            warnings.append(
                f"{ticker}: {n_extreme} days with >{max_daily_move_pct*100}% move. "
                f"Worst: {worst_return:.1%} on {worst_date.date()}"
            )

    passed = True  # Outliers are warnings, not errors (they could be legitimate)
    return {"passed": passed, "errors": errors, "warnings": warnings, "total_outliers": outlier_count}


def check_volume(min_avg_volume: int = 500_000) -> dict:
    """
    Check that ETFs have adequate trading volume.
    Low volume means wide spreads and difficulty executing.
    """
    warnings = []

    tickers = get_all_tickers()
    low_volume = []

    for ticker in tickers:
        df = get_prices(ticker)
        if df.empty:
            continue

        # Check last 21 days of volume
        recent_vol = df["volume"].tail(21).mean()
        if recent_vol < min_avg_volume:
            low_volume.append(f"{ticker}: avg volume {recent_vol:,.0f} (need {min_avg_volume:,}+)")

    if low_volume:
        warnings.append(f"Low volume ETFs (may have wide spreads): {low_volume}")

    return {"passed": True, "warnings": warnings}


def print_quality_report(report: dict):
    """Print a human-readable quality report to the console."""
    print("\n" + "=" * 60)
    print("  JARVIS V2 - DATA QUALITY REPORT")
    print("=" * 60)

    status = "✅ PASSED" if report["passed"] else "❌ FAILED"
    print(f"\n  Overall Status: {status}")

    if report["errors"]:
        print(f"\n  ❌ ERRORS ({len(report['errors'])}):")
        for err in report["errors"]:
            print(f"     • {err}")

    if report["warnings"]:
        print(f"\n  ⚠️  WARNINGS ({len(report['warnings'])}):")
        for warn in report["warnings"]:
            print(f"     • {warn}")

    if not report["errors"] and not report["warnings"]:
        print("\n  All checks passed with no warnings. Data is clean!")

    print("\n" + "=" * 60)
