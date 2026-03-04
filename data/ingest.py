"""
JARVIS V2 - Data Ingestion Pipeline
======================================
Downloads historical ETF prices and macroeconomic data, then stores
them in the PostgreSQL database.

HOW THIS WORKS (for non-coders):
- This file connects to Yahoo Finance and FRED (Federal Reserve)
- It downloads 10 years of daily price data for all 25 ETFs
- It downloads key economic indicators (VIX, yield curve, credit spreads)
- All data is cleaned and saved into your database
- You run this ONCE to fill the historical database, then daily to update
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger

from config.settings import FRED_API_KEY, HISTORY_YEARS
from config.universe import (
    get_all_tickers, MACRO_TICKERS, FRED_SERIES
)
from data.db import save_daily_prices, save_macro_data, get_latest_date


# ============================================================
# ETF PRICE DATA INGESTION
# ============================================================

def download_etf_prices(tickers: list = None, years: int = None) -> dict:
    """
    Download daily OHLCV data for all ETFs from Yahoo Finance.

    Parameters:
        tickers: List of ticker symbols. Defaults to all 25 in the universe.
        years: How many years of history. Defaults to HISTORY_YEARS from settings.

    Returns:
        A dict with 'success' (list of tickers downloaded) and
        'failed' (list of tickers that had errors).

    WHAT HAPPENS STEP BY STEP:
    1. Calculates the start date (today minus 10 years)
    2. For each ETF, asks Yahoo Finance for daily data
    3. Cleans the data (removes gaps, handles missing values)
    4. Saves each ETF's data into the database
    """
    if tickers is None:
        tickers = get_all_tickers()
    if years is None:
        years = HISTORY_YEARS

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    logger.info(f"Downloading price data for {len(tickers)} ETFs from {start_date.date()} to {end_date.date()}")

    success = []
    failed = []

    for ticker in tickers:
        try:
            logger.info(f"  Downloading {ticker}...")

            # Download from Yahoo Finance
            data = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,  # We want both raw close and adjusted close
            )

            if data.empty:
                logger.warning(f"  {ticker}: No data returned from Yahoo Finance")
                failed.append(ticker)
                continue

            # Clean and format the data
            df = _clean_price_data(data, ticker)

            if df.empty:
                logger.warning(f"  {ticker}: Empty after cleaning")
                failed.append(ticker)
                continue

            # Save to database
            save_daily_prices(df)

            logger.info(f"  {ticker}: Saved {len(df)} days ({df['date'].min()} to {df['date'].max()})")
            success.append(ticker)

        except Exception as e:
            logger.error(f"  {ticker}: Error - {str(e)}")
            failed.append(ticker)

    logger.info(f"Download complete: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed tickers: {failed}")

    return {"success": success, "failed": failed}


def update_etf_prices(tickers: list = None) -> dict:
    """
    Incremental update: only download data from the last available date.
    Run this daily to keep the database current.
    """
    if tickers is None:
        tickers = get_all_tickers()

    end_date = datetime.now()
    success = []
    failed = []

    for ticker in tickers:
        try:
            # Find the most recent date we have
            latest = get_latest_date(ticker)

            if latest:
                # Start from the day after the last record
                start = pd.to_datetime(latest) + timedelta(days=1)
                if start.date() >= end_date.date():
                    logger.info(f"  {ticker}: Already up to date")
                    success.append(ticker)
                    continue
            else:
                # No data at all — do a full download
                start = end_date - timedelta(days=HISTORY_YEARS * 365)

            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )

            if data.empty:
                logger.info(f"  {ticker}: No new data available")
                success.append(ticker)
                continue

            df = _clean_price_data(data, ticker)
            if not df.empty:
                save_daily_prices(df)
                logger.info(f"  {ticker}: Updated with {len(df)} new days")

            success.append(ticker)

        except Exception as e:
            logger.error(f"  {ticker}: Update error - {str(e)}")
            failed.append(ticker)

    return {"success": success, "failed": failed}


def _clean_price_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean raw Yahoo Finance data into our standard format.

    Steps:
    1. Flatten multi-level column headers (Yahoo sometimes returns these)
    2. Rename columns to our standard names
    3. Remove rows with missing close prices
    4. Forward-fill small gaps (weekends/holidays are already excluded)
    5. Add the ticker column
    6. Format dates properly
    """
    df = data.copy()

    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize column names
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower == "open":
            col_map[col] = "open"
        elif col_lower == "high":
            col_map[col] = "high"
        elif col_lower == "low":
            col_map[col] = "low"
        elif col_lower == "close":
            col_map[col] = "close"
        elif "adj" in col_lower:
            col_map[col] = "adj_close"
        elif col_lower == "volume":
            col_map[col] = "volume"

    df = df.rename(columns=col_map)

    # Ensure we have the required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"  {ticker}: Missing column '{col}'")
            return pd.DataFrame()

    # If adj_close is missing, use close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # Remove rows where close is NaN or zero
    df = df[df["close"].notna() & (df["close"] > 0)]

    # Forward fill small gaps (max 3 days)
    df = df.ffill(limit=3)

    # Add ticker and format date
    df["ticker"] = ticker
    df = df.reset_index()

    # The index from yfinance is the date
    date_col = [c for c in df.columns if "date" in str(c).lower() or "Date" in str(c)]
    if date_col:
        df = df.rename(columns={date_col[0]: "date"})
    elif "index" in df.columns:
        df = df.rename(columns={"index": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Select only the columns we need
    keep_cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[keep_cols]

    # Remove any remaining NaN rows
    df = df.dropna(subset=["close"])

    return df


# ============================================================
# MACRO DATA INGESTION (FRED)
# ============================================================

def download_fred_data(years: int = None) -> dict:
    """
    Download macroeconomic indicators from the Federal Reserve (FRED).

    This includes: yield curves, credit spreads, VIX, unemployment, CPI, etc.
    These are the inputs for the regime detection model.
    """
    if years is None:
        years = HISTORY_YEARS

    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping macro data download")
        return {"success": [], "failed": list(FRED_SERIES.keys())}

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    success = []
    failed = []

    for series_id, description in FRED_SERIES.items():
        try:
            logger.info(f"  Downloading FRED: {series_id} ({description})")

            data = fred.get_series(
                series_id,
                observation_start=start_date.strftime("%Y-%m-%d"),
                observation_end=end_date.strftime("%Y-%m-%d"),
            )

            if data is None or data.empty:
                logger.warning(f"  {series_id}: No data from FRED")
                failed.append(series_id)
                continue

            # Convert to DataFrame format
            df = pd.DataFrame({
                "series_id": series_id,
                "date": data.index.date,
                "value": data.values,
            })

            # Remove NaN values
            df = df.dropna(subset=["value"])

            if not df.empty:
                save_macro_data(df)
                logger.info(f"  {series_id}: Saved {len(df)} observations")
                success.append(series_id)
            else:
                failed.append(series_id)

        except Exception as e:
            logger.error(f"  {series_id}: Error - {str(e)}")
            failed.append(series_id)

    logger.info(f"FRED download complete: {len(success)} succeeded, {len(failed)} failed")
    return {"success": success, "failed": failed}


def download_vix_from_yahoo(years: int = None) -> bool:
    """
    Download VIX index data from Yahoo Finance as a backup
    (in case FRED's VIXCLS has gaps).
    """
    if years is None:
        years = HISTORY_YEARS

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        data = yf.download(
            "^VIX",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
        )

        if data.empty:
            logger.warning("VIX: No data from Yahoo Finance")
            return False

        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Find the close column
        close_col = None
        for col in data.columns:
            if "close" in str(col).lower():
                close_col = col
                break

        if close_col is None:
            logger.warning("VIX: Could not find close column")
            return False

        df = pd.DataFrame({
            "series_id": "VIX_YAHOO",
            "date": data.index.date,
            "value": data[close_col].values,
        })
        df = df.dropna(subset=["value"])

        if not df.empty:
            save_macro_data(df)
            logger.info(f"VIX (Yahoo): Saved {len(df)} observations")
            return True

    except Exception as e:
        logger.error(f"VIX download error: {str(e)}")

    return False


# ============================================================
# MASTER INGESTION FUNCTION
# ============================================================

def run_full_ingestion():
    """
    Run the complete data ingestion pipeline:
    1. Download all ETF prices (10 years)
    2. Download all FRED macro data
    3. Download VIX from Yahoo as backup

    This is what you run ONCE to fill the database from scratch.
    """
    logger.info("=" * 60)
    logger.info("JARVIS V2 - FULL DATA INGESTION STARTING")
    logger.info("=" * 60)

    # Step 1: ETF Prices
    logger.info("\n--- STEP 1: Downloading ETF Prices ---")
    etf_result = download_etf_prices()

    # Step 2: FRED Macro Data
    logger.info("\n--- STEP 2: Downloading FRED Macro Data ---")
    fred_result = download_fred_data()

    # Step 3: VIX Backup
    logger.info("\n--- STEP 3: Downloading VIX from Yahoo ---")
    vix_ok = download_vix_from_yahoo()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"ETFs:  {len(etf_result['success'])}/25 succeeded")
    logger.info(f"FRED:  {len(fred_result['success'])}/{len(FRED_SERIES)} series succeeded")
    logger.info(f"VIX:   {'OK' if vix_ok else 'Failed'}")

    if etf_result["failed"]:
        logger.warning(f"Failed ETFs: {etf_result['failed']}")
    if fred_result["failed"]:
        logger.warning(f"Failed FRED: {fred_result['failed']}")

    return {
        "etf": etf_result,
        "fred": fred_result,
        "vix": vix_ok,
    }


def run_daily_update():
    """
    Run the daily incremental update.
    Only downloads new data since the last available date.

    This is what the scheduler runs every morning at 6:00 AM ET.
    """
    logger.info("JARVIS V2 - Daily data update starting...")

    etf_result = update_etf_prices()
    fred_result = download_fred_data(years=1)  # Only need recent macro data
    vix_ok = download_vix_from_yahoo(years=1)

    logger.info(f"Daily update complete: {len(etf_result['success'])} ETFs updated")
    return etf_result
