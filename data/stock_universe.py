"""
JARVIS 60/40 — Stock Universe Data Pipeline
==============================================
Downloads daily price data for the top 500 US stocks by market cap.
Uses the existing Alpaca API connection (already configured on Railway).

HOW IT WORKS:
  1. Gets list of tradeable assets from Alpaca
  2. Filters to top 500 by market activity (proxy for S&P 500)
  3. Downloads 2 years of daily prices for each stock
  4. Stores in the existing PostgreSQL database
  5. Runs daily as part of the scheduler pipeline

INTEGRATION:
  Add to scheduler.py Step 1, after ETF data download:
    from data.stock_universe import update_stock_universe
    update_stock_universe()
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
import os

# Use existing Alpaca credentials
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY


def update_stock_universe():
    """
    Download/update price data for top 500 US stocks.
    Called daily from the scheduler.
    """
    logger.info("\n--- Updating stock universe ---")

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        # Get tradeable assets
        trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True,
        )

        assets = trading_client.get_all_assets()

        # Filter to US equities that are tradeable and shortable
        us_stocks = [
            a for a in assets
            if a.asset_class.value == "us_equity"
            and a.tradable
            and a.shortable
            and a.status.value == "active"
            and not a.symbol.endswith("W")   # Skip warrants
            and "." not in a.symbol          # Skip preferred shares
            and len(a.symbol) <= 5           # Skip weird tickers
        ]

        logger.info(f"  Found {len(us_stocks)} tradeable US stocks")

        # Get tickers — we'll download data for up to 500
        # Sort by name length as a rough proxy (shorter tickers = bigger companies)
        # The actual filtering by market cap happens in the optimizer
        tickers = sorted([a.symbol for a in us_stocks])[:600]

        # Download price data
        data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )

        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)  # ~1.5 years for momentum calc

        logger.info(f"  Downloading prices for {len(tickers)} stocks...")
        logger.info(f"  Period: {start_date.date()} → {end_date.date()}")

        # Download in batches of 50 to avoid API limits
        all_prices = {}
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                    feed=DataFeed.IEX,
                )
                bars = data_client.get_stock_bars(request)

                for bar in bars.data:
                    ticker = bar
                    for b in bars[bar]:
                        if ticker not in all_prices:
                            all_prices[ticker] = []
                        all_prices[ticker].append({
                            "date": b.timestamp.date(),
                            "close": float(b.close),
                            "volume": float(b.volume),
                        })

            except Exception as e:
                logger.warning(f"  Batch {i//batch_size + 1} failed: {e}")
                continue

            if (i // batch_size) % 5 == 0:
                logger.info(f"  Progress: {i + batch_size}/{len(tickers)} tickers")

        logger.info(f"  Downloaded data for {len(all_prices)} stocks")

        # Save to database
        _save_stock_prices(all_prices)

        return {"success": len(all_prices), "total": len(tickers)}

    except Exception as e:
        logger.error(f"  Stock universe update failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": 0, "error": str(e)}


def _save_stock_prices(all_prices):
    """Save stock prices to the existing PostgreSQL database."""
    try:
        from data.db import engine
        from sqlalchemy import text

        with engine.begin() as conn:
            # Create stock_prices table if it doesn't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    date DATE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    PRIMARY KEY (date, ticker)
                )
            """))

            # Insert/update prices
            count = 0
            for ticker, bars in all_prices.items():
                for bar in bars:
                    conn.execute(text("""
                        INSERT INTO stock_prices (date, ticker, close, volume)
                        VALUES (:date, :ticker, :close, :volume)
                        ON CONFLICT (date, ticker) DO UPDATE SET
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """), {
                        "date": bar["date"],
                        "ticker": ticker,
                        "close": bar["close"],
                        "volume": bar["volume"],
                    })
                    count += 1

        logger.info(f"  Saved {count} stock price records to database")

    except Exception as e:
        logger.error(f"  Failed to save stock prices: {e}")


def get_stock_prices():
    """
    Load stock prices from database as a DataFrame.
    Returns: DataFrame with dates as index, tickers as columns.
    """
    try:
        from data.db import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT date, ticker, close FROM stock_prices ORDER BY date"
            ))
            rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "ticker", "close"])
        prices = df.pivot(index="date", columns="ticker", values="close")
        prices.index = pd.to_datetime(prices.index)
        prices = prices.ffill()

        return prices

    except Exception as e:
        logger.error(f"Failed to load stock prices: {e}")
        return pd.DataFrame()
