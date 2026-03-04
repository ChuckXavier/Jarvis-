"""
JARVIS V2 - Database Layer
============================
Creates and manages the PostgreSQL database that stores all market data.

HOW THIS WORKS (for non-coders):
- This file creates the "tables" (think: spreadsheets) inside your database
- Table 1: daily_prices  → stores daily price data for all 25 ETFs
- Table 2: macro_data    → stores economic indicators (VIX, yields, etc.)
- Table 3: features      → stores computed trading features
- Table 4: signals       → stores signal scores
- Table 5: trades        → stores every trade Jarvis makes
- Table 6: portfolio     → stores daily portfolio snapshots
"""

import pandas as pd
from sqlalchemy import (
    create_engine, text, Column, String, Float, Date, DateTime,
    Integer, BigInteger, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker
from loguru import logger

from config.settings import DATABASE_URL

# ============================================================
# DATABASE ENGINE
# ============================================================
# This is the "connection" to your PostgreSQL database on Railway
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ============================================================
# TABLE DEFINITIONS
# ============================================================

class DailyPrice(Base):
    """
    Stores daily OHLCV (Open, High, Low, Close, Volume) data for each ETF.
    Think of this as a giant spreadsheet with one row per ETF per day.
    """
    __tablename__ = "daily_prices"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(BigInteger)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uix_ticker_date"),
        Index("ix_daily_prices_ticker", "ticker"),
        Index("ix_daily_prices_date", "date"),
    )


class MacroData(Base):
    """
    Stores macroeconomic indicators from FRED
    (yield curves, credit spreads, VIX, unemployment, etc.)
    """
    __tablename__ = "macro_data"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    series_id = Column(String(50), nullable=False)
    date = Column(Date, nullable=False)
    value = Column(Float)

    __table_args__ = (
        UniqueConstraint("series_id", "date", name="uix_macro_series_date"),
        Index("ix_macro_series", "series_id"),
    )


class FeatureStore(Base):
    """
    Stores computed trading features for each ETF on each day.
    These are the numbers the signal models use to make decisions.
    """
    __tablename__ = "features"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    feature_name = Column(String(50), nullable=False)
    value = Column(Float)

    __table_args__ = (
        UniqueConstraint("ticker", "date", "feature_name", name="uix_feature"),
        Index("ix_features_ticker_date", "ticker", "date"),
    )


class SignalScore(Base):
    """
    Stores signal scores from each of the 4 core signals + ensemble.
    """
    __tablename__ = "signals"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False)
    date = Column(Date, nullable=False)
    signal_name = Column(String(50), nullable=False)
    score = Column(Float)

    __table_args__ = (
        UniqueConstraint("ticker", "date", "signal_name", name="uix_signal"),
    )


class TradeLog(Base):
    """
    Stores every trade Jarvis executes (paper or live).
    Complete audit trail.
    """
    __tablename__ = "trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    ticker = Column(String(10), nullable=False)
    side = Column(String(4), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    order_type = Column(String(10))  # 'limit' or 'market'
    status = Column(String(20))  # PENDING, FILLED, CANCELLED, etc.
    alpaca_order_id = Column(String(100))
    signal_source = Column(String(50))  # Which signal triggered this


class PortfolioSnapshot(Base):
    """
    Stores daily portfolio snapshots for tracking performance.
    """
    __tablename__ = "portfolio_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    total_value = Column(Float)
    cash = Column(Float)
    invested = Column(Float)
    daily_pnl = Column(Float)
    daily_return_pct = Column(Float)
    cumulative_return_pct = Column(Float)
    max_drawdown_pct = Column(Float)
    num_positions = Column(Integer)
    sharpe_ratio = Column(Float)


# ============================================================
# DATABASE SETUP
# ============================================================

def create_all_tables():
    """
    Creates all tables in the database if they don't already exist.
    Run this once when first setting up Jarvis.
    """
    logger.info("Creating database tables...")
    Base.metadata.create_all(engine)
    logger.info("All tables created successfully!")


def drop_all_tables():
    """
    WARNING: Drops all tables. Only use if you want to start completely fresh.
    """
    logger.warning("DROPPING ALL TABLES - this deletes all data!")
    Base.metadata.drop_all(engine)
    logger.info("All tables dropped.")


# ============================================================
# DATA ACCESS HELPERS
# ============================================================

def save_daily_prices(df: pd.DataFrame):
    """
    Save a DataFrame of daily prices to the database.
    The DataFrame must have columns: ticker, date, open, high, low, close, adj_close, volume

    Uses 'upsert' logic: if the data already exists for that ticker+date,
    it updates it. If not, it inserts a new row.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to save_daily_prices")
        return

    records = df.to_dict("records")
    with engine.begin() as conn:
        for record in records:
            conn.execute(text("""
                INSERT INTO daily_prices (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
                ON CONFLICT ON CONSTRAINT uix_ticker_date
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adj_close = EXCLUDED.adj_close,
                    volume = EXCLUDED.volume
            """), record)

    logger.info(f"Saved {len(records)} price records to database")


def save_macro_data(df: pd.DataFrame):
    """
    Save a DataFrame of macro data to the database.
    DataFrame must have columns: series_id, date, value
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to save_macro_data")
        return

    records = df.to_dict("records")
    with engine.begin() as conn:
        for record in records:
            conn.execute(text("""
                INSERT INTO macro_data (series_id, date, value)
                VALUES (:series_id, :date, :value)
                ON CONFLICT ON CONSTRAINT uix_macro_series_date
                DO UPDATE SET value = EXCLUDED.value
            """), record)

    logger.info(f"Saved {len(records)} macro records to database")


def get_prices(ticker: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Retrieve daily prices for a single ticker from the database.
    Returns a DataFrame indexed by date.
    """
    query = "SELECT * FROM daily_prices WHERE ticker = :ticker"
    params = {"ticker": ticker}

    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND date <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY date"

    df = pd.read_sql(text(query), engine, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def get_all_prices(start_date=None, end_date=None) -> pd.DataFrame:
    """
    Retrieve adjusted close prices for ALL ETFs, pivoted so each column
    is a ticker. This is the main format used by the signal engine.

    Returns a DataFrame where:
    - Index = date
    - Columns = ticker symbols (SPY, QQQ, IWM, ...)
    - Values = adjusted close prices
    """
    query = "SELECT date, ticker, adj_close FROM daily_prices WHERE 1=1"
    params = {}

    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND date <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY date"

    df = pd.read_sql(text(query), engine, params=params)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="adj_close")
    return pivot


def get_macro(series_id: str, start_date=None, end_date=None) -> pd.DataFrame:
    """Retrieve a single macro series from the database."""
    query = "SELECT date, value FROM macro_data WHERE series_id = :series_id"
    params = {"series_id": series_id}

    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND date <= :end_date"
        params["end_date"] = end_date

    query += " ORDER BY date"

    df = pd.read_sql(text(query), engine, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def get_record_count(table_name: str) -> int:
    """Get the total number of rows in a table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.scalar()


def get_latest_date(ticker: str) -> str:
    """Get the most recent date we have data for a given ticker."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(date) FROM daily_prices WHERE ticker = :ticker"),
            {"ticker": ticker}
        )
        val = result.scalar()
        return str(val) if val else None


def get_data_summary() -> pd.DataFrame:
    """
    Returns a summary of what data we have in the database.
    Shows each ticker, its earliest date, latest date, and row count.
    Useful for checking data quality at a glance.
    """
    query = """
        SELECT
            ticker,
            MIN(date) as first_date,
            MAX(date) as last_date,
            COUNT(*) as num_days
        FROM daily_prices
        GROUP BY ticker
        ORDER BY ticker
    """
    return pd.read_sql(text(query), engine)
