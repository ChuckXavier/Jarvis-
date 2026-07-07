"""
JARVIS V3 - Data Ingestion Pipeline (600+ tickers)
===================================================
ARCHITECTURAL NOTE (what changed vs V2):
V2 downloaded one ticker per yfinance call and wrote one DB row per INSERT —
both fatal at 500+ names (slow, rate-limited, and a 1.2M-row backfill done
row-by-row). V3 batches downloads (INGEST_CHUNK_SIZE tickers per call, with
retry + exponential backoff), upserts in bulk via SQLAlchemy executemany, and
adds `get_prices_for_universe()` — the single function the signal/portfolio
layer uses to load a clean (date x ticker) matrix, with optional volume, in
one SQL query. Daily updates resume from MAX(date) per ticker computed in ONE
GROUP BY query instead of 600 round trips; the upsert makes overlap harmless.
Yahoo Finance remains the primary historical source (the Alpaca free tier is
IEX-only and is used for live quotes, not history).
"""

import time
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

from config.settings import (
    FRED_API_KEY, HISTORY_YEARS, PRICE_LOOKBACK_DAYS,
    INGEST_CHUNK_SIZE, INGEST_MAX_RETRIES, INGEST_RETRY_BASE_SLEEP,
    INGEST_CHUNK_PAUSE,
)
from config.universe import get_all_tickers, get_full_universe, FRED_SERIES


# ============================================================
# BULK DATABASE WRITES
# ============================================================

def _save_prices_bulk(records: list[dict]) -> int:
    """Upsert price records in batches of 5000 via executemany."""
    if not records:
        return 0
    from sqlalchemy import text
    from data.db import engine
    stmt = text("""
        INSERT INTO daily_prices (ticker, date, open, high, low, close, adj_close, volume)
        VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
        ON CONFLICT ON CONSTRAINT uix_ticker_date DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume
    """)
    written = 0
    with engine.begin() as conn:
        for i in range(0, len(records), 5000):
            chunk = records[i:i + 5000]
            conn.execute(stmt, chunk)
            written += len(chunk)
    return written


def _save_macro_bulk(records: list[dict]) -> int:
    if not records:
        return 0
    from sqlalchemy import text
    from data.db import engine
    stmt = text("""
        INSERT INTO macro_data (series_id, date, value)
        VALUES (:series_id, :date, :value)
        ON CONFLICT ON CONSTRAINT uix_macro_series_date
        DO UPDATE SET value = EXCLUDED.value
    """)
    with engine.begin() as conn:
        for i in range(0, len(records), 5000):
            conn.execute(stmt, records[i:i + 5000])
    return len(records)


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _frame_to_records(df: pd.DataFrame, ticker: str) -> list[dict]:
    """Clean one ticker's OHLCV frame into upsert-ready dicts."""
    if df is None or df.empty:
        return []
    f = df.copy()
    if isinstance(f.columns, pd.MultiIndex):
        f.columns = f.columns.get_level_values(0)
    rename = {}
    for c in f.columns:
        cl = str(c).lower().strip()
        if cl == "open":
            rename[c] = "open"
        elif cl == "high":
            rename[c] = "high"
        elif cl == "low":
            rename[c] = "low"
        elif cl == "close":
            rename[c] = "close"
        elif "adj" in cl:
            rename[c] = "adj_close"
        elif cl == "volume":
            rename[c] = "volume"
    f = f.rename(columns=rename)
    if "close" not in f.columns:
        return []
    if "adj_close" not in f.columns:
        f["adj_close"] = f["close"]
    for col in ("open", "high", "low", "volume"):
        if col not in f.columns:
            f[col] = np.nan
    f = f[f["close"].notna() & (f["close"] > 0)]
    if f.empty:
        return []
    f = f.reset_index()
    date_col = next((c for c in f.columns if "date" in str(c).lower()), None)
    if date_col is None:
        return []
    out = []
    for _, r in f.iterrows():
        try:
            vol = r["volume"]
            out.append({
                "ticker": ticker,
                "date": pd.to_datetime(r[date_col]).date(),
                "open": float(r["open"]) if pd.notna(r["open"]) else None,
                "high": float(r["high"]) if pd.notna(r["high"]) else None,
                "low": float(r["low"]) if pd.notna(r["low"]) else None,
                "close": float(r["close"]),
                "adj_close": float(r["adj_close"]) if pd.notna(r["adj_close"]) else float(r["close"]),
                "volume": int(vol) if pd.notna(vol) else 0,
            })
        except (ValueError, TypeError):
            continue
    return out


def _download_batch(tickers: list[str], start: str, end: str) -> dict[str, list[dict]]:
    """
    One batched yfinance call for a chunk of tickers, with retry/backoff.
    Returns {ticker: records}. Failed tickers simply absent from the dict.
    """
    last_err = None
    for attempt in range(1, INGEST_MAX_RETRIES + 1):
        try:
            data = yf.download(
                tickers, start=start, end=end,
                progress=False, auto_adjust=False,
                group_by="ticker", threads=True,
            )
            if data is None or data.empty:
                raise ValueError("empty response")
            out: dict[str, list[dict]] = {}
            if len(tickers) == 1:
                out[tickers[0]] = _frame_to_records(data, tickers[0])
            else:
                top = set(data.columns.get_level_values(0))
                for t in tickers:
                    if t in top:
                        out[t] = _frame_to_records(data[t].dropna(how="all"), t)
            return out
        except Exception as e:
            last_err = e
            sleep = INGEST_RETRY_BASE_SLEEP * (2 ** (attempt - 1)) + random.random()
            logger.warning(f"  batch attempt {attempt}/{INGEST_MAX_RETRIES} failed "
                           f"({e}); retrying in {sleep:.1f}s")
            time.sleep(sleep)
    logger.error(f"  batch FAILED after {INGEST_MAX_RETRIES} attempts: {last_err}")
    return {}


# ============================================================
# FULL BACKFILL + INCREMENTAL UPDATE
# ============================================================

def download_universe_history(tickers: list[str] | None = None,
                              years: int | None = None) -> dict:
    """Full-history backfill for a ticker list, in batches. Run once / rarely."""
    if tickers is None:
        tickers = get_full_universe()
    if years is None:
        years = HISTORY_YEARS
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    logger.info(f"Backfilling {len(tickers)} tickers, {years}y, "
                f"chunks of {INGEST_CHUNK_SIZE}")
    success, failed, rows = [], [], 0
    for chunk in _chunks(tickers, INGEST_CHUNK_SIZE):
        got = _download_batch(chunk, s, e)
        for t in chunk:
            recs = got.get(t, [])
            if recs:
                rows += _save_prices_bulk(recs)
                success.append(t)
            else:
                failed.append(t)
        logger.info(f"  progress: {len(success)} ok / {len(failed)} failed "
                    f"/ {rows:,} rows")
        time.sleep(INGEST_CHUNK_PAUSE)
    logger.info(f"Backfill complete: {len(success)} ok, {len(failed)} failed, "
                f"{rows:,} rows written")
    if failed:
        logger.warning(f"Failed tickers ({len(failed)}): {failed[:25]}"
                       f"{' ...' if len(failed) > 25 else ''}")
    return {"success": success, "failed": failed, "rows": rows}


def _latest_dates(tickers: list[str]) -> dict[str, datetime.date]:
    """MAX(date) per ticker in ONE query (not 600 round trips)."""
    from sqlalchemy import text, bindparam
    from data.db import engine
    stmt = text(
        "SELECT ticker, MAX(date) FROM daily_prices "
        "WHERE ticker IN :ts GROUP BY ticker"
    ).bindparams(bindparam("ts", expanding=True))
    with engine.begin() as conn:
        rows = conn.execute(stmt, {"ts": list(tickers)}).fetchall()
    return {r[0]: r[1] for r in rows}


def update_universe_prices(tickers: list[str] | None = None) -> dict:
    """
    Incremental daily update. Existing tickers re-fetch a 7-day overlap window
    (idempotent upsert makes the overlap free and restart-proof); brand-new
    tickers get a full backfill.
    """
    if tickers is None:
        tickers = get_full_universe()
    latest = _latest_dates(tickers)
    have = [t for t in tickers if t in latest]
    new = [t for t in tickers if t not in latest]

    end = datetime.now()
    results = {"success": [], "failed": [], "rows": 0}

    if have:
        overall_max = max(latest.values())
        start = pd.Timestamp(overall_max) - timedelta(days=7)
        if start.date() < end.date():
            s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            logger.info(f"Updating {len(have)} tickers from {s} (7d overlap)")
            for chunk in _chunks(have, INGEST_CHUNK_SIZE):
                got = _download_batch(chunk, s, e)
                for t in chunk:
                    recs = got.get(t, [])
                    if recs:
                        results["rows"] += _save_prices_bulk(recs)
                    # Empty response on an update window is normal (holiday/
                    # weekend), not a failure.
                    results["success"].append(t)
                time.sleep(INGEST_CHUNK_PAUSE)
        else:
            results["success"].extend(have)
            logger.info("Prices already current")

    if new:
        logger.info(f"{len(new)} new tickers — running full backfill for them")
        back = download_universe_history(new)
        results["success"].extend(back["success"])
        results["failed"].extend(back["failed"])
        results["rows"] += back["rows"]

    logger.info(f"Daily price update: {len(results['success'])} ok, "
                f"{len(results['failed'])} failed, {results['rows']:,} rows")
    return results


# ============================================================
# UNIVERSE PRICE MATRIX (the read path everything else uses)
# ============================================================

def get_prices_for_universe(tickers: list[str] | None = None,
                            lookback_days: int | None = None,
                            with_volume: bool = False):
    """
    Load adj_close (and optionally volume) for a ticker list as wide matrices:
    index = date, columns = ticker. One SQL query; ~600 x 460 floats ≈ 2 MB,
    comfortably inside the 512MB container.
    """
    from sqlalchemy import text, bindparam
    from data.db import engine
    if tickers is None:
        tickers = get_full_universe()
    if lookback_days is None:
        lookback_days = PRICE_LOOKBACK_DAYS
    cutoff = (datetime.now() - timedelta(days=int(lookback_days * 1.6))).date()

    cols = "date, ticker, adj_close" + (", volume" if with_volume else "")
    stmt = text(
        f"SELECT {cols} FROM daily_prices "
        f"WHERE ticker IN :ts AND date >= :cutoff ORDER BY date"
    ).bindparams(bindparam("ts", expanding=True))

    df = pd.read_sql(stmt, engine, params={"ts": list(tickers), "cutoff": cutoff})
    if df.empty:
        empty = pd.DataFrame()
        return (empty, empty) if with_volume else empty

    df["date"] = pd.to_datetime(df["date"])
    prices = df.pivot(index="date", columns="ticker", values="adj_close")
    prices = prices.tail(lookback_days)
    if with_volume:
        volume = df.pivot(index="date", columns="ticker", values="volume")
        volume = volume.reindex(prices.index)
        return prices, volume
    return prices


# ============================================================
# MACRO DATA (FRED + VIX backup) — bulk-write versions
# ============================================================

def download_fred_data(years: int | None = None) -> dict:
    if years is None:
        years = HISTORY_YEARS
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping macro download")
        return {"success": [], "failed": list(FRED_SERIES.keys())}

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    success, failed = [], []
    for series_id, desc in FRED_SERIES.items():
        try:
            data = fred.get_series(
                series_id,
                observation_start=start.strftime("%Y-%m-%d"),
                observation_end=end.strftime("%Y-%m-%d"),
            )
            if data is None or data.empty:
                failed.append(series_id)
                continue
            recs = [
                {"series_id": series_id, "date": d.date(), "value": float(v)}
                for d, v in data.items() if pd.notna(v)
            ]
            if recs:
                _save_macro_bulk(recs)
                success.append(series_id)
                logger.info(f"  FRED {series_id}: {len(recs)} obs ({desc})")
            else:
                failed.append(series_id)
        except Exception as e:
            logger.error(f"  FRED {series_id}: {e}")
            failed.append(series_id)
    return {"success": success, "failed": failed}


def download_vix_from_yahoo(years: int | None = None) -> bool:
    """VIX backup from Yahoo in case FRED's VIXCLS lags or gaps."""
    if years is None:
        years = HISTORY_YEARS
    try:
        end = datetime.now()
        start = end - timedelta(days=years * 365)
        data = yf.download("^VIX", start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), progress=False)
        if data is None or data.empty:
            return False
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close_col = next((c for c in data.columns if "close" in str(c).lower()), None)
        if close_col is None:
            return False
        recs = [
            {"series_id": "VIX_YAHOO", "date": d.date(), "value": float(v)}
            for d, v in data[close_col].items() if pd.notna(v)
        ]
        if recs:
            _save_macro_bulk(recs)
            logger.info(f"  VIX (Yahoo): {len(recs)} obs")
            return True
    except Exception as e:
        logger.error(f"VIX download error: {e}")
    return False


# ============================================================
# ORCHESTRATION (called by start.py / scheduler.py)
# ============================================================

def run_full_ingestion() -> dict:
    """One-time DB fill: full universe history + macro. Memory-safe, batched."""
    logger.info("=" * 60)
    logger.info("JARVIS V3 — FULL DATA INGESTION")
    logger.info("=" * 60)
    px = download_universe_history()
    fred = download_fred_data()
    vix = download_vix_from_yahoo()
    logger.info(f"Ingestion summary: prices {len(px['success'])} ok / "
                f"{len(px['failed'])} failed; FRED {len(fred['success'])} ok; "
                f"VIX {'ok' if vix else 'failed'}")
    return {"prices": px, "fred": fred, "vix": vix}


def run_daily_update() -> dict:
    """Daily incremental update across the FULL universe (stocks + ETFs)."""
    logger.info("JARVIS V3 — daily data update (full universe)")
    px = update_universe_prices()
    fred = download_fred_data(years=1)
    vix = download_vix_from_yahoo(years=1)
    return {"prices": px, "fred": fred, "vix": vix}


# ── V2-compatible shims (older modules import these names) ──

def download_etf_prices(tickers=None, years=None):
    return download_universe_history(tickers or get_all_tickers(), years)


def update_etf_prices(tickers=None):
    return update_universe_prices(tickers or get_all_tickers())
