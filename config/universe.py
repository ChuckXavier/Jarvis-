"""
JARVIS V3 - Universe Definition (ETFs + dynamic stock universe)
================================================================
ARCHITECTURAL NOTE (what changed vs V2):
V2 hardcoded ~33 ETFs and bolted stocks on via data/stock_universe.py, which
the pipeline never fully integrated — positions got stranded because price
fetching only covered ETFs. V3 makes the stock universe first-class: S&P 500
and Nasdaq-100 constituents (with GICS sectors) are fetched from Wikipedia,
cached in PostgreSQL (`universe_constituents`), and refreshed weekly so the
list survives restarts and constituent churn. The fallback chain is honest:
live fetch -> DB cache -> a small, clearly-labelled mega-cap seed list -> ETFs
only, each step logged loudly. `get_all_tickers()` still returns ETFs only
(backward compatibility with ingest/quality callers); new code should use
`get_stock_universe()` / `get_full_universe()`.
"""

import io
import re
from datetime import datetime, timezone

from loguru import logger

# ============================================================
# THE ETF UNIVERSE (unchanged from V2 — these stay first-class)
# ============================================================
ETF_UNIVERSE = [
    # ── TIER 1: CORE EQUITY ──
    {"ticker": "SPY",  "name": "S&P 500",                  "asset_class": "equity",       "sector": "us_large_cap",   "role": "Core equity anchor"},
    {"ticker": "QQQ",  "name": "Nasdaq 100",               "asset_class": "equity",       "sector": "us_tech",        "role": "Growth + tech exposure"},
    {"ticker": "IWM",  "name": "Russell 2000",             "asset_class": "equity",       "sector": "us_small_cap",   "role": "Small-cap premium"},
    {"ticker": "EFA",  "name": "MSCI EAFE",                "asset_class": "equity",       "sector": "intl_developed", "role": "Geographic diversification"},
    {"ticker": "EEM",  "name": "MSCI Emerging Markets",    "asset_class": "equity",       "sector": "intl_emerging",  "role": "EM growth + carry"},
    {"ticker": "VTV",  "name": "Vanguard Value",           "asset_class": "equity",       "sector": "us_value",       "role": "Value factor exposure"},
    {"ticker": "VUG",  "name": "Vanguard Growth",          "asset_class": "equity",       "sector": "us_growth",      "role": "Growth factor exposure"},
    {"ticker": "XLF",  "name": "Financial Select Sector",  "asset_class": "equity",       "sector": "us_financials",  "role": "Sector rotation candidate"},
    {"ticker": "XLE",  "name": "Energy Select Sector",     "asset_class": "equity",       "sector": "us_energy",      "role": "Commodity/inflation proxy"},
    {"ticker": "XLK",  "name": "Technology Select Sector", "asset_class": "equity",       "sector": "us_technology",  "role": "Sector rotation candidate"},
    {"ticker": "XLV",  "name": "Health Care Select Sector","asset_class": "equity",       "sector": "us_healthcare",  "role": "Defensive sector"},
    {"ticker": "XLI",  "name": "Industrial Select Sector", "asset_class": "equity",       "sector": "us_industrials", "role": "Cyclical sector"},

    # ── TIER 2: FIXED INCOME ──
    {"ticker": "TLT",  "name": "20+ Year Treasury Bond",    "asset_class": "fixed_income", "sector": "us_treasury_long",  "role": "Flight-to-safety hedge"},
    {"ticker": "IEF",  "name": "7-10 Year Treasury Bond",   "asset_class": "fixed_income", "sector": "us_treasury_mid",   "role": "Yield + stability"},
    {"ticker": "LQD",  "name": "Investment Grade Corporate","asset_class": "fixed_income", "sector": "us_corporate",      "role": "Credit spread exposure"},
    {"ticker": "HYG",  "name": "High Yield Corporate",      "asset_class": "fixed_income", "sector": "us_high_yield",     "role": "Risk-on credit"},
    {"ticker": "TIP",  "name": "TIPS Bond",                 "asset_class": "fixed_income", "sector": "us_tips",           "role": "Inflation regime hedge"},

    # ── TIER 3: REAL ASSETS ──
    {"ticker": "GLD",  "name": "Gold",                      "asset_class": "commodity",    "sector": "precious_metals",   "role": "Inflation + crisis hedge"},
    {"ticker": "SLV",  "name": "Silver",                    "asset_class": "commodity",    "sector": "precious_metals",   "role": "Industrial + precious metals"},
    {"ticker": "DBC",  "name": "Commodity Index",           "asset_class": "commodity",    "sector": "broad_commodities", "role": "Inflation protection"},
    {"ticker": "VNQ",  "name": "Real Estate",               "asset_class": "real_estate",  "sector": "us_reits",          "role": "Real asset exposure"},

    # ── TIER 4: TACTICAL & HEDGING ──
    {"ticker": "SHY",  "name": "1-3 Year Treasury",         "asset_class": "fixed_income", "sector": "us_treasury_short", "role": "Cash alternative / parking"},
    {"ticker": "VIXY", "name": "VIX Short-Term Futures",    "asset_class": "volatility",   "sector": "volatility",        "role": "Crisis hedge (short-term)"},
    {"ticker": "SH",   "name": "Short S&P 500",             "asset_class": "inverse",      "sector": "inverse_equity",    "role": "Bear market hedge"},
    {"ticker": "UUP",  "name": "US Dollar Index",           "asset_class": "currency",     "sector": "us_dollar",         "role": "Dollar strength exposure"},

    # ── TIER 5: LEVERAGED (candidates ONLY when settings.ALLOW_LEVERAGE) ──
    {"ticker": "QLD",  "name": "ProShares Ultra QQQ",        "asset_class": "leveraged",   "sector": "us_tech_2x",        "role": "2x Nasdaq-100 (gated)"},
    {"ticker": "SSO",  "name": "ProShares Ultra S&P 500",    "asset_class": "leveraged",   "sector": "us_large_cap_2x",   "role": "2x S&P 500 (gated)"},
    {"ticker": "ROM",  "name": "ProShares Ultra Technology", "asset_class": "leveraged",   "sector": "us_tech_2x",        "role": "2x Technology (gated)"},
    {"ticker": "UYG",  "name": "ProShares Ultra Financials", "asset_class": "leveraged",   "sector": "us_financials_2x",  "role": "2x Financials (gated)"},
    {"ticker": "DIG",  "name": "ProShares Ultra Energy",     "asset_class": "leveraged",   "sector": "us_energy_2x",      "role": "2x Oil & Gas (gated)"},

    # ── TIER 6: CRISIS ALPHA ──
    {"ticker": "GDX",  "name": "VanEck Gold Miners",         "asset_class": "commodity",    "sector": "gold_miners",          "role": "Crisis alpha"},
    {"ticker": "DBMF", "name": "iMGP Managed Futures",       "asset_class": "alternative",  "sector": "managed_futures",      "role": "Crisis alpha"},
    {"ticker": "BIL",  "name": "SPDR 1-3 Month T-Bill",      "asset_class": "fixed_income", "sector": "us_treasury_ultra_short", "role": "Cash equivalent"},
]

# Wikipedia sources for constituents (fetched, never trusted blindly: parsed
# tables are validated for shape before use).
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (JARVIS-V3 research; contact: ops)"}

# Emergency-only seed list: ~48 mega caps whose membership and GICS sector have
# been stable for years. Used ONLY if both the live fetch and the DB cache are
# empty, and its use is logged as a warning. This is not the trading universe;
# it is a parachute.
FALLBACK_SEED = [
    ("AAPL", "Information Technology"), ("MSFT", "Information Technology"),
    ("NVDA", "Information Technology"), ("GOOGL", "Communication Services"),
    ("AMZN", "Consumer Discretionary"), ("META", "Communication Services"),
    ("BRK-B", "Financials"), ("LLY", "Health Care"), ("JPM", "Financials"),
    ("XOM", "Energy"), ("UNH", "Health Care"), ("V", "Financials"),
    ("PG", "Consumer Staples"), ("MA", "Financials"), ("JNJ", "Health Care"),
    ("HD", "Consumer Discretionary"), ("COST", "Consumer Staples"),
    ("ORCL", "Information Technology"), ("ABBV", "Health Care"),
    ("BAC", "Financials"), ("CVX", "Energy"), ("KO", "Consumer Staples"),
    ("MRK", "Health Care"), ("WMT", "Consumer Staples"),
    ("PEP", "Consumer Staples"), ("CSCO", "Information Technology"),
    ("TMO", "Health Care"), ("MCD", "Consumer Discretionary"),
    ("CRM", "Information Technology"), ("ABT", "Health Care"),
    ("ADBE", "Information Technology"), ("WFC", "Financials"),
    ("IBM", "Information Technology"), ("GE", "Industrials"),
    ("QCOM", "Information Technology"), ("CAT", "Industrials"),
    ("DIS", "Communication Services"), ("VZ", "Communication Services"),
    ("AMGN", "Health Care"), ("PFE", "Health Care"),
    ("TXN", "Information Technology"), ("INTU", "Information Technology"),
    ("NEE", "Utilities"), ("UNP", "Industrials"),
    ("NKE", "Consumer Discretionary"), ("RTX", "Industrials"),
    ("LOW", "Consumer Discretionary"), ("HON", "Industrials"),
]


# ============================================================
# ETF HELPERS (V2-compatible)
# ============================================================

def get_all_tickers():
    """ETF tickers only — kept exactly as V2 for backward compatibility."""
    return [etf["ticker"] for etf in ETF_UNIVERSE]


def get_etf_tickers():
    return get_all_tickers()


def get_etf_universe():
    return list(ETF_UNIVERSE)


def get_tickers_by_asset_class(asset_class):
    return [e["ticker"] for e in ETF_UNIVERSE if e["asset_class"] == asset_class]


def get_etf_info(ticker):
    for etf in ETF_UNIVERSE:
        if etf["ticker"] == ticker:
            return etf
    return None


def get_asset_class_map():
    """Ticker -> asset_class for ETFs; stocks default to 'equity' downstream."""
    return {e["ticker"]: e["asset_class"] for e in ETF_UNIVERSE}


# ============================================================
# DATABASE CACHE
# ============================================================

def _ensure_table():
    from sqlalchemy import text
    from data.db import engine
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS universe_constituents (
                ticker     TEXT PRIMARY KEY,
                name       TEXT,
                sector     TEXT,
                source     TEXT,
                active     BOOLEAN DEFAULT TRUE,
                first_seen TIMESTAMP,
                last_seen  TIMESTAMP
            )
        """))


def _normalize_symbol(sym: str) -> str:
    """Wikipedia uses BRK.B; Yahoo/Alpaca use BRK-B. Normalize and validate."""
    s = str(sym).strip().upper().replace(".", "-")
    return s if re.fullmatch(r"[A-Z\-]{1,10}", s) else ""


def _fetch_wiki_table(url: str, symbol_col: str, sector_col: str | None,
                      name_col: str | None) -> list[dict]:
    """Fetch a Wikipedia constituents table. Returns [] on any failure."""
    import requests
    import pandas as pd
    resp = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        if symbol_col in cols:
            rows = []
            for _, r in tbl.iterrows():
                sym = _normalize_symbol(r[symbol_col])
                if not sym:
                    continue
                rows.append({
                    "ticker": sym,
                    "name": str(r[name_col]) if name_col and name_col in cols else "",
                    "sector": str(r[sector_col]) if sector_col and sector_col in cols else "Unknown",
                })
            # Sanity: a constituents table should have a sensible row count.
            if len(rows) >= 50:
                return rows
    return []


def refresh_universe(force: bool = False) -> dict:
    """
    Fetch S&P 500 + Nasdaq-100 constituents and upsert into PostgreSQL.
    Skips the network call if the cache was refreshed within
    UNIVERSE_REFRESH_DAYS (unless force=True). Never raises: on failure the
    existing cache remains authoritative.
    """
    from sqlalchemy import text
    from data.db import engine
    from config.settings import UNIVERSE_REFRESH_DAYS

    result = {"refreshed": False, "count": 0, "source": "cache", "error": None}
    try:
        _ensure_table()
        with engine.begin() as conn:
            row = conn.execute(text(
                "SELECT MAX(last_seen), COUNT(*) FROM universe_constituents WHERE active"
            )).fetchone()
        last_seen, count = row[0], int(row[1] or 0)
        if not force and last_seen is not None and count >= 300:
            age_days = (datetime.now(timezone.utc) - last_seen.replace(tzinfo=timezone.utc)).days
            if age_days < UNIVERSE_REFRESH_DAYS:
                result.update(count=count)
                logger.info(f"Universe cache fresh ({count} names, {age_days}d old) — skipping fetch")
                return result
    except Exception as e:
        logger.warning(f"Universe cache check failed ({e}); attempting live fetch")

    # ── Live fetch ──
    constituents: dict[str, dict] = {}
    try:
        sp = _fetch_wiki_table(WIKI_SP500, "Symbol", "GICS Sector", "Security")
        for r in sp:
            r["source"] = "sp500"
            constituents[r["ticker"]] = r
        logger.info(f"Fetched {len(sp)} S&P 500 constituents")
    except Exception as e:
        logger.warning(f"S&P 500 fetch failed: {e}")
        result["error"] = str(e)
    try:
        ndx = _fetch_wiki_table(WIKI_NDX, "Ticker", "GICS Sector", "Company")
        added = 0
        for r in ndx:
            if r["ticker"] not in constituents:
                r["source"] = "ndx100"
                constituents[r["ticker"]] = r
                added += 1
        logger.info(f"Fetched {len(ndx)} Nasdaq-100 constituents ({added} new)")
    except Exception as e:
        logger.warning(f"Nasdaq-100 fetch failed: {e}")

    # ETFs are handled separately — keep them out of the stock table.
    etf_set = set(get_all_tickers())
    constituents = {t: r for t, r in constituents.items() if t not in etf_set}

    if len(constituents) < 100:
        logger.error(f"Universe fetch yielded only {len(constituents)} names — "
                     f"keeping existing DB cache untouched")
        return result

    # ── Upsert: deactivate-all then mark fetched active (history preserved) ──
    try:
        now = datetime.now(timezone.utc)
        with engine.begin() as conn:
            conn.execute(text("UPDATE universe_constituents SET active = FALSE"))
            stmt = text("""
                INSERT INTO universe_constituents
                    (ticker, name, sector, source, active, first_seen, last_seen)
                VALUES (:ticker, :name, :sector, :source, TRUE, :now, :now)
                ON CONFLICT (ticker) DO UPDATE SET
                    name = EXCLUDED.name, sector = EXCLUDED.sector,
                    source = EXCLUDED.source, active = TRUE,
                    last_seen = EXCLUDED.last_seen
            """)
            conn.execute(stmt, [
                {"ticker": t, "name": r["name"][:200], "sector": r["sector"][:80],
                 "source": r["source"], "now": now}
                for t, r in constituents.items()
            ])
        result.update(refreshed=True, count=len(constituents), source="wikipedia")
        logger.info(f"Universe refreshed: {len(constituents)} stocks cached in PostgreSQL")
    except Exception as e:
        logger.error(f"Universe upsert failed: {e}")
        result["error"] = str(e)
    return result


def get_stock_universe() -> list[str]:
    """
    Active stock tickers. Order of truth: DB cache -> live refresh -> seed list.
    Returns [] only if every layer fails (caller then runs ETF-only, logged).
    """
    from sqlalchemy import text
    from data.db import engine
    try:
        _ensure_table()
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT ticker FROM universe_constituents WHERE active ORDER BY ticker"
            )).fetchall()
        if len(rows) >= 100:
            return [r[0] for r in rows]
        logger.warning(f"Stock cache thin ({len(rows)}); attempting refresh")
        refresh_universe(force=True)
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT ticker FROM universe_constituents WHERE active ORDER BY ticker"
            )).fetchall()
        if len(rows) >= 100:
            return [r[0] for r in rows]
    except Exception as e:
        logger.error(f"get_stock_universe DB path failed: {e}")

    logger.warning(f"FALLBACK universe in use: {len(FALLBACK_SEED)} seed mega-caps "
                   f"(constituent fetch unavailable)")
    return [t for t, _ in FALLBACK_SEED]


def get_full_universe() -> list[str]:
    """Stocks + ETFs, de-duplicated, ETFs guaranteed present."""
    stocks = get_stock_universe()
    etfs = get_all_tickers()
    seen, out = set(), []
    for t in etfs + stocks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def get_sector_map() -> dict:
    """
    Ticker -> sector for everything we know. ETFs use their V2 sector labels;
    stocks use GICS sectors from the constituents cache (or the seed list);
    unknowns map to 'Unknown' and are still subject to sector caps downstream.
    """
    m = {e["ticker"]: e["sector"] for e in ETF_UNIVERSE}
    from sqlalchemy import text
    try:
        from data.db import engine
        with engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT ticker, sector FROM universe_constituents WHERE active"
            )).fetchall()
        for t, s in rows:
            m[t] = s or "Unknown"
    except Exception as e:
        logger.warning(f"sector map: DB unavailable ({e}); using seed sectors")
        for t, s in FALLBACK_SEED:
            m.setdefault(t, s)
    return m


# ── Macro tickers / FRED series (unchanged from V2) ──
MACRO_TICKERS = {
    "^VIX": "VIX Index (Fear Gauge)",
    "^TNX": "10-Year Treasury Yield",
    "^TYX": "30-Year Treasury Yield",
    "^IRX": "13-Week Treasury Bill Rate",
}

FRED_SERIES = {
    "DGS10": "10-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "T10Y2Y": "10Y-2Y Yield Curve Spread",
    "BAMLH0A0HYM2": "High Yield Credit Spread (ICE BofA)",
    "VIXCLS": "VIX Close",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index (CPI)",
    "ICSA": "Initial Jobless Claims",
}

RISK_PARITY_GROUPS = {
    "us_equities": ["SPY", "QQQ", "IWM", "VTV", "VUG", "XLF", "XLE", "XLK", "XLV", "XLI"],
    "intl_equities": ["EFA", "EEM"],
    "fixed_income": ["TLT", "IEF", "LQD", "TIP", "SHY"],
    "real_assets": ["GLD", "SLV", "DBC", "VNQ"],
}
