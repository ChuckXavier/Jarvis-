"""
JARVIS V2 - ETF Universe Definition
=====================================
The 25 ETFs that Jarvis trades, organized by asset class and sector.

HOW THIS WORKS (for non-coders):
- This file defines WHICH ETFs Jarvis watches and trades
- Each ETF has metadata: what sector it belongs to, what asset class, etc.
- The system uses this to ensure diversification (no single sector > 20%)
"""

# ============================================================
# THE 25-ETF UNIVERSE
# ============================================================
# Each ETF is defined with:
#   ticker:      The stock ticker symbol
#   name:        Full name of the ETF
#   asset_class: Broad category (equity, fixed_income, commodity, etc.)
#   sector:      More specific grouping
#   role:        What this ETF does in the portfolio

ETF_UNIVERSE = [
    # ── TIER 1: CORE EQUITY (12 ETFs) ──
    {"ticker": "SPY",  "name": "S&P 500",                 "asset_class": "equity",       "sector": "us_large_cap",   "role": "Core equity anchor"},
    {"ticker": "QQQ",  "name": "Nasdaq 100",              "asset_class": "equity",       "sector": "us_tech",        "role": "Growth + tech exposure"},
    {"ticker": "IWM",  "name": "Russell 2000",            "asset_class": "equity",       "sector": "us_small_cap",   "role": "Small-cap premium"},
    {"ticker": "EFA",  "name": "MSCI EAFE",               "asset_class": "equity",       "sector": "intl_developed", "role": "Geographic diversification"},
    {"ticker": "EEM",  "name": "MSCI Emerging Markets",   "asset_class": "equity",       "sector": "intl_emerging",  "role": "EM growth + carry"},
    {"ticker": "VTV",  "name": "Vanguard Value",          "asset_class": "equity",       "sector": "us_value",       "role": "Value factor exposure"},
    {"ticker": "VUG",  "name": "Vanguard Growth",         "asset_class": "equity",       "sector": "us_growth",      "role": "Growth factor exposure"},
    {"ticker": "XLF",  "name": "Financial Select Sector", "asset_class": "equity",       "sector": "us_financials",  "role": "Sector rotation candidate"},
    {"ticker": "XLE",  "name": "Energy Select Sector",    "asset_class": "equity",       "sector": "us_energy",      "role": "Commodity/inflation proxy"},
    {"ticker": "XLK",  "name": "Technology Select Sector", "asset_class": "equity",      "sector": "us_technology",  "role": "Sector rotation candidate"},
    {"ticker": "XLV",  "name": "Health Care Select Sector","asset_class": "equity",      "sector": "us_healthcare",  "role": "Defensive sector"},
    {"ticker": "XLI",  "name": "Industrial Select Sector", "asset_class": "equity",      "sector": "us_industrials", "role": "Cyclical sector"},

    # ── TIER 2: FIXED INCOME (5 ETFs) ──
    {"ticker": "TLT",  "name": "20+ Year Treasury Bond",  "asset_class": "fixed_income", "sector": "us_treasury_long",    "role": "Flight-to-safety hedge"},
    {"ticker": "IEF",  "name": "7-10 Year Treasury Bond",  "asset_class": "fixed_income", "sector": "us_treasury_mid",     "role": "Yield + stability"},
    {"ticker": "LQD",  "name": "Investment Grade Corporate","asset_class": "fixed_income", "sector": "us_corporate",        "role": "Credit spread exposure"},
    {"ticker": "HYG",  "name": "High Yield Corporate",     "asset_class": "fixed_income", "sector": "us_high_yield",       "role": "Risk-on credit"},
    {"ticker": "TIP",  "name": "TIPS Bond",                "asset_class": "fixed_income", "sector": "us_tips",             "role": "Inflation regime hedge"},

    # ── TIER 3: REAL ASSETS (4 ETFs) ──
    {"ticker": "GLD",  "name": "Gold",                     "asset_class": "commodity",    "sector": "precious_metals",     "role": "Inflation + crisis hedge"},
    {"ticker": "SLV",  "name": "Silver",                   "asset_class": "commodity",    "sector": "precious_metals",     "role": "Industrial + precious metals"},
    {"ticker": "DBC",  "name": "Commodity Index",          "asset_class": "commodity",    "sector": "broad_commodities",   "role": "Inflation protection"},
    {"ticker": "VNQ",  "name": "Real Estate",              "asset_class": "real_estate",  "sector": "us_reits",            "role": "Real asset exposure"},

    # ── TIER 4: TACTICAL & HEDGING (4 ETFs) ──
    {"ticker": "SHY",  "name": "1-3 Year Treasury",        "asset_class": "fixed_income", "sector": "us_treasury_short",   "role": "Cash alternative / parking"},
    {"ticker": "VIXY", "name": "VIX Short-Term Futures",   "asset_class": "volatility",   "sector": "volatility",          "role": "Crisis hedge (short-term)"},
    {"ticker": "SH",   "name": "Short S&P 500",            "asset_class": "inverse",      "sector": "inverse_equity",      "role": "Bear market hedge"},
    {"ticker": "UUP",  "name": "US Dollar Index",          "asset_class": "currency",     "sector": "us_dollar",           "role": "Dollar strength exposure"},
]


# ── Helper functions ──

def get_all_tickers():
    """Return a list of all 25 ticker symbols."""
    return [etf["ticker"] for etf in ETF_UNIVERSE]


def get_tickers_by_asset_class(asset_class):
    """Return tickers filtered by asset class (e.g., 'equity', 'fixed_income')."""
    return [etf["ticker"] for etf in ETF_UNIVERSE if etf["asset_class"] == asset_class]


def get_etf_info(ticker):
    """Return the full info dict for a given ticker."""
    for etf in ETF_UNIVERSE:
        if etf["ticker"] == ticker:
            return etf
    return None


def get_sector_map():
    """Return a dict mapping each ticker to its sector."""
    return {etf["ticker"]: etf["sector"] for etf in ETF_UNIVERSE}


def get_asset_class_map():
    """Return a dict mapping each ticker to its asset class."""
    return {etf["ticker"]: etf["asset_class"] for etf in ETF_UNIVERSE}


# ── Asset class groupings for Risk Parity core ──
RISK_PARITY_GROUPS = {
    "us_equities":    ["SPY", "QQQ", "IWM", "VTV", "VUG", "XLF", "XLE", "XLK", "XLV", "XLI"],
    "intl_equities":  ["EFA", "EEM"],
    "fixed_income":   ["TLT", "IEF", "LQD", "TIP", "SHY"],
    "real_assets":    ["GLD", "SLV", "DBC", "VNQ"],
}

# ── Macro data tickers to pull from Yahoo Finance ──
MACRO_TICKERS = {
    "^VIX": "VIX Index (Fear Gauge)",
    "^TNX": "10-Year Treasury Yield",
    "^TYX": "30-Year Treasury Yield",
    "^IRX": "13-Week Treasury Bill Rate",
}

# ── FRED series IDs for macro data ──
FRED_SERIES = {
    "DGS10":     "10-Year Treasury Yield",
    "DGS2":      "2-Year Treasury Yield",
    "T10Y2Y":    "10Y-2Y Yield Curve Spread",
    "BAMLH0A0HYM2": "High Yield Credit Spread (ICE BofA)",
    "VIXCLS":    "VIX Close",
    "UNRATE":    "Unemployment Rate",
    "CPIAUCSL":  "Consumer Price Index (CPI)",
    "ICSA":      "Initial Jobless Claims",
}
