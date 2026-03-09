"""
JARVIS V5 - ETF Universe (Morningstar-Verified)
==================================================
37 ETFs across 5 tiers, selected for the dual-engine architecture.
"""

UNIVERSE = {
    # ── TIER 1: Leveraged Offense (Return Generators) ──
    "TQQQ": {"name": "ProShares UltraPro QQQ",       "leverage": 3, "underlying": "QQQ",  "tier": 1, "asset_class": "leveraged_equity", "sector": "nasdaq"},
    "UPRO": {"name": "ProShares UltraPro S&P 500",    "leverage": 3, "underlying": "SPY",  "tier": 1, "asset_class": "leveraged_equity", "sector": "sp500"},
    "SOXL": {"name": "Direxion Semi Bull 3X",          "leverage": 3, "underlying": "SOXX", "tier": 1, "asset_class": "leveraged_equity", "sector": "semiconductors"},
    "TECL": {"name": "Direxion Tech Bull 3X",           "leverage": 3, "underlying": "XLK",  "tier": 1, "asset_class": "leveraged_equity", "sector": "technology"},
    "QLD":  {"name": "ProShares Ultra QQQ",             "leverage": 2, "underlying": "QQQ",  "tier": 1, "asset_class": "leveraged_equity", "sector": "nasdaq"},
    "SSO":  {"name": "ProShares Ultra S&P 500",         "leverage": 2, "underlying": "SPY",  "tier": 1, "asset_class": "leveraged_equity", "sector": "sp500"},
    "ROM":  {"name": "ProShares Ultra Technology",      "leverage": 2, "underlying": "XLK",  "tier": 1, "asset_class": "leveraged_equity", "sector": "technology"},
    "SPXL": {"name": "Direxion S&P 500 Bull 3X",        "leverage": 3, "underlying": "SPY",  "tier": 1, "asset_class": "leveraged_equity", "sector": "sp500"},

    # ── TIER 2: Non-Leveraged Offense (Stable Growers) ──
    "SMH":  {"name": "VanEck Semiconductor",            "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "semiconductors"},
    "SOXX": {"name": "iShares Semiconductor",           "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "semiconductors"},
    "VGT":  {"name": "Vanguard IT Index",               "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "technology"},
    "XLK":  {"name": "Technology Select SPDR",           "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "technology"},
    "QQQ":  {"name": "Invesco QQQ Trust",               "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "nasdaq"},
    "SPY":  {"name": "SPDR S&P 500",                    "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "sp500"},
    "IYW":  {"name": "iShares US Technology",           "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "technology"},
    "FTEC": {"name": "Fidelity MSCI IT",                "leverage": 1, "underlying": None,   "tier": 2, "asset_class": "equity", "sector": "technology"},

    # ── TIER 3: Defensive Assets (Crisis Shields) ──
    "GLD":  {"name": "SPDR Gold Shares",                "leverage": 1, "underlying": None,   "tier": 3, "asset_class": "commodity",     "sector": "gold"},
    "GDX":  {"name": "VanEck Gold Miners",              "leverage": 1, "underlying": None,   "tier": 3, "asset_class": "equity",         "sector": "gold_miners"},
    "UGL":  {"name": "ProShares Ultra Gold",             "leverage": 2, "underlying": "GLD",  "tier": 3, "asset_class": "leveraged_commodity", "sector": "gold"},
    "TLT":  {"name": "iShares 20+ Year Treasury",       "leverage": 1, "underlying": None,   "tier": 3, "asset_class": "fixed_income",  "sector": "long_bond"},
    "IEF":  {"name": "iShares 7-10 Year Treasury",      "leverage": 1, "underlying": None,   "tier": 3, "asset_class": "fixed_income",  "sector": "mid_bond"},
    "SGOV": {"name": "iShares 0-3 Month Treasury",      "leverage": 1, "underlying": None,   "tier": 3, "asset_class": "cash_equiv",    "sector": "cash"},
    "BIL":  {"name": "SPDR 1-3 Month T-Bill",           "leverage": 1, "underlying": None,   "tier": 3, "asset_class": "cash_equiv",    "sector": "cash"},

    # ── TIER 4: Crisis Alpha (Bear Market Profit Engines) ──
    "SQQQ": {"name": "ProShares UltraPro Short QQQ",   "leverage": -3, "underlying": "QQQ", "tier": 4, "asset_class": "inverse",       "sector": "nasdaq"},
    "SH":   {"name": "ProShares Short S&P 500",         "leverage": -1, "underlying": "SPY", "tier": 4, "asset_class": "inverse",       "sector": "sp500"},
    "UVXY": {"name": "ProShares Ultra VIX ST Futures",  "leverage": 1.5,"underlying": None,  "tier": 4, "asset_class": "volatility",    "sector": "vix"},
    "DBMF": {"name": "iMGP DBi Managed Futures",        "leverage": 1, "underlying": None,   "tier": 4, "asset_class": "managed_futures","sector": "trend"},
    "CTA":  {"name": "Simplify Managed Futures",         "leverage": 1, "underlying": None,   "tier": 4, "asset_class": "managed_futures","sector": "trend"},
    "TAIL": {"name": "Cambria Tail Risk",                "leverage": 1, "underlying": None,   "tier": 4, "asset_class": "tail_hedge",    "sector": "puts"},
    "BTAL": {"name": "AGF Market Neutral Anti-Beta",    "leverage": 1, "underlying": None,   "tier": 4, "asset_class": "market_neutral", "sector": "neutral"},

    # ── TIER 5: Alternatives (Correlation Breakers) ──
    "COPX": {"name": "Global X Copper Miners",          "leverage": 1, "underlying": None,   "tier": 5, "asset_class": "commodity",     "sector": "copper"},
    "XME":  {"name": "SPDR Metals & Mining",             "leverage": 1, "underlying": None,   "tier": 5, "asset_class": "equity",        "sector": "mining"},
    "XLU":  {"name": "Utilities Select SPDR",            "leverage": 1, "underlying": None,   "tier": 5, "asset_class": "equity",        "sector": "utilities"},
    "XLP":  {"name": "Consumer Staples Select SPDR",     "leverage": 1, "underlying": None,   "tier": 5, "asset_class": "equity",        "sector": "staples"},
    "XLV":  {"name": "Health Care Select SPDR",          "leverage": 1, "underlying": None,   "tier": 5, "asset_class": "equity",        "sector": "healthcare"},
    "UUP":  {"name": "Invesco DB US Dollar Index",       "leverage": 1, "underlying": None,   "tier": 5, "asset_class": "currency",      "sector": "dollar"},
}

def get_all_tickers():
    return list(UNIVERSE.keys())

def get_tier(tier_num):
    return {t: info for t, info in UNIVERSE.items() if info["tier"] == tier_num}

def get_leveraged_tickers():
    return [t for t, info in UNIVERSE.items() if abs(info["leverage"]) > 1]

def get_underlying_map():
    """Map leveraged ETF → its underlying non-leveraged index ETF."""
    return {t: info["underlying"] for t, info in UNIVERSE.items() if info["underlying"]}

def get_tier1_3x():
    return [t for t, info in UNIVERSE.items() if info["tier"] == 1 and info["leverage"] == 3]

def get_tier1_2x():
    return [t for t, info in UNIVERSE.items() if info["tier"] == 1 and info["leverage"] == 2]

def get_offensive_tickers():
    return [t for t, info in UNIVERSE.items() if info["tier"] in (1, 2)]

def get_defensive_tickers():
    return [t for t, info in UNIVERSE.items() if info["tier"] in (3, 4)]

def get_crisis_tickers():
    return [t for t, info in UNIVERSE.items() if info["tier"] == 4]

def get_asset_class_map():
    return {t: info["asset_class"] for t, info in UNIVERSE.items()}

def get_sector_map():
    return {t: info["sector"] for t, info in UNIVERSE.items()}
