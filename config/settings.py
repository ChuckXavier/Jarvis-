"""
JARVIS V2 - Central Configuration
==================================
All system settings in one place. Environment variables are loaded
from Railway's dashboard (never hardcoded).

HOW THIS WORKS (for non-coders):
- This file reads your secret API keys from Railway's environment
- You set the keys once in Railway's dashboard, and this file grabs them
- The actual keys are NEVER written in the code (security best practice)
"""

import os
from dotenv import load_dotenv

# Load .env file if running locally (Railway sets these automatically)
load_dotenv()


# ============================================================
# DATABASE (PostgreSQL on Railway)
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ============================================================
# ALPACA PAPER TRADING
# ============================================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading

# ============================================================
# FRED (Federal Reserve Economic Data)
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ============================================================
# DATA SETTINGS
# ============================================================
# How many years of historical data to download
HISTORY_YEARS = 10

# Data refresh schedule (Eastern Time)
DATA_REFRESH_HOUR = 6
DATA_REFRESH_MINUTE = 0

# ============================================================
# RISK LIMITS (The Risk Fortress - Layer 3-5)
# ============================================================
MAX_SINGLE_POSITION_PCT = 0.25      # V4.2: allow 25% per position (leveraged ETFs need this)
MAX_SECTOR_EXPOSURE_PCT = 0.50      # V4.2: allow 50% sector (leveraged tech can be 2x20%)
MAX_POSITIONS = 35                   # 60/40 Combined: 15 stock longs + 10 stock shorts + 5 ETFs + buffer
MIN_POSITION_PCT = 0.01             # V4.2: allow smaller positions
STOP_LOSS_PCT = -0.25               # V4.2: -25% stop loss (V4.2 uses 200-SMA + circuit breakers for risk)
TRAILING_STOP_PCT = -0.05           # -5% trailing stop from peak

MAX_DAILY_VAR_PCT = 0.02            # 2% daily Value at Risk limit
MAX_DRAWDOWN_PCT = -0.15            # -15% maximum drawdown
MAX_AVG_CORRELATION = 0.70          # Maximum average pairwise correlation
MIN_CASH_RESERVE_PCT = 0.00         # V4.2: ZERO cash reserve (zero cash drag)

# Circuit Breakers
CIRCUIT_DAILY_LOSS_PCT = -0.03      # Halt if down 3% in a day
CIRCUIT_WEEKLY_LOSS_PCT = -0.05     # Reduce if down 5% in a week
CIRCUIT_MAX_DRAWDOWN_PCT = -0.15    # Full halt if down 15% from peak
CIRCUIT_VIX_THRESHOLD = 35.0        # Reduce positions if VIX > 35

# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================
CORE_ALLOCATION_PCT = 0.95          # V4.2: allow 95% invested
SATELLITE_ALLOCATION_PCT = 0.05   # V4.2: minimal (V4.2 doesn't use satellite)
CASH_RESERVE_PCT = 0.00             # V4.2: ZERO cash buffer

KELLY_MULTIPLIER = 1.0              # V4.2: full position sizing (no Kelly scaling)
REBALANCE_DRIFT_THRESHOLD = 0.20    # Rebalance if position drifts 20%

# ============================================================
# SIGNAL SETTINGS
# ============================================================
MOMENTUM_LOOKBACK = 126             # 6 months (trading days)
MOMENTUM_SKIP = 21                  # Skip most recent month
TREND_FAST_LOOKBACK = 21            # 1 month fast trend
TREND_SLOW_LOOKBACK = 252           # 12 month slow trend
RSI_OVERSOLD = 30                   # RSI oversold threshold
RSI_OVERBOUGHT = 70                 # RSI overbought threshold
VIX_CRISIS_THRESHOLD = 35.0         # VIX level for crisis regime
HMM_N_STATES = 3                    # Calm, Transition, Crisis

# ============================================================
# EXECUTION SETTINGS
# ============================================================
EXECUTION_MODE = "AUTONOMOUS"       # SHADOW | SUPERVISED | AUTONOMOUS
MAX_DAILY_TRADES = 40               # 60/40 Combined: up to 30 positions may need rebalancing
MAX_DAILY_CAPITAL_DEPLOYED_PCT = 1.00  # 60/40 Combined: allow full rebalance on strategy switch
ORDER_TIMEOUT_SECONDS = 7200        # Cancel unfilled orders after 2 hours

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "logs/jarvis.log"


def validate_config():
    """
    Check that all required environment variables are set.
    Returns a list of any missing variables.
    """
    missing = []
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not ALPACA_API_KEY:
        missing.append("ALPACA_API_KEY")
    if not ALPACA_SECRET_KEY:
        missing.append("ALPACA_SECRET_KEY")
    if not FRED_API_KEY:
        missing.append("FRED_API_KEY")
    return missing
