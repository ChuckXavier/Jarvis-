"""
JARVIS V3 - Central Configuration
==================================
ARCHITECTURAL NOTE (what changed vs V2):
Risk parameters move from the concentrated V4.2 posture (25% single position,
zero cash, -25% stops) to a diversified long/short book: 5% position cap,
60-80 names, -8% stops, 2% cash buffer. New sections cover the stock universe
(liquidity filters), the regime-driven exposure ladder, and the adaptive
signal-weight machinery. ALLOW_LEVERAGE is the single gate for >100% gross and
2x ETFs; it ships False and should stay False until the walk-forward harness
(backtest/walkforward.py) justifies flipping it. Every name that existed in V2
is still defined here so modules not rewritten in this pass keep importing.
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
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading only

# ============================================================
# FRED (Federal Reserve Economic Data)
# ============================================================
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ============================================================
# DATA SETTINGS
# ============================================================
HISTORY_YEARS = 10            # full-history backfill depth
PRICE_LOOKBACK_DAYS = 460     # trading days loaded for daily signal work
                              # (covers 252d momentum + 21d skip + IC windows)
DATA_REFRESH_HOUR = 6
DATA_REFRESH_MINUTE = 0

INGEST_CHUNK_SIZE = 50        # tickers per yfinance batch call
INGEST_MAX_RETRIES = 3
INGEST_RETRY_BASE_SLEEP = 2.0 # seconds; exponential backoff base
INGEST_CHUNK_PAUSE = 1.0      # polite pause between batch calls

# ============================================================
# UNIVERSE (stocks + ETFs)
# ============================================================
UNIVERSE_REFRESH_DAYS = 7     # constituent lists refreshed weekly
MIN_PRICE = 5.0               # exclude sub-$5 names
MIN_DOLLAR_VOLUME = 10_000_000  # median 21d dollar volume floor
MIN_HISTORY_DAYS = 252        # names with less history score NaN

# ============================================================
# RISK LIMITS — diversified long/short book
# ============================================================
MAX_POSITIONS = 80                  # capacity: ~50 longs + ~30 shorts
TARGET_LONG_POSITIONS = 40          # default long-leg breadth
TARGET_SHORT_POSITIONS = 20         # default short-leg breadth
MAX_SINGLE_POSITION_PCT = 0.05      # 5% cap per name — diversification first
MIN_POSITION_PCT = 0.004            # below this, a position is dust: skip
STOP_LOSS_PCT = -0.08               # -8% per-position stop (was -25%)
TRAILING_STOP_PCT = -0.05           # retained for risk_engine compatibility

# Live rebalance cadence (lab winner 2026-07-07: "rebal 10d + crisis net 0"
# doubled walk-forward Sharpe 0.20 -> 0.42, mostly by halving turnover costs).
# Between full rebalances the daily cron still updates data, feeds the regime
# machine, and enforces STOP_LOSS_PCT — see scheduler.run_stop_check.
LIVE_REBALANCE_DAYS = 10            # trading days between full rebalances

MAX_SECTOR_NET_LONG = 0.20          # sector long-side weight cap (abs of NAV)
MAX_SECTOR_GROSS_SHORT = 0.10       # sector short-side weight cap (abs of NAV)
MAX_SECTOR_EXPOSURE_PCT = 0.25      # legacy name used by risk_engine; long-side

MAX_DAILY_VAR_PCT = 0.02
MAX_DRAWDOWN_PCT = -0.15
MAX_AVG_CORRELATION = 0.70
MIN_CASH_RESERVE_PCT = 0.02         # 2% buffer (was 0%; never 47% again)

# Circuit Breakers (consumed by risk/circuit_breakers.py)
CIRCUIT_DAILY_LOSS_PCT = -0.03
CIRCUIT_WEEKLY_LOSS_PCT = -0.05
CIRCUIT_MAX_DRAWDOWN_PCT = -0.15
CIRCUIT_VIX_THRESHOLD = 35.0

# ============================================================
# LEVERAGE GATE — the Option-A switch
# ============================================================
# False: gross <= 100%, leveraged/inverse ETFs excluded from candidates.
# True : regime ladder may target up to 130% gross and map into 2x ETFs.
# Do not flip this on without walk-forward, cost-aware evidence.
ALLOW_LEVERAGE = False

# ============================================================
# PORTFOLIO CONSTRUCTION (legacy names retained for compatibility)
# ============================================================
CORE_ALLOCATION_PCT = 0.95
SATELLITE_ALLOCATION_PCT = 0.05
CASH_RESERVE_PCT = MIN_CASH_RESERVE_PCT
KELLY_MULTIPLIER = 1.0
REBALANCE_DRIFT_THRESHOLD = 0.20    # relative drift before a position trades
MIN_TRADE_WEIGHT = 0.0035           # absolute |Δw| floor; smaller = noise

INVERSE_VOL_FLOOR = 0.05            # annualized vol floor for 1/vol weights
INVERSE_VOL_LOOKBACK = 63           # days of realized vol for sizing

# ============================================================
# SIGNAL SETTINGS
# ============================================================
MOMENTUM_LOOKBACK = 126
MOMENTUM_SKIP = 21
TREND_FAST_LOOKBACK = 50
TREND_SLOW_LOOKBACK = 200
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VIX_CRISIS_THRESHOLD = 35.0
HMM_N_STATES = 3                    # legacy (signals/vol_regime.py)

# Adaptive IC weighting (live, daily — replaces the orphaned backtest/adapter)
IC_LOOKBACK_PRIMARY = 21            # trailing IC horizon (days)
IC_LOOKBACK_SECONDARY = 63          # stability blend horizon
IC_BLEND = 0.70                     # weight on the 21d IC in the composite
WEIGHT_FLOOR = 0.05                 # never abandon a signal entirely
WEIGHT_CEILING = 0.40               # never bet the book on one signal
WEIGHT_EMA_HALFLIFE_DAYS = 10       # daily EMA toward IC-implied targets
IC_TILT_SCALE = 0.5                 # how strongly positive IC pulls weight up

# ============================================================
# EXECUTION SETTINGS
# ============================================================
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "AUTONOMOUS")  # SHADOW|SUPERVISED|AUTONOMOUS
MAX_DAILY_TRADES = 60
MAX_DAILY_CAPITAL_DEPLOYED_PCT = 1.00
ORDER_TIMEOUT_SECONDS = 7200        # cancel resting orders after 2 hours
SELL_FILL_WAIT_SECONDS = 90         # poll window: let reduces fill before opens
SELL_FILL_POLL_SECONDS = 3
QUOTE_CHUNK_SIZE = 200              # symbols per latest-quote request

# ============================================================
# BACKTEST / RESEARCH DEFAULTS
# ============================================================
BACKTEST_COST_BPS = 5.0             # per-side cost (spread/2 + slippage), liquid US
BACKTEST_REBALANCE_DAYS = 5         # weekly rebalance in the harness

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "logs/jarvis.log"


def validate_config():
    """Return a list of missing required environment variables."""
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
