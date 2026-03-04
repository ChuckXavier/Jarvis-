"""
JARVIS V2 - API Server
========================
A lightweight FastAPI server that exposes Jarvis's data as JSON endpoints.
The React dashboard fetches from these endpoints.

ENDPOINTS:
    GET /api/health          → Server status
    GET /api/account         → Alpaca account info + positions
    GET /api/alpha           → Latest alpha scores from all 4 signals
    GET /api/data-summary    → Database health (ETF coverage, record counts)
    GET /api/risk            → VIX, circuit breakers, macro data
    GET /api/orders          → Recent shadow/live orders
    GET /api/portfolio       → Portfolio snapshots history
    GET /api/performance     → 30-day ETF performance

HOW TO RUN:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import traceback
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Lifespan (startup/shutdown) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("JARVIS V2 API starting...")
    yield
    print("JARVIS V2 API shutting down...")

app = FastAPI(
    title="JARVIS V2 API",
    description="Live data API for the JARVIS V2 trading dashboard",
    version="2.0",
    lifespan=lifespan,
)

# Allow the React dashboard (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_json(obj):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (pd.Series,)):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/health")
def health_check():
    """Basic health check — is the API running?"""
    return {
        "status": "ok",
        "service": "JARVIS V2 API",
        "timestamp": datetime.now().isoformat(),
        "mode": _get_mode(),
    }


@app.get("/api/account")
def get_account():
    """Get Alpaca account info and current positions."""
    try:
        from execution.engine import ExecutionEngine
        executor = ExecutionEngine()

        if not executor.connect():
            return JSONResponse(status_code=503, content={
                "error": "Cannot connect to Alpaca",
                "account": None,
                "positions": [],
            })

        account = executor.get_account_info()
        positions_raw = executor.get_current_positions()

        positions = []
        for ticker, info in positions_raw.items():
            positions.append({
                "ticker": ticker,
                "qty": float(info.get("qty", 0)),
                "market_value": float(info.get("market_value", 0)),
                "current_price": float(info.get("current_price", 0)),
                "entry_price": float(info.get("entry_price", 0)),
                "unrealized_pnl": float(info.get("unrealized_pnl", 0)),
                "unrealized_pnl_pct": float(info.get("unrealized_pnl_pct", 0)),
            })

        positions.sort(key=lambda x: x["market_value"], reverse=True)

        return {
            "account": {
                "portfolio_value": float(account.get("portfolio_value", 0)),
                "cash": float(account.get("cash", 0)),
                "buying_power": float(account.get("buying_power", 0)),
                "equity": float(account.get("equity", 0)),
                "long_market_value": float(account.get("long_market_value", 0)),
            },
            "positions": positions,
            "connected": True,
            "mode": _get_mode(),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "account": None,
            "positions": [],
            "connected": False,
        })


@app.get("/api/alpha")
def get_alpha_scores():
    """Run the full ensemble and return alpha scores for all ETFs."""
    try:
        from data.db import get_all_prices
        from signals.ensemble import compute_ensemble, get_top_bottom_etfs

        prices = get_all_prices()
        if prices.empty:
            return JSONResponse(status_code=503, content={"error": "No price data"})

        result = compute_ensemble(prices)
        latest = result["latest_scores"]
        regime = result["regime"]
        weights = result["weights_used"]

        etfs = []
        for i, (ticker, score) in enumerate(latest.items()):
            score_val = float(score) if pd.notna(score) else 0
            if score_val > 0.5:
                action = "STRONG BUY"
            elif score_val > 0.1:
                action = "BUY"
            elif score_val > -0.1:
                action = "HOLD"
            elif score_val > -0.5:
                action = "REDUCE"
            else:
                action = "SELL"

            etfs.append({
                "rank": i + 1,
                "ticker": ticker,
                "score": round(score_val, 4),
                "action": action,
            })

        top_bottom = get_top_bottom_etfs(latest, top_n=5)

        return {
            "regime": regime,
            "signals_active": 4,
            "weights": safe_json(weights),
            "etfs": etfs,
            "top_buy": safe_json(top_bottom["top_buy"]),
            "top_sell": safe_json(top_bottom["top_sell"]),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


@app.get("/api/data-summary")
def get_data_summary():
    """Get database health: ETF coverage, record counts, freshness."""
    try:
        from data.db import get_data_summary as db_summary, get_record_count

        summary = db_summary()
        price_count = get_record_count("daily_prices")

        try:
            macro_count = get_record_count("macro_data")
        except Exception:
            macro_count = 0

        etfs = []
        if not summary.empty:
            for _, row in summary.iterrows():
                etfs.append({
                    "ticker": row["ticker"],
                    "first_date": str(row["first_date"]),
                    "last_date": str(row["last_date"]),
                    "num_days": int(row["num_days"]),
                })

        return {
            "total_etfs": len(summary) if not summary.empty else 0,
            "price_records": price_count,
            "macro_records": macro_count,
            "latest_date": str(summary["last_date"].max()) if not summary.empty else None,
            "etfs": etfs,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/risk")
def get_risk_data():
    """Get VIX, circuit breaker status, and risk metrics."""
    try:
        from data.db import get_macro, get_all_prices
        from config.settings import (
            CIRCUIT_DAILY_LOSS_PCT, CIRCUIT_WEEKLY_LOSS_PCT,
            CIRCUIT_MAX_DRAWDOWN_PCT, CIRCUIT_VIX_THRESHOLD,
            MAX_SINGLE_POSITION_PCT, MAX_SECTOR_EXPOSURE_PCT,
            MAX_POSITIONS, MAX_DAILY_VAR_PCT, MAX_DRAWDOWN_PCT,
            MIN_CASH_RESERVE_PCT,
        )

        # VIX data
        vix_data = get_macro("VIXCLS")
        if vix_data.empty:
            vix_data = get_macro("VIX_YAHOO")

        current_vix = float(vix_data["value"].iloc[-1]) if not vix_data.empty else None
        vix_5d_change = None
        if not vix_data.empty and len(vix_data) > 5:
            vix_5d_change = float(vix_data["value"].iloc[-1] - vix_data["value"].iloc[-6])

        vix_history = []
        if not vix_data.empty:
            recent_vix = vix_data.tail(252)
            for date, row in recent_vix.iterrows():
                vix_history.append({
                    "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                    "value": float(row["value"]) if pd.notna(row["value"]) else None,
                })

        # SPY drawdown
        prices = get_all_prices()
        spy_drawdown = None
        avg_correlation = None

        if not prices.empty and "SPY" in prices.columns:
            spy = prices["SPY"].dropna()
            peak = spy.rolling(252).max().iloc[-1]
            spy_drawdown = float((spy.iloc[-1] / peak - 1) * 100)

            equity_tickers = [t for t in ["SPY", "QQQ", "IWM", "EFA", "EEM"] if t in prices.columns]
            if len(equity_tickers) >= 3:
                ret = prices[equity_tickers].pct_change().tail(63).dropna()
                corr = ret.corr()
                mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
                avg_correlation = float(corr.values[mask].mean())

        circuit_breakers = [
            {"name": "Daily Loss", "threshold": f"{CIRCUIT_DAILY_LOSS_PCT:.1%}", "status": "clear"},
            {"name": "Weekly Loss", "threshold": f"{CIRCUIT_WEEKLY_LOSS_PCT:.1%}", "status": "clear"},
            {"name": "Max Drawdown", "threshold": f"{CIRCUIT_MAX_DRAWDOWN_PCT:.1%}", "status": "clear"},
            {"name": "VIX Level", "threshold": f"> {CIRCUIT_VIX_THRESHOLD}",
             "status": "triggered" if (current_vix and current_vix >= CIRCUIT_VIX_THRESHOLD) else "clear"},
        ]

        limits = {
            "max_single_position": f"{MAX_SINGLE_POSITION_PCT:.0%}",
            "max_sector_exposure": f"{MAX_SECTOR_EXPOSURE_PCT:.0%}",
            "max_positions": MAX_POSITIONS,
            "max_daily_var": f"{MAX_DAILY_VAR_PCT:.0%}",
            "max_drawdown": f"{MAX_DRAWDOWN_PCT:.0%}",
            "min_cash_reserve": f"{MIN_CASH_RESERVE_PCT:.0%}",
        }

        return {
            "vix": current_vix,
            "vix_5d_change": vix_5d_change,
            "vix_history": vix_history,
            "spy_drawdown": spy_drawdown,
            "avg_correlation": avg_correlation,
            "circuit_breakers": circuit_breakers,
            "risk_limits": limits,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/performance")
def get_etf_performance():
    """Get 30-day return for all ETFs."""
    try:
        from data.db import get_all_prices
        prices = get_all_prices()

        if prices.empty:
            return {"etfs": []}

        recent = prices.tail(30)
        if len(recent) < 2:
            return {"etfs": []}

        returns = ((recent.iloc[-1] / recent.iloc[0]) - 1)
        returns = returns.sort_values(ascending=False)

        etfs = []
        for ticker, ret in returns.items():
            if pd.notna(ret):
                etfs.append({
                    "ticker": ticker,
                    "return_30d": round(float(ret) * 100, 2),
                })

        return {"etfs": etfs}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/portfolio")
def get_portfolio_history():
    """Get portfolio optimization details (risk parity breakdown)."""
    try:
        from data.db import get_all_prices, engine
        from sqlalchemy import text

        # Portfolio snapshots
        snapshots = []
        try:
            df = pd.read_sql(text("SELECT * FROM portfolio_snapshots ORDER BY date"), engine)
            for _, row in df.iterrows():
                snapshots.append({
                    "date": str(row["date"]),
                    "total_value": float(row["total_value"]) if pd.notna(row.get("total_value")) else None,
                    "cash": float(row["cash"]) if pd.notna(row.get("cash")) else None,
                    "num_positions": int(row["num_positions"]) if pd.notna(row.get("num_positions")) else None,
                })
        except Exception:
            pass

        # Risk parity weights
        prices = get_all_prices()
        risk_parity = {}

        if not prices.empty:
            try:
                from portfolio.optimizer import compute_risk_parity
                core_weights = compute_risk_parity(prices)
                risk_parity = safe_json(core_weights)
            except Exception:
                pass

        return {
            "snapshots": snapshots,
            "risk_parity_weights": risk_parity,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def _get_mode():
    try:
        from config.settings import EXECUTION_MODE
        return EXECUTION_MODE
    except Exception:
        return "UNKNOWN"


# ── Run directly ──
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
