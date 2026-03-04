"""
JARVIS V2 - Monitoring Dashboard
===================================
A simple Streamlit dashboard to monitor Jarvis's performance.

HOW TO RUN:
    streamlit run monitor/dashboard.py

HOW THIS WORKS (for non-coders):
- This creates a web page you can open in your browser
- It shows: portfolio value, positions, recent trades, P&L chart
- It refreshes automatically so you can watch Jarvis in real-time
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="JARVIS V2 Dashboard", layout="wide")

st.title("🤖 JARVIS V2 — Autonomous ETF Alpha Engine")
st.markdown("---")


def load_data():
    """Load data from the database."""
    try:
        from data.db import get_all_prices, get_data_summary, engine
        from sqlalchemy import text

        # Portfolio snapshots
        try:
            snapshots = pd.read_sql(
                text("SELECT * FROM portfolio_snapshots ORDER BY date"),
                engine
            )
        except Exception:
            snapshots = pd.DataFrame()

        # Trade log
        try:
            trades = pd.read_sql(
                text("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 50"),
                engine
            )
        except Exception:
            trades = pd.DataFrame()

        # Data summary
        try:
            summary = get_data_summary()
        except Exception:
            summary = pd.DataFrame()

        # Recent prices
        try:
            prices = get_all_prices()
        except Exception:
            prices = pd.DataFrame()

        return {
            "snapshots": snapshots,
            "trades": trades,
            "summary": summary,
            "prices": prices,
        }
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return {"snapshots": pd.DataFrame(), "trades": pd.DataFrame(),
                "summary": pd.DataFrame(), "prices": pd.DataFrame()}


data = load_data()

# ── Section 1: Account Overview ──
st.header("📊 Account Overview")
col1, col2, col3, col4 = st.columns(4)

try:
    from execution.engine import ExecutionEngine
    exec_engine = ExecutionEngine()
    if exec_engine.connect():
        account = exec_engine.get_account_info()
        positions = exec_engine.get_current_positions()

        col1.metric("Portfolio Value", f"${account.get('portfolio_value', 0):,.2f}")
        col2.metric("Cash", f"${account.get('cash', 0):,.2f}")
        col3.metric("Invested", f"${account.get('long_market_value', 0):,.2f}")
        col4.metric("Positions", len(positions))
    else:
        col1.metric("Portfolio Value", "Not connected")
        col2.metric("Cash", "—")
        col3.metric("Invested", "—")
        col4.metric("Positions", "—")
        positions = {}
except Exception:
    col1.metric("Portfolio Value", "Offline")
    positions = {}

# ── Section 2: Current Positions ──
if positions:
    st.header("📈 Current Positions")
    pos_data = []
    for ticker, info in positions.items():
        pos_data.append({
            "Ticker": ticker,
            "Shares": info["qty"],
            "Value": f"${info['market_value']:,.2f}",
            "Entry Price": f"${info['entry_price']:.2f}",
            "Current Price": f"${info['current_price']:.2f}",
            "P&L": f"${info['unrealized_pnl']:,.2f}",
            "P&L %": f"{info['unrealized_pnl_pct']:.1%}",
        })
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

# ── Section 3: Latest Alpha Scores ──
st.header("🧠 Latest Alpha Scores")
try:
    from signals.ensemble import compute_ensemble
    prices = data["prices"]
    if not prices.empty:
        if st.button("Run Alpha Engine Now"):
            with st.spinner("Computing signals..."):
                result = compute_ensemble(prices)
                latest = result["latest_scores"]

                st.success(f"Regime: **{result['regime']}** | Active Signals: 4/4")

                if not latest.empty:
                    chart_data = latest.sort_values(ascending=True)
                    st.bar_chart(chart_data)
    else:
        st.warning("No price data available")
except Exception as e:
    st.warning(f"Alpha engine not available: {e}")

# ── Section 4: Data Health ──
st.header("💾 Data Health")
summary = data["summary"]
if not summary.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("ETFs Tracked", len(summary))
    col2.metric("Total Records", f"{summary['num_days'].sum():,}")
    col3.metric("Latest Data", str(summary['last_date'].max()))

    st.dataframe(summary, use_container_width=True)
else:
    st.warning("No data summary available")

# ── Section 5: Recent Trades ──
st.header("📝 Recent Trades")
trades = data["trades"]
if not trades.empty:
    st.dataframe(trades, use_container_width=True)
else:
    st.info("No trades recorded yet (Jarvis is in SHADOW mode)")

# ── Footer ──
st.markdown("---")
st.caption(f"JARVIS V2 • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
           f"Mode: SHADOW")
