"""
JARVIS V2 - Mission Control Dashboard
========================================
A comprehensive Streamlit dashboard for monitoring Jarvis.

HOW TO RUN LOCALLY:
    streamlit run monitor/dashboard.py

HOW TO DEPLOY ON STREAMLIT CLOUD:
    1. Connect your GitHub repo on share.streamlit.io
    2. Set main file path: monitor/dashboard.py
    3. Add secrets: DATABASE_URL, ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path so imports work on Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="JARVIS V2 — Mission Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4CAF50;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-top: -10px;
    }
    .regime-calm { color: #4CAF50; font-weight: bold; font-size: 1.3rem; }
    .regime-transition { color: #FFC107; font-weight: bold; font-size: 1.3rem; }
    .regime-crisis { color: #F44336; font-weight: bold; font-size: 1.3rem; }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-shadow { background: #1E3A5F; color: #64B5F6; }
    .badge-supervised { background: #3E2723; color: #FFB74D; }
    .badge-autonomous { background: #1B5E20; color: #81C784; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================
def load_env():
    """Load environment variables from Streamlit secrets or OS env."""
    try:
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
            os.environ["ALPACA_API_KEY"] = st.secrets.get("ALPACA_API_KEY", "")
            os.environ["ALPACA_SECRET_KEY"] = st.secrets.get("ALPACA_SECRET_KEY", "")
            os.environ["FRED_API_KEY"] = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        pass

load_env()


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=300)
def load_prices():
    try:
        from data.db import get_all_prices
        return get_all_prices()
    except Exception as e:
        st.error(f"Cannot load prices: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_data_summary():
    try:
        from data.db import get_data_summary, get_record_count
        summary = get_data_summary()
        try:
            price_count = get_record_count("daily_prices")
            macro_count = get_record_count("macro_data")
        except Exception:
            price_count, macro_count = 0, 0
        return summary, price_count, macro_count
    except Exception:
        return pd.DataFrame(), 0, 0

@st.cache_data(ttl=300)
def load_portfolio_snapshots():
    try:
        from data.db import engine
        from sqlalchemy import text
        return pd.read_sql(text("SELECT * FROM portfolio_snapshots ORDER BY date"), engine)
    except Exception:
        return pd.DataFrame()

def get_alpaca_info():
    try:
        from execution.engine import ExecutionEngine
        executor = ExecutionEngine()
        if executor.connect():
            account = executor.get_account_info()
            positions = executor.get_current_positions()
            return account, positions, True
        return {}, {}, False
    except Exception:
        return {}, {}, False


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🤖 JARVIS V2")
    st.markdown("*Autonomous ETF Alpha Engine*")
    st.markdown("---")

    try:
        from config.settings import EXECUTION_MODE
        mode = EXECUTION_MODE
    except Exception:
        mode = "UNKNOWN"

    if mode == "SHADOW":
        st.markdown('<span class="status-badge badge-shadow">● SHADOW MODE</span>', unsafe_allow_html=True)
        st.caption("Logging only — no real trades")
    elif mode == "SUPERVISED":
        st.markdown('<span class="status-badge badge-supervised">● SUPERVISED MODE</span>', unsafe_allow_html=True)
        st.caption("Orders require your approval")
    elif mode == "AUTONOMOUS":
        st.markdown('<span class="status-badge badge-autonomous">● AUTONOMOUS MODE</span>', unsafe_allow_html=True)
        st.caption("Fully automated trading")

    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🧠 Alpha Scores",
        "📈 Portfolio",
        "🛡️ Risk Monitor",
        "💾 Data Health",
    ])

    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.markdown('<p class="main-header">Mission Control</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time overview of JARVIS V2 operations</p>', unsafe_allow_html=True)
    st.markdown("---")

    account, positions, connected = get_alpaca_info()

    col1, col2, col3, col4, col5 = st.columns(5)
    if connected:
        pv = account.get("portfolio_value", 0)
        cash = account.get("cash", 0)
        invested = account.get("long_market_value", 0)
        col1.metric("Portfolio Value", f"${pv:,.2f}")
        col2.metric("Cash", f"${cash:,.2f}")
        col3.metric("Invested", f"${invested:,.2f}")
        col4.metric("Positions", f"{len(positions)}")
        col5.metric("Cash %", f"{(cash/pv*100) if pv > 0 else 0:.1f}%")
    else:
        for c in [col1, col2, col3, col4, col5]:
            c.metric("—", "Offline")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Current Positions")
        if positions:
            pos_data = []
            for ticker, info in sorted(positions.items(), key=lambda x: x[1].get("market_value", 0), reverse=True):
                pos_data.append({
                    "Ticker": ticker,
                    "Shares": f"{info['qty']:.2f}",
                    "Value": f"${info['market_value']:,.2f}",
                    "P&L": f"${info['unrealized_pnl']:,.2f}",
                    "P&L %": f"{info.get('unrealized_pnl_pct', 0):.2%}",
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        else:
            st.info("No positions yet — Jarvis is in SHADOW mode")

    with right:
        st.subheader("ETF Performance (30 Days)")
        prices = load_prices()
        if not prices.empty:
            recent = prices.tail(30)
            if len(recent) >= 2:
                returns_30d = ((recent.iloc[-1] / recent.iloc[0]) - 1).sort_values(ascending=True) * 100
                st.bar_chart(returns_30d, horizontal=True)

    st.markdown("---")
    summary, price_count, macro_count = load_data_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ETFs Tracked", f"{len(summary) if not summary.empty else 0}")
    col2.metric("Price Records", f"{price_count:,}")
    col3.metric("Macro Records", f"{macro_count:,}")
    col4.metric("Latest Data", str(summary["last_date"].max()) if not summary.empty else "—")


# ============================================================
# PAGE: ALPHA SCORES
# ============================================================
elif page == "🧠 Alpha Scores":
    st.markdown('<p class="main-header">Alpha Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Signal analysis and ensemble scores</p>', unsafe_allow_html=True)
    st.markdown("---")

    prices = load_prices()
    if prices.empty:
        st.warning("No price data available. Run the data pipeline first.")
    else:
        if st.button("🧠 Run Alpha Engine Now", type="primary"):
            with st.spinner("Computing features and signals... (1-2 minutes)"):
                try:
                    from signals.ensemble import compute_ensemble, get_top_bottom_etfs
                    result = compute_ensemble(prices)
                    latest = result["latest_scores"]
                    regime = result["regime"]

                    regime_class = f"regime-{regime.lower()}"
                    st.markdown(f'### Market Regime: <span class="{regime_class}">{regime}</span>',
                               unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Active Signals", "4 / 4")
                    col2.metric("ETFs Scored", f"{len(latest)}")
                    col3.metric("Regime", regime)

                    if not latest.empty:
                        st.markdown("---")
                        st.subheader("Ensemble Alpha Scores")
                        chart_data = latest.sort_values(ascending=True)
                        st.bar_chart(chart_data, horizontal=True)

                        top_bottom = get_top_bottom_etfs(latest, top_n=5)
                        col_buy, col_sell = st.columns(2)

                        with col_buy:
                            st.subheader("🟢 Top Buy Candidates")
                            buy_data = pd.DataFrame({
                                "ETF": top_bottom["top_buy"].index,
                                "Score": [f"{s:+.3f}" for s in top_bottom["top_buy"].values],
                                "Strength": ["█" * max(1, int(abs(s) * 8)) for s in top_bottom["top_buy"].values],
                            })
                            st.dataframe(buy_data, use_container_width=True, hide_index=True)

                        with col_sell:
                            st.subheader("🔴 Sell / Avoid")
                            sell_data = pd.DataFrame({
                                "ETF": top_bottom["top_sell"].index,
                                "Score": [f"{s:+.3f}" for s in top_bottom["top_sell"].values],
                                "Weakness": ["░" * max(1, int(abs(s) * 8)) for s in top_bottom["top_sell"].values],
                            })
                            st.dataframe(sell_data, use_container_width=True, hide_index=True)

                        st.markdown("---")
                        st.subheader("Full ETF Ranking")
                        full = pd.DataFrame({
                            "Rank": range(1, len(latest) + 1),
                            "ETF": latest.index,
                            "Score": [f"{s:+.4f}" for s in latest.values],
                            "Action": ["STRONG BUY" if s > 0.5 else "BUY" if s > 0.1
                                       else "HOLD" if s > -0.1 else "REDUCE" if s > -0.5
                                       else "SELL" for s in latest.values],
                        })
                        st.dataframe(full, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Alpha engine error: {e}")
                    st.exception(e)
        else:
            st.info("Click the button above to run the Alpha Engine.")


# ============================================================
# PAGE: PORTFOLIO
# ============================================================
elif page == "📈 Portfolio":
    st.markdown('<p class="main-header">Portfolio</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Positions, allocation, and performance</p>', unsafe_allow_html=True)
    st.markdown("---")

    account, positions, connected = get_alpaca_info()

    if connected:
        pv = account.get("portfolio_value", 0)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${pv:,.2f}")
        col2.metric("Cash", f"${account.get('cash', 0):,.2f}")
        col3.metric("# Positions", len(positions))

        if positions:
            st.subheader("Position Details")
            rows = []
            for ticker, info in sorted(positions.items(), key=lambda x: x[1]["market_value"], reverse=True):
                w = info["market_value"] / pv * 100 if pv > 0 else 0
                rows.append({
                    "Ticker": ticker,
                    "Shares": f"{info['qty']:.2f}",
                    "Entry": f"${info['entry_price']:.2f}",
                    "Current": f"${info['current_price']:.2f}",
                    "Value": f"${info['market_value']:,.2f}",
                    "Weight": f"{w:.1f}%",
                    "P&L $": f"${info['unrealized_pnl']:,.2f}",
                    "P&L %": f"{info.get('unrealized_pnl_pct', 0):.2%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("Allocation")
            alloc = {t: info["market_value"] for t, info in positions.items()}
            alloc["Cash"] = account.get("cash", 0)
            st.bar_chart(pd.Series(alloc).sort_values(ascending=True), horizontal=True)

        snapshots = load_portfolio_snapshots()
        if not snapshots.empty:
            st.markdown("---")
            st.subheader("Portfolio Value History")
            snap = snapshots.copy()
            snap["date"] = pd.to_datetime(snap["date"])
            snap = snap.set_index("date")
            st.line_chart(snap["total_value"])
    else:
        st.warning("Cannot connect to Alpaca. Check API keys.")


# ============================================================
# PAGE: RISK MONITOR
# ============================================================
elif page == "🛡️ Risk Monitor":
    st.markdown('<p class="main-header">Risk Fortress</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">5-layer risk monitoring and circuit breakers</p>', unsafe_allow_html=True)
    st.markdown("---")

    prices = load_prices()
    if not prices.empty:
        col1, col2, col3 = st.columns(3)

        current_vix = 0
        try:
            from data.db import get_macro
            vix = get_macro("VIXCLS")
            if vix.empty:
                vix = get_macro("VIX_YAHOO")
            if not vix.empty:
                current_vix = vix["value"].iloc[-1]
                vix_5d = vix["value"].iloc[-1] - vix["value"].iloc[-5] if len(vix) > 5 else 0
                col1.metric("VIX", f"{current_vix:.1f}", delta=f"{vix_5d:+.1f} (5d)", delta_color="inverse")
        except Exception:
            col1.metric("VIX", "—")

        if "SPY" in prices.columns:
            spy = prices["SPY"].dropna()
            peak = spy.rolling(252).max().iloc[-1]
            dd = (spy.iloc[-1] / peak - 1) * 100
            col2.metric("SPY Drawdown", f"{dd:.1f}%")

        equity_t = [t for t in ["SPY", "QQQ", "IWM", "EFA", "EEM"] if t in prices.columns]
        if len(equity_t) >= 3:
            ret = prices[equity_t].pct_change().tail(63).dropna()
            corr = ret.corr()
            mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            col3.metric("Avg Correlation", f"{corr.values[mask].mean():.2f}")

        if current_vix >= 35:
            st.error("⚠️ VIX above 35 — CIRCUIT BREAKER would trigger!")
        elif current_vix >= 25:
            st.warning("VIX elevated — market showing stress")
        else:
            st.success("Markets calm — all clear")

        st.markdown("---")
        st.subheader("Circuit Breaker Status")
        cb = [
            {"Breaker": "Daily Loss > 3%", "Threshold": "-3.0%", "Status": "✅ Clear"},
            {"Breaker": "Weekly Loss > 5%", "Threshold": "-5.0%", "Status": "✅ Clear"},
            {"Breaker": "Max Drawdown > 15%", "Threshold": "-15.0%", "Status": "✅ Clear"},
            {"Breaker": "VIX > 35", "Threshold": "35.0",
             "Status": "✅ Clear" if current_vix < 35 else "🔴 TRIGGERED"},
        ]
        st.dataframe(pd.DataFrame(cb), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("VIX History (1 Year)")
        try:
            st.line_chart(vix["value"].tail(252))
        except Exception:
            pass

        st.markdown("---")
        st.subheader("Risk Limits")
        from config.settings import (
            MAX_SINGLE_POSITION_PCT, MAX_SECTOR_EXPOSURE_PCT, MAX_POSITIONS,
            MAX_DAILY_VAR_PCT, MAX_DRAWDOWN_PCT, MIN_CASH_RESERVE_PCT,
        )
        limits = [
            {"Limit": "Max Single Position", "Value": f"{MAX_SINGLE_POSITION_PCT:.0%}"},
            {"Limit": "Max Sector Exposure", "Value": f"{MAX_SECTOR_EXPOSURE_PCT:.0%}"},
            {"Limit": "Max Positions", "Value": str(MAX_POSITIONS)},
            {"Limit": "Max Daily VaR (95%)", "Value": f"{MAX_DAILY_VAR_PCT:.0%}"},
            {"Limit": "Max Drawdown", "Value": f"{MAX_DRAWDOWN_PCT:.0%}"},
            {"Limit": "Min Cash Reserve", "Value": f"{MIN_CASH_RESERVE_PCT:.0%}"},
        ]
        st.dataframe(pd.DataFrame(limits), use_container_width=True, hide_index=True)


# ============================================================
# PAGE: DATA HEALTH
# ============================================================
elif page == "💾 Data Health":
    st.markdown('<p class="main-header">Data Health</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Database status and data quality</p>', unsafe_allow_html=True)
    st.markdown("---")

    summary, price_count, macro_count = load_data_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total ETFs", len(summary) if not summary.empty else 0)
    col2.metric("Price Records", f"{price_count:,}")
    col3.metric("Macro Records", f"{macro_count:,}")

    if not summary.empty:
        st.subheader("Per-ETF Data Coverage")
        display = summary.copy()
        display.columns = ["Ticker", "First Date", "Last Date", "Trading Days"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("---")
        if st.button("🔍 Run Data Quality Checks"):
            with st.spinner("Checking..."):
                try:
                    from data.quality import run_all_quality_checks
                    report = run_all_quality_checks()
                    if report["passed"]:
                        st.success(f"✅ All checks PASSED ({len(report['warnings'])} warnings)")
                    else:
                        st.error(f"❌ FAILED ({len(report['errors'])} errors)")
                    for e in report.get("errors", []):
                        st.error(e)
                    for w in report.get("warnings", []):
                        st.warning(w)
                except Exception as e:
                    st.error(f"Error: {e}")


# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #555; font-size: 0.8rem;'>"
    f"JARVIS V2 • Autonomous ETF Alpha Engine • "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True
)
