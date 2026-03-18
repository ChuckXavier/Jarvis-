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
    .metric-positive { color: #4CAF50; font-weight: bold; }
    .metric-negative { color: #F44336; font-weight: bold; }
    .benchmark-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #64B5F6;
        margin-bottom: 5px;
    }
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

@st.cache_data(ttl=300)
def load_portfolio_history():
    """Load Alpaca portfolio history for benchmark comparison."""
    try:
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            return pd.DataFrame()

        import requests
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        # Paper trading endpoint
        url = "https://paper-api.alpaca.markets/v2/account/portfolio/history"
        params = {
            "period": "1A",
            "timeframe": "1D",
            "intraday_reporting": "market_hours",
            "pnl_reset": "per_day",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            timestamps = data.get("timestamp", [])
            equity = data.get("equity", [])
            if timestamps and equity:
                df = pd.DataFrame({
                    "date": pd.to_datetime(timestamps, unit="s"),
                    "equity": equity,
                })
                df = df.set_index("date")
                df = df[df["equity"] > 0]
                return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_spy_history(start_date, end_date):
    """Load SPY price history from Yahoo Finance for benchmark comparison."""
    try:
        import requests
        # Use Yahoo Finance API
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/SPY"
            f"?period1={start_ts}&period2={end_ts}&interval=1d"
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            closes = result["indicators"]["adjclose"][0]["adjclose"]
            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s"),
                "close": closes,
            })
            df = df.set_index("date")
            df = df.dropna()
            return df
        return pd.DataFrame()
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
        "🏆 Benchmark",
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
# PAGE: BENCHMARK — Jarvis vs S&P 500
# ============================================================
elif page == "🏆 Benchmark":
    st.markdown('<p class="main-header">Benchmark Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Jarvis V4.2 performance vs. S&P 500 (SPY)</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Load portfolio history from Alpaca ──
    portfolio_hist = load_portfolio_history()

    if portfolio_hist.empty:
        # Fallback: try portfolio snapshots from database
        snapshots = load_portfolio_snapshots()
        if not snapshots.empty:
            snapshots["date"] = pd.to_datetime(snapshots["date"])
            snapshots = snapshots.set_index("date")
            portfolio_hist = snapshots[["total_value"]].rename(columns={"total_value": "equity"})

    if portfolio_hist.empty:
        st.warning(
            "No portfolio history available yet. Jarvis needs to run for at least a few days "
            "to build up enough data for benchmark comparison. Check back soon!"
        )
    else:
        # ── Determine date range ──
        start_date = portfolio_hist.index.min()
        end_date = portfolio_hist.index.max()
        total_days = (end_date - start_date).days

        # ── Load SPY data for the same period ──
        spy_data = load_spy_history(
            start_date - timedelta(days=5),
            end_date + timedelta(days=1),
        )

        if spy_data.empty:
            # Fallback: try loading from database prices
            prices = load_prices()
            if not prices.empty and "SPY" in prices.columns:
                spy_series = prices["SPY"].dropna()
                spy_data = pd.DataFrame({"close": spy_series})
                spy_data.index = pd.to_datetime(spy_data.index)

        if spy_data.empty:
            st.warning("Cannot load SPY benchmark data.")
        else:
            # ── Time Period Selector ──
            st.markdown("### Select Time Period")
            period_options = ["Since Inception"]
            if total_days >= 7:
                period_options.append("1 Week")
            if total_days >= 30:
                period_options.append("1 Month")
            if total_days >= 90:
                period_options.append("3 Months")
            if total_days >= 180:
                period_options.append("6 Months")
            if total_days >= 365:
                period_options.append("1 Year")

            # Add YTD
            ytd_start = datetime(end_date.year, 1, 1)
            if start_date <= pd.Timestamp(ytd_start):
                period_options.append("YTD")

            period_cols = st.columns(len(period_options))
            selected_period = st.radio(
                "Period",
                period_options,
                horizontal=True,
                label_visibility="collapsed",
            )

            # ── Filter data by selected period ──
            if selected_period == "1 Week":
                cutoff = end_date - timedelta(days=7)
            elif selected_period == "1 Month":
                cutoff = end_date - timedelta(days=30)
            elif selected_period == "3 Months":
                cutoff = end_date - timedelta(days=90)
            elif selected_period == "6 Months":
                cutoff = end_date - timedelta(days=180)
            elif selected_period == "1 Year":
                cutoff = end_date - timedelta(days=365)
            elif selected_period == "YTD":
                cutoff = pd.Timestamp(datetime(end_date.year, 1, 1))
            else:  # Since Inception
                cutoff = start_date

            # Filter portfolio
            port_filtered = portfolio_hist[portfolio_hist.index >= cutoff].copy()
            if port_filtered.empty:
                port_filtered = portfolio_hist.copy()

            # Filter SPY to match
            spy_filtered = spy_data[spy_data.index >= cutoff].copy()

            if len(port_filtered) >= 2 and len(spy_filtered) >= 2:
                # ── Normalize both to 100 at start ──
                port_norm = (port_filtered["equity"] / port_filtered["equity"].iloc[0]) * 100
                spy_norm = (spy_filtered["close"] / spy_filtered["close"].iloc[0]) * 100

                # ── Equity Curve Chart ──
                st.markdown("---")
                st.subheader(f"Performance: Jarvis vs. S&P 500 ({selected_period})")

                chart_df = pd.DataFrame({
                    "Jarvis V4.2": port_norm,
                    "S&P 500 (SPY)": spy_norm,
                })
                # Align dates — forward fill to handle missing dates
                chart_df = chart_df.sort_index().ffill().dropna()

                st.line_chart(chart_df, use_container_width=True)

                # ── Calculate Metrics ──
                port_start = port_filtered["equity"].iloc[0]
                port_end = port_filtered["equity"].iloc[-1]
                port_return = (port_end / port_start - 1) * 100

                spy_start = spy_filtered["close"].iloc[0]
                spy_end = spy_filtered["close"].iloc[-1]
                spy_return = (spy_end / spy_start - 1) * 100

                alpha = port_return - spy_return
                period_days = (port_filtered.index[-1] - port_filtered.index[0]).days
                period_years = max(period_days / 365.25, 0.01)

                # Annualized returns
                port_ann = ((port_end / port_start) ** (1 / period_years) - 1) * 100 if period_years >= 0.25 else port_return
                spy_ann = ((spy_end / spy_start) ** (1 / period_years) - 1) * 100 if period_years >= 0.25 else spy_return

                # Daily returns for Sharpe and drawdown
                port_daily = port_filtered["equity"].pct_change().dropna()
                spy_daily = spy_filtered["close"].pct_change().dropna()

                # Sharpe ratio (annualized, assuming 0% risk-free)
                port_sharpe = (port_daily.mean() / port_daily.std() * np.sqrt(252)) if len(port_daily) > 5 and port_daily.std() > 0 else 0
                spy_sharpe = (spy_daily.mean() / spy_daily.std() * np.sqrt(252)) if len(spy_daily) > 5 and spy_daily.std() > 0 else 0

                # Max drawdown
                port_cummax = port_filtered["equity"].cummax()
                port_dd = ((port_filtered["equity"] / port_cummax) - 1).min() * 100

                spy_cummax = spy_filtered["close"].cummax()
                spy_dd = ((spy_filtered["close"] / spy_cummax) - 1).min() * 100

                # Volatility (annualized)
                port_vol = port_daily.std() * np.sqrt(252) * 100 if len(port_daily) > 5 else 0
                spy_vol = spy_daily.std() * np.sqrt(252) * 100 if len(spy_daily) > 5 else 0

                # ── Summary Metrics ──
                st.markdown("---")
                st.subheader("Performance Summary")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "Jarvis Return",
                    f"{port_return:+.2f}%",
                    delta=f"{alpha:+.2f}% vs SPY",
                    delta_color="normal",
                )
                col2.metric("S&P 500 Return", f"{spy_return:+.2f}%")
                col3.metric("Alpha", f"{alpha:+.2f}%",
                           delta_color="normal")
                col4.metric("Period", f"{period_days} days")

                # ── Detailed Comparison Table ──
                st.markdown("---")
                st.subheader("Head-to-Head Comparison")

                def fmt_pct(v):
                    return f"{v:+.2f}%"

                def fmt_ratio(v):
                    return f"{v:.3f}"

                comparison = pd.DataFrame({
                    "Metric": [
                        "Total Return",
                        "Annualized Return" if period_years >= 0.25 else "Period Return",
                        "Sharpe Ratio",
                        "Max Drawdown",
                        "Volatility (Ann.)",
                        "Best Day",
                        "Worst Day",
                        "Win Rate (Daily)",
                    ],
                    "Jarvis V4.2": [
                        fmt_pct(port_return),
                        fmt_pct(port_ann),
                        fmt_ratio(port_sharpe),
                        fmt_pct(port_dd),
                        fmt_pct(port_vol),
                        fmt_pct(port_daily.max() * 100) if len(port_daily) > 0 else "—",
                        fmt_pct(port_daily.min() * 100) if len(port_daily) > 0 else "—",
                        f"{(port_daily > 0).mean() * 100:.1f}%" if len(port_daily) > 0 else "—",
                    ],
                    "S&P 500 (SPY)": [
                        fmt_pct(spy_return),
                        fmt_pct(spy_ann),
                        fmt_ratio(spy_sharpe),
                        fmt_pct(spy_dd),
                        fmt_pct(spy_vol),
                        fmt_pct(spy_daily.max() * 100) if len(spy_daily) > 0 else "—",
                        fmt_pct(spy_daily.min() * 100) if len(spy_daily) > 0 else "—",
                        f"{(spy_daily > 0).mean() * 100:.1f}%" if len(spy_daily) > 0 else "—",
                    ],
                    "Advantage": [
                        "Jarvis" if port_return > spy_return else "SPY",
                        "Jarvis" if port_ann > spy_ann else "SPY",
                        "Jarvis" if port_sharpe > spy_sharpe else "SPY",
                        "Jarvis" if port_dd > spy_dd else "SPY",  # Less negative = better
                        "Jarvis" if port_vol < spy_vol else "SPY",  # Lower = better
                        "Jarvis" if len(port_daily) > 0 and len(spy_daily) > 0 and port_daily.max() > spy_daily.max() else "SPY",
                        "Jarvis" if len(port_daily) > 0 and len(spy_daily) > 0 and port_daily.min() > spy_daily.min() else "SPY",
                        "Jarvis" if len(port_daily) > 0 and len(spy_daily) > 0 and (port_daily > 0).mean() > (spy_daily > 0).mean() else "SPY",
                    ],
                })
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                # ── Multi-Period Returns Table ──
                st.markdown("---")
                st.subheader("Returns Across All Periods")

                period_map = {
                    "1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365,
                }

                period_rows = []
                for label, days in period_map.items():
                    if total_days >= days:
                        p_cut = end_date - timedelta(days=days)
                        p_port = portfolio_hist[portfolio_hist.index >= p_cut]
                        p_spy = spy_data[spy_data.index >= p_cut]

                        if len(p_port) >= 2 and len(p_spy) >= 2:
                            p_ret = (p_port["equity"].iloc[-1] / p_port["equity"].iloc[0] - 1) * 100
                            s_ret = (p_spy["close"].iloc[-1] / p_spy["close"].iloc[0] - 1) * 100
                            period_rows.append({
                                "Period": label,
                                "Jarvis": f"{p_ret:+.2f}%",
                                "S&P 500": f"{s_ret:+.2f}%",
                                "Alpha": f"{(p_ret - s_ret):+.2f}%",
                                "Winner": "Jarvis" if p_ret > s_ret else "SPY",
                            })

                # Add YTD
                ytd_cut = pd.Timestamp(datetime(end_date.year, 1, 1))
                if start_date <= ytd_cut:
                    ytd_port = portfolio_hist[portfolio_hist.index >= ytd_cut]
                    ytd_spy = spy_data[spy_data.index >= ytd_cut]
                    if len(ytd_port) >= 2 and len(ytd_spy) >= 2:
                        ytd_p = (ytd_port["equity"].iloc[-1] / ytd_port["equity"].iloc[0] - 1) * 100
                        ytd_s = (ytd_spy["close"].iloc[-1] / ytd_spy["close"].iloc[0] - 1) * 100
                        period_rows.append({
                            "Period": "YTD",
                            "Jarvis": f"{ytd_p:+.2f}%",
                            "S&P 500": f"{ytd_s:+.2f}%",
                            "Alpha": f"{(ytd_p - ytd_s):+.2f}%",
                            "Winner": "Jarvis" if ytd_p > ytd_s else "SPY",
                        })

                # Add Since Inception
                si_ret = (portfolio_hist["equity"].iloc[-1] / portfolio_hist["equity"].iloc[0] - 1) * 100
                si_spy_data = spy_data[spy_data.index >= start_date]
                if len(si_spy_data) >= 2:
                    si_spy = (si_spy_data["close"].iloc[-1] / si_spy_data["close"].iloc[0] - 1) * 100
                    period_rows.append({
                        "Period": "Inception",
                        "Jarvis": f"{si_ret:+.2f}%",
                        "S&P 500": f"{si_spy:+.2f}%",
                        "Alpha": f"{(si_ret - si_spy):+.2f}%",
                        "Winner": "Jarvis" if si_ret > si_spy else "SPY",
                    })

                if period_rows:
                    st.dataframe(pd.DataFrame(period_rows), use_container_width=True, hide_index=True)

                # ── Rolling Alpha Chart ──
                if len(port_daily) >= 20 and len(spy_daily) >= 20:
                    st.markdown("---")
                    st.subheader("Rolling 20-Day Alpha (Jarvis - SPY)")

                    # Align the daily returns
                    combined = pd.DataFrame({
                        "jarvis": port_daily,
                        "spy": spy_daily,
                    }).dropna()

                    if len(combined) >= 20:
                        rolling_alpha = (
                            combined["jarvis"].rolling(20).mean() -
                            combined["spy"].rolling(20).mean()
                        ) * 252 * 100  # Annualized

                        rolling_alpha = rolling_alpha.dropna()
                        if not rolling_alpha.empty:
                            st.line_chart(rolling_alpha, use_container_width=True)
                            st.caption(
                                "Positive values = Jarvis outperforming SPY. "
                                "Negative values = SPY outperforming Jarvis. "
                                "Measured as annualized rolling 20-day alpha."
                            )

            else:
                st.info("Not enough data points in the selected period for comparison.")


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
    f"JARVIS V2 • Autonomous ETF Alpha Engine • V4.2 Strategy • "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True
)

