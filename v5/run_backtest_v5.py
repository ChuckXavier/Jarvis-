"""
JARVIS V5 - Medallion Backtest Runner
========================================
Downloads all V5 ETF data and runs the dual-engine backtest.

HOW TO RUN:
    python run_backtest_v5.py

This script is SELF-CONTAINED — it downloads its own data via yfinance
so it can run independently of the Jarvis V1-V4 database.
"""

import sys, os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs("logs", exist_ok=True)
logger.add("logs/backtest_v5.log", rotation="10 MB", level="INFO")


def download_data():
    """Download all V5 ETF + macro data via yfinance and FRED."""
    import yfinance as yf

    from config.universe_v5 import get_all_tickers

    tickers = get_all_tickers()
    # Add underlying indices for SMA filter
    extras = ["QQQ", "SPY", "XLK", "SOXX", "GLD"]
    all_tickers = list(set(tickers + extras))

    logger.info(f"Downloading data for {len(all_tickers)} ETFs...")

    prices = pd.DataFrame()
    for batch_start in range(0, len(all_tickers), 10):
        batch = all_tickers[batch_start:batch_start+10]
        try:
            data = yf.download(batch, period="10y", progress=False, auto_adjust=True)
            
            # Handle yfinance 1.2.0+ multi-level column format
            if isinstance(data.columns, pd.MultiIndex):
                # New format: (Price, Ticker) multi-index
                if "Close" in data.columns.get_level_values(0):
                    adj = data["Close"]
                elif "Adj Close" in data.columns.get_level_values(0):
                    adj = data["Adj Close"]
                else:
                    adj = pd.DataFrame()
            elif len(batch) == 1:
                # Single ticker returns flat columns
                if "Close" in data.columns:
                    adj = data[["Close"]].rename(columns={"Close": batch[0]})
                elif "Adj Close" in data.columns:
                    adj = data[["Adj Close"]].rename(columns={"Adj Close": batch[0]})
                else:
                    adj = pd.DataFrame()
            else:
                # Fallback
                if "Adj Close" in data.columns:
                    adj = data["Adj Close"]
                elif "Close" in data.columns:
                    adj = data["Close"]
                else:
                    adj = pd.DataFrame()

            if isinstance(adj, pd.Series):
                adj = adj.to_frame(batch[0])

            for col in adj.columns:
                if not adj[col].dropna().empty:
                    prices[col] = adj[col]

            logger.info(f"  Downloaded: {', '.join(batch)}")
        except Exception as e:
            logger.warning(f"  Failed batch {batch}: {e}")

    prices = prices.ffill().dropna(how="all")
    logger.info(f"Total: {len(prices.columns)} ETFs, {len(prices)} trading days")

    # VIX data
    vix = None
    try:
        vix_data = yf.download("^VIX", period="10y", progress=False)
        if "Adj Close" in vix_data.columns:
            vix = vix_data["Adj Close"]
        elif "Close" in vix_data.columns:
            vix = vix_data["Close"]
        if vix is not None:
            logger.info(f"VIX: {len(vix)} observations")
    except Exception as e:
        logger.warning(f"VIX download failed: {e}")

    return prices, vix


def compute_spy_benchmark(prices, start_date):
    if "SPY" not in prices.columns:
        return {}
    s = prices["SPY"].loc[start_date:].dropna()
    if len(s) < 2:
        return {}
    tr = (s.iloc[-1] / s.iloc[0]) - 1
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    ar = (1 + tr) ** (1 / yrs) - 1 if yrs > 0 else 0
    dr = s.pct_change().dropna()
    vol = dr.std() * np.sqrt(252)
    sh = (ar - 0.04) / vol if vol > 0 else 0
    ds = dr[dr < 0]
    dsv = ds.std() * np.sqrt(252) if len(ds) > 0 else vol
    so = (ar - 0.04) / dsv if dsv > 0 else 0
    cm = s.cummax()
    mdd = ((s / cm) - 1).min()
    cal = abs(ar / mdd) if mdd != 0 else 0
    mo = s.resample("ME").last().pct_change().dropna()
    wr = (mo > 0).mean()
    best_yr = s.resample("YE").last().pct_change().dropna()
    return {
        "total_return": tr, "annualized_return": ar,
        "annualized_volatility": vol, "sharpe_ratio": sh,
        "sortino_ratio": so, "max_drawdown": mdd,
        "calmar_ratio": cal, "win_rate_monthly": wr,
        "best_month": mo.max() if not mo.empty else 0,
        "worst_month": mo.min() if not mo.empty else 0,
        "best_year": best_yr.max() if not best_yr.empty else 0,
        "worst_year": best_yr.min() if not best_yr.empty else 0,
        "start_value": 100000, "end_value": 100000 * (1 + tr),
    }


def main():
    print()
    print("=" * 75)
    print("  J A R V I S   V 5   —   M E D A L L I O N   B A C K T E S T")
    print("  Dual-Engine: Leveraged Offense + Crisis Alpha")
    print("  37 ETFs | 5 Regimes | 8 Voting Signals | 200-SMA Hard Filter")
    print("=" * 75)
    print()

    # ── Download Data ──
    print("STEP 1: Downloading market data...")
    prices, vix = download_data()

    if prices.empty:
        print("  ❌ Failed to download data. Check network.")
        sys.exit(1)

    print(f"  ✅ {len(prices.columns)} ETFs loaded")
    print(f"  Date range: {prices.index[0].date()} → {prices.index[-1].date()}")

    # Some leveraged ETFs have shorter history
    for t in ["TQQQ", "SOXL", "TECL", "UPRO", "SQQQ", "UVXY"]:
        if t in prices.columns:
            col = prices[t].dropna()
            print(f"    {t}: {len(col)} days ({col.index[0].date()} → {col.index[-1].date()})")
    print()

    # Start date: need enough history for all leveraged ETFs
    # TQQQ launched Feb 2010, so start from 2012 to have 2 years warmup
    start_date = "2012-01-01"
    # But if running on Railway with only 2018+ data, adapt
    if prices.index[0] > pd.Timestamp("2015-01-01"):
        start_date = (prices.index[0] + pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    # ── Run V5 Backtest ──
    print("STEP 2: Running V5 Medallion backtest...")
    print(f"  Period: {start_date} → {prices.index[-1].date()}")
    print(f"  Strategy: Leveraged momentum rotation + crisis alpha")
    print(f"  Rebalancing: Weekly")
    print(f"  Risk: 200-SMA filter, 7-layer protection, leverage budget 2.5x max")
    print()

    from backtest.engine_v5 import BacktestV5

    bt = BacktestV5({
        "start_date": start_date,
        "initial_capital": 100000,
        "rebalance_days": 5,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
    })

    result = bt.run(prices, vix_series=vix)
    m5 = result.get("metrics", {})

    # ── SPY Benchmark ──
    spy = compute_spy_benchmark(prices, start_date)

    # ── Previous versions for reference ──
    # (We include V4 numbers from the last backtest for comparison)
    m4_ref = {"annualized_return": 0.069, "sharpe_ratio": 0.26, "max_drawdown": -0.163,
              "total_return": 0.706, "win_rate_monthly": 0.646, "end_value": 170632}

    # ════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("  J A R V I S   V 5   R E S U L T S")
    print("=" * 75)
    print()

    v5_end = m5.get("end_value", 0)
    spy_end = spy.get("end_value", 0)

    print(f"  {'':32s} {'V4 (ref)':>12s} {'V5':>12s} {'SPY':>12s} {'V5 vs SPY':>12s}")
    print(f"  {'':32s} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Ending Value':32s} ${m4_ref.get('end_value',0):>10,.0f} ${v5_end:>10,.0f} ${spy_end:>10,.0f}")

    rows = [
        ("Total Return", "total_return", "%"),
        ("Annualized Return", "annualized_return", "%"),
        ("Annualized Volatility", "annualized_volatility", "%"),
        ("Sharpe Ratio", "sharpe_ratio", "x"),
        ("Sortino Ratio", "sortino_ratio", "x"),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Calmar Ratio", "calmar_ratio", "x"),
        ("Win Rate (Monthly)", "win_rate_monthly", "%"),
        ("Best Month", "best_month", "%"),
        ("Worst Month", "worst_month", "%"),
        ("Best Year", "best_year", "%"),
        ("Worst Year", "worst_year", "%"),
    ]

    print()
    print(f"  {'Metric':32s} {'V4 (ref)':>12s} {'V5':>12s} {'SPY':>12s} {'V5-SPY':>12s}")
    print(f"  {'─'*32} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

    for name, key, fmt in rows:
        v4_val = m4_ref.get(key, 0) or 0
        v5_val = m5.get(key, 0) or 0
        spy_val = spy.get(key, 0) or 0
        diff = v5_val - spy_val

        if fmt == "%":
            print(f"  {name:32s} {v4_val:>+11.1%} {v5_val:>+11.1%} {spy_val:>+11.1%} {diff:>+11.1%}")
        else:
            print(f"  {name:32s} {v4_val:>11.2f} {v5_val:>11.2f} {spy_val:>11.2f} {diff:>+11.2f}")

    # Trade stats
    print(f"\n  {'Total Trades':32s} {'—':>12s} {m5.get('total_trades',0):>12,d}")
    print(f"  {'Total Costs':32s} {'—':>12s} ${m5.get('total_costs',0):>10,.0f}")

    # Regime stats
    regime_log = result.get("regime_log", [])
    if regime_log:
        from collections import Counter
        rc = Counter(r["regime"] for r in regime_log)
        n = len(regime_log)
        print(f"\n  V5 Regime Distribution:")
        for r in ["EUPHORIA", "CALM", "CAUTION", "STRESS", "CRISIS"]:
            c = rc.get(r, 0)
            bar = "█" * int(c / n * 40)
            print(f"    {r:12s} {c:3d}/{n} ({c/n:5.1%}) {bar}")

    # ═══ VERDICT ═══
    v5_ret = m5.get("annualized_return", 0)
    spy_ret = spy.get("annualized_return", 0)
    v5_sharpe = m5.get("sharpe_ratio", 0)
    v5_dd = m5.get("max_drawdown", 0)
    spy_dd = spy.get("max_drawdown", 0)
    alpha = v5_ret - spy_ret

    print(f"\n  {'═'*75}")
    print(f"  V5 ALPHA vs SPY: {alpha:+.2%} annualized")
    print(f"  V5 Annualized Return: {v5_ret:.1%}")
    print(f"  V5 Sharpe Ratio: {v5_sharpe:.2f}")
    print(f"  V5 Max Drawdown: {v5_dd:.1%}")

    if v5_ret > 0.20 and v5_sharpe > 0.7:
        print(f"\n  🏆 V5 HITS THE TARGET: {v5_ret:.1%} return with {v5_sharpe:.2f} Sharpe!")
        print(f"     This is within the blueprint's 20-30% target range.")
        print(f"     Proceed to paper trading validation.")
    elif v5_ret > 0.15 and v5_sharpe > 0.5:
        print(f"\n  ✅ V5 STRONG: {v5_ret:.1%} return with {v5_sharpe:.2f} Sharpe")
        if alpha > 0:
            print(f"     BEATS SPY by {alpha:.1%}/year — a genuine achievement")
    elif alpha > 0:
        print(f"\n  ⚠️ V5 beats SPY ({alpha:+.1%}/year) but below 20% target")
    else:
        print(f"\n  V5 needs further refinement")

    if abs(v5_dd) < abs(spy_dd):
        print(f"  ✅ Risk management: {v5_dd:.1%} drawdown vs SPY's {spy_dd:.1%}")

    print(f"  {'═'*75}")
    print()

    # Compounding projection
    if v5_ret > 0.10:
        print(f"  Compounding projection at {v5_ret:.0%} CAGR:")
        capital = 100000
        for year in [1, 3, 5, 7, 10, 15, 20]:
            future = capital * (1 + v5_ret) ** year
            print(f"    Year {year:2d}: ${future:>14,.0f}")
        print()


if __name__ == "__main__":
    main()
