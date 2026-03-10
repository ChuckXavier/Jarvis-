"""
JARVIS V6 - Volatility-Aware Momentum Engine Backtest
=======================================================
Self-contained. Downloads data. Compares V4 vs V6 vs SPY.

HOW TO RUN:
    pip install yfinance pandas numpy loguru --break-system-packages
    python run_v6.py
"""

import sys, os, warnings
import pandas as pd
import numpy as np
from loguru import logger

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

os.makedirs("logs", exist_ok=True)
logger.add("logs/v6.log", rotation="10 MB", level="INFO")


def download():
    import yfinance as yf
    tickers = [
        "QQQ", "SPY", "SMH", "SOXX", "VGT", "XLK", "XLV", "XLE",
        "GLD", "TLT", "IEF", "DBMF",
        "QLD", "SSO", "ROM",
        "SHY", "BIL",
    ]
    logger.info(f"Downloading {len(tickers)} ETFs...")
    data = yf.download(tickers, period="max", progress=False)
    adj = data.get("Adj Close", data.get("Close", pd.DataFrame()))
    if isinstance(adj, pd.Series):
        adj = adj.to_frame(tickers[0])
    prices = adj.ffill().dropna(how="all")
    logger.info(f"Loaded {len(prices.columns)} ETFs, {len(prices)} days")

    vix = None
    try:
        vd = yf.download("^VIX", period="max", progress=False)
        vix = vd.get("Adj Close", vd.get("Close"))
        logger.info(f"VIX: {len(vix)} days")
    except:
        pass

    return prices, vix


def spy_benchmark(prices, start):
    s = prices["SPY"].loc[start:].dropna()
    if len(s) < 2: return {}
    tr = (s.iloc[-1] / s.iloc[0]) - 1
    y = (s.index[-1] - s.index[0]).days / 365.25
    ar = (1+tr)**(1/y)-1 if y > 0 else 0
    dr = s.pct_change().dropna()
    vol = dr.std()*np.sqrt(252)
    sh = (ar-0.04)/vol if vol>0 else 0
    cm = s.cummax(); mdd = ((s/cm)-1).min()
    mo = s.resample("ME").last().pct_change().dropna()
    by = s.resample("YE").last().pct_change().dropna()
    return {"total_return": tr, "annualized_return": ar, "annualized_volatility": vol,
            "sharpe_ratio": sh, "max_drawdown": mdd, "calmar_ratio": abs(ar/mdd) if mdd else 0,
            "win_rate_monthly": (mo>0).mean(),
            "best_month": mo.max() if not mo.empty else 0,
            "worst_month": mo.min() if not mo.empty else 0,
            "best_year": by.max() if not by.empty else 0,
            "worst_year": by.min() if not by.empty else 0,
            "end_value": 100000*(1+tr), "total_trades": 0, "total_costs": 0}


def main():
    print()
    print("=" * 75)
    print("  J A R V I S   V 6  —  V O L A T I L I T Y - A W A R E   M O M E N T U M")
    print("  Three Pillars: Momentum Core | Vol Filter | 200-Day SMA Switch")
    print("  No 3x ETFs | No inverse ETFs | Max 2 trades/week | 5% drift threshold")
    print("=" * 75)
    print()

    prices, vix = download()
    if prices.empty:
        print("  ❌ No data"); sys.exit(1)

    # Use same start as V4 backtests for fair comparison
    start_date = (prices.index[0] + pd.Timedelta(days=800)).strftime("%Y-%m-%d")
    if pd.Timestamp(start_date) < pd.Timestamp("2014-01-01"):
        start_date = "2014-01-01"

    print(f"  Data: {len(prices.columns)} ETFs")
    print(f"  Backtest: {start_date} → {prices.index[-1].date()}\n")

    # ── V6 ──
    print("  Running V6 (Volatility-Aware Momentum)...")
    from backtest_v6 import BacktestV6
    r6 = BacktestV6({"initial_capital": 100000}).run(prices.loc[start_date:], vix)
    m6 = r6.get("metrics", {})
    print(f"  → V6: ${m6.get('end_value',0):,.0f} | {m6.get('annualized_return',0):+.1%} | Sharpe {m6.get('sharpe_ratio',0):.2f}\n")

    # ── SPY ──
    spy = spy_benchmark(prices, start_date)

    # V4 reference
    m4 = {"annualized_return": 0.069, "sharpe_ratio": 0.26, "max_drawdown": -0.163,
          "total_return": 0.706, "win_rate_monthly": 0.646, "end_value": 170632,
          "annualized_volatility": 0.113, "calmar_ratio": 0.42, "total_trades": 1665,
          "total_costs": 12323, "best_month": 0.082, "worst_month": -0.089}

    # V5 reference
    m5 = {"annualized_return": -0.010, "sharpe_ratio": -0.23, "max_drawdown": -0.723,
          "end_value": 90999, "total_trades": 5859, "total_costs": 46710}

    # ════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════
    print("=" * 75)
    print("  R E S U L T S")
    print("=" * 75)
    print()

    all_m = [("V4 (ref)", m4), ("V5 (ref)", m5), ("V6", m6), ("SPY", spy)]

    print(f"  {'':32s}", end="")
    for name, _ in all_m: print(f" {name:>12s}", end="")
    print()
    print(f"  {'':32s}", end="")
    for _ in all_m: print(f" {'─'*12}", end="")
    print()

    print(f"  {'Ending Value':32s}", end="")
    for _, m in all_m: print(f" ${m.get('end_value',0):>10,.0f}", end="")
    print()

    rows = [
        ("Total Return", "total_return", "%"),
        ("Annualized Return", "annualized_return", "%"),
        ("Annualized Volatility", "annualized_volatility", "%"),
        ("Sharpe Ratio", "sharpe_ratio", "x"),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Calmar Ratio", "calmar_ratio", "x"),
        ("Win Rate (Monthly)", "win_rate_monthly", "%"),
        ("Best Month", "best_month", "%"),
        ("Worst Month", "worst_month", "%"),
        ("Best Year", "best_year", "%"),
        ("Worst Year", "worst_year", "%"),
    ]

    print()
    print(f"  {'Metric':32s}", end="")
    for name, _ in all_m: print(f" {name:>12s}", end="")
    v6_spy_diff = " V6-SPY"
    print(f" {v6_spy_diff:>12s}")

    print(f"  {'─'*32}", end="")
    for _ in all_m: print(f" {'─'*12}", end="")
    print(f" {'─'*12}")

    for label, key, fmt in rows:
        print(f"  {label:32s}", end="")
        vals = []
        for _, m in all_m:
            v = m.get(key, 0) or 0
            vals.append(v)
            if fmt == "%": print(f" {v:>+11.1%}", end="")
            else: print(f" {v:>11.2f}", end="")
        diff = vals[2] - vals[3]
        if fmt == "%": print(f" {diff:>+11.1%}", end="")
        else: print(f" {diff:>+11.2f}", end="")
        print()

    # Costs
    print()
    print(f"  {'Total Trades':32s}", end="")
    for _, m in all_m:
        t = m.get("total_trades", 0)
        print(f" {int(t):>12,d}" if t else f" {'—':>12s}", end="")
    print()
    print(f"  {'Total Costs':32s}", end="")
    for _, m in all_m:
        c = m.get("total_costs", 0)
        print(f" ${c:>10,.0f}" if c else f" {'—':>12s}", end="")
    print()

    # V6 time in modes
    if m6.get("pct_active_mode") is not None:
        print(f"\n  V6 Mode Distribution:")
        print(f"    ACTIVE:    {m6.get('pct_active_mode',0):.0%} of time")
        print(f"    SAFETY:    {1-m6.get('pct_active_mode',0):.0%} of time")
        print(f"    LEVERAGED: {m6.get('pct_leveraged',0):.0%} of time (within ACTIVE)")

    # ═══ VERDICT ═══
    v6_ret = m6.get("annualized_return", 0)
    spy_ret = spy.get("annualized_return", 0)
    v6_sh = m6.get("sharpe_ratio", 0)
    v6_dd = m6.get("max_drawdown", 0)
    spy_dd = spy.get("max_drawdown", 0)
    alpha = v6_ret - spy_ret

    print(f"\n  {'═'*75}")
    print(f"  V6 ALPHA vs SPY: {alpha:+.2%} annualized")
    print(f"  V6 Return: {v6_ret:.1%} | Sharpe: {v6_sh:.2f} | Max DD: {v6_dd:.1%}")

    if alpha > 0 and v6_sh > 0.5:
        print(f"\n  🏆 V6 BEATS SPY: {alpha:+.1%}/year with Sharpe {v6_sh:.2f}!")
        print(f"     READY for paper trading validation.")
    elif alpha > 0:
        print(f"\n  ✅ V6 beats SPY on raw return ({alpha:+.1%}/year)")
        if abs(v6_dd) < abs(spy_dd):
            print(f"     AND with better risk ({v6_dd:.1%} vs {spy_dd:.1%} drawdown)")
    elif v6_ret > m4.get("annualized_return", 0):
        print(f"\n  ⚠️ V6 beats V4 ({v6_ret:.1%} vs {m4['annualized_return']:.1%}) but trails SPY")
        if abs(v6_dd) < abs(spy_dd):
            print(f"     Risk-adjusted: better than SPY ({v6_dd:.1%} vs {spy_dd:.1%} drawdown)")
    else:
        print(f"\n  ⚠️ V6 needs refinement")

    # Compounding
    if v6_ret > 0.05:
        print(f"\n  Compounding at {v6_ret:.0%} CAGR:")
        for yr in [1, 3, 5, 7, 10, 15, 20]:
            print(f"    Year {yr:2d}: ${100000*(1+v6_ret)**yr:>14,.0f}")

    print(f"\n  {'═'*75}")
    print()


if __name__ == "__main__":
    main()
