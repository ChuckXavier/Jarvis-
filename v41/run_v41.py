"""
JARVIS V4.1 — Zero Cash Drag Backtest
Run: pip install yfinance pandas numpy loguru --break-system-packages && python run_v41.py
"""
import sys, os, warnings
import pandas as pd, numpy as np
from loguru import logger

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.makedirs("logs", exist_ok=True)
logger.add("logs/v41.log", rotation="10 MB", level="INFO")

def download():
    import yfinance as yf
    tickers = ["SPY","QQQ","IWM","EFA","EEM","VTV","VUG","XLK","XLF","XLV","XLE","XLI",
               "TLT","IEF","GLD","BIL","SHY","DBMF","HYG","LQD","DBC","VNQ"]
    logger.info(f"Downloading {len(tickers)} ETFs...")
    data = yf.download(tickers, period="max", progress=False)
    adj = data.get("Adj Close", data.get("Close", pd.DataFrame()))
    if isinstance(adj, pd.Series): adj = adj.to_frame(tickers[0])
    prices = adj.ffill().dropna(how="all")
    logger.info(f"Loaded {len(prices.columns)} ETFs, {len(prices)} days")
    return prices

def spy_bm(prices, start):
    s = prices["SPY"].loc[start:].dropna()
    if len(s)<2: return {}
    tr=(s.iloc[-1]/s.iloc[0])-1; y=(s.index[-1]-s.index[0]).days/365.25
    ar=(1+tr)**(1/y)-1 if y>0 else 0; dr=s.pct_change().dropna()
    vol=dr.std()*np.sqrt(252); sh=(ar-0.04)/vol if vol>0 else 0
    ds=dr[dr<0]; dsv=ds.std()*np.sqrt(252) if len(ds)>0 else vol
    so=(ar-0.04)/dsv if dsv>0 else 0
    cm=s.cummax(); mdd=((s/cm)-1).min()
    cal=abs(ar/mdd) if mdd!=0 else 0
    mo=s.resample("ME").last().pct_change().dropna()
    by=s.resample("YE").last().pct_change().dropna()
    return {"total_return":tr,"annualized_return":ar,"annualized_volatility":vol,
            "sharpe_ratio":sh,"sortino_ratio":so,"max_drawdown":mdd,"calmar_ratio":cal,
            "win_rate_monthly":(mo>0).mean(),
            "best_month":mo.max() if not mo.empty else 0,
            "worst_month":mo.min() if not mo.empty else 0,
            "best_year":by.max() if not by.empty else 0,
            "worst_year":by.min() if not by.empty else 0,
            "end_value":100000*(1+tr),"total_trades":0,"total_costs":0}

def main():
    print()
    print("="*75)
    print("  JARVIS V4.1 — ZERO CASH DRAG MOMENTUM ROTATOR")
    print("  FIX 1: Cash → BIL (earns risk-free rate)")
    print("  FIX 2: Unfilled allocation → SPY (earns market beta)")
    print("  FIX 3: Full position sizing (no fractional Kelly)")
    print("  FIX 4: 100% invested at all times (SMA handles crash protection)")
    print("="*75)
    print()

    prices = download()
    if prices.empty: print("  ❌ No data"); sys.exit(1)

    print(f"  Data: {len(prices.columns)} ETFs, {len(prices)} days\n")

    # ── V4.1 ──
    print("  Running V4.1 (Zero Cash Drag)...")
    from backtest_v41 import BacktestV41
    r41 = BacktestV41({"initial_capital": 100000}).run(prices)
    m41 = r41.get("metrics", {})
    print(f"  → V4.1: ${m41.get('end_value',0):,.0f} | {m41.get('annualized_return',0):+.1%} | "
          f"Sharpe {m41.get('sharpe_ratio',0):.2f} | Avg cash: {m41.get('avg_cash_pct',0):.1%}\n")

    # Get start date from V4.1 history
    hist = r41.get("portfolio_history", pd.DataFrame())
    if not hist.empty:
        start_date = hist.iloc[0]["date"]
    else:
        start_date = prices.index[400]

    spy = spy_bm(prices, start_date)

    # V4 reference (from prior backtests)
    m4 = {"annualized_return":0.069,"sharpe_ratio":0.26,"max_drawdown":-0.163,
          "end_value":170632,"annualized_volatility":0.113,"calmar_ratio":0.42,
          "sortino_ratio":0.34,"win_rate_monthly":0.646,"total_trades":1665,
          "total_costs":12323,"best_month":0.082,"worst_month":-0.089,
          "avg_cash_pct":0.30}

    # ════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════
    print("="*75)
    print("  R E S U L T S:  V4 vs V4.1 (Zero Cash) vs SPY")
    print("="*75)
    print()

    all_m = [("V4", m4), ("V4.1", m41), ("SPY", spy)]

    print(f"  {'':32s}", end="")
    for n,_ in all_m: print(f" {n:>12s}", end="")
    print(f" {'V4.1-SPY':>12s}")
    print(f"  {'─'*32}", end="")
    for _ in all_m: print(f" {'─'*12}", end="")
    print(f" {'─'*12}")

    print(f"  {'Ending Value':32s}", end="")
    for _,m in all_m: print(f" ${m.get('end_value',0):>10,.0f}", end="")
    print()

    for label, key, fmt in [
        ("Total Return","total_return","%"),
        ("Annualized Return","annualized_return","%"),
        ("Annualized Volatility","annualized_volatility","%"),
        ("Sharpe Ratio","sharpe_ratio","x"),
        ("Sortino Ratio","sortino_ratio","x"),
        ("Max Drawdown","max_drawdown","%"),
        ("Calmar Ratio","calmar_ratio","x"),
        ("Win Rate (Monthly)","win_rate_monthly","%"),
        ("Profit Factor","profit_factor","x"),
        ("Best Month","best_month","%"),
        ("Worst Month","worst_month","%"),
        ("Best Year","best_year","%"),
        ("Worst Year","worst_year","%"),
    ]:
        print(f"  {label:32s}", end="")
        vals = []
        for _,m in all_m:
            v = m.get(key,0) or 0; vals.append(v)
            if fmt=="%": print(f" {v:>+11.1%}", end="")
            else: print(f" {v:>11.2f}", end="")
        diff = vals[1]-vals[2]
        if fmt=="%": print(f" {diff:>+11.1%}")
        else: print(f" {diff:>+11.2f}")

    # Cash and trades
    print()
    print(f"  {'Avg Cash Held':32s} {m4.get('avg_cash_pct',0):>+11.1%} {m41.get('avg_cash_pct',0):>+11.1%} {'0.0%':>12s}")
    print(f"  {'Total Trades':32s} {m4.get('total_trades',0):>12,d} {m41.get('total_trades',0):>12,d} {'—':>12s}")
    print(f"  {'Total Costs':32s} ${m4.get('total_costs',0):>10,.0f} ${m41.get('total_costs',0):>10,.0f} {'—':>12s}")

    if m41.get("pct_active_mode") is not None:
        print(f"\n  V4.1 Mode: ACTIVE {m41['pct_active_mode']:.0%} | SAFETY {1-m41['pct_active_mode']:.0%}")

    # Improvement analysis
    v4_ret = m4.get("annualized_return", 0)
    v41_ret = m41.get("annualized_return", 0)
    spy_ret = spy.get("annualized_return", 0)
    improvement = v41_ret - v4_ret
    alpha = v41_ret - spy_ret

    print(f"\n  {'═'*75}")
    print(f"  CASH DRAG ELIMINATION IMPACT:")
    print(f"    V4 return:   {v4_ret:+.1%} (avg ~30% cash)")
    print(f"    V4.1 return: {v41_ret:+.1%} (avg {m41.get('avg_cash_pct',0):.1%} cash)")
    print(f"    Improvement: {improvement:+.1%}")
    print()
    print(f"  V4.1 ALPHA vs SPY: {alpha:+.2%} annualized")

    v41_sh = m41.get("sharpe_ratio", 0)
    v41_dd = m41.get("max_drawdown", 0)
    spy_dd = spy.get("max_drawdown", 0)

    if alpha > 0 and v41_sh > 0.5:
        print(f"\n  🏆 V4.1 BEATS SPY: {alpha:+.1%}/year with Sharpe {v41_sh:.2f}!")
    elif alpha > 0:
        print(f"\n  ✅ V4.1 beats SPY by {alpha:+.1%}/year")
    elif v41_ret > v4_ret:
        print(f"\n  ✅ V4.1 beats V4 by {improvement:+.1%}/year")
        if abs(v41_dd) < abs(spy_dd):
            print(f"     Risk-adjusted: better drawdown ({v41_dd:.1%} vs SPY {spy_dd:.1%})")
    else:
        print(f"\n  ⚠️ Cash elimination didn't help as expected")

    if v41_ret > 0.05:
        print(f"\n  Compounding at {v41_ret:.0%} CAGR ($100K start):")
        for yr in [1,3,5,7,10,15,20]:
            print(f"    Year {yr:2d}: ${100000*(1+v41_ret)**yr:>14,.0f}")

    print(f"\n  {'═'*75}\n")

if __name__ == "__main__":
    main()
