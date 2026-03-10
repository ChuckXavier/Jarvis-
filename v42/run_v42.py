"""
JARVIS V4.2 — Momentum + Simple Conditional Leverage
Run: pip install yfinance pandas numpy loguru --break-system-packages && python run_v42.py
"""
import sys, os, warnings
import pandas as pd, numpy as np
from loguru import logger

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.makedirs("logs", exist_ok=True)
logger.add("logs/v42.log", rotation="10 MB", level="INFO")

def download():
    import yfinance as yf
    tickers = ["SPY","QQQ","IWM","EFA","EEM","VTV","VUG","XLK","XLF","XLV","XLE","XLI",
               "TLT","IEF","GLD","BIL","SHY","DBMF",
               "QLD","SSO","ROM","USD","UYG","DIG",
               "GDX","SLV","CTA","DBC","XLU","XLP"]  # Crisis alpha pool
    logger.info(f"Downloading {len(tickers)} ETFs...")
    data = yf.download(tickers, period="max", progress=False)
    adj = data.get("Adj Close", data.get("Close", pd.DataFrame()))
    if isinstance(adj, pd.Series): adj = adj.to_frame(tickers[0])
    prices = adj.ffill().dropna(how="all")
    logger.info(f"Loaded {len(prices.columns)} ETFs, {len(prices)} days")
    # Show leveraged ETF date ranges
    for t in ["QLD","SSO","ROM","USD","UYG","DIG"]:
        if t in prices.columns:
            col = prices[t].dropna()
            if not col.empty:
                logger.info(f"  {t}: {col.index[0].date()} → {col.index[-1].date()} ({len(col)} days)")
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
    mo=s.resample("ME").last().pct_change().dropna()
    by=s.resample("YE").last().pct_change().dropna()
    return {"total_return":tr,"annualized_return":ar,"annualized_volatility":vol,
            "sharpe_ratio":sh,"sortino_ratio":so,"max_drawdown":mdd,
            "calmar_ratio":abs(ar/mdd) if mdd!=0 else 0,
            "win_rate_monthly":(mo>0).mean(),
            "best_month":mo.max() if not mo.empty else 0,
            "worst_month":mo.min() if not mo.empty else 0,
            "best_year":by.max() if not by.empty else 0,
            "worst_year":by.min() if not by.empty else 0,
            "end_value":100000*(1+tr),"total_trades":0,"total_costs":0}

def main():
    print()
    print("="*78)
    print("  JARVIS V4.2 — MOMENTUM + LEVERAGE + CRISIS ALPHA")
    print("  ACTIVE: Top 5 momentum, top 3 get 2x leverage (QQQ>200-SMA)")
    print("  SAFETY: Crisis Alpha momentum rotation (GLD/GDX/DBMF/TLT by momentum)")
    print("  No regime prediction | No inverse ETFs | No 3x | Monthly rebalance")
    print("="*78)
    print()

    prices = download()
    if prices.empty: print("  ❌ No data"); sys.exit(1)
    print(f"  Loaded {len(prices.columns)} ETFs\n")

    from backtest_v42 import BacktestV42
    r42 = BacktestV42({"initial_capital": 100000}).run(prices)
    m42 = r42.get("metrics", {})

    hist = r42.get("portfolio_history", pd.DataFrame())
    start_date = hist.iloc[0]["date"] if not hist.empty else prices.index[500]
    spy = spy_bm(prices, start_date)

    m4 = {"annualized_return":0.069,"sharpe_ratio":0.26,"max_drawdown":-0.163,
          "end_value":170632,"annualized_volatility":0.113,"calmar_ratio":0.42,
          "sortino_ratio":0.34,"win_rate_monthly":0.646,"total_trades":1665,
          "total_costs":12323,"avg_cash_pct":0.30}

    m41 = {"annualized_return":0.077,"sharpe_ratio":0.28,"max_drawdown":-0.307,
           "end_value":653149,"annualized_volatility":0.13,"calmar_ratio":0.25,
           "sortino_ratio":0.33,"win_rate_monthly":0.65,"total_trades":1119,
           "total_costs":3000,"avg_cash_pct":0.153}

    # ══════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════
    print()
    print("="*78)
    print("  R E S U L T S:  V4 vs V4.1 vs V4.2 vs SPY")
    print("="*78)
    print()

    all_m = [("V4", m4), ("V4.1", m41), ("V4.2", m42), ("SPY", spy)]

    print(f"  {'':32s}", end="")
    for n,_ in all_m: print(f" {n:>10s}", end="")
    print(f" {'V4.2-SPY':>10s}")
    print(f"  {'─'*32}", end="")
    for _ in all_m: print(f" {'─'*10}", end="")
    print(f" {'─'*10}")

    print(f"  {'Ending Value':32s}", end="")
    for _,m in all_m: print(f" ${m.get('end_value',0):>8,.0f}", end="")
    print()

    for label, key, fmt in [
        ("Annualized Return","annualized_return","%"),
        ("Annualized Volatility","annualized_volatility","%"),
        ("Sharpe Ratio","sharpe_ratio","x"),
        ("Sortino Ratio","sortino_ratio","x"),
        ("Max Drawdown","max_drawdown","%"),
        ("Calmar Ratio","calmar_ratio","x"),
        ("Win Rate (Monthly)","win_rate_monthly","%"),
        ("Best Month","best_month","%"),
        ("Worst Month","worst_month","%"),
        ("Best Year","best_year","%"),
        ("Worst Year","worst_year","%"),
    ]:
        print(f"  {label:32s}", end="")
        vals = []
        for _,m in all_m:
            v = m.get(key,0) or 0; vals.append(v)
            if fmt=="%": print(f" {v:>+9.1%}", end="")
            else: print(f" {v:>9.2f}", end="")
        diff = vals[2]-vals[3]
        if fmt=="%": print(f" {diff:>+9.1%}")
        else: print(f" {diff:>+9.2f}")

    print()
    print(f"  {'Avg Cash Held':32s} {'~30%':>10s} {'15.3%':>10s} {m42.get('avg_cash_pct',0):>+9.1%} {'0%':>10s}")
    print(f"  {'Avg in 2x ETFs':32s} {'0%':>10s} {'0%':>10s} {m42.get('avg_leverage_pct',0):>+9.1%} {'0%':>10s}")
    print(f"  {'Total Trades':32s}", end="")
    for _,m in all_m: print(f" {int(m.get('total_trades',0)):>10,d}" if m.get('total_trades') else f" {'—':>10s}", end="")
    print()
    print(f"  {'Total Costs':32s}", end="")
    for _,m in all_m:
        c=m.get('total_costs',0)
        print(f" ${c:>8,.0f}" if c else f" {'—':>10s}", end="")
    print()

    if m42.get("pct_active_mode") is not None:
        print(f"\n  V4.2 Mode: ACTIVE {m42['pct_active_mode']:.0%} | SAFETY {1-m42['pct_active_mode']:.0%}")

    # Sanity check
    ev = m42.get("end_value", 0)
    ar = m42.get("annualized_return", 0)

    print(f"\n  {'═'*78}")
    if ev > 50_000_000:
        print(f"  ⚠️ SUSPICIOUS: ${ev:,.0f} ending value may indicate a bug.")
    elif ar > 0.50:
        print(f"  ⚠️ SUSPICIOUS: {ar:.0%} return seems too high. Verify.")
    else:
        alpha = ar - spy.get("annualized_return", 0)
        sh = m42.get("sharpe_ratio", 0)
        dd = m42.get("max_drawdown", 0)

        print(f"  V4.2 RESULTS:")
        print(f"    Annualized Return: {ar:.1%}")
        print(f"    Sharpe Ratio:      {sh:.2f}")
        print(f"    Max Drawdown:      {dd:.1%}")
        print(f"    Alpha vs SPY:      {alpha:+.2%}/year")

        # Check against targets
        target_met = []
        if ar >= 0.15: target_met.append(f"✅ Return {ar:.1%} ≥ 15%")
        else: target_met.append(f"❌ Return {ar:.1%} < 15% target")
        if sh >= 0.50: target_met.append(f"✅ Sharpe {sh:.2f} ≥ 0.50")
        else: target_met.append(f"❌ Sharpe {sh:.2f} < 0.50 target")
        if alpha > 0: target_met.append(f"✅ Beats SPY by {alpha:+.1%}")
        else: target_met.append(f"❌ Trails SPY by {alpha:.1%}")

        print(f"\n  TARGET CHECK:")
        for t in target_met:
            print(f"    {t}")

        if ar >= 0.15 and sh >= 0.50:
            print(f"\n  🏆 BOTH TARGETS HIT! Ready for deployment.")
        elif ar >= 0.10 and sh >= 0.35:
            print(f"\n  ⚠️ Promising but below targets. Fine-tuning may close the gap.")

    if ar > 0.05 and ar < 0.50:
        print(f"\n  Compounding at {ar:.0%} CAGR ($100K start):")
        for yr in [1,3,5,7,10,15,20]:
            print(f"    Year {yr:2d}: ${100000*(1+ar)**yr:>14,.0f}")

    print(f"\n  {'═'*78}\n")

if __name__ == "__main__":
    main()

