"""
JARVIS V4.3 — FINAL LEVERAGE ATTEMPT
Run: pip install yfinance pandas numpy loguru --break-system-packages && python run_v43.py
"""
import sys,os,warnings
import pandas as pd, numpy as np
from loguru import logger
warnings.filterwarnings('ignore')
os.makedirs("logs",exist_ok=True)
logger.add("logs/v43.log",rotation="10 MB",level="INFO")

def download():
    import yfinance as yf
    tickers = ["SPY","QQQ","IWM","EFA","EEM","VTV","VUG","XLK","XLF","XLV","XLE","XLI",
               "TLT","IEF","GLD","BIL","SHY","DBMF","GDX","SLV","DBC","XLU","XLP",
               "QLD","SSO","ROM","UYG","DIG"]
    data = yf.download(tickers, start="2008-01-01", progress=False)
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
    cm=s.cummax(); mdd=((s/cm)-1).min()
    mo=s.resample("ME").last().pct_change().dropna()
    by=s.resample("YE").last().pct_change().dropna()
    return {"annualized_return":ar,"sharpe_ratio":sh,"max_drawdown":mdd,
            "win_rate_monthly":(mo>0).mean(),"end_value":100000*(1+tr),
            "annualized_volatility":vol,"calmar_ratio":abs(ar/mdd) if mdd else 0,
            "best_month":mo.max() if not mo.empty else 0,
            "worst_month":mo.min() if not mo.empty else 0,
            "best_year":by.max() if not by.empty else 0,
            "worst_year":by.min() if not by.empty else 0}

def main():
    print()
    print("="*75)
    print("  JARVIS V4.3 — FINAL LEVERAGE ATTEMPT")
    print("  Period: 2010+ only (all leveraged ETFs exist)")
    print("  Fixes: CB 90-day ban, cash>=0 enforced, no 3x, no inverse")
    print("  ACTIVE: Top 5 momentum, top 3 → 2x | SAFETY: Crisis alpha momentum")
    print("="*75)
    print()

    prices = download()
    if prices.empty: print("No data"); sys.exit(1)

    from backtest_v43 import BacktestV43
    bt = BacktestV43(capital=100000)
    m = bt.run(prices)

    h = pd.DataFrame(bt.hist)
    start_date = h.iloc[0]["date"] if not h.empty else "2010-06-01"
    spy = spy_bm(prices, start_date)

    # References
    m4 = {"annualized_return":0.069,"sharpe_ratio":0.26,"max_drawdown":-0.163,"end_value":170632}
    m41 = {"annualized_return":0.077,"sharpe_ratio":0.28,"max_drawdown":-0.307,"end_value":653149}

    print("="*75)
    print("  R E S U L T S  (2010 - 2026)")
    print("="*75)
    print()
    print(f"  {'':30s} {'V4 ref':>10s} {'V4.1 ref':>10s} {'V4.3':>10s} {'SPY':>10s} {'V4.3-SPY':>10s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    print(f"  {'Ending Value':30s} ${m4['end_value']:>8,.0f} ${m41['end_value']:>8,.0f} ${m.get('end_value',0):>8,.0f} ${spy.get('end_value',0):>8,.0f}")

    for label,key,fmt in [
        ("Annualized Return","annualized_return","%"),
        ("Annualized Volatility","annualized_volatility","%"),
        ("Sharpe Ratio","sharpe_ratio","x"),
        ("Sortino Ratio","sortino_ratio","x"),
        ("Max Drawdown","max_drawdown","%"),
        ("Calmar Ratio","calmar_ratio","x"),
        ("Win Rate (Monthly)","win_rate_monthly","%"),
        ("Best Month","best_month","%"),("Worst Month","worst_month","%"),
        ("Best Year","best_year","%"),("Worst Year","worst_year","%"),
    ]:
        v4v = m4.get(key,0) or 0
        v41v = m41.get(key,0) or 0
        v43v = m.get(key,0) or 0
        sv = spy.get(key,0) or 0
        d = v43v - sv
        if fmt=="%":
            print(f"  {label:30s} {v4v:>+9.1%} {v41v:>+9.1%} {v43v:>+9.1%} {sv:>+9.1%} {d:>+9.1%}")
        else:
            print(f"  {label:30s} {v4v:>9.2f} {v41v:>9.2f} {v43v:>9.2f} {sv:>9.2f} {d:>+9.2f}")

    print(f"\n  {'Total Trades':30s} {'':>10s} {'':>10s} {m.get('total_trades',0):>10,d}")
    print(f"  {'Total Costs':30s} {'':>10s} {'':>10s} ${m.get('total_costs',0):>8,.0f}")
    print(f"  {'Avg Cash %':30s} {'~30%':>10s} {'15.3%':>10s} {m.get('avg_cash_pct',0):>9.1%}")
    print(f"  {'Active Mode %':30s} {'':>10s} {'':>10s} {m.get('pct_active',0):>9.1%}")

    ar = m.get("annualized_return",0)
    sh = m.get("sharpe_ratio",0)
    dd = m.get("max_drawdown",0)
    alpha = ar - spy.get("annualized_return",0)

    print(f"\n  {'═'*75}")

    # Sanity checks
    ev = m.get("end_value",0)
    if ev > 50_000_000:
        print(f"  ⚠️ BUG: ${ev:,.0f} is unrealistic. Leverage compounding still broken.")
    elif ev < 100:
        print(f"  ⚠️ BUG: ${ev:,.0f} — portfolio went to near zero.")
    elif ar < -0.05:
        print(f"  ⚠️ FAILED: {ar:.1%} return. Strategy is losing money.")
    else:
        print(f"  V4.3 Return: {ar:.1%} | Sharpe: {sh:.2f} | Max DD: {dd:.1%}")
        print(f"  Alpha vs SPY: {alpha:+.2%}/year")

        checks = []
        if ar >= 0.15: checks.append(f"  ✅ Return {ar:.1%} ≥ 15% TARGET HIT")
        else: checks.append(f"  ❌ Return {ar:.1%} < 15% target")
        if sh >= 0.50: checks.append(f"  ✅ Sharpe {sh:.2f} ≥ 0.50 TARGET HIT")
        else: checks.append(f"  ❌ Sharpe {sh:.2f} < 0.50 target")
        if alpha > 0: checks.append(f"  ✅ Beats SPY by {alpha:+.1%}")
        else: checks.append(f"  ❌ Trails SPY by {alpha:.1%}")
        if dd > -0.40: checks.append(f"  ✅ Drawdown {dd:.1%} within tolerance")
        else: checks.append(f"  ⚠️ Drawdown {dd:.1%} is severe")

        print(f"\n  TARGET CHECK:")
        for c in checks: print(f"  {c}")

        if ar >= 0.15 and sh >= 0.50:
            print(f"\n  🏆 BOTH TARGETS HIT!")
        elif ar > m41.get("annualized_return",0):
            print(f"\n  ✅ Beats V4.1 ({ar:.1%} vs {m41['annualized_return']:.1%})")

    if 0.05 < ar < 0.50:
        print(f"\n  Compounding at {ar:.0%} CAGR:")
        for yr in [1,3,5,7,10,15,20]:
            print(f"    Year {yr:2d}: ${100000*(1+ar)**yr:>14,.0f}")

    print(f"\n  {'═'*75}\n")

if __name__=="__main__":
    main()
