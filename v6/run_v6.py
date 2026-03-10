"""
JARVIS V6.1 — Fixed Leverage Backtest Runner
Run: pip install yfinance pandas numpy loguru --break-system-packages && python run_v6.py
"""
import sys, os, warnings
import pandas as pd, numpy as np
from loguru import logger

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.makedirs("logs", exist_ok=True)
logger.add("logs/v6.log", rotation="10 MB", level="INFO")

def download():
    import yfinance as yf
    tickers = ["QQQ","SPY","SMH","SOXX","VGT","XLK","XLV","XLE","GLD","TLT","IEF","DBMF",
               "QLD","SSO","ROM","USD","SHY","BIL"]
    logger.info(f"Downloading {len(tickers)} ETFs...")
    data = yf.download(tickers, period="max", progress=False)
    adj = data.get("Adj Close", data.get("Close", pd.DataFrame()))
    if isinstance(adj, pd.Series): adj = adj.to_frame(tickers[0])
    prices = adj.ffill().dropna(how="all")
    logger.info(f"Loaded {len(prices.columns)} ETFs, {len(prices)} days")
    vix = None
    try:
        vd = yf.download("^VIX", period="max", progress=False)
        vix = vd.get("Adj Close", vd.get("Close"))
    except: pass
    return prices, vix

def spy_bm(prices, start):
    s = prices["SPY"].loc[start:].dropna()
    if len(s)<2: return {}
    tr=(s.iloc[-1]/s.iloc[0])-1; y=(s.index[-1]-s.index[0]).days/365.25
    ar=(1+tr)**(1/y)-1 if y>0 else 0; dr=s.pct_change().dropna()
    vol=dr.std()*np.sqrt(252); sh=(ar-0.04)/vol if vol>0 else 0
    cm=s.cummax(); mdd=((s/cm)-1).min()
    mo=s.resample("ME").last().pct_change().dropna()
    by=s.resample("YE").last().pct_change().dropna()
    return {"total_return":tr,"annualized_return":ar,"annualized_volatility":vol,
            "sharpe_ratio":sh,"max_drawdown":mdd,"calmar_ratio":abs(ar/mdd) if mdd else 0,
            "win_rate_monthly":(mo>0).mean(),"best_month":mo.max() if not mo.empty else 0,
            "worst_month":mo.min() if not mo.empty else 0,
            "best_year":by.max() if not by.empty else 0,
            "worst_year":by.min() if not by.empty else 0,
            "end_value":100000*(1+tr),"total_trades":0,"total_costs":0}

def main():
    print()
    print("="*75)
    print("  JARVIS V6.1 — FIXED LEVERAGE COMPOUNDING")
    print("  Three Pillars: Momentum | Vol Filter | 200-Day SMA")
    print("  Max 2x leverage | No inverse ETFs | Cash-constrained trades")
    print("="*75)
    print()

    prices, vix = download()
    if prices.empty: print("  ❌ No data"); sys.exit(1)

    from backtest_v6 import BacktestV6
    r6 = BacktestV6({"initial_capital":100000}).run(prices, vix)
    m6 = r6.get("metrics", {})

    start_date = pd.DataFrame(r6.get("portfolio_history",[])).iloc[0]["date"] if r6.get("portfolio_history") else prices.index[300]
    spy = spy_bm(prices, start_date)

    m4 = {"annualized_return":0.069,"sharpe_ratio":0.26,"max_drawdown":-0.163,
          "end_value":170632,"annualized_volatility":0.113,"calmar_ratio":0.42,
          "win_rate_monthly":0.646,"total_trades":1665,"total_costs":12323,
          "best_month":0.082,"worst_month":-0.089}

    # ── RESULTS ──
    print()
    print("="*75)
    print("  R E S U L T S   (V6.1 Fixed)")
    print("="*75)
    print()

    all_m = [("V4 (ref)", m4), ("V6.1", m6), ("SPY", spy)]
    print(f"  {'':32s}", end="")
    for n,_ in all_m: print(f" {n:>12s}", end="")
    print(f" {'V6.1-SPY':>12s}")
    print(f"  {'─'*32}", end="")
    for _ in all_m: print(f" {'─'*12}", end="")
    print(f" {'─'*12}")

    print(f"  {'Ending Value':32s}", end="")
    for _,m in all_m: print(f" ${m.get('end_value',0):>10,.0f}", end="")
    print()

    for label, key, fmt in [
        ("Total Return","total_return","%"),("Annualized Return","annualized_return","%"),
        ("Annualized Volatility","annualized_volatility","%"),("Sharpe Ratio","sharpe_ratio","x"),
        ("Max Drawdown","max_drawdown","%"),("Calmar Ratio","calmar_ratio","x"),
        ("Win Rate (Monthly)","win_rate_monthly","%"),
        ("Best Month","best_month","%"),("Worst Month","worst_month","%"),
        ("Best Year","best_year","%"),("Worst Year","worst_year","%"),
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

    print(f"\n  {'Total Trades':32s}", end="")
    for _,m in all_m: print(f" {int(m.get('total_trades',0)):>12,d}" if m.get('total_trades') else f" {'—':>12s}", end="")
    print()
    print(f"  {'Total Costs':32s}", end="")
    for _,m in all_m: print(f" ${m.get('total_costs',0):>10,.0f}" if m.get('total_costs') else f" {'—':>12s}", end="")
    print()

    if m6.get("pct_active_mode") is not None:
        print(f"\n  V6.1 Mode: ACTIVE {m6['pct_active_mode']:.0%} | "
              f"SAFETY {1-m6['pct_active_mode']:.0%} | "
              f"LEVERAGED {m6.get('pct_leveraged',0):.0%} of time")

    # Sanity checks
    ev = m6.get("end_value", 0)
    ar = m6.get("annualized_return", 0)
    print(f"\n  {'═'*75}")

    if ev > 10_000_000:
        print(f"  ⚠️ STILL BROKEN: ${ev:,.0f} ending value is unrealistic.")
        print(f"     The leverage compounding bug may not be fully fixed.")
    elif ar > 0.50:
        print(f"  ⚠️ SUSPICIOUS: {ar:.0%} annualized return seems too high.")
        print(f"     Verify with manual calculation.")
    else:
        alpha = ar - spy.get("annualized_return", 0)
        sh = m6.get("sharpe_ratio", 0)
        dd = m6.get("max_drawdown", 0)
        spy_dd = spy.get("max_drawdown", 0)

        print(f"  V6.1 ALPHA vs SPY: {alpha:+.2%} annualized")
        print(f"  V6.1 Return: {ar:.1%} | Sharpe: {sh:.2f} | Max DD: {dd:.1%}")

        if alpha > 0 and sh > 0.5:
            print(f"\n  🏆 V6.1 BEATS SPY with Sharpe > 0.5!")
        elif alpha > 0:
            print(f"\n  ✅ V6.1 beats SPY on raw return ({alpha:+.1%}/year)")
        elif ar > m4["annualized_return"]:
            print(f"\n  ✅ V6.1 beats V4 ({ar:.1%} vs {m4['annualized_return']:.1%})")
        else:
            print(f"\n  ⚠️ V6.1 needs refinement")

        if ar > 0.05:
            print(f"\n  Compounding at {ar:.0%} CAGR:")
            for yr in [1,3,5,7,10,15,20]:
                print(f"    Year {yr:2d}: ${100000*(1+ar)**yr:>14,.0f}")

    print(f"  {'═'*75}\n")

if __name__ == "__main__":
    main()
