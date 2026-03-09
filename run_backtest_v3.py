"""
JARVIS V3 - Definitive Backtest: V1 vs V2 vs V3 vs SPY
=========================================================
Tests all three versions side by side.

V3 fixes:
1. Simple trend+VIX regime detection (replaces unreliable HMM)
2. Momentum & Trend get 60% weight (proven alpha generators)
3. 30% minimum equity floor (never miss the recovery)
4. 70% satellite allocation in CALM (max alpha capture)
5. Golden Cross confirmation (SMA50 > SMA200 → extra equity boost)

HOW TO RUN:
    python run_backtest_v3.py
"""

import sys, os
import pandas as pd
import numpy as np
from loguru import logger

os.makedirs("logs", exist_ok=True)
logger.add("logs/backtest_v3.log", rotation="10 MB", level="INFO")


def main():
    print()
    print("=" * 75)
    print("  J A R V I S   —   D E F I N I T I V E   B A C K T E S T")
    print("  V1 (Conservative) vs V2 (Regime-HMM) vs V3 (Production) vs SPY")
    print("=" * 75)
    print()

    from data.db import get_all_prices
    from config.settings import validate_config

    missing = validate_config()
    if missing:
        print(f"  ❌ Missing: {missing}")
        sys.exit(1)

    prices = get_all_prices()
    if prices.empty or len(prices) < 500:
        print("  ❌ Insufficient data.")
        sys.exit(1)

    print(f"  Data: {len(prices.columns)} ETFs, {len(prices)} days")
    start_date = (prices.index[0] + pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    print(f"  Backtest: {start_date} → {prices.index[-1].date()}\n")

    common_config = {
        "start_date": start_date,
        "initial_capital": 100000,
        "rebalance_frequency": 21,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
    }

    # ── V1 ──
    print(f"{'─'*75}")
    print("  Running V1 (Conservative — equal weights, static allocation)...")
    print(f"{'─'*75}")
    from backtest.engine import Backtest as BT1
    r1 = BT1(common_config).run(prices)
    m1 = r1.get("metrics", {})
    print(f"  → V1 done: ${m1.get('end_value',0):,.0f}\n")

    # ── V2 ──
    print(f"{'─'*75}")
    print("  Running V2 (HMM Regime — dynamic allocation)...")
    print(f"{'─'*75}")
    from backtest.engine_v2 import BacktestV2 as BT2
    r2 = BT2(common_config).run(prices)
    m2 = r2.get("metrics", {})
    print(f"  → V2 done: ${m2.get('end_value',0):,.0f}\n")

    # ── V3 ──
    print(f"{'─'*75}")
    print("  Running V3 (Simple Regime + Momentum-heavy + Equity Floor)...")
    print(f"{'─'*75}")
    from backtest.engine_v3 import BacktestV3 as BT3
    r3 = BT3(common_config).run(prices)
    m3 = r3.get("metrics", {})
    print(f"  → V3 done: ${m3.get('end_value',0):,.0f}\n")

    # ── SPY ──
    spy = compute_spy(prices, start_date)

    # ════════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ════════════════════════════════════════════════════════════
    print()
    print("=" * 75)
    print("  D E F I N I T I V E   R E S U L T S")
    print("=" * 75)
    print()

    v1e = m1.get("end_value", 0)
    v2e = m2.get("end_value", 0)
    v3e = m3.get("end_value", 0)
    spy_e = 100000 * (1 + spy.get("total_return", 0))

    print(f"  {'':30s} {'V1':>10s} {'V2':>10s} {'V3':>10s} {'SPY':>10s}")
    print(f"  {'':30s} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Ending Value':30s} ${v1e:>8,.0f} ${v2e:>8,.0f} ${v3e:>8,.0f} ${spy_e:>8,.0f}")
    print(f"  {'Profit/Loss':30s} ${v1e-100000:>+8,.0f} ${v2e-100000:>+8,.0f} ${v3e-100000:>+8,.0f} ${spy_e-100000:>+8,.0f}")
    print()

    rows = [
        ("Total Return", "total_return", "%"),
        ("Annualized Return", "annualized_return", "%"),
        ("Annualized Volatility", "annualized_volatility", "%"),
        ("Sharpe Ratio", "sharpe_ratio", "x"),
        ("Sortino Ratio", "sortino_ratio", "x"),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Calmar Ratio", "calmar_ratio", "x"),
        ("Win Rate (Monthly)", "win_rate", "%"),
        ("Profit Factor", "profit_factor", "x"),
        ("Best Month", "best_month", "%"),
        ("Worst Month", "worst_month", "%"),
    ]

    print(f"  {'Metric':30s} {'V1':>10s} {'V2':>10s} {'V3':>10s} {'SPY':>10s} {'V3-SPY':>10s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for name, key, fmt in rows:
        vals = [m1.get(key, 0) or 0, m2.get(key, 0) or 0, m3.get(key, 0) or 0, spy.get(key, 0) or 0]
        diff = vals[2] - vals[3]
        if fmt == "%":
            parts = [f"{v:>+9.1%}" for v in vals]
            d = f"{diff:>+9.1%}"
        else:
            parts = [f"{v:>9.2f}" for v in vals]
            d = f"{diff:>+9.2f}"
        print(f"  {name:30s} {parts[0]} {parts[1]} {parts[2]} {parts[3]} {d}")

    # Trades & costs
    print(f"\n  {'Total Trades':30s} {m1.get('total_trades',0):>10,d} {m2.get('total_trades',0):>10,d} {m3.get('total_trades',0):>10,d} {'—':>10s}")
    print(f"  {'Total Costs':30s} ${m1.get('total_costs',0):>9,.0f} ${m2.get('total_costs',0):>9,.0f} ${m3.get('total_costs',0):>9,.0f} {'—':>10s}")

    # V3 regime breakdown
    regime_log = r3.get("regime_log", [])
    if regime_log:
        n = len(regime_log)
        c = sum(1 for r in regime_log if r["regime"] == "CALM")
        t = sum(1 for r in regime_log if r["regime"] == "TRANSITION")
        cr = sum(1 for r in regime_log if r["regime"] == "CRISIS")
        print(f"\n  V3 Regime Distribution:")
        print(f"    CALM:       {c:3d}/{n} ({c/n:.0%}) — 70% satellite, Half-Kelly, equity boost")
        print(f"    TRANSITION: {t:3d}/{n} ({t/n:.0%}) — 35% satellite, Third-Kelly")
        print(f"    CRISIS:     {cr:3d}/{n} ({cr/n:.0%}) — 10% satellite, Eighth-Kelly")

    # V3 vs V1 improvement
    v3_ret = m3.get("annualized_return", 0)
    v1_ret = m1.get("annualized_return", 0)
    v3_sharpe = m3.get("sharpe_ratio", 0)
    v1_sharpe = m1.get("sharpe_ratio", 0)
    spy_ret = spy.get("annualized_return", 0)
    alpha = v3_ret - spy_ret
    v3_dd = m3.get("max_drawdown", 0)
    spy_dd = spy.get("max_drawdown", 0)

    print(f"\n  {'═'*75}")
    print(f"  V3 IMPROVEMENT OVER V1:")
    print(f"    Return:  {v1_ret:+.1%} → {v3_ret:+.1%} ({v3_ret-v1_ret:+.1%})")
    print(f"    Sharpe:  {v1_sharpe:.2f} → {v3_sharpe:.2f} ({v3_sharpe-v1_sharpe:+.2f})")
    print(f"\n  V3 ALPHA vs SPY: {alpha:+.2%} annualized")

    if alpha > 0:
        print(f"    ✅ V3 OUTPERFORMS SPY by {alpha:.1%}/year")
    elif alpha > -0.03:
        print(f"    ⚠️ V3 is within 3% of SPY ({alpha:+.1%}/year) with {v3_dd:.1%} drawdown vs {spy_dd:.1%}")
    else:
        print(f"    V3 underperforms SPY by {abs(alpha):.1%}/year")

    if abs(v3_dd) < abs(spy_dd):
        dd_saved = abs(spy_dd) - abs(v3_dd)
        print(f"    ✅ V3 drawdown protection: {v3_dd:.1%} vs SPY {spy_dd:.1%} (saved {dd_saved:.1%})")

    # Risk-adjusted verdict
    print(f"\n  {'═'*75}")
    if v3_sharpe >= 0.5:
        print(f"  ✅ VERDICT: V3 is READY for paper trading validation")
        print(f"     Sharpe {v3_sharpe:.2f} exceeds 0.5 threshold")
    elif v3_sharpe >= 0.2 and alpha > -0.05:
        print(f"  ⚠️ VERDICT: V3 is PROMISING — proceed with paper trading")
        print(f"     Sharpe {v3_sharpe:.2f} is positive with {v3_dd:.1%} max drawdown")
        if abs(v3_dd) < abs(spy_dd) * 0.7:
            print(f"     Risk management is significantly better than SPY")
    elif v3_sharpe >= 0:
        print(f"  ⚠️ VERDICT: V3 is MARGINAL — paper trade for 6 months")
    else:
        print(f"  ❌ VERDICT: V3 needs more refinement")
    print(f"  {'═'*75}")
    print()


def compute_spy(prices, start_date):
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
    aw = mo[mo > 0].mean() if (mo > 0).any() else 0
    al = mo[mo < 0].mean() if (mo < 0).any() else 0
    pf = abs(aw / al) if al != 0 else float("inf")
    return {
        "total_return": tr, "annualized_return": ar,
        "annualized_volatility": vol, "sharpe_ratio": sh,
        "sortino_ratio": so, "max_drawdown": mdd,
        "calmar_ratio": cal, "win_rate": wr, "profit_factor": pf,
        "best_month": mo.max() if not mo.empty else 0,
        "worst_month": mo.min() if not mo.empty else 0,
    }


if __name__ == "__main__":
    main()
