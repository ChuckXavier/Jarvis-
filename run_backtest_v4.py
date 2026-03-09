"""
JARVIS V4 - Definitive Backtest: Beat SPY or Go Home
=======================================================
V4 is a fundamentally different strategy: concentrated momentum rotation.
Instead of diversifying across 15 positions in multiple asset classes,
V4 loads the top 7 equity momentum winners with 90% of capital.

HOW TO RUN:
    python run_backtest_v4.py
"""

import sys, os
import pandas as pd
import numpy as np
from loguru import logger

os.makedirs("logs", exist_ok=True)
logger.add("logs/backtest_v4.log", rotation="10 MB", level="INFO")


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
        "total_trades": 0, "total_costs": 0,
        "start_value": 100000, "end_value": 100000 * (1 + tr),
    }


def main():
    print()
    print("=" * 78)
    print("  J A R V I S   V 4   —   B E A T   S P Y   O R   G O   H O M E")
    print("  V1 (Conservative) vs V3 (Risk-Managed) vs V4 (Momentum) vs SPY")
    print("=" * 78)
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

    config = {
        "start_date": start_date,
        "initial_capital": 100000,
        "rebalance_frequency": 21,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
    }

    # ── V1 ──
    print(f"  Running V1 (Conservative)...")
    from backtest.engine import Backtest as BT1
    m1 = BT1(config).run(prices).get("metrics", {})
    print(f"  → ${m1.get('end_value',0):,.0f}")

    # ── V3 ──
    print(f"  Running V3 (Risk-Managed)...")
    from backtest.engine_v3 import BacktestV3 as BT3
    m3 = BT3(config).run(prices).get("metrics", {})
    print(f"  → ${m3.get('end_value',0):,.0f}")

    # ── V4 ──
    print(f"  Running V4 (Concentrated Momentum)...")
    from backtest.engine_v4 import BacktestV4 as BT4
    r4 = BT4(config).run(prices)
    m4 = r4.get("metrics", {})
    print(f"  → ${m4.get('end_value',0):,.0f}")

    # ── SPY ──
    spy = compute_spy(prices, start_date)
    print(f"  → SPY: ${spy.get('end_value',0):,.0f}")

    # ════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════
    print()
    print("=" * 78)
    print("  R E S U L T S")
    print("=" * 78)
    print()

    all_m = {"V1": m1, "V3": m3, "V4": m4, "SPY": spy}

    # Ending values
    print(f"  {'':30s}", end="")
    for name in all_m:
        print(f" {name:>12s}", end="")
    print()
    print(f"  {'':30s}", end="")
    for name in all_m:
        print(f" {'─'*12}", end="")
    print()

    for label in ["Ending Value", "Profit/Loss"]:
        print(f"  {label:30s}", end="")
        for name, m in all_m.items():
            ev = m.get("end_value", 0)
            if label == "Ending Value":
                print(f" ${ev:>10,.0f}", end="")
            else:
                print(f" ${ev-100000:>+10,.0f}", end="")
        print()

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

    # Header with V4 vs SPY column
    print(f"  {'Metric':30s}", end="")
    for name in all_m:
        print(f" {name:>12s}", end="")
    print(f" {'V4-SPY':>12s}")

    print(f"  {'─'*30}", end="")
    for _ in all_m:
        print(f" {'─'*12}", end="")
    print(f" {'─'*12}")

    for name, key, fmt in rows:
        print(f"  {name:30s}", end="")
        vals = []
        for mname, m in all_m.items():
            v = m.get(key, 0) or 0
            vals.append(v)
            if fmt == "%":
                print(f" {v:>+11.1%}", end="")
            else:
                print(f" {v:>11.2f}", end="")

        # V4 vs SPY diff
        diff = vals[2] - vals[3]  # V4 - SPY
        if fmt == "%":
            print(f" {diff:>+11.1%}", end="")
        else:
            print(f" {diff:>+11.2f}", end="")
        print()

    # Trade stats
    print()
    print(f"  {'Total Trades':30s}", end="")
    for name, m in all_m.items():
        t = m.get("total_trades", 0)
        if t:
            print(f" {int(t):>12,d}", end="")
        else:
            print(f" {'—':>12s}", end="")
    print()
    print(f"  {'Total Costs':30s}", end="")
    for name, m in all_m.items():
        c = m.get("total_costs", 0)
        if c:
            print(f" ${c:>10,.0f}", end="")
        else:
            print(f" {'—':>12s}", end="")
    print()

    # V4 regime breakdown
    regime_log = r4.get("regime_log", [])
    if regime_log:
        n = len(regime_log)
        c = sum(1 for r in regime_log if r["regime"] == "CALM")
        t = sum(1 for r in regime_log if r["regime"] == "TRANSITION")
        cr = sum(1 for r in regime_log if r["regime"] == "CRISIS")
        print(f"\n  V4 Regime: CALM {c}/{n} ({c/n:.0%}) | TRANS {t}/{n} ({t/n:.0%}) | CRISIS {cr}/{n} ({cr/n:.0%})")

    # V4 top holdings frequency
    holdings_log = r4.get("holdings_log", [])
    if holdings_log:
        ticker_counts = {}
        for h in holdings_log:
            for t in h.get("holdings", []):
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
        top_held = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  V4 Most Frequently Held ETFs:")
        for t, count in top_held:
            pct = count / len(holdings_log) * 100
            bar = "█" * int(pct / 3)
            print(f"    {t:5s} held {count:3d}/{len(holdings_log)} months ({pct:.0f}%) {bar}")

    # ═══ VERDICT ═══
    v4_ret = m4.get("annualized_return", 0)
    spy_ret = spy.get("annualized_return", 0)
    v4_sharpe = m4.get("sharpe_ratio", 0)
    spy_sharpe = spy.get("sharpe_ratio", 0)
    v4_dd = m4.get("max_drawdown", 0)
    spy_dd = spy.get("max_drawdown", 0)
    alpha = v4_ret - spy_ret

    print(f"\n  {'═'*78}")
    print(f"  V4 ALPHA vs SPY: {alpha:+.2%} annualized")

    if alpha > 0:
        print(f"\n  🏆 V4 BEATS SPY by {alpha:.1%}/year!")
        print(f"     ${100000*(1+m4.get('total_return',0)):,.0f} vs ${100000*(1+spy.get('total_return',0)):,.0f}")
        if abs(v4_dd) < abs(spy_dd):
            print(f"     AND with better risk management ({v4_dd:.1%} vs {spy_dd:.1%} drawdown)")
        if v4_sharpe > spy_sharpe:
            print(f"     AND with better Sharpe ({v4_sharpe:.2f} vs {spy_sharpe:.2f})")
        print(f"\n  ✅ VERDICT: V4 is READY for live paper trading validation")
    elif alpha > -0.03:
        print(f"\n  ⚠️ V4 is within 3% of SPY ({alpha:+.1%}/year)")
        if abs(v4_dd) < abs(spy_dd) * 0.75:
            print(f"     With significantly better risk: {v4_dd:.1%} vs {spy_dd:.1%} drawdown")
            print(f"     Risk-adjusted: BETTER than SPY")
        print(f"\n  ⚠️ VERDICT: V4 is COMPETITIVE — paper trade to confirm")
    else:
        print(f"\n  V4 underperforms SPY by {abs(alpha):.1%}/year")
        if abs(v4_dd) < abs(spy_dd) * 0.6:
            print(f"     But drawdown is dramatically better: {v4_dd:.1%} vs {spy_dd:.1%}")
        print(f"\n  ⚠️ VERDICT: Continue iterating or accept risk-managed approach")

    print(f"  {'═'*78}")
    print()


if __name__ == "__main__":
    main()
