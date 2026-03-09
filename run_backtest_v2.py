"""
JARVIS V2 - Backtest V2 Runner
=================================
Runs BOTH the old (conservative) and new (regime-aware) backtests
side by side, compared against SPY buy-and-hold.

This is the definitive test: does regime-aware allocation fix the problem?

HOW TO RUN:
    python run_backtest_v2.py
"""

import sys
import os
import pandas as pd
import numpy as np
from loguru import logger

os.makedirs("logs", exist_ok=True)
logger.add("logs/backtest_v2.log", rotation="10 MB", level="INFO")


def main():
    print()
    print("=" * 70)
    print("  J A R V I S   V 2   —   R E G I M E - A W A R E   B A C K T E S T")
    print("  Comparing: V1 (Conservative) vs V2 (Regime-Aware) vs SPY")
    print("=" * 70)
    print()

    # Load data
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
    print(f"  Range: {prices.index[0].date()} → {prices.index[-1].date()}")

    start_date = (prices.index[0] + pd.Timedelta(days=730)).strftime("%Y-%m-%d")

    # ── Run V1 (original conservative backtest) ──
    print(f"\n{'─'*70}")
    print("  Running V1 (Conservative — static allocation)...")
    print(f"{'─'*70}")

    from backtest.engine import Backtest as BacktestV1

    bt_v1 = BacktestV1({
        "start_date": start_date,
        "initial_capital": 100000,
        "rebalance_frequency": 21,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
    })
    result_v1 = bt_v1.run(prices)
    m1 = result_v1.get("metrics", {})

    # ── Run V2 (regime-aware backtest) ──
    print(f"\n{'─'*70}")
    print("  Running V2 (Regime-Aware — dynamic allocation)...")
    print(f"{'─'*70}")

    from backtest.engine_v2 import BacktestV2

    bt_v2 = BacktestV2({
        "start_date": start_date,
        "initial_capital": 100000,
        "rebalance_frequency": 21,
        "transaction_cost_bps": 5,
        "slippage_bps": 3,
    })
    result_v2 = bt_v2.run(prices)
    m2 = result_v2.get("metrics", {})

    # ── SPY Benchmark ──
    spy_metrics = compute_spy_benchmark(prices, start_date)

    # ── Display Results ──
    print()
    print("=" * 70)
    print("  H E A D - T O - H E A D   C O M P A R I S O N")
    print("=" * 70)
    print()

    print(f"  Period: {start_date} → {prices.index[-1].date()}")
    print(f"  Starting Capital: $100,000")
    print()

    # Ending values
    v1_end = m1.get("end_value", 0)
    v2_end = m2.get("end_value", 0)
    spy_total = spy_metrics.get("total_return", 0)
    spy_end = 100000 * (1 + spy_total)

    print(f"  {'':28s} {'V1 (Old)':>12s} {'V2 (New)':>12s} {'SPY B&H':>12s}")
    print(f"  {'':28s} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Ending Value':28s} ${v1_end:>10,.0f} ${v2_end:>10,.0f} ${spy_end:>10,.0f}")
    print(f"  {'Profit/Loss':28s} ${v1_end-100000:>+10,.0f} ${v2_end-100000:>+10,.0f} ${spy_end-100000:>+10,.0f}")
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
        ("Total Trades", "total_trades", "n"),
        ("Total Costs", "total_costs", "$"),
    ]

    print(f"  {'Metric':28s} {'V1 (Old)':>12s} {'V2 (New)':>12s} {'SPY B&H':>12s} {'V2 vs SPY':>12s}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

    for name, key, fmt in rows:
        v1_val = m1.get(key, 0) or 0
        v2_val = m2.get(key, 0) or 0
        spy_val = spy_metrics.get(key, 0) or 0
        diff = v2_val - spy_val

        if fmt == "%":
            print(f"  {name:28s} {v1_val:>+11.1%} {v2_val:>+11.1%} {spy_val:>+11.1%} {diff:>+11.1%}")
        elif fmt == "x":
            print(f"  {name:28s} {v1_val:>11.2f} {v2_val:>11.2f} {spy_val:>11.2f} {diff:>+11.2f}")
        elif fmt == "n":
            print(f"  {name:28s} {int(v1_val):>11,d} {int(v2_val):>11,d} {'—':>12s} {'':>12s}")
        elif fmt == "$":
            print(f"  {name:28s} ${v1_val:>10,.0f} ${v2_val:>10,.0f} {'—':>12s} {'':>12s}")

    # ── Improvement ──
    v1_sharpe = m1.get("sharpe_ratio", 0)
    v2_sharpe = m2.get("sharpe_ratio", 0)
    v1_return = m1.get("annualized_return", 0)
    v2_return = m2.get("annualized_return", 0)
    spy_return = spy_metrics.get("annualized_return", 0)
    v2_alpha = v2_return - spy_return
    v2_dd = m2.get("max_drawdown", 0)
    spy_dd = spy_metrics.get("max_drawdown", 0)

    print()
    print(f"  {'═'*70}")
    print(f"  V2 IMPROVEMENT OVER V1:")
    print(f"    Return:  {v1_return:+.1%} → {v2_return:+.1%} ({v2_return - v1_return:+.1%} improvement)")
    print(f"    Sharpe:  {v1_sharpe:.2f} → {v2_sharpe:.2f} ({v2_sharpe - v1_sharpe:+.2f} improvement)")
    print()
    print(f"  V2 ALPHA vs SPY: {v2_alpha:+.2%} annualized")
    if v2_alpha > 0:
        print(f"    → V2 OUTPERFORMS SPY by {v2_alpha:.1%}/year")
    else:
        print(f"    → V2 underperforms SPY by {abs(v2_alpha):.1%}/year")

    if abs(v2_dd) < abs(spy_dd):
        print(f"    → V2 drawdown {v2_dd:.1%} vs SPY {spy_dd:.1%} — BETTER risk management")

    # ── Regime breakdown ──
    regime_hist = result_v2.get("regime_history", [])
    if regime_hist:
        total = len(regime_hist)
        calm = sum(1 for r in regime_hist if r.get("regime") == "CALM")
        trans = sum(1 for r in regime_hist if r.get("regime") == "TRANSITION")
        crisis = sum(1 for r in regime_hist if r.get("regime") == "CRISIS")
        print(f"\n  Regime Distribution:")
        print(f"    CALM:       {calm:3d}/{total} ({calm/total:.0%}) — Half-Kelly, equity tilt")
        print(f"    TRANSITION: {trans:3d}/{total} ({trans/total:.0%}) — Quarter-Kelly, balanced")
        print(f"    CRISIS:     {crisis:3d}/{total} ({crisis/total:.0%}) — Eighth-Kelly, defensive")

    # Verdict
    print(f"\n  {'═'*70}")
    if v2_sharpe >= 0.5 and v2_alpha > -0.02:
        print(f"  ✅ VERDICT: V2 is viable for paper trading validation")
        print(f"     Sharpe of {v2_sharpe:.2f} with {v2_dd:.1%} max drawdown is acceptable")
    elif v2_sharpe >= 0 and v2_return > 0:
        print(f"  ⚠️ VERDICT: V2 is marginal — continue paper trading")
        print(f"     Positive returns but Sharpe needs improvement")
    else:
        print(f"  ❌ VERDICT: V2 needs more work before live capital")
    print(f"  {'═'*70}")
    print()


def compute_spy_benchmark(prices, start_date):
    if "SPY" not in prices.columns:
        return {}
    spy = prices["SPY"].loc[start_date:].dropna()
    if len(spy) < 2:
        return {}

    total_return = (spy.iloc[-1] / spy.iloc[0]) - 1
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    daily_returns = spy.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0
    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = (ann_return - 0.04) / downside_vol if downside_vol > 0 else 0
    cummax = spy.cummax()
    max_dd = ((spy / cummax) - 1).min()
    calmar = abs(ann_return / max_dd) if max_dd != 0 else 0
    monthly = spy.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    avg_win = monthly[monthly > 0].mean() if (monthly > 0).any() else 0
    avg_loss = monthly[monthly < 0].mean() if (monthly < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return {
        "total_return": total_return, "annualized_return": ann_return,
        "annualized_volatility": ann_vol, "sharpe_ratio": sharpe,
        "sortino_ratio": sortino, "max_drawdown": max_dd,
        "calmar_ratio": calmar, "win_rate": win_rate,
        "profit_factor": profit_factor,
        "best_month": monthly.max() if not monthly.empty else 0,
        "worst_month": monthly.min() if not monthly.empty else 0,
    }


if __name__ == "__main__":
    main()
