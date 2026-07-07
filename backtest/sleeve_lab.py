"""
JARVIS V3 - SLEEVE LAB: evidence for (or against) the multi-asset engine
=========================================================================
Runs Engine 2 (signals/tsmom.py) through the same adversarial treatment as
everything else: walk-forward with per-side costs, cost sensitivity, per-year
stability — then the number that actually matters: the TWO-ENGINE COMBINATION.
It re-runs the production equity walk-forward (Engine 1, current deployed
config) to get its daily return series, aligns it with the sleeve's series,
reports their correlation, and shows combined Sharpe/CAGR/MaxDD at several
allocations. The whole thesis of Engine 2 is that low correlation lifts the
combination above either part; if the data disagrees, the sleeve does not
ship. Pre-committed shipping rule: sleeve ships only if (a) standalone
Sharpe at 5 bps >= 0.3, (b) |correlation with Engine 1| <= 0.4, and
(c) the best combination beats Engine 1 alone on Sharpe WITHOUT a deeper
max drawdown. Run INSIDE the Railway container:
    python backtest/sleeve_lab.py
Self-test: python backtest/sleeve_lab.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from signals.tsmom import (SLEEVE_UNIVERSE, tsmom_scores, build_sleeve_targets,
                           TRADING_DAYS)
from backtest.walkforward import _metrics, _sharpe   # same yardstick, same code


def run_sleeve_walkforward(prices: pd.DataFrame, cost_bps: float = 5.0,
                           rebalance_days: int = 10,
                           warmup: int = 260) -> dict:
    """Day-by-day sleeve simulation, structure mirroring the production
    walk-forward: decide at close t, earn t -> t+1, costs on traded notional."""
    rets = prices.pct_change()
    vol = rets.rolling(63).std() * np.sqrt(TRADING_DAYS)
    scores = tsmom_scores(prices)

    port: dict[str, float] = {}
    dates, series, grosses = [], [], []
    traded_total, n_rebals = 0.0, 0
    n = len(prices)
    if n <= warmup + 2:
        raise ValueError("not enough history")

    for i in range(warmup, n - 1):
        cost_today = 0.0
        if (i - warmup) % rebalance_days == 0:
            new = build_sleeve_targets(scores.iloc[i], vol.iloc[i])
            traded = sum(abs(new.get(t, 0.0) - port.get(t, 0.0))
                         for t in set(new) | set(port))
            traded_total += traded
            cost_today = traded * cost_bps / 1e4
            port = new
            n_rebals += 1
        r_next = rets.iloc[i + 1]
        gross_r = float(sum(w * (r_next.get(t, 0.0)
                                 if np.isfinite(r_next.get(t, np.nan)) else 0.0)
                            for t, w in port.items()))
        net_r = gross_r - cost_today
        if port and (1.0 + gross_r) != 0:
            port = {t: w * (1.0 + (r_next.get(t, 0.0)
                                   if np.isfinite(r_next.get(t, np.nan)) else 0.0))
                       / (1.0 + gross_r) for t, w in port.items()}
        dates.append(prices.index[i + 1])
        series.append(net_r)
        grosses.append(sum(abs(w) for w in port.values()))

    s = pd.Series(series, index=pd.DatetimeIndex(dates), name="sleeve")
    bench = rets["SPY"].reindex(s.index).fillna(0.0) if "SPY" in rets else s * 0
    years = len(s) / TRADING_DAYS
    out = _metrics(s, bench)
    out.update({
        "n_days": len(s), "n_rebalances": n_rebals,
        "annual_turnover_x": round(traded_total / max(years, 1e-9) / 2.0, 2),
        "avg_gross": round(float(np.mean(grosses)), 3),
        "cost_bps_per_side": cost_bps,
        "daily_returns": s,
    })
    return out


def combination_table(eq: pd.Series, sleeve: pd.Series,
                      allocations=(0.7, 0.6, 0.5)) -> tuple[pd.DataFrame, float]:
    """Blend the two engines' daily series at fixed allocations (rebalanced
    continuously — a mild idealization, flagged) and report the frontier."""
    idx = eq.index.intersection(sleeve.index)
    e, s = eq.reindex(idx), sleeve.reindex(idx)
    corr = round(float(e.corr(s)), 2)
    rows = []
    def _row(label, r):
        eq_curve = (1 + r).cumprod()
        years = len(r) / TRADING_DAYS
        rows.append({
            "portfolio": label,
            "cagr": round(float(eq_curve.iloc[-1] ** (1 / max(years, 1e-9)) - 1), 4),
            "sharpe": _sharpe(r),
            "max_dd": round(float((eq_curve / eq_curve.cummax() - 1).min()), 4),
        })
    _row("Engine 1 only (equity book)", e)
    _row("Engine 2 only (TS-mom sleeve)", s)
    for a in allocations:
        _row(f"{int(a*100)}/{int((1-a)*100)} blend", a * e + (1 - a) * s)
    return pd.DataFrame(rows), corr


def _report(sleeve_res, cost_rows, per_year, combo_df, corr, meta) -> str:
    lines = [
        "=" * 70,
        f"SLEEVE LAB REPORT — {meta}",
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        "=" * 70, "",
        "ENGINE 2 STANDALONE (net of 5 bps/side):",
        f"  CAGR {sleeve_res['cagr']:+.2%}  vol {sleeve_res['ann_vol']:.1%}  "
        f"Sharpe {sleeve_res['sharpe']}  MaxDD {sleeve_res['max_drawdown']:.1%}",
        f"  avg gross {sleeve_res['avg_gross']:.0%}  "
        f"turnover {sleeve_res['annual_turnover_x']}x/yr  "
        f"({sleeve_res['n_rebalances']} rebalances)",
        "",
        "COST SENSITIVITY:",
        pd.DataFrame(cost_rows).to_string(index=False), "",
        "PER-YEAR:",
        per_year.to_string(index=False) if per_year is not None and not per_year.empty
        else "  (short span)", "",
    ]
    if combo_df is not None:
        lines += [
            f"TWO-ENGINE COMBINATION (daily correlation = {corr:+.2f}):",
            combo_df.to_string(index=False), "",
            "SHIPPING RULE (pre-committed): sleeve ships only if standalone",
            "Sharpe@5bps >= 0.3, |corr| <= 0.4, and a blend beats Engine 1",
            "alone on Sharpe without a deeper max drawdown. Otherwise the",
            "sleeve stays in research. One change at a time still applies:",
            "shipping waits until the current live config has its paper",
            "baseline.", ""]
    lines.append("=" * 70)
    return "\n".join(lines)


def _per_year(s: pd.Series) -> pd.DataFrame:
    rows = []
    for yr, r in s.groupby(s.index.year):
        if len(r) < 40:
            continue
        rows.append({"year": int(yr),
                     "return": round(float((1 + r).prod() - 1), 4),
                     "sharpe": _sharpe(r)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS Sleeve Lab (Engine 2)")
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-combo", action="store_true",
                    help="skip the Engine 1 re-run (faster)")
    args = ap.parse_args()

    if args.synthetic:
        rng = np.random.default_rng(5)
        n_d = 900
        dates = pd.bdate_range("2021-01-04", periods=n_d)
        cols = list(SLEEVE_UNIVERSE)
        r = rng.normal(1e-4, 0.010, (n_d, len(cols)))
        px = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=dates,
                          columns=cols)
        res = run_sleeve_walkforward(px)
        print(_report(res, [{"cost_bps": 5, "sharpe": res["sharpe"]}],
                      _per_year(res["daily_returns"]), None, 0.0,
                      "SYNTHETIC NOISE (numbers are noise, not evidence)"))
        return 0

    from data.ingest import get_prices_for_universe, download_universe_history
    tickers = list(SLEEVE_UNIVERSE)
    lookback = int(args.years * TRADING_DAYS)
    px = get_prices_for_universe(tickers, lookback)
    missing = [t for t in tickers if t not in px.columns
               or px[t].notna().sum() < 300]
    if missing:
        logger.info(f"backfilling {len(missing)} new sleeve tickers: {missing}")
        download_universe_history(missing, years=10)
        px = get_prices_for_universe(tickers, lookback)
    logger.info(f"sleeve matrix: {px.shape[1]} assets x {px.shape[0]} days")

    res5 = run_sleeve_walkforward(px, cost_bps=5.0)
    cost_rows = []
    for bps in (0.0, 5.0, 15.0):
        r_ = res5 if bps == 5.0 else run_sleeve_walkforward(px, cost_bps=bps)
        cost_rows.append({"cost_bps": bps, "cagr": r_["cagr"],
                          "sharpe": r_["sharpe"], "max_dd": r_["max_drawdown"]})

    combo_df, corr = None, 0.0
    if not args.no_combo:
        logger.info("re-running Engine 1 walk-forward for the combination...")
        from config.universe import (get_full_universe, get_sector_map,
                                     get_asset_class_map)
        from backtest.walkforward import run_walkforward
        eq_px, eq_vol = get_prices_for_universe(get_full_universe(), lookback,
                                                with_volume=True)
        eq = run_walkforward(eq_px, volume=eq_vol,
                             sector_map=get_sector_map(),
                             asset_class=get_asset_class_map(),
                             rebalance_days=10)   # the shipped live config
        combo_df, corr = combination_table(eq["daily_returns"],
                                           res5["daily_returns"])

    report = _report(res5, cost_rows, _per_year(res5["daily_returns"]),
                     combo_df, corr,
                     f"REAL DATA ({px.shape[1]} assets, {args.years}y)")
    print("\n" + report + "\n")
    with open("sleeve_lab_report.txt", "w") as f:
        f.write(report)
    logger.info("written: sleeve_lab_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
