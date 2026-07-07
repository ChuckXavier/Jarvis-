"""
JARVIS V3 - One-Command Walk-Forward Evaluation
================================================
PURPOSE: turn "run the walk-forward on the real database" into one command.
Checks data coverage, optionally backfills (--ingest), runs the production
walk-forward at 5 bps, the 0/5/15 bps cost-sensitivity table, and a
per-calendar-year breakdown, then writes everything to walkforward_report.txt
and a `backtest_results` row in PostgreSQL. The numbers this prints are the
evidence the leverage gate and every Tier decision hang on. Nothing here
re-implements strategy logic — it calls the same production functions the
live pipeline uses.

USAGE (on Railway, where DATABASE_URL lives):
    railway run python backtest/run_full_evaluation.py --ingest   # first time
    railway run python backtest/run_full_evaluation.py            # thereafter
Self-test without a database (machinery check only, numbers meaningless):
    python backtest/run_full_evaluation.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

TRADING_DAYS = 252


# ── data coverage ────────────────────────────────────────────────────────────

def check_coverage(min_tickers: int = 400, min_years: float = 6.0) -> dict:
    """How much history is actually in daily_prices?"""
    from sqlalchemy import text
    from data.db import engine
    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT COUNT(DISTINCT ticker), MIN(date), MAX(date), COUNT(*) "
            "FROM daily_prices")).fetchone()
    n_tickers = int(row[0] or 0)
    dmin, dmax, n_rows = row[1], row[2], int(row[3] or 0)
    years = ((pd.Timestamp(dmax) - pd.Timestamp(dmin)).days / 365.25
             if dmin and dmax else 0.0)
    ok = n_tickers >= min_tickers and years >= min_years
    logger.info(f"coverage: {n_tickers} tickers, {years:.1f}y "
                f"({dmin} -> {dmax}), {n_rows:,} rows -> "
                f"{'SUFFICIENT' if ok else 'INSUFFICIENT'}")
    return {"ok": ok, "tickers": n_tickers, "years": round(years, 1),
            "rows": n_rows, "min": str(dmin), "max": str(dmax)}


# ── report helpers ───────────────────────────────────────────────────────────

def yearly_table(daily: pd.Series) -> pd.DataFrame:
    """Per-calendar-year return / vol / Sharpe — the stability check."""
    rows = []
    for yr, r in daily.groupby(daily.index.year):
        if len(r) < 40:
            continue
        vol = r.std() * np.sqrt(TRADING_DAYS)
        rows.append({
            "year": int(yr),
            "days": len(r),
            "return": round(float((1 + r).prod() - 1), 4),
            "vol": round(float(vol), 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)), 2)
                      if r.std() > 0 else None,
        })
    return pd.DataFrame(rows)


def build_report(res: dict, cost_df: pd.DataFrame, coverage: dict | None,
                 title: str) -> str:
    yt = yearly_table(res["daily_returns"])
    lines = [
        "=" * 64,
        f"JARVIS V3 WALK-FORWARD EVALUATION — {title}",
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        "=" * 64,
    ]
    if coverage:
        lines.append(f"data: {coverage['tickers']} tickers, "
                     f"{coverage['years']}y ({coverage['min']} -> {coverage['max']})")
    lines += [
        "",
        f"HEADLINE (net of {res['cost_bps_per_side']:.0f} bps/side):",
        f"  span {res['years']}y ({res['n_days']} days), "
        f"{res['n_rebalances']} rebalances",
        f"  CAGR {res['cagr']:+.2%}   vol {res['ann_vol']:.1%}   "
        f"Sharpe {res['sharpe']}   Sortino {res['sortino']}",
        f"  MaxDD {res['max_drawdown']:.1%}   hit-rate {res['hit_rate']:.0%}   "
        f"SPY CAGR {res['benchmark_cagr']:+.2%}",
        f"  avg gross {res['avg_gross']:.0%}   avg net {res['avg_net']:+.0%}   "
        f"turnover {res['annual_turnover_x']}x/yr",
        "",
        f"REGIME OCCUPANCY: {res['regime_share']}",
        f"PER-REGIME SHARPE: {res['per_regime_sharpe']}",
        "",
        "COST SENSITIVITY (the honesty table):",
        cost_df.to_string(index=False),
        "",
        "PER-YEAR STABILITY:",
        yt.to_string(index=False) if not yt.empty else "  (span too short)",
        "",
        "READING GUIDE — pre-committed, decided before seeing the numbers:",
        "  Sharpe(5bps) < 0.3 : signals not carrying; regime ladder is doing",
        "                       the work. Stay unlevered; build fundamentals",
        "                       feed (Tier 1) before anything else.",
        "  0.3 - 0.7          : real but modest. Tier 1 upgrades first;",
        "                       leverage discussion stays closed.",
        "  >= 0.7 AND survives 15bps AND yearly Sharpe sign-stable:",
        "                       Tier 1 build + revisit ALLOW_LEVERAGE only",
        "                       after 3 months of paper tracking this curve.",
        "  Any result: if Sharpe collapses 5bps -> 15bps, the edge is",
        "  execution-fragile and must not be levered.",
        "=" * 64,
    ]
    return "\n".join(lines)


def _pynum(x):
    """Coerce numpy scalars to native Python numbers (psycopg2 can't adapt
    np.float64 — it errors with 'schema np does not exist')."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return None if np.isnan(v) else v
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def persist_result(res: dict, cost_df: pd.DataFrame) -> None:
    """Store the run in backtest_results so the dashboard can show it."""
    try:
        from sqlalchemy import text
        from data.db import engine
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id SERIAL PRIMARY KEY, run_at TIMESTAMP,
                    years REAL, cost_bps REAL, cagr REAL, sharpe REAL,
                    sortino REAL, max_drawdown REAL, ann_vol REAL,
                    turnover_x REAL, avg_gross REAL, avg_net REAL,
                    detail TEXT)
            """))
            conn.execute(text("""
                INSERT INTO backtest_results
                    (run_at, years, cost_bps, cagr, sharpe, sortino,
                     max_drawdown, ann_vol, turnover_x, avg_gross, avg_net,
                     detail)
                VALUES (:ts,:yr,:bps,:cagr,:sh,:so,:dd,:vol,:to,:g,:n,:d)
            """), {
                "ts": datetime.now(timezone.utc), "yr": _pynum(res["years"]),
                "bps": _pynum(res["cost_bps_per_side"]),
                "cagr": _pynum(res["cagr"]),
                "sh": _pynum(res["sharpe"]), "so": _pynum(res["sortino"]),
                "dd": _pynum(res["max_drawdown"]),
                "vol": _pynum(res["ann_vol"]),
                "to": _pynum(res["annual_turnover_x"]),
                "g": _pynum(res["avg_gross"]), "n": _pynum(res["avg_net"]),
                "d": json.dumps({
                    "regime_share": {k: _pynum(v) for k, v in
                                     res["regime_share"].items()},
                    "per_regime_sharpe": {k: _pynum(v) for k, v in
                                          res["per_regime_sharpe"].items()},
                    "cost_table": [{k: _pynum(v) for k, v in row.items()}
                                   for row in cost_df.to_dict("records")],
                })[:2000],
            })
        logger.info("result persisted to backtest_results")
    except Exception as e:
        logger.warning(f"persist failed (non-fatal): {e}")


# ── main paths ───────────────────────────────────────────────────────────────

def run_real(years: int, do_ingest: bool) -> int:
    from backtest.walkforward import run_walkforward, cost_sensitivity
    from config.universe import (get_full_universe, get_sector_map,
                                 get_asset_class_map, refresh_universe)
    from data.ingest import get_prices_for_universe, run_full_ingestion

    cov = check_coverage()
    if not cov["ok"]:
        if do_ingest:
            logger.info("coverage insufficient — running full ingestion "
                        "(universe refresh + ~10y backfill; this takes a while)")
            refresh_universe(force=True)
            run_full_ingestion()
            cov = check_coverage()
            if not cov["ok"]:
                logger.error("coverage still insufficient after ingest — "
                             "inspect data.ingest logs for failed tickers")
                return 2
        else:
            logger.error("coverage insufficient. Re-run with --ingest to "
                         "backfill the full universe first.")
            return 2

    lookback = int(years * TRADING_DAYS)
    prices, volume = get_prices_for_universe(get_full_universe(), lookback,
                                             with_volume=True)
    logger.info(f"matrix: {prices.shape[1]} tickers x {prices.shape[0]} days")
    sector_map = get_sector_map()
    asset_class = get_asset_class_map()

    res = run_walkforward(prices, volume=volume, sector_map=sector_map,
                          asset_class=asset_class, cost_bps=5.0)
    cost_df = cost_sensitivity(prices, volume=volume, sector_map=sector_map,
                               asset_class=asset_class)
    report = build_report(res, cost_df, cov,
                          f"REAL DATA ({prices.shape[1]} tickers)")
    print("\n" + report + "\n")
    with open("walkforward_report.txt", "w") as f:
        f.write(report)
    logger.info("written: walkforward_report.txt")
    persist_result(res, cost_df)
    return 0


def run_synthetic() -> int:
    """Machinery self-test. Numbers are NOISE and must never be cited."""
    from backtest.walkforward import run_walkforward, cost_sensitivity
    rng = np.random.default_rng(7)
    n_d, n_t = 700, 120
    dates = pd.bdate_range("2022-01-03", periods=n_d)
    names = [f"T{i:03d}" for i in range(n_t)]
    mkt = rng.standard_normal(n_d) * 0.009
    mkt[int(n_d * .55):int(n_d * .62)] -= 0.018
    r = (rng.normal(3e-4, 2e-4, n_t)
         + rng.uniform(.01, .03, n_t)
         * (0.6 * rng.standard_normal((n_d, n_t)) + 0.4 * mkt[:, None]))
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=dates, columns=names)
    px["SPY"] = 100 * np.exp(np.cumsum(mkt * 1.1 + 2e-4))
    sectors = {t: f"S{int(t[1:]) % 8}" for t in names}
    res = run_walkforward(px, sector_map=sectors, cost_bps=5.0, warmup=300)
    cost_df = cost_sensitivity(px, sector_map=sectors, warmup=300)
    print("\n" + build_report(res, cost_df, None,
                              "SYNTHETIC SELF-TEST (noise — not evidence)") + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS V3 walk-forward evaluation")
    ap.add_argument("--years", type=int, default=8,
                    help="history span to evaluate (default 8)")
    ap.add_argument("--ingest", action="store_true",
                    help="backfill the full universe first if coverage is thin")
    ap.add_argument("--synthetic", action="store_true",
                    help="no-database machinery self-test")
    args = ap.parse_args()
    if args.synthetic:
        return run_synthetic()
    return run_real(args.years, args.ingest)


if __name__ == "__main__":
    sys.exit(main())
