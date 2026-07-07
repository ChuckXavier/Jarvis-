"""
JARVIS V3 - Walk-Forward Backtest Harness
==========================================
ARCHITECTURAL NOTE (why this file exists):
V2's optimizer carried an embedded backtest whose signal and portfolio code
had drifted from the live path — its 15.7% CAGR / 0.61 Sharpe self-report
described a system that was not the one trading. This harness imports the
PRODUCTION functions directly: signals from signals.ensemble, portfolio
construction from portfolio.optimizer.build_targets, and the regime decision
from risk.regime.decide_regime, walked forward day by day with per-side
transaction costs. Research and live cannot diverge because they are the
same code. Honest limitation, stated not hidden: the credit vote (HY OAS)
has no point-in-time history in the price matrix, so it is marked
unavailable here and the regime machine runs on trend + realized-vol proxy —
two live signals, which satisfies MIN_SIGNALS_TO_SWITCH. This is the gate
for ALLOW_LEVERAGE: no walk-forward evidence at realistic costs, no leverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import (
    BACKTEST_COST_BPS, BACKTEST_REBALANCE_DAYS,
    IC_LOOKBACK_PRIMARY, IC_LOOKBACK_SECONDARY, IC_BLEND,
    INVERSE_VOL_LOOKBACK, MIN_PRICE,
)
from signals.ensemble import (
    SIGNAL_NAMES, DEFAULT_WEIGHTS,
    compute_signal_matrices, combine_scores, compute_signal_ics, adapt_weights,
)
from portfolio.optimizer import build_targets
from risk.regime import decide_regime, get_target_exposure

TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
# REGIME VOTES FROM PRICES ONLY (backtest proxies, documented)
# ══════════════════════════════════════════════════════════════════════════════

def _proxy_votes(spy_px: float, spy_sma200: float, spy_rvol21: float) -> dict:
    """
    trend  — REAL: SPY vs 200-SMA with the same ±2% buffer as live.
    vol    — PROXY: SPY 21d realized ann. vol (<15% risk-on, >25% risk-off)
             standing in for the live VIX-level vote.
    credit — UNAVAILABLE in backtest (no point-in-time OAS in the matrix).
    """
    if np.isfinite(spy_px) and np.isfinite(spy_sma200) and spy_sma200 > 0:
        if spy_px > spy_sma200 * 1.02:
            tv = 1
        elif spy_px < spy_sma200 * 0.98:
            tv = -1
        else:
            tv = 0
        trend = {"vote": tv, "available": True, "value": spy_px / spy_sma200,
                 "detail": "SPY vs 200-SMA (±2%)"}
    else:
        trend = {"vote": 0, "available": False, "value": None,
                 "detail": "insufficient history"}

    if np.isfinite(spy_rvol21):
        if spy_rvol21 < 0.15:
            vv = 1
        elif spy_rvol21 > 0.25:
            vv = -1
        else:
            vv = 0
        vol = {"vote": vv, "available": True, "value": spy_rvol21,
               "detail": "SPY 21d realized vol proxy"}
    else:
        vol = {"vote": 0, "available": False, "value": None,
               "detail": "insufficient history"}

    credit = {"vote": 0, "available": False, "value": None,
              "detail": "OAS not available in backtest"}
    return {"trend": trend, "vol": vol, "credit": credit}


# ══════════════════════════════════════════════════════════════════════════════
# THE WALK
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward(prices: pd.DataFrame,
                    volume: pd.DataFrame | None = None,
                    sector_map: dict | None = None,
                    asset_class: dict | None = None,
                    cost_bps: float = BACKTEST_COST_BPS,
                    rebalance_days: int = BACKTEST_REBALANCE_DAYS,
                    adapt: bool = True,
                    warmup: int = 300,
                    benchmark: str = "SPY") -> dict:
    """
    Day-by-day simulation. At each rebalance date t the book is built from
    information <= t and earns returns from t -> t+1 onward; costs are charged
    on traded notional at cost_bps per side. Returns a metrics dict plus the
    daily series for inspection.
    """
    if prices is None or prices.empty or benchmark not in prices.columns:
        raise ValueError("prices must be non-empty and include the benchmark")
    sector_map = sector_map or {}
    asset_class = asset_class or {}
    gated = {"leveraged", "inverse", "volatility"}
    no_short = {t for t, c in asset_class.items()
                if c in ("fixed_income", "alternative")}

    rets = prices.pct_change()
    vol_sizing = rets.rolling(INVERSE_VOL_LOOKBACK).std() * np.sqrt(TRADING_DAYS)
    spy = prices[benchmark]
    spy_sma200 = spy.rolling(200).mean()
    spy_rvol21 = spy.pct_change().rolling(21).std() * np.sqrt(TRADING_DAYS)

    signals = compute_signal_matrices(prices)   # point-in-time by construction
    sig_w = DEFAULT_WEIGHTS.copy()
    machine = {"regime": "CAUTIOUS", "bullish_count": 0, "bearish_count": 0}

    excluded = {t for t in prices.columns if asset_class.get(t, "equity") in gated}

    if volume is not None and not volume.empty:
        dollar_med21 = (prices * volume).rolling(21).median()
    else:
        dollar_med21 = None

    port_w: dict[str, float] = {}
    dates, net_rets, regimes, grosses, nets_ = [], [], [], [], []
    total_traded = 0.0
    n_rebals = 0

    n = len(prices)
    if n <= warmup + 2:
        raise ValueError(f"need > {warmup + 2} rows, got {n}")

    for i in range(warmup, n - 1):
        # 1) Regime machine — daily, exactly the production decision function.
        votes = _proxy_votes(float(spy.iloc[i]), float(spy_sma200.iloc[i]),
                             float(spy_rvol21.iloc[i]))
        decision = decide_regime(votes, machine)
        machine = {"regime": decision["regime"],
                   "bullish_count": decision["bullish_count"],
                   "bearish_count": decision["bearish_count"]}
        regime = decision["regime"]

        # 2) Rebalance every K days using information at close of day i.
        cost_today = 0.0
        if (i - warmup) % rebalance_days == 0:
            if adapt:
                ic21 = compute_signal_ics(signals, prices,
                                          IC_LOOKBACK_PRIMARY, asof_idx=i)
                ic63 = compute_signal_ics(signals, prices,
                                          IC_LOOKBACK_SECONDARY, asof_idx=i)
                comp = {s: IC_BLEND * ic21[s] + (1 - IC_BLEND) * ic63[s]
                        for s in SIGNAL_NAMES}
                sig_w = adapt_weights(sig_w, comp)

            sl = {name: df.iloc[[i]] for name, df in signals.items()}
            row = combine_scores(sl, sig_w)
            scores = row.iloc[0].dropna() if not row.empty else pd.Series(dtype=float)

            px_i = prices.iloc[i]
            scores = scores[px_i.reindex(scores.index) >= MIN_PRICE]
            scores = scores[~scores.index.isin(excluded)]
            if dollar_med21 is not None:
                liq = dollar_med21.iloc[i].reindex(scores.index)
                scores = scores[liq.fillna(0) >= 1e7]

            exp = get_target_exposure(regime)
            new_w = build_targets(scores, vol_sizing.iloc[i], sector_map,
                                  exp["gross"], exp["net"], no_short=no_short)

            traded = sum(abs(new_w.get(t, 0.0) - port_w.get(t, 0.0))
                         for t in set(new_w) | set(port_w))
            total_traded += traded
            cost_today = traded * cost_bps / 1e4
            port_w = new_w
            n_rebals += 1

        # 3) Earn day i -> i+1 returns on the signed book.
        r_next = rets.iloc[i + 1]
        gross_r = float(sum(w * (r_next.get(t, 0.0) if np.isfinite(r_next.get(t, np.nan)) else 0.0)
                            for t, w in port_w.items()))
        net_r = gross_r - cost_today

        # 4) Drift weights with their own returns.
        if port_w and (1.0 + gross_r) != 0:
            port_w = {t: w * (1.0 + (r_next.get(t, 0.0)
                                     if np.isfinite(r_next.get(t, np.nan)) else 0.0))
                         / (1.0 + gross_r)
                      for t, w in port_w.items()}

        dates.append(prices.index[i + 1])
        net_rets.append(net_r)
        regimes.append(regime)
        grosses.append(sum(abs(w) for w in port_w.values()))
        nets_.append(sum(port_w.values()))

    series = pd.Series(net_rets, index=pd.DatetimeIndex(dates), name="ret")
    reg_s = pd.Series(regimes, index=series.index)
    bench = rets[benchmark].reindex(series.index).fillna(0.0)
    years = len(series) / TRADING_DAYS

    out = _metrics(series, bench)
    out.update({
        "n_days": len(series),
        "n_rebalances": n_rebals,
        "annual_turnover_x": round(total_traded / max(years, 1e-9) / 2.0, 2),
        "avg_gross": round(float(np.mean(grosses)), 3),
        "avg_net": round(float(np.mean(nets_)), 3),
        "cost_bps_per_side": cost_bps,
        "regime_share": reg_s.value_counts(normalize=True).round(3).to_dict(),
        "per_regime_sharpe": {
            r: _sharpe(series[reg_s == r]) for r in sorted(set(regimes))
        },
        "daily_returns": series,
        "regime_series": reg_s,
    })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(r: pd.Series) -> float:
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    return round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)), 2)


def _metrics(r: pd.Series, bench: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (r.mean() * TRADING_DAYS / downside) if downside and downside > 0 else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    beq = (1 + bench).cumprod()
    bcagr = beq.iloc[-1] ** (1 / max(years, 1e-9)) - 1
    return {
        "years": round(years, 2),
        "cagr": round(float(cagr), 4),
        "ann_vol": round(float(vol), 4),
        "sharpe": _sharpe(r),
        "sortino": round(float(sortino), 2) if np.isfinite(sortino) else None,
        "max_drawdown": round(float(dd), 4),
        "benchmark_cagr": round(float(bcagr), 4),
        "hit_rate": round(float((r > 0).mean()), 3),
    }


def cost_sensitivity(prices, **kwargs) -> pd.DataFrame:
    """The honesty table: same walk at 0 / 5 / 15 bps per side."""
    rows = []
    for bps in (0.0, 5.0, 15.0):
        res = run_walkforward(prices, cost_bps=bps, **kwargs)
        rows.append({"cost_bps": bps, "cagr": res["cagr"],
                     "sharpe": res["sharpe"], "max_dd": res["max_drawdown"],
                     "turnover_x": res["annual_turnover_x"]})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def _print_report(res: dict, title: str):
    logger.info("=" * 58)
    logger.info(f"WALK-FORWARD — {title}")
    logger.info(f"  span {res['years']}y ({res['n_days']} days), "
                f"{res['n_rebalances']} rebalances")
    logger.info(f"  CAGR {res['cagr']:+.1%}  vol {res['ann_vol']:.1%}  "
                f"Sharpe {res['sharpe']}  Sortino {res['sortino']}")
    logger.info(f"  MaxDD {res['max_drawdown']:.1%}  hit {res['hit_rate']:.0%}  "
                f"vs SPY CAGR {res['benchmark_cagr']:+.1%}")
    logger.info(f"  gross {res['avg_gross']:.0%} avg, net {res['avg_net']:+.0%} avg, "
                f"turnover {res['annual_turnover_x']}x/yr "
                f"@ {res['cost_bps_per_side']}bps/side")
    logger.info(f"  regime share: {res['regime_share']}")
    logger.info(f"  per-regime Sharpe: {res['per_regime_sharpe']}")
    logger.info("=" * 58)


def run_from_database(years_back: int = 8) -> dict | None:
    """Production data path: full universe history from PostgreSQL."""
    try:
        from config.universe import get_full_universe, get_sector_map, get_asset_class_map
        from data.ingest import get_prices_for_universe
        lookback = int(years_back * TRADING_DAYS)
        prices, volume = get_prices_for_universe(get_full_universe(),
                                                 lookback, with_volume=True)
        if prices.empty:
            logger.error("no prices in DB — run data.ingest.run_full_ingestion first")
            return None
        res = run_walkforward(prices, volume=volume,
                              sector_map=get_sector_map(),
                              asset_class=get_asset_class_map())
        _print_report(res, f"DB universe ({prices.shape[1]} tickers)")
        logger.info("cost sensitivity:\n"
                    + cost_sensitivity(prices, volume=volume,
                                       sector_map=get_sector_map(),
                                       asset_class=get_asset_class_map()
                                       ).to_string(index=False))
        return res
    except Exception as e:
        logger.error(f"DB walk-forward failed: {e}")
        return None


def synthetic_demo(n_tickers: int = 120, n_days: int = 700, seed: int = 7) -> dict:
    """
    Harness self-test on synthetic data (random walks + a mid-sample crash to
    exercise the CRISIS path). This validates MACHINERY, not edge — synthetic
    Sharpe numbers mean nothing about live performance and are not evidence
    for enabling leverage.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    names = [f"T{i:03d}" for i in range(n_tickers)]
    drift = rng.normal(0.0003, 0.0002, n_tickers)
    volr = rng.uniform(0.010, 0.030, n_tickers)
    shocks = rng.standard_normal((n_days, n_tickers))
    market = rng.standard_normal(n_days) * 0.009
    crash = slice(int(n_days * 0.55), int(n_days * 0.62))
    market[crash] -= 0.018                     # ~7w drawdown window
    r = drift + volr * (0.6 * shocks + 0.4 * market[:, None])
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)),
                      index=dates, columns=names)
    px["SPY"] = 100 * np.exp(np.cumsum(market * 1.1 + 0.0002))
    sectors = {t: f"S{int(t[1:]) % 8}" for t in names}
    sectors["SPY"] = "index"
    res = run_walkforward(px, sector_map=sectors, warmup=280)
    _print_report(res, "synthetic self-test (machinery check only)")
    return res


if __name__ == "__main__":
    if run_from_database() is None:
        logger.info("falling back to synthetic self-test")
        synthetic_demo()
