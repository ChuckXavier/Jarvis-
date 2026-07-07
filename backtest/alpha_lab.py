"""
JARVIS V3 - ALPHA LAB: systematic signal discovery under adversarial rules
===========================================================================
WHAT THIS IS: the honest version of "master code that finds the alpha."
It cannot create edge; it detects edge and refuses to be fooled. Sixteen
candidate signals (documented anomalies computable from your OHLCV data)
plus the five incumbent production signals are evaluated under IDENTICAL
rules: point-in-time construction, out-of-sample confirmation on a held-out
40% of history, cost-aware quintile spreads, and a promotion bar raised for
multiple testing (testing ~21 things means the best of them looks good by
luck alone; the t>=3.0 bar accounts for that). It also runs the SYSTEM
experiments last night's report demanded: rebalance cadence, drift bands,
and CRISIS posture — because the report showed much of the missing return
is in costs and the net-short crisis stance, not in signal choice.

PRE-COMMITTED PROMOTION RULE (decided before seeing any result):
  PROMOTE a candidate only if ALL hold:
    - in-sample IC t-stat >= 3.0 and out-of-sample t >= 2.5, same sign
    - quintile long/short spread Sharpe at 15 bps/side >= 0.5
    - |correlation| with the incumbent composite < 0.5 (adds NEW info)
  WATCH if it passes two of three. KILL otherwise. No exceptions, no
  narrative overrides. Promoted signals still enter the live book only
  after improving the full-system walk-forward.

Run INSIDE the Railway container (standing policy — laptop uplink is
unreliable): python backtest/alpha_lab.py
Self-test without a database: python backtest/alpha_lab.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import BACKTEST_REBALANCE_DAYS
from signals.ensemble import compute_signal_matrices, cross_sectional_zscore

TRADING_DAYS = 252
MIN_NAMES_PER_OBS = 50      # an IC observation needs a real cross-section
OOS_FRACTION = 0.40         # last 40% of history is held out for confirmation

# Promotion bars (multiple-testing aware: with ~21 tests the expected best
# |t| under the pure-noise null is ~2.6-2.9, hence the 3.0 in-sample bar).
BAR_T_IS = 3.0
BAR_T_OOS = 2.5
BAR_SPREAD_SHARPE_15BPS = 0.5
BAR_MAX_CORR_INCUMBENT = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE SIGNAL LIBRARY — each is pure, point-in-time, (date x ticker)
# Sign convention: HIGHER value = MORE attractive to own.
# ══════════════════════════════════════════════════════════════════════════════

def _rets(px):  # daily returns helper
    return px.pct_change()


def sig_rev_5d(px, vol, ohlc):
    """Short-term reversal: fade the 5-day move."""
    return -cross_sectional_zscore(px.pct_change(5))


def sig_rev_21d(px, vol, ohlc):
    """Monthly reversal (classic Jegadeesh 1990 horizon)."""
    return -cross_sectional_zscore(px.pct_change(21))


def sig_mom_12_1(px, vol, ohlc):
    """12-month momentum skipping the reversal month."""
    return cross_sectional_zscore(px.shift(21) / px.shift(252) - 1)


def sig_mom_3m(px, vol, ohlc):
    """3-month momentum, no skip."""
    return cross_sectional_zscore(px.pct_change(63))


def sig_52wk_high(px, vol, ohlc):
    """Proximity to 52-week high (George-Hwang): near-high names persist."""
    return cross_sectional_zscore(px / px.rolling(252).max())


def sig_idio_vol(px, vol, ohlc):
    """Low IDIOSYNCRATIC vol (residual to SPY): the anomaly, refined."""
    r = _rets(px)
    if "SPY" not in r.columns:
        return pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    m = r["SPY"]
    cov = r.rolling(126).cov(m)
    beta = cov.div(m.rolling(126).var(), axis=0)
    resid = r.sub(beta.mul(m, axis=0))
    return -cross_sectional_zscore(resid.rolling(126).std() * np.sqrt(TRADING_DAYS))


def sig_max_lottery(px, vol, ohlc):
    """MAX effect (Bali 2011): lottery-like names (big recent best days)
    underperform. Own the boring ones."""
    r = _rets(px)
    mx = r.rolling(21).apply(lambda x: np.mean(np.sort(x)[-5:]), raw=True)
    return -cross_sectional_zscore(mx)


def sig_skew(px, vol, ohlc):
    """Negative skew preference: right-tail lottery skew is overpriced."""
    return -cross_sectional_zscore(_rets(px).rolling(63).skew())


def sig_bab(px, vol, ohlc):
    """Betting against beta (Frazzini-Pedersen): low-beta names, levered by
    the book not the name, earn more per unit risk."""
    r = _rets(px)
    if "SPY" not in r.columns:
        return pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    m = r["SPY"]
    beta = r.rolling(252).cov(m).div(m.rolling(252).var(), axis=0)
    return -cross_sectional_zscore(beta)


def sig_amihud(px, vol, ohlc):
    """Amihud illiquidity premium: |ret|/dollar volume. CAVEAT: this edge
    pays you for taking the very costs that killed the headline number —
    treat a PROMOTE here with suspicion at high turnover."""
    if vol is None or vol.empty:
        return pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    dollar = (px * vol).replace(0, np.nan)
    illiq = (_rets(px).abs() / dollar).rolling(21).mean()
    return cross_sectional_zscore(np.log(illiq.replace(0, np.nan)))


def sig_volume_surprise(px, vol, ohlc):
    """Abnormal attention: volume spike vs its own 6m norm. High attention
    names tend to be temporarily overpriced -> own the ignored."""
    if vol is None or vol.empty:
        return pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    ratio = vol.rolling(21).mean() / vol.rolling(126).mean()
    return -cross_sectional_zscore(ratio)


def sig_up_day_consistency(px, vol, ohlc):
    """Information discreteness (Da-Gurun-Warachka): smooth, frequent small
    gains continue; jumpy gains reverse. Fraction of up days over 63d,
    conditioned on positive drift."""
    r = _rets(px)
    frac_up = (r > 0).rolling(63).mean()
    drift = px.pct_change(63)
    return cross_sectional_zscore(frac_up.where(drift > 0, 1 - frac_up))


def sig_overnight_reversal(px, vol, ohlc):
    """Cumulative overnight (close->open) return, faded: chronic overnight
    gappers underperform (Lou-Polk-Skouras decomposition). Uses RAW open/
    close; split days are clipped, 21d sum smooths the rest."""
    if not ohlc:
        return pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    o, c = ohlc["open"], ohlc["close"]
    overnight = (o / c.shift(1) - 1).clip(-0.2, 0.2)
    return -cross_sectional_zscore(overnight.rolling(21).sum())


def sig_intraday_mom(px, vol, ohlc):
    """The intraday leg of the same decomposition, followed."""
    if not ohlc:
        return pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    o, c = ohlc["open"], ohlc["close"]
    intraday = (c / o - 1).clip(-0.2, 0.2)
    return cross_sectional_zscore(intraday.rolling(21).sum())


def sig_range_strength(px, vol, ohlc):
    """Where in the daily range do closes land? Persistent close-near-high
    is buying pressure."""
    if not ohlc:
        return pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    h, l, c = ohlc["high"], ohlc["low"], ohlc["close"]
    pos = ((c - l) / (h - l).replace(0, np.nan)).clip(0, 1)
    return cross_sectional_zscore(pos.rolling(10).mean())


def sig_downside_vol(px, vol, ohlc):
    """Semi-vol refinement of low-vol: penalize downside variance only."""
    r = _rets(px)
    down = r.where(r < 0, 0.0)
    return -cross_sectional_zscore(down.rolling(126).std() * np.sqrt(TRADING_DAYS))


def sig_mom_vol_interaction(px, vol, ohlc):
    """Momentum conditioned on calm: trend in quiet names persists longer
    than trend in loud ones."""
    mom = px.shift(21) / px.shift(147) - 1
    v = _rets(px).rolling(63).std() * np.sqrt(TRADING_DAYS)
    return cross_sectional_zscore(mom / v.clip(lower=0.10))


CANDIDATES = {
    "rev_5d": sig_rev_5d,
    "rev_21d": sig_rev_21d,
    "mom_12_1": sig_mom_12_1,
    "mom_3m": sig_mom_3m,
    "hi_52wk": sig_52wk_high,
    "idio_lowvol": sig_idio_vol,
    "anti_lottery": sig_max_lottery,
    "anti_skew": sig_skew,
    "anti_beta": sig_bab,
    "amihud_illiq": sig_amihud,
    "low_attention": sig_volume_surprise,
    "smooth_mom": sig_up_day_consistency,
    "overnight_fade": sig_overnight_reversal,
    "intraday_mom": sig_intraday_mom,
    "range_strength": sig_range_strength,
    "downside_lowvol": sig_downside_vol,
    "calm_mom": sig_mom_vol_interaction,
}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION CORE (pure)
# ══════════════════════════════════════════════════════════════════════════════

def ic_series(sig: pd.DataFrame, px: pd.DataFrame, horizon: int) -> pd.Series:
    """Non-overlapping Spearman ICs: signal at t vs return t -> t+horizon,
    sampled every `horizon` days so t-stats are honest (no overlap games)."""
    from scipy.stats import spearmanr
    out, idx = [], []
    n = len(px)
    for t in range(260, n - horizon, horizon):
        s = sig.iloc[t]
        f = px.iloc[t + horizon] / px.iloc[t] - 1
        valid = s.dropna().index.intersection(f.dropna().index)
        if len(valid) >= MIN_NAMES_PER_OBS:
            ic, _ = spearmanr(s[valid].values, f[valid].values)
            if np.isfinite(ic):
                out.append(float(ic))
                idx.append(px.index[t])
    return pd.Series(out, index=idx)


def t_stat(ics: pd.Series) -> float:
    if len(ics) < 8 or ics.std() == 0:
        return 0.0
    return float(ics.mean() / (ics.std() / np.sqrt(len(ics))))


def quintile_spread(sig: pd.DataFrame, px: pd.DataFrame,
                    rebal: int = 5) -> dict:
    """Q5-minus-Q1 equal-weight spread portfolio with per-leg churn costs."""
    rets_d = px.pct_change()
    n = len(px)
    spread, churn_costs = [], []
    prev_top, prev_bot = set(), set()
    turn_accum, periods = 0.0, 0
    for t in range(260, n - rebal, rebal):
        s = sig.iloc[t].dropna()
        if len(s) < MIN_NAMES_PER_OBS:
            continue
        q = max(int(len(s) * 0.2), 10)
        ranked = s.sort_values()
        bot, top = set(ranked.index[:q]), set(ranked.index[-q:])
        period = rets_d.iloc[t + 1: t + rebal + 1]
        r_top = period[list(top)].mean(axis=1, skipna=True).sum()
        r_bot = period[list(bot)].mean(axis=1, skipna=True).sum()
        churn = 1.0
        if prev_top:
            churn = (1 - len(top & prev_top) / len(top)
                     + 1 - len(bot & prev_bot) / len(bot)) / 2
        spread.append(float(r_top - r_bot))
        churn_costs.append(churn * 2)   # both legs trade their churn fraction
        turn_accum += churn
        periods += 1
        prev_top, prev_bot = top, bot
    if not spread:
        return {"sharpe_gross": 0.0, "sharpe_5": 0.0, "sharpe_15": 0.0,
                "ann_ret_5": 0.0, "ann_turnover": 0.0}
    sp = np.array(spread)
    cc = np.array(churn_costs)
    per_year = TRADING_DAYS / rebal
    def _sh(net):
        return float(net.mean() / net.std() * np.sqrt(per_year)) if net.std() > 0 else 0.0
    return {
        "sharpe_gross": round(_sh(sp), 2),
        "sharpe_5": round(_sh(sp - cc * 5e-4), 2),
        "sharpe_15": round(_sh(sp - cc * 15e-4), 2),
        "ann_ret_5": round(float((sp - cc * 5e-4).mean() * per_year), 4),
        "ann_turnover": round(float(turn_accum / max(periods, 1) * per_year), 1),
    }


def corr_with_incumbent(sig: pd.DataFrame, incumbent: pd.DataFrame) -> float:
    """Average cross-sectional correlation with the production composite."""
    vals = []
    for t in range(300, len(sig), 21):
        a, b = sig.iloc[t], incumbent.iloc[t]
        valid = a.dropna().index.intersection(b.dropna().index)
        if len(valid) >= MIN_NAMES_PER_OBS:
            c = np.corrcoef(a[valid], b[valid])[0, 1]
            if np.isfinite(c):
                vals.append(c)
    return round(float(np.mean(vals)), 2) if vals else 0.0


def evaluate_signal(name: str, sig: pd.DataFrame, px: pd.DataFrame,
                    incumbent_composite: pd.DataFrame | None,
                    horizon: int = 21) -> dict:
    """Full adversarial evaluation of one signal, IS + OOS split."""
    split = int(len(px) * (1 - OOS_FRACTION))
    ics_is = ic_series(sig.iloc[:split], px.iloc[:split], horizon)
    ics_oos = ic_series(sig.iloc[split - 260:], px.iloc[split - 260:], horizon)
    t_is, t_oos = t_stat(ics_is), t_stat(ics_oos)
    same_sign = np.sign(ics_is.mean() if len(ics_is) else 0) == \
                np.sign(ics_oos.mean() if len(ics_oos) else 0) and t_is != 0
    q = quintile_spread(sig, px)
    corr = (corr_with_incumbent(sig, incumbent_composite)
            if incumbent_composite is not None else 0.0)

    passes = [
        t_is >= BAR_T_IS and t_oos >= BAR_T_OOS and same_sign,
        q["sharpe_15"] >= BAR_SPREAD_SHARPE_15BPS,
        abs(corr) < BAR_MAX_CORR_INCUMBENT,
    ]
    verdict = ("PROMOTE" if all(passes)
               else "WATCH" if sum(passes) >= 2
               else "KILL")
    return {
        "signal": name,
        "ic_is": round(float(ics_is.mean()), 4) if len(ics_is) else 0.0,
        "t_is": round(t_is, 2),
        "ic_oos": round(float(ics_oos.mean()), 4) if len(ics_oos) else 0.0,
        "t_oos": round(t_oos, 2),
        "spread_sh_5bps": q["sharpe_5"],
        "spread_sh_15bps": q["sharpe_15"],
        "turnover_x": q["ann_turnover"],
        "corr_incumbent": corr,
        "verdict": verdict,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM EXPERIMENTS — where last night's report said the money is leaking
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def _crisis_net(net_value: float):
    """Temporarily set the CRISIS net exposure in the regime ladder.
    Research-side override only; production table is restored on exit."""
    import risk.regime as rg
    old = dict(rg.EXPOSURE_BY_REGIME["CRISIS"])
    rg.EXPOSURE_BY_REGIME["CRISIS"]["net"] = net_value
    try:
        yield
    finally:
        rg.EXPOSURE_BY_REGIME["CRISIS"] = old


def run_system_experiments(px, vol, sector_map, asset_class) -> pd.DataFrame:
    from backtest.walkforward import run_walkforward
    rows = []

    def record(label, **kw):
        crisis0 = kw.pop("crisis_net", None)
        try:
            if crisis0 is not None:
                with _crisis_net(crisis0):
                    r = run_walkforward(px, volume=vol, sector_map=sector_map,
                                        asset_class=asset_class, **kw)
            else:
                r = run_walkforward(px, volume=vol, sector_map=sector_map,
                                    asset_class=asset_class, **kw)
            rows.append({"experiment": label, "cagr": r["cagr"],
                         "sharpe": r["sharpe"], "max_dd": r["max_drawdown"],
                         "turnover_x": r["annual_turnover_x"]})
            logger.info(f"  experiment {label}: Sharpe {r['sharpe']}, "
                        f"CAGR {r['cagr']:+.2%}, turnover {r['annual_turnover_x']}x")
        except Exception as e:
            logger.error(f"  experiment {label} failed: {e}")
            rows.append({"experiment": label, "cagr": None, "sharpe": None,
                         "max_dd": None, "turnover_x": None})

    record("baseline (5d rebal, crisis net -10%)",
           rebalance_days=BACKTEST_REBALANCE_DAYS)
    record("rebal 10d", rebalance_days=10)
    record("rebal 15d", rebalance_days=15)
    record("crisis net 0 (hedged, never net short)",
           rebalance_days=BACKTEST_REBALANCE_DAYS, crisis_net=0.0)
    record("rebal 10d + crisis net 0", rebalance_days=10, crisis_net=0.0)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def _load_ohlc(tickers, lookback):
    """Raw open/high/low/close pivots for the decomposition signals."""
    from sqlalchemy import text, bindparam
    from data.db import engine
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=int(lookback * 1.6))).date()
    stmt = text("SELECT date, ticker, open, high, low, close FROM daily_prices "
                "WHERE ticker IN :ts AND date >= :cutoff ORDER BY date"
                ).bindparams(bindparam("ts", expanding=True))
    df = pd.read_sql(stmt, engine, params={"ts": list(tickers), "cutoff": cutoff})
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    return {f: df.pivot(index="date", columns="ticker", values=f).tail(lookback)
            for f in ("open", "high", "low", "close")}


def run_lab(px, vol=None, ohlc=None, sector_map=None, asset_class=None,
            run_experiments=True) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    logger.info(f"ALPHA LAB — {px.shape[1]} tickers x {px.shape[0]} days; "
                f"{len(CANDIDATES)} candidates + 5 incumbents; "
                f"OOS holdout = last {OOS_FRACTION:.0%}")

    # Incumbents evaluated under the same rules, from the same production code.
    incumbents = compute_signal_matrices(px)
    incumbent_comp = None
    stack = [df for df in incumbents.values() if not df.empty]
    if stack:
        incumbent_comp = sum(df.fillna(0) for df in stack) / len(stack)

    rows = []
    for name, df in incumbents.items():
        rows.append(evaluate_signal(f"[incumbent] {name}", df, px, None))
    for name, fn in CANDIDATES.items():
        try:
            sig = fn(px, vol, ohlc or {})
            rows.append(evaluate_signal(name, sig, px, incumbent_comp))
            logger.info(f"  {rows[-1]['signal']:26s} t_is={rows[-1]['t_is']:+5.2f} "
                        f"t_oos={rows[-1]['t_oos']:+5.2f} -> {rows[-1]['verdict']}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
    league = pd.DataFrame(rows).sort_values("t_oos", ascending=False)

    experiments = None
    if run_experiments and sector_map is not None:
        logger.info("Running system experiments (5 walk-forwards)...")
        experiments = run_system_experiments(px, vol, sector_map, asset_class)
    return league, experiments


def _report(league: pd.DataFrame, experiments: pd.DataFrame | None,
            meta: str) -> str:
    lines = [
        "=" * 78,
        f"ALPHA LAB REPORT — {meta}",
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        "=" * 78, "",
        "LEAGUE TABLE (sorted by out-of-sample IC t-stat):",
        league.to_string(index=False), "",
        f"Promotion bars (pre-committed): t_is>={BAR_T_IS}, t_oos>={BAR_T_OOS} "
        f"same sign, spread Sharpe@15bps>={BAR_SPREAD_SHARPE_15BPS}, "
        f"|corr incumbent|<{BAR_MAX_CORR_INCUMBENT}.",
        "With ~21 signals tested, expect the BEST pure-noise t-stat to reach",
        "~2.6-2.9 by luck alone. That is why the bar sits at 3.0 and why a",
        "WATCH is not a PROMOTE. Anything below the bar is noise until it",
        "proves otherwise on data it has never seen.", "",
    ]
    if experiments is not None:
        lines += ["SYSTEM EXPERIMENTS (full production walk-forward each):",
                  experiments.to_string(index=False), "",
                  "Reading: an experiment matters if Sharpe improves vs baseline",
                  "WITHOUT max_dd deepening materially. Turnover reduction that",
                  "preserves Sharpe is free money (costs saved, nothing lost).", ""]
    lines += ["NEXT ACTION RULE: at most ONE change ships at a time. A PROMOTE",
              "here earns a shadow slot, then a full-system walk-forward with",
              "the signal included; only beating the baseline end-to-end earns",
              "weight in the live book.", "=" * 78]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="JARVIS Alpha Lab")
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--no-experiments", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        rng = np.random.default_rng(3)
        n_d, n_t = 900, 150
        dates = pd.bdate_range("2021-01-04", periods=n_d)
        names = [f"T{i:03d}" for i in range(n_t)]
        mkt = rng.standard_normal(n_d) * 0.009
        r = (rng.normal(3e-4, 2e-4, n_t)
             + rng.uniform(.01, .03, n_t)
             * (0.6 * rng.standard_normal((n_d, n_t)) + 0.4 * mkt[:, None]))
        px = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=dates,
                          columns=names)
        px["SPY"] = 100 * np.exp(np.cumsum(mkt * 1.1 + 2e-4))
        league, _ = run_lab(px, run_experiments=False)
        print(_report(league, None, "SYNTHETIC NOISE (expect: KILLs everywhere)"))
        return 0

    from config.universe import (get_full_universe, get_sector_map,
                                 get_asset_class_map)
    from data.ingest import get_prices_for_universe
    lookback = int(args.years * TRADING_DAYS)
    tickers = get_full_universe()
    px, vol = get_prices_for_universe(tickers, lookback, with_volume=True)
    if px.empty:
        logger.error("no prices in DB")
        return 2
    ohlc = _load_ohlc(list(px.columns), lookback)
    league, experiments = run_lab(px, vol, ohlc, get_sector_map(),
                                  get_asset_class_map(),
                                  run_experiments=not args.no_experiments)
    report = _report(league, experiments,
                     f"REAL DATA ({px.shape[1]} tickers, {args.years}y)")
    print("\n" + report + "\n")
    with open("alpha_lab_report.txt", "w") as f:
        f.write(report)
    logger.info("written: alpha_lab_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
