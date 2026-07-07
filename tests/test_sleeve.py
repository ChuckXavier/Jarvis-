"""
ENGINE 2 validation — the sleeve must: detect real trends, respect every cap,
compute nothing from the future, earn ~nothing on noise, and combine
correctly. Run: python -B tests/test_sleeve.py
"""

import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/user-data/outputs")

class _L:
    def __getattr__(self, _):
        return lambda *a, **k: None
sys.modules["loguru"] = types.SimpleNamespace(logger=_L())
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
_db = types.ModuleType("data.db")
class _E:
    def begin(self):
        raise RuntimeError("no db")
_db.engine = _E()
sys.modules.setdefault("data", types.ModuleType("data"))
sys.modules["data.db"] = _db

PASS, FAIL = 0, 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")

from signals.tsmom import (SLEEVE_UNIVERSE, tsmom_scores, build_sleeve_targets,
                           SLEEVE_MAX_ASSET, SLEEVE_MAX_CLASS, SLEEVE_GROSS_CAP)
from backtest.sleeve_lab import run_sleeve_walkforward, combination_table

rng = np.random.default_rng(9)
N_D = 900
dates = pd.bdate_range("2021-01-04", periods=N_D)
cols = list(SLEEVE_UNIVERSE)

print("\n[1] trend detection — signs must follow the trends")
r = rng.normal(0, 0.008, (N_D, len(cols)))
px = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=dates, columns=cols)
px["SPY"] = 100 * np.exp(np.cumsum(np.full(N_D, 0.0008)
                                   + rng.normal(0, 0.004, N_D)))   # up-trend
px["TLT"] = 100 * np.exp(np.cumsum(np.full(N_D, -0.0008)
                                   + rng.normal(0, 0.004, N_D)))   # down-trend
sc = tsmom_scores(px)
check("uptrending asset scores strongly positive", sc["SPY"].iloc[-1] >= 0.9,
      str(sc["SPY"].iloc[-1]))
check("downtrending asset scores strongly negative", sc["TLT"].iloc[-1] <= -0.9,
      str(sc["TLT"].iloc[-1]))
short_px = px.copy()
short_px["NEWBIE"] = np.nan
short_px.loc[short_px.index[-100:], "NEWBIE"] = 50.0
check("insufficient history scores NaN",
      pd.isna(tsmom_scores(short_px)["NEWBIE"].iloc[-1]))

print("\n[2] no look-ahead")
t_cut = 700
full = tsmom_scores(px).iloc[t_cut]
cut = tsmom_scores(px.iloc[:t_cut + 1]).iloc[t_cut]
check("scores at t identical when future removed",
      np.allclose(full.fillna(0), cut.fillna(0), atol=1e-12)
      and (full.isna() == cut.isna()).all())

print("\n[3] target construction — every cap holds")
vol_row = pd.Series(rng.uniform(0.05, 0.30, len(px.columns)),
                    index=px.columns)
w = build_sleeve_targets(sc.iloc[-1], vol_row)
gross = sum(abs(x) for x in w.values())
check("gross <= cap", gross <= SLEEVE_GROSS_CAP + 1e-9, f"{gross:.3f}")
check("per-asset cap holds",
      max(abs(x) for x in w.values()) <= SLEEVE_MAX_ASSET + 1e-9)
cls_sum = {}
for t, x in w.items():
    c = SLEEVE_UNIVERSE.get(t, "other")
    cls_sum[c] = cls_sum.get(c, 0) + abs(x)
check("per-class cap holds", max(cls_sum.values()) <= SLEEVE_MAX_CLASS + 1e-9,
      str(cls_sum))
check("signs follow scores",
      all(np.sign(w[t]) == np.sign(sc.iloc[-1][t]) for t in w))
check("SPY long and TLT short in targets",
      w.get("SPY", 0) > 0 and w.get("TLT", 0) < 0)

print("\n[4] no alpha from noise + cost monotonicity")
px_noise = pd.DataFrame(100 * np.exp(np.cumsum(
    rng.normal(0, 0.010, (N_D, len(cols))), 0)), index=dates, columns=cols)
r0 = run_sleeve_walkforward(px_noise, cost_bps=0.0)
r5 = run_sleeve_walkforward(px_noise, cost_bps=5.0)
r15 = run_sleeve_walkforward(px_noise, cost_bps=15.0)
check("noise Sharpe near zero (|Sharpe| < 0.6)", abs(r5["sharpe"]) < 0.6,
      str(r5["sharpe"]))
check("costs are monotone drag", r0["cagr"] >= r5["cagr"] >= r15["cagr"])
check("turnover is low (trend trades slowly, < 6x/yr)",
      r5["annual_turnover_x"] < 6.0, str(r5["annual_turnover_x"]))

print("\n[5] trending world — sleeve must harvest it")
trend_dirs = rng.choice([-1.0, 1.0], len(cols))
r_tr = trend_dirs * 0.0006 + rng.normal(0, 0.008, (N_D, len(cols)))
px_tr = pd.DataFrame(100 * np.exp(np.cumsum(r_tr, 0)), index=dates,
                     columns=cols)
rt = run_sleeve_walkforward(px_tr, cost_bps=5.0)
check("captures persistent trends (Sharpe > 1.5)", rt["sharpe"] > 1.5,
      str(rt["sharpe"]))

print("\n[6] combination math")
idx = pd.bdate_range("2022-01-03", periods=500)
a = pd.Series(rng.normal(4e-4, 0.010, 500), index=idx)
b = pd.Series(rng.normal(3e-4, 0.006, 500), index=idx)
df, corr = combination_table(a, b, allocations=(0.5,))
check("correlation ~0 for independent series", abs(corr) < 0.15, str(corr))
blend = df[df["portfolio"] == "50/50 blend"].iloc[0]
manual = 0.5 * a + 0.5 * b
manual_sh = manual.mean() / manual.std() * np.sqrt(252)
check("blend Sharpe matches manual computation",
      abs(blend["sharpe"] - round(manual_sh, 2)) < 0.02,
      f"{blend['sharpe']} vs {manual_sh:.2f}")
check("diversification visible: blend Sharpe >= max single engine",
      blend["sharpe"] >= max(df["sharpe"].iloc[0], df["sharpe"].iloc[1]) - 0.05)

print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
