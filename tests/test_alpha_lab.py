"""
ALPHA LAB validation: three properties, or the lab is worthless.
  1. DETECTION  — a real (planted) alpha must be found and PROMOTEd.
  2. SKEPTICISM — pure-noise candidates must overwhelmingly KILL, and the
                  planted signal must outrank every noise candidate OOS.
  3. NO LOOK-AHEAD — candidate matrices at time t are unchanged when the
                  future is deleted.
Run: python tests/test_alpha_lab.py
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

from backtest.alpha_lab import (CANDIDATES, evaluate_signal, run_lab,
                                cross_sectional_zscore)

# ── build a market with ONE planted, persistent alpha ────────────────────────
rng = np.random.default_rng(42)
N_D, N_T = 1000, 150
dates = pd.bdate_range("2020-01-02", periods=N_D)
names = [f"T{i:03d}" for i in range(N_T)]
quality = rng.standard_normal(N_T)                      # the hidden truth
qz = (quality - quality.mean()) / quality.std()
K = 1e-3                                                # daily edge injected
mkt = rng.standard_normal(N_D) * 0.009
idio = rng.standard_normal((N_D, N_T))
volr = rng.uniform(0.010, 0.028, N_T)
r = (rng.normal(2e-4, 1.5e-4, N_T) + K * qz
     + volr * (0.65 * idio + 0.35 * mkt[:, None]))
px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates,
                  columns=names)
px["SPY"] = 100 * np.exp(np.cumsum(mkt * 1.1 + 2e-4))

def sig_planted(p, v, o):
    """The true quality, known point-in-time (static -> trivially PIT)."""
    base = pd.DataFrame(np.tile(qz, (len(p), 1)), index=p.index,
                        columns=names)
    return cross_sectional_zscore(base.reindex(columns=p.columns))

print("\n[1] DETECTION — planted alpha must be found")
res_planted = evaluate_signal("planted_quality", sig_planted(px, None, {}),
                              px, None)
print(f"      planted: t_is={res_planted['t_is']} t_oos={res_planted['t_oos']} "
      f"spread15={res_planted['spread_sh_15bps']} -> {res_planted['verdict']}")
check("planted signal PROMOTEd", res_planted["verdict"] == "PROMOTE",
      str(res_planted))
check("planted OOS t-stat is decisive (>4)", res_planted["t_oos"] > 4)

print("\n[2] SKEPTICISM — on a PURE-NOISE market, candidates must die")
# (On the planted market, momentum-family candidates legitimately detect the
# quality drift indirectly — that is correct detection, not gullibility. The
# skepticism property is tested where there is truly nothing to find.)
px_noise = pd.DataFrame(
    100 * np.exp(np.cumsum(rng.normal(2e-4, 0.018, (N_D, 100)), axis=0)),
    index=dates, columns=names[:100])
px_noise["SPY"] = px["SPY"]
noise_rows = []
for name, fn in CANDIDATES.items():
    try:
        sig = fn(px_noise, None, {})
        if sig.dropna(how="all").empty:
            continue        # OHLC/volume candidates inert without that data
        noise_rows.append(evaluate_signal(name, sig, px_noise, None))
    except Exception as e:
        check(f"candidate {name} runs without error", False, str(e))
n_eval = len(noise_rows)
n_promote = sum(1 for r_ in noise_rows if r_["verdict"] == "PROMOTE")
n_kill = sum(1 for r_ in noise_rows if r_["verdict"] == "KILL")
best_noise_toos = max(abs(r_["t_oos"]) for r_ in noise_rows)
print(f"      {n_eval} candidates on noise: {n_kill} KILL, "
      f"{n_eval - n_kill - n_promote} WATCH, {n_promote} PROMOTE; "
      f"best |t_oos| = {best_noise_toos:.2f}")
check("no more than 1 lucky PROMOTE on pure noise", n_promote <= 1,
      f"{n_promote} promoted")
check("majority of candidates KILLed on pure noise", n_kill >= n_eval * 0.6,
      f"{n_kill}/{n_eval}")
check("planted alpha (real market) outranks best noise t-stat",
      res_planted["t_oos"] > best_noise_toos)

print("\n[3] NO LOOK-AHEAD — truncating the future changes nothing")
t_cut = 700
ok = True
for name in ("rev_5d", "mom_12_1", "calm_mom", "anti_beta"):
    full = CANDIDATES[name](px, None, {})
    cut = CANDIDATES[name](px.iloc[:t_cut + 1], None, {})
    a, b = full.iloc[t_cut].astype(float), cut.iloc[t_cut].astype(float)
    if not (np.allclose(a.fillna(0), b.fillna(0), atol=1e-10)
            and (a.isna() == b.isna()).all()):
        ok = False
        print(f"      look-ahead detected in {name}")
check("candidate rows at t identical when future removed", ok)

print("\n[4] END-TO-END — full league on the planted market")
league, _ = run_lab(px, run_experiments=False)
cand = league[~league["signal"].str.startswith("[incumbent]")]
promotes = set(cand[cand["verdict"] == "PROMOTE"]["signal"])
drift_family = {"mom_12_1", "mom_3m", "hi_52wk", "calm_mom", "smooth_mom",
                "mom_6_1"}
strays = promotes - drift_family
print(f"      planted-market promotes: {sorted(promotes)}")
check("planted-market promotes are drift-detecting families only",
      len(strays) == 0, f"unexpected: {strays}")

print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
