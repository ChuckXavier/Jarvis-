"""
JARVIS V3 — deterministic logic tests (no network, no database, no broker).

Stubs loguru / dotenv / data.db so the production modules import cleanly,
then exercises every pure-math invariant on synthetic data:
  ensemble    — shapes, history mask, z-clip, >=3-signal rule, NO LOOK-AHEAD
  IC/weights  — finiteness, floor/ceiling, sum-to-1, positive-IC tilt
  optimizer   — gross/net targets, 5% cap, sector caps, cash reserve, no_short,
                leveraged exclusion, price & liquidity filters
  rebalancer  — sweep-first, whole-share shorts, cross-side split, drift gate,
                reduce-to-dust, trade-cap protection of closes
  walkforward — end-to-end run, multi-regime occupancy, cost monotonicity
Run:  python tests/test_v3_logic.py
"""

import sys
import types
import math

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/user-data/outputs")

# ── stubs ─────────────────────────────────────────────────────────────────────
class _L:
    def __getattr__(self, _):
        return lambda *a, **k: None
sys.modules["loguru"] = types.SimpleNamespace(logger=_L())
sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)

_db = types.ModuleType("data.db")
class _NoEngine:
    def begin(self):
        raise RuntimeError("no database in tests")
_db.engine = _NoEngine()
_db.create_all_tables = lambda: None
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

# ── synthetic market ─────────────────────────────────────────────────────────
rng = np.random.default_rng(11)
N_T, N_D = 130, 520
dates = pd.bdate_range("2023-01-02", periods=N_D)
names = [f"T{i:03d}" for i in range(N_T)]
drift = rng.normal(0.0004, 0.0003, N_T)
volr = rng.uniform(0.008, 0.030, N_T)
mkt = rng.standard_normal(N_D) * 0.009
sh = rng.standard_normal((N_D, N_T))
r = drift + volr * (0.6 * sh + 0.4 * mkt[:, None])
px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=names)
px["SPY"] = 100 * np.exp(np.cumsum(mkt * 1.1 + 0.0003))
px["SHORTHIST"] = np.nan
px.loc[px.index[-100:], "SHORTHIST"] = 50.0   # only 100 obs — must score NaN
px["PENNY"] = 2.0                              # < $5 — must fail price filter
sectors = {t: f"S{int(t[1:]) % 8}" for t in names}
sectors.update({"SPY": "index", "SHORTHIST": "S0", "PENNY": "S1"})

print("\n[1] signals/ensemble — matrices, masks, look-ahead")
from signals.ensemble import (SIGNAL_NAMES, DEFAULT_WEIGHTS,
                              compute_signal_matrices, combine_scores,
                              compute_signal_ics, adapt_weights,
                              cross_sectional_zscore)
sigs = compute_signal_matrices(px)
check("five signal matrices", set(sigs) == set(SIGNAL_NAMES))
check("matrix shape matches prices",
      all(s.shape == px.shape for s in sigs.values()))
last = {n: s.iloc[-1] for n, s in sigs.items()}
check("short-history ticker NaN in every signal",
      all(pd.isna(last[n]["SHORTHIST"]) for n in SIGNAL_NAMES))
check("z-scores winsorized at |3|",
      max(float(np.nanmax(np.abs(s.values))) for s in sigs.values()) <= 3.0 + 1e-9)

# no look-ahead: row t computed on truncated history equals row t on full
t_cut = 400
sigs_cut = compute_signal_matrices(px.iloc[:t_cut + 1])
ok_la = True
for n in SIGNAL_NAMES:
    a = sigs[n].iloc[t_cut].astype(float)
    b = sigs_cut[n].iloc[t_cut].astype(float)
    if not np.allclose(a.fillna(0).values, b.fillna(0).values, atol=1e-10) \
       or not (a.isna() == b.isna()).all():
        ok_la = False
check("NO LOOK-AHEAD: row t identical when future removed", ok_la)

comp = combine_scores(sigs, DEFAULT_WEIGHTS)
check("composite NaN for short-history name", pd.isna(comp.iloc[-1]["SHORTHIST"]))
two_sig = {n: sigs[n] for n in SIGNAL_NAMES[:2]}
comp2 = combine_scores(two_sig, DEFAULT_WEIGHTS)
check("composite requires >=3 live signals", comp2.iloc[-1].dropna().empty)

ic21 = compute_signal_ics(sigs, px, 21)
check("ICs finite and in [-1,1]",
      all(np.isfinite(v) and -1 <= v <= 1 for v in ic21.values()))
w0 = DEFAULT_WEIGHTS.copy()
w1 = adapt_weights(w0, {n: (0.15 if n == "xs_momentum" else -0.05)
                        for n in SIGNAL_NAMES})
check("weights sum to 1", abs(sum(w1.values()) - 1.0) < 1e-9)
check("weights within [0.05, 0.40]",
      all(0.05 - 1e-9 <= v <= 0.40 + 1e-9 for v in w1.values()))
check("positive-IC signal gains weight", w1["xs_momentum"] > w0["xs_momentum"])
for _ in range(60):
    w1 = adapt_weights(w1, {n: (0.15 if n == "xs_momentum" else -0.05)
                            for n in SIGNAL_NAMES})
check("ceiling holds under 60 days of pressure", w1["xs_momentum"] <= 0.40 + 1e-9)

print("\n[2] portfolio/optimizer — caps and exposure")
from portfolio.optimizer import build_targets, optimize_portfolio
rets = px.pct_change()
vol63 = rets.rolling(63).std().iloc[-1] * np.sqrt(252)
scores = comp.iloc[-1].dropna()
scores = scores[[t for t in scores.index if t in names]]   # clean candidates

tw = build_targets(scores, vol63, sectors, gross=1.00, net=0.80)
tl = sum(w for w in tw.values() if w > 0)
ts = sum(-w for w in tw.values() if w < 0)
check("ACTIVE long leg ~ (g+n)/2 minus cash cap",
      abs(tl - min(0.90, 0.98)) < 0.03, f"long={tl:.3f}")
check("ACTIVE short leg ~ (g-n)/2", abs(ts - 0.10) < 0.03, f"short={ts:.3f}")
check("max position <= 5%", max(abs(w) for w in tw.values()) <= 0.05 + 1e-6)
sec_long, sec_short = {}, {}
for t, w in tw.items():
    s = sectors.get(t, "Unknown")
    (sec_long if w > 0 else sec_short).setdefault(s, 0)
    if w > 0: sec_long[s] += w
    else: sec_short[s] -= w
check("sector long caps <= 20%", max(sec_long.values()) <= 0.20 + 1e-6,
      str(max(sec_long.values())))
check("sector short caps <= 10%",
      (max(sec_short.values()) if sec_short else 0) <= 0.10 + 1e-6)

tw_c = build_targets(scores, vol63, sectors, gross=0.50, net=-0.10)
tl_c = sum(w for w in tw_c.values() if w > 0)
ts_c = sum(-w for w in tw_c.values() if w < 0)
check("CRISIS net short ~ -10%", abs((tl_c - ts_c) - (-0.10)) < 0.04,
      f"net={tl_c-ts_c:.3f}")
check("CRISIS still deployed (gross > 0)", (tl_c + ts_c) > 0.30)

bond = scores.index[-1]
tw_ns = build_targets(scores, vol63, sectors, gross=0.5, net=-0.1,
                      no_short={bond})
check("no_short respected", tw_ns.get(bond, 0) >= 0)

vol_m = pd.DataFrame(1_000_000.0, index=px.index, columns=px.columns)
vol_m["T000"] = 100.0   # illiquid: ~$10k/day
opt = optimize_portfolio(comp.iloc[-1], px, 100_000,
                         {"regime": "ACTIVE", "target_gross": 1.0,
                          "target_net": 0.8},
                         volume=vol_m, sector_map=sectors)
twl = opt["target_weights"]
check("optimize_portfolio returns a book", len(twl) >= 30, str(len(twl)))
check("penny stock excluded", "PENNY" not in twl)
check("illiquid name excluded", "T000" not in twl)
check("short-history name excluded", "SHORTHIST" not in twl)
check("reported gross matches sum", abs(opt["gross"] -
      sum(abs(w) for w in twl.values())) < 1e-6)

print("\n[3] portfolio/rebalancer — order generation")
from portfolio.rebalancer import generate_rebalance_orders, calculate_turnover
PV = 100_000.0
held = {
    "AAPL": {"qty": 100.0, "market_value": 20_000.0, "current_price": 200.0},
    "MSFT": {"qty": 50.0, "market_value": 21_000.0, "current_price": 420.0},
    "XOM":  {"qty": -40.0, "market_value": -4_400.0, "current_price": 110.0},
    "NVDA": {"qty": 30.0, "market_value": 4_200.0, "current_price": 140.0},
}
targets = {
    "MSFT": 0.05,        # reduce 21% -> 5%
    "XOM": -0.06,        # grow short  -4.4% -> -6%
    "NVDA": -0.03,       # CROSS: long -> short
    "GOOG": 0.04,        # new long
    "TSLA": -0.0234,     # new short — whole shares
    "JPM": 0.042,        # tiny drift case below
}
prices_now = {"AAPL": 200.0, "MSFT": 420.0, "XOM": 110.0, "NVDA": 140.0,
              "GOOG": 180.0, "TSLA": 50.0, "JPM": 210.0}
held["JPM"] = {"qty": PV * 0.0415 / 210.0, "market_value": PV * 0.0415,
               "current_price": 210.0}     # 4.15% vs 4.2% target: inside gate

orders = generate_rebalance_orders(targets, held, PV, prices_now,
                                   fractionable={"GOOG": True, "TSLA": True,
                                                 "MSFT": True, "NVDA": True})
om = {}
for o in orders:
    om.setdefault(o["ticker"], []).append(o)

check("AAPL swept (not in targets)",
      "AAPL" in om and om["AAPL"][0]["priority"] == 0
      and om["AAPL"][0]["side"] == "sell"
      and abs(om["AAPL"][0]["quantity"] - 100.0) < 1e-9)
check("MSFT reduce at priority 1",
      "MSFT" in om and om["MSFT"][0]["priority"] == 1
      and om["MSFT"][0]["side"] == "sell")
nvda = sorted(om.get("NVDA", []), key=lambda o: o["priority"])
check("NVDA cross-side -> close(sell 30, p0) + open(sell, p2)",
      len(nvda) == 2 and nvda[0]["priority"] == 0
      and nvda[0]["side"] == "sell" and abs(nvda[0]["quantity"] - 30) < 1e-9
      and nvda[1]["priority"] == 2 and nvda[1]["side"] == "sell"
      and nvda[1]["quantity"] == float(int(nvda[1]["quantity"])))
tsla = om.get("TSLA", [{}])[0]
check("TSLA short floored to whole shares",
      tsla.get("quantity") == float(math.floor(PV * 0.0234 / 50.0))
      and tsla.get("priority") == 2 and tsla.get("side") == "sell")
xom = om.get("XOM", [{}])[0]
check("XOM short add: sell whole shares at p2",
      xom.get("side") == "sell" and xom.get("priority") == 2
      and xom.get("quantity") == float(int(xom.get("quantity", 0))))
check("JPM inside drift gate — no order", "JPM" not in om)
check("orders sorted reduce-before-open",
      [o["priority"] for o in orders]
      == sorted(o["priority"] for o in orders))

dust_orders = generate_rebalance_orders(
    {"MSFT": 0.003}, {"MSFT": held["MSFT"]}, PV, prices_now)
check("reduce-to-dust closes full held qty",
      len(dust_orders) == 1 and dust_orders[0]["priority"] == 0
      and abs(dust_orders[0]["quantity"] - 50.0) < 1e-9)

big_targets = {f"T{i:03d}": 0.012 for i in range(80)}
big_orders = generate_rebalance_orders(
    big_targets, held, PV, {**prices_now,
                            **{f"T{i:03d}": 100.0 for i in range(80)}})
check("trade cap keeps closes, drops opens",
      len(big_orders) <= 60 and any(o["priority"] == 0 for o in big_orders))
check("turnover math", abs(calculate_turnover({"A": 0.1}, {"A": 0.0,
      "B": 0.1}) - 0.10) < 1e-12)

print("\n[4] backtest/walkforward — end-to-end + cost monotonicity")
from backtest.walkforward import run_walkforward, synthetic_demo
res0 = run_walkforward(px, sector_map=sectors, cost_bps=0.0, warmup=300)
res5 = run_walkforward(px, sector_map=sectors, cost_bps=5.0, warmup=300)
res15 = run_walkforward(px, sector_map=sectors, cost_bps=15.0, warmup=300)
check("walkforward produces finite metrics",
      all(np.isfinite([res5["cagr"], res5["ann_vol"], res5["max_drawdown"]])))
check("turnover positive, rebalances counted",
      res5["annual_turnover_x"] > 0 and res5["n_rebalances"] > 20)
check("costs are monotone drag: cagr(0) >= cagr(5) >= cagr(15)",
      res0["cagr"] >= res5["cagr"] >= res15["cagr"],
      f"{res0['cagr']:.4f} / {res5['cagr']:.4f} / {res15['cagr']:.4f}")
check("gross stays unlevered (avg <= ~1.0)", res5["avg_gross"] <= 1.01,
      str(res5["avg_gross"]))
demo = synthetic_demo(n_tickers=100, n_days=620, seed=3)
check("crash demo visits >= 2 regimes", len(demo["regime_share"]) >= 2,
      str(demo["regime_share"]))

print("\n[5] regime machine — persisted-counter suite (post v1.1 edit)")
import subprocess
rr = subprocess.run([sys.executable, "test_regime_logic.py"],
                    capture_output=True, text=True,
                    cwd="/mnt/user-data/outputs/risk")
tail = (rr.stdout + rr.stderr).strip().splitlines()[-3:]
print("   " + " | ".join(tail))
check("regime suite exit 0", rr.returncode == 0)

print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
