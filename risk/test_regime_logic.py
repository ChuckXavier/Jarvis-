"""
Unit + simulation tests for the regime engine's PURE decision logic.
Stubs loguru and the DB layer so we can import the module and exercise
decide_regime / get_target_exposure with zero I/O.

The headline test reproduces the production bug: with the OLD in-memory
counters (reset every container restart), a sustained recovery never confirms.
With the NEW persisted counters, it confirms on schedule.
"""
import sys, types

# --- stub loguru (not installed here; present in production) ---
_loguru = types.ModuleType("loguru")
class _L:
    def __getattr__(self, _): return lambda *a, **k: None
_loguru.logger = _L()
sys.modules["loguru"] = _loguru

# --- stub data.db + sqlalchemy so module import never touches a database ---
_data = types.ModuleType("data"); _db = types.ModuleType("data.db")
_db.engine = None
_db.get_all_prices = lambda *a, **k: None
_db.get_macro = lambda *a, **k: None
sys.modules["data"] = _data; sys.modules["data.db"] = _db
_sa = types.ModuleType("sqlalchemy"); _sa.text = lambda s: s
sys.modules["sqlalchemy"] = _sa

import importlib.util
spec = importlib.util.spec_from_file_location("regime", "regime.py")
regime = importlib.util.module_from_spec(spec); spec.loader.exec_module(regime)

def vote(t, v, c, avail=(True, True, True)):
    """Build a votes dict with explicit per-signal votes (+1/0/-1)."""
    def mk(val, ok): return {"vote": val, "value": float(val), "detail": f"v={val}", "available": ok}
    return {"trend": mk(t, avail[0]), "vol": mk(v, avail[1]), "credit": mk(c, avail[2])}

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")

print("=== get_target_exposure (leverage gate) ===")
regime.ALLOW_LEVERAGE = False
check("CRISIS unlevered gross 0.50", regime.get_target_exposure("CRISIS")["gross"] == 0.50)
check("CRISIS deploys (>5% cash myth dead)", regime.get_target_exposure("CRISIS")["gross"] >= 0.50)
check("ACTIVE unlevered gross 1.00", regime.get_target_exposure("ACTIVE")["gross"] == 1.00)
regime.ALLOW_LEVERAGE = True
check("ACTIVE levered gross 1.30 only when enabled", regime.get_target_exposure("ACTIVE")["gross"] == 1.30)
regime.ALLOW_LEVERAGE = False

print("\n=== single-step decisions ===")
m_active = {"regime": "ACTIVE", "bullish_count": 0, "bearish_count": 0}
d = regime.decide_regime(vote(-1, -1, 0), m_active)
check("ACTIVE -> CAUTIOUS on first 2/3 risk-off", d["regime"] == "CAUTIOUS")

m_crisis = {"regime": "CRISIS", "bullish_count": 0, "bearish_count": 0}
d = regime.decide_regime(vote(1, 1, 0), m_crisis)
check("recovery fast-path CRISIS -> ACTIVE (trend+vol up)", d["regime"] == "ACTIVE" and d["recovery"])

d = regime.decide_regime(vote(-1, 0, 0, avail=(True, False, False)), m_active)
check("hold when <2 live signals", d["regime"] == "ACTIVE" and "live signal" in d["reason"])

print("\n=== THE BUG: sustained recovery, old (reset) vs new (persisted) counters ===")
# Environment: trend positive but volatility/credit only neutral (no clean
# recovery), so confirmation must come from the 2-of-3 day count accumulating.
env = vote(1, 0, 1)  # trend +1, vol 0, credit +1  -> risk_on = 2 each day
CONFIRM = regime.CONFIRM_RISK_ON

# OLD behaviour: every "day" the container restarted, so counters reset to 0.
old_confirmed = False
for day in range(10):
    fresh = {"regime": "CRISIS", "bullish_count": 0, "bearish_count": 0}  # reset each restart
    d = regime.decide_regime(env, fresh)
    if d["regime"] == "ACTIVE":
        old_confirmed = True
print(f"  old/in-memory: confirmed ACTIVE within 10 days? {old_confirmed}")
check("old in-memory counters NEVER confirm (reproduces bug)", old_confirmed is False)

# NEW behaviour: counters persist across runs.
machine = {"regime": "CRISIS", "bullish_count": 0, "bearish_count": 0}
day_confirmed = None
for day in range(1, 11):
    d = regime.decide_regime(env, machine)
    machine = {"regime": d["regime"], "bullish_count": d["bullish_count"], "bearish_count": d["bearish_count"]}
    if d["regime"] == "ACTIVE" and day_confirmed is None:
        day_confirmed = day
print(f"  new/persisted: confirmed ACTIVE on day {day_confirmed} (CONFIRM_RISK_ON={CONFIRM})")
check("persisted counters confirm ACTIVE on schedule", day_confirmed == CONFIRM)

print("\n=== confirmed crisis path ===")
machine = {"regime": "ACTIVE", "bullish_count": 0, "bearish_count": 0}
seq = []
for day in range(1, 4):
    d = regime.decide_regime(vote(-1, -1, -1), machine)  # full 3/3 risk-off
    machine = {"regime": d["regime"], "bullish_count": d["bullish_count"], "bearish_count": d["bearish_count"]}
    seq.append(d["regime"])
print(f"  ACTIVE under 3/3 risk-off, day-by-day: {seq}")
check("reaches CRISIS after confirmation window", "CRISIS" in seq)
check("steps through CAUTIOUS first (no binary slam)", seq[0] == "CAUTIOUS")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
