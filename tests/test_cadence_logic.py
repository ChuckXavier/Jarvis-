"""
Cadence + crisis-posture tests for the 2026-07-07 lab winner
("rebal 10d + crisis net 0": walk-forward Sharpe 0.20 -> 0.42).

Pure-logic only: cadence decision, stop-loss breach selection, and the
CRISIS exposure target. No database, no broker, no network.
Run: python -m tests.test_cadence_logic
"""

import sys

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


print("\n[1] cadence decision — is_full_rebalance_day")
from scheduler import is_full_rebalance_day, _stop_loss_breaches

check("no full rebalance on record -> full day",
      is_full_rebalance_day(None, 10))
check("day 0 (rebalanced today) -> skip",
      not is_full_rebalance_day(0, 10))
check("day 9 -> skip", not is_full_rebalance_day(9, 10))
check("day 10 -> full", is_full_rebalance_day(10, 10))
check("day 14 (missed a window) -> full", is_full_rebalance_day(14, 10))
check("cadence 1 behaves like daily", is_full_rebalance_day(1, 1))

print("\n[2] stop-loss breach selection — _stop_loss_breaches")
positions = {
    "AAPL": {"unrealized_pnl_pct": -0.081},   # through the stop
    "MSFT": {"unrealized_pnl_pct": -0.08},    # exactly at the stop
    "NVDA": {"unrealized_pnl_pct": -0.05},    # drawdown, not at stop
    "TSLA": {"unrealized_pnl_pct": 0.12},     # winner
    "JPM": {},                                # no pnl field -> treated as 0
    "XOM": "not-a-dict",                      # malformed entry ignored
}
breached = _stop_loss_breaches(positions, -0.08)
check("through-stop position breaches", "AAPL" in breached)
check("exactly-at-stop position breaches", "MSFT" in breached)
check("mid-drawdown position held", "NVDA" not in breached)
check("winner held", "TSLA" not in breached)
check("missing pnl field held", "JPM" not in breached)
check("malformed entry ignored", "XOM" not in breached)
check("nothing else swept in", sorted(breached) == ["AAPL", "MSFT"])

print("\n[3] crisis posture — hedged, never net short")
from risk.regime import get_target_exposure, EXPOSURE_BY_REGIME

crisis = get_target_exposure("CRISIS")
check("CRISIS net is 0.0 (lab winner)", crisis["net"] == 0.0,
      f"net={crisis['net']}")
check("CRISIS gross unchanged at 50%", crisis["gross"] == 0.50)
check("no regime is net short",
      all(v["net"] >= 0.0 for v in EXPOSURE_BY_REGIME.values()),
      str(EXPOSURE_BY_REGIME))
check("ACTIVE unchanged", get_target_exposure("ACTIVE")["net"] == 0.80)
check("unknown regime falls back safely",
      get_target_exposure("NONSENSE")["gross"]
      == EXPOSURE_BY_REGIME["CAUTIOUS"]["gross"])

print("\n[4] configured cadence")
from config.settings import LIVE_REBALANCE_DAYS

check("LIVE_REBALANCE_DAYS is 10", LIVE_REBALANCE_DAYS == 10)

print(f"\n{'=' * 50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
