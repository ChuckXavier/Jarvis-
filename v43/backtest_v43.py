"""
JARVIS V4.3 — FINAL LEVERAGE ATTEMPT
=======================================
Every bug from V4.2/V5/V6 is explicitly prevented:

BUG 1 (V4.2): ETFs didn't exist before 2006 → FIXED: Start 2010 only
BUG 2 (V4.2): Circuit breaker re-entry loop → FIXED: After CB, no leverage for 90 days
BUG 3 (V4.2): Negative cash → FIXED: Explicit cash >= 0 check on every buy
BUG 4 (V6.0): Leverage compounding → FIXED: Buy with cash only, sell before buy
BUG 5 (V5):   Inverse ETFs → FIXED: None exist in the universe
BUG 6 (V5):   3x ETFs → FIXED: Max 2x only

STRATEGY (unchanged from V4.2):
  ACTIVE (QQQ > 200-SMA): Top 5 momentum ETFs, top 3 get 2x version
  SAFETY (QQQ < 200-SMA): Top 4 crisis alpha assets by momentum
"""

import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# ── UNIVERSES ──
MOMENTUM_POOL = ["SPY","QQQ","IWM","EFA","EEM","VTV","VUG","XLK","XLF","XLV","XLE","XLI"]
LEVERAGE_MAP = {"QQQ":"QLD","SPY":"SSO","XLK":"ROM","XLF":"UYG","XLE":"DIG"}
CRISIS_POOL = ["GLD","GDX","SLV","TLT","IEF","DBMF","DBC","XLU","XLP","XLV","BIL"]
TECH = {"QQQ","VUG","XLK"}

# ── HARDCODED START: All leveraged ETFs exist by 2010 ──
EARLIEST_START = "2010-06-01"


class BacktestV43:
    def __init__(self, capital=100000):
        self.capital = capital
        self.hist = []
        self.trades = []

    def run(self, prices):
        cash = float(self.capital)
        pos = {}          # {ticker: shares}
        mode = "ACTIVE"
        holdings = []
        peak = cash
        lev_banned_until = None   # Date until which leverage is banned (CB recovery)
        sma_dn = 0
        sma_up = 0
        last_rebal = None

        # Force start date
        start = max(pd.Timestamp(EARLIEST_START), prices.index[0])
        dates = prices.loc[start:].index

        logger.info(f"V4.3: {start.date()} → {dates[-1].date()} | {len(dates)} days")

        for date in dates:
            # ── MARK TO MARKET ──
            pv = cash
            for t, q in pos.items():
                if t in prices.columns:
                    px = prices.loc[date, t]
                    if pd.notna(px):
                        pv += q * float(px)
            peak = max(peak, pv)
            self.hist.append({"date": date, "pv": pv, "cash": cash, "mode": mode,
                             "n": len(pos)})

            # ── CIRCUIT BREAKER: -25% from peak ──
            dd = (pv / peak) - 1 if peak > 0 else 0
            if dd < -0.25 and mode == "ACTIVE":
                mode = "SAFETY"
                lev_banned_until = date + pd.Timedelta(days=90)  # BUG 2 FIX
                logger.info(f"  CB at {date.date()}: DD={dd:.1%}, leverage banned until {lev_banned_until.date()}")
                cash = self._sell_all(pos, prices, date)
                peak = cash  # CRITICAL FIX: Reset peak after CB to prevent cascade
                pv = cash
                self._buy_crisis(pos, prices, date, pv, cash)
                cash = self._calc_cash(pos, prices, date, pv)
                continue

            # ── 200-DAY SMA SWITCH ──
            sma = self._sma(prices, date)
            if sma == "BELOW":
                sma_dn += 1; sma_up = 0
                if sma_dn >= 3 and mode == "ACTIVE":
                    mode = "SAFETY"
                    cash = self._sell_all(pos, prices, date)
                    self._buy_crisis(pos, prices, date, pv, cash)
                    cash = self._calc_cash(pos, prices, date, pv)
            elif sma == "ABOVE":
                sma_up += 1; sma_dn = 0
                if sma_up >= 3 and mode == "SAFETY":
                    mode = "ACTIVE"
                    cash = self._sell_all(pos, prices, date)
                    peak = cash  # CRITICAL FIX: Reset peak on mode transition
                    pv = cash
                    last_rebal = None  # force immediate rebal

            # ── SAFETY MODE: monthly crisis alpha rebalance ──
            if mode == "SAFETY":
                if last_rebal is None or (date - last_rebal).days >= 21:
                    last_rebal = date
                    target = self._crisis_momentum(prices, date)
                    if target:
                        cash = self._rebalance(pos, target, prices, date, pv, cash)
                        holdings = list(target.keys())
                continue

            # ── ACTIVE MODE: monthly momentum rebalance ──
            if last_rebal is None or (date - last_rebal).days >= 21:
                last_rebal = date

                # Can we use leverage?
                use_leverage = True
                if lev_banned_until and date < lev_banned_until:
                    use_leverage = False  # BUG 2 FIX: still in CB cooldown

                scores = self._momentum(prices, date, holdings)
                top5 = self._pick_top(scores, 5)
                target = self._build_target(top5, use_leverage)
                cash = self._rebalance(pos, target, prices, date, pv, cash)
                holdings = list(target.keys())

        return self._metrics()

    # ══════════════════════════════════════════
    # MOMENTUM SCORING
    # ══════════════════════════════════════════
    def _momentum(self, prices, date, current):
        scores = {}
        for t in MOMENTUM_POOL:
            if t not in prices.columns: continue
            s = prices[t].loc[:date].dropna()
            if len(s) < 273: continue  # 252 + 21 skip
            s = s.iloc[:-21]  # Skip last month
            r3 = (s.iloc[-1]/s.iloc[-63])-1 if len(s)>=63 else 0
            r6 = (s.iloc[-1]/s.iloc[-126])-1 if len(s)>=126 else 0
            r12 = (s.iloc[-1]/s.iloc[-252])-1 if len(s)>=252 else 0
            sc = 0.40*r3 + 0.35*r6 + 0.25*r12
            if t in current: sc += 0.02
            scores[t] = sc
        return scores

    def _crisis_momentum(self, prices, date):
        scores = {}
        for t in CRISIS_POOL:
            if t not in prices.columns: continue
            s = prices[t].loc[:date].dropna()
            if len(s) < 63: continue
            r1 = (s.iloc[-1]/s.iloc[-21])-1 if len(s)>=21 else 0
            r3 = (s.iloc[-1]/s.iloc[-63])-1 if len(s)>=63 else 0
            scores[t] = 0.60*r1 + 0.40*r3
        if not scores: return {"BIL": 1.0}
        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:4]
        total = sum(max(s, 0.001) for _,s in ranked)
        target = {t: min(max(s,0.001)/total, 0.35) for t,s in ranked}
        rem = 1.0 - sum(target.values())
        if rem > 0.02: target["BIL"] = target.get("BIL",0) + rem
        return target

    def _pick_top(self, scores, n):
        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        picked = []; tc = 0
        for t, _ in ranked:
            if len(picked) >= n: break
            if t in TECH and tc >= 2: continue  # Max 2 tech
            picked.append(t)
            if t in TECH: tc += 1
        return picked

    def _build_target(self, top5, use_leverage):
        w = 1.0 / len(top5) if top5 else 0
        target = {}
        for i, t in enumerate(top5):
            if i < 3 and use_leverage and t in LEVERAGE_MAP:
                target[LEVERAGE_MAP[t]] = w
            else:
                target[t] = w
        # Fill remainder with SPY
        total = sum(target.values())
        if total < 0.98:
            target["SPY"] = target.get("SPY", 0) + (1.0 - total)
        return target

    # ══════════════════════════════════════════
    # EXECUTION (every bug fixed)
    # ══════════════════════════════════════════
    def _sell_all(self, pos, prices, date):
        cash = 0
        for t in list(pos.keys()):
            px = prices.loc[date, t] if t in prices.columns else None
            if px is not None and pd.notna(px):
                px = float(px)
                val = pos[t] * px
                cost = abs(val) * 0.0008
                cash += val - cost
                self.trades.append({"date":date,"t":t,"side":"SELL","val":abs(val)})
        pos.clear()
        return cash

    def _rebalance(self, pos, target, prices, date, pv, cash):
        """BUG 3 FIX: Sells first. Never buys more than cash available."""
        target_val = {t: pv * w for t, w in target.items()}

        # SELL positions not in target or overweight
        for t in list(pos.keys()):
            px = prices.loc[date, t] if t in prices.columns else None
            if px is None or pd.isna(px): continue
            px = float(px)
            cur = pos[t] * px
            tgt = target_val.get(t, 0)

            if t not in target:
                # Sell entire position
                cost = abs(cur) * 0.0008
                cash += cur - cost
                self.trades.append({"date":date,"t":t,"side":"SELL","val":abs(cur)})
                del pos[t]
            elif cur > tgt * 1.05:
                # Trim overweight
                sell = cur - tgt
                shares = sell / px
                cost = sell * 0.0008
                pos[t] -= shares
                cash += sell - cost
                self.trades.append({"date":date,"t":t,"side":"SELL","val":sell})

        # BUY underweight positions — NEVER exceed available cash
        for t, tgt in sorted(target_val.items(), key=lambda x: x[1], reverse=True):
            if cash <= 100: break  # BUG 3 FIX: Stop if no cash
            px = prices.loc[date, t] if t in prices.columns else None
            if px is None or pd.isna(px): continue
            px = float(px)
            if px <= 0: continue

            cur = pos.get(t, 0) * px if t in pos else 0
            needed = tgt - cur
            if needed < pv * 0.03: continue  # Skip small adjustments

            buy = min(needed, cash - 50)  # BUG 3 FIX: Keep $50 buffer
            if buy < 100: continue

            shares = buy / px
            cost = buy * 0.0008
            if buy + cost > cash: continue  # BUG 3 FIX: Double check

            pos[t] = pos.get(t, 0) + shares
            cash -= (buy + cost)
            self.trades.append({"date":date,"t":t,"side":"BUY","val":buy})

        return max(cash, 0)  # BUG 3 FIX: Never return negative cash

    def _buy_crisis(self, pos, prices, date, pv, cash):
        target = self._crisis_momentum(prices, date)
        for t, w in target.items():
            if cash <= 100: break
            px = prices.loc[date, t] if t in prices.columns else None
            if px is None or pd.isna(px): continue
            px = float(px)
            if px <= 0: continue
            buy = min(pv * w, cash - 50)
            if buy < 100: continue
            cost = buy * 0.0008
            if buy + cost > cash: continue
            pos[t] = buy / px
            cash -= (buy + cost)
        return cash

    def _calc_cash(self, pos, prices, date, pv):
        invested = sum(pos.get(t,0) * float(prices.loc[date,t])
                      for t in pos if t in prices.columns and pd.notna(prices.loc[date,t]))
        return max(pv - invested, 0)

    # ══════════════════════════════════════════
    # SMA
    # ══════════════════════════════════════════
    def _sma(self, prices, date):
        if "QQQ" not in prices.columns: return "ABOVE"
        q = prices["QQQ"].loc[:date].dropna()
        if len(q) < 200: return "ABOVE"
        sma = float(q.rolling(200).mean().iloc[-1])
        p = float(q.iloc[-1])
        if p > sma * 1.02: return "ABOVE"
        elif p < sma * 0.98: return "BELOW"
        return "BUFFER"

    # ══════════════════════════════════════════
    # METRICS
    # ══════════════════════════════════════════
    def _metrics(self):
        if not self.hist: return {}
        df = pd.DataFrame(self.hist).set_index("date")
        v = df["pv"]
        if len(v) < 2: return {}
        dr = v.pct_change().dropna()
        tr = (float(v.iloc[-1])/float(v.iloc[0]))-1
        y = (v.index[-1]-v.index[0]).days/365.25
        ar = (1+tr)**(1/y)-1 if y>0 else 0
        vol = dr.std()*np.sqrt(252)
        sh = (ar-0.04)/vol if vol>0 else 0
        ds = dr[dr<0]
        dsv = ds.std()*np.sqrt(252) if len(ds)>0 else vol
        so = (ar-0.04)/dsv if dsv>0 else 0
        cm = v.cummax(); mdd = ((v/cm)-1).min()
        cal = abs(ar/mdd) if mdd!=0 else 0
        mo = v.resample("ME").last().pct_change().dropna()
        by = v.resample("YE").last().pct_change().dropna()
        avg_cash = df["cash"].mean() / df["pv"].mean() if df["pv"].mean() > 0 else 0
        active = (df["mode"]=="ACTIVE").mean()
        return {
            "total_return":tr,"annualized_return":ar,"annualized_volatility":vol,
            "sharpe_ratio":sh,"sortino_ratio":so,"max_drawdown":mdd,"calmar_ratio":cal,
            "win_rate_monthly":(mo>0).mean(),
            "best_month":mo.max() if not mo.empty else 0,
            "worst_month":mo.min() if not mo.empty else 0,
            "best_year":by.max() if not by.empty else 0,
            "worst_year":by.min() if not by.empty else 0,
            "total_trades":len(self.trades),
            "total_costs":sum(t.get("val",0)*0.0008 for t in self.trades),
            "years":y,"start_value":float(v.iloc[0]),"end_value":float(v.iloc[-1]),
            "avg_cash_pct":avg_cash,"pct_active":active,
        }

