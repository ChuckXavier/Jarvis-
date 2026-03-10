"""
JARVIS V6.1 - Volatility-Aware Momentum Engine (LEVERAGE FIX)
================================================================
V6.0 had a compounding bug: $100K → $807B (leverage on leverage).
V6.1 FIX: Sells before buys. Never spends more cash than available.
Leveraged ETFs are treated as regular positions sized to portfolio value,
not as synthetic leverage that multiplies the portfolio itself.

The 2x leverage comes from the ETF INTERNALLY (QLD holds 2x QQQ swaps).
Our position SIZE is still bounded by actual cash. We buy $25K of QLD,
which gives us $50K of QQQ exposure. If QLD goes up 20%, our position
is worth $30K — we made $5K, not $50K.
"""

import pandas as pd
import numpy as np
from loguru import logger

TIER1_MOMENTUM = ["QQQ","SPY","SMH","SOXX","VGT","XLK","XLV","XLE","GLD","TLT","IEF","DBMF"]
TIER2_LEVERAGE = {"QQQ":"QLD","SPY":"SSO","SMH":"USD","SOXX":"USD","VGT":"ROM","XLK":"ROM"}
SAFETY = {"TLT":0.30,"IEF":0.30,"GLD":0.20,"BIL":0.10,"DBMF":0.05,"SHY":0.05}

CFG = {
    "initial_capital":100000,
    "momentum_weights":[0.40,0.35,0.25],
    "top_n":4,
    "continuity_bonus":0.02,
    "max_sector_pct":0.60,
    "drift_threshold":0.05,
    "sma_buffer":0.03,
    "sma_confirm":2,
    "vol_entry_confirm":3,
    "vol_exit_confirm":5,
    "min_lev_hold":10,
    "max_lev_hold":60,
    "circuit_breaker":-0.25,
    "cb_cooldown":30,
    "cost_bps":8,
}

class BacktestV6:
    def __init__(self, config=None):
        self.cfg = {**CFG, **(config or {})}
        self.hist = []
        self.trades = []
        self.states = []

    def run(self, prices, vix=None):
        if prices.empty: return {}
        start = prices.index[max(0, len(prices)//4)]  # Use last 75% for backtest
        # Ensure we have enough SMA history
        if "QQQ" in prices.columns:
            qqq = prices["QQQ"].dropna()
            if len(qqq) > 300:
                start = qqq.index[300]
        end = prices.index[-1]

        logger.info(f"V6.1 Backtest: {start.date()} → {end.date()}")

        cash = float(self.cfg["initial_capital"])
        positions = {}  # {ticker: qty}
        mode = "ACTIVE"
        lev_on = False
        lev_level = 1.0
        vstreak = 0
        vexit = 0
        sma_up = 0
        sma_dn = 0
        cb_active = False
        cb_date = None
        last_mom = None
        top4 = []
        lev_date = None
        peak = cash

        for date in prices.loc[start:end].index:
            # Mark to market
            pv = cash
            for t, qty in positions.items():
                if t in prices.columns:
                    px = prices.loc[date, t]
                    if pd.notna(px): pv += qty * px

            self.hist.append({"date":date, "total_value":pv, "cash":cash,
                             "mode":mode, "leverage":lev_level, "n_pos":len(positions)})
            peak = max(peak, pv)

            # Circuit breaker
            dd = (pv/peak)-1 if peak>0 else 0
            if dd <= self.cfg["circuit_breaker"] and not cb_active:
                cb_active = True; cb_date = date
                mode = "SAFETY"; lev_on = False; lev_level = 1.0
                cash = self._sell_all(positions, prices, date)
                cash = self._buy_safety(positions, prices, date, pv, cash)
                continue
            if cb_active:
                if cb_date and (date-cb_date).days >= self.cfg["cb_cooldown"]:
                    if self._sma_ok(prices, date): cb_active = False
                    else: continue
                else: continue

            # PILLAR 3: SMA
            sma_st = self._sma_status(prices, date)
            if sma_st == "BELOW":
                sma_dn += 1; sma_up = 0
                if sma_dn >= self.cfg["sma_confirm"] and mode == "ACTIVE":
                    mode = "SAFETY"; lev_on = False; lev_level = 1.0
                    cash = self._sell_all(positions, prices, date)
                    cash = self._buy_safety(positions, prices, date, pv, cash)
            elif sma_st == "ABOVE":
                sma_up += 1; sma_dn = 0
                if sma_up >= self.cfg["sma_confirm"] and mode == "SAFETY":
                    mode = "ACTIVE"
                    cash = self._sell_all(positions, prices, date)
                    last_mom = None
            else:  # BUFFER
                lev_on = False; lev_level = 1.0

            if mode == "SAFETY": continue

            # PILLAR 1: Monthly momentum
            new_month = last_mom is None or date.month != last_mom.month
            if new_month and mode == "ACTIVE":
                last_mom = date
                new4 = self._momentum(prices, date, top4)
                if new4 != top4 or not positions:
                    target = self._build_target(new4, lev_on, lev_level)
                    cash = self._rebalance(positions, target, prices, date, pv, cash)
                    top4 = new4

            # PILLAR 2: Vol filter (daily)
            if mode == "ACTIVE" and vix is not None:
                vs = self._vol_filter(prices, vix, date)
                if vs > 0:
                    vstreak += 1; vexit = 0
                else:
                    vexit += 1; vstreak = 0

                # Entry
                if vstreak >= self.cfg["vol_entry_confirm"] and not lev_on and sma_st != "BUFFER":
                    lev_level = 2.0 if vs >= 0.05 else 1.5
                    lev_on = True; lev_date = date
                    if top4:
                        target = self._build_target(top4, True, lev_level)
                        cash = self._rebalance(positions, target, prices, date, pv, cash)

                # Exit
                if vexit >= self.cfg["vol_exit_confirm"] and lev_on:
                    if lev_date and (date-lev_date).days >= self.cfg["min_lev_hold"]:
                        lev_on = False; lev_level = 1.0
                        if top4:
                            target = self._build_target(top4, False, 1.0)
                            cash = self._rebalance(positions, target, prices, date, pv, cash)

                if lev_on and lev_date and (date-lev_date).days >= self.cfg["max_lev_hold"]:
                    lev_on = False; lev_level = 1.0

            self.states.append({"date":date,"mode":mode,"lev_on":lev_on,"lev":lev_level,"sma":sma_st})

        return {"portfolio_history": pd.DataFrame(self.hist),
                "trade_log": self.trades, "state_log": self.states,
                "metrics": self._metrics()}

    def _build_target(self, top4, lev_on, lev_level):
        """Build target allocation. Each ETF gets 25% of portfolio."""
        target = {}
        sorted_top = list(top4)
        for i, t in enumerate(sorted_top):
            if i < 2 and lev_on and t in TIER2_LEVERAGE and lev_level >= 1.5:
                target[TIER2_LEVERAGE[t]] = 0.25
            else:
                target[t] = 0.25
        return target

    def _rebalance(self, positions, target, prices, date, pv, cash):
        """
        V6.1 CRITICAL FIX: Proper rebalance that respects cash constraints.
        1. Compute target dollar amounts based on current pv
        2. Sell positions we don't want (increases cash)
        3. Buy positions we want (decreases cash)
        4. Never buy more than available cash
        """
        cost_bps = self.cfg["cost_bps"]

        # What we currently hold
        current = {}
        for t, qty in positions.items():
            if t in prices.columns:
                px = prices.loc[date, t]
                if pd.notna(px): current[t] = qty * px

        # Target dollar amounts
        target_dollars = {t: pv * w for t, w in target.items()}

        # SELLS first
        for t in list(positions.keys()):
            if t not in target:
                # Sell entire position
                if t in prices.columns:
                    px = prices.loc[date, t]
                    if pd.notna(px) and positions[t] != 0:
                        proceeds = positions[t] * px
                        cost = abs(proceeds) * (cost_bps / 10000)
                        cash += proceeds - cost
                        self.trades.append({"date":date,"ticker":t,"side":"SELL","value":abs(proceeds),"cost":cost})
                        del positions[t]
            else:
                # Partial sell if overweight
                current_val = current.get(t, 0)
                target_val = target_dollars.get(t, 0)
                if current_val > target_val * (1 + self.cfg["drift_threshold"]):
                    sell_val = current_val - target_val
                    px = prices.loc[date, t]
                    if pd.notna(px) and px > 0:
                        sell_shares = sell_val / px
                        cost = sell_val * (cost_bps / 10000)
                        positions[t] -= sell_shares
                        cash += sell_val - cost
                        self.trades.append({"date":date,"ticker":t,"side":"SELL","value":sell_val,"cost":cost})

        # BUYS (with cash constraint)
        for t, target_val in sorted(target_dollars.items(), key=lambda x: x[1], reverse=True):
            current_val = 0
            if t in positions and t in prices.columns:
                px = prices.loc[date, t]
                if pd.notna(px): current_val = positions[t] * px

            needed = target_val - current_val
            if needed < pv * self.cfg["drift_threshold"]:
                continue  # Drift too small

            px = prices.loc[date, t] if t in prices.columns else None
            if px is None or pd.isna(px) or px <= 0:
                continue

            # V6.1 FIX: Never buy more than available cash
            buy_val = min(needed, cash * 0.98)  # Keep 2% cash buffer
            if buy_val < 100:  # Minimum trade size
                continue

            shares = buy_val / px
            cost = buy_val * (cost_bps / 10000)

            if t not in positions:
                positions[t] = 0
            positions[t] += shares
            cash -= buy_val + cost
            self.trades.append({"date":date,"ticker":t,"side":"BUY","value":buy_val,"cost":cost})

        return cash

    def _sell_all(self, positions, prices, date):
        cash = 0
        cost_bps = self.cfg["cost_bps"]
        for t in list(positions.keys()):
            if t in prices.columns:
                px = prices.loc[date, t]
                if pd.notna(px):
                    val = positions[t] * px
                    cost = abs(val) * (cost_bps/10000)
                    cash += val - cost
                    self.trades.append({"date":date,"ticker":t,"side":"SELL","value":abs(val),"cost":cost})
        positions.clear()
        return cash

    def _buy_safety(self, positions, prices, date, pv, cash):
        cost_bps = self.cfg["cost_bps"]
        for t, w in SAFETY.items():
            if t in prices.columns:
                px = prices.loc[date, t]
                if pd.notna(px) and px > 0:
                    buy_val = min(pv * w, cash * 0.90)
                    if buy_val < 50: continue
                    shares = buy_val / px
                    cost = buy_val * (cost_bps/10000)
                    positions[t] = shares
                    cash -= buy_val + cost
        return cash

    def _sma_ok(self, prices, date):
        if "QQQ" not in prices.columns: return True
        q = prices["QQQ"].loc[:date].dropna()
        return len(q)>=200 and float(q.iloc[-1]) > float(q.rolling(200).mean().iloc[-1])

    def _sma_status(self, prices, date):
        if "QQQ" not in prices.columns: return "ABOVE"
        q = prices["QQQ"].loc[:date].dropna()
        if len(q)<200: return "ABOVE"
        sma = float(q.rolling(200).mean().iloc[-1])
        p = float(q.iloc[-1]); b = self.cfg["sma_buffer"]
        if p > sma*(1+b): return "ABOVE"
        elif p < sma*(1-b): return "BELOW"
        return "BUFFER"

    def _vol_filter(self, prices, vix, date):
        if vix is None or "SPY" not in prices.columns: return 0
        vb = vix.loc[:date].dropna()
        if len(vb)<20: return 0
        iv = float(vb.tail(20).mean())/100
        spy = prices["SPY"].loc[:date].dropna()
        if len(spy)<11: return 0
        rv = float(spy.pct_change().tail(10).std())*np.sqrt(252)
        s = iv-rv
        return s if s>0 else -abs(s)

    def _momentum(self, prices, date, current):
        w3,w6,w12 = self.cfg["momentum_weights"]
        scores = {}
        tech = {"QQQ","SMH","SOXX","VGT","XLK"}
        for t in TIER1_MOMENTUM:
            if t not in prices.columns: continue
            s = prices[t].loc[:date].dropna()
            if len(s)<252: continue
            r3 = (float(s.iloc[-1])/float(s.iloc[-63]))-1
            r6 = (float(s.iloc[-1])/float(s.iloc[-126]))-1 if len(s)>=126 else 0
            r12 = (float(s.iloc[-1])/float(s.iloc[-252]))-1
            sc = w3*r3+w6*r6+w12*r12
            if t in current: sc += self.cfg["continuity_bonus"]
            scores[t] = sc

        ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        top = []; tc = 0
        mx = int(self.cfg["top_n"]*self.cfg["max_sector_pct"])
        for t,_ in ranked:
            if len(top)>=self.cfg["top_n"]: break
            if t in tech and tc>=mx: continue
            top.append(t)
            if t in tech: tc+=1
        return top

    def _metrics(self):
        if not self.hist: return {}
        df = pd.DataFrame(self.hist).set_index("date")
        v = df["total_value"]
        if len(v)<2: return {}
        dr = v.pct_change().dropna()
        tr = (v.iloc[-1]/v.iloc[0])-1
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
        wr = (mo>0).mean()
        by = v.resample("YE").last().pct_change().dropna()
        ap = sum(1 for s in self.states if s["mode"]=="ACTIVE")/max(len(self.states),1)
        lp = sum(1 for s in self.states if s.get("lev_on"))/max(len(self.states),1)
        return {
            "total_return":tr,"annualized_return":ar,"annualized_volatility":vol,
            "sharpe_ratio":sh,"sortino_ratio":so,"max_drawdown":mdd,"calmar_ratio":cal,
            "win_rate_monthly":wr,
            "profit_factor":abs(mo[mo>0].mean()/mo[mo<0].mean()) if (mo<0).any() and (mo>0).any() else 0,
            "best_month":mo.max() if not mo.empty else 0,
            "worst_month":mo.min() if not mo.empty else 0,
            "best_year":by.max() if not by.empty else 0,
            "worst_year":by.min() if not by.empty else 0,
            "total_trades":len(self.trades),
            "total_costs":sum(t.get("cost",0) for t in self.trades),
            "years":y,"start_value":v.iloc[0],"end_value":v.iloc[-1],
            "pct_active_mode":ap,"pct_leveraged":lp,
        }
