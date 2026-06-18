"""
short_gamma_t1.py — Short-Gamma strategy, TYPE-1 COMPONENT ONLY.

Type 1 (tightest): max 2 short call lots at ATM and 2 short put lots at ATM-1,
held as up to 2 "pairs" (1 CE + 1 PE each).

  ENTRY  (120-min confidence lines):
      UR120(p75) < ENTRY_UR_P75  AND  DR120(p75) < ENTRY_DR_P75
        -> short up to the cap in one shot (top-up; no slicing).

  EXIT — Level-1 breach (30-min lines):
      trigger:  UR30(p75) > BR_UR_P75  OR  DR30(p75) > BR_DR_P75
        escalate: UR30(p50) > BR_UR_P50 OR DR30(p50) > BR_DR_P50
                    -> square off ALL Type-1 pairs           ("t1_breach_full")
                  else
                    -> square off ceil(n/2) oldest pairs      ("t1_breach_half")

  EXIT — take-profit (per pair, on pair PnL as % of pair margin):
      fires only when the Type-1 ENTRY condition re-evaluated on the 30-min
      lines is NOT currently true  (i.e. not (UR30(p75)<ENTRY_UR_P75 and
      DR30(p75)<ENTRY_DR_P75)).  Disabled while TP_PCT is None.

  EXIT — stop-loss (variables only; disabled while SL_PCT is None):
      per pair; Type-1 closes BOTH legs of the pair when its PnL <= -SL_PCT*margin.

  After ANY square-off (full OR partial) a COOLDOWN_SEC cooldown blocks re-entry.
  Everything is squared off EOD_BUFFER_SEC before the last bar (so the close
  fills before the session ends; the engine's own EOD force-square is then a
  no-op backstop).

The strategy keeps its OWN per-pair sub-ledger in the Context (the broker only
sees the net position). All reporting — pair PnL attribution, enter/exit
condition stats, the strikes-vs-time plot, time-in-market — is reconstructed in
the sandbox from the returned Tradelog + perseclog, which is why every order is
stamped with a structured Reason (state / signal / note(JSON) / alphas).

No engine code is modified; this only uses the public engine API.
"""
import json
import math
import numpy as np

from engine import OrderLeg, StateMachineStrategy, Context
from strat_states.short_gamma_t1_states import Run
from short_gamma_markout import load_bundle, SignalEngine

# ---- default parameters --------------------------------------------------- #
UNIT_LOTS     = 1        # lots per pair-leg
MAX_UNITS     = 2        # Type-1 cap: 2 pairs => 2 CE lots + 2 PE lots
PE_OFFSET     = 1        # PE is at ATM - PE_OFFSET strikes

ENTRY_UR_P75  = 20.0     # bps; entry on 120-min lines
ENTRY_DR_P75  = 20.0

BR_UR_P75     = 12.0     # bps; Level-1 breach trigger on 30-min lines
BR_DR_P75     = 15.0
BR_UR_P50     = 12.0     # bps; escalate half -> full
BR_DR_P50     = 15.0

COOLDOWN_SEC  = 1800     # 30-min per-type cooldown after any square-off
EOD_BUFFER_SEC = 15      # square off / stop entering this many bars before close
                         # (large enough to fully slice out the max position 1 lot/sec)

SLICE_LOTS    = 1        # work every order 1 lot at a time ...
SLICE_PAUSE   = 1        # ... with a 1-second pause between slices

TP_PCT        = None     # take-profit as fraction of pair margin (None = off)
SL_PCT        = None     # stop-loss   as fraction of pair margin (None = off)
MARGIN_PER_LOT = 150.0   # must match BirdsEye(margin_per_lot=...) for TP/SL sizing


class ShortGammaT1(StateMachineStrategy):
    states = {"RUN": Run()}

    def __init__(self, broker, markout_dir=None, signal_engine=None, bundle=None,
                 unit_lots=UNIT_LOTS, max_units=MAX_UNITS, pe_offset=PE_OFFSET,
                 entry_ur_p75=ENTRY_UR_P75, entry_dr_p75=ENTRY_DR_P75,
                 br_ur_p75=BR_UR_P75, br_dr_p75=BR_DR_P75,
                 br_ur_p50=BR_UR_P50, br_dr_p50=BR_DR_P50,
                 cooldown_sec=COOLDOWN_SEC, eod_buffer_sec=EOD_BUFFER_SEC,
                 slice_lots=SLICE_LOTS, slice_pause=SLICE_PAUSE,
                 tp_pct=TP_PCT, sl_pct=SL_PCT, margin_per_lot=MARGIN_PER_LOT,
                 strike_spacing=None, name="ShortGammaT1"):

        # --- signal source: injected engine (tests) or built from the bundle ---
        if signal_engine is not None:
            self.sig = signal_engine
        else:
            if bundle is None:
                if markout_dir is None:
                    raise ValueError("pass markout_dir (or signal_engine / bundle).")
                bundle = load_bundle(markout_dir)
            self.sig = SignalEngine(bundle)

        self.unit_lots   = unit_lots
        self.max_units   = max_units
        self.pe_offset   = pe_offset
        self.entry_ur_p75, self.entry_dr_p75 = entry_ur_p75, entry_dr_p75
        self.br_ur_p75, self.br_dr_p75 = br_ur_p75, br_dr_p75
        self.br_ur_p50, self.br_dr_p50 = br_ur_p50, br_dr_p50
        self.cooldown_sec   = cooldown_sec
        self.eod_buffer_sec = eod_buffer_sec
        self.slice_lots     = slice_lots
        self.slice_pause    = slice_pause
        self.tp_pct, self.sl_pct = tp_pct, sl_pct
        self.margin_per_lot = margin_per_lot
        self._spacing_override = strike_spacing
        self.lot_size = broker.portfolio.lot_size

        ctx = Context()
        ctx["t1_open"]       = []      # FIFO list of open pair dicts
        ctx["t1_cd"]         = None    # cooldown-until timestamp
        ctx["t1_next_pair"]  = 0
        ctx["t1_next_batch"] = 0
        ctx["t1_plan"]       = None
        ctx["t1_inflight"]   = None

        super().__init__(initial_state="RUN", broker=broker, name=name, context=ctx)
        self.min_history = self.sig.min_history
        self.context["strat"] = self
        self._feed = None
        self._n = 0
        self._spacing = None

    # ---- per-day setup -------------------------------------------------------
    def _prepare(self, snap):
        feed = snap._feed
        if self._feed is id(feed):
            return
        self.sig.prepare(feed)
        self._feed = id(feed)
        self._n = len(feed)
        if self._spacing_override is not None:
            self._spacing = float(self._spacing_override)
        else:
            d = np.diff(np.asarray(feed.strikes, dtype=float))
            d = d[d > 0]
            self._spacing = float(np.median(d)) if len(d) else 1.0

    def _nearest_strike(self, snap, target):
        ks = snap._feed.strikes
        if len(ks) == 0:
            return None
        j = int(np.argmin(np.abs(ks - target)))
        k = float(ks[j])
        return k if abs(k - target) <= 0.5 * self._spacing + 1e-6 else None

    # ---- alpha + decision ----------------------------------------------------
    def compute_alphas(self, snap):
        self._prepare(snap)
        ctx = self.context
        ctx["now_sec"] = snap.ts
        s = self.sig.row(snap.i)

        self._decide(snap, s)

        n_open  = len(ctx["t1_open"])
        cd      = ctx["t1_cd"]
        cd_rem  = (cd - snap.ts) if (cd is not None and snap.ts < cd) else 0
        return {**s, "t1_open": float(n_open), "t1_cd_remaining": float(cd_rem)}

    def _decide(self, snap, s):
        ctx   = self.context
        ts    = snap.ts
        open_ = ctx["t1_open"]
        n     = len(open_)
        rem   = self._n - 1 - snap.i
        plan  = None

        # 1) EOD square-off (highest priority)
        if n > 0 and rem <= self.eod_buffer_sec:
            plan = self._close_pairs(snap, open_[:], "EXIT", "t1_eod",
                                     "eod_square", set_cd=False)

        # 2) breach / TP / SL while holding
        elif n > 0:
            ur75, dr75 = s["UR30_p75"], s["DR30_p75"]
            ur50, dr50 = s["UR30_p50"], s["DR30_p50"]
            breach = (ur75 > self.br_ur_p75) or (dr75 > self.br_dr_p75)   # NaN -> False
            if breach:
                full = (ur50 > self.br_ur_p50) or (dr50 > self.br_dr_p50)
                k    = n if full else math.ceil(n / 2)
                sig  = "t1_breach_full" if full else "t1_breach_half"
                plan = self._close_pairs(snap, open_[:k], "EXIT", sig, sig, set_cd=True)
            elif self.tp_pct is not None:
                # TP gate: only when the entry condition on the 30-min lines is NOT true
                gate_open = not ((s["UR30_p75"] < self.entry_ur_p75) and
                                 (s["DR30_p75"] < self.entry_dr_p75))
                if gate_open:
                    thr   = self.tp_pct * self._pair_margin()
                    hits  = [p for p in open_
                             if (pnl := self._pair_pnl(snap, p)) is not None and pnl >= thr]
                    if hits:
                        plan = self._close_pairs(snap, hits, "EXIT", "t1_tp", "t1_tp",
                                                 set_cd=True)
            if plan is None and self.sl_pct is not None:
                thr  = -self.sl_pct * self._pair_margin()
                hits = [p for p in open_
                        if (pnl := self._pair_pnl(snap, p)) is not None and pnl <= thr]
                if hits:
                    plan = self._close_pairs(snap, hits, "EXIT", "t1_sl", "t1_sl",
                                             set_cd=True)

        # 3) entry (only if nothing else this tick, capacity left, not in cooldown,
        #    and not in the EOD buffer)
        if plan is None:
            in_cd = ctx["t1_cd"] is not None and ts < ctx["t1_cd"]
            if (not in_cd) and (rem > self.eod_buffer_sec) and (n < self.max_units):
                if (s["UR120_p75"] < self.entry_ur_p75) and (s["DR120_p75"] < self.entry_dr_p75):
                    plan = self._build_entry(snap)

        ctx["t1_plan"] = plan

    # ---- order builders ------------------------------------------------------
    def _build_entry(self, snap):
        ctx     = self.context
        add     = self.max_units - len(ctx["t1_open"])
        spacing = self._spacing
        ce_k = self._nearest_strike(snap, snap.atm)
        pe_k = self._nearest_strike(snap, snap.atm - self.pe_offset * spacing)
        if ce_k is None or pe_k is None:
            return None
        ce_q = snap.mid_and_half_spread(ce_k, "CE")
        pe_q = snap.mid_and_half_spread(pe_k, "PE")
        if ce_q is None or pe_q is None:
            return None
        ce_mid, pe_mid = ce_q[0], pe_q[0]

        batch = ctx["t1_next_batch"]; ctx["t1_next_batch"] += 1
        pairs = []
        for _ in range(add):
            pid = ctx["t1_next_pair"]; ctx["t1_next_pair"] += 1
            pairs.append(dict(pair_id=pid, batch_id=batch, entry_ts=snap.ts,
                              ce_strike=ce_k, pe_strike=pe_k,
                              ce_mid=ce_mid, pe_mid=pe_mid))
        lots = add * self.unit_lots
        legs = [OrderLeg(ce_k, "CE", lots=lots, action="SELL",
                         slice_lots=self.slice_lots, pause=self.slice_pause),
                OrderLeg(pe_k, "PE", lots=lots, action="SELL",
                         slice_lots=self.slice_lots, pause=self.slice_pause)]
        note = json.dumps(dict(inst="t1", event="entry", batch=batch,
                               pairs=[p["pair_id"] for p in pairs],
                               ce=ce_k, pe=pe_k, units=add))
        return dict(legs=legs, event="t1_entry", state="ENTER", signal="t1_entry",
                    note=note, add_pairs=pairs, close_ids=[], set_cd=False)

    def _close_pairs(self, snap, pairs, state, signal, event, set_cd):
        if not pairs:
            return None
        legs = []
        for p in pairs:
            legs.append(OrderLeg(p["ce_strike"], "CE", lots=self.unit_lots,
                                 action="BUY", slice_lots=self.slice_lots,
                                 pause=self.slice_pause))
            legs.append(OrderLeg(p["pe_strike"], "PE", lots=self.unit_lots,
                                 action="BUY", slice_lots=self.slice_lots,
                                 pause=self.slice_pause))
        note = json.dumps(dict(inst="t1", event=event,
                               pairs=[p["pair_id"] for p in pairs]))
        return dict(legs=legs, event=event, state=state, signal=signal, note=note,
                    add_pairs=[], close_ids=[p["pair_id"] for p in pairs], set_cd=set_cd)

    # ---- commit (fill-complete) ---------------------------------------------
    def _commit(self, ctx):
        plan = ctx.get("t1_inflight")
        if not plan:
            return
        if plan["close_ids"]:
            cl = set(plan["close_ids"])
            ctx["t1_open"] = [p for p in ctx["t1_open"] if p["pair_id"] not in cl]
        for p in plan["add_pairs"]:
            ctx["t1_open"].append(p)
        if plan["set_cd"]:
            ctx["t1_cd"] = ctx["now_sec"] + self.cooldown_sec
        ctx["t1_inflight"] = None

    # ---- pnl helpers ---------------------------------------------------------
    def _pair_margin(self):
        return self.margin_per_lot * self.unit_lots * 2

    def _pair_pnl(self, snap, p):
        ce = snap.mid_and_half_spread(p["ce_strike"], "CE")
        pe = snap.mid_and_half_spread(p["pe_strike"], "PE")
        if ce is None or pe is None:
            return None
        # short: profit when the mid falls below entry
        pts = (p["ce_mid"] - ce[0]) + (p["pe_mid"] - pe[0])
        return pts * self.lot_size * self.unit_lots

    # ---- guard ---------------------------------------------------------------
    def guard_act(self, alphas, ctx):
        plan = ctx.get("t1_plan")
        return bool(plan and plan.get("legs"))
