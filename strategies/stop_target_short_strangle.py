"""Stop/target short strangle — a re-entering short-gamma strategy with bracket exits.

Purpose: a FRAMEWORK STRESS TEST, not an alpha. It is built to exercise as many
engine paths as possible — SLTP brackets, transition priority, the execution
slicer, the cost model, and the re-entry lifecycle. Expect negative PnL; that is
fine and not the point.

Lifecycle (FSM):

    WAIT    --[entry_window]----> SHORT       sell OTM strangle (atm +/- width steps)
    SHORT   --[stop_hit]--------> FLATTEN      priority 1   (open PnL <= -stop_pct * credit)
    SHORT   --[target_hit]------> FLATTEN      priority 2   (open PnL >= +tp_pct  * credit)
    SHORT   --[eod_squareoff]---> FLATTEN      priority 3   (hard square-off near close)
    SHORT   --[time_stop]-------> FLATTEN      priority 4   (max holding time elapsed)
    FLATTEN --[always]----------> WAIT         re-arm for the next cadence slot

What it tests, by design:
  - SLTP / bracket    : stop & target are separate NAMED guards in one state, so the
                        Tradelog `signal` column attributes every exit cleanly.
  - portfolio marking : both bracket guards read `portfolio.unrealized_pnl(snap)`
                        (mid-marking) vs. an entry credit taken from `avg_entry`.
  - transition order  : four competing exits in SHORT, first-hit-wins on the same tick.
  - execution slicer  : lots > slice_lots makes each order take several ticks; no user
                        transition fires mid-EXECUTING.
  - cost / churn       : tight brackets + re-entry => many round-trips => spread charged
                        twice per cycle + txn cost; stresses churn_per_day & frictions.
  - non-ATM strikes    : legs at atm +/- width steps on the discovered grid.
"""

import numpy as np

from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context


# ---- states ----------------------------------------------------------------

class Wait(State):
    name = "WAIT"
    transitions = {"entry_window": "SHORT"}

    def target(self, alphas, ctx):
        return None                                  # flat; nothing to hold


class Short(State):
    name = "SHORT"
    # priority order matters: stop, then target, then forced EOD, then time.
    transitions = {
        "stop_hit":      "FLATTEN",
        "target_hit":    "FLATTEN",
        "eod_squareoff": "FLATTEN",
        "time_stop":     "FLATTEN",
    }

    def target(self, alphas, ctx):
        lots   = ctx["strat"].lots
        ce_k   = alphas["ce_strike"]
        pe_k   = alphas["pe_strike"]
        ctx["legs"] = [(ce_k, "CE"), (pe_k, "PE")]   # remember what to flatten
        return Order(name="short_strangle",
                     legs=[OrderLeg(ce_k, "CE", lots=lots, action="SELL"),
                           OrderLeg(pe_k, "PE", lots=lots, action="SELL")],
                     reason=Reason(state="SHORT", note=f"strangle {pe_k}/{ce_k}"))

    def on_enter(self, ctx):                          # runs at FILL-COMPLETE
        ctx["entry_sec"] = ctx["now_sec"]
        # entry credit ($) from the freshly-filled book: avg_entry is the price received.
        pf, credit = ctx["strat"].broker.portfolio, 0.0
        for key in ctx["legs"]:
            pos = pf.positions.get(key)
            if pos is not None and pos.lots != 0:
                credit += pos.avg_entry * abs(pos.lots) * pf.lot_size
        ctx["entry_credit"] = credit


class Flatten(State):
    name = "FLATTEN"
    transitions = {"always": "WAIT"}

    def target(self, alphas, ctx):
        legs = ctx.get("legs") or []
        return ctx["strat"].close_legs(legs, reason=Reason(state="FLATTEN", note="square off"))

    def on_enter(self, ctx):
        ctx["legs"]         = []
        ctx["entry_credit"] = 0.0


# ---- strategy --------------------------------------------------------------

class StopTargetShortStrangle(StateMachineStrategy):
    states     = {"WAIT": Wait(), "SHORT": Short(), "FLATTEN": Flatten()}
    slice_lots = 5                                    # < lots, so orders slice over ticks
    pause      = 2

    def __init__(self, broker, lots=5, width_steps=3,
                 stop_pct=1.0, tp_pct=0.5,
                 decision_every=1800, hold_max=2400,
                 warmup=300, eod_sec=23100):
        self.lots           = lots
        self.width_steps    = width_steps             # strikes away from ATM per leg
        self.stop_pct       = stop_pct                # stop when loss  >= stop_pct * credit
        self.tp_pct         = tp_pct                  # take when gain  >= tp_pct  * credit
        self.decision_every = decision_every
        self.hold_max       = hold_max
        self.eod_sec        = eod_sec
        self.min_history    = warmup

        ctx = Context()
        ctx["next_decision"] = 0
        ctx["legs"]          = []
        ctx["entry_credit"]  = 0.0
        super().__init__("WAIT", broker, name="stop_target_short_strangle", context=ctx)
        ctx["strat"] = self

    # ---- named guards (ledger-visible) ----
    def guard_entry_window(self, a, c):
        return (a["decision_now"] and a["sec"] < self.eod_sec
                and a["legs_quoted"])                 # both chosen strikes have quotes

    def guard_stop_hit(self, a, c):
        cr = c.get("entry_credit", 0.0)
        return cr > 0 and a["open_pnl"] <= -self.stop_pct * cr

    def guard_target_hit(self, a, c):
        cr = c.get("entry_credit", 0.0)
        return cr > 0 and a["open_pnl"] >= self.tp_pct * cr

    def guard_eod_squareoff(self, a, c):
        return a["sec"] >= self.eod_sec

    def guard_time_stop(self, a, c):
        return a["sec"] - c["entry_sec"] >= self.hold_max

    def guard_always(self, a, c):
        return True

    # ---- alphas (every second) ----
    def _strangle_strikes(self, snap):
        strikes = snap._feed.strikes                  # sorted grid (shared array)
        atm     = snap.atm_strike(quoted_only=True)
        j       = int(np.argmin(np.abs(strikes - atm)))
        jc      = min(len(strikes) - 1, j + self.width_steps)
        jp      = max(0, j - self.width_steps)
        return float(strikes[jc]), float(strikes[jp])

    def compute_alphas(self, snap):
        c   = self.context
        sec = snap.i
        c["now_sec"] = sec

        ce_k, pe_k = self._strangle_strikes(snap)
        legs_quoted = (snap.quote(ce_k, "CE") is not None
                       and snap.quote(pe_k, "PE") is not None)

        decide = sec >= c["next_decision"]
        if decide:
            c["next_decision"] += self.decision_every

        open_pnl = self.broker.portfolio.unrealized_pnl(snap)   # mid-marked open PnL ($)

        return {"sec": sec, "spot": snap.spot,
                "ce_strike": ce_k, "pe_strike": pe_k,
                "legs_quoted": legs_quoted,
                "open_pnl": open_pnl,
                "entry_credit": c.get("entry_credit", 0.0),
                "decision_now": decide}
