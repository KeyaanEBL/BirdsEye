"""Range-conditioned short straddle — strategy as an importable module.

Defining strategies in .py files (instead of notebook cells) makes them
reusable across notebooks, spawn-safe for multiprocessing, and testable.
Use from a notebook:

    from strategies.range_short_straddle import RangeShortStraddle
    be = BirdsEye(..., strategy_cls=RangeShortStraddle, strategy_kwargs={"range_bps_max": 15.0})
"""

from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context


class Wait(State):
    name = "WAIT"
    transitions = {"calm_entry": "SHORT"}

    def target(self, alphas, ctx):
        atm = ctx.get("short_atm")
        if atm is None:
            return None
        return ctx["strat"].close_legs([(atm, "CE"), (atm, "PE")], reason=Reason(state="WAIT", note="square off"))


class Short(State):
    name = "SHORT"
    transitions = {"hold_elapsed": "WAIT"}

    def target(self, alphas, ctx):
        atm = alphas["atm"]
        ctx["short_atm"] = atm
        lots = ctx["strat"].lots
        return Order(name="short_straddle",
                     legs=[OrderLeg(atm, "CE", lots=lots, action="SELL"),
                           OrderLeg(atm, "PE", lots=lots, action="SELL")],
                     reason=Reason(state="SHORT"))

    def on_enter(self, ctx):
        ctx["entry_sec"] = ctx["now_sec"]


class RangeShortStraddle(StateMachineStrategy):
    states = {"WAIT": Wait(), "SHORT": Short()}
    slice_lots = 100
    pause = 0

    def __init__(self, broker, lots=1, range_win=600, decision_every=1200, hold=600, range_bps_max=15.0, session_len=23400):
        self.lots           = lots
        self.range_win      = range_win
        self.decision_every = decision_every
        self.hold           = hold
        self.range_bps_max  = range_bps_max
        self.session_len    = session_len
        self.min_history    = range_win
        
        ctx                  = Context()
        ctx["next_decision"] = decision_every
        ctx["short_atm"]     = None
        super().__init__("WAIT", broker, name="range_short_straddle", context=ctx)
        ctx["strat"]         = self

    def guard_calm_entry(self, a, c):
        return (a["decision_now"] and
                a["range_bps"] < self.range_bps_max and
                a["sec"] < self.session_len - self.hold)

    def guard_hold_elapsed(self, a, c):
        return a["sec"] - c["entry_sec"] >= self.hold

    # ---- alphas (every second) ----
    def compute_alphas(self, snap):
        c   = self.context
        sec = snap.i
        h   = snap.spot_hist(self.range_win)
        rng = (h.max() - h.min()) / snap.spot * 1e4
        dn  = sec >= c["next_decision"]
        if dn:
            c["next_decision"] += self.decision_every
        c["now_sec"] = sec
        return {"sec": sec, "spot": snap.spot, "atm": snap.atm_strike(quoted_only=True), "range_bps": rng, "decision_now": dn}