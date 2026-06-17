"""Simple range-conditioned short strangle.

Entry: every second after SKIP_OPEN, if the 15-min trailing spot range
< RANGE_BPS_MAX and valid strikes exist at least MIN_DIST_BPS from spot.

Hold for HOLD seconds, then square off and repeat.
Stop loss: exit if cost-to-close > STOP_MULT * entry premium.
"""
import numpy as np
from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context

LOTS          = 1
RANGE_WIN     = 900
HOLD          = 1800
RANGE_BPS_MAX = 10.0
MIN_DIST_BPS  = 15.0
STOP_MULT     = 2.0
SKIP_OPEN     = 3600
SESSION_LEN   = 23400


class Wait(State):
    name = "WAIT"
    transitions = {"calm_entry": "SHORT"}

    def target(self, alphas, ctx):
        ce_k = ctx.get("short_ce")
        pe_k = ctx.get("short_pe")
        if ce_k is None and pe_k is None:
            return None
        ctx["short_ce"] = None
        ctx["short_pe"] = None
        strat = ctx["strat"]
        return strat.close_legs(
            [(ce_k, "CE"), (pe_k, "PE")],
            reason      = Reason(state="WAIT", note="square off"),
            slice_lots  = strat.lots,
            pause       = 0,
        )


class Short(State):
    name = "SHORT"
    transitions = {"stop_triggered": "WAIT", "hold_elapsed": "WAIT"}

    def on_enter(self, ctx):
        ctx["entry_sec"] = ctx["now_sec"]

    def target(self, alphas, ctx):
        ce_k = alphas["ce_strike"]
        pe_k = alphas["pe_strike"]
        ctx["short_ce"]   = ce_k
        ctx["short_pe"]   = pe_k
        ctx["entry_prem"] = alphas["ce_mid"] + alphas["pe_mid"]
        lots = ctx["strat"].lots
        return Order(
            name="short_strangle",
            legs=[
                OrderLeg(ce_k, "CE", lots=lots, action="SELL",
                         slice_lots=lots, pause=0),
                OrderLeg(pe_k, "PE", lots=lots, action="SELL",
                         slice_lots=lots, pause=0),
            ],
            reason=Reason(state="SHORT"),
        )


class RangeShortStrangle(StateMachineStrategy):
    states     = {"WAIT": Wait(), "SHORT": Short()}
    # slice_lots and pause removed — now per OrderLeg

    def __init__(
        self,
        broker,
        lots          = LOTS,
        range_win     = RANGE_WIN,
        hold          = HOLD,
        range_bps_max = RANGE_BPS_MAX,
        min_dist_bps  = MIN_DIST_BPS,
        stop_mult     = STOP_MULT,
        skip_open     = SKIP_OPEN,
        session_len   = SESSION_LEN,
        # max_lots: max total lots open at once — used for churn.
        # defaults to 2 * lots (two legs of a strangle).
    ):
        self.lots          = lots
        self.range_win     = range_win
        self.hold          = hold
        self.range_bps_max = range_bps_max
        self.min_dist_bps  = min_dist_bps
        self.stop_mult     = stop_mult
        self.skip_open     = skip_open
        self.session_len   = session_len
        self.min_history   = range_win

        ctx             = Context()
        ctx["short_ce"] = None
        ctx["short_pe"] = None
        super().__init__("WAIT", broker, name="range_short_strangle", context=ctx)
        ctx["strat"] = self

    def guard_calm_entry(self, a, c):
        return (a["sec"] >= self.skip_open               and
                a["range_bps"] < self.range_bps_max       and
                a["ce_strike"] is not None               and
                a["pe_strike"] is not None               and
                a["sec"] < self.session_len - self.hold)

    def guard_hold_elapsed(self, a, c):
        return a["sec"] - c.get("entry_sec", 0) >= self.hold

    def guard_stop_triggered(self, a, c):
        ep = c.get("entry_prem")
        cp = a.get("current_prem")
        if ep is None or cp is None or ep <= 0:
            return False
        return cp > self.stop_mult * ep

    def compute_alphas(self, snap):
        c    = self.context
        sec  = snap.i
        spot = snap.spot
        c["now_sec"] = sec

        h       = snap.spot_hist(self.range_win)
        rng_bps = (h.max() - h.min()) / spot * 1e4

        strikes = snap._feed.strikes
        thr     = self.min_dist_bps

        ce_k = ce_mid = None
        for s in strikes:
            if (s - spot) / spot * 1e4 < thr:
                continue
            q = snap.quote(float(s), "CE")
            if q is None:
                continue
            ce_k, ce_mid = float(s), 0.5 * (q[0] + q[1])
            break

        pe_k = pe_mid = None
        for s in strikes[::-1]:
            if (spot - s) / spot * 1e4 < thr:
                continue
            q = snap.quote(float(s), "PE")
            if q is None:
                continue
            pe_k, pe_mid = float(s), 0.5 * (q[0] + q[1])
            break

        current_prem = None
        sce, spe = c.get("short_ce"), c.get("short_pe")
        if sce is not None and spe is not None:
            qc = snap.quote(sce, "CE")
            qp = snap.quote(spe, "PE")
            if qc is not None and qp is not None:
                current_prem = 0.5 * (qc[0] + qc[1] + qp[0] + qp[1])

        return {
            "sec":          sec,
            "spot":         spot,
            "range_bps":    rng_bps,
            "ce_strike":    ce_k,
            "pe_strike":    pe_k,
            "ce_mid":       ce_mid,
            "pe_mid":       pe_mid,
            "current_prem": current_prem,
        }