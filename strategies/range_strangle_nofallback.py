"""range_strangle_nofallback — tight-range pyramided short strangle, no VWAP.

Single regime: when the trailing `range_win`-second spot range is below
`range_bps_max`, short the two strikes that bracket spot most tightly.
Position is built in 4 pyramid groups (1+2+3+4 = 10 lots/side) at
`group_interval` (2 min) intervals. Gate to add each group: the range over
the last `group_interval` seconds is still below `range_bps_max` (market
stayed calm). If the gate fails the pyramid freezes and we hold to an exit.

When the range regime is not active the strategy stays FLAT.

Risk exits (same as vwap_skew_strangle_nofallback)
--------------------------------------------------
Anchored to MARGIN (margin_per_lot * open_lots * 2).
  * profit_take  : upnl >= profit_take_frac * margin   (default 2%)
  * stop_loss    : upnl <= -stop_loss_frac * margin     (default 1%)
  * trailing     : once up trail_arm_frac * margin, give back trail_frac * margin
  * tight_done   : time backstop (default 2 hr)
  * day_stop     : cumulative day PnL <= -day_stop_frac * total_margin (default 1%)

Absolute $ overrides: profit_target, stop_loss_abs, trail_stop.
EOD: any position still open at the last bar is squared off by the runner.
"""

import numpy as np
from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context


# --------------------------------------------------------------------------- #
# States                                                                       #
# --------------------------------------------------------------------------- #

class Wait(State):
    name = "WAIT"
    transitions = {
        "calm_range": "TIGHT_ADD",
    }

    def target(self, alphas, ctx):
        legs = ctx.get("open_legs")
        if not legs:
            return None
        return ctx["strat"].close_legs(legs, reason=Reason(state="WAIT", note="square off"))

    def on_enter(self, ctx):
        ctx["episode"]   = None
        ctx["open_legs"] = None
        ctx["group"]     = 0
        ctx["peak_upnl"] = 0.0
        ctx["tight_ce"]  = None
        ctx["tight_pe"]  = None


class TightAdd(State):
    """Places ONE pyramid group, then routes to TIGHT_HOLD."""
    name = "TIGHT_ADD"
    transitions = {"always_hold": "TIGHT_HOLD"}

    def target(self, alphas, ctx):
        strat = ctx["strat"]
        if ctx.get("group", 0) == 0:              # first group: lock strikes
            ctx["tight_ce"], ctx["tight_pe"] = alphas["t_ce"], alphas["t_pe"]
        g    = ctx.get("group", 0) + 1
        lots = strat.pyramid_schedule[g - 1]
        ce, pe = ctx["tight_ce"], ctx["tight_pe"]
        return Order(
            name="tight_pyramid",
            legs=[OrderLeg(ce, "CE", lots=lots, action="SELL", slice_lots=lots, pause=0),
                  OrderLeg(pe, "PE", lots=lots, action="SELL", slice_lots=lots, pause=0)],
            reason=Reason(state="TIGHT_ADD", note=f"tight|group{g}x{lots}"),
        )

    def on_enter(self, ctx):
        ctx["group"]   = ctx.get("group", 0) + 1
        ctx["episode"] = "tight"
        ctx["open_legs"] = [(ctx["tight_ce"], "CE"), (ctx["tight_pe"], "PE")]
        ctx["window_start_sec"]       = ctx["now_sec"]
        ctx["window_start_range_bps"] = ctx["now_range_bps"]
        if ctx["group"] == 1:
            ctx["entry_sec"] = ctx["now_sec"]
            ctx["peak_upnl"] = ctx["upnl"]


class TightHold(State):
    name = "TIGHT_HOLD"
    transitions = {
        "day_stop":     "WAIT",
        "profit_take":  "WAIT",
        "stop_loss":    "WAIT",
        "trailing":     "WAIT",
        "tight_done":   "WAIT",
        "tight_add":    "TIGHT_ADD",
        "tight_freeze": "TIGHT_FROZEN",
    }

    def target(self, alphas, ctx):
        return None


class TightFrozen(State):
    """Pyramid frozen; hold to an auto-exit."""
    name = "TIGHT_FROZEN"
    transitions = {
        "day_stop":    "WAIT",
        "profit_take": "WAIT",
        "stop_loss":   "WAIT",
        "trailing":    "WAIT",
        "tight_done":  "WAIT",
    }

    def target(self, alphas, ctx):
        return None


# --------------------------------------------------------------------------- #
# Strategy                                                                      #
# --------------------------------------------------------------------------- #

class RangeStrangleNoFallback(StateMachineStrategy):
    states = {
        "WAIT":         Wait(),
        "TIGHT_ADD":    TightAdd(),
        "TIGHT_HOLD":   TightHold(),
        "TIGHT_FROZEN": TightFrozen(),
    }

    def __init__(self, broker, max_lots=10, lots=10,
                 range_bps_max=15.0,           # trailing-range ceiling to trigger entry (bps)
                 range_win=900,                # trailing window for range alpha + warm-up (15 min)
                 pyramid_schedule=(1, 2, 3, 4),# lots/side per group; must sum to `lots`
                 group_interval=120,           # seconds between pyramid groups (2 min)
                 tight_hold=7200,              # max hold from first group (2 hr)
                 margin_per_lot=10000.0,       # margin per lot per side (for stop sizing)
                 profit_take_frac=0.02,        # profit target as fraction of margin used
                 stop_loss_frac=0.01,          # per-trade stop as fraction of margin used
                 trail_arm_frac=0.015,         # trailing arms once up this fraction of margin
                 trail_frac=0.010,             # trailing give-back as fraction of margin
                 day_stop_frac=0.01,           # daily stop as fraction of total margin
                 profit_target=None,           # absolute profit target ($); overrides frac
                 stop_loss_abs=None,           # absolute per-trade stop ($ loss); overrides frac
                 trail_stop=None,              # absolute trailing give-back ($); overrides frac
                 quote_persist=60):            # secs a strike must stay quoted to be tradable
        self.lots             = max_lots
        self.lots             = lots
        self.range_bps_max    = range_bps_max
        self.range_win        = range_win
        self.pyramid_schedule = tuple(pyramid_schedule)
        self.n_groups         = len(self.pyramid_schedule)
        self.group_interval   = group_interval
        self.tight_hold       = tight_hold
        self.margin_per_lot   = margin_per_lot
        self.profit_take_frac = profit_take_frac
        self.stop_loss_frac   = stop_loss_frac
        self.trail_arm_frac   = trail_arm_frac
        self.trail_frac       = trail_frac
        self.day_stop_frac    = day_stop_frac
        self.profit_target    = profit_target
        self.stop_loss_abs    = stop_loss_abs
        self.trail_stop       = trail_stop
        self.quote_persist    = max(1, int(quote_persist))

        assert sum(self.pyramid_schedule) == lots, \
            f"pyramid_schedule {self.pyramid_schedule} must sum to lots={lots}"

        self.min_history = range_win

        ctx = Context()
        ctx["episode"]    = None
        ctx["open_legs"]  = None
        ctx["group"]      = 0
        ctx["peak_upnl"]  = 0.0
        ctx["tight_ce"]   = None
        ctx["tight_pe"]   = None
        ctx["now_sec"]    = 0
        ctx["now_range_bps"] = 0.0
        ctx["upnl"]       = 0.0
        ctx["starting_equity"] = None
        super().__init__("WAIT", broker, name="range_strangle_nofallback", context=ctx)
        ctx["strat"] = self

    # ------------------------------------------------------------------ #
    # Margin helpers                                                        #
    # ------------------------------------------------------------------ #
    def _margin_used(self):
        pf    = self.broker.portfolio
        total = 0.0
        for pos in pf.positions.values():
            if pos.lots != 0:
                total += self.margin_per_lot * abs(pos.lots)
        return total

    def _total_day_margin(self):
        return self.margin_per_lot * self.lots * 2

    # ------------------------------------------------------------------ #
    # Strike-grid helpers (identical to vwap_skew_strangle_nofallback)    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quoted_ladder(snap):
        feed, i = snap._feed, snap.i
        strikes = feed.strikes
        ok = np.ones(len(strikes), dtype=bool)
        for key in (("ce","bid_0"), ("ce","ask_0"), ("pe","bid_0"), ("pe","ask_0")):
            arr = feed.arrays.get(key)
            if arr is None:
                return strikes
            ok &= ~np.isnan(arr[i])
        return strikes[ok] if ok.any() else strikes

    def _stable_ladder(self, snap):
        feed, i = snap._feed, snap.i
        strikes = feed.strikes
        lo      = max(0, i - self.quote_persist + 1)
        ok      = np.ones(len(strikes), dtype=bool)
        for key in (("ce","bid_0"), ("ce","ask_0"), ("pe","bid_0"), ("pe","ask_0")):
            arr = feed.arrays.get(key)
            if arr is None:
                return self._quoted_ladder(snap)
            ok &= ~np.isnan(arr[lo:i+1]).any(axis=0)
        return strikes[ok] if ok.any() else self._quoted_ladder(snap)

    def _leg_stable(self, snap, strike, opt_type):
        if strike != strike:
            return False
        ot = opt_type.lower()
        b  = snap.field_hist(strike, ot, "bid_0", self.quote_persist)
        a  = snap.field_hist(strike, ot, "ask_0", self.quote_persist)
        if b is None or a is None or len(b) == 0:
            return snap.quote(strike, opt_type) is not None
        return bool((~np.isnan(b)).all() and (~np.isnan(a)).all())

    @staticmethod
    def _bracket(ladder, spot):
        if len(ladder) == 0:
            return float("nan"), float("nan")
        above = ladder[ladder > spot]
        below = ladder[ladder < spot]
        ce = float(above[0])  if len(above) else float(ladder[-1])
        pe = float(below[-1]) if len(below) else float(ladder[0])
        return ce, pe

    # ------------------------------------------------------------------ #
    # Alphas                                                               #
    # ------------------------------------------------------------------ #
    def compute_alphas(self, snap):
        c    = self.context
        sec  = snap.i
        spot = snap.spot

        if c["starting_equity"] is None:
            c["starting_equity"] = self.broker.portfolio.equity(snap)

        h         = snap.spot_hist(self.range_win)
        range_bps = (float(h.max()) - float(h.min())) / spot * 1e4 if len(h) else 0.0

        # range over just the last group_interval (for tight-add gate)
        h_win         = snap.spot_hist(self.group_interval)
        range_win_bps = (float(h_win.max()) - float(h_win.min())) / spot * 1e4 if len(h_win) else 0.0

        ladder = self._stable_ladder(snap)
        t_ce, t_pe = self._bracket(ladder, spot)

        def _ok(ce, pe):
            return 1.0 if (self._leg_stable(snap, ce, "CE")
                           and self._leg_stable(snap, pe, "PE")) else 0.0
        t_ok = _ok(t_ce, t_pe)

        tight_legs_ok = 1.0
        if c.get("tight_ce") is not None:
            tight_legs_ok = 1.0 if (self._leg_stable(snap, c["tight_ce"], "CE")
                                    and self._leg_stable(snap, c["tight_pe"], "PE")) else 0.0

        pf     = self.broker.portfolio
        upnl   = pf.unrealized_pnl(snap)
        margin = self._margin_used()

        if c.get("episode") == "tight":
            c["peak_upnl"] = max(c.get("peak_upnl", upnl), upnl)

        risk_reason  = self._risk_reason(c, upnl, margin)
        day_pnl      = pf.equity(snap) - c["starting_equity"]
        day_stop_hit = day_pnl <= -(self.day_stop_frac * self._total_day_margin())

        c["now_sec"]       = sec
        c["now_range_bps"] = range_bps
        c["upnl"]          = upnl

        return {
            "sec": sec, "spot": spot,
            "range_bps": range_bps, "range_win_bps": range_win_bps,
            "risk_reason": risk_reason,
            "day_stop_hit": float(day_stop_hit),
            "day_pnl": day_pnl,
            "upnl": upnl, "margin": margin,
            "group": float(c.get("group", 0)),
            "t_ce": t_ce, "t_pe": t_pe, "t_ok": t_ok,
            "tight_legs_ok": tight_legs_ok,
        }

    # ------------------------------------------------------------------ #
    # Risk helper (identical logic to vwap_skew_strangle_nofallback)      #
    # ------------------------------------------------------------------ #
    def _risk_reason(self, c, upnl, margin):
        if margin <= 0.0:
            return ""
        pt = self.profit_target if self.profit_target is not None \
             else self.profit_take_frac * margin
        if upnl >= pt:
            return "profit_take"
        sl = self.stop_loss_abs if self.stop_loss_abs is not None \
             else self.stop_loss_frac * margin
        if upnl <= -sl:
            return "stop_loss"
        peak = c.get("peak_upnl", upnl)
        if peak >= self.trail_arm_frac * margin:
            give = self.trail_stop if self.trail_stop is not None \
                   else self.trail_frac * margin
            if upnl <= peak - give:
                return "trailing"
        return ""

    # ------------------------------------------------------------------ #
    # Room helper                                                          #
    # ------------------------------------------------------------------ #
    def _room_tight(self, a): return a["sec"] <= 23400 - self.tight_hold

    # ------------------------------------------------------------------ #
    # Guards                                                               #
    # ------------------------------------------------------------------ #

    def guard_day_stop(self, a, c):
        return bool(a["day_stop_hit"]) and bool(c.get("open_legs"))

    def guard_calm_range(self, a, c):
        return (not a["day_stop_hit"] and
                a["range_bps"] < self.range_bps_max and
                a["t_ok"] > 0 and self._room_tight(a))

    def guard_always_hold(self, a, c):
        return True

    def guard_tight_add(self, a, c):
        if c.get("group", 0) >= self.n_groups:
            return False
        if a["sec"] - c.get("window_start_sec", a["sec"]) < self.group_interval:
            return False
        if a["tight_legs_ok"] <= 0:
            return False
        return a["range_win_bps"] < self.range_bps_max

    def guard_tight_freeze(self, a, c):
        if c.get("group", 0) >= self.n_groups:
            return False
        if a["sec"] - c.get("window_start_sec", a["sec"]) < self.group_interval:
            return False
        return a["range_win_bps"] >= self.range_bps_max or a["tight_legs_ok"] <= 0

    def guard_tight_done(self, a, c):
        return (c.get("episode") == "tight" and
                a["sec"] - c.get("entry_sec", a["sec"]) >= self.tight_hold)

    def guard_profit_take(self, a, c): return a.get("risk_reason") == "profit_take"
    def guard_stop_loss(self, a, c):   return a.get("risk_reason") == "stop_loss"
    def guard_trailing(self, a, c):    return a.get("risk_reason") == "trailing"
