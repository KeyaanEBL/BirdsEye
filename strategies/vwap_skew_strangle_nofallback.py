"""vwap_skew_strangle_nofallback — a three-regime short-options strategy.

When none of the three regimes apply the strategy stays FLAT.

The three regimes (evaluated top-down each second; first match wins)
-------------------------------------------------------------------
1. spot >= VWAP + dev_bps_thresh (+20 bps) -> SKEW-UP:
   short PUT far (ATM-3) + short CALL near (ATM+1).
   Built in 4 pyramid groups (1+2+3+4 = 10 lots/side) at group_interval (2 min)
   intervals. Gate to add the next group: spot is still >= the spot at the
   window start (i.e. the move that triggered us is still intact).
2. spot <= VWAP - dev_bps_thresh (-20 bps) -> SKEW-DOWN:
   Mirror: short CALL far (ATM+3) + short PUT near (ATM-1). Same pyramid rules;
   gate: spot still <= window-start spot.
3. else, trailing range_win range < range_bps_max (15 bps) -> TIGHT:
   Same 4-group pyramid at group_interval (2 min) intervals.
   Gate to add the next group: the spot range over that window is still < range_bps_max
   (market stayed calm). Strikes are the two that bracket spot most tightly.

Pyramid freeze: if any window's gate fails (spot reversed / market woke up),
no more groups are added. We hold what we have to the relevant exits.

Risk exits (every short position — skew AND tight)
--------------------------------------------------
Anchored to MARGIN (margin_per_lot * open_lots * 2 * lot_size), not credit.
  * profit_take  : upnl >= profit_take_frac * margin   (default 2%)
  * stop_loss    : upnl <= -stop_loss_frac * margin     (default 1%)
  * trailing     : once up trail_arm_frac * margin, give back trail_frac * margin
  * skew_maxhold / tight_done: time backstops
  * day_stop     : if cumulative day PnL <= -day_stop_frac * total_margin
                   (default 1%), block ALL new entries and exit any open position.

Absolute $ overrides: profit_target, stop_loss_abs, trail_stop bypass the fractions.

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
        "above_vwap": "SKEW_ADD",
        "below_vwap": "SKEW_ADD",
        "calm_range": "TIGHT_ADD",
    }

    def target(self, alphas, ctx):
        legs = ctx.get("open_legs")
        if not legs:
            return None
        return ctx["strat"].close_legs(legs, reason=Reason(state="WAIT", note="square off"))

    def on_enter(self, ctx):
        ctx["episode"]    = None
        ctx["open_legs"]  = None
        ctx["group"]      = 0
        ctx["credit"]     = 0.0
        ctx["peak_upnl"]  = 0.0
        ctx["skew_ce"]    = None
        ctx["skew_pe"]    = None
        ctx["tight_ce"]   = None
        ctx["tight_pe"]   = None


# ---- SKEW states -----------------------------------------------------------

class SkewAdd(State):
    """Places ONE pyramid group for the skew regime, then routes to SKEW_HOLD."""
    name = "SKEW_ADD"
    transitions = {"always_hold": "SKEW_HOLD"}

    def target(self, alphas, ctx):
        strat = ctx["strat"]
        if ctx.get("group", 0) == 0:                  # first group: lock side & strikes
            if alphas["dev_bps"] >= 0:
                ce, pe, side = alphas["c1_ce"], alphas["c1_pe"], "above"
            else:
                ce, pe, side = alphas["c2_ce"], alphas["c2_pe"], "below"
            ctx["skew_ce"], ctx["skew_pe"], ctx["skew_side"] = ce, pe, side
        g    = ctx.get("group", 0) + 1
        lots = strat.pyramid_schedule[g - 1]
        ce, pe = ctx["skew_ce"], ctx["skew_pe"]
        return Order(
            name="skew_pyramid",
            legs=[OrderLeg(ce, "CE", lots=lots, action="SELL", slice_lots=lots, pause=0),
                  OrderLeg(pe, "PE", lots=lots, action="SELL", slice_lots=lots, pause=0)],
            reason=Reason(state="SKEW_ADD", note=f"{ctx['skew_side']}|group{g}x{lots}"),
        )

    def on_enter(self, ctx):
        ctx["group"]   = ctx.get("group", 0) + 1
        ctx["episode"] = "skew"
        ctx["open_legs"] = [(ctx["skew_ce"], "CE"), (ctx["skew_pe"], "PE")]
        # start a fresh direction-gate window
        ctx["window_start_sec"]  = ctx["now_sec"]
        ctx["window_start_spot"] = ctx["now_spot"]     # gate: spot direction
        if ctx["group"] == 1:
            ctx["entry_sec"]  = ctx["now_sec"]
            ctx["peak_upnl"]  = ctx["upnl"]


class SkewHold(State):
    name = "SKEW_HOLD"
    transitions = {
        "day_stop":      "WAIT",
        "profit_take":   "WAIT",
        "stop_loss":     "WAIT",
        "trailing":      "WAIT",
        "skew_maxhold":  "WAIT",
        "skew_add":      "SKEW_ADD",
        "skew_freeze":   "SKEW_FROZEN",
    }

    def target(self, alphas, ctx):
        return None


class SkewFrozen(State):
    """Pyramid frozen; hold to an auto-exit."""
    name = "SKEW_FROZEN"
    transitions = {
        "day_stop":    "WAIT",
        "profit_take": "WAIT",
        "stop_loss":   "WAIT",
        "trailing":    "WAIT",
        "skew_maxhold":"WAIT",
    }

    def target(self, alphas, ctx):
        return None


# ---- TIGHT states ----------------------------------------------------------

class TightAdd(State):
    """Places ONE pyramid group for the tight-strangle regime, then routes to TIGHT_HOLD."""
    name = "TIGHT_ADD"
    transitions = {"always_hold": "TIGHT_HOLD"}

    def target(self, alphas, ctx):
        strat = ctx["strat"]
        if ctx.get("group", 0) == 0:                  # first group: lock strikes
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
        # start a fresh range-gate window
        ctx["window_start_sec"]        = ctx["now_sec"]
        ctx["window_start_range_bps"]  = ctx["now_range_bps"]  # gate: range stayed calm
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
    """Tight pyramid frozen; hold to an auto-exit."""
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

class VwapSkewStrangleNoFallback(StateMachineStrategy):
    states = {
        "WAIT":         Wait(),
        "SKEW_ADD":     SkewAdd(),
        "SKEW_HOLD":    SkewHold(),
        "SKEW_FROZEN":  SkewFrozen(),
        "TIGHT_ADD":    TightAdd(),
        "TIGHT_HOLD":   TightHold(),
        "TIGHT_FROZEN": TightFrozen(),
    }

    def __init__(self, broker, max_lots=10, lots=10,
                 dev_bps_thresh=20.0,          # |spot-VWAP| trigger for regimes 1/2 (bps)
                 range_bps_max=15.0,           # trailing-range ceiling for regime 3 (bps)
                 range_win=900,                # trailing window for the range alpha (15 min)
                 pyramid_schedule=(1, 2, 3, 4),# lots/side per group; must sum to `lots`
                 group_interval=120,           # seconds between pyramid groups (2 min)
                 skew_max_hold=7200,           # skew max hold from first slice (2 hr)
                 tight_hold=7200,              # tight max hold from first slice (2 hr)
                 margin_per_lot=10000.0,       # margin per lot per side (used for stop sizing)
                 profit_take_frac=0.02,        # profit target as fraction of margin used
                 stop_loss_frac=0.01,          # per-trade stop loss as fraction of margin used
                 trail_arm_frac=0.015,         # trailing arms once up this fraction of margin
                 trail_frac=0.010,             # trailing give-back as fraction of margin
                 day_stop_frac=0.01,           # daily stop loss as fraction of total margin
                 profit_target=None,           # absolute profit target ($); overrides frac
                 stop_loss_abs=None,           # absolute per-trade stop ($ loss); overrides frac
                 trail_stop=None,              # absolute trailing give-back ($); overrides frac
                 vwap_use_volume=True,
                 volume_field="volume",
                 volume_is_cumulative=False,
                 quote_persist=60):
        self.max_lots         = max_lots,
        self.lots             = lots
        self.dev_bps_thresh   = dev_bps_thresh
        self.range_bps_max    = range_bps_max
        self.range_win        = range_win
        self.pyramid_schedule = tuple(pyramid_schedule)
        self.n_groups         = len(self.pyramid_schedule)
        self.group_interval   = group_interval
        self.skew_max_hold    = skew_max_hold
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
        self.vwap_use_volume      = vwap_use_volume
        self.volume_field         = volume_field
        self.volume_is_cumulative = volume_is_cumulative
        self.quote_persist        = max(1, int(quote_persist))

        assert sum(self.pyramid_schedule) == lots, \
            f"pyramid_schedule {self.pyramid_schedule} must sum to lots={lots}"

        self.min_history = range_win

        ctx = Context()
        ctx["episode"]    = None
        ctx["open_legs"]  = None
        ctx["group"]      = 0
        ctx["credit"]     = 0.0
        ctx["peak_upnl"]  = 0.0
        ctx["skew_ce"]    = None
        ctx["skew_pe"]    = None
        ctx["tight_ce"]   = None
        ctx["tight_pe"]   = None
        ctx["now_sec"]    = 0
        ctx["now_spot"]   = 0.0
        ctx["now_range_bps"] = 0.0
        ctx["upnl"]       = 0.0
        ctx["starting_equity"] = None      # set on first compute_alphas call
        super().__init__("WAIT", broker, name="vwap_skew_strangle_nofallback", context=ctx)
        ctx["strat"] = self

    # ------------------------------------------------------------------ #
    # Margin helper                                                        #
    # ------------------------------------------------------------------ #
    def _margin_used(self):
        """Total margin currently posted: margin_per_lot * |open lots| summed
        over all open legs. Lot size is already in margin_per_lot convention
        (margin_per_lot is per contract, not per share)."""
        pf    = self.broker.portfolio
        total = 0.0
        for pos in pf.positions.values():
            if pos.lots != 0:
                total += self.margin_per_lot * abs(pos.lots)
        return total

    def _total_day_margin(self):
        """Maximum possible margin at full deployment: both sides, all lots."""
        return self.margin_per_lot * self.lots * 2

    # ------------------------------------------------------------------ #
    # Strike-grid helpers                                                  #
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
    def _offset(ladder, atm, k):
        if len(ladder) == 0:
            return float("nan")
        j  = int(np.argmin(np.abs(ladder - atm)))
        j2 = min(max(j + k, 0), len(ladder) - 1)
        return float(ladder[j2])

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
    # VWAP                                                                 #
    # ------------------------------------------------------------------ #
    def _vwap_now(self, snap):
        c = self.context
        if "vwap_arr" not in c.data:
            c["vwap_arr"] = self._build_vwap(snap._feed)
        v = c["vwap_arr"][snap.i]
        return float(v) if v == v else float(snap.spot)

    def _build_vwap(self, feed):
        sp  = np.asarray(feed.spot, dtype=float)
        n   = len(sp)
        vol = feed.arrays.get(self.volume_field) if self.vwap_use_volume else None
        if vol is not None:
            vol = np.asarray(vol, dtype=float).reshape(-1)
            vol = vol if vol.shape[0] == n else None
        if vol is not None and np.nansum(vol) > 0:
            if self.volume_is_cumulative:
                per = np.diff(vol, prepend=vol[:1])
                vol = np.where(per < 0, 0.0, per)
            w   = np.where(np.isnan(vol), 0.0, vol)
            p   = np.where(np.isnan(sp),  0.0, sp)
            num = np.cumsum(p * w)
            den = np.cumsum(w)
            return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        valid = ~np.isnan(sp)
        csum  = np.nancumsum(sp)
        ccnt  = np.cumsum(valid)
        return np.where(ccnt > 0, csum / np.maximum(ccnt, 1), np.nan)

    # ------------------------------------------------------------------ #
    # Alphas                                                               #
    # ------------------------------------------------------------------ #
    def compute_alphas(self, snap):
        c    = self.context
        sec  = snap.i
        spot = snap.spot

        # record starting equity once (day PnL = equity - starting_equity)
        if c["starting_equity"] is None:
            c["starting_equity"] = self.broker.portfolio.equity(snap)

        vwap     = self._vwap_now(snap)
        dev_bps  = (spot - vwap) / vwap * 1e4 if vwap else 0.0
        h        = snap.spot_hist(self.range_win)
        range_bps = (float(h.max()) - float(h.min())) / spot * 1e4 if len(h) else 0.0

        # range over just the last group_interval (for tight-add gate)
        h_win = snap.spot_hist(self.group_interval)
        range_win_bps = (float(h_win.max()) - float(h_win.min())) / spot * 1e4 if len(h_win) else 0.0

        atm    = snap.atm_strike(quoted_only=True)
        ladder = self._stable_ladder(snap)
        c1_ce, c1_pe = self._offset(ladder, atm, +1), self._offset(ladder, atm, -3)
        c2_ce, c2_pe = self._offset(ladder, atm, +3), self._offset(ladder, atm, -1)
        t_ce,  t_pe  = self._bracket(ladder, spot)

        def _ok(ce, pe):
            return 1.0 if (self._leg_stable(snap, ce, "CE")
                           and self._leg_stable(snap, pe, "PE")) else 0.0
        c1_ok, c2_ok = _ok(c1_ce, c1_pe), _ok(c2_ce, c2_pe)
        t_ok         = _ok(t_ce, t_pe)

        skew_legs_ok = 1.0
        if c.get("skew_ce") is not None:
            skew_legs_ok = 1.0 if (self._leg_stable(snap, c["skew_ce"], "CE")
                                   and self._leg_stable(snap, c["skew_pe"], "PE")) else 0.0
        tight_legs_ok = 1.0
        if c.get("tight_ce") is not None:
            tight_legs_ok = 1.0 if (self._leg_stable(snap, c["tight_ce"], "CE")
                                    and self._leg_stable(snap, c["tight_pe"], "PE")) else 0.0

        pf       = self.broker.portfolio
        upnl     = pf.unrealized_pnl(snap)
        margin   = self._margin_used()

        if c.get("episode") in ("skew", "tight"):
            c["peak_upnl"] = max(c.get("peak_upnl", upnl), upnl)

        risk_reason = self._risk_reason(c, upnl, margin)

        # day PnL = net equity change since session open
        day_pnl      = pf.equity(snap) - c["starting_equity"]
        day_stop_hit = day_pnl <= -(self.day_stop_frac * self._total_day_margin())

        c["now_sec"]      = sec
        c["now_spot"]     = spot
        c["now_range_bps"]= range_bps
        c["upnl"]         = upnl

        return {
            "sec": sec, "spot": spot, "vwap": round(vwap, 4),
            "dev_bps": dev_bps, "range_bps": range_bps,
            "range_win_bps": range_win_bps,
            "risk_reason": risk_reason,
            "day_stop_hit": float(day_stop_hit),
            "day_pnl": day_pnl,
            "atm": atm, "upnl": upnl, "margin": margin,
            "group": float(c.get("group", 0)),
            "c1_ce": c1_ce, "c1_pe": c1_pe, "c2_ce": c2_ce, "c2_pe": c2_pe,
            "t_ce": t_ce, "t_pe": t_pe,
            "c1_ok": c1_ok, "c2_ok": c2_ok, "t_ok": t_ok,
            "skew_legs_ok": skew_legs_ok, "tight_legs_ok": tight_legs_ok,
        }

    # ------------------------------------------------------------------ #
    # Risk helpers                                                         #
    # ------------------------------------------------------------------ #
    def _risk_reason(self, c, upnl, margin):
        """Check per-trade stops. Anchored to MARGIN (margin currently deployed),
        not credit, so the threshold scales with actual position size."""
        if margin <= 0.0:
            return ""
        # profit target
        pt = self.profit_target if self.profit_target is not None \
             else self.profit_take_frac * margin
        if upnl >= pt:
            return "profit_take"
        # fixed stop-loss
        sl = self.stop_loss_abs if self.stop_loss_abs is not None \
             else self.stop_loss_frac * margin
        if upnl <= -sl:
            return "stop_loss"
        # trailing: arms once up trail_arm_frac * margin, then gives back trail_frac * margin
        peak = c.get("peak_upnl", upnl)
        if peak >= self.trail_arm_frac * margin:
            give = self.trail_stop if self.trail_stop is not None \
                   else self.trail_frac * margin
            if upnl <= peak - give:
                return "trailing"
        return ""

    def _position_notional(self, snap):
        pf, total = self.broker.portfolio, 0.0
        for (strike, opt_type), pos in pf.positions.items():
            if pos.lots == 0:
                continue
            mh = snap.mid_and_half_spread(strike, opt_type)
            if mh is None:
                continue
            total += abs(mh[0] * pos.lots * pf.lot_size)
        return total

    # ------------------------------------------------------------------ #
    # Room helpers                                                         #
    # ------------------------------------------------------------------ #
    def _room_skew(self, a):  return a["sec"] <= 23400 - self.skew_max_hold
    def _room_tight(self, a): return a["sec"] <= 23400 - self.tight_hold

    # ------------------------------------------------------------------ #
    # Guards                                                               #
    # ------------------------------------------------------------------ #

    # --- daily stop (highest priority everywhere) ---
    def guard_day_stop(self, a, c):
        return bool(a["day_stop_hit"]) and bool(c.get("open_legs"))

    # --- WAIT: entry gates ---
    def guard_above_vwap(self, a, c):
        return (not a["day_stop_hit"] and
                a["dev_bps"] >= self.dev_bps_thresh and
                a["c1_ok"] > 0 and self._room_skew(a))

    def guard_below_vwap(self, a, c):
        return (not a["day_stop_hit"] and
                a["dev_bps"] <= -self.dev_bps_thresh and
                a["c2_ok"] > 0 and self._room_skew(a))

    def guard_calm_range(self, a, c):
        return (not a["day_stop_hit"] and
                a["range_bps"] < self.range_bps_max and
                a["t_ok"] > 0 and self._room_tight(a))

    # --- skew pyramid ---
    def guard_always_hold(self, a, c):
        return True

    def guard_skew_add(self, a, c):
        """Add next group if: interval elapsed, spot still moved in our direction,
        and locked strikes still quoted."""
        if c.get("group", 0) >= self.n_groups:
            return False
        if a["sec"] - c.get("window_start_sec", a["sec"]) < self.group_interval:
            return False
        if a["skew_legs_ok"] <= 0:
            return False
        # direction gate: for above_vwap (side=="above") spot must still be >= window-start spot
        side = c.get("skew_side", "above")
        ws   = c.get("window_start_spot", a["spot"])
        return a["spot"] >= ws if side == "above" else a["spot"] <= ws

    def guard_skew_freeze(self, a, c):
        """Freeze if interval elapsed but the direction gate or liquidity gate fails."""
        if c.get("group", 0) >= self.n_groups:
            return False
        if a["sec"] - c.get("window_start_sec", a["sec"]) < self.group_interval:
            return False
        # freeze if would NOT add (direction reversed OR illiquid)
        side = c.get("skew_side", "above")
        ws   = c.get("window_start_spot", a["spot"])
        direction_ok = a["spot"] >= ws if side == "above" else a["spot"] <= ws
        return (not direction_ok) or (a["skew_legs_ok"] <= 0)

    def guard_skew_maxhold(self, a, c):
        return (c.get("episode") == "skew" and
                a["sec"] - c.get("entry_sec", a["sec"]) >= self.skew_max_hold)

    # --- tight pyramid ---
    def guard_tight_add(self, a, c):
        """Add next tight group if: interval elapsed, range is still calm (< max),
        and locked strikes still quoted."""
        if c.get("group", 0) >= self.n_groups:
            return False
        if a["sec"] - c.get("window_start_sec", a["sec"]) < self.group_interval:
            return False
        if a["tight_legs_ok"] <= 0:
            return False
        # calm gate: range over the last group_interval must still be below the ceiling
        return a["range_win_bps"] < self.range_bps_max

    def guard_tight_freeze(self, a, c):
        """Freeze tight pyramid if interval elapsed but calm gate or liquidity gate fails."""
        if c.get("group", 0) >= self.n_groups:
            return False
        if a["sec"] - c.get("window_start_sec", a["sec"]) < self.group_interval:
            return False
        return a["range_win_bps"] >= self.range_bps_max or a["tight_legs_ok"] <= 0

    def guard_tight_done(self, a, c):
        return (c.get("episode") == "tight" and
                a["sec"] - c.get("entry_sec", a["sec"]) >= self.tight_hold)

    # --- risk exits (shared by skew and tight) ---
    def guard_profit_take(self, a, c): return a.get("risk_reason") == "profit_take"
    def guard_stop_loss(self, a, c):   return a.get("risk_reason") == "stop_loss"
    def guard_trailing(self, a, c):    return a.get("risk_reason") == "trailing"