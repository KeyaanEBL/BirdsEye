"""vwap_skew_strangle_nofallback — a three-regime short-options strategy.

This is the no-fallback sibling of `vwap_skew_strangle`: identical skew and tight
logic, but with regime 4 (the wide "park the cash" strangle) removed entirely.
When none of the three regimes apply the strategy simply stays FLAT — capital
sits idle rather than shorting a wide strangle.

This is a strategy *module* (importable, spawn-safe, testable) in the BirdsEye
FSM style. Each trading second it classifies the tape from two alphas —
spot-vs-VWAP deviation and the trailing 15-minute spot range — and holds the
position that regime calls for (or nothing, if no regime fires).

Sizing
------
20 lots total = 10 on the CALL branch + 10 on the PUT branch ("10 pairs").
  * Regimes 1 & 2 build those 10 lots/side in a gated pyramid (1+2+3+4 = 10).
  * Regime 3 takes all 10 lots/side in one shot.

The three regimes (evaluated top-down each second; first match wins)
-------------------------------------------------------------------
1. spot >= VWAP + dev_bps_thresh   (default +20 bps)  -> SKEW: short PUT far
   from ATM (-3 strikes) + short CALL near ATM (+1 strike). Pyramided entry,
   trailing stop + profit limit, max-hold 20 min.
2. spot <= VWAP - dev_bps_thresh   (default -20 bps)  -> SKEW: mirror of (1):
   short CALL far (+3) + short PUT near (-1). Same pyramid / stop / hold rules.
3. else if trailing-15-min range < range_bps_max (default 15 bps) -> TIGHT
   strangle on the two strikes that bracket spot most tightly. All 10/side at
   once, hold 30 min, square off.
4. else -> FLAT. No fallback position; wait for a regime to appear.

Pyramided entry (regimes 1 & 2 only)
------------------------------------
Groups of 1, 2, 3, 4 lots/side go in at 30-second intervals (so the build-out
spans ~2 minutes). The next group only fires if the open position's mark-to-
market profit over the last 30-second window was positive; the first window
that fails freezes the pyramid at its current size (state SKEW_FROZEN) and we
just hold. CALLs == PUTs in every slice. Max-hold and the auto stops are timed
from the first slice.

VWAP
----
Session-anchored, lookahead-safe VWAP of spot. When a per-second volume series
is loaded into the feed (pass it via the `fields` arg to BirdsEye and name it
with `volume_field`), VWAP[i] = sum(spot*vol)[0..i] / sum(vol)[0..i] — a true
volume-weighted price. If that column is absent (or all-zero), we fall back to a
session running mean of spot (a TWAP, == VWAP under uniform volume). Either way
`vwap[i]` depends only on rows 0..i, so no future information leaks even though
the cumulative arrays are built once and cached. Set `volume_is_cumulative=True`
if your volume column is a running total rather than per-second turnover.

Robust strike selection (no execution starvation)
-------------------------------------------------
On real data a deep-OTM strike can be quoted one second and gone the next. Since
the engine starves (raises) if it works an order for 60 ticks with no fill, every
strike this strategy emits is taken from the *stably quoted* ladder — strikes
whose CE and PE bid/ask have been continuously present for the last
`quote_persist` seconds — so the leg is overwhelmingly likely to still be quoted
when the slice fills the next tick. Pyramid adds re-check the locked strikes each
group: if a locked leg has gone illiquid, the pyramid freezes instead of firing
into a strike with no quote.

Run it (notebook / BirdsEye)
----------------------------
    from strategies.vwap_skew_strangle_nofallback import VwapSkewStrangleNoFallback
    be = BirdsEye(
        strategy_cls    = VwapSkewStrangleNoFallback,
        index           = "SPY",
        split           = "train",
        strategy_kwargs = {"lots": 10},          # 10/side -> 20 lots total
        lot_size        = 100,
        starting_cash   = 1_000_000.0,
        n_workers       = 40,
    )
    res = be.run()
"""

import numpy as np

from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context


# --------------------------------------------------------------------------- #
# States                                                                       #
# --------------------------------------------------------------------------- #
class Wait(State):
    """Flat hub. Entered both at session start and after every square-off — so
    its target() flattens any legs still open, and its transitions pick the next
    regime on the following second."""
    name = "WAIT"
    transitions = {
        "above_vwap":    "SKEW_ADD",     # regime 1
        "below_vwap":    "SKEW_ADD",     # regime 2
        "calm_range":    "TIGHT",        # regime 3
        # regime 4 (wide fallback) removed: no match -> stay flat in WAIT
    }

    def target(self, alphas, ctx):
        legs = ctx.get("open_legs")
        if not legs:
            return None
        return ctx["strat"].close_legs(legs, reason=Reason(state="WAIT", note="square off"))

    def on_enter(self, ctx):
        # reset the per-episode memory; we are flat now
        ctx["episode"]    = None
        ctx["open_legs"]  = None
        ctx["group"]      = 0
        ctx["credit"]     = 0.0
        ctx["peak_upnl"]  = 0.0
        ctx["skew_ce"]    = None
        ctx["skew_pe"]    = None


class SkewAdd(State):
    """Places ONE pyramid group (1/2/3/4 lots/side) for regime 1 or 2, at strikes
    locked on the first group. Routes straight to SKEW_HOLD once the group fills."""
    name = "SKEW_ADD"
    transitions = {"eod_flat": "WAIT", "always_hold": "SKEW_HOLD"}

    def target(self, alphas, ctx):
        strat = ctx["strat"]
        if ctx.get("group", 0) == 0:                       # first group: choose side & lock strikes
            if alphas["dev_bps"] >= 0:                     # regime 1: spot above VWAP
                ce, pe, side = alphas["c1_ce"], alphas["c1_pe"], "above"
            else:                                          # regime 2: spot below VWAP
                ce, pe, side = alphas["c2_ce"], alphas["c2_pe"], "below"
            ctx["skew_ce"], ctx["skew_pe"], ctx["skew_side"] = ce, pe, side
        g    = ctx.get("group", 0) + 1                     # this group's number, 1..n
        lots = strat.pyramid_schedule[g - 1]               # 1, 2, 3, 4
        ce, pe = ctx["skew_ce"], ctx["skew_pe"]
        return Order(
            name="skew_pyramid",
            legs=[OrderLeg(ce, "CE", lots=lots, action="SELL"),
                  OrderLeg(pe, "PE", lots=lots, action="SELL")],
            reason=Reason(state="SKEW_ADD", note=f"{ctx['skew_side']}|group{g}x{lots}"),
        )

    def on_enter(self, ctx):                               # runs at FILL-COMPLETE of the group
        ctx["group"]    = ctx.get("group", 0) + 1
        ctx["episode"]  = "skew"
        ctx["open_legs"] = [(ctx["skew_ce"], "CE"), (ctx["skew_pe"], "PE")]
        # open a fresh 30s profit-gate window from this fill
        ctx["window_start_sec"]  = ctx["now_sec"]
        ctx["window_start_upnl"] = ctx["upnl"]
        if ctx["group"] == 1:                              # the first slice anchors the auto-exits
            ctx["entry_sec"] = ctx["now_sec"]
            ctx["peak_upnl"] = ctx["upnl"]                 # ~0 just after entry


class SkewHold(State):
    """Holding a (possibly still-building) skew position. Auto-exits win priority
    over scaling; if the 30s window was unprofitable we freeze the pyramid."""
    name = "SKEW_HOLD"
    transitions = {
        "eod_flat":    "WAIT",          # near close: flatten everything (highest priority)
        "skew_exit":   "WAIT",          # trailing stop / profit limit / 20-min max-hold
        "skew_add":    "SKEW_ADD",      # 30s elapsed AND window profitable AND group < n
        "skew_freeze": "SKEW_FROZEN",   # 30s elapsed AND window NOT profitable
    }

    def target(self, alphas, ctx):
        return None                     # pure holding state; never opens a trade itself


class SkewFrozen(State):
    """Pyramid stopped scaling (a 30s window failed). Just hold to an auto-exit."""
    name = "SKEW_FROZEN"
    transitions = {"eod_flat": "WAIT", "skew_exit": "WAIT"}

    def target(self, alphas, ctx):
        return None


class TightStrangle(State):
    """Regime 3: short the two strikes bracketing spot, all 10/side at once, 30-min hold."""
    name = "TIGHT"
    transitions = {"eod_flat": "WAIT", "tight_done": "WAIT"}

    def target(self, alphas, ctx):
        strat = ctx["strat"]
        ce, pe = alphas["t_ce"], alphas["t_pe"]
        ctx["tight_ce"], ctx["tight_pe"] = ce, pe
        return Order(
            name="tight_strangle",
            legs=[OrderLeg(ce, "CE", lots=strat.lots, action="SELL"),
                  OrderLeg(pe, "PE", lots=strat.lots, action="SELL")],
            reason=Reason(state="TIGHT", note="bracket strangle"),
        )

    def on_enter(self, ctx):
        ctx["episode"]   = "tight"
        ctx["open_legs"] = [(ctx["tight_ce"], "CE"), (ctx["tight_pe"], "PE")]
        ctx["entry_sec"] = ctx["now_sec"]


# --------------------------------------------------------------------------- #
# Strategy                                                                      #
# --------------------------------------------------------------------------- #
class VwapSkewStrangleNoFallback(StateMachineStrategy):
    states = {
        "WAIT":        Wait(),
        "SKEW_ADD":    SkewAdd(),
        "SKEW_HOLD":   SkewHold(),
        "SKEW_FROZEN": SkewFrozen(),
        "TIGHT":       TightStrangle(),
    }
    # Every emitted order is <= `lots` per leg, so a large slice fills each one in
    # a single tick. The 30s pyramid spacing is enforced by guard_skew_add — NOT
    # by the slicer — so a one-tick fill per group is exactly what we want.
    slice_lots = 10
    pause      = 0

    def __init__(self, broker, lots=10,
                 dev_bps_thresh=20.0,          # |spot-VWAP| trigger for regimes 1/2 (bps)
                 range_bps_max=15.0,           # trailing-range ceiling for regime 3 (bps)
                 range_win=900,                # trailing window for the range alpha (15 min)
                 pyramid_schedule=(1, 2, 3, 4),# lots/side per slice; must sum to `lots`
                 group_interval=30,            # seconds between pyramid groups
                 skew_max_hold=1200,           # regimes 1/2 max hold from first slice (20 min)
                 tight_hold=1800,              # regime 3 hold (30 min)
                 profit_take_frac=0.40,        # profit limit as a fraction of credit collected
                 trail_frac=0.25,              # trailing stop as a fraction of credit, off the peak
                 profit_target=None,           # absolute profit limit ($); overrides the frac if set
                 trail_stop=None,              # absolute trailing give-back ($); overrides the frac
                 eod_buffer=120,               # flatten any open position this many secs before close
                 session_len=23400,            # legacy; EOD now uses the feed's actual length
                 vwap_use_volume=True,         # weight VWAP by traded volume when available
                 volume_field="volume",        # feed field holding per-second underlying volume
                 volume_is_cumulative=False,   # True if that column is a running total, not per-sec
                 quote_persist=60):            # secs a strike must stay quoted to be tradable
        self.lots             = lots
        self.dev_bps_thresh   = dev_bps_thresh
        self.range_bps_max    = range_bps_max
        self.range_win        = range_win
        self.pyramid_schedule = tuple(pyramid_schedule)
        self.n_groups         = len(self.pyramid_schedule)
        self.group_interval   = group_interval
        self.skew_max_hold    = skew_max_hold
        self.tight_hold       = tight_hold
        self.profit_take_frac = profit_take_frac
        self.trail_frac       = trail_frac
        self.profit_target    = profit_target
        self.trail_stop       = trail_stop
        self.eod_buffer       = eod_buffer
        self.session_len      = session_len
        self.vwap_use_volume      = vwap_use_volume
        self.volume_field         = volume_field
        self.volume_is_cumulative = volume_is_cumulative
        self.quote_persist        = max(1, int(quote_persist))

        assert sum(self.pyramid_schedule) == lots, \
            f"pyramid_schedule {self.pyramid_schedule} must sum to lots={lots}"

        # warm-up: need a full range window of history before any decision
        self.min_history = range_win

        ctx = Context()
        ctx["episode"]   = None
        ctx["open_legs"] = None
        ctx["group"]     = 0
        ctx["credit"]    = 0.0
        ctx["peak_upnl"] = 0.0
        super().__init__("WAIT", broker, name="vwap_skew_strangle_nofallback", context=ctx)
        ctx["strat"] = self

    # ------------------------------------------------------------------ #
    # strike-grid helpers (the public snapshot API exposes ATM but not    #
    # the ladder, so we read the feed's sorted strike array the same way  #
    # MarketSnapshot.atm_strike(quoted_only=True) does)                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quoted_ladder(snap):
        """Sorted strikes that have live CE & PE bid/ask right now."""
        feed, i = snap._feed, snap.i
        strikes = feed.strikes
        ok = np.ones(len(strikes), dtype=bool)
        for key in (("ce", "bid_0"), ("ce", "ask_0"), ("pe", "bid_0"), ("pe", "ask_0")):
            arr = feed.arrays.get(key)
            if arr is None:
                return strikes                     # can't filter -> use the full grid
            ok &= ~np.isnan(arr[i])
        return strikes[ok] if ok.any() else strikes

    def _stable_ladder(self, snap):
        """Strikes whose CE & PE bid/ask have BOTH been continuously quoted over
        the last `quote_persist` seconds. A strike alive that long is very unlikely
        to vanish before the slice fills next tick — which is what keeps the engine
        from starving. Falls back to the quoted-now ladder if nothing qualifies (or
        if quote arrays aren't loaded)."""
        feed, i = snap._feed, snap.i
        strikes = feed.strikes
        lo      = max(0, i - self.quote_persist + 1)
        ok      = np.ones(len(strikes), dtype=bool)
        for key in (("ce", "bid_0"), ("ce", "ask_0"), ("pe", "bid_0"), ("pe", "ask_0")):
            arr = feed.arrays.get(key)
            if arr is None:
                return self._quoted_ladder(snap)
            ok &= ~np.isnan(arr[lo:i + 1]).any(axis=0)
        return strikes[ok] if ok.any() else self._quoted_ladder(snap)

    def _leg_stable(self, snap, strike, opt_type):
        """True if this one (strike, opt_type) leg has had non-NaN bid & ask for the
        last `quote_persist` secs. NOTE: field_hist keys arrays case-sensitively, so
        opt_type is lowercased here to match the feed's 'ce'/'pe' arrays."""
        if strike != strike:                       # NaN strike
            return False
        ot = opt_type.lower()
        b  = snap.field_hist(strike, ot, "bid_0", self.quote_persist)
        a  = snap.field_hist(strike, ot, "ask_0", self.quote_persist)
        if b is None or a is None or len(b) == 0:
            return snap.quote(strike, opt_type) is not None
        return bool((~np.isnan(b)).all() and (~np.isnan(a)).all())

    @staticmethod
    def _offset(ladder, atm, k):
        """Strike k grid-steps from ATM (clamped to the ladder ends)."""
        if len(ladder) == 0:
            return float("nan")
        j  = int(np.argmin(np.abs(ladder - atm)))
        j2 = min(max(j + k, 0), len(ladder) - 1)
        return float(ladder[j2])

    @staticmethod
    def _bracket(ladder, spot):
        """The two strikes that most tightly bracket spot: (first above, first below)."""
        if len(ladder) == 0:
            return float("nan"), float("nan")
        above = ladder[ladder > spot]
        below = ladder[ladder < spot]
        ce = float(above[0])  if len(above) else float(ladder[-1])
        pe = float(below[-1]) if len(below) else float(ladder[0])
        return ce, pe

    def _vwap_now(self, snap):
        """Lookahead-safe session VWAP at the current second. Built once per day
        and cached; vwap[i] uses only rows 0..i."""
        c = self.context
        if "vwap_arr" not in c.data:
            c["vwap_arr"] = self._build_vwap(snap._feed)
        v = c["vwap_arr"][snap.i]
        return float(v) if v == v else float(snap.spot)    # NaN guard -> spot

    def _build_vwap(self, feed):
        """sum(spot*vol)/sum(vol) cumulatively when a usable volume column is
        loaded; otherwise a session running mean of spot (TWAP). Both are pure
        prefix sums, so value at i never depends on rows after i."""
        sp = np.asarray(feed.spot, dtype=float)
        n  = len(sp)

        vol = feed.arrays.get(self.volume_field) if self.vwap_use_volume else None
        if vol is not None:
            vol = np.asarray(vol, dtype=float).reshape(-1)
            vol = vol if vol.shape[0] == n else None

        if vol is not None and np.nansum(vol) > 0:
            if self.volume_is_cumulative:                  # running total -> per-second turnover
                per = np.diff(vol, prepend=vol[:1])
                vol = np.where(per < 0, 0.0, per)          # ignore resets/rollbacks
            w   = np.where(np.isnan(vol), 0.0, vol)
            p   = np.where(np.isnan(sp), 0.0, sp)
            num = np.cumsum(p * w)
            den = np.cumsum(w)
            return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

        # fallback: session running mean of spot (TWAP == VWAP under uniform volume)
        valid = ~np.isnan(sp)
        csum  = np.nancumsum(sp)
        ccnt  = np.cumsum(valid)
        return np.where(ccnt > 0, csum / np.maximum(ccnt, 1), np.nan)

    # ------------------------------------------------------------------ #
    # alphas — computed every (non-executing) second                      #
    # ------------------------------------------------------------------ #
    def compute_alphas(self, snap):
        c   = self.context
        sec = snap.i
        spot = snap.spot
        bars_left = (len(snap._feed.ts) - 1) - sec     # seconds until this day's last bar

        # --- the two headline alphas ---
        vwap     = self._vwap_now(snap)
        dev_bps  = (spot - vwap) / vwap * 1e4 if vwap else 0.0
        h        = snap.spot_hist(self.range_win)
        range_bps = (float(h.max()) - float(h.min())) / spot * 1e4 if len(h) else 0.0

        # --- candidate strikes for every regime, off the STABLY quoted ladder ---
        atm    = snap.atm_strike(quoted_only=True)
        ladder = self._stable_ladder(snap)
        c1_ce, c1_pe = self._offset(ladder, atm, +1), self._offset(ladder, atm, -3)   # regime 1
        c2_ce, c2_pe = self._offset(ladder, atm, +3), self._offset(ladder, atm, -1)   # regime 2
        t_ce,  t_pe  = self._bracket(ladder, spot)                                     # regime 3

        # per-regime tradability: BOTH legs (the exact option type we'd sell) must be
        # persistently quoted, so the slice can't fire into a strike with no quote.
        def _ok(ce, pe):
            return 1.0 if (self._leg_stable(snap, ce, "CE")
                           and self._leg_stable(snap, pe, "PE")) else 0.0
        c1_ok, c2_ok = _ok(c1_ce, c1_pe), _ok(c2_ce, c2_pe)
        t_ok         = _ok(t_ce, t_pe)

        # locked skew legs (pyramid adds re-check these every group)
        skew_legs_ok = 1.0
        if c.get("skew_ce") is not None:
            skew_legs_ok = 1.0 if (self._leg_stable(snap, c["skew_ce"], "CE")
                                   and self._leg_stable(snap, c["skew_pe"], "PE")) else 0.0

        # --- book state the guards/states need (no snap is passed to them) ---
        pf       = self.broker.portfolio
        upnl     = pf.unrealized_pnl(snap)
        notional = self._position_notional(snap)

        # track the credit anchor (peak short notional) and the profit peak while in a skew
        if c.get("episode") == "skew":
            c["credit"]    = max(c.get("credit", 0.0), notional)
            c["peak_upnl"] = max(c.get("peak_upnl", upnl), upnl)

        # stash scalars for on_enter hooks (which only receive ctx)
        c["now_sec"] = sec
        c["upnl"]    = upnl

        return {
            "sec": sec, "spot": spot, "vwap": round(vwap, 4),
            "bars_left": bars_left,
            "dev_bps": dev_bps, "range_bps": range_bps,
            "atm": atm, "upnl": upnl, "notional": notional,
            "group": float(c.get("group", 0)),
            "c1_ce": c1_ce, "c1_pe": c1_pe, "c2_ce": c2_ce, "c2_pe": c2_pe,
            "t_ce": t_ce, "t_pe": t_pe,
            "c1_ok": c1_ok, "c2_ok": c2_ok, "t_ok": t_ok,
            "skew_legs_ok": skew_legs_ok,
        }

    def _position_notional(self, snap):
        """|mid * lots * lot_size| summed over open legs — the short's market value."""
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
    # session-room helpers                                                #
    # ------------------------------------------------------------------ #
    # session-room checks use the day's ACTUAL remaining bars (from the feed), not a
    # fixed session_len guess — so a position never outlives the day. Each also refuses
    # to open inside the EOD buffer, so we never open something we'd instantly flatten.
    def _room_skew(self, a):   return a["bars_left"] >= self.skew_max_hold and a["bars_left"] > self.eod_buffer
    def _room_tight(self, a):  return a["bars_left"] >= self.tight_hold    and a["bars_left"] > self.eod_buffer

    # ------------------------------------------------------------------ #
    # named guards (ledger-visible: the `signal` column records which fired)
    # ------------------------------------------------------------------ #
    # --- WAIT: regime selection (priority = insertion order) ---
    def guard_above_vwap(self, a, c):
        return a["dev_bps"] >= self.dev_bps_thresh and a["c1_ok"] > 0 and self._room_skew(a)

    def guard_below_vwap(self, a, c):
        return a["dev_bps"] <= -self.dev_bps_thresh and a["c2_ok"] > 0 and self._room_skew(a)

    def guard_calm_range(self, a, c):
        return a["range_bps"] < self.range_bps_max and a["t_ok"] > 0 and self._room_tight(a)

    # --- pyramid plumbing ---
    def guard_always_hold(self, a, c):
        return True                                        # SKEW_ADD -> SKEW_HOLD next tick

    def guard_skew_add(self, a, c):
        return (c.get("group", 0) < self.n_groups
                and a["sec"] - c.get("window_start_sec", a["sec"]) >= self.group_interval
                and a["upnl"] - c.get("window_start_upnl", 0.0) > 0.0
                and a["skew_legs_ok"] > 0)                 # locked strikes still tradable

    def guard_skew_freeze(self, a, c):
        # 30s window elapsed but we won't (or can't) add: unprofitable window OR a
        # locked leg has gone illiquid. Freezing here avoids firing into a dead quote.
        if not (c.get("group", 0) < self.n_groups
                and a["sec"] - c.get("window_start_sec", a["sec"]) >= self.group_interval):
            return False
        profitable = a["upnl"] - c.get("window_start_upnl", 0.0) > 0.0
        return (not profitable) or (a["skew_legs_ok"] <= 0)

    def guard_skew_exit(self, a, c):
        """Trailing stop OR profit limit OR 20-min max-hold (from the first slice)."""
        if c.get("episode") != "skew":
            return False
        sec, upnl = a["sec"], a["upnl"]
        if sec - c.get("entry_sec", sec) >= self.skew_max_hold:        # max-hold
            return True
        credit = c.get("credit", 0.0)
        if credit > 0.0:
            target = self.profit_target if self.profit_target is not None else self.profit_take_frac * credit
            if upnl >= target:                                          # profit limit
                return True
            trail = self.trail_stop if self.trail_stop is not None else self.trail_frac * credit
            peak  = c.get("peak_upnl", upnl)
            if upnl < peak and upnl <= peak - trail:                    # trailing stop
                return True
        return False

    # --- TIGHT (regime 3) ---
    def guard_tight_done(self, a, c):
        return a["sec"] - c.get("entry_sec", a["sec"]) >= self.tight_hold

    # --- end-of-day: flatten ANY open position within eod_buffer of the close ---
    def guard_eod_flat(self, a, c):
        return bool(c.get("open_legs")) and a["bars_left"] <= self.eod_buffer