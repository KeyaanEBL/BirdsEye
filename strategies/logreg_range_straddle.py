"""Logistic-regression regime-gated short straddle.

Entry: at each decision tick, compute the 10 signed-distance features from the
last 30 minutes of spot + ATM straddle, run the fitted LogRegPredictor, and
enter a short ATM straddle when P(range_bound) >= entry_prob.

Exit (in priority order):
  stop_hit       — open PnL <= -stop_pct * entry_credit
  target_hit     — open PnL >=  tp_pct  * entry_credit
  regime_flip    — P(range_bound) drops below exit_prob at next decision tick
  eod_squareoff  — hard close near session end
  time_stop      — max holding time elapsed

The logreg model and REGIME_CONFIG are injected at construction so that the
strategy is testable and the same class works for different indices/models.
"""

import os, sys

_BIRDSEYE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT  = os.path.dirname(_BIRDSEYE_ROOT)
for _p in (
    os.path.join(_PROJECT_ROOT, "Intern-Project"),
    os.path.join(_PROJECT_ROOT, "Regime_Classifier", "Regime_Classifier"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import regime_classifier as rc
import regime_config     as rcfg

from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context


# ---- feature constants (must match training in regime_prediction_logreg.ipynb) ----
_REGIMES = [f"{tr} x {dc}"
            for tr in ("uptrend", "downtrend", "mean_reverting", "range_bound")
            for dc in ("good_decay", "low_decay")]
_RANGE_IDX = [i for i, r in enumerate(_REGIMES) if r.startswith("range_bound")]

_FEAT_KEYS   = ("slope_gap", "slope_sign", "er_gap", "range_gap", "decay_gap")
FEATURE_COLS = [f"{w}_{k}" for w in ("cur", "prv") for k in _FEAT_KEYS]

WIN_SEC      = 15 * 60    # 15-min feature window in seconds
HIST_SEC     = 30 * 60    # 30-min total history needed
BAR_SECONDS  = 1.0


def _win_features(spot, strad, regime_config, day_open, window_start_ts_ns):
    """5 signed gap features for one 15-min window. Returns dict or None on failure."""
    # forward-fill any NaN straddle values
    st = pd.Series(strad).ffill().bfill().to_numpy(float)
    res = rc.classify_core(
        spot, st, None, regime_config,
        day_open_straddle=day_open,
        window_start=pd.Timestamp(int(window_start_ts_ns)),
        bar_seconds=BAR_SECONDS,
    )
    if res.get("status") != "ok":
        return None
    th = regime_config.resolve(res["window_min"], res["tod"])
    g  = lambda v: 0.0 if not np.isfinite(v) else float(v)
    return {
        "slope_gap":  g(abs(res["slope"]) - th.trending),
        "slope_sign": g(np.sign(res["slope"])),
        "er_gap":     g(res["er"]         - th.er),
        "range_gap":  g(res["spot_range"] - th.spot_range),
        "decay_gap":  g(res["decay"]      - th.decay),
    }


def _range_prob(logreg, regime_config, spot_buf, strad_buf, ts_buf, day_open, Lm=15):
    """P(range_bound) from 30-min buffers. Returns float in [0,1] or None."""
    n = len(spot_buf)
    if n < HIST_SEC:
        return None
    sp = np.asarray(spot_buf, float)
    st = np.asarray(strad_buf, float)
    ts = np.asarray(ts_buf,   dtype=np.int64)

    cur = _win_features(sp[-WIN_SEC:],         st[-WIN_SEC:],         regime_config, day_open, ts[-WIN_SEC])
    prv = _win_features(sp[-HIST_SEC:-WIN_SEC], st[-HIST_SEC:-WIN_SEC], regime_config, day_open, ts[-HIST_SEC])
    if cur is None or prv is None:
        return None

    row = {f"cur_{k}": cur[k] for k in _FEAT_KEYS}
    row.update({f"prv_{k}": prv[k] for k in _FEAT_KEYS})
    df   = pd.DataFrame([row])[FEATURE_COLS].fillna(0.0)
    prob = logreg.predict_proba(df, Lm)   # shape (1, n_regimes)
    return float(prob[0, _RANGE_IDX].sum())


# ---- states ----------------------------------------------------------------

class Wait(State):
    name = "WAIT"
    transitions = {"range_bound_entry": "SHORT"}

    def target(self, alphas, ctx):
        return None


class Short(State):
    name = "SHORT"
    transitions = {
        "stop_hit":      "FLATTEN",
        "target_hit":    "FLATTEN",
        "regime_flip":   "FLATTEN",
        "eod_squareoff": "FLATTEN",
        "time_stop":     "FLATTEN",
    }

    def target(self, alphas, ctx):
        atm  = alphas["atm"]
        lots = ctx["strat"].lots
        ctx["short_atm"] = atm
        return Order(
            name="range_straddle",
            legs=[OrderLeg(atm, "CE", lots=lots, action="SELL", slice_lots=10, pause=2),
                  OrderLeg(atm, "PE", lots=lots, action="SELL", slice_lots=10, pause=2)],
            reason=Reason(state="SHORT", note=f"range_prob={alphas['range_prob']:.2f}"),
        )

    def on_enter(self, ctx):
        ctx["entry_sec"] = ctx["now_sec"]
        pf = ctx["strat"].broker.portfolio
        atm = ctx["short_atm"]
        credit = 0.0
        for opt in ("CE", "PE"):
            pos = pf.positions.get((atm, opt))
            if pos is not None and pos.lots != 0:
                credit += pos.avg_entry * abs(pos.lots) * pf.lot_size
        ctx["entry_credit"] = credit


class Flatten(State):
    name = "FLATTEN"
    transitions = {"always": "WAIT"}

    def target(self, alphas, ctx):
        atm = ctx.get("short_atm")
        if atm is None:
            return None
        return ctx["strat"].close_legs(
            [(atm, "CE"), (atm, "PE")],
            reason=Reason(state="FLATTEN", note="square off"),
        )

    def on_enter(self, ctx):
        ctx["short_atm"]    = None
        ctx["entry_credit"] = 0.0
        ctx["last_exit_sec"] = ctx["now_sec"]   # record when we went flat


# ---- strategy --------------------------------------------------------------

class LogRegRangeStraddle(StateMachineStrategy):
    states = {"WAIT": Wait(), "SHORT": Short(), "FLATTEN": Flatten()}

    def __init__(self, broker, logreg, regime_config, index="SPY",
                 lots=1,
                 entry_prob=0.45,    # enter when P(range_bound) >= this
                 exit_prob=0.30,     # exit if P(range_bound) drops below this
                 pred_horizon=20,    # forward horizon (min) matching training
                 decision_every=60,  # recheck model this often (seconds)
                 stop_pct=1.0,
                 tp_pct=0.50,
                 hold_max=3600,
                 min_flat_sec=900,   # cooldown after any exit before re-entry (15 min)
                 min_hold_sec=900,   # minimum hold before regime_flip can fire (15 min)
                 eod_buffer=600,
                 max_lots = 1):    # close when <= this many seconds remain in the day
        self.logreg         = logreg
        self.regime_config  = regime_config
        self.lots           = lots
        self.entry_prob     = entry_prob
        self.exit_prob      = exit_prob
        self.pred_horizon   = pred_horizon
        self.decision_every = decision_every
        self.stop_pct       = stop_pct
        self.tp_pct         = tp_pct
        self.hold_max       = hold_max
        self.min_flat_sec   = min_flat_sec
        self.min_hold_sec   = min_hold_sec
        self.eod_buffer     = eod_buffer
        self.min_history    = HIST_SEC   # 30-min warm-up before any decisions
        self.max_lots      = max_lots   

        # set session for the correct index so classify_core uses right market hours
        rc.SESSION = rcfg.session_dict(index)

        ctx = Context()
        # First 30 min fills the buffer (HIST_SEC); next 30 min the buffer still
        # contains the trending morning open so predictions are near-zero. Start
        # decisions only after the buffer has fully slid past the open (2×HIST_SEC).
        ctx["next_decision"]  = 2 * HIST_SEC
        ctx["short_atm"]      = None
        ctx["entry_credit"]   = 0.0
        ctx["range_prob"]     = 0.0
        ctx["last_exit_sec"]  = -999999   # far in the past so first entry is never blocked
        ctx["spot_buf"]       = []
        ctx["strad_buf"]      = []
        ctx["ts_buf"]         = []
        ctx["day_open"]       = None
        super().__init__("WAIT", broker, name="logreg_range_straddle", context=ctx)
        ctx["strat"] = self

    # ---- guards ----
    def guard_range_bound_entry(self, a, c):
        cooled_down = a["sec"] - c["last_exit_sec"] >= self.min_flat_sec
        return (a["decision_now"] and
                cooled_down and
                a["range_prob"] >= self.entry_prob and
                a["secs_remaining"] > self.eod_buffer and
                a["atm"] is not None)

    def guard_stop_hit(self, a, c):
        cr = c.get("entry_credit", 0.0)
        return cr > 0 and a["open_pnl"] <= -self.stop_pct * cr

    def guard_target_hit(self, a, c):
        cr = c.get("entry_credit", 0.0)
        return cr > 0 and a["open_pnl"] >= self.tp_pct * cr

    def guard_regime_flip(self, a, c):
        held_long_enough = a["sec"] - c.get("entry_sec", 0) >= self.min_hold_sec
        return (a["decision_now"] and
                held_long_enough and
                a["range_prob"] < self.exit_prob)

    def guard_eod_squareoff(self, a, c):
        return a["secs_remaining"] <= self.eod_buffer

    def guard_time_stop(self, a, c):
        return a["sec"] - c.get("entry_sec", 0) >= self.hold_max

    def guard_always(self, a, c):
        return True

    # ---- buffer + alpha computation ----
    def _update_buffers(self, snap, c):
        atm   = snap.atm_strike(quoted_only=True)
        strad = np.nan
        if atm is not None:
            ce = snap.mid_and_half_spread(atm, "CE")
            pe = snap.mid_and_half_spread(atm, "PE")
            if ce is not None and pe is not None:
                strad = ce[0] + pe[0]
                if c["day_open"] is None:
                    c["day_open"] = strad

        c["spot_buf"].append(snap.spot)
        c["strad_buf"].append(strad)
        c["ts_buf"].append(snap.ts)

        if len(c["spot_buf"]) > HIST_SEC:
            c["spot_buf"]  = c["spot_buf"][-HIST_SEC:]
            c["strad_buf"] = c["strad_buf"][-HIST_SEC:]
            c["ts_buf"]    = c["ts_buf"][-HIST_SEC:]

    def compute_alphas(self, snap):
        c   = self.context
        sec = snap.i
        c["now_sec"] = sec

        self._update_buffers(snap, c)

        decide = sec >= c["next_decision"]
        if decide:
            c["next_decision"] += self.decision_every
            if c["day_open"] is not None:
                p = _range_prob(
                    self.logreg, self.regime_config,
                    c["spot_buf"], c["strad_buf"], c["ts_buf"],
                    c["day_open"], self.pred_horizon,
                )
                if p is not None:
                    c["range_prob"] = p

        open_pnl      = self.broker.portfolio.unrealized_pnl(snap)
        atm           = snap.atm_strike(quoted_only=True)
        secs_remaining = len(snap._feed) - snap.i

        return {
            "sec":           sec,
            "secs_remaining": secs_remaining,
            "spot":          snap.spot,
            "atm":           atm,
            "range_prob":    c["range_prob"],
            "open_pnl":      open_pnl,
            "entry_credit":  c.get("entry_credit", 0.0),
            "decision_now":  decide,
        }
