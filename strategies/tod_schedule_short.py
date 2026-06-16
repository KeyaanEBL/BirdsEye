"""All-day +3/-1 short strangle — a DETERMINISTIC hold-to-expiry short, not learned.

Unlike `tod_instrument_short.py` (which learns the best instrument per time-of-day
bin and rolls on change), this strategy shorts ONE fixed instrument (the +3/-1
strangle) ONCE at the open and holds it all the way to the close:

    [0, close) -> SHORT +3/-1   (entered at the open, squared off a bar before EOD)

The +3/-1 strangle = SHORT CE three strikes above ATM + SHORT PE one strike below
ATM, struck at the open ATM. `lots` scales both legs. Mid fills + half-spread per
side — exactly BirdsEye's broker model.

NOTE: TRAIN only for now, same as the sibling strategy — keep split="train" until
you deliberately want to spend val.

Usage (train only)
------------------
    from engine import BirdsEye
    from strategies.tod_schedule_short import TodScheduleShort

    be = BirdsEye(
        strategy_cls    = TodScheduleShort,
        index           = "SPY",
        split           = "train",
        strategy_kwargs = {"lots": 10},          # +3/-1 (defaults)
        lot_size        = 100,
        starting_cash   = 1_000_000.0,
        n_workers       = 40,
    )
    res = be.run()

Tune the strikes via strategy_kwargs, e.g. {"lots": 10, "ce_off": 3, "pe_off": -1}.
"""

from typing import List, Optional, Tuple

import numpy as np

from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context

# (opt_type, signed strike-step offset from ATM, weight). weight > 0 = SHORT leg.
Leg = Tuple[str, int, float]


# ---------------------------------------------------------------------------
# FSM — enter once at the open, hold, square off a bar before the close
# ---------------------------------------------------------------------------

class _Wait(State):
    name = "WAIT"
    transitions = {"enter": "SHORT"}

    def target(self, _alphas, ctx):
        # on arriving at WAIT (from SHORT, at EOD): square off whatever we hold
        keys = ctx.get("open_keys")
        if not keys:
            return None
        note = ctx.get("exit_note", "square off")
        ctx["exit_note"] = None
        ctx["open_keys"] = None
        ctx["open_name"] = None
        return ctx["strat"].close_legs(keys, reason=Reason(state="WAIT", note=note))


class _Short(State):
    name = "SHORT"
    transitions = {"stop_loss": "WAIT", "eod": "WAIT"}

    def target(self, alphas, ctx):
        legs = ctx.get("pending_legs")          # resolved (strike, opt, weight)
        if not legs:
            return None
        order_legs = [OrderLeg(k, opt, lots=ctx["strat"].lots * abs(w),
                               action="SELL" if w > 0 else "BUY")
                      for (k, opt, w) in legs]
        ctx["open_keys"] = [(k, opt) for (k, opt, _) in legs]
        ctx["open_name"] = ctx.get("pending_name", "")
        ctx["exit_note"] = None
        return Order(name=f"tod_sched_short_{ctx.get('pending_name','')}", legs=order_legs,
                     reason=Reason(state="SHORT", signal=f"open:{ctx.get('pending_name','')}"))

    def on_enter(self, ctx):                    # runs at FILL-COMPLETE
        ctx["entry_sec"] = ctx["now_sec"]


class TodScheduleShort(StateMachineStrategy):
    """Short the +3/-1 strangle once at the open and hold it to expiry.

    ce_off/pe_off: strike-step offsets for the strangle legs (default +3 / -1).
    lots         : scales both legs.

    stop_loss_pct (optional, default None): exit when the open basket's loss reaches
    that multiple of the collected credit. Once stopped out, the day stays flat (no
    re-entry — there's nothing to re-enter on with a single all-day short).
    """
    states     = {"WAIT": _Wait(), "SHORT": _Short()}
    slice_lots = 100_000        # fill each (small) order in one tick
    pause      = 0

    def __init__(self, broker, lots: float = 1.0, ce_off: int = 2, pe_off: int = 0,
                 min_hold: int = 60, stop_loss_pct: Optional[float] = None):
        self.lots          = lots
        self.min_hold      = min_hold
        self.stop_loss_pct = stop_loss_pct
        self.inst_name     = f"sg {ce_off:+d}/{pe_off:+d}"
        self._legs: List[Leg] = [("CE", ce_off, 1.0), ("PE", pe_off, 1.0)]
        self._start_cash   = broker.portfolio.cash      # fresh book -> cum P&L baseline
        ctx = Context()
        ctx["open_keys"]    = None
        ctx["open_name"]    = None        # instrument currently held (None = flat)
        ctx["entered"]      = False       # one entry per day -> no re-short after square-off
        ctx["pending_legs"] = None
        ctx["pending_name"] = ""
        ctx["exit_note"]    = None
        super().__init__("WAIT", broker, name="tod_schedule_short", context=ctx)
        ctx["strat"] = self

    # ---- legs (the fixed +3/-1), resolved to live strikes ----
    def _resolve_legs(self, snap):
        """(strike, opt, weight) per leg, or None if a leg is off-grid / not quoted."""
        strikes = snap._feed.strikes
        ai = int(np.argmin(np.abs(strikes - snap._feed.atm_strike[snap.i])))
        out = []
        for opt, off, w in self._legs:
            j = ai + off
            if j < 0 or j >= len(strikes):
                return None                                  # off the listed grid
            k = float(strikes[j])
            if snap.quote(k, opt) is None:                   # not quoted -> can't fill
                return None
            out.append((k, opt, w))
        return out

    # ---- guards ----
    def guard_enter(self, a, c):
        # enter ONCE per day, at/after the open, with room before the close to fill
        # the entry and later square off.
        day_len = c.get("day_len", a["sec"])
        return (a["quoted"] and not c.get("entered") and not c.get("open_keys")
                and a["sec"] < day_len - self.min_hold)

    def guard_eod(self, a, c):
        day_len = c.get("day_len", a["sec"])
        if a["sec"] >= day_len - 2:
            c["exit_note"] = "square off (eod)"
            return True
        return False

    def guard_stop_loss(self, a, c):
        if self.stop_loss_pct is None:
            return False
        stop_loss = a.get("stop_loss")
        if not stop_loss or stop_loss <= 0:
            return False
        hit = max(0.0, -a.get("basket_pnl", 0.0)) >= stop_loss
        if hit:
            c["exit_note"] = f"stop loss {self.stop_loss_pct:g}x credit"
        return hit

    def _open_basket_metrics(self, snap):
        """Live PnL/loss and stop threshold for the currently open basket."""
        keys = self.context.get("open_keys") or []
        pnl = 0.0
        credit = 0.0
        for key in keys:
            pos = self.broker.portfolio.positions.get(key)
            if pos is None or pos.lots == 0:
                continue
            mh = snap.mid_and_half_spread(*key)
            if mh is None:
                continue
            mid, _ = mh
            lot_size = self.broker.portfolio.lot_size
            pnl += (mid - pos.avg_entry) * pos.lots * lot_size
            credit += -pos.lots * pos.avg_entry * lot_size

        loss = max(0.0, -pnl)
        stop_loss = 0.0
        if self.stop_loss_pct is not None and credit > 0:
            stop_loss = credit * self.stop_loss_pct
        return pnl, loss, credit, stop_loss

    # ---- alphas (every second) ----
    def compute_alphas(self, snap):
        c = self.context
        sec = snap.i
        c["now_sec"] = sec
        c["day_len"] = len(snap._feed)
        legs = self._resolve_legs(snap)
        # once we've held a position, mark the day's single entry as spent so we
        # don't re-short after a stop-out / square-off.
        if c.get("open_keys"):
            c["entered"] = True
        c["pending_legs"] = legs
        c["pending_name"] = self.inst_name
        pnl, loss, credit, stop_loss = self._open_basket_metrics(snap)
        cum_pnl = self.broker.portfolio.equity(snap) - self._start_cash   # running day P&L
        return {"sec": sec, "target_name": self.inst_name,
                "quoted": int(legs is not None),
                "cum_pnl": cum_pnl,
                "basket_pnl": pnl,
                "basket_credit": credit,
                "stop_loss": stop_loss}
