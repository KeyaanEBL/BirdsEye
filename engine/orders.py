"""
order.py — an order is what a strategy emits: a bundle of option legs to trade.

It just states the legs and signed lots it wants, and hands the Order to the
execution/scheduler, which prices and fills it.

  - OrderLeg : one option — strike, opt_type ("CE"/"PE"), and signed lots
               (lots > 0 long, lots < 0 short; fractional allowed for ratios).
  - Reason   : structured record of WHY the FSM placed the order — the state
               that fired, the signal name, its value, and free-text notes.
  - Order    : a named bundle of legs (a single leg, a straddle, a strangle, a
               pyramid — all just different leg sets), plus its Reason.

Slicing (lots per second) and the pause between slices are NOT on the order —
they are strategy-level settings, applied by the scheduler when it works the
order. The Order only says *what* to trade, not *how fast*.
"""


from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .snapshot import MarketSnapshot

Quote = Optional[Tuple[float, float, float]]   # (bid, ask, mid)


@dataclass
class OrderLeg:
    strike     : float
    opt_type   : str            # "CE" or "PE"
    lots       : float = 1.0    
    action     : str   = "BUY"
    slice_lots : float = 1.0    # lots to release per tick
    pause      : int   = 0      # ticks to wait between slices

    def __post_init__(self):
        self.opt_type = self.opt_type.upper()
        self.action   = self.action.upper()
        
        assert self.lots > 0,       f"lots must be positive (got {self.lots})"
        assert self.slice_lots > 0, f"slice_lots must be positive (got {self.slice_lots})"
        assert self.pause >= 0,     f"pause must be >= 0 (got {self.pause})"
        assert self.action in ("BUY", "SELL")

    @property
    def signed_lots(self) -> float:
        return self.lots if self.action == "BUY" else -self.lots


@dataclass
class Reason:
    """Why the FSM placed this order — stamped at order time, carried to the Tradelog."""
    state  : str = ""                       # FSM state that emitted the order (e.g. "ENTERING")
    signal : str = ""                       # signal/alpha that fired          (e.g. "iv_skew")
    alphas : Optional[dict] = None          # the signal's values at the time
    note   : str = ""                       # free-text for anything extra

    def __str__(self) -> str:
        v    = "" if self.value is None else f"={self.value:.4g}"
        bits = [b for b in (self.state, f"{self.signal}{v}" if self.signal else "", self.note) if b]
        return " | ".join(bits)


@dataclass
class Order:
    legs   : List[OrderLeg]
    name   : str    = ""
    reason : Reason = field(default_factory=Reason)

    def __post_init__(self):
        if isinstance(self.reason, str):    # bare string -> structured note
            self.reason = Reason(note=self.reason)