"""execution.py — execution IS a state. Executing slices ONE order into the
broker over ticks, then the FSM completes the move to `dest`. Pure countdown
(no portfolio reads, no queue). Includes the starvation guard: if it releases
slices for 60 straight ticks with zero fills, it raises naming the legs."""


from typing import Dict, Tuple
from .orders import Order, OrderLeg

STARVE_LIMIT = 60      # ticks of zero fills before failing loudly


class Executing:
    name = "EXECUTING"

    def __init__(self, order: Order, dest: str):
        self.order, self.dest = order, dest
        self.remaining        : Dict[Tuple[float, str], float] = {}   # signed lots left
        for leg in order.legs:
            key                 = (leg.strike, leg.opt_type)
            self.remaining[key] = self.remaining.get(key, 0.0) + leg.signed_lots
        self._pause_ctr = 0
        self._starved   = 0

    def tick(self, snap, broker, slice_lots: float, pause: int):
        if self._pause_ctr > 0:
            self._pause_ctr -= 1
            return []
        legs = []

        for key, rem in self.remaining.items():
            if rem == 0:
                continue
            step = min(slice_lots, abs(rem))
            legs.append(OrderLeg(key[0], key[1], lots = step, action = "BUY" if rem > 0 else "SELL"))
        
        fills = []
        if legs:
            sub = Order(legs = legs, name = self.order.name, reason = self.order.reason)
            fills = broker.execute(sub, snap)
            for f in fills:
                self.remaining[(f.strike, f.opt_type)] -= f.signed_lots
            self._pause_ctr = pause
        
        if legs and not fills:
            self._starved += 1
            if self._starved >= STARVE_LIMIT:
                starving = [k for k, r in self.remaining.items() if r != 0]
                raise RuntimeError(f"Executing starved: no quotes for {starving}")
        else:
            self._starved = 0
        
        return fills

    def is_done(self) -> bool:
        return all(r == 0 for r in self.remaining.values())