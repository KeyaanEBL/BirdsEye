"""execution.py — execution IS a state. Executing slices ONE order into the
broker over ticks, then the FSM completes the move to `dest`. Pure countdown
(no portfolio reads, no queue). Includes the starvation guard: if it releases
slices for 60 straight ticks with zero fills, it raises naming the legs."""


from typing import Dict, Tuple
from .orders import Order, OrderLeg

STARVE_LIMIT = 60

class Executing:
    name = "EXECUTING"

    def __init__(self, order: Order, dest: str):
        self.order, self.dest = order, dest
        self.remaining    : Dict[Tuple[float, str], float] = {}
        self._slice_lots  : Dict[Tuple[float, str], float] = {}
        self._pause       : Dict[Tuple[float, str], int]   = {}
        self._pause_ctrs  : Dict[Tuple[float, str], int]   = {}

        for leg in order.legs:
            key = (leg.strike, leg.opt_type)
            self.remaining[key]   = self.remaining.get(key, 0.0) + leg.signed_lots
            self._slice_lots[key] = leg.slice_lots
            self._pause[key]      = leg.pause
            self._pause_ctrs[key] = 0

        self._starved = 0

    def tick(self, snap, broker) -> list:          # no more slice_lots/pause args
        legs = []
        for key, rem in self.remaining.items():
            if rem == 0:
                continue
            if self._pause_ctrs[key] > 0:
                self._pause_ctrs[key] -= 1
                continue
            step = min(self._slice_lots[key], abs(rem))
            legs.append(OrderLeg(
                key[0], key[1],
                lots       = step,
                action     = "BUY" if rem > 0 else "SELL",
                slice_lots = self._slice_lots[key],
                pause      = self._pause[key],
            ))

        fills = []
        if legs:
            sub   = Order(legs=legs, name=self.order.name, reason=self.order.reason)
            fills = broker.execute(sub, snap)
            for f in fills:
                k = (f.strike, f.opt_type)
                self.remaining[k]  -= f.signed_lots
                self._pause_ctrs[k] = self._pause[k]

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