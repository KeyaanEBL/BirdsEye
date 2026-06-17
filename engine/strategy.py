"""
strategy.py — a strategy is a finite state machine that emits Orders.

The FSM is STATELESS about execution: each tick it computes its alphas, may
transition to a new state, and forwards an Order (an updated target position)
to the scheduler. It tracks no fills/residuals — the scheduler owns all of that.
slice_lots and pause are strategy-level: every order this strategy fires is
worked by the scheduler with these same two params.

Define a strategy by subclassing StateMachineStrategy and declaring:
  - states      : name -> State   (each says what to hold while in it)
  - transitions : [Transition]    (guarded edges; first matching guard wins)
  - compute_alphas(snap)          (the per-second alpha values)
  - slice_lots, pause             (execution params, read by the scheduler)
"""


from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .snapshot import MarketSnapshot
from .orders import Order, OrderLeg
from .execution import Executing
from .ledger import PerSecLog


@dataclass
class Context:
    """Rolling memory across ticks. Freeform; states/guards read & write it."""
    data: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, k):           return self.data[k]
    def __setitem__(self, k, v):        self.data[k] = v
    def get(self, k, default=None):     return self.data.get(k, default)


@dataclass
class Transition:
    dest  : str
    guard : Callable[[Dict[str, float], Context], bool]


class State:
    name: str = ""
    transitions: Dict[Union[str, Callable], str] = {}   # guard -> dest, first hit wins
    # guard is a string name (resolved to strategy method guard_<name>)
    # or a callable(alphas, ctx) -> bool (quick lambdas still work)
    
    def target(self, alphas, ctx): return None
    def on_enter(self, ctx: Context) -> None:   pass
    def on_exit(self, ctx: Context)  -> None:   pass


class StateMachineStrategy:
    states      : Dict[str, State] = {}
    min_history : int   = 0                                 # warm-up: skip ticks until enough history

    def __init__(self, initial_state, broker, name="", context=None):
        self.state      = initial_state
        self.broker     = broker                            # strategy owns its broker ref now
        self.name       = name or type(self).__name__
        self.context    = context or Context()
        self._executing : Optional[Executing] = None        # Executing instance while an order works
        self.perseclog    = PerSecLog()
        self._guard_cache : dict = {}
        self.states[self.state].on_enter(self.context)

    def next(self, snap):
        if snap.i < self.min_history:                       # warm-up
            self._log(snap, None)
            return
        
        if self._executing is not None:                     # execution IS a state
            self._executing.tick(snap, self.broker)
            if self._executing.is_done():
                dest            = self._executing.dest
                self._executing = None
                self.state      = dest
                self.states[dest].on_enter(self.context)    # enter at fill-complete
            return
        
        alphas = self.compute_alphas(snap)
        self._log(snap, alphas)

        for key, dest in self._transition_items(self.states[self.state]):
            guard, gname = self._resolve_guard(key)
            if guard(alphas, self.context):
                self.states[self.state].on_exit(self.context)
                order = self.states[dest].target(alphas, self.context)
                if order is None:
                    self.state = dest
                    self.states[dest].on_enter(self.context)
                else:
                    order.reason.strategy = self.name
                    if not order.reason.signal:
                        order.reason.signal = gname
                    if not order.reason.alphas:             # log gets the alpha VALUES at fire time
                        order.reason.alphas = {k: round(v, 2) for k, v in alphas.items() if isinstance(v, (int, float))}
                    self._executing = Executing(order, dest)
                    self.state = "EXECUTING"
                break
    
    def close_legs(self, keys, reason="", slice_lots=1.0, pause=0):
        legs = []
        for key in keys:
            pos = self.broker.portfolio.positions.get(key)
            if pos is not None and pos.lots != 0:
                legs.append(OrderLeg(
                    key[0], key[1],
                    lots       = abs(pos.lots),
                    action     = "SELL" if pos.lots > 0 else "BUY",
                    slice_lots = slice_lots,
                    pause      = pause,
                ))
        return Order(legs=legs, name="close", reason=reason) if legs else None
    
    @staticmethod
    def _transition_items(state):
        t = state.transitions
        if isinstance(t, dict):
            return list(t.items())                       # insertion order = priority
        return [(tr.guard, tr.dest) for tr in t]         # legacy [Transition] list

    def _resolve_guard(self, key):
        if key in self._guard_cache:
            return self._guard_cache[key]
        result = (key, getattr(key, "__name__", "guard")) if callable(key) \
                else (getattr(self, f"guard_{key}"), key)
        self._guard_cache[key] = result
        return result
    
    def _log(self, snap, alphas):
        row = {"timestamp": snap.ts, "spot": snap.spot, "atm": snap.atm, "state": self.state}
        if alphas:
            row.update({k: v for k, v in alphas.items() if isinstance(v, (int, float))})
        self.perseclog.add(**row)

    def describe(self) -> str:
        """The state machine as a readable table: state --[guard]--> dest."""
        lines = []
        for name, st in self.states.items():
            for key, dest in self._transition_items(st):
                g = key if isinstance(key, str) else getattr(key, "__name__", "<lambda>")
                lines.append(f"{name:>10s} --[{g}]--> {dest}")
        return "\n".join(lines)