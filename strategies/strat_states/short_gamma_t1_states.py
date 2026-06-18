"""
short_gamma_t1_states.py — FSM states for the Type-1-only Short-Gamma strategy.

Per the project convention, the strategy's State subclasses live HERE, not in the
strategy module. There is a single working state, RUN, that self-loops: every
non-executing tick the strategy decides what (if anything) to trade and stashes a
"plan" on the context; RUN emits at most one (multi-leg) order per tick to enact
it. All the position/cooldown/pair bookkeeping is committed at fill-complete via
the strategy's _commit hook (on_enter fires once the order is fully filled).

The states stay thin and strategy-agnostic: they read the plan and call back into
ctx["strat"], so this same skeleton extends unchanged to the full 3-type version.
"""
from engine import State, Order, Reason


class Run(State):
    name = "RUN"
    # self-loop: act whenever the strategy has staged a non-empty plan this tick.
    transitions = {"act": "RUN"}

    def on_enter(self, ctx):
        # Fires at fill-complete (and once at construction). Commit the order that
        # just filled into the strategy's sub-ledger / cooldown state.
        strat = ctx.get("strat")
        if strat is not None:
            strat._commit(ctx)

    def target(self, alphas, ctx):
        plan = ctx.get("t1_plan")
        if not plan or not plan.get("legs"):
            return None
        ctx["t1_plan"]     = None          # consumed
        ctx["t1_inflight"] = plan          # _commit (on_enter) applies it post-fill
        return Order(
            legs   = plan["legs"],
            name   = plan["event"],
            reason = Reason(state=plan["state"], signal=plan["signal"],
                            note=plan["note"]),
        )
