"""plots.py — plotting helpers (matplotlib imported lazily). Read-only."""


from typing import List, Tuple
import numpy as np

EquityCurve = List[Tuple[int, float]]


def _plot_cum(ax, series, title, color):
    """One cumulative-P&L curve on `ax` with its drawdown (area below the running
    peak) shaded red. X axis is days, with no tick labels."""
    n = len(series)
    x = np.arange(n)
    ax.plot(x, series, color=color, lw=1.3, label=title)
    if n:
        peak = np.maximum.accumulate(series)
        ax.fill_between(x, series, peak, where=series < peak, color="tab:red",
                        alpha=0.25, interpolate=True, label="drawdown")
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_title(title); ax.set_ylabel("cum P&L ($)")
    ax.set_xticks([])                              # no labels on the x axis
    ax.legend(loc="best", fontsize=8)


def plot_equity(summary, axes=None):
    """Cumulative P&L broken down by cost layer — 4 panels:
      (1) gross        : mid-only, zero costs
      (2) mid - txn    : gross minus transaction costs only
      (3) mid - spread : gross minus spread costs only
      (4) net          : gross minus all costs
    Requires a Results object (not just the summary df) so it can
    pull per-fill cost breakdown from the tradelog."""

    import matplotlib.pyplot as plt

    if hasattr(summary, "summary"):
        # Results object — pull both summary and tradelog
        res = summary
        df  = res.summary
        tl  = res.tradelog   # cached_property, no ()

        # aggregate spread and txn costs per day from tradelog
        day_costs = (tl.groupby("day")[["spread_cost", "txn_cost"]]
                       .sum()
                       .rename(columns={"spread_cost": "spd($)",
                                        "txn_cost":    "txn($)"}))
        df = df.join(day_costs, how="left").fillna(0)
    else:
        # plain summary df — only combined costs available, can't split
        raise ValueError(
            "pass the Results object, not just summary — "
            "need tradelog for per-fill cost breakdown"
        )

    gross  = np.asarray(df["gross($)"], dtype=float)
    txn    = np.asarray(df["txn($)"],   dtype=float)
    spread = np.asarray(df["spd($)"],   dtype=float)

    curves = [
        (gross.cumsum(),                     "mid-only (no costs)",        "tab:green"),
        ((gross - txn).cumsum(),             "mid — txn costs only",       "tab:orange"),
        ((gross - spread).cumsum(),          "mid — spread costs only",    "tab:purple"),
        ((gross - txn - spread).cumsum(),    "net (all costs)",            "tab:blue"),
    ]

    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    for ax, (curve, title, color) in zip(axes, curves):
        _plot_cum(ax, curve, title, color)

    axes[-1].set_xlabel("days")
    return axes


def plot_drawdown(curve, ax=None):
    import matplotlib.pyplot as plt
    eq = np.array([e for _, e in curve], dtype=float)
    peak = np.maximum.accumulate(eq) if len(eq) else eq
    ax = ax or plt.subplots(figsize=(11, 3))[1]
    ax.fill_between(np.arange(len(eq)), eq - peak, 0, color="tab:red", alpha=0.4)
    ax.set_title("Drawdown"); ax.set_xlabel("tick"); ax.set_ylabel("eq - peak")
    return ax


def plot_daily_pnl(day_pnls, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(11, 3))[1]
    colors = ["tab:green" if v >= 0 else "tab:red" for v in day_pnls]
    ax.bar(range(len(day_pnls)), day_pnls, color=colors)
    ax.set_title("Daily PnL"); ax.set_xlabel("day"); ax.set_ylabel("pnl ($)")
    return ax