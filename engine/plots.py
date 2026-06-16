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
    """Day-by-day cumulative P&L across the whole run, built from the per-day
    summary table (Results.summary, or a Results). TWO SEPARATE plots, each with
    its own drawdown shaded red: (1) net($) — after costs, (2) gross($) — mid-only,
    no transaction/spread cost. Both cumulative-summed over days; x axis is days
    (no tick labels)."""
    import matplotlib.pyplot as plt
    df = summary.summary if hasattr(summary, "summary") else summary
    net   = np.asarray(df["net($)"],   dtype=float).cumsum()
    gross = np.asarray(df["gross($)"], dtype=float).cumsum()
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(11, 7))
    _plot_cum(axes[0], net,   "net P&L (after costs)",   "tab:blue")
    _plot_cum(axes[1], gross, "mid-only P&L (no costs)", "tab:green")
    axes[1].set_xlabel("days")
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