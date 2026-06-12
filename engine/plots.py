"""plots.py — plotting helpers (matplotlib imported lazily). Read-only."""


from typing import List, Tuple
import numpy as np

EquityCurve = List[Tuple[int, float]]


def plot_equity(curve, ax=None, label=None):
    import matplotlib.pyplot as plt
    ts = np.arange(len(curve)); eq = np.array([e for _, e in curve], dtype=float)
    ax = ax or plt.subplots(figsize=(11, 4))[1]
    ax.plot(ts, eq, label=label); ax.set_title("Equity"); ax.set_xlabel("tick"); ax.set_ylabel("equity")
    if label: ax.legend()
    return ax


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