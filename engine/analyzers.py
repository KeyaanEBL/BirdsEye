"""analyzers.py — performance stats from equity curve(s) + Tradelog. Pure funcs."""


from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from .ledger import Tradelog

EquityCurve = List[Tuple[int, float]]


def _eq(curve): 
    ts = np.array([t for t, _ in curve], dtype=np.int64)
    eq = np.array([e for _, e in curve], dtype=float)
    return ts, eq


def max_drawdown(curve: EquityCurve) -> float:
    _, eq = _eq(curve)
    if len(eq) == 0: return 0.0
    peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def total_return(curve: EquityCurve) -> float:
    _, eq = _eq(curve)
    return float(eq[-1] - eq[0]) if len(eq) else 0.0


def daily_stats(day_pnls: List[float]) -> Dict[str, float]:
    """Stats over a list of per-day PnLs."""
    a = np.array(day_pnls, dtype=float)
    
    if len(a) == 0: return {}
    up, dn = a[a > 0], a[a < 0]
    return {
        "n_days":       int(len(a)),
        "total_pnl":    float(a.sum()),
        "avg_day":      float(a.mean()),
        "pct_pos_days": float(len(up) / len(a)),
        "pct_neg_days": float(len(dn) / len(a)),
        "avg_win":      float(up.mean()) if len(up) else 0.0,
        "avg_loss":     float(dn.mean()) if len(dn) else 0.0,
        "best_day":     float(a.max()),
        "worst_day":    float(a.min()),
        "win_rate":     float(len(up) / len(a)),
    }


def cagr(day_pnls, capital, periods_per_year=252):
    r      = np.array(day_pnls) / capital
    growth = np.prod(1.0 + r)
    
    if growth <= 0: return -1.0
    return float(growth ** (periods_per_year / len(r)) - 1.0)


def max_drawdown_pct(day_pnls, capital):
    eq   = capital * np.cumprod(1.0 + np.array(day_pnls) / capital)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def calmar(day_pnls, capital, periods_per_year=252):    # classic: CAGR / |maxDD%|
    dd = abs(max_drawdown_pct(day_pnls, capital))
    return float("nan") if dd == 0 else cagr(day_pnls, capital, periods_per_year) / dd


def churn(total_notional, capital, n_days):
    """Capital cycles/day: total traded value / (capital * n_days).
    churn=2 -> an average day trades 2x the capital in and out."""
    return float(total_notional / (capital * n_days))


def cost_stats(Tradelog: Tradelog) -> Dict[str, float]:
    df = Tradelog.as_dataframe()
    if df.empty: return {"n_fills": 0, "total_cost": 0.0}
    return {
        "n_fills"     : int(len(df)),
        "txn_cost"    : float(df["txn_cost"].sum()),
        "spread_cost" : float(df["spread_cost"].sum()),
        "brokerage"   : float(df["brokerage"].sum()),
        "total_cost"  : float(df["cost"].sum()),
    }