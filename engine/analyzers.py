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
    }


def cagr(day_pnls         : List[float],
         total_costs      : float,
         margin_per_lot   : float,
         max_lots         : float,
         periods_per_year : int = 252) -> Tuple[float, float]:
    """
    Annualised return on margin, two variants.
      gross : mid_pnl              (net + costs, no friction)
      net   : mid_pnl - total_costs
    Formula: (total_pnl * periods_per_year) / (margin * n_days)
    margin = margin_per_lot * max_lots
    periods_per_year: 252 SPY daily, 52 NIFTY/SENSEX weekly-expiry
    """
    n      = len(day_pnls)
    margin = margin_per_lot * max_lots
    if n == 0 or margin == 0:
        return 0.0, 0.0
    net_pnl   = sum(day_pnls)
    gross_pnl = net_pnl + total_costs
    denom     = margin * n
    return float(gross_pnl * periods_per_year / denom) * 100, \
           float(net_pnl   * periods_per_year / denom) * 100


def max_drawdown(day_pnls       : List[float],
                     margin_per_lot : float,
                     max_lots       : float) -> float:
    """Max drawdown as a fraction of deployed margin."""
    cs   = np.cumsum(day_pnls)
    peak = np.maximum.accumulate(cs)
    margin = margin_per_lot * max_lots
    if margin == 0:
        return 0.0
    return float((peak - cs).max()) * 100 / margin


def calmar(day_pnls         : List[float],
           total_costs      : float,
           margin_per_lot   : float,
           max_lots         : float,
           periods_per_year : int = 252) -> Tuple[float, float]:
    """calmar_gross, calmar_net = cagr / |max_drawdown_pct|."""
    dd = abs(max_drawdown(day_pnls, margin_per_lot, max_lots))
    if dd == 0:
        return float("nan"), float("nan")
    cagr_gross, cagr_net = cagr(day_pnls, total_costs, margin_per_lot, max_lots, periods_per_year)
    return cagr_gross / dd, cagr_net / dd


def churn(total_traded_lots: float) -> float:
    """
    Round-trip count: total lots traded (all buys + sells) / 2.
    churn=1 -> one complete open+close cycle across all legs.
    """
    return float(total_traded_lots / 2)


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