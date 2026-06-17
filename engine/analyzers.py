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


def max_drawdown(day_pnls_net  : List[float],
                 day_pnls_gross: List[float],
                 margin_per_lot: float,
                 max_lots      : float) -> Tuple[float, float]:
    """Max drawdown as % of deployed margin, gross and net separately."""
    margin = margin_per_lot * max_lots
    if margin == 0:
        return 0.0, 0.0

    def _dd(pnls):
        cs   = np.cumsum(pnls)
        peak = np.maximum.accumulate(cs)
        return float((peak - cs).max()) * 100 / margin

    return _dd(day_pnls_gross), _dd(day_pnls_net)


def cagr(day_pnls         : List[float],
         total_costs      : float,
         margin_per_lot   : float,
         max_lots         : float,
         periods_per_year : int = 252) -> Tuple[float, float]:
    """(total_pnl * periods_per_year) / (margin * n_days), gross and net."""
    n      = len(day_pnls)
    margin = margin_per_lot * max_lots
    if n == 0 or margin == 0:
        return 0.0, 0.0
    net_pnl   = sum(day_pnls)
    gross_pnl = net_pnl + total_costs
    denom     = margin * n
    return (float(gross_pnl * periods_per_year / denom) * 100,
            float(net_pnl   * periods_per_year / denom) * 100)


def calmar(cagr_gross: float,
           cagr_net  : float,
           dd_gross  : float,
           dd_net    : float) -> Tuple[float, float]:
    """cagr / |maxDD| — pass already-computed values to avoid recomputing."""
    calmar_g = cagr_gross / dd_gross if dd_gross != 0 else float("nan")
    calmar_n = cagr_net   / dd_net   if dd_net   != 0 else float("nan")
    return calmar_g, calmar_n


def churn(total_traded_lots: float, max_lots: float) -> float:
    """Total lots traded / 2 — one complete open+close = 1."""
    return float(total_traded_lots / 4) / max_lots if max_lots != 0 else float("nan")


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

def time_in_market(exposure_curves: List[List[float]]) -> float:
    if not exposure_curves: return 0.0
    all_vals = np.concatenate([np.asarray(c, dtype=float) for c in exposure_curves])
    return float((all_vals > 0).mean()) if len(all_vals) else 0.0


def avg_hold_time(exposure_curves: List[List[float]]) -> float:
    segments = []
    for vals in exposure_curves:
        a = np.asarray(vals, dtype=float)
        if len(a) == 0: continue
        pad         = np.empty(len(a) + 2, dtype=np.int8)
        pad[0]      = pad[-1] = 0
        pad[1:-1]   = (a > 0).astype(np.int8)
        entries     = np.where((pad[:-1] == 0) & (pad[1:] != 0))[0]
        exits       = np.where((pad[:-1] != 0) & (pad[1:] == 0))[0]
        n           = min(len(entries), len(exits))
        if n: segments.append(exits[:n] - entries[:n])
    if not segments: return 0.0
    return float(np.concatenate(segments).mean())