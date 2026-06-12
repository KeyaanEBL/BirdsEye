"""
Tradelog.py — the immutable trade record. Single source of truth for fills.

Every executed leg becomes one Fill row. PnL, costs, and stats all derive from
the Tradelog downstream, so it stays pure storage: append + read, no logic.
"""


from dataclasses import dataclass, field
from typing import List
import pandas as pd

from .orders import Reason


@dataclass
class Fill:
    ts          : int
    strike      : float
    opt_type    : str        
    lots        : float
    action      : str
    fill_price  : float      
    txn_cost    : float
    brokerage   : float
    spread_cost : float
    reason      : Reason = field(default_factory=Reason)

    @property
    def execution_cost(self) -> float:
        return self.txn_cost + self.brokerage + self.spread_cost
    
    @property
    def signed_lots(self) -> float:
        return self.lots if self.action == "BUY" else -self.lots


class Tradelog:
    def __init__(self):
        self.fills : List[Fill] = []

    def add(self, fill: Fill) -> None:
        self.fills.append(fill)

    @property
    def total_costs(self) -> float:
        return sum(f.execution_cost for f in self.fills)
    
    def total_notional(self, lots_size: float) -> float:
        return sum(f.fill_price * lots_size * f.lots for f in self.fills)

    def as_dataframe(self) -> pd.DataFrame:
        rows, alpha_rows = [], []
        for f in self.fills:
            rows.append({
                "timestamp"   : f.ts,
                "strike"      : f.strike,
                "opt_type"    : f.opt_type,
                "action"      : f.action,
                "lots"        : f.lots,
                "fill_price"  : f.fill_price,
                "txn_cost"    : f.txn_cost,
                "brokerage"   : f.brokerage,
                "spread_cost" : f.spread_cost,
                "exe_cost"    : f.execution_cost,
                "state"       : f.reason.state,
                "signal"      : f.reason.signal,
                "note"        : f.reason.note,
            })
            alpha_rows.append({f"alpha_{k}": v for k, v in (f.reason.alphas or {}).items()})
        df = pd.DataFrame(rows)
        if any(alpha_rows):
            df = pd.concat([df, pd.DataFrame(alpha_rows)], axis=1)
        return df.round(2)
 
 
class PerSecLog:
    """Per-second record: ts, i, spot, atm, FSM state, and every alpha value."""
    def __init__(self):
        self.rows: List[dict] = []
 
    def add(self, **row) -> None:
        self.rows.append(row)
 
    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).round(2)
