"""
portfolio.py — the book: positions, cash, realized/unrealized PnL, equity curve.

Positions keyed by (strike, opt_type). Weighted-average cost basis: realize PnL
when a position is reduced/closed, against the average entry. MtM marks open
positions at MID (never re-charging the spread — that was paid once at fill).

Per-unit P&L is in PRICE points * lot_size. Costs are subtracted from cash at
fill, so equity = cash + realized + unrealized already nets all frictions.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .snapshot import MarketSnapshot


@dataclass
class Position:
    lots      : float = 0.0     # signed net position
    avg_entry : float = 0.0     # weighted-average entry price (per unit)


class Portfolio:
    def __init__(self, lot_size: float = 1.0, max_lots: float = 1.0):
        self.lot_size     = lot_size
        self.max_lots     = max_lots
        self.cash         = 0.0
        self.realized_pnl = 0.0
        self.positions    : Dict[Tuple[float, str], Position] = {}
        self._active_keys  : set = set()
        self._equity_values  : List[float] = []
        self._exposure_values: List[float] = []

    # ---- fills -------------------------------------------------------------
    def apply_fill(self, fill) -> None:
        """Update position (weighted-avg), book realized PnL on reductions,
        and debit costs from cash. `fill` is a Tradelog.Fill."""
        key = (fill.strike, fill.opt_type)
        pos = self.positions.setdefault(key, Position())
        new_lots = pos.lots + fill.signed_lots

        ot         = fill.opt_type.lower()
        type_total = sum(abs(self.positions[k].lots) for k in self._active_keys if k[1].lower() == ot)
        projected  = type_total - abs(pos.lots) + abs(new_lots)
        if projected > self.max_lots:
            raise RuntimeError(f"max_lots exceeded: {fill.opt_type} projected={projected:.1f} limit={self.max_lots}")

        old_lots = pos.lots

        # realized PnL when reducing/closing (sign change or shrink toward 0)
        if old_lots != 0 and (old_lots > 0) != (fill.signed_lots > 0):
            closed             = min(abs(fill.signed_lots), abs(old_lots))
            direction          = 1.0 if old_lots > 0 else -1.0
            self.realized_pnl += direction * (fill.fill_price - pos.avg_entry) * closed * self.lot_size

        # update average entry: keep on adds/opens, reset if it flips through zero
        if old_lots == 0 or (old_lots > 0) == (fill.signed_lots > 0):          # opening/adding
            denom         = old_lots + fill.signed_lots
            pos.avg_entry = ((pos.avg_entry * old_lots) + (fill.fill_price * fill.signed_lots)) / denom if denom != 0 else 0.0
        elif (new_lots > 0) != (old_lots > 0) and new_lots != 0:        # flipped past zero
            pos.avg_entry = fill.fill_price

        # frictions hit cash immediately at fill
        pos.lots = new_lots
        if new_lots == 0:
            pos.avg_entry = 0.0
            self._active_keys.discard(key)
        else:
            self._active_keys.add(key)
        self.cash -= fill.execution_cost                              

    # ---- marking -----------------------------------------------------------
    def unrealized_pnl(self, snap: MarketSnapshot) -> float:
        total = 0.0
        for key in self._active_keys:
            pos = self.positions[key]
            mh  = snap.mid_and_half_spread(key[0], key[1])
            if mh is None: continue
            mid, _ = mh
            total += (mid - pos.avg_entry) * pos.lots * self.lot_size
        return total

    def equity(self, snap: MarketSnapshot) -> float:
        return self.cash + self.realized_pnl + self.unrealized_pnl(snap)

    def mark_to_market(self, snap: MarketSnapshot) -> float:
        eq = self.equity(snap)
        self._equity_values.append(eq)
        return eq
    
    def record_exposure(self, snap: MarketSnapshot) -> float:
        total = sum(abs(self.positions[k].lots) for k in self._active_keys)
        self._exposure_values.append(total)
        return total