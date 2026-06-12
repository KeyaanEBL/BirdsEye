"""
costs.py — the friction model. Charges the three costs at FILL time only.

All numbers are PLACEHOLDERS — set them in config later. CostModel is pure:
it computes costs from a fill's price/lots/quote, holds no state.

Conventions:
  - lot_size          : contract multiplier         
  - txn_cost_bps      : transaction cost as bps of notional
  - brokerage_per_lot : flat brokerage per lot traded
  - spread cost is derived from the quote, not a parameter:
        half_spread = (ask - bid) / 2 ; cost = half_spread * lot_size * |lots|
    i.e. the gap you cross going from mid to the fill price, charged once.
"""


from dataclasses import dataclass


@dataclass
class CostModel:
    lot_size          : float = 1.0
    txn_cost_bps      : float = 0.0
    brokerage_per_lot : float = 0.0
    txn_cost_per_lot  : float = 0.0

    def transaction_cost(self, fill_price: float, lots: float) -> float:
        notional = abs(fill_price) * self.lot_size * abs(lots)
        return notional * self.txn_cost_bps * 1e-4 + self.txn_cost_per_lot * abs(lots)

    def brokerage(self, lots: float) -> float:
        return self.brokerage_per_lot * abs(lots)

    def spread_cost(self, half_spread: float, lots: float) -> float:
        return half_spread * self.lot_size * abs(lots)

    def execution_costs(self, half_spread: float, fill_price: float, lots: float) -> dict:
        """All three costs for one leg fill, as a dict (so the Tradelog keeps them split)."""
        return {
            "txn_cost"    : self.transaction_cost(fill_price, lots),
            "brokerage"   : self.brokerage(lots),
            "spread_cost" : self.spread_cost(half_spread, lots),
        }