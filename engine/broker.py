"""
broker.py — turns an Order into recorded reality: fills, costs, Tradelog, book.
execute(order, snapshot): for each leg, read the quote, fill at the CROSSED side
(buy -> ask, sell -> bid), compute the three costs, write a Fill to the Tradelog,
and apply it to the portfolio. Fills the whole order at once for now — the
scheduler (tomorrow) will sit in FRONT of this and hand it sliced sub-orders,
with no change needed here.
"""


from typing import List
from .snapshot import MarketSnapshot
from .orders import Order, OrderLeg, Reason
from .costs import CostModel
from .ledger import Tradelog, Fill
from .portfolio import Portfolio


class Broker:
    def __init__(self, portfolio: Portfolio, costs: CostModel, tradelog: Tradelog = None):
        self.portfolio = portfolio
        self.costs     = costs
        self.tradelog  = tradelog or Tradelog()

    def execute(self, order: Order, snap: MarketSnapshot) -> List[Fill]:
        """Fill every leg of the order immediately at this second's quotes."""
        fills = []
        for leg in order.legs:
            mh = snap.mid_and_half_spread(leg.strike, leg.opt_type)
            if mh is None:
                continue
            mid, half_spread = mh
            costs = self.costs.execution_costs(half_spread, mid, leg.lots)
            fill  = Fill(
                ts          = snap.ts,
                strike      = leg.strike,
                opt_type    = leg.opt_type,
                lots        = leg.lots,
                action      = leg.action,
                fill_price  = mid,
                txn_cost    = costs["txn_cost"],
                brokerage   = costs["brokerage"],
                spread_cost = costs["spread_cost"],
                reason      = order.reason,
            )
            self.portfolio.apply_fill(fill)
            self.tradelog.add(fill)
            fills.append(fill)
        return fills

    def mark_to_market(self, snap: MarketSnapshot) -> float:
        self.portfolio.record_exposure(snap)
        return self.portfolio.mark_to_market(snap)

    def eod_square_off(self, snap) -> list:
        fills = []
        for (strike, opt_type), pos in self.portfolio.positions.items():
            if pos.lots == 0:
                continue
            leg = OrderLeg(
                strike,
                opt_type,
                lots       = abs(pos.lots),
                action     = "SELL" if pos.lots > 0 else "BUY",
                slice_lots = abs(pos.lots),
                pause      = 0,
            )
            order = Order(
                legs   = [leg],
                name   = "eod_square_off",
                reason = Reason(state="EOD", note="end of day force close"),
            )
            fills += self.execute(order, snap)
        return fills