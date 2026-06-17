"""
snapshot.py — one second of market state (array-backed).

A MarketSnapshot is a read-only view of a single row. It does NOT own per-strike
dicts; it holds the row index plus references to the Feed's shared numpy arrays
and the shared strike->column map. So building a snapshot is O(1) and a quote
lookup is a single array index — no dict is built per second.

Strikes are floats (indices like can have non-integer strikes).
"""


from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class MarketSnapshot:
    __slots__ = ('ts', 'spot', '_i', '_feed')
    ts    : int
    spot  : float
    _i    : int
    _feed : any

    @property
    def i(self) -> int:
        return self._i
    
    @property
    def atm(self) -> float:
        return float(self._feed.atm_strike[self._i])
    
    def field(self, strike, opt_type, field):
        """Current value of ANY loaded per-strike field (iv, delta, ttv, ...)."""
        col = self._feed.strike_to_col.get(strike)
        arr = self._feed.arrays.get((opt_type.lower(), field))
        if col is None or arr is None:
            return None
        v = arr[self._i, col]
        return None if v != v else float(v)           # NaN guard

    def quote(self, strike, opt_type):
        col = self._feed.strike_to_col.get(strike)
        if col is None: return None
        feed = self._feed
        ot   = opt_type.lower()
        if ot == "ce":
            b_arr, a_arr = feed._ce_bid, feed._ce_ask
        else:
            b_arr, a_arr = feed._pe_bid, feed._pe_ask
        if b_arr is None: return None
        i    = self._i
        b, a = b_arr[i, col], a_arr[i, col]
        if b != b or a != a: return None
        return float(b), float(a)

    def mid_and_half_spread(self, strike, opt_type):
        q = self.quote(strike, opt_type)              # (bid, ask) or None
        if q is None:
            return None
        bid, ask = q
        return 0.5 * (bid + ask), 0.5 * (ask - bid)   # (mid, half_spread)

    def _lo(self, n):
        return max(0, self._i - n + 1)

    def spot_hist(self, n):
        return self._feed.spot[self._lo(n): self._i + 1]

    def ts_hist(self, n):
        return self._feed.ts[self._lo(n): self._i + 1]

    def field_hist(self, strike, opt_type, field, n):
        """Last n values of ANY loaded field for one option, ending at now."""
        col = self._feed.strike_to_col.get(strike)
        arr = self._feed.arrays.get((opt_type.lower(), field))
        if col is None or arr is None:
            return None
        return arr[self._lo(n): self._i + 1, col]
