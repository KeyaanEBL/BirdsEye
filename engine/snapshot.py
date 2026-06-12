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


# AFTER
@dataclass(frozen=True)
class MarketSnapshot:
    ts    : int
    spot  : float
    _i    : int         # cursor: row index of "now"
    _feed : any         # whole day; all accessors clamp to [0, _i]

    @property
    def i(self) -> int:
        return self._i
    
    # AFTER
    def field(self, strike, opt_type, field):
        """Current value of ANY loaded per-strike field (iv, delta, ttv, ...)."""
        col = self._feed.strike_to_col.get(strike)
        arr = self._feed.arrays.get((opt_type.lower(), field))
        if col is None or arr is None:
            return None
        v = arr[self._i, col]
        return None if v != v else float(v)           # NaN guard

    def quote(self, strike, opt_type):
        b = self.field(strike, opt_type, "bid_0")
        a = self.field(strike, opt_type, "ask_0")
        if b is None or a is None:
            return None
        return b, a

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
        
    def atm_strike(self, quoted_only: bool = False) -> float:
        strikes = self._feed.strikes
        if quoted_only:
            i = self._i
            ok = np.ones(len(strikes), dtype=bool)
            for key in (("ce","bid_0"), ("ce","ask_0"), ("pe","bid_0"), ("pe","ask_0")):
                arr = self._feed.arrays.get(key)
                ok &= ~np.isnan(arr[i]) if arr is not None else False
            if ok.any():
                strikes = strikes[ok]               # only strikes quoted RIGHT NOW
        j = int(np.argmin(np.abs(strikes - self._feed.atm_strike[self._i])))
        return float(strikes[j])
