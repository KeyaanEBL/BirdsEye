"""
feed.py — turn a day's parquet into a stream of array-backed MarketSnapshots.

- Loads ONLY the needed columns from the parquet (timestamp, spot, atm_strike,
  and per-strike CE/PE bid_0/ask_0), never the whole file.
- Discovers strikes from the `{strike}_{opt_type}_premium` column names
  (decimal-safe), then stacks bid/ask into shared (n_rows, n_strikes) arrays
  once. Per-second iteration just hands out row views — no per-row dicts.
- Assumes bid_0 / ask_0 are present (no synthesis).
"""


from typing import Dict, Iterator, List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .snapshot import MarketSnapshot

BID, ASK, IV = "bid_0", "ask_0", "iv"          # quote column suffixes in the data
DEFAULT_FIELDS = (BID, ASK)

def _parse_strike(token: str):
    """Return float(token) if it's a (possibly decimal) number, else None."""
    try:
        return float(token)
    except ValueError:
        return None


class Feed:
    def __init__(self, df: pd.DataFrame, strikes: List[float], base=None, fields=DEFAULT_FIELDS, tokens=None):
        if tokens is None:                       # constructed directly from a df
            _, tokens = self._discover_strikes(list(df.columns))
        self.strike_tokens = tokens
        self.strikes    = np.asarray(sorted(strikes), dtype=float)
        self.strike_to_col: Dict[float, int] = {float(k): i for i, k in enumerate(self.strikes)}
        self.ts         = df['timestamp'].to_numpy(dtype=np.int64)
        self.spot       = df['spot'].to_numpy(dtype=float)
        self.atm_strike = df['atm_strike'].to_numpy(dtype=float)
        
        self.fields = tuple(fields)
        self.arrays = {}                              
        for ot in ("ce", "pe"):
            for f in self.fields:
                self.arrays[(ot, f)] = self._stack(df, ot, f)
        
        self.ce_bid, self.ce_ask = self.arrays[("ce", BID)], self.arrays[("ce", ASK)]
        self.pe_bid, self.pe_ask = self.arrays[("pe", BID)], self.arrays[("pe", ASK)]

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_parquet(cls, path: str, fields=DEFAULT_FIELDS) -> "Feed":
        columns = pq.ParquetFile(path).schema.names
        strikes, tokens = cls._discover_strikes(columns)
        base = [col for col in ('spot', 'atm_strike') if col in columns]
        wanted = set(base)
        
        for strike in strikes:
            tok = tokens[strike]                       # original token for this strike
            for ot in ("ce", "pe"):
                for suf in fields:
                    col = f"{tok}_{ot}_{suf}"
                    if col in columns:
                        wanted.add(col)

        df = pd.read_parquet(path, columns=list(wanted))
        df = df.reset_index()
        return cls(df, strikes, base, fields, tokens)

    @staticmethod
    def _discover_strikes(columns: List[str]) -> List[float]:
        tokens = {}                          # float strike -> original token, e.g. 471.0 -> "471.00"
        for c in columns:
            if c.endswith("_premium"):
                tok = c.split("_")[0]
                k = _parse_strike(tok)
                if k is not None:
                    tokens[k] = tok
        return sorted(tokens), tokens

    def _stack(self, df, opt_type, field):
        n = len(df)
        arr = np.full((n, len(self.strikes)), np.nan)
        cols = set(df.columns)
        for i, k in enumerate(self.strikes):
            tok = self.strike_tokens[k]
            col = f"{tok}_{opt_type}_{field}"
            if col in cols:
                arr[:, i] = df[col].to_numpy(dtype=float)
        return arr

    # ---- iteration ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ts)

    def __iter__(self) -> Iterator[MarketSnapshot]:
        for i in range(len(self.ts)):
            yield MarketSnapshot(ts = int(self.ts[i]), spot = float(self.spot[i]), _i = i, _feed = self)