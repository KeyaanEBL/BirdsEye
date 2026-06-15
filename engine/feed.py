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

BID, ASK       = "bid_0", "ask_0"
DEFAULT_FIELDS = ('spot', 'atm_strike', 'ce_bid_0', 'ce_ask_0', 'pe_bid_0', 'pe_ask_0')

def _parse_strike(token: str):
    """Return float(token) if it's a (possibly decimal) number, else None."""
    try:
        return float(token)
    except ValueError:
        return None

def _split_fields(fields):
    """Split fields into plain column names and (opt_type, suffix) pairs.
    
    'spot'       -> plain field
    'atm_strike' -> plain field
    'ce_bid_0'   -> opt field: ('ce', 'bid_0')
    'pe_ask_0'   -> opt field: ('pe', 'ask_0')
    """
    plain, opt = [], []
    for f in fields:
        if f.startswith('ce_'):
            opt.append(('ce', f[3:]))
        elif f.startswith('pe_'):
            opt.append(('pe', f[3:]))
        else:
            plain.append(f)
    return plain, opt

def _merge_fields(fields):
    """Always include DEFAULT_FIELDS; user fields are additive on top."""
    return tuple(dict.fromkeys((*DEFAULT_FIELDS, *fields)))

class Feed:
    def __init__(self, df: pd.DataFrame, strikes: List[float], base=None, fields=DEFAULT_FIELDS, tokens=None):
        fields = _merge_fields(fields)
        if tokens is None:                       # constructed directly from a df
            _, tokens = self._discover_strikes(list(df.columns))
        self.strike_tokens = tokens
        self.strikes       = np.asarray(sorted(strikes), dtype=float)
        self.strike_to_col: Dict[float, int] = {float(k): i for i, k in enumerate(self.strikes)}
        self.ts     = df['timestamp'].to_numpy(dtype=np.int64)
        self.fields = tuple(fields)

        plain_fields, opt_fields = _split_fields(self.fields)

        self.arrays: Dict[object, np.ndarray] = {}
        for f in plain_fields:
            if f in df.columns:
                self.arrays[f] = df[f].to_numpy(dtype=float)

        for (ot, suf) in opt_fields:
            self.arrays[(ot, suf)] = self._stack(df, ot, suf)

        # shortcuts — None if the field wasn't requested
        self.spot       = self.arrays.get('spot',       np.full(len(self.ts), np.nan))
        self.atm_strike = self.arrays.get('atm_strike', np.full(len(self.ts), np.nan))
        self.ce_bid     = self.arrays.get(('ce', BID))
        self.ce_ask     = self.arrays.get(('ce', ASK))
        self.pe_bid     = self.arrays.get(('pe', BID))
        self.pe_ask     = self.arrays.get(('pe', ASK))

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_parquet(cls, path: str, fields=DEFAULT_FIELDS) -> "Feed":
        columns = pq.ParquetFile(path).schema.names
        strikes, tokens = cls._discover_strikes(columns)
        fields = _merge_fields(fields)
        plain_fields, opt_fields = _split_fields(fields)
        
        wanted = set()
        for f in plain_fields:
            if f in columns:
                wanted.add(f)

        for strike in strikes:
            tok = tokens[strike]
            for (ot, suf) in opt_fields:
                col = f"{tok}_{ot}_{suf}"
                if col in columns:
                    wanted.add(col)
        df = pd.read_parquet(path, columns=list(wanted))
        df = df.reset_index()
        return cls(df, strikes, fields=fields, tokens=tokens)
    
    @classmethod
    def from_csv(cls, path: str, fields=DEFAULT_FIELDS) -> "Feed":
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        strikes, tokens = cls._discover_strikes(columns)
        fields = _merge_fields(fields)
        plain_fields, opt_fields = _split_fields(fields)
        wanted = set()
        for f in plain_fields:
            if f in columns:
                wanted.add(f)

        for strike in strikes:
            tok = tokens[strike]
            for (ot, suf) in opt_fields:
                col = f"{tok}_{ot}_{suf}"
                if col in columns:
                    wanted.add(col)

        df = pd.read_csv(path, usecols=list(wanted))
        return cls(df, strikes, fields=fields, tokens=tokens)

    @classmethod
    def from_file(cls, path: str, fields=DEFAULT_FIELDS) -> "Feed":
        if path.endswith('.csv'):
            return cls.from_csv(path, fields)
        return cls.from_parquet(path, fields)

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