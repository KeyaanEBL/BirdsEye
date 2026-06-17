"""
feed.py — turn a day's feed file into a stream of array-backed MarketSnapshots.

PRIMARY path (raw /mnt, my preprocessing): `Feed.from_raw(path, index, fields)`
reads the raw source the SAME way markouts_custom_instruments.ipynb does —
through Intern-Project's `data.load_columns`, which preprocesses on the fly
(session filter, 0->NaN/ffill cleaning, decimal-strike normalisation) and returns
standardised columns: plain (`spot`, `atm_strike`) and strike-wildcard dicts
(`*_ce_bid_0`, `*_pe_ask_0`, …) keyed by integer strike. No preprocessed parquet
directory is assumed; the manifest only assigns dates to splits.

FIELD MODEL: a `fields` entry is either a PLAIN column
('spot', 'atm_strike') or an OPTION field ('ce_bid_0', 'pe_ask_0' -> (opt, suf)).
`_merge_fields` always folds in DEFAULT_FIELDS, so spot/atm_strike + ce/pe bid/ask
are present regardless of what the caller asks for; extra fields are additive.

All per-strike fields are stacked into shared (n_rows, n_strikes) numpy arrays
once, so per-second iteration just hands out row views — no per-row dicts.
Arrays are keyed by the plain name (1-D) or by (opt_type, suffix) (2-D).
"""

from typing import Dict, Iterator
import numpy as np

from .snapshot import MarketSnapshot
from .env import load_columns, get_column_map

BID, ASK = "bid_0", "ask_0"
DEFAULT_FIELDS = ("spot", "atm_strike", "ce_bid_0", "ce_ask_0", "pe_bid_0", "pe_ask_0")


def _split_fields(fields):
    """Split `fields` into plain column names and (opt_type, suffix) pairs.
        'spot' / 'atm_strike' -> plain   ;   'ce_bid_0' -> ('ce','bid_0')."""
    plain, opt = [], []
    for f in fields:
        if f.startswith("ce_"):
            opt.append(("ce", f[3:]))
        elif f.startswith("pe_"):
            opt.append(("pe", f[3:]))
        else:
            plain.append(f)
    return plain, opt


def _merge_fields(fields):
    """Always include DEFAULT_FIELDS; user fields are additive on top (order-stable)."""
    return tuple(dict.fromkeys((*DEFAULT_FIELDS, *fields)))


class Feed:
    """Array-backed view of one day. Build it from a raw /mnt file with
    `Feed.from_raw(path, index, fields)`, or directly from preassembled arrays
    with `from_arrays` (tests)."""

    def __init__(self, ts, strikes, arrays: Dict[object, np.ndarray], fields):
        self.ts      = np.asarray(ts, dtype=np.int64)
        self.strikes = np.asarray(sorted(strikes), dtype=float)
        self.strike_to_col: Dict[float, int] = {float(k): i for i, k in enumerate(self.strikes)}
        self.fields  = tuple(fields)
        self.arrays  = arrays            # plain name -> (n,) ; (opt, suffix) -> (n, n_strikes)

        n = len(self.ts)
        # plain shortcuts (NaN-filled if the field wasn't loaded)
        self.spot       = arrays.get("spot",       np.full(n, np.nan))
        self.atm_strike = arrays.get("atm_strike", np.full(n, np.nan))
        # quote shortcuts (None if not requested)
        self.ce_bid, self.ce_ask = arrays.get(("ce", BID)), arrays.get(("ce", ASK))
        self.pe_bid, self.pe_ask = arrays.get(("pe", BID)), arrays.get(("pe", ASK))
        self._ce_bid = self.ce_bid   # direct refs for quote() — no dict lookup per tick
        self._ce_ask = self.ce_ask
        self._pe_bid = self.pe_bid
        self._pe_ask = self.pe_ask

    # ---- construction: raw /mnt via the Intern-Project pipeline (primary) ---
    @classmethod
    def from_raw(cls, path: str, index: str, fields=DEFAULT_FIELDS) -> "Feed":
        """Load+preprocess a raw /mnt day for `index` (e.g. 'SPY') via
        `data.load_columns`. Plain fields come back as 1-D arrays; option fields
        as strike-wildcard dicts that get stacked into (n, n_strikes) arrays.
        Raises ValueError on a day with no usable bars (caller skips it)."""
        fields     = _merge_fields(fields)
        plain, opt = _split_fields(fields)
        cmap       = get_column_map(index)

        # only request plain fields the column map can actually resolve (so a
        # stale bare 'bid_0'/'ask_0' from older callers is silently dropped — the
        # option defaults below still provide ce/pe bid/ask).
        plain = [f for f in plain if _resolvable(cmap, f)]
        wild  = [f"*_{ot}_{suf}" for (ot, suf) in opt]
        cols  = load_columns(path, [*plain, *wild], cmap, index)

        arrays: Dict[object, np.ndarray] = {}
        n = None
        for f in plain:
            arrays[f] = np.asarray(cols[f], dtype=float)
            n = len(arrays[f])

        # strike universe = union of strikes quoted in any requested option field
        strike_set = set()
        for (ot, suf) in opt:
            strike_set |= set(cols[f"*_{ot}_{suf}"].keys())
        strikes = sorted(float(s) for s in strike_set)
        col_of  = {float(k): i for i, k in enumerate(strikes)}

        if n is None:                                  # no plain field -> infer n from an option dict
            n = 0
            for (ot, suf) in opt:
                d = cols[f"*_{ot}_{suf}"]
                if d:
                    n = len(next(iter(d.values())))
                    break

        for (ot, suf) in opt:
            arr = np.full((n, len(strikes)), np.nan)
            for s, series in cols[f"*_{ot}_{suf}"].items():
                arr[:, col_of[float(s)]] = series
            arrays[(ot, suf)] = arr

        # bar index is seconds-from-open after the session filter — what the
        # strategy uses as "now". Real epoch timestamps aren't needed downstream.
        ts = np.arange(n, dtype=np.int64)
        return cls(ts, strikes, arrays, fields)

    @classmethod
    def from_arrays(cls, ts, spot, atm_strike, strikes, arrays, fields=DEFAULT_FIELDS) -> "Feed":
        """Build a Feed from preassembled arrays (tests / synthetic days). `spot`
        and `atm_strike` are folded into the arrays dict if not already there."""
        a = dict(arrays)
        a.setdefault("spot", spot)
        a.setdefault("atm_strike", atm_strike)
        return cls(ts, strikes, a, fields)

    # ---- iteration ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ts)

    def __iter__(self) -> Iterator[MarketSnapshot]:
        for i in range(len(self.ts)):
            yield MarketSnapshot(ts=int(self.ts[i]), spot=float(self.spot[i]), _i=i, _feed=self)


def _resolvable(cmap, field) -> bool:
    """True if the column map can resolve a plain abstract field name (e.g. 'spot',
    'atm_strike'); False for stray suffixes like a bare 'bid_0'."""
    try:
        cmap[field]
        return True
    except KeyError:
        return False
