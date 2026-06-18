"""
short_gamma_features.py — the 5 features that feed the Short-Gamma score.

SINGLE SOURCE OF TRUTH. Both the offline fit (sandbox notebook) and the live
strategy compute features through THESE functions, so the score the strategy
sees at runtime is bit-for-bit the score the ridge combo was fitted on. Each fn
has the Intern-Project signature `fn(arrays, ts) -> np.ndarray`, where:

    arrays : dict[str, np.ndarray | dict]   abstract-column -> full-day data
             "spot", "volume", "atm_strike"            -> 1-D arrays (n,)
             "*_ce_premium", "*_pe_premium"            -> {int strike -> (n,)}
    ts     : np.ndarray[int]                 bar indices to evaluate at

The formulas are copied verbatim from spot-range_battery.ipynb (cell 2). If you
change a feature here, refit the markout bundle — the cache signature in the
research repo will also change, which is the intended invalidation.
"""
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# premium-grid helper (verbatim from the notebook)
# --------------------------------------------------------------------------- #
def _build_premium_grids(ce_prem, pe_prem):
    sorted_strikes = np.array(sorted(ce_prem.keys()))
    if len(sorted_strikes) < 2:
        return None, None, None, None
    min_strike     = sorted_strikes[0]
    strike_spacing = sorted_strikes[1] - sorted_strikes[0]
    day_length     = len(next(iter(ce_prem.values())))
    ce_grid = np.array([ce_prem[k] for k in sorted_strikes])
    pe_grid = np.array([pe_prem[k] if k in pe_prem else np.full(day_length, np.nan)
                        for k in sorted_strikes])
    return ce_grid, pe_grid, min_strike, strike_spacing


# --------------------------------------------------------------------------- #
# feature fns  (name, fn, lookback, columns)
# --------------------------------------------------------------------------- #
def _rv(window=900):
    def fn(arrays, ts):
        spot    = arrays["spot"]
        log_ret = np.log(spot[1:] / spot[:-1])
        sq      = log_ret ** 2
        cs      = np.empty(len(sq) + 1); cs[0] = 0.0
        np.cumsum(sq, out=cs[1:])
        return np.sqrt(cs[ts] - cs[ts - window])
    return (f"rv_{window}s", fn, window, ["spot"])


def _straddle_std(window=10 * 60):
    def fn(arrays, ts):
        atm = arrays["atm_strike"]
        ce_grid, pe_grid, min_k, spacing = _build_premium_grids(
            arrays["*_ce_premium"], arrays["*_pe_premium"])
        out = np.full(len(ts), np.nan)
        if ce_grid is None:
            return out
        n_strikes, N = ce_grid.shape
        rows  = (np.asarray(atm, dtype=np.int64) - min_k) // spacing
        cols  = np.arange(N)
        valid = (rows >= 0) & (rows < n_strikes)
        straddle = np.full(N, np.nan)
        straddle[valid] = (ce_grid[rows[valid], cols[valid]]
                           + pe_grid[rows[valid], cols[valid]])
        std_full = (pd.Series(straddle)
                    .rolling(window, min_periods=max(2, window // 4)).std().values)
        return std_full[np.asarray(ts, dtype=np.int64)]
    return (f"rolling_straddle_std_{window // 60}min", fn, window - 1,
            ["atm_strike", "*_ce_premium", "*_pe_premium"])


def _vwap_gap_abs_bps():
    def _session_vwap(spot, volume):
        cumv  = np.cumsum(volume)
        cumpv = np.cumsum(spot * volume)
        return np.where(cumv > 0, cumpv / cumv, np.nan)
    def fn(arrays, ts):
        spot   = arrays["spot"]
        volume = arrays["volume"]
        vwap   = _session_vwap(spot, volume)
        full   = np.where(np.isfinite(vwap) & (vwap > 0),
                          np.abs(spot - vwap) / vwap * 1e4, np.nan)
        return full[ts]
    return ("vwap_gap_abs_bps", fn, 0, ["spot", "volume"])


def _abs_long_vol_accel(short=600, long_w=1800):
    def fn(arrays, ts):
        s     = pd.Series(arrays["spot"])
        adiff = s.diff().abs()
        def atr_bps(n):
            atr  = adiff.rolling(n,     min_periods=n    ).mean().to_numpy()
            mean = s.rolling(n + 1, min_periods=n + 1).mean().to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                return atr / mean * 1e4
        return np.abs(atr_bps(short) - atr_bps(long_w))[ts]
    return (f"abs_long_vol_accel_{short}_{long_w}", fn, long_w, ["spot"])


def _bps_below_high(window=900):
    def fn(arrays, ts):
        spot = arrays["spot"]
        swv  = np.lib.stride_tricks.sliding_window_view(spot, window + 1)
        rmax = swv.max(axis=1)
        local_high = rmax[ts - window]
        return (spot[ts] - local_high) / local_high * 100
    return (f"bps_below_high_{window}s", fn, window, ["spot"])


# --------------------------------------------------------------------------- #
# canonical registry — ORDER MATTERS (matches the fitted weight vectors)
# --------------------------------------------------------------------------- #
def make_features():
    """The 5 Short-Gamma features in fixed order: (name, fn, lookback, columns)."""
    return [
        _rv(900),
        _straddle_std(10 * 60),
        _vwap_gap_abs_bps(),
        _abs_long_vol_accel(600, 1800),
        _bps_below_high(900),
    ]


FEATURE_NAMES = [f[0] for f in make_features()]
MAX_LOOKBACK  = max(f[2] for f in make_features())

# feed fields the features need loaded (on top of the bid/ask defaults).
FEED_FIELDS = ("volume", "ce_premium", "pe_premium")


def feature_matrix_from_feed(feed):
    """(n, 5) matrix of the 5 features over the WHOLE day, computed off a Feed's
    full-day arrays. Columns ordered as FEATURE_NAMES. Rows < a feature's
    lookback are NaN for that column (and so the score is NaN there)."""
    n  = len(feed)
    ts = np.arange(n, dtype=np.int64)

    arrays = {
        "spot":       np.asarray(feed.spot, dtype=float),
        "atm_strike": np.asarray(feed.atm_strike, dtype=float),
    }
    vol = feed.arrays.get("volume")
    if vol is not None:
        arrays["volume"] = np.asarray(vol, dtype=float)

    ce = feed.arrays.get(("ce", "premium"))
    pe = feed.arrays.get(("pe", "premium"))
    if ce is not None and pe is not None:
        arrays["*_ce_premium"] = {int(k): ce[:, c] for k, c in feed.strike_to_col.items()}
        arrays["*_pe_premium"] = {int(k): pe[:, c] for k, c in feed.strike_to_col.items()}

    cols = []
    for name, fn, lookback, needed in make_features():
        missing = [c for c in needed
                   if (c.startswith("*_") and c not in arrays) or
                      (not c.startswith("*_") and c not in arrays)]
        if missing:
            raise KeyError(
                f"feature {name!r} needs {needed} but the feed is missing {missing}. "
                f"Pass fields={FEED_FIELDS} to BirdsEye(...).")
        with np.errstate(all="ignore"):
            v = np.asarray(fn(arrays, ts), dtype=float)
        v[ts < lookback] = np.nan          # undefined during warm-up
        cols.append(v)
    return np.column_stack(cols)
