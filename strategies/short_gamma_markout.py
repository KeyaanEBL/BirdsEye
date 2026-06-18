"""
short_gamma_markout.py — turn a day's Feed into the UR/DR confidence lines.

The OFFLINE fit (sandbox notebook) saves a `bundle` to the markout directory:
fitted ridge weights + score-bucket edges + per-bucket percentile tables, for
each (side, horizon) in {up,down} x {7200s(=120min), 1800s(=30min)}. This module
loads that bundle and, per day, recomputes the 5 features off the Feed, scores
them, assigns each bar to its training-defined score bucket, and reads the
percentile value out of the table. Nothing here looks ahead: every per-bar input
is a feature computed from data up to t, and the tables are frozen from train.

Bundle layout (pickle), see build_bundle() for the writer used by the sandbox:
  {
    "feat_names" : [...5...],
    "lookbacks"  : {name: int},
    "n_buckets"  : 8,
    "pct_levels" : [5, 25, 50, 75, 95],
    "models"     : { "up_7200": M, "down_7200": M, "up_1800": M, "down_1800": M },
    "meta"       : { "index": "SPY", "fwd_long": 7200, "fwd_short": 1800, ... },
  }
  where M = {"mu":(5,), "sd":(5,), "coef":(5,), "intercept":float,
             "edges":(n_buckets-1,), "pct_table":(n_buckets, len(pct_levels))}

Confidence-line accessor names returned by SignalEngine.row(i):
  UR120_p50 UR120_p75 DR120_p50 DR120_p75   (120-min = entry signals)
  UR30_p50  UR30_p75  DR30_p50  DR30_p75    (30-min  = exit / TP-gate signals)
plus score_/bucket_ diagnostics. Up uses the up model, down the down model.
"""
import os
import pickle
import numpy as np

from short_gamma_features import feature_matrix_from_feed, FEATURE_NAMES, MAX_LOOKBACK

BUNDLE_NAME = "short_gamma_markout.pkl"

# (line key prefix, side, horizon-seconds) for the four signal families
_FAMILIES = [
    ("UR120", "up",   7200),
    ("DR120", "down", 7200),
    ("UR30",  "up",   1800),
    ("DR30",  "down", 1800),
]


def load_bundle(markout_dir, name=BUNDLE_NAME):
    path = os.path.join(markout_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"markout bundle not found at {path}. Run the precompute cells in "
            f"the sandbox notebook first (build_bundle writes it there).")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build_bundle(models, meta, n_buckets, pct_levels):
    """Assemble (not write) a bundle dict from fitted per-(side,horizon) models.
    Used by the sandbox; kept here so the on-disk schema has one definition."""
    return {
        "feat_names": list(FEATURE_NAMES),
        "lookbacks" : {name: lb for name, _, lb, _ in __import__(
                          "short_gamma_features").make_features()},
        "n_buckets" : int(n_buckets),
        "pct_levels": list(pct_levels),
        "models"    : models,
        "meta"      : dict(meta),
    }


def model_key(side, horizon):
    return f"{side}_{int(horizon)}"


class SignalEngine:
    """Holds the fitted bundle; turns a Feed into per-bar confidence lines.

    Per day: call prepare(feed) once (cached by object identity), then row(i)
    for the dict of confidence values at bar i. A custom engine exposing the
    same prepare()/row()/min_history interface can be injected into the strategy
    for testing (see tests)."""

    def __init__(self, bundle):
        self.bundle     = bundle
        self.feat_names = bundle["feat_names"]
        self.n_buckets  = bundle["n_buckets"]
        self.pct_levels = bundle["pct_levels"]
        self._jp50      = self.pct_levels.index(50)
        self._jp75      = self.pct_levels.index(75)
        self._feed_id   = None
        self._series    = None
        self.min_history = MAX_LOOKBACK + 1

    # ---- per-day preparation ---------------------------------------------- #
    def prepare(self, feed):
        if self._feed_id == id(feed) and self._series is not None:
            return
        X = feature_matrix_from_feed(feed)                 # (n, 5)
        n = X.shape[0]
        out = {}
        for prefix, side, horizon in _FAMILIES:
            m      = self.bundle["models"][model_key(side, horizon)]
            score  = self._score(X, m)                     # (n,) NaN where any feat NaN
            bucket = self._bucket(score, m["edges"])       # (n,) int, -1 where NaN
            p50    = self._lookup(bucket, m["pct_table"], self._jp50)
            p75    = self._lookup(bucket, m["pct_table"], self._jp75)
            out[f"{prefix}_p50"]   = p50
            out[f"{prefix}_p75"]   = p75
            out[f"score_{prefix}"] = score
            out[f"bucket_{prefix}"] = np.where(bucket < 0, np.nan, bucket + 1).astype(float)
        self._series  = out
        self._feed_id = id(feed)
        self._n       = n

    def _score(self, X, m):
        z = (X - m["mu"]) / m["sd"]
        s = z @ m["coef"] + m["intercept"]
        s[np.isnan(X).any(axis=1)] = np.nan
        return s

    def _bucket(self, score, edges):
        b = np.full(len(score), -1, dtype=np.int64)
        ok = ~np.isnan(score)
        if ok.any():
            idx = np.searchsorted(np.asarray(edges, dtype=float), score[ok], side="right")
            b[ok] = np.clip(idx, 0, self.n_buckets - 1)
        return b

    def _lookup(self, bucket, table, jcol):
        out = np.full(len(bucket), np.nan)
        ok  = bucket >= 0
        out[ok] = np.asarray(table, dtype=float)[bucket[ok], jcol]
        return out

    # ---- per-bar read ------------------------------------------------------ #
    def row(self, i):
        s = self._series
        return {k: float(v[i]) for k, v in s.items()}
