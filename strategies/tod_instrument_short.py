"""Time-of-day instrument selection short — learn the best instrument to short
in each time-of-day bucket on TRAIN, and (for now) trade that map on TRAIN too.

This is the tradeable form of the bucketed markout grid in
`markouts_custom_instruments.ipynb`:

  • Split the session into `n_tod_bins` equal-width time-of-day bins.
  • For every candidate instrument (ATM straddle, strangles, …) and every entry
    second on TRAIN, compute the NET-of-spread short P&L over a fixed holding
    window `hold`:  value(t) − value(t+hold) − spread paid at entry and exit.
    (Mid pricing + half-spread per side = exactly BirdsEye's broker model, so the
    learned edge matches what the live fills will realise.)
  • Bucket those P&Ls by the entry's time-of-day bin, score each instrument per
    bin (mean / Sharpe / tail), and keep the best instrument per bin.
  • Live: at each entry the FSM looks up the current bin and shorts that bin's
    optimal instrument, holds `hold` seconds, squares off, and re-enters — so the
    day is tiled with back-to-back shorts whose instrument follows the clock.

NOTE: stay on TRAIN only for now — both the learn step and the backtest use
split="train". This is an in-sample sanity run, NOT an out-of-sample evaluation.
Do not switch the backtest to split="val" until you deliberately want to spend
the val set.

Usage (train only)
------------------
    from engine import BirdsEye
    from strategies.tod_instrument_short import learn_from_manifest, TodInstrumentShort

    # manifest + intern-project paths come from BirdsEye/.env (no hardcoding)
    model = learn_from_manifest(index="SPY", split="train",
                                n_tod_bins=6, hold=3600, score="exp_sigma")
    print(model.scores)          # bins × instruments score table
    print(model.name_by_bin)     # chosen instrument per bin

    be = BirdsEye(
        strategy_cls    = TodInstrumentShort,
        index           = "SPY",
        split           = "train",                       # in-sample; val left untouched
        strategy_kwargs = {"model": model, "lots": 1, "stop_loss_pct": None},
        lot_size        = 100,
        starting_cash   = 1_000_000.0,
        n_workers       = 40,
    )
    res = be.run()

The model is learned once and passed into the parallel run; each per-day worker
only does the lookup.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine import Feed, Order, OrderLeg, Reason, State, StateMachineStrategy, Context
from engine.env import get_manifest_files, env

# A leg is (opt_type, offset_in_strikes, weight). offset is signed STRIKE STEPS
# from the entry ATM (CE +1 = one strike above ATM, PE −1 = one below). weight > 0
# is a SHORT leg, weight < 0 a LONG leg (so spreads are just sign-flipped legs).
Leg = Tuple[str, int, float]


def _straddle(off: int) -> List[Leg]:
    return [("CE", off, 1.0), ("PE", off, 1.0)]


def _strangle(ce_off: int, pe_off: int) -> List[Leg]:
    return [("CE", ce_off, 1.0), ("PE", pe_off, 1.0)]


def _pyramid(ce_legs, pe_legs) -> List[Leg]:
    """Weighted multi-strike strangle (a 'pyramid'): ce_legs / pe_legs are
    (offset, weight) lists. Wider, lower-gamma short-vol than a single straddle.
    Mirrors targets.py's p1/p2/p3."""
    return ([("CE", off, w) for off, w in ce_legs]
            + [("PE", off, w) for off, w in pe_legs])


# Default candidate universe — the notebook's grid: ATM + shifted straddles,
# a spread of strangles, and the p2/p3 pyramids (targets.py). Kept to moderate
# strikes (deep-OTM strikes are sparsely quoted on 0-DTE and add noise, not edge).
DEFAULT_CANDIDATES: Dict[str, List[Leg]] = {
    # straddles (both legs same strike), ATM and shifted
    "ATM":       _straddle(0),
    "str +1":    _straddle(1),
    "str -1":    _straddle(-1),
    "str +2":    _straddle(2),
    "str -2":    _straddle(-2),
    # symmetric strangles
    "sg +1/-1":  _strangle(1, -1),
    "sg +2/-2":  _strangle(2, -2),
    # "sg +3/-3":  _strangle(3, -3),
    # "sg +4/-4":  _strangle(4, -4),
    # skewed strangles
    "sg +1/-2":  _strangle(1, -2),
    "sg +2/-1":  _strangle(2, -1),
    "sg +2/-3":  _strangle(2, -3),
    "sg +3/-2":  _strangle(3, -2),
    "sg +1/0":   _strangle(1, 0),
    "sg 0/-1":   _strangle(0, -1),
    "sg +3/-1":  _strangle(3, -1),
    "sg +1/-3":  _strangle(1, -3),
    # weighted pyramids (p1 == sg +1/-1, omitted as a dup)
    "p2": _pyramid([(1, 0.5), (2, 0.5)], [(-1, 0.5), (-2, 0.5)]),
    "p3": _pyramid([(1, 0.25), (2, 0.35), (3, 0.40)],
                   [(-1, 0.25), (-2, 0.35), (-3, 0.40)]),
}


def tod_bin(sec: int, n_bins: int, session_len: int) -> int:
    """Map a second-since-open to an equal-width time-of-day bin in [0, n_bins-1]."""
    if sec < 0:
        return 0
    w = session_len / n_bins
    b = int(sec // w)
    return b if b < n_bins else n_bins - 1


def _score(x: np.ndarray, how: str, risk_aversion: float = 1.0) -> float:
    """Higher = better for all modes. `x` is the per-trade net short P&L sample.
      mean       — average net P&L (most decay captured)
      sharpe     — mean / std (decay per unit of P&L dispersion)
      tail       — mean / |ES5%| (decay per unit of expected shortfall; tail-aware)
      exp_sigma  — mean / exp(std): exponential variance penalty (scale-dependent)
      cara       — CARA certainty equivalent  -1/a * ln(mean(exp(-a*x)))  with
                   risk-aversion a=risk_aversion. The exact exponential-utility CE
                   from the empirical distribution: rewards mean, penalises both
                   variance and a fat left tail. In the same units as `x`."""
    if len(x) == 0:
        return np.nan
    mu = float(x.mean())
    if how == "mean":
        return mu
    if how == "sharpe":
        sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        return mu / sd if sd and sd > 0 else np.nan
    if how == "tail":
        q = np.quantile(x, 0.05)
        tail = x[x <= q]
        es = float(tail.mean()) if len(tail) else np.nan
        return mu / -es if (np.isfinite(es) and es < 0) else np.nan
    if how in ("exp_sigma", "mean_exp_sigma"):
        sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        return mu / np.exp(sd) if np.isfinite(sd) else np.nan
    if how == "cara":
        a = risk_aversion
        if a is None or a <= 0:
            return mu                              # no risk aversion -> just the mean
        y = -a * x                                  # CE = -1/a * log-mean-exp(-a*x)
        m = float(np.max(y))                        # log-sum-exp shift for stability
        lme = m + np.log(np.mean(np.exp(y - m)))
        return float(-lme / a)
    raise ValueError(f"unknown score {how!r} "
                     f"(use 'mean' / 'sharpe' / 'tail' / 'exp_sigma' / 'cara')")


@dataclass
class TodModel:
    """Learned time-of-day → instrument map plus the binning meta the live FSM
    needs to reproduce the bins. Passed whole into the strategy."""
    n_tod_bins:  int
    hold:        int
    session_len: int
    min_hold:    int
    legs_by_bin: Dict[int, List[Leg]]      # bin -> chosen instrument's legs
    name_by_bin: Dict[int, str]            # bin -> chosen instrument's name
    scores:      pd.DataFrame              # bins × candidates score table (inspection)
    counts:      pd.DataFrame              # bins × candidates sample counts
    means:       pd.DataFrame              # bins × candidates mean net P&L (inspection)


# ---------------------------------------------------------------------------
# Learning (offline, on train) — re-implements the notebook grid via Feed
# ---------------------------------------------------------------------------

def _day_net_pnls(feed: Feed, candidates, hold, stride, min_hold):
    """For one day, yield (name, entry_secs, net_pnls) per candidate — net-of-spread
    short P&L value(t) − value(exit) − spread(entry) − spread(exit), strikes pinned
    at the entry ATM, on a stride grid of entry seconds. Exits clip to THIS day's
    last bar. Binning is deferred to the caller (so the session length can be
    chosen robustly across all days first)."""
    n = len(feed)                                           # this day's actual length
    if n <= min_hold + 1:
        return
    strikes = feed.strikes                                  # sorted ascending
    S       = len(strikes)
    atm     = feed.atm_strike
    entries = np.arange(0, n - min_hold, stride, dtype=np.int64)
    if len(entries) == 0:
        return
    exits   = entries + np.minimum(hold, n - 1 - entries)
    # nearest strike index to the day's ATM at each entry second
    atm_idx = np.abs(strikes[None, :] - atm[entries][:, None]).argmin(axis=1)

    arr = feed.arrays   # {(opt_lower, field): (n, S)}
    for name, legs in candidates.items():
        gross = np.zeros(len(entries))
        cost  = np.zeros(len(entries))                      # entry + exit spread, summed legs
        valid = np.ones(len(entries), dtype=bool)
        for opt, off, w in legs:
            bid = arr[(opt.lower(), "bid_0")]
            ask = arr[(opt.lower(), "ask_0")]
            col = atm_idx + off
            in_range = (col >= 0) & (col < S)
            colc = np.clip(col, 0, S - 1)
            bn, an = bid[entries, colc], ask[entries, colc]            # entry quotes
            bf, af = bid[exits,   colc], ask[exits,   colc]            # exit  quotes
            mid_n, mid_f = 0.5 * (bn + an), 0.5 * (bf + af)
            half_n, half_f = 0.5 * (an - bn), 0.5 * (af - bf)
            leg_ok = (in_range & np.isfinite(mid_n) & np.isfinite(mid_f)
                      & np.isfinite(half_n) & np.isfinite(half_f))
            valid &= leg_ok
            gross += w * (mid_n - mid_f)                    # short: sell now, buy back later
            cost  += abs(w) * (half_n + half_f)             # half-spread per side, both sides
        net = gross - cost
        yield name, entries[valid], net[valid]


def learn_tod_instrument_map(paths: List[str],
                             index: str = "SPY",
                             candidates: Optional[Dict[str, List[Leg]]] = None,
                             n_tod_bins: int = 6,
                             hold: int = 3600,
                             stride: int = 1,
                             score: str = "exp_sigma",
                             session_len: Optional[int] = None,
                             risk_aversion: float = 1.0,
                             min_hold: int = 60,
                             min_count: int = 50,
                             fields=("bid_0", "ask_0"),
                             verbose: bool = True) -> TodModel:
    """Learn the best instrument to short per time-of-day bin from TRAIN days.

    paths        : list of RAW /mnt day file paths (read via Feed.from_raw).
    index        : index key for the column map / preprocessing (e.g. 'SPY').
    candidates   : {name: legs}; defaults to DEFAULT_CANDIDATES.
    n_tod_bins   : equal-width time-of-day bins over [0, session_len).
    hold         : preferred holding window in seconds (default 3600 = 1h). Late-day
                   entries with less than `hold` left clip the exit to the close.
    stride       : entry-second grid used for learning (every `stride` seconds).
    score        : 'mean' | 'sharpe' | 'tail' | 'exp_sigma' | 'cara' (default) —
                   how a bin ranks its instruments (see _score). Whatever the
                   ranking, a bin whose top instrument has mean<=0 is left flat.
    session_len  : session length (s) used for binning. If None, inferred ROBUSTLY
                   as the MODAL day length across all usable days (not the first
                   day — a short first day would mis-bin everything).
    risk_aversion: CARA risk-aversion `a` (only used when score='cara').
    min_hold     : shortest clipped late-day holding window to score/trade.
    min_count    : a (bin, instrument) needs at least this many entries to be eligible.

    Returns a TodModel (legs_by_bin / name_by_bin / scores / counts).
    """
    candidates = candidates or DEFAULT_CANDIDATES
    names = list(candidates)
    # one pass: collect per-instrument (entry_sec, net) and every day's length.
    # Binning is deferred until we know a robust session length.
    secs_by_name: Dict[str, List[np.ndarray]] = {nm: [] for nm in names}
    nets_by_name: Dict[str, List[np.ndarray]] = {nm: [] for nm in names}
    day_lengths: List[int] = []
    n_days = 0
    for path in paths:
        try:
            feed = Feed.from_raw(path, index, fields=fields)
        except Exception as e:               # skip unreadable days, keep going
            if verbose:
                print(f"  skip {path}: {e}")
            continue
        day_lengths.append(len(feed))
        n_days += 1
        for name, secs, nets in _day_net_pnls(feed, candidates, hold, stride, min_hold):
            if len(nets):
                secs_by_name[name].append(secs)
                nets_by_name[name].append(nets)

    if not day_lengths:
        raise FileNotFoundError("no usable days to learn from")

    # robust session length = modal day length (most days are full; a handful are
    # short — half-days / data gaps — and must not define the bins).
    if session_len is None:
        session_len = int(Counter(day_lengths).most_common(1)[0][0])

    # bin the collected entries by tod and score each (bin, instrument)
    w = session_len / n_tod_bins
    score_mat = pd.DataFrame(np.nan, index=range(n_tod_bins), columns=names)
    count_mat = pd.DataFrame(0,      index=range(n_tod_bins), columns=names)
    mean_mat  = pd.DataFrame(np.nan, index=range(n_tod_bins), columns=names)
    for name in names:
        if not secs_by_name[name]:
            continue
        secs = np.concatenate(secs_by_name[name])
        nets = np.concatenate(nets_by_name[name])
        bins = np.clip((secs / w).astype(np.int64), 0, n_tod_bins - 1)
        for bi in range(n_tod_bins):
            x = nets[bins == bi]
            count_mat.loc[bi, name] = len(x)
            if len(x) >= min_count:
                score_mat.loc[bi, name] = _score(x, score, risk_aversion)
                mean_mat.loc[bi, name]  = float(x.mean())

    # pick the top-scoring instrument per bin, but DON'T short a bin whose best
    # instrument has a negative mean net P&L — shorting it loses on average, so we
    # sit out that time-of-day window (no legs -> guard_enter stays flat).
    legs_by_bin, name_by_bin = {}, {}
    for bi in range(n_tod_bins):
        row = score_mat.loc[bi].dropna()
        if row.empty:
            continue
        best = row.idxmax()
        if not (mean_mat.loc[bi, best] > 0):       # negative / NaN mean -> skip the bin
            continue
        legs_by_bin[bi] = candidates[best]
        name_by_bin[bi] = best

    score_mat.index.name = count_mat.index.name = mean_mat.index.name = "tod_bin"
    if verbose:
        bin_min = session_len / n_tod_bins / 60.0
        short = sum(1 for L in day_lengths if L < session_len)
        print(f"learned on {n_days} day(s): session_len={session_len}s "
              f"(modal; {short} shorter day(s)), {n_tod_bins} bins of {bin_min:.0f} min, "
              f"hold={hold}s, stride={stride}s, score={score!r}")
        for bi in range(n_tod_bins):
            lo, hi = bi * bin_min, (bi + 1) * bin_min
            if bi in name_by_bin:
                nm = name_by_bin[bi]
                print(f"  bin {bi:>2d} [{lo:6.0f}-{hi:6.0f} min]: {nm:10s}  "
                      f"score={score_mat.loc[bi, nm]:.4f}  mean={mean_mat.loc[bi, nm]:+.4f}")
            else:
                # nothing eligible, or the top instrument had a negative mean
                row = score_mat.loc[bi].dropna()
                why = "no eligible instrument"
                if not row.empty:
                    b = row.idxmax()
                    why = f"flat: best {b} mean={mean_mat.loc[bi, b]:+.4f} <= 0"
                print(f"  bin {bi:>2d} [{lo:6.0f}-{hi:6.0f} min]: —  ({why})")

    return TodModel(n_tod_bins, hold, session_len, min_hold,
                    legs_by_bin, name_by_bin,
                    score_mat.round(4), count_mat, mean_mat.round(4))


def paths_from_manifest(manifest_path: Optional[str] = None, index: str = "SPY",
                        split: str = "train", mode: str = "0-dte",
                        data_dir: Optional[str] = None) -> List[str]:
    """RAW /mnt day-file paths for a manifest split. `manifest_path` defaults to
    the .env MANIFEST_PATH. Delegates to Intern-Project's get_manifest_files,
    which maps each manifest date to its raw source file under the index's
    data_dir (config.get_data_dir(index) unless `data_dir` overrides)."""
    manifest_path = manifest_path or env("MANIFEST_PATH")
    if not manifest_path:
        raise ValueError("no manifest_path given and MANIFEST_PATH is unset in .env")
    recs = get_manifest_files(index, split, mode, manifest_path, data_dir)
    return [r["path"] for r in recs]


def learn_from_manifest(manifest_path: Optional[str] = None, index: str = "SPY",
                        split: str = "train", mode: str = "0-dte",
                        data_dir: Optional[str] = None,
                        days: Optional[List[str]] = None, **learn_kwargs) -> TodModel:
    """Pull a manifest split's RAW days and learn the tod->instrument map on them.
    `manifest_path` defaults to the .env MANIFEST_PATH. `days` optionally subsets
    to those date strings. Extra kwargs pass through to learn_tod_instrument_map
    (n_tod_bins, hold, stride, score, session_len, ...)."""
    import os as _os
    manifest_path = manifest_path or env("MANIFEST_PATH")
    paths = paths_from_manifest(manifest_path, index, split, mode, data_dir)
    if days:
        keep  = set(days)
        paths = [p for p in paths if _os.path.basename(p).split(".")[0] in keep]
    print(f"{index}/{mode}/{split}: {len(paths)} day(s) from {manifest_path}")
    if not paths:
        raise FileNotFoundError(f"no {index}/{mode}/{split} days resolved")
    return learn_tod_instrument_map(paths, index=index, **learn_kwargs)


# ---------------------------------------------------------------------------
# Trading (online, on val/test) — FSM that shorts the bin's instrument
# ---------------------------------------------------------------------------

class _Wait(State):
    name = "WAIT"
    transitions = {"enter": "SHORT"}

    def target(self, _alphas, ctx):
        # on arriving at WAIT (from SHORT): square off whatever we hold
        keys = ctx.get("open_keys")
        if not keys:
            return None
        note = ctx.get("exit_note", "square off")
        ctx["exit_note"] = None
        ctx["open_keys"] = None
        ctx["open_name"] = None
        return ctx["strat"].close_legs(keys, reason=Reason(state="WAIT", note=note))


class _Short(State):
    name = "SHORT"
    transitions = {"stop_loss": "WAIT", "roll": "WAIT", "eod": "WAIT"}

    def target(self, alphas, ctx):
        legs = ctx.get("pending_legs")          # resolved (strike, opt, weight) for now's bin
        if not legs:
            return None
        order_legs = [OrderLeg(k, opt, lots=ctx["strat"].lots * abs(w),
                               action="SELL" if w > 0 else "BUY")
                      for (k, opt, w) in legs]
        ctx["open_keys"] = [(k, opt) for (k, opt, _) in legs]
        ctx["open_name"] = ctx.get("pending_name", "")     # instrument we now hold
        ctx["exit_note"] = None
        return Order(name=f"tod_short_{ctx.get('pending_name','')}", legs=order_legs,
                     reason=Reason(state="SHORT",
                                   signal=f"bin{alphas['bin']}:{ctx.get('pending_name','')}"))

    def on_enter(self, ctx):                    # runs at FILL-COMPLETE
        ctx["entry_sec"] = ctx["now_sec"]


class TodInstrumentShort(StateMachineStrategy):
    """Short the time-of-day-optimal instrument from a learned TodModel.

    HOLD-THROUGH-UNCHANGED-BINS: consecutive time-of-day bins often resolve to the
    SAME instrument. Rather than square off at each bin boundary and re-short the
    identical legs (paying the round-trip spread for nothing), this strategy holds
    the position across same-instrument bins and only ROLLS when the bin's optimal
    instrument actually changes — square off the old legs, short the new ones. A
    bin that maps to no instrument (negative-mean window) squares off and stays
    flat until a tradeable bin returns. Everything is squared off at the close.
    `lots` scales every leg.

    stop_loss_pct is measured against the collected basket credit (0.5 => exit at
    50% of entry credit). Default None (disabled). When enabled, a stop-out blocks
    re-entry of the SAME instrument until the bin's optimal instrument changes (so
    the stop isn't immediately undone by re-shorting the identical legs).

    NOTE on sizing the stop: credit = premium x lots x lot_size, so a short
    straddle's credit is thousands of dollars. A LOOSE multiple (e.g. 1.5x) lands
    the threshold ABOVE the realised intraday adverse move on most days — it then
    almost never fires and you eat the full short-vol loss. Worse, credit grows
    with IV, so a credit-multiple stop is loosest exactly on the high-vol days.
    Keep any stop CONSERVATIVE (~0.3-0.5x) so it actually binds.
    """
    states     = {"WAIT": _Wait(), "SHORT": _Short()}
    slice_lots = 100_000        # fill each (small) order in one tick
    pause      = 0

    def __init__(self, broker, model: TodModel, lots: float = 1.0,
                 stop_loss_pct: Optional[float] = None):
        self.model         = model
        self.lots          = lots
        self.stop_loss_pct = stop_loss_pct
        self._start_cash   = broker.portfolio.cash      # fresh book -> cum P&L baseline
        ctx = Context()
        ctx["open_keys"]     = None
        ctx["open_name"]     = None         # instrument currently held (None = flat)
        ctx["blocked_name"]  = None         # instrument we stopped out of -> no re-entry until it changes
        ctx["pending_legs"]  = None
        ctx["pending_name"]  = ""
        ctx["exit_note"]     = None
        super().__init__("WAIT", broker, name="tod_instrument_short", context=ctx)
        ctx["strat"] = self

    # ---- legs for the current bin, resolved to live strikes ----
    def _resolve_legs(self, snap):
        """(strike, opt, weight) per leg for the current bin, or None if the bin
        has no instrument / a leg is off-grid / a leg is not quoted right now."""
        m   = self.model
        b   = tod_bin(snap.i, m.n_tod_bins, m.session_len)
        legs = m.legs_by_bin.get(b)
        if legs is None:
            return b, None
        strikes = snap._feed.strikes                         # shared sorted grid
        ai = int(np.argmin(np.abs(strikes - snap._feed.atm_strike[snap.i])))
        out = []
        for opt, off, w in legs:
            j = ai + off
            if j < 0 or j >= len(strikes):
                return b, None                               # off the listed grid
            k = float(strikes[j])
            if snap.quote(k, opt) is None:                   # not quoted -> can't fill
                return b, None
            out.append((k, opt, w))
        return b, out

    # ---- guards ----
    def guard_enter(self, a, c):
        # flat, this bin maps to a quoted instrument that we're not blocked from
        # (no re-entry of an instrument we just stopped out of), and there is room
        # before the close to enter and later square off.
        min_hold = getattr(self.model, "min_hold", 60)
        day_len  = c.get("day_len", self.model.session_len)
        return (a["quoted"] and not c.get("open_keys")
                and a["target_name"] != c.get("blocked_name")
                and a["sec"] < day_len - min_hold)

    def guard_roll(self, a, c):
        # square off ONLY when the bin's optimal instrument differs from the one
        # we hold (a real change, or the bin went flat). Consecutive bins with the
        # same instrument keep the position open — no wasted round-trip. Leave a
        # bar before the close so the buy-back can fill (Executing fills NEXT tick).
        day_len = c.get("day_len", self.model.session_len)
        if a["sec"] >= day_len - 2:                       # close handled by guard_eod
            return False
        if a["target_name"] == c.get("open_name"):        # unchanged -> hold through
            return False
        c["exit_note"] = (f"roll {c.get('open_name')}->{a['target_name']}"
                          if a["target_name"] else f"flat (no inst) from {c.get('open_name')}")
        return True

    def guard_eod(self, a, c):
        # always flat by the close: issue the square-off a bar early so it fills.
        day_len = c.get("day_len", self.model.session_len)
        if a["sec"] >= day_len - 2:
            c["exit_note"] = "square off (eod)"
            return True
        return False

    def guard_stop_loss(self, a, c):
        if self.stop_loss_pct is None:
            return False
        stop_loss = a.get("stop_loss")
        if not stop_loss or stop_loss <= 0:
            return False
        hit = max(0.0, -a.get("basket_pnl", 0.0)) >= stop_loss
        if hit:
            c["exit_note"]    = f"stop loss {self.stop_loss_pct:g}x credit"
            c["blocked_name"] = c.get("open_name")        # don't re-short the same legs until it changes
        return hit

    def _open_basket_metrics(self, snap):
        """Return live PnL/loss and stop threshold for the currently open basket."""
        keys = self.context.get("open_keys") or []
        pnl = 0.0
        credit = 0.0
        for key in keys:
            pos = self.broker.portfolio.positions.get(key)
            if pos is None or pos.lots == 0:
                continue
            mh = snap.mid_and_half_spread(*key)
            if mh is None:
                continue
            mid, _ = mh
            lot_size = self.broker.portfolio.lot_size
            pnl += (mid - pos.avg_entry) * pos.lots * lot_size
            credit += -pos.lots * pos.avg_entry * lot_size

        loss = max(0.0, -pnl)
        stop_loss = 0.0
        if self.stop_loss_pct is not None and credit > 0:
            stop_loss = credit * self.stop_loss_pct
        return pnl, loss, credit, stop_loss

    # ---- alphas (every second) ----
    def compute_alphas(self, snap):
        c = self.context
        sec = snap.i
        c["now_sec"] = sec
        c["day_len"] = len(snap._feed)          # actual bars this day (square-off / entry room)
        b, legs = self._resolve_legs(snap)
        target_name = self.model.name_by_bin.get(b, "")    # bin's optimal instrument ("" = flat)
        c["pending_legs"] = legs
        c["pending_name"] = target_name
        # a stop-block only lasts until the optimal instrument actually changes
        if c.get("blocked_name") is not None and target_name != c["blocked_name"]:
            c["blocked_name"] = None
        pnl, loss, credit, stop_loss = self._open_basket_metrics(snap)
        cum_pnl = self.broker.portfolio.equity(snap) - self._start_cash   # running day P&L (net of costs)
        return {"sec": sec, "bin": b, "target_name": target_name,
                "have_inst": int(b in self.model.legs_by_bin),
                "quoted": int(legs is not None),
                "cum_pnl": cum_pnl,
                "basket_pnl": pnl,
                "basket_credit": credit,
                "stop_loss": stop_loss}
