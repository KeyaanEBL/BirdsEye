"""
runner.py — BirdsEye entry point.

Parallelises a backtest over days, returns a Results object.

    be = BirdsEye(
        data_dir        = ".../SPY/0-dte/train",
        strategy_cls    = RangeShortStrangle,
        index           = "SPY",                      # sets periods_per_year
        strategy_kwargs = {"lots": 10},
        lot_size        = 100,
        starting_cash   = 1_000_000.0,
        margin_per_lot  = 150.0,
        cost_kwargs     = {"txn_cost_per_lot": 0.85},
        n_workers       = 40,
        collect_perseclog = True,
    )
    res = be.run()
    res.summary          # cached DataFrame
    res.tradelog         # cached DataFrame
    res.stats()
    res.perseclog(day)   # per-second log, one day
"""

import os
import glob
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from .feed import Feed, DEFAULT_FIELDS
from .costs import CostModel
from .ledger import Tradelog
from .portfolio import Portfolio
from .broker import Broker
from .env import env, get_manifest_files, get_data_dir
from .logsetup import make_logger
from . import analyzers, plots

# periods per year by underlying — override with periods_per_year kwarg if needed
PERIODS_PER_YEAR: Dict[str, int] = {
    "SPY": 252, "QQQ": 252, "IWM": 252, "SPX": 252,
    "NIFTY": 52, "BANKNIFTY": 52, "SENSEX": 52,
}


@dataclass
class RunConfig:
    """Single source of truth for run-level parameters shared by BirdsEye and Results."""
    lot_size        : float
    starting_cash   : float
    margin_per_lot  : float
    periods_per_year: int


# ---------------------------------------------------------------------------
# worker (one day, one process)
# ---------------------------------------------------------------------------

def _run_one_day(args):
    """Run one day in a fresh process. Pure: raw path + index + config in,
    results out. A day with no usable bars (load error) returns a skip sentinel
    (curve=None, last field = the error) instead of aborting the whole run."""
    (path, index, strategy_cls, strategy_kwargs, fields, cfg, cost_kwargs, curve_every, collect_perseclog) = args

    day = os.path.basename(path).split(".")[0]
    try:
        feed = Feed.from_raw(path, index, fields=fields)     # raw /mnt -> arrays
    except Exception as e:
        return day, None, None, None, None, f"load error: {e}"

    pf    = Portfolio(lot_size=cfg.lot_size, starting_cash=cfg.starting_cash)
    cm    = CostModel(lot_size=cfg.lot_size, **cost_kwargs)
    led   = Tradelog()
    br    = Broker(pf, cm, led)
    strat = strategy_cls(broker=br, **strategy_kwargs)

    try:
        for snap in feed:
            strat.next(snap)
            br.mark_to_market(snap)
            last_snap = snap
    except Exception as e:
        raise RuntimeError(f"[{day}] {e}") from e

    if last_snap is not None:
        br.eod_square_off(last_snap)
        br.mark_to_market(last_snap)

    raw_curve = np.array([e for _, e in pf.equity_curve], dtype=np.float64)
    if curve_every > 1:
        idx = list(range(0, len(raw_curve), curve_every))
        if idx[-1] != len(raw_curve) - 1:
            idx.append(len(raw_curve) - 1)
        raw_curve = raw_curve[idx]

    perseclog_data = None
    if collect_perseclog:
        df = strat.perseclog.as_dataframe()
        perseclog_data = {col: df[col].to_numpy() for col in df.columns}

    return (
        day,
        raw_curve,
        led,
        float(raw_curve[-1] - cfg.starting_cash),
        perseclog_data,
        sum(abs(f.lots) for f in led.fills),   # total traded lots (raw, not /2)
        getattr(strat, "max_lots", 1.0),
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class Results:
    def __init__(self, days, curves, tradelogs, day_pnls, cfg: RunConfig,
                 perseclogs=None, traded_lots_list=None, max_lots=None):
        self.days         = days
        self.curves       = curves
        self.tradelogs    = tradelogs
        self.day_pnls     = day_pnls
        self.cfg          = cfg
        self._perseclogs  = perseclogs or {}
        self._traded_lots = traded_lots_list or []
        self._max_lots    = max_lots or 1.0

    # --- tables (cached — concat is expensive over many days) ---------------

    @cached_property
    def tradelog(self) -> pd.DataFrame:
        frames = []
        for day in self.days:
            df = self.tradelogs[day].as_dataframe()
            if not df.empty:
                df.insert(0, "day", day)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @cached_property
    def summary(self) -> pd.DataFrame:
        rows = []
        for day, pnl in zip(self.days, self.day_pnls):
            led   = self.tradelogs[day]
            costs = led.total_costs
            rows.append({
                "day":      day,
                "fills":    len(led.fills),
                "gross($)": round(pnl + costs, 2),
                "costs($)": round(costs, 2),
                "net($)":   round(pnl, 2),
            })
        return pd.DataFrame(rows).set_index("day")

    @cached_property
    def all_curve(self) -> np.ndarray:
        return np.concatenate(list(self.curves.values()))

    def perseclog(self, day: str) -> pd.DataFrame:
        data = self._perseclogs.get(day)
        if data is None:
            raise ValueError(f"No perseclog for {day}. Re-run with collect_perseclog=True.")
        return pd.DataFrame(data)

    # --- stats --------------------------------------------------------------

    def stats(self) -> Dict:
        tl          = self.tradelog
        total_costs = float(tl["exe_cost"].sum()) if not tl.empty else 0.0
        total_lots  = np.mean(self._traded_lots)
        cfg         = self.cfg

        cagr_g, cagr_n     = analyzers.cagr(
            self.day_pnls, total_costs,
            cfg.margin_per_lot, self._max_lots, cfg.periods_per_year,
        )
        calmar_g, calmar_n = analyzers.calmar(
            self.day_pnls, total_costs,
            cfg.margin_per_lot, self._max_lots, cfg.periods_per_year,
        )

        out = dict(analyzers.daily_stats(self.day_pnls))
        out.update({
            "cagr_gross"   : cagr_g,
            "cagr_net"     : cagr_n,
            "maxDD_pct"    : analyzers.max_drawdown(self.day_pnls, cfg.margin_per_lot, self._max_lots),
            "calmar_gross" : calmar_g,
            "calmar_net"   : calmar_n,
            "churn"        : analyzers.churn(total_lots),
            "total_costs"  : total_costs,
            "n_fills"      : len(tl) if not tl.empty else 0,
        })
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# BirdsEye
# ---------------------------------------------------------------------------

class BirdsEye:
    def __init__(self,
                 data_dir          : str,
                 strategy_cls      : Type,
                 index             : str = "SPY",
                 strategy_kwargs   : Optional[dict]   = None,
                 fields            : Tuple[str, ...]  = DEFAULT_FIELDS,
                 lot_size          : float = 1.0,
                 starting_cash     : float = 0.0,
                 margin_per_lot    : float = 0.0,
                 cost_kwargs       : Optional[dict]   = None,
                 n_workers         : Optional[int]    = None,
                 days              : Optional[List[str]] = None,
                 curve_every       : int  = 1,
                 collect_perseclog : bool = False,
                 periods_per_year  : Optional[int]    = None):

        self.cfg = RunConfig(
            lot_size         = lot_size,
            starting_cash    = starting_cash,
            margin_per_lot   = margin_per_lot,
            periods_per_year = periods_per_year
                               or PERIODS_PER_YEAR.get(index.upper(), 252),
        )
        self.data_dir          = data_dir
        self.strategy_cls      = strategy_cls
        self.strategy_kwargs   = strategy_kwargs or {}
        self.fields            = tuple(fields)
        self.cost_kwargs       = cost_kwargs or {}
        self.n_workers         = n_workers
        self.days              = days
        self.curve_every       = max(1, curve_every)
        self.collect_perseclog = collect_perseclog
        # day selection comes from the manifest (train split, 0-dte), same as before:
        # MANIFEST_PATH lists which dates are train; data_dir is where the raw files
        # live. _paths() resolves the train dates to raw paths under data_dir.
        self.index             = index
        self.manifest_path     = env("MANIFEST_PATH")
        self.split             = "train"
        self.mode              = "0-dte"

    def _paths(self) -> List[str]:
        """Raw /mnt source paths for the run. From the manifest split when a
        manifest is available (constructor arg, else .env MANIFEST_PATH), else
        every raw file in the index's data_dir. `days` further subsets either way.
        Paths default to .env (MANIFEST_PATH / DATA_DIR) — nothing hardcoded here."""
        manifest = self.manifest_path or env("MANIFEST_PATH")
        data_dir = self.data_dir or env("DATA_DIR")
        if manifest:
            recs  = get_manifest_files(self.index, self.split, self.mode,
                                       manifest, data_dir)
            paths = [r["path"] for r in recs]
        else:
            ddir  = data_dir or get_data_dir(self.index)
            paths = sorted(glob.glob(os.path.join(ddir, "*.parquet"))
                           + glob.glob(os.path.join(ddir, "*.csv")))
        if self.days:
            keep  = set(self.days)
            paths = [p for p in paths if os.path.basename(p).split(".")[0] in keep]
        if not paths:
            raise FileNotFoundError(
                f"no raw {self.index} days found "
                f"({'manifest ' + self.manifest_path if self.manifest_path else self.data_dir})")
        return paths

    def run(self, parallel: bool = True) -> Results:
        log   = make_logger(f"{self.index}_{self.split}")
        paths = self._paths()
        log.info("run start | index=%s split=%s strategy=%s | %d day(s)",
                 self.index, self.split, self.strategy_cls.__name__, len(paths))
        jobs  = [(p, self.index, self.strategy_cls, self.strategy_kwargs, self.fields,
                  self.cfg , self.cost_kwargs, self.curve_every, self.collect_perseclog)
                 for p in paths]

        n = min(self.n_workers or (os.cpu_count() or 1), len(paths))
        if parallel and n > 1:
            with ProcessPoolExecutor(max_workers=n) as ex:   # context manager — critical
                outs = list(ex.map(_run_one_day, jobs, chunksize=1))
        else:
            outs = [_run_one_day(j) for j in jobs]

        # drop days that failed to load (skip sentinel: curve is None) so Results
        # only ever holds usable days — the failed day's slot would otherwise be a
        # None ledger/curve and break summary/stats. Log each skip with its error.
        good = [o for o in outs if o[1] is not None]
        for o in outs:
            if o[1] is None:
                log.warning("skip %s | %s", o[0], o[5])
        if not good:
            raise RuntimeError(f"all {len(outs)} day(s) failed to load — see {log.log_path}")
        if len(good) < len(outs):
            log.info("ran %d/%d day(s) (%d skipped)", len(good), len(outs), len(outs) - len(good))

        return Results(
            days             = [o[0] for o in good],
            curves           = {o[0]: o[1] for o in good},
            tradelogs        = {o[0]: o[2] for o in good},
            day_pnls         = [o[3] for o in good],
            cfg              = self.cfg,
            perseclogs       = {o[0]: o[4] for o in good},
            traded_lots_list = [o[5] for o in good],
            max_lots         = good[0][6],
        )
