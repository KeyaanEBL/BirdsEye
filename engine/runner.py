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
from . import analyzers

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
    (path, strategy_cls, strategy_kwargs, fields,
     cfg, cost_kwargs, curve_every, collect_perseclog) = args

    feed  = Feed.from_file(path, fields=fields)
    pf    = Portfolio(lot_size=cfg.lot_size, starting_cash=cfg.starting_cash)
    cm    = CostModel(lot_size=cfg.lot_size, **cost_kwargs)
    led   = Tradelog()
    br    = Broker(pf, cm, led)
    strat = strategy_cls(broker=br, **strategy_kwargs)

    day = os.path.basename(path).split(".")[0]
    last_snap = None
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

    def _paths(self) -> List[str]:
        paths = sorted(glob.glob(os.path.join(self.data_dir, "*.parquet")))
        if self.days:
            keep  = set(self.days)
            paths = [p for p in paths if os.path.basename(p).split(".")[0] in keep]
        if not paths:
            raise FileNotFoundError(f"no parquet days found in {self.data_dir}")
        return paths

    def run(self, parallel: bool = True) -> Results:
        paths = self._paths()
        jobs  = [
            (p, self.strategy_cls, self.strategy_kwargs, self.fields,
            self.cfg, self.cost_kwargs, self.curve_every, self.collect_perseclog)
            for p in paths
        ]

        n = min(self.n_workers or (os.cpu_count() or 1), len(paths))
        if parallel and n > 1:
            with ProcessPoolExecutor(max_workers=n) as ex:   # context manager — critical
                outs = list(ex.map(_run_one_day, jobs, chunksize=1))
        else:
            outs = [_run_one_day(j) for j in jobs]

        return Results(
            days             = [o[0] for o in outs],
            curves           = {o[0]: o[1] for o in outs},
            tradelogs        = {o[0]: o[2] for o in outs},
            day_pnls         = [o[3] for o in outs],
            cfg              = self.cfg,
            perseclogs       = {o[0]: o[4] for o in outs},
            traded_lots_list = [o[5] for o in outs],
            max_lots         = outs[0][6],
        )