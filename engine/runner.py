"""
runner.py — the consolidated BirdsEye entry point.

Give it a strategy class + the parameters for every layer, and it runs the whole
backtest in parallel over days, returning a Results object that computes stats
and plots on demand.

    from engine import BirdsEye

    be = BirdsEye(
        data_dir   = "/home/keyaan/Project/data/SPY/0-dte/train",
        strategy_cls = RangeShortStraddle,
        strategy_kwargs = {},                      # -> strategy __init__ (besides broker)
        fields     = ("bid_0", "ask_0"),           # -> Feed (per-strike fields to load)
        lot_size   = 100,                          # -> Portfolio + CostModel
        starting_cash = 1000.0,                    # -> Portfolio (fresh per day)
        cost_kwargs = {"txn_cost_per_trade": 0.85},# -> CostModel (any of its params)
        n_workers  = 36,                           # -> ProcessPoolExecutor
    )
    res = be.run()          # parallel over all days, chronological order kept
    res.summary            # per-day dataframe (gross / costs / net / fills)
    res.stats()            # aggregate dict (daily stats, calmar, drawdowns, costs)
    res.tearsheet()        # daily PnL + stitched equity + drawdown figure
    res.Tradelog()           # all fills across days, one dataframe

NOTE on multiprocessing: workers fork on Linux, so notebook-defined strategy
classes work as-is. If you ever run on a spawn platform (macOS/Windows), move
the strategy class into an importable .py module.
"""
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from .feed import Feed, DEFAULT_FIELDS
from .costs import CostModel
from .ledger import Tradelog, PerSecLog
from .portfolio import Portfolio
from .broker import Broker
from .env import env, get_manifest_files, get_data_dir
from .logsetup import make_logger
from . import analyzers, plots


def _run_one_day(args):
    """Run one day in a fresh process. Pure: raw path + index + config in,
    results out. A day with no usable bars (load error) returns a skip sentinel
    (curve=None, last field = the error) instead of aborting the whole run."""
    (path, index, strategy_cls, strategy_kwargs, fields,
     lot_size, starting_cash, cost_kwargs, curve_every) = args

    day = os.path.basename(path).split(".")[0]
    try:
        feed = Feed.from_raw(path, index, fields=fields)     # raw /mnt -> arrays
    except Exception as e:
        return day, None, None, None, None, f"load error: {e}"

    pf    = Portfolio(lot_size=lot_size, starting_cash=starting_cash)
    cm    = CostModel(lot_size=lot_size, **cost_kwargs)
    led   = Tradelog()
    br    = Broker(pf, cm, led)
    strat = strategy_cls(broker=br, **strategy_kwargs)

    try:
        for snap in feed:
            strat.next(snap)
            br.mark_to_market(snap)
    except Exception as e:
        raise RuntimeError(f"[{day}] {e}") from e

    curve = pf.equity_curve
    if curve_every > 1:               # optional downsample (always keep last point)
        curve = curve[::curve_every] + ([curve[-1]] if (len(curve) - 1) % curve_every else [])

    perseclog = strat.perseclog.as_dataframe()
    return day, curve, led, pf.equity_curve[-1][1] - starting_cash, perseclog, None


class Results:
    def __init__(self, days, curves, Tradelogs, day_pnls, starting_cash, lot_size, perseclogs=None):
        self.days          = days                      # chronological day keys
        self.curves        = curves                    # day -> [(ts, equity)]
        self.Tradelogs     = Tradelogs                 # day -> Tradelog
        self.day_pnls      = day_pnls                  # chronological net PnL per day
        self.starting_cash = starting_cash
        self.perseclogs    = perseclogs or {}
        self.lot_size      = lot_size

    # --- tables ---
    @property
    def summary(self) -> pd.DataFrame:
        rows = []
        for day, pnl in zip(self.days, self.day_pnls):
            led   = self.Tradelogs[day]
            costs = led.total_costs
            rows.append({"day": day, "fills": len(led.fills), "gross($)": pnl + costs, "costs($)": costs, "net($)": pnl})
        return pd.DataFrame(rows).set_index("day").round(2)

    def Tradelog(self) -> pd.DataFrame:
        frames = []
        for day in self.days:
            df = self.Tradelogs[day].as_dataframe()
            if not df.empty:
                df.insert(0, "day", day)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    
    def perseclog(self, day: str) -> pd.DataFrame:
        """Per-second log for one day: ts, i, spot, atm, state, every alpha."""
        return self.perseclogs[day]

    # --- stats ---
    def stats(self) -> Dict[str, float]:
        cap                   = self.starting_cash
        out                   = dict(analyzers.daily_stats(self.day_pnls))
        out["cagr"]           = analyzers.cagr(self.day_pnls, cap)
        out["calmar"]         = analyzers.calmar(self.day_pnls, cap)
        out["maxDD_pct"]      = analyzers.max_drawdown_pct(self.day_pnls, cap)
        notional = sum(led.total_notional(self.lot_size) for led in self.Tradelogs.values())
        out["churn_per_day"]  = analyzers.churn(notional, cap, len(self.days))
        out["daily_maxDD"]    = analyzers.max_drawdown(list(enumerate(np.cumsum(self.day_pnls))))
        out["intraday_maxDD"] = analyzers.max_drawdown(self.all_curve)
        
        led_all = self.Tradelog()
        if not led_all.empty:
            out["total_costs"] = float(led_all["exe_cost"].sum())
            out["n_fills"]     = int(len(led_all))
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}

    @property
    def all_curve(self) -> List[Tuple[int, float]]:
        """Per-second equity stitched across days (each day starts fresh)."""
        curve = []
        for day in self.days:
            curve += self.curves[day]
        return curve

    def tearsheet(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 9))
        plots.plot_daily_pnl(self.day_pnls, ax=ax[0])
        ax[0].set_xticks(range(len(self.days)))
        ax[0].set_xticklabels(self.days, rotation=45)
        plots.plot_equity(self.all_curve, ax=ax[1])
        plots.plot_drawdown(self.all_curve, ax=ax[2])
        fig.tight_layout()
        return fig

    def plot_day(self, day: str):
        """Single-day equity with fill markers (entries v, exits ^ by lots sign)."""
        import matplotlib.pyplot as plt
        curve   = self.curves[day]
        eq      = np.array([e for _, e in curve]); t = np.arange(len(eq))
        df      = self.Tradelogs[day].as_dataframe()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, eq, lw=1, color="0.3", label="equity (MtM @ mid)")
        if not df.empty:
            t0          = curve[0][0]
            sec         = ((df["timestamp"] - t0) // 1_000_000_000).astype(int).clip(0, len(eq) - 1)
            sells, buys = df["lots"] < 0, df["lots"] > 0
            ax.scatter(sec[sells], eq[sec[sells]], marker="v", color="tab:red",   s=40, zorder=3, label="sell")
            ax.scatter(sec[buys],  eq[sec[buys]],  marker="^", color="tab:green", s=40, zorder=3, label="buy")
        ax.set_title(f"{day} — intraday equity"); ax.set_xlabel("seconds"); ax.set_ylabel("equity ($)")
        ax.legend()
        return fig


class BirdsEye:
    def __init__(self,
                 strategy_cls    : Type,
                 index           : str = "SPY",                        # picks column map + raw data_dir
                 manifest_path   : Optional[str] = None,               # split source (e.g. manifest_tvt.json)
                 split           : str = "train",                      # which manifest split to run
                 mode            : str = "0-dte",                      # manifest mode key
                 data_dir        : Optional[str] = None,               # raw /mnt dir; default = config.get_data_dir(index)
                 strategy_kwargs : Optional[dict] = None,
                 fields          : Tuple[str, ...] = DEFAULT_FIELDS,   # -> Feed (per-strike fields)
                 lot_size        : float = 1.0,                        # -> Portfolio + CostModel
                 starting_cash   : float = 0.0,                        # -> Portfolio (fresh per day)
                 cost_kwargs     : Optional[dict] = None,              # -> CostModel
                 n_workers       : Optional[int] = None,               # -> ProcessPoolExecutor
                 days            : Optional[List[str]] = None,         # subset, e.g. ["20240208"]
                 curve_every     : int = 1):                           # equity downsample factor
        self.strategy_cls    = strategy_cls
        self.index           = index
        self.manifest_path   = manifest_path
        self.split           = split
        self.mode            = mode
        self.data_dir        = data_dir
        self.strategy_kwargs = strategy_kwargs or {}
        self.fields          = tuple(fields)
        self.lot_size        = lot_size
        self.starting_cash   = starting_cash
        self.cost_kwargs     = cost_kwargs or {}
        self.n_workers       = n_workers
        self.days            = days
        self.curve_every     = max(1, curve_every)

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
                  self.lot_size, self.starting_cash, self.cost_kwargs, self.curve_every)
                 for p in paths]

        n = min(self.n_workers or (os.cpu_count() or 1), len(paths))
        if parallel and n > 1:
            with ProcessPoolExecutor(max_workers=n) as ex:
                outs = list(ex.map(_run_one_day, jobs))      # preserves order
        else:
            outs = [_run_one_day(j) for j in jobs]

        good    = [o for o in outs if o[1] is not None]      # curve is None on a load error
        skipped = [(o[0], o[5]) for o in outs if o[1] is None]
        if skipped:
            log.info("%d day(s) skipped: %s", len(skipped),
                     "; ".join(f"{d} ({m})" for d, m in skipped))

        days       = [o[0] for o in good]
        curves     = {o[0]: o[1] for o in good}
        Tradelogs  = {o[0]: o[2] for o in good}
        pnls       = [o[3] for o in good]
        perseclogs = {o[0]: o[4] for o in good}
        res = Results(days, curves, Tradelogs, pnls, self.starting_cash, self.lot_size, perseclogs)
        res.log_path = log.log_path                          # where this run was logged

        # log what the Results object produces, instead of printing it
        if days:
            log.info("per-day summary:\n%s", res.summary.to_string())
            log.info("aggregate stats:\n%s",
                     "\n".join(f"  {k:<16}: {v}" for k, v in res.stats().items()))
        else:
            log.info("no successful days")
        for h in log.handlers:
            h.flush()
        print(f"[birdseye] log -> {log.log_path}")          # one-line pointer, not the logs themselves
        return res