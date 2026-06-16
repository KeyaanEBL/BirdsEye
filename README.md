# BirdsEye

**A lightweight, per-second, event-driven options backtesting framework — strategies as finite state machines, parallel by default.**

BirdsEye replays one trading day per second from per-strike options data (0-DTE NIFTY / SPY style parquet files), runs a strategy modelled as an explicit **finite state machine**, fills orders through a cost-aware broker, and reports per-day and aggregate performance — with every day running in its own process across all available cores.

It is deliberately small. The design borrows the good ideas from [Backtrader](https://github.com/mementum/backtrader) (cursor-style history access, warm-up periods, pluggable analyzers) and strips everything else, so the entire engine can be read, understood, and extended in an afternoon.

---

## How it works — one sentence per layer

```
Feed ──► Snapshot ──► Strategy (FSM) ──► Executing ──► Broker ──► Portfolio
                          │                                          │
                      PerSecLog                                  Tradelog
                          └────────────► Results ◄───────────────────┘
```

- **`Feed`** loads one day's parquet (only the columns it needs), discovers the strike grid from the column names, and stacks every requested per-strike field into `(n_seconds × n_strikes)` numpy arrays — once.
- **`MarketSnapshot`** is a *cursor* over that day: "now" plus all history up to now. Strategies read current values (`quote`, `mid_half`, `field`) and history (`spot_hist(n)`, `field_hist(strike, opt_type, field, n)`) — every history accessor is clamped to the current second, so **lookahead is impossible by construction**.
- **`StateMachineStrategy`** is the strategy layer: the user declares states, per-state transition tables of **named guards**, and a per-second alpha function. When a guard fires, the destination state builds an `Order`.
- **`Executing`** — *execution is a state*. Firing an order moves the FSM into `EXECUTING`, which slices the order into the broker over ticks (respecting `slice_lots` / `pause`). No user transitions are evaluated mid-execution; the destination state's `on_enter` runs at **fill-complete**. One order in flight per strategy, ever.
- **`Broker`** fills every leg at **mid**; the bid/ask spread is charged once as an explicit cost (see *Cost model*). Fills are recorded in the **`Tradelog`** and applied to the **`Portfolio`**, which marks the book to market at mid every second.
- **`BirdsEye`** (the runner) wires all of the above per day, fans days out over a `ProcessPoolExecutor`, and returns a **`Results`** object that computes stats, tables, and plots on demand.

---

## Quickstart

### 1. Define a strategy (as an importable module in `strategies/`)

```python
# strategies/range_short_strangle.py
from engine import Order, OrderLeg, Reason, State, StateMachineStrategy, Context


class Wait(State):
    name = "WAIT"
    transitions = {"calm_entry": "SHORT"}           # guard name -> destination

    def target(self, alphas, ctx):
        atm = ctx.get("short_atm")
        if atm is None:
            return None                             # nothing to unwind yet
        return ctx["strat"].close_legs([(atm, "CE"), (atm, "PE")],
                                       reason=Reason(state="WAIT", note="square off"))


class Short(State):
    name = "SHORT"
    transitions = {
        "stop_hit":      "WAIT",
        "hold_elapsed":  "WAIT",
    }

    def target(self, alphas, ctx):
        ce_k = alphas["ce_strike"]
        pe_k = alphas["pe_strike"]
        lots = ctx["strat"].lots
        ctx["short_ce"]   = ce_k
        ctx["short_pe"]   = pe_k
        ctx["entry_prem"] = alphas["ce_mid"] + alphas["pe_mid"]
        return Order(
            name="short_strangle",
            legs=[
                OrderLeg(ce_k, "CE", lots=lots, action="SELL",
                         slice_lots=lots, pause=0),
                OrderLeg(pe_k, "PE", lots=lots, action="SELL",
                         slice_lots=lots, pause=0),
            ],
            reason=Reason(state="SHORT"),
        )

    def on_enter(self, ctx):                       # runs at FILL-COMPLETE
        ctx["entry_sec"] = ctx["now_sec"]


class RangeShortStrangle(StateMachineStrategy):
    states   = {"WAIT": Wait(), "SHORT": Short()}
    max_lots = LOTS
    # slice_lots and pause are NOT class-level — they live on each OrderLeg

    def __init__(
        self,
        broker,
        lots          = LOTS,
        max_lots      = None,
        range_win     = RANGE_WIN,
        hold          = HOLD,
        range_bps_max = RANGE_BPS_MAX,
        min_dist_bps  = MIN_DIST_BPS,
        stop_mult     = STOP_MULT,
        skip_open     = SKIP_OPEN,
        session_len   = SESSION_LEN,
    ):
        self.lots          = lots
        self.max_lots      = max_lots if max_lots is not None else lots * 2
        self.range_win     = range_win
        self.hold          = hold
        self.range_bps_max = range_bps_max
        self.min_dist_bps  = min_dist_bps
        self.stop_mult     = stop_mult
        self.skip_open     = skip_open
        self.session_len   = session_len
        self.min_history   = range_win               # warm-up: no decisions before this

        ctx             = Context()
        ctx["short_ce"] = None
        ctx["short_pe"] = None
        super().__init__("WAIT", broker, name="range_short_strangle", context=ctx)
        ctx["strat"] = self

    # ---- named guards: documented, testable, ledger-visible ----
    def guard_calm_entry(self, a, c):
        return (a["sec"] >= self.skip_open
                and a["range_bps"] < self.range_bps_max
                and a["ce_strike"] is not None
                and a["pe_strike"] is not None
                and a["sec"] < self.session_len - self.hold)

    def guard_stop_triggered(self, a, c):
        ep = c.get("entry_prem")
        cp = a.get("current_prem")
        if ep is None or cp is None or ep <= 0:
            return False
        return cp > self.stop_mult * ep

    def guard_hold_elapsed(self, a, c):
        return a["sec"] - c.get("entry_sec", 0) >= self.hold

    # ---- alphas, computed every second -------------------------
    def compute_alphas(self, snap):
        c    = self.context
        sec  = snap.i
        spot = snap.spot
        c["now_sec"] = sec

        h   = snap.spot_hist(self.range_win)
        rng = (h.max() - h.min()) / spot * 1e4

        strikes = snap._feed.strikes
        thr     = self.min_dist_bps

        def _quoted(s, opt_type):
            try:
                b, a = snap.quote(float(s), opt_type)
                return np.isfinite(b) and np.isfinite(a)
            except Exception:
                return False

        ce_k = next((float(s) for s in strikes
                     if (s - spot) / spot * 1e4 >= thr and _quoted(s, "CE")), None)
        pe_k = next((float(s) for s in strikes[::-1]
                     if (spot - s) / spot * 1e4 >= thr and _quoted(s, "PE")), None)

        ce_mid = pe_mid = np.nan
        if ce_k is not None:
            cb, ca = snap.quote(ce_k, "CE"); ce_mid = (cb + ca) / 2
        if pe_k is not None:
            pb, pa = snap.quote(pe_k, "PE"); pe_mid = (pb + pa) / 2

        current_prem = None
        if c.get("short_ce") is not None:
            try:
                cb, ca = snap.quote(c["short_ce"], "CE")
                pb, pa = snap.quote(c["short_pe"], "PE")
                current_prem = (cb + ca + pb + pa) / 2
            except Exception:
                pass

        return {
            "sec":          sec,
            "spot":         spot,
            "range_bps":    rng,
            "ce_strike":    ce_k,
            "pe_strike":    pe_k,
            "ce_mid":       ce_mid,
            "pe_mid":       pe_mid,
            "current_prem": current_prem,
        }
```

### 2. Run it

```python
from engine import BirdsEye
from strategies.range_short_strangle import RangeShortStrangle

be = BirdsEye(
    data_dir          = "/path/to/SPY/0-dte/train",   # one parquet per day
    strategy_cls      = RangeShortStrangle,
    index             = "SPY",                         # "SPY" (252) or "NIFTY" (52)
    lot_size          = 100,
    starting_cash     = 1_000_000.0,                  # fresh capital every day
    margin_per_lot    = 10_000.0,                      # used for CAGR / Calmar
    strategy_kwargs   = {"lots": 10, "max_lots": 10},
    cost_kwargs       = {"txn_cost_per_lot": 0.85},    # or {"txn_cost_bps": 15}
    n_workers         = 20,
    collect_perseclog = True,                          # False by default; ~5–10x IPC cost
)
res = be.run()          # one process per day, chronological order preserved
```

**`index`** controls the annualisation multiplier used in CAGR and Calmar:
`"SPY"` → 252 trading days/year, `"NIFTY"` → 52 expiry weeks/year.

**`margin_per_lot`** is the per-lot capital requirement. `margin = margin_per_lot × max_lots` is the denominator for all return and drawdown normalisations. Set it to match the actual margin your broker charges for the strategy's maximum open position.

**`collect_perseclog`** defaults to `False` because the per-second flight recorder adds roughly 5–10× to the IPC pickle cost per worker. Set it to `True` when you need intraday diagnostics (`res.perseclog(day)`).

### 3. Inspect everything

```python
res.summary                 # per-day table: fills / gross / costs / net
res.stats()                 # CAGR, Calmar, churn, win rate, drawdowns, costs … (2dp)
res.tearsheet()             # daily PnL + stitched equity + drawdown figure
res.plot_day("20240208")    # one day's intraday MtM with buy/sell markers
res.Tradelog()              # every fill, every day, one DataFrame
res.perseclog("20240208")   # the per-second flight recorder for one day
```

---

## The FSM strategy model

A strategy is a **lifecycle**, not one decision repeated: flat → entering → holding → exiting. BirdsEye makes that explicit. You declare three things:

| You declare | What it is |
|---|---|
| **States** | What to hold while in each phase. `target(alphas, ctx) -> Order` builds the trade on entry; `on_enter` / `on_exit` hooks manage context. |
| **Transitions** | A per-state dict `{guard_name: destination}` — a readable switch-case, evaluated in insertion order, first hit wins. Guards are named methods (`guard_<name>`) on the strategy; bare callables are also accepted for quick experiments. |
| **Context** | The strategy's rolling *decision* memory (entry second, locked strike, …). Market history does **not** live here — it comes from the snapshot. |

```python
print(strat.describe())
#       WAIT --[calm_entry]--> SHORT
#      SHORT --[stop_hit]--> WAIT
#      SHORT --[hold_elapsed]--> WAIT
```

Because guards are named, the **Tradelog records exactly which condition fired every trade** (`signal` column), along with the full alpha values at fire time (`alpha_*` columns). "Are all my losses coming from stops?" is a one-line groupby.

### Execution is a state

```
WAIT --guard fires--> EXECUTING(order, dest=SHORT) --all lots filled--> SHORT
```

While `EXECUTING`, the FSM evaluates no user transitions — a new intent must wait until the in-flight order is fully placed. Each leg is worked independently according to its own `slice_lots` and `pause` — set these on the `OrderLeg`, not on the strategy class. If execution releases slices for 60 straight ticks with zero fills (an unquoted strike), it **raises loudly** naming the starving legs instead of silently eating the day.

### Orders are deltas

An `Order` means *"trade these lots now"* — `lots` is always positive, direction lives in `action` (`"BUY"` / `"SELL"`). Execution pacing is set **per leg** via `slice_lots` and `pause` on `OrderLeg`, not on the strategy:

```python
OrderLeg(strike, "CE", lots=10, action="SELL", slice_lots=10, pause=0)
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                       how many lots to release per tick, and how long to wait between slices
```

Every leg in the same `Order` can have different pacing if needed. Flattening reads the live position once, through one helper:

```python
self.close_legs([(strike, "CE"), (strike, "PE")], reason="square off")
```

---

## Data format

One parquet file per day, named `YYYYMMDD.parquet`. Expected layout:

- **Base columns**: `spot`, `atm_strike` (required); `timestamp` as a column *or* the index (optional — row order is used if absent). Common aliases (`ts`, `spotPrice`, …) resolve automatically, case-insensitively.
- **Per-strike columns**: `{strike}_{ce|pe}_{field}` — e.g. `21150_ce_bid_0`, `471.00_pe_premium`. Strike tokens are taken **verbatim from the file** (integer, decimal, two-decimal — all fine) and matched case-insensitively.
- Strikes are discovered from the `*_premium` columns; `Feed.from_parquet(path, fields=(...))` controls which per-strike fields get loaded (`bid_0`/`ask_0` by default — add `"iv"`, `"delta"`, `"ttv"`, … for greeks/flow alphas, then read them via `snap.field(...)` / `snap.field_hist(...)`).

Only the requested columns are read from disk; everything is stacked into shared numpy arrays once per day.

---

## Cost & fill model

Fills happen **at mid** for every leg, regardless of direction. The bid/ask spread is charged exactly once, explicitly:

| Cost | Formula | Configure |
|---|---|---|
| Spread | `half_spread × lot_size × lots` | derived from the live quote |
| Transaction (absolute) | `txn_cost_per_lot × lots` | e.g. SPY: `{"txn_cost_per_lot": 0.85}` |
| Transaction (percentage) | `bps of (mid × lot_size × lots)` | e.g. NIFTY: `{"txn_cost_bps": 15}` |
| Brokerage | `brokerage_per_lot × lots` | optional, additive |

Both transaction modes can be combined (they sum). Mark-to-market is always at mid, so the equity curve never double-counts the spread: a fresh position shows zero unrealized PnL and a cash debit equal to its frictions — a handy correctness check.

PnL accounting uses weighted-average cost basis; realized PnL books when a position is reduced or closed.

---

## Stats

`res.stats()` reports (all at 2dp):

**Per-day breakdown**
- total / average PnL, win rate, % positive/negative days, average win/loss, best/worst day

**Return metrics**

| Metric | Formula |
|---|---|
| **CAGR (gross/net)** | `(total_pnl × periods_per_year) / (margin × n_days)` |
| **Calmar** | `CAGR / \|max % drawdown\|` on the compounded daily curve |
| **churn_per_day** | `total_traded_lots / 2` — round-trips per day |

Where `margin = margin_per_lot × max_lots` and `periods_per_year` is set by the `index` parameter (252 for SPY, 52 for NIFTY).

**Drawdown**
- Max drawdown three ways: % and $ on the compounded daily curve; $ intraday across every second (`max(cumsum.cummax − cumsum)`, normalised by margin)

**Costs**
- Total frictions (spread + transaction), total fill count, average cost per round-trip

---

## Repository layout

```
BirdsEye/
├── engine/
│   ├── feed.py          # parquet -> numpy arrays; strike/field discovery
│   ├── snapshot.py      # the cursor: now + lookahead-clamped history
│   ├── orders.py        # Order / OrderLeg (lots>0 + action) / Reason
│   ├── strategy.py      # State / Context / StateMachineStrategy (named guards)
│   ├── execution.py     # Executing — execution-as-a-state + starvation guard
│   ├── broker.py        # mid fills + costs -> Tradelog + Portfolio
│   ├── portfolio.py     # positions, cash, realized/unrealized PnL, equity curve
│   ├── costs.py         # CostModel: abs + bps transaction modes
│   ├── ledger.py        # Tradelog (per fill) + PerSecLog (per second)
│   ├── analyzers.py     # cagr / calmar / churn / drawdowns / daily stats
│   ├── plots.py         # equity / drawdown / daily-PnL helpers
│   └── runner.py        # BirdsEye + Results — parallel orchestration
├── strategies/          # one importable module per strategy
│   └── range_short_strangle.py
└── README.md
```

Requirements: Python ≥ 3.9, `numpy`, `pandas`, `pyarrow`, `matplotlib`.

---

## Design decisions (and why)

- **Target the lifecycle, not the tick.** FSMs make the legal phases and the legal moves between them explicit; whole classes of bugs (re-entering while exiting, scaling while flat) become unrepresentable.
- **History through the snapshot, never through context.** One source of truth for market data, clamped to *now* — strategies cannot peek at the future even by accident.
- **Mid + spread/2 everywhere.** No crossed-side branching; the spread is a visible, auditable cost line instead of an implicit fill penalty.
- **One order in flight per strategy.** "Wait out the old order" removes the queue, the residual reconciliation, and the stale-intent problem in one move.
- **Fail loudly.** Unquoted strikes raise with the leg names; worker errors carry their day name. A silent empty day costs an afternoon; a loud one costs a minute.
- **Pure per-day workers.** Each day builds a fresh feed/portfolio/broker/strategy, which makes day-level parallelism trivially safe and results order-independent.

---

## Roadmap

- Declarative FSM definitions (YAML guards, nested states)
- Multiple strategies on one book with sleeved per-strategy attribution
- Execution realism: TWAP/participation schedules, slippage & impact models
- Richer analytics and a graphviz view of `describe()`
- Live/paper-trading parity behind the existing `Feed`/`Broker` seams
