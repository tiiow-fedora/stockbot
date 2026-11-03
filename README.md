# StockBot

A modular, deterministic paper-trading research system built for a high
school stock market simulation class. The classroom platform isn't known
yet, so the engine is built against clean abstract interfaces
(`MarketDataProvider`, `NewsDataProvider`, `Strategy`, `OrderExecutor`)
and ships with a synthetic/CSV/Alpaca data provider today. When the class
platform is known, only a new provider + executor adapter need to be
written - nothing else in the pipeline changes.

**This is a paper-trading simulator. It never places real trades or
touches real money.**

## Pipeline

```
DATA -> ANALYSIS -> STRATEGY -> RISK MANAGER -> ORDER -> EXECUTOR -> PORTFOLIO -> DATABASE
```

Every stage only knows about the standardized types in [core/models.py](core/models.py)
(`MarketSnapshot`, `Candle`, `NewsItem`, `Decision`, `Order`, `Fill`,
`Position`) - never about where data came from or where orders end up.

## Project layout

```
stockbot/
├── config/           # config.yaml (non-secret settings)
├── core/             # models.py (shared types), config.py, logging.py
├── data/             # MarketDataProvider / NewsDataProvider interfaces, market_calendar.py + providers/
│   └── providers/    # synthetic (dev), csv (backtesting), alpaca (real paper data)
├── analysis/         # indicators.py (SMA/EMA/RSI/MACD/...), sentiment.py (lexicon + optional LLM)
├── strategy/         # base.py interface, scoring.py, technical.py, mean_reversion.py, registry.py
├── portfolio/         # portfolio.py (cash/positions ledger), risk.py (RiskManager)
├── execution/        # base.py interface, paper.py, manual.py, alpaca_mirror.py (optional), order_builder.py (shared)
├── backtesting/       # engine.py (no-lookahead sim), compare.py, optimize.py (grid search)
├── database/          # database.py: SQLAlchemy models (SQLite or Postgres)
├── dashboard/          # app.py: read-only FastAPI dashboard + static/index.html
├── tests/             # pytest suite
├── main.py            # CLI: run / backtest [--compare] / optimize / init-db / dashboard
├── Dockerfile, docker-compose.yml
└── requirements.txt
```

## Why the structure differs slightly from the original spec

Two small additions beyond the original tree, both in the spirit of
"preserve separation of concerns":

- **`core/`** - `models.py` holds every shared type so `strategy/`, `data/`,
  `execution/`, etc. never import from each other directly, only from
  `core`. `config.py` and `logging.py` also live here since every layer
  needs them.
- **`execution/order_builder.py`** - the "Decision -> Order -> risk check ->
  fill" logic is identical between the live loop and the backtester. It's
  factored into one shared module so those two paths can never quietly
  diverge (which would make backtests stop predicting live behavior).
- **`strategy/registry.py`** - single source of truth for "which
  strategies exist" (`technical`, `mean_reversion`). `main.py`,
  `backtesting/compare.py`, and `backtesting/optimize.py` all read from
  it, so adding a third strategy means adding one entry, not touching
  three call sites.

## Quick start (local, no Docker)

Use a virtual environment - this machine has other Python tools installed
globally (e.g. `spotdl`) with their own pinned dependency versions, and
`pip install -r requirements.txt` at the system level can silently break
them (it did once during development; see git history).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # only needed if you set provider: alpaca

python main.py init-db                          # creates stockbot.db (SQLite)
python main.py backtest --days 365 --persist    # backtests all configured symbols, saves results
python main.py backtest --symbol AAPL --compare # run every registered strategy side by side
python main.py optimize --symbol AAPL           # grid-search strategy parameters (research tool)
python main.py dashboard                        # http://localhost:8000
python main.py run                              # continuous paper-trading loop (Ctrl+C to stop)
```

With the default `config/config.yaml`, everything runs against the
built-in **synthetic** data provider - a seeded random walk - so there is
zero setup required and results are reproducible. Switch `data.provider`
to `csv` (put files in `data_files/`) or `alpaca` (set API keys in `.env`)
when you want real historical data.

## Quick start (Docker)

```bash
docker compose up -d
docker compose logs -f stockbot
```

This starts Postgres, the trading loop, and the dashboard (on
`localhost:8000`) together; `DATABASE_URL` is injected via the compose
file so the bot talks to the containerized Postgres instead of SQLite.
Edit `config/config.yaml` before building to change symbols, strategy
weights, or risk limits.

## Running tests

```bash
pytest
```

183 tests across indicators, scoring, portfolio accounting, risk limits,
execution (slippage/commission, the Alpaca mirror executor, and the
live-loop market-hours heartbeat - all mocked, not live), backtesting
(including an explicit no-look-ahead check and optional-news bounds),
data validation/resilience, the dashboard API, Alpaca/Anthropic
credential handling, sentiment scoring, both strategies (including a
direct check that they read the same signal in opposite directions),
strategy comparison, parameter grid search, and the market calendar
(deterministic date math, no network). None are placeholders - each
asserts a specific numeric or behavioral outcome. (Tests never hit the
network - they exercise the synthetic provider and mock credentials/
clients, not live API calls.)

## Configuration

All tunables live in [config/config.yaml](config/config.yaml): symbols,
starting balance, strategy weights/thresholds/indicator periods, risk
limits, execution costs, database URL, polling interval. Secrets
(`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) go in `.env` (copy from
`.env.example`) and are never read from YAML.

## What currently works

- Full pipeline wired end-to-end: data -> indicators -> weighted scoring
  -> BUY/HOLD/SELL -> risk checks (position size, exposure, open
  positions, trade frequency, cash reserve) -> paper or manual execution
  -> portfolio accounting -> SQLite/Postgres persistence.
- Deterministic `TechnicalStrategy` using SMA/EMA crossover, RSI, MACD,
  momentum, volume change, and volatility, combined via a fully
  configurable weighted scoring engine. No LLM in the decision path.
- Backtesting engine with an explicit no-look-ahead design (decisions use
  only data through bar *i-1*'s close; fills happen at bar *i*'s open)
  and a buy-and-hold benchmark run through the same executor for a fair
  comparison. Reports total return, win rate, max drawdown, Sharpe ratio.
- Risk manager that the strategy cannot bypass: max position size, max
  portfolio exposure, max open positions, max trades/symbol/day, stop
  loss, take profit, minimum cash reserve - all configurable.
- Every decision, order, fill, risk rejection, and portfolio snapshot is
  persisted with enough detail (full indicator values, score breakdown,
  reason string) to reconstruct *why* a trade did or didn't happen.
- Synthetic data provider (seeded, reproducible, no API key) plus CSV and
  **Alpaca** providers behind the same interface. Alpaca is now the
  configured default and has been verified live (real historical bars,
  a real latest-trade snapshot, and real news) against a free paper
  account - not just written and assumed to work. One real bug this
  caught: free/paper accounts only have `IEX` feed access, not the
  full-market `SIP` feed, which 403s on recent data without a paid
  subscription; both `get_historical_candles` and `get_snapshot` now
  request `DataFeed.IEX` explicitly. Switch `data.provider` back to
  `synthetic` any time for a zero-setup, offline, reproducible run
  (e.g. for the tests, which never hit the network).
- News is fully optional (`NullNewsProvider` by default); the scoring
  engine redistributes sentiment's weight across the other components
  when no news is available rather than silently zeroing part of the
  score.
- **News sentiment scoring** (`analysis/sentiment.py`): `LexiconSentimentScorer`
  is the default - deterministic keyword matching, no API key, no network
  call, verified against real Alpaca headlines. It returns `None` (not a
  fake `0.0`) when a headline has no detectable signal, which the scoring
  engine treats as "no opinion" rather than "confirmed neutral." An
  optional `LLMSentimentScorer` exists behind the exact same interface
  (`news.sentiment_scorer: llm` + `ANTHROPIC_API_KEY`) but is off by
  default and untested against a live key - the strategy works
  identically either way, which is the point of the interface.
- **A second strategy** (`strategy/mean_reversion.py`): `MeanReversionStrategy`
  is a contrarian counterpart to `TechnicalStrategy` - it reads the same
  price-vs-SMA divergence in the opposite direction (buy the dip instead
  of following the trend) and fades momentum instead of following it.
  Proves the `Strategy` interface is genuinely swappable, and gives
  `backtest --compare` two meaningfully different strategies to compare,
  not two copies of the same logic.
- **Strategy comparison** (`backtesting/compare.py`, `main.py backtest --compare`):
  runs every registered strategy against the same historical data with
  independent portfolios and prints a side-by-side table (return, vs.
  benchmark, trades, win rate, max drawdown, Sharpe).
- **Parameter grid search** (`backtesting/optimize.py`, `main.py optimize`):
  sweeps `buy_threshold` / `sell_threshold` / `sma_short_period` /
  `sma_long_period` and ranks combinations by return or Sharpe. Prints an
  explicit overfitting warning every time - a top result is fit to the
  exact historical window it was tested on, not a promise about the future.
- Read-only web dashboard (`python main.py dashboard`, or the `dashboard`
  service in `docker-compose.yml`) showing current portfolio value/cash/
  return, an equity curve, open positions, current signals per symbol,
  recent trades, and the full recent decision log with each reason
  string. It only reads from the database - it never talks to the
  trading engine directly, so it can't affect a live run. The backtester
  now also persists a portfolio snapshot per bar (previously only the
  live loop did), so a `--persist` backtest is fully viewable here too.
- **Optional Alpaca mirroring** (`execution/alpaca_mirror.py`,
  `execution.mirror_to_alpaca: true`): Alpaca is otherwise used only as a
  read-only data source - the local `PaperExecutor`/`Portfolio` is always
  the authoritative record of what StockBot "owns," which is why Alpaca's
  own dashboard shows nothing by default even while StockBot is actively
  trading. `AlpacaMirrorExecutor` wraps the local executor and, purely for
  visualization, best-effort mirrors every fill to your own Alpaca PAPER
  account too (hard-coded `paper=True` - can never touch a live account).
  Verified live: submitted a real order, confirmed via Alpaca's own API
  that it was accepted (queued for the next market open, since the
  market was closed at the time - completely normal). If the mirror call
  fails for any reason, it's logged and swallowed; the local fill that
  already happened is never affected, so the two ledgers can drift out of
  sync over time - that's an accepted tradeoff for a cosmetic feature.
- **Market-hours awareness** (`data/market_calendar.py`,
  `data.market_calendar: auto`): the live loop (`main.py run`) now checks
  whether the market is open before running the strategy/order pipeline
  at all - it used to poll blindly around the clock, which meant the
  local paper simulator would happily fill orders instantly at a stale
  price even at 2am or on a weekend (Alpaca's own paper account correctly
  queues orders until the next open; ours didn't model that at all until
  now). `AlpacaMarketCalendar` uses Alpaca's real clock (exact hours,
  real holidays); `USEquityMarketCalendar` is a deterministic offline
  fallback (Mon-Fri 9:30-16:00 ET, no holiday awareness) for when you're
  not using Alpaca; `AlwaysOpenCalendar` is the default for synthetic/CSV
  data. While the market's closed, the loop still writes a heartbeat
  portfolio snapshot every 30 min (so the dashboard shows "running," not
  "stale," across a long overnight/weekend gap) but places no trades.
  Verified live against the real closed market: correctly logged "Market
  closed - next open 2026-08-12T09:30:00-04:00" and skipped trading.

## Classroom simulator adapter (Milestone 6)

Platform-agnostic, paper-only, no network or credentials:

- `data.provider: class_simulator` reads quotes from a JSON file
  (`data.class_simulator_quotes_path`) and history from the CSV dir; the
  file contract and units (dollars per share, shares-so-far volume) are in
  `data/providers/class_simulator.py`.
- `execution.mode: class_simulator` appends whole-share order tickets to a
  CSV outbox (`execution.class_simulator_outbox`) for you to place on the
  platform. Like `manual`, it never fills locally, and tickets are
  idempotent across restarts.
- Set `data.market_calendar: always_open` (or `us_equity`) as appropriate.

## Unit-consistency guarantees

Regression tests in `tests/test_units.py`: snapshot volume must be in the
same unit as daily history volume (today's partial Alpaca bar is projected
to a full day and excluded from history so it isn't duplicated); volume 0
means "unavailable", not a collapse; `*_pct` risk settings are fractions
(0.08 = 8%) and are validated; `slippage_bps` is basis points; order
quantities are whole shares.

## Limitations / not yet built

- **Portfolio state is in-memory per process** - restarting `main.py run`
  starts a fresh paper portfolio at `starting_balance`. Snapshots are
  persisted to the DB for audit, but the live loop doesn't yet reload the
  last snapshot on startup. Worth fixing before a long unattended run.
- **Classroom adapter is file-based, not a vendor API** (Milestone 6) -
  see "Classroom simulator adapter" below. A direct API integration is
  still future work once the platform is known.
- **Single-symbol-per-run backtester** - `main.py backtest` loops over
  symbols and reports each individually plus an average; there's no
  portfolio-level multi-symbol backtest (shared cash across symbols
  simultaneously) yet. `compare`/`optimize` are also single-symbol.
- **The lexicon sentiment scorer is intentionally simple** - keyword
  matching has no contextual understanding (e.g. "costs could jump" can
  score positive purely because "jump" is in the positive word list, even
  though the headline is about rising costs). It's an honest, cheap,
  deterministic signal - not a claim of accuracy.
- **Backtesting with real historical news is supported but opt-in and
  slow** - `Backtester(..., news_provider=..., sentiment_scorer=...)`
  fetches news per bar bounded to what was known as of that bar (no
  look-ahead), but a live provider like Alpaca means one network call per
  bar, which is slow and can hit rate limits over a long backtest.
  `main.py backtest`/`--compare`/`optimize` don't wire this up by default
  for that reason; it's proven correct in tests with an in-memory fake
  provider instead.
- **The parameter grid search only sweeps top-level scalar fields** (not
  the nested `weights`), and has no walk-forward/out-of-sample splitting
  built in - the printed warning is the whole safeguard right now.

## Recommended next steps

1. Persist/restore portfolio state across `main.py run` restarts.
2. Once you know the classroom platform: add
   `data/providers/class_simulator.py` implementing `MarketDataProvider`
   and `execution/class_simulator.py` implementing `OrderExecutor` (or a
   `ManualExecutor`-style suggestion queue if automated trading isn't
   allowed). Everything upstream is already platform-agnostic.
3. If you want to actually exercise the optional LLM sentiment path, set
   `news.sentiment_scorer: llm` and `ANTHROPIC_API_KEY` and try it against
   real headlines - it's implemented and credential-guarded but has never
   been run live (unlike the Alpaca providers, which have been).
4. Add walk-forward validation to `backtesting/optimize.py`: split the
   history into a fit window and a held-out test window, and only trust
   parameters that hold up on the window they weren't tuned on.
