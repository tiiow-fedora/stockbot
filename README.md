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
├── data/             # MarketDataProvider / NewsDataProvider interfaces + providers/
│   └── providers/    # synthetic (dev), csv (backtesting), alpaca (real paper data)
├── analysis/         # indicators.py: SMA, EMA, RSI, MACD, momentum, volume, volatility
├── strategy/         # base.py interface, scoring.py (weighted composite), technical.py
├── portfolio/         # portfolio.py (cash/positions ledger), risk.py (RiskManager)
├── execution/        # base.py interface, paper.py, manual.py, order_builder.py (shared)
├── backtesting/       # engine.py: no-lookahead simulation + buy-and-hold benchmark
├── database/          # database.py: SQLAlchemy models (SQLite or Postgres)
├── tests/             # pytest suite
├── main.py            # CLI: run / backtest / init-db
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

## Quick start (local, no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env      # only needed if you set provider: alpaca

python main.py init-db                 # creates stockbot.db (SQLite)
python main.py backtest --days 365     # backtests all configured symbols against synthetic data
python main.py run                     # continuous paper-trading loop (Ctrl+C to stop)
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

This starts Postgres + the bot together; `DATABASE_URL` is injected via
the compose file so the bot talks to the containerized Postgres instead
of SQLite. Edit `config/config.yaml` before building to change symbols,
strategy weights, or risk limits.

## Running tests

```bash
pytest
```

62 tests across indicators, scoring, portfolio accounting, risk limits,
execution (slippage/commission), backtesting (including an explicit
no-look-ahead check), and data validation/resilience. None are
placeholders - each asserts a specific numeric or behavioral outcome.

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
  Alpaca provider implementations behind the same interface.
- News is fully optional (`NullNewsProvider` by default); the scoring
  engine redistributes sentiment's weight across the other components
  when no news is available rather than silently zeroing part of the
  score.

## Limitations / not yet built

- **No dashboard yet** (Milestone 3). `database.get_recent_decisions()`
  already exists as a data source for one.
- **Portfolio state is in-memory per process** - restarting `main.py run`
  starts a fresh paper portfolio at `starting_balance`. Snapshots are
  persisted to the DB for audit, but the live loop doesn't yet reload the
  last snapshot on startup. Worth fixing before a long unattended run.
- **News sentiment scoring is a stub** - `AlpacaNewsProvider` fetches
  headlines but does not score sentiment (Alpaca doesn't provide it).
  `NewsItem.sentiment` stays `None` until a sentiment module is added; the
  scoring engine already handles that case correctly.
- **No classroom platform adapter yet** - intentionally deferred until
  the platform is known (Milestone 6). The interfaces it will implement
  (`MarketDataProvider`, `OrderExecutor`) already exist and are exercised
  by two other providers each, so the adapter should be a small, isolated
  addition.
- **Single-symbol-per-run backtester** - `main.py backtest` loops over
  symbols and reports each individually plus an average; there's no
  portfolio-level multi-symbol backtest (shared cash across symbols
  simultaneously) yet.

## Recommended next steps

1. **Milestone 3**: a small dashboard (FastAPI + a single HTML page, or a
   Streamlit app) reading from `DatabaseManager` - portfolio value, open
   positions, recent decisions/trades, current signals. The read-only
   query methods on `DatabaseManager` are the intended seam.
2. Persist/restore portfolio state across `main.py run` restarts.
3. Once you know the classroom platform: add
   `data/providers/class_simulator.py` implementing `MarketDataProvider`
   and `execution/class_simulator.py` implementing `OrderExecutor` (or a
   `ManualExecutor`-style suggestion queue if automated trading isn't
   allowed). Everything upstream is already platform-agnostic.
4. If/when real news becomes available, add a sentiment scorer (rule-based
   or, if you choose to, an optional LLM-based one) - it plugs into
   `TechnicalStrategy._sentiment_score` as just one more component score,
   never as the decision-maker.
