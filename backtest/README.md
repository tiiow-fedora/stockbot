# backtest/

Additive harness over the existing `backtesting.engine.Backtester` (same no-lookahead
fills, risk manager, paper executor). **Paper simulation only** - no broker imports, no credentials.

```bash
python -m backtest --synthetic --strategy mean_reversion --out backtest_reports
python -m backtest --csv-dir data_files --symbols AAPL MSFT --strategy technical
uvicorn --factory backtest.web:create_app   # read-only dashboard (env: BACKTEST_REPORTS_DIR, STOCKBOT_DB_URL)
```

- `scenarios.py` - seeded synthetic regimes: bull, bear, sideways, volatile, crash
- `loaders.py` - CSV (via existing `CSVMarketDataProvider`) or synthetic series
- `runner.py` - `BacktestHarness`: one engine run per series, aggregated as equal-capital sleeves
- `metrics.py` - win rate, profit factor, avg win/loss, drawdown (% and $, peak/trough times)
- `report.py` - JSON `RunReport` save/load
- `web.py` - GET-only FastAPI app: latest paper positions (from DB) + saved reports
