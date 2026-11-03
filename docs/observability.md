# Observability

Paper trading only; nothing here touches order placement behaviour.

| Feature | How |
|---|---|
| Metrics | `GET :9108/metrics` while `main.py run` is up (`METRICS_PORT`, `0` disables) |
| Health | `/healthz` (liveness: loop progressed within a staleness window), `/readyz` (first cycle done + DB reachable) |
| JSON logs | `LOG_FORMAT=json` |
| Graceful shutdown | SIGTERM/SIGINT: finish the current cycle, or wake from sleep immediately, then exit 0 |
| Config validation | `python main.py validate-config`; `run` and `dashboard` refuse to start on errors (exit 2) |
| Grafana | `grafana/stockbot-dashboard.json` (import manually, or use the compose overlay) |

Monitoring stack: `docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d`
(Prometheus on 127.0.0.1:9090, Grafana on 127.0.0.1:3000).

## Metrics
`stockbot_portfolio_value_dollars`, `_portfolio_cash_dollars`, `_positions_value_dollars`, `_open_positions`,
`_decisions_total{symbol,action}`, `_orders_total{side,result}`, `_risk_rejections_total{reason}`,
`_data_errors_total{symbol}`, `_cycles_total{outcome}`, `_cycle_duration_seconds`, `_consecutive_failures`,
`_last_successful_cycle_timestamp_seconds`, `_market_open`, `_info{execution_mode,data_provider,strategy}`.

Backtests reuse `route_order`, so a backtest run in the same process would also bump order counters; the
metrics server is only started by `main.py run`.

## Validation highlights
Fractions in (0,1] (catches `20` for 20%), threshold ordering, indicator periods, known providers/strategy,
credentials present when Alpaca is configured, and a paper-only guard that rejects a non-paper
`APCA_API_BASE_URL` or `ALPACA_PAPER=false`.
