# CSV market data

Drop one file per symbol here, named `{SYMBOL}.csv`, e.g. `AAPL.csv`, with columns:

```
timestamp,open,high,low,close,volume
2023-01-03T00:00:00Z,130.28,130.90,124.17,125.07,112117500
2023-01-04T00:00:00Z,126.89,128.66,125.08,126.36,89113600
```

`timestamp` must be ISO-8601; naive timestamps are assumed UTC. Used when
`data.provider: csv` in `config/config.yaml` - mainly for backtesting
against real historical data you've downloaded in advance.
