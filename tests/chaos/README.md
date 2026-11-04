# Chaos suite

A fake Alpaca **paper** broker plus fault-injection tests that drive the *real*
StockBot pipeline (Alpaca data provider, Alpaca calendar, `AlpacaMirrorExecutor`,
`RiskManager`, DB, `run_live_loop`) with the failures a long-running bot actually
meets. Fully offline: sockets are blocked by an autouse fixture and the fake
refuses to build a non-paper `TradingClient`.

```bash
pytest tests/chaos            # ~40s
pytest tests/chaos -k seed7   # replay one fuzz seed
pytest tests/chaos --runxfail # see each documented bug actually fail
```

## Layout

| file | what it is |
|---|---|
| `fake_alpaca.py` | `FakeAlpacaBroker` (orders, partial fills, rejections, buying power, `client_order_id` uniqueness, clock, bars) + `FakeTradingClient`/`FakeDataClient` + `install()` |
| `bot.py` | `Bot` = one "process" of the live pipeline; make a second `Bot` over the same DB to simulate a restart. `patch_clock` makes `datetime.now` read a skewable fake clock |
| `loop_driver.py` | runs the real `main.run_live_loop` for N iterations without sleeping |
| `test_fake_broker.py` | self-tests, so a failure elsewhere is a bot bug and not a harness bug |
| `test_chaos_data.py` | timeouts/5xx/429, empty BarSet, corrupt bars, NaN, stale and gappy bars, duplicates, unsorted |
| `test_chaos_orders.py` | mirror under timeout/5xx/reject/partial/ambiguous-timeout; `route_order` with partial, unfilled and raising executors |
| `test_chaos_restart.py` | restart between cycles, kill between broker submit and DB write |
| `test_chaos_clock.py` | DST, weekends, skew, backwards NTP step, UTC-midnight rollover, non-UTC timestamps |
| `test_chaos_duplicates.py` | repeated signals, duplicate symbols, stop-loss + strategy SELL, two processes |
| `test_chaos_live_loop.py` | the real loop: outage backoff, market flapping, liveness guard ("an oversold stock **is** bought") |
| `test_chaos_fuzz.py` | seeded soak: random faults every cycle, ledger/DB/broker invariants after each |
| `test_chaos_properties.py` | indicator/strategy/risk property tests (does the sizer's order always pass the gate?), config unit mistakes, backtester on hostile series |
| `test_chaos_misc.py` | poisoned news, portfolio op fuzz, valuation during outages |

Tests for bugs that are **not** fixed are `xfail(strict=True)`: the suite stays
green, and the day someone fixes the bug the test flips to a hard failure that
says "remove the xfail".

## Findings

### Fixed in this branch (small, each with a regression test)

| id | severity | bug | fix |
|---|---|---|---|
| BUG-1 | **med** | Live history included today's still-forming daily bar, and the snapshot is built from that *same* bar, so the strategy saw today's price/volume twice (extra zero-return day in RSI/momentum; partial volume in its own baseline). Backtester uses `bars[:i]` + bar `i`, so live and backtest disagreed. | `main._completed_history` drops bars on/after the snapshot's date (daily timeframe). **Changes live signals slightly**: they now match the backtester. |
| BUG-2 | med | `bars[symbol]` raised bare `KeyError` for a symbol with no bars (halted/delisted/bad ticker). A single malformed bar (zero/NaN price, high<low) raised out of `get_historical_candles` and discarded that symbol's entire history. | provider returns `[]` for no data (→ `MarketDataError` from `get_snapshot`) and skips+logs malformed bars |
| BUG-2b | med | `NaN`/`inf` passed `price <= 0` validation in `MarketSnapshot`/`Candle` and flowed into indicators/sizing. | `core/models.py` requires finite and `> 0` |
| BUG-3 | **high** | Mirrored orders had no `client_order_id`. alpaca-py transparently retries `POST /orders` on 429/504; a 504 can mean "processed but gateway timed out" → second order. | mirror sets a UUID `client_order_id` (reused if the same `Order` is re-mirrored); Alpaca rejects duplicates |
| BUG-4 | **high** | alpaca-py never passes `timeout=` to `requests`. A half-open connection blocks the live loop **forever**: no exits, no heartbeat, nothing logged. The loop's retry logic never runs because the call never fails. | `core/http.apply_default_timeout` (5s connect / 20s read) applied to trading, calendar, data and news clients |
| BUG-8 | **high** | The stop-loss loop in `_run_cycle` was unguarded: one failing exit aborted the whole cycle (all other exits, all strategy decisions, the portfolio snapshot) and, since the same exit re-fires, wedged the loop for as long as the failure lasted. | per-order `try/except`, retried next cycle |
| BUG-15 | med | One news article with a naive timestamp raised from `get_news`; `_run_cycle` then skipped the *whole symbol* every cycle while that article was in the 1-day window. | skip+log malformed articles |
| BUG-17 | med | `size_position` ignored `max_portfolio_exposure_pct`, so near the cap it proposed a full-size order that `evaluate_order` vetoed outright. The bot stopped buying ~10% short of its own limit, with only a `risk_rejections` row to show for it. | sizer clips to remaining exposure room. **Changes live sizing near the cap** (buys smaller instead of nothing). |
| BUG-18 | **high** (the original bug class) | `RiskConfig` accepted anything. `min_cash_reserve_pct: 5` (meaning 5%) → every BUY sizes to 0 shares, silently. `max_position_size_pct: 20` removes the cap; `stop_loss_pct: 8` never fires. | pydantic bounds on risk/execution/app scalars (fractions in (0,1], etc.). `max_open_positions`/`max_trades_*` allow 0 as a kill switch |

### Found, **not** fixed (each has a strict-xfail test)

| id | severity | finding | why not fixed here / suggested direction |
|---|---|---|---|
| **BUG-9** | **critical** | **Restart amnesia.** `run_live_loop` starts with `Portfolio(config.starting_balance)` and never restores cash/positions from the DB or Alpaca. After every restart the bot believes it is flat with the full balance: it re-buys the same signal (broker ends up with 2× the intended position), the closed-market heartbeat writes a "reset" equity point to the DB, and existing positions lose their stop-loss/take-profit coverage. The only durable state is the per-day trade count. (Production logged 336 restarts before PR #1.) | Needs design, touches the live path other sessions may be editing. Direction: at startup replay `fills` (ordered) into a fresh `Portfolio`, then reconcile against `TradingClient.get_all_positions()`/`get_account()` and refuse to trade on mismatch. |
| BUG-5 | med | `get_snapshot` stamps `now` on whatever the latest bar is; if the feed stalls or a symbol is halted for the whole 5-day lookback, a 5-day-old close is traded as current. | Needs a threshold decision (long weekends make 4 days legitimate). |
| BUG-6 | med | No outlier/sanity filter: one bad print (-60% for one poll) trips the 8% stop-loss and liquidates at that price. | Design call (N-of-M confirmation vs % band). |
| BUG-7 | low | `route_order` executes before `Portfolio.apply_fill` checks affordability, so with `min_cash_reserve_pct: 0` slippage can make a fill unaffordable *after* the order reached the mirror. Default 5% reserve absorbs it (tested). | Only reachable with reserve=0. |
| BUG-11 | med | No local-vs-broker clock check. A skewed machine clock stamps fills/decisions wrongly and mis-buckets the daily trade limit (permission to trade still follows the broker clock, tested). | Compare `get_clock().timestamp` to local time each cycle; halt beyond N seconds. |
| BUG-12 | low (latent) | DB stores tz-aware non-UTC timestamps as naive wall clock without converting; `count_trades_today` then mis-buckets them. Everything currently passes UTC. | Normalise in every `save_*`. |
| BUG-13 | med | After a stop-loss exit the strategy's BUY (price now even more oversold) re-enters in the same cycle: realises the loss then buys the knife again, up to 3×/day. | Needs a cooldown policy. |
| BUG-14 | med | Two bot processes (deploy overlap, accidental second container) each keep a private `Portfolio`; together they exceed the position cap. Only the trade count is shared via DB. | Falls out of fixing BUG-9 (reconcile with broker) plus a DB advisory lock. |
| BUG-16 | low | A held symbol whose quote fails this cycle is valued at its **entry** price in the saved snapshot (`prices.get(sym, avg_entry_price)`), so the equity curve jumps (+4% in the test) during data outages. | Carry last-known prices across cycles. |
| BUG-10 | med | (characterisation test, passes) `route_order` mutates the ledger before the DB writes, so a transient DB error leaves the position in memory and at the broker but not in the audit trail. | Outbox/transaction design. |

### By design / observations (no test failure)

* The Alpaca mirror is best-effort, so books drift when a mirror call fails or the broker partially fills (pinned by `test_known_drift_*`). Reconciliation is the real fix.
* `DatabaseManager.save_candles` runs one existence query **per candle** (~180 per symbol per cycle, ~5k queries/min for the default 28 symbols) and never updates an existing row, so today's forming bar is stored once, stale. Not a correctness bug for trading; the fuzz stubs it for speed.
* `ManualExecutor` re-suggests the same BUY every cycle until a human acts.
* `_heartbeat_while_closed` sleeps 1s if the local clock is ahead of Alpaca's `next_open` by more than the wait (rapid snapshot polling while closed).
* `USEquityMarketCalendar` has no holidays (documented already).

### Healthy under chaos

Transient calendar/data outages back off and recover; closed market → no orders
even with a screaming BUY; open/close flapping doesn't double-trade; the daily
trade limit survives restarts; partial fills keep ledger = DB = broker (except
the documented mirror drift); no seed of the random soak produced negative cash
or positions, ledger/DB divergence, or an exception escaping `_run_cycle`.
