# StockBot Research Lab

A strategy research lab built **on top of** the StockBot engine: 16 documented strategies behind the
engine's own `Strategy` interface, a walk-forward parameter optimizer, a risk-metrics module, Monte Carlo
robustness tests, synthetic data generators with *known* structure, and a pytest suite (~1,200 tests).

> **Paper trading / simulation only.** Everything runs through the engine's `PaperExecutor` (simulated fills).
> Nothing here talks to a broker, places orders, or touches real money. No existing engine file was modified
> apart from one line in `pyproject.toml` (`testpaths` now includes `research/tests`).

> **Read the results with care.** All numbers below come from *synthetic* markets whose behaviour is built to
> favour or punish particular ideas (a trending series favours trend followers by construction). They show
> *which strategy fits which regime* and whether the code does what its docs claim - **not** that any strategy
> will make money on real stocks. Treat any real-data conclusion as a hypothesis to test with the walk-forward
> and Monte Carlo tools, on real history, before believing it.

## Quick start

```bash
pip install -r requirements.txt
python -m research.lab list                      # strategies by family
python -m research.lab compare --bars 500        # strategy x scenario matrix (about 20 s)
python -m research.lab walkforward --strategy donchian_breakout --scenario random_walk
python -m research.lab montecarlo  --strategy donchian_breakout --scenario uptrend
python -m pytest research/tests -q               # full lab suite (~75 s)
```

```python
from research import synth
from research.registry import build
from research.harness import run_backtest

candles = synth.regime_switching(600, seed=1)
run = run_backtest(build("supertrend_trend"), candles)       # through the REAL engine Backtester
print(run.report.summary())                                  # sharpe, sortino, maxDD, exposure, trades...
```

Use a research strategy in the engine's normal config (`strategy.name: <name>`) without editing
`strategy/registry.py`:

```python
from strategy.registry import STRATEGY_BUILDERS
from research.registry import engine_builders
STRATEGY_BUILDERS.update(engine_builders())     # partner strategies (pairs/relative strength) are excluded
```

## Layout

| file | purpose |
|---|---|
| `base.py` | `ResearchStrategy` adapter -> engine `Strategy`; `Bars`, `Signal`, validated frozen `StrategyParams` |
| `indicators.py` | numpy indicators (ATR, ADX, Bollinger, Keltner, Donchian, SuperTrend, MACD, RSI, IBS, rolling beta...) with NaN warm-up and a no-look-ahead property test |
| `strategies/` | one module per strategy, auto-registered by `registry.py` (adding a file is all it takes) |
| `synth.py` | deterministic generators: GBM, trend, AR(1) momentum, Ornstein-Uhlenbeck, regime switching, GARCH, jumps, range->breakout, cycles, V-shape, cointegrated pairs |
| `metrics.py` | Sharpe, Sortino, Calmar, max drawdown (+duration, ulcer), VaR/CVaR, exposure, trade stats |
| `harness.py` | `run_backtest` through the real `Backtester`, exposure reconstruction, research risk preset |
| `walkforward.py` | walk-forward optimizer (rolling/anchored), stitched OOS equity, efficiency, parameter stability, overfit flag |
| `montecarlo.py` | trade bootstrap, stationary block bootstrap, permutation test, synthetic price-path robustness, parameter perturbation |
| `lab.py` | CLI and comparison tables |
| `tests/` | contract tests (run on *every* strategy automatically) + per-module tests |

## How strategies plug into the engine

The engine's contract is one method returning a `Decision` (BUY/SELL/HOLD + confidence); the engine is
long-only and the `RiskManager` sizes positions. A research strategy therefore only implements
`signal(bars, in_position) -> ENTER | EXIT | HOLD (+ strength in [0,1])`; `ResearchStrategy` turns that into
a fully populated `Decision` (score breakdown, indicator dict, human-readable reason), and `strength`
becomes `confidence`, which the risk manager uses to scale position size. Design rules, enforced by
`tests/test_contract.py` for every strategy:

* **Stateless** - a pure function of the visible bars and whether you are long, so backtests, walk-forward
  windows and a restarted live process all behave identically.
* **No look-ahead** - tested by corrupting every future bar and asserting decisions are unchanged; history at/after the
  snapshot is dropped defensively. The engine itself fills at the *next bar's open*.
* **Long-only** - ENTER while long and EXIT while flat both become HOLD. (Pairs/relative-strength therefore trade the long leg only.)
* **NaN-safe / validated** - NaN means "no signal"; invalid parameter sets (including non-integer periods) cannot be constructed.

The lab uses `RESEARCH_RISK` (up to 95% of the account in the single symbol, a loose 50% stop) so the
*strategy's* signals, not the risk overlay, drive results; `production_risk()` gives the engine's real defaults
(20% max position, 8% stop-loss, 20% take-profit). Costs: 5 bps slippage per fill, as in the engine default.

## The strategies

| # | name | family | entry | exit | wins when | loses when |
|---|---|---|---|---|---|---|
| 1 | `donchian_breakout` | breakout | close > N-bar high | close < M-bar low | persistent trends, range->breakout cycles | choppy / mean-reverting (whipsaw) |
| 2 | `squeeze_breakout` | breakout | Bollinger-inside-Keltner squeeze releases, price above mid and rising | close below mid by k ATR / momentum flips | compression -> expansion | trendless noise; trades rarely |
| 3 | `atr_expansion_breakout` | volatility | thrust bar: range and gain vs ATR, close near high | Chandelier exit | impulsive moves, V-recoveries | jumpy / mean-reverting (false thrusts) |
| 4 | `tsmom` | momentum | trailing (lookback-skip) return above a hurdle | return < 0 | slow persistent drifts | short-lived momentum, cycles |
| 5 | `high_proximity_momentum` | momentum | within x% of N-bar high, high is recent, momentum > 0 | sags y% below the high | steady uptrends | cycles, reversals |
| 6 | `macd_trend` | trend | bullish MACD cross (zero-line + slope filters) | histogram < 0 | medium trends; few trades | chop |
| 7 | `ema_adx_trend` | trend | fast EMA > slow EMA and ADX strong | EMA cross down or ADX collapses | strong trends | cycles near the EMA lag (-70% on `cyclical`) |
| 8 | `supertrend_trend` | trend | SuperTrend flips up (does not chase stale trends) | flips down | trends with pullbacks | mean reversion; misses trends already under way at start |
| 9 | `ma_ribbon` | trend | fast>mid>slow EMAs, slow rising, fan wide enough | fast < mid | clean trends | cycles shorter than the slow EMA lag |
| 10 | `vol_regime_trend` | volatility | SMA uptrend **and** realised vol in a calm percentile | trend breaks or vol spikes | calm uptrends; lower drawdown than ungated | gives up return in violent rallies |
| 11 | `bollinger_reversion` | mean reversion | close below lower band (panic-cap, optional trend filter) | back to mid/upper band | OU-like ranges | persistent downtrends (falling knife) |
| 12 | `rsi2_reversion` | mean reversion | RSI(2) oversold inside an uptrend | RSI rebound / close > short SMA | dips in uptrends, OU series | breakouts, fast crashes |
| 13 | `ibs_reversion` | mean reversion | close near bar low **and** z-score stretched | IBS recovers or z > 0 | OU-like noise | trends of any direction |
| 14 | `vol_spike_reversion` | volatility | vol spike + sharp drop from recent high | vol normalises / price recovers | V-shaped panics | persistent crashes; trades rarely (~2 per 500 bars) |
| 15 | `pairs_zscore` | pairs | long leg when log-spread vs partner z < -k | z reverts / relationship breaks | cointegrated partners | unrelated partner (no edge) |
| 16 | `relative_strength` | pairs | own return beats the partner by a margin and is positive (dual momentum) | relative or absolute strength fails | leaders in uptrends | mean-reverting leadership |

Each module docstring contains the exact rules, the parameter table and an honest "Expected behaviour" section.
`PARAM_GRID` on every class is the default optimizer grid.

## Comparison across market regimes

500 bars per scenario, seed 0, default parameters, 5 bps slippage, through the real engine
(`python -m research.lab compare --bars 500 --seed 0`). `n/a` Sharpe means a perfectly flat equity curve (never in the market).
The "buy & hold" row uses the same executor and costs. Partner strategies use a fixed random-walk benchmark as the partner,
except in `cointegrated_pair` where the partner is the cointegrated leg.

### Sharpe

| strategy | uptrend | downtrend | momentum | mean_reverting | random_walk | regime_switching | vol_clustering | jumpy | range_breakout | cyclical | v_shape | cointegrated_pair | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `atr_expansion_breakout` | 5.32 | -1.71 | -0.89 | -0.46 | 0.67 | 0.29 | -0.61 | -0.82 | 9.46 | 1.34 | 9.14 | -1.08 | 1.72 |
| `bollinger_reversion` | 1.46 | -5.99 | 0.47 | 0.43 | -0.33 | -0.24 | 0.50 | -0.69 | -4.94 | -2.63 | -5.13 | 0.84 | -1.35 |
| `donchian_breakout` | 4.85 | n/a | -1.19 | -1.04 | 1.00 | 1.34 | -0.07 | -1.21 | 9.75 | 2.67 | 9.28 | -1.20 | 2.20 |
| `ema_adx_trend` | 5.13 | n/a | -1.21 | -1.06 | 0.52 | 0.52 | 0.46 | -1.65 | 6.04 | -7.93 | 8.70 | -0.18 | 0.85 |
| `high_proximity_momentum` | 4.39 | n/a | -0.54 | -0.77 | 1.40 | 0.21 | 0.32 | -1.04 | 3.59 | -3.99 | 7.92 | -1.12 | 0.94 |
| `ibs_reversion` | 0.09 | -5.08 | -0.39 | 0.71 | -1.40 | -0.62 | 0.72 | -0.48 | -5.35 | -6.25 | -7.15 | 0.66 | -2.05 |
| `ma_ribbon` | 4.98 | n/a | -1.62 | -1.90 | 1.14 | 0.35 | -0.16 | -1.22 | 6.74 | -6.85 | 8.56 | -0.85 | 0.83 |
| `macd_trend` | 2.96 | n/a | -0.50 | 0.01 | 0.46 | 0.24 | -0.55 | -0.96 | 6.29 | n/a | 6.51 | -1.44 | 1.30 |
| `pairs_zscore` | 1.49 | -5.40 | -0.10 | 0.12 | -0.53 | 0.47 | -0.59 | -0.38 | -2.00 | 1.78 | -3.63 | 0.40 | -0.70 |
| `relative_strength` | 4.51 | n/a | -1.62 | -0.97 | 1.39 | -0.64 | 0.15 | -0.85 | 5.28 | -1.59 | 8.54 | -0.98 | 1.20 |
| `rsi2_reversion` | 1.06 | n/a | 0.36 | 0.30 | 0.32 | 0.60 | 0.08 | n/a | -3.58 | -8.45 | n/a | -0.58 | -1.10 |
| `squeeze_breakout` | 1.27 | n/a | -0.69 | -0.19 | 1.03 | 1.49 | -0.77 | -0.48 | 6.16 | n/a | n/a | -0.05 | 0.86 |
| `supertrend_trend` | 3.70 | -1.78 | -0.89 | -0.92 | 1.20 | 0.73 | 0.04 | -0.41 | 7.48 | 3.75 | 9.46 | -1.50 | 1.74 |
| `tsmom` | 4.37 | n/a | -1.64 | -1.19 | 0.69 | -0.06 | 0.06 | -0.44 | -0.51 | n/a | 7.89 | -0.79 | 0.84 |
| `vol_regime_trend` | 2.99 | -0.80 | -1.02 | -0.79 | 0.61 | 1.34 | 0.16 | -0.38 | 5.44 | -4.23 | 7.62 | -0.76 | 0.85 |
| `vol_spike_reversion` | n/a | -1.16 | n/a | n/a | n/a | 0.28 | -0.05 | -0.04 | -3.43 | n/a | -2.12 | 0.89 | -0.80 |
| *buy & hold* | 5.50 | -7.12 | -0.63 | 0.28 | 0.60 | 0.17 | 0.50 | -1.27 | 5.14 | 0.13 | 1.91 | 0.02 | 0.44 |


### Total return

| strategy | uptrend | downtrend | momentum | mean_reverting | random_walk | regime_switching | vol_clustering | jumpy | range_breakout | cyclical | v_shape | cointegrated_pair | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `atr_expansion_breakout` | 229.6% | -5.6% | -17.9% | -12.8% | 17.7% | 4.9% | -9.5% | -15.1% | 557.3% | 13.4% | 392.7% | -31.7% | 93.6% |
| `bollinger_reversion` | 5.9% | -67.0% | 5.4% | 6.8% | -6.1% | -6.0% | 4.2% | -12.1% | -45.2% | -31.2% | -47.5% | 21.1% | -14.3% |
| `donchian_breakout` | 188.2% | 0.0% | -18.0% | -18.7% | 24.7% | 27.2% | -1.3% | -12.1% | 588.2% | 41.0% | 407.7% | -30.7% | 99.7% |
| `ema_adx_trend` | 217.5% | 0.0% | -12.7% | -12.1% | 9.8% | 8.3% | 4.3% | -8.2% | 266.6% | -70.3% | 344.8% | -5.2% | 61.9% |
| `high_proximity_momentum` | 149.4% | 0.0% | -4.8% | -12.1% | 39.5% | 2.3% | 4.6% | -14.4% | 73.6% | -24.7% | 270.6% | -19.1% | 38.7% |
| `ibs_reversion` | 0.2% | -57.1% | -6.1% | 10.4% | -19.5% | -11.1% | 5.2% | -7.3% | -48.0% | -53.4% | -65.5% | 15.6% | -19.7% |
| `ma_ribbon` | 199.9% | 0.0% | -20.8% | -31.8% | 29.9% | 6.2% | -2.1% | -17.5% | 254.7% | -58.7% | 327.4% | -20.8% | 55.5% |
| `macd_trend` | 58.6% | 0.0% | -4.8% | -0.4% | 6.4% | 3.0% | -4.7% | -5.9% | 148.0% | 0.0% | 153.7% | -20.2% | 27.8% |
| `pairs_zscore` | 8.4% | -60.3% | -2.8% | 1.3% | -10.6% | 7.5% | -5.1% | -8.2% | -19.5% | 15.0% | -36.1% | 7.1% | -8.6% |
| `relative_strength` | 157.5% | 0.0% | -18.5% | -17.4% | 29.9% | -13.8% | 1.4% | -9.5% | 193.0% | -19.7% | 328.8% | -27.9% | 50.3% |
| `rsi2_reversion` | 6.3% | 0.0% | 1.7% | 1.7% | 3.5% | 6.6% | 0.4% | 0.0% | -26.4% | -63.7% | 0.0% | -8.5% | -6.5% |
| `squeeze_breakout` | 10.4% | 0.0% | -6.8% | -3.5% | 17.4% | 24.3% | -5.4% | -4.5% | 158.1% | 0.0% | 0.0% | -2.4% | 15.6% |
| `supertrend_trend` | 86.5% | -4.6% | -15.5% | -20.6% | 33.1% | 16.4% | 0.1% | -8.3% | 282.0% | 87.4% | 423.2% | -31.7% | 70.7% |
| `tsmom` | 148.2% | 0.0% | -17.0% | -16.3% | 15.6% | -1.9% | 0.3% | -4.2% | -9.7% | 0.0% | 212.5% | -15.2% | 26.0% |
| `vol_regime_trend` | 64.8% | -0.7% | -11.7% | -14.0% | 11.6% | 24.1% | 1.4% | -4.6% | 140.6% | -35.4% | 223.4% | -16.6% | 31.9% |
| `vol_spike_reversion` | 0.0% | -5.2% | 0.0% | 0.0% | 0.0% | 2.7% | -0.3% | -0.2% | -21.9% | 0.0% | -7.8% | 8.0% | -2.1% |
| *buy & hold* | 266.3% | -77.9% | -19.6% | 6.7% | 19.1% | 2.8% | 9.5% | -34.6% | 292.0% | 1.5% | 65.6% | -6.7% | 43.7% |


### Max drawdown

| strategy | uptrend | downtrend | momentum | mean_reverting | random_walk | regime_switching | vol_clustering | jumpy | range_breakout | cyclical | v_shape | cointegrated_pair | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `atr_expansion_breakout` | 5.3% | 5.7% | 22.8% | 18.3% | 14.5% | 12.2% | 17.8% | 15.1% | 3.8% | 5.0% | 1.8% | 38.7% | 13.4% |
| `bollinger_reversion` | 0.2% | 68.1% | 6.8% | 9.0% | 21.4% | 19.0% | 5.9% | 20.9% | 45.7% | 34.2% | 48.0% | 15.0% | 24.5% |
| `donchian_breakout` | 5.4% | 0.0% | 20.6% | 21.8% | 9.7% | 10.6% | 8.9% | 12.1% | 3.2% | 4.2% | 1.8% | 33.2% | 11.0% |
| `ema_adx_trend` | 5.4% | 0.0% | 13.1% | 15.8% | 13.1% | 12.3% | 6.1% | 8.5% | 17.1% | 70.3% | 1.7% | 19.4% | 15.2% |
| `high_proximity_momentum` | 5.4% | 0.0% | 8.6% | 15.4% | 9.4% | 11.2% | 8.4% | 14.4% | 10.9% | 25.1% | 1.7% | 21.6% | 11.0% |
| `ibs_reversion` | 1.1% | 57.1% | 13.5% | 4.4% | 28.3% | 16.7% | 6.2% | 14.7% | 48.1% | 55.0% | 66.0% | 17.3% | 27.4% |
| `ma_ribbon` | 5.4% | 0.0% | 21.0% | 32.3% | 6.9% | 14.2% | 12.3% | 18.8% | 8.8% | 59.2% | 1.7% | 23.6% | 17.0% |
| `macd_trend` | 2.9% | 0.0% | 8.3% | 6.9% | 5.6% | 13.8% | 12.8% | 6.7% | 1.2% | 0.0% | 2.1% | 23.3% | 7.0% |
| `pairs_zscore` | 0.9% | 61.2% | 15.8% | 9.4% | 28.3% | 16.5% | 7.5% | 17.2% | 22.1% | 5.4% | 42.6% | 14.7% | 20.1% |
| `relative_strength` | 5.3% | 0.0% | 20.4% | 21.0% | 6.4% | 17.5% | 7.8% | 10.4% | 16.7% | 19.8% | 1.7% | 36.2% | 13.6% |
| `rsi2_reversion` | 2.9% | 0.0% | 2.1% | 2.8% | 13.9% | 6.2% | 3.6% | 0.0% | 27.6% | 63.7% | 0.0% | 10.7% | 11.1% |
| `squeeze_breakout` | 2.8% | 0.0% | 10.5% | 7.2% | 5.8% | 6.5% | 8.5% | 6.7% | 2.3% | 0.0% | 0.0% | 15.5% | 5.5% |
| `supertrend_trend` | 3.4% | 4.6% | 18.1% | 25.3% | 10.1% | 12.1% | 15.5% | 8.9% | 5.7% | 5.3% | 1.8% | 36.3% | 12.2% |
| `tsmom` | 5.4% | 0.0% | 18.2% | 18.7% | 17.6% | 14.5% | 5.7% | 6.4% | 35.2% | 0.0% | 1.6% | 24.1% | 12.3% |
| `vol_regime_trend` | 3.4% | 0.7% | 11.8% | 17.2% | 10.8% | 7.3% | 9.5% | 11.6% | 5.2% | 36.2% | 1.1% | 16.9% | 11.0% |
| `vol_spike_reversion` | 0.0% | 7.0% | 0.0% | 0.0% | 0.0% | 6.4% | 2.6% | 2.7% | 21.9% | 0.0% | 7.8% | 3.2% | 4.3% |
| *buy & hold* | 5.6% | 77.9% | 28.7% | 13.5% | 33.9% | 22.7% | 10.5% | 36.6% | 33.9% | 15.4% | 71.0% | 30.0% | 31.6% |


(Average exposure and closed-trade count tables: run the command above, or see `research/results/`.)

## What the comparison actually says

* **Scenario design drives the winners.** `range_breakout` and `v_shape` are built around sustained directional bursts, so
  breakout/trend strategies post Sharpe 5-9 there. That validates the *code* (a breakout strategy should win a breakout
  series) - it is not evidence of an edge in real markets. The mean-Sharpe column is dominated by those two rows.
* **Most robust across regimes (by construction of this suite):** `donchian_breakout` (mean Sharpe 2.2, median drawdown 9%),
  `supertrend_trend` (1.7, profitable in 7/12), `atr_expansion_breakout` (1.7, but drawdowns up to 39% on the pair series).
  `macd_trend` and `squeeze_breakout` have the lowest drawdowns of the always-active strategies (median 6%) at the price of
  low exposure and smaller returns.
* **Trend/breakout followers lose in the mean-reverting, jumpy and AR(1)-momentum scenarios** (Sharpe about -0.5 to -1.9):
  whipsaws plus 5 bps of slippage per fill. The `momentum` scenario (phi=0.25) is too weak a signal for slow lookbacks such as `tsmom`.
* **Mean reversion is the weak family here.** It only clearly wins on the Ornstein-Uhlenbeck `mean_reverting` and
  `cointegrated_pair` scenarios (small, often positive), and it takes large losses in `downtrend`
  (`bollinger_reversion` -67%, `ibs_reversion` -57%; under the engine's production risk preset the same runs lose about -16% each) and on
  one-way moves. On the default period-40 `cyclical` sine, *every* reversion strategy and the lagging trend filters lose: the cycle is
  slow relative to the 20-bar bands and the EMAs lag it. They only win on fast cycles (tests pin this both ways).
* **Defensive behaviour works as designed.** In `downtrend` the trend/breakout/momentum strategies are flat (0% return vs -78% buy-and-hold)
  because they never enter; `vol_regime_trend` and `squeeze_breakout` cut max drawdown versus their ungated/always-on counterparts
  (tested on GARCH seeds).
* **Buy-and-hold is hard to beat in the clean `uptrend`** (266% vs 229% for the best strategy): strategies pay slippage and are
  out of the market part of the time. Their benefit shows up in *drawdown*: median max drawdown across scenarios is 6-13% for most trend/breakout strategies versus a buy-and-hold mean of 32% (see the drawdown table).
* **Pairs strategies need a genuinely related partner.** `pairs_zscore` earns on the cointegrated pair (+7% return, Sharpe 0.40; on four other
  `cointegrated_pair(500)` seeds (0-3) it returned +7% to +21% with 4-6 trades each) and does nothing useful against an unrelated benchmark. A *rolling* hedge ratio
  removes most of the edge on short windows, so the default is a fixed beta of 1 (a test pins this).
* **`vol_spike_reversion` is nearly inert** with defaults (about 2 trades per 500 bars on regime-switching data, mean return slightly
  negative). Looser parameters trade more but do not earn more; the docstring says so.

### Impact of the risk overlay

The lab's `RESEARCH_RISK` (95% position, loose stop) isolates strategy signals. The engine's production defaults
(`harness.production_risk()`: 20% max position, 8% stop-loss, 20% take-profit) shrink both gains and losses a lot (over the 12 scenarios, mean max drawdown falls from 11% to 2.5% for `donchian_breakout`, 24.5% to 4.4% for `bollinger_reversion`, while mean absolute return falls from 113% to 13% and 22% to 4%) and
add stop-out/re-entry churn (capped at 3 trades per symbol per day by the RiskManager). Always evaluate a candidate under the risk
settings you will trade with.

## Walk-forward optimisation

`research/walkforward.py` grid-searches `PARAM_GRID` on a training window, then evaluates the winner on the *next, unseen* block;
blocks are stitched into one out-of-sample equity curve.

```bash
python -m research.lab walkforward --strategy donchian_breakout --scenario random_walk
```

```
Walk-forward donchian_breakout: 6 windows (rolling, train=200, test=60, step=60, objective=sharpe)
  OOS (stitched): ret=8.2% cagr=5.8% sharpe=0.55 maxDD=7.3% exposure=39% trades=6 win=50% pf=1.33
  mean objective  IS=1.22  OOS=0.16  efficiency=0.13
  stability: modal params {'entry_period': 10, 'exit_period': 10} in 50% of windows
  overfit_flag=True
```

On a random walk the in-sample Sharpe (1.22) is mostly selection luck and collapses out of sample (0.16): efficiency 0.13 and
the overfit flag fires - exactly what the tool is for. Key properties (all tested): the first decision of each test block is exactly
its first bar and the strategy never sees later bars; corrupting test-slice candles never changes parameter selection; test blocks do not
overlap; ties resolve deterministically; windows with no eligible candidate are skipped (held in cash); an optional
process pool gives results identical to serial. Options: rolling or anchored training, objective (`sharpe|sortino|calmar|total_return|callable`),
`min_trades`, `max_combos` (seeded random subsampling for big grids), `warmup_bars`.

## Risk metrics

`research/metrics.py` (all on equity curves; undefined results are `nan`, never a silent 0): total return, CAGR, annualised vol,
**Sharpe**, **Sortino** (downside semideviation over all bars), Calmar, **max drawdown** (+ duration and ulcer index), historical VaR/CVaR,
**exposure** (time in market, average and peak capital deployed - reconstructed by `harness.py` from the trade log because the stock
`BacktestResult` lacks it), and trade statistics (win rate, profit factor, expectancy ratio, best/worst). Closed-trade counts exclude a
position still open at the end, so a strategy riding one trend reports 0 trades with a positive return.

## Monte Carlo robustness

```bash
python -m research.lab montecarlo --strategy donchian_breakout --scenario uptrend --sims 100
```

* `bootstrap_trades` - resample trade P&Ls: distribution of final equity, drawdown, probability of ruin (needs >= 5 closed trades).
* `block_bootstrap_returns` - Politis-Romano stationary / circular / moving-block bootstrap of per-bar returns (keeps autocorrelation).
* `permutation_test` - circularly shifted position vs market returns: is the *timing* better than chance? One-sided, +1-corrected p-value.
* `price_path_robustness` - bootstrap or GBM-matched synthetic price paths rebuilt from a real series, strategy re-run on each: fraction of paths profitable / beating buy-and-hold.
* `parameter_perturbation` - jitter parameters and report a plateau score (a high score on a *negative* base result is a stable loser, so `positive` is reported alongside).
* Everything is seeded (`SeedSequence.spawn` streams); no global RNG.

Sample output (`donchian_breakout`, `uptrend`): real run +288% Sharpe 5.2; block-bootstrap total return p5/p50/p95 = +172%/+284%/+456%;
profitable on 100% of synthetic paths but beats buy-and-hold on 0%; the timing permutation is *not* significant (p=0.93) - the strategy
is simply long almost all the time in a trending market, which is beta, not timing skill. That nuance is the kind of thing these tests are for.

## Testing and review

`python -m pytest research/tests -q` (about 1,200 tests, ~75 s):

* `test_contract.py` runs ~20 checks on **every** registered strategy automatically (metadata/docs, parameter validation, statelessness,
  no look-ahead, long-only semantics, degenerate data, live-quote snapshots without OHLC, full run through the real `Backtester`).
* Per-strategy files test rules on hand-built bars (boundaries included), parameter validation, and falsifiable claims on synthetic data -
  including honest *negative* results (where the strategy must not win).
* Indicators are cross-checked against the engine's own list-based indicators and for the no-look-ahead property by truncation.
* Generators are tested for the structure they claim (autocorrelation, OU half-life, GARCH clustering, cointegration, fat tails).

Process: strategies were written by five parallel agents against a shared contract, then **cross-reviewed by different agents** who
mutation-tested the assertions. Review findings that were fixed: a pair-generator beta bug in `synth.py`; `tsmom` never exiting after a
flat tape; `atr_expansion_breakout` entering bars its own Chandelier exit would sell next bar (about 10% of entries); NaN/inf and
non-integer parameters passing validation; stale z-score signals on a bad bar and silent duplicate partner timestamps in the pairs code;
a NaN equity curve reporting 0 drawdown; several docstrings that overclaimed.

## Known limitations

* **Synthetic data only.** No real-market validation was done; use the walk-forward and Monte Carlo tools on real history before drawing conclusions.
* Long-only, single symbol per run (engine limitation): pairs strategies trade one leg; no portfolio-level allocation or shorting.
* Stateless design means recursive indicators depend slightly on where the `lookback` window starts, and there are no per-trade trailing stops
  (the engine's stop-loss/take-profit is the only price stop).
* The timing-permutation test weights bars by end-of-bar exposure (fills occur at the next open), which is mildly anti-conservative.
* A position still open at the end of a walk-forward test block is marked to market and carried as cash, without an exit cost.
* Reversion strategies hold losers until their exit rule fires; in the lab's risk preset the only protection is the 50% stop.
* Results depend on the scenario seeds; the committed tables are `research/results/compare_500bars_seed0.*`.
