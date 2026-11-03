"""
TechnicalStrategy: the default, deterministic, quantitative strategy.

It is a straightforward rules-based scorer, not a black box - every
number that goes into the decision is recorded on the returned Decision
so it can be reproduced and audited later. It does NOT ask an LLM
anything; if you later add an LLM-based sentiment module, it plugs in as
just another optional component score (see `sentiment` below), never as
the decision-maker itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from analysis import indicators
from core.config import StrategyConfig
from core.models import Candle, Decision, MarketSnapshot, NewsItem
from strategy.base import Strategy
from strategy.scoring import ScoringEngine, clip


class TechnicalStrategy(Strategy):
    def __init__(self, config: StrategyConfig, scoring_engine: ScoringEngine):
        self.config = config
        self.scoring_engine = scoring_engine

    def generate_decision(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        history: List[Candle],
        news: List[NewsItem],
        current_position_qty: float,
        cash: float,
        portfolio_value: float,
    ) -> Decision:
        cfg = self.config
        closes = [c.close for c in history] + [snapshot.price]
        volumes = [c.volume for c in history] + [snapshot.volume]

        indicator_values = self._compute_indicators(closes, volumes)
        if snapshot.volume <= 0:
            # Volume 0 means "this source gave no volume", not "volume
            # collapsed to nothing" - don't let it read as a -100% change.
            indicator_values["volume_change"] = None

        if len(closes) < cfg.min_history_bars or indicator_values["sma_long"] is None:
            return self._hold_decision(
                symbol,
                snapshot,
                indicator_values,
                current_position_qty,
                cash,
                portfolio_value,
                reason=(
                    f"insufficient history: have {len(closes)} bars, "
                    f"need >= {cfg.min_history_bars}"
                ),
            )

        technical_score = self._technical_score(indicator_values)
        momentum_score = clip((indicator_values["momentum"] or 0.0) * 5.0)
        volume_score = clip((indicator_values["volume_change"] or 0.0) * 2.0)
        # Volatility is a headwind, not a tailwind: scale it into [0, 1] and
        # apply it as a *negative* contribution regardless of price direction.
        risk_score = -clip((indicator_values["volatility"] or 0.0) * 8.0, 0.0, 1.0)

        sentiment_score = self._sentiment_score(news)

        breakdown = self.scoring_engine.score(
            technical=technical_score,
            momentum=momentum_score,
            volume=volume_score,
            risk=risk_score,
            sentiment=sentiment_score,
        )
        action = self.scoring_engine.decide(breakdown.composite)
        confidence = round(abs(breakdown.composite), 4)

        reason = self._build_reason(action, breakdown, indicator_values, news)

        return Decision(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            price=snapshot.price,
            action=action,
            confidence=confidence,
            scores=breakdown,
            indicators=indicator_values,
            current_position_qty=current_position_qty,
            cash=cash,
            portfolio_value=portfolio_value,
            reason=reason,
        )

    # -- internals ------------------------------------------------------

    def _compute_indicators(self, closes: List[float], volumes: List[float]) -> dict:
        cfg = self.config
        sma_short = indicators.sma(closes, cfg.sma_short_period)[-1]
        sma_long = indicators.sma(closes, cfg.sma_long_period)[-1]
        ema_val = indicators.ema(closes, cfg.ema_period)[-1]
        rsi_val = indicators.rsi(closes, cfg.rsi_period)[-1]
        macd_line, signal_line, hist = indicators.macd(
            closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        momentum_val = indicators.momentum(closes, cfg.momentum_period)[-1]
        volume_change_val = indicators.volume_change(volumes, cfg.volume_period)[-1]
        volatility_val = indicators.volatility(closes, cfg.volatility_period)[-1]

        return {
            "price": closes[-1],
            "sma_short": sma_short,
            "sma_long": sma_long,
            "ema": ema_val,
            "rsi": rsi_val,
            "macd": macd_line[-1],
            "macd_signal": signal_line[-1],
            "macd_histogram": hist[-1],
            "momentum": momentum_val,
            "volume_change": volume_change_val,
            "volatility": volatility_val,
        }

    def _technical_score(self, ind: dict) -> float:
        cfg = self.config
        components = []

        # SMA crossover: short above long is bullish.
        if ind["sma_short"] is not None and ind["sma_long"] is not None and ind["sma_long"] != 0:
            spread = (ind["sma_short"] - ind["sma_long"]) / ind["sma_long"]
            components.append(clip(spread * 10.0))

        # RSI: overbought is bearish, oversold is bullish, neutral zone contributes ~0.
        if ind["rsi"] is not None:
            if ind["rsi"] >= cfg.rsi_overbought:
                components.append(-clip((ind["rsi"] - cfg.rsi_overbought) / (100 - cfg.rsi_overbought)))
            elif ind["rsi"] <= cfg.rsi_oversold:
                components.append(clip((cfg.rsi_oversold - ind["rsi"]) / cfg.rsi_oversold))
            else:
                components.append(0.0)

        # MACD histogram sign/magnitude relative to price.
        if ind["macd_histogram"] is not None and ind["price"]:
            components.append(clip((ind["macd_histogram"] / ind["price"]) * 20.0))

        if not components:
            return 0.0
        return clip(sum(components) / len(components))

    def _sentiment_score(self, news: List[NewsItem]) -> Optional[float]:
        scored = [n.sentiment for n in news if n.sentiment is not None]
        if not scored:
            return None
        return clip(sum(scored) / len(scored))

    def _hold_decision(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        indicator_values: dict,
        current_position_qty: float,
        cash: float,
        portfolio_value: float,
        reason: str,
    ) -> Decision:
        from core.models import Action, ScoreBreakdown

        breakdown = ScoreBreakdown(
            technical=0.0, momentum=0.0, volume=0.0, risk=0.0, sentiment=None, composite=0.0
        )
        return Decision(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            price=snapshot.price,
            action=Action.HOLD,
            confidence=0.0,
            scores=breakdown,
            indicators=indicator_values,
            current_position_qty=current_position_qty,
            cash=cash,
            portfolio_value=portfolio_value,
            reason=reason,
        )

    def _build_reason(self, action, breakdown, ind: dict, news: List[NewsItem]) -> str:
        parts = [
            f"composite={breakdown.composite:.3f}",
            f"technical={breakdown.technical:.3f}",
            f"momentum={breakdown.momentum:.3f}",
            f"volume={breakdown.volume:.3f}",
            f"risk={breakdown.risk:.3f}",
        ]
        if breakdown.sentiment is not None:
            parts.append(f"sentiment={breakdown.sentiment:.3f} ({len(news)} news items)")
        else:
            parts.append("sentiment=n/a (no news available)")
        parts.append(
            f"sma{self.config.sma_short_period}={_fmt(ind['sma_short'])} "
            f"sma{self.config.sma_long_period}={_fmt(ind['sma_long'])} "
            f"rsi={_fmt(ind['rsi'])} macd_hist={_fmt(ind['macd_histogram'])}"
        )
        return f"{action.value}: " + ", ".join(parts)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"
