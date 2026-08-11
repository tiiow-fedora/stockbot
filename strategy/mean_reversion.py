"""
MeanReversionStrategy: a contrarian counterpart to TechnicalStrategy.

Where TechnicalStrategy trades *with* an SMA crossover and momentum
(trend-following), this strategy trades *against* short-term price
extremes (mean-reversion): it gets more bullish the further price has
fallen below its long SMA and fades recent momentum instead of
following it. RSI's already-contrarian overbought/oversold read carries
over unchanged from TechnicalStrategy.

It exists to prove the `Strategy` interface is genuinely swappable
(Milestone 5) and to give `backtesting/compare.py` something meaningfully
different to compare against on the same data - neither strategy is
assumed profitable; that's what backtesting answers, not the code.
"""

from __future__ import annotations

from typing import List, Optional

from analysis import indicators
from core.config import StrategyConfig
from core.models import Candle, Decision, MarketSnapshot, NewsItem
from strategy.base import Strategy
from strategy.scoring import ScoringEngine, clip


class MeanReversionStrategy(Strategy):
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

        ind = self._compute_indicators(closes, volumes)

        if len(closes) < cfg.min_history_bars or ind["sma_long"] is None:
            return self._hold_decision(
                symbol,
                snapshot,
                ind,
                current_position_qty,
                cash,
                portfolio_value,
                reason=f"insufficient history: have {len(closes)} bars, need >= {cfg.min_history_bars}",
            )

        technical_score = self._contrarian_technical_score(ind)
        # Fade momentum instead of following it - the opposite sign from
        # TechnicalStrategy's momentum component.
        momentum_score = -clip((ind["momentum"] or 0.0) * 5.0)
        volume_score = clip((ind["volume_change"] or 0.0) * 2.0)
        risk_score = -clip((ind["volatility"] or 0.0) * 8.0, 0.0, 1.0)
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
        reason = self._build_reason(action, breakdown, ind, news)

        return Decision(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            price=snapshot.price,
            action=action,
            confidence=confidence,
            scores=breakdown,
            indicators=ind,
            current_position_qty=current_position_qty,
            cash=cash,
            portfolio_value=portfolio_value,
            reason=reason,
        )

    # -- internals ------------------------------------------------------

    def _compute_indicators(self, closes: List[float], volumes: List[float]) -> dict:
        cfg = self.config
        return {
            "price": closes[-1],
            "sma_short": indicators.sma(closes, cfg.sma_short_period)[-1],
            "sma_long": indicators.sma(closes, cfg.sma_long_period)[-1],
            "rsi": indicators.rsi(closes, cfg.rsi_period)[-1],
            "momentum": indicators.momentum(closes, cfg.momentum_period)[-1],
            "volume_change": indicators.volume_change(volumes, cfg.volume_period)[-1],
            "volatility": indicators.volatility(closes, cfg.volatility_period)[-1],
        }

    def _contrarian_technical_score(self, ind: dict) -> float:
        cfg = self.config
        components = []

        # The further price has strayed below its long SMA, the more
        # bullish (expecting reversion back up) - and vice versa. This is
        # the mirror image of TechnicalStrategy's trend-following spread.
        if ind["sma_long"]:
            deviation = (ind["price"] - ind["sma_long"]) / ind["sma_long"]
            components.append(-clip(deviation * 10.0))

        # RSI is already a contrarian/reversal signal - unchanged from
        # TechnicalStrategy: oversold is bullish, overbought is bearish.
        if ind["rsi"] is not None:
            if ind["rsi"] >= cfg.rsi_overbought:
                components.append(-clip((ind["rsi"] - cfg.rsi_overbought) / (100 - cfg.rsi_overbought)))
            elif ind["rsi"] <= cfg.rsi_oversold:
                components.append(clip((cfg.rsi_oversold - ind["rsi"]) / cfg.rsi_oversold))
            else:
                components.append(0.0)

        if not components:
            return 0.0
        return clip(sum(components) / len(components))

    def _sentiment_score(self, news: List[NewsItem]) -> Optional[float]:
        scored = [n.sentiment for n in news if n.sentiment is not None]
        if not scored:
            return None
        return clip(sum(scored) / len(scored))

    def _hold_decision(self, symbol, snapshot, ind, current_position_qty, cash, portfolio_value, reason) -> Decision:
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
            indicators=ind,
            current_position_qty=current_position_qty,
            cash=cash,
            portfolio_value=portfolio_value,
            reason=reason,
        )

    def _build_reason(self, action, breakdown, ind: dict, news: List[NewsItem]) -> str:
        parts = [
            f"composite={breakdown.composite:.3f}",
            f"technical(contrarian)={breakdown.technical:.3f}",
            f"momentum(faded)={breakdown.momentum:.3f}",
            f"volume={breakdown.volume:.3f}",
            f"risk={breakdown.risk:.3f}",
        ]
        if breakdown.sentiment is not None:
            parts.append(f"sentiment={breakdown.sentiment:.3f} ({len(news)} news items)")
        else:
            parts.append("sentiment=n/a (no news available)")
        parts.append(
            f"price_vs_sma{self.config.sma_long_period}="
            f"{_fmt(_pct_dev(ind['price'], ind['sma_long']))} rsi={_fmt(ind['rsi'])}"
        )
        return f"{action.value}: " + ", ".join(parts)


def _pct_dev(price: Optional[float], sma: Optional[float]) -> Optional[float]:
    if price is None or not sma:
        return None
    return (price - sma) / sma


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"
