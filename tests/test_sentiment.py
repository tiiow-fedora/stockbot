from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analysis.sentiment import LexiconSentimentScorer, build_sentiment_scorer
from core.models import NewsItem

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_item(headline: str, sentiment=None) -> NewsItem:
    return NewsItem(symbol="AAPL", timestamp=NOW, headline=headline, source="test", sentiment=sentiment)


def test_lexicon_scores_clearly_positive_headline():
    scorer = LexiconSentimentScorer()
    score = scorer.score("Apple beats earnings expectations, shares surge on record profit")
    assert score is not None
    assert score > 0


def test_lexicon_scores_clearly_negative_headline():
    scorer = LexiconSentimentScorer()
    score = scorer.score("Apple misses estimates as shares plunge amid fraud investigation")
    assert score is not None
    assert score < 0


def test_lexicon_returns_none_for_no_signal():
    scorer = LexiconSentimentScorer()
    assert scorer.score("Apple to hold quarterly shareholder meeting next Tuesday") is None


def test_lexicon_returns_none_for_empty_headline():
    scorer = LexiconSentimentScorer()
    assert scorer.score("") is None


def test_lexicon_is_deterministic():
    scorer = LexiconSentimentScorer()
    headline = "Company beats and misses in a mixed quarter"
    assert scorer.score(headline) == scorer.score(headline)


def test_lexicon_mixed_signal_lands_between_extremes():
    scorer = LexiconSentimentScorer()
    positive_only = scorer.score("Strong growth and record profit")
    mixed = scorer.score("Strong growth but record loss")
    assert positive_only is not None and mixed is not None
    assert mixed < positive_only


def test_score_news_items_fills_missing_sentiment_only():
    scorer = LexiconSentimentScorer()
    already_scored = make_item("Apple beats earnings", sentiment=0.5)
    unscored_positive = make_item("Apple beats earnings expectations")
    unscored_ambiguous = make_item("Apple to host developer conference")

    result = scorer.score_news_items([already_scored, unscored_positive, unscored_ambiguous])

    assert result[0].sentiment == pytest.approx(0.5)  # provider-native sentiment untouched
    assert result[1].sentiment is not None and result[1].sentiment > 0
    assert result[2].sentiment is None  # no signal - stays None, not forced to neutral


def test_build_sentiment_scorer_none():
    assert build_sentiment_scorer("none") is None


def test_build_sentiment_scorer_lexicon():
    assert isinstance(build_sentiment_scorer("lexicon"), LexiconSentimentScorer)


def test_build_sentiment_scorer_unknown_raises():
    with pytest.raises(ValueError):
        build_sentiment_scorer("not_a_real_scorer")


def test_llm_sentiment_scorer_requires_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from analysis.sentiment import LLMSentimentScorer

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMSentimentScorer()


def test_build_sentiment_scorer_llm_requires_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_sentiment_scorer("llm")
