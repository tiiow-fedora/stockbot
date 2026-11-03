# signals_sentiment

News/filings sentiment signal. Paper-trading research only; fully offline (fixtures / injected fetchers).

```
DocumentSource (RSS/Atom, filings JSON, static, composite)
  -> DocumentScorer (LexiconScorer default | LLMScorer interface, StubLLMScorer)
  -> ScoreCache (score each doc once; optional JSON persistence)
  -> decay.aggregate (half-life decay, filings slower/heavier, no lookahead)
  -> SentimentSignal.value(symbol, as_of) in [-1, 1] or None
```

Plugs into the existing pipeline via `SentimentNewsProvider` (a `NewsDataProvider`,
items carry pre-filled `.sentiment`) and `HeadlineAdapter` (a `SentimentScorer`).
`None` means "no opinion", matching `strategy/scoring.py`'s weight redistribution.

A real LLM backend subclasses `LLMScorer` and implements `_complete(prompt) -> str`;
wrap with `FallbackScorer(llm, LexiconScorer())`. A real RSS deployment passes a
`fetcher(url) -> str` to `RSSSource`. Neither is wired by default.
