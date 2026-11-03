from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.models import NewsItem
from data.base import NewsDataProvider
from signals_sentiment import (
    CachedScorer, CachedSource, CompositeSource, DocKind, Document, DocumentScorer, FallbackScorer,
    FilingsSource, HeadlineAdapter, LexiconScorer, RSSSource, ScoreCache, ScoredDocument,
    SentimentNewsProvider, SentimentSignal, SignalConfig, StaticSource, StubLLMScorer, aggregate, decay_weight,
)
from signals_sentiment.sources import file_fetcher, parse_feed

FIX = Path(__file__).resolve().parents[2] / "signals_sentiment" / "fixtures"
T0 = datetime(2024, 3, 8, 12, 0, tzinfo=timezone.utc)
WIDE = (datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc))


def doc(title, ts=T0, body="", kind=DocKind.NEWS, symbol="AAPL", source="t"):
    return Document(symbol=symbol, timestamp=ts, title=title, body=body, kind=kind, source=source)


class CountingScorer(DocumentScorer):
    name = "counting"

    def __init__(self, value=0.5):
        self.calls, self.value = 0, value

    def score(self, d):
        self.calls += 1
        return self.value


# ---------------- models ----------------
def test_document_requires_tz():
    with pytest.raises(ValueError):
        Document(symbol="A", timestamp=datetime(2024, 1, 1), title="x")


def test_doc_id_stable_and_distinct():
    assert doc("a").doc_id == doc("a").doc_id
    assert doc("a").doc_id != doc("b").doc_id
    assert doc("a").doc_id != doc("a", ts=T0 + timedelta(seconds=1)).doc_id


# ---------------- lexicon ----------------
def test_lexicon_positive_negative_none():
    s = LexiconScorer()
    assert s.score(doc("Company beats estimates, record profit")) > 0
    assert s.score(doc("Company misses estimates, plunge, fraud")) < 0
    assert s.score(doc("Company holds annual meeting")) is None


def test_lexicon_bounded():
    s = LexiconScorer()
    v = s.score(doc(" ".join(["surge"] * 500)))
    assert 0 < v <= 1.0


def test_lexicon_negation_flips():
    s = LexiconScorer()
    assert s.score(doc("results were not strong")) < 0
    assert s.score(doc("results were strong")) > 0


def test_lexicon_intensifier_amplifies():
    s = LexiconScorer()
    assert abs(s.score(doc("sharply weak demand"))) > abs(s.score(doc("weak demand")))
    assert abs(s.score(doc("slightly weak demand"))) < abs(s.score(doc("weak demand")))


def test_lexicon_title_outweighs_body():
    s = LexiconScorer()
    assert s.score(doc("strong", body="")) > s.score(doc("neutral words", body="strong"))


def test_filing_vocabulary_only_for_filings():
    text = "material weakness and impairment"
    assert LexiconScorer().score(doc(text, kind=DocKind.FILING)) < 0
    assert LexiconScorer().score(doc(text, kind=DocKind.NEWS)) is None


def test_lexicon_deterministic():
    s = LexiconScorer()
    d = doc("mixed: beats but also layoffs")
    assert s.score(d) == s.score(d)


# ---------------- LLM stub ----------------
def test_stub_llm_scripted_and_clamped():
    s = StubLLMScorer(scripted={"great": "0.8", "wild": "7.5", "awful": "-3"})
    assert s.score(doc("great")) == pytest.approx(0.8)
    assert s.score(doc("wild")) == 1.0
    assert s.score(doc("awful")) == -1.0
    assert len(s.prompts) == 3 and "AAPL" in s.prompts[0]


def test_stub_llm_unparseable_and_failure_give_none():
    assert StubLLMScorer(scripted={"x": "no idea"}).score(doc("x")) is None
    assert StubLLMScorer(fail=True).score(doc("x")) is None


def test_llm_parse_reply_edge_cases():
    p = StubLLMScorer.parse_reply
    assert p("Score: -0.25.") == pytest.approx(-0.25)
    assert p("") is None and p(None) is None
    assert p("nan") is None


def test_fallback_scorer_order():
    llm_fail = StubLLMScorer(fail=True)
    fb = FallbackScorer(llm_fail, LexiconScorer())
    assert fb.score(doc("strong profit")) > 0
    assert fb.score(doc("nothing here")) is None
    with pytest.raises(ValueError):
        FallbackScorer()


def test_headline_adapter_matches_existing_interface():
    from analysis.sentiment import SentimentScorer

    adapter = HeadlineAdapter(LexiconScorer())
    assert isinstance(adapter, SentimentScorer)
    assert adapter.score("shares surge on record profit") > 0
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    items = adapter.score_news_items([NewsItem("AAPL", now, "shares plunge on fraud", "t")])
    assert items[0].sentiment < 0


# ---------------- sources ----------------
def test_rss_fixture_parse_and_html_stripped():
    docs = parse_feed((FIX / "aapl_news.xml").read_text(), "aapl", "rss")
    assert len(docs) == 3  # undated item dropped
    assert docs[0].body == "Strong growth in services."
    assert docs[1].timestamp == datetime(2024, 3, 5, 14, 30, tzinfo=timezone.utc)  # -0500 normalised to UTC


def test_atom_parse():
    docs = parse_feed((FIX / "aapl_atom.xml").read_text(), "AAPL", "atom")
    assert len(docs) == 1 and docs[0].url == "https://example.test/aapl/5"


def test_feed_with_entity_declaration_rejected():
    evil = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel/></rss>'
    with pytest.raises(ValueError):
        parse_feed(evil, "AAPL", "rss")


def test_rss_source_window_and_ordering():
    src = RSSSource({"AAPL": ["u1", "u2"]}, file_fetcher({"u1": FIX / "aapl_news.xml", "u2": FIX / "aapl_atom.xml"}))
    docs = src.fetch("aapl", datetime(2024, 3, 5, tzinfo=timezone.utc), datetime(2024, 3, 6, 23, tzinfo=timezone.utc))
    assert [d.url for d in docs] == ["https://example.test/aapl/2", "https://example.test/aapl/3"]
    assert src.fetch("MSFT", *WIDE) == []


def test_filings_source_filters_symbol_and_requires_acceptance():
    src = FilingsSource(FIX / "filings.json")
    docs = src.fetch("AAPL", *WIDE)
    assert len(docs) == 2 and all(d.kind == DocKind.FILING for d in docs)
    assert docs[0].source == "filings:8-K"
    assert [d.symbol for d in FilingsSource(FIX / "filings.json").fetch("MSFT", *WIDE)] == ["MSFT"]


def test_filings_source_from_list_and_callable():
    recs = json.loads((FIX / "filings.json").read_text())
    assert len(FilingsSource(recs).fetch("AAPL", *WIDE)) == 2
    assert len(FilingsSource(lambda: recs).fetch("AAPL", *WIDE)) == 2


def test_static_source_window_inclusive():
    d = doc("x")
    s = StaticSource([d])
    assert s.fetch("aapl", T0, T0) == [d]
    assert s.fetch("AAPL", T0 + timedelta(seconds=1), T0 + timedelta(days=1)) == []


def test_composite_dedupes_and_isolates_failures():
    class Boom(StaticSource):
        def fetch(self, *a):
            raise RuntimeError("down")

    d = doc("x")
    comp = CompositeSource([Boom([]), StaticSource([d]), StaticSource([d])])
    assert comp.fetch("AAPL", *WIDE) == [d]


# ---------------- cache ----------------
def test_cached_scorer_scores_once_including_none():
    inner = CountingScorer()
    cache = ScoreCache()
    s = CachedScorer(inner, cache)
    d = doc("x")
    s.score(d), s.score(d)
    assert inner.calls == 1 and cache.hits == 1 and cache.misses == 1
    inner.value = None
    n = CachedScorer(inner, ScoreCache())
    n.score(d), n.score(d)
    assert inner.calls == 2  # one more call total for the second cache, then hit


def test_cache_keyed_by_scorer_name():
    cache = ScoreCache()
    a, b = CountingScorer(0.1), CountingScorer(0.9)
    b.name = "other"
    d = doc("x")
    assert CachedScorer(a, cache).score(d) == 0.1
    assert CachedScorer(b, cache).score(d) == 0.9


def test_cache_persistence_roundtrip_and_corruption(tmp_path):
    p = tmp_path / "sub" / "cache.json"
    c1 = ScoreCache(p)
    inner = CountingScorer(0.3)
    CachedScorer(inner, c1).score(doc("x"))
    c1.save()
    c2 = ScoreCache(p)
    CachedScorer(inner, c2).score(doc("x"))
    assert inner.calls == 1 and len(c2) == 1
    p.write_text("{not json")
    assert len(ScoreCache(p)) == 0
    assert not list(p.parent.glob("*.tmp"))


def test_cached_source_ttl():
    now = [0.0]
    calls = []

    class Src(StaticSource):
        def fetch(self, *a):
            calls.append(1)
            return super().fetch(*a)

    cs = CachedSource(Src([doc("x")]), ttl_seconds=60, clock=lambda: now[0])
    cs.fetch("AAPL", *WIDE), cs.fetch("AAPL", *WIDE)
    assert len(calls) == 1
    now[0] = 61
    cs.fetch("AAPL", *WIDE)
    assert len(calls) == 2


# ---------------- decay ----------------
def test_decay_weight_half_life():
    assert decay_weight(0, 3600) == 1.0
    assert decay_weight(3600, 3600) == pytest.approx(0.5)
    assert decay_weight(7200, 3600) == pytest.approx(0.25)
    assert decay_weight(-1, 3600) == 0.0
    with pytest.raises(ValueError):
        decay_weight(1, 0)


def sd(score, age_h, kind=DocKind.NEWS):
    return ScoredDocument(doc("x", ts=T0 - timedelta(hours=age_h), kind=kind), score, "t")


def test_aggregate_none_when_empty_or_all_none():
    assert aggregate([], T0) is None
    assert aggregate([sd(None, 1)], T0) is None


def test_aggregate_fresh_beats_stale():
    fresh_pos_old_neg = aggregate([sd(0.8, 1), sd(-0.8, 72)], T0)
    assert fresh_pos_old_neg > 0
    assert aggregate([sd(0.8, 72), sd(-0.8, 1)], T0) < 0


def test_aggregate_signal_fades_over_time():
    docs = [sd(0.9, 0)]
    vals = [aggregate(docs, T0 + timedelta(hours=h)) for h in (0, 24, 72)]
    assert vals[0] > vals[1] > vals[2] > 0


def test_aggregate_ignores_future_docs():
    assert aggregate([sd(1.0, -5)], T0) is None


def test_aggregate_old_doc_dropped_below_min_weight():
    assert aggregate([sd(1.0, 24 * 365)], T0) is None


def test_aggregate_more_evidence_more_confidence():
    one = aggregate([sd(0.5, 0)], T0)
    many = aggregate([sd(0.5, 0)] * 5, T0)
    assert 0 < one < many < 0.5 + 1e-9


def test_aggregate_filing_decays_slower_and_weighs_more():
    news = aggregate([sd(0.5, 96)], T0)
    filing = aggregate([sd(0.5, 96, DocKind.FILING)], T0)
    assert filing > news


def test_aggregate_always_bounded():
    v = aggregate([sd(1.0, 0, DocKind.FILING)] * 100, T0)
    assert -1.0 <= v <= 1.0 and math.isfinite(v)


# ---------------- signal end to end ----------------
def build_signal(**kw):
    src = CompositeSource([
        RSSSource({"AAPL": ["u1", "u2"]}, file_fetcher({"u1": FIX / "aapl_news.xml", "u2": FIX / "aapl_atom.xml"})),
        FilingsSource(FIX / "filings.json"),
    ])
    return SentimentSignal(src, **kw)


def test_signal_no_lookahead():
    sig = build_signal()
    before = datetime(2024, 3, 4, 13, 59, tzinfo=timezone.utc)
    assert sig.value("AAPL", before) is None
    docs = sig.score_documents("AAPL", datetime(2024, 3, 4, 14, 0, tzinfo=timezone.utc))
    assert len(docs) == 1 and docs[0].document.timestamp <= datetime(2024, 3, 4, 14, 0, tzinfo=timezone.utc)


def test_signal_evolves_with_news_flow():
    sig = build_signal()
    after_beat = sig.value("AAPL", datetime(2024, 3, 4, 22, 0, tzinfo=timezone.utc))
    after_lawsuit = sig.value("AAPL", datetime(2024, 3, 5, 15, 0, tzinfo=timezone.utc))
    after_bad_10q = sig.value("AAPL", datetime(2024, 3, 6, 22, 0, tzinfo=timezone.utc))
    assert after_beat > 0.3
    assert after_lawsuit < after_beat
    assert after_bad_10q < after_lawsuit


def test_signal_unknown_symbol_is_none():
    assert build_signal().value("ZZZZ", T0) is None


def test_signal_uses_cache_across_calls():
    inner = CountingScorer(0.4)
    sig = build_signal(scorer=inner)
    sig.value("AAPL", T0)
    first = inner.calls
    sig.value("AAPL", T0 + timedelta(hours=1))
    assert inner.calls == first  # nothing re-scored


def test_signal_llm_stub_pluggable():
    llm = StubLLMScorer(scripted={"AAPL": "0.9"})
    sig = build_signal(scorer=llm)
    assert sig.value("AAPL", T0) > 0
    assert llm.prompts


def test_signal_max_docs_cap():
    docs = [doc(f"strong {i}", ts=T0 - timedelta(minutes=i)) for i in range(50)]
    sig = SentimentSignal(StaticSource(docs), config=SignalConfig(max_docs=10))
    assert len(sig.score_documents("AAPL", T0)) == 10


def test_news_provider_is_drop_in():
    prov = SentimentNewsProvider(build_signal())
    assert isinstance(prov, NewsDataProvider)
    items = prov.get_news("AAPL", datetime(2024, 3, 5, tzinfo=timezone.utc), datetime(2024, 3, 7, tzinfo=timezone.utc))
    assert items and all(isinstance(i, NewsItem) for i in items)
    assert all(datetime(2024, 3, 5, tzinfo=timezone.utc) <= i.timestamp <= datetime(2024, 3, 7, tzinfo=timezone.utc) for i in items)
    assert any(i.sentiment is None for i in items)  # the neutral "developer conference" item
    assert all(i.sentiment is None or -1 <= i.sentiment <= 1 for i in items)


def test_package_has_no_network_or_trading_imports():
    import ast

    import signals_sentiment as pkg

    banned = {"alpaca", "httpx", "requests", "anthropic", "urllib", "socket", "execution", "portfolio"}
    for f in Path(pkg.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                assert m.split(".")[0] not in banned, f"{f.name} imports {m}"
