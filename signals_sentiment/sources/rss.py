"""RSS / Atom news source.

Network access is injected as a `fetcher(url) -> str` callable, so tests
and the cloud environment use fixtures while a real deployment can pass
e.g. an httpx-based fetcher. This module never opens a socket itself.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..models import DocKind, Document
from .base import DocumentSource

Fetcher = Callable[[str], str]

_TAG_RE = re.compile(r"<[^>]+>")
_ATOM = "{http://www.w3.org/2005/Atom}"


def file_fetcher(mapping: Dict[str, Path]) -> Fetcher:
    """Fetcher serving local fixture files by URL."""

    def fetch(url: str) -> str:
        return Path(mapping[url]).read_text(encoding="utf-8")

    return fetch


def _parse_date(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    text = text.strip()
    try:
        dt = parsedate_to_datetime(text)  # RFC 822 (RSS)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))  # ISO 8601 (Atom)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # RFC 822 "-0000"/naive: assume UTC
    return dt.astimezone(timezone.utc)


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text or "")).strip()


def parse_feed(xml_text: str, symbol: str, source: str) -> List[Document]:
    """Parse RSS 2.0 or Atom into Documents. Items with no parseable date
    are dropped: an undated item cannot be placed in time safely."""
    head = xml_text[:4096].upper()
    if "<!DOCTYPE" in head or "<!ENTITY" in xml_text.upper():
        raise ValueError("refusing feed with DOCTYPE/ENTITY declarations")
    root = ET.fromstring(xml_text)
    docs: List[Document] = []

    entries = root.findall(".//item") + root.findall(f".//{_ATOM}entry")
    for e in entries:
        def get(tag: str) -> Optional[str]:
            el = e.find(tag)
            if el is None:
                el = e.find(_ATOM + tag)
            return el.text if el is not None else None

        title = _clean(get("title"))
        ts = _parse_date(get("pubDate") or get("published") or get("updated"))
        if not title or ts is None:
            continue
        link = get("link")
        if link is None:
            link_el = e.find(_ATOM + "link")
            link = link_el.get("href") if link_el is not None else None
        body = _clean(get("description") or get("summary"))
        docs.append(Document(symbol=symbol.upper(), timestamp=ts, title=title, body=body,
                             source=source, kind=DocKind.NEWS, url=link))
    return docs


class RSSSource(DocumentSource):
    """One feed URL per symbol (e.g. a per-ticker Yahoo/Google News feed)."""

    def __init__(self, feeds: Dict[str, List[str]], fetcher: Fetcher, name: str = "rss"):
        self.name = name
        self._feeds = {k.upper(): list(v) for k, v in feeds.items()}
        self._fetch = fetcher

    def fetch(self, symbol: str, start: datetime, end: datetime) -> List[Document]:
        docs: List[Document] = []
        for url in self._feeds.get(symbol.upper(), []):
            docs.extend(parse_feed(self._fetch(url), symbol, self.name))
        return sorted((d for d in docs if start <= d.timestamp <= end), key=lambda d: d.timestamp)
