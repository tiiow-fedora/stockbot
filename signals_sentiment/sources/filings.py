"""SEC-filing-style source (8-K / 10-K / 10-Q excerpts).

Reads a JSON list of filing records (the shape a thin EDGAR adapter would
produce). Fixture-driven; no network.

Look-ahead note: a filing is public at its `accepted` (acceptance)
timestamp, NOT its `period_end`. Records without `accepted` are dropped.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Union

from ..models import DocKind, Document
from .base import DocumentSource


def _ts(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_filings(records: list, name: str = "filings") -> List[Document]:
    docs = []
    for r in records:
        if not r.get("accepted") or not r.get("symbol"):
            continue
        form = r.get("form", "FILING")
        title = f"{form}: {r.get('title', '').strip()}".strip(": ")
        docs.append(Document(
            symbol=r["symbol"].upper(), timestamp=_ts(r["accepted"]), title=title,
            body=r.get("excerpt", ""), source=f"{name}:{form}", kind=DocKind.FILING, url=r.get("url"),
        ))
    return docs


class FilingsSource(DocumentSource):
    def __init__(self, data: Union[Path, str, list, Callable[[], list]], name: str = "filings"):
        self.name = name
        self._data = data

    def _records(self) -> list:
        d = self._data
        if callable(d):
            return d()
        if isinstance(d, (str, Path)):
            return json.loads(Path(d).read_text(encoding="utf-8"))
        return d

    def fetch(self, symbol: str, start: datetime, end: datetime) -> List[Document]:
        sym = symbol.upper()
        docs = [d for d in parse_filings(self._records(), self.name) if d.symbol == sym and start <= d.timestamp <= end]
        return sorted(docs, key=lambda d: d.timestamp)
