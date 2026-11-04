"""
Shared partner-instrument lookup for the ``requires_partner`` strategies.

Private module (leading underscore): it defines no ``ResearchStrategy`` subclass, so the
registry ignores it. The single rule it enforces is **no look-ahead**: partner closes
are fetched by *exact timestamp* for the timestamps already present in the decision
bars, never by position or by "latest available", so a partner bar later than the
decision bar can never be read.

Gap policy: if the partner has no bar at a timestamp, the last known partner close is
carried forward for at most ``MAX_FFILL`` consecutive missing bars; beyond that the
aligned value is NaN, which strategies treat as "no signal" (HOLD).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Sequence

import numpy as np

from core.models import Candle

MAX_FFILL = 3


def partner_close_map(partner: Sequence[Candle]) -> Dict[datetime, float]:
    """``{timestamp: close}`` copied from the partner series (later mutation of the list is
    ignored); order-independent. Rejects an empty partner and conflicting duplicate timestamps."""
    if partner is None or len(partner) == 0:
        raise ValueError("partner candles must be a non-empty list")
    out: Dict[datetime, float] = {}
    for c in partner:
        k, v = c.timestamp, float(c.close)
        if out.setdefault(k, v) != v:
            raise ValueError(f"duplicate partner timestamp with conflicting closes: {c.timestamp}")
    return out


def align_partner(closes: Dict[datetime, float], timestamps: Sequence, max_ffill: int = MAX_FFILL) -> np.ndarray:
    """Partner close for every timestamp in ``timestamps`` (NaN where unavailable)."""
    out = np.full(len(timestamps), np.nan)
    last, missing = np.nan, 0
    for i, ts in enumerate(timestamps):
        v = closes.get(ts)
        if v is not None and np.isfinite(v) and v > 0:
            last, missing = v, 0
            out[i] = v
        else:
            missing += 1
            if missing <= max_ffill:
                out[i] = last
    return out


def usable_mask(*arrays: np.ndarray) -> np.ndarray:
    """Rows where every array is finite and positive (so logs are safe)."""
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a) & (a > 0)
    return mask
