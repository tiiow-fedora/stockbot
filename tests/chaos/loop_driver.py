"""Drive main.run_live_loop for a fixed number of iterations against the
fake broker, advancing the shared fake clock instead of really sleeping."""

from __future__ import annotations

from typing import Callable, List, Optional

import main
from tests.chaos.fake_alpaca import FakeAlpacaBroker
from tests.chaos.bot import patch_clock


def run_loop(monkeypatch, broker: FakeAlpacaBroker, config, iterations: int,
             on_sleep: Optional[Callable[[int, float], None]] = None) -> dict:
    """Runs the real live loop. Each time the loop sleeps, broker time advances
    by the slept amount; after `iterations` sleeps a KeyboardInterrupt ends it
    (the loop's documented clean-exit path). Returns captured state."""
    patch_clock(monkeypatch, broker.clock)
    sleeps: List[float] = []
    portfolios = []
    real_portfolio = main.Portfolio

    def recording_portfolio(*a, **k):
        p = real_portfolio(*a, **k)
        portfolios.append(p)
        return p

    monkeypatch.setattr(main, "Portfolio", recording_portfolio)

    def fake_sleep(seconds):
        sleeps.append(seconds)
        broker.clock.advance(seconds=seconds)
        if on_sleep:
            on_sleep(len(sleeps), seconds)
        if len(sleeps) >= iterations:
            raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", fake_sleep)
    main.run_live_loop(config)
    return {"sleeps": sleeps, "portfolio": portfolios[-1] if portfolios else None}
