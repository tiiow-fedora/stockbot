"""
Registry of research strategies, discovered automatically.

Every module in ``research/strategies/`` that defines a concrete
``ResearchStrategy`` subclass with a non-empty ``name`` is registered under
that name - adding a strategy is just adding a file, with nothing to edit here
(so parallel contributors never conflict).

Usage::

    from research.registry import STRATEGIES, build
    strat = build("donchian_breakout", entry_period=30)
    STRATEGIES["rsi2_reversion"].PARAM_GRID

Hooking into the engine's own registry (``strategy/registry.py``) without
editing it::

    from strategy.registry import STRATEGY_BUILDERS
    from research.registry import engine_builders
    STRATEGY_BUILDERS.update(engine_builders())   # config.yaml `strategy.name: <research name>` now works

Partner strategies (``requires_partner``) are skipped by ``engine_builders``
because the engine's builder signature (``StrategyConfig -> Strategy``) has no
way to supply the second instrument.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any, Callable, Dict, Type

from core.config import StrategyConfig
from research.base import ResearchStrategy
from strategy.base import Strategy


def discover() -> Dict[str, Type[ResearchStrategy]]:
    import research.strategies as pkg

    found: Dict[str, Type[ResearchStrategy]] = {}
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{mod_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, ResearchStrategy)
                and cls is not ResearchStrategy
                and cls.__module__ == module.__name__
                and not inspect.isabstract(cls)
                and cls.name
            ):
                if cls.name in found and found[cls.name] is not cls:
                    raise RuntimeError(f"duplicate research strategy name {cls.name!r}")
                found[cls.name] = cls
    return dict(sorted(found.items()))


STRATEGIES: Dict[str, Type[ResearchStrategy]] = discover()

FAMILIES = ("momentum", "trend", "mean_reversion", "breakout", "pairs", "volatility")


def build(name: str, **params: Any) -> ResearchStrategy:
    """Instantiate a strategy by name; ``params`` override its defaults.
    A ``partner=`` kwarg is forwarded for partner strategies."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown research strategy {name!r}. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](**params)


def by_family() -> Dict[str, list]:
    out: Dict[str, list] = {}
    for cls in STRATEGIES.values():
        out.setdefault(cls.family, []).append(cls.name)
    return out


def engine_builders() -> Dict[str, Callable[[StrategyConfig], Strategy]]:
    """``{name: builder(StrategyConfig) -> Strategy}`` compatible with ``strategy.registry.STRATEGY_BUILDERS``.
    Research strategies use their own defaults; ``StrategyConfig`` is accepted only for signature
    compatibility."""
    return {
        name: (lambda _cfg, _cls=cls: _cls())
        for name, cls in STRATEGIES.items()
        if not cls.requires_partner
    }
