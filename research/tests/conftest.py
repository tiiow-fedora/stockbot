"""Shared fixtures for the research-lab test-suite (the root tests/conftest.py is untouched)."""

from __future__ import annotations

import logging

import pytest

from research import synth

# Keep test output readable: the engine logs every simulated fill at INFO.
logging.getLogger("stockbot").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def scenarios():
    """One 400-bar series per market condition (see research.synth.scenario_suite)."""
    return synth.scenario_suite(400, seed=11)


@pytest.fixture(scope="session")
def uptrend():
    return synth.trending(300, 0.004, 0.006, seed=1)


@pytest.fixture(scope="session")
def downtrend():
    return synth.trending(300, -0.004, 0.006, seed=2)


@pytest.fixture(scope="session")
def ou_series():
    return synth.mean_reverting(500, half_life=6.0, sigma=0.015, seed=3)


@pytest.fixture(scope="session")
def pair():
    return synth.cointegrated_pair(500, seed=4)
