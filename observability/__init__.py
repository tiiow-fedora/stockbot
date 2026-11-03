"""Operational tooling: metrics, JSON logs, health checks, graceful shutdown
and config validation. Everything here is additive and opt-in; the trading
pipeline only touches it through a handful of one-line hooks."""
