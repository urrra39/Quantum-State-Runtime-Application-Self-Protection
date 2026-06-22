"""Active-defense policies for Q-RASP-Engine (the decision layer).

This subpackage centralizes *what to do* about an anomaly, separate from the
monitors that detect one and the gateway that transports it. Today it holds the
escalation policy; future active-defense policies (rate limiting, quarantine,
error-correction strategy selection) belong here too.
"""
from __future__ import annotations

from .escalation import (
    DEFAULT_ESCALATING_KINDS,
    NOMINAL,
    PURITY_DROP,
    TRACE_VIOLATION,
    EscalationDecision,
    EscalationPolicy,
)

__all__ = [
    "DEFAULT_ESCALATING_KINDS",
    "NOMINAL",
    "PURITY_DROP",
    "TRACE_VIOLATION",
    "EscalationDecision",
    "EscalationPolicy",
]
