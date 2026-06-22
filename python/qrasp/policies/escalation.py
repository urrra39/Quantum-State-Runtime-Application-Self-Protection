"""Escalation policy: map an anomaly classification to a defensive decision.

This module is the **decision layer** of the engine. The core observer (Rust)
*classifies* a quantum state; the monitors *produce* those classifications; the
gateway *transports* them. None of those layers should privately decide what
counts as an attack. That judgement lives here, in one declarative place, so
that a circuit analyzed locally by the simulator adapter and one analyzed
through the FastAPI gateway apply byte-for-byte identical rules and can never
disagree about whether a state warrants active defense.

The canonical anomaly-kind labels mirror ``qrasp_core::observer::Anomaly`` and
the gateway's ``AnomalyKind`` enum. They are plain strings here so the policy
has no dependency on either Rust or Pydantic and is trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet

#: State is within expected bounds.
NOMINAL: str = "nominal"
#: Purity ``Tr(rho^2)`` fell by more than the configured threshold between steps
#: (decoherence / side-channel information leakage).
PURITY_DROP: str = "purity_drop"
#: Trace ``Tr(rho)`` deviated from 1.0 (non-trace-preserving op: tamper/clone).
TRACE_VIOLATION: str = "trace_violation"

#: Anomaly kinds that, by default, warrant escalation to active defense.
DEFAULT_ESCALATING_KINDS: FrozenSet[str] = frozenset({PURITY_DROP, TRACE_VIOLATION})

#: Human-readable rationale strings (kept stable: the gateway returns these to
#: clients and the bridge logs them).
_ESCALATE_REASON: str = "active-defense trigger recommended"
_NOMINAL_REASON: str = "state nominal"


@dataclass(frozen=True)
class EscalationDecision:
    """The outcome of evaluating one anomaly classification against the policy."""

    #: Whether the anomaly warrants an active-defense response (rollback /
    #: error-correction injection).
    escalate: bool
    #: The anomaly kind that was evaluated.
    kind: str
    #: Stable, human-readable rationale for the decision.
    reason: str


@dataclass(frozen=True)
class EscalationPolicy:
    """Decides whether an anomaly classification triggers active defense.

    The policy is intentionally simple and declarative: a configurable set of
    anomaly kinds escalates, and everything else is treated as nominal. It is
    immutable (``frozen``) so a single shared instance can be used concurrently
    by the gateway and the bridge without risk of mutation.

    Example:
        >>> policy = EscalationPolicy()
        >>> policy.should_escalate("purity_drop")
        True
        >>> policy.should_escalate("nominal")
        False
        >>> policy.decide("trace_violation").reason
        'active-defense trigger recommended'
    """

    #: The set of anomaly kinds that escalate. Defaults to purity drops and
    #: trace violations; nominal never escalates.
    escalating_kinds: FrozenSet[str] = field(default=DEFAULT_ESCALATING_KINDS)

    def should_escalate(self, kind: str) -> bool:
        """Return ``True`` if ``kind`` warrants active defense."""
        return kind in self.escalating_kinds

    def decide(self, kind: str) -> EscalationDecision:
        """Evaluate ``kind`` and return a structured :class:`EscalationDecision`."""
        escalate = self.should_escalate(kind)
        reason = _ESCALATE_REASON if escalate else _NOMINAL_REASON
        return EscalationDecision(escalate=escalate, kind=kind, reason=reason)
