"""Tests for the escalation decision layer (``qrasp.policies``).

These exercise the pure-Python decision layer that maps an anomaly
classification to an active-defense decision. No native extension, qiskit, or
numpy is required, so they always run locally.
"""
import dataclasses

import pytest

from qrasp.policies import (
    DEFAULT_ESCALATING_KINDS,
    NOMINAL,
    PURITY_DROP,
    TRACE_VIOLATION,
    EscalationDecision,
    EscalationPolicy,
)

_ESCALATE_REASON = "active-defense trigger recommended"
_NOMINAL_REASON = "state nominal"


# -- canonical label constants ---------------------------------------------


def test_canonical_kind_labels_match_core_and_gateway() -> None:
    """The string labels must mirror the Rust core / gateway enum values."""
    assert NOMINAL == "nominal"
    assert PURITY_DROP == "purity_drop"
    assert TRACE_VIOLATION == "trace_violation"


def test_default_escalating_kinds_is_purity_drop_and_trace_violation() -> None:
    """The shipped default escalates exactly the two real anomaly kinds."""
    assert DEFAULT_ESCALATING_KINDS == frozenset({PURITY_DROP, TRACE_VIOLATION})
    assert NOMINAL not in DEFAULT_ESCALATING_KINDS


# -- default policy: should_escalate ----------------------------------------


def test_default_policy_does_not_escalate_nominal() -> None:
    """A nominal state never warrants active defense."""
    policy = EscalationPolicy()
    assert policy.should_escalate(NOMINAL) is False


def test_default_policy_escalates_purity_drop() -> None:
    """A purity drop (decoherence / side-channel leak) escalates by default."""
    policy = EscalationPolicy()
    assert policy.should_escalate(PURITY_DROP) is True


def test_default_policy_escalates_trace_violation() -> None:
    """A trace violation (tamper / clone) escalates by default."""
    policy = EscalationPolicy()
    assert policy.should_escalate(TRACE_VIOLATION) is True


# -- default policy: decide() -----------------------------------------------


def test_decide_returns_escalation_decision_instance() -> None:
    """``decide`` returns a structured :class:`EscalationDecision`."""
    decision = EscalationPolicy().decide(NOMINAL)
    assert isinstance(decision, EscalationDecision)


def test_decide_nominal_returns_non_escalating_decision() -> None:
    """``decide`` on a nominal kind yields a non-escalating decision."""
    decision = EscalationPolicy().decide(NOMINAL)
    assert decision.escalate is False
    assert decision.kind == NOMINAL
    assert decision.reason == _NOMINAL_REASON


def test_decide_purity_drop_returns_escalating_decision() -> None:
    """``decide`` on a purity drop yields an escalating decision with reason."""
    decision = EscalationPolicy().decide(PURITY_DROP)
    assert decision.escalate is True
    assert decision.kind == PURITY_DROP
    assert decision.reason == _ESCALATE_REASON


def test_decide_trace_violation_returns_escalating_decision() -> None:
    """``decide`` on a trace violation yields an escalating decision."""
    decision = EscalationPolicy().decide(TRACE_VIOLATION)
    assert decision.escalate is True
    assert decision.kind == TRACE_VIOLATION
    assert decision.reason == _ESCALATE_REASON


def test_decide_echoes_kind_verbatim() -> None:
    """``decide`` echoes back the exact kind string it was given."""
    for kind in (NOMINAL, PURITY_DROP, TRACE_VIOLATION):
        assert EscalationPolicy().decide(kind).kind == kind


def test_decide_is_consistent_with_should_escalate() -> None:
    """``decide().escalate`` must agree with ``should_escalate`` for every kind."""
    policy = EscalationPolicy()
    for kind in (NOMINAL, PURITY_DROP, TRACE_VIOLATION, "garbage"):
        assert policy.decide(kind).escalate == policy.should_escalate(kind)


# -- configurability --------------------------------------------------------


def test_custom_policy_escalates_only_configured_kinds() -> None:
    """A policy restricted to trace violations ignores purity drops."""
    policy = EscalationPolicy(escalating_kinds=frozenset({TRACE_VIOLATION}))
    assert policy.should_escalate(TRACE_VIOLATION) is True
    assert policy.should_escalate(PURITY_DROP) is False
    assert policy.should_escalate(NOMINAL) is False


def test_custom_policy_decide_reflects_restricted_set() -> None:
    """``decide`` honors the custom escalating set in both flag and reason."""
    policy = EscalationPolicy(escalating_kinds=frozenset({TRACE_VIOLATION}))

    trace = policy.decide(TRACE_VIOLATION)
    assert trace.escalate is True
    assert trace.reason == _ESCALATE_REASON

    purity = policy.decide(PURITY_DROP)
    assert purity.escalate is False
    assert purity.reason == _NOMINAL_REASON


def test_empty_policy_escalates_nothing() -> None:
    """An empty escalating set is fail-open by construction: nothing escalates."""
    policy = EscalationPolicy(escalating_kinds=frozenset())
    for kind in (NOMINAL, PURITY_DROP, TRACE_VIOLATION):
        assert policy.should_escalate(kind) is False
        assert policy.decide(kind).escalate is False


# -- adversarial / unknown kinds --------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        "",
        "PURITY_DROP",  # wrong case
        "purity-drop",  # wrong separator
        "tracе_violation",  # cyrillic 'е' homoglyph, not ASCII 'e'
        "unknown",
        "nominal ",  # trailing whitespace
        "purity_drop\x00",  # embedded NUL
        "drop_table",
    ],
)
def test_unknown_or_garbage_kinds_never_escalate(kind: str) -> None:
    """Any kind not in the escalating set is treated as nominal (no escalation)."""
    policy = EscalationPolicy()
    assert policy.should_escalate(kind) is False
    decision = policy.decide(kind)
    assert decision.escalate is False
    assert decision.reason == _NOMINAL_REASON
    assert decision.kind == kind


# -- immutability -----------------------------------------------------------


def test_policy_is_frozen() -> None:
    """The policy is immutable so a shared instance is concurrency-safe."""
    policy = EscalationPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.escalating_kinds = frozenset({NOMINAL})  # type: ignore[misc]


def test_decision_is_frozen() -> None:
    """An emitted decision is immutable; callers cannot rewrite the verdict."""
    decision = EscalationPolicy().decide(PURITY_DROP)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.escalate = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.kind = NOMINAL  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.reason = "tampered"  # type: ignore[misc]
