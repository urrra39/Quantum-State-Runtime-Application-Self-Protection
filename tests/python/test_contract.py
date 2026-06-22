"""Cross-layer contract tests for the anomaly-kind labels.

The three canonical labels (``nominal`` / ``purity_drop`` / ``trace_violation``)
are defined independently in four places that must stay byte-for-byte aligned:

  1. the Rust core enum ``qrasp_core::observer::Anomaly`` (mapped to strings in
     ``crates/qrasp-py/src/lib.rs``),
  2. the policy constants in ``qrasp.policies.escalation``,
  3. the gateway's Pydantic ``AnomalyKind`` enum, and
  4. the bridge's ``is_anomaly`` check (``kind != "nominal"``).

Nothing enforces that alignment at compile time, so these tests pin the two
pure-Python definitions (2) and (3) to each other. The Rust side (1) is covered
by the Rust integration tests; (4) is exercised in ``test_monitor.py``. If a
label is ever renamed on one side only, this fails loudly instead of silently
misclassifying an attack.
"""
from qrasp.gateway.schemas import AnomalyKind
from qrasp.policies import NOMINAL, PURITY_DROP, TRACE_VIOLATION
from qrasp.policies import escalation as esc


def test_policy_constants_match_gateway_enum_values() -> None:
    """The policy label constants equal the gateway enum's wire values."""
    assert NOMINAL == AnomalyKind.NOMINAL.value
    assert PURITY_DROP == AnomalyKind.PURITY_DROP.value
    assert TRACE_VIOLATION == AnomalyKind.TRACE_VIOLATION.value


def test_label_sets_are_identical_across_layers() -> None:
    """The full set of labels is the same in the policy layer and the gateway."""
    policy_labels = {NOMINAL, PURITY_DROP, TRACE_VIOLATION}
    gateway_labels = {k.value for k in AnomalyKind}
    assert policy_labels == gateway_labels


def test_default_escalating_kinds_are_real_gateway_kinds() -> None:
    """Every kind the default policy escalates is a kind the gateway can emit.

    Guards against a policy escalating on a label the rest of the system can
    never produce (a dead rule), which would mask a missing real rule.
    """
    gateway_labels = {k.value for k in AnomalyKind}
    assert esc.DEFAULT_ESCALATING_KINDS <= gateway_labels
    # ...and nominal is deliberately NOT among them.
    assert NOMINAL not in esc.DEFAULT_ESCALATING_KINDS
