"""Pure-Python tests for the monitor data layer (``qrasp.bridge.qiskit_adapter``).

These cover the data structures and backend-selection logic that do not require
the Rust native extension or qiskit/qiskit-aer. Only numpy is needed, so they
always run locally. The ``QiskitObserverAdapter.analyze`` path (which needs the
native observer and a real circuit) is intentionally not exercised here.
"""
import sys

import numpy as np
import pytest

from qrasp.bridge.qiskit_adapter import (
    AnalysisReport,
    Backend,
    QiskitObserverAdapter,
    StateGuard,
    StepResult,
    _resolve_backend,
    _validate_gateway_url,
)


def _step(step: int, kind: str) -> StepResult:
    """Build a minimal StepResult with the given classification kind."""
    return StepResult(
        step=step,
        gate="h",
        qubits=[0],
        kind=kind,
        purity=1.0,
        trace=1.0,
        delta=None,
    )


# -- StepResult.is_anomaly --------------------------------------------------


def test_step_result_nominal_is_not_anomaly() -> None:
    """A nominal classification is not an anomaly."""
    assert _step(0, "nominal").is_anomaly is False


@pytest.mark.parametrize("kind", ["purity_drop", "trace_violation", "weird"])
def test_step_result_non_nominal_is_anomaly(kind: str) -> None:
    """Any non-nominal classification is reported as an anomaly."""
    assert _step(0, kind).is_anomaly is True


# -- AnalysisReport.anomalies -----------------------------------------------


def test_report_anomalies_filters_and_preserves_order() -> None:
    """``anomalies`` returns only anomalous steps, in their original order."""
    results = [
        _step(0, "nominal"),
        _step(1, "purity_drop"),
        _step(2, "nominal"),
        _step(3, "trace_violation"),
        _step(4, "nominal"),
    ]
    report = AnalysisReport(
        run_id="run-x",
        backend=Backend.QUANTUM_INFO,
        num_qubits=1,
        num_steps=len(results),
        results=results,
    )
    anomalies = report.anomalies
    assert [r.step for r in anomalies] == [1, 3]
    assert [r.kind for r in anomalies] == ["purity_drop", "trace_violation"]


def test_report_with_no_anomalies_returns_empty_list() -> None:
    """A clean run yields an empty anomaly list."""
    report = AnalysisReport(
        run_id="run-clean",
        backend=Backend.QUANTUM_INFO,
        num_qubits=2,
        num_steps=2,
        results=[_step(0, "nominal"), _step(1, "nominal")],
    )
    assert report.anomalies == []


def test_report_defaults_results_and_rollbacks_to_empty() -> None:
    """A freshly constructed report has independent empty result/rollback lists."""
    report = AnalysisReport(
        run_id="run-empty",
        backend=Backend.AER,
        num_qubits=1,
        num_steps=0,
    )
    assert report.results == []
    assert report.rollbacks == []
    assert report.anomalies == []


# -- Backend enum -----------------------------------------------------------


def test_backend_enum_string_values() -> None:
    """Backend members carry the expected wire string values."""
    assert Backend.AER.value == "aer"
    assert Backend.QUANTUM_INFO.value == "quantum_info"


def test_backend_is_str_enum() -> None:
    """Backend is a str subclass, so members compare equal to their values."""
    assert Backend.AER == "aer"
    assert isinstance(Backend.AER, str)


# -- _resolve_backend -------------------------------------------------------


def test_resolve_backend_honors_quantum_info_preference() -> None:
    """An explicit quantum_info preference is always honored, never probed."""
    assert _resolve_backend(Backend.QUANTUM_INFO) is Backend.QUANTUM_INFO


def test_resolve_backend_falls_back_when_aer_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no preference and qiskit-aer unimportable, it degrades to quantum_info."""
    # Setting the module entry to None makes ``import qiskit_aer`` raise
    # ImportError, simulating an environment without qiskit-aer installed.
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)
    assert _resolve_backend(None) is Backend.QUANTUM_INFO


def test_resolve_backend_prefers_aer_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no preference and qiskit-aer importable, it selects the aer backend."""
    # Inject a dummy module so the optional ``import qiskit_aer`` succeeds even
    # though qiskit-aer is not installed in this environment.
    import types

    monkeypatch.setitem(sys.modules, "qiskit_aer", types.ModuleType("qiskit_aer"))
    assert _resolve_backend(None) is Backend.AER


def test_resolve_backend_quantum_info_preference_ignores_available_aer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit quantum_info preference wins even if aer could be imported."""
    import types

    monkeypatch.setitem(sys.modules, "qiskit_aer", types.ModuleType("qiskit_aer"))
    assert _resolve_backend(Backend.QUANTUM_INFO) is Backend.QUANTUM_INFO


# -- _validate_gateway_url (SSRF / scheme hardening) ------------------------


def test_validate_gateway_url_accepts_http_and_strips_trailing_slash() -> None:
    """A well-formed http(s) URL is accepted with any trailing slash removed."""
    assert _validate_gateway_url("http://localhost:8000/") == "http://localhost:8000"
    assert _validate_gateway_url("https://gw.example:9000") == "https://gw.example:9000"


def test_validate_gateway_url_passes_through_none_and_empty() -> None:
    """No gateway configured is a valid state (local policy decides)."""
    assert _validate_gateway_url(None) is None
    assert _validate_gateway_url("") is None


@pytest.mark.parametrize(
    "bad_url",
    [
        "file:///etc/passwd",  # local file read
        "ftp://host/x",  # non-http scheme
        "gopher://host",  # legacy SSRF vector
        "http://",  # no host
    ],
)
def test_validate_gateway_url_rejects_non_http_or_hostless(bad_url: str) -> None:
    """Non-http(s) schemes and host-less URLs are rejected (CWE-22 guard)."""
    with pytest.raises(ValueError):
        _validate_gateway_url(bad_url)


# -- StateGuard edge cases (NEW; not duplicating test_rollback.py) ----------


def test_rollback_returns_independent_copies_across_calls() -> None:
    """Each rollback() yields a fresh array; mutating one must not affect later ones.

    This guards the active-defense mechanism: a caller that scribbles on a
    restored state must not poison the cached snapshot or any subsequent
    restoration.
    """
    guard = StateGuard()
    nominal = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    guard.snapshot(nominal, purity=1.0)

    first = guard.rollback()
    second = guard.rollback()
    assert first is not None and second is not None
    # Distinct array objects, not aliases of the same buffer.
    assert first is not second
    assert not np.shares_memory(first, second)

    # Corrupt the first restored copy.
    first[0, 0] = 0.0
    # A later rollback is unaffected by the mutation of an earlier copy.
    third = guard.rollback()
    assert third is not None
    assert third[0, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(second, nominal)


def test_second_snapshot_supersedes_first_as_rollback_target() -> None:
    """The guard rolls back to the most recently cached nominal state."""
    guard = StateGuard()
    first_state = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    second_state = np.array([[0, 0], [0, 1]], dtype=np.complex128)

    guard.snapshot(first_state, purity=1.0)
    guard.snapshot(second_state, purity=1.0)

    restored = guard.rollback()
    assert restored is not None
    np.testing.assert_allclose(restored, second_state)


# -- monitors facade re-exports ---------------------------------------------


def test_monitors_facade_reexports_are_importable() -> None:
    """The stable ``qrasp.monitors`` surface re-exports the bridge symbols."""
    from qrasp.monitors import (
        AnalysisReport as FacadeAnalysisReport,
    )
    from qrasp.monitors import (
        Backend as FacadeBackend,
    )
    from qrasp.monitors import (
        QiskitObserverAdapter as FacadeAdapter,
    )
    from qrasp.monitors import (
        StateGuard as FacadeStateGuard,
    )
    from qrasp.monitors import (
        StepResult as FacadeStepResult,
    )

    # The facade must re-export the very same objects as the bridge module,
    # not look-alikes, so both import paths agree.
    assert FacadeAdapter is QiskitObserverAdapter
    assert FacadeAnalysisReport is AnalysisReport
    assert FacadeBackend is Backend
    assert FacadeStateGuard is StateGuard
    assert FacadeStepResult is StepResult
