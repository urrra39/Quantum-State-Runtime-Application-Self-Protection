"""Integration tests for the Qiskit bridge + typed Anomaly payload.

Two independent assertions:
  1. A noiseless Bell circuit, run through the full adapter, stays pure
     (purity ~ 1) and produces no anomalies.
  2. A deterministic depolarizing channel applied via ``quantum_info`` Kraus
     operators (backend-independent, no Aer) drops the Bell state's purity and
     is cleanly caught by the Rust core ``StateObserver`` through the typed
     ``AnomalyEvent`` payload.

The tests skip automatically if Qiskit or the native extension are unavailable,
so they never produce false failures in a pure-Python lint job.
"""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
native = pytest.importorskip("qrasp.qrasp_native")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import DensityMatrix, Kraus  # noqa: E402

from qrasp.bridge.qiskit_adapter import Backend, QiskitObserverAdapter  # noqa: E402


def _bell_circuit() -> "QuantumCircuit":
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _depolarizing_kraus(p: float) -> Kraus:
    """Single-qubit depolarizing channel as explicit Kraus operators.

    rho -> (1 - p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
    """
    identity = np.array([[1, 0], [0, 1]], dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    k0 = np.sqrt(1.0 - p) * identity
    k1 = np.sqrt(p / 3.0) * x
    k2 = np.sqrt(p / 3.0) * y
    k3 = np.sqrt(p / 3.0) * z
    return Kraus([k0, k1, k2, k3])


def test_clean_bell_state_is_nominal() -> None:
    """A noiseless Bell circuit stays pure (purity ~ 1) throughout."""
    adapter = QiskitObserverAdapter(
        purity_drop_threshold=0.01, backend=Backend.QUANTUM_INFO
    )
    report = adapter.analyze(_bell_circuit(), run_id="bell-clean")

    assert report.num_steps == 2
    assert report.anomalies == []
    for r in report.results:
        assert r.purity == pytest.approx(1.0, abs=1e-9)
        assert r.trace == pytest.approx(1.0, abs=1e-9)


def test_depolarizing_kraus_triggers_purity_drop() -> None:
    """A deterministic depolarizing channel on the Bell state must drop purity
    and raise a `purity_drop` anomaly via the typed AnomalyEvent payload.

    Backend-independent: builds the trajectory with quantum_info only and feeds
    matrices straight into the Rust StateObserver.
    """
    observer = native.StateObserver(0.01)

    # Step 0: clean, pure Bell state (purity == 1).
    bell = DensityMatrix(_bell_circuit())
    rho0 = np.ascontiguousarray(bell.data, dtype=np.complex128)
    event0 = observer.observe(0, rho0)
    assert event0.kind == "nominal"
    assert event0.purity == pytest.approx(1.0, abs=1e-9)
    assert event0.trace == pytest.approx(1.0, abs=1e-9)

    # Step 1: apply a strong depolarizing channel to qubit 0.
    channel = _depolarizing_kraus(0.75)
    noisy = bell.evolve(channel, qargs=[0])
    rho1 = np.ascontiguousarray(noisy.data, dtype=np.complex128)
    event1 = observer.observe(1, rho1)

    # The channel is trace-preserving (trace stays ~1) but reduces purity.
    assert event1.trace == pytest.approx(1.0, abs=1e-9)
    assert event1.kind == "purity_drop"
    assert event1.purity < 1.0
    assert event1.delta is not None and event1.delta > 0.01
