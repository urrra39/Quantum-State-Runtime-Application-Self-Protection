"""Qiskit bridge adapter for Q-RASP-Engine.

Bridges a Qiskit ``QuantumCircuit`` to the Rust core ``StateObserver`` (exposed
via the ``qrasp.qrasp_native`` PyO3 extension) by reconstructing the system's
density matrix after every gate and streaming each snapshot into the observer.

Design note (read this):
    Qiskit/Aer provide **no supported per-gate runtime callback** during a
    single ``run()``. To obtain the state after every step we instrument the
    circuit: for each prefix of length ``k`` we simulate ``circuit[:k]`` and
    capture its density matrix. This is O(n) simulations for an n-gate circuit
    and is intended for research-scale circuits, not large production jobs.

    The prefix-reconstruction path assumes **unitary** instructions. Explicit
    non-unitary channels (noise) are out of scope for the circuit path and
    should be applied directly to a ``DensityMatrix`` and fed to the observer.

    Two backends are supported, selected automatically:
      * ``aer``          - AerSimulator with ``save_density_matrix`` (preferred).
      * ``quantum_info`` - pure ``qiskit.quantum_info`` evolution (no Aer).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Instructions that are not norm/trace-preserving unitaries and must be skipped
# when reconstructing the density-matrix trajectory.
_SKIP_OPS = frozenset({"measure", "barrier", "reset", "snapshot", "delay"})


class Backend(str, Enum):
    """State-reconstruction backend."""

    AER = "aer"
    QUANTUM_INFO = "quantum_info"


@dataclass
class StepResult:
    """Observer classification for the state after one gate."""

    step: int
    gate: str
    qubits: List[int]
    kind: str
    purity: float
    trace: float
    delta: Optional[float] = None

    @property
    def is_anomaly(self) -> bool:
        return self.kind != "nominal"


@dataclass
class AnalysisReport:
    """Full anomaly timeline for one circuit run."""

    run_id: str
    backend: Backend
    num_qubits: int
    num_steps: int
    results: List[StepResult] = field(default_factory=list)

    # Steps at which an escalation triggered a state rollback.
    rollbacks: List[int] = field(default_factory=list)

    @property
    def anomalies(self) -> List[StepResult]:
        """Steps the observer did not classify as nominal."""
        return [r for r in self.results if r.is_anomaly]


def _resolve_backend(prefer: Optional[Backend]) -> Backend:
    """Pick a backend, honoring preference but degrading gracefully."""
    if prefer is Backend.QUANTUM_INFO:
        return Backend.QUANTUM_INFO
    try:
        import qiskit_aer  # noqa: F401

        return Backend.AER
    except ImportError:
        logger.warning(
            "qiskit-aer not available; falling back to quantum_info backend."
        )
        return Backend.QUANTUM_INFO


class QiskitObserverAdapter:
    """Streams per-gate density matrices from a Qiskit circuit into the
    Rust ``StateObserver``.

    Example:
        adapter = QiskitObserverAdapter(purity_drop_threshold=0.01)
        report = adapter.analyze(circuit, run_id="demo")
        for a in report.anomalies:
            print(a.step, a.gate, a.kind, a.delta)
    """

    def __init__(
        self,
        purity_drop_threshold: float = 0.01,
        backend: Optional[Backend] = None,
        gateway_url: Optional[str] = None,
    ) -> None:
        # Import the PyO3 core lazily so the module is importable even before
        # the Rust extension has been built (e.g. during pure-Python CI lint).
        try:
            from qrasp.qrasp_native import StateObserver
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The qrasp native extension is not built. Run "
                "`pip install -e .` (maturin) before using the Qiskit adapter."
            ) from exc

        self._StateObserver = StateObserver
        self._purity_drop_threshold = purity_drop_threshold
        self._backend = _resolve_backend(backend)
        # Optional FastAPI gateway base URL (e.g. "http://localhost:8000").
        # When set, the adapter POSTs events and honors the gateway's
        # `escalated` decision; when unset, it escalates locally on any anomaly.
        self._gateway_url = gateway_url.rstrip("/") if gateway_url else None

    # -- public API ---------------------------------------------------------

    def analyze(self, circuit, run_id: str) -> AnalysisReport:
        """Reconstruct the density matrix after each gate and classify it.

        Args:
            circuit: a Qiskit ``QuantumCircuit`` of unitary gates. Measurements
                and resets in the prefix are skipped by the reconstruction.
            run_id: logical id for this execution.

        Returns:
            An ``AnalysisReport`` with the full per-step classification timeline.
        """
        observer = self._StateObserver(self._purity_drop_threshold)

        instructions = self._gate_instructions(circuit)
        report = AnalysisReport(
            run_id=run_id,
            backend=self._backend,
            num_qubits=circuit.num_qubits,
            num_steps=len(instructions),
        )

        guard = StateGuard()
        for step, (name, qubit_indices) in enumerate(instructions):
            rho = self._density_matrix_after(circuit, step + 1)
            rho = np.ascontiguousarray(rho, dtype=np.complex128)
            event = observer.observe(step, rho)
            # Clamp reconstructed purity into [0, 1] for reporting; tiny
            # floating-point overshoots above 1.0 are physically meaningless.
            reported_purity = float(np.clip(event.purity, 0.0, 1.0))
            report.results.append(
                StepResult(
                    step=step,
                    gate=name,
                    qubits=qubit_indices,
                    kind=event.kind,
                    purity=reported_purity,
                    trace=event.trace,
                    delta=event.delta,
                )
            )
            if event.is_anomaly():
                logger.warning(
                    "Q-RASP anomaly at step %d (%s on %s): %s (purity=%.6f)",
                    step, name, qubit_indices, event.kind, event.purity,
                )
                # Active defense: ask the policy whether to escalate, and if so
                # roll the system back to the last known-nominal snapshot.
                if self._escalate(event):
                    restored = guard.rollback()
                    if restored is not None:
                        report.rollbacks.append(step)
                        logger.warning(
                            "Q-RASP rolled back step %d to last nominal snapshot "
                            "(purity=%.6f)", step, guard.last_nominal_purity,
                        )
            else:
                # Cache this clean state as a rollback target.
                guard.snapshot(rho, event.purity)

        return report

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _gate_instructions(circuit):
        """Return [(gate_name, [qubit indices])] for analyzed unitary ops."""
        out = []
        for instruction in circuit.data:
            op = instruction.operation
            if op.name in _SKIP_OPS:
                continue
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            out.append((op.name, qubits))
        return out

    def _prefix_circuit(self, circuit, length: int):
        """Build a copy of ``circuit`` truncated to ``length`` analyzed ops."""
        prefix = circuit.copy_empty_like()
        appended = 0
        for instruction in circuit.data:
            if instruction.operation.name in _SKIP_OPS:
                continue
            if appended >= length:
                break
            prefix.append(
                instruction.operation, instruction.qubits, instruction.clbits
            )
            appended += 1
        return prefix

    def _density_matrix_after(self, circuit, length: int) -> np.ndarray:
        """Return the density matrix after the first ``length`` analyzed ops."""
        prefix = self._prefix_circuit(circuit, length)
        if self._backend is Backend.AER:
            return self._density_matrix_aer(prefix)
        return self._density_matrix_quantum_info(prefix)

    @staticmethod
    def _density_matrix_aer(prefix) -> np.ndarray:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        instrumented = prefix.copy()
        instrumented.save_density_matrix()
        sim = AerSimulator(method="density_matrix")
        result = sim.run(transpile(instrumented, sim)).result()
        return np.asarray(result.data(0)["density_matrix"], dtype=np.complex128)

    @staticmethod
    def _density_matrix_quantum_info(prefix) -> np.ndarray:
        from qiskit.quantum_info import DensityMatrix

        return np.asarray(DensityMatrix(prefix).data, dtype=np.complex128)

    # -- active defense -----------------------------------------------------

    def _escalate(self, event) -> bool:
        """Decide whether an anomaly warrants rollback.

        If a gateway URL is configured, the gateway is authoritative: the
        adapter POSTs the event and obeys the returned `escalated` flag. If the
        gateway is unreachable, the adapter fails safe and escalates locally.
        With no gateway configured, any anomaly escalates.
        """
        if self._gateway_url is None:
            return True
        try:
            return self._report_to_gateway(event)
        except Exception as exc:  # network/JSON errors -> fail safe
            logger.warning("Gateway unreachable (%s); escalating locally.", exc)
            return True

    def _report_to_gateway(self, event, run_id: str = "adapter") -> bool:
        """POST an anomaly event to the gateway and return its escalation flag.

        Uses only the stdlib so the adapter has no hard HTTP dependency.
        """
        import json
        import urllib.request

        purity = float(np.clip(event.purity, 0.0, 1.0))
        payload = json.dumps(
            {
                "run_id": run_id,
                "step": int(event.step),
                "kind": event.kind,
                "purity": purity,
                "trace": float(event.trace),
                "delta": (None if event.delta is None else float(event.delta)),
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._gateway_url}/v1/events",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return bool(body.get("escalated", True))


class StateGuard:
    """Caches the last known-nominal density matrix and restores it on demand.

    This is the simulator-side active-defense *mechanism*. On a classical
    simulator a "rollback" is the faithful restoration of a previously captured
    state vector / density matrix, which is a legitimate and demonstrable
    capability (it is not a claim about physical QPU state restoration).

    FUTURE RESEARCH GOAL (not implemented): error-correction *gate injection*,
    i.e. synthesizing and applying corrective gates / a stabilizer round to
    actively repair the state in place rather than discarding the corrupted
    trajectory. This is meaningful only for specific code structures and is
    deliberately out of scope here.
    """

    def __init__(self) -> None:
        self._snapshot: Optional[np.ndarray] = None
        self.last_nominal_purity: float = float("nan")

    def snapshot(self, rho: np.ndarray, purity: float) -> None:
        """Cache a copy of a known-nominal density matrix as a rollback target."""
        self._snapshot = np.array(rho, dtype=np.complex128, copy=True)
        self.last_nominal_purity = float(np.clip(purity, 0.0, 1.0))

    def rollback(self) -> Optional[np.ndarray]:
        """Return a copy of the last cached nominal state, or None if none."""
        if self._snapshot is None:
            return None
        return np.array(self._snapshot, dtype=np.complex128, copy=True)

    @property
    def has_snapshot(self) -> bool:
        return self._snapshot is not None
