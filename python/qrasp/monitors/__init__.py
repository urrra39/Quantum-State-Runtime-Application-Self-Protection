"""Runtime monitors for Q-RASP-Engine (the detection layer).

A *monitor* observes a running quantum program and streams per-step state
snapshots into the Rust core ``StateObserver`` for classification. This package
is the canonical home of that layer.

The first concrete monitor is the Qiskit bridge, which instruments a
``QuantumCircuit`` by reconstructing the density matrix after every gate. It
physically lives under :mod:`qrasp.bridge` (simulator-specific bridges) and is
re-exported here so callers can depend on the stable ``qrasp.monitors`` surface
regardless of which simulator backs a given monitor:

    from qrasp.monitors import QiskitObserverAdapter, Backend

Future monitors (Cirq, hardware telemetry taps) should expose the same
``analyze(program, run_id) -> AnalysisReport`` shape and be re-exported here.
"""
from __future__ import annotations

from qrasp.bridge.qiskit_adapter import (
    AnalysisReport,
    Backend,
    QiskitObserverAdapter,
    StateGuard,
    StepResult,
)

__all__ = [
    "AnalysisReport",
    "Backend",
    "QiskitObserverAdapter",
    "StateGuard",
    "StepResult",
]
