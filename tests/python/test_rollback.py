"""Tests for the Phase 5 StateGuard active-defense rollback mechanism."""
import numpy as np
import pytest

from qrasp.bridge.qiskit_adapter import StateGuard


def _pure_state() -> np.ndarray:
    # |0><0|
    return np.array([[1, 0], [0, 0]], dtype=np.complex128)


def _mixed_state() -> np.ndarray:
    # maximally mixed I/2
    return np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)


def test_rollback_without_snapshot_returns_none() -> None:
    guard = StateGuard()
    assert guard.has_snapshot is False
    assert guard.rollback() is None


def test_rollback_restores_last_nominal_snapshot() -> None:
    guard = StateGuard()
    nominal = _pure_state()
    guard.snapshot(nominal, purity=1.0)

    assert guard.has_snapshot is True
    assert guard.last_nominal_purity == pytest.approx(1.0)

    restored = guard.rollback()
    assert restored is not None
    np.testing.assert_allclose(restored, nominal)


def test_snapshot_is_defensively_copied() -> None:
    """Mutating the source array must not corrupt the cached snapshot."""
    guard = StateGuard()
    source = _pure_state()
    guard.snapshot(source, purity=1.0)

    source[0, 0] = 0.0  # mutate after snapshot
    restored = guard.rollback()
    assert restored[0, 0] == pytest.approx(1.0)


def test_purity_is_clamped_into_unit_interval() -> None:
    guard = StateGuard()
    # A floating-point overshoot above 1.0 must be clamped.
    guard.snapshot(_pure_state(), purity=1.0000000002)
    assert guard.last_nominal_purity <= 1.0
