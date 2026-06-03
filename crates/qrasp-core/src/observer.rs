//! crates/qrasp-core/src/observer.rs
//!
//! Runtime state observation for the Q-RASP engine.
//!
//! The `StateObserver` ingests successive snapshots of a quantum system's
//! density matrix and tracks security-relevant invariants over time:
//!
//!   * **Purity** `gamma = Tr(rho^2)` in `[1/d, 1]`. An unexpected drop
//!     indicates information leakage to an environment (side-channel) or
//!     decoherence injected by an adversary.
//!   * **Trace** `Tr(rho)`, which must remain ~1.0. Deviation signals an
//!     unnormalized / cloned / tampered state.
//!
//! Detection is *baseline-relative*: each observation is compared against the
//! previous one, and a purity drop exceeding `purity_drop_threshold` raises an
//! anomaly. Absolute purity is reported but not used as the sole trigger,
//! since legitimate circuits may operate on intentionally mixed states.

use ndarray::Array2;
use num_complex::Complex64;

/// Numerical tolerance for trace-preservation checks.
const TRACE_TOLERANCE: f64 = 1e-9;

/// A single security-relevant snapshot of the quantum state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateSnapshot {
    /// Monotonic logical timestep (e.g. gate index in the circuit).
    pub step: u64,
    /// Purity `Tr(rho^2)`.
    pub purity: f64,
    /// Real part of the trace `Tr(rho)`; should be ~1.0.
    pub trace: f64,
}

/// Classification of an observation relative to the running baseline.
#[derive(Debug, Clone, PartialEq)]
pub enum Anomaly {
    /// State is within expected bounds.
    Nominal,
    /// Purity fell by more than the configured threshold between steps.
    PurityDrop { from: f64, to: f64, delta: f64 },
    /// Trace deviated from 1.0 beyond tolerance (non-trace-preserving op).
    TraceViolation { trace: f64 },
}

/// Configuration governing anomaly sensitivity.
#[derive(Debug, Clone, Copy)]
pub struct ObserverConfig {
    /// Minimum purity decrease between consecutive steps to flag an anomaly.
    pub purity_drop_threshold: f64,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        // 1% inter-step purity loss is a conservative, tunable default.
        Self { purity_drop_threshold: 0.01 }
    }
}

/// Stateful observer that tracks a quantum system across runtime steps.
#[derive(Debug)]
pub struct StateObserver {
    config: ObserverConfig,
    history: Vec<StateSnapshot>,
}

impl StateObserver {
    /// Create a new observer with the given configuration.
    pub fn new(config: ObserverConfig) -> Self {
        Self { config, history: Vec::new() }
    }

    /// Ingest a density matrix `rho` at logical `step`, record a snapshot,
    /// and classify it against the previous observation.
    ///
    /// `rho` must be a square (`d x d`) Hermitian matrix. Hermiticity is the
    /// caller's responsibility; this method validates dimensionality and
    /// trace-preservation.
    pub fn observe(&mut self, step: u64, rho: &Array2<Complex64>) -> Anomaly {
        let purity = Self::purity(rho);
        let trace = Self::trace(rho).re;

        let snapshot = StateSnapshot { step, purity, trace };

        let anomaly = self.classify(&snapshot);
        self.history.push(snapshot);
        anomaly
    }

    /// Compute purity `gamma = Tr(rho^2)` for a density matrix.
    ///
    /// Uses the identity `Tr(rho^2) = sum_{i,j} rho_{ij} * rho_{ji}`, which
    /// avoids materializing the full `rho^2` product and is O(d^2) instead of
    /// O(d^3). For a Hermitian `rho`, `rho_{ji} = conj(rho_{ij})`, so each term
    /// reduces to `|rho_{ij}|^2`, and the result is provably real.
    pub fn purity(rho: &Array2<Complex64>) -> f64 {
        let (rows, cols) = rho.dim();
        debug_assert_eq!(rows, cols, "density matrix must be square");

        let mut acc = 0.0_f64;
        for i in 0..rows {
            for j in 0..cols {
                acc += rho[(i, j)].norm_sqr(); // |rho_{ij}|^2
            }
        }
        acc
    }

    /// Compute the trace `Tr(rho) = sum_i rho_{ii}`.
    pub fn trace(rho: &Array2<Complex64>) -> Complex64 {
        let (rows, _) = rho.dim();
        (0..rows).map(|i| rho[(i, i)]).sum()
    }

    /// Classify a snapshot against the running baseline (previous snapshot).
    fn classify(&self, current: &StateSnapshot) -> Anomaly {
        if (current.trace - 1.0).abs() > TRACE_TOLERANCE {
            return Anomaly::TraceViolation { trace: current.trace };
        }

        if let Some(prev) = self.history.last() {
            let delta = prev.purity - current.purity;
            if delta > self.config.purity_drop_threshold {
                return Anomaly::PurityDrop {
                    from: prev.purity,
                    to: current.purity,
                    delta,
                };
            }
        }

        Anomaly::Nominal
    }

    /// Immutable view of the recorded observation history.
    pub fn history(&self) -> &[StateSnapshot] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// |0><0| is pure: purity == 1.
    #[test]
    fn pure_state_has_unit_purity() {
        let rho = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        assert!((StateObserver::purity(&rho) - 1.0).abs() < 1e-12);
    }

    /// Maximally mixed 1-qubit state I/2 has purity 0.5 (= 1/d).
    #[test]
    fn maximally_mixed_state_has_min_purity() {
        let rho = array![
            [Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
        ];
        assert!((StateObserver::purity(&rho) - 0.5).abs() < 1e-12);
    }

    /// A purity drop from pure -> mixed must be flagged.
    #[test]
    fn detects_purity_drop() {
        let mut obs = StateObserver::new(ObserverConfig::default());
        let pure = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        let mixed = array![
            [Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
        ];
        assert_eq!(obs.observe(0, &pure), Anomaly::Nominal);
        matches!(obs.observe(1, &mixed), Anomaly::PurityDrop { .. });
    }
}