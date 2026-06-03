//! PyO3 bindings exposing `qrasp-core` to Python.
//!
//! This crate is a marshalling boundary only: it converts NumPy complex
//! arrays into `ndarray` matrices and delegates all computation to
//! `qrasp-core`. `observe()` returns a structured `AnomalyEvent` whose `kind`
//! field aligns with the gateway's `AnomalyKind` enum
//! (`nominal` / `purity_drop` / `trace_violation`).

use numpy::{Complex64 as NpComplex64, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use qrasp_core::observer::{Anomaly, ObserverConfig, StateObserver};

/// Structured classification returned to Python for a single observation.
///
/// `kind` is one of `"nominal"`, `"purity_drop"`, `"trace_violation"` and maps
/// 1:1 onto the FastAPI gateway's `AnomalyKind` enum. `delta` is populated only
/// for purity drops.
#[pyclass(name = "AnomalyEvent", get_all)]
#[derive(Clone)]
struct PyAnomalyEvent {
    kind: String,
    step: u64,
    purity: f64,
    trace: f64,
    delta: Option<f64>,
}

#[pymethods]
impl PyAnomalyEvent {
    fn __repr__(&self) -> String {
        format!(
            "AnomalyEvent(kind='{}', step={}, purity={:.6}, trace={:.6}, delta={:?})",
            self.kind, self.step, self.purity, self.trace, self.delta
        )
    }

    /// True if this event is anything other than nominal.
    fn is_anomaly(&self) -> bool {
        self.kind != "nominal"
    }
}

#[pyclass(name = "StateObserver")]
struct PyStateObserver {
    inner: StateObserver,
}

#[pymethods]
impl PyStateObserver {
    #[new]
    #[pyo3(signature = (purity_drop_threshold = 0.01))]
    fn new(purity_drop_threshold: f64) -> Self {
        let config = ObserverConfig { purity_drop_threshold };
        Self { inner: StateObserver::new(config) }
    }

    /// Ingest a density matrix `rho` (complex, square) at logical `step` and
    /// return a structured `AnomalyEvent`.
    fn observe(
        &mut self,
        step: u64,
        rho: PyReadonlyArray2<NpComplex64>,
    ) -> PyResult<PyAnomalyEvent> {
        let view = rho.as_array();
        let (rows, cols) = view.dim();
        if rows != cols {
            return Err(PyValueError::new_err("rho must be a square matrix"));
        }

        let owned = view.mapv(|c| num_complex::Complex64::new(c.re, c.im));
        let result = self.inner.observe(step, &owned);

        // The snapshot just recorded carries the canonical purity/trace.
        let snap = self
            .inner
            .history()
            .last()
            .expect("observe() always records a snapshot");

        let (kind, delta) = match result {
            Anomaly::Nominal => ("nominal", None),
            Anomaly::PurityDrop { delta, .. } => ("purity_drop", Some(delta)),
            Anomaly::TraceViolation { .. } => ("trace_violation", None),
        };

        Ok(PyAnomalyEvent {
            kind: kind.to_string(),
            step: snap.step,
            purity: snap.purity,
            trace: snap.trace,
            delta,
        })
    }
}

/// The native extension module: `from qrasp import qrasp_native`.
#[pymodule]
fn qrasp_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStateObserver>()?;
    m.add_class::<PyAnomalyEvent>()?;
    Ok(())
}
