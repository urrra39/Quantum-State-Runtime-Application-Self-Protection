use numpy::{PyReadonlyArray2, Complex64 as NpComplex64};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use qrasp_core::observer::{Anomaly, ObserverConfig, StateObserver};

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

    fn observe(&mut self, step: u64, rho: PyReadonlyArray2<NpComplex64>) -> PyResult<String> {
        let view = rho.as_array();
        let (rows, cols) = view.dim();
        if rows != cols {
            return Err(PyValueError::new_err("rho must be a square matrix"));
        }

        let owned = view.mapv(|c| num_complex::Complex64::new(c.re, c.im));

        let result = self.inner.observe(step, &owned);
        Ok(match result {
            Anomaly::Nominal => "nominal".to_string(),
            Anomaly::PurityDrop { from, to, delta } =>
                format!("purity_drop:{from:.6}->{to:.6}(delta={delta:.6})"),
            Anomaly::TraceViolation { trace } =>
                format!("trace_violation:{trace:.6}"),
        })
    }
}

#[pymodule]
fn qrasp_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStateObserver>()?;
    Ok(())
}