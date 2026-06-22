"""Q-RASP-Engine: Adversarial Quantum Emulation Sandbox.

Runtime self-protection for quantum state. The engine is organized as four
layers, separated so each can be tested and reasoned about in isolation:

* **core** (Rust, ``qrasp.qrasp_native``) - classifies a density matrix:
  purity ``Tr(rho^2)`` drops and trace violations.
* **monitors** (:mod:`qrasp.monitors`) - instrument a running quantum program
  and stream per-step snapshots into the core.
* **policies** (:mod:`qrasp.policies`) - decide whether a classification
  warrants an active-defense response (rollback / error-correction).
* **gateway** (:mod:`qrasp.gateway`) - FastAPI diagnostics & alert sink that
  records timelines and applies the policy over HTTP.
"""

__version__ = "0.1.0"

# The native extension is built by maturin from the qrasp-py crate. It may be
# absent during pure-Python development before a build, so import defensively.
try:
    from . import qrasp_native  # noqa: F401
    _NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NATIVE_AVAILABLE = False


def native_available() -> bool:
    """Return True if the compiled Rust core extension is importable."""
    return _NATIVE_AVAILABLE
