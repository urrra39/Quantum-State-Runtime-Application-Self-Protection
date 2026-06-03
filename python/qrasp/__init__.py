"""Q-RASP-Engine: Adversarial Quantum Emulation Sandbox.

This package provides the Python integration layer for the Rust core observer
(exposed as the native extension ``qrasp.qrasp_native``), simulator bridges,
and the FastAPI diagnostics & alert gateway.
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
