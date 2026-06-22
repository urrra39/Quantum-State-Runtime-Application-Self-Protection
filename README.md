# Q-RASP-Engine

[![CI](https://github.com/urrra39/Quantum-State-Runtime-Application-Self-Protection/actions/workflows/ci.yml/badge.svg)](https://github.com/urrra39/Quantum-State-Runtime-Application-Self-Protection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

**Runtime Application Self-Protection (RASP) for quantum state.**

Q-RASP-Engine — the *Adversarial Quantum Emulation Sandbox* — watches a quantum
system's density matrix `ρ` as a circuit executes and flags security-relevant
anomalies in real time. A Rust core continuously checks two physical invariants,
a shared policy layer decides whether an anomaly warrants intervention, and an
active-defense mechanism can roll the system back to the last known-good state.

This is a **research- and simulation-grade** engine. A "rollback" here is the
faithful restoration of a previously captured *simulator* state — a real,
demonstrable capability — **not** a claim about repairing physical QPU state in
hardware. That distinction is deliberate and preserved throughout the codebase.

---

## Threat model

The protected asset is the **integrity and confidentiality of a quantum state in
flight**. Two physical invariants of a density matrix expose the attacks we care
about:

- **Purity** `γ = Tr(ρ²)` ∈ `[1/d, 1]` — a pure state has `γ = 1`. Purity can
  only *fall* when the system couples to an environment. A drop is the signature
  of information leaking out (a side channel) or of decoherence being injected.
- **Trace** `Tr(ρ)` — must remain `≈ 1.0` for any legitimate, trace-preserving
  evolution. Deviation means the state was renormalized, cloned, truncated, or
  otherwise tampered with by a non-trace-preserving operation.

| Asset | Adversary capability | Observable | Core signal | Response |
|-------|----------------------|------------|-------------|----------|
| State confidentiality | Couple a subsystem to a hidden environment to siphon information (side channel) | Inter-step purity decreases beyond the configured tolerance | `purity_drop` | Escalate → rollback to last nominal snapshot |
| State integrity (decoherence attack) | Inject a noise channel (e.g. depolarizing) to corrupt the computation | Sharp purity drop between consecutive steps | `purity_drop` | Escalate → rollback |
| State integrity (tamper / clone) | Apply a non-trace-preserving op: unauthorized measurement, cloning, renormalization, truncation | `Tr(ρ)` departs from 1.0 beyond numerical tolerance | `trace_violation` | Escalate → rollback |
| Normal operation | — | Stable purity, unit trace | `nominal` | Record telemetry, continue |

**Detection is baseline-relative.** Each observation is compared against the
*previous* step rather than an absolute threshold, so legitimately mixed states
do not produce false positives — only an unexpected *change* trips the detector.
The purity-drop threshold is configurable (default: 1% inter-step loss).

**Trust boundary.** The Rust core and the in-process bridge are trusted: the core
emits the canonical anomaly label, never the adversary. At the HTTP boundary the
gateway re-validates every event against a strict `AnomalyKind` enum, so an
attacker who can reach the gateway cannot inject an unknown `kind` to suppress
escalation (a malformed event is rejected with HTTP 422). The escalation policy
is an **allow-list** of escalating kinds — unknown input fails to the safe,
non-escalating side and cannot *force* a self-inflicted rollback.

## Detection & mitigation

1. **Detect** — `crates/qrasp-core` (`StateObserver`) computes purity in `O(d²)`
   via the Hermitian identity `Tr(ρ²) = Σ|ρ_ij|²` and classifies each step as
   `nominal`, `purity_drop`, or `trace_violation`.
2. **Decide** — `qrasp.policies.EscalationPolicy` is the single source of truth
   for which anomaly kinds escalate. Both the local bridge and the gateway apply
   the *same* policy, so a circuit analyzed in-process and one analyzed over HTTP
   can never disagree about what counts as an attack.
3. **Respond** — on escalation, `StateGuard` restores the last known-nominal
   density matrix and the intervention is logged on the run's timeline. (The
   restored state is recorded as a defensive action but not re-injected into the
   trajectory; corrective re-execution / error-correction injection is a
   documented future research goal.)
4. **Harden** — gateway URLs are validated to `http(s)` hosts before use, so the
   reporting path cannot be coerced into a `file://` read; an unreachable gateway
   fails safe to the local policy.

## Architecture

The engine is split into four layers, each independently testable, with a
strictly one-directional (acyclic) dependency graph:

```
                      quantum program (Qiskit QuantumCircuit)
                                     │
   ┌─────────────────────────────────────────────────────────────────────┐
   │ MONITORS  (Python)  qrasp.monitors  ──►  qrasp.bridge.qiskit_adapter  │
   │   reconstruct ρ after every gate; stream snapshots into the core      │
   └─────────────────────────────────────────────────────────────────────┘
                                     │  ρ (NumPy complex128, d×d)
                                     ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ PyO3 BRIDGE  (Rust)   crates/qrasp-py  →  module qrasp.qrasp_native   │
   │   marshals NumPy complex arrays into ndarray; delegates to the core   │
   └─────────────────────────────────────────────────────────────────────┘
                                     │  Array2<Complex64>
                                     ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ CORE  (Rust)   crates/qrasp-core  (src/observer.rs)                   │
   │   StateObserver: O(d²) purity via the Hermitian identity, trace       │
   │   check; classifies Anomaly = Nominal | PurityDrop | TraceViolation   │
   └─────────────────────────────────────────────────────────────────────┘
                                     │  AnomalyEvent(kind, step, purity, …)
                                     ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ POLICIES  (Python)   qrasp.policies.escalation                        │
   │   EscalationPolicy: single source of truth for which anomaly kinds    │
   │   escalate to active defense. Used by BOTH the bridge and the gateway │
   └─────────────────────────────────────────────────────────────────────┘
                       │ escalate?                    ▲ same policy
                       ▼                              │
   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
   │ active defense (in-process)  │   │ GATEWAY  (Python, FastAPI)        │
   │ qrasp.bridge StateGuard      │   │ qrasp.gateway.app                 │
   │   roll back to last nominal  │   │   POST /v1/events  (ingest+decide)│
   │   snapshot; record rollback  │   │   GET  /v1/runs/{id}/events       │
   └──────────────────────────────┘   │   GET  /v1/runs/{id}/rollbacks    │
                                       │   GET  /health                    │
                                       └──────────────────────────────────┘
```

**Data flow.** A monitor (`QiskitObserverAdapter`, re-exported via
`qrasp.monitors`) instruments a `QuantumCircuit` by reconstructing the density
matrix after each gate — using either the Aer `density_matrix` backend or pure
`qiskit.quantum_info`, selected automatically. Each `ρ` crosses the PyO3 boundary
(`crates/qrasp-py`) into the Rust core, where `StateObserver` computes purity and
trace and returns a classification. The shared `EscalationPolicy` decides whether
to escalate; when it fires, the in-process `StateGuard` restores the last
known-nominal snapshot, and the gateway records the event and rollback on the
run's timeline.

## Quickstart

> **Prerequisite:** installing the package compiles the Rust core extension with
> [maturin](https://www.maturin.rs/), so you need a **Rust toolchain**
> ([rustup](https://rustup.rs/)) on your `PATH` in addition to Python 3.9+.

**1. Install (builds `qrasp.qrasp_native` via maturin):**

```bash
pip install -e ".[gateway,qiskit,dev]"
```

**2. Run the diagnostics & alert gateway:**

```bash
uvicorn qrasp.gateway.app:app --reload
# GET http://127.0.0.1:8000/health  ->  {"status": "ok"}
```

**3. Analyze a circuit for anomalies:**

```python
from qiskit import QuantumCircuit
from qrasp.monitors import QiskitObserverAdapter

# A noiseless Bell circuit: should stay pure (no anomalies).
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

adapter = QiskitObserverAdapter(purity_drop_threshold=0.01)
report = adapter.analyze(circuit, run_id="demo")

print(f"steps={report.num_steps}  anomalies={len(report.anomalies)}")
for a in report.anomalies:
    print(a.step, a.gate, a.kind, a.delta)   # purity_drop / trace_violation
```

A clean Bell circuit reports **no anomalies** (purity stays ~1.0 throughout). To
see a `purity_drop` fire, apply a depolarizing channel to the Bell state and feed
the resulting density matrices straight into the core observer — see
`tests/python/test_qiskit_adapter.py` for a backend-independent example using
`quantum_info` Kraus operators.

## Testing & CI

```bash
cargo test -p qrasp-core        # Rust core: unit + integration anomaly tests
pytest tests/python/ -v         # Python: policies, monitors, gateway, rollback
```

The Python tests skip automatically if Qiskit or the compiled native extension
are unavailable, so a pure-Python lint job never produces false failures.

CI runs on [GitHub Actions](./.github/workflows/ci.yml) on every push and PR to
`main`:

- **Rust** — `cargo test` (blocking), plus `rustfmt` and `clippy` (advisory).
- **Python** — `ruff`, `mypy` (strict), `bandit` (SAST), and `pytest` (all
  blocking) after a real maturin build of the native extension.
- **Audits** — `cargo-audit` and `pip-audit` for dependency CVEs (advisory).

## Project layout

```
crates/
  qrasp-core/        Rust: StateObserver, Anomaly classification (+ tests/)
  qrasp-py/          Rust: PyO3 bindings -> module qrasp.qrasp_native
python/qrasp/
  monitors/          runtime monitors (facade over the Qiskit bridge)
  bridge/            Qiskit adapter + StateGuard rollback mechanism
  policies/          EscalationPolicy: the active-defense decision layer
  gateway/           FastAPI diagnostics & alert gateway
tests/python/        policies, monitors, gateway, rollback, contract tests
.github/workflows/   CI: tests + security linters
```

## Scope & limitations

- **Simulation-grade.** Detection and rollback operate on simulated density
  matrices. Rollback restores a captured simulator state; it does not repair a
  physical QPU.
- **Per-gate reconstruction is O(n) simulations** for an n-gate circuit (each
  prefix is re-simulated), suitable for research-scale circuits, not large jobs.
- **Hermiticity is assumed** by the `O(d²)` purity formula. The Qiskit path
  always supplies Hermitian density matrices; arbitrary callers of the raw PyO3
  `observe()` are responsible for supplying a valid density matrix.
- **The gateway is a single-process diagnostics sink.** Its in-memory timelines
  are not durable and not coherent across multiple workers.

## License

[MIT](./LICENSE) © Q-RASP-Engine contributors
