"""FastAPI diagnostics & security alert gateway for Q-RASP-Engine.

The gateway is a thin telemetry + alerting sink. The simulator bridge runs the
Rust observer per gate and POSTs anomaly classifications here; the gateway logs
them per run and decides whether to recommend an active-defense escalation
(e.g. error-correction gate injection or state rollback).

Run locally:
    pip install -e ".[gateway]"
    uvicorn qrasp.gateway.app:app --reload
"""
from collections import defaultdict
from typing import Dict, List

from fastapi import FastAPI

from qrasp.policies import EscalationPolicy

from .schemas import AlertResponse, AnomalyEvent, RollbackRecord

app = FastAPI(
    title="Q-RASP Gateway",
    description=(
        "Diagnostics & security alert gateway for the Adversarial Quantum "
        "Emulation Sandbox."
    ),
    version="0.1.0",
)

# In-memory logs, keyed by run_id. Swap for a durable store later.
_EVENT_LOG: Dict[str, List[AnomalyEvent]] = defaultdict(list)
_ROLLBACK_LOG: Dict[str, List[RollbackRecord]] = defaultdict(list)

# The escalation decision lives in the shared policy layer so the gateway and
# the simulator-side bridge cannot disagree about what counts as an attack.
_POLICY = EscalationPolicy()


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/v1/events", response_model=AlertResponse)
def ingest_event(event: AnomalyEvent) -> AlertResponse:
    """Ingest one anomaly classification from the bridge and decide whether to
    escalate to active defense (e.g. rollback / error-correction)."""
    _EVENT_LOG[event.run_id].append(event)

    decision = _POLICY.decide(event.kind)
    if decision.escalate:
        # Record the defensive intervention so it can be exposed alongside the
        # anomaly timeline. This closes the detection -> defense telemetry loop.
        _ROLLBACK_LOG[event.run_id].append(
            RollbackRecord(
                run_id=event.run_id,
                step=event.step,
                triggered_by=event.kind,
                purity=event.purity,
            )
        )
    return AlertResponse(
        accepted=True, escalated=decision.escalate, message=decision.reason
    )


@app.get("/v1/runs/{run_id}/events", response_model=List[AnomalyEvent])
def get_run_events(run_id: str) -> List[AnomalyEvent]:
    """Return the recorded anomaly timeline for a given circuit run."""
    return _EVENT_LOG.get(run_id, [])


@app.get("/v1/runs/{run_id}/rollbacks", response_model=List[RollbackRecord])
def get_run_rollbacks(run_id: str) -> List[RollbackRecord]:
    """Return the history of defensive rollback interventions for a run."""
    return _ROLLBACK_LOG.get(run_id, [])
