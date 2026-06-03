"""Pydantic schemas for the Q-RASP diagnostics & alert gateway.

These mirror the classification produced by the Rust core ``Anomaly`` enum so
that anomaly events emitted by the simulator bridge serialize cleanly across
the HTTP boundary.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AnomalyKind(str, Enum):
    """Classification labels mirroring qrasp_core::observer::Anomaly."""

    NOMINAL = "nominal"
    PURITY_DROP = "purity_drop"
    TRACE_VIOLATION = "trace_violation"


class AnomalyEvent(BaseModel):
    """A single classification emitted by the core observer at one step."""

    run_id: str = Field(..., description="Logical id for one circuit execution")
    step: int = Field(..., ge=0, description="Gate index / logical timestep")
    kind: AnomalyKind
    # Upper bound carries a small tolerance: a reconstructed density matrix can
    # yield purity marginally above 1.0 due to floating-point error.
    purity: float = Field(..., ge=0.0, le=1.0 + 1e-6)
    trace: float
    delta: Optional[float] = Field(
        None, description="Purity drop magnitude between steps, if any"
    )


class AlertResponse(BaseModel):
    """Gateway response to an ingested anomaly event."""

    accepted: bool
    escalated: bool = Field(
        ..., description="True if an active-defense trigger is recommended"
    )
    message: str


class RollbackRecord(BaseModel):
    """A recorded defensive intervention: the gateway recommended escalation
    for an anomaly, so a state rollback was triggered for this step."""

    run_id: str = Field(..., description="Logical id for one circuit execution")
    step: int = Field(..., ge=0, description="Step at which rollback fired")
    triggered_by: AnomalyKind = Field(
        ..., description="The anomaly kind that triggered the rollback"
    )
    purity: float = Field(..., ge=0.0, le=1.0 + 1e-6)
