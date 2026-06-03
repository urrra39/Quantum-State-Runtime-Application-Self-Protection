"""Tests for the Q-RASP FastAPI diagnostics & alert gateway."""
from fastapi.testclient import TestClient

from qrasp.gateway import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_nominal_event_is_not_escalated() -> None:
    payload = {
        "run_id": "run-1",
        "step": 0,
        "kind": "nominal",
        "purity": 1.0,
        "trace": 1.0,
    }
    resp = client.post("/v1/events", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
    assert body["escalated"] is False


def test_purity_drop_is_escalated() -> None:
    payload = {
        "run_id": "run-2",
        "step": 5,
        "kind": "purity_drop",
        "purity": 0.5,
        "trace": 1.0,
        "delta": 0.5,
    }
    resp = client.post("/v1/events", json=payload)
    assert resp.status_code == 200
    assert resp.json()["escalated"] is True


def test_run_event_timeline_is_recorded() -> None:
    payload = {
        "run_id": "run-3",
        "step": 0,
        "kind": "nominal",
        "purity": 1.0,
        "trace": 1.0,
    }
    client.post("/v1/events", json=payload)
    resp = client.get("/v1/runs/run-3/events")
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) == 1
    assert events[0]["run_id"] == "run-3"


def test_escalating_event_records_a_rollback() -> None:
    payload = {
        "run_id": "run-4",
        "step": 7,
        "kind": "purity_drop",
        "purity": 0.5,
        "trace": 1.0,
        "delta": 0.5,
    }
    client.post("/v1/events", json=payload)
    resp = client.get("/v1/runs/run-4/rollbacks")
    assert resp.status_code == 200
    rollbacks = resp.json()
    assert len(rollbacks) == 1
    assert rollbacks[0]["step"] == 7
    assert rollbacks[0]["triggered_by"] == "purity_drop"


def test_nominal_event_records_no_rollback() -> None:
    payload = {
        "run_id": "run-5",
        "step": 0,
        "kind": "nominal",
        "purity": 1.0,
        "trace": 1.0,
    }
    client.post("/v1/events", json=payload)
    resp = client.get("/v1/runs/run-5/rollbacks")
    assert resp.status_code == 200
    assert resp.json() == []
