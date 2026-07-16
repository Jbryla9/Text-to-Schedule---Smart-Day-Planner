from dataclasses import asdict

from fastapi.testclient import TestClient

from app import main as main_module
from app.models import Schedule
from app.parser import ParseResult
from app.pipeline import PipelineResult
from app.planner import DayPlan


def valid_result() -> PipelineResult:
    schedule = Schedule(
        wake_up="07:00",
        sleep="23:00",
        fixed_events=[],
        tasks=[],
        ambiguities=[],
        metadata=None,
    )
    parsed = ParseResult(
        schedule=schedule,
        raw_json="{}",
        attempts=1,
        model="openai/gpt-oss-20b",
        latency_ms=50,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )
    return PipelineResult(
        parse=parsed,
        plan=DayPlan(wake_up="07:00", sleep="23:00"),
        success=True,
        run_id="test-run",
    )


def test_health_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_SCHEDULER_LOG_DIR", str(tmp_path))
    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_schedule_endpoint_returns_structured_result(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_SCHEDULER_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "run", lambda _: valid_result())
    with TestClient(main_module.app) as client:
        response = client.post("/schedule", json={"text": "plan my day"})

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "test-run"
    assert body["token_usage"]["total"] == 30
    assert body["schedule"]["wake_up"] == "07:00"
    assert body["plan"] == asdict(valid_result().plan)


def test_schedule_endpoint_requires_nonempty_text(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_SCHEDULER_LOG_DIR", str(tmp_path))
    with TestClient(main_module.app) as client:
        response = client.post("/schedule", json={"text": ""})

    assert response.status_code == 422


def test_schedule_endpoint_reports_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_SCHEDULER_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with TestClient(main_module.app) as client:
        response = client.post("/schedule", json={"text": "plan my day"})

    assert response.status_code == 503
    assert "GROQ_API_KEY is not set" in response.json()["detail"]["errors"][0]

