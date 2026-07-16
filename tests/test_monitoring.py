import json

from app.monitoring import record_event


def test_jsonl_monitoring_writes_operational_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_SCHEDULER_LOG_DIR", str(tmp_path))
    record_event("schedule_run", {"success": True, "latency_ms": 25})

    record = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert record["event_type"] == "schedule_run"
    assert record["payload"] == {"success": True, "latency_ms": 25}

