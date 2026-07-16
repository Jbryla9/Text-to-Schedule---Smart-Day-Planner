import json
from types import SimpleNamespace

from openai import OpenAIError

from app.parser import ScheduleParser


def schedule_json(*, wake: str = "07:00", sleep: str = "23:00") -> str:
    return json.dumps({
        "wake_up": wake,
        "sleep": sleep,
        "fixed_events": [{
            "name": "Standup",
            "start": "10:00",
            "end": "10:30",
            "recurrence": "weekdays",
        }],
        "tasks": [{
            "name": "Deep work",
            "duration_min": 120,
            "priority": "high",
            "preferred_window": "morning",
        }],
        "ambiguities": [],
        "metadata": {"parse_confidence": 0.95, "warnings": []},
    })


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))],
            usage=SimpleNamespace(
                prompt_tokens=40,
                completion_tokens=60,
                total_tokens=100,
            ),
        )


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def test_parser_uses_gpt_oss_strict_schema_and_collects_usage():
    client = FakeClient([schedule_json()])
    result = ScheduleParser(client=client).parse("standup and deep work")

    assert result.success
    assert result.model == "openai/gpt-oss-20b"
    assert result.total_tokens == 100
    call = client.completions.calls[0]
    assert call["reasoning_effort"] == "low"
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True


def test_parser_retries_semantic_error_then_succeeds():
    client = FakeClient([
        schedule_json(wake="23:00", sleep="07:00"),
        schedule_json(),
    ])
    result = ScheduleParser(client=client).parse("my day")

    assert result.success
    assert result.attempts == 2
    assert result.total_tokens == 200


def test_parser_returns_configuration_error_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = ScheduleParser().parse("plan my day")

    assert not result.success
    assert "GROQ_API_KEY is not set" in result.errors[0]


def test_parser_converts_provider_error_to_failed_result():
    client = FakeClient([OpenAIError("provider unavailable")])
    result = ScheduleParser(client=client).parse("plan my day")

    assert not result.success
    assert result.errors[0].startswith("Provider error")


def test_parser_rejects_empty_input_without_provider_call():
    client = FakeClient([schedule_json()])
    result = ScheduleParser(client=client).parse("   ")

    assert not result.success
    assert result.attempts == 0
    assert client.completions.calls == []

