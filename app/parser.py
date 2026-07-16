"""Groq-backed natural-language parser for the smart scheduler."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from .models import Schedule

logger = logging.getLogger("smart_scheduler.parser")


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = field(
        default_factory=lambda: os.getenv(
            "SMART_SCHEDULER_MODEL", "openai/gpt-oss-20b"
        )
    )
    api_key_env: str = "GROQ_API_KEY"
    max_tokens: int = 2048
    reasoning_effort: str = "low"
    max_retries: int = 1


SCHEDULE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "wake_up": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
        "sleep": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
        "fixed_events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "start": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
                    "end": {"type": "string", "pattern": r"^\d{2}:\d{2}$"},
                    "recurrence": {
                        "type": "string",
                        "enum": ["none", "daily", "weekdays", "weekly"],
                    },
                },
                "required": ["name", "start", "end", "recurrence"],
            },
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "duration_min": {"type": "integer", "minimum": 1, "maximum": 720},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                    "preferred_window": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "any"],
                    },
                },
                "required": ["name", "duration_min", "priority", "preferred_window"],
            },
        },
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "metadata": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "parse_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["parse_confidence", "warnings"],
        },
    },
    "required": [
        "wake_up",
        "sleep",
        "fixed_events",
        "tasks",
        "ambiguities",
        "metadata",
    ],
}


SYSTEM_PROMPT = """You extract a one-day schedule from natural language.

Interpret fixed events as commitments with explicit start times. Interpret flexible
work as tasks with durations. Use 24-hour HH:MM times. If wake or sleep is missing,
use a cautious default and describe the inference in ambiguities. Never invent an
event the user did not request. If two fixed events overlap, preserve both; the
planner will report the conflict. Return data matching the supplied JSON schema.
"""

RETRY_PROMPT = """Correct the schedule using the validation errors. Preserve the
user's stated times and tasks. Return data matching the supplied JSON schema."""


@dataclass
class ParseResult:
    schedule: Optional[Schedule]
    raw_json: str
    attempts: int
    errors: list[str] = field(default_factory=list)
    model: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def success(self) -> bool:
        return self.schedule is not None


class ScheduleParser:
    """Parse user text into a validated :class:`Schedule`."""

    def __init__(self, config: Optional[LLMConfig] = None, client: Any = None):
        self.cfg = config or LLMConfig()
        self._client = client

    def parse(self, user_input: str) -> ParseResult:
        started = time.perf_counter()
        user_text = user_input.strip()
        if not user_text:
            return self._failure(started, 0, ["Input text cannot be empty."])

        if self._client is None:
            api_key = os.getenv(self.cfg.api_key_env)
            if not api_key:
                return self._failure(
                    started,
                    0,
                    [f"Configuration error: {self.cfg.api_key_env} is not set."],
                )
            self._client = OpenAI(base_url=self.cfg.base_url, api_key=api_key)

        raw = ""
        errors_seen: list[str] = []
        prompt_tokens = completion_tokens = total_tokens = 0

        for attempt in range(1, self.cfg.max_retries + 2):
            if attempt == 1:
                system = SYSTEM_PROMPT
                user = user_text
            else:
                system = RETRY_PROMPT
                user = (
                    f"Original input:\n{user_text}\n\n"
                    f"Previous structured output:\n{raw}\n\n"
                    "Validation errors:\n"
                    + "\n".join(f"- {error}" for error in errors_seen[-5:])
                )

            try:
                raw, usage = self._call(system=system, user=user)
            except OpenAIError as exc:
                logger.warning("Groq request failed: %s", type(exc).__name__)
                return self._failure(
                    started,
                    attempt,
                    [f"Provider error: {type(exc).__name__}. Check the Groq key, quota, and network."],
                )
            except Exception as exc:  # protects the API boundary from SDK shape changes
                logger.exception("Unexpected parser failure")
                return self._failure(
                    started,
                    attempt,
                    [f"Unexpected provider response: {type(exc).__name__}."],
                )

            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]
            schedule, errors = self._validate(raw)
            if schedule is not None:
                return ParseResult(
                    schedule=schedule,
                    raw_json=raw,
                    attempts=attempt,
                    model=self.cfg.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            errors_seen.extend(errors)

        return self._failure(
            started,
            self.cfg.max_retries + 1,
            errors_seen,
            raw=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def _call(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        response = self._client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            reasoning_effort=self.cfg.reasoning_effort,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "daily_schedule",
                    "strict": True,
                    "schema": SCHEDULE_JSON_SCHEMA,
                },
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return content.strip(), {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _failure(
        self,
        started: float,
        attempts: int,
        errors: list[str],
        *,
        raw: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> ParseResult:
        return ParseResult(
            schedule=None,
            raw_json=raw,
            attempts=attempts,
            errors=errors,
            model=self.cfg.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _validate(raw: str) -> tuple[Optional[Schedule], list[str]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, [f"Invalid JSON: {exc}"]

        try:
            schedule = Schedule.model_validate(data)
        except ValidationError as exc:
            return None, [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]

        semantic_errors = _semantic_checks(schedule)
        if semantic_errors:
            return None, semantic_errors
        return schedule, []


def _semantic_checks(schedule: Schedule) -> list[str]:
    errors: list[str] = []
    wake = _to_min(schedule.wake_up)
    sleep = _to_min(schedule.sleep)
    if sleep <= wake:
        errors.append(f"sleep ({schedule.sleep}) must be after wake_up ({schedule.wake_up})")

    for event in schedule.fixed_events:
        start, end = _to_min(event.start), _to_min(event.end)
        if end <= start:
            errors.append(f"'{event.name}': end ({event.end}) not after start ({event.start})")
        if start < wake:
            errors.append(f"'{event.name}' starts before wake_up ({schedule.wake_up})")
        if end > sleep:
            errors.append(f"'{event.name}' ends after sleep ({schedule.sleep})")
    return errors


def _to_min(value: str) -> int:
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute
