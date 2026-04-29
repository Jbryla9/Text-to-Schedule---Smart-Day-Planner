"""
parser.py — Phase 2: LLM Parser
Converts a chaotic user query → validated Schedule object.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

from models import Schedule


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.1-8b-instant"
    api_key_env: str = "GROQ_API_KEY"
    max_tokens: int = 1024
    temperature: float = 0.0
    max_retries: int = 2


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scheduling data extractor. Convert natural language into JSON.

CRITICAL: Return ONLY a raw JSON object — no markdown, no backticks, no explanation.

REQUIRED JSON SCHEMA:
{
  "wake_up": "HH:MM",
  "sleep": "HH:MM",
  "fixed_events": [
    {
      "name": "string",
      "start": "HH:MM",
      "end": "HH:MM",
      "recurrence": "none|daily|weekdays|weekly"
    }
  ],
  "tasks": [
    {
      "name": "string",
      "duration_min": integer,
      "priority": "low|medium|high",
      "preferred_window": "morning|afternoon|evening|any"
    }
  ],
  "ambiguities": ["string"],
  "metadata": {
    "parse_confidence": float,
    "warnings": ["string"]
  }
}

RULES:
1. All times → HH:MM 24-hour format
2. Missing end time → start + 60 min
3. "1h"→60, "1.5h"→90, "30min"→30, "45 minutes"→45
4. No wake_up stated → infer from first event or use "07:00"
5. No sleep stated → infer from last event or use "23:00"
6. fixed_event = explicit or strongly implied start time
   task = duration only, no anchor time
7. Priority:
   high   = meeting, standup, appointment, doctor, kids, school, commute
   medium = gym, exercise, cook, shopping, project, work
   low    = read, walk, relax, hobby, generic errands
8. preferred_window: <12:00=morning, 12-17=afternoon, >=17:00=evening, unknown=any
9. Uncertain inference → add string to ambiguities[]
10. parse_confidence: 1.0=all explicit, 0.5=many inferences, 0.0=barely any data"""


RETRY_SYSTEM_PROMPT = """You are a JSON fixer. The previous output had validation errors.
Return ONLY the corrected raw JSON object — no markdown, no explanation."""


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    schedule: Optional[Schedule]
    raw_json: str
    attempts: int
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.schedule is not None


# ─── Parser ───────────────────────────────────────────────────────────────────

class ScheduleParser:
    """
    Phase 2 LLM Parser using Groq (llama-3.1-8b-instant).

    - Uses `response_format={"type": "json_object"}` (JSON mode)
      guarantees syntactically valid JSON, eliminating the most common failure class.
    - temperature=0.0 for deterministic extraction.
    - Retry messages use a separate tighter system prompt to save tokens.

    Usage:
        parser = ScheduleParser()
        result = parser.parse("wake 7, gym 1h, meeting 3pm, sleep 23")
        if result.success:
            print(result.schedule.model_dump_json(indent=2))
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.cfg = config or LLMConfig()
        self.client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=os.environ.get(self.cfg.api_key_env, ""),
        )

    # ── public ────────────────────────────────────────────────────────────────

    def parse(self, user_input: str) -> ParseResult:
        """
        Parse raw user text → validated Schedule.
        - Attempt 1: standard extraction prompt + JSON mode
        - Attempt 2+: error feedback loop (retry with specific errors)
        """
        errors_seen: list[str] = []
        raw = ""
        user_text = user_input.strip()

        for attempt in range(1, self.cfg.max_retries + 2):
            if attempt == 1:
                raw = self._call(
                    system=SYSTEM_PROMPT,
                    user=user_text,
                )
            else:
                raw = self._call(
                    system=RETRY_SYSTEM_PROMPT,
                    user=(
                        f"Original input:\n\"\"\"\n{user_text}\n\"\"\"\n\n"
                        f"Your previous output:\n{raw}\n\n"
                        f"Validation errors to fix:\n"
                        + "\n".join(f"- {e}" for e in errors_seen[-5:])
                    ),
                )

            schedule, errors = self._validate(raw)

            if schedule is not None:
                return ParseResult(
                    schedule=schedule,
                    raw_json=raw,
                    attempts=attempt,
                )

            errors_seen.extend(errors)

        return ParseResult(
            schedule=None,
            raw_json=raw,
            attempts=self.cfg.max_retries + 1,
            errors=errors_seen,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _call(self, system: str, user: str) -> str:
        """
        Single LLM call with JSON mode enabled.
        """
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            response_format={"type": "json_object"},   # ← JSON mode: key difference
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _validate(raw: str) -> tuple[Optional[Schedule], list[str]]:
        """
        Two-stage validation:
        1. JSON parse
        2. Pydantic schema + semantic checks
        """
        clean = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            return None, [f"Invalid JSON: {e}"]

        try:
            schedule = Schedule.model_validate(data)
        except ValidationError as e:
            return None, [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                          for err in e.errors()]

        semantic_errors = _semantic_checks(schedule)
        if semantic_errors:
            return None, semantic_errors

        return schedule, []


# ─── Semantic checks ──────────────────────────────────────────────────────────

def _semantic_checks(s: Schedule) -> list[str]:
    """Checks Pydantic can't express: time ordering, event bounds."""
    errors = []
    wake = _to_min(s.wake_up)
    sleep = _to_min(s.sleep)

    if sleep <= wake:
        errors.append(f"sleep ({s.sleep}) must be after wake_up ({s.wake_up})")

    for ev in s.fixed_events:
        start, end = _to_min(ev.start), _to_min(ev.end)
        if end <= start:
            errors.append(f"'{ev.name}': end ({ev.end}) not after start ({ev.start})")
        if start < wake:
            errors.append(f"'{ev.name}' starts before wake_up ({s.wake_up})")
        if end > sleep:
            errors.append(f"'{ev.name}' ends after sleep ({s.sleep})")

    return errors


def _to_min(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m
