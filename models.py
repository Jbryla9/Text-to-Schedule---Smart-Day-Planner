"""
models.py — Phase 1: Contract / JSON Schema
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re

TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def _validate_time(v: str) -> str:
    if not TIME_RE.match(v):
        raise ValueError(f"Time must be HH:MM (24h), got: {v!r}")
    h, m = map(int, v.split(":"))
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid time value: {v!r}")
    return v


class FixedEvent(BaseModel):
    """A time-anchored event — immovable by the planner."""
    name: str = Field(..., min_length=1)
    start: str = Field(..., description="HH:MM 24h")
    end: str = Field(..., description="HH:MM 24h")
    recurrence: Literal["none", "daily", "weekdays", "weekly"] = "none"

    @field_validator("start", "end")
    @classmethod
    def validate_time(cls, v: str) -> str:
        return _validate_time(v)

    @property
    def duration_min(self) -> int:
        sh, sm = map(int, self.start.split(":"))
        eh, em = map(int, self.end.split(":"))
        return (eh * 60 + em) - (sh * 60 + sm)


class Task(BaseModel):
    """A flexible, duration-based item — scheduled by the greedy planner."""
    name: str = Field(..., min_length=1)
    duration_min: int = Field(..., gt=0, le=720)
    priority: Literal["low", "medium", "high"]
    preferred_window: Literal["morning", "afternoon", "evening", "any"] = "any"


class ParseMetadata(BaseModel):
    parse_confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class Schedule(BaseModel):
    """Root contract object — the output of the LLM parser."""
    wake_up: str = Field(..., description="HH:MM 24h")
    sleep: str = Field(..., description="HH:MM 24h")
    fixed_events: list[FixedEvent] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    metadata: Optional[ParseMetadata] = None

    @field_validator("wake_up", "sleep")
    @classmethod
    def validate_time(cls, v: str) -> str:
        return _validate_time(v)
