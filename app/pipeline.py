"""End-to-end text-to-schedule pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from .monitoring import record_event
from .parser import ParseResult, ScheduleParser
from .planner import DayPlan, GreedyPlanner


@dataclass
class PipelineResult:
    parse: ParseResult
    plan: Optional[DayPlan]
    success: bool
    run_id: str

    def summary(self) -> str:
        if not self.success or self.plan is None:
            return f"Pipeline failed at parse step:\n  {self.parse.errors}"
        return self.plan.summary()


def run(
    user_input: str,
    *,
    parser: Optional[ScheduleParser] = None,
    planner: Optional[GreedyPlanner] = None,
    monitor: bool = True,
) -> PipelineResult:
    run_id = str(uuid4())
    parse_result = (parser or ScheduleParser()).parse(user_input)

    if not parse_result.success or parse_result.schedule is None:
        result = PipelineResult(parse_result, None, False, run_id)
        if monitor:
            _record(result)
        return result

    plan = (planner or GreedyPlanner()).plan(parse_result.schedule)
    result = PipelineResult(parse_result, plan, True, run_id)
    if monitor:
        _record(result)
    return result


def _record(result: PipelineResult) -> None:
    schedule = result.parse.schedule
    plan = result.plan
    record_event(
        "schedule_run",
        {
            "run_id": result.run_id,
            "success": result.success,
            "model": result.parse.model,
            "latency_ms": result.parse.latency_ms,
            "attempts": result.parse.attempts,
            "prompt_tokens": result.parse.prompt_tokens,
            "completion_tokens": result.parse.completion_tokens,
            "total_tokens": result.parse.total_tokens,
            "validation_error_count": len(result.parse.errors),
            "fixed_event_count": len(schedule.fixed_events) if schedule else 0,
            "task_count": len(schedule.tasks) if schedule else 0,
            "conflict_count": len(plan.conflicts) if plan else 0,
            "unscheduled_count": len(plan.unscheduled) if plan else 0,
        },
    )

