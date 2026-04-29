"""
pipeline.py — End-to-end pipeline: raw text → DayPlan

Wires Phase 2 (parser) + Phase 4 (planner) together.

Usage:
    from pipeline import run
    plan = run("wake 7, gym 1h, meeting 3pm, sleep 23")
    print(plan.summary())
"""

from __future__ import annotations
from dataclasses import dataclass

from parser import ScheduleParser, ParseResult
from planner import GreedyPlanner, DayPlan


@dataclass
class PipelineResult:
    parse:  ParseResult
    plan:   DayPlan | None
    success: bool

    def summary(self) -> str:
        if not self.success:
            return f"Pipeline failed at parse step:\n  {self.parse.errors}"
        return self.plan.summary()


_parser  = ScheduleParser()
_planner = GreedyPlanner()


def run(user_input: str) -> PipelineResult:
    parse_result = _parser.parse(user_input)
    if not parse_result.success:
        return PipelineResult(parse=parse_result, plan=None, success=False)
    plan = _planner.plan(parse_result.schedule)
    return PipelineResult(parse=parse_result, plan=plan, success=True)
