"""
test_planner.py — Phase 4: Greedy Planner Unit Tests

These tests are fully deterministic — no API calls, no LLM.
They test the planner directly by feeding it hand-crafted Schedule objects.

Usage:
    python test_planner.py
    python test_planner.py -v    # verbose
    python -m pytest test_planner.py -v
"""

from __future__ import annotations
import sys
import traceback
from dataclasses import dataclass

from models import Schedule, FixedEvent, Task, ParseMetadata
from planner import GreedyPlanner, SlotKind, DayPlan


# ─── Test harness ─────────────────────────────────────────────────────────────

@dataclass
class Case:
    id: int
    description: str
    schedule: Schedule
    assertions: list[tuple[str, callable]]   # (label, fn(DayPlan) -> bool)


PLANNER = GreedyPlanner()
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


def _run(cases: list[Case]) -> None:
    passed = failed = 0

    print(f"\n{'─'*60}")
    print(f"  Greedy Planner · Unit Test Suite ({len(cases)} cases)")
    print(f"{'─'*60}\n")

    for c in cases:
        plan = PLANNER.plan(c.schedule)
        case_pass = True
        fail_msgs = []

        for label, fn in c.assertions:
            try:
                ok = fn(plan)
            except Exception as e:
                ok = False
                fail_msgs.append(f"{label}: exception — {e}")
                if VERBOSE:
                    traceback.print_exc()
            if not ok:
                case_pass = False
                if label not in fail_msgs:
                    fail_msgs.append(label)

        status = "PASS" if case_pass else "FAIL"
        print(f"  [{c.id:02d}] {c.description}  →  {status}")

        if VERBOSE or not case_pass:
            if VERBOSE:
                print(plan.summary())
            if fail_msgs:
                for m in fail_msgs:
                    print(f"         ✗ {m}")
            print()

        if case_pass:
            passed += 1
        else:
            failed += 1

    print(f"{'─'*60}")
    print(f"  {passed}/{passed+failed} passed  ({100*passed//(passed+failed)}% accuracy)")
    print(f"{'─'*60}\n")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _slot_names(plan: DayPlan) -> list[str]:
    return [s.name for s in plan.slots if s.kind != SlotKind.FREE]


def _task_slots(plan: DayPlan) -> list:
    return [s for s in plan.slots if s.kind == SlotKind.TASK]


def _fixed_slots(plan: DayPlan) -> list:
    return [s for s in plan.slots if s.kind == SlotKind.FIXED_EVENT]


def _start_min(slot) -> int:
    h, m = map(int, slot.start.split(":"))
    return h * 60 + m


def _no_overlap(plan: DayPlan) -> bool:
    """Verify no two non-free slots overlap."""
    active = [s for s in plan.slots if s.kind != SlotKind.FREE]
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            a_end = int(a.end[:2]) * 60 + int(a.end[3:])
            b_start = int(b.start[:2]) * 60 + int(b.start[3:])
            a_start = int(a.start[:2]) * 60 + int(a.start[3:])
            b_end = int(b.end[:2]) * 60 + int(b.end[3:])
            if not (a_end <= b_start or b_end <= a_start):
                return False
    return True


# ─── Test cases ───────────────────────────────────────────────────────────────

CASES: list[Case] = [

    Case(
        id=1,
        description="High priority task placed before low — both fit",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[],
            tasks=[
                Task(name="Read book",   duration_min=30, priority="low",    preferred_window="any"),
                Task(name="Deep work",   duration_min=90, priority="high",   preferred_window="any"),
            ],
        ),
        assertions=[
            ("deep work placed before read book",
             lambda p: _start_min(_task_slots(p)[0]) <= _start_min(_task_slots(p)[1])
             if len(_task_slots(p)) == 2 else False),
            ("no unscheduled",           lambda p: len(p.unscheduled) == 0),
            ("no overlap",               _no_overlap),
        ],
    ),

    Case(
        id=2,
        description="30-min morning buffer respected",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[],
            tasks=[Task(name="Gym", duration_min=60, priority="high", preferred_window="any")],
        ),
        assertions=[
            ("gym starts at 07:30 or later",
             lambda p: _start_min(_task_slots(p)[0]) >= 7 * 60 + 30
             if _task_slots(p) else False),
            ("no unscheduled", lambda p: len(p.unscheduled) == 0),
        ],
    ),

    Case(
        id=3,
        description="Fixed events blocked — task fills gap correctly",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[
                FixedEvent(name="School run", start="08:00", end="09:00"),
                FixedEvent(name="Lunch",      start="12:00", end="13:00"),
            ],
            tasks=[Task(name="Work",  duration_min=120, priority="high", preferred_window="any")],
        ),
        assertions=[
            ("work placed in a free gap",
             lambda p: all(
                 not (
                     _start_min(s) < 9 * 60 and  # overlaps school
                     _start_min(s) >= 8 * 60
                 )
                 for s in _task_slots(p)
             )),
            ("no overlap",      _no_overlap),
            ("no unscheduled",  lambda p: len(p.unscheduled) == 0),
        ],
    ),

    Case(
        id=4,
        description="Task that doesn't fit → unscheduled with reason",
        schedule=Schedule(
            wake_up="07:00", sleep="08:00",   # only 1h active day
            fixed_events=[],
            tasks=[Task(name="Long project", duration_min=120, priority="high", preferred_window="any")],
        ),
        assertions=[
            ("task is unscheduled",       lambda p: len(p.unscheduled) == 1),
            ("reason is not empty",       lambda p: len(p.unscheduled[0].reason) > 0),
            ("no task slots placed",      lambda p: len(_task_slots(p)) == 0),
        ],
    ),

    Case(
        id=5,
        description="Priority order: high → medium → low all placed",
        schedule=Schedule(
            wake_up="06:00", sleep="23:00",
            fixed_events=[],
            tasks=[
                Task(name="Relax",    duration_min=30,  priority="low",    preferred_window="any"),
                Task(name="Shopping", duration_min=60,  priority="medium", preferred_window="any"),
                Task(name="Meeting",  duration_min=60,  priority="high",   preferred_window="any"),
            ],
        ),
        assertions=[
            ("all 3 tasks placed",  lambda p: len(_task_slots(p)) == 3),
            ("no unscheduled",      lambda p: len(p.unscheduled) == 0),
            ("meeting earliest",
             lambda p: _start_min(next(s for s in _task_slots(p) if s.name == "Meeting"))
             < _start_min(next(s for s in _task_slots(p) if s.name == "Shopping"))
             < _start_min(next(s for s in _task_slots(p) if s.name == "Relax"))),
            ("no overlap", _no_overlap),
        ],
    ),

    Case(
        id=6,
        description="Preferred window respected (morning task placed before noon)",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[],
            tasks=[Task(name="Run", duration_min=45, priority="medium", preferred_window="morning")],
        ),
        assertions=[
            ("run ends before 12:00",
             lambda p: int(_task_slots(p)[0].end[:2]) < 12
             if _task_slots(p) else False),
            ("no unscheduled", lambda p: len(p.unscheduled) == 0),
        ],
    ),

    Case(
        id=7,
        description="Preferred window fallback — morning blocked, task placed in afternoon",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[
                # Block entire morning
                FixedEvent(name="Morning block", start="07:00", end="12:00"),
            ],
            tasks=[Task(name="Gym", duration_min=60, priority="medium", preferred_window="morning")],
        ),
        assertions=[
            ("gym placed (fallback)",  lambda p: len(_task_slots(p)) == 1),
            ("gym starts at 12:00+",
             lambda p: _start_min(_task_slots(p)[0]) >= 12 * 60
             if _task_slots(p) else False),
            ("no overlap",             _no_overlap),
        ],
    ),

    Case(
        id=8,
        description="Dense day — multiple fixed events + tasks, no overlap",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[
                FixedEvent(name="School drop", start="08:00", end="09:00"),
                FixedEvent(name="Standup",     start="10:00", end="10:30"),
                FixedEvent(name="Lunch",       start="12:30", end="13:30"),
                FixedEvent(name="School pick", start="16:00", end="16:30"),
            ],
            tasks=[
                Task(name="Deep work",  duration_min=90, priority="high",   preferred_window="morning"),
                Task(name="Errands",    duration_min=60, priority="medium",  preferred_window="afternoon"),
                Task(name="Reading",    duration_min=30, priority="low",     preferred_window="evening"),
            ],
        ),
        assertions=[
            ("all tasks placed",   lambda p: len(_task_slots(p)) == 3),
            ("no unscheduled",     lambda p: len(p.unscheduled) == 0),
            ("no overlap",         _no_overlap),
            ("fixed events present", lambda p: len(_fixed_slots(p)) == 4),
        ],
    ),

    Case(
        id=9,
        description="Shorter high-priority task before longer same-priority task",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[],
            tasks=[
                Task(name="Long high",  duration_min=120, priority="high", preferred_window="any"),
                Task(name="Short high", duration_min=30,  priority="high", preferred_window="any"),
            ],
        ),
        assertions=[
            ("both placed",       lambda p: len(_task_slots(p)) == 2),
            ("shorter placed first",
             lambda p: _start_min(next(s for s in _task_slots(p) if s.name == "Short high"))
             < _start_min(next(s for s in _task_slots(p) if s.name == "Long high"))),
            ("no overlap",        _no_overlap),
        ],
    ),

    Case(
        id=10,
        description="Free gaps are generated between placed slots",
        schedule=Schedule(
            wake_up="07:00", sleep="22:00",
            fixed_events=[FixedEvent(name="Meeting", start="10:00", end="11:00")],
            tasks=[Task(name="Gym", duration_min=60, priority="high", preferred_window="any")],
        ),
        assertions=[
            ("at least one free gap",
             lambda p: any(s.kind == SlotKind.FREE for s in p.slots)),
            ("no overlap", _no_overlap),
        ],
    ),
]


if __name__ == "__main__":
    _run(CASES)
