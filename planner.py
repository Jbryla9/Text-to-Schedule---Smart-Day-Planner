"""
planner.py — Phase 4: Greedy Day Planner

Takes a validated Schedule (output of parser.py) and returns a complete
DayPlan: every minute of the day accounted for as either a placed event,
a placed task, a free gap, or an unscheduled task with a reason.

Algorithm
---------
1. Build a minute-resolution timeline [wake_up_min, sleep_min).
2. Mark all fixed_event slots as BLOCKED.
3. Sort tasks by priority (high→medium→low), then duration (shorter first
   within same priority — maximises the number of tasks that fit).
4. For each task:
   a. Scan from (wake_up + 30 min) forward in 1-minute steps.
   b. Find first contiguous free window >= task.duration_min that also
      respects the task's preferred_window hint (soft constraint — if no
      window found within the hint, fall back to any free window).
   c. Place the task; mark those minutes BLOCKED.
   d. If no window found anywhere → add to unscheduled[] with reason.
5. Collect all contiguous free gaps and label them.
6. Return DayPlan sorted by start time.

Design notes
------------
- No LLM, no randomness, fully deterministic and unit-testable.
- preferred_window is a *soft* constraint: the planner tries it first,
  then falls back to any free slot. This is intentional — a task that
  prefers "morning" should still be placed in the afternoon rather than
  dropped entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models import Schedule, FixedEvent, Task


# ─── Constants ────────────────────────────────────────────────────────────────

MORNING_BUFFER_MIN: int = 30   # buffer after wake_up before first task
PRIORITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}
WINDOW_RANGES: dict[str, tuple[int, int]] = {
    "morning":   (0,    719),   # 00:00–11:59
    "afternoon": (720,  1019),  # 12:00–16:59
    "evening":   (1020, 1439),  # 17:00–23:59
    "any":       (0,    1439),
}


# ─── Output models ────────────────────────────────────────────────────────────

class SlotKind(str, Enum):
    FIXED_EVENT  = "fixed_event"
    TASK         = "task"
    FREE         = "free"


@dataclass
class Slot:
    kind:     SlotKind
    name:     str
    start:    str          # HH:MM
    end:      str          # HH:MM
    priority: Optional[str] = None   # only for tasks
    note:     Optional[str] = None   # e.g. "preferred morning → placed afternoon"


@dataclass
class UnscheduledTask:
    name:        str
    duration_min: int
    priority:    str
    reason:      str


@dataclass
class DayPlan:
    wake_up:     str
    sleep:       str
    slots:       list[Slot]             = field(default_factory=list)
    unscheduled: list[UnscheduledTask]  = field(default_factory=list)

    # ── convenience ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Day Plan  {self.wake_up} → {self.sleep}",
            f"{'─'*48}",
        ]
        for s in self.slots:
            tag = f"[{s.kind.value:11s}]"
            pri = f" ({s.priority})" if s.priority else ""
            note = f"  ← {s.note}" if s.note else ""
            lines.append(f"  {s.start}–{s.end}  {tag}  {s.name}{pri}{note}")
        if self.unscheduled:
            lines.append(f"{'─'*48}")
            lines.append("  Unscheduled:")
            for u in self.unscheduled:
                lines.append(f"    ✗ {u.name} ({u.duration_min}min, {u.priority}): {u.reason}")
        lines.append(f"{'─'*48}")
        placed = sum(1 for s in self.slots if s.kind == SlotKind.TASK)
        free_min = sum(
            _to_min(s.end) - _to_min(s.start)
            for s in self.slots if s.kind == SlotKind.FREE
        )
        lines.append(
            f"  Tasks placed: {placed}  |  "
            f"Unscheduled: {len(self.unscheduled)}  |  "
            f"Free time: {free_min}min"
        )
        return "\n".join(lines)


# ─── Planner ──────────────────────────────────────────────────────────────────

class GreedyPlanner:
    """
    Deterministic greedy day planner.

    Usage:
        planner = GreedyPlanner()
        plan = planner.plan(schedule)
        print(plan.summary())
    """

    def plan(self, schedule: Schedule) -> DayPlan:
        wake = _to_min(schedule.wake_up)
        sleep = _to_min(schedule.sleep)
        day_len = sleep - wake              # minutes in the active day

        # ── 1. Timeline: index 0 = wake_up minute ─────────────────────────────
        # True = free, False = blocked
        free: list[bool] = [True] * day_len

        # ── 2. Block fixed events ─────────────────────────────────────────────
        placed_slots: list[Slot] = []

        for ev in schedule.fixed_events:
            ev_start = _to_min(ev.start) - wake
            ev_end   = _to_min(ev.end)   - wake
            if ev_start < 0 or ev_end > day_len:
                continue                   # outside active window — skip
            for m in range(max(ev_start, 0), min(ev_end, day_len)):
                free[m] = False
            placed_slots.append(Slot(
                kind=SlotKind.FIXED_EVENT,
                name=ev.name,
                start=ev.start,
                end=ev.end,
            ))

        # ── 3. Sort tasks: priority first, then shorter tasks first ───────────
        sorted_tasks = sorted(
            schedule.tasks,
            key=lambda t: (PRIORITY_ORDER[t.priority], t.duration_min),
        )

        # ── 4. Place each task ────────────────────────────────────────────────
        earliest_offset = MORNING_BUFFER_MIN   # 30-min buffer after wake_up

        for task in sorted_tasks:
            note: Optional[str] = None

            # Try preferred window first (soft constraint)
            offset = self._find_slot(
                free, task.duration_min, earliest_offset, day_len,
                window_hint=task.preferred_window, wake_abs=wake,
            )

            # Fallback: any free window
            if offset is None and task.preferred_window != "any":
                offset = self._find_slot(
                    free, task.duration_min, earliest_offset, day_len,
                    window_hint="any", wake_abs=wake,
                )
                if offset is not None:
                    note = f"preferred {task.preferred_window} → placed {_window_of(offset + wake)}"

            if offset is None:
                schedule_unscheduled = UnscheduledTask(
                    name=task.name,
                    duration_min=task.duration_min,
                    priority=task.priority,
                    reason=f"no contiguous {task.duration_min}min free slot found",
                )
                # Will collect below
                placed_slots.append(Slot(
                    kind=SlotKind.TASK,
                    name=f"[UNSCHEDULED] {task.name}",
                    start="--:--",
                    end="--:--",
                    priority=task.priority,
                    note=schedule_unscheduled.reason,
                ))
                # Store as unscheduled — handled at end
                placed_slots[-1].__dict__["_unscheduled_obj"] = schedule_unscheduled
                continue

            # Mark minutes as blocked
            for m in range(offset, offset + task.duration_min):
                free[m] = False

            task_start_abs = offset + wake
            task_end_abs   = offset + task.duration_min + wake
            placed_slots.append(Slot(
                kind=SlotKind.TASK,
                name=task.name,
                start=_to_hhmm(task_start_abs),
                end=_to_hhmm(task_end_abs),
                priority=task.priority,
                note=note,
            ))

        # ── 5. Collect free gaps ───────────────────────────────────────────────
        gap_slots = _collect_gaps(free, wake, min_gap_min=5)

        # ── 6. Separate real slots from unscheduled markers ───────────────────
        real_slots: list[Slot] = []
        unscheduled: list[UnscheduledTask] = []

        for s in placed_slots:
            obj = s.__dict__.pop("_unscheduled_obj", None)
            if obj is not None:
                unscheduled.append(obj)
            else:
                real_slots.append(s)

        # ── 7. Merge + sort by start time ─────────────────────────────────────
        all_slots = sorted(
            real_slots + gap_slots,
            key=lambda s: _to_min(s.start) if s.start != "--:--" else 9999,
        )

        return DayPlan(
            wake_up=schedule.wake_up,
            sleep=schedule.sleep,
            slots=all_slots,
            unscheduled=unscheduled,
        )

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_slot(
        free: list[bool],
        duration: int,
        earliest: int,
        day_len: int,
        window_hint: str,
        wake_abs: int,
    ) -> Optional[int]:
        """
        Find the first offset (relative to wake_up) where `duration` contiguous
        free minutes exist, starting from `earliest`, within `window_hint`.
        Returns offset or None.
        """
        w_start, w_end = WINDOW_RANGES[window_hint]
        # Convert absolute window bounds to relative offsets
        rel_start = max(earliest, w_start - wake_abs)
        rel_end   = min(day_len,  w_end   - wake_abs)

        if rel_start >= rel_end:
            return None

        consecutive = 0
        for i in range(rel_start, rel_end):
            if i >= day_len:
                break
            if free[i]:
                consecutive += 1
                if consecutive >= duration:
                    return i - duration + 1   # start of the window
            else:
                consecutive = 0

        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _collect_gaps(free: list[bool], wake: int, min_gap_min: int = 5) -> list[Slot]:
    """Turn contiguous True runs in the free[] array into FREE slots."""
    gaps: list[Slot] = []
    start: Optional[int] = None

    for i, is_free in enumerate(free):
        if is_free and start is None:
            start = i
        elif not is_free and start is not None:
            length = i - start
            if length >= min_gap_min:
                gaps.append(Slot(
                    kind=SlotKind.FREE,
                    name="Free",
                    start=_to_hhmm(start + wake),
                    end=_to_hhmm(i + wake),
                ))
            start = None

    # Trailing gap
    if start is not None:
        length = len(free) - start
        if length >= min_gap_min:
            gaps.append(Slot(
                kind=SlotKind.FREE,
                name="Free",
                start=_to_hhmm(start + wake),
                end=_to_hhmm(len(free) + wake),
            ))

    return gaps


def _to_min(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _to_hhmm(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def _window_of(abs_min: int) -> str:
    if abs_min < 720:
        return "morning"
    if abs_min < 1020:
        return "afternoon"
    return "evening"
