"""
planner.py — Phase 4: Greedy Day Planner

Takes a validated Schedule (output of parser.py) and returns a complete
DayPlan: every minute of the day accounted for as a placed event, placed
task, free gap, or unscheduled task with a reason.


Algorithm
---------
1. Build a minute-resolution timeline [0, day_len) where 0 = wake_up.
2. Sort fixed events by start time. Merge/clamp overlapping events so
   every rendered slot is non-overlapping.
3. Block all fixed-event minutes in free[].
4. Deduplicate tasks against fixed events by name.
5. Sort tasks: high → medium → low, then shorter first within same tier.
6. For each task: scan forward from (wake + 30 min buffer), find first
   contiguous free window ≥ duration_min within preferred_window (soft).
   Fall back to any free window. If nothing fits → unscheduled[].
7. Collect free gaps ≥ 5 min.
8. Merge + sort all slots by start time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models import Schedule, FixedEvent, Task


# ─── Constants ────────────────────────────────────────────────────────────────

MORNING_BUFFER_MIN: int = 30
PRIORITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

# Absolute minute-of-day ranges for preferred_window
WINDOW_RANGES: dict[str, tuple[int, int]] = {
    "morning":   (0,    719),
    "afternoon": (720,  1019),
    "evening":   (1020, 1439),
    "any":       (0,    1439),
}


# ─── Output models ────────────────────────────────────────────────────────────

class SlotKind(str, Enum):
    FIXED_EVENT = "fixed_event"
    TASK        = "task"
    FREE        = "free"


@dataclass
class Slot:
    kind:     SlotKind
    name:     str
    start:    str
    end:      str
    priority: Optional[str] = None
    note:     Optional[str] = None


@dataclass
class UnscheduledTask:
    name:         str
    duration_min: int
    priority:     str
    reason:       str


@dataclass
class DayPlan:
    wake_up:     str
    sleep:       str
    slots:       list[Slot]            = field(default_factory=list)
    unscheduled: list[UnscheduledTask] = field(default_factory=list)
    warnings:    list[str]             = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Day Plan  {self.wake_up} → {self.sleep}", f"{'─'*52}"]
        for s in self.slots:
            tag  = f"[{s.kind.value:11s}]"
            pri  = f" ({s.priority})" if s.priority else ""
            note = f"  ← {s.note}"   if s.note     else ""
            lines.append(f"  {s.start}–{s.end}  {tag}  {s.name}{pri}{note}")
        if self.warnings:
            lines.append(f"{'─'*52}")
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")
        if self.unscheduled:
            lines.append(f"{'─'*52}")
            lines.append("  Unscheduled:")
            for u in self.unscheduled:
                lines.append(f"    ✗ {u.name} ({u.duration_min}min, {u.priority}): {u.reason}")
        lines.append(f"{'─'*52}")
        placed   = sum(1 for s in self.slots if s.kind == SlotKind.TASK)
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
    Overlap-safe greedy day planner.

    Usage:
        plan = GreedyPlanner().plan(schedule)
        print(plan.summary())
    """

    def plan(self, schedule: Schedule) -> DayPlan:
        wake    = _to_min(schedule.wake_up)
        sleep   = _to_min(schedule.sleep)
        day_len = sleep - wake
        warnings: list[str] = []

        # ── 1. free[] timeline: index i = minute (wake + i) ───────────────────
        free: list[bool] = [True] * day_len

        # ── 2. Sort + merge overlapping fixed events ──────────
        sorted_events = sorted(schedule.fixed_events, key=lambda e: _to_min(e.start))
        merged_events = _merge_overlapping(sorted_events, warnings)

        # ── 3. Block fixed events in free[] ───────────────────────────────────
        event_slots: list[Slot] = []
        for ev in merged_events:
            ev_start_rel = _to_min(ev.start) - wake
            ev_end_rel   = _to_min(ev.end)   - wake
            # Clamp to active window
            block_start = max(ev_start_rel, 0)
            block_end   = min(ev_end_rel, day_len)
            if block_start >= block_end:
                warnings.append(f"'{ev.name}' is outside wake/sleep window — skipped")
                continue
            for m in range(block_start, block_end):
                free[m] = False
            event_slots.append(Slot(
                kind=SlotKind.FIXED_EVENT,
                name=ev.name,
                start=_to_hhmm(wake + block_start),
                end=_to_hhmm(wake + block_end),
            ))

        # ── 4. Build set of fixed-event names for deduplication  ───
        event_names = {_normalise(e.name) for e in merged_events}

        # ── 5. Sort tasks: priority → duration (shorter first within tier) ─────
        sorted_tasks = sorted(
            schedule.tasks,
            key=lambda t: (PRIORITY_ORDER[t.priority], t.duration_min),
        )

        # ── 6. Place tasks ─────────────────────────────────────────────────────
        task_slots:  list[Slot]            = []
        unscheduled: list[UnscheduledTask] = []
        earliest_rel = MORNING_BUFFER_MIN  # 30-min buffer after wake_up

        for task in sorted_tasks:
            #  skip task that duplicates a fixed event name
            if _normalise(task.name) in event_names:
                warnings.append(
                    f"Task '{task.name}' matches a fixed event — skipped to avoid duplicate"
                )
                continue

            note: Optional[str] = None

            # Try preferred window first (soft constraint)
            offset = _find_slot(free, task.duration_min, earliest_rel, day_len,
                                 task.preferred_window, wake)

            # Fallback: any free slot
            if offset is None and task.preferred_window != "any":
                offset = _find_slot(free, task.duration_min, earliest_rel, day_len,
                                     "any", wake)
                if offset is not None:
                    placed_window = _window_of(wake + offset)
                    note = f"preferred {task.preferred_window} → placed {placed_window}"

            if offset is None:
                unscheduled.append(UnscheduledTask(
                    name=task.name,
                    duration_min=task.duration_min,
                    priority=task.priority,
                    reason=f"no contiguous {task.duration_min}min free slot in day",
                ))
                continue

            for m in range(offset, offset + task.duration_min):
                free[m] = False

            task_slots.append(Slot(
                kind=SlotKind.TASK,
                name=task.name,
                start=_to_hhmm(wake + offset),
                end=_to_hhmm(wake + offset + task.duration_min),
                priority=task.priority,
                note=note,
            ))

        # ── 7. Collect free gaps ≥ 5 min ──────────────────────────────────────
        gap_slots = _collect_gaps(free, wake, min_gap_min=5)

        # ── 8. Merge + sort by start time ─────────────────────────────────────
        all_slots = sorted(
            event_slots + task_slots + gap_slots,
            key=lambda s: _to_min(s.start),
        )

        return DayPlan(
            wake_up=schedule.wake_up,
            sleep=schedule.sleep,
            slots=all_slots,
            unscheduled=unscheduled,
            warnings=warnings,
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _merge_overlapping(events: list[FixedEvent], warnings: list[str]) -> list[FixedEvent]:
    """
    merge fixed events that overlap so they render as a single
    non-overlapping slot. Events must be sorted by start time before calling.

    Merging strategy: extend the running end time; concatenate names.
    E.g. dinner(13:30–14:30) + english(14:00–14:30) → dinner + english(13:30–14:30)
    """
    if not events:
        return []

    merged: list[FixedEvent] = []
    current = events[0]

    for nxt in events[1:]:
        cur_end  = _to_min(current.end)
        nxt_start = _to_min(nxt.start)
        nxt_end   = _to_min(nxt.end)

        if nxt_start < cur_end:
            # Overlap detected — merge
            new_end  = max(cur_end, nxt_end)
            new_name = f"{current.name} + {nxt.name}"
            warnings.append(
                f"Fixed events '{current.name}' and '{nxt.name}' overlap "
                f"({current.start}–{current.end} / {nxt.start}–{nxt.end}) "
                f"— merged into '{new_name}' ({current.start}–{_to_hhmm(new_end)})"
            )
            # Rebuild as a plain object (FixedEvent is a Pydantic model)
            current = _make_event(new_name, current.start, _to_hhmm(new_end))
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def _make_event(name: str, start: str, end: str) -> FixedEvent:
    """Create a FixedEvent without triggering Pydantic validators (used internally)."""
    ev = object.__new__(FixedEvent)
    object.__setattr__(ev, "name", name)
    object.__setattr__(ev, "start", start)
    object.__setattr__(ev, "end", end)
    object.__setattr__(ev, "recurrence", "none")
    return ev


def _normalise(name: str) -> str:
    """Lowercase + strip for fuzzy name comparison."""
    return name.lower().strip()


def _find_slot(
    free: list[bool],
    duration: int,
    earliest_rel: int,
    day_len: int,
    window_hint: str,
    wake_abs: int,
) -> Optional[int]:
    """
    Find the first relative offset in free[] where `duration` contiguous True
    values exist, starting from earliest_rel, within window_hint.
    """
    w_start_abs, w_end_abs = WINDOW_RANGES[window_hint]

    # Convert absolute window bounds → relative to wake
    rel_start = max(earliest_rel, w_start_abs - wake_abs)
    rel_end   = min(day_len,      w_end_abs   - wake_abs + 1)

    if rel_start >= rel_end:
        return None

    consecutive = 0
    for i in range(rel_start, rel_end):
        if free[i]:
            consecutive += 1
            if consecutive >= duration:
                return i - duration + 1
        else:
            consecutive = 0

    return None


def _collect_gaps(free: list[bool], wake: int, min_gap_min: int = 5) -> list[Slot]:
    gaps: list[Slot] = []
    start: Optional[int] = None

    for i, is_free in enumerate(free):
        if is_free and start is None:
            start = i
        elif not is_free and start is not None:
            if i - start >= min_gap_min:
                gaps.append(Slot(
                    kind=SlotKind.FREE,
                    name="Free",
                    start=_to_hhmm(wake + start),
                    end=_to_hhmm(wake + i),
                ))
            start = None

    if start is not None and len(free) - start >= min_gap_min:
        gaps.append(Slot(
            kind=SlotKind.FREE,
            name="Free",
            start=_to_hhmm(wake + start),
            end=_to_hhmm(wake + len(free)),
        ))

    return gaps


def _to_min(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _to_hhmm(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


def _window_of(abs_min: int) -> str:
    if abs_min < 720:  return "morning"
    if abs_min < 1020: return "afternoon"
    return "evening"
