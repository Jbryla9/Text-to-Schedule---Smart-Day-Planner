from app.models import FixedEvent, Schedule, Task
from app.planner import GreedyPlanner, SlotKind


def make_schedule(*, events=None, tasks=None):
    return Schedule(
        wake_up="07:00",
        sleep="23:00",
        fixed_events=events or [],
        tasks=tasks or [],
        ambiguities=[],
        metadata=None,
    )


def test_high_priority_task_is_placed_before_low_priority_task():
    plan = GreedyPlanner().plan(make_schedule(tasks=[
        Task(name="Read", duration_min=30, priority="low"),
        Task(name="Important work", duration_min=60, priority="high"),
    ]))
    tasks = [slot for slot in plan.slots if slot.kind == SlotKind.TASK]

    assert [slot.name for slot in tasks] == ["Important work", "Read"]


def test_planner_respects_thirty_minute_morning_buffer():
    plan = GreedyPlanner().plan(make_schedule(tasks=[
        Task(name="Work", duration_min=60, priority="high"),
    ]))
    task = next(slot for slot in plan.slots if slot.kind == SlotKind.TASK)

    assert task.start == "07:30"


def test_task_that_does_not_fit_is_reported_unscheduled():
    plan = GreedyPlanner().plan(Schedule(
        wake_up="07:00",
        sleep="08:00",
        fixed_events=[],
        tasks=[Task(name="Long work", duration_min=90, priority="high")],
    ))

    assert len(plan.unscheduled) == 1
    assert plan.unscheduled[0].name == "Long work"


def test_overlapping_fixed_events_are_preserved_and_reported():
    plan = GreedyPlanner().plan(make_schedule(events=[
        FixedEvent(name="Client call", start="10:00", end="11:00"),
        FixedEvent(name="Doctor", start="10:30", end="11:30"),
    ]))
    fixed = [slot for slot in plan.slots if slot.kind == SlotKind.FIXED_EVENT]

    assert [slot.name for slot in fixed] == ["Client call", "Doctor"]
    assert len(plan.conflicts) == 1
    assert plan.conflicts[0].overlap_start == "10:30"
    assert plan.conflicts[0].overlap_end == "11:00"


def test_fixed_events_block_task_minutes():
    plan = GreedyPlanner().plan(make_schedule(
        events=[FixedEvent(name="Meeting", start="08:00", end="09:00")],
        tasks=[Task(name="Work", duration_min=60, priority="high")],
    ))
    task = next(slot for slot in plan.slots if slot.kind == SlotKind.TASK)

    assert task.end <= "08:00" or task.start >= "09:00"

