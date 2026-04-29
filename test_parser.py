"""
test_parser.py — Phase 2: Test Suite
Measures parser accuracy across 10 canonical test cases.

Usage:
    python test_parser.py
    python test_parser.py --verbose
    python test_parser.py --case 3       # run single case
"""

from __future__ import annotations
import argparse
import time
from dataclasses import dataclass, field
from typing import Callable

from parser import ScheduleParser, ParseResult, LLMConfig


# ─── Check helpers ────────────────────────────────────────────────────────────

def _has_field(name: str) -> Callable:
    def check(r: ParseResult) -> tuple[bool, str]:
        if not r.success:
            return False, f"parse failed: {r.errors}"
        val = getattr(r.schedule, name, None)
        if val is None:
            return False, f"missing: {name}"
        if isinstance(val, list) and len(val) == 0 and name != "ambiguities":
            return False, f"empty list: {name}"
        return True, f"{name} ok"
    return check


def _has_events(n: int = 1) -> Callable:
    def check(r: ParseResult) -> tuple[bool, str]:
        if not r.success:
            return False, "parse failed"
        count = len(r.schedule.fixed_events)
        return count >= n, f"fixed_events={count} (want >={n})"
    return check


def _has_tasks(n: int = 1) -> Callable:
    def check(r: ParseResult) -> tuple[bool, str]:
        if not r.success:
            return False, "parse failed"
        count = len(r.schedule.tasks)
        return count >= n, f"tasks={count} (want >={n})"
    return check


def _has_ambiguities() -> Callable:
    def check(r: ParseResult) -> tuple[bool, str]:
        if not r.success:
            return False, "parse failed"
        n = len(r.schedule.ambiguities)
        return n > 0, f"ambiguities={n} (want >0 for vague input)"
    return check


# ─── Test cases ───────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: int
    description: str
    input: str
    checks: list[Callable[[ParseResult], tuple[bool, str]]]


TEST_CASES: list[TestCase] = [
    TestCase(
        id=1,
        description="Full explicit day",
        input="wake up 7am, kids school 8 to 9, gym 1h, lunch 12:30 for 30min, work 3h, sleep 23",
        checks=[_has_field("wake_up"), _has_field("sleep"), _has_events(2), _has_tasks(1)],
    ),
    TestCase(
        id=2,
        description="Implicit wake, one anchored event",
        input="start day 9, pick kids 16:30 for 30min, sleep 22",
        checks=[_has_events(1), _has_field("wake_up")],
    ),
    TestCase(
        id=3,
        description="Mixed anchored + duration only",
        input="morning run 6am 45min, standup 10 to 10:30, deep work 4h, lunch, sleep midnight",
        checks=[_has_events(2), _has_tasks(1), _has_field("wake_up")],
    ),
    TestCase(
        id=4,
        description="Commute + full work day",
        input="wake 6:30, commute 45min, work 9-17, lunch 13, evening walk, sleep 22:30",
        checks=[_has_field("wake_up"), _has_field("sleep"), _has_events(2)],
    ),
    TestCase(
        id=5,
        description="No wake/sleep stated — must infer",
        input="gym 1h, work 8h, dinner 19:00",
        checks=[_has_tasks(1), _has_events(1), _has_field("wake_up")],
    ),
    TestCase(
        id=6,
        description="Meeting + project tasks",
        input="meeting 3pm for 1 hour, project 3h, shopping 1h",
        checks=[_has_events(1), _has_tasks(2)],
    ),
    TestCase(
        id=7,
        description="Family day with appointments",
        input="wake 8, breakfast, kids dentist 10:30 for 45min, groceries 1h, cook dinner 45min, sleep 23",
        checks=[_has_events(1), _has_tasks(2), _has_field("wake_up")],
    ),
    TestCase(
        id=8,
        description="Tech worker day",
        input="6am wake, run 30min, work 9-18, lunch 12, gym 19:00 1h, sleep 23",
        checks=[_has_field("wake_up"), _has_events(3), _has_tasks(1)],
    ),
    TestCase(
        id=9,
        description="Pure work schedule",
        input="standup 10 to 10:30, design review 14:00 30min, coding 3h, deploy 1h",
        checks=[_has_events(2), _has_tasks(2)],
    ),
    TestCase(
        id=10,
        description="Very vague — must surface ambiguities",
        input="lazy morning, some errands, maybe gym, dinner 7pm, early sleep",
        checks=[_has_ambiguities(), _has_events(1)],
    ),
    TestCase(
        id=11,
        description="",
        input="8 am wake, breakfast 9pm, gym 10pm, learning python 11, dinner 13:30, english 30min 14-14:30, learning python 14:30, football training 18, dinner 20. I need to also spent 15min to order ball in amazon",
        checks=[_has_ambiguities(), _has_events(1)],
    )
]


# ─── Runner ───────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    case: TestCase
    parse_result: ParseResult
    outcomes: list[tuple[bool, str]]
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.outcomes)


def run_suite(verbose: bool = False, case_filter: Optional[int] = None) -> None:
    parser = ScheduleParser()
    cases = [tc for tc in TEST_CASES if case_filter is None or tc.id == case_filter]
    results: list[TestResult] = []

    model_str = f"{parser.cfg.model} via {parser.cfg.base_url.split('/')[2]}"
    print(f"\n{'─'*62}")
    print(f"  Smart Scheduler · Parser Test Suite")
    print(f"  Model : {model_str}")
    print(f"  Cases : {len(cases)}")
    print(f"{'─'*62}\n")

    for tc in cases:
        print(f"  [{tc.id:02d}] {tc.description}")
        if verbose:
            print(f"       input : {tc.input[:80]}")

        t0 = time.time()
        pr = parser.parse(tc.input)
        elapsed = time.time() - t0

        outcomes = [chk(pr) for chk in tc.checks]
        tr = TestResult(tc, pr, outcomes, elapsed)
        results.append(tr)

        status = "PASS" if tr.passed else "FAIL"
        retry_note = f" (retried {pr.attempts-1}x)" if pr.attempts > 1 else ""
        print(f"       {status}  {elapsed:.1f}s{retry_note}")

        if verbose or not tr.passed:
            for ok, msg in outcomes:
                print(f"         {'✓' if ok else '✗'} {msg}")
            if pr.success and verbose:
                s = pr.schedule
                conf = f"  conf={s.metadata.parse_confidence:.2f}" if s.metadata else ""
                print(f"         wake={s.wake_up}  sleep={s.sleep}"
                      f"  events={len(s.fixed_events)}  tasks={len(s.tasks)}{conf}")
            if not pr.success:
                print(f"         errors: {pr.errors[:3]}")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    accuracy = passed / total * 100 if total else 0
    avg_t = sum(r.elapsed_s for r in results) / total if total else 0
    retried = sum(1 for r in results if r.parse_result.attempts > 1)

    print(f"{'─'*62}")
    print(f"  Accuracy : {passed}/{total}  ({accuracy:.0f}%)")
    print(f"  Retries  : {retried} case(s) needed >1 attempt")
    print(f"  Avg time : {avg_t:.1f}s per case")
    print(f"{'─'*62}\n")

    if passed < total:
        print("  Failed cases:")
        for r in results:
            if not r.passed:
                bad = [msg for ok, msg in r.outcomes if not ok]
                print(f"    [{r.case.id:02d}] {r.case.description}: {'; '.join(bad)}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--case", "-c", type=int, help="Run only this test case ID")
    args = ap.parse_args()
    run_suite(verbose=args.verbose, case_filter=args.case)
