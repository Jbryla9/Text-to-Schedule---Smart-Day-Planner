"""
main.py — Full pipeline demo: raw text → structured JSON → day plan

Usage:
    python main.py
    python main.py "wake 7, gym 1h, meeting 3pm, dinner 19, sleep 23"
    python main.py --plan-only   # skip JSON dump, show plan only
"""

from __future__ import annotations
import argparse
import sys

from pipeline import run


DEMO_QUERIES = [
    "wake up 7am, take kids to school 8 to 9, gym for 1 hour, work on project 3 hours high priority, lunch 12:30 for 30min, design review 14:00 for 45min, dinner 19, sleep 23",
    "standup 10 to 10:30, deep work 4h high priority, design review 14:00 45min, shopping 1h, gym 18:00 1h, read before bed 30min, sleep 23",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="Scheduling query text")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    queries = args.query if args.query else DEMO_QUERIES

    for i, query in enumerate(queries, 1):
        print(f"\n{'═'*62}")
        print(f"  Input {i}: {query[:65]}{'…' if len(query) > 65 else ''}")
        print(f"{'═'*62}")

        result = run(query)

        if not result.success:
            print(f"\n  ✗ Parse failed: {result.parse.errors}")
            continue

        s = result.parse.schedule
        p = result.plan

        if not args.plan_only:
            conf = f"{s.metadata.parse_confidence:.0%}" if s.metadata else "—"
            print(f"\n  Parse  (attempt {result.parse.attempts}, confidence {conf})")
            print(f"  wake={s.wake_up}  sleep={s.sleep}"
                  f"  events={len(s.fixed_events)}  tasks={len(s.tasks)}")
            if s.ambiguities:
                for a in s.ambiguities:
                    print(f"  ! {a}")
            print()

        print(p.summary())

    print()


if __name__ == "__main__":
    main()
