"""Optional command-line interface using the same pipeline as the web API."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .monitoring import configure_logging
from .pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a daily schedule from text")
    parser.add_argument("text", help="Natural-language plan (wrap it in quotes)")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    result = run(args.text)
    if not result.success:
        print(f"Could not create schedule: {result.parse.errors}")
        return 1

    if not args.plan_only:
        print(result.parse.schedule.model_dump_json(indent=2))
        print()
    print(result.plan.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
