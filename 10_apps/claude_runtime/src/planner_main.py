"""
AI Influencer OS — Content Planner V1

Commands:

    python3 src/planner_main.py --today
    python3 src/planner_main.py --date 2026-07-15
    python3 src/planner_main.py --today --run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONTENT_PLAN_DIR = (
    PROJECT_ROOT
    / "12_content"
    / "content_plans"
    / "monthly"
)

RUNTIME_DIR = (
    PROJECT_ROOT
    / "10_apps"
    / "claude_runtime"
)

MAIN_SCRIPT = (
    RUNTIME_DIR
    / "src"
    / "main.py"
)


from core.loaders.content_plan_loader import ContentPlanLoader


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read and run Aiko's content schedule."
    )

    date_group = parser.add_mutually_exclusive_group(
        required=True
    )

    date_group.add_argument(
        "--today",
        action="store_true",
        help="Use today's date.",
    )

    date_group.add_argument(
        "--date",
        help="Use a specific date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the scheduled image-request workflow.",
    )

    return parser.parse_args()


def resolve_date(
    args: argparse.Namespace,
) -> str:
    if args.today:
        return date.today().isoformat()

    selected_date = str(args.date).strip()

    try:
        date.fromisoformat(selected_date)
    except ValueError as error:
        raise ValueError(
            "Date must use YYYY-MM-DD format."
        ) from error

    return selected_date


def resolve_plan_path(
    selected_date: str,
) -> Path:
    month_name = selected_date[:7]

    return (
        CONTENT_PLAN_DIR
        / f"{month_name}.yaml"
    )


def run_scheduled_request(
    request_name: str,
) -> None:
    command = [
        sys.executable,
        str(MAIN_SCRIPT),
        request_name,
    ]

    subprocess.run(
        command,
        cwd=RUNTIME_DIR,
        check=True,
    )


def main() -> None:
    args = parse_arguments()

    selected_date = resolve_date(args)
    plan_path = resolve_plan_path(selected_date)

    loader = ContentPlanLoader(plan_path)
    item = loader.get_date(selected_date)

    request_name = str(item["request"])
    content_type = str(
        item.get("content_type", "image_post")
    )
    objective = str(
        item.get("objective", "")
    )

    print("\n" + "=" * 70)
    print("AIKO CONTENT PLAN")
    print("=" * 70)
    print(f"Date: {selected_date}")
    print(f"Request: {request_name}")
    print(f"Content type: {content_type}")

    if objective:
        print(f"Objective: {objective}")

    if not args.run:
        print("\nTo generate this content, run:")
        print(
            f"python3 src/main.py {request_name}"
        )
        return

    print("\nStarting image workflow...")
    run_scheduled_request(request_name)


if __name__ == "__main__":
    main()