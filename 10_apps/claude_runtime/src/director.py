"""
AI Influencer OS — Director V1

The Director provides one central entry point for existing workflows.

Examples:

    # List image requests
    python3 src/director.py list

    # Create a new request
    python3 src/director.py new airport duty_free

    # Generate one image prompt
    python3 src/director.py image airport/lounge

    # Generate one caption prompt
    python3 src/director.py caption airport/lounge

    # Generate all multi-shot prompts
    python3 src/director.py multishot luxury_hotel/pool --all

    # Send one multi-shot prompt to ChatGPT
    python3 src/director.py multishot luxury_hotel/pool --shot cover --send

    # View content scheduled for a date
    python3 src/director.py plan --date 2026-07-15

    # Run the scheduled image workflow
    python3 src/director.py plan --date 2026-07-15 --run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


# ============================================================
# Runtime paths
# ============================================================

RUNTIME_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = RUNTIME_DIR / "src"

IMAGE_SCRIPT = SRC_DIR / "main.py"
CAPTION_SCRIPT = SRC_DIR / "caption_main.py"
MULTISHOT_SCRIPT = SRC_DIR / "multishot_main.py"
PLANNER_SCRIPT = SRC_DIR / "planner_main.py"


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Influencer OS central task director.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 src/director.py list\n"
            "  python3 src/director.py new airport duty_free\n"
            "  python3 src/director.py image airport/lounge\n"
            "  python3 src/director.py caption airport/lounge\n"
            "  python3 src/director.py multishot luxury_hotel/pool --all\n"
            "  python3 src/director.py plan --date 2026-07-15\n"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    subparsers.add_parser(
        "list",
        help="List all available image requests.",
    )

    # --------------------------------------------------------
    # new
    # --------------------------------------------------------

    new_parser = subparsers.add_parser(
        "new",
        help="Create a new image-request YAML file.",
    )

    new_parser.add_argument(
        "path",
        nargs="+",
        help=(
            "Request path.\n"
            "Examples: airport duty_free\n"
            "          luxury_hotel breakfast"
        ),
    )

    # --------------------------------------------------------
    # image
    # --------------------------------------------------------

    image_parser = subparsers.add_parser(
        "image",
        help="Generate and send one image prompt.",
    )

    image_parser.add_argument(
        "request",
        help="Example: airport/lounge",
    )

    # --------------------------------------------------------
    # caption
    # --------------------------------------------------------

    caption_parser = subparsers.add_parser(
        "caption",
        help="Generate and send one caption prompt.",
    )

    caption_parser.add_argument(
        "request",
        help="Example: airport/lounge",
    )

    # --------------------------------------------------------
    # multishot
    # --------------------------------------------------------

    multishot_parser = subparsers.add_parser(
        "multishot",
        help="Generate multi-shot prompts.",
    )

    multishot_parser.add_argument(
        "request",
        help="Example: luxury_hotel/pool",
    )

    multishot_actions = (
        multishot_parser.add_mutually_exclusive_group(
            required=True
        )
    )

    multishot_actions.add_argument(
        "--all",
        action="store_true",
        help="Generate all shot prompt files.",
    )

    multishot_actions.add_argument(
        "--shot",
        metavar="NAME",
        help="Generate one shot, such as cover or closeup.",
    )

    multishot_actions.add_argument(
        "--list-shots",
        action="store_true",
        help="Show available shot names.",
    )

    multishot_parser.add_argument(
        "--send",
        action="store_true",
        help="Copy the selected shot prompt and open ChatGPT.",
    )

    # --------------------------------------------------------
    # plan
    # --------------------------------------------------------

    plan_parser = subparsers.add_parser(
        "plan",
        help="View or run scheduled content.",
    )

    plan_date_group = (
        plan_parser.add_mutually_exclusive_group(
            required=True
        )
    )

    plan_date_group.add_argument(
        "--today",
        action="store_true",
        help="Use today's date.",
    )

    plan_date_group.add_argument(
        "--date",
        help="Specific date in YYYY-MM-DD format.",
    )

    plan_parser.add_argument(
        "--run",
        action="store_true",
        help="Run the scheduled image workflow.",
    )

    return parser


# ============================================================
# Script execution
# ============================================================

def validate_script(script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(
            f"Required workflow script was not found: {script_path}"
        )

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Workflow path is not a file: {script_path}"
        )


def run_script(
    script_path: Path,
    arguments: Sequence[str],
) -> None:
    """
    Run an existing workflow with the same Python interpreter
    currently used by Director.
    """

    validate_script(script_path)

    command = [
        sys.executable,
        str(script_path),
        *arguments,
    ]

    print("\n" + "=" * 70)
    print("AI INFLUENCER OS DIRECTOR")
    print("=" * 70)
    print(f"Workflow: {script_path.stem}")
    print(f"Command: {' '.join(command)}")
    print("=" * 70 + "\n")

    subprocess.run(
        command,
        cwd=RUNTIME_DIR,
        check=True,
    )


# ============================================================
# Task routing
# ============================================================

def handle_list() -> None:
    run_script(
        IMAGE_SCRIPT,
        ["--list"],
    )


def handle_new(path_parts: list[str]) -> None:
    run_script(
        IMAGE_SCRIPT,
        ["--new", *path_parts],
    )


def handle_image(request: str) -> None:
    run_script(
        IMAGE_SCRIPT,
        [request],
    )


def handle_caption(request: str) -> None:
    run_script(
        CAPTION_SCRIPT,
        [request],
    )


def handle_multishot(
    request: str,
    all_shots: bool,
    shot_name: str | None,
    list_shots: bool,
    send: bool,
) -> None:
    arguments = [request]

    if all_shots:
        arguments.append("--all")

    elif shot_name:
        arguments.extend(
            ["--shot", shot_name]
        )

        if send:
            arguments.append("--send")

    elif list_shots:
        arguments.append("--list-shots")

    run_script(
        MULTISHOT_SCRIPT,
        arguments,
    )


def handle_plan(
    use_today: bool,
    selected_date: str | None,
    run: bool,
) -> None:
    arguments: list[str] = []

    if use_today:
        arguments.append("--today")
    else:
        arguments.extend(
            ["--date", str(selected_date)]
        )

    if run:
        arguments.append("--run")

    run_script(
        PLANNER_SCRIPT,
        arguments,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        handle_list()
        return

    if args.command == "new":
        handle_new(args.path)
        return

    if args.command == "image":
        handle_image(args.request)
        return

    if args.command == "caption":
        handle_caption(args.request)
        return

    if args.command == "multishot":
        if args.send and not args.shot:
            raise ValueError(
                "--send must be used together with --shot."
            )

        handle_multishot(
            request=args.request,
            all_shots=args.all,
            shot_name=args.shot,
            list_shots=args.list_shots,
            send=args.send,
        )
        return

    if args.command == "plan":
        handle_plan(
            use_today=args.today,
            selected_date=args.date,
            run=args.run,
        )
        return

    parser.error(
        f"Unsupported command: {args.command}"
    )


if __name__ == "__main__":
    main()