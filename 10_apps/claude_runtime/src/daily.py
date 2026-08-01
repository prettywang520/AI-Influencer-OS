from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AIKO daily production workflow."
    )

    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Production date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not write the generated plan to planning history.",
    )

    parser.add_argument(
        "--next",
        action="store_true",
        help="Prepare the first image task after production is created.",
    )

    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    print()

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    arguments = parse_arguments()

    production_command = [
        sys.executable,
        "-u",
        "-m",
        "src.production_service",
        "--date",
        arguments.date,
    ]

    if arguments.no_history:
        production_command.append("--no-history")

    print()
    print("AIKO Daily Production")
    print("---------------------")
    print(f"date: {arguments.date}")

    run_command(production_command)

    if arguments.next:
        run_command(
            [
                sys.executable,
                "-u",
                "-m",
                "src.production_worker",
                "--date",
                arguments.date,
                "--next",
            ]
        )
        return

    print()
    print("Daily production created successfully.")
    print()
    print("Start the first image task with:")
    print(
        f"python3 -u -m src.production_worker "
        f"--date {arguments.date} --next"
    )


if __name__ == "__main__":
    main()