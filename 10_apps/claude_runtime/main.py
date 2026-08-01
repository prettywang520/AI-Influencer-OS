from __future__ import annotations

import argparse
from datetime import date

from src.content_service import build_service


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO OS Daily Content Runtime"
    )

    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Production date in YYYY-MM-DD format.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    service = build_service()

    plan = service.generate(
        target_date=args.date,
    )

    print()
    print("AIKO Daily Production")
    print("---------------------")
    print(f"date:          {plan.date}")
    print(f"country:       {plan.country}")
    print(f"city:          {plan.city}")
    print(f"venue:         {plan.venue}")
    print(f"theme:         {plan.theme}")
    print("feed:          1")
    print(f"stories:       {len(plan.stories)}")
    print(f"reel scenes:   {len(plan.reel_scenes)}")
    print(
        "quality gate:  "
        f"{'passed' if plan.validation.get('passed') else 'failed'}"
    )
    print()


if __name__ == "__main__":
    main()