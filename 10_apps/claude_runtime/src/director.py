from datetime import datetime
from pathlib import Path

import yaml

try:
    from .executor import Executor
except ImportError:
    from executor import Executor


class Director:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.calendar = self.project_root / "config" / "calendar.yaml"

    def load_calendar(self) -> dict:
        if not self.calendar.exists():
            raise FileNotFoundError(
                f"Calendar file not found: {self.calendar}"
            )

        with self.calendar.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError("calendar.yaml content is empty or invalid.")

        if "schedule" not in data:
            raise ValueError(
                "calendar.yaml must contain a top-level 'schedule:' section."
            )

        return data

    def run(self) -> None:
        calendar = self.load_calendar()

        weekday = datetime.now().strftime("%A").lower()
        schedule = calendar["schedule"]

        if weekday not in schedule:
            raise ValueError(
                f"No schedule found for '{weekday}' in calendar.yaml."
            )

        today = schedule[weekday]

        required_fields = ["theme", "feed", "stories"]
        missing_fields = [
            field for field in required_fields
            if field not in today
        ]

        if missing_fields:
            raise ValueError(
                f"Missing fields for {weekday}: "
                + ", ".join(missing_fields)
            )

        print()
        print(f"Today   : {weekday}")
        print(f"Theme   : {today['theme']}")
        print(f"Feed    : {today['feed']}")
        print("Stories :")

        for story in today["stories"]:
            print(f"  - {story}")

        executor = Executor()
        executor.run(today)