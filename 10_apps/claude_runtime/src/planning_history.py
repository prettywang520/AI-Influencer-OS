from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .content_models import DailyContentPlan


class PlanningHistoryError(RuntimeError):
    """Raised when planning history cannot be read or written safely."""


@dataclass(slots=True)
class PlanningHistoryEntry:
    production_date: str
    country: str
    city: str
    venue: str
    theme: str

    behaviors: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    interactions: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    story_stages: list[str] = field(default_factory=list)

    feed_behavior: str | None = None
    feed_emotion: str | None = None
    feed_interaction: str | None = None
    feed_camera: str | None = None

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecentPlanningContext:
    cities: set[str] = field(default_factory=set)
    venues: set[str] = field(default_factory=set)
    themes: set[str] = field(default_factory=set)
    behaviors: set[str] = field(default_factory=set)
    emotions: set[str] = field(default_factory=set)
    interactions: set[str] = field(default_factory=set)
    cameras: set[str] = field(default_factory=set)
    story_stages: set[str] = field(default_factory=set)


class PlanningHistory:
    """
    Persistent history for AIKO Dynamic Content Planning.

    It prevents recent content from repeatedly using the same:

    - city
    - venue
    - theme
    - behavior
    - emotion
    - interaction
    - camera story
    - story stage
    """

    def __init__(
        self,
        history_path: str | Path,
        *,
        maximum_entries: int = 365,
    ) -> None:
        self.path = Path(history_path).expanduser().resolve()
        self.maximum_entries = maximum_entries
        self._lock = Lock()

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.path.exists():
            self._write(
                {
                    "version": "1.0",
                    "history": [],
                }
            )

    @staticmethod
    def _normalise(value: str | None) -> str:
        if not value:
            return ""

        return (
            value.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _parse_date(value: str) -> date | None:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    def _read(self) -> dict[str, Any]:
        try:
            with self.path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except FileNotFoundError:
            return {
                "version": "1.0",
                "history": [],
            }
        except json.JSONDecodeError as exc:
            raise PlanningHistoryError(
                f"Invalid JSON in planning history: {exc}"
            ) from exc
        except OSError as exc:
            raise PlanningHistoryError(
                f"Unable to read planning history: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise PlanningHistoryError(
                "Planning history root must be a JSON object"
            )

        history = data.get("history")

        if not isinstance(history, list):
            raise PlanningHistoryError(
                "Planning history must contain a 'history' list"
            )

        return data

    def _write(
        self,
        data: dict[str, Any],
    ) -> None:
        temporary_path = self.path.with_suffix(".tmp")

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            temporary_path.replace(self.path)

        except OSError as exc:
            raise PlanningHistoryError(
                f"Unable to write planning history: {exc}"
            ) from exc

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()

        history = data.get("history", [])

        return [
            item
            for item in history
            if isinstance(item, dict)
        ]

    def last(
        self,
        count: int = 7,
    ) -> list[dict[str, Any]]:
        if count < 1:
            return []

        history = self.entries()

        history.sort(
            key=lambda item: str(
                item.get("production_date", "")
            ),
            reverse=True,
        )

        return history[:count]

    def recent_days(
        self,
        *,
        reference_date: str,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        target_date = self._parse_date(reference_date)

        if target_date is None:
            raise PlanningHistoryError(
                f"Invalid reference date: {reference_date}"
            )

        earliest_date = target_date - timedelta(days=days)

        output: list[dict[str, Any]] = []

        for item in self.entries():
            production_date = self._parse_date(
                str(item.get("production_date", ""))
            )

            if production_date is None:
                continue

            if earliest_date <= production_date < target_date:
                output.append(item)

        output.sort(
            key=lambda item: str(
                item.get("production_date", "")
            ),
            reverse=True,
        )

        return output

    def recent_context(
        self,
        *,
        reference_date: str,
        days: int = 7,
    ) -> RecentPlanningContext:
        context = RecentPlanningContext()

        for item in self.recent_days(
            reference_date=reference_date,
            days=days,
        ):
            city = self._normalise(
                str(item.get("city", ""))
            )
            venue = self._normalise(
                str(item.get("venue", ""))
            )
            theme = self._normalise(
                str(item.get("theme", ""))
            )

            if city:
                context.cities.add(city)

            if venue:
                context.venues.add(venue)

            if theme:
                context.themes.add(theme)

            for behavior in item.get("behaviors", []):
                value = self._normalise(str(behavior))

                if value:
                    context.behaviors.add(value)

            for emotion in item.get("emotions", []):
                value = self._normalise(str(emotion))

                if value:
                    context.emotions.add(value)

            for interaction in item.get(
                "interactions",
                [],
            ):
                value = self._normalise(
                    str(interaction)
                )

                if value:
                    context.interactions.add(value)

            for camera in item.get("cameras", []):
                value = self._normalise(str(camera))

                if value:
                    context.cameras.add(value)

            for stage in item.get(
                "story_stages",
                [],
            ):
                value = self._normalise(str(stage))

                if value:
                    context.story_stages.add(value)

        return context

    def add_entry(
        self,
        entry: PlanningHistoryEntry,
        *,
        replace_same_date: bool = True,
    ) -> None:
        with self._lock:
            data = self._read()
            history = data.get("history", [])

            if replace_same_date:
                history = [
                    item
                    for item in history
                    if str(
                        item.get("production_date", "")
                    )
                    != entry.production_date
                ]

            history.append(entry.to_dict())

            history.sort(
                key=lambda item: str(
                    item.get("production_date", "")
                )
            )

            data["history"] = history[
                -self.maximum_entries:
            ]

            self._write(data)

    def remember_plan(
        self,
        plan: DailyContentPlan,
    ) -> PlanningHistoryEntry:
        moments = [
            plan.feed,
            *plan.stories,
        ]

        behaviors = [
            self._normalise(moment.behavior)
            for moment in moments
            if moment.behavior
        ]

        emotions = [
            self._normalise(moment.emotion)
            for moment in moments
            if moment.emotion
        ]

        interactions = [
            self._normalise(moment.interaction)
            for moment in moments
            if moment.interaction
        ]

        cameras = [
            self._normalise(moment.camera_story)
            for moment in moments
            if moment.camera_story
        ]

        story_stages = [
            self._normalise(moment.story_stage)
            for moment in moments
            if moment.story_stage
        ]

        for scene in plan.reel_scenes:
            behaviors.append(
                self._normalise(scene.behavior)
            )
            emotions.append(
                self._normalise(scene.emotion)
            )
            interactions.append(
                self._normalise(scene.interaction)
            )
            cameras.append(
                self._normalise(scene.camera_story)
            )

        entry = PlanningHistoryEntry(
            production_date=plan.date,
            country=plan.country,
            city=plan.city,
            venue=plan.venue,
            theme=plan.theme,
            behaviors=self._deduplicate(behaviors),
            emotions=self._deduplicate(emotions),
            interactions=self._deduplicate(
                interactions
            ),
            cameras=self._deduplicate(cameras),
            story_stages=self._deduplicate(
                story_stages
            ),
            feed_behavior=self._normalise(
                plan.feed.behavior
            ),
            feed_emotion=self._normalise(
                plan.feed.emotion
            ),
            feed_interaction=self._normalise(
                plan.feed.interaction
            ),
            feed_camera=self._normalise(
                plan.feed.camera_story
            ),
        )

        self.add_entry(entry)
        return entry

    @staticmethod
    def _deduplicate(
        values: list[str],
    ) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()

        for value in values:
            if not value or value in seen:
                continue

            seen.add(value)
            output.append(value)

        return output

    def was_city_used_recently(
        self,
        city: str,
        *,
        reference_date: str,
        days: int = 3,
    ) -> bool:
        context = self.recent_context(
            reference_date=reference_date,
            days=days,
        )

        return self._normalise(city) in context.cities

    def was_venue_used_recently(
        self,
        venue: str,
        *,
        reference_date: str,
        days: int = 30,
    ) -> bool:
        context = self.recent_context(
            reference_date=reference_date,
            days=days,
        )

        return self._normalise(venue) in context.venues

    def was_theme_used_recently(
        self,
        theme: str,
        *,
        reference_date: str,
        days: int = 7,
    ) -> bool:
        context = self.recent_context(
            reference_date=reference_date,
            days=days,
        )

        return self._normalise(theme) in context.themes

    def clear(self) -> None:
        with self._lock:
            self._write(
                {
                    "version": "1.0",
                    "history": [],
                }
            )

    def remove_date(
        self,
        production_date: str,
    ) -> bool:
        with self._lock:
            data = self._read()
            history = data.get("history", [])

            original_count = len(history)

            history = [
                item
                for item in history
                if str(
                    item.get("production_date", "")
                )
                != production_date
            ]

            data["history"] = history
            self._write(data)

        return len(history) < original_count

    def status(self) -> dict[str, Any]:
        history = self.entries()

        latest_date = None

        if history:
            latest_date = max(
                str(
                    item.get(
                        "production_date",
                        "",
                    )
                )
                for item in history
            )

        return {
            "path": str(self.path),
            "entry_count": len(history),
            "latest_date": latest_date,
            "maximum_entries": self.maximum_entries,
        }


def build_planning_history() -> PlanningHistory:
    """
    Build PlanningHistory using:

    10_apps/claude_runtime/output/planning_history.json
    """
    runtime_root = Path(__file__).resolve().parents[1]

    return PlanningHistory(
        runtime_root
        / "output"
        / "planning_history.json"
    )