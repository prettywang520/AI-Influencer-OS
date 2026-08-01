from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


class MotionEngineError(RuntimeError):
    """Raised when a suitable motion cannot be selected."""


class MotionEngine:
    def __init__(
        self,
        *,
        library_path: str | Path,
        rules_path: str | Path,
        history_path: str | Path,
        random_seed: int | None = None,
    ) -> None:
        self.library_path = Path(library_path).expanduser().resolve()
        self.rules_path = Path(rules_path).expanduser().resolve()
        self.history_path = Path(history_path).expanduser().resolve()
        self.random = random.Random(random_seed)

        self.library = self._read_yaml(self.library_path)
        self.rules_data = self._read_yaml(self.rules_path)

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.history_path.exists():
            self._write_history(
                {
                    "version": "2.0",
                    "history": {},
                }
            )

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise MotionEngineError(
                f"Motion YAML file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise MotionEngineError(
                f"Invalid motion YAML file: {path}"
            )

        return data

    def _read_history(self) -> dict[str, Any]:
        try:
            data = json.loads(
                self.history_path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise MotionEngineError(
                f"Invalid motion history: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise MotionEngineError(
                "Motion history must be a JSON object"
            )

        data.setdefault("version", "2.0")
        data.setdefault("history", {})

        return data

    def _write_history(
        self,
        data: dict[str, Any],
    ) -> None:
        temporary_path = self.history_path.with_suffix(
            ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            self.history_path
        )

    @staticmethod
    def _normalise(value: str) -> str:
        return (
            value.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _rules(self) -> dict[str, Any]:
        rules = self.rules_data.get("rules", {})

        if not isinstance(rules, dict):
            return {}

        return rules

    def _stage_categories(
        self,
        stage: str,
    ) -> list[str]:
        mapping = self.rules_data.get(
            "stage_categories",
            {},
        )

        if not isinstance(mapping, dict):
            return []

        categories = mapping.get(
            self._normalise(stage),
            [],
        )

        if not isinstance(categories, list):
            return []

        return [
            self._normalise(str(category))
            for category in categories
        ]

    def _recent_motion_ids(
        self,
        *,
        production_date: str,
    ) -> set[str]:
        history = self._read_history().get(
            "history",
            {},
        )

        if not isinstance(history, dict):
            return set()

        days = int(
            self._rules().get(
                "no_repeat_previous_days",
                3,
            )
        )

        target_date = date.fromisoformat(
            production_date
        )

        recent_ids: set[str] = set()

        for offset in range(1, days + 1):
            day = (
                target_date
                - timedelta(days=offset)
            ).isoformat()

            entry = history.get(day, {})

            if not isinstance(entry, dict):
                continue

            motions = entry.get("motions", [])

            if not isinstance(motions, list):
                continue

            for motion in motions:
                if isinstance(motion, dict):
                    motion_id = motion.get("id")
                else:
                    motion_id = motion

                if motion_id:
                    recent_ids.add(
                        self._normalise(
                            str(motion_id)
                        )
                    )

        return recent_ids

    def _candidates(
        self,
        *,
        stage: str,
        activity: str | None = None,
    ) -> list[dict[str, str]]:
        motions = self.library.get(
            "motions",
            {},
        )

        if not isinstance(motions, dict):
            raise MotionEngineError(
                "motion_library.yaml must contain motions"
            )

        categories = self._stage_categories(
            stage
        )

        activity_key = self._normalise(
            activity or ""
        )

        activity_aliases = {
            "luxury_hotel": "hotel",
            "night_market": "food",
            "street_food": "food",
            "coffee": "coffee_shop",
            "cafe": "coffee_shop",
            "old_town": "sightseeing",
            "tourism": "sightseeing",
        }

        activity_category = activity_aliases.get(
            activity_key,
            activity_key,
        )

        # 優先加入當前主題對應動作池
        if activity_category in motions:
            if activity_category not in categories:
                categories.insert(
                    0,
                    activity_category,
                )

        # Stage alias
        stage_key = self._normalise(stage)

        stage_aliases = {
            "check_in": "hotel",
            "room_discovery": "hotel",
            "room_reveal": "hotel",
            "room_entry": "hotel",
            "suite_discovery": "hotel",
            "lobby": "hotel",
            "breakfast": "hotel",
            "pool": "hotel",
            "spa": "hotel",

            "order": "coffee_shop",
            "ordering": "coffee_shop",
            "receive_drink": "coffee_shop",
            "first_sip": "coffee_shop",

            "first_bite": "food",
            "receive_food": "food",
            "taste": "food",

            "arrival": "arrival",
            "departure": "leaving",
            "checkout": "leaving",
            "check_out": "leaving",
            "leaving": "leaving",
        }

        stage_category = stage_aliases.get(
            stage_key
        )

        if (
            stage_category
            and stage_category in motions
            and stage_category not in categories
        ):
            categories.insert(
                0,
                stage_category,
            )

        # 最終 fallback：使用主題分類
        if not categories and activity_category in motions:
            categories = [
                activity_category
            ]

        # 最終安全 fallback：使用所有有效動作分類
        if not categories:
            categories = [
                category
                for category, values in motions.items()
                if isinstance(values, list) and values
            ]

        if not categories:
            raise MotionEngineError(
                "motion_library.yaml contains no usable motion categories"
            )

        candidates: list[dict[str, str]] = []
        seen: set[str] = set()

        for category in categories:
            category_motions = motions.get(
                category,
                [],
            )

            if not isinstance(category_motions, list):
                continue

            for motion in category_motions:
                if not isinstance(motion, dict):
                    continue

                motion_id = self._normalise(
                    str(motion.get("id", ""))
                )

                motion_text = str(
                    motion.get("text", "")
                ).strip()

                if not motion_id or not motion_text:
                    continue

                if motion_id in seen:
                    continue

                seen.add(motion_id)

                candidates.append(
                    {
                        "id": motion_id,
                        "text": motion_text,
                        "category": category,
                    }
                )

        return candidates

    def choose(
        self,
        *,
        production_date: str,
        stage: str,
        activity: str | None = None,
        used_today: set[str] | None = None,
        previous_category: str | None = None,
    ) -> dict[str, str]:
        used_today = {
            self._normalise(value)
            for value in (
                used_today or set()
            )
        }

        recent_ids = self._recent_motion_ids(
            production_date=production_date
        )

        candidates = self._candidates(
            stage=stage,
            activity=activity,
        )

        if not candidates:
            raise MotionEngineError(
                f"No motion candidates for stage: {stage}"
            )

        rules = self._rules()

        filtered = []

        for candidate in candidates:
            motion_id = candidate["id"]

            if (
                rules.get(
                    "no_repeat_within_day",
                    True,
                )
                and motion_id in used_today
            ):
                continue

            if (
                motion_id in recent_ids
                and rules.get(
                    "prefer_unused_motion",
                    True,
                )
            ):
                continue

            if (
                rules.get(
                    "no_same_category_consecutively",
                    True,
                )
                and previous_category
                and candidate["category"]
                == previous_category
            ):
                continue

            filtered.append(candidate)

        if not filtered:
            filtered = [
                candidate
                for candidate in candidates
                if candidate["id"]
                not in used_today
            ]

        if not filtered:
            filtered = candidates

        return self.random.choice(
            filtered
        )

    def remember_day(
        self,
        *,
        production_date: str,
        motions: list[dict[str, str]],
    ) -> None:
        data = self._read_history()
        history = data.setdefault(
            "history",
            {},
        )

        history[production_date] = {
            "motions": motions,
        }

        self._write_history(data)