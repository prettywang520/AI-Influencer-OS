from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PlanningLoaderError(RuntimeError):
    """Raised when planning data cannot be loaded or validated."""


class PlanningLoader:
    """
    Central loader for AIKO dynamic-planning YAML files.

    All planning modules should load YAML through this class instead of
    opening files directly.

    Expected planning directory:

    03_personas/aiko/planning/
    ├── calendar.yaml
    ├── location_rotation.yaml
    ├── behavior_database.yaml
    ├── camera_database.yaml
    ├── emotion_database.yaml
    ├── interaction_database.yaml
    ├── weather_rules.yaml          optional
    ├── seasonal_rules.yaml         optional
    └── story_templates.yaml        optional
    """

    REQUIRED_FILES: dict[str, str] = {
        "calendar": "calendar.yaml",
        "locations": "location_rotation.yaml",
        "behaviors": "behavior_database.yaml",
        "cameras": "camera_database.yaml",
        "emotions": "emotion_database.yaml",
        "interactions": "interaction_database.yaml",
    }

    OPTIONAL_FILES: dict[str, str] = {
        "weather_rules": "weather_rules.yaml",
        "seasonal_rules": "seasonal_rules.yaml",
        "story_templates": "story_templates.yaml",
    }

    def __init__(self, planning_root: str | Path) -> None:
        self.root = Path(planning_root).expanduser().resolve()
        self._cache: dict[str, dict[str, Any]] = {}

        if not self.root.exists():
            raise PlanningLoaderError(
                f"Planning directory does not exist: {self.root}"
            )

        if not self.root.is_dir():
            raise PlanningLoaderError(
                f"Planning root is not a directory: {self.root}"
            )

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise PlanningLoaderError(
                f"Planning YAML file not found: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise PlanningLoaderError(
                f"Invalid YAML in {path}: {exc}"
            ) from exc
        except OSError as exc:
            raise PlanningLoaderError(
                f"Unable to read {path}: {exc}"
            ) from exc

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise PlanningLoaderError(
                f"Expected a YAML mapping in {path}, "
                f"received {type(data).__name__}"
            )

        return data

    def _load(
        self,
        key: str,
        filename: str,
        *,
        required: bool,
    ) -> dict[str, Any]:
        if key in self._cache:
            return self._cache[key]

        path = self.root / filename

        if not path.exists() and not required:
            self._cache[key] = {}
            return {}

        data = self._read_yaml(path)
        self._cache[key] = data
        return data

    def calendar(self) -> dict[str, Any]:
        return self._load(
            "calendar",
            self.REQUIRED_FILES["calendar"],
            required=True,
        )

    def locations(self) -> dict[str, Any]:
        return self._load(
            "locations",
            self.REQUIRED_FILES["locations"],
            required=True,
        )

    def behaviors(self) -> dict[str, Any]:
        return self._load(
            "behaviors",
            self.REQUIRED_FILES["behaviors"],
            required=True,
        )

    def cameras(self) -> dict[str, Any]:
        return self._load(
            "cameras",
            self.REQUIRED_FILES["cameras"],
            required=True,
        )

    def emotions(self) -> dict[str, Any]:
        return self._load(
            "emotions",
            self.REQUIRED_FILES["emotions"],
            required=True,
        )

    def interactions(self) -> dict[str, Any]:
        return self._load(
            "interactions",
            self.REQUIRED_FILES["interactions"],
            required=True,
        )

    def weather_rules(self) -> dict[str, Any]:
        return self._load(
            "weather_rules",
            self.OPTIONAL_FILES["weather_rules"],
            required=False,
        )

    def seasonal_rules(self) -> dict[str, Any]:
        return self._load(
            "seasonal_rules",
            self.OPTIONAL_FILES["seasonal_rules"],
            required=False,
        )

    def story_templates(self) -> dict[str, Any]:
        return self._load(
            "story_templates",
            self.OPTIONAL_FILES["story_templates"],
            required=False,
        )

    def load_all(self) -> dict[str, dict[str, Any]]:
        """
        Load all required and optional planning databases.

        Optional files return an empty dictionary when they do not exist.
        """
        return {
            "calendar": self.calendar(),
            "locations": self.locations(),
            "behaviors": self.behaviors(),
            "cameras": self.cameras(),
            "emotions": self.emotions(),
            "interactions": self.interactions(),
            "weather_rules": self.weather_rules(),
            "seasonal_rules": self.seasonal_rules(),
            "story_templates": self.story_templates(),
        }

    def validate_required_files(self) -> list[str]:
        """
        Return validation errors without raising an exception.
        """
        errors: list[str] = []

        for key, filename in self.REQUIRED_FILES.items():
            path = self.root / filename

            if not path.exists():
                errors.append(
                    f"missing_required_file:{filename}"
                )
                continue

            try:
                data = self._read_yaml(path)
            except PlanningLoaderError as exc:
                errors.append(str(exc))
                continue

            if not data:
                errors.append(
                    f"empty_required_file:{filename}"
                )

            if not isinstance(data, dict):
                errors.append(
                    f"invalid_required_file:{filename}"
                )

        return errors

    def file_status(self) -> dict[str, dict[str, Any]]:
        """
        Return a readable status report for all planning YAML files.
        """
        status: dict[str, dict[str, Any]] = {}

        combined_files = {
            **self.REQUIRED_FILES,
            **self.OPTIONAL_FILES,
        }

        for key, filename in combined_files.items():
            path = self.root / filename
            required = key in self.REQUIRED_FILES

            status[key] = {
                "filename": filename,
                "path": str(path),
                "required": required,
                "exists": path.exists(),
                "cached": key in self._cache,
            }

        return status

    def clear_cache(self, key: str | None = None) -> None:
        """
        Clear one cached dataset or the entire cache.

        Useful after editing YAML files while the runtime is active.
        """
        if key is None:
            self._cache.clear()
            return

        self._cache.pop(key, None)

    def reload(self, key: str) -> dict[str, Any]:
        """
        Reload one planning dataset from disk.
        """
        all_files = {
            **self.REQUIRED_FILES,
            **self.OPTIONAL_FILES,
        }

        if key not in all_files:
            raise PlanningLoaderError(
                f"Unknown planning dataset: {key}"
            )

        self.clear_cache(key)

        return self._load(
            key,
            all_files[key],
            required=key in self.REQUIRED_FILES,
        )


def build_planning_loader() -> PlanningLoader:
    """
    Build PlanningLoader from the AI-Influencer-OS repository structure.

    Current file:

    AI-Influencer-OS/
    └── 10_apps/
        └── claude_runtime/
            └── src/
                └── planning_loader.py

    Therefore parents[3] resolves to the repository root.
    """
    project_root = Path(__file__).resolve().parents[3]

    planning_root = (
        project_root
        / "03_personas"
        / "aiko"
        / "planning"
    )

    return PlanningLoader(planning_root)