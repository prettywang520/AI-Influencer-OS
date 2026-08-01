from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ReplyLoaderError(RuntimeError):
    """Raised when Reply Brain data cannot be loaded safely."""


class ReplyLoader:
    def __init__(self, reply_brain_root: str | Path) -> None:
        self.root = Path(reply_brain_root).expanduser().resolve()
        self.libraries_dir = self.root / "libraries"
        self.engines_dir = self.root / "engines"
        self.runtime_dir = self.root / "runtime"

        self._libraries: dict[str, dict[str, Any]] = {}
        self._engines: dict[str, dict[str, Any]] = {}
        self._runtime: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ReplyLoaderError(f"YAML file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ReplyLoaderError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ReplyLoaderError(f"Expected YAML mapping in {path}")

        return data

    def load_library(self, name: str) -> dict[str, Any]:
        if name in self._libraries:
            return self._libraries[name]

        path = self.libraries_dir / f"{name}.yaml"

        # Support the earlier flat reply_brain structure.
        if not path.exists():
            path = self.root / f"{name}.yaml"
            print(f"[ReplyLoader] loading library from: {path}")

        data = self._read_yaml(path)
        self._libraries[name] = data
        return data

    def load_engine(self, name: str) -> dict[str, Any]:
        if name in self._engines:
            return self._engines[name]

        path = self.engines_dir / f"{name}.yaml"
        data = self._read_yaml(path)
        self._engines[name] = data
        return data

    def load_runtime_config(self, name: str) -> dict[str, Any]:
        if name in self._runtime:
            return self._runtime[name]

        path = self.runtime_dir / f"{name}.yaml"
        data = self._read_yaml(path)
        self._runtime[name] = data
        return data

    def load_all_libraries(self) -> dict[str, dict[str, Any]]:
        search_locations = [self.libraries_dir, self.root]

        for directory in search_locations:
            if not directory.exists():
                continue

            for path in sorted(directory.glob("*.yaml")):
                data = self._read_yaml(path)
                library_name = str(
                    data.get("library")
                    or data.get("category")
                    or path.stem
                )

                if library_name not in self._libraries:
                    self._libraries[library_name] = data

        if not self._libraries:
            raise ReplyLoaderError(
                f"No reply libraries found under {self.root}"
            )

        return self._libraries.copy()

    def validate_required_files(self) -> list[str]:
        errors: list[str] = []

        required_libraries = {
            "compliments",
            "travel",
            "location",
            "food",
            "shopping",
            "hotel",
            "camera",
            "engagement",
            "japanese",
            "outfit",
            "flirting",
            "negative",
            "spam",
        }

        required_engines = {
            "intent_engine",
            "emotion_engine",
            "emoji_engine",
            "context_engine",
            "japanese_engine",
            "anti_repeat_engine",
        }

        for library in required_libraries:
            flat_path = self.root / f"{library}.yaml"
            nested_path = self.libraries_dir / f"{library}.yaml"

            if not flat_path.exists() and not nested_path.exists():
                errors.append(f"Missing library: {library}.yaml")

        for engine in required_engines:
            path = self.engines_dir / f"{engine}.yaml"

            if not path.exists():
                errors.append(f"Missing engine: engines/{engine}.yaml")

        return errors