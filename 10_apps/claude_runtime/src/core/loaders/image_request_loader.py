"""
Load image-generation requests from YAML files.
"""

from pathlib import Path
from typing import Any

import yaml


class ImageRequestLoader:
    """Load and validate one image request YAML file."""

    REQUIRED_FIELDS = (
        "persona",
        "location",
        "scene",
        "outfit",
        "pose",
        "expression",
        "lighting",
    )

    def __init__(self, request_path: str | Path) -> None:
        self.request_path = Path(request_path).resolve()

    def load(self) -> dict[str, Any]:
        if not self.request_path.exists():
            raise FileNotFoundError(
                f"Image request file not found: {self.request_path}"
            )

        if self.request_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                f"Image request must be YAML: {self.request_path}"
            )

        with self.request_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Image request must contain a YAML object: "
                f"{self.request_path}"
            )

        missing_fields = [
            field
            for field in self.REQUIRED_FIELDS
            if not str(data.get(field, "")).strip()
        ]

        if missing_fields:
            raise ValueError(
                "Missing image request fields: "
                + ", ".join(missing_fields)
            )

        return data