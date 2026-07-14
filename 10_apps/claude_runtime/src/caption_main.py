"""
AI Influencer OS — Caption Generator V1

Usage:

    python3 src/caption_main.py airport/lounge
    python3 src/caption_main.py coffee_shop/latte
    python3 src/caption_main.py luxury_hotel/pool
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PLUGIN_ROOT = PROJECT_ROOT / "05_plugins"

IMAGE_REQUEST_DIR = (
    PROJECT_ROOT
    / "08_content"
    / "image_requests"
)

OUTPUT_CAPTION_DIR = (
    PROJECT_ROOT
    / "10_apps"
    / "claude_runtime"
    / "output"
    / "captions"
)

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

# ============================================================
# Internal imports
# ============================================================

from core.builders.caption_prompt_builder import CaptionPromptBuilder
from core.loaders.image_request_loader import ImageRequestLoader
from core.loaders.persona_loader import PersonaLoader


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an Instagram caption prompt "
            "from an image-request YAML file."
        )
    )

    parser.add_argument(
        "request",
        help=(
            "Request path without .yaml. "
            "Example: airport/lounge"
        )
    )

    return parser.parse_args()


# ============================================================
# Request path helpers
# ============================================================

def normalize_segment(value: str) -> str:
    clean_value = value.strip().lower()

    clean_value = re.sub(
        r"[\s\-]+",
        "_",
        clean_value,
    )

    clean_value = re.sub(
        r"[^a-z0-9_]",
        "",
        clean_value,
    )

    clean_value = re.sub(
        r"_+",
        "_",
        clean_value,
    ).strip("_")

    if not clean_value:
        raise ValueError(
            f"Invalid request path segment: {value!r}"
        )

    return clean_value


def normalize_request_path(
    request_name: str,
) -> tuple[str, ...]:
    prepared_name = (
        request_name.strip()
        .replace("\\", "/")
    )

    prepared_name = re.sub(
        r"\.(yaml|yml)$",
        "",
        prepared_name,
        flags=re.IGNORECASE,
    )

    path_object = PurePosixPath(
        prepared_name
    )

    parts: list[str] = []

    for part in path_object.parts:
        if part in {"", ".", "/"}:
            continue

        if part == "..":
            raise ValueError(
                "Parent-folder references are not allowed."
            )

        parts.append(
            normalize_segment(part)
        )

    if not parts:
        raise ValueError(
            "Request path cannot be empty."
        )

    return tuple(parts)


def resolve_request_path(
    request_name: str,
) -> Path:
    request_parts = normalize_request_path(
        request_name
    )

    *folder_parts, file_stem = request_parts

    yaml_path = (
        IMAGE_REQUEST_DIR
        / Path(*folder_parts)
        / f"{file_stem}.yaml"
    )

    yml_path = (
        IMAGE_REQUEST_DIR
        / Path(*folder_parts)
        / f"{file_stem}.yml"
    )

    if yaml_path.exists():
        return yaml_path

    if yml_path.exists():
        return yml_path

    normalized_name = "/".join(
        request_parts
    )

    raise FileNotFoundError(
        "Image request was not found:\n"
        f"{normalized_name}"
    )


def build_output_path(
    request_path: Path,
) -> Path:
    relative_path = request_path.relative_to(
        IMAGE_REQUEST_DIR
    )

    return (
        OUTPUT_CAPTION_DIR
        / relative_path.parent
        / f"{relative_path.stem}_caption_prompt.txt"
    )


# ============================================================
# Caption workflow
# ============================================================

def run_caption_request(
    request_path: Path,
) -> None:
    request_loader = ImageRequestLoader(request_path)
    request = request_loader.load()

    persona_name = str(
        request["persona"]
    ).strip().lower()

    persona_path = (
        PROJECT_ROOT
        / "03_personas"
        / persona_name
    )

    persona_loader = PersonaLoader(persona_path)
    persona = persona_loader.load()

    caption_builder = CaptionPromptBuilder(persona)
    caption_prompt = caption_builder.build(request)

    output_path = build_output_path(request_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        caption_prompt,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("CAPTION PROMPT")
    print("=" * 70)
    print(caption_prompt)

    print("\n" + "=" * 70)
    print("SAVED CAPTION PROMPT")
    print("=" * 70)
    print(output_path)

    from chatgpt_connector.plugin import ChatGPTConnector

    connector = ChatGPTConnector()
    connector.send(caption_prompt)

    # 5. Copy prompt and open ChatGPT.
    


def main() -> None:
    args = parse_arguments()

    request_path = resolve_request_path(
        args.request
    )

    run_caption_request(
        request_path
    )


if __name__ == "__main__":
    main()