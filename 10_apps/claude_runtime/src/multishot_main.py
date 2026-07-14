"""
AI Influencer OS — Multi-Shot Generator V1

Commands:

    # Show available shots
    python3 src/multishot_main.py luxury_hotel/pool --list-shots

    # Create all six prompt files
    python3 src/multishot_main.py luxury_hotel/pool --all

    # Create one selected shot prompt
    python3 src/multishot_main.py luxury_hotel/pool --shot cover

    # Create one prompt, copy it and open ChatGPT
    python3 src/multishot_main.py luxury_hotel/pool --shot cover --send
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

PLUGIN_ROOT = PROJECT_ROOT / "11_plugins"

IMAGE_REQUEST_DIR = (
    PROJECT_ROOT
    / "12_content"
    / "image_requests"
)

OUTPUT_MULTISHOT_DIR = (
    PROJECT_ROOT
    / "10_apps"
    / "claude_runtime"
    / "output"
    / "multishot_prompts"
)


if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


# ============================================================
# Internal imports
# ============================================================

from core.builders.prompt_builder import ImageBrief, PromptBuilder
from core.builders.shot_prompt_builder import (
    DEFAULT_SHOTS,
    ShotDefinition,
    ShotPromptBuilder,
)
from core.loaders.image_request_loader import ImageRequestLoader
from core.loaders.persona_loader import PersonaLoader


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create multiple editorial shot prompts "
            "from one image-request YAML file."
        )
    )

    parser.add_argument(
        "request",
        help=(
            "Request path without .yaml. "
            "Example: luxury_hotel/pool"
        ),
    )

    action_group = parser.add_mutually_exclusive_group(
        required=True
    )

    action_group.add_argument(
        "--all",
        action="store_true",
        help="Create prompt files for every available shot.",
    )

    action_group.add_argument(
        "--shot",
        metavar="NAME",
        help=(
            "Create one selected shot. "
            "Example: --shot cover"
        ),
    )

    action_group.add_argument(
        "--list-shots",
        action="store_true",
        help="Show all available shot names.",
    )

    parser.add_argument(
        "--send",
        action="store_true",
        help=(
            "Copy the selected shot prompt and open ChatGPT. "
            "Use together with --shot."
        ),
    )

    return parser.parse_args()


# ============================================================
# Path helpers
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


def normalize_request_parts(
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
    parts = normalize_request_parts(
        request_name
    )

    *folder_parts, file_stem = parts

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

    raise FileNotFoundError(
        "Image request was not found:\n"
        f"{'/'.join(parts)}"
    )


def build_output_path(
    request_path: Path,
    shot: ShotDefinition,
) -> Path:
    relative_path = request_path.relative_to(
        IMAGE_REQUEST_DIR
    )

    filename = (
        f"{relative_path.stem}_"
        f"{shot.number:02d}_"
        f"{shot.name}_prompt.txt"
    )

    return (
        OUTPUT_MULTISHOT_DIR
        / relative_path.parent
        / filename
    )


# ============================================================
# Prompt building
# ============================================================

def build_base_prompt(
    request_path: Path,
) -> tuple[
    dict,
    str,
    str,
]:
    request_loader = ImageRequestLoader(
        request_path
    )

    request = request_loader.load()

    persona_name = str(
        request["persona"]
    ).strip().lower()

    persona_path = (
        PROJECT_ROOT
        / "03_personas"
        / persona_name
    )

    persona_loader = PersonaLoader(
        persona_path
    )

    persona = persona_loader.load()

    brief = ImageBrief(
        location=str(request["location"]),
        scene=str(request["scene"]),
        outfit=str(request["outfit"]),
        pose=str(request["pose"]),
        expression=str(request["expression"]),
        lighting=str(request["lighting"]),
        camera_angle=str(
            request.get(
                "camera_angle",
                "eye level",
            )
        ),
        aspect_ratio=str(
            request.get(
                "aspect_ratio",
                "4:5",
            )
        ),
        platform=str(
            request.get(
                "platform",
                "Instagram",
            )
        ),
    )

    prompt_builder = PromptBuilder(
        persona
    )

    image_prompt = (
        prompt_builder.build_image_prompt(
            brief
        )
    )

    negative_prompt = (
        prompt_builder.build_negative_prompt()
    )

    return (
        request,
        image_prompt,
        negative_prompt,
    )


def build_shot_prompt(
    request: dict,
    image_prompt: str,
    negative_prompt: str,
    shot: ShotDefinition,
) -> str:
    shot_builder = ShotPromptBuilder()

    shot_image_prompt = shot_builder.build(
        base_image_prompt=image_prompt,
        request=request,
        shot=shot,
    )

    return f"""
Please generate one image based on the instructions below.

IMAGE PROMPT

{shot_image_prompt}

AVOID

{negative_prompt}

Generate the image directly.
Do not return, summarize or rewrite the prompt.
""".strip()


def save_shot_prompt(
    request_path: Path,
    shot: ShotDefinition,
    final_prompt: str,
) -> Path:
    output_path = build_output_path(
        request_path,
        shot,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        final_prompt,
        encoding="utf-8",
    )

    return output_path


# ============================================================
# Commands
# ============================================================

def list_shots() -> None:
    print("\nAvailable multi-shot directions")
    print("=" * 70)

    for shot in DEFAULT_SHOTS:
        print(
            f"{shot.number:02d}  "
            f"{shot.name:<12} "
            f"{shot.framing}"
        )


def generate_one_shot(
    request_path: Path,
    shot_name: str,
    send: bool,
) -> None:
    shot_builder = ShotPromptBuilder()

    shot = shot_builder.get_shot(
        shot_name
    )

    request, image_prompt, negative_prompt = (
        build_base_prompt(
            request_path
        )
    )

    final_prompt = build_shot_prompt(
        request=request,
        image_prompt=image_prompt,
        negative_prompt=negative_prompt,
        shot=shot,
    )

    output_path = save_shot_prompt(
        request_path=request_path,
        shot=shot,
        final_prompt=final_prompt,
    )

    print("\n" + "=" * 70)
    print("MULTI-SHOT PROMPT CREATED")
    print("=" * 70)
    print(f"Shot: {shot.number:02d} — {shot.name}")
    print(f"Saved: {output_path}")

    if not send:
        print("\nTo send this shot to ChatGPT, run:")
        print(
            "python3 src/multishot_main.py "
            f"{request_path.relative_to(IMAGE_REQUEST_DIR).with_suffix('').as_posix()} "
            f"--shot {shot.name} --send"
        )
        return

    from chatgpt_connector.plugin import ChatGPTConnector

    connector = ChatGPTConnector()

    connector.send(
        final_prompt
    )


def generate_all_shots(
    request_path: Path,
) -> None:
    request, image_prompt, negative_prompt = (
        build_base_prompt(
            request_path
        )
    )

    saved_files: list[Path] = []

    for shot in DEFAULT_SHOTS:
        final_prompt = build_shot_prompt(
            request=request,
            image_prompt=image_prompt,
            negative_prompt=negative_prompt,
            shot=shot,
        )

        output_path = save_shot_prompt(
            request_path=request_path,
            shot=shot,
            final_prompt=final_prompt,
        )

        saved_files.append(
            output_path
        )

    request_name = (
        request_path
        .relative_to(IMAGE_REQUEST_DIR)
        .with_suffix("")
        .as_posix()
    )

    print("\n" + "=" * 70)
    print("MULTI-SHOT SERIES CREATED")
    print("=" * 70)
    print(f"Request: {request_name}")
    print(f"Total prompts: {len(saved_files)}")

    print("\nSaved files:")

    for output_path in saved_files:
        print(f"- {output_path}")

    print("\nSend one shot to ChatGPT, for example:")

    print(
        "python3 src/multishot_main.py "
        f"{request_name} --shot cover --send"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_arguments()

    if args.send and not args.shot:
        raise ValueError(
            "--send must be used together with --shot."
        )

    if args.list_shots:
        list_shots()
        return

    request_path = resolve_request_path(
        args.request
    )

    if args.all:
        generate_all_shots(
            request_path
        )
        return

    if args.shot:
        generate_one_shot(
            request_path=request_path,
            shot_name=args.shot,
            send=args.send,
        )
        return


if __name__ == "__main__":
    main()