"""
AI Influencer OS — Image Request CLI

Supported commands:

    # List all available requests
    python3 src/main.py --list

    # Run a top-level request
    python3 src/main.py coffee_shop

    # Run a nested scene request
    python3 src/main.py airport/lounge
    python3 src/main.py luxury_hotel/pool

    # Create a top-level request
    python3 src/main.py --new airport

    # Create a nested scene request
    python3 src/main.py --new airport lounge
    python3 src/main.py --new luxury_hotel spa

Workflow:

1. Read an image-request YAML file
2. Load the selected persona
3. Build the image prompt
4. Save the prompt locally
5. Copy the final prompt to the clipboard
6. Open ChatGPT
"""

from __future__ import annotations

import argparse
import re
import shutil
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
)

IMAGE_REQUEST_TEMPLATE = (
    IMAGE_REQUEST_DIR
    / "_template.yaml"
)

OUTPUT_PROMPT_DIR = (
    PROJECT_ROOT
    / "10_apps"
    / "claude_runtime"
    / "output"
    / "prompts"
    
)
SCENE_SERIES: dict[str, tuple[str, ...]] = {
    "airport": (
        "arrival",
        "boarding",
        "immigration",
        "lounge",
    ),
    "coffee_shop": (
        "latte",
        "outdoor",
        "reading",
        "window_seat",
    ),
    "luxury_hotel": (
        "lobby",
        "suite",
        "pool",
        "rooftop",
        "restaurant",
        "spa",
        "breakfast",
    ),
    "beach": (
        "sunrise",
        "sunset",
        "resort",
    ),
    "yacht": (
        "deck",
        "sunset",
        "dinner",
    ),
}

# Make project plugins importable.
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


# ============================================================
# Internal imports
# ============================================================

from core.builders.prompt_builder import ImageBrief, PromptBuilder
from core.loaders.image_request_loader import ImageRequestLoader
from core.loaders.persona_loader import PersonaLoader


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create, list and run AI Influencer image requests."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 src/main.py --list\n"
            "  python3 src/main.py coffee_shop\n"
            "  python3 src/main.py airport/lounge\n"
            "  python3 src/main.py --new airport\n"
            "  python3 src/main.py --new airport lounge\n"
        ),
    )

    parser.add_argument(
        "request",
        nargs="?",
        help=(
            "Request path without .yaml.\n"
            "Examples: coffee_shop, airport/lounge"
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available image requests.",
    )

    parser.add_argument(
        "--new",
        nargs="+",
        metavar="PATH",
        help=(
            "Create a request from _template.yaml.\n"
            "Examples:\n"
            "  --new airport\n"
            "  --new airport lounge\n"
            "  --new airport/lounge"
        ),
    )

    return parser.parse_args()


# ============================================================
# Request-name helpers
# ============================================================

def normalize_path_segment(value: str) -> str:
    """
    Convert one folder or filename segment into safe snake_case.

    Examples:

        Airport Lounge -> airport_lounge
        luxury-hotel   -> luxury_hotel
        Coffee_Shop    -> coffee_shop
    """

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
            f"Invalid request-name segment: {value!r}"
        )

    return clean_value


def normalize_request_parts(
    values: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """
    Normalize a request path.

    Both of these forms are supported:

        ["airport", "lounge"]
        ["airport/lounge"]

    The result will be:

        ("airport", "lounge")
    """

    raw_parts: list[str] = []

    for value in values:
        prepared_value = (
            value.strip()
            .replace("\\", "/")
        )

        prepared_value = re.sub(
            r"\.(yaml|yml)$",
            "",
            prepared_value,
            flags=re.IGNORECASE,
        )

        path_object = PurePosixPath(
            prepared_value
        )

        for part in path_object.parts:
            if part in {"", ".", "/"}:
                continue

            if part == "..":
                raise ValueError(
                    "Parent-folder references are not allowed."
                )

            raw_parts.append(part)

    if not raw_parts:
        raise ValueError(
            "Request name cannot be empty."
        )

    return tuple(
        normalize_path_segment(part)
        for part in raw_parts
    )


def request_parts_to_relative_path(
    request_parts: tuple[str, ...],
    suffix: str = ".yaml",
) -> Path:
    """Convert normalized parts into a relative request path."""

    if not request_parts:
        raise ValueError(
            "At least one request-name segment is required."
        )

    *folder_parts, file_stem = request_parts

    return (
        Path(*folder_parts)
        / f"{file_stem}{suffix}"
    )


def relative_path_without_suffix(
    file_path: Path,
) -> str:
    """
    Return a request path relative to IMAGE_REQUEST_DIR.

    Example:

        airport/lounge.yaml -> airport/lounge
    """

    relative_path = file_path.relative_to(
        IMAGE_REQUEST_DIR
    )

    return relative_path.with_suffix(
        ""
    ).as_posix()


# ============================================================
# Available-request discovery
# ============================================================

def is_usable_request_file(
    file_path: Path,
) -> bool:
    """
    Return True when a YAML file should appear in --list.

    Files or folders beginning with an underscore are excluded.
    """

    if file_path.suffix.lower() not in {
        ".yaml",
        ".yml",
    }:
        return False

    try:
        relative_path = file_path.relative_to(
            IMAGE_REQUEST_DIR
        )
    except ValueError:
        return False

    return not any(
        part.startswith("_")
        for part in relative_path.parts
    )


def get_available_request_files() -> list[Path]:
    """
    Find top-level and nested YAML request files.

    Examples:

        coffee_shop.yaml
        airport/lounge.yaml
        luxury_hotel/pool.yaml
    """

    if not IMAGE_REQUEST_DIR.exists():
        return []

    request_files = [
        file_path
        for file_path in IMAGE_REQUEST_DIR.rglob("*")
        if file_path.is_file()
        and is_usable_request_file(file_path)
    ]

    return sorted(
        request_files,
        key=lambda file_path: (
            relative_path_without_suffix(
                file_path
            )
        ),
    )


# ============================================================
# List requests
# ============================================================

def list_requests() -> None:
    """Display all available top-level and nested requests."""

    if not IMAGE_REQUEST_DIR.exists():
        raise FileNotFoundError(
            "Image-request folder was not found:\n"
            f"{IMAGE_REQUEST_DIR}"
        )

    request_files = get_available_request_files()

    print("\nAvailable image requests")
    print("=" * 60)

    if not request_files:
        print("No completed image requests were found.")
        return

    grouped_requests: dict[str, list[str]] = {}
    root_requests: list[str] = []

    for request_file in request_files:
        relative_name = (
            relative_path_without_suffix(
                request_file
            )
        )

        parts = relative_name.split("/")

        if len(parts) == 1:
            root_requests.append(parts[0])
            continue

        category = parts[0]
        scene = "/".join(parts[1:])

        grouped_requests.setdefault(
            category,
            [],
        ).append(scene)

    if root_requests:
        print("\nTop-level requests")
        print("-" * 60)

        for request_name in root_requests:
            print(f"- {request_name}")

    if grouped_requests:
        print("\nScene library")
        print("-" * 60)

        for category in sorted(grouped_requests):
            print(f"\n{category}")

            scenes = sorted(
                grouped_requests[category]
            )

            for index, scene in enumerate(scenes):
                is_last = index == len(scenes) - 1

                branch = (
                    "└──"
                    if is_last
                    else "├──"
                )

                print(f"  {branch} {scene}")

    print()


# ============================================================
# Create new request
# ============================================================

def create_new_request(
    request_values: list[str],
) -> list[Path]:
    """
    Create one request or a complete preset scene series.

    Examples:

        --new airport
            Creates the complete airport series.

        --new luxury_hotel
            Creates the complete luxury-hotel series.

        --new airport duty_free
            Creates only airport/duty_free.yaml.

        --new airport/duty_free
            Creates only airport/duty_free.yaml.
    """

    if not IMAGE_REQUEST_TEMPLATE.exists():
        raise FileNotFoundError(
            "Image-request template was not found:\n"
            f"{IMAGE_REQUEST_TEMPLATE}"
        )

    request_parts = normalize_request_parts(
        request_values
    )

    created_files: list[Path] = []
    skipped_files: list[Path] = []

    # One category name: create the complete preset series.
    if (
        len(request_parts) == 1
        and request_parts[0] in SCENE_SERIES
    ):
        category = request_parts[0]
        category_dir = IMAGE_REQUEST_DIR / category

        category_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for scene_name in SCENE_SERIES[category]:
            scene_path = (
                category_dir
                / f"{scene_name}.yaml"
            )

            if scene_path.exists():
                skipped_files.append(scene_path)
                continue

            shutil.copyfile(
                IMAGE_REQUEST_TEMPLATE,
                scene_path,
            )

            created_files.append(scene_path)

        print("\n" + "=" * 70)
        print("SCENE SERIES CREATED")
        print("=" * 70)
        print(f"Category: {category}")

        if created_files:
            print("\nCreated:")

            for file_path in created_files:
                print(
                    "- "
                    + file_path.relative_to(
                        IMAGE_REQUEST_DIR
                    ).as_posix()
                )

        if skipped_files:
            print("\nAlready existed, skipped:")

            for file_path in skipped_files:
                print(
                    "- "
                    + file_path.relative_to(
                        IMAGE_REQUEST_DIR
                    ).as_posix()
                )

        print("\nNext steps:")
        print("1. Open each new YAML file.")
        print("2. Complete the location, scene, outfit and pose.")
        print("3. Save the YAML file.")
        print(
            "4. Run a scene, for example:\n"
            f"   python3 src/main.py {category}/lobby"
        )

        return created_files

    # More than one path segment: create one individual scene.
    relative_request_path = (
        request_parts_to_relative_path(
            request_parts
        )
    )

    new_request_path = (
        IMAGE_REQUEST_DIR
        / relative_request_path
    )

    if new_request_path.exists():
        raise FileExistsError(
            "Image request already exists:\n"
            f"{new_request_path}"
        )

    new_request_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        IMAGE_REQUEST_TEMPLATE,
        new_request_path,
    )

    request_command_name = (
        relative_request_path
        .with_suffix("")
        .as_posix()
    )

    print("\n" + "=" * 70)
    print("NEW IMAGE REQUEST CREATED")
    print("=" * 70)
    print(new_request_path)

    print("\nNext steps:")
    print(f"1. Open: {new_request_path.name}")
    print("2. Complete all YAML fields.")
    print("3. Save the YAML file.")
    print(
        "4. Run:\n"
        f"   python3 src/main.py {request_command_name}"
    )

    return [new_request_path]

# ============================================================
# Resolve request path
# ============================================================

def resolve_request_path(
    request_name: str,
) -> Path:
    """
    Find a top-level or nested image request.

    Examples:

        coffee_shop
        airport/lounge
        luxury_hotel/pool
    """

    request_parts = normalize_request_parts(
        [request_name]
    )

    yaml_relative_path = (
        request_parts_to_relative_path(
            request_parts,
            suffix=".yaml",
        )
    )

    yml_relative_path = (
        request_parts_to_relative_path(
            request_parts,
            suffix=".yml",
        )
    )

    yaml_path = (
        IMAGE_REQUEST_DIR
        / yaml_relative_path
    )

    yml_path = (
        IMAGE_REQUEST_DIR
        / yml_relative_path
    )

    if yaml_path.exists() and yaml_path.is_file():
        return yaml_path

    if yml_path.exists() and yml_path.is_file():
        return yml_path

    available_names = [
        relative_path_without_suffix(
            file_path
        )
        for file_path
        in get_available_request_files()
    ]

    available_text = (
        "\n".join(
            f"  - {name}"
            for name in available_names
        )
        if available_names
        else "  No image requests available"
    )

    normalized_request_name = "/".join(
        request_parts
    )

    raise FileNotFoundError(
        "Image request was not found:\n"
        f"  {normalized_request_name}\n\n"
        "Available requests:\n"
        f"{available_text}"
    )


# ============================================================
# Prompt-output path
# ============================================================

def build_prompt_output_path(
    image_request_path: Path,
) -> Path:
    """
    Mirror the image-request folder structure inside output/prompts.

    Examples:

        coffee_shop.yaml
            -> output/prompts/coffee_shop_prompt.txt

        airport/lounge.yaml
            -> output/prompts/airport/lounge_prompt.txt
    """

    relative_request_path = (
        image_request_path.relative_to(
            IMAGE_REQUEST_DIR
        )
    )

    output_relative_path = (
        relative_request_path.parent
        / f"{relative_request_path.stem}_prompt.txt"
    )

    return (
        OUTPUT_PROMPT_DIR
        / output_relative_path
    )


# ============================================================
# Run image request
# ============================================================

def run_image_request(
    image_request_path: Path,
) -> None:
    """Build, save and send one image request to ChatGPT."""

    # 1. Load YAML request.
    request_loader = ImageRequestLoader(
        image_request_path
    )

    request = request_loader.load()

    # 2. Determine which persona should be used.
    persona_name = str(
        request["persona"]
    ).strip().lower()

    persona_path = (
        PROJECT_ROOT
        / "03_personas"
        / persona_name
    )

    # 3. Load persona documents.
    persona_loader = PersonaLoader(
        persona_path
    )

    persona = persona_loader.load()

    # 4. Convert YAML data into ImageBrief.
    brief = ImageBrief(
        location=str(
            request["location"]
        ),
        scene=str(
            request["scene"]
        ),
        outfit=str(
            request["outfit"]
        ),
        pose=str(
            request["pose"]
        ),
        expression=str(
            request["expression"]
        ),
        lighting=str(
            request["lighting"]
        ),
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

    # 5. Build image and negative prompts.
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

    # 6. Define output path.
    output_file = build_prompt_output_path(
        image_request_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 7. Save the prompt.
    output_content = f"""IMAGE PROMPT
{'=' * 70}

{image_prompt}

NEGATIVE PROMPT
{'=' * 70}

{negative_prompt}
"""

    output_file.write_text(
        output_content,
        encoding="utf-8",
    )

    # 8. Display result.
    request_display_name = (
        relative_path_without_suffix(
            image_request_path
        )
    )

    print("\n" + "=" * 70)
    print("IMAGE REQUEST")
    print("=" * 70)
    print(request_display_name)

    print("\n" + "=" * 70)
    print(
        f"{persona_name.upper()} IMAGE PROMPT"
    )
    print("=" * 70)
    print(image_prompt)

    print("\n" + "=" * 70)
    print("NEGATIVE PROMPT")
    print("=" * 70)
    print(negative_prompt)

    print("\n" + "=" * 70)
    print("SAVED PROMPT")
    print("=" * 70)
    print(output_file)

    # 9. Build the final ChatGPT instruction.
    final_generation_prompt = f"""
Please generate one image based on the instructions below.

IMAGE PROMPT
{image_prompt}

AVOID
{negative_prompt}

Generate the image directly.
Do not return, summarize or rewrite the prompt.
""".strip()

    # 10. Import the connector only when it is required.
    from chatgpt_connector.plugin import ChatGPTConnector

    connector = ChatGPTConnector()

    connector.send(
        final_generation_prompt
    )


# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    """Run the selected CLI action."""

    args = parse_arguments()

    selected_actions = sum(
        [
            bool(args.list),
            bool(args.new),
            bool(args.request),
        ]
    )

    if selected_actions == 0:
        print(
            "\nNo command was provided.\n\n"
            "Examples:\n"
            "  python3 src/main.py --list\n"
            "  python3 src/main.py coffee_shop\n"
            "  python3 src/main.py airport/lounge\n"
            "  python3 src/main.py --new airport\n"
            "  python3 src/main.py --new airport lounge\n"
        )
        return

    if selected_actions > 1:
        raise ValueError(
            "Use only one action at a time:\n"
            "  --list\n"
            "  --new PATH\n"
            "  REQUEST"
        )

    if args.list:
        list_requests()
        return

    if args.new:
        create_new_request(
            args.new
        )
        return

    if args.request:
        image_request_path = (
            resolve_request_path(
                args.request
            )
        )

        run_image_request(
            image_request_path
        )
        return


if __name__ == "__main__":
    main()