from __future__ import annotations

from .planning_loader import build_planning_loader


def main() -> None:
    loader = build_planning_loader()

    errors = loader.validate_required_files()

    if errors:
        print("Planning Loader validation failed:")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    datasets = loader.load_all()

    print("Planning Loader test passed.")
    print()

    for name, data in datasets.items():
        status = "loaded" if data else "optional / empty"

        print(
            f"{name:<18} "
            f"{status:<18} "
            f"keys={len(data)}"
        )

    print()
    print("Planning root:")
    print(loader.root)


if __name__ == "__main__":
    main()