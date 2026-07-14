from pathlib import Path


class PersonaLoader:
    """Load all Markdown files belonging to one persona."""

    def __init__(self, persona_path: str | Path) -> None:
        self.persona_path = Path(persona_path).resolve()

    def load(self) -> dict[str, str]:
        if not self.persona_path.exists():
            raise FileNotFoundError(
                f"Persona folder not found: {self.persona_path}"
            )

        if not self.persona_path.is_dir():
            raise NotADirectoryError(
                f"Persona path is not a folder: {self.persona_path}"
            )

        persona: dict[str, str] = {}

        for file_path in sorted(self.persona_path.glob("*.md")):
            persona[file_path.stem] = file_path.read_text(encoding="utf-8")

        if not persona:
            raise FileNotFoundError(
                f"No Markdown files found in: {self.persona_path}"
            )

        return persona