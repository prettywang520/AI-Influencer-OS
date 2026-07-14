"""
OpenAI image-generation plugin for AI Influencer OS.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


class OpenAIImagePlugin:
    """Generate one image using the OpenAI Images API."""

    def __init__(
        self,
        runtime_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.runtime_dir = Path(runtime_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

        env_path = self.runtime_dir / ".env"

        if not env_path.exists():
            raise FileNotFoundError(
                f"Runtime .env file not found: {env_path}"
            )

        load_dotenv(
            dotenv_path=env_path,
            override=True,
        )

        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing from claude_runtime/.env"
            )

        if not api_key.isascii():
            raise RuntimeError(
                "OPENAI_API_KEY contains non-ASCII characters."
            )

        if not api_key.startswith("sk-"):
            raise RuntimeError(
                "OPENAI_API_KEY does not look like a valid OpenAI API key."
            )

        self.model = os.getenv(
            "OPENAI_IMAGE_MODEL",
            "gpt-image-1",
        ).strip()

        self.size = os.getenv(
            "OPENAI_IMAGE_SIZE",
            "1024x1536",
        ).strip()

        self.quality = os.getenv(
            "OPENAI_IMAGE_QUALITY",
            "high",
        ).strip()

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        filename: str,
    ) -> Path:
        """Generate and save one PNG image."""

        clean_prompt = prompt.strip()

        if not clean_prompt:
            raise ValueError("Image prompt cannot be empty.")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        response = self.client.images.generate(
            model=self.model,
            prompt=clean_prompt,
            size=self.size,
            quality=self.quality,
            output_format="png",
            n=1,
        )

        if not response.data:
            raise RuntimeError(
                "OpenAI returned no image data."
            )

        image_base64 = response.data[0].b64_json

        if not image_base64:
            raise RuntimeError(
                "OpenAI returned an empty image result."
            )

        safe_filename = (
            filename
            if filename.lower().endswith(".png")
            else f"{filename}.png"
        )

        output_path = self.output_dir / safe_filename

        try:
            image_bytes = base64.b64decode(
                image_base64,
                validate=True,
            )
        except (ValueError, TypeError) as error:
            raise RuntimeError(
                "OpenAI returned invalid Base64 image data."
            ) from error

        output_path.write_bytes(image_bytes)

        return output_path