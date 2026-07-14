from dataclasses import dataclass
from typing import Mapping

"""
Image Prompt Builder for AI Influencer OS.

Builds a structured image-generation prompt from selected
Aiko persona documents and a user-provided creative brief.
"""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ImageBrief:
    """Creative instructions for one image."""

    location: str
    scene: str
    outfit: str
    pose: str
    expression: str
    lighting: str
    camera_angle: str = "eye-level"
    aspect_ratio: str = "4:5"
    platform: str = "Instagram"


class PromptBuilder:
    """Build image prompts from persona context and an image brief."""

    REQUIRED_PERSONA_SECTIONS = (
        "00_identity",
        "02_appearance",
        "09_fashion",
        "14_image_prompt",
        "19_boundaries",
        "28_photo_style",
    )

    def __init__(self, persona: Mapping[str, str]) -> None:
        if not persona:
            raise ValueError("Persona context cannot be empty.")

        self.persona = dict(persona)

    def build_image_prompt(self, brief: ImageBrief) -> str:
        """Return one complete image-generation prompt."""

        self._validate_brief(brief)

        persona_context = self._build_persona_context()

        return f"""
Create one ultra-photorealistic editorial travel photograph for {brief.platform}.

CHARACTER CONTEXT
{persona_context}

CREATIVE BRIEF
Location: {brief.location}
Scene: {brief.scene}
Outfit: {brief.outfit}
Pose: {brief.pose}
Expression: {brief.expression}
Lighting: {brief.lighting}
Camera angle: {brief.camera_angle}
Aspect ratio: {brief.aspect_ratio}

CHARACTER CONSISTENCY
Keep Aiko's facial identity, ethnicity, age, hairstyle, eye colour,
skin tone and body proportions consistent with the approved persona.
Do not redesign or reinterpret the character.

VISUAL DIRECTION
Natural human skin texture.
Premium travel editorial photography.
Realistic anatomy and hands.
Elegant but approachable styling.
Cinematic depth.
Professional composition.
Authentic Asian travel atmosphere.
No text, watermark or visible logo.

OUTPUT
Return only the final image prompt.
""".strip()

    def build_negative_prompt(self) -> str:
        """Return a reusable negative prompt."""

        return (
            "low quality, worst quality, blurry, out of focus, "
            "deformed face, asymmetrical eyes, crossed eyes, "
            "bad anatomy, bad hands, extra fingers, missing fingers, "
            "extra limbs, duplicate person, distorted body proportions, "
            "plastic skin, waxy skin, oversaturated colours, cartoon, anime, "
            "cgi appearance, watermark, text, logo, cropped face"
        )

    def _build_persona_context(self) -> str:
        sections: list[str] = []

        for section_name in self.REQUIRED_PERSONA_SECTIONS:
            content = self.persona.get(section_name)

            if content and content.strip():
                sections.append(
                    f"\n--- {section_name} ---\n{content.strip()}"
                )

        if not sections:
            raise ValueError(
                "None of the required persona sections were found."
            )

        return "\n".join(sections)

    @staticmethod
    def _validate_brief(brief: ImageBrief) -> None:
        required_values = {
            "location": brief.location,
            "scene": brief.scene,
            "outfit": brief.outfit,
            "pose": brief.pose,
            "expression": brief.expression,
            "lighting": brief.lighting,
        }

        missing = [
            name
            for name, value in required_values.items()
            if not value.strip()
        ]

        if missing:
            raise ValueError(
                f"Missing required image brief fields: {', '.join(missing)}"
            )