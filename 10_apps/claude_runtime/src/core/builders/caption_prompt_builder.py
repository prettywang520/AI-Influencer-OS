"""
Instagram caption prompt builder for AI Influencer OS.
"""

from __future__ import annotations

from typing import Any, Mapping


class CaptionPromptBuilder:
    """Build an Instagram caption prompt from persona and scene data."""

    PERSONA_SECTIONS = (
        "00_identity",
        "01_personality",
        "03_voice",
        "04_response_logic",
        "05_response_style",
        "11_social_media",
        "12_content_rules",
        "17_do_not_say",
        "19_boundaries",
    )

    def __init__(
        self,
        persona: Mapping[str, str],
    ) -> None:
        if not persona:
            raise ValueError(
                "Persona context cannot be empty."
            )

        self.persona = dict(persona)

    def build(
        self,
        request: Mapping[str, Any],
    ) -> str:
        """Build one complete Instagram caption instruction."""

        persona_context = self._build_persona_context()

        location = str(
            request.get("location", "")
        ).strip()

        scene = str(
            request.get("scene", "")
        ).strip()

        outfit = str(
            request.get("outfit", "")
        ).strip()

        pose = str(
            request.get("pose", "")
        ).strip()

        expression = str(
            request.get("expression", "")
        ).strip()

        platform = str(
            request.get("platform", "Instagram")
        ).strip()

        return f"""
Create one Instagram caption for Aiko based on the scene below.

PERSONA CONTEXT
{persona_context}

CONTENT CONTEXT
Platform: {platform}
Location: {location}
Scene: {scene}
Outfit: {outfit}
Pose: {pose}
Expression: {expression}

CAPTION REQUIREMENTS
- Write in Aiko's established voice.
- Cute but mature.
- Gentle, warm and slightly playful.
- Use natural lowercase English.
- Aiko is Japanese. Naturally include one short Japanese sentence or phrase when it suits the scene.
- Keep the main caption in natural lowercase English.
- Japanese should feel personal and conversational, not decorative.
- Use no more than one or two Japanese lines per caption.
- Use short sentences.
- Use suitable emojis without overloading the caption.
- Never sound like customer service.
- Never discuss politics or religion.
- Do not make unverifiable claims about the location.
- Keep the caption under 80 words.
- End with one natural question.
- Do not include hashtags.
- Do not explain your writing process.


OUTPUT
Return only the final Instagram caption.
""".strip()

    def _build_persona_context(self) -> str:
        sections: list[str] = []

        for section_name in self.PERSONA_SECTIONS:
            content = self.persona.get(section_name)

            if content and content.strip():
                sections.append(
                    f"--- {section_name} ---\n"
                    f"{content.strip()}"
                )

        if not sections:
            raise ValueError(
                "No suitable persona sections were found."
            )

        return "\n\n".join(sections)