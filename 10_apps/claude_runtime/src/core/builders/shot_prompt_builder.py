"""
Multi-shot prompt builder for AI Influencer OS.

This module creates several visually different shots
from one existing image-request YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ShotDefinition:
    """Visual direction for one shot."""

    number: int
    name: str
    pose_direction: str
    expression: str
    camera_angle: str
    framing: str
    composition: str


DEFAULT_SHOTS: tuple[ShotDefinition, ...] = (
    ShotDefinition(
        number=1,
        name="cover",
        pose_direction=(
            "Create a strong hero pose suitable for an Instagram cover. "
            "The subject should be clearly visible and visually dominant."
        ),
        expression="soft confident smile",
        camera_angle="eye level",
        framing="three-quarter body portrait",
        composition=(
            "Premium editorial hero composition with a clear subject, "
            "balanced background and strong visual impact."
        ),
    ),
    ShotDefinition(
        number=2,
        name="candid",
        pose_direction=(
            "Create a natural candid moment. "
            "The subject should appear unaware of the camera, "
            "interacting naturally with the environment."
        ),
        expression="relaxed natural expression",
        camera_angle="slightly off-centre eye level",
        framing="medium lifestyle portrait",
        composition=(
            "Documentary-style travel photograph with natural movement "
            "and an authentic unposed atmosphere."
        ),
    ),
    ShotDefinition(
        number=3,
        name="closeup",
        pose_direction=(
            "Create an elegant close-up portrait. "
            "Keep the face, eyes, hair and skin texture highly detailed."
        ),
        expression="gentle warm smile",
        camera_angle="eye level close-up",
        framing="close portrait",
        composition=(
            "Editorial beauty portrait with shallow depth of field, "
            "soft background separation and natural facial detail."
        ),
    ),
    ShotDefinition(
        number=4,
        name="wide",
        pose_direction=(
            "Place the subject naturally within the wider environment. "
            "Show more architecture, landscape or location context."
        ),
        expression="calm confident expression",
        camera_angle="wide eye-level view",
        framing="full-body environmental portrait",
        composition=(
            "Wide cinematic travel composition showing the relationship "
            "between the subject and the destination."
        ),
    ),
    ShotDefinition(
        number=5,
        name="detail",
        pose_direction=(
            "Create a lifestyle detail shot focused on the subject's hands, "
            "accessories, drink, luggage, table setting or surrounding object. "
            "The subject may remain partially visible."
        ),
        expression="natural relaxed mood",
        camera_angle="slightly above detail angle",
        framing="editorial detail photograph",
        composition=(
            "Luxury lifestyle detail composition with realistic hands, "
            "materials, textures and premium visual styling."
        ),
    ),
    ShotDefinition(
        number=6,
        name="back_view",
        pose_direction=(
            "Photograph the subject from behind or from a three-quarter "
            "back angle while she looks toward the location. "
            "Her identity, hairstyle and proportions must remain consistent."
        ),
        expression="subtle over-the-shoulder expression",
        camera_angle="three-quarter rear eye level",
        framing="three-quarter or full-body portrait",
        composition=(
            "Cinematic travel storytelling composition with depth, "
            "destination atmosphere and elegant body language."
        ),
    ),
)


class ShotPromptBuilder:
    """Add one shot direction to an existing image prompt."""

    def __init__(
        self,
        shots: tuple[ShotDefinition, ...] = DEFAULT_SHOTS,
    ) -> None:
        if not shots:
            raise ValueError("At least one shot definition is required.")

        self.shots = shots

    def get_shot(
        self,
        shot_name: str,
    ) -> ShotDefinition:
        clean_name = shot_name.strip().lower()

        for shot in self.shots:
            if shot.name == clean_name:
                return shot

        available = ", ".join(
            shot.name
            for shot in self.shots
        )

        raise ValueError(
            f"Unknown shot: {shot_name}\n"
            f"Available shots: {available}"
        )

    def build(
        self,
        base_image_prompt: str,
        request: Mapping[str, Any],
        shot: ShotDefinition,
    ) -> str:
        """Build the final prompt for one specific shot."""

        original_pose = str(
            request.get("pose", "")
        ).strip()

        original_expression = str(
            request.get("expression", "")
        ).strip()

        original_camera_angle = str(
            request.get("camera_angle", "eye level")
        ).strip()

        return f"""
{base_image_prompt.strip()}

MULTI-SHOT DIRECTION

Shot number: {shot.number:02d}
Shot name: {shot.name}

Original scene pose:
{original_pose}

Shot-specific pose and action:
{shot.pose_direction}

Expression:
Use {shot.expression}.
Maintain the emotional tone of the original expression:
{original_expression}

Camera:
Original camera direction: {original_camera_angle}
Shot camera direction: {shot.camera_angle}
Framing: {shot.framing}

Composition:
{shot.composition}

CONTINUITY REQUIREMENTS

- This image belongs to the same editorial photo session.
- Keep the same Aiko identity.
- Keep the same age, ethnicity, facial structure and hairstyle.
- Keep the same outfit, accessories, location and lighting.
- Do not redesign the character.
- Change only the pose, action, framing and camera composition.
- The result must look like another photograph from the same shoot.
""".strip()