from __future__ import annotations

from .content_models import ContentMoment, ReelScene


class PromptEngine:
    CHARACTER = """
Aiko Sato, a 24-year-old Japanese luxury travel creator with long
silky dark brown hair, natural radiant makeup, a slim elegant figure,
delicate gold jewellery, a premium cream cardigan over a soft beige
travel dress, a leather shoulder bag and a mirrorless camera worn
naturally across her body.
""".strip()

    NEGATIVE_RULES = """
No posing.
No fashion catalogue composition.
No static standing portrait.
No generic smile toward the camera.
No empty hand gesture.
No duplicated limbs.
No incorrect hands.
No artificial plastic skin.
No text overlay.
""".strip()

    STYLE = """
Luxury travel documentary photography.
Authentic human movement.
Natural candid interaction.
Editorial visual storytelling.
Realistic human proportions.
Soft cinematic natural light.
Ultra photorealistic.
Vertical composition.
8K.
""".strip()

    def build_moment_prompt(
        self,
        moment: ContentMoment,
    ) -> str:
        details = "\n".join(
            f"- {detail}"
            for detail in moment.daily_details
        )

        japanese = (
            moment.japanese_detail
            if moment.japanese_detail
            else "No forced text."
        )

        return f"""
Create one ultra-photorealistic editorial Instagram
{moment.content_type.capitalize()} image.

CHARACTER

{self.CHARACTER}

LOCATION

{moment.location}

STORY STAGE

{moment.story_stage}

REAL LIFE

{moment.real_life}

MAIN BEHAVIOR

{moment.behavior}

The behavior is the visual centre of the image.
Aiko must be captured in the middle of this action.

EMOTION

{moment.emotion}

Show one clear natural emotion only.

SOCIAL INTERACTION

{moment.interaction}

BODY MOTION

{moment.body_motion}

CAMERA STORY

{moment.camera_story}

DAILY DETAILS

{details}

JAPANESE IDENTITY DETAIL

{japanese}

QUALITY REQUIREMENTS

The image must show a real-life activity.
The location must support the behavior rather than replace it.
Aiko must not look like a model posing for a catalogue.
The interaction must be visibly happening in the frame.
At least three daily details must be clearly visible.

STYLE

{self.STYLE}

NEGATIVE RULES

{self.NEGATIVE_RULES}
""".strip()

    def build_reel_scene_prompt(
        self,
        scene: ReelScene,
        location: str,
    ) -> str:
        details = ", ".join(scene.daily_details)

        return f"""
Create one ultra-photorealistic vertical cinematic travel video scene.

LOCATION

{location}

SCENE {scene.scene_number}

{scene.title}

MAIN BEHAVIOR

{scene.behavior}

EMOTION

{scene.emotion}

SOCIAL INTERACTION

{scene.interaction}

BODY MOTION

{scene.body_motion}

CAMERA MOVEMENT

{scene.camera_story}

DAILY DETAILS

{details}

DURATION

REAL LIFE

Approximately {scene.duration_seconds} seconds.

MOVEMENT REQUIREMENTS

Natural continuous human motion.
Clear beginning and ending action.
No frozen pose.
No direct model-style camera performance.
Subtle hair, clothing, hand and eye movement.

STYLE

Luxury travel documentary.
Natural cinematic movement.
Vertical Instagram Reel.
Ultra photorealistic.
""".strip()

    