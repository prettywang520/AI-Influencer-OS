from __future__ import annotations

import hashlib
from pathlib import Path

from reply_classifier import ReplyClassifier
from reply_loader import ReplyLoader
from reply_models import (
    CommentInput,
    ContentContext,
    ReplyCandidate,
    ReplyResult,
    ValidationResult,
)
from reply_selector import ReplySelectionError, ReplySelector
from reply_validator import ReplyValidator


class InstagramReplyAgent:
    """
    Main execution agent for AIKO Instagram replies.

    Flow:
    comment
    -> intent classification
    -> emotion classification
    -> library routing
    -> reply group selection
    -> weighted reply selection
    -> validation
    -> publish / review / ignore
    """

    def __init__(
        self,
        reply_brain_root: str | Path,
        maximum_attempts: int = 3,
        random_seed: int | None = None,
    ) -> None:
        self.loader = ReplyLoader(reply_brain_root)
        self.maximum_attempts = maximum_attempts

        intent_config = self.loader.load_engine("intent_engine")
        emotion_config = self.loader.load_engine("emotion_engine")

        self.classifier = ReplyClassifier(
            intent_config=intent_config,
            emotion_config=emotion_config,
        )

        self.selector = ReplySelector(
            random_seed=random_seed,
        )

        self.validator = ReplyValidator(
            maximum_words=25,
            maximum_emojis=3,
        )

    @staticmethod
    def hash_follower_id(follower_id: str) -> str:
        """
        Return a privacy-safe hash for logging or memory storage.
        """
        return hashlib.sha256(
            follower_id.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _empty_validation(
        failed_rule: str | None = None,
    ) -> ValidationResult:
        """
        Create a default failed validation object.
        """
        failed_rules = []

        if failed_rule:
            failed_rules.append(failed_rule)

        return ValidationResult(
            passed=False,
            failed_rules=failed_rules,
        )

    @staticmethod
    def _normalise_group_name(value: str | None) -> str | None:
        """
        Convert values such as 'New York City' into 'new_york_city'.
        """
        if not value:
            return None

        return (
            value.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _resolve_location_group(
        self,
        content: ContentContext,
    ) -> str:
        """
        Choose the most suitable reply group inside location.yaml.

        Priority:
        1. verified venue reply
        2. district or city-specific group
        3. generic fallback
        """
        if content.venue and content.city:
            return "verified_venue"

        searchable_values = [
            content.location,
            content.venue,
            content.city,
        ]

        combined = " ".join(
            value.lower()
            for value in searchable_values
            if value
        )

        location_group_map = {
            "daikanyama": "tokyo",
            "asakusa": "tokyo",
            "ginza": "tokyo",
            "shibuya": "tokyo",
            "shinjuku": "tokyo",
            "tsukiji": "tokyo",
            "harajuku": "tokyo",
            "tokyo": "tokyo",
            "kyoto": "kyoto",
            "osaka": "osaka",
            "taipei": "taiwan",
            "taiwan": "taiwan",
            "hong kong": "hong_kong",
            "seoul": "korea",
            "korea": "korea",
            "singapore": "singapore",
            "bangkok": "thailand",
            "thailand": "thailand",
        }

        for keyword, group_name in location_group_map.items():
            if keyword in combined:
                return group_name

        city_group = self._normalise_group_name(content.city)

        if city_group:
            return city_group

        return "generic"

    def _resolve_hotel_group(
        self,
        comment_text: str,
        content: ContentContext,
    ) -> str | None:
        """
        Select a hotel reply group based on the follower's question
        and current hotel content.
        """
        text = comment_text.lower()

        if any(
            phrase in text
            for phrase in (
                "which hotel",
                "what hotel",
                "hotel name",
                "where did you stay",
            )
        ):
            return "hotel_identification"

        if "breakfast" in text:
            return "breakfast"

        if "pool" in text:
            return "pool"

        if "spa" in text or "massage" in text:
            return "spa"

        if "view" in text or "window" in text or "sunset" in text:
            return "view"

        if "room" in text or "suite" in text or "bed" in text:
            return "room"

        if "check-in" in text or "check in" in text or "reception" in text:
            return "check_in"

        if "service" in text or "staff" in text:
            return "service"

        if (
            "worth it" in text
            or "recommend" in text
            or "expensive" in text
        ):
            return "value_and_recommendation"

        topic = (content.topic or "").lower()

        topic_map = {
            "hotel_breakfast": "breakfast",
            "hotel_pool": "pool",
            "hotel_spa": "spa",
            "hotel_room": "room",
            "hotel_suite": "room",
            "hotel_view": "view",
            "hotel_lobby": "check_in",
        }

        return topic_map.get(topic, "overall_stay")

    @staticmethod
    def _resolve_food_group(
        comment_text: str,
        content: ContentContext,
    ) -> str | None:
        """
        Select a food reply group from food.yaml.
        """
        text = comment_text.lower()
        topic = (content.topic or "").lower()

        if "coffee" in text or "latte" in text or "café" in text or "cafe" in text:
            return "coffee"

        if "matcha" in text:
            return "matcha"

        if "dessert" in text or "cake" in text or "sweet" in text:
            return "dessert"

        if "ramen" in text:
            return "ramen"

        if "sushi" in text:
            return "sushi"

        topic_map = {
            "coffee": "coffee",
            "cafe": "coffee",
            "matcha": "matcha",
            "dessert": "dessert",
            "ramen": "ramen",
            "sushi": "sushi",
        }

        return topic_map.get(topic, "delicious")

    @staticmethod
    def _resolve_shopping_group(
        comment_text: str,
        content: ContentContext,
    ) -> str | None:
        """
        Select a shopping reply group from shopping.yaml.
        """
        text = comment_text.lower()
        topic = (content.topic or "").lower()

        if "perfume" in text or "fragrance" in text:
            return "perfume"

        if "book" in text or "bookstore" in text:
            return "bookstore"

        if "souvenir" in text or "gift" in text:
            return "souvenirs"

        if "handmade" in text or "local maker" in text:
            return "local_products"

        if "fashion" in text or "clothes" in text:
            return "fashion"

        topic_map = {
            "bookstore": "bookstore",
            "perfume": "perfume",
            "souvenir": "souvenirs",
            "shopping": "fashion",
        }

        return topic_map.get(topic, "souvenirs")

    @staticmethod
    def _resolve_camera_group(
        comment_text: str,
        content: ContentContext,
    ) -> str | None:
        """
        Select a camera reply group from camera.yaml.
        """
        text = comment_text.lower()

        if "what camera" in text or "which camera" in text:
            return "camera_identification"

        if "lens" in text or "focal length" in text:
            return "lens_identification"

        if "phone" in text:
            return "phone_or_camera"

        if any(
            phrase in text
            for phrase in (
                "settings",
                "aperture",
                "shutter",
                "iso",
            )
        ):
            return "camera_settings"

        if "light" in text or "lighting" in text:
            return "natural_light"

        if (
            "composition" in text
            or "angle" in text
            or "frame" in text
        ):
            return "composition"

        if "candid" in text or "natural" in text or "unposed" in text:
            return "candid"

        if "who took" in text or "photographer" in text:
            return "photographer_credit"

        if (
            "edit" in text
            or "preset" in text
            or "filter" in text
            or "colour grade" in text
            or "color grade" in text
        ):
            return "editing"

        if content.ai_generated:
            return "photography_compliments"

        return "photography_compliments"

    @staticmethod
    def _resolve_outfit_group(
        comment_text: str,
    ) -> str | None:
        """
        Select an outfit reply group from outfit.yaml.
        """
        text = comment_text.lower()

        if "where is your dress" in text or "dress from" in text:
            return "dress_source"

        if "outfit details" in text or "what are you wearing" in text:
            return "outfit_details"

        if "bag" in text or "handbag" in text:
            return "bag"

        if "shoe" in text or "shoes" in text:
            return "shoes"

        if (
            "jewellery" in text
            or "jewelry" in text
            or "sunglasses" in text
            or "accessory" in text
        ):
            return "accessories"

        if "style" in text or "how do you style" in text:
            return "styling"

        if "comfortable" in text or "travel outfit" in text:
            return "travel_comfort"

        if "link" in text:
            return "link_request"

        return "outfit_compliment"

    @staticmethod
    def _resolve_engagement_group(
        comment_text: str,
    ) -> str | None:
        """
        Select an engagement reply group.
        """
        text = comment_text.lower()

        if "take me with you" in text:
            return "take_me_with_you"

        if "bucket list" in text:
            return "bucket_list"

        if (
            "come to" in text
            or "visit my country" in text
            or "visit my city" in text
        ):
            return "visit_my_country"

        if (
            "i recommend" in text
            or "you should visit" in text
            or "you must try" in text
        ):
            return "follower_recommendation"

        if "i've been" in text or "i have been" in text:
            return "ask_experience"

        if "me too" in text or "same" in text or "i agree" in text:
            return "simple_agreement"

        if "which one" in text or "i would choose" in text:
            return "ask_preference"

        return "friendly_check_in"

    @staticmethod
    def _resolve_flirting_group(
        comment_text: str,
    ) -> str | None:
        """
        Select a safe flirting reply group.
        """
        text = comment_text.lower()

        if "marry me" in text:
            return "marry_me"

        if "i love you" in text:
            return "i_love_you"

        if "date me" in text or "go on a date" in text:
            return "date_request"

        if (
            "my wife" in text
            or "my girlfriend" in text
            or "you're mine" in text
        ):
            return "possessive_comment"

        if "handsome" in text:
            return "handsome"

        if any(
            phrase in text
            for phrase in (
                "beautiful",
                "gorgeous",
                "stunning",
                "pretty",
                "cute",
            )
        ):
            return "appearance_flirt"

        return "gentle_compliment"

    @staticmethod
    def _resolve_negative_group(
        comment_text: str,
    ) -> str | None:
        """
        Select a negative reply group.
        """
        text = comment_text.lower()

        if (
            "wrong" in text
            or "incorrect" in text
            or "lying" in text
        ):
            return "accusation"

        if (
            "too expensive" in text
            or "waste of money" in text
            or "not worth" in text
        ):
            return "price_criticism"

        if (
            "fake" in text
            or "ai" in text
            or "not authentic" in text
            or "too edited" in text
        ):
            return "authenticity"

        if any(
            phrase in text
            for phrase in (
                "ugly",
                "stupid",
                "terrible person",
            )
        ):
            return "personal_insult"

        if "i disagree" in text or "not my taste" in text:
            return "different_taste"

        return "constructive_criticism"

    def _resolve_preferred_group(
        self,
        library_name: str,
        comment_text: str,
        content: ContentContext,
    ) -> str | None:
        """
        Route each reply library to the most relevant reply group.
        """
        if library_name == "location":
            return self._resolve_location_group(content)

        if library_name == "hotel":
            return self._resolve_hotel_group(
                comment_text,
                content,
            )

        if library_name == "food":
            return self._resolve_food_group(
                comment_text,
                content,
            )

        if library_name == "shopping":
            return self._resolve_shopping_group(
                comment_text,
                content,
            )

        if library_name == "camera":
            return self._resolve_camera_group(
                comment_text,
                content,
            )

        if library_name == "outfit":
            return self._resolve_outfit_group(
                comment_text,
            )

        if library_name == "engagement":
            return self._resolve_engagement_group(
                comment_text,
            )

        if library_name == "flirting":
            return self._resolve_flirting_group(
                comment_text,
            )

        if library_name == "negative":
            return self._resolve_negative_group(
                comment_text,
            )

        return None

    def process(
        self,
        comment: CommentInput,
        content: ContentContext,
        excluded_reply_ids: set[str] | None = None,
    ) -> ReplyResult:
        """
        Process one follower comment and return one reply decision.
        """
        classification = self.classifier.classify(
            comment.text
        )

        non_publish_actions = {
            "ignore",
            "escalate",
            "block_recommended",
            "hold_for_review",
        }

        if classification.action in non_publish_actions:
            return ReplyResult(
                request_id=comment.request_id,
                action=classification.action,
                approved=False,
                reply_text=None,
                classification=classification,
                validation=self._empty_validation(),
            )
        library_name = classification.selected_library

        if (
            not library_name
            or library_name == "brand_collaboration_agent"
        ):
            return ReplyResult(
                request_id=comment.request_id,
                action="hold_for_review",
                approved=False,
                reply_text=None,
                classification=classification,
                validation=self._empty_validation(
                    "invalid_or_external_library"
                ),
            )

        # Use a deterministic reply when both venue and city are known.
        # This prevents a verified Tokyo location from randomly selecting
        # a Kyoto, Osaka or generic reply group.
        if (
            library_name == "location"
            and content.venue
            and content.city
        ):
            country_emoji = ""

            if (content.country or "").strip().lower() == "japan":
                country_emoji = " 🇯🇵"

            candidate = ReplyCandidate(
                reply_id="LOCATION-VERIFIED-001",
                library="location",
                reply_group="verified_venue",
                text=(
                    f"this is {content.venue} "
                    f"in {content.city}{country_emoji}"
                ),
                weight=100,
                emotion="helpful",
            )

            validation = self.validator.validate(
                candidate=candidate,
                context=content,
            )

            if validation.passed:
                return ReplyResult(
                    request_id=comment.request_id,
                    action="publish",
                    approved=True,
                    reply_text=candidate.text,
                    classification=classification,
                    validation=validation,
                    reply_id=candidate.reply_id,
                    library=candidate.library,
                    reply_group=candidate.reply_group,
                    language=comment.language_hint or "english",
                    japanese_injected=False,
                    selected_emojis=[],
                    attempts=1,
                )

        try:
            library = self.loader.load_library(
                library_name
            )
        except Exception as exc:
            return ReplyResult(
                request_id=comment.request_id,
                action="hold_for_review",
                approved=False,
                reply_text=None,
                classification=classification,
                validation=ValidationResult(
                    passed=False,
                    failed_rules=[
                        f"library_load_failed:{type(exc).__name__}"
                    ],
                ),
            )

        preferred_group = self._resolve_preferred_group(
            library_name=library_name,
            comment_text=comment.text,
            content=content,
        )

        rejected_ids = set(
            excluded_reply_ids or set()
        )

        final_validation = self._empty_validation(
            "no_candidate_selected"
        )

        attempts_completed = 0

        for attempt in range(
            1,
            self.maximum_attempts + 1,
        ):
            attempts_completed = attempt

            try:
                candidate = self.selector.select(
                    library_name=library_name,
                    library=library,
                    context=content,
                    preferred_group=preferred_group,
                    preferred_emotion=classification.reply_emotion,
                    excluded_reply_ids=rejected_ids,
                )
            except ReplySelectionError:
                # Retry without preferred group if the specific group
                # has no valid context-aware replies.
                if preferred_group is not None:
                    preferred_group = None
                    continue

                break

            final_validation = self.validator.validate(
                candidate=candidate,
                context=content,
            )

            if final_validation.passed:
                return ReplyResult(
                    request_id=comment.request_id,
                    action="publish",
                    approved=True,
                    reply_text=candidate.text,
                    classification=classification,
                    validation=final_validation,
                    reply_id=candidate.reply_id,
                    library=candidate.library,
                    reply_group=candidate.reply_group,
                    language=(
                        comment.language_hint
                        or "english"
                    ),
                    japanese_injected=False,
                    selected_emojis=[],
                    attempts=attempt,
                )

            rejected_ids.add(
                candidate.reply_id
            )

        return ReplyResult(
            request_id=comment.request_id,
            action="hold_for_review",
            approved=False,
            reply_text=None,
            classification=classification,
            validation=final_validation,
            language=comment.language_hint or "english",
            attempts=attempts_completed,
        )


def build_agent() -> InstagramReplyAgent:
    """
    Build the agent using the repository root.

    reply_agent.py location:
    AI-Influencer-OS/
    └── 10_apps/
        └── claude_runtime/
            └── src/
                └── reply_agent.py

    parents[3] therefore resolves to AI-Influencer-OS.
    """
    project_root = Path(__file__).resolve().parents[3]

    reply_brain_root = (
        project_root
        / "03_personas"
        / "aiko"
        / "reply_brain"
    )

    return InstagramReplyAgent(
        reply_brain_root=reply_brain_root,
        maximum_attempts=3,
    )