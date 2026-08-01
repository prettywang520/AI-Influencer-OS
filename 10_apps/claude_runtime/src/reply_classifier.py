from __future__ import annotations

import re
from typing import Any

from reply_models import ClassificationResult


class ReplyClassifier:
    def __init__(
        self,
        intent_config: dict[str, Any],
        emotion_config: dict[str, Any],
    ) -> None:
        self.intent_config = intent_config
        self.emotion_config = emotion_config

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _contains_trigger(text: str, trigger: str) -> bool:
        return trigger.lower() in text

    def _score_intent(
        self,
        text: str,
        definition: dict[str, Any],
    ) -> tuple[float, list[str]]:
        triggers = definition.get("triggers", {})
        score = 0.0
        matched: list[str] = []

        for phrase in triggers.get("exact", []):
            if text == str(phrase).lower():
                score += 1.0
                matched.append(str(phrase))

        for phrase in triggers.get("contains", []):
            if self._contains_trigger(text, str(phrase)):
                score += 0.55
                matched.append(str(phrase))

        for pattern in triggers.get("regex", []):
            try:
                if re.search(pattern, text):
                    score += 0.65
                    matched.append(pattern)
            except re.error:
                continue

        return min(score, 1.0), matched

    def detect_intent(self, message_text: str) -> ClassificationResult:
        text = self._normalise(message_text)
        definitions = self.intent_config.get("intent_definitions", {})
        priority = self.intent_config.get(
            "intent_priority",
            list(definitions.keys()),
        )

        scored: list[tuple[str, float, list[str], dict[str, Any]]] = []

        for intent_name, definition in definitions.items():
            score, matched = self._score_intent(text, definition)

            if score > 0:
                scored.append((intent_name, score, matched, definition))

        if not scored:
            return ClassificationResult(
                primary_intent="unknown",
                selected_library="engagement",
                action="clarification_required",
                confidence=0.40,
            )

        priority_index = {
            intent: index for index, intent in enumerate(priority)
        }

        scored.sort(
            key=lambda item: (
                -item[1],
                priority_index.get(item[0], 999),
            )
        )

        intent, score, matched, definition = scored[0]
        secondary = [item[0] for item in scored[1:4]]

        return ClassificationResult(
            primary_intent=intent,
            secondary_intents=secondary,
            selected_library=definition.get("library"),
            action=definition.get("action", "reply"),
            confidence=max(score, 0.55),
            matched_triggers=matched,
        )

    def detect_emotion(self, message_text: str) -> tuple[str, str, float]:
        text = self._normalise(message_text)
        categories = self.emotion_config.get("emotion_categories", {})
        priority = self.emotion_config.get(
            "emotion_priority",
            list(categories.keys()),
        )

        scored: list[tuple[str, float, dict[str, Any]]] = []

        for emotion, definition in categories.items():
            markers = definition.get("markers", {})
            score = 0.0

            for word in markers.get("words", []):
                if str(word).lower() in text:
                    score += 0.35

            for phrase in markers.get("phrases", []):
                if str(phrase).lower() in text:
                    score += 0.45

            for question_word in markers.get("question_words", []):
                if re.search(
                    rf"\b{re.escape(str(question_word).lower())}\b",
                    text,
                ):
                    score += 0.15

            for emoji in markers.get("emojis", []):
                if emoji in message_text:
                    score += 0.25

            if score > 0:
                scored.append((emotion, min(score, 1.0), definition))

        if not scored:
            return "neutral", "friendly", 0.50

        priority_index = {
            emotion: index for index, emotion in enumerate(priority)
        }

        scored.sort(
            key=lambda item: (
                -item[1],
                priority_index.get(item[0], 999),
            )
        )

        emotion, score, definition = scored[0]

        return (
            emotion,
            str(definition.get("default_reply_emotion", "friendly")),
            max(score, 0.55),
        )

    def classify(self, message_text: str) -> ClassificationResult:
        result = self.detect_intent(message_text)
        emotion, reply_emotion, emotion_confidence = self.detect_emotion(
            message_text
        )

        result.detected_emotion = emotion
        result.reply_emotion = reply_emotion
        result.confidence = round(
            (result.confidence + emotion_confidence) / 2,
            3,
        )

        return result