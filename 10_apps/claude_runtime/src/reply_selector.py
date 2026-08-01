from __future__ import annotations

import random
import re
from typing import Any

from reply_models import ContentContext, ReplyCandidate


class ReplySelectionError(RuntimeError):
    pass


class ReplySelector:
    def __init__(self, random_seed: int | None = None) -> None:
        self.random = random.Random(random_seed)

    @staticmethod
    def _available_context(context: ContentContext) -> dict[str, Any]:
        values: dict[str, Any] = {
            "location": context.location,
            "current_location": context.location,
            "city": context.city,
            "current_city": context.city,
            "country": context.country,
            "current_country": context.country,
            "venue": context.venue,
            "venue_name": context.venue,
            "activity": context.activity,
            "current_activity": context.activity,
            "current_emotion": context.emotion,
        }

        values.update(context.metadata)
        return values

    @staticmethod
    def _extract_replies(
        library_name: str,
        library: dict[str, Any],
    ) -> list[ReplyCandidate]:
        candidates: list[ReplyCandidate] = []
        groups = library.get("reply_groups", {})

        if isinstance(groups, list):
            group_items = [
                (str(group.get("id", "general")), group)
                for group in groups
                if isinstance(group, dict)
            ]
        else:
            group_items = list(groups.items())

        for group_name, group_data in group_items:
            if isinstance(group_data, list):
                replies = group_data
            elif isinstance(group_data, dict):
                replies = group_data.get("replies", [])

                if not replies:
                    # Supports earlier YAML where replies sit directly
                    # under the group.
                    replies = [
                        value
                        for value in group_data.values()
                        if isinstance(value, dict) and "text" in value
                    ]
            else:
                continue

            for reply in replies:
                if not isinstance(reply, dict) or "text" not in reply:
                    continue

                candidates.append(
                    ReplyCandidate(
                        reply_id=str(
                            reply.get(
                                "id",
                                f"{library_name}-{group_name}-{len(candidates)}",
                            )
                        ),
                        library=library_name,
                        reply_group=str(group_name),
                        text=str(reply["text"]).strip(),
                        weight=max(int(reply.get("weight", 1)), 1),
                        emotion=reply.get("emotion"),
                        requires_context=list(
                            reply.get("requires_context", [])
                        ),
                        source_data=reply,
                    )
                )

        # Supports simple top-level replies from early files.
        for reply in library.get("replies", []):
            if isinstance(reply, dict) and "text" in reply:
                candidates.append(
                    ReplyCandidate(
                        reply_id=str(reply.get("id", "unknown")),
                        library=library_name,
                        reply_group="general",
                        text=str(reply["text"]).strip(),
                        weight=max(int(reply.get("weight", 1)), 1),
                        emotion=reply.get("emotion"),
                        requires_context=list(
                            reply.get("requires_context", [])
                        ),
                        source_data=reply,
                    )
                )

        return candidates

    @staticmethod
    def _context_is_available(
        candidate: ReplyCandidate,
        values: dict[str, Any],
    ) -> bool:
        return all(values.get(field) not in (None, "") for field in candidate.requires_context)

    @staticmethod
    def _format_text(text: str, values: dict[str, Any]) -> str:
        fields = re.findall(r"\{([a-zA-Z0-9_]+)\}", text)

        for field in fields:
            if values.get(field) in (None, ""):
                raise ReplySelectionError(
                    f"Missing required context field: {field}"
                )

        try:
            return text.format(**values)
        except KeyError as exc:
            raise ReplySelectionError(
                f"Unable to format reply: {exc}"
            ) from exc

    def select(
        self,
        library_name: str,
        library: dict[str, Any],
        context: ContentContext,
        preferred_group: str | None = None,
        preferred_emotion: str | None = None,
        excluded_reply_ids: set[str] | None = None,
    ) -> ReplyCandidate:
        excluded_reply_ids = excluded_reply_ids or set()
        values = self._available_context(context)

        candidates = [
            candidate
            for candidate in self._extract_replies(
                library_name,
                library,
            )
            if candidate.reply_id not in excluded_reply_ids
            and self._context_is_available(candidate, values)
        ]

        if preferred_group:
            group_candidates = [
                candidate
                for candidate in candidates
                if candidate.reply_group == preferred_group
            ]

            if group_candidates:
                candidates = group_candidates

        if preferred_emotion:
            emotional_candidates = [
                candidate
                for candidate in candidates
                if candidate.emotion in (
                    preferred_emotion,
                    None,
                )
            ]

            if emotional_candidates:
                candidates = emotional_candidates

        if not candidates:
            raise ReplySelectionError(
                f"No eligible replies found in library: {library_name}"
            )

        weights = [candidate.weight for candidate in candidates]
        selected = self.random.choices(
            candidates,
            weights=weights,
            k=1,
        )[0]

        selected.text = self._format_text(selected.text, values)
        return selected