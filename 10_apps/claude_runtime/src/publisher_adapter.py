from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from instagram_connector import (
    InstagramConnectorProtocol,
    PublishResponse,
)


@dataclass(slots=True)
class DispatchRequest:
    request_id: str
    platform: str
    action: str
    comment_id: str
    approved: bool
    reply_text: str | None
    moderation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PublisherAdapter:
    """
    Converts Reply Brain actions into platform connector operations.
    """

    def __init__(
        self,
        instagram_connector: InstagramConnectorProtocol,
    ) -> None:
        self.instagram_connector = instagram_connector

    def dispatch(
        self,
        request: DispatchRequest,
    ) -> PublishResponse:
        if request.platform not in {
            "instagram_feed",
            "instagram_reel",
            "instagram_story",
            "instagram_thread",
        }:
            return PublishResponse(
                success=False,
                platform=request.platform,
                action=request.action,
                comment_id=request.comment_id,
                error="unsupported_platform",
            )

        if request.action == "publish":
            if not request.approved:
                return PublishResponse(
                    success=False,
                    platform=request.platform,
                    action=request.action,
                    comment_id=request.comment_id,
                    error="reply_not_approved",
                )

            if not request.reply_text:
                return PublishResponse(
                    success=False,
                    platform=request.platform,
                    action=request.action,
                    comment_id=request.comment_id,
                    error="missing_reply_text",
                )

            return self.instagram_connector.publish_comment_reply(
                comment_id=request.comment_id,
                reply_text=request.reply_text,
            )

        if request.action == "ignore":
            return PublishResponse(
                success=True,
                platform=request.platform,
                action="ignore",
                comment_id=request.comment_id,
            )

        if request.action == "block_recommended":
            return self.instagram_connector.hide_comment(
                comment_id=request.comment_id,
            )

        if request.action in {
            "hold_for_review",
            "escalate",
            "clarification_required",
        }:
            return PublishResponse(
                success=True,
                platform=request.platform,
                action=request.action,
                comment_id=request.comment_id,
            )

        return PublishResponse(
            success=False,
            platform=request.platform,
            action=request.action,
            comment_id=request.comment_id,
            error="unsupported_action",
        )