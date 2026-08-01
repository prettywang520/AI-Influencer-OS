from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(slots=True)
class PublishResponse:
    success: bool
    platform: str
    action: str
    comment_id: str
    reply_text: str | None = None
    platform_reply_id: str | None = None
    error: str | None = None
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstagramConnectorProtocol(Protocol):
    def publish_comment_reply(
        self,
        *,
        comment_id: str,
        reply_text: str,
    ) -> PublishResponse:
        ...

    def hide_comment(
        self,
        *,
        comment_id: str,
    ) -> PublishResponse:
        ...

    def delete_comment(
        self,
        *,
        comment_id: str,
    ) -> PublishResponse:
        ...


class DryRunInstagramConnector:
    """
    Safe local connector.

    It simulates Instagram publishing but does not send anything online.
    Use this until a real approved Instagram integration is available.
    """

    platform = "instagram"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def publish_comment_reply(
        self,
        *,
        comment_id: str,
        reply_text: str,
    ) -> PublishResponse:
        print(
            "[DryRunInstagramConnector] "
            f"reply to {comment_id}: {reply_text}"
        )

        return PublishResponse(
            success=True,
            platform=self.platform,
            action="publish_reply",
            comment_id=comment_id,
            reply_text=reply_text,
            platform_reply_id=f"dry-run-reply-{comment_id}",
            published_at=self._now(),
        )

    def hide_comment(
        self,
        *,
        comment_id: str,
    ) -> PublishResponse:
        print(
            "[DryRunInstagramConnector] "
            f"hide comment {comment_id}"
        )

        return PublishResponse(
            success=True,
            platform=self.platform,
            action="hide_comment",
            comment_id=comment_id,
            published_at=self._now(),
        )

    def delete_comment(
        self,
        *,
        comment_id: str,
    ) -> PublishResponse:
        print(
            "[DryRunInstagramConnector] "
            f"delete comment {comment_id}"
        )

        return PublishResponse(
            success=True,
            platform=self.platform,
            action="delete_comment",
            comment_id=comment_id,
            published_at=self._now(),
        )