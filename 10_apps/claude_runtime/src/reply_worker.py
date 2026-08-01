from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from reply_agent import InstagramReplyAgent
from reply_logger import ReplyLogger
from reply_memory import ReplyMemory
from reply_models import CommentInput, ContentContext
from reply_queue import ReplyQueue


class ReplyWorker:
    """
    Processes queued reply requests one at a time.

    Current version:
    - reads local JSON queue
    - calls InstagramReplyAgent
    - stores result
    - writes logs
    - updates local reply memory

    It does not publish to Instagram yet.
    """

    def __init__(
        self,
        *,
        agent: InstagramReplyAgent,
        queue: ReplyQueue,
        logger: ReplyLogger,
        memory: ReplyMemory,
        maximum_queue_attempts: int = 3,
    ) -> None:
        self.agent = agent
        self.queue = queue
        self.logger = logger
        self.memory = memory
        self.maximum_queue_attempts = maximum_queue_attempts

    @staticmethod
    def _parse_comment(
        payload: dict[str, Any],
    ) -> CommentInput:
        comment_data = payload.get("comment", {})
        content_data = payload.get("content", {})

        return CommentInput(
            request_id=str(payload["request_id"]),
            comment_id=str(comment_data["comment_id"]),
            follower_id=str(comment_data["follower_id"]),
            text=str(comment_data["text"]),
            platform=str(payload.get("platform", "instagram_feed")),
            content_id=str(content_data["content_id"]),
            follower_username=comment_data.get(
                "follower_username"
            ),
            parent_comment_id=comment_data.get(
                "parent_comment_id"
            ),
            language_hint=comment_data.get(
                "language_hint"
            ),
            timestamp=str(
                payload.get("timestamp")
                or comment_data.get("created_at")
                or ""
            ),
        )

    @staticmethod
    def _parse_content(
        payload: dict[str, Any],
    ) -> ContentContext:
        content_data = payload.get("content", {})
        production_metadata = payload.get(
            "production_metadata"
        ) or {}

        known_fields = {
            "content_id",
            "content_type",
            "topic",
            "caption",
            "location",
            "city",
            "country",
            "venue",
            "activity",
            "emotion",
            "story_stage",
            "ai_generated",
            "sponsored",
            "gifted",
            "hosted",
        }

        extra_metadata = {
            key: value
            for key, value in content_data.items()
            if key not in known_fields
        }

        extra_metadata.update(production_metadata)

        return ContentContext(
            content_type=str(
                content_data.get(
                    "content_type",
                    "feed",
                )
            ),
            topic=content_data.get("topic"),
            caption=content_data.get("caption"),
            location=content_data.get("location"),
            city=content_data.get("city"),
            country=content_data.get("country"),
            venue=content_data.get("venue"),
            activity=content_data.get("activity"),
            emotion=content_data.get("emotion"),
            story_stage=content_data.get(
                "story_stage"
            ),
            ai_generated=bool(
                content_data.get(
                    "ai_generated",
                    True,
                )
            ),
            sponsored=bool(
                content_data.get(
                    "sponsored",
                    False,
                )
            ),
            gifted=bool(
                content_data.get(
                    "gifted",
                    False,
                )
            ),
            hosted=bool(
                content_data.get(
                    "hosted",
                    False,
                )
            ),
            metadata=extra_metadata,
        )

    def process_item(
        self,
        queue_item: dict[str, Any],
    ) -> bool:
        queue_id = str(queue_item["queue_id"])
        payload = queue_item.get("payload", {})
        request_id = str(
            payload.get("request_id", queue_id)
        )

        self.logger.log_started(
            queue_id=queue_id,
            request_id=request_id,
        )

        try:
            comment = self._parse_comment(payload)
            content = self._parse_content(payload)

            excluded_reply_ids = (
                self.memory.get_excluded_reply_ids(
                    follower_id=comment.follower_id
                )
            )

            result = self.agent.process(
                comment=comment,
                content=content,
                excluded_reply_ids=excluded_reply_ids,
            )

            result_data = result.to_dict()

            self.queue.complete(
                queue_id=queue_id,
                result=result_data,
            )

            if result.approved:
                self.memory.remember_reply(
                    follower_id=comment.follower_id,
                    content_id=comment.content_id,
                    request_id=comment.request_id,
                    reply_id=result.reply_id,
                    reply_text=result.reply_text,
                    library=result.library,
                    reply_group=result.reply_group,
                    primary_intent=(
                        result.classification.primary_intent
                    ),
                    detected_emotion=(
                        result.classification.detected_emotion
                    ),
                )

            self.logger.log_completed(
                queue_id=queue_id,
                result=result_data,
            )

            print(
                f"[ReplyWorker] completed {queue_id}: "
                f"{result.action}"
            )
            return True

        except Exception as exc:
            attempts = int(
                queue_item.get("attempts", 1)
            )

            retry = attempts < self.maximum_queue_attempts

            error_text = (
                f"{type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            )

            self.queue.fail(
                queue_id=queue_id,
                error=error_text,
                retry=retry,
            )

            self.logger.log_failed(
                queue_id=queue_id,
                error=error_text,
                retry=retry,
            )

            print(
                f"[ReplyWorker] failed {queue_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def run_once(self) -> bool:
        queue_item = self.queue.next_pending()

        if queue_item is None:
            print("[ReplyWorker] no pending items")
            return False

        return self.process_item(queue_item)

    def run_forever(
        self,
        poll_seconds: float = 5.0,
    ) -> None:
        print(
            "[ReplyWorker] started "
            f"(poll interval: {poll_seconds}s)"
        )

        try:
            while True:
                queue_item = self.queue.next_pending()

                if queue_item is None:
                    time.sleep(poll_seconds)
                    continue

                self.process_item(queue_item)

        except KeyboardInterrupt:
            print("\n[ReplyWorker] stopped")


def build_worker() -> ReplyWorker:
    runtime_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[3]

    reply_brain_root = (
        project_root
        / "03_personas"
        / "aiko"
        / "reply_brain"
    )

    data_root = runtime_root / "output" / "reply_runtime"

    agent = InstagramReplyAgent(
        reply_brain_root=reply_brain_root,
        maximum_attempts=3,
    )

    queue = ReplyQueue(
        data_root / "reply_queue.json"
    )

    logger = ReplyLogger(
        data_root / "reply_events.jsonl"
    )

    memory = ReplyMemory(
        data_root / "reply_memory.json"
    )

    return ReplyWorker(
        agent=agent,
        queue=queue,
        logger=logger,
        memory=memory,
        maximum_queue_attempts=3,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Reply Worker"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one pending queue item and exit.",
    )

    parser.add_argument(
        "--poll",
        type=float,
        default=5.0,
        help="Polling interval in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    worker = build_worker()

    if arguments.once:
        worker.run_once()
        return

    worker.run_forever(
        poll_seconds=arguments.poll
    )


if __name__ == "__main__":
    main()