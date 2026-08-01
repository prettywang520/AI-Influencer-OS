from __future__ import annotations

from pathlib import Path

from reply_queue import ReplyQueue
from reply_worker import build_worker


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]

    queue = ReplyQueue(
        runtime_root
        / "output"
        / "reply_runtime"
        / "reply_queue.json"
    )

    queue_id = queue.enqueue(
        {
            "request_id": "worker-test-001",
            "platform": "instagram_feed",
            "timestamp": "2026-07-26T15:45:00+08:00",
            "comment": {
                "comment_id": "comment-worker-001",
                "follower_id": "sample-follower-001",
                "follower_username": "sample_user",
                "text": "beautiful photo! where is this?",
                "language_hint": "english",
            },
            "content": {
                "content_id": "feed-2026-07-26",
                "content_type": "feed",
                "topic": "bookstore",
                "caption": "a rainy afternoon in tokyo",
                "location": "Daikanyama, Tokyo",
                "city": "Tokyo",
                "country": "Japan",
                "venue": "Daikanyama T-Site",
                "activity": "browsing a photography book",
                "emotion": "inspired",
                "story_stage": "hero_moment",
                "ai_generated": True,
                "sponsored": False,
                "gifted": False,
                "hosted": False,
            },
        }
    )

    print(f"queued: {queue_id}")

    worker = build_worker()
    worker.run_once()


if __name__ == "__main__":
    main()