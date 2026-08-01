from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class ReplyMemory:
    """
    Stores lightweight reply history for anti-repeat and continuity.

    This is local runtime memory only.
    It does not publish or send anything.
    """

    def __init__(self, memory_path: str | Path) -> None:
        self.path = Path(memory_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

        if not self.path.exists():
            self._write(
                {
                    "replies": [],
                    "followers": {},
                }
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def hash_follower_id(follower_id: str) -> str:
        return hashlib.sha256(
            follower_id.encode("utf-8")
        ).hexdigest()

    def _read(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "replies": [],
                "followers": {},
            }

        if not isinstance(data, dict):
            return {
                "replies": [],
                "followers": {},
            }

        data.setdefault("replies", [])
        data.setdefault("followers", {})
        return data

    def _write(self, data: dict[str, Any]) -> None:
        temporary_path = self.path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(self.path)

    def get_excluded_reply_ids(
        self,
        follower_id: str,
        limit: int = 30,
    ) -> set[str]:
        follower_hash = self.hash_follower_id(follower_id)

        with self._lock:
            data = self._read()

        replies = data.get("replies", [])
        matching = [
            item
            for item in replies
            if item.get("follower_hash") == follower_hash
            and item.get("reply_id")
        ]

        matching.sort(
            key=lambda item: item.get("timestamp", ""),
            reverse=True,
        )

        return {
            str(item["reply_id"])
            for item in matching[:limit]
        }

    def remember_reply(
        self,
        *,
        follower_id: str,
        content_id: str,
        request_id: str,
        reply_id: str | None,
        reply_text: str | None,
        library: str | None,
        reply_group: str | None,
        primary_intent: str,
        detected_emotion: str,
    ) -> None:
        follower_hash = self.hash_follower_id(follower_id)

        with self._lock:
            data = self._read()

            data["replies"].append(
                {
                    "timestamp": self._now(),
                    "follower_hash": follower_hash,
                    "content_id": content_id,
                    "request_id": request_id,
                    "reply_id": reply_id,
                    "reply_text": reply_text,
                    "library": library,
                    "reply_group": reply_group,
                    "primary_intent": primary_intent,
                    "detected_emotion": detected_emotion,
                }
            )

            followers = data["followers"]
            follower = followers.setdefault(
                follower_hash,
                {
                    "first_seen": self._now(),
                    "last_seen": self._now(),
                    "reply_count": 0,
                    "recent_intents": [],
                },
            )

            follower["last_seen"] = self._now()
            follower["reply_count"] = int(
                follower.get("reply_count", 0)
            ) + 1

            recent_intents = list(
                follower.get("recent_intents", [])
            )

            recent_intents.append(primary_intent)
            follower["recent_intents"] = recent_intents[-10:]

            # Keep local memory from growing forever.
            data["replies"] = data["replies"][-5000:]

            self._write(data)