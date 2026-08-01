from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from instagram_connector import DryRunInstagramConnector
from publisher_adapter import DispatchRequest, PublisherAdapter
from reply_logger import ReplyLogger
from reply_queue import ReplyQueue


class ReplyDispatcher:
    """
    Dispatches completed Reply Worker results to a platform connector.

    Current version uses DryRunInstagramConnector, so nothing is
    published online.
    """

    def __init__(
        self,
        *,
        queue: ReplyQueue,
        publisher: PublisherAdapter,
        logger: ReplyLogger,
        dispatch_state_path: str | Path,
    ) -> None:
        self.queue = queue
        self.publisher = publisher
        self.logger = logger
        self.state_path = Path(
            dispatch_state_path
        ).expanduser().resolve()

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.state_path.exists():
            self._write_state(
                {
                    "dispatched_queue_ids": [],
                }
            )

    def _read_state(self) -> dict[str, Any]:
        try:
            with self.state_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
        ):
            return {
                "dispatched_queue_ids": [],
            }

        if not isinstance(data, dict):
            return {
                "dispatched_queue_ids": [],
            }

        data.setdefault(
            "dispatched_queue_ids",
            [],
        )
        return data

    def _write_state(
        self,
        data: dict[str, Any],
    ) -> None:
        temporary_path = self.state_path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(self.state_path)

    @staticmethod
    def _build_request(
        queue_item: dict[str, Any],
    ) -> DispatchRequest:
        payload = queue_item.get("payload", {})
        result = queue_item.get("result") or {}
        comment = payload.get("comment", {})

        return DispatchRequest(
            request_id=str(
                payload.get(
                    "request_id",
                    queue_item.get("queue_id"),
                )
            ),
            platform=str(
                payload.get(
                    "platform",
                    "instagram_feed",
                )
            ),
            action=str(
                result.get(
                    "action",
                    "hold_for_review",
                )
            ),
            comment_id=str(
                comment.get(
                    "comment_id",
                    "",
                )
            ),
            approved=bool(
                result.get(
                    "approved",
                    False,
                )
            ),
            reply_text=result.get(
                "reply_text"
            ),
            moderation_reason=(
                result.get("classification", {})
                .get("moderation_reason")
            ),
        )

    def dispatch_item(
        self,
        queue_item: dict[str, Any],
    ) -> bool:
        queue_id = str(queue_item["queue_id"])
        request = self._build_request(queue_item)

        response = self.publisher.dispatch(
            request
        )

        self.logger.write(
            "reply_dispatch_result",
            {
                "queue_id": queue_id,
                "dispatch_request": (
                    request.to_dict()
                ),
                "publish_response": (
                    response.to_dict()
                ),
            },
        )

        if not response.success:
            print(
                "[ReplyDispatcher] failed "
                f"{queue_id}: {response.error}"
            )
            return False

        state = self._read_state()
        dispatched = set(
            state["dispatched_queue_ids"]
        )
        dispatched.add(queue_id)

        state["dispatched_queue_ids"] = sorted(
            dispatched
        )
        self._write_state(state)

        print(
            "[ReplyDispatcher] completed "
            f"{queue_id}: {response.action}"
        )
        return True

    def dispatch_once(self) -> bool:
        state = self._read_state()
        dispatched = set(
            state["dispatched_queue_ids"]
        )

        completed_items = self.queue.list_items(
            status="completed"
        )

        for item in completed_items:
            queue_id = str(item["queue_id"])

            if queue_id in dispatched:
                continue

            return self.dispatch_item(item)

        print(
            "[ReplyDispatcher] "
            "no completed undispatched items"
        )
        return False


def build_dispatcher() -> ReplyDispatcher:
    runtime_root = Path(__file__).resolve().parents[1]

    data_root = (
        runtime_root
        / "output"
        / "reply_runtime"
    )

    queue = ReplyQueue(
        data_root / "reply_queue.json"
    )

    logger = ReplyLogger(
        data_root / "reply_events.jsonl"
    )

    connector = DryRunInstagramConnector()

    publisher = PublisherAdapter(
        instagram_connector=connector
    )

    return ReplyDispatcher(
        queue=queue,
        publisher=publisher,
        logger=logger,
        dispatch_state_path=(
            data_root / "dispatch_state.json"
        ),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Reply Dispatcher"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Dispatch one completed reply.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dispatcher = build_dispatcher()

    if arguments.once:
        dispatcher.dispatch_once()
        return

    dispatcher.dispatch_once()


if __name__ == "__main__":
    main()