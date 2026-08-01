from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from instagram_connector import DryRunInstagramConnector
from publisher_adapter import PublisherAdapter
from reply_agent import InstagramReplyAgent
from reply_dispatcher import ReplyDispatcher
from reply_logger import ReplyLogger
from reply_memory import ReplyMemory
from reply_queue import ReplyQueue
from reply_worker import ReplyWorker


@dataclass(slots=True)
class ServiceStatus:
    pending: int
    processing: int
    completed: int
    failed: int
    dispatched: int
    undispatched: int
    queue_total: int
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplyService:
    """
    Main runtime service for AIKO Auto Reply.

    Responsibilities:
    - process queued comments
    - generate validated replies
    - update reply memory
    - log executions
    - dispatch completed replies
    - prevent duplicate dispatch
    - report runtime status

    Current publisher:
    DryRunInstagramConnector

    No online Instagram reply is sent.
    """

    def __init__(
        self,
        *,
        worker: ReplyWorker,
        dispatcher: ReplyDispatcher,
        queue: ReplyQueue,
        state_path: str | Path,
    ) -> None:
        self.worker = worker
        self.dispatcher = dispatcher
        self.queue = queue
        self.state_path = Path(state_path).expanduser().resolve()

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.state_path.exists():
            self._write_service_state(
                {
                    "running": False,
                    "mode": None,
                    "started_at": None,
                    "stopped_at": None,
                    "last_cycle_at": None,
                    "processed_count": 0,
                    "dispatch_count": 0,
                    "error_count": 0,
                }
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read_service_state(self) -> dict[str, Any]:
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
            return {}

        return data if isinstance(data, dict) else {}

    def _write_service_state(
        self,
        state: dict[str, Any],
    ) -> None:
        temporary_path = self.state_path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                state,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(self.state_path)

    def _update_service_state(
        self,
        **updates: Any,
    ) -> None:
        state = self._read_service_state()
        state.update(updates)
        self._write_service_state(state)

    def process_once(
        self,
        *,
        dispatch: bool = True,
    ) -> bool:
        """
        Process one pending queue item.

        Returns True when an item was processed.
        """
        processed = self.worker.run_once()

        if not processed:
            return False

        state = self._read_service_state()

        self._update_service_state(
            processed_count=(
                int(state.get("processed_count", 0)) + 1
            ),
            last_cycle_at=self._now(),
        )

        if dispatch:
            dispatched = self.dispatcher.dispatch_once()

            if dispatched:
                current = self._read_service_state()

                self._update_service_state(
                    dispatch_count=(
                        int(current.get("dispatch_count", 0)) + 1
                    )
                )

        return True

    def process_batch(
        self,
        *,
        limit: int = 20,
        dispatch: bool = True,
    ) -> int:
        """
        Process up to `limit` pending queue items.
        """
        processed_count = 0

        for _ in range(limit):
            processed = self.process_once(
                dispatch=dispatch
            )

            if not processed:
                break

            processed_count += 1

        print(
            "[ReplyService] batch completed: "
            f"{processed_count} item(s)"
        )

        return processed_count

    def dispatch_all(
        self,
        *,
        limit: int = 100,
    ) -> int:
        """
        Dispatch completed but undispatched replies.
        """
        dispatched_count = 0

        for _ in range(limit):
            dispatched = self.dispatcher.dispatch_once()

            if not dispatched:
                break

            dispatched_count += 1

        state = self._read_service_state()

        self._update_service_state(
            dispatch_count=(
                int(state.get("dispatch_count", 0))
                + dispatched_count
            ),
            last_cycle_at=self._now(),
        )

        print(
            "[ReplyService] dispatch completed: "
            f"{dispatched_count} item(s)"
        )

        return dispatched_count

    def watch(
        self,
        *,
        poll_seconds: float = 5.0,
        batch_size: int = 10,
        dispatch: bool = True,
    ) -> None:
        """
        Continuously process queue items.
        """
        self._update_service_state(
            running=True,
            mode="watch",
            started_at=self._now(),
            stopped_at=None,
        )

        print(
            "[ReplyService] watch mode started "
            f"(poll={poll_seconds}s, batch={batch_size})"
        )

        try:
            while True:
                processed = self.process_batch(
                    limit=batch_size,
                    dispatch=dispatch,
                )

                self._update_service_state(
                    last_cycle_at=self._now()
                )

                if processed == 0:
                    time.sleep(poll_seconds)

        except KeyboardInterrupt:
            self._update_service_state(
                running=False,
                stopped_at=self._now(),
            )

            print("\n[ReplyService] stopped")

        except Exception:
            state = self._read_service_state()

            self._update_service_state(
                running=False,
                stopped_at=self._now(),
                error_count=(
                    int(state.get("error_count", 0)) + 1
                ),
            )

            raise

    def get_status(self) -> ServiceStatus:
        queue_items = self.queue.list_items()

        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }

        for item in queue_items:
            status = str(
                item.get("status", "unknown")
            )

            if status in status_counts:
                status_counts[status] += 1

        dispatch_state = self.dispatcher._read_state()

        dispatched_ids = set(
            dispatch_state.get(
                "dispatched_queue_ids",
                [],
            )
        )

        completed_ids = {
            str(item.get("queue_id"))
            for item in queue_items
            if item.get("status") == "completed"
        }

        undispatched = len(
            completed_ids - dispatched_ids
        )

        return ServiceStatus(
            pending=status_counts["pending"],
            processing=status_counts["processing"],
            completed=status_counts["completed"],
            failed=status_counts["failed"],
            dispatched=len(dispatched_ids),
            undispatched=undispatched,
            queue_total=len(queue_items),
            updated_at=self._now(),
        )

    def print_status(self) -> None:
        status = self.get_status()
        service_state = self._read_service_state()

        print("\nAIKO Reply Service")
        print("------------------")
        print(f"running:      {service_state.get('running', False)}")
        print(f"mode:         {service_state.get('mode')}")
        print(f"pending:      {status.pending}")
        print(f"processing:   {status.processing}")
        print(f"completed:    {status.completed}")
        print(f"failed:       {status.failed}")
        print(f"dispatched:   {status.dispatched}")
        print(f"undispatched: {status.undispatched}")
        print(f"queue total:  {status.queue_total}")
        print(
            "processed:    "
            f"{service_state.get('processed_count', 0)}"
        )
        print(
            "errors:       "
            f"{service_state.get('error_count', 0)}"
        )
        print()


def build_service() -> ReplyService:
    """
    Build the complete local Reply Service.
    """
    src_path = Path(__file__).resolve()
    runtime_root = src_path.parents[1]
    project_root = src_path.parents[3]

    reply_brain_root = (
        project_root
        / "03_personas"
        / "aiko"
        / "reply_brain"
    )

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

    memory = ReplyMemory(
        data_root / "reply_memory.json"
    )

    agent = InstagramReplyAgent(
        reply_brain_root=reply_brain_root,
        maximum_attempts=3,
    )

    worker = ReplyWorker(
        agent=agent,
        queue=queue,
        logger=logger,
        memory=memory,
        maximum_queue_attempts=3,
    )

    connector = DryRunInstagramConnector()

    publisher = PublisherAdapter(
        instagram_connector=connector
    )

    dispatcher = ReplyDispatcher(
        queue=queue,
        publisher=publisher,
        logger=logger,
        dispatch_state_path=(
            data_root / "dispatch_state.json"
        ),
    )

    return ReplyService(
        worker=worker,
        dispatcher=dispatcher,
        queue=queue,
        state_path=(
            data_root / "service_state.json"
        ),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Auto Reply Service"
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--once",
        action="store_true",
        help="Process one pending item and dispatch it.",
    )

    mode.add_argument(
        "--batch",
        type=int,
        metavar="COUNT",
        help="Process up to COUNT pending items.",
    )

    mode.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor and process the queue.",
    )

    mode.add_argument(
        "--dispatch-only",
        action="store_true",
        help="Dispatch completed undispatched replies only.",
    )

    mode.add_argument(
        "--status",
        action="store_true",
        help="Display current runtime status.",
    )

    parser.add_argument(
        "--poll",
        type=float,
        default=5.0,
        help="Watch-mode polling interval in seconds.",
    )

    parser.add_argument(
        "--watch-batch-size",
        type=int,
        default=10,
        help="Maximum items per watch cycle.",
    )

    parser.add_argument(
        "--no-dispatch",
        action="store_true",
        help="Process replies without dispatching.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    service = build_service()

    dispatch_enabled = not arguments.no_dispatch

    if arguments.status:
        service.print_status()
        return

    if arguments.once:
        service.process_once(
            dispatch=dispatch_enabled
        )
        return

    if arguments.batch is not None:
        if arguments.batch < 1:
            raise SystemExit(
                "--batch must be at least 1"
            )

        service.process_batch(
            limit=arguments.batch,
            dispatch=dispatch_enabled,
        )
        return

    if arguments.dispatch_only:
        service.dispatch_all()
        return

    if arguments.watch:
        service.watch(
            poll_seconds=arguments.poll,
            batch_size=arguments.watch_batch_size,
            dispatch=dispatch_enabled,
        )


if __name__ == "__main__":
    main()