from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProductionWorkerError(RuntimeError):
    """Raised when a production queue cannot be processed safely."""


class ProductionWorker:
    """
    Manual image-production worker for AIKO OS.

    This worker does not call an image-generation API.

    It:
    - reads image_queue.json
    - finds the next pending task
    - loads the prompt
    - copies the prompt to the macOS clipboard
    - displays the expected image output path
    - checks whether the generated image exists
    - updates queue and manifest status
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    def __init__(
        self,
        *,
        production_root: str | Path,
    ) -> None:
        self.production_root = Path(
            production_root
        ).expanduser().resolve()

        self.queue_path = (
            self.production_root
            / "queues"
            / "image_queue.json"
        )

        self.manifest_path = (
            self.production_root
            / "production_manifest.json"
        )

        if not self.production_root.exists():
            raise ProductionWorkerError(
                f"Production folder does not exist: "
                f"{self.production_root}"
            )

        if not self.queue_path.exists():
            raise ProductionWorkerError(
                f"Image queue does not exist: "
                f"{self.queue_path}"
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict[str, Any]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except FileNotFoundError as exc:
            raise ProductionWorkerError(
                f"JSON file not found: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProductionWorkerError(
                f"Invalid JSON in {path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ProductionWorkerError(
                f"Expected JSON object in {path}"
            )

        return data

    @staticmethod
    def _write_json(
        path: Path,
        data: dict[str, Any],
    ) -> None:
        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
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

        temporary_path.replace(path)

    @staticmethod
    def _copy_to_clipboard(
        text: str,
    ) -> None:
        """
        Copy text to the macOS clipboard using pbcopy.
        """
        try:
            subprocess.run(
                ["pbcopy"],
                input=text,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise ProductionWorkerError(
                "pbcopy is unavailable. "
                "This worker currently expects macOS."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ProductionWorkerError(
                "Unable to copy prompt to clipboard."
            ) from exc

    @staticmethod
    def _resolve_path(
        value: str,
        *,
        base: Path,
    ) -> Path:
        path = Path(value).expanduser()

        if path.is_absolute():
            return path.resolve()

        return (
            base
            / path
        ).resolve()

    def _load_queue(
        self,
    ) -> dict[str, Any]:
        queue = self._read_json(
            self.queue_path
        )

        tasks = queue.get("tasks")

        if not isinstance(tasks, list):
            raise ProductionWorkerError(
                "image_queue.json must contain a tasks list"
            )

        return queue

    def _save_queue(
        self,
        queue: dict[str, Any],
    ) -> None:
        tasks = queue.get("tasks", [])

        completed = sum(
            1
            for task in tasks
            if task.get("status") == "completed"
        )

        failed = sum(
            1
            for task in tasks
            if task.get("status") == "failed"
        )

        pending = sum(
            1
            for task in tasks
            if task.get("status") == "pending"
        )

        if tasks and completed == len(tasks):
            queue["status"] = "completed"
        elif failed:
            queue["status"] = "attention_required"
        elif completed:
            queue["status"] = "in_progress"
        else:
            queue["status"] = "pending"

        queue["completed_count"] = completed
        queue["failed_count"] = failed
        queue["pending_count"] = pending
        queue["updated_at"] = self._now()

        self._write_json(
            self.queue_path,
            queue,
        )

        self._update_manifest(
            queue=queue
        )

    def _update_manifest(
        self,
        *,
        queue: dict[str, Any],
    ) -> None:
        if not self.manifest_path.exists():
            return

        manifest = self._read_json(
            self.manifest_path
        )

        tasks = queue.get("tasks", [])

        manifest["image_tasks"] = tasks

        completed = sum(
            1
            for task in tasks
            if task.get("status") == "completed"
        )

        failed = sum(
            1
            for task in tasks
            if task.get("status") == "failed"
        )

        if tasks and completed == len(tasks):
            manifest["status"] = "images_complete"
        elif failed:
            manifest["status"] = "images_in_progress"
        elif completed:
            manifest["status"] = "images_in_progress"
        else:
            manifest["status"] = "ready_for_images"

        manifest["updated_at"] = self._now()

        self._write_json(
            self.manifest_path,
            manifest,
        )

    @staticmethod
    def _task_status(
        task: dict[str, Any],
    ) -> str:
        return str(
            task.get("status", "pending")
        ).strip().lower()

    def next_task(
        self,
    ) -> dict[str, Any] | None:
        """
        Return the current in-progress task first.

        Only return a new pending task when no task is already in progress.
        This prevents --next from skipping unfinished images.
        """
        queue = self._load_queue()
        tasks = queue.get("tasks", [])

        if not isinstance(tasks, list):
            raise ProductionWorkerError(
            "image_queue.json must contain a tasks list"
            )

        # Never skip an unfinished task.
        for task in tasks:
            if not isinstance(task, dict):
                continue

            if self._task_status(task) == "in_progress":
                return task

        # No current task: return the first pending task.
        for task in tasks:
            if not isinstance(task, dict):
                continue

            if self._task_status(task) == "pending":
                return task
            
        return None

    def prepare_next(
        self,
    ) -> dict[str, Any] | None:
        """
        Prepare one image task.

        Rules:
        - Keep returning an existing in-progress task.
        - Do not skip unfinished tasks.
        - Only start the first pending task.
        """
        queue = self._load_queue()
        tasks = queue.get("tasks", [])

        if not isinstance(tasks, list):
            raise ProductionWorkerError(
                "image_queue.json must contain a tasks list"
            )

        task: dict[str, Any] | None = None

        # Keep the current unfinished task.
        for candidate in tasks:
            if not isinstance(candidate, dict):
                continue

            if self._task_status(candidate) == "in_progress":
                task = candidate
                break

        # Otherwise select the first pending task.
        if task is None:
            for candidate in tasks:
                if not isinstance(candidate, dict):
                    continue

                if self._task_status(candidate) == "pending":
                    task = candidate
                    break

        if task is None:
            print("[ProductionWorker] no pending image tasks")
            return None

        prompt_value = str(
            task.get("prompt_file", "")
        ).strip()

        output_value = str(
            task.get("expected_output_file", "")
        ).strip()

        if not prompt_value:
            raise ProductionWorkerError(
                f"Task {task.get('task_id')} has no prompt_file"
            )

        if not output_value:
            raise ProductionWorkerError(
                f"Task {task.get('task_id')} "
                "has no expected_output_file"
            )

        prompt_path = self._resolve_path(
            prompt_value,
            base=self.production_root,
        )

        output_path = self._resolve_path(
            output_value,
            base=self.production_root,
        )

        if not prompt_path.exists():
            raise ProductionWorkerError(
                f"Prompt file not found: {prompt_path}"
            )

        prompt = prompt_path.read_text(
            encoding="utf-8"
        ).strip()

        if not prompt:
            raise ProductionWorkerError(
                f"Prompt file is empty: {prompt_path}"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_status = self._task_status(task)

        if current_status == "in_progress":
            print()
            print("AIKO Image Production Task")
            print("--------------------------")
            print(f"task ID:      {task.get('task_id')}")
            print(f"type:         {task.get('content_type')}")
            print(f"title:        {task.get('title')}")
            print(f"prompt:       {prompt_path}")
            print(f"save image:   {output_path}")
            print()
            print(
                "This task is still in progress. "
                "Save the image and run --complete."
            )
            print()

            return task

        self._copy_to_clipboard(prompt)

        task["status"] = "in_progress"
        task["started_at"] = self._now()
        task["attempts"] = (
            int(task.get("attempts", 0)) + 1
        )
        task["error"] = None
        task.pop("failed_at", None)

        self._save_queue(queue)

        print()
        print("AIKO Image Production Task")
        print("--------------------------")
        print(f"task ID:      {task.get('task_id')}")
        print(f"type:         {task.get('content_type')}")
        print(f"title:        {task.get('title')}")
        print(f"prompt:       {prompt_path}")
        print(f"save image:   {output_path}")
        print()
        print("Prompt copied to Clipboard.")
        print(
            "Generate the image in ChatGPT, "
            "then save it to the path above."
        )
        print()

        return task
    
    def current_task(
        self,
    ) -> dict[str, Any] | None:
        queue = self._load_queue()

        for task in queue["tasks"]:
            if not isinstance(task, dict):
                continue

            if self._task_status(task) == "in_progress":
                return task

        return None

    def complete_current(
        self,
    ) -> bool:
        """
        Confirm the current image task after its file has been saved.
        """
        queue = self._load_queue()

        for task in queue["tasks"]:
            if not isinstance(task, dict):
                continue

            if self._task_status(task) != "in_progress":
                continue

            output_path = self._resolve_path(
                str(task["expected_output_file"]),
                base=self.production_root,
            )

            if not output_path.exists():
                print(
                    "[ProductionWorker] image file not found:"
                )
                print(output_path)
                return False

            if not output_path.is_file():
                print(
                    "[ProductionWorker] output path is not a file:"
                )
                print(output_path)
                return False

            if output_path.suffix.lower() not in (
                self.IMAGE_EXTENSIONS
            ):
                print(
                    "[ProductionWorker] unsupported image type:"
                )
                print(output_path.suffix)
                return False

            if output_path.stat().st_size == 0:
                print(
                    "[ProductionWorker] image file is empty:"
                )
                print(output_path)
                return False

            task["status"] = "completed"
            task["generated_at"] = self._now()
            task["completed_at"] = self._now()
            task["file_size_bytes"] = (
                output_path.stat().st_size
            )
            task["error"] = None

            self._save_queue(queue)

            print()
            print("[ProductionWorker] completed")
            print(f"task:  {task.get('task_id')}")
            print(f"file:  {output_path}")
            print()

            return True

        print(
            "[ProductionWorker] "
            "no image task is currently in progress"
        )
        return False

    def mark_current_failed(
        self,
        *,
        error: str,
        retry: bool = True,
    ) -> bool:
        queue = self._load_queue()

        for task in queue["tasks"]:
            if not isinstance(task, dict):
                continue

            if self._task_status(task) != "in_progress":
                continue

            task["status"] = (
                "pending"
                if retry
                else "failed"
            )
            task["error"] = error
            task["failed_at"] = self._now()

            self._save_queue(queue)

            return True

        return False

    def sync_files(
        self,
    ) -> int:
        """
        Mark tasks completed when their expected image files already exist.

        Useful if images were saved before running --complete.
        """
        queue = self._load_queue()
        completed_count = 0

        for task in queue["tasks"]:
            if not isinstance(task, dict):
                continue

            if self._task_status(task) == "completed":
                continue

            output_path = self._resolve_path(
                str(task.get("expected_output_file", "")),
                base=self.production_root,
            )

            if not output_path.exists():
                continue

            if not output_path.is_file():
                continue

            if output_path.suffix.lower() not in (
                self.IMAGE_EXTENSIONS
            ):
                continue

            if output_path.stat().st_size == 0:
                continue

            task["status"] = "completed"
            task["generated_at"] = (
                task.get("generated_at")
                or self._now()
            )
            task["completed_at"] = self._now()
            task["file_size_bytes"] = (
                output_path.stat().st_size
            )
            task["error"] = None

            completed_count += 1

        if completed_count:
            self._save_queue(queue)

        print(
            "[ProductionWorker] synced "
            f"{completed_count} image task(s)"
        )

        return completed_count

    def status(
        self,
    ) -> dict[str, Any]:
        queue = self._load_queue()
        tasks = queue["tasks"]

        status_counts = {
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
        }

        for task in tasks:
            if not isinstance(task, dict):
                continue

            status = self._task_status(task)

            if status in status_counts:
                status_counts[status] += 1

        current = self.current_task()

        return {
            "production_root": str(
                self.production_root
            ),
            "queue_status": queue.get(
                "status",
                "unknown",
            ),
            "total": len(tasks),
            **status_counts,
            "current_task": (
                current.get("task_id")
                if current
                else None
            ),
        }

    def print_status(
        self,
    ) -> None:
        status = self.status()

        print()
        print("AIKO Image Worker Status")
        print("------------------------")
        print(
            f"queue status: {status['queue_status']}"
        )
        print(f"total:        {status['total']}")
        print(f"pending:      {status['pending']}")
        print(
            f"in progress:  {status['in_progress']}"
        )
        print(f"completed:    {status['completed']}")
        print(f"failed:       {status['failed']}")
        print(
            f"current task: {status['current_task']}"
        )
        print()


def build_production_worker(
    production_date: str,
) -> ProductionWorker:
    runtime_root = (
        Path(__file__).resolve().parents[1]
    )

    production_root = (
        runtime_root
        / "output"
        / production_date
    )

    return ProductionWorker(
        production_root=production_root
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Manual Image Production Worker"
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Production date in YYYY-MM-DD format.",
    )

    action = parser.add_mutually_exclusive_group(
        required=True
    )

    action.add_argument(
        "--next",
        action="store_true",
        help="Prepare the next pending image task.",
    )

    action.add_argument(
        "--complete",
        action="store_true",
        help="Complete the current task after saving its image.",
    )

    action.add_argument(
        "--sync",
        action="store_true",
        help="Mark existing image files as completed.",
    )

    action.add_argument(
        "--status",
        action="store_true",
        help="Show image queue status.",
    )

    action.add_argument(
        "--retry",
        action="store_true",
        help="Return the current task to pending.",
    )

    action.add_argument(
        "--fail",
        action="store_true",
        help="Mark the current task as failed.",
    )

    parser.add_argument(
        "--error",
        default="manual production failed",
        help="Error text used with --retry or --fail.",
    )

    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()

    worker = build_production_worker(
        arguments.date
    )

    if arguments.next:
        worker.prepare_next()
        return

    if arguments.complete:
        worker.complete_current()
        return

    if arguments.sync:
        worker.sync_files()
        return

    if arguments.status:
        worker.print_status()
        return

    if arguments.retry:
        changed = worker.mark_current_failed(
            error=arguments.error,
            retry=True,
        )

        print(
            "[ProductionWorker] "
            + (
                "current task returned to pending"
                if changed
                else "no current task found"
            )
        )
        return

    if arguments.fail:
        changed = worker.mark_current_failed(
            error=arguments.error,
            retry=False,
        )

        print(
            "[ProductionWorker] "
            + (
                "current task marked failed"
                if changed
                else "no current task found"
            )
        )
        return

    raise ProductionWorkerError(
        "No worker action was selected."
    )


if __name__ == "__main__":
    main()