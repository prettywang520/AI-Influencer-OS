from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import ProductionWorker


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        production_root = Path(temp_dir)

        prompts_root = (
            production_root
            / "prompts"
        )

        queues_root = (
            production_root
            / "queues"
        )

        images_root = (
            production_root
            / "images"
            / "feed"
        )

        prompts_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        queues_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        images_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        prompt_path = (
            prompts_root
            / "feed_prompt.txt"
        )

        output_path = (
            images_root
            / "feed_01.png"
        )

        prompt_path.write_text(
            "Create one test travel image.",
            encoding="utf-8",
        )

        queue = {
            "version": "1.0",
            "status": "pending",
            "tasks": [
                {
                    "task_id": "test-image-001",
                    "content_type": "feed",
                    "title": "Test Feed",
                    "prompt_file": str(
                        prompt_path
                    ),
                    "expected_output_file": str(
                        output_path
                    ),
                    "status": "pending",
                    "attempts": 0,
                }
            ],
        }

        (
            queues_root
            / "image_queue.json"
        ).write_text(
            json.dumps(
                queue,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        worker = ProductionWorker(
            production_root=production_root
        )

        task = worker.prepare_next()

        assert task is not None
        assert task["task_id"] == "test-image-001"

        status = worker.status()

        assert status["in_progress"] == 1

        # Minimal non-empty fake file for queue testing.
        output_path.write_bytes(
            b"test-image-data"
        )

        assert worker.complete_current()

        status = worker.status()

        assert status["completed"] == 1
        assert status["pending"] == 0

        print("Production Worker test passed.")


if __name__ == "__main__":
    main()