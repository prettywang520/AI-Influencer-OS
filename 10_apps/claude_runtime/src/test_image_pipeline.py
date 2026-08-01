from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from .image_pipeline import ImagePipeline


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        production_root = Path(temp_dir)

        queues_root = (
            production_root
            / "queues"
        )

        image_root = (
            production_root
            / "images"
            / "feed"
        )

        queues_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = (
            image_root
            / "feed_01.png"
        )

        image = Image.new(
            "RGB",
            (1080, 1350),
            "white",
        )

        image.save(image_path)

        queue = {
            "version": "1.0",
            "status": "completed",
            "tasks": [
                {
                    "task_id": "test-feed-001",
                    "content_type": "feed",
                    "title": "Test Feed",
                    "expected_output_file": str(
                        image_path
                    ),
                    "status": "completed",
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

        pipeline = ImagePipeline(
            production_root=production_root
        )

        manifest = pipeline.run(
            create_thumbnails=True,
            require_all=True,
        )

        assert manifest.status == "complete"
        assert manifest.completed == 1
        assert manifest.missing == 0
        assert manifest.invalid == 0
        assert manifest.feed_count == 1
        assert manifest.thumbnails_created == 1

        metadata_path = (
            production_root
            / "metadata"
            / "images"
            / "test-feed-001.json"
        )

        assert metadata_path.exists()
        assert pipeline.image_manifest_path.exists()

        print("Image Pipeline test passed.")


if __name__ == "__main__":
    main()