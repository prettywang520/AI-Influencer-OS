from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .approval_service import ApprovalService, build_approval_service
from .instagram_models import (
    ApprovalRecord,
    InstagramSessionError,
    PrepareRepliesResult,
)
from .reply_brain import ReplyBrain, build_reply_brain
from .reply_safety import resolve_safety_route

COMMENT_QUEUE_FILENAME = "comment_queue.json"


def _runtime_root() -> Path:
    """
    community_service.py location:

    10_apps/claude_runtime/src/social/community_service.py

    parents[2] resolves to 10_apps/claude_runtime.
    """
    return Path(__file__).resolve().parents[2]


def _default_comment_queue_path() -> Path:
    return (
        _runtime_root()
        / "output"
        / "community"
        / "queues"
        / COMMENT_QUEUE_FILENAME
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_comment_queue(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": "1.0", "updated_at": None, "comments": []}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise InstagramSessionError(
            f"Invalid JSON in comment queue: {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise InstagramSessionError(
            f"Comment queue must be a JSON object: {path}"
        )

    data.setdefault("comments", [])
    return data


def _save_comment_queue(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def _build_approval_id(stable_id: str) -> str:
    return "appr-" + stable_id.replace(":", "-")


def prepare_replies(
    *,
    comment_queue_path: Path,
    approval_service: ApprovalService,
    reply_brain: ReplyBrain,
) -> PrepareRepliesResult:
    """
    Read pending comments, classify each one, route it through the
    safety gate, propose an Aiko-style reply where a reply is allowed,
    and append new records to the approval queue.

    Never sends anything to Instagram.
    """
    comment_queue = _load_comment_queue(comment_queue_path)

    existing_ids = approval_service.existing_stable_ids()
    used_reply_texts = approval_service.existing_reply_texts()

    comments_processed = 0
    replies_proposed = 0
    auto_eligible_count = 0
    human_review_count = 0
    skipped_count = 0
    duplicates_skipped = 0

    new_records: list[ApprovalRecord] = []

    for comment in comment_queue.get("comments", []):
        if comment.get("status") != "pending":
            continue

        comments_processed += 1
        stable_id = str(comment.get("stable_id"))

        if stable_id in existing_ids:
            duplicates_skipped += 1
            continue

        classification_result = reply_brain.classify(
            comment.get("comment_text")
        )
        classification = classification_result.classification
        safety_route = resolve_safety_route(classification)
        created_at = _now()

        if safety_route == "never_reply":
            record = ApprovalRecord(
                approval_id=_build_approval_id(stable_id),
                stable_id=stable_id,
                comment_id=comment.get("comment_id"),
                post_url=comment.get("post_url", ""),
                username=comment.get("username"),
                comment_text=comment.get("comment_text"),
                classification=classification,
                proposed_reply=None,
                safety_route=safety_route,
                status="skipped",
                created_at=created_at,
            )

            skipped_count += 1
            comment["status"] = "skipped"

        else:
            proposed_reply = reply_brain.generate_reply(
                classification,
                used_reply_texts=used_reply_texts,
            )
            used_reply_texts.add(proposed_reply)

            record = ApprovalRecord(
                approval_id=_build_approval_id(stable_id),
                stable_id=stable_id,
                comment_id=comment.get("comment_id"),
                post_url=comment.get("post_url", ""),
                username=comment.get("username"),
                comment_text=comment.get("comment_text"),
                classification=classification,
                proposed_reply=proposed_reply,
                safety_route=safety_route,
                status="pending_approval",
                created_at=created_at,
            )

            replies_proposed += 1

            if safety_route == "auto_eligible":
                auto_eligible_count += 1
            else:
                human_review_count += 1

            comment["status"] = "processed"

        new_records.append(record)
        existing_ids.add(stable_id)

    approval_service.add_records(new_records)
    _save_comment_queue(comment_queue_path, comment_queue)

    return PrepareRepliesResult(
        comments_processed=comments_processed,
        replies_proposed=replies_proposed,
        auto_eligible=auto_eligible_count,
        human_review_required=human_review_count,
        skipped=skipped_count,
        duplicates_skipped=duplicates_skipped,
        approval_queue_path=str(approval_service.queue_path),
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIKO Community Service (Phase 8C)"
    )

    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--prepare-replies",
        action="store_true",
        help="Classify pending comments and propose replies.",
    )

    action.add_argument(
        "--list-pending",
        action="store_true",
        help="List approval records awaiting a decision.",
    )

    action.add_argument(
        "--approve",
        metavar="APPROVAL_ID",
        default=None,
        help="Approve one proposed reply (updates local JSON only).",
    )

    action.add_argument(
        "--reject",
        metavar="APPROVAL_ID",
        default=None,
        help="Reject one proposed reply (updates local JSON only).",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate reply_rules.yaml config file.",
    )

    return parser.parse_args(argv)


def _print_prepare_result(result: PrepareRepliesResult) -> None:
    print()
    print("AIKO Community Service — Prepare Replies")
    print("------------------------------------------")
    print(f"comments processed:    {result.comments_processed}")
    print(f"replies proposed:      {result.replies_proposed}")
    print(f"auto eligible:         {result.auto_eligible}")
    print(f"human review required: {result.human_review_required}")
    print(f"skipped:               {result.skipped}")
    print(f"duplicates skipped:    {result.duplicates_skipped}")
    print(f"approval queue path:   {result.approval_queue_path}")
    print()


def _print_pending(records: list[dict[str, Any]]) -> None:
    print()
    print(f"AIKO Community Service — Pending Approvals ({len(records)})")
    print("-------------------------------------------------")

    for record in records:
        print(
            f"[{record.get('approval_id')}] "
            f"{record.get('classification')} / {record.get('safety_route')}"
        )
        print(f"  from:     @{record.get('username')}")
        print(f"  comment:  {record.get('comment_text')}")
        print(f"  proposed: {record.get('proposed_reply')}")
        print()

    print()


def main() -> None:
    arguments = parse_arguments()
    approval_service = build_approval_service()

    if arguments.prepare_replies:
        reply_brain = build_reply_brain(config_path=arguments.config)

        result = prepare_replies(
            comment_queue_path=_default_comment_queue_path(),
            approval_service=approval_service,
            reply_brain=reply_brain,
        )

        _print_prepare_result(result)
        return

    if arguments.list_pending:
        _print_pending(approval_service.list_pending())
        return

    if arguments.approve:
        try:
            record = approval_service.approve(arguments.approve)
        except InstagramSessionError as exc:
            print(f"[CommunityService] approve failed: {exc}")
            raise SystemExit(1)

        print(f"[CommunityService] approved {record['approval_id']}")
        return

    if arguments.reject:
        try:
            record = approval_service.reject(arguments.reject)
        except InstagramSessionError as exc:
            print(f"[CommunityService] reject failed: {exc}")
            raise SystemExit(1)

        print(f"[CommunityService] rejected {record['approval_id']}")
        return


if __name__ == "__main__":
    main()
