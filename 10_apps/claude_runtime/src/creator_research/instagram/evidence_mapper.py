"""Converts observed Instagram records into
`creator_intelligence.evidence.Evidence` -- reused directly, never
redefined. `Evidence` has exactly 5 real fields besides `tags`/
`evidence_id` (no metadata dict), so every function here follows the
same convention `creator_intelligence.intake`'s own `to_evidence_from_*()`
functions already established: rich provenance goes into
`source_description` (a descriptive sentence) and `tags` (free-form
classification strings); `content_excerpt` is always capped via the
shared `caption_excerpt_max_chars` policy, reused rather than
duplicated. Every function returns `(Evidence, list[str] warnings)`.
"""
from __future__ import annotations

from src.creator_intelligence.evidence import Evidence, EvidenceType
from src.creator_intelligence.models import RelationshipBasis

from .models import (
    CaptionObservation,
    CommentObservation,
    CreatorReplyObservation,
    DiscoveredItem,
    HighlightRecord,
    PostRecord,
    ProfileRecord,
    ReelRecord,
    RelationshipEvidenceBasis,
    RelationshipEvidenceRecord,
    VisualExampleRecord,
)

_RELATIONSHIP_BASIS_MAP = {
    RelationshipEvidenceBasis.CAPTION_DECLARED: RelationshipBasis.PRESENTED_NARRATIVE,
    RelationshipEvidenceBasis.TAGGED_ACCOUNT: RelationshipBasis.VISUALLY_DEPICTED,
    RelationshipEvidenceBasis.COLLABORATION_LABEL: RelationshipBasis.VISUALLY_DEPICTED,
    RelationshipEvidenceBasis.VISIBLE_PUBLIC_CONTEXT: RelationshipBasis.INFERRED,
}


def _excerpt_cap() -> int:
    try:
        from src.creator_intelligence.config import load_framework_config

        return load_framework_config().caption_excerpt_max_chars
    except Exception:  # pragma: no cover - defensive fallback only
        return 280


def _capped_excerpt(text: str, max_chars: int) -> tuple[str, list[str]]:
    if len(text) <= max_chars:
        return text, []
    return text[:max_chars], [f"content truncated to {max_chars} chars (was {len(text)})"]


def profile_to_evidence(profile: ProfileRecord) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(profile.bio or "", _excerpt_cap())
    evidence = Evidence(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=(
            f"instagram profile | username={profile.username} | followers={profile.follower_count} "
            f"| following={profile.following_count} | posts={profile.post_count} | verified={profile.verified} "
            f"| category={profile.category}"
        ),
        content_excerpt=excerpt,
        collected_by="instagram_research_connector",
        tags=["instagram", "profile", "persona", "reliability:high"],
    )
    return evidence, warnings


def discovered_item_to_evidence(item: DiscoveredItem) -> tuple[Evidence, list[str]]:
    evidence_type = EvidenceType.SCREENSHOT if item.thumbnail_reference else EvidenceType.OPERATOR_OBSERVATION
    evidence = Evidence(
        evidence_type=evidence_type,
        source_description=(
            f"instagram grid item | url={item.source_url} | type={item.content_type} | position={item.position}"
        ),
        content_excerpt="",
        collected_by="instagram_research_connector",
        tags=["instagram", "grid", item.content_type, "reliability:medium"],
    )
    return evidence, []


def post_to_evidence(post: PostRecord) -> tuple[Evidence, list[str]]:
    tags = ["instagram", "posts", "reliability:medium"]
    if post.is_carousel:
        tags.append("carousel")
    if post.location_label:
        tags.append("location_tagged")
    evidence = Evidence(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=(
            f"instagram post | url={post.post_url} | published_at={post.published_at} "
            f"| likes={post.like_count} | comments={post.comment_count} | carousel={post.is_carousel} "
            f"| location={post.location_label}"
        ),
        content_excerpt="",
        collected_by="instagram_research_connector",
        tags=tags,
    )
    return evidence, []


def caption_to_evidence(caption: CaptionObservation) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(caption.caption_text, _excerpt_cap())
    tags = ["instagram", "caption", "reliability:high"]
    tags.extend(f"hashtag:{tag}" for tag in caption.hashtags)
    tags.extend(f"mention:{mention}" for mention in caption.mentions)
    evidence = Evidence(
        evidence_type=EvidenceType.TEXT_EXCERPT,
        source_description=f"instagram caption | post={caption.post_reference} | published_at={caption.published_at}",
        content_excerpt=excerpt,
        collected_by="instagram_research_connector",
        tags=tags,
    )
    return evidence, warnings


def comment_to_evidence(comment: CommentObservation, *, redact_usernames: bool) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(comment.comment_text, _excerpt_cap())
    author_display = "[redacted]" if redact_usernames else (comment.author_username or "unknown")
    evidence = Evidence(
        evidence_type=EvidenceType.TEXT_EXCERPT,
        source_description=(
            f"instagram audience comment | post={comment.post_reference} | author={author_display} "
            f"| timestamp={comment.timestamp_text}"
        ),
        content_excerpt=excerpt,
        collected_by="instagram_research_connector",
        tags=["instagram", "comment", "audience", "reliability:medium"],
    )
    return evidence, warnings


def creator_reply_to_evidence(reply: CreatorReplyObservation) -> tuple[Evidence, list[str]]:
    combined = f"comment: {reply.parent_comment_text} | reply: {reply.creator_reply_text}"
    excerpt, warnings = _capped_excerpt(combined, _excerpt_cap())
    evidence = Evidence(
        evidence_type=EvidenceType.TEXT_EXCERPT,
        source_description=(
            f"instagram creator reply | post={reply.post_reference} | timestamp={reply.timestamp_text} "
            f"| thread_depth={reply.thread_depth}"
        ),
        content_excerpt=excerpt,
        collected_by="instagram_research_connector",
        tags=["instagram", "reply", "creator_reply", "reliability:high"],
    )
    return evidence, warnings


def reel_to_evidence(reel: ReelRecord) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(reel.caption or "", _excerpt_cap())
    evidence_type = EvidenceType.SCREENSHOT if reel.thumbnail_reference else EvidenceType.OPERATOR_OBSERVATION
    evidence = Evidence(
        evidence_type=evidence_type,
        source_description=(
            f"instagram reel | url={reel.reel_url} | published_at={reel.published_at} | views={reel.views} "
            f"| likes={reel.likes} | comments={reel.comments} | duration_seconds={reel.duration_seconds}"
        ),
        content_excerpt=excerpt,
        collected_by="instagram_research_connector",
        tags=["instagram", "reels", "reliability:medium"],
    )
    return evidence, warnings


def highlight_to_evidence(highlight: HighlightRecord) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(highlight.caption_text or "", _excerpt_cap())
    evidence_type = EvidenceType.SCREENSHOT if highlight.reference else EvidenceType.OPERATOR_OBSERVATION
    evidence = Evidence(
        evidence_type=evidence_type,
        source_description=(
            f"instagram highlight | title={highlight.highlight_title} | item_index={highlight.item_index} "
            f"| content_type={highlight.content_type}"
        ),
        content_excerpt=excerpt,
        collected_by="instagram_research_connector",
        tags=["instagram", "highlights", "storytelling", "reliability:medium"],
    )
    return evidence, warnings


def relationship_to_evidence(record: RelationshipEvidenceRecord) -> tuple[Evidence, list[str]]:
    excerpt, warnings = _capped_excerpt(record.description, _excerpt_cap())
    mapped_basis = _RELATIONSHIP_BASIS_MAP.get(record.basis, RelationshipBasis.UNKNOWN)
    evidence = Evidence(
        evidence_type=EvidenceType.OPERATOR_OBSERVATION,
        source_description=(
            f"instagram relationship evidence | basis={record.basis} | account={record.related_account} "
            f"| post={record.post_reference}"
        ),
        content_excerpt=excerpt,
        collected_by="instagram_research_connector",
        tags=[
            "instagram",
            "relationship",
            "human_authenticity",
            record.basis,
            f"relationship_basis:{mapped_basis}",
            "reliability:medium",
        ],
    )
    return evidence, warnings


def visual_example_to_evidence(example: VisualExampleRecord) -> tuple[Evidence, list[str]]:
    evidence_type = EvidenceType.SCREENSHOT if example.screenshot_reference else EvidenceType.OPERATOR_OBSERVATION
    evidence = Evidence(
        evidence_type=evidence_type,
        source_description=(
            f"instagram visual example | source={example.source_reference} | bucket={example.sampling_bucket}"
        ),
        content_excerpt="",
        collected_by="instagram_research_connector",
        tags=["instagram", "visual_realism", "photography", example.sampling_bucket, "reliability:low"],
    )
    return evidence, []
