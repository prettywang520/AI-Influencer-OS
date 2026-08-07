"""Section-by-section observation logic, adapter-driven. Delegates all
text interpretation to `parser.py` (pure functions) and all element
lookup to `navigator.InstagramBrowserAdapter` (via `selectors.py`'s
layered candidate lists) -- this module contains no Playwright import
and no text-parsing regex of its own.

Convention: an "item" element (a grid post, a comment, a reel, a
highlight item) is expected to expose its primary text via
`adapter.get_text(element)` and every secondary field (username,
timestamp, href, content_type, ...) via
`adapter.get_attribute(element, name)`. A real per-item DOM often
needs its own sub-tree lookup to fill those attributes -- that's an
implementation detail of a specific `InstagramBrowserAdapter`, not
something `observer.py` needs to know about.

Two observation functions are pure (no adapter): `observe_relationships()`
and `observe_visual_examples()` operate on already-collected records.
"""
from __future__ import annotations

from .dedupe import Deduplicator, identity_for
from .exceptions import (
    AccessLimitedError,
    AuthenticationRequiredError,
    ChallengeDetectedError,
    RateLimitedError,
)
from .models import (
    AccessEvent,
    AccessEventKind,
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
    VisualSamplingBucket,
)
from . import parser
from .selectors import AccessStateSelectors, InstagramSelectors

# Consecutive scroll rounds with no newly-discovered items before a
# scroll-and-collect loop gives up early. Not config-driven: the
# task's own suggested config only caps *total* rounds
# (max_scroll_rounds_per_section); this is a small, fixed safety valve
# on top of that, not an operational policy that needs live tuning.
_STAGNANT_ROUND_LIMIT = 2

_ACCESS_EXCEPTIONS = {
    AccessEventKind.ACCESS_LIMITED: AccessLimitedError,
    AccessEventKind.AUTHENTICATION_REQUIRED: AuthenticationRequiredError,
    AccessEventKind.RATE_LIMITED: RateLimitedError,
    AccessEventKind.CHALLENGE_DETECTED: ChallengeDetectedError,
}


def exception_for_access_event(event: AccessEvent):
    return _ACCESS_EXCEPTIONS[event.kind](event.detail or event.kind)


def detect_access_event(adapter, selectors: AccessStateSelectors, section: str) -> AccessEvent | None:
    if adapter.query(selectors.login_wall) is not None:
        return AccessEvent(kind=AccessEventKind.AUTHENTICATION_REQUIRED, section=section, detail="login wall detected")
    if adapter.query(selectors.private_account_marker) is not None:
        return AccessEvent(kind=AccessEventKind.ACCESS_LIMITED, section=section, detail="private account marker detected")
    if adapter.query(selectors.rate_limit_marker) is not None:
        return AccessEvent(kind=AccessEventKind.RATE_LIMITED, section=section, detail="rate limit marker detected")
    if adapter.query(selectors.challenge_marker) is not None:
        return AccessEvent(kind=AccessEventKind.CHALLENGE_DETECTED, section=section, detail="challenge marker detected")
    return None


def _text(adapter, candidates: list[str]) -> str | None:
    element = adapter.query(candidates)
    return adapter.get_text(element) if element is not None else None


def _attr(adapter, candidates: list[str], name: str) -> str | None:
    element = adapter.query(candidates)
    return adapter.get_attribute(element, name) if element is not None else None


def _present(adapter, candidates: list[str]) -> bool:
    return adapter.query(candidates) is not None


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def observe_profile(adapter, job, selectors: InstagramSelectors, config) -> ProfileRecord:
    adapter.open_profile(job.profile_url)
    adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
    event = detect_access_event(adapter, selectors.access_state, "profile")
    if event is not None:
        raise exception_for_access_event(event)

    raw = {
        "username": _text(adapter, selectors.profile.username) or job.username,
        "display_name": _text(adapter, selectors.profile.display_name),
        "bio": _text(adapter, selectors.profile.bio),
        "post_count": _text(adapter, selectors.profile.post_count),
        "follower_count": _text(adapter, selectors.profile.follower_count),
        "following_count": _text(adapter, selectors.profile.following_count),
        "category": _text(adapter, selectors.profile.category),
        "external_link": _attr(adapter, selectors.profile.external_link, "href"),
        "verified": "yes" if _present(adapter, selectors.profile.verified_badge) else None,
    }
    parsed = parser.parse_profile_text(raw)
    return ProfileRecord(
        username=parsed["username"] or job.username,
        display_name=parsed["display_name"],
        bio=parsed["bio"],
        profile_url=job.profile_url,
        post_count=parsed["post_count"],
        follower_count=parsed["follower_count"],
        following_count=parsed["following_count"],
        category=parsed["category"],
        external_link=parsed["external_link"],
        verified=parsed["verified"],
        profile_image_reference=_attr(adapter, selectors.profile.profile_image, "src"),
    )


# ---------------------------------------------------------------------------
# Grid / discovery (shared scroll-and-collect loop, reused by reels)
# ---------------------------------------------------------------------------


def _scroll_and_collect(
    adapter,
    selectors: InstagramSelectors,
    *,
    section: str,
    item_selector_candidates: list[str],
    href_attribute: str,
    max_items: int,
    max_rounds: int,
    dedupe: Deduplicator,
    build_item,
) -> tuple[list, list[str]]:
    collected: list = []
    warnings: list[str] = []
    stagnant_rounds = 0

    for _round_index in range(max_rounds):
        elements = adapter.query_all(item_selector_candidates)
        new_this_round = 0
        for element in elements:
            href = adapter.get_attribute(element, href_attribute)
            try:
                identity = identity_for(url=href, stable_id=adapter.get_attribute(element, "id"))
            except ValueError:
                continue
            if dedupe.is_duplicate(identity):
                dedupe.duplicates_skipped += 1
                continue
            dedupe.mark_seen(identity)
            collected.append(build_item(element, href, len(collected)))
            new_this_round += 1
            if len(collected) >= max_items:
                break
        if len(collected) >= max_items:
            break
        if new_this_round == 0:
            stagnant_rounds += 1
            if stagnant_rounds >= _STAGNANT_ROUND_LIMIT:
                break
        else:
            stagnant_rounds = 0

        event = detect_access_event(adapter, selectors.access_state, section)
        if event is not None:
            warnings.append(f"{event.kind}: {event.detail}")
            break

        adapter.scroll()

    return collected, warnings


def observe_grid(adapter, job, selectors: InstagramSelectors, config, dedupe: Deduplicator) -> tuple[list[DiscoveredItem], list[str]]:
    adapter.open_profile(job.profile_url)
    adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
    event = detect_access_event(adapter, selectors.access_state, "grid")
    if event is not None:
        raise exception_for_access_event(event)

    def build_item(element, href, position):
        content_type = adapter.get_attribute(element, "content_type") or "post"
        return DiscoveredItem(
            source_url=href or "",
            source_id=adapter.get_attribute(element, "id"),
            content_type="reel" if content_type == "reel" else "post",
            thumbnail_reference=adapter.get_attribute(element, "thumbnail"),
            position=position,
        )

    max_items = config.max_posts_per_job + config.max_reels_per_job
    return _scroll_and_collect(
        adapter,
        selectors,
        section="grid",
        item_selector_candidates=selectors.grid.grid_item,
        href_attribute="href",
        max_items=max_items,
        max_rounds=config.max_scroll_rounds_per_section,
        dedupe=dedupe,
        build_item=build_item,
    )


# ---------------------------------------------------------------------------
# Posts / captions
# ---------------------------------------------------------------------------


def observe_post(adapter, url: str, selectors: InstagramSelectors, config) -> PostRecord:
    adapter.open_profile(url)
    adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
    event = detect_access_event(adapter, selectors.access_state, "posts")
    if event is not None:
        raise exception_for_access_event(event)

    raw = {
        "caption": _text(adapter, selectors.post.caption),
        "like_count": _text(adapter, selectors.post.like_count),
        "comment_count": _text(adapter, selectors.post.comment_count),
        "timestamp": _attr(adapter, selectors.post.timestamp, "datetime") or _text(adapter, selectors.post.timestamp),
        "carousel": _present(adapter, selectors.post.carousel_indicator),
        "location": _text(adapter, selectors.post.location_label),
        "tagged_accounts": [adapter.get_text(e) for e in adapter.query_all(selectors.post.tagged_accounts)],
        "collaboration_labels": [adapter.get_text(e) for e in adapter.query_all(selectors.post.collaboration_label)],
    }
    parsed = parser.parse_post_metadata(raw)
    return PostRecord(
        post_url=url,
        published_at=parsed["timestamp"],
        caption_reference=parsed["caption"].text or None,
        is_carousel=parsed["carousel"],
        like_count=parsed["like_count"],
        comment_count=parsed["comment_count"],
        tagged_accounts=parsed["tagged_accounts"],
        collaboration_labels=parsed["collaboration_labels"],
        location_label=parsed["location"],
        content_type="post",
        visual_reference=adapter.get_screenshot_reference(f"post_{len(url)}_{hash(url) & 0xFFFF:x}"),
    )


def observe_captions(
    adapter, items: list[DiscoveredItem], selectors: InstagramSelectors, config
) -> tuple[list[CaptionObservation], list[str]]:
    captions: list[CaptionObservation] = []
    warnings: list[str] = []
    limit = min(len(items), config.max_posts_per_job)
    for item in items[:limit]:
        adapter.open_profile(item.source_url)
        adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
        event = detect_access_event(adapter, selectors.access_state, "captions")
        if event is not None:
            warnings.append(f"{event.kind}: {event.detail}")
            continue
        raw_caption = _text(adapter, selectors.post.caption)
        parsed = parser.parse_caption(raw_caption)
        timestamp = _attr(adapter, selectors.post.timestamp, "datetime") or _text(adapter, selectors.post.timestamp)
        captions.append(
            CaptionObservation(
                caption_text=parsed.text,
                post_reference=item.source_url,
                published_at=timestamp,
                hashtags=parsed.hashtags,
                mentions=parsed.mentions,
            )
        )
    return captions, warnings


# ---------------------------------------------------------------------------
# Comments / creator replies
# ---------------------------------------------------------------------------


def observe_comments(
    adapter, post_url: str, selectors: InstagramSelectors, config
) -> tuple[list[CommentObservation], list[str]]:
    adapter.open_profile(post_url)
    adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
    event = detect_access_event(adapter, selectors.access_state, "comments")
    if event is not None:
        raise exception_for_access_event(event)

    elements = adapter.query_all(selectors.comments.comment_item)[: config.max_comments_per_post]
    raw_items = [
        {
            "text": adapter.get_text(element),
            "username": adapter.get_attribute(element, "username"),
            "timestamp": adapter.get_attribute(element, "timestamp"),
            "parent_id": adapter.get_attribute(element, "parent_id"),
        }
        for element in elements
    ]
    cleaned = parser.parse_comment_thread(raw_items)
    observations = [
        CommentObservation(
            comment_text=item["text"],
            post_reference=post_url,
            author_username=item["username"],
            timestamp_text=item["timestamp"],
            thread_parent_id=item["parent_id"],
            is_creator=False,  # comments never self-classify creator identity -- see observe_creator_replies()
        )
        for item in cleaned
    ]
    return observations, []


def observe_creator_replies(
    adapter, post_url: str, selectors: InstagramSelectors, config, *, username: str
) -> tuple[list[CreatorReplyObservation], list[str]]:
    """Only classifies a reply as a creator reply when the
    visibly-attributed replying username case-insensitively equals
    `username` -- never inferred from avatar/tone/position."""
    adapter.open_profile(post_url)
    adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
    event = detect_access_event(adapter, selectors.access_state, "creator_replies")
    if event is not None:
        raise exception_for_access_event(event)

    elements = adapter.query_all(selectors.comments.reply_item)[: config.max_creator_replies_per_post]
    observations: list[CreatorReplyObservation] = []
    target = username.strip().lower()
    for element in elements:
        reply_username = adapter.get_attribute(element, "username")
        if not reply_username or reply_username.strip().lower() != target:
            continue
        reply_text = adapter.get_text(element) or ""
        parent_text = adapter.get_attribute(element, "parent_comment_text") or ""
        observations.append(
            CreatorReplyObservation(
                parent_comment_text=parent_text,
                creator_reply_text=reply_text,
                post_reference=post_url,
                emoji=parser.extract_emoji(reply_text),
                thread_depth=int(adapter.get_attribute(element, "thread_depth") or 1),
                timestamp_text=adapter.get_attribute(element, "timestamp"),
            )
        )
    return observations, []


# ---------------------------------------------------------------------------
# Reels
# ---------------------------------------------------------------------------


def observe_reels(
    adapter, job, selectors: InstagramSelectors, config, dedupe: Deduplicator
) -> tuple[list[ReelRecord], list[str]]:
    adapter.open_profile(job.profile_url)
    adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
    event = detect_access_event(adapter, selectors.access_state, "reels")
    if event is not None:
        raise exception_for_access_event(event)

    def build_item(element, href, _position):
        raw = {
            "caption": adapter.get_text(element),
            "views": adapter.get_attribute(element, "views"),
            "likes": adapter.get_attribute(element, "likes"),
            "comments": adapter.get_attribute(element, "comments"),
            "duration": adapter.get_attribute(element, "duration"),
            "text_overlays": [adapter.get_attribute(element, "text_overlay")]
            if adapter.get_attribute(element, "text_overlay")
            else [],
        }
        parsed = parser.parse_reel_metadata(raw)
        return ReelRecord(
            reel_url=href or "",
            caption=parsed["caption"].text or None,
            published_at=adapter.get_attribute(element, "timestamp"),
            views=parsed["views"],
            likes=parsed["likes"],
            comments=parsed["comments"],
            duration_seconds=parsed["duration"],
            text_overlays=parsed["text_overlays"],
            thumbnail_reference=adapter.get_screenshot_reference(f"reel_{href}"),
        )

    return _scroll_and_collect(
        adapter,
        selectors,
        section="reels",
        item_selector_candidates=selectors.reels.reel_item,
        href_attribute="href",
        max_items=config.max_reels_per_job,
        max_rounds=config.max_scroll_rounds_per_section,
        dedupe=dedupe,
        build_item=build_item,
    )


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------


def observe_highlights(adapter, job, selectors: InstagramSelectors, config) -> tuple[list[HighlightRecord], list[str]]:
    adapter.open_profile(job.profile_url)
    adapter.wait_for_page_ready(config.page_ready_timeout_seconds)
    event = detect_access_event(adapter, selectors.access_state, "highlights")
    if event is not None:
        return [], [f"{event.kind}: {event.detail}"]

    avatars = adapter.query_all(selectors.highlights.highlight_avatar)
    if not avatars:
        return [], ["no highlights discovered, or the Highlights UI was inaccessible"]

    records: list[HighlightRecord] = []
    warnings: list[str] = []
    for avatar in avatars:
        if len(records) >= config.max_highlight_items:
            break
        title = adapter.get_attribute(avatar, "title") or ""
        adapter.click(avatar)
        items = adapter.query_all(selectors.highlights.highlight_item)
        for index, item in enumerate(items):
            if len(records) >= config.max_highlight_items:
                break
            raw = {
                "title": title,
                "item_index": index,
                "content_type": adapter.get_attribute(item, "content_type"),
                "caption": adapter.get_text(item),
            }
            parsed = parser.parse_highlight_metadata(raw)
            records.append(
                HighlightRecord(
                    highlight_title=parsed["title"],
                    item_index=parsed["item_index"],
                    content_type=parsed["content_type"],
                    caption_text=parsed["caption"],
                    reference=adapter.get_screenshot_reference(f"highlight_{title}_{index}"),
                )
            )
    return records, warnings


# ---------------------------------------------------------------------------
# Relationships / visual examples -- pure functions over already-collected data
# ---------------------------------------------------------------------------


def observe_relationships(
    posts: list[PostRecord], captions: list[CaptionObservation]
) -> list[RelationshipEvidenceRecord]:
    """Captures only explicit public context: tagged accounts,
    collaboration labels, and caption mentions. Never infers a
    relationship *type* (romantic/family/friendship/employment) from
    appearance -- every record's `description` states only what was
    observed (a tag, a label, a mention), never a conclusion about it."""
    records: list[RelationshipEvidenceRecord] = []
    for post in posts:
        for account in post.tagged_accounts:
            records.append(
                RelationshipEvidenceRecord(
                    description=f"tagged account @{account} in post",
                    basis=RelationshipEvidenceBasis.TAGGED_ACCOUNT,
                    related_account=account,
                    post_reference=post.post_url,
                )
            )
        for label in post.collaboration_labels:
            records.append(
                RelationshipEvidenceRecord(
                    description=f"collaboration label observed: {label}",
                    basis=RelationshipEvidenceBasis.COLLABORATION_LABEL,
                    post_reference=post.post_url,
                )
            )
    for caption in captions:
        for mention in caption.mentions:
            records.append(
                RelationshipEvidenceRecord(
                    description=f"mentioned in caption: @{mention}",
                    basis=RelationshipEvidenceBasis.CAPTION_DECLARED,
                    related_account=mention,
                    post_reference=caption.post_reference,
                )
            )
    return records


def observe_visual_examples(items: list[DiscoveredItem]) -> list[VisualExampleRecord]:
    """Deterministic bucket assignment (position index modulo the
    fixed bucket list) -- selection only. No visual-realism judgement,
    no face recognition, no person identification is performed here or
    anywhere in this package."""
    buckets = VisualSamplingBucket.ALL
    records = []
    for index, item in enumerate(items):
        bucket = buckets[index % len(buckets)]
        records.append(
            VisualExampleRecord(
                source_reference=item.source_url,
                sampling_bucket=bucket,
                screenshot_reference=item.thumbnail_reference,
            )
        )
    return records
