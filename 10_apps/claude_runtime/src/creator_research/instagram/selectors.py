"""Layered selector candidate lists, one dataclass per UI area. Every
field is a `list[str]` tried in order by `navigator.py`'s
`query()`/`query_all()` -- never a single brittle CSS string. These
are untested placeholder defaults (matching the same convention and
caveat already established by `src/social/instagram_models.py`'s
`CommentSelectors`) and should be tuned against a live page before any
real run; nothing in this module depends on them being exactly right
for the unit test suite to pass, since tests drive `FakeBrowserAdapter`
directly with whatever selector strings they choose to register.

Every candidate here is a plain DOM selector a normal browser session
would render -- none of them name a private or internal-API endpoint
of any kind.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AccessStateSelectors:
    login_wall: list[str] = field(
        default_factory=lambda: [
            "input[type='password']",
            "input[name='username']",
            "button:has-text('Log in')",
        ]
    )
    private_account_marker: list[str] = field(
        default_factory=lambda: [
            "text=This Account is Private",
            "h2:has-text('This Account is Private')",
        ]
    )
    rate_limit_marker: list[str] = field(
        default_factory=lambda: [
            "text=Please wait a few minutes",
            "text=Try Again Later",
        ]
    )
    challenge_marker: list[str] = field(
        default_factory=lambda: [
            "text=Help us confirm",
            "text=suspicious activity",
            "iframe[title*='recaptcha' i]",
        ]
    )


@dataclass(slots=True)
class ProfileSelectors:
    username: list[str] = field(default_factory=lambda: ["header h2", "header h1"])
    display_name: list[str] = field(default_factory=lambda: ["header section span"])
    bio: list[str] = field(default_factory=lambda: ["header section div span", "header div.-vDIg span"])
    post_count: list[str] = field(default_factory=lambda: ["header li:nth-child(1) span"])
    follower_count: list[str] = field(default_factory=lambda: ["header li:nth-child(2) span"])
    following_count: list[str] = field(default_factory=lambda: ["header li:nth-child(3) span"])
    category: list[str] = field(default_factory=lambda: ["header div[data-testid='user-category']"])
    external_link: list[str] = field(default_factory=lambda: ["header a[href^='http']"])
    verified_badge: list[str] = field(default_factory=lambda: ["span[aria-label='Verified']"])
    profile_image: list[str] = field(default_factory=lambda: ["header img"])


@dataclass(slots=True)
class GridSelectors:
    grid_item: list[str] = field(default_factory=lambda: ["article a[href*='/p/']", "article a[href*='/reel/']"])
    grid_item_link: list[str] = field(default_factory=lambda: ["a[href*='/p/']", "a[href*='/reel/']"])
    reel_indicator: list[str] = field(default_factory=lambda: ["svg[aria-label='Reel']"])
    scroll_sentinel: list[str] = field(default_factory=lambda: ["article"])


@dataclass(slots=True)
class PostSelectors:
    caption: list[str] = field(default_factory=lambda: ["div[data-testid='post-caption']", "h1"])
    like_count: list[str] = field(default_factory=lambda: ["section span:has-text('likes')"])
    comment_count: list[str] = field(default_factory=lambda: ["a:has-text('comments')"])
    timestamp: list[str] = field(default_factory=lambda: ["time"])
    carousel_indicator: list[str] = field(default_factory=lambda: ["button[aria-label='Next']"])
    location_label: list[str] = field(default_factory=lambda: ["a[href*='/explore/locations/']"])
    tagged_accounts: list[str] = field(default_factory=lambda: ["a[href*='/'][data-testid='tagged-account']"])
    collaboration_label: list[str] = field(default_factory=lambda: ["text=Collaborators"])


@dataclass(slots=True)
class InstagramCommentSelectors:
    comment_container: list[str] = field(default_factory=lambda: ["ul.comments-list", "div[role='menu']"])
    comment_item: list[str] = field(default_factory=lambda: ["li.comment-item"])
    comment_username: list[str] = field(default_factory=lambda: ["a.comment-username"])
    comment_text: list[str] = field(default_factory=lambda: ["span.comment-text"])
    comment_timestamp: list[str] = field(default_factory=lambda: ["time"])
    reply_item: list[str] = field(default_factory=lambda: ["li.comment-reply"])
    load_more_button: list[str] = field(default_factory=lambda: ["button:has-text('Load more comments')"])


@dataclass(slots=True)
class ReelSelectors:
    reel_item: list[str] = field(default_factory=lambda: ["a[href*='/reel/']"])
    reel_caption: list[str] = field(default_factory=lambda: ["div[data-testid='reel-caption']"])
    reel_views: list[str] = field(default_factory=lambda: ["span:has-text('views')"])
    reel_likes: list[str] = field(default_factory=lambda: ["section span:has-text('likes')"])
    reel_comments: list[str] = field(default_factory=lambda: ["a:has-text('comments')"])
    reel_duration: list[str] = field(default_factory=lambda: ["span.reel-duration"])
    reel_text_overlay: list[str] = field(default_factory=lambda: ["div.reel-text-overlay"])


@dataclass(slots=True)
class HighlightSelectors:
    highlight_avatar: list[str] = field(default_factory=lambda: ["li canvas + button"])
    highlight_title: list[str] = field(default_factory=lambda: ["div.highlight-title"])
    highlight_item: list[str] = field(default_factory=lambda: ["div.highlight-item"])
    highlight_next_button: list[str] = field(default_factory=lambda: ["button[aria-label='Next']"])


@dataclass(slots=True)
class InstagramSelectors:
    access_state: AccessStateSelectors = field(default_factory=AccessStateSelectors)
    profile: ProfileSelectors = field(default_factory=ProfileSelectors)
    grid: GridSelectors = field(default_factory=GridSelectors)
    post: PostSelectors = field(default_factory=PostSelectors)
    comments: InstagramCommentSelectors = field(default_factory=InstagramCommentSelectors)
    reels: ReelSelectors = field(default_factory=ReelSelectors)
    highlights: HighlightSelectors = field(default_factory=HighlightSelectors)


def default_instagram_selectors() -> InstagramSelectors:
    return InstagramSelectors()
