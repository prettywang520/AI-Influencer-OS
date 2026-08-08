"""Adapter-native records: InstagramReelEvidencePacket (per-Reel
input schema), InstagramReelLearningRecord (per-Reel output record,
task's own §18 shape), and BatchLearningReport (task's own §25 shape).

Naming note: the task's own COMPLETENESS section spells one section
"lip_sync" (with underscore); video_intelligence.production_dna.TRAIT_FIELDS
spells the real trait name "lipsync" (no underscore, matching
video_intelligence/lipsync.py's own TRAIT_NAME). This module uses the
real trait name everywhere so completeness/coverage keys line up
exactly with VideoDNA's own field names -- not a second vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.video_intelligence.audio import TAGS as AUDIO_TAGS
from src.video_intelligence.camera import TAGS as CAMERA_TAGS
from src.video_intelligence.cta import TAGS as CTA_TAGS
from src.video_intelligence.editing import TAGS as EDITING_TAGS
from src.video_intelligence.emotion import TAGS as EMOTION_TAGS
from src.video_intelligence.evidence import VideoEvidence, VideoPlatform, video_id_for
from src.video_intelligence.framing import TAGS as FRAMING_TAGS
from src.video_intelligence.gesture import TAGS as GESTURE_TAGS
from src.video_intelligence.hook import TAGS as HOOK_TAGS
from src.video_intelligence.lipsync import TAGS as LIPSYNC_TAGS
from src.video_intelligence.pacing import TAGS as PACING_TAGS
from src.video_intelligence.production_dna import TRAIT_FIELDS, VideoDNA
from src.video_intelligence.speech import TAGS as SPEECH_TAGS
from src.video_intelligence.storytelling import TAGS as STORYTELLING_TAGS
from src.video_intelligence.subtitles import TAGS as SUBTITLES_TAGS

# trait_name -> that analyzer's own scoping TAGS tuple, reused directly
# (never redefined) so completeness reporting can never drift out of
# sync with what the analyzers themselves actually scope by.
TRAIT_TAGS: dict[str, tuple[str, ...]] = {
    "hook": HOOK_TAGS,
    "pacing": PACING_TAGS,
    "speech": SPEECH_TAGS,
    "lipsync": LIPSYNC_TAGS,
    "gesture": GESTURE_TAGS,
    "camera": CAMERA_TAGS,
    "framing": FRAMING_TAGS,
    "editing": EDITING_TAGS,
    "subtitles": SUBTITLES_TAGS,
    "audio": AUDIO_TAGS,
    "storytelling": STORYTELLING_TAGS,
    "cta": CTA_TAGS,
    "emotion": EMOTION_TAGS,
}

assert tuple(TRAIT_TAGS.keys()) == TRAIT_FIELDS  # stays in lockstep with VideoDNA's own fields


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_completeness(evidence: list[VideoEvidence]) -> dict[str, float]:
    """Binary evidence-coverage score per trait domain (1.0 = at
    least one evidence item carries that domain's own tag, 0.0 =
    none). Not quality, not performance -- coverage only, computable
    before any analysis runs."""
    tag_set = {tag for item in evidence for tag in item.tags}
    return {
        trait_name: 1.0 if tag_set.intersection(trait_tags) else 0.0
        for trait_name, trait_tags in TRAIT_TAGS.items()
    }


@dataclass(slots=True)
class InstagramReelsConfig:
    """Loaded from config/video_intelligence/instagram_reels.yaml (see
    workflow.load_instagram_reels_config()). Lives here, not in
    workflow.py, so sampler.py/models.py can depend on the type
    without a circular import -- mirrors
    video_intelligence.models.VideoIntelligenceConfig's own placement
    reasoning exactly. `priorities` is keyed by the real trait_name
    (e.g. "lipsync", not the task's "lip_sync" spelling) so it lines
    up with TRAIT_TAGS/VideoDNA's own field names."""

    version: str
    schema_version: str
    adapter_version: str
    platform: str
    minimum_reels_for_dna: int
    recommended_reels: int
    strong_sample: int
    deterministic: bool
    priorities: dict[str, float] = field(default_factory=dict)
    deidentify_creator: bool = True
    preserve_verbatim_scripts: bool = False
    minimum_pattern_corroboration: int = 2

    def priority_for(self, trait_name: str) -> float:
        return float(self.priorities.get(trait_name, 1.0))


@dataclass(slots=True)
class InstagramReelEvidencePacket:
    """One Reel's normalized-but-not-yet-VideoEvidence input. Built
    either from real creator_research Evidence
    (mapper.reel_evidence_from_creator_research()) or by hand
    (tests/demo/a future operator or Phase 12D.2 annotation tool)."""

    reel_url: str
    published_at: str | None = None
    duration_seconds: float | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    caption: str | None = None
    creator_label: str = ""
    source_evidence_ids: list[str] = field(default_factory=list)
    annotations: list[VideoEvidence] = field(default_factory=list)
    collected_at: str = field(default_factory=_now_iso)
    metadata: dict = field(default_factory=dict)
    reel_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.reel_id = video_id_for(VideoPlatform.INSTAGRAM_REELS, self.creator_label, self.reel_url)

    def visible_engagement_ratio(self) -> float | None:
        """Optional derived metric (task §17) -- only computed when
        the denominator (views) exists and is positive, AND both
        numerator terms (likes, comments) are actually known. A
        missing likes/comments value is never treated as 0 here --
        that would silently understate true engagement, exactly the
        kind of invented metric §17 forbids."""
        if not self.views or self.views <= 0:
            return None
        if self.likes is None or self.comments is None:
            return None
        return (self.likes + self.comments) / self.views

    def comments_per_1000_views(self) -> float | None:
        if not self.views or self.views <= 0 or self.comments is None:
            return None
        return self.comments / self.views * 1000

    def likes_per_1000_views(self) -> float | None:
        if not self.views or self.views <= 0 or self.likes is None:
            return None
        return self.likes / self.views * 1000


@dataclass(slots=True)
class InstagramReelLearningRecord:
    """One Reel's outcome after learn_reel() -- task's own §18 shape."""

    reel_id: str
    source_url: str
    source_evidence_ids: list[str] = field(default_factory=list)
    collected_at: str = field(default_factory=_now_iso)
    published_at: str | None = None
    video_evidence: list[VideoEvidence] = field(default_factory=list)
    dna: VideoDNA | None = None
    completeness: dict[str, float] = field(default_factory=dict)
    patterns_touched: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class BatchLearningReport:
    """Task's own §25 batch report shape. `video_dna_ids` is the set
    of dna_ids produced this batch -- there is no cross-reel "Production
    DNA ID" hash of its own; the knowledge base is where cross-reel
    generalization already lives (video_intelligence.knowledge_base)."""

    total_reels: int
    reels_analyzed: int
    reels_skipped: int
    evidence_completeness: dict[str, float] = field(default_factory=dict)
    analyzer_coverage: dict[str, int] = field(default_factory=dict)
    video_dna_ids: list[str] = field(default_factory=list)
    knowledge_patterns_added: int = 0
    warnings: list[str] = field(default_factory=list)
    missing_evidence: dict[str, int] = field(default_factory=dict)
    generated_at: str = field(default_factory=_now_iso)
