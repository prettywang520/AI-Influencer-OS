"""Granular learning-target tag catalog + coverage reporting.

This is a reporting view over `Evidence.tags`, layered underneath
CreatorDNA's existing 13 trait fields -- it is not a new analyzer and
it gates/validates nothing, matching
creator_intelligence.intake_manifest's own
count_categories()/compute_completeness() precedent ("not quality, not
confidence"). Zero coverage of a topic is never an error; "unknown
remains unknown."
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.creator_intelligence.evidence import Evidence

# trait field name (on CreatorDNA) -> granular tag names catalogued
# underneath it. A tag may legitimately appear under more than one
# trait (e.g. "family" informs both human_authenticity and
# relationships).
LEARNING_TARGETS: dict[str, tuple[str, ...]] = {
    "visual_realism": (
        "camera_feeling", "lighting", "skin_texture", "film_grain",
        "ccd_feeling", "luxury_feeling", "environment", "color_palette",
        "editing_style", "pose", "composition", "emotion",
    ),
    "photography": ("camera_feeling", "lighting", "editing_style", "composition"),
    "captions": ("emoji", "japanese", "chinese", "english"),
    "replies": ("emoji", "japanese", "chinese", "english"),
    "human_authenticity": ("family", "friends", "childhood_photos", "pet", "home", "relationship"),
    "relationships": ("family", "friends", "relationship"),
    "posting": ("posting_rhythm", "seasonal_changes"),
    "branding": ("brand_collaboration", "hotel_preference", "fashion"),
    "storytelling": ("coffee", "travel", "daily_lifestyle"),
}


def all_target_tags() -> tuple[str, ...]:
    seen: list[str] = []
    for tags in LEARNING_TARGETS.values():
        for tag in tags:
            if tag not in seen:
                seen.append(tag)
    return tuple(seen)


@dataclass(slots=True)
class TraitCoverage:
    trait_name: str
    tag_counts: dict[str, int] = field(default_factory=dict)

    @property
    def covered_tags(self) -> tuple[str, ...]:
        return tuple(sorted(tag for tag, count in self.tag_counts.items() if count > 0))

    @property
    def missing_tags(self) -> tuple[str, ...]:
        return tuple(sorted(tag for tag, count in self.tag_counts.items() if count == 0))


def compute_learning_target_coverage(evidence: list[Evidence]) -> dict[str, TraitCoverage]:
    """Counts, per trait, how much evidence exists for each of that
    trait's granular learning-target tags -- the same tag-counting
    approach intake_manifest.py already uses, just re-scoped per
    trait."""
    coverage: dict[str, TraitCoverage] = {
        trait: TraitCoverage(trait_name=trait, tag_counts={tag: 0 for tag in tags})
        for trait, tags in LEARNING_TARGETS.items()
    }
    for item in evidence:
        item_tags = set(item.tags)
        for trait_coverage in coverage.values():
            for tag in trait_coverage.tag_counts:
                if tag in item_tags:
                    trait_coverage.tag_counts[tag] += 1
    return coverage


def missing_tags_by_trait(evidence: list[Evidence]) -> dict[str, tuple[str, ...]]:
    coverage = compute_learning_target_coverage(evidence)
    return {trait: trait_coverage.missing_tags for trait, trait_coverage in coverage.items()}
