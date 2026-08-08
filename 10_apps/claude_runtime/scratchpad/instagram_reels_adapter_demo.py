"""Safe demo for Phase 12D.1 -- Instagram Reels Learning Adapter.

Entirely synthetic: 12 fake Instagram Reels (talking, travel, cafe,
daily-life, brand, selfie, B-roll, fast-cut, slow-talking, plus 3 more
for variety) run through the full target flow:

Instagram Reel Evidence -> Adapter -> VideoEvidence -> Video
Intelligence -> VideoDNA -> Production DNA -> Knowledge Base

Shows sample completeness, VideoDNA ids, deterministic output across
independent runs, generalized patterns, and a de-identification proof.
No real creator, no network, no browser. Not part of the automated
test suite.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.video_intelligence.evidence import VideoEvidence, VideoEvidenceType
from src.video_intelligence.production_dna import load_engine_config
from src.video_intelligence.workflow import VideoIntelligenceWorkflow
from src.video_intelligence.instagram.models import InstagramReelEvidencePacket
from src.video_intelligence.instagram.workflow import learn_reels, load_instagram_reels_config


def _annotation(tags, excerpt="", timestamp=None):
    return VideoEvidence(
        video_id="", evidence_type=VideoEvidenceType.OPERATOR_OBSERVATION, source_description="synthetic observation",
        content_excerpt=excerpt, timestamp_seconds=timestamp, tags=tags,
    )


def _build_reels(creator_label: str) -> list[InstagramReelEvidencePacket]:
    reels = []

    def add(url_suffix, *, duration, views, likes, comments, published_at, caption, annotations):
        reels.append(
            InstagramReelEvidencePacket(
                reel_url=f"https://www.instagram.com/reel/{url_suffix}/", creator_label=creator_label,
                duration_seconds=duration, views=views, likes=likes, comments=comments,
                published_at=published_at, caption=caption, annotations=annotations,
            )
        )

    # 1. Talking reel -- direct to camera, question hook
    add(
        "talking_1", duration=18.0, views=42000, likes=3100, comments=210, published_at="2026-07-01T09:00:00Z",
        caption="answering the question everyone asks me",
        annotations=[
            _annotation(["hook", "question"], "have you ever wondered why...", 0.5),
            _annotation(["camera", "close_up"], "", 1.0),
            _annotation(["speech", "english", "warm"], "", 2.0),
            _annotation(["cta", "comment"], "let me know below", 16.0),
        ],
    )
    # 2. Travel reel
    add(
        "travel_1", duration=22.0, views=88000, likes=6200, comments=340, published_at="2026-07-03T09:00:00Z",
        caption="a weekend in the mountains",
        annotations=[_annotation(["camera", "wide"], "", 0.5), _annotation(["storytelling", "beginning"], "arriving at the trailhead")],
    )
    # 3. Cafe reel -- daily life
    add(
        "cafe_1", duration=12.0, views=15000, likes=900, comments=60, published_at="2026-07-05T09:00:00Z",
        caption="morning coffee ritual",
        annotations=[_annotation(["pacing", "medium"], ""), _annotation(["audio", "music"], "")],
    )
    # 4. Daily-life reel
    add(
        "daily_1", duration=15.0, views=21000, likes=1200, comments=80, published_at="2026-07-07T09:00:00Z",
        caption="a normal Tuesday",
        annotations=[_annotation(["gesture", "hand_movement"], "")],
    )
    # 5. Brand/collaboration reel
    add(
        "brand_1", duration=20.0, views=61000, likes=4200, comments=190, published_at="2026-07-09T09:00:00Z",
        caption="so excited to share this with you",
        annotations=[_annotation(["cta", "follow"], "", 18.0), _annotation(["emotion", "primary_emotion"], "excited")],
    )
    # 6. Selfie reel
    add(
        "selfie_1", duration=10.0, views=33000, likes=2600, comments=150, published_at="2026-07-11T09:00:00Z",
        caption="get ready with me",
        annotations=[_annotation(["camera", "selfie"], "")],
    )
    # 7. B-roll reel (non-talking)
    add(
        "broll_1", duration=25.0, views=19000, likes=800, comments=30, published_at="2026-07-13T09:00:00Z",
        caption=None, annotations=[_annotation(["editing", "zoom"], "")],
    )
    # 8. Fast-cut reel
    add(
        "fastcut_1", duration=9.0, views=52000, likes=4100, comments=220, published_at="2026-07-15T09:00:00Z",
        caption="pov: your morning in 9 seconds",
        annotations=[_annotation(["pacing", "fast"], ""), _annotation(["editing", "jump_cut"], "")],
    )
    # 9. Slow-talking reel
    add(
        "slowtalk_1", duration=45.0, views=8000, likes=300, comments=20, published_at="2026-07-17T09:00:00Z",
        caption="a longer story about my week",
        annotations=[_annotation(["speech", "slow"], ""), _annotation(["storytelling", "payoff"], "how it all worked out")],
    )
    # 10. Friend/social reel
    add(
        "friends_1", duration=14.0, views=27000, likes=1900, comments=95, published_at="2026-07-19T09:00:00Z",
        caption="with my best friend",
        annotations=[_annotation(["camera", "friend_shot"], "")],
    )
    # 11. Another talking reel (corroborates the question-hook pattern)
    add(
        "talking_2", duration=16.0, views=39000, likes=2900, comments=180, published_at="2026-07-21T09:00:00Z",
        caption="you asked, I'm answering",
        annotations=[_annotation(["hook", "question"], "so many of you asked...", 0.3), _annotation(["cta", "comment"], "", 14.0)],
    )
    # 12. Cinematic reel
    add(
        "cinematic_1", duration=30.0, views=71000, likes=5300, comments=260, published_at="2026-07-23T09:00:00Z",
        caption="golden hour",
        annotations=[_annotation(["camera", "tracking"], ""), _annotation(["subtitles", "bold_caption"], "")],
    )

    return reels


def run_demo(output_root: Path) -> None:
    video_config = load_engine_config()
    video_config.output_root = str(output_root)
    adapter_config = load_instagram_reels_config()

    packets = _build_reels(creator_label="reference_creator_demo")
    report, records = learn_reels(packets, config=adapter_config, video_config=video_config, sample=False)

    print("=== 1. Batch summary ===")
    print("  total_reels:", report.total_reels)
    print("  reels_analyzed:", report.reels_analyzed)
    print("  reels_skipped:", report.reels_skipped)
    print("  knowledge_patterns_added:", report.knowledge_patterns_added)

    print("\n=== 2. Evidence completeness (coverage, not quality) ===")
    for trait, score in report.evidence_completeness.items():
        print(f"  {trait}: {score:.0%}")

    print("\n=== 3. VideoDNA ids ===")
    for dna_id in report.video_dna_ids:
        print(" ", dna_id)

    print("\n=== 4. Generalized patterns in the Knowledge Base ===")
    workflow = VideoIntelligenceWorkflow(video_config)
    patterns = workflow.knowledge_base().list_patterns()
    for pattern in patterns:
        print(f"  [{pattern.category}] {pattern.description} (supported by {pattern.supporting_video_count} reel(s), confidence {pattern.confidence})")

    print("\n=== 5. De-identification proof ===")
    all_patterns_json = json.dumps([p.to_dict() for p in patterns])
    assert "reference_creator_demo" not in all_patterns_json, "creator_label leaked into a pattern!"
    assert "answering the question everyone asks me" not in all_patterns_json, "verbatim caption leaked into a pattern!"
    print("  confirmed: no creator_label, no verbatim caption/script in any ProductionPattern")

    return report.video_dna_ids


with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
    print("########## RUN 1 ##########")
    ids_a = run_demo(Path(tmp_a) / "vi_output")

    print("\n\n########## RUN 2 (independent output dir, same synthetic evidence) ##########")
    ids_b = run_demo(Path(tmp_b) / "vi_output")

    print("\n=== 6. Deterministic VideoDNA ids across independent runs ===")
    print("  identical:", sorted(ids_a) == sorted(ids_b))

print("\nDemo complete. Synthetic reels only. No network access. No real creator analyzed.")
