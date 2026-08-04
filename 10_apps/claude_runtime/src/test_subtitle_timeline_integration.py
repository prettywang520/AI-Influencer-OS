from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import subtitle_engine, timeline_engine
from .subtitle_timeline_integration import (
    IntegratedTimelineValidationError,
    IntegrationOutputExistsError,
    SourceValidationError,
    SubtitleDurationExceedsTimelineError,
    SubtitleLoadError,
    SubtitleStyleReferenceError,
    SubtitleTimelineIntegrationConfig,
    SubtitleTrackExistsError,
    SubtitleTrackMappingError,
    TimelineIDMismatchError,
    TimelineLoadError,
    UnsafeIntegrationOutputError,
    attach_subtitles_to_timeline,
    load_integration_config,
    map_subtitle_cue,
    map_subtitle_track,
    parse_arguments,
    remove_existing_subtitle_track,
    run_integration,
    validate_subtitle_timeline_compatibility,
)

MODULE_PATH = Path(__file__).resolve().parent / "subtitle_timeline_integration.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


def _build_timeline(
    *, num_clips: int = 3, clip_duration: float = 3.0, timeline_id: str = "tl1", order_offset: int = 0
) -> timeline_engine.Timeline:
    clips = []
    running = 0.0
    for index in range(1, num_clips + 1):
        clips.append(
            timeline_engine.VideoClip(
                clip_id=f"clip_video_{index:02d}",
                track_id="track_video",
                source_path=f"/scenes/scene_{index:02d}.mp4",
                start=running,
                end=running + clip_duration,
                duration_seconds=clip_duration,
                source_out=clip_duration,
                scene_number=index,
            )
        )
        running += clip_duration

    video_track = timeline_engine.TimelineTrack(
        track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=order_offset, clips=clips
    )
    return timeline_engine.Timeline(
        timeline_id=timeline_id,
        schema_version="1.0",
        production_date="2026-08-05",
        duration_seconds=running,
        tracks=[video_track],
    )


def _build_subtitle_document(
    *,
    cues: list[dict] | None = None,
    document_id: str = "doc1",
    language: str = "en",
    timeline_id: str | None = None,
    style_ids: list[str] | None = None,
) -> subtitle_engine.SubtitleDocument:
    if cues is None:
        cues = [
            {"cue_id": "cue_0001", "text": "hello there", "start": 0.0, "end": 2.0},
            {"cue_id": "cue_0002", "text": "general kenobi", "start": 2.0, "end": 4.0},
        ]

    default_config = subtitle_engine.load_subtitle_config()
    style_ids = style_ids or ["default"]
    styles = {style_id: default_config.styles[style_id] for style_id in style_ids}

    subtitle_cues = []
    for entry in cues:
        duration = entry["end"] - entry["start"]
        subtitle_cues.append(
            subtitle_engine.SubtitleCue(
                cue_id=entry["cue_id"],
                track_id="track_subtitles",
                start_seconds=entry["start"],
                end_seconds=entry["end"],
                duration_seconds=duration,
                text=entry["text"],
                style_id=entry.get("style_id", "default"),
                position=subtitle_engine.SubtitlePosition(x=540, y=1450),
                speaker=entry.get("speaker"),
                metadata=entry.get("metadata", {}),
            )
        )

    track = subtitle_engine.SubtitleTrack(
        track_id="track_subtitles", language=language, default_style_id="default", cues=subtitle_cues
    )
    duration = max((c.end_seconds for c in subtitle_cues), default=0.0)
    return subtitle_engine.SubtitleDocument(
        schema_version="1.0",
        document_id=document_id,
        language=language,
        duration_seconds=duration,
        tracks=[track],
        styles=styles,
        timeline_id=timeline_id,
    )


class IntegrationTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = load_integration_config()

    def _write_timeline(self, timeline: timeline_engine.Timeline, *, name: str = "timeline.json") -> Path:
        path = self.temp_dir / name
        timeline_engine.save_timeline(timeline, path)
        return path

    def _write_subtitles(
        self, document: subtitle_engine.SubtitleDocument, *, name: str = "subtitles.json"
    ) -> Path:
        path = self.temp_dir / name
        subtitle_engine.save_subtitle_document(document, path)
        return path

    def _hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------


class LoadingValidationTests(IntegrationTempTestCase):
    def test_valid_timeline_and_subtitles_integrate(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        output_path = self.temp_dir / "out.json"

        _integrated, result = run_integration(timeline_path, subtitle_path, self.config, output_path=output_path)
        self.assertTrue(result.validation_passed)
        self.assertTrue(output_path.is_file())

    def test_invalid_timeline_rejected(self) -> None:
        bad_timeline = timeline_engine.Timeline(timeline_id="bad", duration_seconds=0.0, tracks=[])
        timeline_path = self._write_timeline(bad_timeline)
        subtitle_path = self._write_subtitles(_build_subtitle_document())
        with self.assertRaises(SourceValidationError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

    def test_invalid_subtitle_document_rejected(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        bad_doc = subtitle_engine.SubtitleDocument(document_id="bad", duration_seconds=0.0, tracks=[])
        subtitle_path = self._write_subtitles(bad_doc)
        with self.assertRaises(SourceValidationError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

    def test_malformed_timeline_json_rejected(self) -> None:
        timeline_path = self.temp_dir / "timeline.json"
        timeline_path.write_text("{not valid json")
        subtitle_path = self._write_subtitles(_build_subtitle_document())
        with self.assertRaises(TimelineLoadError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

    def test_malformed_subtitle_json_rejected(self) -> None:
        timeline_path = self._write_timeline(_build_timeline())
        subtitle_path = self.temp_dir / "subtitles.json"
        subtitle_path.write_text("{not valid json")
        with self.assertRaises(SubtitleLoadError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

    def test_source_files_unchanged_byte_for_byte(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        before_t, before_s = self._hash(timeline_path), self._hash(subtitle_path)

        run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

        self.assertEqual(self._hash(timeline_path), before_t)
        self.assertEqual(self._hash(subtitle_path), before_s)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


class MappingTests(IntegrationTempTestCase):
    def test_exactly_one_subtitle_track_added(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        integrated, _mapping, _warnings = attach_subtitles_to_timeline(timeline, document, self.config)
        subtitle_tracks = [t for t in integrated.tracks if t.track_type == timeline_engine.TrackType.SUBTITLE]
        self.assertEqual(len(subtitle_tracks), 1)

    def test_track_type_is_subtitle(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        new_track = integrated.tracks[-1]
        self.assertEqual(new_track.track_type, timeline_engine.TrackType.SUBTITLE)

    def test_configured_track_id_used(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config, track_id="captions-fr")
        self.assertEqual(integrated.tracks[-1].track_id, "captions-fr")

    def test_track_order_deterministic(self) -> None:
        timeline = _build_timeline(order_offset=0)
        timeline.tracks.append(
            timeline_engine.TimelineTrack(track_id="track_audio", track_type=timeline_engine.TrackType.AUDIO, order=1)
        )
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(integrated.tracks[-1].order, 2)

    def test_every_cue_mapped_once(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, mapping, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(len(integrated.tracks[-1].clips), 2)
        self.assertEqual(mapping.cue_count, 2)

    def test_timing_preserved_exactly(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        clip = integrated.tracks[-1].clips[0]
        self.assertEqual(clip.start, 0.0)
        self.assertEqual(clip.end, 2.0)
        self.assertEqual(clip.duration_seconds, 2.0)
        self.assertEqual(clip.source_in, 0.0)
        self.assertEqual(clip.source_out, 2.0)

    def test_text_preserved_exactly(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(integrated.tracks[-1].clips[0].content, "hello there")

    def test_unicode_cjk_emoji_preserved(self) -> None:
        text = "京都の朝 🌸 早晨"
        timeline = _build_timeline()
        document = _build_subtitle_document(cues=[{"cue_id": "cue_0001", "text": text, "start": 0.0, "end": 2.0}])
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(integrated.tracks[-1].clips[0].content, text)

    def test_style_id_preserved(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(integrated.tracks[-1].clips[0].metadata["style_id"], "default")

    def test_full_style_metadata_preserved(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        style_snapshot = integrated.tracks[-1].clips[0].metadata["style"]
        for field_name in (
            "font_family", "font_size", "font_weight", "italic", "underline",
            "primary_color", "outline_color", "background_color",
            "outline_width", "shadow_depth", "margin_left", "margin_right",
            "margin_vertical", "alignment",
        ):
            self.assertIn(field_name, style_snapshot)

    def test_position_and_alignment_preserved(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        position = integrated.tracks[-1].clips[0].metadata["position"]
        self.assertEqual(position["x"], 540)
        self.assertEqual(position["y"], 1450)

    def test_speaker_and_line_break_metadata_preserved(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document(cues=[
            {"cue_id": "cue_0001", "text": "hi", "start": 0.0, "end": 2.0, "speaker": "narrator"},
        ])
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        clip_metadata = integrated.tracks[-1].clips[0].metadata
        self.assertEqual(clip_metadata["speaker"], "narrator")
        self.assertIn("line_break_mode", clip_metadata)

    def test_cue_metadata_preserved(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document(cues=[
            {"cue_id": "cue_0001", "text": "hi", "start": 0.0, "end": 2.0, "metadata": {"take": 3}},
        ])
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(integrated.tracks[-1].clips[0].metadata["cue_metadata"]["take"], 3)

    def test_no_fake_media_source_path(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        for clip in integrated.tracks[-1].clips:
            self.assertEqual(clip.source_path, "")

    def test_unresolved_style_raises_in_map_subtitle_cue(self) -> None:
        document = _build_subtitle_document()
        cue = document.tracks[0].cues[0]
        cue.style_id = "does_not_exist"
        with self.assertRaises(SubtitleStyleReferenceError):
            map_subtitle_cue(cue, document, document.tracks[0], self.config, track_id="subtitle-main", enabled=True)

    def test_missing_position_raises_when_required(self) -> None:
        document = _build_subtitle_document()
        document.tracks[0].cues[0].position = None
        config = SubtitleTimelineIntegrationConfig(require_safe_area_position=True)
        with self.assertRaises(SubtitleTrackMappingError):
            map_subtitle_cue(
                document.tracks[0].cues[0], document, document.tracks[0], config,
                track_id="subtitle-main", enabled=True,
            )


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


class CompatibilityTests(IntegrationTempTestCase):
    def test_duration_within_timeline_passes(self) -> None:
        timeline = _build_timeline(num_clips=3, clip_duration=3.0)  # 9s
        document = _build_subtitle_document(timeline_id=timeline.timeline_id)
        errors = validate_subtitle_timeline_compatibility(timeline, document, self.config)
        self.assertEqual(errors, [])

    def test_cue_beyond_timeline_fails(self) -> None:
        timeline = _build_timeline(num_clips=1, clip_duration=1.0)  # 1s
        document = _build_subtitle_document(timeline_id=timeline.timeline_id)
        errors = validate_subtitle_timeline_compatibility(timeline, document, self.config)
        self.assertTrue(errors)

    def test_timeline_id_match_passes(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        errors = validate_subtitle_timeline_compatibility(timeline, document, self.config)
        self.assertEqual(errors, [])

    def test_timeline_id_mismatch_fails(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl2")
        errors = validate_subtitle_timeline_compatibility(timeline, document, self.config)
        self.assertTrue(errors)

    def test_timeline_id_mismatch_raises_specific_error(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl2"))
        with self.assertRaises(TimelineIDMismatchError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

    def test_duration_exceeds_raises_specific_error(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(num_clips=1, clip_duration=1.0))
        subtitle_path = self._write_subtitles(_build_subtitle_document())
        with self.assertRaises(SubtitleDurationExceedsTimelineError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

    def test_mismatch_warning_policy_when_check_disabled(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl2")
        config = SubtitleTimelineIntegrationConfig(timeline_id_must_match=False)
        _integrated, _mapping, warnings = attach_subtitles_to_timeline(timeline, document, config)
        self.assertTrue(any(w.code == "timeline_id_mismatch" for w in warnings))

    def test_duplicate_cue_id_rejected_upstream(self) -> None:
        timeline_path = self._write_timeline(_build_timeline())
        document = _build_subtitle_document()
        document.tracks[0].cues[1].cue_id = document.tracks[0].cues[0].cue_id
        subtitle_path = self._write_subtitles(document)
        with self.assertRaises(SourceValidationError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")

    def test_overlapping_cue_policy_preserved(self) -> None:
        timeline_path = self._write_timeline(_build_timeline())
        document = _build_subtitle_document(cues=[
            {"cue_id": "cue_0001", "text": "a", "start": 0.0, "end": 3.0},
            {"cue_id": "cue_0002", "text": "b", "start": 1.0, "end": 3.0},
        ])
        subtitle_path = self._write_subtitles(document)
        with self.assertRaises(SourceValidationError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json")


# ---------------------------------------------------------------------------
# Existing tracks
# ---------------------------------------------------------------------------


class ExistingTrackTests(IntegrationTempTestCase):
    def test_existing_target_track_fails_by_default(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        with self.assertRaises(SubtitleTrackExistsError):
            attach_subtitles_to_timeline(integrated, document, self.config)

    def test_replace_existing_replaces_exact_target(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)

        new_document = _build_subtitle_document(
            cues=[{"cue_id": "cue_9999", "text": "replaced", "start": 0.0, "end": 2.0}], document_id="doc2"
        )
        integrated2, mapping2, warnings2 = attach_subtitles_to_timeline(
            integrated, new_document, self.config, replace_existing=True
        )
        subtitle_tracks = [t for t in integrated2.tracks if t.track_type == timeline_engine.TrackType.SUBTITLE]
        self.assertEqual(len(subtitle_tracks), 1)
        self.assertEqual(subtitle_tracks[0].clips[0].metadata["source_cue_id"], "cue_9999")
        self.assertTrue(any(w.code == "replaced_existing_track" for w in warnings2))

    def test_unrelated_subtitle_track_preserved(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(
            timeline, document, self.config, track_id="captions-fr"
        )
        document2 = _build_subtitle_document(document_id="doc2")
        integrated2, _m2, _w2 = attach_subtitles_to_timeline(
            integrated, document2, self.config, track_id="captions-en"
        )
        subtitle_track_ids = {
            t.track_id for t in integrated2.tracks if t.track_type == timeline_engine.TrackType.SUBTITLE
        }
        self.assertEqual(subtitle_track_ids, {"captions-fr", "captions-en"})

    def test_video_audio_tracks_unchanged(self) -> None:
        timeline = _build_timeline()
        original_video_track_dict = timeline_engine._track_to_dict(timeline.tracks[0])
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        integrated_video_track_dict = timeline_engine._track_to_dict(integrated.tracks[0])
        self.assertEqual(original_video_track_dict, integrated_video_track_dict)

    def test_multiple_subtitle_tracks_follow_config(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(
            timeline, document, self.config, track_id="captions-fr"
        )
        document2 = _build_subtitle_document(document_id="doc2")
        config = SubtitleTimelineIntegrationConfig(allow_multiple_subtitle_tracks=False)
        with self.assertRaises(SubtitleTrackExistsError):
            attach_subtitles_to_timeline(integrated, document2, config, track_id="captions-en")

    def test_disabled_track_created_correctly(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, mapping, _w = attach_subtitles_to_timeline(timeline, document, self.config, disable_track=True)
        new_track = integrated.tracks[-1]
        self.assertFalse(new_track.enabled)
        self.assertFalse(mapping.cues[0].enabled)

    def test_conflicting_non_subtitle_track_id_rejected(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        with self.assertRaises(SubtitleTrackMappingError):
            attach_subtitles_to_timeline(timeline, document, self.config, track_id="track_video")

    def test_fail_on_existing_target_track_false_auto_replaces(self) -> None:
        timeline = _build_timeline()
        document = _build_subtitle_document()
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)

        document2 = _build_subtitle_document(document_id="doc2")
        config = SubtitleTimelineIntegrationConfig(fail_on_existing_target_track=False)
        integrated2, _m2, warnings2 = attach_subtitles_to_timeline(integrated, document2, config)
        self.assertTrue(any(w.code == "replaced_existing_track" for w in warnings2))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityTests(IntegrationTempTestCase):
    def test_stable_integration_id(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        _t1, r1 = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out1.json"
        )
        _t2, r2 = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out2.json", force=True
        )
        self.assertEqual(r1.integration_id, r2.integration_id)

    def test_stable_integrated_timeline_id(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        _t1, r1 = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out1.json"
        )
        _t2, r2 = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out2.json", force=True
        )
        self.assertEqual(r1.integrated_timeline_id, r2.integrated_timeline_id)

    def test_created_timestamp_excluded_from_ids(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        t1, r1 = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out1.json"
        )
        t2, r2 = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out2.json", force=True
        )
        attached_at_1 = t1.metadata["subtitle_integrations"][0]["attached_at"]
        attached_at_2 = t2.metadata["subtitle_integrations"][0]["attached_at"]
        self.assertNotEqual(attached_at_1, attached_at_2)
        self.assertEqual(r1.integration_id, r2.integration_id)

    def test_base_timeline_id_preserved(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        integrated, result = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json"
        )
        self.assertEqual(integrated.timeline_id, "tl1")
        self.assertEqual(result.base_timeline_id, "tl1")

    def test_subtitle_integration_metadata_recorded(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        integrated, _result = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json"
        )
        entries = integrated.metadata["subtitle_integrations"]
        self.assertEqual(len(entries), 1)
        for field_name in ("integration_id", "subtitle_document_id", "track_id", "attached_at", "enabled", "cue_count", "language"):
            self.assertIn(field_name, entries[0])

    def test_repeated_identical_integration_does_not_duplicate_metadata(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        integrated1, mapping1, warnings1 = attach_subtitles_to_timeline(timeline, document, self.config)

        # Simulate calling run_integration() twice against the same
        # already-integrated result the way the orchestrator does.
        from src.subtitle_timeline_integration import _compute_integration_id

        integration_id = _compute_integration_id(
            base_timeline_id=timeline.timeline_id, subtitle_document_id=document.document_id,
            track_id="subtitle-main", replace_existing=False, disable_track=False,
            schema_version=self.config.schema_version,
        )
        integrated1.metadata.setdefault("subtitle_integrations", []).append(
            {"integration_id": integration_id, "subtitle_document_id": document.document_id,
             "track_id": "subtitle-main", "attached_at": "t1", "enabled": True, "cue_count": 2, "language": "en"}
        )
        # Re-apply the same entry as run_integration() would.
        entries = integrated1.metadata["subtitle_integrations"]
        entries[:] = [e for e in entries if e.get("integration_id") != integration_id]
        entries.append(
            {"integration_id": integration_id, "subtitle_document_id": document.document_id,
             "track_id": "subtitle-main", "attached_at": "t2", "enabled": True, "cue_count": 2, "language": "en"}
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["attached_at"], "t2")


# ---------------------------------------------------------------------------
# Post-integration validation
# ---------------------------------------------------------------------------


class PostIntegrationValidationTests(IntegrationTempTestCase):
    def test_integrated_timeline_passes(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        _integrated, result = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json"
        )
        self.assertTrue(result.validation_passed)

    def test_duplicate_track_id_fails(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        # Force a duplicate track_id onto the integrated result directly.
        duplicate = timeline_engine.TimelineTrack(
            track_id="subtitle-main", track_type=timeline_engine.TrackType.SUBTITLE, order=99
        )
        integrated.tracks.append(duplicate)
        result = timeline_engine.validate_timeline(integrated)
        self.assertFalse(result.passed)

    def test_duplicate_clip_id_fails(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        integrated.tracks[-1].clips[1].clip_id = integrated.tracks[-1].clips[0].clip_id
        result = timeline_engine.validate_timeline(integrated)
        self.assertFalse(result.passed)

    def test_invalid_track_reference_fails(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        integrated.tracks[-1].clips[0].track_id = "wrong-track"
        result = timeline_engine.validate_timeline(integrated)
        self.assertFalse(result.passed)

    def test_duration_unchanged(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        original_duration = timeline.duration_seconds
        document = _build_subtitle_document(timeline_id="tl1")
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(integrated.duration_seconds, original_duration)

    def test_subtitle_clip_timing_math_correct(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        result = timeline_engine.validate_timeline(integrated)
        self.assertTrue(result.passed)

    def test_metadata_json_serializable(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        integrated, _m, _w = attach_subtitles_to_timeline(timeline, document, self.config)
        for clip in integrated.tracks[-1].clips:
            json.dumps(clip.metadata)  # must not raise

    def test_forced_invalid_metadata_caught_by_adapter_check(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        integrated, _result = run_integration(
            timeline_path, subtitle_path, self.config, output_path=self.temp_dir / "out.json"
        )
        integrated.tracks[-1].clips[0].metadata["bad"] = object()
        from src.subtitle_timeline_integration import _validate_integrated_subtitle_tracks

        errors = _validate_integrated_subtitle_tracks(integrated, self.config)
        self.assertTrue(any("not JSON-serializable" in e for e in errors))


# ---------------------------------------------------------------------------
# JSON / CLI
# ---------------------------------------------------------------------------


class JsonCliTests(IntegrationTempTestCase):
    def test_atomic_output_no_tmp_left_behind(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        output_path = self.temp_dir / "out.json"
        run_integration(timeline_path, subtitle_path, self.config, output_path=output_path)
        self.assertTrue(output_path.is_file())
        self.assertFalse((self.temp_dir / "out.json.tmp").exists())

    def test_overwrite_refused_without_force(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        output_path = self.temp_dir / "out.json"
        run_integration(timeline_path, subtitle_path, self.config, output_path=output_path)
        with self.assertRaises(IntegrationOutputExistsError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=output_path)

    def test_force_overwrites(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        output_path = self.temp_dir / "out.json"
        run_integration(timeline_path, subtitle_path, self.config, output_path=output_path)
        _integrated, result = run_integration(
            timeline_path, subtitle_path, self.config, output_path=output_path, force=True
        )
        self.assertTrue(result.validation_passed)

    def test_validate_only_writes_nothing(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        errors = validate_subtitle_timeline_compatibility(
            timeline_engine.load_timeline(timeline_path),
            subtitle_engine.load_subtitle_document(subtitle_path),
            self.config,
        )
        self.assertEqual(errors, [])
        self.assertEqual(sorted(p.name for p in self.temp_dir.iterdir()), ["subtitles.json", "timeline.json"])

    def test_output_round_trips(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        output_path = self.temp_dir / "out.json"
        integrated, _result = run_integration(
            timeline_path, subtitle_path, self.config, output_path=output_path
        )
        reloaded = timeline_engine.load_timeline(output_path)
        self.assertEqual(
            timeline_engine.timeline_to_dict(reloaded), timeline_engine.timeline_to_dict(integrated)
        )

    def test_parent_directory_creation_works(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        output_path = self.temp_dir / "nested" / "dir" / "out.json"
        run_integration(timeline_path, subtitle_path, self.config, output_path=output_path)
        self.assertTrue(output_path.is_file())

    def test_output_same_as_timeline_source_rejected(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        with self.assertRaises(UnsafeIntegrationOutputError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=timeline_path)

    def test_output_same_as_subtitle_source_rejected(self) -> None:
        timeline_path = self._write_timeline(_build_timeline(timeline_id="tl1"))
        subtitle_path = self._write_subtitles(_build_subtitle_document(timeline_id="tl1"))
        with self.assertRaises(UnsafeIntegrationOutputError):
            run_integration(timeline_path, subtitle_path, self.config, output_path=subtitle_path)

    def test_cli_output_required_unless_validate_only(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--timeline", "t.json", "--subtitles", "s.json"])

    def test_cli_output_and_validate_only_conflict(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(
                ["--timeline", "t.json", "--subtitles", "s.json", "--output", "o.json", "--validate-only"]
            )

    def test_cli_optional_flags_accepted(self) -> None:
        args = parse_arguments(
            [
                "--timeline", "t.json", "--subtitles", "s.json", "--output", "o.json",
                "--track-id", "captions", "--replace-existing", "--disable-track", "--force", "--json",
            ]
        )
        self.assertEqual(args.track_id, "captions")
        self.assertTrue(args.replace_existing)
        self.assertTrue(args.disable_track)
        self.assertTrue(args.force)
        self.assertTrue(args.as_json)


# ---------------------------------------------------------------------------
# Deep-copy / no in-place mutation
# ---------------------------------------------------------------------------


class DeepCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_integration_config()

    def test_original_timeline_not_mutated(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        document = _build_subtitle_document(timeline_id="tl1")
        original_track_count = len(timeline.tracks)
        attach_subtitles_to_timeline(timeline, document, self.config)
        self.assertEqual(len(timeline.tracks), original_track_count)

    def test_remove_existing_subtitle_track_leaves_others_untouched(self) -> None:
        timeline = _build_timeline(timeline_id="tl1")
        timeline.tracks.append(
            timeline_engine.TimelineTrack(track_id="captions-fr", track_type=timeline_engine.TrackType.SUBTITLE, order=1)
        )
        timeline.tracks.append(
            timeline_engine.TimelineTrack(track_id="captions-en", track_type=timeline_engine.TrackType.SUBTITLE, order=2)
        )
        removed = remove_existing_subtitle_track(timeline, "captions-fr")
        self.assertIsNotNone(removed)
        self.assertEqual([t.track_id for t in timeline.tracks], ["track_video", "captions-en"])


# ---------------------------------------------------------------------------
# Structural safety
# ---------------------------------------------------------------------------


class StructuralSafetyTests(unittest.TestCase):
    def test_no_playwright_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import playwright"))
            self.assertFalse(stripped.startswith("from playwright"))

    def test_no_instagram_session_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.social"))
            self.assertFalse(stripped.startswith("from .social"))
        self.assertNotIn("InstagramSession(", MODULE_SOURCE)

    def test_no_publishing_or_social_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.publishing"))
            self.assertFalse(stripped.startswith("from .publishing"))

    def test_no_ffmpeg_ffprobe_or_subprocess(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))
        for forbidden in ("subprocess.run(", "subprocess.Popen(", "ffmpeg_binary", "ffprobe_binary"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_media_file_writes(self) -> None:
        for forbidden in (".mp4", ".mov", ".m4v", ".wav", ".mp3"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_speech_recognition_or_whisper(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip().lower()
            self.assertFalse(stripped.startswith("import whisper"))
            self.assertFalse(stripped.startswith("from whisper"))
            self.assertFalse(stripped.startswith("import speech_recognition"))
            self.assertFalse(stripped.startswith("from speech_recognition"))

    def test_no_font_download_or_network_code(self) -> None:
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download_font"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_rendering_implementation(self) -> None:
        for forbidden in ("cv2.", "PIL.", "moviepy", "render_frame", "draw_text"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_never_writes_source_timeline_or_subtitle_files(self) -> None:
        # The module must never call subtitle_engine.save_subtitle_document
        # at all, and must only call timeline_engine.save_timeline() on
        # the *integrated* result, never on the loaded source Timeline
        # object directly under a source-path variable name.
        self.assertNotIn("save_subtitle_document(", MODULE_SOURCE)


if __name__ == "__main__":
    unittest.main()
