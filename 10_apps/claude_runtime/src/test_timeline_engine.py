from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from .timeline_engine import (
    AudioClip,
    ClipType,
    DuplicateSceneNumberError,
    MissingSceneNumberError,
    OverlayClip,
    SceneDiscoveryError,
    SceneDurationError,
    SceneFile,
    Timeline,
    TimelineConfig,
    TimelineDurationLimitError,
    TimelineFileNotFoundError,
    TimelineInputError,
    TimelineJSONError,
    TimelineOutputExistsError,
    TimelineTrack,
    TimelineValidationResult,
    TrackType,
    UnsafeTimelineOutputError,
    VideoClip,
    build_timeline_from_scenes,
    discover_scenes,
    load_timeline,
    load_timeline_config,
    main,
    parse_arguments,
    resolve_scene_duration,
    save_timeline,
    timeline_from_dict,
    timeline_to_dict,
    validate_timeline,
)

PRODUCTION_DATE = "2026-08-04"
MODULE_PATH = Path(__file__).resolve().parent / "timeline_engine.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


class TimelineTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.scenes_dir = self.temp_dir / "output" / PRODUCTION_DATE / "videos" / "reel_scenes"
        self.scenes_dir.mkdir(parents=True)
        self.config = TimelineConfig()

    def _write_scene(self, name: str, *, size: int = 16) -> Path:
        path = self.scenes_dir / name
        path.write_bytes(b"x" * size)
        return path

    def _write_three_scenes(self) -> list[Path]:
        return [self._write_scene(f"scene_{index:02d}.mp4") for index in (1, 2, 3)]

    def _write_sidecar(self, scene_path: Path, duration_seconds) -> None:
        scene_path.with_suffix(".json").write_text(json.dumps({"duration_seconds": duration_seconds}))

    def _write_manifest(self, entries: list[dict]) -> Path:
        manifest_path = self.temp_dir / "output" / PRODUCTION_DATE / "production_manifest.json"
        manifest_path.write_text(json.dumps({"reel_tasks": entries}))
        return manifest_path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_loads_from_repo(self) -> None:
        config = load_timeline_config()
        self.assertEqual(config.schema_version, "1.0")
        self.assertIn("sidecar", config.duration_priority)
        self.assertIn(".mp4", config.supported_extensions)

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(Exception):
            load_timeline_config("/nonexistent/timeline.yaml")


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------


class SceneDiscoveryTests(TimelineTempTestCase):
    def test_plain_numbered_names(self) -> None:
        for index in (1, 2, 3):
            self._write_scene(f"{index:02d}.mp4")
        scenes = discover_scenes(self.scenes_dir, self.config)
        self.assertEqual([s.index for s in scenes], [1, 2, 3])

    def test_scene_prefixed_names(self) -> None:
        self._write_three_scenes()
        scenes = discover_scenes(self.scenes_dir, self.config)
        self.assertEqual([s.index for s in scenes], [1, 2, 3])

    def test_mixed_naming_conventions(self) -> None:
        self._write_scene("scene_01.mp4")
        self._write_scene("02.mp4")
        self._write_scene("scene_03.mp4")
        scenes = discover_scenes(self.scenes_dir, self.config)
        self.assertEqual([s.index for s in scenes], [1, 2, 3])

    def test_duplicate_scene_number_across_naming_conventions(self) -> None:
        self._write_scene("01.mp4")
        self._write_scene("scene_01.mp4")
        with self.assertRaises(DuplicateSceneNumberError):
            discover_scenes(self.scenes_dir, self.config)

    def test_missing_scene_number_gap_raises_by_default(self) -> None:
        self._write_scene("scene_01.mp4")
        self._write_scene("scene_03.mp4")
        with self.assertRaises(MissingSceneNumberError):
            discover_scenes(self.scenes_dir, self.config)

    def test_keep_gaps_allows_missing_number(self) -> None:
        self._write_scene("scene_01.mp4")
        self._write_scene("scene_03.mp4")
        scenes = discover_scenes(self.scenes_dir, self.config, keep_gaps=True)
        self.assertEqual([s.index for s in scenes], [1, 3])

    def test_zero_byte_scene_rejected(self) -> None:
        self._write_scene("scene_01.mp4", size=0)
        with self.assertRaises(SceneDiscoveryError):
            discover_scenes(self.scenes_dir, self.config)

    def test_unsupported_extension_skipped_by_default(self) -> None:
        self._write_three_scenes()
        (self.scenes_dir / "scene_04.txt").write_text("not a scene")
        scenes = discover_scenes(self.scenes_dir, self.config)
        self.assertEqual([s.index for s in scenes], [1, 2, 3])

    def test_unsupported_extension_rejected_when_configured(self) -> None:
        self._write_three_scenes()
        (self.scenes_dir / "scene_04.txt").write_text("not a scene")
        config = TimelineConfig(reject_unsupported_extensions=True)
        with self.assertRaises(SceneDiscoveryError):
            discover_scenes(self.scenes_dir, config)

    def test_deterministic_sorted_order(self) -> None:
        self._write_scene("scene_03.mp4")
        self._write_scene("scene_01.mp4")
        self._write_scene("scene_02.mp4")
        scenes = discover_scenes(self.scenes_dir, self.config)
        self.assertEqual([s.index for s in scenes], [1, 2, 3])

    def test_unicode_and_space_filenames_ignored_as_non_scene(self) -> None:
        self._write_three_scenes()
        (self.scenes_dir / "bloopers reel — outtakes.mp4").write_bytes(b"x")
        scenes = discover_scenes(self.scenes_dir, self.config)
        self.assertEqual([s.index for s in scenes], [1, 2, 3])

    def test_resolved_duplicate_path_rejected(self) -> None:
        real = self._write_scene("scene_01.mp4")
        symlink_path = self.scenes_dir / "scene_02.mp4"
        os.symlink(real, symlink_path)
        with self.assertRaises(SceneDiscoveryError):
            discover_scenes(self.scenes_dir, self.config)

    def test_no_scenes_found_raises(self) -> None:
        with self.assertRaises(SceneDiscoveryError):
            discover_scenes(self.scenes_dir, self.config)

    def test_missing_directory_raises_input_error(self) -> None:
        with self.assertRaises(TimelineInputError):
            discover_scenes(self.temp_dir / "does_not_exist", self.config)


# ---------------------------------------------------------------------------
# Duration resolution
# ---------------------------------------------------------------------------


class DurationResolutionTests(TimelineTempTestCase):
    def test_sidecar_duration_used(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        self._write_sidecar(scene_path, 4.2)
        duration, source = resolve_scene_duration(SceneFile(1, scene_path), self.config, None)
        self.assertEqual(duration, 4.2)
        self.assertEqual(source, "sidecar")

    def test_invalid_sidecar_raises_even_though_manifest_available(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        scene_path.with_suffix(".json").write_text(json.dumps({"duration_seconds": -1}))
        manifest = {"reel_tasks": [{"scene_number": 1, "duration_seconds": 3.0}]}
        with self.assertRaises(SceneDurationError):
            resolve_scene_duration(SceneFile(1, scene_path), self.config, manifest)

    def test_malformed_sidecar_json_raises(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        scene_path.with_suffix(".json").write_text("{not json")
        with self.assertRaises(SceneDurationError):
            resolve_scene_duration(SceneFile(1, scene_path), self.config, None)

    def test_manifest_duration_used_when_no_sidecar(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        manifest = {"reel_tasks": [{"scene_number": 1, "duration_seconds": 3.3}]}
        duration, source = resolve_scene_duration(SceneFile(1, scene_path), self.config, manifest)
        self.assertEqual(duration, 3.3)
        self.assertEqual(source, "manifest")

    def test_bad_manifest_entry_falls_through_to_default(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        manifest = {"reel_tasks": [{"scene_number": 1, "duration_seconds": -5}]}
        duration, source = resolve_scene_duration(SceneFile(1, scene_path), self.config, manifest)
        self.assertEqual(source, "default")
        self.assertEqual(duration, self.config.default_scene_duration_seconds)

    def test_default_duration_used_and_flagged(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        duration, source = resolve_scene_duration(SceneFile(1, scene_path), self.config, None)
        self.assertEqual(source, "default")
        self.assertEqual(duration, self.config.default_scene_duration_seconds)

    def test_missing_duration_raises_when_default_disabled(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        config = TimelineConfig(allow_default_duration=False)
        with self.assertRaises(SceneDurationError):
            resolve_scene_duration(SceneFile(1, scene_path), config, None)

    def test_zero_sidecar_duration_rejected(self) -> None:
        scene_path = self._write_scene("scene_01.mp4")
        self._write_sidecar(scene_path, 0)
        with self.assertRaises(SceneDurationError):
            resolve_scene_duration(SceneFile(1, scene_path), self.config, None)


# ---------------------------------------------------------------------------
# Timeline construction
# ---------------------------------------------------------------------------


class ConstructionTests(TimelineTempTestCase):
    def test_single_video_track_created(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(self.scenes_dir, self.config, production_date=PRODUCTION_DATE)
        video_tracks = [t for t in timeline.tracks if t.track_type == TrackType.VIDEO]
        self.assertEqual(len(video_tracks), 1)
        self.assertEqual(len(video_tracks[0].clips), 3)

    def test_sequential_start_end_math(self) -> None:
        scenes = self._write_three_scenes()
        for scene, duration in zip(scenes, (2.0, 3.0, 1.5)):
            self._write_sidecar(scene, duration)
        timeline = build_timeline_from_scenes(self.scenes_dir, self.config)
        clips = timeline.tracks[0].clips
        self.assertEqual((clips[0].start, clips[0].end), (0.0, 2.0))
        self.assertEqual((clips[1].start, clips[1].end), (2.0, 5.0))
        self.assertEqual((clips[2].start, clips[2].end), (5.0, 6.5))
        self.assertAlmostEqual(timeline.duration_seconds, 6.5)

    def test_scene_order_preserved(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(self.scenes_dir, self.config)
        scene_numbers = [c.scene_number for c in timeline.tracks[0].clips]
        self.assertEqual(scene_numbers, [1, 2, 3])

    def test_no_music_track_without_music_path(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(self.scenes_dir, self.config)
        self.assertEqual(len(timeline.tracks), 1)

    def test_music_track_created_when_requested(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(
            self.scenes_dir, self.config, music_path="/music/bg.mp3"
        )
        audio_tracks = [t for t in timeline.tracks if t.track_type == TrackType.AUDIO]
        self.assertEqual(len(audio_tracks), 1)
        clip = audio_tracks[0].clips[0]
        self.assertEqual(clip.source_path, "/music/bg.mp3")
        self.assertEqual((clip.start, clip.end), (0.0, timeline.duration_seconds))
        self.assertEqual(clip.loop, self.config.music_loop)
        self.assertEqual(clip.volume, self.config.music_volume)

    def test_music_overrides_applied(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(
            self.scenes_dir,
            self.config,
            music_path="/music/bg.mp3",
            music_overrides={"volume": 0.5, "fade_in_seconds": 1.0},
        )
        clip = timeline.tracks[1].clips[0]
        self.assertEqual(clip.volume, 0.5)
        self.assertEqual(clip.fade_in_seconds, 1.0)

    def test_no_overlay_or_subtitle_clips_created(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(
            self.scenes_dir, self.config, music_path="/music/bg.mp3"
        )
        for track in timeline.tracks:
            for clip in track.clips:
                self.assertNotIsInstance(clip, OverlayClip)
                self.assertNotIn(track.track_type, (TrackType.SUBTITLE,))

    def test_transitions_always_none_and_playback_rate_always_one(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(self.scenes_dir, self.config)
        for clip in timeline.tracks[0].clips:
            self.assertEqual(clip.transition_in, "none")
            self.assertEqual(clip.transition_out, "none")
            self.assertEqual(clip.playback_rate, 1.0)

    def test_default_duration_records_warning(self) -> None:
        self._write_three_scenes()
        timeline = build_timeline_from_scenes(self.scenes_dir, self.config)
        self.assertTrue(timeline.metadata["build_warnings"])


# ---------------------------------------------------------------------------
# Duration limit handling
# ---------------------------------------------------------------------------


class DurationLimitTests(TimelineTempTestCase):
    def _build_three_two_second_scenes(self, config=None):
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        return build_timeline_from_scenes(self.scenes_dir, config or self.config, duration_limit_seconds=10.0)

    def test_under_limit_passes_unchanged(self) -> None:
        timeline = self._build_three_two_second_scenes()
        self.assertAlmostEqual(timeline.duration_seconds, 6.0)

    def test_over_limit_fails_by_default(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        with self.assertRaises(TimelineDurationLimitError):
            build_timeline_from_scenes(self.scenes_dir, self.config, duration_limit_seconds=3.0)

    def test_trim_final_clip_when_allowed(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        config = TimelineConfig(allow_trim_final_clip=True)
        timeline = build_timeline_from_scenes(self.scenes_dir, config, duration_limit_seconds=5.0)
        clips = timeline.tracks[0].clips
        self.assertAlmostEqual(timeline.duration_seconds, 5.0)
        self.assertAlmostEqual(clips[-1].duration_seconds, 1.0)
        self.assertAlmostEqual(clips[-1].end, 5.0)

    def test_trim_records_exact_amount_in_warning(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        config = TimelineConfig(allow_trim_final_clip=True)
        timeline = build_timeline_from_scenes(self.scenes_dir, config, duration_limit_seconds=5.0)
        warnings_text = " ".join(timeline.metadata["build_warnings"])
        self.assertIn("1.000s", warnings_text)

    def test_trim_never_touches_earlier_clips(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        config = TimelineConfig(allow_trim_final_clip=True)
        timeline = build_timeline_from_scenes(self.scenes_dir, config, duration_limit_seconds=5.0)
        clips = timeline.tracks[0].clips
        self.assertEqual((clips[0].start, clips[0].end), (0.0, 2.0))
        self.assertEqual((clips[1].start, clips[1].end), (2.0, 4.0))

    def test_trim_cannot_shrink_below_zero_raises(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        config = TimelineConfig(allow_trim_final_clip=True)
        with self.assertRaises(TimelineDurationLimitError):
            build_timeline_from_scenes(self.scenes_dir, config, duration_limit_seconds=0.5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _basic_valid_timeline() -> Timeline:
    track = TimelineTrack(
        track_id="track_video",
        track_type=TrackType.VIDEO,
        order=0,
        clips=[
            VideoClip(
                clip_id="clip_01",
                track_id="track_video",
                source_path="/scenes/scene_01.mp4",
                start=0.0,
                end=2.0,
                duration_seconds=2.0,
                source_in=0.0,
                source_out=2.0,
                scene_number=1,
            ),
            VideoClip(
                clip_id="clip_02",
                track_id="track_video",
                source_path="/scenes/scene_02.mp4",
                start=2.0,
                end=4.0,
                duration_seconds=2.0,
                source_in=0.0,
                source_out=2.0,
                scene_number=2,
            ),
        ],
    )
    return Timeline(
        timeline_id="abc123",
        duration_seconds=4.0,
        tracks=[track],
    )


class ValidationTests(unittest.TestCase):
    def test_valid_timeline_passes(self) -> None:
        result = validate_timeline(_basic_valid_timeline())
        self.assertTrue(result.passed)
        self.assertEqual(result.errors, [])

    def test_duplicate_track_id_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks.append(TimelineTrack(track_id="track_video", track_type=TrackType.AUDIO, order=1))
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("duplicate track_id" in e for e in result.errors))

    def test_duplicate_clip_id_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[1].clip_id = "clip_01"
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("duplicate clip_id" in e for e in result.errors))

    def test_bad_duration_math_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].duration_seconds = 99.0
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("does not match end-start" in e for e in result.errors))

    def test_end_not_greater_than_start_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].end = 0.0
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_negative_start_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].start = -1.0
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_overlap_within_video_track_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[1].start = 1.0
        timeline.tracks[0].clips[1].duration_seconds = 3.0
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("overlap" in e for e in result.errors))

    def test_gap_disallowed_by_default(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[1].start = 3.0
        timeline.tracks[0].clips[1].end = 5.0
        timeline.duration_seconds = 5.0
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("gap" in e for e in result.errors))

    def test_gap_allowed_when_configured(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[1].start = 3.0
        timeline.tracks[0].clips[1].end = 5.0
        timeline.duration_seconds = 5.0
        config = TimelineConfig(allow_video_gaps=True)
        result = validate_timeline(timeline, config)
        self.assertTrue(result.passed)

    def test_audio_overlap_disallowed_by_default(self) -> None:
        timeline = _basic_valid_timeline()
        audio_track = TimelineTrack(
            track_id="track_audio",
            track_type=TrackType.AUDIO,
            order=1,
            clips=[
                AudioClip(clip_id="a1", track_id="track_audio", source_path="/m.mp3", start=0.0, end=3.0, duration_seconds=3.0, source_out=3.0),
                AudioClip(clip_id="a2", track_id="track_audio", source_path="/m.mp3", start=1.0, end=4.0, duration_seconds=3.0, source_out=3.0),
            ],
        )
        timeline.tracks.append(audio_track)
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_audio_overlap_allowed_when_configured(self) -> None:
        timeline = _basic_valid_timeline()
        audio_track = TimelineTrack(
            track_id="track_audio",
            track_type=TrackType.AUDIO,
            order=1,
            clips=[
                AudioClip(clip_id="a1", track_id="track_audio", source_path="/m.mp3", start=0.0, end=3.0, duration_seconds=3.0, source_out=3.0),
                AudioClip(clip_id="a2", track_id="track_audio", source_path="/m.mp3", start=1.0, end=4.0, duration_seconds=3.0, source_out=3.0),
            ],
        )
        timeline.tracks.append(audio_track)
        config = TimelineConfig(allow_audio_overlap=True)
        result = validate_timeline(timeline, config)
        self.assertTrue(result.passed)

    def test_no_video_track_fails(self) -> None:
        timeline = Timeline(timeline_id="x", duration_seconds=0.0, tracks=[])
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("no video track" in e for e in result.errors))

    def test_empty_video_track_fails(self) -> None:
        timeline = Timeline(
            timeline_id="x",
            duration_seconds=0.0,
            tracks=[TimelineTrack(track_id="t", track_type=TrackType.VIDEO, order=0, clips=[])],
        )
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_timeline_duration_mismatch_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.duration_seconds = 999.0
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("does not match" in e and "maximum enabled clip end" in e for e in result.errors))

    def test_transition_not_none_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].transition_in = "crossfade"
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_invalid_volume_fails(self) -> None:
        timeline = _basic_valid_timeline()
        audio_track = TimelineTrack(
            track_id="track_audio",
            track_type=TrackType.AUDIO,
            order=1,
            clips=[AudioClip(clip_id="a1", track_id="track_audio", source_path="/m.mp3", start=0.0, end=4.0, duration_seconds=4.0, source_out=4.0, volume=-0.1)],
        )
        timeline.tracks.append(audio_track)
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_invalid_fade_totals_fail(self) -> None:
        timeline = _basic_valid_timeline()
        audio_track = TimelineTrack(
            track_id="track_audio",
            track_type=TrackType.AUDIO,
            order=1,
            clips=[
                AudioClip(
                    clip_id="a1", track_id="track_audio", source_path="/m.mp3",
                    start=0.0, end=4.0, duration_seconds=4.0, source_out=4.0,
                    fade_in_seconds=3.0, fade_out_seconds=3.0,
                )
            ],
        )
        timeline.tracks.append(audio_track)
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_non_serializable_metadata_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].metadata = {"bad": object()}
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("not JSON-serializable" in e for e in result.errors))

    def test_duplicate_scene_number_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[1].scene_number = 1
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)
        self.assertTrue(any("duplicate scene_number" in e for e in result.errors))

    def test_empty_source_path_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].source_path = ""
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)

    def test_playback_rate_zero_fails(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].playback_rate = 0.0
        result = validate_timeline(timeline)
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------


class JsonTests(TimelineTempTestCase):
    def test_round_trip_equality(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        timeline = build_timeline_from_scenes(
            self.scenes_dir, self.config, production_date=PRODUCTION_DATE, music_path="/m.mp3"
        )
        rebuilt = timeline_from_dict(timeline_to_dict(timeline))
        self.assertEqual(timeline_to_dict(rebuilt), timeline_to_dict(timeline))

    def test_atomic_write_no_tmp_left_behind(self) -> None:
        timeline = _basic_valid_timeline()
        out = self.temp_dir / "timeline.json"
        save_timeline(timeline, out)
        self.assertTrue(out.is_file())
        self.assertFalse((self.temp_dir / "timeline.json.tmp").exists())

    def test_overwrite_refused_without_force(self) -> None:
        timeline = _basic_valid_timeline()
        out = self.temp_dir / "timeline.json"
        save_timeline(timeline, out)
        with self.assertRaises(TimelineOutputExistsError):
            save_timeline(timeline, out)

    def test_overwrite_allowed_with_force(self) -> None:
        timeline = _basic_valid_timeline()
        out = self.temp_dir / "timeline.json"
        save_timeline(timeline, out)
        save_timeline(timeline, out, force=True)
        self.assertTrue(out.is_file())

    def test_output_path_is_directory_raises(self) -> None:
        timeline = _basic_valid_timeline()
        out_dir = self.temp_dir / "a_directory"
        out_dir.mkdir()
        with self.assertRaises(UnsafeTimelineOutputError):
            save_timeline(timeline, out_dir)

    def test_same_inputs_yield_same_timeline_id(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        first = build_timeline_from_scenes(self.scenes_dir, self.config, production_date=PRODUCTION_DATE)
        second = build_timeline_from_scenes(self.scenes_dir, self.config, production_date=PRODUCTION_DATE)
        self.assertEqual(first.timeline_id, second.timeline_id)
        self.assertNotEqual(first.timeline_id, "")

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(TimelineFileNotFoundError):
            load_timeline(self.temp_dir / "nope.json")

    def test_malformed_json_file_raises(self) -> None:
        bad = self.temp_dir / "bad.json"
        bad.write_text("{not valid json")
        with self.assertRaises(TimelineJSONError):
            load_timeline(bad)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(TimelineTempTestCase):
    def test_from_scenes_and_validate_together_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--from-scenes", "x", "--validate", "y"])

    def test_from_scenes_requires_one_mode(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments([])

    def test_music_flag_accepted(self) -> None:
        args = parse_arguments(["--from-scenes", "x", "--music", "/tmp/m.mp3"])
        self.assertEqual(args.music, "/tmp/m.mp3")

    def test_output_is_optional(self) -> None:
        args = parse_arguments(["--from-scenes", "x"])
        self.assertIsNone(args.output)

    def test_main_from_scenes_prints_summary(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["--from-scenes", str(self.scenes_dir)])
        self.assertIn("timeline_id:", buffer.getvalue())

    def test_main_from_scenes_json_output(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["--from-scenes", str(self.scenes_dir), "--json"])
        data = json.loads(buffer.getvalue())
        self.assertIn("timeline_id", data)

    def test_main_writes_output_only_when_requested(self) -> None:
        scenes = self._write_three_scenes()
        for scene in scenes:
            self._write_sidecar(scene, 2.0)
        out = self.temp_dir / "timeline.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["--from-scenes", str(self.scenes_dir), "--output", str(out)])
        self.assertTrue(out.is_file())

    def test_main_validate_mode(self) -> None:
        timeline = _basic_valid_timeline()
        out = self.temp_dir / "timeline.json"
        save_timeline(timeline, out)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            main(["--validate", str(out)])
        self.assertIn("PASSED", buffer.getvalue())

    def test_main_validate_mode_failure_exits_nonzero(self) -> None:
        timeline = _basic_valid_timeline()
        timeline.tracks[0].clips[0].end = 0.0
        out = self.temp_dir / "timeline.json"
        save_timeline(timeline, out)
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            main(["--validate", str(out)])


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

    def test_no_ffmpeg_or_ffprobe_invocation(self) -> None:
        for forbidden in ("ffmpeg_binary", "ffprobe_binary", "subprocess.run", "subprocess.Popen"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_media_inspector_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .media_inspector"))
            self.assertFalse(stripped.startswith("from src.media_inspector"))
            self.assertFalse(stripped.startswith("import media_inspector"))

    def test_no_video_engine_or_music_mixer_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .video_engine"))
            self.assertFalse(stripped.startswith("from .music_mixer"))
            self.assertFalse(stripped.startswith("from .reel_builder"))

    def test_no_write_mode_open_against_scene_paths(self) -> None:
        self.assertNotIn('"w"', MODULE_SOURCE)
        self.assertNotIn("'w'", MODULE_SOURCE)

    def test_no_automatic_music_selection(self) -> None:
        for forbidden in ("random.choice", "glob(", "rglob(", "listdir(", "default_music"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_network_or_download_code(self) -> None:
        for forbidden in ("requests.", "urllib.request", "http.client", "socket."):
            self.assertNotIn(forbidden, MODULE_SOURCE)


if __name__ == "__main__":
    unittest.main()
