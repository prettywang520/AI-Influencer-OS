from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import timeline_engine
from .overlay_plan_engine import (
    CoordinateSpace,
    OverlayCollision,
    OverlayLayer,
    OverlayMappingError,
    OverlayPlan,
    OverlayPlanConfig,
    OverlayPlanJSONError,
    OverlayPlanOutputExistsError,
    OverlayPosition,
    OverlaySafeArea,
    OverlaySeverity,
    OverlayStyleSnapshot,
    OverlayTimelineLoadError,
    OverlayTimelineValidationError,
    OverlayType,
    OverlayTypeConfig,
    PlannedOverlay,
    UnsafeOverlayPlanOutputError,
    UnsupportedOverlayTypeError,
    build_overlay_plan,
    calculate_overlay_layers,
    calculate_safe_area_status,
    detect_overlay_collisions,
    discover_overlay_clips,
    load_overlay_plan,
    load_overlay_plan_config,
    map_timeline_overlay_clip,
    overlay_plan_from_dict,
    overlay_plan_to_dict,
    parse_arguments,
    resolve_overlay_type,
    save_overlay_plan,
    validate_overlay_plan,
)

MODULE_PATH = Path(__file__).resolve().parent / "overlay_plan_engine.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _video_track(*, num_clips: int = 1, clip_duration: float = 10.0, order: int = 0) -> timeline_engine.TimelineTrack:
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
    return timeline_engine.TimelineTrack(track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=order, clips=clips)


def _subtitle_clip(
    clip_id: str = "sub_1", *, start: float = 0.0, end: float = 2.0, text: str = "hello there",
    enabled: bool = True, position: dict | None = None, style: dict | None = None, extra_metadata: dict | None = None,
) -> timeline_engine.OverlayClip:
    metadata: dict = {
        "position": position if position is not None else {"x": 540, "y": 1450, "anchor": "bottom_center"},
        "style": style if style is not None else {"style_id": "default", "font_family": "Arial", "font_size": 54},
        "language": "en",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return timeline_engine.OverlayClip(
        clip_id=clip_id, track_id="track_subtitles", source_path="", overlay_type="subtitle",
        start=start, end=end, duration_seconds=end - start, source_out=end - start,
        content=text, enabled=enabled, metadata=metadata,
    )


def _subtitle_track(clips: list[timeline_engine.OverlayClip], *, order: int = 1) -> timeline_engine.TimelineTrack:
    return timeline_engine.TimelineTrack(
        track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=order, clips=clips
    )


def _overlay_clip(
    clip_id: str, overlay_type: str, *, start: float = 0.0, end: float = 3.0, content: str = "",
    position: dict | None = None, asset: dict | None = None, enabled: bool = True, extra_metadata: dict | None = None,
    track_id: str = "track_overlay",
) -> timeline_engine.OverlayClip:
    metadata: dict = {}
    if position is not None:
        metadata["position"] = position
    if asset is not None:
        metadata["asset"] = asset
    if extra_metadata:
        metadata.update(extra_metadata)
    return timeline_engine.OverlayClip(
        clip_id=clip_id, track_id=track_id, source_path="", overlay_type=overlay_type,
        start=start, end=end, duration_seconds=end - start, source_out=end - start,
        content=content, enabled=enabled, metadata=metadata,
    )


def _overlay_track(
    track_id: str, clips: list[timeline_engine.OverlayClip], *, order: int = 2
) -> timeline_engine.TimelineTrack:
    return timeline_engine.TimelineTrack(track_id=track_id, track_type=timeline_engine.TrackType.OVERLAY, order=order, clips=clips)


def _timeline(
    tracks: list[timeline_engine.TimelineTrack], *, timeline_id: str = "tl1", duration: float = 10.0,
    metadata: dict | None = None, width: int = 1080, height: int = 1920,
) -> timeline_engine.Timeline:
    return timeline_engine.Timeline(
        timeline_id=timeline_id, duration_seconds=duration, tracks=tracks,
        metadata=metadata or {}, width=width, height=height,
    )


class OverlayPlanTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = load_overlay_plan_config()

    def _write_timeline(self, timeline: timeline_engine.Timeline, *, name: str = "timeline.json") -> Path:
        path = self.temp_dir / name
        timeline_engine.save_timeline(timeline, path)
        return path

    def _hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class LoadingTests(OverlayPlanTempTestCase):
    def test_valid_timeline_loads(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        self.assertEqual(plan.overlay_count, 1)

    def test_malformed_timeline_json_fails(self) -> None:
        path = self.temp_dir / "timeline.json"
        path.write_text("{not valid json")
        with self.assertRaises(OverlayTimelineLoadError):
            build_overlay_plan(path, self.config)

    def test_invalid_timeline_fails(self) -> None:
        bad_timeline = timeline_engine.Timeline(timeline_id="bad", duration_seconds=0.0, tracks=[])
        timeline_path = self._write_timeline(bad_timeline)
        with self.assertRaises(OverlayTimelineValidationError):
            build_overlay_plan(timeline_path, self.config)

    def test_source_timeline_unchanged(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        before = self._hash(timeline_path)
        build_overlay_plan(timeline_path, self.config)
        self.assertEqual(self._hash(timeline_path), before)

    def test_integrated_timeline_id_preserved(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])], metadata={"integrated_timeline_id": "itl-123"})
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        self.assertEqual(plan.integrated_timeline_id, "itl-123")

    def test_unknown_timeline_metadata_preserved_in_plan_metadata_path(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])], metadata={"custom_field": "value"})
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        self.assertEqual(plan.metadata["source_timeline_path"], str(timeline_path))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class DiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_overlay_plan_config()

    def test_subtitle_track_discovered(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        pairs = discover_overlay_clips(timeline, self.config)
        self.assertEqual(len(pairs), 1)

    def test_overlay_track_discovered(self) -> None:
        timeline = _timeline([_video_track(), _overlay_track("track_cta", [_overlay_clip("cta_1", "cta")])])
        pairs = discover_overlay_clips(timeline, self.config)
        self.assertEqual(len(pairs), 1)

    def test_video_audio_metadata_tracks_ignored(self) -> None:
        audio_track = timeline_engine.TimelineTrack(
            track_id="track_audio", track_type=timeline_engine.TrackType.AUDIO, order=1,
            clips=[timeline_engine.AudioClip(clip_id="a1", track_id="track_audio", source_path="/m.mp3", start=0.0, end=5.0, duration_seconds=5.0, source_out=5.0)],
        )
        metadata_track = timeline_engine.TimelineTrack(track_id="track_meta", track_type=timeline_engine.TrackType.METADATA, order=2)
        timeline = _timeline([_video_track(), audio_track, metadata_track])
        pairs = discover_overlay_clips(timeline, self.config)
        self.assertEqual(pairs, [])

    def test_disabled_overlay_excluded_by_default(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip(enabled=False)])])
        pairs = discover_overlay_clips(timeline, self.config)
        self.assertEqual(pairs, [])

    def test_disabled_overlay_included_with_option(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip(enabled=False)])])
        pairs = discover_overlay_clips(timeline, self.config, include_disabled=True)
        self.assertEqual(len(pairs), 1)

    def test_no_overlays_timeline_creates_valid_empty_plan(self) -> None:
        timeline = _timeline([_video_track()])
        with tempfile.TemporaryDirectory() as tmp:
            timeline_path = Path(tmp) / "timeline.json"
            timeline_engine.save_timeline(timeline, timeline_path)
            plan = build_overlay_plan(timeline_path, self.config)
            self.assertEqual(plan.overlay_count, 0)
            self.assertTrue(any(w.code == "no_overlays_found" for w in plan.warnings))

    def test_fail_on_no_overlays_configured(self) -> None:
        timeline = _timeline([_video_track()])
        config = OverlayPlanConfig(fail_on_no_overlays=True, overlay_types=self.config.overlay_types)
        with tempfile.TemporaryDirectory() as tmp:
            timeline_path = Path(tmp) / "timeline.json"
            timeline_engine.save_timeline(timeline, timeline_path)
            with self.assertRaises(OverlayMappingError):
                build_overlay_plan(timeline_path, config)

    def test_custom_overlay_type_allowed_when_configured(self) -> None:
        timeline = _timeline([_video_track(), _overlay_track("track_x", [_overlay_clip("x1", "billboard")])])
        pairs = discover_overlay_clips(timeline, self.config)
        track, clip = pairs[0]
        overlay_type, custom_type = resolve_overlay_type(clip, track, self.config)
        self.assertEqual(overlay_type, OverlayType.CUSTOM)
        self.assertEqual(custom_type, "billboard")

    def test_custom_overlay_type_blocked_when_configured(self) -> None:
        timeline = _timeline([_video_track(), _overlay_track("track_x", [_overlay_clip("x1", "billboard")])])
        pairs = discover_overlay_clips(timeline, self.config)
        track, clip = pairs[0]
        config = OverlayPlanConfig(allow_custom_overlay_types=False, overlay_types=self.config.overlay_types)
        with self.assertRaises(UnsupportedOverlayTypeError):
            resolve_overlay_type(clip, track, config)

    def test_video_clip_with_explicit_overlay_descriptor_discovered(self) -> None:
        video_track = _video_track()
        video_track.clips[0].metadata["is_overlay_descriptor"] = True
        timeline = _timeline([video_track])
        pairs = discover_overlay_clips(timeline, self.config)
        self.assertEqual(len(pairs), 1)

    def test_discovery_order_deterministic(self) -> None:
        clips = [
            _subtitle_clip("sub_b", start=2.0, end=4.0),
            _subtitle_clip("sub_a", start=0.0, end=2.0),
        ]
        timeline = _timeline([_video_track(), _subtitle_track(clips)])
        pairs = discover_overlay_clips(timeline, self.config)
        self.assertEqual([c.clip_id for _t, c in pairs], ["sub_a", "sub_b"])


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


class MappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_overlay_plan_config()

    def _map_single(self, clip, track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY):
        track = timeline_engine.TimelineTrack(track_id=track_id, track_type=track_type, order=1, clips=[clip])
        return map_timeline_overlay_clip(track, clip, self.config)

    def test_subtitle_overlay_maps_once(self) -> None:
        overlay = self._map_single(_subtitle_clip(), track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE)
        self.assertEqual(overlay.overlay_type, OverlayType.SUBTITLE)

    def test_logo_maps_once(self) -> None:
        overlay = self._map_single(_overlay_clip("logo_1", "logo", position={"x": 100, "y": 100}))
        self.assertEqual(overlay.overlay_type, OverlayType.LOGO)

    def test_watermark_maps_once(self) -> None:
        overlay = self._map_single(_overlay_clip("wm_1", "watermark", position={"x": 100, "y": 100}))
        self.assertEqual(overlay.overlay_type, OverlayType.WATERMARK)

    def test_location_maps_once(self) -> None:
        overlay = self._map_single(_overlay_clip("loc_1", "location", content="Kyoto, Japan"))
        self.assertEqual(overlay.overlay_type, OverlayType.LOCATION)

    def test_cta_maps_once(self) -> None:
        overlay = self._map_single(_overlay_clip("cta_1", "cta", content="Follow us"))
        self.assertEqual(overlay.overlay_type, OverlayType.CTA)

    def test_sticker_maps_once(self) -> None:
        overlay = self._map_single(_overlay_clip("sticker_1", "sticker"))
        self.assertEqual(overlay.overlay_type, OverlayType.STICKER)

    def test_custom_future_maps_once(self) -> None:
        overlay = self._map_single(_overlay_clip("future_1", "poll_widget"))
        self.assertEqual(overlay.overlay_type, OverlayType.CUSTOM)
        self.assertEqual(overlay.custom_overlay_type, "poll_widget")

    def test_content_preserved_exactly(self) -> None:
        overlay = self._map_single(_subtitle_clip(text="hello there"), track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE)
        self.assertEqual(overlay.content, "hello there")

    def test_unicode_cjk_emoji_preserved(self) -> None:
        text = "京都の朝 🌸 早晨"
        overlay = self._map_single(_subtitle_clip(text=text), track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE)
        self.assertEqual(overlay.content, text)

    def test_style_snapshot_preserved(self) -> None:
        overlay = self._map_single(
            _subtitle_clip(style={"style_id": "luxury_editorial", "font_family": "Helvetica Neue", "font_size": 48}),
            track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE,
        )
        self.assertEqual(overlay.style_snapshot.font_family, "Helvetica Neue")
        self.assertEqual(overlay.style_snapshot.font_size, 48)

    def test_asset_reference_preserved(self) -> None:
        overlay = self._map_single(_overlay_clip("logo_1", "logo", position={"x": 100, "y": 100}, asset={"asset_reference": "logo.png", "width": 80, "height": 40}))
        self.assertEqual(overlay.style_snapshot.asset_reference, "logo.png")
        self.assertEqual(overlay.style_snapshot.width, 80)

    def test_metadata_preserved(self) -> None:
        overlay = self._map_single(_subtitle_clip(extra_metadata={"cue_metadata": {"take": 3}}), track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE)
        self.assertEqual(overlay.metadata["take"], 3)

    def test_source_clip_and_track_ids_preserved(self) -> None:
        overlay = self._map_single(_subtitle_clip(clip_id="sub_99"), track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE)
        self.assertEqual(overlay.source_clip_id, "sub_99")
        self.assertEqual(overlay.source_track_id, "track_subtitles")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class TimingTests(OverlayPlanTempTestCase):
    def test_valid_timing_passes(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        result = validate_overlay_plan(plan, self.config)
        self.assertTrue(result.passed)

    def test_negative_start_fails(self) -> None:
        plan = OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[OverlayLayer(layer_id="layer_subtitle")],
                            overlays=[PlannedOverlay(overlay_id="o1", layer_id="layer_subtitle", timeline_start_seconds=-1.0, timeline_end_seconds=2.0, duration_seconds=3.0, content="hi")])
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_end_not_greater_than_start_fails(self) -> None:
        plan = OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[OverlayLayer(layer_id="layer_subtitle")],
                            overlays=[PlannedOverlay(overlay_id="o1", layer_id="layer_subtitle", timeline_start_seconds=2.0, timeline_end_seconds=2.0, duration_seconds=0.0, content="hi")])
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_duration_mismatch_fails(self) -> None:
        plan = OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[OverlayLayer(layer_id="layer_subtitle")],
                            overlays=[PlannedOverlay(overlay_id="o1", layer_id="layer_subtitle", timeline_start_seconds=0.0, timeline_end_seconds=2.0, duration_seconds=5.0, content="hi")])
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_cue_beyond_timeline_fails(self) -> None:
        timeline = _timeline([_video_track(num_clips=1, clip_duration=1.0)])
        timeline.tracks.append(_subtitle_track([_subtitle_clip(start=0.0, end=5.0)]))
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(OverlayTimelineValidationError):
            build_overlay_plan(timeline_path, self.config)

    def test_exact_timeline_end_overlay_passes(self) -> None:
        timeline = _timeline([_video_track(num_clips=1, clip_duration=10.0), _subtitle_track([_subtitle_clip(start=8.0, end=10.0)])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        result = validate_overlay_plan(plan, self.config)
        self.assertTrue(result.passed)

    def test_disabled_overlay_timing_excluded_by_default(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip(enabled=False, start=0.0, end=2.0)])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        self.assertEqual(plan.overlay_count, 0)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class PositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_overlay_plan_config()

    def _plan_with_overlay(self, position: OverlayPosition, overlay_type: str = OverlayType.SUBTITLE) -> OverlayPlan:
        overlay = PlannedOverlay(
            overlay_id="o1", layer_id=f"layer_{overlay_type}", overlay_type=overlay_type,
            timeline_start_seconds=0.0, timeline_end_seconds=2.0, duration_seconds=2.0,
            position=position, content="hi",
        )
        layer = OverlayLayer(layer_id=f"layer_{overlay_type}", overlay_type=overlay_type)
        return OverlayPlan(timeline_id="tl1", duration_seconds=10.0, canvas_width=1080, canvas_height=1920, layers=[layer], overlays=[overlay])

    def test_valid_pixel_position_passes(self) -> None:
        plan = self._plan_with_overlay(OverlayPosition(x=540, y=1450, coordinate_space=CoordinateSpace.PIXEL))
        result = validate_overlay_plan(plan, self.config)
        self.assertTrue(result.passed)

    def test_invalid_pixel_position_fails(self) -> None:
        plan = self._plan_with_overlay(OverlayPosition(x=5000, y=1450, coordinate_space=CoordinateSpace.PIXEL))
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_normalized_position_passes(self) -> None:
        plan = self._plan_with_overlay(OverlayPosition(x=0.5, y=0.8, coordinate_space=CoordinateSpace.NORMALIZED))
        result = validate_overlay_plan(plan, self.config)
        self.assertTrue(result.passed)

    def test_normalized_position_outside_range_fails(self) -> None:
        plan = self._plan_with_overlay(OverlayPosition(x=1.5, y=0.8, coordinate_space=CoordinateSpace.NORMALIZED))
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_invalid_anchor_fails(self) -> None:
        plan = self._plan_with_overlay(OverlayPosition(x=540, y=1450, anchor="middle_of_nowhere"))
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_negative_width_fails(self) -> None:
        plan = self._plan_with_overlay(OverlayPosition(x=540, y=1450, width=-10))
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_negative_height_fails(self) -> None:
        plan = self._plan_with_overlay(OverlayPosition(x=540, y=1450, height=-10))
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_custom_position_retained(self) -> None:
        clip = _overlay_clip("logo_1", "logo", position={"x": 123, "y": 456})
        track = timeline_engine.TimelineTrack(track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY, order=1, clips=[clip])
        overlay = map_timeline_overlay_clip(track, clip, self.config)
        self.assertEqual(overlay.position.x, 123)
        self.assertEqual(overlay.position.y, 456)

    def test_default_position_applied_only_when_configured(self) -> None:
        clip = _overlay_clip("logo_1", "logo")  # no position metadata at all
        track = timeline_engine.TimelineTrack(track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY, order=1, clips=[clip])
        overlay = map_timeline_overlay_clip(track, clip, self.config)
        self.assertIsNone(overlay.position.x)
        self.assertIsNone(overlay.position.y)
        self.assertEqual(overlay.position.anchor, self.config.default_anchor)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


class StyleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_overlay_plan_config()

    def _plan_with_overlay(self, **kwargs) -> OverlayPlan:
        style = OverlayStyleSnapshot(scale=kwargs.pop("scale", 1.0))
        overlay = PlannedOverlay(
            overlay_id="o1", layer_id="layer_subtitle", overlay_type=OverlayType.SUBTITLE,
            timeline_start_seconds=0.0, timeline_end_seconds=2.0, duration_seconds=2.0,
            content="hi", style_snapshot=style, **kwargs,
        )
        layer = OverlayLayer(layer_id="layer_subtitle", overlay_type=OverlayType.SUBTITLE)
        return OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[layer], overlays=[overlay])

    def test_valid_opacity_passes(self) -> None:
        plan = self._plan_with_overlay(opacity=0.8)
        result = validate_overlay_plan(plan, self.config)
        self.assertTrue(result.passed)

    def test_opacity_outside_range_fails(self) -> None:
        plan = self._plan_with_overlay(opacity=1.5)
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_scale_limits_enforced(self) -> None:
        plan = self._plan_with_overlay(scale=100.0)
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_rotation_preserved(self) -> None:
        clip = _overlay_clip("sticker_1", "sticker", extra_metadata={"rotation_degrees": 45.0})
        track = timeline_engine.TimelineTrack(track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY, order=1, clips=[clip])
        overlay = map_timeline_overlay_clip(track, clip, self.config)
        self.assertEqual(overlay.style_snapshot.rotation_degrees, 45.0)

    def test_style_metadata_serializable(self) -> None:
        clip = _subtitle_clip(style={"style_id": "default", "metadata": {"note": "x"}})
        track = timeline_engine.TimelineTrack(track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=1, clips=[clip])
        overlay = map_timeline_overlay_clip(track, clip, self.config)
        json.dumps(overlay.style_snapshot.metadata)  # must not raise

    def test_missing_required_subtitle_content_fails(self) -> None:
        plan = self._plan_with_overlay()
        plan.overlays[0].content = ""
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_logo_asset_reference_optional(self) -> None:
        clip = _overlay_clip("logo_1", "logo", position={"x": 100, "y": 100})  # no asset metadata
        track = timeline_engine.TimelineTrack(track_id="track_overlay", track_type=timeline_engine.TrackType.OVERLAY, order=1, clips=[clip])
        overlay = map_timeline_overlay_clip(track, clip, self.config)
        self.assertIsNone(overlay.style_snapshot.asset_reference)

    def test_no_font_file_access(self) -> None:
        # .open( is used legitimately for config/JSON files -- checked
        # precisely for actual font-file patterns only, not a bare
        # substring that would false-fail on ordinary file I/O.
        for forbidden in (".ttf", ".otf", ".woff", "font_path", "FontFile"):
            self.assertNotIn(forbidden, MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Safe area
# ---------------------------------------------------------------------------


class SafeAreaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_overlay_plan_config()

    def test_subtitle_inside_safe_area_passes(self) -> None:
        status = calculate_safe_area_status(OverlayPosition(x=540, y=1450), OverlayType.SUBTITLE, 1080, 1920, self.config)
        self.assertTrue(status.inside_safe_area)

    def test_subtitle_outside_safe_area_fails_policy(self) -> None:
        status = calculate_safe_area_status(OverlayPosition(x=540, y=50), OverlayType.SUBTITLE, 1080, 1920, self.config)
        self.assertFalse(status.inside_safe_area)
        self.assertEqual(status.severity, OverlaySeverity.ERROR)

    def test_cta_outside_action_safe_area_fails(self) -> None:
        status = calculate_safe_area_status(OverlayPosition(x=540, y=1919), OverlayType.CTA, 1080, 1920, self.config)
        self.assertFalse(status.inside_safe_area)
        self.assertEqual(status.severity, OverlaySeverity.ERROR)

    def test_watermark_edge_warns(self) -> None:
        status = calculate_safe_area_status(OverlayPosition(x=1079, y=50), OverlayType.WATERMARK, 1080, 1920, self.config)
        self.assertFalse(status.inside_safe_area)
        self.assertEqual(status.severity, OverlaySeverity.WARNING)

    def test_location_violation_warns(self) -> None:
        status = calculate_safe_area_status(OverlayPosition(x=10, y=1450), OverlayType.LOCATION, 1080, 1920, self.config)
        self.assertFalse(status.inside_safe_area)
        self.assertEqual(status.severity, OverlaySeverity.WARNING)

    def test_safe_area_recommendation_generated(self) -> None:
        status = calculate_safe_area_status(OverlayPosition(x=540, y=50), OverlayType.SUBTITLE, 1080, 1920, self.config)
        self.assertIsNotNone(status.recommended_position)

    def test_no_automatic_reposition_occurs(self) -> None:
        original = OverlayPosition(x=540, y=50)
        calculate_safe_area_status(original, OverlayType.SUBTITLE, 1080, 1920, self.config)
        self.assertEqual(original.x, 540)
        self.assertEqual(original.y, 50)

    def test_no_position_means_no_violation(self) -> None:
        status = calculate_safe_area_status(OverlayPosition(), OverlayType.SUBTITLE, 1080, 1920, self.config)
        self.assertTrue(status.inside_safe_area)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class LayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_overlay_plan_config()

    def test_deterministic_layer_ordering(self) -> None:
        overlays = [
            PlannedOverlay(overlay_id="o_wm", overlay_type=OverlayType.WATERMARK, layer_id="layer_watermark", timeline_start_seconds=0, timeline_end_seconds=1, duration_seconds=1),
            PlannedOverlay(overlay_id="o_sub", overlay_type=OverlayType.SUBTITLE, layer_id="layer_subtitle", timeline_start_seconds=0, timeline_end_seconds=1, duration_seconds=1),
            PlannedOverlay(overlay_id="o_cta", overlay_type=OverlayType.CTA, layer_id="layer_cta", timeline_start_seconds=0, timeline_end_seconds=1, duration_seconds=1),
        ]
        layers = calculate_overlay_layers(overlays, self.config)
        self.assertEqual([l.overlay_type for l in layers], [OverlayType.SUBTITLE, OverlayType.CTA, OverlayType.WATERMARK])

    def test_deterministic_overlay_ordering_within_layer(self) -> None:
        overlays = [
            PlannedOverlay(overlay_id="o_b", overlay_type=OverlayType.SUBTITLE, layer_id="layer_subtitle", timeline_start_seconds=2.0, timeline_end_seconds=3.0, duration_seconds=1.0),
            PlannedOverlay(overlay_id="o_a", overlay_type=OverlayType.SUBTITLE, layer_id="layer_subtitle", timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0),
        ]
        layers = calculate_overlay_layers(overlays, self.config)
        self.assertEqual(layers[0].overlay_ids, ["o_a", "o_b"])

    def test_configured_default_z_index_used(self) -> None:
        clip = _subtitle_clip()
        track = timeline_engine.TimelineTrack(track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=1, clips=[clip])
        overlay = map_timeline_overlay_clip(track, clip, self.config)
        self.assertEqual(overlay.z_index, self.config.overlay_types[OverlayType.SUBTITLE].default_z_index)

    def test_source_z_index_override_preserved(self) -> None:
        clip = _subtitle_clip(extra_metadata={"z_index": 999})
        track = timeline_engine.TimelineTrack(track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=1, clips=[clip])
        overlay = map_timeline_overlay_clip(track, clip, self.config)
        self.assertEqual(overlay.z_index, 999)

    def test_duplicate_z_index_deterministic(self) -> None:
        overlays = [
            PlannedOverlay(overlay_id="o_b", overlay_type=OverlayType.SUBTITLE, layer_id="layer_subtitle", z_index=100, timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0),
            PlannedOverlay(overlay_id="o_a", overlay_type=OverlayType.SUBTITLE, layer_id="layer_subtitle", z_index=100, timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0),
        ]
        layers1 = calculate_overlay_layers(overlays, self.config)
        layers2 = calculate_overlay_layers(list(reversed(overlays)), self.config)
        self.assertEqual(layers1[0].overlay_ids, layers2[0].overlay_ids)

    def test_negative_z_index_policy_enforced(self) -> None:
        plan = OverlayPlan(
            timeline_id="tl1", duration_seconds=10.0,
            layers=[OverlayLayer(layer_id="layer_subtitle")],
            overlays=[PlannedOverlay(overlay_id="o1", layer_id="layer_subtitle", z_index=-1, timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0, content="hi")],
        )
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_negative_z_index_allowed_when_configured(self) -> None:
        config = OverlayPlanConfig(allow_negative_z_index=True, overlay_types=self.config.overlay_types)
        plan = OverlayPlan(
            timeline_id="tl1", duration_seconds=10.0,
            layers=[OverlayLayer(layer_id="layer_subtitle")],
            overlays=[PlannedOverlay(overlay_id="o1", layer_id="layer_subtitle", z_index=-1, timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0, content="hi")],
        )
        result = validate_overlay_plan(plan, config)
        self.assertTrue(result.passed)

    def test_unique_layer_ids(self) -> None:
        plan = OverlayPlan(
            timeline_id="tl1", duration_seconds=10.0,
            layers=[OverlayLayer(layer_id="layer_subtitle"), OverlayLayer(layer_id="layer_subtitle")],
        )
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_overlay_references_valid_layer(self) -> None:
        plan = OverlayPlan(
            timeline_id="tl1", duration_seconds=10.0, layers=[OverlayLayer(layer_id="layer_subtitle")],
            overlays=[PlannedOverlay(overlay_id="o1", layer_id="layer_missing", timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0, content="hi")],
        )
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)


# ---------------------------------------------------------------------------
# Collisions
# ---------------------------------------------------------------------------


class CollisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_overlay_plan_config()

    def _bounded(self, overlay_id, overlay_type, start, end, x, y, w=100, h=50, collision_group=None):
        return PlannedOverlay(
            overlay_id=overlay_id, overlay_type=overlay_type, layer_id=f"layer_{overlay_type}",
            timeline_start_seconds=start, timeline_end_seconds=end, duration_seconds=end - start,
            position=OverlayPosition(x=x, y=y, width=w, height=h),
            collision_group=collision_group or self.config.overlay_types.get(overlay_type, self.config.overlay_types[OverlayType.CUSTOM]).collision_group,
            content="x",
        )

    def test_no_time_overlap_no_collision(self) -> None:
        overlays = [self._bounded("a", OverlayType.CTA, 0.0, 2.0, 500, 500), self._bounded("b", OverlayType.CTA, 3.0, 5.0, 500, 500)]
        collisions = detect_overlay_collisions(overlays, self.config)
        self.assertEqual(collisions, [])

    def test_time_and_spatial_overlap_detected(self) -> None:
        overlays = [self._bounded("a", OverlayType.CTA, 0.0, 5.0, 500, 500), self._bounded("b", OverlayType.LOCATION, 1.0, 3.0, 500, 500)]
        collisions = detect_overlay_collisions(overlays, self.config)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0].collision_type, "spatial_overlap")

    def test_missing_bounds_creates_unknown_bounds_warning(self) -> None:
        overlays = [
            PlannedOverlay(overlay_id="a", overlay_type=OverlayType.SUBTITLE, layer_id="layer_subtitle", timeline_start_seconds=0.0, timeline_end_seconds=5.0, duration_seconds=5.0, content="x", position=OverlayPosition()),
            self._bounded("b", OverlayType.CTA, 1.0, 3.0, 500, 500),
        ]
        collisions = detect_overlay_collisions(overlays, self.config)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0].collision_type, "unknown_bounds")
        self.assertEqual(collisions[0].severity, OverlaySeverity.WARNING)

    def test_subtitle_cta_conflict_error_by_config(self) -> None:
        overlays = [self._bounded("sub", OverlayType.SUBTITLE, 0.0, 5.0, 500, 1450), self._bounded("cta", OverlayType.CTA, 1.0, 3.0, 500, 1450)]
        collisions = detect_overlay_collisions(overlays, self.config)
        self.assertEqual(collisions[0].collision_type, "cta_subtitle_conflict")
        self.assertEqual(collisions[0].severity, OverlaySeverity.ERROR)

    def test_subtitle_cta_overlap_not_error_when_disabled(self) -> None:
        config = OverlayPlanConfig(fail_on_subtitle_cta_overlap=False, overlay_types=self.config.overlay_types)
        overlays = [self._bounded("sub", OverlayType.SUBTITLE, 0.0, 5.0, 500, 1450), self._bounded("cta", OverlayType.CTA, 1.0, 3.0, 500, 1450)]
        collisions = detect_overlay_collisions(overlays, config)
        self.assertEqual(collisions[0].collision_type, "spatial_overlap")

    def test_branding_overlays_follow_branding_policy(self) -> None:
        overlays = [self._bounded("logo", OverlayType.LOGO, 0.0, 5.0, 500, 500, collision_group="branding"), self._bounded("wm", OverlayType.WATERMARK, 1.0, 3.0, 500, 500, collision_group="branding")]
        collisions = detect_overlay_collisions(overlays, self.config)
        self.assertEqual(collisions[0].collision_type, "layer_conflict")

    def test_disabled_overlay_excluded_from_active_collision(self) -> None:
        a = self._bounded("a", OverlayType.CTA, 0.0, 5.0, 500, 500)
        b = self._bounded("b", OverlayType.LOCATION, 1.0, 3.0, 500, 500)
        b.enabled = False
        collisions = detect_overlay_collisions([a, b], self.config)
        self.assertEqual(collisions, [])

    def test_collision_id_stable(self) -> None:
        overlays = [self._bounded("a", OverlayType.CTA, 0.0, 5.0, 500, 500), self._bounded("b", OverlayType.LOCATION, 1.0, 3.0, 500, 500)]
        collisions1 = detect_overlay_collisions(overlays, self.config)
        collisions2 = detect_overlay_collisions(overlays, self.config)
        self.assertEqual(collisions1[0].collision_id, collisions2[0].collision_id)

    def test_suggested_resolution_is_metadata_only(self) -> None:
        overlays = [self._bounded("a", OverlayType.CTA, 0.0, 5.0, 500, 500), self._bounded("b", OverlayType.LOCATION, 1.0, 3.0, 500, 500)]
        collisions = detect_overlay_collisions(overlays, self.config)
        self.assertIsInstance(collisions[0].suggested_resolution, str)
        # positions themselves must be untouched
        self.assertEqual(overlays[0].position.x, 500)
        self.assertEqual(overlays[1].position.x, 500)

    def test_no_overlay_automatically_moved(self) -> None:
        a = self._bounded("a", OverlayType.CTA, 0.0, 5.0, 500, 500)
        b = self._bounded("b", OverlayType.LOCATION, 1.0, 3.0, 500, 500)
        original_positions = [(a.position.x, a.position.y), (b.position.x, b.position.y)]
        detect_overlay_collisions([a, b], self.config)
        self.assertEqual([(a.position.x, a.position.y), (b.position.x, b.position.y)], original_positions)

    def test_duplicate_position_detected(self) -> None:
        a = self._bounded("a", OverlayType.LOGO, 0.0, 5.0, 500, 500, w=10, h=10, collision_group="a")
        b = self._bounded("b", OverlayType.LOGO, 3.0, 8.0, 500, 500, w=10, h=10, collision_group="b")
        # non-overlapping tiny boxes but identical position and time-overlapping
        collisions = detect_overlay_collisions([a, b], self.config)
        self.assertTrue(any(c.collision_type in ("spatial_overlap", "duplicate_position") for c in collisions))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityTests(OverlayPlanTempTestCase):
    _counter = 0

    def _timeline_path(self, **kwargs) -> Path:
        IdentityTests._counter += 1
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip(**kwargs)])])
        return self._write_timeline(timeline, name=f"timeline_{IdentityTests._counter}.json")

    def test_stable_plan_id(self) -> None:
        timeline_path = self._timeline_path()
        plan1 = build_overlay_plan(timeline_path, self.config)
        plan2 = build_overlay_plan(timeline_path, self.config)
        self.assertEqual(plan1.plan_id, plan2.plan_id)

    def test_created_at_excluded_from_plan_id(self) -> None:
        timeline_path = self._timeline_path()
        plan1 = build_overlay_plan(timeline_path, self.config)
        plan2 = build_overlay_plan(timeline_path, self.config)
        self.assertNotEqual(plan1.created_at, plan2.created_at)
        self.assertEqual(plan1.plan_id, plan2.plan_id)

    def test_include_disabled_changes_plan_id(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip(enabled=False)])])
        timeline_path = self._write_timeline(timeline)
        plan1 = build_overlay_plan(timeline_path, self.config, include_disabled=False)
        plan2 = build_overlay_plan(timeline_path, self.config, include_disabled=True)
        self.assertNotEqual(plan1.plan_id, plan2.plan_id)

    def test_style_change_changes_plan_id(self) -> None:
        p1 = self._timeline_path(style={"style_id": "default", "font_family": "Arial"})
        p2 = self._timeline_path(style={"style_id": "default", "font_family": "Helvetica"})
        plan1 = build_overlay_plan(p1, self.config)
        plan2 = build_overlay_plan(p2, self.config)
        self.assertNotEqual(plan1.plan_id, plan2.plan_id)

    def test_timing_change_changes_plan_id(self) -> None:
        p1 = self._timeline_path(start=0.0, end=2.0)
        p2 = self._timeline_path(start=0.0, end=3.0)
        plan1 = build_overlay_plan(p1, self.config)
        plan2 = build_overlay_plan(p2, self.config)
        self.assertNotEqual(plan1.plan_id, plan2.plan_id)

    def test_position_change_changes_plan_id(self) -> None:
        p1 = self._timeline_path(position={"x": 540, "y": 1450})
        p2 = self._timeline_path(position={"x": 300, "y": 1450})
        plan1 = build_overlay_plan(p1, self.config)
        plan2 = build_overlay_plan(p2, self.config)
        self.assertNotEqual(plan1.plan_id, plan2.plan_id)

    def test_z_index_change_changes_plan_id(self) -> None:
        p1 = self._timeline_path(extra_metadata={"z_index": 100})
        p2 = self._timeline_path(extra_metadata={"z_index": 200})
        plan1 = build_overlay_plan(p1, self.config)
        plan2 = build_overlay_plan(p2, self.config)
        self.assertNotEqual(plan1.plan_id, plan2.plan_id)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class JsonTests(OverlayPlanTempTestCase):
    def test_round_trip_preserved(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        rebuilt = overlay_plan_from_dict(overlay_plan_to_dict(plan))
        self.assertEqual(overlay_plan_to_dict(rebuilt), overlay_plan_to_dict(plan))

    def test_atomic_output_no_tmp_left_behind(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        out = self.temp_dir / "overlay_plan.json"
        save_overlay_plan(plan, out)
        self.assertTrue(out.is_file())
        self.assertFalse((self.temp_dir / "overlay_plan.json.tmp").exists())

    def test_overwrite_refused_without_force(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        out = self.temp_dir / "overlay_plan.json"
        save_overlay_plan(plan, out)
        with self.assertRaises(OverlayPlanOutputExistsError):
            save_overlay_plan(plan, out)

    def test_force_overwrites(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        out = self.temp_dir / "overlay_plan.json"
        save_overlay_plan(plan, out)
        save_overlay_plan(plan, out, force=True)
        self.assertTrue(out.is_file())

    def test_output_same_as_source_rejected_via_cli_guard(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        from .overlay_plan_engine import main as cli_main

        with self.assertRaises(SystemExit):
            cli_main(["--timeline", str(timeline_path), "--output", str(timeline_path)])

    def test_parent_directory_created(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        out = self.temp_dir / "nested" / "dir" / "overlay_plan.json"
        save_overlay_plan(plan, out)
        self.assertTrue(out.is_file())

    def test_malformed_overlay_plan_json_fails(self) -> None:
        bad = self.temp_dir / "bad.json"
        bad.write_text("{not valid json")
        with self.assertRaises(OverlayPlanJSONError):
            load_overlay_plan(bad)

    def test_unknown_plan_metadata_preserved(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        plan.metadata["custom_field"] = {"nested": True}
        rebuilt = overlay_plan_from_dict(overlay_plan_to_dict(plan))
        self.assertEqual(rebuilt.metadata["custom_field"], {"nested": True})

    def test_deterministic_serialization(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        json1 = json.dumps(overlay_plan_to_dict(plan), sort_keys=True)
        json2 = json.dumps(overlay_plan_to_dict(plan), sort_keys=True)
        self.assertEqual(json1, json2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationTests(OverlayPlanTempTestCase):
    def test_duplicate_overlay_ids_fail(self) -> None:
        plan = OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[OverlayLayer(layer_id="layer_subtitle")],
                            overlays=[
                                PlannedOverlay(overlay_id="dup", layer_id="layer_subtitle", timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0, content="a"),
                                PlannedOverlay(overlay_id="dup", layer_id="layer_subtitle", timeline_start_seconds=1.0, timeline_end_seconds=2.0, duration_seconds=1.0, content="b"),
                            ])
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_duplicate_layer_ids_fail(self) -> None:
        plan = OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[OverlayLayer(layer_id="dup"), OverlayLayer(layer_id="dup")])
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_invalid_layer_reference_fails(self) -> None:
        plan = OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[],
                            overlays=[PlannedOverlay(overlay_id="o1", layer_id="missing", timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0, content="a")])
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_invalid_metadata_serialization_fails(self) -> None:
        plan = OverlayPlan(timeline_id="tl1", duration_seconds=10.0, layers=[OverlayLayer(layer_id="layer_subtitle")],
                            overlays=[PlannedOverlay(overlay_id="o1", layer_id="layer_subtitle", timeline_start_seconds=0.0, timeline_end_seconds=1.0, duration_seconds=1.0, content="a", metadata={"bad": object()})])
        result = validate_overlay_plan(plan, self.config)
        self.assertFalse(result.passed)

    def test_disabled_and_enabled_counts_correct(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip("s1"), _subtitle_clip("s2", start=2.0, end=4.0, enabled=False)])])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config, include_disabled=True)
        self.assertEqual(plan.enabled_overlay_count, 1)
        self.assertEqual(plan.disabled_overlay_count, 1)

    def test_collision_count_correct(self) -> None:
        timeline = _timeline([
            _video_track(),
            _overlay_track("track_cta", [_overlay_clip("cta_1", "cta", start=0.0, end=5.0, position={"x": 500, "y": 500, "width": 100, "height": 50}, track_id="track_cta")]),
            _overlay_track("track_loc", [_overlay_clip("loc_1", "location", start=1.0, end=3.0, position={"x": 500, "y": 500, "width": 100, "height": 50}, track_id="track_loc")], order=3),
        ])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        result = validate_overlay_plan(plan, self.config)
        self.assertEqual(result.collision_count, 1)

    def test_empty_plan_validation_passes_with_warning(self) -> None:
        timeline = _timeline([_video_track()])
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        result = validate_overlay_plan(plan, self.config)
        self.assertTrue(result.passed)
        self.assertTrue(any("no_overlays_found" in w or "no active subtitle" in w for w in [w.message for w in plan.warnings]))

    def test_timeline_duration_preserved_cross_check(self) -> None:
        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])], duration=10.0)
        timeline_path = self._write_timeline(timeline)
        plan = build_overlay_plan(timeline_path, self.config)
        reloaded_timeline = timeline_engine.load_timeline(timeline_path)
        result = validate_overlay_plan(plan, self.config, timeline=reloaded_timeline)
        self.assertTrue(result.passed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(OverlayPlanTempTestCase):
    def test_timeline_and_output_works(self) -> None:
        from .overlay_plan_engine import main as cli_main

        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        out = self.temp_dir / "overlay_plan.json"
        cli_main(["--timeline", str(timeline_path), "--output", str(out)])
        self.assertTrue(out.is_file())

    def test_validate_only_writes_nothing(self) -> None:
        from .overlay_plan_engine import main as cli_main

        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        cli_main(["--timeline", str(timeline_path), "--validate-only"])
        self.assertEqual(sorted(p.name for p in self.temp_dir.iterdir()), ["timeline.json"])

    def test_json_flag_valid(self) -> None:
        import contextlib
        import io

        from .overlay_plan_engine import main as cli_main

        timeline = _timeline([_video_track(), _subtitle_track([_subtitle_clip()])])
        timeline_path = self._write_timeline(timeline)
        out = self.temp_dir / "overlay_plan.json"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cli_main(["--timeline", str(timeline_path), "--output", str(out), "--json"])
        data = json.loads(buffer.getvalue())
        self.assertIn("plan_id", data)

    def test_include_disabled_flag_works(self) -> None:
        args = parse_arguments(["--timeline", "t.json", "--output", "o.json", "--include-disabled"])
        self.assertTrue(args.include_disabled)

    def test_force_flag_works(self) -> None:
        args = parse_arguments(["--timeline", "t.json", "--output", "o.json", "--force"])
        self.assertTrue(args.force)

    def test_missing_timeline_fails(self) -> None:
        from .overlay_plan_engine import main as cli_main

        with self.assertRaises(SystemExit):
            cli_main(["--timeline", str(self.temp_dir / "does_not_exist.json"), "--output", str(self.temp_dir / "o.json")])

    def test_output_required_unless_validate_only(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--timeline", "t.json"])

    def test_expected_failure_has_no_traceback(self) -> None:
        import contextlib
        import io

        from .overlay_plan_engine import main as cli_main

        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            cli_main(["--timeline", str(self.temp_dir / "does_not_exist.json"), "--output", str(self.temp_dir / "o.json")])
        self.assertIn("[OverlayPlanEngine]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())


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

    def test_no_rendering(self) -> None:
        for forbidden in ("cv2.", "PIL.", "moviepy", "draw_text(", "render_frame("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_media_mutation(self) -> None:
        for forbidden in (".mp4", ".mov", ".wav", ".mp3"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_speech_recognition_or_whisper(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip().lower()
            self.assertFalse(stripped.startswith("import whisper"))
            self.assertFalse(stripped.startswith("from whisper"))
            self.assertFalse(stripped.startswith("import speech_recognition"))

    def test_no_font_download_or_network_code(self) -> None:
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download_font"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_automatic_asset_selection(self) -> None:
        for forbidden in ("random.choice", "glob(", "rglob(", "listdir(", "default_asset"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_does_not_import_subtitle_engine_or_integration(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .subtitle_engine"))
            self.assertFalse(stripped.startswith("from .subtitle_timeline_integration"))


if __name__ == "__main__":
    unittest.main()
