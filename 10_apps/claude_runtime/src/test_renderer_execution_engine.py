from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import overlay_plan_engine, renderer_plan_engine, timeline_engine
from .video_engine import ProcessResult
from .renderer_execution_engine import (
    PassExecutionResult,
    RendererExecutionConfig,
    RendererExecutionEngineError,
    RendererExecutionMusicError,
    RendererExecutionOutputExistsError,
    RendererExecutionPlanLoadError,
    RendererExecutionPlanMismatchError,
    RendererExecutionProbeError,
    RendererExecutionResult,
    RendererExecutionTimelineLoadError,
    RendererExecutionUnsupportedPassError,
    RendererExecutionVideoError,
    UnsafeRendererExecutionOutputError,
    execute_renderer_plan,
    load_renderer_execution_config,
    parse_arguments,
    renderer_execution_result_to_dict,
)

MODULE_PATH = Path(__file__).resolve().parent / "renderer_execution_engine.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fake ffprobe/ffmpeg runner — no real subprocess is ever invoked.
# ---------------------------------------------------------------------------


def _video_probe_json(*, duration: float = 10.0, width: int = 1080, height: int = 1920, has_audio: bool = True) -> dict:
    streams = [
        {
            "index": 0, "codec_type": "video", "codec_name": "h264", "width": width, "height": height,
            "r_frame_rate": "30/1", "avg_frame_rate": "30/1", "duration": str(duration), "pix_fmt": "yuv420p",
            "tags": {}, "disposition": {"default": 1, "attached_pic": 0},
        }
    ]
    if has_audio:
        streams.append(
            {
                "index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2,
                "duration": str(duration), "tags": {}, "disposition": {"default": 1, "attached_pic": 0},
            }
        )
    return {
        "streams": streams,
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": str(duration), "bit_rate": "8000000"},
        "chapters": [],
    }


def _music_probe_json(*, duration: float = 8.0) -> dict:
    return {
        "streams": [
            {
                "index": 0, "codec_type": "audio", "codec_name": "mp3", "sample_rate": "44100", "channels": 2,
                "duration": str(duration), "tags": {}, "disposition": {"default": 1, "attached_pic": 0},
            }
        ],
        "format": {"format_name": "mp3", "duration": str(duration), "bit_rate": "192000"},
        "chapters": [],
    }


class FakeRunner:
    """Replaces both ffprobe and ffmpeg across media_inspector.py/
    video_engine.py/music_mixer.py simultaneously. probes maps a
    RESOLVED absolute path (str) to a probe spec dict."""

    def __init__(self, probes: dict[str, dict] | None = None, *, ffmpeg_should_fail: bool = False) -> None:
        self.probes = probes or {}
        self.ffmpeg_should_fail = ffmpeg_should_fail
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, timeout: int) -> ProcessResult:
        self.calls.append(command)
        binary_name = Path(command[0]).name

        if binary_name == "ffprobe":
            path = command[-1]
            spec = self.probes.get(path)
            if spec is None:
                return ProcessResult(returncode=1, stderr=f"no such file: {path}")
            return ProcessResult(returncode=0, stdout=json.dumps(spec))

        if self.ffmpeg_should_fail:
            return ProcessResult(returncode=1, stderr="simulated ffmpeg failure")

        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4-bytes")
        return ProcessResult(returncode=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class RendererExecutionTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = load_renderer_execution_config()
        self.overlay_config = overlay_plan_engine.load_overlay_plan_config()
        self.renderer_plan_config = renderer_plan_engine.load_renderer_plan_config()

    def _video_clip(self, clip_id="clip_1", *, start=0.0, end=10.0, source_path=None) -> timeline_engine.VideoClip:
        path = source_path or (self.temp_dir / "scene.mp4")
        if source_path is None and not path.exists():
            path.write_bytes(b"x" * 16)
        return timeline_engine.VideoClip(
            clip_id=clip_id, track_id="track_video", source_path=str(path),
            start=start, end=end, duration_seconds=end - start, source_out=end - start, scene_number=1,
        )

    def _subtitle_clip(self, clip_id="sub_1", *, start=0.0, end=2.0) -> timeline_engine.OverlayClip:
        return timeline_engine.OverlayClip(
            clip_id=clip_id, track_id="track_subtitles", source_path="", overlay_type="subtitle",
            start=start, end=end, duration_seconds=end - start, source_out=end - start, content="hello there",
            metadata={"position": {"x": 540, "y": 1450}, "style": {"style_id": "default"}, "language": "en"},
        )

    def _build_timeline(self, *, with_subtitle: bool = False, duration: float = 10.0, video_path=None) -> tuple[timeline_engine.Timeline, Path]:
        video_path = video_path or (self.temp_dir / "scene.mp4")
        if not video_path.exists():
            video_path.write_bytes(b"x" * 16)
        tracks = [
            timeline_engine.TimelineTrack(
                track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
                clips=[self._video_clip(end=duration, source_path=video_path)],
            )
        ]
        if with_subtitle:
            tracks.append(
                timeline_engine.TimelineTrack(
                    track_id="track_subtitles", track_type=timeline_engine.TrackType.SUBTITLE, order=1,
                    clips=[self._subtitle_clip()],
                )
            )
        timeline = timeline_engine.Timeline(timeline_id="tl1", duration_seconds=duration, tracks=tracks)
        return timeline, video_path

    def _write_music_plan(self, music_path: Path, **overrides) -> Path:
        data = {"music_path": str(music_path), "volume": 0.2, "mode": "loop"}
        data.update(overrides)
        path = self.temp_dir / "music_plan.json"
        path.write_text(json.dumps(data))
        return path

    def _build_chain(
        self, *, with_subtitle: bool = False, with_music: bool = False, duration: float = 10.0
    ) -> dict:
        timeline, video_path = self._build_timeline(with_subtitle=with_subtitle, duration=duration)
        timeline_path = self.temp_dir / "timeline.json"
        timeline_engine.save_timeline(timeline, timeline_path)

        overlay_plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config)
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path)

        music_plan_path = None
        music_path = None
        if with_music:
            music_path = self.temp_dir / "music.mp3"
            music_path.write_bytes(b"y" * 16)
            music_plan_path = self._write_music_plan(music_path)

        renderer_plan = renderer_plan_engine.build_renderer_plan(
            timeline_path, overlay_plan_path, self.renderer_plan_config, music_plan_path=music_plan_path
        )
        renderer_plan_path = self.temp_dir / "renderer_plan.json"
        renderer_plan_engine.save_renderer_plan(renderer_plan, renderer_plan_path)

        output_path = self.temp_dir / "final.mp4"
        probes: dict[str, dict] = {str(video_path.resolve()): _video_probe_json(duration=duration)}
        if with_music:
            probes[str(music_path.resolve())] = _music_probe_json()
            intermediate_path = output_path.with_name(f"final{self.config.intermediate_filename_suffix}.mp4")
            probes[str(intermediate_path.resolve())] = _video_probe_json(duration=duration)

        return {
            "timeline": timeline, "timeline_path": timeline_path,
            "overlay_plan": overlay_plan, "overlay_plan_path": overlay_plan_path,
            "renderer_plan": renderer_plan, "renderer_plan_path": renderer_plan_path,
            "output_path": output_path, "video_path": video_path,
            "music_path": music_path, "probes": probes,
        }


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------


class DryRunTests(RendererExecutionTempTestCase):
    def test_dry_run_writes_no_file(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        self.assertFalse(chain["output_path"].exists())

    def test_dry_run_writes_no_log(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        log_path = chain["output_path"].with_name(f"final{self.config.log_filename_suffix}")
        self.assertFalse(log_path.exists())

    def test_dry_run_result_status(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        self.assertEqual(result.result, "dry_run")
        self.assertTrue(result.dry_run)

    def test_dry_run_video_pass_returns_real_command(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        video_pass = next(p for p in result.passes if p.pass_type == "video")
        self.assertEqual(video_pass.status, "dry_run")
        self.assertTrue(video_pass.command)
        self.assertIn("ffmpeg", Path(video_pass.command[0]).name)

    def test_dry_run_never_calls_runner_for_execution(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        self.assertEqual(ffmpeg_calls, [])

    def test_dry_run_still_probes_real_assets(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        ffprobe_calls = [c for c in runner.calls if Path(c[0]).name == "ffprobe"]
        self.assertEqual(len(ffprobe_calls), 1)

    def test_dry_run_music_pass_shows_request_only(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        music_pass = next(p for p in result.passes if p.pass_type == "music")
        self.assertEqual(music_pass.status, "dry_run")
        self.assertIsNone(music_pass.command)
        self.assertTrue(music_pass.warnings)

    def test_config_dry_run_default_used_when_not_specified(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        config = RendererExecutionConfig(dry_run_default=True)
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], config, output_path=chain["output_path"], runner=runner)
        self.assertTrue(result.dry_run)

    def test_config_dry_run_default_false_executes(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        config = RendererExecutionConfig(dry_run_default=False)
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], config, output_path=chain["output_path"], runner=runner)
        self.assertFalse(result.dry_run)
        self.assertTrue(chain["output_path"].exists())


# ---------------------------------------------------------------------------
# Video-only execution
# ---------------------------------------------------------------------------


class VideoOnlyExecutionTests(RendererExecutionTempTestCase):
    def test_video_only_execution_writes_output(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertTrue(chain["output_path"].is_file())
        self.assertEqual(result.result, "success")

    def test_video_only_final_output_path_is_output(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(result.final_output_path, str(chain["output_path"]))

    def test_video_only_no_intermediate_file_created(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        intermediate_path = chain["output_path"].with_name(f"final{self.config.intermediate_filename_suffix}.mp4")
        self.assertFalse(intermediate_path.exists())

    def test_video_pass_status_executed(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        video_pass = next(p for p in result.passes if p.pass_type == "video")
        self.assertEqual(video_pass.status, "executed")
        self.assertIsInstance(video_pass.duration_seconds, float)

    def test_final_encode_pass_verified(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        final_pass = next(p for p in result.passes if p.pass_type == "final_encode")
        self.assertEqual(final_pass.status, "verified")

    def test_video_ffmpeg_failure_raises(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"], ffmpeg_should_fail=True)
        with self.assertRaises(RendererExecutionVideoError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)

    def test_missing_probe_raises_probe_error(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner({})  # no probe entries at all
        with self.assertRaises(RendererExecutionProbeError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)


# ---------------------------------------------------------------------------
# Video + music execution
# ---------------------------------------------------------------------------


class VideoMusicExecutionTests(RendererExecutionTempTestCase):
    def test_video_and_music_execution_writes_final_output(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertTrue(chain["output_path"].is_file())
        self.assertEqual(result.result, "success")

    def test_video_pass_writes_intermediate_path(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        video_pass = next(p for p in result.passes if p.pass_type == "video")
        self.assertIn(self.config.intermediate_filename_suffix, video_pass.output_path)

    def test_music_pass_consumes_video_pass_output(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        self.assertEqual(len(ffmpeg_calls), 2)  # video concat + music mix

    def test_music_pass_status_executed(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        music_pass = next(p for p in result.passes if p.pass_type == "music")
        self.assertEqual(music_pass.status, "executed")

    def test_music_request_volume_and_mode_applied(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        music_pass = next(p for p in result.passes if p.pass_type == "music")
        self.assertTrue(music_pass.command)

    def test_music_ffmpeg_failure_raises(self) -> None:
        chain = self._build_chain(with_music=True)
        probes = dict(chain["probes"])
        runner = FakeRunner(probes)
        # Force the second (music) ffmpeg call to fail by making the runner
        # fail unconditionally once the video pass has already written its
        # own intermediate file via a first, successful pass. Simplest:
        # remove the music probe so Media Inspector's own inspection fails,
        # which mix_music() surfaces as a MusicMixerError.
        del probes[str(chain["music_path"].resolve())]
        with self.assertRaises(RendererExecutionMusicError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)

    def test_final_output_recorded_after_music_pass(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(result.final_output_path, str(chain["output_path"]))


# ---------------------------------------------------------------------------
# Subtitle / overlay pass handling
# ---------------------------------------------------------------------------


class UnsupportedPassTests(RendererExecutionTempTestCase):
    def test_subtitle_pass_fails_loud_by_default(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionUnsupportedPassError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)

    def test_subtitle_pass_fails_loud_in_execute_mode_too(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionUnsupportedPassError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)

    def test_no_output_written_when_subtitle_pass_blocks(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionUnsupportedPassError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertFalse(chain["output_path"].exists())

    def test_allow_partial_execution_skips_subtitle(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, allow_partial_execution=True, runner=runner)
        subtitle_pass = next(p for p in result.passes if p.pass_type == "subtitle")
        self.assertEqual(subtitle_pass.status, "skipped")

    def test_allow_partial_execution_still_executes_video(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, allow_partial_execution=True, runner=runner)
        self.assertTrue(chain["output_path"].is_file())
        self.assertEqual(result.result, "partial")

    def test_skipped_pass_warning_recorded(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, allow_partial_execution=True, runner=runner)
        self.assertTrue(any("skipped" in w for w in result.warnings))

    def test_subtitle_never_fabricated_in_output(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, allow_partial_execution=True, runner=runner)
        subtitle_pass = next(p for p in result.passes if p.pass_type == "subtitle")
        self.assertIsNone(subtitle_pass.command)
        self.assertIsNone(subtitle_pass.output_path)

    def test_config_allow_partial_execution_default_used(self) -> None:
        chain = self._build_chain(with_subtitle=True)
        runner = FakeRunner(chain["probes"])
        config = RendererExecutionConfig(allow_partial_execution_default=True)
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(result.result, "partial")


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------


class LoadingValidationTests(RendererExecutionTempTestCase):
    def test_malformed_renderer_plan_fails(self) -> None:
        chain = self._build_chain()
        chain["renderer_plan_path"].write_text("{not valid json")
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionPlanLoadError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)

    def test_malformed_timeline_fails(self) -> None:
        chain = self._build_chain()
        chain["timeline_path"].write_text("{not valid json")
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionTimelineLoadError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)

    def test_timeline_id_mismatch_fails(self) -> None:
        chain = self._build_chain()
        other_timeline, _p = self._build_timeline()
        other_timeline.timeline_id = "different"
        other_timeline_path = self.temp_dir / "other_timeline.json"
        timeline_engine.save_timeline(other_timeline, other_timeline_path)
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionPlanMismatchError):
            execute_renderer_plan(chain["renderer_plan_path"], other_timeline_path, self.config, output_path=chain["output_path"], dry_run=True, runner=runner)

    def test_source_files_unchanged(self) -> None:
        import hashlib

        chain = self._build_chain()
        before_plan = hashlib.sha256(chain["renderer_plan_path"].read_bytes()).hexdigest()
        before_timeline = hashlib.sha256(chain["timeline_path"].read_bytes()).hexdigest()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(hashlib.sha256(chain["renderer_plan_path"].read_bytes()).hexdigest(), before_plan)
        self.assertEqual(hashlib.sha256(chain["timeline_path"].read_bytes()).hexdigest(), before_timeline)

    def test_invalid_renderer_plan_json_shape_fails(self) -> None:
        chain = self._build_chain()
        chain["renderer_plan_path"].write_text(json.dumps({"not": "a real plan"}))
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionPlanLoadError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)


# ---------------------------------------------------------------------------
# Output / overwrite protection
# ---------------------------------------------------------------------------


class OutputProtectionTests(RendererExecutionTempTestCase):
    def test_overwrite_refused_without_force(self) -> None:
        chain = self._build_chain()
        chain["output_path"].write_bytes(b"existing")
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionOutputExistsError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)

    def test_force_overwrites(self) -> None:
        chain = self._build_chain()
        chain["output_path"].write_bytes(b"existing")
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, force=True, runner=runner)
        self.assertEqual(result.result, "success")
        self.assertEqual(chain["output_path"].read_bytes(), b"fake-mp4-bytes")

    def test_dry_run_ignores_existing_output(self) -> None:
        chain = self._build_chain()
        chain["output_path"].write_bytes(b"existing")
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)
        self.assertEqual(result.result, "dry_run")

    def test_output_same_as_renderer_plan_rejected(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(UnsafeRendererExecutionOutputError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["renderer_plan_path"], dry_run=True, runner=runner)

    def test_output_same_as_timeline_rejected(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(UnsafeRendererExecutionOutputError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["timeline_path"], dry_run=True, runner=runner)


# ---------------------------------------------------------------------------
# Execution log
# ---------------------------------------------------------------------------


class ExecutionLogTests(RendererExecutionTempTestCase):
    def test_log_written_on_success(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        log_path = chain["output_path"].with_name(f"final{self.config.log_filename_suffix}")
        self.assertTrue(log_path.is_file())

    def test_log_atomic_no_tmp_left_behind(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        log_path = chain["output_path"].with_name(f"final{self.config.log_filename_suffix}")
        self.assertFalse(Path(str(log_path) + ".tmp").exists())

    def test_log_shape_has_all_fields(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        log_path = chain["output_path"].with_name(f"final{self.config.log_filename_suffix}")
        payload = json.loads(log_path.read_text())
        for field_name in ("renderer_plan_id", "timeline_id", "dry_run", "started_at", "finished_at", "passes", "final_output_path", "warnings", "errors", "result"):
            self.assertIn(field_name, payload)

    def test_log_written_on_failure(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"], ffmpeg_should_fail=True)
        with self.assertRaises(RendererExecutionVideoError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        log_path = chain["output_path"].with_name(f"final{self.config.log_filename_suffix}")
        self.assertTrue(log_path.is_file())
        payload = json.loads(log_path.read_text())
        self.assertEqual(payload["result"], "failed")
        self.assertTrue(payload["errors"])

    def test_renderer_execution_result_to_dict_matches_log(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        log_path = chain["output_path"].with_name(f"final{self.config.log_filename_suffix}")
        payload = json.loads(log_path.read_text())
        self.assertEqual(payload, renderer_execution_result_to_dict(result))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class CliTests(RendererExecutionTempTestCase):
    def test_default_is_dry_run(self) -> None:
        args = parse_arguments(["--renderer-plan", "r.json", "--timeline", "t.json", "--output", "o.mp4"])
        self.assertFalse(args.execute)

    def test_execute_flag_accepted(self) -> None:
        args = parse_arguments(["--renderer-plan", "r.json", "--timeline", "t.json", "--output", "o.mp4", "--execute"])
        self.assertTrue(args.execute)

    def test_allow_partial_execution_flag_accepted(self) -> None:
        args = parse_arguments(["--renderer-plan", "r.json", "--timeline", "t.json", "--output", "o.mp4", "--allow-partial-execution"])
        self.assertTrue(args.allow_partial_execution)

    def test_force_flag_accepted(self) -> None:
        args = parse_arguments(["--renderer-plan", "r.json", "--timeline", "t.json", "--output", "o.mp4", "--force"])
        self.assertTrue(args.force)

    def test_cli_dry_run_without_ffprobe_fails_cleanly(self) -> None:
        # main() never has a way to inject a FakeRunner (by design --
        # that's only for direct Python API use in tests), so it always
        # uses each delegated engine's own real default_runner. In this
        # dev environment ffmpeg/ffprobe is not installed, so even
        # --dry-run (which still probes real, pre-existing source
        # assets) must fail -- but cleanly, via RendererExecutionProbeError,
        # never a raw traceback.
        import contextlib
        import io

        from .renderer_execution_engine import main as cli_main

        chain = self._build_chain()
        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            cli_main(["--renderer-plan", str(chain["renderer_plan_path"]), "--timeline", str(chain["timeline_path"]), "--output", str(chain["output_path"])])
        self.assertFalse(chain["output_path"].exists())
        self.assertIn("[RendererExecutionEngine]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())

    def test_cli_missing_renderer_plan_fails_no_traceback(self) -> None:
        import contextlib
        import io

        from .renderer_execution_engine import main as cli_main

        buffer = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buffer):
            cli_main(["--renderer-plan", str(self.temp_dir / "missing.json"), "--timeline", str(self.temp_dir / "t.json"), "--output", str(self.temp_dir / "o.mp4")])
        self.assertIn("[RendererExecutionEngine]", buffer.getvalue())
        self.assertNotIn("Traceback", buffer.getvalue())

    def test_cli_requires_all_three_paths(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--renderer-plan", "r.json", "--timeline", "t.json"])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_loads_from_repo(self) -> None:
        config = load_renderer_execution_config()
        self.assertTrue(config.dry_run_default)
        self.assertFalse(config.allow_partial_execution_default)

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(Exception):
            load_renderer_execution_config("/nonexistent/renderer_execution.yaml")

    def test_config_defaults(self) -> None:
        config = RendererExecutionConfig()
        self.assertEqual(config.intermediate_filename_suffix, "_video_only")
        self.assertEqual(config.log_filename_suffix, "_renderer_execution_log.json")


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

    def test_no_font_download(self) -> None:
        for forbidden in (".ttf", ".otf", ".woff", "font_path", "FontFile"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_network_or_download_code(self) -> None:
        for forbidden in ("requests.", "urllib.request", "http.client", "socket.", "download("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_direct_subprocess_import(self) -> None:
        # This module must never construct/run a raw ffmpeg command
        # itself -- only through video_engine.py/music_mixer.py/
        # media_inspector.py's own runners.
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import subprocess"))
            self.assertFalse(stripped.startswith("from subprocess"))
        self.assertNotIn("subprocess.run(", MODULE_SOURCE)
        self.assertNotIn("subprocess.Popen(", MODULE_SOURCE)

    def test_no_raw_ffmpeg_binary_config(self) -> None:
        # ffmpeg/ffprobe binary configuration stays owned by
        # video_engine.py/media_inspector.py/music_mixer.py's own
        # configs -- never duplicated here.
        self.assertNotIn("ffmpeg_binary", MODULE_SOURCE)
        self.assertNotIn("ffprobe_binary", MODULE_SOURCE)

    def test_only_imports_expected_engines(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from .subtitle_engine"))
            self.assertFalse(stripped.startswith("from .subtitle_timeline_integration"))
            self.assertFalse(stripped.startswith("from .reel_builder"))

    def test_all_real_commands_traced_to_delegated_engines(self) -> None:
        # Every PassExecutionResult.command must come from
        # video_engine.py's build_*_concat_command()/result.command or
        # music_mixer's result.command -- never a locally constructed list.
        self.assertNotIn('command = ["ffmpeg"', MODULE_SOURCE)
        self.assertNotIn("command = ['ffmpeg'", MODULE_SOURCE)


# ---------------------------------------------------------------------------
# Additional loading / cross-validation coverage
# ---------------------------------------------------------------------------


class MoreLoadingValidationTests(RendererExecutionTempTestCase):
    def test_invalid_timeline_fails(self) -> None:
        chain = self._build_chain()
        bad_timeline = timeline_engine.Timeline(timeline_id="tl1", duration_seconds=0.0, tracks=[])
        timeline_engine.save_timeline(bad_timeline, chain["timeline_path"], force=True)
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionTimelineLoadError):
            execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)

    def test_missing_renderer_plan_file_fails(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionPlanLoadError):
            execute_renderer_plan(self.temp_dir / "does_not_exist.json", chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=True, runner=runner)

    def test_missing_timeline_file_fails(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        with self.assertRaises(RendererExecutionTimelineLoadError):
            execute_renderer_plan(chain["renderer_plan_path"], self.temp_dir / "does_not_exist.json", self.config, output_path=chain["output_path"], dry_run=True, runner=runner)

    def test_overlay_plan_source_unchanged(self) -> None:
        import hashlib

        chain = self._build_chain()
        before = hashlib.sha256(chain["overlay_plan_path"].read_bytes()).hexdigest()
        runner = FakeRunner(chain["probes"])
        execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(hashlib.sha256(chain["overlay_plan_path"].read_bytes()).hexdigest(), before)


# ---------------------------------------------------------------------------
# PassExecutionResult / RendererExecutionResult field content
# ---------------------------------------------------------------------------


class ResultFieldTests(RendererExecutionTempTestCase):
    def test_result_renderer_plan_id_matches_plan(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(result.renderer_plan_id, chain["renderer_plan"].renderer_plan_id)

    def test_result_timeline_id_matches_timeline(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(result.timeline_id, chain["timeline"].timeline_id)

    def test_started_at_before_finished_at(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertLessEqual(result.started_at, result.finished_at)

    def test_passes_recorded_in_plan_order(self) -> None:
        chain = self._build_chain(with_music=True)
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual([p.pass_type for p in result.passes], [pp.pass_type for pp in chain["renderer_plan"].passes])

    def test_no_warnings_or_errors_on_clean_success(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        self.assertEqual(result.errors, [])

    def test_video_pass_return_code_zero_on_success(self) -> None:
        chain = self._build_chain()
        runner = FakeRunner(chain["probes"])
        result = execute_renderer_plan(chain["renderer_plan_path"], chain["timeline_path"], self.config, output_path=chain["output_path"], dry_run=False, runner=runner)
        video_pass = next(p for p in result.passes if p.pass_type == "video")
        self.assertEqual(video_pass.return_code, 0)

    def test_pass_execution_result_defaults(self) -> None:
        result = PassExecutionResult()
        self.assertEqual(result.status, "")
        self.assertEqual(result.warnings, [])
        self.assertIsNone(result.error)

    def test_renderer_execution_result_defaults(self) -> None:
        result = RendererExecutionResult()
        self.assertTrue(result.dry_run)
        self.assertEqual(result.passes, [])


# ---------------------------------------------------------------------------
# Multiple video clips
# ---------------------------------------------------------------------------


class MultiClipTests(RendererExecutionTempTestCase):
    def test_two_video_clips_both_probed(self) -> None:
        video_path_1 = self.temp_dir / "scene1.mp4"
        video_path_1.write_bytes(b"x" * 16)
        video_path_2 = self.temp_dir / "scene2.mp4"
        video_path_2.write_bytes(b"y" * 16)

        video_track = timeline_engine.TimelineTrack(
            track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0,
            clips=[
                timeline_engine.VideoClip(clip_id="c1", track_id="track_video", source_path=str(video_path_1), start=0.0, end=5.0, duration_seconds=5.0, source_out=5.0, scene_number=1),
                timeline_engine.VideoClip(clip_id="c2", track_id="track_video", source_path=str(video_path_2), start=5.0, end=10.0, duration_seconds=5.0, source_out=5.0, scene_number=2),
            ],
        )
        timeline = timeline_engine.Timeline(timeline_id="tl1", duration_seconds=10.0, tracks=[video_track])
        timeline_path = self.temp_dir / "timeline.json"
        timeline_engine.save_timeline(timeline, timeline_path)

        overlay_plan = overlay_plan_engine.build_overlay_plan(timeline_path, self.overlay_config)
        overlay_plan_path = self.temp_dir / "overlay_plan.json"
        overlay_plan_engine.save_overlay_plan(overlay_plan, overlay_plan_path)

        renderer_plan = renderer_plan_engine.build_renderer_plan(timeline_path, overlay_plan_path, self.renderer_plan_config)
        renderer_plan_path = self.temp_dir / "renderer_plan.json"
        renderer_plan_engine.save_renderer_plan(renderer_plan, renderer_plan_path)

        probes = {
            str(video_path_1.resolve()): _video_probe_json(duration=5.0),
            str(video_path_2.resolve()): _video_probe_json(duration=5.0),
        }
        runner = FakeRunner(probes)
        output_path = self.temp_dir / "final.mp4"
        execute_renderer_plan(renderer_plan_path, timeline_path, self.config, output_path=output_path, dry_run=False, runner=runner)

        ffprobe_calls = [c for c in runner.calls if Path(c[0]).name == "ffprobe"]
        self.assertEqual(len(ffprobe_calls), 2)


# ---------------------------------------------------------------------------
# Config file content
# ---------------------------------------------------------------------------


class MoreConfigTests(unittest.TestCase):
    def test_repo_config_does_not_own_ffmpeg_binary_settings(self) -> None:
        content = (Path(__file__).resolve().parents[1] / "config" / "video" / "renderer_execution.yaml").read_text()
        self.assertNotIn("ffmpeg_binary", content)
        self.assertNotIn("ffprobe_binary", content)

    def test_config_output_settings(self) -> None:
        config = load_renderer_execution_config()
        self.assertTrue(config.overwrite_requires_force)
        self.assertTrue(config.atomic_write)


if __name__ == "__main__":
    unittest.main()
