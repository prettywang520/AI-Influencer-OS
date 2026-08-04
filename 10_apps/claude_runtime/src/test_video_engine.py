from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .video_engine import (
    DISALLOWED_ACTIONS,
    PLAN_LOSSLESS_COPY,
    PLAN_NORMALIZED_RENDER,
    PLAN_REJECTED,
    AudioStreamSpec,
    ConcatManifestError,
    DuplicateVideoInputError,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    FFmpegTimeoutError,
    ProcessResult,
    UnsafeVideoOutputError,
    VideoEngine,
    VideoEngineConfig,
    VideoEngineConfigError,
    VideoInput,
    VideoInputError,
    VideoOutputExistsError,
    VideoOutputSpec,
    VideoOutputVerificationError,
    VideoPlanRejectedError,
    VideoStreamSpec,
    default_engine_config_path,
    default_runner,
    load_engine_config,
)

MODULE_PATH = Path(__file__).resolve().parent / "video_engine.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fake runner — no real ffmpeg/ffprobe is ever invoked in these tests.
# ---------------------------------------------------------------------------


class FakeRunner:
    def __init__(
        self,
        *,
        should_fail: bool = False,
        write_output: bool = True,
        output_bytes: bytes = b"fake-mp4-bytes",
        raise_exc: Exception | None = None,
    ) -> None:
        self.should_fail = should_fail
        self.write_output = write_output
        self.output_bytes = output_bytes
        self.raise_exc = raise_exc
        self.calls: list[list[str]] = []
        self.timeouts: list[int] = []

    def __call__(self, command: list[str], *, timeout: int) -> ProcessResult:
        self.calls.append(command)
        self.timeouts.append(timeout)

        if self.raise_exc is not None:
            raise self.raise_exc

        if self.should_fail:
            return ProcessResult(returncode=1, stderr="simulated ffmpeg failure")

        if self.write_output:
            output_path = Path(command[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(self.output_bytes)

        return ProcessResult(returncode=0, stdout="", stderr="")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _video_spec(**overrides) -> VideoStreamSpec:
    defaults = dict(width=1080, height=1920, fps=30.0, codec_name="h264", pixel_format="yuv420p")
    defaults.update(overrides)
    return VideoStreamSpec(**defaults)


def _audio_spec(**overrides) -> AudioStreamSpec:
    defaults = dict(present=True, codec_name="aac", sample_rate=48000, channels=2)
    defaults.update(overrides)
    return AudioStreamSpec(**defaults)


class VideoEngineTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = VideoEngineConfig()

    def _write_clip(self, name: str, *, size: int = 100) -> Path:
        path = self.temp_dir / name
        path.write_bytes(b"x" * size)
        return path

    def _input(self, name: str, **overrides) -> VideoInput:
        path = self._write_clip(name)
        video = overrides.pop("video", _video_spec())
        audio = overrides.pop("audio", _audio_spec())
        return VideoInput(path=path, video=video, audio=audio, duration_seconds=5.0)

    def _output(self, name: str = "output.mp4", **overrides) -> VideoOutputSpec:
        return VideoOutputSpec(path=self.temp_dir / name, **overrides)

    def _engine(self, *, runner: FakeRunner | None = None, config: VideoEngineConfig | None = None) -> VideoEngine:
        return VideoEngine(config or self.config, runner=runner or FakeRunner())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_path_exists(self) -> None:
        self.assertTrue(default_engine_config_path().is_file())

    def test_config_loads_with_expected_defaults(self) -> None:
        config = load_engine_config()
        self.assertEqual(config.ffmpeg_binary, "ffmpeg")
        self.assertEqual(config.ffmpeg_timeout_seconds, 300)
        self.assertEqual(config.ffprobe_binary, "ffprobe")
        self.assertEqual(config.ffprobe_timeout_seconds, 30)
        self.assertTrue(config.faststart)
        self.assertTrue(config.normalize_when_needed)

    def test_missing_config_fails_clearly(self) -> None:
        with self.assertRaises(VideoEngineConfigError):
            load_engine_config("/nonexistent/path/engine.yaml")

    def test_executable_path_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "engine.yaml"
            config_path.write_text(
                "ffmpeg:\n  binary: /custom/ffmpeg\n", encoding="utf-8"
            )
            config = load_engine_config(config_path)
            self.assertEqual(config.ffmpeg_binary, "/custom/ffmpeg")

    def test_timeout_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "engine.yaml"
            config_path.write_text(
                "ffmpeg:\n  timeout_seconds: 42\n", encoding="utf-8"
            )
            config = load_engine_config(config_path)
            self.assertEqual(config.ffmpeg_timeout_seconds, 42)

    def test_no_hardcoded_override_of_configured_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "engine.yaml"
            config_path.write_text(
                "video:\n  default_codec: prores\n  default_crf: 5\n",
                encoding="utf-8",
            )
            config = load_engine_config(config_path)
            self.assertEqual(config.default_video_codec, "prores")
            self.assertEqual(config.default_crf, 5)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RunnerTests(VideoEngineTempTestCase):
    def test_command_is_a_list(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan([clip], output)
        engine.execute_plan(plan)
        self.assertIsInstance(engine.runner.calls[0], list)

    def test_shell_true_never_used(self) -> None:
        self.assertNotIn("shell=True", MODULE_SOURCE)

    def test_timeout_applied(self) -> None:
        config = VideoEngineConfig(ffmpeg_timeout_seconds=77)
        runner = FakeRunner()
        engine = VideoEngine(config, runner=runner)
        clip = self._input("a.mp4")
        output = self._output()
        plan = engine.build_concat_plan([clip], output)
        engine.execute_plan(plan)
        self.assertEqual(runner.timeouts[0], 77)

    def test_non_zero_return_raises_ffmpeg_execution_error(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine(runner=FakeRunner(should_fail=True))
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(FFmpegExecutionError):
            engine.execute_plan(plan)

    def test_timeout_raises_ffmpeg_timeout_error(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        runner = FakeRunner(raise_exc=FFmpegTimeoutError("simulated timeout"))
        engine = self._engine(runner=runner)
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(FFmpegTimeoutError):
            engine.execute_plan(plan)

    def test_missing_binary_raises_ffmpeg_not_found_error(self) -> None:
        with self.assertRaises(FFmpegNotFoundError):
            default_runner(["/definitely/not/a/real/binary/ffmpeg_xyz"], timeout=5)

    def test_no_automatic_retry_on_failure(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        runner = FakeRunner(should_fail=True)
        engine = self._engine(runner=runner)
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(FFmpegExecutionError):
            engine.execute_plan(plan)
        self.assertEqual(len(runner.calls), 1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


class PathTests(VideoEngineTempTestCase):
    def test_missing_input_rejected(self) -> None:
        missing = VideoInput(path=self.temp_dir / "missing.mp4")
        output = self._output()
        engine = self._engine()
        with self.assertRaises(VideoInputError):
            engine.validate_inputs([missing], output)

    def test_directory_input_rejected(self) -> None:
        directory_input = VideoInput(path=self.temp_dir)
        output = self._output()
        engine = self._engine()
        with self.assertRaises(VideoInputError):
            engine.validate_inputs([directory_input], output)

    def test_zero_byte_input_rejected(self) -> None:
        path = self._write_clip("empty.mp4", size=0)
        zero_input = VideoInput(path=path)
        output = self._output()
        engine = self._engine()
        with self.assertRaises(VideoInputError):
            engine.validate_inputs([zero_input], output)

    def test_duplicate_resolved_input_rejected(self) -> None:
        path = self._write_clip("clip.mp4")
        differently_spelled = self.temp_dir / "." / "clip.mp4"
        inputs = [VideoInput(path=path), VideoInput(path=differently_spelled)]
        output = self._output()
        engine = self._engine()
        with self.assertRaises(DuplicateVideoInputError):
            engine.validate_inputs(inputs, output)

    def test_output_equal_to_input_rejected(self) -> None:
        path = self._write_clip("clip.mp4")
        output = VideoOutputSpec(path=path)
        engine = self._engine()
        with self.assertRaises(UnsafeVideoOutputError):
            engine.validate_inputs([VideoInput(path=path)], output)

    def test_existing_output_refused_without_force(self) -> None:
        clip = self._input("a.mp4")
        output_path = self.temp_dir / "output.mp4"
        output_path.write_bytes(b"already-here")
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(VideoOutputExistsError):
            engine.execute_plan(plan, force=False)
        self.assertEqual(output_path.read_bytes(), b"already-here")

    def test_force_adds_dash_y(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan([clip], output)
        manifest_path = self.temp_dir / "manifest.txt"
        command = engine.build_lossless_concat_command(plan, manifest_path, force=True)
        self.assertIn("-y", command)
        self.assertNotIn("-n", command)

    def test_spaces_and_unicode_paths_work(self) -> None:
        path = self.temp_dir / "clip with spaces and 日本語.mp4"
        path.write_bytes(b"x" * 50)
        clip = VideoInput(path=path, video=_video_spec(), audio=_audio_spec())
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan([clip], output)
        result = engine.execute_plan(plan)
        self.assertTrue(result.output_exists)
        self.assertIn(str(path), result.input_files)

    def test_manifest_escaping_is_correct(self) -> None:
        path = self.temp_dir / "clip'with'quotes.mp4"
        path.write_bytes(b"x" * 50)
        clip = VideoInput(path=path)
        manifest_path = self.temp_dir / "manifest.txt"
        engine = self._engine()
        engine.write_concat_manifest([clip], manifest_path)
        content = manifest_path.read_text(encoding="utf-8")
        self.assertIn("clip'\\''with'\\''quotes.mp4", content)

    def test_manifest_path_equal_to_input_rejected(self) -> None:
        path = self._write_clip("clip.mp4")
        clip = VideoInput(path=path)
        engine = self._engine()
        with self.assertRaises(ConcatManifestError):
            engine.write_concat_manifest([clip], path)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlanningTests(VideoEngineTempTestCase):
    def test_fully_compatible_clips_choose_lossless_copy(self) -> None:
        clips = [self._input(f"c{i}.mp4") for i in range(3)]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_LOSSLESS_COPY)
        self.assertEqual(plan.reasons, ["compatible_for_stream_copy"])

    def test_resolution_mismatch_chooses_normalized_render(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(width=1080, height=1920)),
            self._input("c1.mp4", video=_video_spec(width=720, height=1280)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_NORMALIZED_RENDER)
        self.assertIn("resolution_mismatch", plan.reasons)

    def test_fps_mismatch_chooses_normalized_render(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(fps=30.0)),
            self._input("c1.mp4", video=_video_spec(fps=24.0)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_NORMALIZED_RENDER)
        self.assertIn("fps_mismatch", plan.reasons)

    def test_fps_within_tolerance_is_not_a_mismatch(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(fps=30.0)),
            self._input("c1.mp4", video=_video_spec(fps=30.01)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_LOSSLESS_COPY)

    def test_codec_mismatch_chooses_normalized_render(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(codec_name="h264")),
            self._input("c1.mp4", video=_video_spec(codec_name="hevc")),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_NORMALIZED_RENDER)
        self.assertIn("codec_mismatch", plan.reasons)

    def test_pixel_format_mismatch_chooses_normalized_render(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(pixel_format="yuv420p")),
            self._input("c1.mp4", video=_video_spec(pixel_format="yuv422p")),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_NORMALIZED_RENDER)
        self.assertIn("pixel_format_mismatch", plan.reasons)

    def test_normalization_disabled_causes_rejected(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(width=1080, height=1920)),
            self._input("c1.mp4", video=_video_spec(width=720, height=1280)),
        ]
        output = self._output()
        config = VideoEngineConfig(normalize_when_needed=False)
        engine = self._engine(config=config)
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_REJECTED)
        self.assertIn("resolution_mismatch", plan.reasons)
        self.assertIn("normalization_disabled", plan.reasons)

    def test_mixed_audio_presence_fails_clearly(self) -> None:
        clips = [
            self._input("c0.mp4", audio=_audio_spec(present=True)),
            self._input("c1.mp4", audio=_audio_spec(present=False)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_REJECTED)
        self.assertEqual(plan.reasons, ["audio_presence_mismatch"])

    def test_uniform_missing_audio_is_not_a_mismatch(self) -> None:
        clips = [
            self._input("c0.mp4", audio=_audio_spec(present=False, codec_name=None)),
            self._input("c1.mp4", audio=_audio_spec(present=False, codec_name=None)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_LOSSLESS_COPY)

    def test_audio_codec_mismatch_chooses_normalized_render(self) -> None:
        clips = [
            self._input("c0.mp4", audio=_audio_spec(present=True, codec_name="aac")),
            self._input("c1.mp4", audio=_audio_spec(present=True, codec_name="mp3")),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_NORMALIZED_RENDER)
        self.assertIn("audio_codec_mismatch", plan.reasons)

    def test_plan_reasons_are_deterministic(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(width=1080, height=1920)),
            self._input("c1.mp4", video=_video_spec(width=720, height=1280)),
        ]
        output = self._output()
        engine = self._engine()
        plan_a = engine.build_concat_plan(clips, output)
        plan_b = engine.build_concat_plan(clips, output)
        self.assertEqual(plan_a.plan_type, plan_b.plan_type)
        self.assertEqual(plan_a.reasons, plan_b.reasons)

    def test_execute_plan_rejects_a_rejected_plan(self) -> None:
        clips = [
            self._input("c0.mp4", audio=_audio_spec(present=True)),
            self._input("c1.mp4", audio=_audio_spec(present=False)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        with self.assertRaises(VideoPlanRejectedError):
            engine.execute_plan(plan)
        self.assertEqual(engine.runner.calls, [])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class CommandTests(VideoEngineTempTestCase):
    def test_lossless_command_uses_concat_demuxer_and_copy(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan([clip], output)
        command = engine.build_lossless_concat_command(plan, self.temp_dir / "manifest.txt", force=False)
        self.assertIn("-c", command)
        self.assertIn("copy", command)
        self.assertIn("-f", command)
        self.assertIn("concat", command)
        self.assertNotIn("-filter_complex", command)

    def test_normalized_command_includes_scale_fps_pixel_format(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(width=1080, height=1920)),
            self._input("c1.mp4", video=_video_spec(width=720, height=1280)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        command = engine.build_normalized_concat_command(plan, force=False)
        filter_complex = command[command.index("-filter_complex") + 1]
        self.assertIn("scale=", filter_complex)
        self.assertIn("fps=", filter_complex)
        self.assertIn("format=", filter_complex)

    def test_configured_codec_crf_preset_bitrate_are_used(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(codec_name="h264")),
            self._input("c1.mp4", video=_video_spec(codec_name="hevc")),
        ]
        output = self._output()
        config = VideoEngineConfig(
            default_video_codec="libx265", default_crf=23, default_preset="fast", default_video_bitrate="5M"
        )
        engine = self._engine(config=config)
        plan = engine.build_concat_plan(clips, output)
        command = engine.build_normalized_concat_command(plan, force=False)
        self.assertIn("libx265", command)
        self.assertIn("23", command)
        self.assertIn("fast", command)
        self.assertIn("5M", command)

    def test_faststart_included_when_enabled(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        config = VideoEngineConfig(faststart=True)
        engine = self._engine(config=config)
        plan = engine.build_concat_plan([clip], output)
        command = engine.build_lossless_concat_command(plan, self.temp_dir / "manifest.txt", force=False)
        self.assertIn("-movflags", command)
        self.assertIn("+faststart", command)

    def test_faststart_excluded_when_disabled(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        config = VideoEngineConfig(faststart=False)
        engine = self._engine(config=config)
        plan = engine.build_concat_plan([clip], output)
        command = engine.build_lossless_concat_command(plan, self.temp_dir / "manifest.txt", force=False)
        self.assertNotIn("-movflags", command)

    def test_output_path_appears_exactly_once(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(width=1080, height=1920)),
            self._input("c1.mp4", video=_video_spec(width=720, height=1280)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        command = engine.build_normalized_concat_command(plan, force=False)
        self.assertEqual(command.count(str(output.path)), 1)

    def test_inputs_remain_in_correct_order(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(width=1080, height=1920)),
            self._input("c1.mp4", video=_video_spec(width=720, height=1280)),
            self._input("c2.mp4", video=_video_spec(width=480, height=854)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        command = engine.build_normalized_concat_command(plan, force=False)
        indices = [command.index(str(c.path)) for c in clips]
        self.assertEqual(indices, sorted(indices))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionTests(VideoEngineTempTestCase):
    def test_successful_execution_verifies_non_zero_output(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan([clip], output)
        result = engine.execute_plan(plan)
        self.assertTrue(result.output_exists)
        self.assertGreater(result.output_size_bytes, 0)
        self.assertEqual(result.return_code, 0)
        self.assertIsNone(result.error)

    def test_missing_output_after_success_fails(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine(runner=FakeRunner(write_output=False))
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(VideoOutputVerificationError):
            engine.execute_plan(plan)

    def test_zero_byte_output_after_success_fails(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine(runner=FakeRunner(output_bytes=b""))
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(VideoOutputVerificationError):
            engine.execute_plan(plan)

    def test_result_captures_timing_and_return_code(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan([clip], output)
        result = engine.execute_plan(plan)
        self.assertIsInstance(result.started_at, str)
        self.assertIsInstance(result.finished_at, str)
        self.assertGreaterEqual(result.duration_seconds, 0)
        self.assertEqual(result.return_code, 0)

    def test_temp_manifest_cleanup_when_configured(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        config = VideoEngineConfig(cleanup_temporary_files=True)
        engine = self._engine(config=config)
        plan = engine.build_concat_plan([clip], output)
        result = engine.execute_plan(plan)
        self.assertEqual(result.cleanup_result, "cleaned")
        self.assertFalse(Path(result.manifest_path).exists())

    def test_temp_manifest_retained_when_cleanup_disabled(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        config = VideoEngineConfig(cleanup_temporary_files=False)
        engine = self._engine(config=config)
        plan = engine.build_concat_plan([clip], output)
        result = engine.execute_plan(plan)
        self.assertEqual(result.cleanup_result, "retained")
        self.assertTrue(Path(result.manifest_path).exists())

    def test_previous_output_not_pre_deleted_on_failure(self) -> None:
        clip = self._input("a.mp4")
        output_path = self.temp_dir / "output.mp4"
        output_path.write_bytes(b"original-content")
        output = self._output()
        engine = self._engine(runner=FakeRunner(should_fail=True))
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(FFmpegExecutionError):
            engine.execute_plan(plan, force=True)
        self.assertEqual(output_path.read_bytes(), b"original-content")

    def test_failure_never_triggers_fallback_command(self) -> None:
        clip = self._input("a.mp4")
        output = self._output()
        runner = FakeRunner(should_fail=True)
        engine = self._engine(runner=runner)
        plan = engine.build_concat_plan([clip], output)
        with self.assertRaises(FFmpegExecutionError):
            engine.execute_plan(plan)
        self.assertEqual(len(runner.calls), 1)

    def test_normalized_render_execution_succeeds(self) -> None:
        clips = [
            self._input("c0.mp4", video=_video_spec(width=1080, height=1920)),
            self._input("c1.mp4", video=_video_spec(width=720, height=1280)),
        ]
        output = self._output()
        engine = self._engine()
        plan = engine.build_concat_plan(clips, output)
        self.assertEqual(plan.plan_type, PLAN_NORMALIZED_RENDER)
        result = engine.execute_plan(plan)
        self.assertTrue(result.output_exists)
        self.assertIsNone(result.manifest_path)


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

    def test_no_publishing_or_social_module_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.publishing"))
            self.assertFalse(stripped.startswith("from .publishing"))

    def test_no_upload_publish_code(self) -> None:
        self.assertNotIn("requests.post", MODULE_SOURCE)
        self.assertNotIn("upload_to_instagram", MODULE_SOURCE)

    def test_no_music_subtitle_logo_implementation(self) -> None:
        # Checked as actual ffmpeg filter/argument signatures, not bare
        # words — this module's own docstrings legitimately name
        # "music"/"subtitle"/"logo" when documenting what it does NOT
        # implement (the same false-fail risk fixed in prior phases'
        # structural tests).
        for forbidden in ("subtitles=", "drawtext", "watermark.png", "amix", "-i music"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_shell_true(self) -> None:
        self.assertNotIn("shell=True", MODULE_SOURCE)

    def test_does_not_mutate_input_media_files(self) -> None:
        # This module never writes binary content to any file itself
        # (only text manifests via write_text, and whatever ffmpeg itself
        # produces as a subprocess). cleanup_temporary_files() legitimately
        # unlinks *temporary manifest files it created itself* — never an
        # input media path — so unlink() usage is checked separately in
        # execute_plan() to be scoped to `temporary_files`, not asserted
        # absent outright.
        self.assertNotIn(".write_bytes(", MODULE_SOURCE)
        self.assertNotIn("os.remove(", MODULE_SOURCE)

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("publish", DISALLOWED_ACTIONS)
        self.assertIn("upload", DISALLOWED_ACTIONS)
        self.assertIn("add_music", DISALLOWED_ACTIONS)
        self.assertIn("add_subtitles", DISALLOWED_ACTIONS)
        self.assertIn("add_logo", DISALLOWED_ACTIONS)


if __name__ == "__main__":
    unittest.main()
