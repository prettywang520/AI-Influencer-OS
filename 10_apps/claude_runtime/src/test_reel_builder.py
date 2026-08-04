from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .reel_builder import (
    BuilderConfig,
    DuplicateSceneError,
    EmptySceneFileError,
    FfmpegRenderError,
    MissingSceneError,
    MixedFpsError,
    MixedResolutionError,
    ProcessResult,
    ReelAlreadyExistsError,
    ReelBuilderConfigError,
    ReelBuildOutputMissingError,
    SceneClip,
    SceneDirectoryNotFoundError,
    UnrecognizedSceneFileError,
    build_ffmpeg_command,
    build_reel,
    default_builder_config_path,
    discover_scene_clips,
    load_builder_config,
    validate_clip_consistency,
    validate_date,
)

PRODUCTION_DATE = "2026-08-01"


# ---------------------------------------------------------------------------
# Fake ffmpeg/ffprobe runner — no real subprocess is ever invoked in tests.
# ---------------------------------------------------------------------------


def _probe_json(
    *,
    width: int = 1080,
    height: int = 1920,
    fps: str = "30/1",
    codec: str = "h264",
    duration: float = 5.0,
    has_audio: bool = True,
) -> dict:
    streams = [
        {
            "codec_type": "video",
            "width": width,
            "height": height,
            "r_frame_rate": fps,
            "codec_name": codec,
            "duration": str(duration),
        }
    ]
    if has_audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    return {"streams": streams, "format": {"duration": str(duration)}}


class FakeRunner:
    """
    Replaces both ffprobe and ffmpeg. probes maps an absolute clip path
    (str) to a probe spec dict (from _probe_json). ffmpeg calls either
    write ffmpeg_output_bytes to the command's final argument (the output
    path) and return 0, or return a non-zero exit when configured to fail.
    """

    def __init__(
        self,
        *,
        probes: dict[str, dict] | None = None,
        ffmpeg_should_fail: bool = False,
        ffmpeg_write_output: bool = True,
        ffmpeg_output_bytes: bytes = b"fake-mp4-bytes",
    ) -> None:
        self.probes = probes or {}
        self.ffmpeg_should_fail = ffmpeg_should_fail
        self.ffmpeg_write_output = ffmpeg_write_output
        self.ffmpeg_output_bytes = ffmpeg_output_bytes
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

        # ffmpeg
        if self.ffmpeg_should_fail:
            return ProcessResult(returncode=1, stderr="simulated ffmpeg failure")

        if self.ffmpeg_write_output:
            output_path = Path(command[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(self.ffmpeg_output_bytes)

        return ProcessResult(returncode=0, stdout="", stderr="")


class ReelBuilderTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.scenes_dir = self.temp_dir / "output" / PRODUCTION_DATE / "videos" / "reel_scenes"
        self.scenes_dir.mkdir(parents=True)
        self.config = BuilderConfig()

    def _write_scene(self, name: str, *, size: int = 100) -> Path:
        path = self.scenes_dir / name
        path.write_bytes(b"x" * size)
        return path

    def _write_five_scenes(self) -> list[Path]:
        return [self._write_scene(f"scene_{index:02d}.mp4") for index in range(1, 6)]

    def _probes_for(self, paths: list[Path], **kwargs) -> dict[str, dict]:
        return {str(path): _probe_json(**kwargs) for path in paths}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_path_exists(self) -> None:
        self.assertTrue(default_builder_config_path().is_file())

    def test_real_config_loads_with_expected_defaults(self) -> None:
        config = load_builder_config()
        self.assertEqual(config.scene_count, 5)
        self.assertEqual(config.transition, "none")
        self.assertEqual(config.fps, 30)
        self.assertEqual(config.video_codec, "libx264")
        self.assertEqual(config.audio_codec, "aac")
        self.assertTrue(config.normalize_resolution)
        self.assertTrue(config.normalize_fps)

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(ReelBuilderConfigError):
            load_builder_config("/nonexistent/path/builder.yaml")

    def test_unimplemented_transition_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "builder.yaml"
            config_path.write_text(
                "rendering:\n  transition: crossfade\n", encoding="utf-8"
            )
            with self.assertRaises(ReelBuilderConfigError):
                load_builder_config(config_path)

    def test_date_validation(self) -> None:
        validate_date("2026-08-01")  # does not raise
        with self.assertRaises(Exception):
            validate_date("08-01-2026")
        with self.assertRaises(Exception):
            validate_date("not-a-date")


# ---------------------------------------------------------------------------
# Scene discovery (pure, no ffmpeg)
# ---------------------------------------------------------------------------


class SceneDiscoveryTests(ReelBuilderTempTestCase):
    def test_successful_discovery_returns_five_ordered_clips(self) -> None:
        self._write_five_scenes()
        clips = discover_scene_clips(self.scenes_dir, 5)
        self.assertEqual([clip.index for clip in clips], [1, 2, 3, 4, 5])
        self.assertTrue(all(clip.path.is_file() for clip in clips))

    def test_missing_clip_fails(self) -> None:
        for index in range(1, 5):  # only 1..4, missing 5
            self._write_scene(f"scene_{index:02d}.mp4")
        with self.assertRaises(MissingSceneError):
            discover_scene_clips(self.scenes_dir, 5)

    def test_duplicate_clip_across_naming_conventions_fails(self) -> None:
        self._write_five_scenes()
        self._write_scene("01.mp4")  # same index (1) as scene_01.mp4
        with self.assertRaises(DuplicateSceneError):
            discover_scene_clips(self.scenes_dir, 5)

    def test_wrong_numbering_fails(self) -> None:
        self._write_scene("scene_1.mp4")  # single digit, not zero-padded
        self._write_scene("scene_02.mp4")
        self._write_scene("scene_03.mp4")
        self._write_scene("scene_04.mp4")
        self._write_scene("scene_05.mp4")
        with self.assertRaises(UnrecognizedSceneFileError):
            discover_scene_clips(self.scenes_dir, 5)

    def test_empty_file_fails(self) -> None:
        for index in range(1, 5):
            self._write_scene(f"scene_{index:02d}.mp4")
        self._write_scene("scene_05.mp4", size=0)
        with self.assertRaises(EmptySceneFileError):
            discover_scene_clips(self.scenes_dir, 5)

    def test_scenes_directory_missing_fails(self) -> None:
        missing_dir = self.temp_dir / "does_not_exist"
        with self.assertRaises(SceneDirectoryNotFoundError):
            discover_scene_clips(missing_dir, 5)

    def test_extra_clip_beyond_scene_count_fails(self) -> None:
        self._write_five_scenes()
        self._write_scene("scene_06.mp4")
        with self.assertRaises(UnrecognizedSceneFileError):
            discover_scene_clips(self.scenes_dir, 5)

    def test_non_video_files_are_ignored(self) -> None:
        self._write_five_scenes()
        (self.scenes_dir / "notes.txt").write_text("not a video", encoding="utf-8")
        clips = discover_scene_clips(self.scenes_dir, 5)
        self.assertEqual(len(clips), 5)


# ---------------------------------------------------------------------------
# Clip consistency validation
# ---------------------------------------------------------------------------


class ClipConsistencyTests(unittest.TestCase):
    def _probe(self, **kwargs):
        from .reel_builder import VideoProbe

        return VideoProbe(
            path=kwargs.pop("path", "x.mp4"),
            width=kwargs.pop("width", 1080),
            height=kwargs.pop("height", 1920),
            fps=kwargs.pop("fps", 30.0),
            duration_seconds=kwargs.pop("duration_seconds", 5.0),
            codec_name=kwargs.pop("codec_name", "h264"),
            has_audio=kwargs.pop("has_audio", True),
        )

    def test_identical_clips_do_not_require_normalization(self) -> None:
        probes = [self._probe(), self._probe(), self._probe()]
        config = BuilderConfig(normalize_resolution=True, normalize_fps=True)
        self.assertFalse(validate_clip_consistency(probes, config))

    def test_mixed_resolution_with_normalize_true_requires_normalization(self) -> None:
        probes = [self._probe(width=1080, height=1920), self._probe(width=720, height=1280)]
        config = BuilderConfig(normalize_resolution=True, normalize_fps=True)
        self.assertTrue(validate_clip_consistency(probes, config))

    def test_mixed_resolution_with_normalize_false_raises(self) -> None:
        probes = [self._probe(width=1080, height=1920), self._probe(width=720, height=1280)]
        config = BuilderConfig(normalize_resolution=False, normalize_fps=True)
        with self.assertRaises(MixedResolutionError):
            validate_clip_consistency(probes, config)

    def test_mixed_fps_with_normalize_true_requires_normalization(self) -> None:
        probes = [self._probe(fps=30.0), self._probe(fps=24.0)]
        config = BuilderConfig(normalize_resolution=True, normalize_fps=True)
        self.assertTrue(validate_clip_consistency(probes, config))

    def test_mixed_fps_with_normalize_false_raises(self) -> None:
        probes = [self._probe(fps=30.0), self._probe(fps=24.0)]
        config = BuilderConfig(normalize_resolution=True, normalize_fps=False)
        with self.assertRaises(MixedFpsError):
            validate_clip_consistency(probes, config)


# ---------------------------------------------------------------------------
# ffmpeg command construction
# ---------------------------------------------------------------------------


class FfmpegCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BuilderConfig()
        self.clips = [SceneClip(index=i, path=Path(f"/tmp/scene_{i:02d}.mp4")) for i in range(1, 6)]

    def test_lossless_command_uses_concat_demuxer(self) -> None:
        command = build_ffmpeg_command(
            clips=self.clips,
            output_path=Path("/tmp/reel_final.mp4"),
            config=self.config,
            concat_list_path=Path("/tmp/concat_list.txt"),
            normalize=False,
            force=False,
        )
        self.assertIn("-c", command)
        self.assertIn("copy", command)
        self.assertNotIn("-filter_complex", command)
        self.assertIn("-n", command)

    def test_normalize_command_uses_filter_complex(self) -> None:
        command = build_ffmpeg_command(
            clips=self.clips,
            output_path=Path("/tmp/reel_final.mp4"),
            config=self.config,
            concat_list_path=Path("/tmp/concat_list.txt"),
            normalize=True,
            force=False,
            target_width=1080,
            target_height=1920,
        )
        self.assertIn("-filter_complex", command)
        self.assertIn(self.config.video_codec, command)
        self.assertIn(self.config.audio_codec, command)

    def test_force_uses_overwrite_flag(self) -> None:
        command = build_ffmpeg_command(
            clips=self.clips,
            output_path=Path("/tmp/reel_final.mp4"),
            config=self.config,
            concat_list_path=Path("/tmp/concat_list.txt"),
            normalize=False,
            force=True,
        )
        self.assertIn("-y", command)
        self.assertNotIn("-n", command)


# ---------------------------------------------------------------------------
# Full build_reel orchestration
# ---------------------------------------------------------------------------


class BuildReelTests(ReelBuilderTempTestCase):
    def test_successful_build(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir
        )

        self.assertEqual(len(result.scene_clips), 5)
        self.assertEqual(result.clip_durations, [5.0, 5.0, 5.0, 5.0, 5.0])
        self.assertFalse(result.normalized)
        self.assertGreater(result.output_size_bytes, 0)

        output_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertTrue(output_path.is_file())
        self.assertEqual(output_path.read_bytes(), b"fake-mp4-bytes")

    def test_successful_build_with_mixed_resolution_normalizes(self) -> None:
        clip_paths = self._write_five_scenes()
        probes = self._probes_for(clip_paths)
        # Make the last clip a different resolution.
        probes[str(clip_paths[-1])] = _probe_json(width=720, height=1280)
        runner = FakeRunner(probes=probes)

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir
        )
        self.assertTrue(result.normalized)

        ffmpeg_calls = [call for call in runner.calls if Path(call[0]).name == "ffmpeg"]
        self.assertEqual(len(ffmpeg_calls), 1)
        self.assertIn("-filter_complex", ffmpeg_calls[0])

    def test_overwrite_refused_without_force(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        output_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"existing-reel")

        with self.assertRaises(ReelAlreadyExistsError):
            build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        # Refusal happens before any ffmpeg/ffprobe call.
        self.assertEqual(runner.calls, [])
        self.assertEqual(output_path.read_bytes(), b"existing-reel")

    def test_force_overwrite_succeeds(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        output_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"existing-reel")

        result = build_reel(
            PRODUCTION_DATE, self.config, force=True, runner=runner, root=self.temp_dir
        )
        self.assertEqual(output_path.read_bytes(), b"fake-mp4-bytes")
        self.assertIn("-y", result.ffmpeg_command)

    def test_ffmpeg_failure_raises_and_leaves_no_output(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths), ffmpeg_should_fail=True)

        with self.assertRaises(FfmpegRenderError):
            build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        output_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertFalse(output_path.exists())

    def test_ffmpeg_success_but_no_file_written_raises(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths), ffmpeg_write_output=False)

        with self.assertRaises(ReelBuildOutputMissingError):
            build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

    def test_missing_scene_fails_before_any_ffmpeg_call(self) -> None:
        for index in range(1, 4):
            self._write_scene(f"scene_{index:02d}.mp4")
        runner = FakeRunner()

        with self.assertRaises(MissingSceneError):
            build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        self.assertEqual(runner.calls, [])


# ---------------------------------------------------------------------------
# Diagnostics / build log
# ---------------------------------------------------------------------------


class BuildLogTests(ReelBuilderTempTestCase):
    def test_log_written_on_success_with_required_fields(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertTrue(log_path.is_file())

        payload = json.loads(log_path.read_text(encoding="utf-8"))
        for field_name in (
            "started_at",
            "finished_at",
            "production_date",
            "input_clips",
            "clip_durations",
            "render_duration_seconds",
            "ffmpeg_command",
            "output_size_bytes",
            "warnings",
            "errors",
            "result",
        ):
            self.assertIn(field_name, payload)

        self.assertEqual(payload["result"], "success")
        self.assertEqual(payload["errors"], [])
        self.assertEqual(len(payload["input_clips"]), 5)

    def test_log_written_on_failure_with_error_recorded(self) -> None:
        for index in range(1, 4):
            self._write_scene(f"scene_{index:02d}.mp4")
        runner = FakeRunner()

        with self.assertRaises(MissingSceneError):
            build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertTrue(log_path.is_file())

        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"], "failed")
        self.assertEqual(len(payload["errors"]), 1)
        self.assertIn("Missing scene clip", payload["errors"][0])

    def test_warning_recorded_when_a_clip_has_no_audio(self) -> None:
        clip_paths = self._write_five_scenes()
        probes = self._probes_for(clip_paths)
        probes[str(clip_paths[0])] = _probe_json(has_audio=False)
        runner = FakeRunner(probes=probes)

        result = build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)
        self.assertTrue(
            any("audio" in warning for warning in result.warnings),
            result.warnings,
        )


if __name__ == "__main__":
    unittest.main()
