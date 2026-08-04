from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import music_mixer, timeline_engine
from .reel_builder import (
    BuilderConfig,
    DuplicateSceneError,
    EmptySceneFileError,
    FfmpegRenderError,
    MissingSceneError,
    MixedAudioPresenceError,
    MixedFpsError,
    MixedResolutionError,
    MusicMixIntegrationError,
    ProcessResult,
    ReelAlreadyExistsError,
    ReelBuilderConfigError,
    ReelBuildOutputMissingError,
    SceneClip,
    SceneDirectoryNotFoundError,
    TimelineIntegrationError,
    UnrecognizedSceneFileError,
    build_ffmpeg_command,
    build_reel,
    default_builder_config_path,
    discover_scene_clips,
    load_builder_config,
    parse_arguments,
    validate_clip_consistency,
    validate_date,
)

PRODUCTION_DATE = "2026-08-01"
MODULE_PATH = Path(__file__).resolve().parent / "reel_builder.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


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
# Phase 11B.1 — Music integration fixtures
#
# media_inspector.inspect_file() (called internally by music_mixer.mix_music())
# resolves paths before probing — unlike reel_builder's own probe_video(),
# which does not — so probes used for the intermediate video / music file
# must be keyed by the RESOLVED path (same macOS /var vs /private/var
# symlink fix already applied in test_music_mixer.py), while scene-clip
# probes stay keyed by the unresolved path as before.
# ---------------------------------------------------------------------------


def _music_video_probe_json(*, duration: float = 12.5, has_audio: bool = True) -> dict:
    streams = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1080,
            "height": 1920,
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30/1",
            "duration": str(duration),
            "pix_fmt": "yuv420p",
            "tags": {},
            "disposition": {"default": 1, "attached_pic": 0},
        }
    ]
    if has_audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "duration": str(duration),
                "tags": {},
                "disposition": {"default": 1, "attached_pic": 0},
            }
        )
    return {
        "streams": streams,
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": str(duration), "bit_rate": "8000000"},
        "chapters": [],
    }


def _music_track_probe_json(*, duration: float = 8.0) -> dict:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "mp3",
                "sample_rate": "44100",
                "channels": 2,
                "duration": str(duration),
                "tags": {},
                "disposition": {"default": 1, "attached_pic": 0},
            }
        ],
        "format": {"format_name": "mp3", "duration": str(duration), "bit_rate": "192000"},
        "chapters": [],
    }


def _fake_mix_plan(*, video_path: Path, music_path: Path, output_path: Path) -> music_mixer.MusicMixPlan:
    return music_mixer.MusicMixPlan(
        video_path=video_path,
        music_path=music_path,
        output_path=output_path,
        video_duration_seconds=12.5,
        music_duration_seconds=8.0,
        video_has_audio=True,
        music_mode=music_mixer.MusicMode.LOOP,
        music_loop_required=True,
        music_trim_required=False,
        fade_in_seconds=0.5,
        fade_out_seconds=1.0,
        music_volume=0.2,
        source_audio_volume=1.0,
        ducking_mode=music_mixer.DuckingMode.NONE,
        audio_decision=music_mixer.AudioMixDecision(case="mix_source_and_music", use_amix=True),
        command=["ffmpeg", "-n", "-i", str(video_path), "-i", str(music_path), str(output_path)],
        warnings=[],
    )


def _fake_mix_result(
    *, video_path: Path, music_path: Path, output_path: Path, warnings: list[str] | None = None
) -> music_mixer.MusicMixResult:
    plan = _fake_mix_plan(video_path=video_path, music_path=music_path, output_path=output_path)
    return music_mixer.MusicMixResult(
        started_at="2026-08-04T00:00:00+00:00",
        finished_at="2026-08-04T00:00:01+00:00",
        duration_seconds=1.0,
        video_path=str(video_path),
        music_path=str(music_path),
        output_path=str(output_path),
        command=plan.command,
        return_code=0,
        stdout="",
        stderr="",
        output_exists=True,
        output_size_bytes=999,
        plan=plan,
        warnings=warnings or [],
        error=None,
    )


class FakeMixer:
    """
    Injectable stand-in for music_mixer.mix_music, matching its exact
    call shape: (request, config, *, runner) -> MusicMixResult. Never
    touches real ffmpeg/ffprobe/Media Inspector — used for tests that
    only care about how reel_builder.build_reel() wires into Music
    Mixer, not Music Mixer's own internals (those are covered by
    test_music_mixer.py, unchanged).
    """

    def __init__(self, *, should_fail: bool = False, write_output: bool = True, output_bytes: bytes = b"fake-final-bytes") -> None:
        self.should_fail = should_fail
        self.write_output = write_output
        self.output_bytes = output_bytes
        self.calls: list[music_mixer.MusicMixRequest] = []

    def __call__(self, request: music_mixer.MusicMixRequest, config: music_mixer.MusicMixerConfig, *, runner):
        self.calls.append(request)

        if self.should_fail:
            raise music_mixer.FFmpegExecutionError("simulated mixer failure")

        if self.write_output:
            Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(request.output_path).write_bytes(self.output_bytes)

        return _fake_mix_result(
            video_path=request.video_path, music_path=request.music_path, output_path=request.output_path
        )


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

    def test_lossless_build_uses_video_engine_concat_manifest(self) -> None:
        # Phase 11A.3: a lossless build now writes its concat manifest
        # via VideoEngine.write_concat_manifest() under a
        # .video_engine_tmp directory next to the output, and cleans it
        # up afterward (VideoEngineConfig.cleanup_temporary_files
        # defaults true) — proving the build genuinely routed through
        # Video Engine's plan/execute path, not just an equivalent
        # command built by hand.
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        output_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        tmp_dir = output_path.parent / ".video_engine_tmp"
        # Manifest was written and then cleaned up (default config).
        self.assertFalse(list(tmp_dir.glob("*")) if tmp_dir.exists() else False)

    def test_normalized_build_uses_video_engine_plan(self) -> None:
        clip_paths = self._write_five_scenes()
        probes = self._probes_for(clip_paths)
        probes[str(clip_paths[-1])] = _probe_json(width=720, height=1280)
        runner = FakeRunner(probes=probes)

        build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["engine_plan_type"], "normalized_render")


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

    def test_warning_recorded_when_all_clips_uniformly_lack_audio(self) -> None:
        # Phase 11A.3 note: this fixture was adapted from "one of five
        # clips silent" to "all five clips silent" — the former was
        # actually a mixed-audio-presence layout, which Phase 11A.3 now
        # correctly rejects via MixedAudioPresenceError (see
        # test_mixed_audio_presence_fails_clearly below) rather than
        # silently warning and building anyway. A uniformly audio-less
        # set of clips is still a legitimate, successful build that
        # should warn about the resulting silent output — that is what
        # this test now verifies.
        clip_paths = self._write_five_scenes()
        probes = self._probes_for(clip_paths, has_audio=False)
        runner = FakeRunner(probes=probes)

        result = build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)
        self.assertTrue(
            any("audio" in warning for warning in result.warnings),
            result.warnings,
        )

    def test_mixed_audio_presence_fails_clearly(self) -> None:
        # New in Phase 11A.3 (requirement 6): mixed audio presence across
        # scene clips must fail clearly rather than silently building an
        # output with potentially desynchronized/incomplete audio.
        clip_paths = self._write_five_scenes()
        probes = self._probes_for(clip_paths)
        probes[str(clip_paths[0])] = _probe_json(has_audio=False)
        runner = FakeRunner(probes=probes)

        with self.assertRaises(MixedAudioPresenceError):
            build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        output_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertFalse(output_path.exists())

        ffmpeg_calls = [call for call in runner.calls if Path(call[0]).name == "ffmpeg"]
        self.assertEqual(ffmpeg_calls, [])


# ---------------------------------------------------------------------------
# Phase 11B.1 — Reel Builder Music Integration
# ---------------------------------------------------------------------------


class BackwardCompatibilityTests(ReelBuilderTempTestCase):
    def test_no_music_keeps_original_final_output_path(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        result = build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)
        expected = str(self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir))
        self.assertEqual(result.output_path, expected)

    def test_no_music_does_not_create_intermediate(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)
        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertFalse(intermediate_path.exists())

    def test_no_music_never_calls_music_mixer(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("music_mixer_callable must not be called without --music")

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            music_mixer_callable=_fail_if_called,
        )
        self.assertIsNone(result.music_mix)

    def test_old_positional_and_keyword_callers_still_work(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        # Exactly the call shape every one of the original 34 tests uses —
        # no new keyword required.
        result = build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)
        self.assertTrue(result.output_path)


class CliTests(unittest.TestCase):
    def test_music_flag_accepted(self) -> None:
        args = parse_arguments(["--date", PRODUCTION_DATE, "--music", "/tmp/music.mp3"])
        self.assertEqual(args.music, "/tmp/music.mp3")

    def test_music_volume_accepted(self) -> None:
        args = parse_arguments(["--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3", "--music-volume", "0.5"])
        self.assertEqual(args.music_volume, 0.5)

    def test_source_audio_volume_accepted(self) -> None:
        args = parse_arguments(
            ["--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3", "--source-audio-volume", "0.8"]
        )
        self.assertEqual(args.source_audio_volume, 0.8)

    def test_music_mode_accepted(self) -> None:
        args = parse_arguments(["--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3", "--music-mode", "trim"])
        self.assertEqual(args.music_mode, "trim")

    def test_ducking_accepted(self) -> None:
        args = parse_arguments(["--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3", "--ducking", "fixed"])
        self.assertEqual(args.ducking_mode, "fixed")

    def test_fade_options_accepted(self) -> None:
        args = parse_arguments(
            [
                "--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3",
                "--fade-in-seconds", "0.25", "--fade-out-seconds", "0.75",
            ]
        )
        self.assertEqual(args.fade_in_seconds, 0.25)
        self.assertEqual(args.fade_out_seconds, 0.75)

    def test_keep_intermediate_accepted(self) -> None:
        args = parse_arguments(["--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3", "--keep-intermediate"])
        self.assertTrue(args.keep_intermediate)

    def test_music_specific_flag_without_music_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--date", PRODUCTION_DATE, "--music-volume", "0.5"])

    def test_keep_intermediate_without_music_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--date", PRODUCTION_DATE, "--keep-intermediate"])

    def test_invalid_music_mode_choice_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3", "--music-mode", "bogus"])

    def test_invalid_ducking_choice_fails(self) -> None:
        with self.assertRaises(SystemExit):
            parse_arguments(["--date", PRODUCTION_DATE, "--music", "/tmp/m.mp3", "--ducking", "bogus"])


class MusicFlowEndToEndTests(ReelBuilderTempTestCase):
    """Uses the real, unmodified music_mixer.mix_music() end-to-end,
    through a single shared FakeRunner — proves genuine wiring, not just
    that reel_builder calls *something*."""

    def _run_with_music(self, **build_kwargs):
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)

        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)

        probes = self._probes_for(clip_paths)
        probes[str(intermediate_path.resolve())] = _music_video_probe_json()
        probes[str(music_path.resolve())] = _music_track_probe_json()

        runner = FakeRunner(probes=probes)
        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            **build_kwargs,
        )
        return result, runner, intermediate_path, final_path, music_path

    def test_music_path_renders_intermediate_first(self) -> None:
        result, runner, intermediate_path, final_path, _music_path = self._run_with_music()
        # ffmpeg is called twice: once for the scene concat (writing the
        # intermediate), once for the mix (writing the final).
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        self.assertEqual(len(ffmpeg_calls), 2)
        self.assertEqual(Path(ffmpeg_calls[0][-1]), intermediate_path)
        self.assertEqual(Path(ffmpeg_calls[1][-1]), final_path)

    def test_final_output_exists_and_intermediate_cleaned_by_default(self) -> None:
        result, _runner, intermediate_path, final_path, _music_path = self._run_with_music()
        self.assertTrue(final_path.is_file())
        self.assertFalse(intermediate_path.exists())
        self.assertEqual(result.output_path, str(final_path))

    def test_exact_music_path_passed_once(self) -> None:
        _result, runner, _i, _f, music_path = self._run_with_music()
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        mix_command = ffmpeg_calls[1]
        self.assertEqual(mix_command.count(str(music_path)), 1)

    def test_no_fallback_to_no_music_final_on_mixer_failure(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)

        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)

        probes = self._probes_for(clip_paths)
        probes[str(intermediate_path.resolve())] = _music_video_probe_json()
        # Deliberately no probe entry for the music file -> Media
        # Inspector's ffprobe call fails -> mix_music() raises.
        runner = FakeRunner(probes=probes)

        with self.assertRaises(MusicMixIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, music_path=music_path
            )

        self.assertFalse(final_path.exists())
        self.assertTrue(intermediate_path.exists())


class IntermediateHandlingTests(ReelBuilderTempTestCase):
    def _run_with_fake_mixer(self, mixer: FakeMixer, **build_kwargs):
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            music_mixer_callable=mixer,
            **build_kwargs,
        )
        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        return result, intermediate_path

    def test_success_deletes_intermediate_by_default(self) -> None:
        _result, intermediate_path = self._run_with_fake_mixer(FakeMixer())
        self.assertFalse(intermediate_path.exists())

    def test_success_keeps_intermediate_with_keep_intermediate(self) -> None:
        _result, intermediate_path = self._run_with_fake_mixer(FakeMixer(), keep_intermediate=True)
        self.assertTrue(intermediate_path.exists())

    def test_failure_preserves_intermediate(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        mixer = FakeMixer(should_fail=True)

        with self.assertRaises(MusicMixIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=mixer,
            )

        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertTrue(intermediate_path.is_file())

    def test_cleanup_only_removes_exact_intermediate(self) -> None:
        unrelated = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir).parent
        unrelated.mkdir(parents=True, exist_ok=True)
        unrelated_file = unrelated / "unrelated_asset.mp4"
        unrelated_file.write_bytes(b"do-not-touch")

        self._run_with_fake_mixer(FakeMixer())

        self.assertTrue(unrelated_file.is_file())
        self.assertEqual(unrelated_file.read_bytes(), b"do-not-touch")

    def test_cleanup_result_recorded_in_music_mix(self) -> None:
        result, _intermediate_path = self._run_with_fake_mixer(FakeMixer())
        self.assertFalse(result.music_mix["intermediate_retained"])

    def test_cleanup_result_recorded_when_kept(self) -> None:
        result, _intermediate_path = self._run_with_fake_mixer(FakeMixer(), keep_intermediate=True)
        self.assertTrue(result.music_mix["intermediate_retained"])


class FailureBehaviorTests(ReelBuilderTempTestCase):
    def test_render_failure_prevents_mixer_call(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths), ffmpeg_should_fail=True)
        mixer = FakeMixer()

        with self.assertRaises(FfmpegRenderError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=mixer,
            )

        self.assertEqual(mixer.calls, [])

    def test_mixer_failure_marks_build_failed(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        mixer = FakeMixer(should_fail=True)

        with self.assertRaises(MusicMixIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=mixer,
            )

    def test_mixer_failure_logs_error(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        mixer = FakeMixer(should_fail=True)

        with self.assertRaises(MusicMixIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=mixer,
            )

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["errors"], ["music mix failed", "simulated mixer failure"])
        self.assertEqual(payload["music_mix"]["status"], "failed")

    def test_mixer_failure_does_not_create_final_output(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        mixer = FakeMixer(should_fail=True)

        with self.assertRaises(MusicMixIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=mixer,
            )

        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertFalse(final_path.exists())


class OverwriteTests(ReelBuilderTempTestCase):
    def test_existing_final_refused_without_force(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"existing-final")
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        with self.assertRaises(ReelAlreadyExistsError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=FakeMixer(),
            )
        self.assertEqual(runner.calls, [])

    def test_existing_intermediate_refused_without_force(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        intermediate_path.write_bytes(b"stale-intermediate")
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        with self.assertRaises(ReelAlreadyExistsError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=FakeMixer(),
            )
        self.assertEqual(runner.calls, [])

    def test_force_propagates_to_render_and_mixer(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"existing-final")
        intermediate_path.write_bytes(b"stale-intermediate")

        runner = FakeRunner(probes=self._probes_for(clip_paths))
        mixer = FakeMixer()

        build_reel(
            PRODUCTION_DATE,
            self.config,
            force=True,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            music_mixer_callable=mixer,
        )

        self.assertEqual(len(mixer.calls), 1)
        self.assertTrue(mixer.calls[0].force)

    def test_force_does_not_predelete_unrelated_files(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        unrelated_file = final_path.parent / "unrelated.mp4"
        unrelated_file.write_bytes(b"keep-me")

        runner = FakeRunner(probes=self._probes_for(clip_paths))
        build_reel(
            PRODUCTION_DATE,
            self.config,
            force=True,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            music_mixer_callable=FakeMixer(),
        )

        self.assertEqual(unrelated_file.read_bytes(), b"keep-me")


class BuildLogMusicTests(ReelBuilderTempTestCase):
    def test_music_mix_disabled_without_music(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["music_mix"], {"enabled": False})

    def test_music_mix_success_fields_written(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            music_mixer_callable=FakeMixer(),
        )

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        music_mix = payload["music_mix"]

        self.assertTrue(music_mix["enabled"])
        self.assertEqual(music_mix["status"], "success")
        for field_name in (
            "music_path",
            "music_filename",
            "music_volume",
            "source_audio_volume",
            "music_mode",
            "ducking_mode",
            "fade_in_seconds",
            "fade_out_seconds",
            "intermediate_path",
            "intermediate_retained",
            "mix_duration_seconds",
            "diagnostic_log",
            "warnings",
        ):
            self.assertIn(field_name, music_mix)

    def test_music_mix_failure_fields_written(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        with self.assertRaises(MusicMixIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=runner,
                root=self.temp_dir,
                music_path=music_path,
                music_mixer_callable=FakeMixer(should_fail=True),
            )

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        music_mix = payload["music_mix"]
        self.assertTrue(music_mix["enabled"])
        self.assertEqual(music_mix["status"], "failed")
        self.assertTrue(music_mix["intermediate_retained"])
        self.assertIn("error", music_mix)

    def test_original_build_log_fields_remain_present(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            music_mixer_callable=FakeMixer(),
        )

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
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

    def test_warnings_preserved_in_music_mix(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        mixer = FakeMixer()

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            music_mixer_callable=mixer,
        )
        self.assertEqual(result.music_mix["warnings"], [])

    def test_diagnostic_log_path_recorded_when_available(self) -> None:
        clip_paths = self._write_five_scenes()
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            music_path=music_path,
            music_mixer_callable=FakeMixer(),
        )
        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        expected_suffix = "_music_mix_log.json"
        self.assertTrue(result.music_mix["diagnostic_log"].endswith(f"{final_path.stem}{expected_suffix}"))


# ---------------------------------------------------------------------------
# Phase 11C.1 — Timeline -> Reel Builder Integration
# ---------------------------------------------------------------------------


class TimelineTestCase(ReelBuilderTempTestCase):
    """Shared helpers for building a real timeline_engine.Timeline over
    the temp scene files, matching FakeRunner's probe fixtures. Uses the
    real (already-tested) timeline_engine.save_timeline()/load_timeline()/
    validate_timeline() by default -- no timeline_engine internals are
    reimplemented here."""

    def _write_scenes(self, count: int, *, duration: float = 5.0, size: int = 100) -> list[Path]:
        return [self._write_scene(f"scene_{index:02d}.mp4", size=size) for index in range(1, count + 1)]

    def _build_timeline(
        self,
        clip_paths: list[Path],
        *,
        production_date: str | None = PRODUCTION_DATE,
        scene_duration: float = 5.0,
        music_path: str | Path | None = None,
    ) -> timeline_engine.Timeline:
        video_track = timeline_engine.TimelineTrack(
            track_id="track_video", track_type=timeline_engine.TrackType.VIDEO, order=0
        )
        running = 0.0
        for index, path in enumerate(clip_paths):
            video_track.clips.append(
                timeline_engine.VideoClip(
                    clip_id=f"clip_video_{index + 1:02d}",
                    track_id="track_video",
                    source_path=str(path),
                    start=running,
                    end=running + scene_duration,
                    duration_seconds=scene_duration,
                    source_in=0.0,
                    source_out=scene_duration,
                    scene_number=index + 1,
                )
            )
            running += scene_duration

        tracks = [video_track]
        if music_path is not None:
            audio_track = timeline_engine.TimelineTrack(
                track_id="track_audio_music", track_type=timeline_engine.TrackType.AUDIO, order=1
            )
            audio_track.clips.append(
                timeline_engine.AudioClip(
                    clip_id="clip_audio_music",
                    track_id="track_audio_music",
                    source_path=str(music_path),
                    start=0.0,
                    end=running,
                    duration_seconds=running,
                    source_in=0.0,
                    source_out=running,
                )
            )
            tracks.append(audio_track)

        return timeline_engine.Timeline(
            timeline_id="test-timeline-id",
            schema_version="1.0",
            production_date=production_date,
            created_at="2026-08-01T00:00:00+00:00",
            duration_seconds=running,
            tracks=tracks,
        )

    def _write_timeline(self, timeline: timeline_engine.Timeline, *, name: str = "timeline.json") -> Path:
        path = self.temp_dir / name
        timeline_engine.save_timeline(timeline, path)
        return path


class TimelineBackwardCompatibilityTests(TimelineTestCase):
    def test_no_timeline_never_calls_timeline_loader(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("timeline_loader_callable must not be called without --timeline")

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            timeline_loader_callable=_fail_if_called,
            timeline_validator_callable=_fail_if_called,
        )
        self.assertIsNone(result.timeline)

    def test_no_timeline_keeps_original_scene_discovery(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        result = build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)
        self.assertEqual(len(result.scene_clips), 5)
        self.assertIsNone(result.timeline)

    def test_old_positional_and_keyword_callers_still_work_with_timeline_params_present(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        result = build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)
        self.assertTrue(result.output_path)


class TimelineCliTests(unittest.TestCase):
    def test_timeline_flag_accepted(self) -> None:
        args = parse_arguments(["--date", PRODUCTION_DATE, "--timeline", "/tmp/timeline.json"])
        self.assertEqual(args.timeline, "/tmp/timeline.json")

    def test_timeline_with_music_accepted(self) -> None:
        args = parse_arguments(
            ["--date", PRODUCTION_DATE, "--timeline", "/tmp/t.json", "--music", "/tmp/m.mp3"]
        )
        self.assertEqual(args.timeline, "/tmp/t.json")
        self.assertEqual(args.music, "/tmp/m.mp3")

    def test_timeline_omitted_defaults_to_none(self) -> None:
        args = parse_arguments(["--date", PRODUCTION_DATE])
        self.assertIsNone(args.timeline)


class TimelineLoadingTests(TimelineTestCase):
    def test_missing_timeline_file_rejected(self) -> None:
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=FakeRunner(),
                root=self.temp_dir,
                timeline_path=self.temp_dir / "does_not_exist.json",
            )

    def test_empty_timeline_file_rejected(self) -> None:
        timeline_path = self.temp_dir / "timeline.json"
        timeline_path.write_bytes(b"")
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_malformed_timeline_file_rejected(self) -> None:
        timeline_path = self.temp_dir / "timeline.json"
        timeline_path.write_text("{not valid json")
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_invalid_timeline_rejected(self) -> None:
        # Structurally valid JSON but fails timeline_engine.validate_timeline()
        # (no video track at all).
        bad_timeline = timeline_engine.Timeline(timeline_id="bad", duration_seconds=0.0, tracks=[])
        timeline_path = self._write_timeline(bad_timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_timeline_date_mismatch_rejected(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, production_date="2020-01-01")
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_timeline_date_mismatch_allowed_when_config_disables_check(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, production_date="2020-01-01")
        timeline_path = self._write_timeline(timeline)
        config = BuilderConfig(timeline_production_date_must_match=False)
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        result = build_reel(
            PRODUCTION_DATE, config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertFalse(result.timeline["production_date_match"])

    def test_timeline_integration_disabled_rejects_timeline_flag(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths)
        timeline_path = self._write_timeline(timeline)
        config = BuilderConfig(timeline_integration_enabled=False)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )


class TimelineAuthorityTests(TimelineTestCase):
    def test_scene_order_overrides_directory_order(self) -> None:
        clip_paths = self._write_scenes(3)
        reversed_paths = list(reversed(clip_paths))
        timeline = self._build_timeline(reversed_paths)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertEqual(result.scene_clips, [str(p) for p in reversed_paths])

    def test_disabled_clip_ignored(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths)
        timeline.tracks[0].clips[1].enabled = False
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertEqual(len(result.scene_clips), 2)
        self.assertEqual(result.timeline["ignored_disabled_clips"], 1)

    def test_duplicate_resolved_source_path_rejected(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths)
        timeline.tracks[0].clips[1].source_path = str(clip_paths[0])
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_missing_source_file_rejected(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths)
        timeline.tracks[0].clips[1].source_path = str(self.temp_dir / "does_not_exist.mp4")
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_active_transition_rejected(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths)
        timeline.tracks[0].clips[0].transition_out = "crossfade"
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_non_unit_playback_rate_rejected(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths)
        timeline.tracks[0].clips[0].playback_rate = 1.5
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_gap_rejected(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths)
        timeline.tracks[0].clips[1].start += 2.0
        timeline.tracks[0].clips[1].end += 2.0
        timeline.duration_seconds += 2.0
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )

    def test_overlap_rejected(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths)
        timeline.tracks[0].clips[1].start -= 2.0
        timeline_path = self._write_timeline(timeline)
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=FakeRunner(), root=self.temp_dir, timeline_path=timeline_path
            )


class TimelineDurationReconciliationTests(TimelineTestCase):
    def test_exact_duration_passes(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertEqual(result.timeline["trimmed_clip_count"], 0)

    def test_mismatch_within_tolerance_passes(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        # Actual media is 5.02s, planned is 5.0s -- within the default 0.05s tolerance.
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.02))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertIsNotNone(result.timeline)

    def test_actual_source_shorter_than_source_out_fails(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        # Actual media is only 3s -- shorter than the planned 5s trim range.
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=3.0))

        with self.assertRaises(TimelineIntegrationError) as ctx:
            build_reel(
                PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
            )
        self.assertIn("clip_video_01", str(ctx.exception))

    def test_trim_range_shorter_than_actual_file_passes(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=3.0)
        timeline_path = self._write_timeline(timeline)
        # Actual media is 5s but the timeline only uses the first 3s.
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertEqual(result.timeline["trimmed_clip_count"], 2)

    def test_negative_source_in_fails(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline.tracks[0].clips[0].source_in = -1.0
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        with self.assertRaises(TimelineIntegrationError) as ctx:
            build_reel(
                PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
            )
        self.assertIn("clip_video_01", str(ctx.exception))

    def test_source_out_not_greater_than_source_in_fails(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline.tracks[0].clips[0].source_out = timeline.tracks[0].clips[0].source_in
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
            )

    def test_planned_duration_mismatch_fails(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        # duration_seconds says 5.0 but source_out - source_in is 4.0.
        timeline.tracks[0].clips[0].source_out = 4.0
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        with self.assertRaises(TimelineIntegrationError) as ctx:
            build_reel(
                PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
            )
        self.assertIn("clip_video_01", str(ctx.exception))


class TimelinePlanningTests(TimelineTestCase):
    def test_full_compatible_clips_use_lossless_copy(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertFalse(result.normalized)

    def test_trimmed_clip_forces_normalized_render(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=3.0)
        timeline_path = self._write_timeline(timeline)
        # Actual media is 5s, timeline only uses the first 3s -> trim required.
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertTrue(result.normalized)
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        self.assertEqual(len(ffmpeg_calls), 1)
        self.assertIn("-filter_complex", ffmpeg_calls[0])
        self.assertIn("-ss", ffmpeg_calls[0])
        self.assertIn("-t", ffmpeg_calls[0])

    def test_plan_reasons_contain_timeline_reasons(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertTrue(any(r.startswith("timeline_") for r in payload["engine_reasons"]))

    def test_source_order_preserved_in_render(self) -> None:
        clip_paths = self._write_scenes(3)
        reversed_paths = list(reversed(clip_paths))
        timeline = self._build_timeline(reversed_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        ffprobe_calls = [c for c in runner.calls if Path(c[0]).name == "ffprobe"]
        probed_paths = [call[-1] for call in ffprobe_calls]
        self.assertEqual(probed_paths, [str(p) for p in reversed_paths])

    def test_each_input_path_appears_exactly_once(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=3.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        command = ffmpeg_calls[0]
        for path in clip_paths:
            self.assertEqual(command.count(str(path)), 1)


class TimelineMusicTests(TimelineTestCase):
    def test_timeline_music_does_not_auto_enable_mixing(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0, music_path="/music/bg.mp3")
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("music_mixer_callable must not be called without --music")

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            timeline_path=timeline_path,
            music_mixer_callable=_fail_if_called,
        )
        self.assertIsNone(result.music_mix)

    def test_warning_emitted_when_timeline_music_exists_without_cli_music(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0, music_path="/music/bg.mp3")
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        result = build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        self.assertTrue(any("bg.mp3" in w for w in result.timeline["warnings"]))
        self.assertTrue(any("bg.mp3" in w for w in result.warnings))

    def test_cli_music_overrides_timeline_music(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0, music_path="/music/timeline-track.mp3")
        timeline_path = self._write_timeline(timeline)
        cli_music_path = self.temp_dir / "cli-music.mp3"
        cli_music_path.write_bytes(b"y" * 100)

        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        probes = self._probes_for(clip_paths, duration=5.0)
        probes[str(intermediate_path.resolve())] = _music_video_probe_json()
        probes[str(cli_music_path.resolve())] = _music_track_probe_json()
        runner = FakeRunner(probes=probes)

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            timeline_path=timeline_path,
            music_path=cli_music_path,
            music_mixer_callable=FakeMixer(),
        )
        self.assertEqual(result.music_mix["music_path"], str(cli_music_path))

    def test_conflict_warning_recorded_when_cli_and_timeline_music_differ(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0, music_path="/music/timeline-track.mp3")
        timeline_path = self._write_timeline(timeline)
        cli_music_path = self.temp_dir / "cli-music.mp3"
        cli_music_path.write_bytes(b"y" * 100)

        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        probes = self._probes_for(clip_paths, duration=5.0)
        probes[str(intermediate_path.resolve())] = _music_video_probe_json()
        probes[str(cli_music_path.resolve())] = _music_track_probe_json()
        runner = FakeRunner(probes=probes)

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            timeline_path=timeline_path,
            music_path=cli_music_path,
            music_mixer_callable=FakeMixer(),
        )
        self.assertTrue(any("authoritative" in w for w in result.timeline["warnings"]))

    def test_existing_music_mixer_flow_unchanged_with_timeline(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)

        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        final_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        probes = self._probes_for(clip_paths, duration=5.0)
        probes[str(intermediate_path.resolve())] = _music_video_probe_json()
        probes[str(music_path.resolve())] = _music_track_probe_json()
        runner = FakeRunner(probes=probes)

        result = build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            timeline_path=timeline_path,
            music_path=music_path,
        )
        self.assertTrue(final_path.is_file())
        self.assertEqual(result.output_path, str(final_path))
        self.assertEqual(result.music_mix["status"], "success")


class TimelineBuildLogTests(TimelineTestCase):
    def test_timeline_disabled_field_without_timeline(self) -> None:
        clip_paths = self._write_five_scenes()
        runner = FakeRunner(probes=self._probes_for(clip_paths))
        build_reel(PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir)

        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["timeline"], {"enabled": False})

    def test_successful_timeline_fields_written(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        timeline_info = payload["timeline"]
        for field_name in (
            "enabled",
            "timeline_path",
            "timeline_id",
            "schema_version",
            "validation_passed",
            "production_date_match",
            "video_clip_count",
            "ignored_disabled_clips",
            "ignored_tracks",
            "planned_duration_seconds",
            "reconciled_duration_seconds",
            "duration_tolerance_seconds",
            "trimmed_clip_count",
            "warnings",
        ):
            self.assertIn(field_name, timeline_info)
        self.assertTrue(timeline_info["validation_passed"])
        self.assertEqual(timeline_info["video_clip_count"], 3)

    def test_failure_fields_written(self) -> None:
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=FakeRunner(),
                root=self.temp_dir,
                timeline_path=self.temp_dir / "does_not_exist.json",
            )
        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        timeline_info = payload["timeline"]
        self.assertTrue(timeline_info["enabled"])
        self.assertFalse(timeline_info["validation_passed"])
        self.assertIn("error", timeline_info)

    def test_trim_count_recorded(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=3.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["timeline"]["trimmed_clip_count"], 2)

    def test_duration_values_recorded(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["timeline"]["planned_duration_seconds"], 10.0)
        self.assertEqual(payload["timeline"]["reconciled_duration_seconds"], 10.0)

    def test_original_build_log_fields_preserved_with_timeline(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
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
            "engine_plan_type",
            "engine_reasons",
        ):
            self.assertIn(field_name, payload)
        self.assertEqual(payload["result"], "success")

    def test_music_mix_section_preserved_with_timeline(self) -> None:
        clip_paths = self._write_scenes(3)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)

        intermediate_path = self.config.reel_without_music_path(PRODUCTION_DATE, root=self.temp_dir)
        probes = self._probes_for(clip_paths, duration=5.0)
        probes[str(intermediate_path.resolve())] = _music_video_probe_json()
        probes[str(music_path.resolve())] = _music_track_probe_json()
        runner = FakeRunner(probes=probes)

        build_reel(
            PRODUCTION_DATE,
            self.config,
            runner=runner,
            root=self.temp_dir,
            timeline_path=timeline_path,
            music_path=music_path,
        )
        log_path = self.config.build_log_path(PRODUCTION_DATE, root=self.temp_dir)
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["music_mix"]["enabled"])
        self.assertEqual(payload["music_mix"]["status"], "success")


class TimelineSafetyTests(TimelineTestCase):
    def test_invalid_timeline_prevents_video_engine_call(self) -> None:
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=FakeRunner(),
                root=self.temp_dir,
                timeline_path=self.temp_dir / "does_not_exist.json",
            )
        output_path = self.config.reel_final_path(PRODUCTION_DATE, root=self.temp_dir)
        self.assertFalse(output_path.exists())

    def test_invalid_timeline_prevents_music_mixer_call(self) -> None:
        music_path = self.temp_dir / "music.mp3"
        music_path.write_bytes(b"y" * 100)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("music_mixer_callable must not be called when the timeline is invalid")

        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=FakeRunner(),
                root=self.temp_dir,
                timeline_path=self.temp_dir / "does_not_exist.json",
                music_path=music_path,
                music_mixer_callable=_fail_if_called,
            )

    def test_no_fallback_to_directory_discovery_on_invalid_timeline(self) -> None:
        # Real scene files exist on disk (discoverable the old way), but
        # the --timeline file is invalid -- must fail, never silently
        # fall back to scanning the directory.
        self._write_five_scenes()
        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE,
                self.config,
                runner=FakeRunner(),
                root=self.temp_dir,
                timeline_path=self.temp_dir / "does_not_exist.json",
            )

    def test_no_automatic_retry_on_reconciliation_failure(self) -> None:
        clip_paths = self._write_scenes(2)
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=3.0))

        with self.assertRaises(TimelineIntegrationError):
            build_reel(
                PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
            )
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        self.assertEqual(ffmpeg_calls, [])

    def test_no_input_media_mutation(self) -> None:
        clip_paths = self._write_scenes(3)
        original_bytes = [path.read_bytes() for path in clip_paths]
        timeline = self._build_timeline(clip_paths, scene_duration=5.0)
        timeline_path = self._write_timeline(timeline)
        runner = FakeRunner(probes=self._probes_for(clip_paths, duration=5.0))

        build_reel(
            PRODUCTION_DATE, self.config, runner=runner, root=self.temp_dir, timeline_path=timeline_path
        )
        for path, original in zip(clip_paths, original_bytes):
            self.assertEqual(path.read_bytes(), original)


class TimelineIntegrationStructuralSafetyTests(unittest.TestCase):
    def test_no_playwright_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("import playwright"))
            self.assertFalse(stripped.startswith("from playwright"))

    def test_no_publishing_or_social_import(self) -> None:
        for line in MODULE_SOURCE.splitlines():
            stripped = line.strip()
            self.assertFalse(stripped.startswith("from src.publishing"))
            self.assertFalse(stripped.startswith("from .publishing"))
            self.assertFalse(stripped.startswith("from src.social"))
            self.assertFalse(stripped.startswith("from .social"))

    def test_no_automatic_timeline_discovery(self) -> None:
        for forbidden in ("glob(", "rglob(", "listdir(", "default_timeline"):
            self.assertNotIn(forbidden, MODULE_SOURCE)


class MusicIntegrationStructuralSafetyTests(unittest.TestCase):
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

    def test_no_automatic_music_selection(self) -> None:
        for forbidden in ("random.choice", "glob(", "rglob(", "listdir(", "default_music"):
            self.assertNotIn(forbidden, MODULE_SOURCE)


if __name__ == "__main__":
    unittest.main()
