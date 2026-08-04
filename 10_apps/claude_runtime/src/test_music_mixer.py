from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from .media_inspector import AudioStreamInfo, MediaInfo, VideoStreamInfo
from .music_mixer import (
    DISALLOWED_ACTIONS,
    AudioMixDecision,
    DuckingMode,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    FFmpegTimeoutError,
    MusicDurationError,
    MusicFileEmptyError,
    MusicFileNotFoundError,
    MusicInputError,
    MusicMixerConfig,
    MusicMixerConfigError,
    MusicMixPlanError,
    MusicMixRequest,
    MusicMode,
    MusicOutputExistsError,
    MusicOutputVerificationError,
    MusicTooShortError,
    ProcessResult,
    UnsafeMusicOutputError,
    UnsupportedMusicExtensionError,
    _validate_paths,
    build_music_mix_plan,
    default_mixer_config_path,
    default_runner,
    execute_music_mix_plan,
    load_music_mixer_config,
    mix_music,
)

MODULE_PATH = Path(__file__).resolve().parent / "music_mixer.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fake runner — no real ffmpeg/ffprobe is ever invoked in these tests.
# ---------------------------------------------------------------------------


class FakeRunner:
    def __init__(
        self,
        *,
        probes: dict[str, dict] | None = None,
        ffmpeg_should_fail: bool = False,
        ffmpeg_write_output: bool = True,
        ffmpeg_output_bytes: bytes = b"fake-mp4-bytes",
        raise_exc: Exception | None = None,
    ) -> None:
        self.probes = probes or {}
        self.ffmpeg_should_fail = ffmpeg_should_fail
        self.ffmpeg_write_output = ffmpeg_write_output
        self.ffmpeg_output_bytes = ffmpeg_output_bytes
        self.raise_exc = raise_exc
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, timeout: int) -> ProcessResult:
        self.calls.append(command)

        if self.raise_exc is not None:
            raise self.raise_exc

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


# ---------------------------------------------------------------------------
# ffprobe-shaped fixtures
# ---------------------------------------------------------------------------


def _video_probe_json(*, duration: float = 12.5, has_audio: bool = True, width: int = 1080, height: int = 1920) -> dict:
    streams = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": width,
            "height": height,
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
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": str(duration),
            "bit_rate": "8000000",
        },
        "chapters": [],
    }


def _music_probe_json(*, duration: float = 8.0) -> dict:
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


# ---------------------------------------------------------------------------
# MediaInfo fixture builders (for pure planning tests — no ffprobe at all)
# ---------------------------------------------------------------------------


def _video_info(path: Path, *, duration: float = 12.5, has_audio: bool = True) -> MediaInfo:
    video_streams = [
        VideoStreamInfo(stream_index=0, codec_name="h264", width=1080, height=1920, fps=30.0, duration_seconds=duration)
    ]
    audio_streams = (
        [AudioStreamInfo(stream_index=1, codec_name="aac", sample_rate=48000, channels=2, duration_seconds=duration)]
        if has_audio
        else []
    )
    return MediaInfo(
        path=str(path),
        filename=path.name,
        extension=path.suffix.lower(),
        file_size_bytes=1000,
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        format_long_name="QuickTime / MOV",
        duration_seconds=duration,
        overall_bitrate=8_000_000,
        start_time_seconds=0.0,
        stream_count=len(video_streams) + len(audio_streams),
        video_streams=video_streams,
        audio_streams=audio_streams,
        chapters=[],
        has_video=True,
        has_audio=has_audio,
        primary_video_stream_index=0,
        primary_audio_stream_index=1 if has_audio else None,
        warnings=[],
    )


def _music_info(path: Path, *, duration: float = 8.0) -> MediaInfo:
    audio_streams = [AudioStreamInfo(stream_index=0, codec_name="mp3", sample_rate=44100, channels=2, duration_seconds=duration)]
    return MediaInfo(
        path=str(path),
        filename=path.name,
        extension=path.suffix.lower(),
        file_size_bytes=1000,
        format_name="mp3",
        format_long_name="MP2/3 (MPEG audio layer 2/3)",
        duration_seconds=duration,
        overall_bitrate=192_000,
        start_time_seconds=0.0,
        stream_count=1,
        video_streams=[],
        audio_streams=audio_streams,
        chapters=[],
        has_video=False,
        has_audio=True,
        primary_video_stream_index=None,
        primary_audio_stream_index=0,
        warnings=[],
    )


class MusicMixerTempTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir_ctx = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir_ctx.name)
        self.addCleanup(self._temp_dir_ctx.cleanup)
        self.config = load_music_mixer_config()

    def _write_file(self, name: str, *, size: int = 1000) -> Path:
        path = self.temp_dir / name
        path.write_bytes(b"x" * size)
        return path

    def _video(self, name: str = "video.mp4", **kwargs) -> Path:
        return self._write_file(name, **kwargs)

    def _music(self, name: str = "music.mp3", **kwargs) -> Path:
        return self._write_file(name, **kwargs)

    def _request(self, **overrides) -> MusicMixRequest:
        defaults = dict(
            video_path=self.temp_dir / "video.mp4",
            music_path=self.temp_dir / "music.mp3",
            output_path=self.temp_dir / "output.mp4",
            force=False,
        )
        defaults.update(overrides)
        return MusicMixRequest(**defaults)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_path_exists(self) -> None:
        self.assertTrue(default_mixer_config_path().is_file())

    def test_config_loads_with_expected_defaults(self) -> None:
        config = load_music_mixer_config()
        self.assertEqual(config.default_music_mode, MusicMode.LOOP)
        self.assertAlmostEqual(config.default_music_volume, 0.20)
        self.assertAlmostEqual(config.default_source_audio_volume, 1.00)
        self.assertEqual(config.default_ducking_mode, DuckingMode.NONE)
        self.assertAlmostEqual(config.fixed_music_multiplier, 0.55)
        self.assertTrue(config.preserve_source_audio)
        self.assertTrue(config.faststart)
        self.assertTrue(config.shortest)
        self.assertEqual(config.audio_codec, "aac")

    def test_missing_config_fails_clearly(self) -> None:
        with self.assertRaises(MusicMixerConfigError):
            load_music_mixer_config("/nonexistent/path/music_mixer.yaml")

    def test_cli_override_takes_precedence_over_config(self) -> None:
        config = load_music_mixer_config()
        video_info = _video_info(Path("/tmp/video.mp4"))
        music_info = _music_info(Path("/tmp/music.mp3"))
        request = MusicMixRequest(
            video_path=Path("/tmp/video.mp4"),
            music_path=Path("/tmp/music.mp3"),
            output_path=Path("/tmp/output.mp4"),
            music_volume=0.9,
        )
        plan = build_music_mix_plan(request, video_info, music_info, config)
        self.assertAlmostEqual(plan.music_volume, 0.9)
        self.assertNotAlmostEqual(plan.music_volume, config.default_music_volume)

    def test_invalid_volume_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "music_mixer.yaml"
            config_path.write_text("mix:\n  music_volume: -1\n", encoding="utf-8")
            with self.assertRaises(MusicMixerConfigError):
                load_music_mixer_config(config_path)

    def test_invalid_mode_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "music_mixer.yaml"
            config_path.write_text("mix:\n  music_mode: bogus\n", encoding="utf-8")
            with self.assertRaises(MusicMixerConfigError):
                load_music_mixer_config(config_path)

    def test_invalid_fade_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "music_mixer.yaml"
            config_path.write_text("mix:\n  fade_in_seconds: -0.5\n", encoding="utf-8")
            with self.assertRaises(MusicMixerConfigError):
                load_music_mixer_config(config_path)

    def test_copy_video_stream_false_not_implemented(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            config_path = Path(temp_name) / "music_mixer.yaml"
            config_path.write_text("output:\n  copy_video_stream: false\n", encoding="utf-8")
            with self.assertRaises(MusicMixerConfigError):
                load_music_mixer_config(config_path)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class InputValidationTests(MusicMixerTempTestCase):
    def test_missing_video_fails(self) -> None:
        self._music()
        request = self._request()
        with self.assertRaises(MusicInputError):
            _validate_paths(request, self.config)

    def test_empty_video_fails(self) -> None:
        self._video(size=0)
        self._music()
        request = self._request()
        with self.assertRaises(MusicInputError):
            _validate_paths(request, self.config)

    def test_missing_music_fails(self) -> None:
        self._video()
        request = self._request()
        with self.assertRaises(MusicFileNotFoundError):
            _validate_paths(request, self.config)

    def test_empty_music_fails(self) -> None:
        self._video()
        self._music(size=0)
        request = self._request()
        with self.assertRaises(MusicFileEmptyError):
            _validate_paths(request, self.config)

    def test_unsupported_video_extension_fails(self) -> None:
        self._video(name="video.webm")
        self._music()
        request = self._request(video_path=self.temp_dir / "video.webm")
        with self.assertRaises(MusicInputError):
            _validate_paths(request, self.config)

    def test_unsupported_music_extension_fails(self) -> None:
        self._video()
        self._music(name="music.ogg")
        request = self._request(music_path=self.temp_dir / "music.ogg")
        with self.assertRaises(UnsupportedMusicExtensionError):
            _validate_paths(request, self.config)

    def test_video_and_music_same_path_rejected(self) -> None:
        video_path = self._video()
        request = self._request(music_path=video_path)
        with self.assertRaises(MusicInputError):
            _validate_paths(request, self.config)

    def test_output_equals_video_fails(self) -> None:
        video_path = self._video()
        self._music()
        request = self._request(output_path=video_path)
        with self.assertRaises(UnsafeMusicOutputError):
            _validate_paths(request, self.config)

    def test_output_equals_music_fails(self) -> None:
        self._video()
        music_path = self._music()
        request = self._request(output_path=music_path)
        with self.assertRaises(UnsafeMusicOutputError):
            _validate_paths(request, self.config)

    def test_existing_output_fails_without_force(self) -> None:
        self._video()
        self._music()
        output_path = self.temp_dir / "output.mp4"
        output_path.write_bytes(b"existing")
        request = self._request(force=False)
        with self.assertRaises(MusicOutputExistsError):
            _validate_paths(request, self.config)

    def test_force_allows_overwrite_flag(self) -> None:
        self._video()
        self._music()
        output_path = self.temp_dir / "output.mp4"
        output_path.write_bytes(b"existing")
        request = self._request(force=True)
        _validate_paths(request, self.config)  # does not raise

    def test_resolved_symlink_collision_rejected(self) -> None:
        video_path = self._video()
        self._music()
        symlinked_output = self.temp_dir / "link_to_video.mp4"
        os.symlink(video_path, symlinked_output)
        request = self._request(output_path=symlinked_output)
        with self.assertRaises(UnsafeMusicOutputError):
            _validate_paths(request, self.config)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class PlanningTests(MusicMixerTempTestCase):
    def _plan(self, *, video_duration=12.5, music_duration=8.0, video_has_audio=True, **request_overrides):
        video_path = self.temp_dir / "video.mp4"
        music_path = self.temp_dir / "music.mp3"
        video_info = _video_info(video_path, duration=video_duration, has_audio=video_has_audio)
        music_info = _music_info(music_path, duration=music_duration)
        request = self._request(**request_overrides)
        return build_music_mix_plan(request, video_info, music_info, self.config)

    def test_long_music_in_trim_mode_trims(self) -> None:
        plan = self._plan(video_duration=10.0, music_duration=20.0, music_mode=MusicMode.TRIM)
        self.assertTrue(plan.music_trim_required)
        self.assertFalse(plan.music_loop_required)

    def test_short_music_in_trim_mode_fails(self) -> None:
        with self.assertRaises(MusicTooShortError):
            self._plan(video_duration=20.0, music_duration=10.0, music_mode=MusicMode.TRIM)

    def test_short_music_in_loop_mode_loops(self) -> None:
        plan = self._plan(video_duration=20.0, music_duration=10.0, music_mode=MusicMode.LOOP)
        self.assertTrue(plan.music_loop_required)
        self.assertFalse(plan.music_trim_required)

    def test_long_music_in_loop_mode_trims(self) -> None:
        plan = self._plan(video_duration=10.0, music_duration=20.0, music_mode=MusicMode.LOOP)
        self.assertTrue(plan.music_trim_required)
        self.assertFalse(plan.music_loop_required)

    def test_exact_duration_match_requires_neither(self) -> None:
        plan = self._plan(video_duration=10.0, music_duration=10.0, music_mode=MusicMode.LOOP)
        self.assertFalse(plan.music_loop_required)
        self.assertFalse(plan.music_trim_required)

    def test_video_with_source_audio_uses_amix(self) -> None:
        plan = self._plan(video_has_audio=True)
        self.assertEqual(plan.audio_decision.case, "mix_source_and_music")
        self.assertTrue(plan.audio_decision.use_amix)

    def test_video_without_source_audio_uses_music_only(self) -> None:
        plan = self._plan(video_has_audio=False)
        self.assertEqual(plan.audio_decision.case, "music_only_no_source_audio")
        self.assertFalse(plan.audio_decision.use_amix)

    def test_preserve_source_audio_false_discards_source(self) -> None:
        config = MusicMixerConfig(preserve_source_audio=False)
        video_info = _video_info(self.temp_dir / "video.mp4", has_audio=True)
        music_info = _music_info(self.temp_dir / "music.mp3")
        request = self._request()
        plan = build_music_mix_plan(request, video_info, music_info, config)
        self.assertEqual(plan.audio_decision.case, "music_only_discarded_source")
        self.assertFalse(plan.audio_decision.use_amix)

    def test_fixed_ducking_multiplier_applied(self) -> None:
        plan = self._plan(video_has_audio=True, ducking_mode=DuckingMode.FIXED, music_volume=0.5)
        expected = 0.5 * self.config.fixed_music_multiplier
        self.assertIn(f"volume={expected}", "".join(plan.command))

    def test_none_ducking_keeps_configured_volume(self) -> None:
        plan = self._plan(video_has_audio=True, ducking_mode=DuckingMode.NONE, music_volume=0.5)
        self.assertIn("volume=0.5", "".join(plan.command))

    def test_fade_positions_calculated_correctly(self) -> None:
        plan = self._plan(video_duration=10.0, fade_in_seconds=1.0, fade_out_seconds=2.0)
        self.assertEqual(plan.fade_in_seconds, 1.0)
        self.assertEqual(plan.fade_out_seconds, 2.0)
        joined = "".join(plan.command)
        self.assertIn("afade=t=in:st=0:d=1.0", joined)
        self.assertIn("afade=t=out:st=8.0:d=2.0", joined)

    def test_excessive_fades_clamp_with_warning(self) -> None:
        plan = self._plan(video_duration=1.0, fade_in_seconds=5.0, fade_out_seconds=5.0)
        self.assertLessEqual(plan.fade_in_seconds + plan.fade_out_seconds, 1.0 + 1e-9)
        self.assertTrue(any("clamp" in w for w in plan.warnings))

    def test_zero_fades_disable_filters(self) -> None:
        plan = self._plan(fade_in_seconds=0.0, fade_out_seconds=0.0)
        joined = "".join(plan.command)
        self.assertNotIn("afade", joined)

    def test_deterministic_command(self) -> None:
        plan_a = self._plan()
        plan_b = self._plan()
        self.assertEqual(plan_a.command, plan_b.command)

    def test_invalid_music_mode_in_request_rejected(self) -> None:
        with self.assertRaises(MusicMixPlanError):
            self._plan(music_mode="bogus")

    def test_missing_duration_metadata_fails(self) -> None:
        video_path = self.temp_dir / "video.mp4"
        music_path = self.temp_dir / "music.mp3"
        video_info = _video_info(video_path, duration=12.5)
        video_info.duration_seconds = None
        music_info = _music_info(music_path)
        request = self._request()
        with self.assertRaises(MusicDurationError):
            build_music_mix_plan(request, video_info, music_info, self.config)


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


class CommandTests(MusicMixerTempTestCase):
    def _plan(self, **request_overrides):
        video_path = self.temp_dir / "video.mp4"
        music_path = self.temp_dir / "music.mp3"
        video_info = _video_info(video_path, duration=12.5, has_audio=True)
        music_info = _music_info(music_path, duration=8.0)
        request = self._request(**request_overrides)
        return build_music_mix_plan(request, video_info, music_info, self.config)

    def test_command_is_list(self) -> None:
        plan = self._plan()
        self.assertIsInstance(plan.command, list)
        self.assertTrue(all(isinstance(token, str) for token in plan.command))

    def test_shell_true_never_used_in_source(self) -> None:
        self.assertNotIn("shell=True", MODULE_SOURCE)

    def test_video_input_appears_once(self) -> None:
        plan = self._plan()
        self.assertEqual(plan.command.count(str(plan.video_path)), 1)

    def test_music_input_appears_once(self) -> None:
        plan = self._plan()
        self.assertEqual(plan.command.count(str(plan.music_path)), 1)

    def test_output_appears_once(self) -> None:
        plan = self._plan()
        self.assertEqual(plan.command.count(str(plan.output_path)), 1)

    def test_loop_mode_uses_stream_loop(self) -> None:
        plan = self._plan(music_mode=MusicMode.LOOP)  # music (8s) shorter than video (12.5s) -> loop
        self.assertIn("-stream_loop", plan.command)
        self.assertIn("-1", plan.command)

    def test_trim_mode_does_not_use_stream_loop(self) -> None:
        video_path = self.temp_dir / "video.mp4"
        music_path = self.temp_dir / "music.mp3"
        video_info = _video_info(video_path, duration=5.0, has_audio=True)
        music_info = _music_info(music_path, duration=8.0)  # longer -> trim, not loop
        request = self._request(music_mode=MusicMode.TRIM)
        plan = build_music_mix_plan(request, video_info, music_info, self.config)
        self.assertNotIn("-stream_loop", plan.command)

    def test_music_only_path_maps_music_audio_without_amix(self) -> None:
        video_path = self.temp_dir / "video.mp4"
        music_path = self.temp_dir / "music.mp3"
        video_info = _video_info(video_path, duration=12.5, has_audio=False)
        music_info = _music_info(music_path, duration=8.0)
        request = self._request()
        plan = build_music_mix_plan(request, video_info, music_info, self.config)
        joined = "".join(plan.command)
        self.assertNotIn("amix", joined)
        self.assertIn("[aout]", joined)

    def test_mixed_path_maps_amix_result(self) -> None:
        plan = self._plan()  # video has audio by default
        joined = "".join(plan.command)
        self.assertIn("amix", joined)
        self.assertIn("[src][music]amix", joined)

    def test_configured_aac_bitrate_rate_channels_used(self) -> None:
        config = MusicMixerConfig(audio_codec="aac", audio_bitrate="256k", audio_sample_rate=44100, audio_channels=1)
        video_info = _video_info(self.temp_dir / "video.mp4")
        music_info = _music_info(self.temp_dir / "music.mp3")
        request = self._request()
        plan = build_music_mix_plan(request, video_info, music_info, config)
        self.assertIn("aac", plan.command)
        self.assertIn("256k", plan.command)
        self.assertIn("44100", plan.command)
        self.assertIn("1", plan.command)

    def test_c_v_copy_used(self) -> None:
        plan = self._plan()
        index = plan.command.index("-c:v")
        self.assertEqual(plan.command[index + 1], "copy")

    def test_faststart_included_when_enabled(self) -> None:
        config = MusicMixerConfig(faststart=True)
        video_info = _video_info(self.temp_dir / "video.mp4")
        music_info = _music_info(self.temp_dir / "music.mp3")
        request = self._request()
        plan = build_music_mix_plan(request, video_info, music_info, config)
        self.assertIn("-movflags", plan.command)
        self.assertIn("+faststart", plan.command)

    def test_shortest_included_when_enabled(self) -> None:
        config = MusicMixerConfig(shortest=True)
        video_info = _video_info(self.temp_dir / "video.mp4")
        music_info = _music_info(self.temp_dir / "music.mp3")
        request = self._request()
        plan = build_music_mix_plan(request, video_info, music_info, config)
        self.assertIn("-shortest", plan.command)

    def test_force_uses_dash_y(self) -> None:
        plan = self._plan(force=True)
        self.assertIn("-y", plan.command)
        self.assertNotIn("-n", plan.command)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionTests(MusicMixerTempTestCase):
    def _run_mix(self, *, video_duration=12.5, music_duration=8.0, runner=None, **request_overrides):
        video_path = self._video()
        music_path = self._music()
        probes = {str(video_path.resolve()): _video_probe_json(duration=video_duration), str(music_path.resolve()): _music_probe_json(duration=music_duration)}
        runner = runner or FakeRunner(probes=probes)
        request = self._request(**request_overrides)
        return mix_music(request, self.config, runner=runner), runner

    def test_successful_execution_verifies_non_zero_output(self) -> None:
        result, runner = self._run_mix()
        self.assertTrue(result.output_exists)
        self.assertGreater(result.output_size_bytes, 0)
        self.assertEqual(result.return_code, 0)
        self.assertIsNone(result.error)

    def test_missing_output_fails(self) -> None:
        video_path = self._video()
        music_path = self._music()
        probes = {str(video_path.resolve()): _video_probe_json(), str(music_path.resolve()): _music_probe_json()}
        runner = FakeRunner(probes=probes, ffmpeg_write_output=False)
        request = self._request()
        with self.assertRaises(MusicOutputVerificationError):
            mix_music(request, self.config, runner=runner)

    def test_zero_byte_output_fails(self) -> None:
        video_path = self._video()
        music_path = self._music()
        probes = {str(video_path.resolve()): _video_probe_json(), str(music_path.resolve()): _music_probe_json()}
        runner = FakeRunner(probes=probes, ffmpeg_output_bytes=b"")
        request = self._request()
        with self.assertRaises(MusicOutputVerificationError):
            mix_music(request, self.config, runner=runner)

    def test_timeout_raises_clear_error(self) -> None:
        video_path = self._video()
        music_path = self._music()
        probes = {str(video_path.resolve()): _video_probe_json(), str(music_path.resolve()): _music_probe_json()}
        runner = FakeRunner(probes=probes, raise_exc=FFmpegTimeoutError("simulated timeout"))
        request = self._request()
        with self.assertRaises(FFmpegTimeoutError):
            mix_music(request, self.config, runner=runner)

    def test_non_zero_return_raises_clear_error(self) -> None:
        video_path = self._video()
        music_path = self._music()
        probes = {str(video_path.resolve()): _video_probe_json(), str(music_path.resolve()): _music_probe_json()}
        runner = FakeRunner(probes=probes, ffmpeg_should_fail=True)
        request = self._request()
        with self.assertRaises(FFmpegExecutionError):
            mix_music(request, self.config, runner=runner)

    def test_missing_binary_raises_clear_error(self) -> None:
        with self.assertRaises(FFmpegNotFoundError):
            default_runner(["/definitely/not/a/real/binary/ffmpeg_xyz"], timeout=5)

    def test_runner_called_expected_number_of_times_on_failure(self) -> None:
        video_path = self._video()
        music_path = self._music()
        probes = {str(video_path.resolve()): _video_probe_json(), str(music_path.resolve()): _music_probe_json()}
        runner = FakeRunner(probes=probes, ffmpeg_should_fail=True)
        request = self._request()
        with self.assertRaises(FFmpegExecutionError):
            mix_music(request, self.config, runner=runner)
        # Exactly 2 ffprobe calls (video + music) + 1 ffmpeg call — no
        # automatic fallback or retry of any of them.
        ffmpeg_calls = [c for c in runner.calls if Path(c[0]).name == "ffmpeg"]
        self.assertEqual(len(ffmpeg_calls), 1)

    def test_diagnostics_log_is_atomic_json(self) -> None:
        result, runner = self._run_mix()
        output_path = self.temp_dir / "output.mp4"
        log_path = self.temp_dir / f"output{self.config.log_filename_suffix}"
        self.assertTrue(log_path.is_file())
        self.assertFalse(log_path.with_suffix(log_path.suffix + ".tmp").exists())

        payload = json.loads(log_path.read_text(encoding="utf-8"))
        for field_name in (
            "schema_version",
            "started_at",
            "finished_at",
            "video_path",
            "music_path",
            "output_path",
            "video_duration_seconds",
            "music_duration_seconds",
            "video_has_audio",
            "music_mode",
            "music_loop_required",
            "music_trim_required",
            "music_volume",
            "source_audio_volume",
            "ducking_mode",
            "fade_in_seconds",
            "fade_out_seconds",
            "command",
            "return_code",
            "output_size_bytes",
            "warnings",
            "result",
            "error",
        ):
            self.assertIn(field_name, payload)
        self.assertEqual(payload["result"], "success")

    def test_diagnostics_log_written_on_failure(self) -> None:
        video_path = self._video()
        music_path = self._music()
        probes = {str(video_path.resolve()): _video_probe_json(), str(music_path.resolve()): _music_probe_json()}
        runner = FakeRunner(probes=probes, ffmpeg_should_fail=True)
        request = self._request()
        with self.assertRaises(FFmpegExecutionError):
            mix_music(request, self.config, runner=runner)

        log_path = self.temp_dir / f"output{self.config.log_filename_suffix}"
        self.assertTrue(log_path.is_file())
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["result"], "failed")
        self.assertIsNotNone(payload["error"])

    def test_inputs_remain_byte_for_byte_unchanged(self) -> None:
        video_path = self._video()
        music_path = self._music()
        video_before = video_path.read_bytes()
        music_before = music_path.read_bytes()

        self._run_mix(video_duration=12.5, music_duration=8.0)

        self.assertEqual(video_path.read_bytes(), video_before)
        self.assertEqual(music_path.read_bytes(), music_before)


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

    def test_no_network_download_code(self) -> None:
        for forbidden in ("requests.get(", "requests.post(", "urlopen(", "urllib.request"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_subtitle_logo_transition_implementation(self) -> None:
        # Checked as actual ffmpeg filter/argument signatures, not bare
        # words — this module's own docstrings/comments legitimately name
        # "subtitles"/"logo"/"transition" when documenting what it does
        # NOT implement (same false-fail risk fixed in prior phases).
        for forbidden in ("subtitles=", "drawtext", "watermark.png", "xfade"):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_no_shell_true(self) -> None:
        self.assertNotIn("shell=True", MODULE_SOURCE)

    def test_no_write_mode_open_against_inputs(self) -> None:
        self.assertNotIn(".write_bytes(", MODULE_SOURCE)

    def test_no_automatic_music_selection(self) -> None:
        for forbidden in ("random.choice", "glob(", "rglob(", "listdir("):
            self.assertNotIn(forbidden, MODULE_SOURCE)

    def test_disallowed_actions_constant_covers_spec(self) -> None:
        self.assertIn("publish", DISALLOWED_ACTIONS)
        self.assertIn("select_music", DISALLOWED_ACTIONS)
        self.assertIn("download_music", DISALLOWED_ACTIONS)
        self.assertIn("add_subtitles", DISALLOWED_ACTIONS)
        self.assertIn("generate_voice", DISALLOWED_ACTIONS)


if __name__ == "__main__":
    unittest.main()
