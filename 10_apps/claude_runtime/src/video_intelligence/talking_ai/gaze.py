"""GazeAnalyzer -- DNA field gaze (task §7). Naturalness concerns may
include a constant unbroken stare, mechanically periodic gaze shifts,
or zero gaze variation -- never infers emotional state from gaze alone.
"""
from __future__ import annotations

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, ratio_true, sorted_by_timestamp, variability
from .evidence import GazeDirection
from .models import TalkingAIMetricResult

TRAIT_NAME = "gaze"

_GAZE_SHIFT_NEAR_PAUSE_WINDOW_SECONDS = 0.5


class GazeAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = sorted_by_timestamp(evidence_for_video(context.evidence, context.video_id))
        gaze_evidence = [item for item in own_evidence if item.gaze_direction is not None]
        eye_contact_evidence = [item for item in own_evidence if item.eye_contact is not None]

        direct_flags = [item.gaze_direction == GazeDirection.DIRECT_CAMERA for item in gaze_evidence]

        # runs of consecutive same-direction observations, to derive
        # away-duration / fixed-stare-duration / return-to-camera behavior
        runs: list[tuple[str, float, float]] = []  # (direction, start_ts, end_ts)
        for item in gaze_evidence:
            if runs and runs[-1][0] == item.gaze_direction:
                runs[-1] = (runs[-1][0], runs[-1][1], item.timestamp_seconds)
            else:
                runs.append((item.gaze_direction, item.timestamp_seconds, item.timestamp_seconds))

        away_runs = [run for run in runs if run[0] != GazeDirection.DIRECT_CAMERA]
        direct_runs = [run for run in runs if run[0] == GazeDirection.DIRECT_CAMERA]
        away_durations = [end - start for _direction, start, end in away_runs]
        direct_durations = [end - start for _direction, start, end in direct_runs]

        returns_to_camera = 0
        for index, (direction, _start, _end) in enumerate(runs[:-1]):
            if direction != GazeDirection.DIRECT_CAMERA and runs[index + 1][0] == GazeDirection.DIRECT_CAMERA:
                returns_to_camera += 1

        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        pause_windows = [
            (segment.start_seconds - segment.pause_before, segment.start_seconds)
            for segment in own_segments
            if segment.pause_before is not None and segment.pause_before >= context.config.pause_min_seconds
        ]
        shift_timestamps = [run[1] for run in runs[1:]]  # start of every run after the first = a gaze shift
        shifts_near_pause = sum(
            1 for ts in shift_timestamps
            if any(start - _GAZE_SHIFT_NEAR_PAUSE_WINDOW_SECONDS <= ts <= end + _GAZE_SHIFT_NEAR_PAUSE_WINDOW_SECONDS for start, end in pause_windows)
        )

        if direct_flags:
            direct_camera_gaze_ratio = ratio_true(direct_flags)
        elif eye_contact_evidence:
            direct_camera_gaze_ratio = ratio_true([item.eye_contact for item in eye_contact_evidence])
        else:
            direct_camera_gaze_ratio = None

        metrics = {
            "direct_camera_gaze_ratio": direct_camera_gaze_ratio,
            "gaze_away_frequency": (len(away_runs) / len(runs)) if runs else None,
            "average_gaze_away_duration": (sum(away_durations) / len(away_durations)) if away_durations else None,
            "return_to_camera_ratio": (returns_to_camera / len(away_runs)) if away_runs else None,
            "gaze_shift_near_pause_ratio": (shifts_near_pause / len(shift_timestamps)) if shift_timestamps else None,
            "fixed_stare_duration": max(direct_durations) if direct_durations else None,
            "eye_contact_variability": variability([1.0 if item.eye_contact else 0.0 for item in eye_contact_evidence]),
            "gaze_shift_count": float(len(runs) - 1) if runs else 0.0,
        }

        evidence_count = len(gaze_evidence) + len(eye_contact_evidence)
        confidence = confidence_for(context, evidence_count=evidence_count, corroboration_count=evidence_count)
        score = min(1.0, evidence_count / (evidence_count + 2)) if evidence_count else 0.0
        rationale = (
            f"Gaze pattern observed across {evidence_count} evidence item(s)."
            if evidence_count
            else "No gaze evidence supplied; confidence is unknown."
        )
        evidence_ids = sorted({item.evidence_id for item in gaze_evidence} | {item.evidence_id for item in eye_contact_evidence})
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=score, confidence=confidence,
            sample_count=evidence_count, evidence_ids=evidence_ids, rationale=rationale,
        )
