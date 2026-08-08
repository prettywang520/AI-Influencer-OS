"""LipSyncAnalyzer -- DNA field lip_sync (task §4). Every result is
honestly labeled "coarse-timing lip sync" -- this schema defines no
phoneme-level evidence type at all, so no code path here could ever
claim phoneme-level validation.
"""
from __future__ import annotations

from statistics import mean

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, score_from_ratio, variability
from .evidence import SpeechSegment
from .models import TalkingAIMetricResult
from .timing import mouth_motion_events, nearest_event_after, nearest_event_near, speech_active_intervals

TRAIT_NAME = "lip_sync"

_MOUTH_CLOSED_THRESHOLD = 0.15
_JAW_MOTION_THRESHOLD = 0.10
_OVER_ARTICULATION_THRESHOLD = 0.70
_UNDER_ARTICULATION_THRESHOLD = 0.05


def _pause_windows(segments: list[SpeechSegment], pause_min_seconds: float) -> list[tuple[float, float]]:
    windows = []
    for segment in segments:
        if segment.pause_before is not None and segment.pause_before >= pause_min_seconds:
            windows.append((segment.start_seconds - segment.pause_before, segment.start_seconds))
    return windows


class LipSyncAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        own_evidence = evidence_for_video(context.evidence, context.video_id)
        own_segments = [segment for segment in context.speech_segments if segment.video_id == context.video_id]
        speech_intervals = speech_active_intervals(own_segments)
        mouth_events = mouth_motion_events(own_evidence)

        max_window = context.config.maximum_reasonable_sync_latency_seconds
        search_window = max(max_window * 4, 1.0)

        onset_latencies: list[float] = []
        onset_hits = onset_checked = 0
        offset_hits = offset_checked = 0
        talking_without_mouth_motion = 0
        contributing_evidence_ids: set[str] = {item.evidence_id for item in own_evidence}

        for start, end in speech_intervals:
            onset_checked += 1
            nearest_start = nearest_event_after(start, mouth_events, max_window=search_window)
            if nearest_start is not None:
                latency = nearest_start - start
                onset_latencies.append(latency)
                if latency <= max_window:
                    onset_hits += 1
            elif mouth_events:
                talking_without_mouth_motion += 1

            offset_checked += 1
            nearest_end = nearest_event_near(end, mouth_events, max_window=search_window)
            if nearest_end is not None and abs(nearest_end - end) <= max_window:
                offset_hits += 1

        mouth_motion_without_speech = sum(
            1 for ts in mouth_events
            if not any(start - max_window <= ts <= end + max_window for start, end in speech_intervals)
        )

        frozen_mouth_events = 0
        for start, end in speech_intervals:
            window_evidence = [
                item for item in own_evidence
                if start <= item.timestamp_seconds <= end
                and (item.mouth_open_ratio is not None or item.lip_motion_intensity is not None)
            ]
            if window_evidence and not any(start <= ts <= end for ts in mouth_events):
                frozen_mouth_events += 1

        pause_windows = _pause_windows(own_segments, context.config.pause_min_seconds)
        closure_hits = closure_evidenced = 0
        for start, end in pause_windows:
            for item in own_evidence:
                if start <= item.timestamp_seconds <= end and item.mouth_open_ratio is not None:
                    closure_evidenced += 1
                    if item.mouth_open_ratio < _MOUTH_CLOSED_THRESHOLD:
                        closure_hits += 1

        jaw_evidenced = jaw_hits = 0
        for item in own_evidence:
            if item.jaw_motion is None:
                continue
            in_speech = any(start <= item.timestamp_seconds <= end for start, end in speech_intervals)
            if in_speech:
                jaw_evidenced += 1
                if item.jaw_motion > _JAW_MOTION_THRESHOLD:
                    jaw_hits += 1

        lip_motion_samples = [item.lip_motion_intensity for item in own_evidence if item.lip_motion_intensity is not None]
        mouth_ratio_samples = [item.mouth_open_ratio for item in own_evidence if item.mouth_open_ratio is not None]
        over_articulated = sum(1 for value in mouth_ratio_samples if value >= _OVER_ARTICULATION_THRESHOLD)
        under_articulated = sum(1 for value in mouth_ratio_samples if value <= _UNDER_ARTICULATION_THRESHOLD)

        lip_motion_variability = variability(lip_motion_samples)
        metrics = {
            "speech_to_mouth_latency": mean(onset_latencies) if onset_latencies else None,
            "mouth_onset_alignment": (onset_hits / onset_checked) if onset_checked else None,
            "mouth_offset_alignment": (offset_hits / offset_checked) if offset_checked else None,
            "pause_closure_alignment": (closure_hits / closure_evidenced) if closure_evidenced else None,
            "jaw_motion_alignment": (jaw_hits / jaw_evidenced) if jaw_evidenced else None,
            "syllabic_motion_consistency": (
                1.0 - min(1.0, lip_motion_variability) if lip_motion_variability is not None else None
            ),
            "lip_motion_variation": lip_motion_variability,
            "over_articulation": (over_articulated / len(mouth_ratio_samples)) if mouth_ratio_samples else None,
            "under_articulation": (under_articulated / len(mouth_ratio_samples)) if mouth_ratio_samples else None,
            "frozen_mouth_events": float(frozen_mouth_events),
            "talking_without_mouth_motion": float(talking_without_mouth_motion),
            "mouth_motion_without_speech": float(mouth_motion_without_speech),
            "sync_stability": variability(onset_latencies),
            "overall_lip_sync_naturalness": score_from_ratio(onset_checked, onset_hits) if onset_checked else 0.0,
        }

        evidence_count = onset_checked
        confidence = confidence_for(context, evidence_count=evidence_count, corroboration_count=evidence_count)
        rationale = (
            "coarse-timing lip sync (frame/segment-level evidence only) -- "
            f"{onset_hits}/{onset_checked} speech onsets had a mouth-motion event within "
            f"{max_window:.2f}s" if onset_checked else "No speech/mouth-motion evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME, metrics=metrics, score=metrics["overall_lip_sync_naturalness"],
            confidence=confidence, sample_count=evidence_count, evidence_ids=sorted(contributing_evidence_ids),
            rationale=rationale,
        )
