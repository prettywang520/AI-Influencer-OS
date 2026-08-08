"""SpeechSyncAnalyzer -- DNA fields speech_rhythm (task §3) and
pause_behavior (task §5). Both are derived from SpeechSegment +
TalkingAIEvidence timing, so one analyzer file owns both. Uses
"breath-like pause behavior" language, never "breathing detected" --
actual breathing is never inferred, only pause timing and its
cross-channel correlates (mouth closure, blink, head motion).
"""
from __future__ import annotations

from statistics import mean

from .analyzer import TalkingAIAnalyzerContext, confidence_for, evidence_for_video, ratio_true, segments_for_video, variability
from .models import TalkingAIMetricResult

TRAIT_NAME_RHYTHM = "speech_rhythm"
TRAIT_NAME_PAUSE = "pause_behavior"

_MOUTH_CLOSED_THRESHOLD = 0.15
_HEAD_MOTION_THRESHOLD = 0.10


class SpeechSyncAnalyzer:
    name = TRAIT_NAME_RHYTHM

    def analyze_speech_rhythm(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        segments = segments_for_video(context.speech_segments, context.video_id)
        evidence_count = len(segments)

        rates = [segment.speech_rate_optional for segment in segments if segment.speech_rate_optional is not None]
        questions = [segment.question_like for segment in segments if segment.question_like is not None]
        emphasis = [segment.emphasis for segment in segments if segment.emphasis is not None]
        languages = {segment.language for segment in segments if segment.language is not None}
        code_switching = [segment.code_switching for segment in segments if segment.code_switching is not None]

        metrics = {
            "average_speech_rate": mean(rates) if rates else None,
            "speech_rate_variability": variability(rates),
            "question_ratio": ratio_true(questions),
            "emphasis_ratio": ratio_true(emphasis),
            "language_mix_count": float(len(languages)) if languages else None,
            "code_switching_ratio": ratio_true(code_switching),
        }
        confidence = confidence_for(context, evidence_count=evidence_count, corroboration_count=evidence_count)
        score = min(1.0, evidence_count / (evidence_count + 2)) if evidence_count else 0.0
        rationale = (
            "Speech rhythm derived from SpeechSegment evidence."
            if evidence_count
            else "No speech segments supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME_RHYTHM, metrics=metrics, score=score, confidence=confidence,
            sample_count=evidence_count, evidence_ids=[segment.segment_id for segment in segments], rationale=rationale,
        )

    def analyze_pause_behavior(self, context: TalkingAIAnalyzerContext) -> TalkingAIMetricResult:
        segments = segments_for_video(context.speech_segments, context.video_id)
        own_evidence = evidence_for_video(context.evidence, context.video_id)

        pause_windows: list[tuple[float, float]] = []
        for segment in segments:
            if segment.pause_before is not None and segment.pause_before >= context.config.pause_min_seconds:
                pause_windows.append((segment.start_seconds - segment.pause_before, segment.start_seconds))

        pause_durations = [end - start for start, end in pause_windows]

        closure_hits = closure_evidenced = 0
        blink_hits = blink_evidenced = 0
        head_motion_hits = head_motion_evidenced = 0
        contributing_evidence_ids: set[str] = set()

        for start, end in pause_windows:
            nearby = [item for item in own_evidence if start <= item.timestamp_seconds <= end]
            for item in nearby:
                contributing_evidence_ids.add(item.evidence_id)
                if item.mouth_open_ratio is not None:
                    closure_evidenced += 1
                    if item.mouth_open_ratio < _MOUTH_CLOSED_THRESHOLD:
                        closure_hits += 1
                if item.blink is not None:
                    blink_evidenced += 1
                    if item.blink:
                        blink_hits += 1
                if item.head_motion_intensity is not None:
                    head_motion_evidenced += 1
                    if item.head_motion_intensity > _HEAD_MOTION_THRESHOLD:
                        head_motion_hits += 1

        metrics = {
            "pause_frequency": float(len(pause_windows)) if segments else None,
            "average_pause_duration": mean(pause_durations) if pause_durations else None,
            "pause_duration_variability": variability(pause_durations),
            "mouth_closure_during_pause_ratio": (closure_hits / closure_evidenced) if closure_evidenced else None,
            "blink_near_pause_ratio": (blink_hits / blink_evidenced) if blink_evidenced else None,
            "head_motion_near_pause_ratio": (head_motion_hits / head_motion_evidenced) if head_motion_evidenced else None,
            "breath_like_pause_count": float(len(pause_windows)),
        }
        evidence_count = len(pause_windows)
        confidence = confidence_for(context, evidence_count=evidence_count, corroboration_count=evidence_count)
        score = min(1.0, evidence_count / (evidence_count + 2)) if evidence_count else 0.0
        rationale = (
            "Breath-like pause behavior observed across speech-segment pause windows "
            "(pause timing and cross-channel correlates only -- actual breathing is never inferred)."
            if evidence_count
            else "No pause evidence supplied; confidence is unknown."
        )
        return TalkingAIMetricResult(
            trait_name=TRAIT_NAME_PAUSE, metrics=metrics, score=score, confidence=confidence,
            sample_count=evidence_count, evidence_ids=sorted(contributing_evidence_ids), rationale=rationale,
        )
