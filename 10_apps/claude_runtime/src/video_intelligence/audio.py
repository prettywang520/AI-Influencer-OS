"""Audio Analyzer -- scores evidence coverage for voice/music/ambient
balance, silence, and hook sound.
"""
from __future__ import annotations

from .analyzer import AnalyzerContext, AnalyzerResult, analyze_by_tag

TRAIT_NAME = "audio"
TAGS = ("audio", "voice", "music", "ambient", "audio_balance", "silence", "hook_sound")


class AudioAnalyzer:
    name = TRAIT_NAME

    def analyze(self, context: AnalyzerContext) -> AnalyzerResult:
        return analyze_by_tag(
            context, trait_name=TRAIT_NAME, tags=TAGS,
            rationale_with_evidence="Audio balance pattern observed across cited evidence.",
            rationale_without_evidence="No audio-tagged evidence supplied; confidence is unknown.",
        )
