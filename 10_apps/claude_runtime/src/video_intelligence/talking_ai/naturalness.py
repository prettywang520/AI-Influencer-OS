"""NaturalnessAnalyzer -- DNA field naturalness (task §15). Aggregates
the 10 already-computed domain TalkingAIMetricResults into one
TalkingNaturalnessResult; computes no independent signal of its own.
Never a human/AI classifier: there is no boolean/enum field anywhere
in TalkingNaturalnessResult that could hold such a verdict, and the
rationale text is built exclusively from a fixed vocabulary of
comparative phrases ("more naturalistic" / "more mechanically
regular" / "insufficient evidence") -- never a claim about a video's
actual human or AI origin.
"""
from __future__ import annotations

from statistics import median

from .analyzer import TalkingAIAnalyzerContext
from .models import (
    ConfidenceLevel,
    NATURALNESS_DIMENSION_ALIASES,
    TalkingAIMetricResult,
    TalkingNaturalnessResult,
    min_confidence,
)

TRAIT_NAME = "naturalness"


class NaturalnessAnalyzer:
    name = TRAIT_NAME

    def analyze(
        self,
        context: TalkingAIAnalyzerContext,
        domain_results: dict[str, TalkingAIMetricResult],
    ) -> TalkingNaturalnessResult:
        dimension_confidence: dict[str, str] = {}
        evidenced_dimension_scores: dict[str, float] = {}
        all_evidence_ids: set[str] = set()
        warnings: list[str] = []

        for trait_name, dimension_name in NATURALNESS_DIMENSION_ALIASES.items():
            result = domain_results.get(trait_name)
            if result is None:
                dimension_confidence[dimension_name] = ConfidenceLevel.UNKNOWN
                warnings.append(f"No domain result supplied for '{trait_name}'.")
                continue
            dimension_confidence[dimension_name] = result.confidence
            all_evidence_ids.update(result.evidence_ids)
            warnings.extend(result.warnings)
            if result.confidence != ConfidenceLevel.UNKNOWN:
                evidenced_dimension_scores[dimension_name] = result.score

        if evidenced_dimension_scores:
            overall_score = sum(evidenced_dimension_scores.values()) / len(evidenced_dimension_scores)
            overall_confidence = min_confidence(
                [dimension_confidence[name] for name in evidenced_dimension_scores]
            )
        else:
            overall_score = 0.0
            overall_confidence = ConfidenceLevel.UNKNOWN

        rationale = self._build_rationale(evidenced_dimension_scores)

        return TalkingNaturalnessResult(
            dimension_scores=evidenced_dimension_scores,
            dimension_confidence=dimension_confidence,
            score=overall_score,
            confidence=overall_confidence,
            sample_count=len(all_evidence_ids),
            evidence_ids=sorted(all_evidence_ids),
            warnings=warnings,
            rationale=rationale,
        )

    @staticmethod
    def _build_rationale(evidenced_dimension_scores: dict[str, float]) -> str:
        if not evidenced_dimension_scores:
            return "Insufficient evidence across all naturalness dimensions."

        if len(evidenced_dimension_scores) == 1:
            (only_dimension,) = evidenced_dimension_scores.keys()
            return (
                f"Insufficient evidence to compare dimensions; only '{only_dimension}' "
                "had enough evidence to score."
            )

        midpoint = median(evidenced_dimension_scores.values())
        more_naturalistic = sorted(
            name for name, score in evidenced_dimension_scores.items() if score > midpoint
        )
        more_mechanical = sorted(
            name for name, score in evidenced_dimension_scores.items() if score < midpoint
        )

        clauses = []
        if more_naturalistic:
            clauses.append(f"more naturalistic on {', '.join(more_naturalistic)}")
        if more_mechanical:
            clauses.append(f"more mechanically regular on {', '.join(more_mechanical)}")
        missing = sorted(set(NATURALNESS_DIMENSION_ALIASES.values()) - set(evidenced_dimension_scores))
        if missing:
            clauses.append(f"insufficient evidence on {', '.join(missing)}")

        return "; ".join(clauses) + "." if clauses else "Insufficient evidence to characterize naturalness."
