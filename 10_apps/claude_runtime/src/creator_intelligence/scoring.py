"""Shared scoring math for the Creator Intelligence Framework.
One implementation of clamping/weighted-average, reused by every
analyzer and the summary builder, so scoring behavior never drifts
between per-analyzer copies.
"""
from __future__ import annotations

SCORE_MIN = 0.0
SCORE_MAX = 1.0


def clamp_score(value: float) -> float:
    """Clamps a raw score into the closed [0.0, 1.0] range."""
    if value < SCORE_MIN:
        return SCORE_MIN
    if value > SCORE_MAX:
        return SCORE_MAX
    return float(value)


def weighted_average(values: list[float], weights: list[float] | None = None) -> float:
    """Weighted mean of `values`, clamped to [0, 1]. An empty `values`
    list returns 0.0 (callers should gate on evidence presence before
    calling this -- an empty score is meaningless without a
    corresponding UNKNOWN confidence, which this function does not
    itself decide)."""
    if not values:
        return SCORE_MIN
    if weights is None:
        weights = [1.0] * len(values)
    if len(weights) != len(values):
        raise ValueError("values and weights must be the same length")
    total_weight = sum(weights)
    if total_weight <= 0:
        return SCORE_MIN
    total = sum(v * w for v, w in zip(values, weights))
    return clamp_score(total / total_weight)
