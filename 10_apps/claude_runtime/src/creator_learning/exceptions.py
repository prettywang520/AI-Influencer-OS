"""Shared error hierarchy for the Creator Learning Engine."""
from __future__ import annotations


class LearningError(RuntimeError):
    """Base error for the Creator Learning Engine."""


class LearningConfigError(LearningError):
    """Raised when config/creator_learning/engine.yaml is missing or invalid."""


class InvalidCreatorUrlError(LearningError):
    """Raised when a supplied profile URL cannot be parsed into a
    platform/username."""


class KnowledgeBaseError(LearningError):
    """Raised for knowledge-base read/write/consistency failures."""


class LearningSessionError(LearningError):
    """Raised for learning-session orchestration failures."""


class ReportError(LearningError):
    """Raised for report-generation failures."""
