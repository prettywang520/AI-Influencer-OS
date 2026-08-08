"""Phase 12C.0 -- Creator Learning Engine.

Turns Creator Research (src/creator_research/) into Creator
Intelligence (src/creator_intelligence/) by adding the missing
memory/synthesis layer: a durable per-creator knowledge base that
accumulates Evidence across sessions, re-derives CreatorDNA as that
store grows, tracks how it changes over time ("style evolution"), and
produces human-readable reports. This package performs NO evidence
collection of its own -- it only accumulates Evidence handed to it by
an already-built Connector (run through
creator_research.orchestrator.run_job()) or an existing
creator_intelligence.intake evidence bundle. No live network access,
no browser automation, and no writes to Aiko's persona happen
anywhere in this package.
"""
from .engine import CreatorLearningEngine, LearningResult
from .config import LearningConfig, load_learning_config
from .creator_url import ParsedCreatorUrl, parse_creator_url
from .knowledge_base import CreatorKnowledgeBase
from .learning_history import LearningHistoryEntry
from .style_evolution import StyleEvolutionRecord
from .workflow import WorkflowConfig, load_workflow_config
from .workflow_models import WorkflowCheckpoint
from .workflow_runner import WorkflowRunner, WorkflowRunResult
from .workflow_state import WorkflowState

__all__ = [
    "CreatorLearningEngine",
    "LearningResult",
    "LearningConfig",
    "load_learning_config",
    "ParsedCreatorUrl",
    "parse_creator_url",
    "CreatorKnowledgeBase",
    "LearningHistoryEntry",
    "StyleEvolutionRecord",
    "WorkflowConfig",
    "load_workflow_config",
    "WorkflowCheckpoint",
    "WorkflowRunner",
    "WorkflowRunResult",
    "WorkflowState",
]
