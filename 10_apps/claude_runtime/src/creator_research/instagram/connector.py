"""InstagramResearchConnector -- the concrete, read-only Connector
implementation. Ties together navigator.py (adapter),
selectors.py/observer.py (page observation), evidence_mapper.py
(Evidence conversion), dedupe.py/checkpoint.py (resume-safe identity),
rate_limit.py (polite pacing), and diagnostics.py (structured run
reporting). Runs through the existing
creator_research.orchestrator.run_job() unchanged -- this module never
creates a second orchestration system.

Checkpointing here saves once per section, after that section's
observation completes (not mid-scroll-loop) -- resume is still fully
duplicate-safe (dedupe is seeded from the checkpoint's
processed_source_ids before a section re-runs), just not
sub-section-granular. `checkpoint_every_items` is accepted from config
but this phase doesn't yet act on it mid-section; a future phase could
wire it into observer.py's scroll loop if a live run's section length
makes that worthwhile.
"""
from __future__ import annotations

from pathlib import Path

from src.creator_intelligence.evidence import Evidence

from ..connector import BaseConnector, ConnectorRegistry
from ..interfaces import ConnectorSection, EvidenceBundle
from . import evidence_mapper, observer
from .checkpoint import InstagramCheckpoint, load_checkpoint, save_checkpoint, validate_checkpoint
from .config import InstagramConnectorConfig
from .dedupe import Deduplicator
from .diagnostics import DiagnosticsRecorder, new_connector_run_id, save_diagnostics
from .exceptions import (
    AccessLimitedError,
    AuthenticationRequiredError,
    ChallengeDetectedError,
    RateLimitedError,
)
from .models import CaptionObservation, DiscoveredItem, PostRecord
from .rate_limit import RateLimiter
from .selectors import InstagramSelectors, default_instagram_selectors

_ACCESS_EXCEPTION_KIND = {
    AccessLimitedError: "access_limited",
    AuthenticationRequiredError: "authentication_required",
    RateLimitedError: "rate_limited",
    ChallengeDetectedError: "challenge_detected",
}


class InstagramResearchConnector(BaseConnector):
    name = "instagram"

    def __init__(
        self,
        adapter,
        config: InstagramConnectorConfig,
        *,
        selectors: InstagramSelectors | None = None,
        checkpoint_dir: str | Path | None = None,
        diagnostics_dir: str | Path | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.selectors = selectors or default_instagram_selectors()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else config.resolved_checkpoint_directory()
        self.diagnostics_dir = Path(diagnostics_dir) if diagnostics_dir else config.resolved_diagnostics_directory()
        self.rate_limiter = rate_limiter or RateLimiter(
            minimum_delay_seconds=config.minimum_delay_seconds,
            maximum_delay_seconds=config.maximum_delay_seconds,
            backoff_seconds=config.backoff_seconds,
            max_retries_per_navigation=config.max_retries_per_navigation,
        )

        self._grid_cache: dict[str, list[DiscoveredItem]] = {}
        self._posts_cache: dict[str, list[PostRecord]] = {}
        self._captions_cache: dict[str, list[CaptionObservation]] = {}
        self._dedupe_by_job: dict[str, Deduplicator] = {}
        self._diagnostics_by_job: dict[str, DiagnosticsRecorder] = {}

    # -- shared helpers -----------------------------------------------------

    def _dedupe_for(self, job) -> Deduplicator:
        dedupe = self._dedupe_by_job.get(job.job_id)
        if dedupe is None:
            dedupe = Deduplicator()
            self._dedupe_by_job[job.job_id] = dedupe
        return dedupe

    def _diagnostics_for(self, job) -> DiagnosticsRecorder:
        recorder = self._diagnostics_by_job.get(job.job_id)
        if recorder is None:
            recorder = DiagnosticsRecorder(
                connector_run_id=new_connector_run_id(), job_id=job.job_id, username=job.username
            )
            self._diagnostics_by_job[job.job_id] = recorder
        return recorder

    def save_diagnostics_for(self, job) -> Path | None:
        """Finalizes and persists this job's diagnostics. Not called
        automatically by orchestrator.run_job() (which doesn't know
        about per-connector diagnostics) -- a caller invokes this
        explicitly after a job finishes."""
        recorder = self._diagnostics_by_job.get(job.job_id)
        if recorder is None:
            return None
        recorder.finish()
        return save_diagnostics(self.diagnostics_dir, recorder)

    def _load_or_new_checkpoint(self, job, section: str) -> InstagramCheckpoint:
        if not self.config.checkpoint_enabled:
            return InstagramCheckpoint(
                job_id=job.job_id, creator_id=job.creator_id, connector_version=self.config.connector_version,
                section=section,
            )
        existing = load_checkpoint(self.checkpoint_dir, job.job_id, section)
        if existing is None:
            return InstagramCheckpoint(
                job_id=job.job_id, creator_id=job.creator_id, connector_version=self.config.connector_version,
                section=section,
            )
        mismatch_warnings = validate_checkpoint(
            existing, job_id=job.job_id, creator_id=job.creator_id, connector_version=self.config.connector_version
        )
        existing.warnings.extend(mismatch_warnings)
        self._dedupe_for(job).seed(existing.processed_source_ids)
        return existing

    def _save_checkpoint(self, checkpoint: InstagramCheckpoint, *, processed_source_ids, evidence_ids) -> None:
        if not self.config.checkpoint_enabled:
            return
        checkpoint.processed_source_ids = sorted(set(checkpoint.processed_source_ids) | set(processed_source_ids))
        checkpoint.evidence_ids = sorted(set(checkpoint.evidence_ids) | set(evidence_ids))
        save_checkpoint(self.checkpoint_dir, checkpoint)

    def _handle_access_exception(self, job, section: str, exc: Exception) -> EvidenceBundle:
        kind = _ACCESS_EXCEPTION_KIND.get(type(exc), "access_limited")
        self._diagnostics_for(job).record_access_event(kind=kind, section=section, detail=str(exc))
        if self.config.access_limit_behavior == "stop":
            raise exc
        return EvidenceBundle(section=section, warnings=[f"{kind}: {exc}"])

    def _map_items(self, mapper, records) -> tuple[list[Evidence], list[str]]:
        items: list[Evidence] = []
        warnings: list[str] = []
        for record in records:
            evidence, item_warnings = mapper(record)
            items.append(evidence)
            warnings.extend(item_warnings)
        return items, warnings

    # -- grid / posts / captions caches --------------------------------------

    def _ensure_grid(self, job) -> list[DiscoveredItem]:
        cached = self._grid_cache.get(job.job_id)
        if cached is not None:
            return cached
        self._diagnostics_for(job).record_section_attempt(ConnectorSection.GRID)
        if self.config.rate_limit_enabled:
            self.rate_limiter.polite_delay()
        checkpoint = self._load_or_new_checkpoint(job, ConnectorSection.GRID)
        dedupe = self._dedupe_for(job)
        items, warnings = observer.observe_grid(self.adapter, job, self.selectors, self.config, dedupe)
        for warning in warnings:
            self._diagnostics_for(job).record_warning(warning)
        # processed_source_ids stores the same identity strings dedupe.py
        # computes (url:.../id:.../hash:...), not raw URLs -- seed() on
        # resume compares against exactly this format.
        self._save_checkpoint(checkpoint, processed_source_ids=dedupe.seen_identities(), evidence_ids=[])
        self._grid_cache[job.job_id] = items
        return items

    def _ensure_posts(self, job) -> list[PostRecord]:
        cached = self._posts_cache.get(job.job_id)
        if cached is not None:
            return cached
        items = [item for item in self._ensure_grid(job) if item.content_type == "post"]
        limit = min(len(items), self.config.max_posts_per_job)
        posts = [observer.observe_post(self.adapter, item.source_url, self.selectors, self.config) for item in items[:limit]]
        self._posts_cache[job.job_id] = posts
        return posts

    def _ensure_captions(self, job) -> list[CaptionObservation]:
        cached = self._captions_cache.get(job.job_id)
        if cached is not None:
            return cached
        items = [item for item in self._ensure_grid(job) if item.content_type == "post"]
        captions, warnings = observer.observe_captions(self.adapter, items, self.selectors, self.config)
        for warning in warnings:
            self._diagnostics_for(job).record_warning(warning)
        self._captions_cache[job.job_id] = captions
        return captions

    # -- Connector Protocol methods ------------------------------------------

    def collect_profile(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.PROFILE)
        if self.config.rate_limit_enabled:
            self.rate_limiter.polite_delay()
        try:
            profile = observer.observe_profile(self.adapter, job, self.selectors, self.config)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.PROFILE, exc)
        evidence, warnings = evidence_mapper.profile_to_evidence(profile)
        diagnostics.record_section_completed(ConnectorSection.PROFILE, 1)
        return EvidenceBundle(section=ConnectorSection.PROFILE, items=[evidence], warnings=warnings)

    def collect_grid(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        try:
            items = self._ensure_grid(job)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.GRID, exc)
        evidence_items, warnings = self._map_items(evidence_mapper.discovered_item_to_evidence, items)
        diagnostics.record_section_completed(ConnectorSection.GRID, len(evidence_items))
        diagnostics.record_duplicates_skipped(ConnectorSection.GRID, self._dedupe_for(job).duplicates_skipped)
        return EvidenceBundle(section=ConnectorSection.GRID, items=evidence_items, warnings=warnings)

    def collect_posts(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.POSTS)
        try:
            posts = self._ensure_posts(job)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.POSTS, exc)
        evidence_items, warnings = self._map_items(evidence_mapper.post_to_evidence, posts)
        diagnostics.record_section_completed(ConnectorSection.POSTS, len(evidence_items))
        return EvidenceBundle(section=ConnectorSection.POSTS, items=evidence_items, warnings=warnings)

    def collect_captions(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.CAPTIONS)
        try:
            captions = self._ensure_captions(job)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.CAPTIONS, exc)
        evidence_items, warnings = self._map_items(evidence_mapper.caption_to_evidence, captions)
        diagnostics.record_section_completed(ConnectorSection.CAPTIONS, len(evidence_items))
        return EvidenceBundle(section=ConnectorSection.CAPTIONS, items=evidence_items, warnings=warnings)

    def collect_comments(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.COMMENTS)
        if self.config.rate_limit_enabled:
            self.rate_limiter.polite_delay()
        try:
            items = self._ensure_grid(job)
            all_comments = []
            all_warnings = []
            limit = min(len(items), self.config.max_posts_per_job)
            for item in items[:limit]:
                comments, warnings = observer.observe_comments(self.adapter, item.source_url, self.selectors, self.config)
                all_comments.extend(comments)
                all_warnings.extend(warnings)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.COMMENTS, exc)
        mapper = lambda record: evidence_mapper.comment_to_evidence(
            record, redact_usernames=self.config.redact_usernames_in_audience_comments
        )
        evidence_items, mapping_warnings = self._map_items(mapper, all_comments)
        diagnostics.record_section_completed(ConnectorSection.COMMENTS, len(evidence_items))
        return EvidenceBundle(
            section=ConnectorSection.COMMENTS, items=evidence_items, warnings=all_warnings + mapping_warnings
        )

    def collect_creator_replies(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.CREATOR_REPLIES)
        if self.config.rate_limit_enabled:
            self.rate_limiter.polite_delay()
        try:
            items = self._ensure_grid(job)
            all_replies = []
            all_warnings = []
            limit = min(len(items), self.config.max_posts_per_job)
            for item in items[:limit]:
                replies, warnings = observer.observe_creator_replies(
                    self.adapter, item.source_url, self.selectors, self.config, username=job.username
                )
                all_replies.extend(replies)
                all_warnings.extend(warnings)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.CREATOR_REPLIES, exc)
        evidence_items, mapping_warnings = self._map_items(evidence_mapper.creator_reply_to_evidence, all_replies)
        diagnostics.record_section_completed(ConnectorSection.CREATOR_REPLIES, len(evidence_items))
        return EvidenceBundle(
            section=ConnectorSection.CREATOR_REPLIES, items=evidence_items, warnings=all_warnings + mapping_warnings
        )

    def collect_reels(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.REELS)
        if self.config.rate_limit_enabled:
            self.rate_limiter.polite_delay()
        checkpoint = self._load_or_new_checkpoint(job, ConnectorSection.REELS)
        dedupe = self._dedupe_for(job)
        try:
            reels, warnings = observer.observe_reels(self.adapter, job, self.selectors, self.config, dedupe)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.REELS, exc)
        evidence_items, mapping_warnings = self._map_items(evidence_mapper.reel_to_evidence, reels)
        self._save_checkpoint(
            checkpoint, processed_source_ids=dedupe.seen_identities(), evidence_ids=[e.evidence_id for e in evidence_items]
        )
        diagnostics.record_section_completed(ConnectorSection.REELS, len(evidence_items))
        return EvidenceBundle(section=ConnectorSection.REELS, items=evidence_items, warnings=warnings + mapping_warnings)

    def collect_highlights(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.HIGHLIGHTS)
        if self.config.rate_limit_enabled:
            self.rate_limiter.polite_delay()
        try:
            highlights, warnings = observer.observe_highlights(self.adapter, job, self.selectors, self.config)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.HIGHLIGHTS, exc)
        evidence_items, mapping_warnings = self._map_items(evidence_mapper.highlight_to_evidence, highlights)
        diagnostics.record_section_completed(ConnectorSection.HIGHLIGHTS, len(evidence_items))
        return EvidenceBundle(section=ConnectorSection.HIGHLIGHTS, items=evidence_items, warnings=warnings + mapping_warnings)

    def collect_relationships(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.RELATIONSHIPS)
        try:
            posts = self._ensure_posts(job)
            captions = self._ensure_captions(job)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.RELATIONSHIPS, exc)
        records = observer.observe_relationships(posts, captions)
        evidence_items, warnings = self._map_items(evidence_mapper.relationship_to_evidence, records)
        diagnostics.record_section_completed(ConnectorSection.RELATIONSHIPS, len(evidence_items))
        return EvidenceBundle(section=ConnectorSection.RELATIONSHIPS, items=evidence_items, warnings=warnings)

    def collect_visual_examples(self, job) -> EvidenceBundle:
        diagnostics = self._diagnostics_for(job)
        diagnostics.record_section_attempt(ConnectorSection.VISUAL_EXAMPLES)
        try:
            items = self._ensure_grid(job)
        except (AccessLimitedError, AuthenticationRequiredError, RateLimitedError, ChallengeDetectedError) as exc:
            return self._handle_access_exception(job, ConnectorSection.VISUAL_EXAMPLES, exc)
        records = observer.observe_visual_examples(items)
        evidence_items, warnings = self._map_items(evidence_mapper.visual_example_to_evidence, records)
        diagnostics.record_section_completed(ConnectorSection.VISUAL_EXAMPLES, len(evidence_items))
        return EvidenceBundle(section=ConnectorSection.VISUAL_EXAMPLES, items=evidence_items, warnings=warnings)


def register_instagram_connector(registry: ConnectorRegistry, adapter, config: InstagramConnectorConfig, **kwargs) -> None:
    """Registers this connector under the name "instagram". Never
    called automatically on import -- an explicit call only, so
    importing this package has zero side effects."""
    registry.register("instagram", lambda: InstagramResearchConnector(adapter, config, **kwargs))
