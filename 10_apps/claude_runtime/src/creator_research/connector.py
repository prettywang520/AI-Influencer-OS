"""BaseConnector -- the inheritable abstract form of interfaces.py's
Connector Protocol, plus a name-based registry so orchestration code
never imports a concrete connector class directly. Zero implementation
logic here: every method exists only to be overridden. This module
never imports a network, browser-automation, or platform-specific
library, and never will -- that is the entire point of this
interface/registry split (see docs/creator_research/connector_contract.md).
"""
from __future__ import annotations

from typing import Callable

from .exceptions import ConnectorNotImplementedError, ConnectorNotRegisteredError
from .interfaces import EvidenceBundle


class BaseConnector:
    """Base for every research connector. Deliberately NOT a strict
    `abc.ABC` -- a concrete subclass may override any subset of the 10
    collect_* methods (e.g. a connector that only supports
    profile+captions); every method left un-overridden raises
    ConnectorNotImplementedError if called, rather than blocking
    instantiation of a partial connector."""

    name: str = "base"

    def collect_profile(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_profile is not implemented")

    def collect_grid(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_grid is not implemented")

    def collect_posts(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_posts is not implemented")

    def collect_captions(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_captions is not implemented")

    def collect_comments(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_comments is not implemented")

    def collect_creator_replies(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_creator_replies is not implemented")

    def collect_reels(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_reels is not implemented")

    def collect_highlights(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_highlights is not implemented")

    def collect_relationships(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_relationships is not implemented")

    def collect_visual_examples(self, job) -> EvidenceBundle:
        raise ConnectorNotImplementedError(f"{type(self).__name__}.collect_visual_examples is not implemented")


class ConnectorRegistry:
    """Maps a plain connector_name string (as stored on ResearchJob)
    to a factory that builds a BaseConnector -- lets orchestration
    code resolve a connector by name without ever importing a
    concrete connector class itself."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BaseConnector]] = {}

    def register(self, name: str, factory: Callable[[], BaseConnector]) -> None:
        self._factories[name] = factory

    def get(self, name: str) -> BaseConnector:
        factory = self._factories.get(name)
        if factory is None:
            raise ConnectorNotRegisteredError(f"No connector registered under name {name!r}")
        return factory()

    def registered_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
