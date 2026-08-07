"""EvidenceQueue -- in-memory accumulator for the EvidenceBundles a
Connector returns during one orchestrator.run_job() call. Owns no file
I/O of its own; a future integration phase decides whether/how to
persist its contents (e.g. via creator_intelligence.intake's add_*
functions), which this package never calls.
"""
from __future__ import annotations

from src.creator_intelligence.evidence import Evidence

from .interfaces import EvidenceBundle


class EvidenceQueue:
    def __init__(self) -> None:
        self._bundles: list[EvidenceBundle] = []

    def add(self, bundle: EvidenceBundle) -> None:
        self._bundles.append(bundle)

    def bundles(self) -> list[EvidenceBundle]:
        return list(self._bundles)

    def all_items(self) -> list[Evidence]:
        items: list[Evidence] = []
        for bundle in self._bundles:
            items.extend(bundle.items)
        return items

    def items_for_section(self, section: str) -> list[Evidence]:
        items: list[Evidence] = []
        for bundle in self._bundles:
            if bundle.section == section:
                items.extend(bundle.items)
        return items

    def counts_by_section(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for bundle in self._bundles:
            counts[bundle.section] = counts.get(bundle.section, 0) + len(bundle.items)
        return counts

    def warnings(self) -> list[str]:
        collected: list[str] = []
        for bundle in self._bundles:
            collected.extend(bundle.warnings)
        return collected
