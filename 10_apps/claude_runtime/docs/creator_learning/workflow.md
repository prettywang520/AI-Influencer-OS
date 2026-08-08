# Creator Learning Engine -- Workflow

See `architecture.md` for the design. This is the practical "how do I
run a session" guide.

## 1. Zero-network: re-learn from evidence already on disk

Useful for regenerating reports, or in any test/demo, without
touching a browser:

```python
from src.creator_learning.config import load_learning_config
from src.creator_learning.engine import CreatorLearningEngine

engine = CreatorLearningEngine(load_learning_config())
result = engine.learn("https://www.instagram.com/some_creator/")
print(result.dna.overall_confidence)
print(result.reports["learning"])  # rendered Markdown
```

If nothing has ever been learned for this creator, this call still
succeeds -- it produces an all-`unknown`-confidence `CreatorDNA` from
zero evidence, matching "unknown remains unknown."

## 2. Ingesting a manual intake workspace

If a `creator_intelligence.intake` workspace already has an
`evidence_bundle.json` (built via `intake.py`'s `add_*` functions,
Phase 12A.1), point `learn()` at it:

```python
result = engine.learn(
    "https://www.instagram.com/some_creator/",
    intake_evidence_bundle_path="output/creator_intelligence/some_creator/evidence_bundle.json",
)
```

## 3. Running an existing Connector through the engine

`learn()` accepts any already-built `creator_research.connector.BaseConnector`
subclass (a `FakeBrowserAdapter`-backed one in tests/demos; a real
`InstagramResearchConnector` only when explicitly authorized, same
gate Phase 12B.2.1 established). The engine runs it through
`creator_research.orchestrator.run_job()` itself -- it never opens a
browser directly:

```python
from src.creator_research.instagram.connector import InstagramResearchConnector
from src.creator_research.interfaces import ConnectorSection

connector = InstagramResearchConnector(adapter, instagram_config)
result = engine.learn(
    "https://www.instagram.com/some_creator/",
    connector=connector,
    requested_sections=(ConnectorSection.PROFILE, ConnectorSection.GRID),
)
```

`connector=None` is always safe. Supplying a real, network-capable
connector is the caller's decision and the caller's authorization to
make -- this package does not construct connectors itself.

## 4. Reading back what has been learned

```python
engine.latest_dna(profile_url)      # CreatorDNA | None
engine.history(profile_url)         # list[LearningHistoryEntry]
engine.reports(profile_url)         # dict[str, str] -- 6 Markdown + 1 JSON (creator_dna)
engine.reports(profile_url, session_id="...")   # a specific past session
```

## 5. What gets written to disk

Every `learn()` call, whether or not it added new evidence, appends a
new session to `output/creator_learning/<creator_id>/learning_history.json`
and saves a new `dna_snapshots/<session_id>.json` -- even a "zero new
evidence" re-run gets its own session id (each call is a distinct
learning event) but always produces the same `dna_id`, since
`CreatorDNA` is a pure function of the evidence store's contents.

`output/` is gitignored; nothing under `output/creator_learning/` is
ever committed by this package.
