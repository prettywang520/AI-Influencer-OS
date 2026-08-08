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

## 6. Workflow Automation (Phase 12C.0.1)

`CreatorLearningEngine.learn()` (above) is a single, atomic,
non-resumable call. `WorkflowRunner` (`workflow_runner.py`) adds a
**checkpointed** layer on top of the exact same building blocks
(`orchestrator.run_job()`, `run_learning_session()`,
`reports.builder`/`reports.markdown` -- none of them modified), so a
run can be tracked, resumed, retried, or cancelled instead of being
all-or-nothing.

### State machine

```
CREATED -> RESEARCHING -> LEARNING -> KNOWLEDGE_UPDATED -> DNA_UPDATED -> REPORTING -> COMPLETED
                                                                              FAILED (from any non-terminal state)
                                                                              CANCELLED (from any non-terminal state)
```

A `WorkflowCheckpoint` is saved to
`output/creator_learning/<creator_id>/workflow/<workflow_id>.json`
after every transition (plus a `latest.json` pointer, same pattern as
`learning_state.json`'s `last_session_id`). `workflow_id` is a fresh
`uuid4`-based token per run (like `learning_session.py`'s own
`session_id`) -- re-learning the same creator next week is a new,
independent workflow, not a collision.

### Running it

```python
from src.creator_learning.workflow_runner import WorkflowRunner

runner = WorkflowRunner()  # loads config/creator_learning/{engine,workflow}.yaml
result = runner.start("https://www.instagram.com/some_creator/")  # zero-network
# or, with an already-built Connector (same authorization rules as engine.learn()):
result = runner.start("https://www.instagram.com/some_creator/", connector=my_connector)

print(result.state)               # "completed"
print(result.checkpoint.workflow_id)
print(result.report.evidence_statistics, result.report.dna_statistics)
```

`python3 -m src.creator_learning.workflow --creator-url <url>` runs
the same thing from the shell, zero-network only. The full CLI
(`workflow_cli.py`) additionally supports:

```
--start --creator-url URL [--intake-evidence-bundle PATH]
--resume WORKFLOW_ID     # auto-detects FAILED -> retry(), else -> resume()
--cancel WORKFLOW_ID
--status [WORKFLOW_ID]
--validate
```

**Neither CLI can construct or reference a live Connector** -- the
same structural restraint `creator_research/cli.py` already
established. A real, network-capable connector can only be supplied
by a direct Python caller of `WorkflowRunner.start()`/`resume()`/
`retry()`/`restart()`.

### resume() vs. retry() vs. restart() vs. cancel()

- **`resume(workflow_id)`**: continues a workflow that is
  non-terminal on disk (the process died mid-run without ever
  reaching FAILED). Raises if the workflow already reached
  COMPLETED/CANCELLED/FAILED.
- **`retry(workflow_id)`**: only valid once a workflow reached FAILED.
  Resets to the step after its last completed one and re-attempts,
  capped by `workflow.yaml`'s `max_retry_attempts`.
- **`restart(workflow_id)`**: the one operation that *deliberately*
  redoes finished work -- resets the same `workflow_id` back to
  `CREATED` and runs the whole thing again (e.g. to pick up a supplied
  connector after research previously ran with none).
- **`cancel(workflow_id)`**: valid from any non-terminal state, same
  rule as FAILED.

### Two documented tradeoffs

1. **Research is never re-run once LEARNING has started.** If a
   connector was supplied and RESEARCHING already completed (whether
   it collected real evidence or was a no-op with `connector=None`),
   `resume()`/`retry()` will **not** invoke the connector again, even
   if one is passed again -- re-running research after evidence has
   already been (or is about to be) merged would just be a wasted, or
   worse duplicated, live call. If the process crashes *between*
   RESEARCHING finishing and LEARNING starting, the collected evidence
   (which `orchestrator.run_job()` only ever keeps in memory) is lost;
   resuming in that narrow window requires supplying the connector
   again so research reruns -- `_current_index(checkpoint) < LEARNING`
   is what still allows that. Once LEARNING has begun, this window is
   closed for good.
2. **`run_learning_session()` is reused as one atomic call, not
   decomposed.** If the process dies inside it, the checkpoint still
   shows `LEARNING`, and resuming re-invokes the whole call. This is
   safe (evidence merge is dedup-by-id idempotent; `CreatorDNA`
   rebuild is a deterministic function of the merged evidence -- same
   `dna_id` out for the same evidence in) but does add one extra,
   harmless `learning_history.json` entry for the retried attempt.

### Output

```
output/creator_learning/<creator_id>/
    ...                       [unchanged files from engine.py/knowledge_base.py, see §5]
    workflow/<workflow_id>.json   one checkpoint per run
    workflow/latest.json          {"workflow_id": ..., "state": ...} pointer
    creator_dna.json              convenience "latest DNA" export (JSON)
    workflow_report.json          overwritten with the latest run's report
```
