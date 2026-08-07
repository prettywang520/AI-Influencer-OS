# Creator Research Agent -- Architecture

**Phase:** 12B.1
**Status:** Orchestration layer only. No connector implementation
exists in this package; the only concrete connector anywhere is the
synthetic `DummyConnector` used in tests/demos.

## Pipeline

```
ResearchAgent (this package)
    |
    v
Connector (abstract interface -- no implementation here)
    |
    v
Evidence Collector (whatever the concrete connector does internally)
    |
    v
Evidence Bundle (interfaces.EvidenceBundle)
    |
    v
Creator Intelligence (src/creator_intelligence/ -- Phase 12A.0/12A.1,
                       unmodified by this phase)
    |
    v
Creator DNA (creator_intelligence.models.CreatorDNA)
```

This package owns everything up to and including "Evidence Bundle."
It never calls into `creator_intelligence` itself (no
`intake.add_*()`, no `summary_builder.build_creator_dna()`) -- handing
an `EvidenceBundle`'s contents to `creator_intelligence` is an
explicit, separate integration decision for a future phase.

## Why the agent knows nothing about Instagram

`connector.py`'s `BaseConnector` and `interfaces.py`'s `Connector`
Protocol define 10 `collect_*` methods with **zero implementation
logic**. `orchestrator.run_job()` only ever calls
`getattr(connector, f"collect_{section}")` -- it has no branch, import,
or string anywhere that names a specific platform. A future phase that
adds a real connector (e.g. an Instagram one) would create a new
`BaseConnector` subclass in its own module, register it via
`ConnectorRegistry`, and pass an instance to `run_job()` directly --
this package never needs to change to support it. See
[connector_contract.md](connector_contract.md) for what such a
connector must (and must not) do.

## Component responsibilities

| Module | Owns |
| --- | --- |
| `interfaces.py` | `EvidenceBundle`, `ConnectorSection`, the `Connector` Protocol -- pure typing |
| `connector.py` | `BaseConnector` (raises if a method isn't overridden), `ConnectorRegistry` |
| `state.py` | `JobStatus` + the transition table `advance()` validates against |
| `jobs.py` | `ResearchJob` (deterministic `job_id`) |
| `planner.py` | Section->state mapping, section ordering, resume point, retry policy |
| `progress.py` | `ResearchProgress` -- step-count-based tracking |
| `evidence_queue.py` | `EvidenceQueue` -- in-memory accumulator for one run |
| `report.py` | `ResearchReport` -- coverage/duration/missing-sections summary |
| `session.py` | `ResearchSession` -- the runtime container `run_job()` builds |
| `queue.py` | `ResearchQueue` -- file-backed job persistence/lifecycle |
| `orchestrator.py` | `run_job()` -- the actual execution entry point |
| `cli.py` | Job **lifecycle** management only (create/resume/cancel/status/validate) |

## Why the state machine has 8 states but the connector has 10 methods

`state.JobStatus.WORKING_STATES` is `PROFILE, GRID, CAPTIONS,
COMMENTS, REPLIES, REELS, HIGHLIGHTS, VISUAL` -- 8 states, matching the
task's own linear diagram. `interfaces.ConnectorSection` has 10
members (one per `collect_*` method). `planner.SECTION_STATE_MAP`
reconciles this: `posts` folds into `GRID`, `relationships` folds into
`VISUAL`. Every job flows through all 8 working states regardless of
which sections it actually requested -- a state with no requested
sections simply collects nothing and the job still transitions through
it (this is why `orchestrator.run_job()` iterates
`JobStatus.WORKING_STATES` directly, not a per-job subset).

## Why the CLI can never run a job

`cli.py`'s parser has no flag that accepts or constructs a `Connector`
-- every flag operates on the queue file only. `orchestrator.run_job()`
is the only way to actually execute collection, and it must be called
from code that has a real `Connector` instance in hand. This is
deliberate: it means "no network" is a structural property of the CLI
binary itself, not just something today's implementation happens not
to do.
