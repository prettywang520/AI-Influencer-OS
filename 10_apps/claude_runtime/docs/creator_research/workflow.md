# Creator Research Agent -- Workflow

A worked example using the CLI for job lifecycle management and
`orchestrator.run_job()` (called from Python, since the CLI never
executes a job -- see [architecture.md](architecture.md)) for actual
collection against a connector.

## 1. Create a job

```
python3 -m src.creator_research.cli \
  --create-job \
  --creator-id demo_creator \
  --platform instagram \
  --username demo_creator \
  --profile-url "https://example.invalid/demo_creator" \
  --sections profile,grid,captions \
  --priority 5 \
  --queue output/creator_research/queue.json
```

Creates (or, if the same creator/sections/connector combination
already exists, idempotently returns) a `ResearchJob` in `QUEUED`
status. `--queue` defaults to `config/creator_research/queue.yaml`'s
`default_queue_path` (`output/creator_research/queue.json`, gitignored
-- real job data is never auto-tracked by git).

## 2. Check status

```
python3 -m src.creator_research.cli --status --queue output/creator_research/queue.json
python3 -m src.creator_research.cli --status JOB_ID --queue output/creator_research/queue.json
```

## 3. Run it (Python, not the CLI)

```python
from src.creator_research.jobs import ResearchJob
from src.creator_research.orchestrator import run_job, load_orchestrator_config
from src.creator_research.queue import ResearchQueue, load_queue_config

queue_config = load_queue_config()
queue = ResearchQueue(queue_config.resolved_default_queue_path())
job = queue.get("JOB_ID")

connector = MyConcreteConnector()  # a future phase's own class, never provided here
config = load_orchestrator_config()
session = run_job(job, connector, config)

print(session.report.summary)
print(session.report.coverage)
print(session.report.missing_sections)
```

`run_job()` mutates `job.status` in place (`QUEUED -> STARTING -> ...
-> COMPLETE`/`FAILED`). If you're driving jobs out of the queue file
rather than a one-off script, save the mutated job back with
`queue._replace(...)` or re-`enqueue()` it (idempotent re-enqueue is a
no-op for an already-stored `job_id`, so persisting status changes
currently means writing through `queue.py`'s own lifecycle methods --
`pause`/`resume`/`cancel`/`retry` -- rather than `run_job()` writing to
the queue file itself, which it does not do).

## 4. Handle a failure and resume

If `run_job()' leaves a job `FAILED` (e.g. a required section's
connector call raised), the CLI's `--resume` transitions it back to
`QUEUED` (via a retry, bounded by `queue.yaml`'s `max_attempts`):

```
python3 -m src.creator_research.cli --resume JOB_ID --queue output/creator_research/queue.json
```

Then call `run_job()` again. Because `orchestrator.run_job()` doesn't
persist `ResearchProgress`/`EvidenceQueue` across process runs in this
phase, a fresh `run_job()` call starts its progress tracking over --
`planner.resume_point(job, progress)` is available as a utility for a
future phase that does persist progress and wants to skip
already-completed states.

## 5. Pause / cancel

```
python3 -m src.creator_research.cli --cancel JOB_ID --queue ...   # -> FAILED, metadata.cancelled=true
```

Pausing is a queue-level operation (`ResearchQueue.pause()`/`resume()`)
not currently exposed as its own CLI flag beyond what `--resume`
already covers for a paused-vs-failed job -- both paths converge on
the same `--resume JOB_ID` command, which inspects the job's current
state and does the right thing (unpause if paused, retry if failed).

## 6. Validate the queue file

```
python3 -m src.creator_research.cli --validate --queue output/creator_research/queue.json
```

Read-only. Confirms every job's `status` is recognized and every
stored `job_id` matches a fresh recompute (tamper/corruption check).

## Safe demo

See `scratchpad/creator_research_demo.py` (generated during Phase
12B.1's implementation, not part of the automated suite) for a full
run against a synthetic `DummyConnector` -- job creation, `run_job()`,
progress, report, and `--validate`-equivalent structural check, all in
a temp directory.
