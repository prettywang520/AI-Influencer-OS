# Talking AI Live Learning Adapter -- Runbook

See `talking_ai_live_adapter.md` for the design. This document covers
two very different things: (1) running the adapter **today**, entirely
offline, against hand-built or already-collected evidence; and (2)
what a **future, separately authorized** live learning run against
`https://www.instagram.com/howdoyoulikedurian/` would look like. Part
2 has not happened and does not happen as part of this phase.

## Part 1 -- Running the adapter today (offline, no browser)

### From hand-built evidence

```python
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence
from src.video_intelligence.talking_ai.live.models import LiveTalkingReelRecord
from src.video_intelligence.talking_ai.live.workflow import learn_live_talking_reel

record = LiveTalkingReelRecord(
    source_url="https://www.instagram.com/reel/<public reel>/",
    creator_label="reference_creator",  # free text, never a real handle requirement
    duration_seconds=14.5,
    language_observations=["cantonese"],
    timeline_observations=[
        TalkingAIEvidence(video_id="", timestamp_seconds=0.4, mouth_open_ratio=0.32, blink=False, gaze_direction="direct_camera"),
        TalkingAIEvidence(video_id="", timestamp_seconds=1.8, blink=True),
    ],
    speech_segments=[
        SpeechSegment(video_id="", start_seconds=0.4, end_seconds=2.6, language="cantonese", pause_before=0.0),
    ],
    camera_observations=["static", "direct_to_camera"],
)

outcome = learn_live_talking_reel(record)
print(outcome.dna.dna_id, outcome.overall_confidence)
```

### From an already-collected `ReelRecord` (recovers subtitle text_overlays)

```python
from src.creator_research.instagram.models import ReelRecord
from src.video_intelligence.talking_ai.live.observer_bridge import live_talking_reel_from_reel_record

reel = ReelRecord(
    reel_url="https://www.instagram.com/reel/<public reel>/",
    text_overlays=["hello everyone", "let's talk about durian"],
)
record = live_talking_reel_from_reel_record(reel, creator_label="reference_creator")
```

### From an already-collected creator_research `Evidence` list

```python
from src.video_intelligence.talking_ai.live.mapper import live_talking_reels_from_creator_research
from src.video_intelligence.talking_ai.live.workflow import learn_live_talking_reels

records = live_talking_reels_from_creator_research(evidence, creator_label="reference_creator")
report, outcomes, diagnostics = learn_live_talking_reels(records)
print(report.reels_analyzed, report.production_dna_ids)
```

### Ordinal-only timing (no seconds known, only sequence)

```python
from src.video_intelligence.talking_ai.live.timing_bridge import assign_ordinal_timestamps

ordered = assign_ordinal_timestamps(items, step_seconds=0.5)  # never fakes real elapsed time
```

### CLI

```
python3 -m src.video_intelligence.talking_ai.live.workflow \
    --evidence path/to/live_talking_reels.json \
    --output output/video_intelligence/talking_ai/live/_cli

python3 -m src.video_intelligence.talking_ai.live.workflow \
    --evidence path/to/live_talking_reels.json --output ... --validate-only
```

No subcommand can construct a browser, call an external API, or run
computer-vision/voice inference -- every input is a local JSON file.
Writes `talking_ai_live_learning_report.json`/`.md` and
`diagnostics_<run_id>.json` to `--output`.

## Part 2 -- A future, separately authorized live run

**Nothing below has been run. Nothing below runs automatically.** It
requires a separate, explicit go-ahead from the user, in addition to
this phase's own implementation approval.

### Target

`https://www.instagram.com/howdoyoulikedurian/` -- named by the task as
the future approved reference account for studying generalized
Talking AI production techniques (Cantonese speech rhythm, mouth/
speech timing, pause behavior, blink/gaze variation, head micro-
motion, gesture/speech sync, direct-to-camera framing, subtitle
timing, hook pacing, editing rhythm, visible temporal artifacts).

### Planned scope, if and when separately authorized

- Reel limit: matches `config/video_intelligence/talking_ai_live.yaml`'s
  `sampling.recommended_talking_reels` (currently 15), never an
  unbounded crawl.
- Sections to collect: **Reels only** (`ConnectorSection.REELS` via
  `creator_research.instagram.connector.InstagramResearchConnector`),
  reusing that connector exactly as-is -- this package creates no new
  browser code, no new selectors, no new crawler, no new login flow.
- Browser profile: `browser_profile/creator_research_instagram` (the
  research connector's own dedicated profile, never
  `browser_profile/instagram`, the reply-production system's profile).
- Output location: `output/video_intelligence/talking_ai/live/` (this
  package) plus `output/creator_research/instagram/` (the connector's
  own checkpoints/diagnostics) -- both already gitignored.
- Checkpoint/resume behavior: unchanged from
  `docs/creator_research/instagram_connector.md` -- one JSON file per
  `(job_id, section)`, resumed only after `job_id`/`creator_id`/
  `connector_version` revalidation.

### Actions that WILL occur, if authorized

- Navigate the public profile.
- Read visible Reel references.
- Open visible public Reels.
- Observe visible captions/metrics/frames.
- Observe publicly visible subtitles.
- Checkpoint research progress.

### Actions that will NEVER occur

Follow, like, comment, DM, save, share, publish, edit the profile,
bypass a CAPTCHA or login wall, access a private account, reverse-
engineer a hidden API, rotate a proxy, or extract a credential --
exactly the same forbidden-action list
`docs/creator_research/connector_contract.md` already enforces for
every connector in this codebase, unchanged here.

### Before that future run

1. Re-read `docs/creator_research/instagram_connector.md`,
   `connector_contract.md`, and `instagram_live_research_runbook.md`
   in full -- this package's own future live step would still be
   `creator_research/instagram`'s connector, governed entirely by
   those documents, not by anything new here.
2. Confirm this package's own offline test suite still passes:
   `python3 -m unittest discover -s src/video_intelligence/talking_ai/live -p "test_*.py" -t .`
3. Get the user's separate, explicit authorization for the specific
   scope above -- this runbook describes the plan, it does not
   constitute that authorization.
