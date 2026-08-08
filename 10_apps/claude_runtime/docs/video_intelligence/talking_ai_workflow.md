# Talking AI Intelligence -- Workflow

See `talking_ai_architecture.md` for the design and `talking_ai_dna.md`
for the data model. This is the practical "how do I run an analysis"
guide.

## 1. Recording evidence about a reference talking-AI video

`TalkingAIEvidence` is always operator-supplied -- no scraping, no
CV/face-landmark inference, no voice cloning. Each item is a
*simultaneous snapshot* at a timestamp; evidence may be sparse -- only
set the fields actually observed, everything else stays `None`
("not observed," never guessed as `False`/`0`). `SpeechSegment` is a
coarser, speech-level companion record (start/end, language, pauses,
emphasis):

```python
from src.video_intelligence.talking_ai.evidence import SpeechSegment, TalkingAIEvidence

video_id = "reference_talking_video_001"

segments = [
    SpeechSegment(
        video_id=video_id, start_seconds=0.0, end_seconds=2.4,
        language="cantonese", emphasis=True, pause_before=0.0,
    ),
    SpeechSegment(
        video_id=video_id, start_seconds=3.0, end_seconds=6.0,
        language="cantonese", pause_before=0.6,
    ),
]

evidence = [
    TalkingAIEvidence(
        video_id=video_id, timestamp_seconds=0.1,
        speech_active=True, mouth_open_ratio=0.32, blink=False,
        gaze_direction="direct_camera", camera_motion="static",
    ),
    TalkingAIEvidence(
        video_id=video_id, timestamp_seconds=2.7,
        speech_active=False, mouth_open_ratio=0.02, blink=True,
        gaze_direction="direct_camera",
    ),
]
```

## 2. Running an analysis

```python
from src.video_intelligence.talking_ai.workflow import TalkingAIWorkflow

workflow = TalkingAIWorkflow()  # loads config/video_intelligence/talking_ai.yaml
result = workflow.analyze_talking_video(video_id, "demo talking video", evidence, segments)

print(result.dna.dna_id, result.dna.overall_confidence)
print(result.dna.naturalness.rationale)
print(result.report_markdown)
print([p.description for p in result.patterns])
```

For a batch: `workflow.analyze_talking_videos([(video_id, label, evidence, segments), ...])`.

This single call: builds a `TalkingAIProductionDNA` (all 10 domain
analyzers + the naturalness aggregator), saves it to
`output/video_intelligence/talking_ai/<video_id>/talking_ai_dna.json`,
renders and saves a report to `.../talking_ai_report.md` and
`.../talking_ai_report.json`, and -- unless
`ingest_into_knowledge_base=False` -- folds the video's structured
patterns into the shared, de-identified pattern library.

## 3. Reading back what's been analyzed

```python
workflow.load_dna(video_id)                       # TalkingAIProductionDNA | None
workflow.load_report_markdown(video_id)            # str | None
workflow.knowledge_base().list_patterns()           # list[TalkingAIProductionPattern], ranked by support
workflow.knowledge_base().list_patterns("gaze")      # scoped to one category
```

## 4. CLI

```
python3 -m src.video_intelligence.talking_ai.workflow \
    --evidence path/to/talking_ai_batch.json \
    --output output/video_intelligence/talking_ai/_cli \
    --json
```

`talking_ai_batch.json` is a JSON list (or a single object) of
`{"video_id", "subject_label", "evidence": [...], "speech_segments": [...]}`
entries -- `evidence`/`speech_segments` items are the same fields
`TalkingAIEvidence`/`SpeechSegment` accept, with `video_id` supplied
once at the batch level. No subcommand can construct a browser, call
an external API, or run computer-vision/voice inference -- every input
is a local JSON file.

## 5. Optional: bridging from an already-analyzed Instagram Reel

```python
from src.video_intelligence.talking_ai.workflow import talking_evidence_from_reel_record

bridged_evidence = talking_evidence_from_reel_record(instagram_reel_learning_record)
```

Best-effort and deliberately lossy: only `VideoEvidence` tags with an
unambiguous, non-magnitude meaning (`"blink_timing"`, `"eye_contact"`,
`"subtitles"`, `"handheld"`, `"tripod"`, `"tracking"`, `"smile_timing"`,
`"finger_pointing"`) are mapped; a `VideoEvidence` item with no
timestamp, or whose tags map to nothing this bridge understands,
contributes no `TalkingAIEvidence` at all. This reads
`video_intelligence.instagram`'s own record type only -- it never
modifies that package and performs no live Instagram access itself.

## 6. What gets written to disk

```
output/video_intelligence/talking_ai/<video_id>/
    talking_ai_dna.json
    talking_ai_report.md
    talking_ai_report.json
output/video_intelligence/talking_ai/knowledge_base/
    patterns.json       de-identified TalkingAIProductionPattern records
    membership.json      internal, uncapped video_id-per-pattern coverage
                          tracking (never exposed directly -- see talking_ai_dna.md)
```

`output/` is gitignored; nothing under
`output/video_intelligence/talking_ai/` is ever committed by this
package.
