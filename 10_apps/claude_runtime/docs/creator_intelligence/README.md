# Creator Intelligence Framework

**Phase:** 12A.0
**Status:** Framework only. No real creator has been analyzed.

## What this is

A framework for turning manually-supplied *Evidence* about a public
creator's observable content patterns into a structured,
confidence-scored `CreatorDNA` record, for later creative-inspiration
use. It is deliberately narrow in this phase: typed models, analyzers,
scoring, validation, serialization, and a CLI -- no evidence
collection, no scraping, no real analysis.

## What this is not

- Not a scraper. Nothing in this package fetches data from the
  internet, logs into an account, or automates a browser.
- Not identity cloning. `CreatorDNA` describes observable *patterns*,
  never a real person's private identity.
- Not connected to Aiko's active persona. Nothing here reads or
  writes `03_personas/`; any future creative use of a `CreatorDNA`
  record is an explicit, separate, future integration decision.

See [ethics.md](ethics.md) for the governing principles,
[evidence_guide.md](evidence_guide.md) for how to supply evidence, and
[phase_12a1_handoff.md](phase_12a1_handoff.md) for what a future phase
would need to analyze the two named real creators -- which this phase
does not do.

## Module map

| Module | Purpose |
| --- | --- |
| `evidence.py` | `Evidence` record + closed `EvidenceType` allow-list |
| `confidence.py` | `ConfidenceLevel` vocabulary + evidence-driven computation |
| `scoring.py` | Shared clamping/weighted-average math |
| `models.py` | `TraitScore`, `RelationshipClaim`, `CreatorDNA` |
| `analyzer_base.py` | Common `Analyzer` contract + shared scoring helper |
| `*_analyzer.py` (9 files) | One analyzer per trait category |
| `summary_builder.py` | Runs all analyzers, aggregates into one `CreatorDNA` |
| `comparison.py` | Typed reshape for a future comparison phase (no comparison logic) |
| `validation.py` | `validate_evidence()` / `validate_creator_dna()` |
| `serialization.py` | Atomic JSON save/load (+ optional YAML) |
| `config.py` | Loads `config/creator_intelligence/*.yaml` |
| `cli.py` | `validate` / `build` / `show` subcommands |
| `intake_template.py` | Generates the Phase 12A.1 intake checklist |

## Related, not reused

`docs/02_Market/Competitor.md` at the repo root sketches an unrelated
competitor-analysis outline; this framework doesn't depend on it.
