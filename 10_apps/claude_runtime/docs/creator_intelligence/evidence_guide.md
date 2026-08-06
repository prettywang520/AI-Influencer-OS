# Supplying Evidence

`Evidence` is the only input the Creator Intelligence Framework ever
analyzes. This guide describes how an operator supplies it, and the
hard boundaries on how it may be collected.

## Allowed evidence types

| `evidence_type` | Meaning |
| --- | --- |
| `screenshot` | A screenshot of public content, manually reviewed by a human operator |
| `manual_note` | A free-text note an operator writes after manually reviewing public content |
| `text_excerpt` | A short, manually-copied excerpt of public text (caption, bio, reply) |
| `exported_data` | Data the creator or platform has directly provided to the operator (e.g. a creator-provided media kit) |
| `operator_observation` | A general observation an operator records while reviewing public content |

This list is closed in code (`evidence.py`'s `EvidenceType`) and
enforced by `validate_evidence()` -- an evidence record with any other
`evidence_type` fails validation.

## What is never permitted, at any phase

- Scraping a profile, post, or comment section with automated tooling.
- Logging into any account, including the creator's own or a
  throwaway account, to view otherwise-inaccessible content.
- Bypassing an access control (private account, follower-only content,
  rate limits) to reach content that would not otherwise be visible.
- Collecting private messages, DMs, or any non-public communication.
- Inferring sensitive personal data (health, precise location,
  immigration status, etc.) that is not the creator's own stated
  public content.

## Practical intake steps

1. A human operator manually browses the creator's *public* profile
   in a normal browser session, as any member of the public could.
2. For each observation worth recording, the operator writes an
   `Evidence` entry: a `source_description` (what was reviewed and
   how), a short `content_excerpt` if quoting text (kept under
   `caption_excerpt_max_chars`, currently 280), and `tags` identifying
   which analyzer(s) the observation is relevant to (see the tag list
   in `intake_template.EVIDENCE_CATEGORIES`).
3. Evidence entries are validated (`creator_intelligence validate
   --evidence FILE`) before being used to build a `CreatorDNA`
   (`creator_intelligence build ...`).
4. Relationship-related evidence should be tagged with its basis
   (`presented_narrative`, `visually_depicted`, `verified`,
   `inferred`) so `HumanAuthenticityAnalyzer` can extract a
   correctly-qualified `RelationshipClaim` -- never assume a stronger
   basis than what was actually observed.

## Corroboration

Confidence rises with independently-sourced evidence
(`collected_by` differing between entries), not merely with more
entries from the same source. When multiple operators can
independently confirm the same observation, record each as a separate
`Evidence` entry with its own `collected_by`.
