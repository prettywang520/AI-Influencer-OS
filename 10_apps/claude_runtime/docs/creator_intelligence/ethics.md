# Creator Intelligence Framework -- Ethics

This document governs every current and future use of the Creator
Intelligence Framework, including the real-creator analysis a future
Phase 12A.1 would perform. It is binding, not aspirational: code in
this package is structured (closed `EvidenceType` allow-list,
basis-qualified `RelationshipClaim`s, excerpt length caps) specifically
to make several of these rules hard to violate by accident.

## Core principles

1. **Analyze public content patterns, not private identity.** A
   `CreatorDNA` record describes observable style (tone, framing,
   posting cadence, caption structure) -- never a real person's
   private life, location, or identity beyond what they have
   themselves made public.

2. **Never impersonate a creator.** Nothing produced by this
   framework, or by any future phase building on it, may present
   itself as being from, or speaking as, the analyzed creator.

3. **Never clone a unique voice verbatim.** Caption/reply excerpts
   captured as evidence are short and citation-only (enforced by
   `caption_excerpt_max_chars`); a `CreatorDNA` record is a *pattern
   description*, not a corpus for reproduction.

4. **Never falsely claim relationships or shared history.** Every
   relationship-related observation is a `RelationshipClaim` with an
   explicit `basis` (`presented_narrative`, `visually_depicted`,
   `verified`, `inferred`, `unknown`). A claim is only ever `verified`
   when at least two independently-sourced pieces of evidence
   corroborate it -- a single source's own assertion is never
   sufficient, no matter how it's tagged.

5. **Never use childhood or family imagery to manufacture a false
   real-person biography.** A childhood photo is evidence of a
   childhood-photo *narrative* only -- it is never treated as
   verifying a real biographical fact.

6. **Any future Aiko-facing adaptation must stay clearly fictional and
   separated from the source creator.** If a future phase uses a
   `CreatorDNA` record as creative inspiration for Aiko, that
   adaptation must be presented as fictional, must not claim to be
   "based on" the real creator publicly, and must go through its own
   explicit, separate integration phase -- never an automatic or
   implicit step of this framework.

## Structural enforcement in this phase

- `EvidenceType` is a closed set (`screenshot`, `manual_note`,
  `text_excerpt`, `exported_data`, `operator_observation`) -- there is
  no automated-collection member to construct evidence from in the
  first place.
- No module in this package imports a network, browser-automation, or
  Instagram-session library, or anything under `03_personas/`,
  `social/`, or `publishing/` -- enforced by
  `test_structural_safety.py`.
- Confidence is always derived from evidence quantity/corroboration,
  never asserted by an analyzer's own judgement; zero evidence is
  always `unknown`.

## Scope of Phase 12A.0

This phase builds the framework only. It does not collect evidence
about, or analyze, `carlysuen112` or `aitana_10_01` (see
[phase_12a1_handoff.md](phase_12a1_handoff.md)). Any future phase that
does perform real analysis must continue to honor every principle
above.
