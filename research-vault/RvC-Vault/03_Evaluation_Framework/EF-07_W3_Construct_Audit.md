# EF-07 W3 construct audit: nonce rename vs cover-story isomorph

status: verified-raw (bank inspection 2026-09-25) · addresses: [[EF-01_Probe1_Surface_Invariance]], [[D05_Cross_Linguistic]], [[D06_Direction_Asymmetry]] · evidence: [[docs/audit/REPO_AUDIT_2026-09-25]]
gate impact: none on G0; Paper I text must change before resubmission.

## What the documents say W3 is
- Paper I and vault: entity rename to a vocabulary-disjoint, held-out, length-matched nonce token set.

## What the bank W3 actually is
- GSM: cover-story swap into a new real-word domain, entity mapping stored in `notes`.
- ALGO: domain relabel with real words (coins to stamps, node indices to city names, events to named sessions).
- BW: action-verb and frame relabel with real words (recruit, dismiss, promote; HR reporting chain).
- Plus a substring bug in 12/65 BW rows ("You are Alice robot arm").

## Why it matters scientifically
- A nonce rename tests **entity binding** with no competing semantics. A cover-story swap tests **transfer across problem isomorphs**, where the new domain brings its own priors (stamps have "postage", hikers have "calories").
- These predict different things:
  - Nonce rename cost should shrink with a tokenizer-aware length match and should not depend on the new domain.
  - Isomorph cost should depend on the new domain's distance from the canonical one and on whether the new domain's priors conflict with the procedure.
- Human baseline exists for isomorphs: Tower of Hanoi isomorphs differ several-fold in solve time by cover story (Kotovsky, Hayes & Simon 1985); analogical transfer across cover stories is poor without a hint (Gick & Holyoak 1980). Both quarantined until read ([[P-00_Ingestion_Queue]]).

## Two ways to fix, choose one and record the choice in [[01_Program_State]]
- **Option R (relabel, cheap, no new runs).** Rename the construct in Paper I to "W3 cover-story isomorph". Rewrite F1 as "numeric regeneration within a template is nearly free; moving the same template into a new domain is expensive". Drop the nonce-token tokenisation defence. Cite the isomorph literature. Cost: text only.
- **Option S (split, stronger, small API spend).** Keep current W3 as **W3b cover-story isomorph** and add **W3a true nonce rename** (vocabulary-disjoint, length-matched in tokens per model tokenizer, same frame, same verbs). Run W3a on the canonical-correct subset only. The 2 × 2 that matters: {nonce, real-word} × {frame kept, frame swapped}. This separates binding from domain transfer, which is the cleanest possible version of F1 and directly feeds H2 and D05.
  - Recommended: Option S. It turns a documentation error into a new dissociation.
  - Pre-register before running: primary contrast Acc(W3a) vs Acc(W3b) on items with Acc_can ≥ .30; cluster bootstrap for ALGO, Wilson for GSM/BW; no threshold changes afterwards.

## Downstream edits required
- [[D05_Cross_Linguistic]]: its "entities kept vs localized" 2 × 2 must specify which W3 it compares against.
- [[HP-11_D6_Direction_Probe]]: its "W3 nonce rename" cell needs W3a, not the current bank W3.
- Site: Probe 1 "Cannot discharge alone" copy about length-matched nonces.
- Paper I: method paragraph, F1 wording, related work (GSM-Symbolic Vary Name is names only).
