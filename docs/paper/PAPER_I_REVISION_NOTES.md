# Paper I revision notes (2026-09-25)

target file: `paper/main.tex` ("What Survives a Rename", NeurIPS 2026 E&D draft)
source of every issue: `docs/audit/REPO_AUDIT_2026-09-25.md`, [[EF-07_W3_Construct_Audit]], [[EF-08_Position_On_Thinking]]
rule: no number in these snippets is new. Every number is copied from the current draft. Anything marked TODO-VERIFY must be checked in the primary source before the edit is kept.

## R1. W3 description (required)
Current (method, around line 120): "W3 entity rename to a vocabulary-disjoint held-out token set".
The bank does not do this (EF-07). Pick the variant that matches the option chosen in 01_Program_State.

Option R (text only):
```latex
W3 \emph{cover-story isomorph}: entities, actions and framing are mapped into a
different real-world domain (e.g.\ coins to stamps, a phone tariff to a calorie
budget, block moves to a reporting chain) while numbers, structure and gold answer
are preserved; the mapping is persisted and consumed by the verifier.
```

Option S (after HP-23 runs), replace with two lines:
```latex
W3a \emph{nonce rename}: every entity name is replaced by a held-out nonce token,
length-matched within one token under each evaluated tokenizer; frame, verbs and
numbers are unchanged. W3b \emph{cover-story isomorph}: entities, actions and
framing are mapped into a different real-world domain with numbers, structure and
gold preserved. Both mappings are persisted and consumed by the verifier.
```

## R2. Abstract, result (i)
Current: "renaming its entities costs a great deal".
Option R:
```latex
while moving the same problem into a different cover story costs a great deal
($.909 \rightarrow .523$), on the same algorithm at the same difficulty.
```
Option S: report W3a and W3b separately and state which one carries the drop.

## R3. Related work, GSM-Symbolic (required)
Replace the sentence calling Vary Name "the direct predecessor of our entity-rename transformation":
```latex
\citet{mirzadeh2025gsmsymbolic} regenerate GSM8K templates with new names and
values and report group-level decline. % TODO-VERIFY: exact direction of their
% name-vs-number result, with section and figure number.
Their \emph{Vary Name} condition changes proper names only; our W3b changes the
entire cover story, and our W6 changes values within a fixed template. The two
manipulations are therefore not comparable one to one, and we report the contrast
explicitly rather than treating our W3 as a replication of theirs.
```
If TODO-VERIFY confirms they found numbers more damaging than names, add:
```latex
Their larger sensitivity to value changes than to name changes runs opposite to
our W6/W3 contrast; the natural reconciliation is that our W6 preserves the
template while our W3b replaces the domain.
```

## R4. F3 prior art (required)
Add the forward/backward planning asymmetry citation (arXiv:2411.01790; fetch the BibTeX from arXiv, key suggestion `fwdbwd2024`) to Related Work and to the F3 paragraph:
```latex
Direction asymmetry in planning has been reported before and linked to search
complexity \citep{fwdbwd2024}. We replicate it on PlanBench Blocksworld items and
leave its mechanism open; a pre-registered discriminating test (backward-reasoning
instruction on canonical items versus constraint asymmetry per item) is specified in
our repository.
```

## R5. Exploratory labelling (required)
Findings computed before PREREGISTRATION.md was committed are exploratory. Add one sentence at the start of Results:
```latex
All results in this section are exploratory with respect to the pre-registration,
which governs subsequent calibration and mechanism phases; thresholds reported here
(e.g.\ the $\mathrm{Acc}_{\mathrm{can}} \geq .30$ retention floor) were fixed before
the corresponding analyses were run.
```
Check the second clause is true for each threshold before keeping it.

## R6. Limitations: per-instance status and noise floor
```latex
The instrument is designed for per-instance diagnosis, but we do not issue
per-instance retrieval or computation labels: across a 270-configuration threshold
sweep the retrieval-consistent count ranges from 0 to 170 of 440, and mixture Rasch
models select a single class on arithmetic. Per-instance verdicts may also be
sensitive to the inference stack (batch size, kernel, accumulation precision) even at
temperature zero; we quantify this floor on open-weight models in follow-up work.
```

## R7. Discussion paragraph (optional, recommended)
Use the one-paragraph version in EF-08. Remove the human-literature sentence if the P-notes are not written by submission time.

## R8. Table and figure hygiene
- Model display names: regenerate any table that reads `configs/models.yaml` after fixing `Claude 3.7 Sonnet` to `Claude Sonnet 4`.
- Confirm "six models" vs "five primary models" per family against `results/derived/COVERAGE_MASTER.csv` and state per-family n in Table 2.
