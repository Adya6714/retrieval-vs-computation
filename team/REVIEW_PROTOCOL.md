# Review Protocol

This document defines how every contribution to the problem set is reviewed before it enters the repo. Nothing merges without following this process.

---

## Why this exists

The problem set is the foundation of the paper. A wrong answer in the CSV means a wrong result in the paper. A missing source means a reviewer can't reproduce the instance. An inconsistent variant means the surface invariance test is invalid. Review catches these before they propagate.

---

## What requires review

| Contribution type | Reviewer required |
|-------------------|-------------------|
| New instances in `probe1_instances.csv` | Yes — one reviewer, not the selector |
| New instances in `probe2_instances.csv` | Yes — one reviewer, not the selector |
| New variants in `probe1_variants.csv` | Yes — one reviewer, not the writer |
| Changes to existing instances or variants | Yes — Adya reviews all edits to existing rows |
| New entries in `INSTANCE_SELECTION_CRITERIA.md` | Adya reviews |
| Code changes (scripts, verifiers) | Adya reviews all code |

---

## How to submit for review (GitHub pull request)

1. Make your changes on a new branch. Branch name format: `your-name/description`. Example: `shaswat/blocksworld-instances` or `nandini/sp-variants-w1`.
2. Open a pull request (PR) against `main`.
3. PR title format: `[instances] family_name: what you added` or `[variants] problem_id: variant types added`.
   - Example: `[instances] blocksworld: 8 canonical instances`
   - Example: `[variants] SP_001-SP_004: W1 and W2`
4. In the PR description, fill in the following:

```
## What's in this PR
- [ ] How many instances / variants added
- [ ] Which families / sub-types
- [ ] Which instances are procedurally generated (list seeds)
- [ ] Which instances are W5 (answer differs from canonical)

## Self-check completed
- [ ] Every answer independently verified
- [ ] Source documented for every instance
- [ ] No LLM used for any variant
- [ ] Format rules followed (correct_answer column, problem_id format)

## Flagged for reviewer
(Anything you were unsure about — list here. Do not omit and hope the reviewer doesn't notice.)
```

5. Assign Nandini as reviewer. Nandini will assign a second reviewer if needed or escalate to Adya.

---

## Reviewer checklist — instances

For each instance in the PR:

**Answer correctness**
- [ ] Independently solved the problem and got the same correct answer
- [ ] Answer is in canonical format (see format rules in `shasshy.md`)

**Source documentation**
- [ ] `source_dataset` is filled and refers to a real source
- [ ] `source_instance_id` is filled (ID or `generated_seed_N`)
- [ ] If procedural: generation script is committed and seed is documented

**Difficulty and contamination**
- [ ] Difficulty tier matches `INSTANCE_SELECTION_CRITERIA.md` definitions
- [ ] Contamination pole matches the source type

**Instance validity**
- [ ] Answer is unambiguous (unique optimal answer, or bounded valid set)
- [ ] Problem is not trivially solvable by inspection
- [ ] For Blocksworld: plan was simulated step by step and reaches goal state

If any item fails: comment on the specific row in the PR. Do not approve until fixed.

---

## Reviewer checklist — variants

For each variant in the PR:

**Answer correctness**
- [ ] For W1–W4, W6: correct answer matches canonical exactly
- [ ] For W5: correct answer was independently computed from the reversed problem
- [ ] Independently solved the variant and got the same answer as `correct_answer`

**Variant validity**
- [ ] W1: wording is genuinely rewritten, not just synonym substitution
- [ ] W2: all information from original is present, nothing dropped
- [ ] W3: entity mapping is consistent throughout, no label missed
- [ ] W4: formal notation is correct and complete
- [ ] W6: generated instance is valid (answer correct, difficulty appropriate)
- [ ] W5: reversal is well-defined and unambiguous

**Meta**
- [ ] `variant_id` follows format `PROBLEMID_WTYPE`
- [ ] No LLM used (ask the writer directly if unsure)
- [ ] Variant is not trivially identical to another variant of the same problem

---

## Review tracking

Nandini maintains a review log at `team/review_log.csv` with these columns:

| Column | What to put |
|--------|-------------|
| `contribution_id` | PR number or a short descriptor |
| `contribution_type` | `instance` or `variant` |
| `item_id` | `problem_id` or `variant_id` |
| `submitted_by` | Name |
| `reviewer` | Name |
| `status` | `pending`, `approved`, `needs_fix`, `merged` |
| `date_submitted` | Date |
| `date_resolved` | Date when approved or closed |
| `notes` | Any flags or issues |

Update this log when a PR is opened, when review is assigned, and when it closes.

---

## Escalation

Escalate to Adya when:
- Reviewer and submitter disagree on whether an answer is correct
- An instance or variant doesn't fit cleanly into any tier definition in `INSTANCE_SELECTION_CRITERIA.md`
- A source cannot be verified
- Any code needs to be written or changed
- A W5 reversal is ambiguous and you can't resolve it

Do not merge anything escalated until Adya responds.

---

## What happens if something wrong gets merged

Flag it immediately. Open a new PR titled `[fix] description of what's wrong`. Do not silently edit a merged row — the fix must go through review too so there's a record of what changed and why.
