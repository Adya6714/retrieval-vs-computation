# Handoff 2026-09-25: what this is and how to apply it

Produced from a full clone of the repo on 2026-09-25 plus a review of the site. Every "verified-raw" statement was checked against committed files; everything from memory or outside literature is marked quarantined or TODO-VERIFY, per operating rule 2.

## The one thing to read first
`docs/audit/REPO_AUDIT_2026-09-25.md` section A1: W3 in the bank is a cover-story / domain isomorph (real words), not the vocabulary-disjoint nonce rename that Paper I and the vault describe. Paper I must change before resubmission. `EF-07_W3_Construct_Audit.md` gives the two fixes; Option S (add a true nonce W3a) is recommended and turns the error into a new dissociation.

## File map (drop the whole zip at repo root; paths already match)
| File | Purpose |
|---|---|
| `docs/audit/REPO_AUDIT_2026-09-25.md` | All verified problems: W3 construct, Alice bug, hygiene, doc drift |
| `research-vault/RvC-Vault/THE_PLAN_AMENDMENT_A1.md` | Track T: T4-feasible work while Phase 1 is blocked; H8 to H13, H7b; kill criteria |
| `.../03_Evaluation_Framework/EF-07_W3_Construct_Audit.md` | W3 nonce vs isomorph; Option R vs S |
| `.../03_Evaluation_Framework/EF-08_Position_On_Thinking.md` | The "can LLMs think" position, tied to findings |
| `.../04_Directions/D11` to `D16` | Noise floor, precision, continued pretraining, distillation, RLVR, surface ensembling |
| `.../06_Operating_Manual/HP-16` to `HP-23` | Cold-executable handoffs for each |
| `.../07_Broader_Impact/BI-04_Engineering_Relevance.md` | What the instrument does and does not offer engineering |
| `docs/paper/PAPER_I_REVISION_NOTES.md` | LaTeX snippets R1 to R8 |
| `docs/site/SITE_REDESIGN_SPEC.md` | Full site spec incl. Pipeline Explorer |
| `scripts/site/build_pipeline_explorer_data.py` | Builds explorer JSON from committed CSVs (tested) |
| `tests/site/test_pipeline_explorer_data.py` | Guards JSON against bank drift (4 tests pass) |
| `site/data/pipeline_explorer.json` | Current build: GSM_047, BW_497, CC_035, WIS_001 |
| `HANDOFF_2026-09-25/appends/*` | Append blocks for existing notes and PREREGISTRATION.md |
| `HANDOFF_2026-09-25/CURSOR_PROMPTS.md` | C0 to C9 |

## Order
1. Commit the files as-is.
2. Run C0 (placement, appends, archive handoff).
3. Commit PREREGISTRATION Track T (done by C0) before any Track T run.
4. Run C1 (hygiene), then C2 (W3 relabel, Alice fix).
5. Run C3 (paper), C4 (README).
6. Run C5, C6, C7 (site) in that order.
7. Run C8 (notebooks), then execute HP-16 on Colab.
8. Run C9.
9. After each Track T report lands, open a reasoning session (not Cursor) to interpret and write back to 01_Program_State.

## Decisions only Adya can make
- EF-07: Option R (text only) or Option S (add W3a). Recommended: S.
- ALGO ID canonical form (C1 step 4).
- Model count wording (five primary + one planning-only, or six).
