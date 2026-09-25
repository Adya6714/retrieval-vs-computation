# Cursor prompts (run in order; one prompt per Cursor session; commit after each)

Standing preamble. Paste this at the top of every prompt:
> You are working in github.com/Adya6714/retrieval-vs-computation. Operating rules: raw CSVs are truth and are append-only; derived tables regenerate by script; never invent or edit a number by hand; do not interpret results, only produce code, files and tables; if a file this prompt names does not exist or differs from what the prompt assumes, stop and report instead of guessing. Commit with a message that starts with the prompt ID (e.g. "C1:").

---

## C0. Place the handoff files
The folder `HANDOFF_2026-09-25/` was added at the repo root together with files already at their final paths. Do this:
1. Verify these files exist at their final paths (they were added directly): `docs/audit/REPO_AUDIT_2026-09-25.md`, `docs/paper/PAPER_I_REVISION_NOTES.md`, `docs/site/SITE_REDESIGN_SPEC.md`, `scripts/site/build_pipeline_explorer_data.py`, `tests/site/test_pipeline_explorer_data.py`, `site/data/pipeline_explorer.json`, `research-vault/RvC-Vault/THE_PLAN_AMENDMENT_A1.md`, `research-vault/RvC-Vault/03_Evaluation_Framework/EF-07_W3_Construct_Audit.md`, `.../EF-08_Position_On_Thinking.md`, `research-vault/RvC-Vault/04_Directions/D11_...` to `D16_...`, `research-vault/RvC-Vault/06_Operating_Manual/HP-16_...` to `HP-23_...`, `research-vault/RvC-Vault/07_Broader_Impact/BI-04_Engineering_Relevance.md`.
2. For each file in `HANDOFF_2026-09-25/appends/`, the first line is an HTML comment naming the target file and where to append. Append the content (without the comment) to that target exactly as instructed. Never rewrite or delete existing lines in the target.
3. Add a one-line pointer to `THE_PLAN_AMENDMENT_A1.md` at the top of `research-vault/RVC_MASTER_DOCUMENT.md` and at the top of `BOOK.md`'s roadmap section.
4. Replace the content of `NEXT_STAGE.MD` with: "Current plan: research-vault/RvC-Vault/THE_PLAN.md, amended by THE_PLAN_AMENDMENT_A1.md (2026-09-25). Priority list: 00_MOC.md, section 'Added 2026-09-25'."
5. Run `PYTHONPATH=. python -m pytest tests/site -q`. Report the result.
6. Check every [[wikilink]] in the new and appended notes resolves to an existing note name in the vault; list any that do not. Do not create stub notes; just report.
7. Move `HANDOFF_2026-09-25/` to `research-vault/handoffs/2026-09-25/` so the chat handoff is archived with the vault.

## C1. Data hygiene (audit section B)
Read `docs/audit/REPO_AUDIT_2026-09-25.md` section B. Then:
1. Find every row with `model == "mock"` in `results/derived/*_rescored.csv`. Trace which script writes derived rescored files; make it drop mock rows; regenerate derived files; confirm no metric in `results/derived/` or `results/paper/` changes. If any metric changes, stop and report the before/after table.
2. Find the row in `results/derived/BW_P1_behavioral_rescored.csv` whose `model` is not a model ID (e.g. "The answer is 42."). Trace it to raw; fix the parser or CSV quoting; regenerate. Report whether any BW metric changed.
3. Normalise `verifier_function` in `data/problems/question_bank_algo.csv` to one value per subtype (`verify_coinchange`, `verify_sp`, `verify_wis`); update any code that dispatches on these strings; add `tests/test_bank_schema.py` asserting the allowed set per family.
4. ALGO IDs: report every place that joins on `problem_id` for ALGO and whether 2-digit (`CC_01`) and 3-digit (`CC_031`) forms collide or fail to join. Propose (do not apply) a canonical form in the report; wait for approval before renaming IDs.
5. `configs/models.yaml`: set `display_name` of `claude-sonnet-4` to `Claude Sonnet 4`. Regenerate any table that uses display names.
6. Append a dated entry to `results/derived/C3_corrections_changelog.md` listing every change.

## C2. W3 relabel and substring bug (HP-23 steps 1 and 2 only)
Follow `research-vault/RvC-Vault/06_Operating_Manual/HP-23_W3a_Nonce_Bank.md` steps 1 and 2 only. Do not generate W3a and do not call any model. Report: number of W3 rows per family relabelled `W3b`; every row where substring replacement was detected (show a before/after snippet for each); the regenerated text for the 12 BW "You are Alice robot arm" rows. Rerun only the affected rows with the existing sweep scripts' `--resume` if and only if API keys are present; otherwise list them in `docs/trackT/T0_PENDING_RERUNS.md`.

## C3. Paper I edits
Apply `docs/paper/PAPER_I_REVISION_NOTES.md` to `paper/main.tex`:
- R1 and R2 using **Option R** text for now.
- R3 first sentence and the Vary Name sentence; leave the `% TODO-VERIFY` comment in place and do NOT add the conditional paragraph.
- R4: fetch the BibTeX for arXiv:2411.01790 from arxiv.org, add to `paper/references.bib` under key `fwdbwd2024`, insert both snippets.
- R5 and R6 as written.
- R8: regenerate tables after C1.
- Leave R7 out.
Build with the commands in `paper/README.md`; report page count and any LaTeX warnings about undefined references.

## C4. README and docs consistency
1. Root `README.md`: change the "Accepted paper" rows so `paper/main.tex` is described as "Paper I draft: What Survives a Rename (NeurIPS 2026 E&D format)" and the CAISc paper points to `paper/venue/caisc2026/`.
2. Replace the hard-coded behavioural roster sentence with a table generated from `configs/models.yaml` by a new script `scripts/consolidate/make_model_table.py` (write output into README between markers `<!-- MODELS:START -->` and `<!-- MODELS:END -->`).
3. Add a "Status" section near the top with exactly the status strip text from `docs/site/SITE_REDESIGN_SPEC.md` section 2.
4. Add a "Plan" row to the "Where everything lives" table pointing to `THE_PLAN_AMENDMENT_A1.md` and `docs/audit/REPO_AUDIT_2026-09-25.md`.
5. GitHub repo "About" cannot be set from code; write the suggested description, website URL and topics into `docs/site/GITHUB_ABOUT.md` so Adya can paste them.

## C5. Site, part 1: Pipeline Explorer
Read `docs/site/SITE_REDESIGN_SPEC.md` sections 0, 1 and 3 fully before touching code.
1. Run `PYTHONPATH=. python scripts/site/build_pipeline_explorer_data.py` and `python -m pytest tests/site -q`.
2. Implement section `#pipeline` in `site/index.html` exactly as specified in section 3: data fetch with error state, item chips, stage rail, eight stages, word-level diff for Transform (write a small LCS word-diff function; no new library), mapping table for W3, verifier animation, Probe 1 grid, Probe 2 hand-off to the existing staged player, Probe 3 strip plot with D3, honest verdict block, phase track that reuses the existing cross-tab highlight.
3. Put the explorer's JS in `site/js/pipeline.js` and its CSS in `site/css/pipeline.css`, loaded from index.html. Reuse existing CSS variables; add no new colours.
4. Do not modify the existing Probe 2 staged player or `animateTrace`, except to expose a function the explorer can call to scroll to and start the player.
5. Add `site/data/**` and `scripts/site/**` to the Pages workflow `paths` trigger. Add a workflow step that runs the build script before upload, so the JSON never goes stale. The build step must fail the deploy if the pytest guard fails.
6. Verify: keyboard-only walkthrough, reduced motion, 360 px width, both themes. Report screenshots of stages 2, 4 and 8 in both themes.

## C6. Site, part 2: content and structure
Implement `docs/site/SITE_REDESIGN_SPEC.md` sections 2, 4, 5, 6, 7, 8. Source text comes only from these notes: THE_PLAN_AMENDMENT_A1, EF-07, EF-08, BI-04, the D11 to D16 notes. Do not write new scientific claims. For CIs in section 4: read them from the linked CSVs; if a CSV has no CI column for a number, display "CI pending" and create a GitHub issue draft in `docs/site/CI_PENDING.md` listing each. Report the final list of sections in rail order.

## C7. Site, part 3: polish pass
Audit the whole page against spec section 1:
- Remove middle-dot meta strings, arrow suffixes on link text and all-caps eyebrows where they carry no information.
- Apply tabular numerals to all numbers.
- Make sure no section except `#pipeline` gains a new entrance animation.
- Run Lighthouse (accessibility, best practices) in both themes and fix anything under 95.
- Report a before/after list of every change.

## C8. Colab notebooks for Track T
Create `colab/T1_noise_floor.ipynb`, `colab/T2_qwen_family.ipynb` and `colab/T3_precision.ipynb` that clone the repo, install `requirements.txt` plus the HP's extra packages, and run the scripts specified in HP-16, HP-17 and HP-18 with `--resume`. Each notebook writes raw CSVs back via a `git push` cell that requires Adya's token as a Colab secret (never hard-code). First cell prints GPU name, CUDA, torch and transformers versions. Do not run them.

## C9. Final consistency check
1. Grep the repo for "vocabulary-disjoint", "nonce", "length-matched", "five primary models", "six models", "Claude 3.7". Report every occurrence with file and line, and whether it is now consistent with EF-07 and the model-count decision.
2. Confirm every hypothesis ID H1 to H13 and H7b appears identically in THE_PLAN_AMENDMENT_A1, 01_Program_State, the site registry and `site/data/pipeline_explorer.json`.
