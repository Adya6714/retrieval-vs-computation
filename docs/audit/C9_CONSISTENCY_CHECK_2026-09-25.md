# C9 consistency check (2026-09-25)

Standing decision (EF-07 Option R + Option S pending): bank W3 is **W3b cover-story isomorph**; true **W3a nonce** (vocabulary-disjoint, length-matched) is planned via HP-23, not yet in the bank.

Model-count decision: **five primary models**, plus **R1-distill-70B on planning only** (`COVERAGE_MASTER.csv` / SITE_REDESIGN_SPEC §2).

---

## 1. Grep report

Paths under `Retrieval vs Computation…/`, `Speaker_Notes*`, and archived handoff *copies* are omitted below except where the live vault still carries the same line. Verdicts use: **OK** = consistent with EF-07 / model-count; **HIST** = historical audit/revision note describing the old bug; **STALE** = still asserts the pre-EF-07 or wrong model-count claim in an active artefact; **ARCHIVE** = frozen CAISc / extract.

### 1.1 `vocabulary-disjoint`

| File:line | Snippet / role | Verdict |
|-----------|----------------|---------|
| `research-vault/RvC-Vault/03_Evaluation_Framework/EF-07_W3_Construct_Audit.md:7` | Documents what Paper I *said* W3 was | **OK** (audit of mismatch) |
| `…/EF-07_….md:24` | Defines future W3a as vocabulary-disjoint | **OK** (Option S target) |
| `docs/audit/REPO_AUDIT_2026-09-25.md:10` | Quotes old Paper I wording | **HIST** |
| `docs/paper/PAPER_I_REVISION_NOTES.md:8` | Same quote in revision checklist | **HIST** |
| `paper/appendix.tex:93` | Paper I appendix still describes nonce W3 tokens | **STALE** — body Option R done; appendix nonce wording not fully rewritten |
| `paper/venue/caisc2026/appendix_caisc.tex:81` | CAISc predecessor appendix | **ARCHIVE** (accepted predecessor; leave) |
| `results/paper/_pdf_extract.txt:240` | Extract of older PDF | **ARCHIVE** |

### 1.2 `nonce`

| File:line | Role | Verdict |
|-----------|------|---------|
| `README.md:79` | W3 cover-story isomorph (W3b; nonce W3a in progress) | **OK** |
| `PREREGISTRATION.md:302` | T0 H13 W3a vs W3b | **OK** |
| `docs/site/SITE_REDESIGN_SPEC.md:64,92` | Spec for W3a caption / Probe 1 copy | **OK** |
| `site/js/pipeline.js:192`, `site/index.html` Probe 1 / H13 / T0 | Isomorph now; W3a pending | **OK** |
| `site/data/pipeline_explorer.json` W3 meanings | Points to EF-07 / W3a | **OK** |
| `paper/main.tex` | Method uses cover-story isomorph / W3b (C3 Option R) | **OK** |
| `paper/appendix.tex:39,90` | Still “nonce tokens” for W3 | **STALE** |
| `paper/venue/caisc2026/*` | Predecessor “nonce-token rename” | **ARCHIVE** |
| `research-vault/…/EF-07_*.md`, HP-23, D05/D06 appends, Program State | Construct audit + pending W3a | **OK** |
| `site/index.html` probe1 viz `nonce:` field keys | UI demo labels for renamed entities (not W3 construct claim) | **OK** (unrelated key name) |
| `docs/audit/*`, `PAPER_I_REVISION_NOTES` | Record the mismatch | **HIST** |

### 1.3 `length-matched`

| File:line | Role | Verdict |
|-----------|------|---------|
| `EF-07_….md:7,24` | Old claim vs W3a design | **OK** |
| `EF-01_Probe1_Surface_Invariance.md:9` | Still “length-matched nonce vocabulary” as Probe 1 bound | **STALE** — should cite EF-07 / W3a pending (site Probe 1 copy already updated) |
| `SITE_REDESIGN_SPEC.md:92` | Instructs replacing that site copy | **OK** (done in C6) |
| `REPO_AUDIT…:17`, `PAPER_I_REVISION_NOTES.md:22` | Audit / W3a spec | **HIST** / **OK** |
| `paper/appendix.tex:91`, `caisc2026/appendix_caisc.tex:79`, `_pdf_extract.txt` | Appendix / archive | **STALE** / **ARCHIVE** |

### 1.4 `five primary models`

| File:line | Verdict |
|-----------|---------|
| `README.md:46`, `paper/main.tex:162,279`, `site/index.html` (position, F4, questions, experiments) | **OK** |
| `SITE_REDESIGN_SPEC.md:29`, `REPO_AUDIT…:42`, `PAPER_I_REVISION_NOTES.md:90` | Spec / checklist | **OK** / **HIST** |

### 1.5 `six models`

| File:line | Verdict |
|-----------|---------|
| `EF-08_Position_On_Thinking.md:27` | Was **STALE**; **fixed in C9** → “five primary models” (matches site) |
| `paper/tables/table_bw_accuracy.tex:3` + `emit_paper_neurips_tables.py:297` | “Four of six models” on BW matched pairs — includes R1-distill on planning | **OK** (planning roster, not the five-primary claim) |
| `REPO_AUDIT…:42`, `PAPER_I_REVISION_NOTES.md:90` | Checklist of the discrepancy | **HIST** |

### 1.6 `Claude 3.7`

| File:line | Verdict |
|-----------|---------|
| `configs/models.yaml` | `display_name: Claude Sonnet 4` | **OK** (C1 fixed) |
| `01_Program_State.md:61` | Hygiene bullet still *says* yaml had Claude 3.7 | **HIST** (changelog of the bug; name is fixed in yaml) |
| `REPO_AUDIT…:37`, `PAPER_I_REVISION_NOTES.md:89` | Audit / revision notes | **HIST** |

---

## 2. Hypothesis IDs H1–H13 and H7b

Wanted set: `H1…H13`, `H7b`.

| Source | Present | Missing | Notes |
|--------|---------|---------|-------|
| `THE_PLAN_AMENDMENT_A1.md` | H7b, H8–H13 | H1, H3–H6 | Amendment only **adds** Track T hyps; does not restate H1–H7 registry. **Expected.** |
| `01_Program_State.md` | all 14 | — | Full registry + H8–H13/H7b append |
| Site `DATA.hypotheses` (`site/index.html`) | all 14 | — | IDs match; short titles aligned with Program State / A1 |
| `site/data/pipeline_explorer.json` `phase_routing[].claims` | all 14 | — | **H2 added to phase 1 in C9** (was missing) |

### Wording identity (H8–H13 / H7b)

IDs match across A1, Program State, and site. **Prose is not byte-identical**: A1 has full hypothesis sentences; Program State has short status lines with wikilinks; site has intermediate “text” / “detail” fields. Semantically aligned; not verbatim clones.

---

## 3. Remaining follow-ups (not blocking C9 report)

1. Rewrite `paper/appendix.tex` W3 nonce / vocabulary-disjoint / length-matched paragraphs to Option R (W3b) + W3a pending — body already done in C3.
2. Update `EF-01_Probe1_Surface_Invariance.md` “Assumption this probe alone cannot discharge” to EF-07 caption (site already updated).
3. Optionally add a one-line pointer in A1: “H1–H7 remain as in 01_Program_State” so a naive ID grep is complete.
4. Leave CAISc `paper/venue/caisc2026/` archive wording as-is.

---

## 4. Fixes applied in this C9 pass

- `EF-08_Position_On_Thinking.md`: “six models” → “five primary models”.
- `scripts/site/build_pipeline_explorer_data.py`: phase 1 `claims` includes `H2`; JSON regenerated; `tests/site` pass.
