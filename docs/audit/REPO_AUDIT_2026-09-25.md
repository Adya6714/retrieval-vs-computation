# Repo audit, 2026-09-25

status: verified-raw (every item below was checked against committed files at the commit cloned on 2026-09-25)
addresses: [[EF-07_W3_Construct_Audit]], [[01_Program_State]]
owner: Adya. Fixes are execution tasks for Cursor; interpretation stays in reasoning sessions (rule 6).

## A. Blocking for Paper I (fix before any resubmission)

### A1. W3 in the bank is not the W3 the paper describes
- `paper/main.tex` (method section, around line 120) says: "W3 entity rename to a vocabulary-disjoint held-out token set".
- The vault (D05, HP-11) also assumes "W3 nonce rename".
- The committed banks do something different:
  - GSM (24/44 W3 rows carry a `chosen_domain` note): the whole cover story is swapped into a new real-word domain. Example GSM_001: a hotel phone call becomes a hiker burning calories. Numbers and gold are unchanged. The `notes` column stores the `entity_mapping` (e.g. `oranges -> widgets`, `harvest -> produce`).
  - ALGO coin_change: coins become stamps and "postage". Real words.
  - ALGO shortest_path: node labels become real city names (Berlin, Hamburg, London, Oxford).
  - BW: action verbs are renamed to real words (`recruit`, `dismiss`, `promote`, `retrieve`, `shelve`), sometimes with a new frame ("You are an HR manager building a reporting chain").
- Consequence: W3 is a cover-story / domain isomorph, not a nonce rename. That is a different construct with a different literature (problem isomorphs, analogical transfer). F1's interpretation ("renaming entities costs a great deal") must be rewritten, and the site's "length-matched nonce tokens bound tokenisation noise" defence does not apply to these items.
- Fixable, and arguably makes the finding more interesting. See [[EF-07_W3_Construct_Audit]].

### A2. Rename artefact: "You are Alice robot arm"
- 12/65 BW W3 rows contain "You are Alice robot arm". The generator replaced the article "a" with an entity name: a substring-replacement bug.
- Action: audit every W3 row for whole-word vs substring replacement; regenerate affected rows; rerun only those rows (`--resume`); log in `C3_corrections_changelog.md`.

### A3. GSM-Symbolic comparison is unaddressed
- Related Work calls GSM-Symbolic's Vary Name "the direct predecessor" of W3. Vary Name changes proper names only; our W3 changes the whole domain. Say so.
- Quarantined recollection (verify in the primary source before writing): GSM-Symbolic reported larger drops for changed numbers than for changed names. If confirmed, F1 runs the opposite way on numbers, and the likely reconciliation is that our W6 regenerates numbers inside the same template while our W3 swaps the cover story. Write this explicitly.

### A4. Forward/backward planning prior work is not cited in Paper I
- HP-11 already cites arXiv:2411.01790 (search-complexity asymmetry); D06 says "do not pitch as a new phenomenon".
- Paper I presents F3 without it. Add the citation; reframe F3 as "replicates direction asymmetry on PlanBench items; mechanism open"; point to D06 and HP-22.

## B. Data hygiene (fix, regenerate, recommit derived)
- `results/derived/*_P1_behavioral_*_rescored.csv` still contain `model == "mock"` rows (ALGO claude, ALGO llama, BW combined). Confirm exclusion upstream of every metric, then drop them from derived files.
- `results/derived/BW_P1_behavioral_rescored.csv` has a row whose `model` is `"The answer is 42."`: a column shift. Trace to raw, fix the parser.
- `question_bank_algo.csv` `verifier_function` values are inconsistent: `veryify_WIS` (typo), `verify_wis`, `verify_sp`, `Dijkstra`, `Bellman-Ford`, `verify_coinchange`. Normalise to one per subtype; assert in a test.
- ALGO IDs mix widths: 10 are 2-digit (`CC_01`), 100 are 3-digit (`CC_031`). `ALGO_P2_per_instance_cci.csv` uses 2-digit. Pick one canonical form; add a join test so no CCI row silently drops.
- `configs/models.yaml`: `claude-sonnet-4` has `display_name: Claude 3.7 Sonnet`. Wrong name in any generated table.

## C. Documentation consistency
- Root `README.md` calls `paper/main.tex` the accepted CAISc paper. `paper/README.md` correctly says `main.tex` is the NeurIPS 2026 E&D draft and CAISc lives in `paper/venue/caisc2026/`.
- Root README roster lists 3 models; the paper uses 6. Generate the table from `configs/models.yaml`.
- Site: "fail all five primary models"; paper: six models. Standardise on "five primary models, plus R1-distill-70B on planning only", or whatever `COVERAGE_MASTER.csv` supports.
- `NEXT_STAGE.MD` is a stray chat instruction. Replace with a pointer to `THE_PLAN_AMENDMENT_A1.md`.

## D. Corrections to the 2026-09-25 chat review
- `secrets/` holds only `gpu.local.env.example` with no values. Not a leak.
- `PREREGISTRATION.md` and `BOOK.md` exist at root.
- Site nav links are real hrefs; only `#bank-src` starts as `#` and is filled by JS. F7 is on the site.
