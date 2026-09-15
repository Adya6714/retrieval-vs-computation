# C1 / DS-02 — Matching rules (methods note)

Frozen with the pre-registration. No undocumented fuzzy matching.

## Shared

- **Normalize text:** lowercase for BW verbs/objects; strip numbered list prefixes (`1.`, `2)`).
- **Variant gold differs:** structured extract of canonical gold ≠ structured extract of variant gold (family rules below). String equality alone is not used when a structured extract exists.
- **Surface-incorrect (error gate):** response structured extract ≠ variant gold structured extract, **without** W3 reverse-mapping. Differs from `rescored_correct`, which may accept isomorphic canonical forms.
- **INTRUSION** requires match to canonical structured gold (raw and/or after reverse-map) while remaining surface-incorrect vs variant gold.
- **PARTIAL overlap threshold:** strictly greater than **0.50** (i.e. `overlap > 0.5`). Documented; not tuned after seeing results.
- **Overlap definition:** Jaccard index on multisets of plan steps / solution tokens (see family).

## ALGO

### Shortest path (`SP_*`)

- Extract last `Path: …` token sequence (split on `→` / `->`; alphanumeric tokens; digits kept as digit strings; Hub labels uppercased).
- **W3 reverse map:** if bank notes / params yield `entity_mapping` (id→label), invert label→id and rewrite path tokens before compare.
- **INTRUSION:** mapped path tokens == canonical path tokens.
- **PARTIAL:** Jaccard(model path tokens, canonical path tokens) > 0.5, or canonical path token sequence is a contiguous subsequence of the model path.

### Coin change (`CC_*`)

- Extract `(Count|Total)` integer and sorted coin/scoop multiset from `[…]`.
- **INTRUSION:** coin multiset == canonical multiset (count may differ only if multiset matches; both compared).
- **PARTIAL:** Jaccard on coin multisets > 0.5.

### WIS (`WIS_*`)

- Extract `Selected: {…}` as a set of tokens (uppercased).
- **W3 reverse map:** invert `entity_mapping` / item rename (Item A↔0 style) when present; rewrite selected tokens.
- **INTRUSION:** selected set == canonical selected set.
- **PARTIAL:** Jaccard(selected sets) > 0.5.

## BW (blocksworld / mystery)

- Extract action lines matching canonical verbs **or** W3-renamed verbs.
- **W3 reverse map (required):** invert `action_mapping` (canonical→renamed) and `entity_mapping` (block id→name) from bank `notes`; rewrite each action’s verb and object args to canonical space; then compare to canonical gold plan.
- **INTRUSION:** full action tuple equality with canonical gold.
- **PARTIAL:** Jaccard on action-step multisets > 0.5, or canonical action sequence is a contiguous subsequence of the model plan.

## GSM

- **Final answer:** last `####` number if present, else last numeric token in the response (commas/`$` stripped). Tolerance **0.01**.
- **INTRUSION:** final number equals canonical gold (within tol) and not variant gold.
- **Intermediate chain:** all numbers extracted in order from the response. PARTIAL if canonical gold appears as any number in that chain (subsequence of the number stream) **or** Jaccard between response number multiset and `{canonical_gold}` is undefined for multi-step gold — bank stores only the final number, so intermediate PARTIAL reduces to: canonical gold appears among extracted numbers while final ≠ canonical (else INTRUSION). No fabricated solution traces.
- GSM W3: numeric gold equals canonical by construction → **excluded** by gold-diff filter. Primary GSM cell is **W6**.

## Chance baseline matcher

Uses the **INTRUSION** rule only (not PARTIAL) against the focal item’s canonical gold.

## Cluster bootstrap

`clone_family` from `results/derived/bank_clone_audit.csv` (`clone_family_id`, else `SINGLETON_{problem_id}`). Rate CIs: `cluster_bootstrap_ci` (B=10000, seed=42). Association CIs: `cluster_bootstrap_assoc` Spearman (B=5000, seed=42).
