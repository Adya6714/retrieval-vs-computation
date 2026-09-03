# W1–W6 Variant Specification

Source: research vault `EF-01_Probe1_Surface_Invariance.md`, `docs/workbench/EVALUATION_WALKTHROUGH.md`, and `scripts/generation/stage2_generate_variants.py`.

Shared design (all families): six answer-preserving or answer-changing surface transforms of each canonical item. Variants must pass the family verifier before any model call. Zero-shot CoT, T=0 at evaluation.

| Code | Cross-family intent | Gold answer | Problem text |
|------|---------------------|-------------|--------------|
| canonical | Base item | baseline | baseline |
| W1 | Lexical paraphrase | **unchanged** | **changes** (numbers/lists/block names preserved) |
| W2 | Structural reformat | **unchanged** | **changes** (layout/format only) |
| W3 | Entity rename → nonce / alternate domain labels (diagnostic) | **unchanged numerically / isomorphic labels** | **changes** |
| W4 | Formal notation | **unchanged** | **changes** |
| W5 | Direction / role reversal (RCS; excluded from CSS) | **changes** | **changes** |
| W6 | Procedural regeneration (new numbers, same algorithm/template) | **changes** | **changes** |

---

## ALGO (coin_change / shortest_path / WIS)

| Variant | Transformation | Gold answer changes? | Problem text changes? |
|---------|----------------|----------------------|------------------------|
| W1 | LLM paraphrase; **lists and numbers must be preserved verbatim** | No | Yes (wording only) |
| W2 | Deterministic or prompted reformat (tables / structured layout by subtype) | No | Yes (layout) |
| W3 | Domain rename via mapping JSON: CC → alternate units (“scoops”); SP → letter/nonce node labels; WIS → renamed interval labels. Round-trip verified. | No for numeric content; label strings in gold may be remapped for SP/WIS | Yes |
| W4 | Formal / mathematical notation rewrite | No | Yes |
| W5 | **SP only** in bank (50 rows): reverse s–t / invert path query; CC and WIS have **no W5** | Yes (new optimal for reversed query) | Yes |
| W6 | Procedural regen: CC new denominations/target; SP/WIS new graph/intervals from seed. Bank has 90 W6 (not all 110) | Yes | Yes |

Notes: ALGO bank = 110 canonical; W5 only on shortest_path; W6 missing for some IDs (`missing_bank_row` exclusions).

---

## BW (blocksworld)

| Variant | Transformation | Gold answer changes? | Problem text changes? |
|---------|----------------|----------------------|------------------------|
| W1 | LLM paraphrase; **block letter names preserved** | No | Yes |
| W2 | Deterministic **Current/Goal table** markdown | No | Yes |
| W3 | Bijective rename of block names **and** action verbs (`pick-up`/`stack`/… → nonce); plan gold remapped | Labels in gold remapped; plan structure isomorphic | Yes |
| W4 | Formal / PDDL-flavored notation of state+goal | No (same plan under mapping) | Yes |
| W5 | Init↔goal swap (or procedural new instance when PDDL path missing); Fast Downward replan | **Yes** (new plan) | Yes |
| W6 | New random init/goal with same `n_blocks`; FD writes new plan | **Yes** | Yes |

Bank: 65 items × all 7 variants.

---

## GSM (arithmetic_reasoning)

| Variant | Transformation | Gold answer changes? | Problem text changes? |
|---------|----------------|----------------------|------------------------|
| W1 | Lexical paraphrase; numeric literals preserved | No | Yes |
| W2 | Prompted structural reformat (sections / bullet layout) | No | Yes |
| W3 | Entity/role rename to nonce or alternate narrative (e.g. hotel→hiker); answer number unchanged | No | Yes |
| W4 | Formal / piecewise-function notation | No | Yes |
| W5 | Invert: given answer ask for a different unknown (e.g. cost→minutes) | **Yes** | Yes |
| W6 | New instance from same `gsm-symbolic` template_id (instance≠0); new numbers | **Yes** | Yes |

Bank: 44 canonical; W6 only for 24 templates with a secondary instance.

---

## Scoring implications

- **CSS** uses {W1, W2, W3, W4, W6} (answer-preserving set).
- **RCS** uses W5 (answer changes).
- **Retention R_W3** = Acc_W3 / Acc_canonical; undefined when Acc_canonical < 0.30   (`MIN_CANONICAL_FOR_RETENTION`).
