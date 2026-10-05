# D18 Protected-attribute surface (W8)

status: proposed · Track B, Amendment A2 · execution: pending handoff · not run · hypotheses: H16, H17 · serves: CC-5
amendment: [[THE_PLAN_AMENDMENT_A2]] · prerequisites: [[HP-23_W3a_Nonce_Bank]], T1 ([[D11_Inference_Stack_Noise_Floor]]) · related: [[D15_RLVR_Surface_Diversity]], [[D17_Commitment_Depth_T4]], [[DS-03_Representational_Invariance_RSA]], T3 ([[D12_Precision_Compression_Invariance]]), [[D18b_Social_Role_Framing]]

**Claim tested.** On verifiable-gold items with a neutral-name baseline, swapping only protected-attribute-bearing names (W8) produces a group-vs-group accuracy gap beyond the T1 noise floor (H16). Secondary: per-item W8 cost correlates with per-item W3a (nonce rename) cost. At matched accuracy on the neutral-name baseline, CCI and intrusion rate differ by protagonist group (H17).

**Question.** Is a protected attribute in a gold-fixed problem one more answer-irrelevant surface the procedure can bind to?

**Hypothesis.** H16, H17 (see [[THE_PLAN_AMENDMENT_A2]]).

**Primary contrast (H16).** Acc(W8_group_A) - Acc(W8_group_B) with all protagonist names swapped between groups, token-count and frequency matched. Canonical text with the original memorised name is a **reference arm only**, not the primary contrast, because canonical-name binding can confound the gap.

**Secondary contrasts.** Per-item cost vs W3a on the same neutral-name baseline; CCI and intrusion by group (H17). All contrasts relative to the T1/H8 flip-rate floor.

**Family order.** GSM first. Then coin_change (and shortest_path where the Acc_can floor is cleared).

**Neutral-name baseline (GSM and ALGO).** Before any W8 variant, install a neutral-name canonical as the working baseline for that item. Adding or changing the protagonist relative to the shipped bank canonical is a baseline change and must be recorded. W8 swaps are name-only off that neutral-name baseline.

**GSM-Symbolic name slots in the bank (audit, 2026-10-05).** All 44 GSM canonicals are sourced from GSM-Symbolic (`template_id` in `source`), but the bank does **not** expose structured name-slot fields (`difficulty_params` empty; no name-slot column). Name variation is only whatever appears in rendered `problem_text`. Canonical items whose rendered text contains a given name (person protagonist): GSM_005, GSM_006, GSM_009, GSM_010, GSM_012, GSM_013, GSM_015, GSM_017, GSM_018, GSM_020, GSM_042, GSM_044, GSM_045, GSM_048, GSM_049, GSM_052, GSM_053, GSM_055, GSM_056, GSM_057, GSM_058, GSM_059. Remaining canonicals have no given-name protagonist in text (honorifics such as Mrs/Prof excluded). Do not invent name slots on nameless items.

**Protocol steps.**
1. Prerequisites: HP-23 W3a bank present; T1 noise floor measured for the models used.
2. GSM first: build neutral-name baselines; reuse items with Acc_can ≥ .30 on that baseline; restrict W8 to items with a rendered name slot (list above) unless an expanded GSM set is pre-registered.
3. Generate W8 by swapping only names drawn from documented public name lists per group (gender, region, religion, caste-associated surname as audit dimensions). Gold unchanged. No other lexical change.
4. Same instruction template as Probe 1. Matched name length in tokens and frequency bands across groups; report the tokenizer used (at minimum Llama-3 and Qwen2.5 tokenizers for open-weight arms).
5. At least 3 name samples per group per item (repeated measures within item); persist mappings in `notes` as JSON; verifier consumes them.
6. ALGO: same neutral-name baseline rule as GSM before W8.
7. Gold-in-gold-out: every W8 gold passes the family verifier before any model call.
8. Pre-register contrasts and the power analysis in `PREREGISTRATION.md` Track B before the first W8 call. No W8 data exists at time of writing.

**Compute.** Open-weight GPU (A100-class when available) or API. No run scheduled; costs estimated at preregistration.

**Kill criterion.** If the W8 group-vs-group gap is within the T1 noise floor on all models, report the null; do not search for subgroups post hoc. Proposed fallback: [[D18b_Social_Role_Framing]] (not built, not scheduled).

**Why this tier.** Behavioural prerequisite for D19 geometry and D20 repair; cheapest falsifier of CC-5.
