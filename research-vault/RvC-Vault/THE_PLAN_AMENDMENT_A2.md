# THE PLAN, Amendment A2 (2026-10-05): Track B, bias as surface binding

status: proposed · amends: [[THE_PLAN]] v1.0 · recorded in: [[01_Program_State]] E · decision: [[02_Decision_Memo]] addendum 2026-10-05
THE_PLAN still wins every conflict. This amendment adds work; it removes no gate and moves no claim. Track B is proposed, not run; no W8 data exists at the time of this amendment.

## Why
The programme's position is that the object of study is which answer-irrelevant surface features a solution procedure is bound to. A protected attribute (gender, region, religion, caste-associated surname) in a verifiable-gold task is such a feature. Bias is therefore measurable with the existing instrument as one more surface. W3 is a cover-story isomorph, not a name swap, so the matched comparator for W8 is W3a; Track B therefore depends on HP-23.

## What is not new
Counterfactual name-swap bias testing is not new (BBQ, IndiBias, counterfactual augmentation). What is new: (1) differential reasoning quality by protagonist on tasks with verifiable gold, (2) testing whether bias and rename fragility share an internal mechanism, (3) a single repair targeting both.

## Scope rules for Track B
- Probe-level only (accuracy gap, retention, flip rate, CCI, intrusion) until Gate G1 passes.
- No per-instance retrieval/computation labels.
- Floor rule unchanged: Acc_can ≥ .30. Track B uses GSM + coin_change (+ shortest_path where the floor is cleared).
- Any W8 gap is reported only relative to the T1/H8 inference noise floor.
- Pre-register each direction's primary contrast in `PREREGISTRATION.md` (section "Track B") before its first run.
- Compute: GPU and API credits are expected; no Track B run is scheduled yet. Costs are estimated at preregistration, not here.

## Track B directions
D18 prerequisites: [[HP-23_W3a_Nonce_Bank]] (W3a bank) and T1 ([[D11_Inference_Stack_Noise_Floor]]) noise floor. Do not run W8 until both exist.
| ID | Direction | Hypothesis | Serves | Compute |
|---|---|---|---|---|
| D18 | [[D18_Protected_Attribute_Surface]] W8 protected-attribute surface | H16, H17 | CC-5 | open-weight GPU (A100-class when available) or API |
| D19 | [[D19_Bias_Geometry]] attribute direction vs W3a binding direction, same layer band | H18 | AC-5, DS-03 | open-weight GPU (A100-class when available; 1.5B-3B FP16 is T4-feasible) or API |
| D20 | [[D20_Debias_By_Repair]] steering / band-constrained ablation; RLVR with demographic-diverse surfaces | H19, H20 | AC-5, AC-2, AC-3 | steering: open-weight GPU (A100-class when available) or API; RLVR gated behind G2 like D15 |

Note on claim ID: DS-13 already registers AC-4 as the moonshot binding-bottleneck module. The Track B architecture claim is therefore **AC-5** (not AC-4).

## Dependency and sequencing
A1 Track T remains valid and is not superseded; if GPU arrives, Track T items can run on it unchanged.

THE_PLAN gates AC-2 repair behind G2 (H6 verdict). Track B steering is proposed under G2, not G1. If this amendment ever read otherwise, THE_PLAN wins.

| Direction | Prerequisite | Gate | Earliest start |
|---|---|---|---|
| D18 | T1 (noise floor); HP-23 (W3a bank); power analysis completed | None (probe-level claims only) | After T1 + HP-23 + Track B power analysis |
| D19 | D18 complete; T3/T7 layer band identified; open-weight GPU | Per-instance claims only after G1 | After D18 behavioural battery |
| D20 steering, joint arm | D19 verdict supported (shared substrate) | G2 (AC-2 / THE_PLAN) | After G2 and a supporting D19 verdict |
| D20 steering, attribute-only arm | D18 non-null (W8 gap above T1 floor) | G2 (AC-2 / THE_PLAN) | After G2 and a non-null D18 |
| D20 RLVR | Same envelope as D15 | G2 (like D15) | After G2 |

## New claims
- **CC-5** Protected-attribute invariance is a case of surface invariance: on verifiable-gold items, attribute-swap cost covaries per item with W3a (nonce rename) cost.
- **AC-5** Bias and W3a (nonce rename) fragility share a localisable representational substrate; repairing it reduces both.

## New hypotheses (registered here, status in [[01_Program_State]] C)
- H16 Primary contrast is W8 group A vs W8 group B (all names swapped, token-count and frequency matched) on a neutral-name baseline; canonical is a reference arm only. Secondary: per-item W8 cost correlates with per-item W3a (nonce rename) cost.
- H17 At matched canonical accuracy, plan-execution consistency (CCI) and intrusion rate differ by protagonist group (differential reasoning, not only differential answers).
- H18 The residual-stream direction separating W8 variants overlaps the canonical-vs-W3a direction above a permutation null (cosine and CKA) in the same layer band identified by T3/T7.
- H19 Projecting out or steering against the shared direction in that band reduces the W8 gap and W3a cost, with canonical accuracy loss ≤ 2 points.
- H20 (extends H7b) RLVR with demographic-diverse surfaces reduces the W8 gap on held-out attribute groups more than canonical-only RLVR at matched steps and equal canonical accuracy.

## Kill criteria (pre-registered)
- D18: if W8 gap is within the T1 noise floor on all models, report the null; do not search for subgroups post hoc; register [[D18b_Social_Role_Framing]] as the proposed fallback (not built, not scheduled).
- D19: if overlap is not above the permutation null, record "bias and rename fragility are separate mechanisms" as the finding and drop the joint-repair claim; attribute-only repair continues as a separate arm.
- D20 (both steering arms): if canonical accuracy drops > 2 points or the gap closes only on attribute groups seen during steering-vector extraction, report as failed installation.

## Threats and confounds
| Threat | Planned control |
|---|---|
| Name token length and frequency confounds | Match token count and corpus frequency bands; report tokenizer. |
| Gender or region cues correlating with name rarity | Frequency-matched pools. |
| Instruction-tuned models refusing or hedging on some names | Log refusals separately; excluded with reason, never scored, per the repo's verification rule. |
| Position or format effects from longer names | Length-matched nonce W3a comparator. |
| Multiple comparisons across groups and models | Pre-registered primary contrast; correction method named at preregistration. |
| Item-count power | Power analysis before the first run; expand item set if minimum detectable gap exceeds 5 accuracy points. |
| Steering side effects | Canonical accuracy loss cap; held-out attribute groups. |
| Construct validity | Gold is independent of the attribute only on these task types; no claim extends to open-ended generation. |
| Effect near zero on arithmetic | D18 kill criterion; registered fallback [[D18b_Social_Role_Framing]] (proposed, conditional on null D18; not built, not scheduled). |
| Names as proxies | Names signal perceived group, not actual identity; no claim about any individual. |

## Method borrowing (for the Strategy catalogue)
Apply the site's four-question generative principle to two methods:

1. **Correspondence audit** (matched CVs differing only in name).
   - Hidden property: differential evaluation by inferred group membership.
   - Observable signature: outcome gap under matched credentials, name only varied.
   - LLM analogue: W8 name swap on verifiable-gold items; gold independent of the name.
   - Per-instance signal: accuracy gap, retention, flip rate, and (with Probe 2) CCI by protagonist group.

2. **Matched-guise technique** (same speaker, different accent; links to W7).
   - Hidden property: evaluative response to a surface cue that is content-irrelevant.
   - Observable signature: rating or decision shift under guise change with content held fixed.
   - LLM analogue: answer-preserving surface change that carries a social cue (W7 language; W8 attribute-bearing name) while gold is fixed.
   - Per-instance signal: same metrics as W8; W7 is the linguistic cousin already on Track T.

## Ethics note
Names are proxies for perceived group membership, sampled from documented public name lists per group; no claim about any real person; caste-associated surnames used only as an audit dimension, following IndiBias practice; attributes are chosen only where the gold answer is provably independent of them.

## Paper mapping
- Paper II gains D19 as an AC-5 arm.
- New short paper candidate "Same Problem, Different Person" (D18 + D19) for a fairness or evaluation workshop.
- D20 RLVR arm joins Paper III.

## Outside the paper (site `#who`)
Track B extends the same deployment frame as Paper I: same Acc_can, different protagonist surface, different user-visible outcome. The site fairness tab (proposed until W8 runs) lists the shipping decision (trusting flat benchmarks for audit-sensitive tasks), the instrument catch (W8 group A vs B on a neutral-name baseline, covariation with W3a), and the anchor finding (cover-story vs numeric dissociation from Finding 1; CC-5 proposed). See [[BI-05_Fairness_And_Bias]] and [[BI-03_Practitioner_Guidance]].
