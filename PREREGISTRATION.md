# PREREGISTRATION.md — RvC Phase-0 / paper-facing analysis plan

**Document date:** 2026-09-03  
**Authority:** Phase-0 gate G0 (vault `THE_PLAN` / `RVC_MASTER_DOCUMENT`).  
**Companion specs:** `results/derived/C1_INTRUSION_PREREG.md`, `results/derived/C2_IRT_CODING.md`, `colab/README.md`.

---

## 0. How to read this document (non-negotiable)

| Tag | Meaning |
|-----|---------|
| **PROSPECTIVE** | Spec frozen here *before* first execution of that analysis. May be reported as pre-registered. |
| **BRIEF-LOCKED** | Spec fixed in a task brief / dedicated prereg file immediately before the script ran, but **not** prospectively registered in a public registry via *this* file. Report as “pre-specified before run,” not as OSF-style pre-registration of this document. |
| **EXPLORATORY** | Spec or thresholds chosen after seeing patterns, or analysis was fishing / sensitivity / appendix. Must be labeled exploratory in the paper. |
| **OPERATIONAL** | Corrections, audits, denominators — not hypothesis tests. |

**Do not retrofit.** An EXPLORATORY or post-hoc finding must not be rewritten as PROSPECTIVE. Null and kill outcomes are first-class; they are listed in §4 for the paper.

**Default inference contract** (unless a row overrides):

- Significance: two-sided; **95% CI excludes 0** (or excludes the null value stated), equivalently α = 0.05.
- Clustering: prefer **clone/family cluster** for ALGO; **problem_id** otherwise; state the clustering variable on every estimate.
- Seeds: **42** unless noted; bootstrap **B ≥ 5000** for primary associations; permutation **N = 10000** for Kendall W.

---

## 1. Status snapshot

| ID | Analysis | Registration tag | Execution status (2026-09-03) |
|----|----------|------------------|------------------------------|
| C1 | DS-02 intrusion errors | **BRIEF-LOCKED** (`C1_INTRUSION_PREREG.md` before run) | Run |
| C2 | DS-01 mixture IRT | **BRIEF-LOCKED** (coding note before fit) | Run |
| C3 | Corrections changelog | **OPERATIONAL** | Done |
| C4 | TEP/CCI discriminant validity | **BRIEF-LOCKED** | Run |
| C5 | Oracle bias item-level | **BRIEF-LOCKED** | Run |
| C6 | O10 Bernoulli floor + EFA null | **BRIEF-LOCKED** | Run |
| C7 | Kendall W degeneracy filter | **BRIEF-LOCKED** | Run |
| C8 | DS-14 crossover battery | **BRIEF-LOCKED** | Run |
| C9 | This document | — | — |
| Colab O5–O8, O14b, O15–O16, DS-16, Llama greedy, Mech-freq | GPU / open-model battery | **PROSPECTIVE** (below) | Partially pending / incomplete downloads |

Analyses already in the draft paper from earlier sweeps (O10 G-theory, N3/N4, P1/P2 discriminant cells, triangulation, etc.) are inventoried in §3 with honest tags — many are **EXPLORATORY**.

---

## 2. Phase-0 battery (C1, C2, C4–C8)

Each entry: hypothesis → primary outcome → estimator → inference → significance → kill.

### C1 — DS-02 intrusion-error analysis

- **Tag:** BRIEF-LOCKED (`results/derived/C1_INTRUSION_PREREG.md`).
- **Hypothesis:** Among surface-incorrect answers on gold-changing W3/W6, a non-zero fraction are **canonical intrusions**; intrusion rate correlates **positively** with Infini-gram contamination within (family, model).
- **Primary outcomes:** (i) `intrusion_rate` vs explicit chance baseline; (ii) Spearman ρ(intrusion, contamination) among eligible errors.
- **Estimator:** Rule-based error classes (INTRUSION / PARTIAL / OTHER); chance = leave-one-out match rate to this item’s canonical gold among other errors in the same cell.
- **Inference:** Cluster bootstrap on ρ; clustering = **clone_family** (ALGO) / **problem_id** (else); B=5000, seed=42.
- **Significance:** CI for (rate − chance) excludes 0 **or** ρ CI excludes 0 with correct sign.
- **Kill:** Raw response text missing → stop. Do not pool families. Do not promote PARTIAL to primary. Null (rate ≤ chance and/or ρ CI∋0) is publishable — drop *positive retrieval-fingerprint claim*, keep the null table.

### C2 — DS-01 mixture IRT (strategy classes)

- **Tag:** BRIEF-LOCKED (`results/derived/C2_IRT_CODING.md`).
- **Hypothesis:** Variant-response patterns are better described by **K≥2** strategy classes than by a single Rasch dimension (Mislevy–Verhelst).
- **Primary outcome:** Preferred K by **minimum BIC** among K∈{1,2,3,4}; class profiles secondary.
- **Estimator:** Mixture Rasch (class-specific difficulties); 1PL/2PL baselines; person = (model, problem_id), items = variant indicators.
- **Inference:** BIC primary; bootstrap LRT secondary (B=40). No cluster bootstrap on BIC.
- **Significance:** Prefer K>1 only if BIC selects it (LRT disagreement → still BIC).
- **Kill:** If O10-style item variance / identifiability check shows near-zero item dispersion for a family, **do not** interpret that family’s mixture as a problem-bank difficulty ranking; report identifiability failure and drop strategy-class claims for that family.

### C4 — TEP / CCI discriminant validity

- **Tag:** BRIEF-LOCKED.
- **Hypothesis:** TEP (and CCI) are **capability-independent** robustness measures: |Spearman(metric, canonical accuracy)| is small (CI includes 0) at model-mean and item-within-model levels. Contrast: P1 retention ρ≈0.136 n.s., phi ρ≈−0.426 n.s.
- **Primary outcome:** Spearman ρ(TEP or CCI, canonical accuracy); scopes ALGO, GSM, pooled.
- **Estimator:** Model means from P2 TEP/CCI tables; item-level within model; BW P2 excluded (execution floor).
- **Inference:** Cluster bootstrap; clustering = **models** (model-mean), **clone_family** (ALGO items), **problem_id** (GSM items), **family×model cells** (pooled). B=5000, seed=42. Family-cluster with K=2 is **undefined** — do not use.
- **Significance:** Discriminant *failure* if CI excludes 0 and |ρ| large; discriminant *OK* if CI∋0.
- **Kill:** Degenerate binary outcome (`min(n_pos,n_neg)<5`) → undefined, withdraw. If TEP/CCI track capability at the pre-registered primary pooled cell, **drop** “TEP/CCI = pure robustness” as a main claim; keep as measurement caveat.

### C5 — Oracle / rescoring bias (item-level)

- **Tag:** BRIEF-LOCKED.
- **Hypothesis:** Perturbed (rescored) accuracy gains exceed matched canonical gains (positive excess δ_pert − δ_can); O13’s n=3 sign test is underpowered.
- **Primary outcome:** Mean paired excess over matched (model × problem) pairs.
- **Estimator:** Paired Δ on scored instances; report defect-equal-weight O13 reference alongside.
- **Inference:** Cluster bootstrap H0: mean excess = 0; **primary clustering = problem_id**; also report defect clustering (n=3, low power). B=5000, seed=42.
- **Significance:** 95% CI for excess excludes 0.
- **Kill:** If both primary CIs include 0, drop “systematic oracle bias” as a headline; keep O13 underpowered note. Do not switch clustering post-hoc to chase significance.

### C6 — O10 noise floor + EFA null calibration

- **Tag:** BRIEF-LOCKED.
- **Hypothesis (a):** Residual VC on binary LPM is largely Bernoulli sampling noise; G coefficients are underestimates until the simulated floor is subtracted.  
  **Hypothesis (b):** “W1–W6 one factor” may be an artifact of tiny item variance (null matrices also yield one factor).
- **Primary outcomes:** (a) Raw vs Bernoulli-floor-corrected VCs and `G_item_vs_surface_noise` with CIs; (b) fraction of null matrices with n_factors=1 under identical parallel analysis.
- **Estimator:** Fitted cell p = LPM mains+two-ways; null Y~Bern(p); EFA null = additive item+variant + iid residual, no common factor. N_null=1000.
- **Inference:** Item bootstrap CIs for corrected quantities; descriptive null fractions for EFA.
- **Significance:** Paper cites **corrected** G; EFA uninformative if null_frac_one_factor ≥ 0.90.
- **Kill:** If BW corrected G rises to a *substantial* item signal (materially above ~0.01 with CI excluding ~0), rewrite the BW “no item signal” headline before drafting. If EFA null also returns one factor, **drop unidimensionality-as-evidence**; report uninformative.

### C7 — Kendall W degeneracy filter (N4 sensitivity)

- **Tag:** BRIEF-LOCKED (criterion fixed before filtering).
- **Hypothesis:** ALGO’s marginal W (p≈0.049) may be inflated/deflated by a degenerate ranker; after excluding degenerates, concordance may change.
- **Primary outcome:** Full-sample vs filtered Kendall W and permutation p (O2 within-row null).
- **Estimator / criterion (preregistered):** Ranker degenerate if range(W1–W6 acc) < 0.10 **or** mean acc > 0.95 **or** mean acc < 0.10.
- **Inference:** Within-row rank permutation; N=10000; seed=42 (full); filtered uses seed+10000 when the ranker set changes.
- **Significance:** α=0.05 on permutation p; always report **both** full and filtered.
- **Kill:** If filtered ALGO is clearly n.s. while full was marginal, do not claim ALGO concordance without the filter table. Do not add post-hoc criteria aimed at dropping o4-mini unless it meets the frozen rule.

### C8 — DS-14 double-dissociation crossovers

- **Tag:** BRIEF-LOCKED.
- **Hypothesis:** Reliable item×model crossovers (M1 solves A fails B; M2 solves B fails A) exceed the additive single-dimension + Bernoulli null — evidence of separable capabilities.
- **Primary outcome:** Count of **reliable** rate-based crossovers per family vs null expectation.
- **Estimator:** Rate = mean over ≥4 variants; τ_hi=2/3, τ_lo=1/3; both legs beyond exact Bernoulli pooled-p q95; family-native IDs only. Null: additive LPM item+model, Bin(n,P)/n, N=1000.
- **Inference:** Compare observed count to null 95% interval / p_ge; binary cell counts are diagnostic.
- **Significance:** Above-chance if observed > null 97.5% quantile (equivalently small p_ge).
- **Kill:** If all families are at/below chance, **drop** DS-14 as positive architecture evidence; report chance-rate crossovers as consistent with one latent dimension (aligns with C6 EFA null / O10). Do not elevate GSM post-hoc without the null table.

### C3 — OPERATIONAL (not a hypothesis test)

Corrections to known bad figures (N3 Llama degeneracy, BW P2 floor prose, Qwen rank, tower parser, coverage notes). No kill criterion beyond “fix the false number.”

---

## 3. Colab items — PROSPECTIVE registrations

Freeze before (re)running or before treating downloads as confirmatory. Outputs land per `colab/README.md`. Default GPU: Colab T4 unless noted.

### COLAB-LG — Llama greedy behavioural P1

- **Hypothesis:** Greedy (T=0) Llama-3.1-8B P1 accuracies are stable enough to serve as the open-model behavioural anchor; GSM canonical @768 is a separate file (never overwrite the main greedy CSV).
- **Primary outcome:** Per-(family, variant, problem) correctness CSV + manifest.
- **Estimator:** Greedy generation + existing verifiers.
- **Inference:** Descriptive coverage audit vs API Llama rows; no α test required for the dump itself.
- **Kill:** If parse/verifier failure rate > 20% on a family, do not use that family as mechanistic–behavioural link input until fixed.

### COLAB-MF — Mechanistic frequency-controlled sweep

- **Hypothesis:** Gold-token rank / surprisal shifts under frequency-controlled renames track behavioural fragility beyond chance.
- **Primary outcome:** Per-instance layer profiles (`mechanistic_frequency_controlled*.csv`); ALGO+BW in the dedicated algo_bw file only.
- **Estimator:** Teacher-forced / rank-at-layer metrics as in notebook.
- **Inference:** Downstream N3/O8-style Spearman with cluster bootstrap (clone_family / problem_id).
- **Kill:** Constant or near-constant behavioural outcome (`min(n_pos,n_neg)<5`) → correlation **undefined** (do not report ρ). GSM-only file must not overwrite ALGO/BW.

### COLAB-O5 — Teacher-forced likelihood grid

- **Hypothesis:** Canonical→variant Δ mean logprob is a continuous, non-degenerate behavioural–mechanistic bridge outcome.
- **Primary outcome:** `O5_teacher_forced_likelihood.csv` (full P1 grid × open models); no premature aggregates.
- **Estimator:** Teacher-forced NLL / mean logprob on gold tokens.
- **Inference:** Used as input to O8/O10-style analyses; primary associations use cluster bootstrap.
- **Kill:** If Δ logprob is constant within a model×family, drop that cell from correlational claims.

### COLAB-O6 — Quantization sensitivity

- **Hypothesis:** fp16 vs int8 vs nf4 does not change qualitative mechanistic conclusions (pairwise bounds small).
- **Primary outcome:** Pairwise metric bounds in `O6_quantization_sensitivity.csv`.
- **Estimator:** Same forward metrics under three precisions.
- **Inference:** Descriptive bounds; flag if |Δ| exceeds pre-set tolerance in notebook summary.
- **Kill:** If nf4 (or int8) flips a primary sign or exceeds the notebook’s bound on >10% of items, **do not** treat T4-quantized runs as interchangeable with fp16 for that metric.

### COLAB-O7 — GSM mechanistic degeneracy gate

- **Hypothesis:** GSM open-model mechanistic outcomes are non-degenerate enough to enter O8.
- **Primary outcome:** Pass/fail verdict in `O7_gsm_degeneracy_verdict.txt`.
- **Estimator:** Variance / positive-rate checks on W3 (or notebook metric).
- **Inference:** Binary gate, not a significance test.
- **Kill:** FAIL → do not run or do not interpret O8 on GSM; report measurement failure.

### COLAB-O8 — Mechanistic↔behavioural link (replaces failed N3)

- **Hypothesis:** Per-layer canonical→W3 gold-token **rank shift** correlates with continuous O5 Δ logprob (not binary W3 correctness).
- **Primary outcome:** Spearman ρ(rank-shift, Δ logprob) per model/family; layer profile secondary.
- **Estimator:** As in `o8_mech_behavior_link.ipynb`; binary correctness reported only to show degeneracy.
- **Inference:** Cluster bootstrap; clustering = **clone_family** (ALGO) / **problem_id**; B=5000, seed=42.
- **Significance:** CI excludes 0.
- **Kill:** Degenerate continuous outcome or failed O7 gate → withdraw. Never revive N3 binary ρ on Llama 1/60 or Qwen 0/61.

### COLAB-O15 — Surprisal contamination

- **Hypothesis:** Problem-statement surprisal / min-k% tracks Infini-gram contamination after length residualization.
- **Primary outcome:** Spearman ρ(surprisal residual, contamination); `O15_surprisal_vs_infinigram.csv`.
- **Estimator:** Notebook NLL + `scripts/consolidate/o15_surprisal_vs_infinigram.py`.
- **Inference:** Cluster bootstrap by problem_id / clone_family as applicable.
- **Significance:** CI excludes 0 in the predicted direction.
- **Kill:** If ρ CI∋0 after length residual, drop “surprisal = contamination proxy” as confirmatory; keep as exploratory appendix.

### COLAB-O16 — Open-model proxy calibration

- **Hypothesis:** Open-corpus membership (Dolma/Pile) validates Infini-gram / O15/O5 proxies (AUC > 0.5); contaminated items show **lower** W3 retention.
- **Primary outcome:** ROC AUC vs corpus GT; retention delta member−clean (`O16_proxy_calibration.csv`, `O16_groundtruth_retention_test.csv`).
- **Estimator:** Part A corpus GT scripts (CPU); Part B open-model scores from Colab (still **scores_missing** until run).
- **Inference:** Cluster bootstrap for AUC≠0.5 and retention associations.
- **Significance:** AUC CI excludes 0.5; retention effect CI excludes 0 in the predicted direction.
- **Kill:** Insufficient GT positives → `insufficient_data`, no claim. If retention effect is null or wrong-signed after clustering, **do not** claim “contamination ⇒ fragility” from this calibration arm. Closed models have no GT — never pretend otherwise.

### COLAB-O14b — Naming likelihood (teacher-forced)

- **Hypothesis:** Named-entity / naming interventions shift gold `mean_logprob` in a measurable, non-degenerate way on the O14 bank (Qwen 1.5B/3B).
- **Primary outcome:** `O14b_naming_likelihood.csv` + `O14b_naming_analysis.csv`.
- **Estimator:** Same O5 teacher-forced path on `O14_naming_bank.jsonl`.
- **Kill:** Empty bank or constant Δ → no confirmatory claim.

### COLAB-DS16 — Recognition vs recall (Memory-Signature Suite)

- **Hypothesis:** Canonical→W3 **drop** in `recognition_recall_gap` (= recognition accuracy − recall accuracy) correlates positively with Infini-gram `contamination_score` and with C1 item intrusion rate (W3), yielding convergent retrieval signatures alongside C1.
- **Primary outcome:** `DS16_recognition_recall.csv` (per-option scores logged); `DS16_gap_correlations.csv`.
- **Estimator:** Recall = O5 TF `mean_logprob` + greedy verify; recognition = argmax TF `mean_logprob` over gold + **k=4** scripted distractors (`scripts/consolidate/ds16_distractors.py`). Models: Qwen2.5-1.5B/3B fp16.
- **Inference:** Spearman + cluster bootstrap (ALGO `clone_family`, else singleton); B=5000, seed=42.
- **Significance:** CI excludes 0 in the predicted (positive) direction for drop vs contamination and/or C1 intrusion.
- **Kill:** Null-compatible CIs on both correlates → recognition–recall does **not** add convergent validity with C1/Infini-gram on these open models; still report cell gaps descriptively. Distractors must remain deterministic (no hand-written options).

---

## 4. Hypotheses already tested and NOT supported

Report these explicitly (nulls are results). Do not quietly omit.

| Claim | Result | Where |
|-------|--------|--------|
| **item×model variance > item variance** (fragility not an item property) | **Not supported in any family** on primary O10 designs (`item_model_gt_item` false for ALGO/BW/GSM primary rows) | `O10_hypothesis_tests.csv` |
| **variant×model > variant** (fragility not a perturbation property) | **BW only** (primary full_7); not ALGO, not GSM | `O10_hypothesis_tests.csv` |
| **“Scalar robustness not generalizable”** (both interaction ratios hold) | **False everywhere** (`both_hold_scalar_robustness_not_generalizable` false all primary rows) | `O10_hypothesis_tests.csv` |
| **CCI predicts W3 retention** | **Not supported once clustered** at family scope (ALGO/GSM point-biserial CIs include 0; model-mean GSM n.s.). Pooled cells that exclude 0 are secondary and must not override the clustered family tests | `P2_P1_convergence.csv` |
| **TEP predicts W3 retention** | **Not supported** as a clean positive (item/model associations null or capability-entangled; BW TEP unusable on execution floor) | `C4_tep_discriminant_validity.csv`, `O13_measurement_failures.csv` |
| **TEP correlates with CCI** | **Not supported** (Spearman/Pearson CIs include 0 for GSM, ALGO, pooled) | `TEP_CCI_correlation.csv` |

Related measurement withdrawals (also report):

- **N3 Llama mechanistic–behavioural ρ** — withdrawn (degenerate 1/60 W3-correct).
- **O10 EFA “one factor ⇒ unidimensional W1–W6”** — uninformative under C6 null calibration.
- **BW Probe-2 CCI/TEP fingerprinting** — blocked (goal_reached=0).

---

## 5. Retrospective inventory — prior analyses (pre-registered vs exploratory)

Honesty pass for the paper methods / supplements. Tags below are **retrospective**; they do not upgrade EXPLORATORY work.

### Closer to pre-specified / audit-locked

| Analysis | Tag | Notes |
|----------|-----|-------|
| O2 / N4 Kendall W with **within-row** permutation null | BRIEF-LOCKED after null bugfix | Report corrected p; C7 filter is additional sensitivity |
| O10 ANOVA G-theory pipeline + stated interaction hypotheses | Pre-specified in O10 script/hypothesis table | Estimator (Gaussian LPM on binary) known-weak; C6 is the required noise correction |
| O13 measurement-failure ledger | OPERATIONAL / audit | Not a confirmatory test |
| C1 intrusion (dedicated prereg file) | BRIEF-LOCKED | Strongest Phase-0 prereg artifact |
| Cluster-bootstrap inference contract in `ALL_INFERENCE_AUDIT.csv` | Process standard | Clustering variable must be named every time |

### Exploratory (must be labeled as such)

| Analysis | Why exploratory |
|----------|-----------------|
| Triangulation / convergence labels & 270-config sweeps | Thresholds and “confident label” rates are sensitivity analyses; frozen rules were never locked in `LABEL_THRESHOLDS_PREREG.md` (HP-04 still open) |
| Cross-probe agreement counts (triple retrieval/computation) | Flag thresholds post-hoc / draft-driven |
| CC > SP > WIS exposure narrative | Confounded with difficulty until HP-05 matched WIS bank |
| Proximity → VRI / contamination regressions (per-model mixed) | Multiple looks; pooled often null |
| Algorithm-invocation paradox | Small n, observational |
| CC inversion / SP-adv headline on tiny subtype n | Appendix-scoped; not a pre-registered crossover battery (see C8 / solidify T3) |
| Qwen 0.5B–7B mechanistic pilots | Single-arch / pilot |
| Rebuild/solidify T3 subtype crossover (Claude×GPT-4o SP×CC) | Formal criterion written in-analysis; treat as **suggestive / exploratory** unless re-run under C8-style null control |
| Model-mean CCI–retention ρ with n=2 ALGO models | Degenerate sample; not confirmatory |
| Any claim using unrecovered prose numbers corrected in C3 | Invalid |

### HP-04 still outstanding (PROSPECTIVE when executed)

Vault `HP-04_Threshold_Prereg_and_MTMM`: freeze `docs/LABEL_THRESHOLDS_PREREG.md`, MTMM matrix with bootstrap CIs, pooled mixed-effects + 2PL with exposure covariate. Until that file exists, **convergence-label headline rates remain exploratory.**

---

## 6. Paper reporting checklist

1. Every confirmatory estimate names: hypothesis ID, estimator, clustering variable, B/seed, CI, and registration tag.  
2. §4 nulls appear in the main text or a dedicated “unsupported hypotheses” subsection — not only the supplement.  
3. C6 corrected G and EFA uninformative verdict gate any O10 factor/G prose.  
4. C7 full vs filtered W both appear if ALGO concordance is cited.  
5. C8: report per-family observed vs null; do not claim DS-14 globally if only GSM is above chance.  
6. Colab downloads are not confirmatory until the matching PROSPECTIVE row’s kill checks pass.  
7. Never describe EXPLORATORY triangulation thresholds as pre-registered.

---

## 7. Document history

| Date | Change |
|------|--------|
| 2026-09-03 | Added COLAB-DS16 (recognition/recall) and COLAB-O14b to §3 PROSPECTIVE Colab registrations. |
| 2026-09-03 | Initial consolidated PREREGISTRATION.md (C9). Records C1–C8 as BRIEF-LOCKED; Colab as PROSPECTIVE; §4 unsupported list; retrospective inventory. |
