# Claim tagging draft (Step 12)

Maps intended **main-paper claims** to evidence artifacts and a pre-submission status.
Statuses: **supported** (audited data, scoped as in text) · **exploratory** (report with n/denominator caveats) · **blocked** (missing data or protocol failure prevents strong claim).

Regenerate inputs: Phase 2 summaries + `scientific_file_deductions.csv` + `PRE_API_MASTER_AUDIT.md` + `PRE_API_RECOVERY_AUDIT.md`.

---

## Summary counts

| Status | Main-text claims | Notes |
|--------|------------------|-------|
| **supported** | 13 | Pending Step 13 user decision for C12 |
| **exploratory** | 8 | Includes C12 triangulation + C14 exposure until Step 13–14 locked |
| **blocked** | 4 | BW process, ALGO P2A partial, WIS exposure (if Option B not run), GSM P1 partial |

---

## Master table

| ID | Claim (paper intent) | Primary evidence | Status | Blocker / scope |
|----|----------------------|------------------|--------|-----------------|
| C01 | Identical benchmark accuracy can hide different solving strategies | `deep_p1_transitions.csv`, `P1_metrics_by_model_family.csv`, abstract fig | **supported** | Qualitative framing; numbers per family |
| C02 | Three orthogonal probes (rename, plan/execution, exposure) apply per instance | Setup §3, `triangulation_v2_labels.csv`, teaser fig | **supported** | BW P2 uses tolerant rerun (see C18) |
| C03 | Rename fragility is **family- and model-dependent**, not rankable from accuracy alone | `deep_p1_transitions.csv`, `vri_by_model.csv`, `fig1_robustness_scatter` | **supported** | Five models on ALGO/GSM/BW bank |
| C04 | **SP inversion:** Claude 64.7→0% vs GPT-4o 41.2→26.5% under W3 (matched SP, Fisher p=0.0021) | `scientific_file_deductions.csv`, AUDIT Fisher tables, `deep_p1_transitions.csv` | **supported** | n=34 matched SP subset |
| C05 | **CC inversion** (same-direction robustness reversal) | `deep_p1_transitions.csv` (coin_change) | **exploratory** | n=10 — already appendix-scoped in draft |
| C06 | **WIS collapses to ~0%** under entity rename despite high canonical accuracy | `deep_p1_transitions.csv`, `vri_by_subtype.csv` | **supported** | ALGO WIS subtype; CC>SP>WIS exposure gradient confounded (C14) |
| C07 | **VRI** quantifies rename-specific drop beyond vocabulary-preserving variants | `vri_by_model.csv`, `vri_analysis_summary.md` | **supported** | GSM GPT-4o/Llama VRI on **20/44** IDs only |
| C08 | Plan fidelity (CCI) **does not predict** final correctness | `GSM_P2_cci.csv`, `tep_dissociation_summary.md`, `ALGO_P2_per_instance_cci.csv` | **supported** | GSM/ALGO; BW CCI sparse |
| C09 | Models **accept injected wrong states** at high rates yet recover correct finals ~half the time | `deep_p2b_reactivity_delta.csv`, `deep_p2b_response_profile.csv` | **supported** | ALGO P2B n=61 by design; cite denominator |
| C10 | **TEP dissociation:** high trajectory divergence + correct final answer | `tep_dissociation_summary.md`, `tep_dissociation_by_model.csv` | **supported** | GSM/ALGO; 127/546 TEP-valid sessions; BW excluded |
| C11 | Training **contamination/proximity** correlates with surface fragility (P3 × P1) | `vri_proximity_correlation.csv`, `ALGO_P3_contamination_regression_v3.txt` | **exploratory** | Pooled ALGO Spearman ≈ −0.04 n.s.; per-model mixed |
| C12 | **Cross-probe convergence** labels retrieval vs computation instances | `triangulation_framing_decision.md`, `triangulation_v2_labels.csv` | **exploratory** | **Step 13 `[D]`** — brief ready; you pick thresholds + scope |
| C13 | Triple-probe **agreement** (P1 fragile + P2 low CCI + P3 high contam) identifies retrieval-like instances | `cross_probe_correlation_summary.md`, `cross_probe_agreement_instances.csv` | **exploratory** | 0/1249 retrieval-agree at default flags; 1.4% computation-agree — threshold-sensitive |
| C14 | **Exposure gradient** CC > SP > WIS isolates training prevalence with algorithm held constant | `vri_by_subtype.csv`, `PRE_API_MASTER_AUDIT.md` §WIS | **exploratory** | WIS harder + lower exposure confound; Step 14 or Step 17 matched bank |
| C15 | **Accuracy vs W3 retention** correlates across models within probe | `cross_probe_acc_vs_w3retention.csv` | **exploratory** | GSM ρ=0.63 (p=0.25); ALGO ρ=1.0 (n=5 models — rank artifact) |
| C16 | **o4-mini** (reasoning-trained) shows distinct robustness / proximity profile | `vri_by_model.csv`, `GSM_P2_phase1_o1mini.csv`, `deep_p1_transitions.csv` | **supported** | GSM P2 44/44 recovered — no P2 API needed |
| C17 | **Algorithm-invocation paradox** (more algorithm mention → worse W3) | `algorithm_invocation_clean.csv`, appendix | **exploratory** | n=13 adversarial cases; observational only |
| C18 | **BW P2 strict PDDL aborts** 84–100% of sessions — measurement-protocol finding | `tep_dissociation_quality_audit.md`, `BW_P2_*`, checklist BW flags | **supported** | Scope BW **process** claims to tolerant rerun / pilot |
| C19 | BW rename can **improve** some models (negative VRI) | `vri_by_model.csv` (BW GPT-4o, Llama negative VRI) | **supported** | Low absolute accuracy on BW canonical |
| C20 | **Five-model coverage** on core GSM+ALGO P1/P3 | `COVERAGE_AUDIT_SUMMARY.md`, `master_coverage_table.csv` | **supported** | **34/40** slices complete; 6 cells partial |
| C21 | **GSM five-model P2** comparison (e.g. Claude vs GPT-4o CCI) | `load_gsm_p2_merged()`, `COVERAGE_AUDIT_SUMMARY.md` | **supported** | Merged loader includes o4-mini from phase1_o1mini file |
| C22 | **ALGO P2A elicited** five-model phase-1 elicitation | `deep_p2a_phase_link.csv`, raw `ALGO_P2_phase1_*` | **exploratory** | Claude/Gemini/Llama **61/110**; GPT-4o/o4-mini full |
| C23 | **Mechanistic rank dissociation** (Qwen pilot) | `PROBE3_mechanistic_*`, appendix | **exploratory** | Single open-weight backbone — appendix only |
| C24 | P2A **first-decision match** after normalization | `deep_p2a_decision_schema_audit.csv`, `deep_metrics_summary.md` | **supported** | Raw 0% → normalized ~27.5%; methods must cite normalization |
| C25 | **No blank-row corruption** in raw behavioral CSVs | `PRE_API_MASTER_AUDIT.md`, Step 6 manifest | **supported** | Missing ≠ corrupt |
| C26 | Per-instance labels **re-derivable** from audit bundle | `results/paper/AUDIT/README.md`, `rederive_all_metrics.py` | **supported** | Paper path only for tables |

---

## By paper section

### Abstract / §5.1 — Robustness inversion (C03–C07)

| Claim | Status | Action |
|-------|--------|--------|
| Same rename op improves / destroys / leaves untouched | **supported** | Keep; cite per-model bars |
| WIS collapse under rename | **supported** | Keep; note ALGO subtype |
| SP Claude vs GPT-4o inversion | **supported** | Keep Fisher + n=34 |
| CC reversal | **exploratory** | Appendix or “exploratory” tag |

### §5.2 — Plan / execution / injection (C08–C10)

| Claim | Status | Action |
|-------|--------|--------|
| CCI ⊥ final correctness | **supported** | GSM paired tests OK |
| Injection compliance vs recovery | **supported** | State ALGO n=61 |
| TEP dissociation | **supported** | Exclude BW from TEP stats |

### §5.3 — Contamination / exposure (C11, C14)

| Claim | Status | Action |
|-------|--------|--------|
| Proximity predicts fragility (class-level) | **exploratory** | Soften to “two of five models” or per-family |
| CC>SP>WIS exposure control | **exploratory** | Step 14: suggestive OR Step 17 API |

### §5.4 — BW protocol (C18–C19)

| Claim | Status | Action |
|-------|--------|--------|
| Strict PDDL abort rate | **supported** | Main-text measurement finding |
| BW process / TEP / CCI claims | **blocked** | Step 20 or scope to appendix pilot |

### §5.5 — Convergence / triangulation (C12)

| Claim | Status | Action |
|-------|--------|--------|
| Three-probe convergence labels | **exploratory** | **Step 13 `[D]`** — you decide |
| “Existence proof” instances | **exploratory** | Depends on Step 13 threshold + scope |
| Legacy 3% strong | **historical** | Appendix comparison |
| BW convergence | **blocked** | Appendix / process only |

### Reasoning model / o4-mini (C16)

| Claim | Status | Action |
|-------|--------|--------|
| o4-mini robustness patterns (ALGO, GSM P1) | **exploratory** | OK with P1 data |
| o4-mini plan-execution profile | **blocked** | **Step 16** GSM P2 616 calls |

### Coverage / completeness (C20–C22)

| Gap | API step | Est. calls |
|-----|----------|------------|
| ~~GSM P2 o4-mini 0/44~~ | ~~Step 16~~ | **0** (recovered) |
| GSM P1 GPT-4o/Llama 20/44 | Step 16 | ~336 |
| ALGO P2A elicited 61/110 ×3 | Step 18 | ~588 |
| BW P2 five-model full bank | Step 20 | ~2300 |

Source: `pre_api_api_budget.csv`, `PRE_API_RECOVERY_AUDIT.md`.

---

## Evidence file index (Phase 2)

| Step | Summary | Key CSVs |
|------|---------|----------|
| 6b | `PRE_API_MASTER_AUDIT.md` | `pre_api_slice_inventory.csv`, `pre_api_api_budget.csv` |
| 7 | `deep_metrics_summary.md` | `deep_p1_transitions.csv`, `deep_p2a_phase_link.csv` |
| 8 | `tep_dissociation_summary.md` | `tep_dissociation_by_model.csv` |
| 9 | `vri_analysis_summary.md` | `vri_by_model.csv`, `vri_proximity_correlation.csv` |
| 10 | `triangulation_exploratory_summary.md` | `triangulation_label_distribution.csv` |
| 11 | `cross_probe_correlation_summary.md` | `cross_probe_spearman_by_model.csv` |
| 5 | `scientific_filewise_audit.md` | `scientific_file_deductions.csv` (114 rows) |

---

## Recommended Phase 3 actions (from tags)

1. **Step 13 `[D]`:** Triangulation — `triangulation_framing_decision.md`
2. **Step 14 `[D]`:** WIS exposure — `wis_exposure_framing_brief.md`
3. **Step 15 `[D]`:** Submission gate after 13–14 locked

---

## Regenerate

```bash
python scripts/runs/scientific_filewise_audit.py
python scripts/runs/pre_api_master_audit.py
python scripts/runs/cross_probe_correlation_analysis.py
# Phase 2 summaries: tep_dissociation, vri, triangulation_exploratory (see CHECKLIST.md)
```

This draft is **manual synthesis** — edit claim IDs when `main.tex` changes; Step 15 checks every abstract bullet maps to a row here.
