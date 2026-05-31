# Scientific file-wise audit

This report audits each file with concrete deductions, then summarizes cross-probe relations.

## Global inventory

- Raw files audited: 42
- Derived files audited: 58
- Total deductions generated: 114

## File profile table (compact)

file,rows,cols,n_models,n_problem_ids,n_subtypes,primary_correct_col,primary_correct_rate,parse_failed_rate
ALGO_P1_4model_frozen_labels.csv,150,7,4,0,3,,,
ALGO_P1_VWC_exploratory.csv,18,5,3,0,3,,,
ALGO_P1_behavioral_claude.csv,693,13,2,110,0,verified,0.3246753246753247,0.01875901875901876
ALGO_P1_behavioral_gemini.csv,690,13,1,110,0,verified,0.3144927536231884,0.04927536231884058
ALGO_P1_behavioral_gpt4o.csv,710,13,1,110,0,verified,0.2802816901408451,0.1
ALGO_P1_behavioral_llama.csv,712,13,2,110,0,verified,0.054775280898876406,0.06320224719101124
ALGO_P1_behavioral_o1mini.csv,690,13,1,110,0,verified,0.6260869565217392,0.1391304347826087
ALGO_P1_lenient_4model.csv,124,7,4,0,3,,,
ALGO_P1_metrics.csv,66,7,3,0,3,,,
ALGO_P1_review_queue.csv,319,10,8,107,0,,,
ALGO_P2_metrics.csv,24,6,4,0,3,,,
ALGO_P2_per_instance_cci.csv,244,6,4,61,0,,,
ALGO_P2_phase1_claude_new.csv,110,12,1,110,3,,,
ALGO_P2_phase1_gemini.csv,110,12,1,110,3,,,
ALGO_P2_phase1_gpt4o.csv,20,12,1,20,3,,,
ALGO_P2_phase1_gpt4o_new.csv,110,12,1,110,3,,,
ALGO_P2_phase1_llama.csv,20,12,1,20,3,,,
ALGO_P2_phase1_llama_new.csv,110,12,1,110,3,,,
ALGO_P2_phase2_injected.csv,905,17,5,61,3,post_injection_correct,0.3901639344262295,0.0022099447513812156
ALGO_P2_phase2_injected_gemini.csv,181,17,1,61,3,post_injection_correct,0.3114754098360656,0.0
ALGO_P2_phase2_injected_implausible.csv,905,17,5,61,3,post_injection_correct,0.4098360655737705,0.0
ALGO_P2_phase2_normal.csv,2250,14,5,110,3,final_answer_correct,0.09555555555555556,0.0017777777777777779
ALGO_P2_phase2_normal_elicited.csv,1443,14,5,110,3,final_answer_correct,0.1101871101871102,0.001386001386001386
ALGO_P2_phase2_normal_gemini.csv,181,14,1,61,3,final_answer_correct,0.11049723756906077,0.0055248618784530384
ALGO_P3_contamination.csv,116,11,0,110,3,,,
ALGO_P3_mechanistic.csv,20,7,1,20,0,,,
ALGO_P3_triangulation.csv,330,39,3,110,3,,,
ALGO_P3_triangulation_v2.csv,330,39,3,110,3,,,
ALGO_P3_triangulation_v3.csv,440,39,4,110,3,,,
BW_P1_behavioral.csv,2760,10,6,124,0,behavioral_correct,0.3352685050798258,
BW_P1_behavioral_gemini.csv,455,9,1,65,0,behavioral_correct,0.23956043956043957,
BW_P1_behavioral_o1mini.csv,455,9,1,65,0,behavioral_correct,0.6285714285714286,
BW_P1_metrics.csv,88,10,4,0,0,,,
BW_P2_cci.csv,150,27,3,50,0,,,
BW_P2_cci_nl.csv,150,22,3,50,0,,,
BW_P2_plans.csv,150,8,3,50,0,,,
BW_P2_tep.csv,536,18,3,50,0,,,
BW_P3_contamination.csv,65,11,0,65,2,,,
BW_P3_mechanistic.csv,35,7,1,35,0,,,
BW_P3_triangulation_claude.csv,124,18,0,124,0,,,
BW_P3_triangulation_gpt4o.csv,124,18,0,124,0,,,
BW_P3_triangulation_llama.csv,124,18,0,124,0,,,
GSM_P1_behavioral_claude.csv,288,9,1,44,0,behavioral_correct,0.7743055555555556,
GSM_P1_behavioral_gemini.csv,288,9,1,44,0,behavioral_correct,0.6875,
GSM_P1_behavioral_gpt4o.csv,280,9,1,40,0,behavioral_correct,0.5714285714285714,
GSM_P1_behavioral_llama.csv,280,9,1,40,0,behavioral_correct,0.4357142857142857,
GSM_P1_behavioral_o1mini.csv,288,9,1,44,0,behavioral_correct,0.8229166666666666,
GSM_P1_metrics.csv,45,8,3,0,0,,,
GSM_P2_cci.csv,176,14,4,44,2,session_b_correct,0.7272727272727273,
GSM_P2_metrics.csv,20,5,4,0,0,,,
GSM_P2_phase1_claude.csv,44,18,1,44,0,session_b_correct,0.8636363636363636,
GSM_P2_phase1_gemini.csv,44,18,1,44,0,session_b_correct,0.8863636363636364,
GSM_P2_phase1_gpt4o.csv,44,18,1,44,0,session_b_correct,0.7045454545454546,
GSM_P2_phase1_llama.csv,44,18,1,44,0,session_b_correct,0.45454545454545453,
GSM_P2_phase1_o1mini.csv,44,18,1,44,0,session_b_correct,0.9545454545454546,
GSM_P3_contamination.csv,44,7,0,44,0,,,
GSM_P3_mechanistic.csv,20,7,1,20,0,,,
GSM_P3_triangulation_per_instance_claude.csv,44,16,0,44,0,,,
GSM_P3_triangulation_per_instance_gpt4o.csv,64,16,0,64,0,,,
P1_metrics_by_model_family.csv,9,15,3,0,0,,,
P1_metrics_by_model_subtype.csv,20,16,3,0,7,,,
P1_per_problem_var.csv,657,12,3,219,6,,,
P1_per_problem_var_5model.csv,940,12,5,219,6,,,
algorithm_invocation_clean.csv,4,4,4,0,0,,,
bw_violation_profile.csv,3,9,3,0,0,,,
coverage_matrix.csv,40,4,5,0,0,,,
coverage_pivot.csv,8,6,0,0,0,,,
cross_family_universally_fragile.csv,4,4,0,4,0,,,
cross_probe_acc_vs_w3retention.csv,3,5,0,0,0,,,
cross_probe_agreement_instances.csv,15,11,5,0,0,,,
cross_probe_corr_within_model.csv,30,7,5,0,0,,,
cross_probe_instance_frame.csv,1249,44,5,254,0,,,
cross_probe_spearman_by_model.csv,75,9,5,0,0,,,
cross_probe_spearman_model_level.csv,12,6,0,0,0,,,
cross_probe_triple_flagged_instances.csv,1249,10,0,254,0,,,
deep_p1_pairwise.csv,803,11,7,0,0,,,
deep_p1_transitions.csv,101,10,7,0,0,,,
deep_p2a_decision_schema_audit.csv,440,10,4,110,3,,,
deep_p2a_phase_link.csv,12,11,4,0,3,,,
deep_p2b_reactivity_delta.csv,15,5,5,0,3,,,
deep_p2b_response_profile.csv,46,7,5,0,3,,,
deep_probe3_mech_links.csv,3,7,1,0,0,,,
deep_triangulation_plus.csv,1128,12,7,219,0,,,
gemini_ALGO_perproblem.csv,110,12,1,110,3,,,
gemini_BW_perproblem.csv,65,12,1,65,2,,,
gemini_GSM_perproblem.csv,44,12,1,44,1,,,
implausibility_detection.csv,5,6,5,0,0,,,
master_coverage_gaps.csv,483,5,4,0,0,,,
master_coverage_table.csv,40,31,5,0,0,,,
master_per_problem_5model.csv,1249,16,5,254,0,,,
o4mini_ALGO_perproblem.csv,20,12,1,20,3,,,
o4mini_GSM_perproblem.csv,44,12,1,44,1,,,
o4mini_algo_partial.csv,5,5,1,0,0,,,
o4mini_bw_partial.csv,1,6,1,0,0,,,
o4mini_gsm_partial.csv,7,5,1,0,0,,,
probe1_per_model_variant.csv,105,6,5,0,0,,,
probe1_w3_retention.csv,15,5,5,0,0,,,
probe2_algo_metrics.csv,20,6,5,0,0,,,
probe2_gsm_metrics.csv,5,6,5,0,0,,,
table_denominator_flags.csv,11,6,0,0,0,,,

## File-by-file deductions

### `ALGO_P1_behavioral_claude.csv`
- Shape: rows=693, cols=13, models=2, problem_ids=110.
- Worst canonical->variant fragility: `anthropic/claude-sonnet-4` on `W5` with drop-rate 1.000 (n=50).
- Highest rescue effect: `anthropic/claude-sonnet-4` on `W4` with rescue-rate 0.543.
- Mean canonical minus W3 gap across models: 0.273 (max=0.273, min=0.273).

### `ALGO_P1_behavioral_gemini.csv`
- Shape: rows=690, cols=13, models=1, problem_ids=110.
- Worst canonical->variant fragility: `google/gemini-2.5-flash` on `W5` with drop-rate 0.971 (n=50).
- Highest rescue effect: `google/gemini-2.5-flash` on `W4` with rescue-rate 0.271.
- Mean canonical minus W3 gap across models: 0.209 (max=0.209, min=0.209).

### `ALGO_P1_behavioral_gpt4o.csv`
- Shape: rows=710, cols=13, models=1, problem_ids=110.
- Worst canonical->variant fragility: `openai/gpt-4o` on `W5` with drop-rate 1.000 (n=50).
- Highest rescue effect: `openai/gpt-4o` on `W4` with rescue-rate 0.381.
- Mean canonical minus W3 gap across models: 0.255 (max=0.255, min=0.255).

### `ALGO_P1_behavioral_llama.csv`
- Shape: rows=712, cols=13, models=2, problem_ids=110.
- Worst canonical->variant fragility: `meta-llama/llama-3.1-8b-instruct` on `W3` with drop-rate 1.000 (n=110).
- Highest rescue effect: `meta-llama/llama-3.1-8b-instruct` on `W1` with rescue-rate 0.077.
- Mean canonical minus W3 gap across models: 0.036 (max=0.036, min=0.036).

### `ALGO_P1_behavioral_o1mini.csv`
- Shape: rows=690, cols=13, models=1, problem_ids=110.
- Worst canonical->variant fragility: `openai/o4-mini` on `W5` with drop-rate 0.960 (n=50).
- Mean canonical minus W3 gap across models: 0.391 (max=0.391, min=0.391).

### `ALGO_P1_review_queue.csv`
- Shape: rows=319, cols=10, models=8, problem_ids=107.
- No specialized deduction emitted (schema-only or sparse file).

### `ALGO_P2_phase1_claude_new.csv`
- Shape: rows=110, cols=12, models=1, problem_ids=110.
- `phase1_parseable` rate: 0.764 over 110 rows.
- `greedy_assessment_correct` rate: 0.769 over 91 rows.
- `critical_point_identified` rate: 0.327 over 110 rows.
- `predicted_first_decision` non-empty rate: 0.964.

### `ALGO_P2_phase1_gemini.csv`
- Shape: rows=110, cols=12, models=1, problem_ids=110.
- `phase1_parseable` rate: 0.655 over 110 rows.
- `greedy_assessment_correct` rate: 0.740 over 77 rows.
- `critical_point_identified` rate: 0.291 over 110 rows.
- `predicted_first_decision` non-empty rate: 0.991.

### `ALGO_P2_phase1_gpt4o.csv`
- Shape: rows=20, cols=12, models=1, problem_ids=20.
- `phase1_parseable` rate: 0.900 over 20 rows.
- `greedy_assessment_correct` rate: 0.737 over 19 rows.
- `predicted_first_decision` non-empty rate: 0.950.

### `ALGO_P2_phase1_gpt4o_new.csv`
- Shape: rows=110, cols=12, models=1, problem_ids=110.
- `phase1_parseable` rate: 0.855 over 110 rows.
- `greedy_assessment_correct` rate: 0.788 over 99 rows.
- `critical_point_identified` rate: 0.155 over 110 rows.
- `predicted_first_decision` non-empty rate: 0.973.

### `ALGO_P2_phase1_llama.csv`
- Shape: rows=20, cols=12, models=1, problem_ids=20.
- `phase1_parseable` rate: 0.900 over 20 rows.
- `greedy_assessment_correct` rate: 0.737 over 19 rows.
- `predicted_first_decision` non-empty rate: 1.000.

### `ALGO_P2_phase1_llama_new.csv`
- Shape: rows=110, cols=12, models=1, problem_ids=110.
- `phase1_parseable` rate: 0.691 over 110 rows.
- `greedy_assessment_correct` rate: 0.722 over 90 rows.
- `critical_point_identified` rate: 0.218 over 110 rows.
- `predicted_first_decision` non-empty rate: 0.945.

### `ALGO_P2_phase2_injected.csv`
- Shape: rows=905, cols=17, models=5, problem_ids=61.
- Top `response_type` distribution: compliant:0.61, format_ignored:0.37, partial_compliance:0.02, refusal:0.00.
- `post_injection_correct` rate: 0.390 (final-step rows only).
- `injection_applied=True` prevalence: 0.337.

### `ALGO_P2_phase2_injected_gemini.csv`
- Shape: rows=181, cols=17, models=1, problem_ids=61.
- Top `response_type` distribution: format_ignored:0.99, compliant:0.01.
- `post_injection_correct` rate: 0.311 (final-step rows only).
- `injection_applied=True` prevalence: 0.337.

### `ALGO_P2_phase2_injected_implausible.csv`
- Shape: rows=905, cols=17, models=5, problem_ids=61.
- Top `response_type` distribution: compliant:0.63, format_ignored:0.36, partial_compliance:0.01.
- `post_injection_correct` rate: 0.410 (final-step rows only).
- `injection_applied=True` prevalence: 0.337.

### `ALGO_P2_phase2_normal.csv`
- Shape: rows=2250, cols=14, models=5, problem_ids=110.
- Top `response_type` distribution: compliant:0.57, format_ignored:0.39, partial_compliance:0.03, refusal:0.00.
- `final_answer_correct` rate: 0.096.

### `ALGO_P2_phase2_normal_elicited.csv`
- Shape: rows=1443, cols=14, models=5, problem_ids=110.
- Top `response_type` distribution: compliant:0.75, format_ignored:0.24, partial_compliance:0.01, refusal:0.00.
- `final_answer_correct` rate: 0.110.

### `ALGO_P2_phase2_normal_gemini.csv`
- Shape: rows=181, cols=14, models=1, problem_ids=61.
- Top `response_type` distribution: format_ignored:0.99, refusal:0.01.
- `final_answer_correct` rate: 0.110.

### `ALGO_P3_contamination.csv`
- Shape: rows=116, cols=11, models=0, problem_ids=110.
- `instance_contamination_score` quantiles q10/q50/q90: 0.000/0.000/0.564.
- `contamination_score` quantiles q10/q50/q90: 0.000/0.054/0.109.
- `template_contamination_score` quantiles q10/q50/q90: 0.000/0.062/1.000.

### `ALGO_P3_mechanistic.csv`
- Shape: rows=20, cols=7, models=1, problem_ids=20.
- Normalized crystallization depth mean/std: -0.042/0.000.
- `layer_cosine_similarities` non-empty rate: 1.000.

### `BW_P1_behavioral.csv`
- Shape: rows=2760, cols=10, models=6, problem_ids=124.
- Worst canonical->variant fragility: `deepseek/deepseek-r1-distill-llama-70b` on `W4` with drop-rate 0.914 (n=49).
- Highest rescue effect: `anthropic/claude-sonnet-4` on `W5` with rescue-rate 0.540.
- Mean canonical minus W3 gap across models: 0.162 (max=0.607, min=0.000).

### `BW_P1_behavioral_gemini.csv`
- Shape: rows=455, cols=9, models=1, problem_ids=65.
- Worst canonical->variant fragility: `google/gemini-2.5-flash` on `W4` with drop-rate 0.960 (n=65).
- Highest rescue effect: `google/gemini-2.5-flash` on `W5` with rescue-rate 0.450.
- Mean canonical minus W3 gap across models: 0.277 (max=0.277, min=0.277).

### `BW_P1_behavioral_o1mini.csv`
- Shape: rows=455, cols=9, models=1, problem_ids=65.
- Worst canonical->variant fragility: `openai/o4-mini` on `W3` with drop-rate 0.860 (n=65).
- Highest rescue effect: `openai/o4-mini` on `W3` with rescue-rate 0.333.
- Mean canonical minus W3 gap across models: 0.585 (max=0.585, min=0.585).

### `BW_P2_cci.csv`
- Shape: rows=150, cols=27, models=3, problem_ids=50.
- No specialized deduction emitted (schema-only or sparse file).

### `BW_P2_cci_nl.csv`
- Shape: rows=150, cols=22, models=3, problem_ids=50.
- No specialized deduction emitted (schema-only or sparse file).

### `BW_P2_plans.csv`
- Shape: rows=150, cols=8, models=3, problem_ids=50.
- No specialized deduction emitted (schema-only or sparse file).

### `BW_P2_tep.csv`
- Shape: rows=536, cols=18, models=3, problem_ids=50.
- No specialized deduction emitted (schema-only or sparse file).

### `BW_P3_contamination.csv`
- Shape: rows=65, cols=11, models=0, problem_ids=65.
- `instance_contamination_score` quantiles q10/q50/q90: 0.058/0.086/0.109.
- `contamination_score` quantiles q10/q50/q90: 0.036/0.042/0.053.
- `template_contamination_score` quantiles q10/q50/q90: 0.000/0.000/0.000.

### `BW_P3_mechanistic.csv`
- Shape: rows=35, cols=7, models=1, problem_ids=35.
- Normalized crystallization depth mean/std: -0.013/0.169.
- `layer_cosine_similarities` non-empty rate: 1.000.

### `GSM_P1_behavioral_claude.csv`
- Shape: rows=288, cols=9, models=1, problem_ids=44.
- Worst canonical->variant fragility: `anthropic/claude-sonnet-4` on `W4` with drop-rate 0.297 (n=44).
- Highest rescue effect: `anthropic/claude-sonnet-4` on `W5` with rescue-rate 0.857.
- Mean canonical minus W3 gap across models: 0.091 (max=0.091, min=0.091).

### `GSM_P1_behavioral_gemini.csv`
- Shape: rows=288, cols=9, models=1, problem_ids=44.
- Worst canonical->variant fragility: `google/gemini-2.5-flash` on `W4` with drop-rate 0.500 (n=44).
- Highest rescue effect: `google/gemini-2.5-flash` on `W5` with rescue-rate 0.750.
- Mean canonical minus W3 gap across models: 0.386 (max=0.386, min=0.386).

### `GSM_P1_behavioral_gpt4o.csv`
- Shape: rows=280, cols=9, models=1, problem_ids=40.
- Worst canonical->variant fragility: `openai/gpt-4o` on `W3` with drop-rate 0.667 (n=40).
- Highest rescue effect: `openai/gpt-4o` on `W6` with rescue-rate 0.571.
- Mean canonical minus W3 gap across models: 0.475 (max=0.475, min=0.475).

### `GSM_P1_behavioral_llama.csv`
- Shape: rows=280, cols=9, models=1, problem_ids=40.
- Worst canonical->variant fragility: `meta-llama/llama-3.1-8b-instruct` on `W5` with drop-rate 0.939 (n=40).
- Highest rescue effect: `meta-llama/llama-3.1-8b-instruct` on `W6` with rescue-rate 0.429.
- Mean canonical minus W3 gap across models: 0.625 (max=0.625, min=0.625).

### `GSM_P1_behavioral_o1mini.csv`
- Shape: rows=288, cols=9, models=1, problem_ids=44.
- Worst canonical->variant fragility: `openai/o4-mini` on `W4` with drop-rate 0.216 (n=44).
- Highest rescue effect: `openai/o4-mini` on `W5` with rescue-rate 0.714.
- Mean canonical minus W3 gap across models: 0.000 (max=0.000, min=0.000).

### `GSM_P2_cci.csv`
- Shape: rows=176, cols=14, models=4, problem_ids=44.
- No specialized deduction emitted (schema-only or sparse file).

### `GSM_P2_phase1_claude.csv`
- Shape: rows=44, cols=18, models=1, problem_ids=44.
- `phase1_parseable` rate: 0.909 over 44 rows.

### `GSM_P2_phase1_gemini.csv`
- Shape: rows=44, cols=18, models=1, problem_ids=44.
- `phase1_parseable` rate: 0.818 over 44 rows.

### `GSM_P2_phase1_gpt4o.csv`
- Shape: rows=44, cols=18, models=1, problem_ids=44.
- `phase1_parseable` rate: 0.455 over 44 rows.

### `GSM_P2_phase1_llama.csv`
- Shape: rows=44, cols=18, models=1, problem_ids=44.
- `phase1_parseable` rate: 0.455 over 44 rows.

### `GSM_P2_phase1_o1mini.csv`
- Shape: rows=44, cols=18, models=1, problem_ids=44.
- `phase1_parseable` rate: 0.977 over 44 rows.

### `GSM_P3_contamination.csv`
- Shape: rows=44, cols=7, models=0, problem_ids=44.
- `contamination_score` quantiles q10/q50/q90: 0.017/0.108/0.187.

### `GSM_P3_mechanistic.csv`
- Shape: rows=20, cols=7, models=1, problem_ids=20.
- Normalized crystallization depth mean/std: -0.042/0.000.
- `layer_cosine_similarities` non-empty rate: 1.000.

### `ALGO_P1_4model_frozen_labels.csv`
- Shape: rows=150, cols=7, models=4, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `ALGO_P1_VWC_exploratory.csv`
- Shape: rows=18, cols=5, models=3, problem_ids=0.
- Metrics present (2): VWC_CSS, VWC_VAR.
- `metric_value` mean/min/max: -0.314/-0.809/0.414.

### `ALGO_P1_lenient_4model.csv`
- Shape: rows=124, cols=7, models=4, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `ALGO_P1_metrics.csv`
- Shape: rows=66, cols=7, models=3, problem_ids=0.
- Metrics present (2): VAR, W6_Gap.
- `metric_value` mean/min/max: 0.203/-0.044/0.873.

### `ALGO_P2_metrics.csv`
- Shape: rows=24, cols=6, models=4, problem_ids=0.
- Metrics present (2): CCI, TEP.
- `metric_value` mean/min/max: 0.299/0.000/1.000.

### `ALGO_P2_per_instance_cci.csv`
- Shape: rows=244, cols=6, models=4, problem_ids=61.
- No specialized deduction emitted (schema-only or sparse file).

### `ALGO_P3_triangulation.csv`
- Shape: rows=330, cols=39, models=3, problem_ids=110.
- `convergence_label` distribution: ambiguous:0.62, mixed:0.35, retrieval_signal:0.02, computation_signal:0.01.

### `ALGO_P3_triangulation_v2.csv`
- Shape: rows=330, cols=39, models=3, problem_ids=110.
- `convergence_label` distribution: ambiguous:0.62, mixed:0.35, retrieval_signal:0.02, computation_signal:0.01.

### `ALGO_P3_triangulation_v3.csv`
- Shape: rows=440, cols=39, models=4, problem_ids=110.
- `convergence_label` distribution: ambiguous:0.62, mixed:0.36, retrieval_signal:0.02, computation_signal:0.01.

### `BW_P1_metrics.csv`
- Shape: rows=88, cols=10, models=4, problem_ids=0.
- Metrics present (2): PDAS, VAR.
- `metric_value` mean/min/max: 0.255/-0.588/1.000.

### `BW_P3_triangulation_claude.csv`
- Shape: rows=124, cols=18, models=0, problem_ids=124.
- No specialized deduction emitted (schema-only or sparse file).

### `BW_P3_triangulation_gpt4o.csv`
- Shape: rows=124, cols=18, models=0, problem_ids=124.
- No specialized deduction emitted (schema-only or sparse file).

### `BW_P3_triangulation_llama.csv`
- Shape: rows=124, cols=18, models=0, problem_ids=124.
- No specialized deduction emitted (schema-only or sparse file).

### `GSM_P1_metrics.csv`
- Shape: rows=45, cols=8, models=3, problem_ids=0.
- Metrics present (2): VAR, W6_Gap.
- `metric_value` mean/min/max: 0.549/0.000/0.944.

### `GSM_P2_metrics.csv`
- Shape: rows=20, cols=5, models=4, problem_ids=0.
- Metrics present (4): mean_cci_by_contamination_excluding_valid_divergence, mean_cci_excluding_valid_divergence, mean_tep, valid_divergence_rate.
- `value` mean/min/max: 0.246/0.000/0.773.

### `GSM_P3_triangulation_per_instance_claude.csv`
- Shape: rows=44, cols=16, models=0, problem_ids=44.
- No specialized deduction emitted (schema-only or sparse file).

### `GSM_P3_triangulation_per_instance_gpt4o.csv`
- Shape: rows=64, cols=16, models=0, problem_ids=64.
- No specialized deduction emitted (schema-only or sparse file).

### `P1_metrics_by_model_family.csv`
- Shape: rows=9, cols=15, models=3, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `P1_metrics_by_model_subtype.csv`
- Shape: rows=20, cols=16, models=3, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `P1_per_problem_var.csv`
- Shape: rows=657, cols=12, models=3, problem_ids=219.
- No specialized deduction emitted (schema-only or sparse file).

### `P1_per_problem_var_5model.csv`
- Shape: rows=940, cols=12, models=5, problem_ids=219.
- No specialized deduction emitted (schema-only or sparse file).

### `algorithm_invocation_clean.csv`
- Shape: rows=4, cols=4, models=4, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `bw_violation_profile.csv`
- Shape: rows=3, cols=9, models=3, problem_ids=0.
- `mean_cci` mean/min/max: 0.041/0.000/0.109.

### `coverage_matrix.csv`
- Shape: rows=40, cols=4, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `coverage_pivot.csv`
- Shape: rows=8, cols=6, models=0, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_family_universally_fragile.csv`
- Shape: rows=4, cols=4, models=0, problem_ids=4.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_probe_acc_vs_w3retention.csv`
- Shape: rows=3, cols=5, models=0, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_probe_agreement_instances.csv`
- Shape: rows=15, cols=11, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_probe_corr_within_model.csv`
- Shape: rows=30, cols=7, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_probe_instance_frame.csv`
- Shape: rows=1249, cols=44, models=5, problem_ids=254.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_probe_spearman_by_model.csv`
- Shape: rows=75, cols=9, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_probe_spearman_model_level.csv`
- Shape: rows=12, cols=6, models=0, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `cross_probe_triple_flagged_instances.csv`
- Shape: rows=1249, cols=10, models=0, problem_ids=254.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_p1_pairwise.csv`
- Shape: rows=803, cols=11, models=7, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_p1_transitions.csv`
- Shape: rows=101, cols=10, models=7, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_p2a_decision_schema_audit.csv`
- Shape: rows=440, cols=10, models=4, problem_ids=110.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_p2a_phase_link.csv`
- Shape: rows=12, cols=11, models=4, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_p2b_reactivity_delta.csv`
- Shape: rows=15, cols=5, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_p2b_response_profile.csv`
- Shape: rows=46, cols=7, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_probe3_mech_links.csv`
- Shape: rows=3, cols=7, models=1, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `deep_triangulation_plus.csv`
- Shape: rows=1128, cols=12, models=7, problem_ids=219.
- `tri_plus_label` distribution: insufficient:0.43, mixed:0.19, weak_computation:0.13, computation_signal_plus:0.13, weak_retrieval:0.09, retrieval_signal_plus:0.03.

### `gemini_ALGO_perproblem.csv`
- Shape: rows=110, cols=12, models=1, problem_ids=110.
- No specialized deduction emitted (schema-only or sparse file).

### `gemini_BW_perproblem.csv`
- Shape: rows=65, cols=12, models=1, problem_ids=65.
- No specialized deduction emitted (schema-only or sparse file).

### `gemini_GSM_perproblem.csv`
- Shape: rows=44, cols=12, models=1, problem_ids=44.
- No specialized deduction emitted (schema-only or sparse file).

### `implausibility_detection.csv`
- Shape: rows=5, cols=6, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `master_coverage_gaps.csv`
- Shape: rows=483, cols=5, models=4, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `master_coverage_table.csv`
- Shape: rows=40, cols=31, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `master_per_problem_5model.csv`
- Shape: rows=1249, cols=16, models=5, problem_ids=254.
- No specialized deduction emitted (schema-only or sparse file).

### `o4mini_ALGO_perproblem.csv`
- Shape: rows=20, cols=12, models=1, problem_ids=20.
- No specialized deduction emitted (schema-only or sparse file).

### `o4mini_GSM_perproblem.csv`
- Shape: rows=44, cols=12, models=1, problem_ids=44.
- No specialized deduction emitted (schema-only or sparse file).

### `o4mini_algo_partial.csv`
- Shape: rows=5, cols=5, models=1, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `o4mini_bw_partial.csv`
- Shape: rows=1, cols=6, models=1, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `o4mini_gsm_partial.csv`
- Shape: rows=7, cols=5, models=1, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `probe1_per_model_variant.csv`
- Shape: rows=105, cols=6, models=5, problem_ids=0.
- `accuracy` mean/min/max: 0.391/0.000/1.000.

### `probe1_w3_retention.csv`
- Shape: rows=15, cols=5, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `probe2_algo_metrics.csv`
- Shape: rows=20, cols=6, models=5, problem_ids=0.
- No specialized deduction emitted (schema-only or sparse file).

### `probe2_gsm_metrics.csv`
- Shape: rows=5, cols=6, models=5, problem_ids=0.
- `mean_cci` mean/min/max: 0.198/0.108/0.270.
- `mean_tep` mean/min/max: 0.640/0.539/0.773.

### `table_denominator_flags.csv`
- Shape: rows=11, cols=6, models=0, problem_ids=0.
- Metrics present (8): Algorithm elicitation sessions, BW P1 canonical accuracy, Claude vs GPT-4o paired CCI, GSM P2 mean CCI, GSM Probe 2 five-model comparison, P1, P1 canonical / W3 accuracy, P2A_elicited.

## Cross-probe relation summary

relation,family,model,n,value,interpretation
contamination_vs_tri_plus_score,ALGO,all,551,-0.38997237256550316,negative => higher contamination associates with retrieval-side votes
contamination_vs_tri_plus_score,BW,all,405,0.022919888524520118,negative => higher contamination associates with retrieval-side votes
contamination_vs_tri_plus_score,GSM,all,172,0.013759134353290669,negative => higher contamination associates with retrieval-side votes
p2a_first_decision_match_rate,ALGO,Claude,25,0.48,subtype=coin_change; final_correct_rate=0.680
p2a_first_decision_match_rate,ALGO,Claude,55,0.3454545454545454,subtype=shortest_path; final_correct_rate=0.436
p2a_first_decision_match_rate,ALGO,Claude,30,0.1333333333333333,subtype=wis; final_correct_rate=0.467
p2a_first_decision_match_rate,ALGO,Gemini,25,0.36,subtype=coin_change; final_correct_rate=0.480
p2a_first_decision_match_rate,ALGO,Gemini,55,0.1454545454545454,subtype=shortest_path; final_correct_rate=0.091
p2a_first_decision_match_rate,ALGO,Gemini,30,0.0333333333333333,subtype=wis; final_correct_rate=0.533
p2a_first_decision_match_rate,ALGO,Llama,25,0.32,subtype=coin_change; final_correct_rate=0.160
p2a_first_decision_match_rate,ALGO,Llama,55,0.2545454545454545,subtype=shortest_path; final_correct_rate=0.091
p2a_first_decision_match_rate,ALGO,Llama,30,0.2666666666666666,subtype=wis; final_correct_rate=0.500
p2a_first_decision_match_rate,ALGO,GPT-4o,25,0.6,subtype=coin_change; final_correct_rate=0.720
p2a_first_decision_match_rate,ALGO,GPT-4o,55,0.2,subtype=shortest_path; final_correct_rate=0.345
p2a_first_decision_match_rate,ALGO,GPT-4o,30,0.1666666666666666,subtype=wis; final_correct_rate=0.600
p2b_plausible_minus_implausible,ALGO,Claude,1,0.1,subtype=coin_change; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Claude,1,0.0,subtype=shortest_path; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Claude,1,-0.0588235294117646,subtype=wis; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,GPT-4o,1,-0.1,subtype=coin_change; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,GPT-4o,1,-0.0588235294117647,subtype=shortest_path; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,GPT-4o,1,0.0,subtype=wis; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Gemini,1,0.0,subtype=coin_change; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Gemini,1,0.0294117647058823,subtype=shortest_path; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Gemini,1,0.0588235294117647,subtype=wis; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Llama,1,0.3,subtype=coin_change; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Llama,1,0.0588235294117647,subtype=shortest_path; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,Llama,1,-0.4117647058823528,subtype=wis; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,o4-mini,1,-0.3,subtype=coin_change; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,o4-mini,1,0.0,subtype=shortest_path; positive => better on plausible than implausible.
p2b_plausible_minus_implausible,ALGO,o4-mini,1,0.0,subtype=wis; positive => better on plausible than implausible.
triangulation_original_non_ambiguous_rate,ALGO,all,330,0.3787878787878788,fraction with label != ambiguous in current pipeline
triangulation_plus_strong_signal_rate,ALGO,all,551,0.2268602540834846,fraction with strong +/-2 vote margin
triangulation_plus_strong_signal_rate,BW,all,405,0.03209876543209877,fraction with strong +/-2 vote margin
triangulation_plus_strong_signal_rate,GSM,all,172,0.20930232558139536,fraction with strong +/-2 vote margin

