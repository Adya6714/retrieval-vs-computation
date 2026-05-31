# Paper audit bundle

Generated artifacts for paper tables and checklist §0.1 (coverage & metric audit).

## Canonical derivation path

Run once after any raw CSV change:

```bash
python scripts/runs/rederive_all_metrics.py
```

This is the **single source of truth** for derived metrics. It calls:

| Step | Script / function | Outputs |
|------|---------------------|---------|
| 0 | `scripts/runs/coverage_audit.py` | Master coverage, gaps, denominator flags, cells_needing_runs, GSM P2 sensitivity |
| 1–5 | `rederive_all_metrics.py` | `coverage_pivot.csv`, P1/P2 metrics, Spearman cross-probe |

Do **not** use `scripts/compute_p1_metrics_unified.py` for paper tables — it covers 3 models only.

## §0.1 audit outputs

| File | Purpose |
|------|---------|
| `master_coverage_table.csv` | model × family × probe × canonical n × total n × coverage label |
| `COVERAGE_AUDIT_SUMMARY.md` | Executive summary (incomplete slices, key findings) |
| `../derived/master_coverage_gaps.csv` | Long-form missing IDs / sessions |
| `../derived/cells_needing_runs.csv` | Incomplete cells prioritized for API runs |
| `../derived/table_denominator_flags.csv` | Paper tables needing partial/imputed labels |
| `gsm_cci_wilcoxon_sensitivity.csv` | Claude vs GPT-4o: zero-imputed vs complete-case |
| `../INVESTIGATION/gsm_p2_gap.json` | Machine-readable GSM P2 gap + paired-test stats |

Coverage labels: `full_bank` · `partial_canonical` · `partial` · `contaminated_extra_ids` · `missing`

## Legacy audit scripts

The following were referenced in an earlier revision but are **not** in this repo:

- `scripts/audit/regenerate_paper_tables.py`
- `scripts/audit/five_model_var.py`
- etc.

Existing CSVs below were produced manually or from removed scripts; regenerate via `rederive_all_metrics.py` where noted above.

## Other key outputs

- `table1_gsm.csv`, `table1_algo_adversarial.csv`
- `appendix_c_full_var_5model.csv`, `appendix_c_full_var_5model.tex`
- `contamination_vri_algo_adversarial.csv`
- `o4mini_gsm_canonical_w3_comparison.csv`
- `../robustness_scatter_data.csv`
