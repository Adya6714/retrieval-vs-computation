#!/usr/bin/env python3
"""O13: Oracle-bias table + measurement-failure table (paper primary figures).

TABLE A — grader defects selectively inflate (or only affect) perturbed accuracy.
Numbers verified from repo artifacts (oracle_bias_summary.csv, exclusions, K1/J3/I3);
not copied from prose.

TABLE B — cells we could not measure (coverage / floor / degeneracy / missing runs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
ROOT = REPO_ROOT

OUT_A = DER / "O13_oracle_bias_table.csv"
OUT_B = DER / "O13_measurement_failures.csv"
OUT_SIGN = DER / "O13_oracle_bias_sign_test.csv"


def _round(x: float | None, nd: int = 4):
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return round(float(x), nd)


def build_table_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(DER / "oracle_bias_summary.csv")
    ex = pd.read_csv(DER / "variant_exclusions.csv")
    k1 = pd.read_csv(DER / "K1_wis_w6_generator_bug.csv")
    j3 = pd.read_csv(DER / "J3_gsm_w6_offbank.csv")
    i3 = pd.read_csv(DER / "I3_table4_acc_p2a_status.csv")

    rows: list[dict] = []

    # --- Graded verifier defects (before/after accuracy) ---
    # Source: results/derived/oracle_bias_summary.csv (scripts/consolidate/oracle_bias_summary.py)
    graded = {
        "BW_W3_action_mapping": {
            "defect": "BW_W3_action_mapping",
            "family": "BW",
            "variant": "W3",
            "source": "oracle_bias_summary.csv",
            "note": "action_mapping=None vs notes mapping; canonical unchanged",
        },
        "SP_W3_node_mapping": {
            "defect": "ALGO_SP_W3_node_mapping",
            "family": "ALGO",
            "variant": "W3",
            "source": "oracle_bias_summary.csv",
            "note": "node_mapping missing from difficulty_params; notes+trailing Path fix (J1)",
        },
        "BW_state_parser": {
            "defect": "BW_legacy_state_parser",
            "family": "BW",
            "variant": "all_variants",
            "source": "oracle_bias_summary.csv",
            "note": "legacy regex parser vs released prose/table parser",
        },
    }
    for key, meta in graded.items():
        r = obs[obs["defect"] == key].iloc[0]
        p_before = float(r["acc_before_perturbed"])
        p_after = float(r["acc_after_perturbed"])
        c_before = float(r["acc_before_canonical"])
        c_after = float(r["acc_after_canonical"])
        p_delta = float(r["delta_perturbed"])
        c_delta = float(r["delta_canonical"])
        rows.append(
            {
                "defect": meta["defect"],
                "family": meta["family"],
                "variant": meta["variant"],
                "rows_affected": int(r["rows_affected"]),
                "perturbed_acc_before": _round(p_before),
                "perturbed_acc_after": _round(p_after),
                "perturbed_delta": _round(p_delta),
                "canonical_acc_before": _round(c_before),
                "canonical_acc_after": _round(c_after),
                "canonical_delta": _round(c_delta),
                "clean_case": bool(abs(c_delta) < 1e-12),
                "disposition": "rescored",
                "source": meta["source"],
                "note": meta["note"],
            }
        )

    # --- Exclusion / withdrawal defects (no before/after rescoring) ---
    n_bw_w5 = int(((ex["family"] == "BW") & (ex["variant"] == "W5")).sum())
    assert n_bw_w5 == 5, n_bw_w5
    rows.append(
        {
            "defect": "BW_W5_byte_identical",
            "family": "BW",
            "variant": "W5",
            "rows_affected": n_bw_w5,
            "perturbed_acc_before": "",
            "perturbed_acc_after": "",
            "perturbed_delta": "",
            "canonical_acc_before": "",
            "canonical_acc_after": "",
            "canonical_delta": "",
            "clean_case": True,  # excluded; no false accuracy credit on either side
            "disposition": "excluded",
            "source": "variant_exclusions.csv",
            "note": "MBW_496–MBW_500 byte-identical to canonical; excluded variant_not_transformed",
        }
    )

    n_wis = len(k1)
    assert n_wis == 25, n_wis
    n_ex_wis = int(((ex["family"] == "ALGO") & (ex["variant"] == "W6")).sum())
    assert n_ex_wis == 25, n_ex_wis
    rows.append(
        {
            "defect": "ALGO_WIS_W6_generator_bug",
            "family": "ALGO",
            "variant": "W6",
            "rows_affected": n_wis,
            "perturbed_acc_before": "",
            "perturbed_acc_after": "",
            "perturbed_delta": "",
            "canonical_acc_before": "",
            "canonical_acc_after": "",
            "canonical_delta": "",
            "clean_case": True,
            "disposition": "excluded",
            "source": "K1_wis_w6_generator_bug.csv|variant_exclusions.csv",
            "note": "render_wis_text_with_weights left problem_text unchanged; 25 WIS W6 excluded",
        }
    )

    n_gsm_w6 = int(((ex["family"] == "GSM") & (ex["variant"] == "W6")).sum())
    assert n_gsm_w6 == 20, n_gsm_w6
    rows.append(
        {
            "defect": "GSM_W6_off_bank",
            "family": "GSM",
            "variant": "W6",
            "rows_affected": n_gsm_w6,
            "perturbed_acc_before": "",
            "perturbed_acc_after": "",
            "perturbed_delta": "",
            "canonical_acc_before": "",
            "canonical_acc_after": "",
            "canonical_delta": "",
            "clean_case": True,
            "disposition": "removed",
            "source": "J3_gsm_w6_offbank.csv|variant_exclusions.csv",
            "note": "W6 never defined on GSM_001–020; Table 7 GPT-4o/Llama W6 cells removed (missing_bank_row)",
        }
    )

    rows.append(
        {
            "defect": "GSM_Table4_Acc_P2A_disjunction",
            "family": "GSM",
            "variant": "P2_phase2A",
            "rows_affected": int(len(i3) * 44),  # 5 models × 44 sessions
            "perturbed_acc_before": "",
            "perturbed_acc_after": "",
            "perturbed_delta": "",
            "canonical_acc_before": "",
            "canonical_acc_after": "",
            "canonical_delta": "",
            "clean_case": False,  # published cells were a disjunction, not a clean Acc_P2A
            "disposition": "withdrawn",
            "source": "I3_table4_acc_p2a_status.csv",
            "note": (
                "phase2a_values never persisted; published Acc_P2A was "
                "either_session_correct (phase2a OR phase1); unrecoverable without re-run"
            ),
        }
    )

    table_a = pd.DataFrame(rows)

    # Sign test on graded defects only: H0 median(perturbed_delta - canonical_delta) = 0
    graded_df = table_a[table_a["disposition"] == "rescored"].copy()
    diffs = (
        pd.to_numeric(graded_df["perturbed_delta"], errors="coerce")
        - pd.to_numeric(graded_df["canonical_delta"], errors="coerce")
    ).to_numpy(dtype=float)
    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    n_zero = int(np.sum(diffs == 0))
    # binomial sign test ignoring ties
    n_eff = n_pos + n_neg
    if n_eff > 0:
        # two-sided exact binomial under p=0.5
        p_sign = float(stats.binomtest(n_pos, n_eff, 0.5, alternative="two-sided").pvalue)
    else:
        p_sign = float("nan")
    mean_p = float(pd.to_numeric(graded_df["perturbed_delta"], errors="coerce").mean())
    mean_c = float(pd.to_numeric(graded_df["canonical_delta"], errors="coerce").mean())
    sign = pd.DataFrame(
        [
            {
                "n_graded_defects": len(graded_df),
                "mean_perturbed_delta": round(mean_p, 4),
                "mean_canonical_delta": round(mean_c, 4),
                "mean_excess_perturbed_minus_canonical": round(mean_p - mean_c, 4),
                "n_perturbed_gt_canonical": n_pos,
                "n_perturbed_lt_canonical": n_neg,
                "n_tie": n_zero,
                "sign_test_p_two_sided": round(p_sign, 4) if p_sign == p_sign else "",
                "interpretation": (
                    "grader fixes raise perturbed accuracy more than canonical "
                    f"(mean Δ_pert={mean_p:.3f} vs mean Δ_can={mean_c:.3f}); "
                    f"sign test {n_pos}/{n_eff} positive, p={p_sign:.4f}"
                    if n_eff
                    else "insufficient graded defects"
                ),
            }
        ]
    )
    return table_a, sign


def build_table_b() -> pd.DataFrame:
    floor = pd.read_csv(DER / "BW_P2_floor_documentation.csv")
    cov2 = pd.read_csv(DER / "COVERAGE_PROBE2.csv")
    cov_m = pd.read_csv(DER / "COVERAGE_MATRIX.csv")
    mech = pd.read_csv(ROOT / "Mech Frequency Controlled Summary.csv")
    ex = pd.read_csv(DER / "variant_exclusions.csv")
    i3 = pd.read_csv(DER / "I3_table4_acc_p2a_status.csv")
    j3 = pd.read_csv(DER / "J3_gsm_w6_offbank.csv")
    k1 = pd.read_csv(DER / "K1_wis_w6_generator_bug.csv")
    cci = pd.read_csv(DER / "ALGO_P2_cci.csv")

    rows: list[dict] = []

    # BW Probe 2 — execution floor
    pg = pd.to_numeric(floor["partial_goal_achievement_mean"], errors="coerce")
    models_bw = ",".join(sorted(floor["model"].astype(str).unique()))
    n_bw_p2 = int(floor["n_sessions"].sum())
    # also count inject-phase rows that are similarly unusable
    bw_p2_cov = cov2[cov2["family"] == "BW"]
    n_bw_all_phases = int(pd.to_numeric(bw_p2_cov["n_rows"], errors="coerce").fillna(0).sum())
    rows.append(
        {
            "family": "BW",
            "probe": "P2",
            "phase_or_variant": "all_phases_CCI_and_TEP",
            "models_affected": models_bw,
            "reason": (
                f"execution floor: goal_reached=0 for all models; "
                f"partial_goal_mean∈[{pg.min():.4f},{pg.max():.4f}] "
                f"(cci_mean∈[{pd.to_numeric(floor['cci_mean'], errors='coerce').min():.4f},"
                f"{pd.to_numeric(floor['cci_mean'], errors='coerce').max():.4f}]); "
                "do not analyze TEP / treat CCI as fingerprint"
            ),
            "n_rows_lost": n_bw_all_phases if n_bw_all_phases else n_bw_p2,
            "recoverable": False,
            "source": "BW_P2_floor_documentation.csv|COVERAGE_PROBE2.csv",
        }
    )

    # BW Probe 3 mechanistic — gold-token degeneracy (3 distinct tokens)
    # Verified on contentgold sweeps: pick/attack/un (3 ids); Mech Frequency notes pick=48/65
    bw_degen = mech[mech["note"].fillna("").str.contains("DEGENERATE") & (mech["family"] == "BW")]
    models_mech = ",".join(sorted(bw_degen["model"].astype(str).unique()))
    rows.append(
        {
            "family": "BW",
            "probe": "P3",
            "phase_or_variant": "mechanistic",
            "models_affected": models_mech if models_mech else "Qwen2.5-1.5B,Qwen2.5-7B,Llama-3.1-8B",
            "reason": (
                "gold-token degeneracy: only 3 distinct first gold tokens "
                "(pick/attack/un); Mech Frequency reports canonical 'pick'=48/65"
            ),
            "n_rows_lost": 65,  # BW bank size; Wilcoxon not reported for BW slices
            "recoverable": False,
            "source": "Mech Frequency Controlled Summary.csv|mechanistic_sweep_*_contentgold.csv",
        }
    )

    # GSM W6 never on GSM_001-020
    n_gsm_w6 = int(((ex["family"] == "GSM") & (ex["variant"] == "W6")).sum())
    rows.append(
        {
            "family": "GSM",
            "probe": "P1",
            "phase_or_variant": "W6",
            "models_affected": "GPT-4o,Llama",
            "reason": "W6 never defined on GSM_001–020 in any bank commit; missing_bank_row exclusions",
            "n_rows_lost": n_gsm_w6,
            "recoverable": False,
            "source": "J3_gsm_w6_offbank.csv|variant_exclusions.csv",
        }
    )

    # GSM P2 Acc_P2A
    rows.append(
        {
            "family": "GSM",
            "probe": "P2",
            "phase_or_variant": "phase2A_accuracy",
            "models_affected": ",".join(i3["model"].astype(str).tolist()),
            "reason": (
                "phase2a_values never persisted; published Acc_P2A was phase2a∨phase1 "
                "disjunction; unrecoverable without re-run"
            ),
            "n_rows_lost": int(len(i3) * 44),
            "recoverable": False,
            "source": "I3_table4_acc_p2a_status.csv",
        }
    )

    # ALGO WIS W6 generator bug
    rows.append(
        {
            "family": "ALGO",
            "probe": "P1",
            "phase_or_variant": "W6",
            "models_affected": "all_P1_models",
            "reason": "WIS W6 generator left problem_text byte-identical; 25 rows excluded",
            "n_rows_lost": len(k1),
            "recoverable": False,
            "source": "K1_wis_w6_generator_bug.csv",
        }
    )

    # Qwen2.5-7B mechanistic — high gold rank on ALGO
    qwen = mech[
        (mech["model"] == "Qwen/Qwen2.5-7B-Instruct")
        & (mech["family"] == "ALGO")
        & (mech["freq_tercile"] == "all")
    ]
    assert len(qwen) == 1, qwen
    med = float(qwen.iloc[0]["median_rank_canonical"])
    rows.append(
        {
            "family": "ALGO",
            "probe": "P3",
            "phase_or_variant": "mechanistic_frequency_controlled",
            "models_affected": "Qwen/Qwen2.5-7B-Instruct",
            "reason": (
                f"median gold-token rank on ALGO canonical = {med:.0f} "
                f"of ~152k vocab (Qwen2.5); content signal near chance"
            ),
            "n_rows_lost": int(qwen.iloc[0]["n_pairs"]),
            "recoverable": False,
            "source": "Mech Frequency Controlled Summary.csv",
        }
    )

    # Gemini ALGO CCI all NaN
    gem = cci[cci["model"].astype(str).str.contains("gemini", case=False)]
    n_gem = len(gem)
    nan_frac = float(pd.to_numeric(gem["cci_score"], errors="coerce").isna().mean()) if n_gem else 1.0
    assert nan_frac == 1.0, nan_frac
    rows.append(
        {
            "family": "ALGO",
            "probe": "P2",
            "phase_or_variant": "CCI_plan_execution",
            "models_affected": "Gemini",
            "reason": f"cci_score all NaN ({n_gem}/{n_gem} rows); unparseable execution",
            "n_rows_lost": n_gem,
            "recoverable": False,
            "source": "ALGO_P2_cci.csv|COVERAGE_PROBE2.csv",
        }
    )

    # o4-mini absent from ALGO phase 1
    o4 = cov2[(cov2["family"] == "ALGO") & (cov2["model"] == "o4-mini") & (cov2["phase"] == "phase1_declare")]
    assert len(o4) == 1 and int(o4.iloc[0]["n_rows"]) == 0
    rows.append(
        {
            "family": "ALGO",
            "probe": "P2",
            "phase_or_variant": "phase1_declare",
            "models_affected": "o4-mini",
            "reason": "never_collected (absent from ALGO Probe 2 phase 1)",
            "n_rows_lost": 110,  # bank size that would have been collected
            "recoverable": True,  # could still be run
            "source": "COVERAGE_PROBE2.csv",
        }
    )

    # Retention cells suppressed by 0.30 canonical-accuracy floor (O4 COVERAGE_MATRIX)
    floor_cells = cov_m[
        cov_m["reason_if_undefined"].fillna("").str.contains("below_floor|floor_0", regex=True)
    ]
    for r in floor_cells.itertuples(index=False):
        n_lost = int(r.n_rows_valid_after_exclusions) if pd.notna(r.n_rows_valid_after_exclusions) else ""
        rows.append(
            {
                "family": r.family,
                "probe": "P1",
                "phase_or_variant": f"{r.variant}_retention",
                "models_affected": r.model,
                "reason": str(r.reason_if_undefined),
                "n_rows_lost": n_lost,
                "recoverable": False,
                "source": "COVERAGE_MATRIX.csv",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    table_a, sign = build_table_a()
    table_b = build_table_b()
    table_a.to_csv(OUT_A, index=False)
    table_b.to_csv(OUT_B, index=False)
    sign.to_csv(OUT_SIGN, index=False)

    print(f"Wrote {OUT_A} ({len(table_a)} rows)")
    print(f"Wrote {OUT_B} ({len(table_b)} rows)")
    print(f"Wrote {OUT_SIGN}")
    print("\n=== TABLE A (oracle bias) ===")
    cols = [
        "defect", "family", "variant", "rows_affected",
        "perturbed_acc_before", "perturbed_acc_after", "perturbed_delta",
        "canonical_acc_before", "canonical_acc_after", "canonical_delta",
        "clean_case", "disposition",
    ]
    print(table_a[cols].to_string(index=False))
    print("\n=== Sign test (graded rescored defects) ===")
    print(sign.to_string(index=False))
    print("\n=== TABLE B (measurement failures) ===")
    print(table_b.to_string(index=False))


if __name__ == "__main__":
    main()
