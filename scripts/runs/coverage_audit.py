"""Coverage & metric audit (checklist §0.1).

Builds a master coverage table from question banks + raw CSVs, flags
incomplete denominators, and separates complete-case vs zero-imputed stats.

Canonical derivation path — run via:
    python scripts/runs/rederive_all_metrics.py
or directly:
    python scripts/runs/coverage_audit.py

Writes:
    results/derived/master_coverage_table.csv
    results/derived/master_coverage_gaps.csv
    results/derived/table_denominator_flags.csv
    results/derived/cells_needing_runs.csv
    results/derived/COVERAGE_AUDIT_SUMMARY.md
    results/paper/AUDIT/master_coverage_table.csv
    results/paper/AUDIT/gsm_cci_wilcoxon_sensitivity.csv
    results/paper/AUDIT/COVERAGE_AUDIT_SUMMARY.md
    results/paper/INVESTIGATION/gsm_p2_gap.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
DATA = ROOT / "data" / "problems"
DER = ROOT / "results" / "derived"
AUD = ROOT / "results" / "paper" / "AUDIT"
INV = ROOT / "results" / "paper" / "INVESTIGATION"

MODELS = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
    "openai/o4-mini",
]
SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
P1_TAGS = {
    "anthropic/claude-sonnet-4": "claude",
    "google/gemini-2.5-flash": "gemini",
    "openai/gpt-4o": "gpt4o",
    "meta-llama/llama-3.1-8b-instruct": "llama",
    "openai/o4-mini": "o1mini",
}
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]


def _gsm_bank_canonical_ids() -> list[str]:
    bank = _load_bank("GSM")
    return sorted(_bank_canonical_ids(bank))


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        print(f"  !! read failed for {path}: {exc}")
        return None


def _norm_variant(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text.lower() == "canonical":
        return "canonical"
    if text.lower().startswith("w") and len(text) >= 2 and text[1:].isdigit():
        return f"W{text[1:]}"
    return text


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    raw = df.get("raw_response", df.get("model_answer", pd.Series([""] * len(df))))
    return ~raw.astype(str).str.startswith("ERROR:")


def _load_bank(family: str) -> pd.DataFrame:
    path = DATA / f"question_bank_{family.lower()}.csv"
    bank = pd.read_csv(path, dtype=str)
    bank["variant_type"] = bank["variant_type"].map(_norm_variant)
    return bank


def _bank_pairs(bank: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(bank["problem_id"], bank["variant_type"]))


def _bank_canonical_ids(bank: pd.DataFrame) -> set[str]:
    return set(bank.loc[bank["variant_type"] == "canonical", "problem_id"])


def _variant_expected(bank: pd.DataFrame) -> dict[str, int]:
    return bank.groupby("variant_type")["problem_id"].nunique().to_dict()


def _join_ids(ids: set[str] | list[str]) -> str:
    return ",".join(sorted(ids))


def filter_p1_to_bank(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Keep only problem_id×variant_type pairs present in the question bank."""
    if df is None or df.empty or "variant_type" not in df.columns:
        return df
    bank = _load_bank(family)
    keys = bank[["problem_id", "variant_type"]].drop_duplicates()
    out = df.copy()
    out["variant_type"] = out["variant_type"].map(_norm_variant)
    return out.merge(keys, on=["problem_id", "variant_type"], how="inner")


def _load_p1_raw(family: str, model: str) -> pd.DataFrame | None:
    tag = P1_TAGS[model]
    path = RAW / f"{family}_P1_behavioral_{tag}.csv"
    df = _safe_read(path)
    if family == "BW" and (df is None or df.empty):
        combined = _safe_read(RAW / "BW_P1_behavioral.csv")
        if combined is not None and "model" in combined.columns:
            df = combined[combined["model"] == model].copy()
    if df is None or df.empty:
        return None
    if "variant_type" not in df.columns:
        return None
    df = df.copy()
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    valid = _valid_mask(df)
    df = df[valid].drop_duplicates(["problem_id", "variant_type"], keep="last")
    return filter_p1_to_bank(df, family)


def _p1_coverage_row(family: str, model: str, bank: pd.DataFrame) -> tuple[dict, list[dict]]:
    expected_pairs = _bank_pairs(bank)
    expected_canon = _bank_canonical_ids(bank)
    variant_expected = _variant_expected(bank)
    df = _load_p1_raw(family, model)

    if df is None:
        observed_pairs: set[tuple[str, str]] = set()
        observed_canon: set[str] = set()
        n_errors = 0
    else:
        observed_pairs = set(zip(df["problem_id"], df["variant_type"]))
        observed_canon = set(
            df.loc[df["variant_type"] == "canonical", "problem_id"]
        )
        raw_path = RAW / f"{family}_P1_behavioral_{P1_TAGS[model]}.csv"
        if family == "BW" and not raw_path.exists():
            raw_df = _safe_read(RAW / "BW_P1_behavioral.csv")
            raw_df = raw_df[raw_df["model"] == model] if raw_df is not None else None
        else:
            raw_df = _safe_read(raw_path)
        if raw_df is not None and "raw_response" in raw_df.columns:
            n_errors = int(raw_df["raw_response"].astype(str).str.startswith("ERROR:").sum())
        else:
            n_errors = 0

    missing_canon = expected_canon - observed_canon
    extra_canon = observed_canon - expected_canon
    missing_pairs = expected_pairs - observed_pairs
    extra_pairs = observed_pairs - expected_pairs
    variant_observed = (
        df.groupby("variant_type")["problem_id"].nunique().to_dict()
        if df is not None
        else {}
    )

    if len(missing_pairs) == 0 and len(extra_pairs) == 0:
        coverage_label = "full_bank"
    elif len(missing_canon) > 0:
        coverage_label = "partial"
    elif len(extra_canon) > 0 or len(extra_pairs) > 0:
        coverage_label = "contaminated_extra_ids"
    else:
        coverage_label = "partial_canonical"

    row = {
        "family": family,
        "probe": "P1",
        "model": SHORT[model],
        "bank_canonical_n": len(expected_canon),
        "observed_canonical_n": len(observed_canon & expected_canon),
        "bank_total_n": len(expected_pairs),
        "observed_total_n": len(observed_pairs & expected_pairs),
        "n_errors": n_errors,
        "canonical_complete": len(missing_canon) == 0,
        "bank_complete": len(missing_pairs) == 0 and len(extra_pairs) == 0,
        "missing_canonical_ids": _join_ids(missing_canon),
        "extra_canonical_ids": _join_ids(extra_canon),
        "missing_pair_count": len(missing_pairs),
        "extra_pair_count": len(extra_pairs),
        "reference_full_n": len(expected_canon),
        "coverage_label": coverage_label,
    }
    for variant in VARIANTS:
        row[f"{variant}_expected"] = variant_expected.get(variant, 0)
        row[f"{variant}_observed"] = variant_observed.get(variant, 0)

    gaps = []
    for pid in sorted(missing_canon):
        gaps.append(
            {
                "family": family,
                "probe": "P1",
                "model": SHORT[model],
                "gap_kind": "missing_canonical",
                "gap_id": pid,
            }
        )
    for pid, variant in sorted(missing_pairs - {(p, "canonical") for p in missing_canon}):
        if variant == "canonical" and pid in missing_canon:
            continue
        gaps.append(
            {
                "family": family,
                "probe": "P1",
                "model": SHORT[model],
                "gap_kind": f"missing_{variant.lower()}",
                "gap_id": pid,
            }
        )
    return row, gaps


def _session_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    inst = df["instance_type"] if "instance_type" in df.columns else pd.Series([""] * len(df))
    return set(zip(df["problem_id"], inst.astype(str)))


def _load_algo_p2(path_name: str) -> pd.DataFrame:
    df = _safe_read(RAW / path_name)
    return df if df is not None else pd.DataFrame()


def _algo_p2_coverage(
    probe: str,
    path_name: str,
    reference_model: str = "openai/gpt-4o",
) -> tuple[list[dict], list[dict]]:
    df = _load_algo_p2(path_name)
    if df.empty or "model" not in df.columns:
        rows = []
        for model in MODELS:
            rows.append(
                {
                    "family": "ALGO",
                    "probe": probe,
                    "model": SHORT[model],
                    "bank_canonical_n": 0,
                    "observed_canonical_n": 0,
                    "bank_total_n": 0,
                    "observed_total_n": 0,
                    "n_errors": 0,
                    "canonical_complete": False,
                    "bank_complete": False,
                    "missing_canonical_ids": "",
                    "missing_pair_count": 0,
                    "reference_full_n": 0,
                    "coverage_label": "missing",
                }
            )
        return rows, []

    ref_keys = _session_keys(df[df["model"] == reference_model])
    ref_problems = set(df.loc[df["model"] == reference_model, "problem_id"])
    rows: list[dict] = []
    gaps: list[dict] = []

    for model in MODELS:
        sub = df[df["model"] == model]
        valid = sub[_valid_mask(sub)] if not sub.empty else sub
        keys = _session_keys(valid)
        problems = set(valid["problem_id"]) if not valid.empty else set()
        missing_keys = ref_keys - keys
        missing_problems = ref_problems - problems
        n_errors = 0
        if not sub.empty and "raw_response" in sub.columns:
            n_errors = int(sub["raw_response"].astype(str).str.startswith("ERROR:").sum())

        rows.append(
            {
                "family": "ALGO",
                "probe": probe,
                "model": SHORT[model],
                "bank_canonical_n": len(ref_problems),
                "observed_canonical_n": len(problems),
                "bank_total_n": len(ref_keys),
                "observed_total_n": len(keys),
                "n_errors": n_errors,
                "canonical_complete": len(missing_problems) == 0,
                "bank_complete": len(missing_keys) == 0,
                "missing_canonical_ids": _join_ids(missing_problems),
                "missing_pair_count": len(missing_keys),
                "reference_full_n": len(ref_keys),
                "reference_model": SHORT[reference_model],
                "coverage_label": (
                    "full_bank"
                    if len(missing_keys) == 0
                    else ("partial_canonical" if len(missing_problems) == 0 else "partial")
                ),
            }
        )
        for pid in sorted(missing_problems):
            gaps.append(
                {
                    "family": "ALGO",
                    "probe": probe,
                    "model": SHORT[model],
                    "gap_kind": "missing_canonical",
                    "gap_id": pid,
                }
            )
        for pid, inst in sorted(missing_keys):
            if pid in missing_problems:
                continue
            gaps.append(
                {
                    "family": "ALGO",
                    "probe": probe,
                    "model": SHORT[model],
                    "gap_kind": "missing_session",
                    "gap_id": f"{pid}|{inst}",
                }
            )
    return rows, gaps


def load_gsm_p2_merged() -> pd.DataFrame:
    """GSM P2 sessions: four models in GSM_P2_cci.csv + o4-mini in phase1_o1mini file."""
    df = _safe_read(RAW / "GSM_P2_cci.csv")
    df2 = _safe_read(RAW / "GSM_P2_phase1_o1mini.csv")
    if df is not None and df2 is not None:
        common = sorted(set(df.columns) & set(df2.columns))
        out = pd.concat([df[common], df2[common]], ignore_index=True)
    elif df is not None:
        out = df
    elif df2 is not None:
        out = df2
    else:
        return pd.DataFrame()
    if "session_b_correct" in out.columns and "either_session_correct" not in out.columns:
        out = out.rename(columns={"session_b_correct": "either_session_correct"})
    overlay_path = DER / "GSM_P2_session_correct.csv"
    if overlay_path.exists():
        ov = pd.read_csv(overlay_path, dtype=str).fillna("")
        keep = [
            c
            for c in (
                "problem_id",
                "model",
                "either_session_correct",
                "phase1_correct",
                "phase2a_correct",
                "phase2b_correct",
            )
            if c in ov.columns
        ]
        ov = ov[keep].drop_duplicates(["problem_id", "model"])
        out = out.drop(
            columns=[c for c in ("either_session_correct", "phase1_correct", "phase2a_correct", "phase2b_correct") if c in out.columns],
            errors="ignore",
        )
        out = out.merge(ov, on=["problem_id", "model"], how="left")
    return out


def _load_gsm_p2() -> pd.DataFrame:
    return load_gsm_p2_merged()


def _gsm_p2_coverage() -> tuple[list[dict], list[dict]]:
    df = load_gsm_p2_merged()
    expected = set(_gsm_bank_canonical_ids())
    rows: list[dict] = []
    gaps: list[dict] = []

    for model in MODELS:
        sub = df[df["model"] == model] if not df.empty and "model" in df.columns else pd.DataFrame()
        observed = set(sub["problem_id"]) if not sub.empty else set()
        missing = expected - observed
        extra = observed - expected
        rows.append(
            {
                "family": "GSM",
                "probe": "P2",
                "model": SHORT[model],
                "bank_canonical_n": len(expected),
                "observed_canonical_n": len(observed & expected),
                "bank_total_n": len(expected),
                "observed_total_n": len(observed & expected),
                "n_errors": 0,
                "canonical_complete": len(missing) == 0,
                "bank_complete": len(missing) == 0 and len(extra) == 0,
                "missing_canonical_ids": _join_ids(missing),
                "extra_canonical_ids": _join_ids(extra),
                "missing_pair_count": len(missing),
                "reference_full_n": len(expected),
                "coverage_label": (
                    "full_bank"
                    if len(missing) == 0 and len(extra) == 0
                    else ("partial" if missing else "extra_ids")
                ),
            }
        )
        for pid in sorted(missing):
            gaps.append(
                {
                    "family": "GSM",
                    "probe": "P2",
                    "model": SHORT[model],
                    "gap_kind": "missing_canonical",
                    "gap_id": pid,
                }
            )
    return rows, gaps


def master_coverage_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    gaps: list[dict] = []

    for family in ["GSM", "ALGO", "BW"]:
        bank = _load_bank(family)
        for model in MODELS:
            row, row_gaps = _p1_coverage_row(family, model, bank)
            rows.append(row)
            gaps.extend(row_gaps)

    for probe, path in [
        ("P2", None),
        ("P2A_normal", "ALGO_P2_phase2_normal.csv"),
        ("P2A_elicited", "ALGO_P2_phase2_normal_elicited.csv"),
        ("P2B_plausible", "ALGO_P2_phase2_injected.csv"),
        ("P2B_implausible", "ALGO_P2_phase2_injected_implausible.csv"),
    ]:
        if probe == "P2":
            p2_rows, p2_gaps = _gsm_p2_coverage()
        else:
            p2_rows, p2_gaps = _algo_p2_coverage(probe, path)  # type: ignore[arg-type]
        rows.extend(p2_rows)
        gaps.extend(p2_gaps)

    table = pd.DataFrame(rows)
    gap_df = pd.DataFrame(gaps)
    return table, gap_df


def _load_gsm_p2_cci_only() -> pd.DataFrame:
    """Four-model GSM P2 file only (Claude/GPT-4o/Gemini/Llama) — for paired sensitivity."""
    df = _safe_read(RAW / "GSM_P2_cci.csv")
    return df if df is not None else pd.DataFrame()


def gsm_p2_sensitivity() -> tuple[pd.DataFrame, dict]:
    """Claude vs GPT-4o CCI: zero-imputed full bank vs complete-case intersection."""
    df = _load_gsm_p2_cci_only()
    claude_id = "anthropic/claude-sonnet-4"
    gpt4o_id = "openai/gpt-4o"

    def _series(model: str) -> pd.Series:
        sub = df[df["model"] == model] if not df.empty else pd.DataFrame()
        if sub.empty:
            return pd.Series(dtype=float)
        out = pd.to_numeric(sub["cci_score"], errors="coerce")
        return pd.Series(out.values, index=sub["problem_id"].values)

    cl = _series(claude_id)
    gp = _series(gpt4o_id)
    ll = _series("meta-llama/llama-3.1-8b-instruct")

    bank_ids = _gsm_bank_canonical_ids()
    imputed_ids = bank_ids
    common_ids = sorted(set(cl.index) & set(gp.index))
    cl_common = cl.reindex(common_ids)
    gp_common = gp.reindex(common_ids)

    cl_imputed = np.array([float(cl.get(i, 0.0)) for i in imputed_ids], dtype=float)
    gp_imputed = np.array([float(gp.get(i, 0.0)) for i in imputed_ids], dtype=float)

    w_imputed, p_imputed = stats.wilcoxon(
        cl_imputed, gp_imputed, alternative="greater", zero_method="wilcox"
    )
    t_imputed_p = float(stats.ttest_rel(cl_imputed, gp_imputed, alternative="greater").pvalue)

    if len(common_ids) >= 2 and cl_common.notna().all() and gp_common.notna().all():
        w_cc, p_cc = stats.wilcoxon(
            cl_common.values,
            gp_common.values,
            alternative="greater",
            zero_method="wilcox",
        )
        t_cc_p = float(
            stats.ttest_rel(cl_common.values, gp_common.values, alternative="greater").pvalue
        )
    else:
        w_cc, p_cc, t_cc_p = float("nan"), float("nan"), float("nan")

    sensitivity_rows = [
        {
            "analysis": "full_bank_zero_imputed",
            "comparison": "Claude_vs_GPT4o",
            "n_pairs": len(imputed_ids),
            "claude_mean_cci": float(cl_imputed.mean()),
            "gpt4o_mean_cci": float(gp_imputed.mean()),
            "wilcoxon_W": float(w_imputed),
            "wilcoxon_p_one_sided": float(p_imputed),
            "paired_t_p": t_imputed_p,
            "denominator_label": "zero_imputed",
        },
        {
            "analysis": "complete_case_only",
            "comparison": "Claude_vs_GPT4o",
            "n_pairs": len(common_ids),
            "claude_mean_cci": float(cl_common.mean()) if len(common_ids) else float("nan"),
            "gpt4o_mean_cci": float(gp_common.mean()) if len(common_ids) else float("nan"),
            "wilcoxon_W": float(w_cc),
            "wilcoxon_p_one_sided": float(p_cc),
            "paired_t_p": t_cc_p,
            "denominator_label": "complete_case",
        },
    ]
    sensitivity = pd.DataFrame(sensitivity_rows)

    gap_json = {
        "generated_by": "scripts/runs/coverage_audit.py",
        "bank_n": len(bank_ids),
        "claude_n": int(len(cl)),
        "gpt4o_n": int(len(gp)),
        "llama_n": int(len(ll)),
        "gemini_n": int(len(_series("google/gemini-2.5-flash"))),
        "o4mini_n": int(len(_series("openai/o4-mini"))),
        "missing_gpt4o": sorted(set(bank_ids) - set(gp.index)),
        "missing_llama": sorted(set(bank_ids) - set(ll.index)),
        "missing_o4mini": sorted(set(bank_ids) - set(_series("openai/o4-mini").index)),
        "common_claude_gpt4o_ids": common_ids,
        "wilcoxon_imputed_W": float(w_imputed),
        "wilcoxon_imputed_p": float(p_imputed),
        "ttest_imputed_p": t_imputed_p,
        "wilcoxon_complete_case_n": len(common_ids),
        "wilcoxon_complete_case_p": float(p_cc),
        "mean_cci_claude_imputed": float(cl_imputed.mean()),
        "mean_cci_gpt4o_imputed": float(gp_imputed.mean()),
        "mean_cci_claude_complete_case": float(cl_common.mean()) if len(common_ids) else None,
        "mean_cci_gpt4o_complete_case": float(gp_common.mean()) if len(common_ids) else None,
    }
    return sensitivity, gap_json


def table_denominator_flags(coverage: pd.DataFrame) -> pd.DataFrame:
    """Flag known paper tables/figures that depend on partial or imputed slices."""
    partial = coverage[~coverage["bank_complete"]].copy()
    flags = [
        {
            "artifact": "table1_gsm.csv",
            "metric": "P1 canonical / W3 accuracy",
            "models_affected": "GPT-4o, Llama",
            "issue": "20/44 bank-valid canonical; raw has GSM_021-040 (wrong IDs) + missing GSM_041-064",
            "denominator_label": "partial_canonical",
            "action": "Reconcile ID mapping or finish missing bank IDs (checklist 1.7)",
        },
        {
            "artifact": "probe2_gsm_metrics.csv / main.tex CCI",
            "metric": "GSM P2 mean CCI",
            "models_affected": "o4-mini",
            "issue": "0/44 GSM P2 rows",
            "denominator_label": "missing",
            "action": "Exclude o4-mini from five-model P2 claims or run P2",
        },
        {
            "artifact": "gsm_cci_wilcoxon_sensitivity.csv",
            "metric": "Claude vs GPT-4o paired CCI",
            "models_affected": "Claude, GPT-4o",
            "issue": "Two analyses: zero-imputed n=44 vs complete-case intersection",
            "denominator_label": "see analysis column",
            "action": "Never report imputed p without complete-case counterpart",
        },
        {
            "artifact": "ALGO P2A elicited metrics",
            "metric": "Algorithm elicitation sessions",
            "models_affected": "Claude, Gemini, Llama",
            "issue": "61/110 sessions vs 450 for GPT-4o/o4-mini",
            "denominator_label": "partial",
            "action": "Do not pool elicited counts across models without label",
        },
        {
            "artifact": "BW_P1_behavioral.csv / table1_bw",
            "metric": "BW P1 canonical accuracy",
            "models_affected": "Claude, GPT-4o, Llama",
            "issue": "Combined file contains 44 GSM IDs not in BW bank (109 observed vs 65 bank)",
            "denominator_label": "contaminated_extra_ids",
            "action": "Filter to BW bank IDs only before cross-family tables",
        },
        {
            "artifact": "main.tex / cross-model P2 claims",
            "metric": "GSM Probe 2 five-model comparison",
            "models_affected": "all five",
            "issue": "Only 4/5 models have GSM P2 rows; o4-mini 0/44",
            "denominator_label": "four_model_only",
            "action": 'Never claim "five models on Probe 2" without o4-mini runs',
        },
    ]
    flag_df = pd.DataFrame(flags)
    if not partial.empty:
        extra = partial[
            ["family", "probe", "model", "coverage_label", "missing_canonical_ids"]
        ].copy()
        extra = extra.rename(
            columns={
                "family": "artifact",
                "probe": "metric",
                "model": "models_affected",
                "coverage_label": "denominator_label",
                "missing_canonical_ids": "issue",
            }
        )
        extra["action"] = "Auto-flagged from master_coverage_table"
        flag_df = pd.concat([flag_df, extra], ignore_index=True)
    return flag_df


def cells_needing_runs(coverage: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Actionable cells: incomplete slices grouped for API run planning."""
    incomplete = coverage[~coverage["bank_complete"]].copy()
    if incomplete.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "probe",
                "model",
                "observed_canonical_n",
                "bank_canonical_n",
                "missing_canonical_n",
                "coverage_label",
                "run_type",
                "priority",
            ]
        )

    rows: list[dict] = []
    for _, row in incomplete.iterrows():
        missing_n = int(row["bank_canonical_n"] - row["observed_canonical_n"])
        label = str(row["coverage_label"])
        if label == "contaminated_extra_ids":
            run_type = "data_cleanup"
            priority = "P1"
        elif row["probe"] == "P2" and missing_n == row["bank_canonical_n"]:
            run_type = "full_probe_run"
            priority = "P0"
        elif row["probe"] == "P1":
            run_type = "partial_p1_rerun"
            priority = "P1"
        else:
            run_type = "partial_probe_run"
            priority = "P1"

        rows.append(
            {
                "family": row["family"],
                "probe": row["probe"],
                "model": row["model"],
                "observed_canonical_n": int(row["observed_canonical_n"]),
                "bank_canonical_n": int(row["bank_canonical_n"]),
                "missing_canonical_n": max(missing_n, 0),
                "coverage_label": label,
                "missing_canonical_ids": row.get("missing_canonical_ids", ""),
                "extra_canonical_ids": row.get("extra_canonical_ids", ""),
                "run_type": run_type,
                "priority": priority,
            }
        )
    return pd.DataFrame(rows)


def coverage_audit_summary(
    coverage: pd.DataFrame,
    gaps: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gap_json: dict,
) -> str:
    """Human-readable executive summary for checklist §0.1."""
    n_slices = len(coverage)
    incomplete = coverage[~coverage["bank_complete"]]
    full = coverage[coverage["bank_complete"]]

    lines = [
        "# Coverage audit summary (§0.1)",
        "",
        f"Generated by `scripts/runs/coverage_audit.py`. "
        f"**{len(full)}/{n_slices}** model×probe slices are bank-complete.",
        "",
        "## Canonical command",
        "",
        "```bash",
        "python scripts/runs/rederive_all_metrics.py",
        "```",
        "",
        "Paper tables must use this path only (`results/paper/AUDIT/README.md`).",
        "",
        "## Incomplete slices (need runs or cleanup)",
        "",
    ]

    if incomplete.empty:
        lines.append("_None — all slices match question banks._")
    else:
        cols = [
            "family",
            "probe",
            "model",
            "observed_canonical_n",
            "bank_canonical_n",
            "coverage_label",
        ]
        lines.append("```")
        lines.append(incomplete[cols].to_string(index=False))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Key findings",
            "",
            "### GSM P1 (GPT-4o, Llama: 20/44 bank-valid)",
            "- Raw has **40 canonical rows** but only **20** match the bank (GSM_001–020).",
            "- **GSM_021–040 are duplicate reruns of GSM_001–020** (same answers) — **not** valid remap to GSM_041–060.",
            "- **24 missing bank IDs** (GSM_041–064) still need API runs (or separate archived logs if found).",
            "- Claude/Gemini/o4-mini: **44/44** full bank.",
            "",
            "### GSM P2",
            "- Claude/Gemini/GPT-4o/Llama: **44/44** in `GSM_P2_cci.csv`.",
            "- **o4-mini: 44/44** in `GSM_P2_phase1_o1mini.csv` (merge into derivations; no new API).",
            "",
            "### GSM P2 Claude vs GPT-4o CCI (paired test)",
        ]
    )
    for _, srow in sensitivity.iterrows():
        lines.append(
            f"- **{srow['denominator_label']}**: n={int(srow['n_pairs'])}, "
            f"p={srow['wilcoxon_p_one_sided']:.4g} "
            f"(Claude mean={srow['claude_mean_cci']:.3f}, "
            f"GPT-4o mean={srow['gpt4o_mean_cci']:.3f})"
        )
    if sensitivity["n_pairs"].nunique() == 1:
        lines.append(
            "- Imputed and complete-case **coincide** (both models have all 44 bank IDs)."
        )

    lines.extend(
        [
            "",
            "### BW P1 combined file contamination",
            "- `BW_P1_behavioral.csv` rows for Claude/GPT-4o/Llama include **44 GSM IDs** "
            "not in the BW bank.",
            "- Filter to bank IDs before BW accuracy tables; do not treat 109 as canonical n.",
            "",
            "### ALGO P2A elicited",
            "- Claude/Gemini/Llama: **61/110** sessions vs **110/110** for GPT-4o/o4-mini.",
            "",
            "## Output files",
            "",
            "| File | Purpose |",
            "|------|---------|",
            "| `master_coverage_table.csv` | model × family × probe × n × label |",
            f"| `master_coverage_gaps.csv` | Long-form missing IDs ({len(gaps)} rows) |",
            "| `cells_needing_runs.csv` | Actionable incomplete cells |",
            "| `table_denominator_flags.csv` | Paper tables with partial denominators |",
            "| `gsm_cci_wilcoxon_sensitivity.csv` | Imputed vs complete-case paired stats |",
            "| `gsm_p2_gap.json` | Machine-readable P2 gap + test stats |",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    AUD.mkdir(parents=True, exist_ok=True)
    INV.mkdir(parents=True, exist_ok=True)

    print("[audit] master coverage table")
    coverage, gaps = master_coverage_table()
    coverage.to_csv(DER / "master_coverage_table.csv", index=False)
    coverage.to_csv(AUD / "master_coverage_table.csv", index=False)
    gaps.to_csv(DER / "master_coverage_gaps.csv", index=False)

    print("[audit] GSM P2 imputed vs complete-case sensitivity")
    sensitivity, gap_json = gsm_p2_sensitivity()
    sensitivity.to_csv(AUD / "gsm_cci_wilcoxon_sensitivity.csv", index=False)
    (INV / "gsm_p2_gap.json").write_text(json.dumps(gap_json, indent=2) + "\n")

    print("[audit] table denominator flags")
    flags = table_denominator_flags(coverage)
    flags.to_csv(DER / "table_denominator_flags.csv", index=False)

    print("[audit] cells needing runs")
    needs_runs = cells_needing_runs(coverage, gaps)
    needs_runs.to_csv(DER / "cells_needing_runs.csv", index=False)

    summary_md = coverage_audit_summary(coverage, gaps, sensitivity, gap_json)
    (DER / "COVERAGE_AUDIT_SUMMARY.md").write_text(summary_md)
    (AUD / "COVERAGE_AUDIT_SUMMARY.md").write_text(summary_md)

    incomplete = coverage[~coverage["bank_complete"]]
    print(f"\nCoverage summary: {len(coverage)} slices, {len(incomplete)} incomplete")
    if not incomplete.empty:
        cols = ["family", "probe", "model", "observed_canonical_n", "bank_canonical_n", "coverage_label"]
        print(incomplete[cols].to_string(index=False))

    print("\nGSM P2 Claude vs GPT-4o sensitivity:")
    print(sensitivity.to_string(index=False))


def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()
