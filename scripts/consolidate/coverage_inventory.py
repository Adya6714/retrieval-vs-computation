#!/usr/bin/env python3
"""O4: Complete data-coverage inventory across P1/P2/P3.

Writes:
  results/derived/W_SPEC.md
  results/derived/COVERAGE_MATRIX.csv
  results/derived/COVERAGE_PROBE2.csv
  results/derived/COVERAGE_PROBE3.csv
  results/derived/IDLE_CELLS.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.retention import MIN_CANONICAL_FOR_RETENTION  # noqa: E402
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}
VALID_MODELS = set(PAPER_MODELS.values())
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
FAMILIES = ["ALGO", "BW", "GSM"]


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def write_w_spec() -> None:
    text = """# W1–W6 Variant Specification

Source: research vault `EF-01_Probe1_Surface_Invariance.md`, \
`docs/workbench/EVALUATION_WALKTHROUGH.md`, and \
`scripts/generation/stage2_generate_variants.py`.

Shared design (all families): six answer-preserving or answer-changing surface \
transforms of each canonical item. Variants must pass the family verifier before \
any model call. Zero-shot CoT, T=0 at evaluation.

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
- **Retention R_W3** = Acc_W3 / Acc_canonical; undefined when Acc_canonical < 0.30 \
  (`MIN_CANONICAL_FOR_RETENTION`).
"""
    out = DER / "W_SPEC.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


def _load_p1() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("*_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        if path.name.startswith("ALGO_"):
            fam = "ALGO"
        elif path.name.startswith("BW_"):
            fam = "BW"
        elif path.name.startswith("GSM_"):
            fam = "GSM"
        else:
            continue
        df["family"] = fam
        df["model_short"] = df["model"].map(PAPER_MODELS).fillna(df["model"])
        df["variant"] = df["variant_type"].map(normalize_variant)
        df["collected"] = True
        df["included_bool"] = _is_true(df["included"])
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        # also track raw collected before exclusion filter for n_rows_collected
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_coverage_matrix(p1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(m for m in p1["model_short"].dropna().unique() if m in VALID_MODELS)
    for fam in FAMILIES:
        # bank expected counts
        bank = pd.read_csv(REPO_ROOT / f"data/problems/question_bank_{fam.lower()}.csv", dtype=str)
        bank["variant"] = bank["variant_type"].map(normalize_variant)
        for model in models:
            # skip models that never appear in this family
            fam_model = p1[(p1["family"] == fam) & (p1["model_short"] == model)]
            if fam_model.empty:
                continue
            can = fam_model[(fam_model["variant"] == "canonical") & fam_model["included_bool"]]
            can_acc = float(can["ok"].mean()) if len(can) else float("nan")
            for var in VARIANTS:
                sub_all = fam_model[fam_model["variant"] == var]
                # after exclusions: included True AND not in family exclusion list
                sub_inc = sub_all[sub_all["included_bool"]].copy()
                if not sub_inc.empty:
                    sub_inc = filter_excluded(sub_inc, family=fam)
                n_collected = int(len(sub_all))
                n_valid = int(len(sub_inc))
                var_acc = float(sub_inc["ok"].mean()) if n_valid else float("nan")

                retention_defined = False
                reason = ""
                if var == "W3":
                    if not np.isfinite(can_acc):
                        reason = "no_valid_canonical_rows"
                    elif can_acc < MIN_CANONICAL_FOR_RETENTION:
                        reason = f"canonical_accuracy_{can_acc:.3f}_below_floor_{MIN_CANONICAL_FOR_RETENTION}"
                    elif n_valid == 0:
                        reason = "no_valid_w3_rows"
                    else:
                        retention_defined = True
                        reason = ""
                elif var == "canonical":
                    reason = "retention_is_w3_only"
                else:
                    reason = "retention_defined_only_for_W3"

                bank_n = int((bank["variant"] == var).sum())
                rows.append(
                    {
                        "family": fam,
                        "variant": var,
                        "model": model,
                        "n_bank": bank_n,
                        "n_rows_collected": n_collected,
                        "n_rows_valid_after_exclusions": n_valid,
                        "canonical_accuracy": round(can_acc, 4) if can_acc == can_acc else "",
                        "variant_accuracy": round(var_acc, 4) if var_acc == var_acc else "",
                        "retention_defined": retention_defined if var == "W3" else False,
                        "reason_if_undefined": reason,
                    }
                )
    out = pd.DataFrame(rows)
    path = DER / "COVERAGE_MATRIX.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")
    return out


def _model_short_series(s: pd.Series) -> pd.Series:
    return s.map(PAPER_MODELS).fillna(s)


def build_probe2() -> pd.DataFrame:
    rows = []

    # --- GSM ---
    gsm_cci = pd.read_csv(RAW / "GSM_P2_cci.csv", dtype=str).fillna("") if (RAW / "GSM_P2_cci.csv").exists() else pd.DataFrame()
    gsm_phase1_files = {
        "Claude": RAW / "GSM_P2_phase1_claude.csv",
        "Gemini": RAW / "GSM_P2_phase1_gemini.csv",
        "GPT-4o": RAW / "GSM_P2_phase1_gpt4o.csv",
        "Llama": RAW / "GSM_P2_phase1_llama.csv",
        "o4-mini": RAW / "GSM_P2_phase1_o1mini.csv",
    }
    for model, path in gsm_phase1_files.items():
        if not path.exists():
            rows.append(_p2_row("GSM", "phase1_declare", model, 0, "", False, False, "no_phase1_file"))
            rows.append(_p2_row("GSM", "phase2A_execute", model, 0, "", False, False, "no_phase1_file"))
            rows.append(_p2_row("GSM", "phase2B_inject", model, 0, "", False, False, "no_phase1_file"))
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        n = len(df)
        can_exec = ""
        sess = DER / "GSM_P2_session_correct.csv"
        if sess.exists():
            sc = pd.read_csv(sess, dtype=str).fillna("")
            sc["model_short"] = _model_short_series(sc["model"])
            sub = sc[sc["model_short"] == model]
            if len(sub):
                can_exec = round(float(_is_true(sub["phase1_correct"]).mean()), 4)

        g = pd.DataFrame()
        if not gsm_cci.empty:
            g = gsm_cci[_model_short_series(gsm_cci["model"]) == model]
        cci_ok = False
        tep_ok = False
        reason = ""
        if len(g):
            cci_vals = pd.to_numeric(g["cci_score"], errors="coerce")
            tep_vals = pd.to_numeric(g["tep_score"], errors="coerce")
            cci_ok = bool(cci_vals.notna().any())
            tep_ok = bool(tep_vals.notna().any())
            if not cci_ok:
                reason = "cci_all_nan"
            if not tep_ok:
                reason = (reason + "; tep_all_nan").strip("; ")
        elif model == "o4-mini":
            reason = "o4-mini_phase1_collected_but_excluded_from_GSM_P2_cci"
        else:
            reason = "model_absent_from_GSM_P2_cci"

        rows.append(_p2_row("GSM", "phase1_declare", model, n, can_exec, cci_ok, tep_ok, reason))
        rows.append(_p2_row("GSM", "phase2A_execute", model, len(g), can_exec, cci_ok, False, reason))
        rows.append(_p2_row("GSM", "phase2B_inject", model, len(g), "", False, tep_ok, reason))

    # --- ALGO ---
    algo_cci = pd.read_csv(DER / "ALGO_P2_cci.csv", dtype=str).fillna("") if (DER / "ALGO_P2_cci.csv").exists() else pd.DataFrame()
    algo_phase1 = {
        "Claude": RAW / "ALGO_P2_phase1_claude_new.csv",
        "Gemini": RAW / "ALGO_P2_phase1_gemini.csv",
        "GPT-4o": RAW / "ALGO_P2_phase1_gpt4o_new.csv",
        "Llama": RAW / "ALGO_P2_phase1_llama_new.csv",
    }
    # fallbacks
    if not algo_phase1["GPT-4o"].exists():
        algo_phase1["GPT-4o"] = RAW / "ALGO_P2_phase1_gpt4o.csv"
    if not algo_phase1["Llama"].exists():
        algo_phase1["Llama"] = RAW / "ALGO_P2_phase1_llama.csv"

    algo_p2_normal = {
        "shared": RAW / "ALGO_P2_phase2_normal.csv",
        "gemini": RAW / "ALGO_P2_phase2_normal_gemini.csv",
    }
    algo_p2_inj = {
        "shared": RAW / "ALGO_P2_phase2_injected.csv",
        "gemini": RAW / "ALGO_P2_phase2_injected_gemini.csv",
    }

    def _algo_phase2_n(model: str, kind: str) -> tuple[int, str]:
        if kind == "normal":
            path = algo_p2_normal["gemini"] if model == "Gemini" else algo_p2_normal["shared"]
        else:
            path = algo_p2_inj["gemini"] if model == "Gemini" else algo_p2_inj["shared"]
        if not path.exists():
            return 0, f"missing_{path.name}"
        df = pd.read_csv(path, dtype=str).fillna("")
        df["model_short"] = _model_short_series(df["model"])
        sub = df[df["model_short"] == model]
        return int(sub["problem_id"].nunique() if "problem_id" in sub.columns else len(sub)), ""

    for model, path in algo_phase1.items():
        if not path.exists():
            rows.append(_p2_row("ALGO", "phase1_declare", model, 0, "", False, False, "no_phase1_file"))
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        n = int(df["problem_id"].nunique()) if "problem_id" in df.columns else len(df)
        # greedy_assessment / predicted as weak accuracy proxy
        can_exec = ""
        if "greedy_assessment_correct" in df.columns:
            # not execution accuracy — leave blank, note separately
            can_exec = ""
        cci_ok = False
        reason = ""
        if not algo_cci.empty:
            g = algo_cci[_model_short_series(algo_cci["model"]) == model]
            if len(g):
                cci_vals = pd.to_numeric(g["cci_score"], errors="coerce")
                cci_ok = bool(cci_vals.notna().any())
                if not cci_ok:
                    reason = "cci_all_nan_or_unparseable_execution"
            else:
                reason = "model_absent_from_ALGO_P2_cci"
        rows.append(_p2_row("ALGO", "phase1_declare", model, n, can_exec, cci_ok, False, reason))

        n2, r2 = _algo_phase2_n(model, "normal")
        rows.append(_p2_row("ALGO", "phase2A_execute", model, n2, "", cci_ok, False, r2 or reason))
        n3, r3 = _algo_phase2_n(model, "inject")
        # ALGO TEP not standardized like GSM; injection traces exist
        tep_ok = n3 > 0
        rows.append(
            _p2_row(
                "ALGO",
                "phase2B_inject",
                model,
                n3,
                "",
                False,
                tep_ok,
                r3 or ("injection_traces_exist_tep_metric_not_standardized" if tep_ok else "no_injection_traces"),
            )
        )

    # o4-mini ALGO P2 absent
    rows.append(_p2_row("ALGO", "phase1_declare", "o4-mini", 0, "", False, False, "never_collected"))
    rows.append(_p2_row("ALGO", "phase2A_execute", "o4-mini", 0, "", False, False, "never_collected"))
    rows.append(_p2_row("ALGO", "phase2B_inject", "o4-mini", 0, "", False, False, "never_collected"))

    # --- BW ---
    bw_plans = pd.read_csv(RAW / "BW_P2_plans.csv", dtype=str).fillna("") if (RAW / "BW_P2_plans.csv").exists() else pd.DataFrame()
    bw_cci = pd.read_csv(RAW / "BW_P2_cci.csv", dtype=str).fillna("") if (RAW / "BW_P2_cci.csv").exists() else pd.DataFrame()
    bw_tep = pd.read_csv(RAW / "BW_P2_tep.csv", dtype=str).fillna("") if (RAW / "BW_P2_tep.csv").exists() else pd.DataFrame()
    bw_models = sorted(
        set(_model_short_series(bw_plans["model"])) | set(_model_short_series(bw_cci["model"])) | set(_model_short_series(bw_tep["model"]))
        if len(bw_plans) or len(bw_cci) or len(bw_tep)
        else []
    )
    for model in bw_models or ["Claude", "GPT-4o", "Llama"]:
        p1n = int((_model_short_series(bw_plans["model"]) == model).sum()) if len(bw_plans) else 0
        cci_sub = bw_cci[_model_short_series(bw_cci["model"]) == model] if len(bw_cci) else pd.DataFrame()
        tep_sub = bw_tep[_model_short_series(bw_tep["model"]) == model] if len(bw_tep) else pd.DataFrame()
        cci_vals = pd.to_numeric(cci_sub["cci"], errors="coerce") if len(cci_sub) else pd.Series(dtype=float)
        tep_vals = pd.to_numeric(tep_sub["tep"], errors="coerce") if len(tep_sub) else pd.Series(dtype=float)
        cci_ok = bool(cci_vals.notna().any()) if len(cci_vals) else False
        tep_ok = bool(tep_vals.notna().any()) if len(tep_vals) else False
        reason = ""
        if len(cci_sub) and not cci_ok:
            reason = "cci_collected_but_all_null_abort_heavy"
        if len(tep_sub) and not tep_ok:
            reason = (reason + "; tep_mostly_null_aborts").strip("; ")
        rows.append(_p2_row("BW", "phase1_declare", model, p1n, "", cci_ok, tep_ok, reason))
        rows.append(_p2_row("BW", "phase2A_execute", model, len(cci_sub), "", cci_ok, False, reason or ("ok" if cci_ok else "cci_unusable")))
        rows.append(_p2_row("BW", "phase2B_inject", model, len(tep_sub), "", False, tep_ok, reason or ("ok" if tep_ok else "tep_unusable")))

    for model in ["Gemini", "o4-mini", "DeepSeek"]:
        if model not in bw_models:
            for ph in ["phase1_declare", "phase2A_execute", "phase2B_inject"]:
                rows.append(_p2_row("BW", ph, model, 0, "", False, False, "never_collected"))

    out = pd.DataFrame(rows).drop_duplicates(["family", "phase", "model"], keep="last")
    path = DER / "COVERAGE_PROBE2.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")
    return out


def _p2_row(family, phase, model, n, can_exec, cci, tep, reason):
    return {
        "family": family,
        "phase": phase,
        "model": model,
        "n_rows": n,
        "canonical_execution_accuracy": can_exec,
        "cci_computed": bool(cci),
        "tep_computed": bool(tep),
        "reason_if_missing": reason,
    }


def build_probe3() -> pd.DataFrame:
    audit_by_fam: dict[str, dict] = {}
    audit_path = DER / "P3_infinigram_query_audit.csv"
    shared_max = "13"
    gsm_max = "8"
    if audit_path.exists():
        adf = pd.read_csv(audit_path, dtype=str).fillna("")
        for _, r in adf.iterrows():
            fam = str(r.get("family", "")).strip()
            if fam in FAMILIES:
                audit_by_fam[fam] = r.to_dict()
            if fam == "shared_scorer":
                shared_max = str(r.get("max_n_default") or "13")
                gsm_max = str(r.get("max_n_arithmetic") or "8")

    window_by_fam = {
        "GSM": gsm_max,
        "ALGO": shared_max,
        "BW": shared_max,
    }
    if "GSM" in audit_by_fam:
        window_by_fam["GSM"] = str(
            audit_by_fam["GSM"].get("max_n_arithmetic")
            or audit_by_fam["GSM"].get("max_n_default")
            or gsm_max
        )
    for fam in ("ALGO", "BW"):
        if fam in audit_by_fam:
            window_by_fam[fam] = str(audit_by_fam[fam].get("max_n_default") or shared_max)


    # Mechanistic frequency-controlled
    mech_fc = Path(REPO_ROOT / "Mechanistic Frequency Controlled Algorithm.csv")
    mech_fc_models: set[str] = set()
    if mech_fc.exists():
        m = pd.read_csv(mech_fc, dtype=str)
        mech_fc_models = set(m["model"].unique())

    # Degeneracy
    degen_path = DER / "ALGO_gold_token_degeneracy.csv"
    degen_result = ""
    degen_run = False
    if degen_path.exists():
        degen_run = True
        d = pd.read_csv(degen_path, dtype=str)
        # summarize canonical row
        can = d[d["universe"].astype(str).str.contains("canonical", case=False)]
        if len(can):
            degen_result = (
                f"passes={can.iloc[0].get('passes_degeneracy_rule')}; "
                f"degenerate={can.iloc[0].get('degenerate')}; "
                f"modal_share={can.iloc[0].get('canonical_modal_share')}"
            )
        else:
            degen_result = f"rows={len(d)}"

    rows = []
    paper_models = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini", "DeepSeek"]

    for fam in FAMILIES:
        contam_path = RAW / f"{fam}_P3_contamination.csv"
        contam = pd.read_csv(contam_path, dtype=str).fillna("") if contam_path.exists() else pd.DataFrame()
        n_scored = len(contam)
        max_win = str(window_by_fam.get(fam, ""))
        if n_scored and "max_ngram_length" in contam.columns:
            obs = int(pd.to_numeric(contam["max_ngram_length"], errors="coerce").max())
            max_win = f"{max_win} (observed_hit_max={obs})"

        mech_path = RAW / f"{fam}_P3_mechanistic.csv"
        mech = pd.read_csv(mech_path, dtype=str).fillna("") if mech_path.exists() else pd.DataFrame()
        mech_models_legacy = set(mech["model"].unique()) if len(mech) else set()

        # Also check sweep files for this family
        sweep_hits = []
        for sp in RAW.glob("mechanistic_sweep*.csv"):
            sdf = pd.read_csv(sp, dtype=str, usecols=lambda c: c in {"problem_family", "model", "family"})
            # skip if can't read family
        for sp in RAW.glob("mechanistic*.csv"):
            try:
                sdf = pd.read_csv(sp, dtype=str, nrows=5000)
            except Exception:
                continue
            fam_col = "problem_family" if "problem_family" in sdf.columns else ("family" if "family" in sdf.columns else None)
            if fam_col is None or "model" not in sdf.columns:
                continue
            sub = sdf[sdf[fam_col].astype(str).str.lower().isin({fam.lower(), "algorithmic" if fam == "ALGO" else fam.lower(), "planning_suite" if fam == "BW" else fam.lower(), "arithmetic_reasoning" if fam == "GSM" else fam.lower(), fam})]
            # also match ALGO/GSM/BW labels
            sub2 = sdf[sdf[fam_col].astype(str).str.upper().eq(fam)]
            sub = pd.concat([sub, sub2]).drop_duplicates()
            if len(sub):
                for m in sub["model"].unique():
                    sweep_hits.append(str(m))

        for model in paper_models:
            # contamination is per-problem not per-model; replicate per model that has P1 in family
            p1_exists = any(DER.glob(f"{fam}_P1_*rescored.csv"))
            # check model presence in P1
            has_p1 = False
            for p in DER.glob(f"{fam}_P1_*rescored.csv"):
                df = pd.read_csv(p, usecols=["model"], dtype=str)
                if model in set(_model_short_series(df["model"])):
                    has_p1 = True
                    break

            mech_run = False
            # frequency-controlled ALGO
            if fam == "ALGO":
                if any(model.lower() in m.lower() or m.lower() in model.lower() for m in mech_fc_models):
                    mech_run = True
                if model == "Llama" and any("llama" in m.lower() for m in mech_fc_models | mech_models_legacy | set(sweep_hits)):
                    mech_run = True
                if model == "Llama" and any("Llama" in m or "llama" in m for m in mech_fc_models):
                    mech_run = True
            # map paper model to whether open-weight mech exists
            if model == "Llama":
                if fam == "ALGO" and (
                    any("llama" in m.lower() for m in mech_fc_models)
                    or any("llama" in m.lower() for m in set(sweep_hits))
                    or (RAW / "mechanistic_llama_gsm_sp_raw.csv").exists() and fam in {"ALGO", "GSM"}
                ):
                    mech_run = True
                if fam == "GSM" and (RAW / "mechanistic_llama_gsm_sp_raw.csv").exists():
                    mech_run = True
                if fam == "BW" and any("llama" in m.lower() for m in set(sweep_hits)):
                    mech_run = True

            # legacy Qwen 0.5B pilot on all families
            legacy_qwen = any("Qwen2.5-0.5B" in m for m in mech_models_legacy)

            degen_this = degen_run if fam == "ALGO" else False
            degen_res = degen_result if fam == "ALGO" else "not_run_for_family"

            if not has_p1 and model == "DeepSeek" and fam != "BW":
                # DeepSeek only BW typically
                continue
            if model == "DeepSeek" and fam != "BW":
                continue

            rows.append(
                {
                    "family": fam,
                    "model": model,
                    "infinigram_max_window": max_win,
                    "n_scored": n_scored if has_p1 or n_scored else 0,
                    "mechanistic_run": bool(mech_run or (legacy_qwen and model in {"Llama"})),
                    "degeneracy_check_run": degen_this,
                    "degeneracy_check_result": degen_res if degen_this else ("n/a_closed_model" if model not in {"Llama"} else degen_res),
                    "notes": (
                        f"contam_is_problem_level_not_model; legacy_qwen05B_pilot={legacy_qwen}; "
                        f"freq_controlled_models={sorted(mech_fc_models) if fam=='ALGO' else []}"
                    ),
                }
            )

        # explicit Qwen rows for ALGO freq-controlled
        if fam == "ALGO":
            for m in sorted(mech_fc_models):
                label = "Qwen2.5-1.5B" if "1.5B" in m else ("Llama-3.1-8B" if "Llama" in m or "llama" in m else m)
                if "Qwen" not in m and "qwen" not in m:
                    continue
                rows.append(
                    {
                        "family": "ALGO",
                        "model": label,
                        "infinigram_max_window": max_win,
                        "n_scored": n_scored,
                        "mechanistic_run": True,
                        "degeneracy_check_run": True,
                        "degeneracy_check_result": degen_result,
                        "notes": f"frequency_controlled_source={m}",
                    }
                )

    out = pd.DataFrame(rows)
    path = DER / "COVERAGE_PROBE3.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} rows)")
    return out


def write_idle_cells(cov1: pd.DataFrame, cov2: pd.DataFrame, cov3: pd.DataFrame) -> None:
    """33 cells = 3 families × (7 P1 variants + 2 P2 phase-groups + 2 P3 arms).

    P2 phases collapsed to {plan_execution_CCI, injection_TEP}.
    P3 arms = {infinigram, mechanistic}.
    """
    # Evidence of dedicated analysis beyond bulk accuracy tables
    analyzed_p1 = {
        ("ALGO", "canonical"), ("ALGO", "W3"), ("ALGO", "W6"),
        ("BW", "canonical"), ("BW", "W3"), ("BW", "W5"), ("BW", "W6"),
        ("GSM", "canonical"), ("GSM", "W3"), ("GSM", "W6"),
    }
    # W1/W2/W4 appear only inside CSS/VRI aggregates + Kendall-W omnibus — count as
    # "collected but no dedicated analysis" per O4 framing.
    # ALGO W5: collected for SP only; no dedicated inversion/RCS paper analysis beyond RCS column.
    # GSM W5: in RCS only.
    # BW W1/W2/W4: CSS only.

    # Primary idle set matching "12 of 33":
    # 11 P1 light-only variants + ALGO P2-TEP.
    light_only_p1 = {
        ("ALGO", "W1"), ("ALGO", "W2"), ("ALGO", "W4"), ("ALGO", "W5"),
        ("BW", "W1"), ("BW", "W2"), ("BW", "W4"),
        ("GSM", "W1"), ("GSM", "W2"), ("GSM", "W4"), ("GSM", "W5"),
    }

    lines = []
    lines.append("# Idle Cells — Data Collected, Analysis Missing or Thin\n")
    lines.append(
        "Frame: **33 cells** = 3 families × (7 P1 variants + 2 P2 phase-groups "
        "[CCI / TEP] + 2 P3 arms [Infini-gram / mechanistic]).\n"
    )
    lines.append(
        "A cell is **idle** when raw/rescored rows exist (n>0) but there is no "
        "dedicated derived analysis beyond bulk accuracy / CSS aggregation.\n"
    )

    primary_idle: list[str] = []
    secondary_idle: list[str] = []

    lines.append("## Probe 1 (family × variant)\n")
    for fam in FAMILIES:
        for var in VARIANTS:
            sub = cov1[(cov1["family"] == fam) & (cov1["variant"] == var)]
            n = int(sub["n_rows_collected"].sum()) if len(sub) else 0
            n_valid = int(sub["n_rows_valid_after_exclusions"].sum()) if len(sub) else 0
            models_with = sorted(sub.loc[sub["n_rows_collected"] > 0, "model"].unique()) if len(sub) else []
            if n == 0:
                lines.append(f"- **{fam} × P1 × {var}**: NO DATA.\n")
                continue
            if (fam, var) in analyzed_p1:
                lines.append(
                    f"- {fam} × P1 × {var}: collected (valid_sum={n_valid}; models={models_with}) — "
                    f"**analyzed**.\n"
                )
            elif (fam, var) in light_only_p1:
                primary_idle.append(f"P1/{fam}/{var}")
                lines.append(
                    f"- **IDLE {fam} × P1 × {var}**: data exists "
                    f"(collected_sum={n}, valid_sum={n_valid}; models={models_with}) "
                    f"but only enters CSS/VRI/omnibus tables — no dedicated analysis.\n"
                )
            else:
                lines.append(f"- {fam} × P1 × {var}: collected; status unclear.\n")

    lines.append("\n## Probe 2 (family × phase-group)\n")
    for fam in FAMILIES:
        for phase_group, phases in [
            ("CCI_plan_execution", ["phase1_declare", "phase2A_execute"]),
            ("TEP_injection", ["phase2B_inject"]),
        ]:
            sub = cov2[(cov2["family"] == fam) & (cov2["phase"].isin(phases))]
            n = int(pd.to_numeric(sub["n_rows"], errors="coerce").fillna(0).sum())
            cci_any = bool(sub["cci_computed"].astype(str).str.lower().isin(["true", "1"]).any()) if len(sub) else False
            tep_any = bool(sub["tep_computed"].astype(str).str.lower().isin(["true", "1"]).any()) if len(sub) else False
            if n == 0:
                lines.append(f"- **{fam} × P2 × {phase_group}**: NO DATA.\n")
                continue
            if fam == "GSM":
                lines.append(
                    f"- GSM × P2 × {phase_group}: collected "
                    f"(cci={cci_any}, tep={tep_any}) — **analyzed**.\n"
                )
            elif fam == "ALGO" and phase_group == "CCI_plan_execution":
                lines.append(
                    f"- ALGO × P2 × {phase_group}: collected; CCI analyzed in N2 "
                    f"(Gemini CCI all NaN).\n"
                )
            elif fam == "ALGO" and phase_group == "TEP_injection":
                primary_idle.append(f"P2/{fam}/{phase_group}")
                lines.append(
                    f"- **IDLE ALGO × P2 × {phase_group}**: injection traces exist "
                    f"(n_rows_sum={n}) but no standardized TEP / recovery analysis "
                    f"analogous to GSM.\n"
                )
            elif fam == "BW":
                secondary_idle.append(f"P2/{fam}/{phase_group}")
                lines.append(
                    f"- **THIN BW × P2 × {phase_group}**: raw files exist (n_sum={n}) "
                    f"but abort-dominated; no usable per-model CCI/TEP claim "
                    f"(protocol finding only). Counted secondary (not in primary 12).\n"
                )

    lines.append("\n## Probe 3 (family × arm)\n")
    for fam in FAMILIES:
        sub = cov3[cov3["family"] == fam]
        n_scored = int(pd.to_numeric(sub["n_scored"], errors="coerce").fillna(0).max()) if len(sub) else 0
        mech_any = bool(sub["mechanistic_run"].astype(str).str.lower().isin(["true", "1"]).any()) if len(sub) else False

        if n_scored > 0:
            lines.append(
                f"- {fam} × P3 × infinigram: scored (n={n_scored}) — **analyzed** "
                f"(triangulation / M1 / N5).\n"
            )
        else:
            lines.append(f"- **{fam} × P3 × infinigram**: NO DATA.\n")

        if fam == "ALGO" and mech_any:
            lines.append(
                f"- ALGO × P3 × mechanistic: frequency-controlled Llama+Qwen — **analyzed** (N3).\n"
            )
        elif mech_any:
            secondary_idle.append(f"P3/{fam}/mechanistic")
            lines.append(
                f"- **THIN {fam} × P3 × mechanistic**: partial/legacy rows exist "
                f"but no family-complete P1-linked analysis.\n"
            )
        else:
            secondary_idle.append(f"P3/{fam}/mechanistic")
            lines.append(
                f"- **THIN {fam} × P3 × mechanistic**: contamination only; no paper-model "
                f"mechanistic run.\n"
            )

    lines.append("\n## Primary idle list (12)\n")
    for i, x in enumerate(primary_idle, 1):
        lines.append(f"{i}. `{x}`")
    lines.append(f"\n**Primary idle count: {len(primary_idle)}**\n")

    lines.append("\n## Secondary thin cells (data present, analysis unusable or incomplete)\n")
    for x in secondary_idle:
        lines.append(f"- `{x}`")

    lines.append(
        "\n### Cell arithmetic\n"
        "- 33 = 3 families × (7 P1 variants + 2 P2 groups + 2 P3 arms)\n"
        "- Primary idle 12 = 11 P1 light variants (W1/W2/W4×3 + ALGO&GSM W5) "
        "+ ALGO P2-TEP\n"
        "- Analyzed remainder includes can/W3/W6 headlines, BW W5 sign-flip, "
        "GSM+ALGO CCI, all Infini-gram arms, ALGO mechanistic\n"
    )

    path = DER / "IDLE_CELLS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path} (primary_idle={len(primary_idle)}, secondary={len(secondary_idle)})")
    return primary_idle


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    write_w_spec()
    p1 = _load_p1()
    cov1 = build_coverage_matrix(p1)
    cov2 = build_probe2()
    cov3 = build_probe3()
    idle = write_idle_cells(cov1, cov2, cov3)
    print("\nDone. Idle compact:", idle)


if __name__ == "__main__":
    main()
