#!/usr/bin/env python3
"""Trace Probe 2 phase 2A (uninjected) vs 2B (injected) without calling APIs.

Operational labels follow scripts/runs/coverage_audit.py:
  2A = uninjected stepwise execution (CCI / normal)
  2B = injected stepwise execution (TEP / injected)

Does not write results/raw/.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW = REPO_ROOT / "results/raw"
DATA = REPO_ROOT / "data/problems"
DERIVED = REPO_ROOT / "results/derived"

COUNTS_OUT = DERIVED / "P2_phase2A_2B_trace.csv"
PROTO_OUT = DERIVED / "P2_phase2A_2B_protocol.csv"


def _bank_crit(params: str):
    raw = str(params or "").strip()
    if not raw or raw in ("{}", "null", "nan"):
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "critical_step_index" in obj:
        try:
            return int(obj["critical_step_index"])
        except (TypeError, ValueError):
            return None
    return None


def _source_has_history(path: Path, class_name: str) -> str:
    """Static check: does complete() send more than a single user message?"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            src = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
            if "conversation" in src.lower() or "history" in src.lower():
                # ModelClient documents "No conversation history ever accumulates."
                if "no conversation history" in src.lower():
                    return "fresh_single_user_message"
            if '"messages": [{"role": "user"' in src.replace("'", '"') or (
                "role" in src and "user" in src and "messages" in src
            ):
                return "fresh_single_user_message"
            return "inspect_manually"
    return "class_not_found"


def _count_file(path: Path, session_keys: list[str]) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str).fillna("")
    if "model" not in df.columns:
        return [
            {
                "source_file": str(path.relative_to(REPO_ROOT)),
                "model": "(no model column)",
                "n_rows": len(df),
                "n_sessions": "",
            }
        ]
    out = []
    for model, sub in df.groupby(df["model"].astype(str), dropna=False):
        keys = [k for k in session_keys if k in sub.columns]
        n_sess = sub.groupby(keys).ngroups if keys else len(sub)
        out.append(
            {
                "source_file": str(path.relative_to(REPO_ROOT)),
                "model": str(model),
                "n_rows": int(len(sub)),
                "n_sessions": int(n_sess),
            }
        )
    return out


def _algo_injection_check(path: Path) -> dict:
    if not path.exists():
        return {"checked": False}
    df = pd.read_csv(path, dtype=str).fillna("")
    if not {"step_index", "critical_step_index", "injection_applied"}.issubset(df.columns):
        return {"checked": False, "reason": "missing columns"}
    step = pd.to_numeric(df["step_index"], errors="coerce")
    crit = pd.to_numeric(df["critical_step_index"], errors="coerce")
    applied = df["injection_applied"].astype(str).str.lower().eq("true")
    should = step == crit
    mismatch = int((applied != should).sum())
    return {
        "checked": True,
        "n_rows": int(len(df)),
        "n_injection_applied": int(applied.sum()),
        "n_step_equals_critical": int(should.sum()),
        "n_mismatch": mismatch,
        "lands_at_critical_step_index": mismatch == 0,
    }


def _bw_tep_vs_bank() -> dict:
    tep_path = RAW / "BW_P2_tep.csv"
    bank_path = DATA / "question_bank_bw.csv"
    if not tep_path.exists():
        return {"checked": False}
    tep = pd.read_csv(tep_path, dtype=str).fillna("")
    bank = pd.read_csv(bank_path, dtype=str).fillna("")
    can = bank[bank["variant_type"].astype(str).str.strip().str.lower() == "canonical"].copy()
    can["bank_crit"] = can["difficulty_params"].map(_bank_crit)
    m = tep.merge(can[["problem_id", "bank_crit"]], on="problem_id", how="left")
    inj = pd.to_numeric(m["inject_at_step"], errors="coerce")
    bc = pd.to_numeric(m["bank_crit"], errors="coerce")
    both = inj.notna() & bc.notna()
    match = int(((inj == bc) & both).sum())
    n_both = int(both.sum())
    n_sess = m.groupby(["problem_id", "model"]).ngroups
    n_sites = m.groupby(["problem_id", "model"]).size()
    return {
        "checked": True,
        "n_tep_rows": int(len(m)),
        "n_sessions": int(n_sess),
        "mean_inject_sites_per_session": round(float(n_sites.mean()), 4) if len(n_sites) else None,
        "n_rows_with_bank_crit": n_both,
        "n_inject_equals_bank_crit": match,
        "lands_at_bank_critical_step_index": False,
        "note": "TEP uses injection_schedule(plan_length), not bank critical_step_index",
    }


def _gsm_inject_vs_bank() -> dict:
    path = RAW / "GSM_P2_cci.csv"
    bank = pd.read_csv(DATA / "question_bank_gsm.csv", dtype=str).fillna("")
    can = bank[bank["variant_type"].astype(str).str.strip().str.lower() == "canonical"].copy()
    can["bank_crit"] = can["difficulty_params"].map(_bank_crit)
    n_with_crit = int(can["bank_crit"].notna().sum())
    gsm = pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()
    cols = list(gsm.columns) if len(gsm) else []
    return {
        "checked": True,
        "n_canonical_with_bank_crit": n_with_crit,
        "raw_has_phase1_steps_json": "phase1_steps_json" in cols,
        "raw_has_inject_at_step": "inject_at_step" in cols,
        "n_raw_rows": int(len(gsm)),
        "note": (
            "GSM bank has no critical_step_index; runner falls back to "
            "max(1, floor(n_phase1/2)). On-disk GSM_P2_cci.csv is one row "
            "per (problem, model) with both cci_score (2A) and tep_score (2B); "
            "it does not store per-step traces, so landing cannot be rechecked "
            "from the file."
        ),
    }


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)

    model_client = _source_has_history(
        REPO_ROOT / "probes/behavioral/model_client.py", "ModelClient"
    )
    or_client = _source_has_history(
        REPO_ROOT / "probes/behavioral/openai_client.py", "OpenRouterClient"
    )

    algo_inj = _algo_injection_check(RAW / "ALGO_P2_phase2_injected.csv")
    algo_impl = _algo_injection_check(RAW / "ALGO_P2_phase2_injected_implausible.csv")
    bw_tep = _bw_tep_vs_bank()
    gsm_inj = _gsm_inject_vs_bank()

    protocol = [
        {
            "family": "ALGO",
            "phase": "2A",
            "label": "uninjected stepwise (CCI / normal)",
            "script": "scripts/ALGO_P2_SCR_run_phase2.py --condition normal",
            "writes": "results/raw/ALGO_P2_phase2_normal.csv; also ALGO_P2_phase2_normal_elicited.csv (--elicit-algorithm) and ALGO_P2_phase2_normal_gemini.csv (gemini-only run, rows also present in the main normal file)",
            "distinguishes_from_other_phase": "No injection. Full canonical pool (110). Each step is client.complete() with the true state in the prompt.",
            "injection_lands_at_critical_step_index": "n/a (no injection)",
            "sessions_separate_api_calls": "yes",
            "shared_api_context": "no",
            "client": f"OpenRouterClient ({or_client})",
            "notes": "coverage_audit maps this file to P2A_normal. Docs (PROBE_PIPELINE_REFERENCE) instead call ALGO_P2_SCR_run_phase1.py 'Phase 2A'; that is strategy declaration, not this file.",
        },
        {
            "family": "ALGO",
            "phase": "2B",
            "label": "injected stepwise (TEP / plausible or implausible)",
            "script": "scripts/ALGO_P2_SCR_run_phase2.py --condition injected [--injection-mode implausible]",
            "writes": "results/raw/ALGO_P2_phase2_injected.csv (plausible); ALGO_P2_phase2_injected_implausible.csv; ALGO_P2_phase2_injected_gemini.csv (subset also in the main injected file)",
            "distinguishes_from_other_phase": "Separate process invocation, separate output file. Injected session only runs instance_type==adversarial (61). At step==critical_step_index the prompt carries injected_state. diverged_from_normal is step>=critical_step (index gate, not a content comparison).",
            "injection_lands_at_critical_step_index": f"yes (plausible mismatch={algo_inj.get('n_mismatch')}; implausible mismatch={algo_impl.get('n_mismatch')})",
            "sessions_separate_api_calls": "yes",
            "shared_api_context": "no",
            "client": f"OpenRouterClient ({or_client})",
            "notes": "2A and 2B are not the same Python process. Each complete() sends messages=[{role:user, content:prompt}] with no chat history.",
        },
        {
            "family": "GSM",
            "phase": "2A",
            "label": "uninjected stepwise (CCI)",
            "script": "scripts/GSM_P2_SCR_run_probe2.py (_run_phase2a without inject_at_step)",
            "writes": "results/raw/GSM_P2_cci.csv (cci_score / cci_matched / cci_total columns). Phase-1 plans also live in GSM_P2_phase1_{claude,gpt4o,llama,gemini,o1mini}.csv",
            "distinguishes_from_other_phase": "Same script, same output row as 2B. First _run_phase2a loop has inject_at_step=None. CCI compares phase1 plan values to this loop.",
            "injection_lands_at_critical_step_index": "n/a (no injection)",
            "sessions_separate_api_calls": "yes",
            "shared_api_context": "no",
            "client": f"OpenRouterClient or AnthropicClient (fresh single-user message); {or_client}",
            "notes": gsm_inj["note"],
        },
        {
            "family": "GSM",
            "phase": "2B",
            "label": "injected stepwise (TEP)",
            "script": "scripts/GSM_P2_SCR_run_probe2.py (second _run_phase2a with inject_at_step)",
            "writes": "results/raw/GSM_P2_cci.csv (tep_score / inject_at_step / injected_value / session_b_correct). Not a separate file.",
            "distinguishes_from_other_phase": "Second loop in the same process. Injection is written into the NEXT-step prompt (k == inject_at_step + 1 overwrites the prior displayed value). Session B starts prior=[] — it does not inherit session A values. TEP compares post-injection values across the two loops.",
            "injection_lands_at_critical_step_index": (
                "bank has 0 canonical rows with critical_step_index; "
                "runner uses max(1, floor(n_phase1/2)). On-disk file cannot confirm landing."
            ),
            "sessions_separate_api_calls": "yes",
            "shared_api_context": "no — but session_b_correct is computed from phase2a_values (session A) plus phase1_final, not session B",
            "client": f"OpenRouterClient or AnthropicClient ({or_client})",
            "notes": "BUG: session_b_correct uses phase2a_values[-1] (and/or phase1_final), never phase2b_values. 2A and 2B scores share one CSV row.",
        },
        {
            "family": "BW",
            "phase": "2A",
            "label": "uninjected stepwise (CCI)",
            "script": "scripts/BW_P2_SCR_run_cci.py; NL parallel scripts/BW_P2_SCR_run_cci_nl.py and scripts/MBW_P2_SCR_run_cci_nl.py",
            "writes": "results/raw/BW_P2_cci.csv; BW_P2_cci_nl.csv; MBW_P2_cci_nl.csv. Plans: BW_P2_plans.csv via BW_P2_SCR_extract_phase1_plans.py",
            "distinguishes_from_other_phase": "No displayed-state injection. Distinct script and file from TEP.",
            "injection_lands_at_critical_step_index": "n/a (no injection)",
            "sessions_separate_api_calls": "yes",
            "shared_api_context": "no",
            "client": f"ModelClient ({model_client})",
            "notes": "Each step rebuilds the prompt from the current true state. ModelClient.complete sends one user message and keeps no history.",
        },
        {
            "family": "BW",
            "phase": "2B",
            "label": "injected stepwise (TEP)",
            "script": "scripts/BW_P2_SCR_run_tep.py",
            "writes": "results/raw/BW_P2_tep.csv",
            "distinguishes_from_other_phase": "Distinct script. Injection is a false displayed_state at inject_at_step via seeded_inject_error. Multiple inject sites per (problem, model) from injection_schedule(plan_length) — typically up to 4, not a single bank critical_step_index.",
            "injection_lands_at_critical_step_index": (
                f"no — {bw_tep.get('n_inject_equals_bank_crit')}/{bw_tep.get('n_rows_with_bank_crit')} "
                f"rows equal bank critical_step_index; mean sites/session={bw_tep.get('mean_inject_sites_per_session')}"
            ),
            "sessions_separate_api_calls": "yes",
            "shared_api_context": "no",
            "client": f"ModelClient ({model_client})",
            "notes": "CCI and TEP are separate process invocations. TEP injection is prompt-narrative only; true_state is tracked locally for classification.",
        },
    ]

    proto_fields = list(protocol[0].keys())
    with PROTO_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=proto_fields)
        w.writeheader()
        w.writerows(protocol)

    count_specs = [
        ("ALGO", "2A", "normal", RAW / "ALGO_P2_phase2_normal.csv", ["problem_id", "model"]),
        ("ALGO", "2A", "normal_elicited", RAW / "ALGO_P2_phase2_normal_elicited.csv", ["problem_id", "model"]),
        ("ALGO", "2A", "normal_gemini_sidecar", RAW / "ALGO_P2_phase2_normal_gemini.csv", ["problem_id", "model"]),
        ("ALGO", "2B", "injected_plausible", RAW / "ALGO_P2_phase2_injected.csv", ["problem_id", "model"]),
        ("ALGO", "2B", "injected_implausible", RAW / "ALGO_P2_phase2_injected_implausible.csv", ["problem_id", "model"]),
        ("ALGO", "2B", "injected_gemini_sidecar", RAW / "ALGO_P2_phase2_injected_gemini.csv", ["problem_id", "model"]),
        ("GSM", "2A", "cci_columns_in_shared_file", RAW / "GSM_P2_cci.csv", ["problem_id", "model"]),
        ("GSM", "2B", "tep_columns_in_shared_file", RAW / "GSM_P2_cci.csv", ["problem_id", "model"]),
        ("BW", "2A", "cci_strict", RAW / "BW_P2_cci.csv", ["problem_id", "model"]),
        ("BW", "2A", "cci_nl", RAW / "BW_P2_cci_nl.csv", ["problem_id", "model"]),
        ("BW", "2A", "mbw_cci_nl", RAW / "MBW_P2_cci_nl.csv", ["problem_id", "model"]),
        ("BW", "2B", "tep", RAW / "BW_P2_tep.csv", ["problem_id", "model", "inject_at_step"]),
    ]

    count_rows = []
    for family, phase, track, path, keys in count_specs:
        for rec in _count_file(path, keys):
            rec["family"] = family
            rec["phase"] = phase
            rec["track"] = track
            count_rows.append(rec)

    # Order columns
    fields = ["family", "phase", "track", "source_file", "model", "n_rows", "n_sessions"]
    with COUNTS_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(count_rows)

    print(f"Wrote {PROTO_OUT}")
    print(f"Wrote {COUNTS_OUT} ({len(count_rows)} rows)")
    print("\nRow counts per family × phase × model:")
    df = pd.DataFrame(count_rows)
    print(
        df.groupby(["family", "phase", "track", "model"], dropna=False)[["n_rows", "n_sessions"]]
        .sum()
        .to_string()
    )
    print("\nALGO injection check (plausible):", algo_inj)
    print("ALGO injection check (implausible):", algo_impl)
    print("BW TEP vs bank critical_step:", bw_tep)
    print("GSM inject vs bank:", gsm_inj)


if __name__ == "__main__":
    main()
