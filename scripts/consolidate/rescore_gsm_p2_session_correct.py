#!/usr/bin/env python3
"""H1: Rename the GSM P2 accuracy disjunction and recover what is stored.

Does not write results/raw/. Does not call any model API.

On-disk GSM P2 files never persisted phase2a_values or phase2b_values.
phase1_final_answer is in GSM_P2_phase1_*.csv only.
  either_session_correct  = the old session_b_correct column (phase2a OR phase1)
  phase1_correct          = verify(phase1_final_answer)
  phase2a_correct         = unrecoverable (not stored)
  phase2b_correct         = unrecoverable (not stored)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify import verify_gsm_answer  # noqa: E402

RAW = REPO_ROOT / "results/raw"
DERIVED = REPO_ROOT / "results/derived"
OUT = DERIVED / "GSM_P2_session_correct.csv"
REPORT = DERIVED / "H1_gsm_p2_acc_before_after.csv"
CORR = DERIVED / "H1_p2acc_vs_contamination.csv"

PHASE1_FILES = [
    RAW / "GSM_P2_phase1_claude.csv",
    RAW / "GSM_P2_phase1_gpt4o.csv",
    RAW / "GSM_P2_phase1_llama.csv",
    RAW / "GSM_P2_phase1_gemini.csv",
    RAW / "GSM_P2_phase1_o1mini.csv",
]
CCI = RAW / "GSM_P2_cci.csv"
CONTAM = RAW / "GSM_P3_contamination.csv"

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _either_col(df: pd.DataFrame) -> pd.Series:
    if "either_session_correct" in df.columns:
        return df["either_session_correct"]
    if "session_b_correct" in df.columns:
        return df["session_b_correct"]
    return pd.Series([""] * len(df))


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in PHASE1_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df["_source"] = path.name
        frames.append(df)
    cci = pd.read_csv(CCI, dtype=str).fillna("") if CCI.exists() else pd.DataFrame()
    if not cci.empty:
        cci["_source"] = "GSM_P2_cci.csv"

    phase1 = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Prefer phase1 files (they have phase1_final_answer). Add cci-only rows.
    if not phase1.empty:
        keys = set(zip(phase1["problem_id"].astype(str), phase1["model"].astype(str)))
    else:
        keys = set()
    extra = []
    if not cci.empty:
        for _, r in cci.iterrows():
            k = (str(r["problem_id"]), str(r["model"]))
            if k not in keys:
                extra.append(r)
    if extra:
        phase1 = pd.concat([phase1, pd.DataFrame(extra)], ignore_index=True)

    rows = []
    for _, r in phase1.iterrows():
        gold = str(r.get("correct_answer", ""))
        p1_final = str(r.get("phase1_final_answer", "")).strip()
        has_p1_final = "phase1_final_answer" in r.index and p1_final not in ("", "nan")
        phase1_correct = ""
        if has_p1_final:
            phase1_correct = str(bool(verify_gsm_answer(p1_final, gold)))
        either = _either_col(pd.DataFrame([r])).iloc[0]
        rows.append(
            {
                "problem_id": str(r["problem_id"]),
                "model": str(r["model"]),
                "source_file": str(r.get("_source", "")),
                "either_session_correct": str(bool(_is_true(pd.Series([either])).iloc[0]))
                if str(either).strip()
                else "",
                "phase1_correct": phase1_correct,
                "phase2a_correct": "",
                "phase2b_correct": "",
                "phase2a_values_persisted": "false",
                "phase2b_values_persisted": "false",
                "correct_answer": gold,
                "phase1_final_answer": p1_final,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)

    # Table 4 Acc_P2A before = disjunction treated as Acc_P2A.
    # After = phase2a_correct, which is not recoverable.
    report = []
    for model, sub in out.groupby("model"):
        n = len(sub)
        before = float(_is_true(sub["either_session_correct"]).mean()) if n else float("nan")
        n_p1 = int(sub["phase1_correct"].astype(str).str.strip().ne("").sum())
        p1_rate = (
            float(_is_true(sub["phase1_correct"]).mean()) if n_p1 else float("nan")
        )
        report.append(
            {
                "model": SHORT.get(str(model), str(model)),
                "model_full": str(model),
                "n": n,
                "acc_p2a_before_disjunction": round(before, 6),
                "acc_p2a_after_phase2a": "",
                "phase2a_recoverable": "false",
                "phase1_correct_rate": "" if pd.isna(p1_rate) else round(p1_rate, 6),
                "n_phase1_scored": n_p1,
                "note": "phase2a_values and phase2b_values were never written to CSV; Acc_P2A cannot be recovered without a re-run",
            }
        )
    pd.DataFrame(report).to_csv(REPORT, index=False)

    # Spearman P2.acc (disjunction) vs contamination — the number that used to
    # be reported. After: undefined because true P2.acc (phase2a) is missing.
    contam = pd.read_csv(CONTAM, dtype=str).fillna("") if CONTAM.exists() else pd.DataFrame()
    if not contam.empty and "contamination_score" in contam.columns:
        contam["contamination_score"] = pd.to_numeric(
            contam["contamination_score"], errors="coerce"
        )
        cmap = dict(
            zip(contam["problem_id"].astype(str), contam["contamination_score"])
        )
    else:
        cmap = {}
    corr_rows = []
    for model, sub in out.groupby("model"):
        acc = _is_true(sub["either_session_correct"]).astype(float)
        c = sub["problem_id"].map(cmap)
        pair = pd.DataFrame({"acc": acc, "contam": c}).dropna()
        if len(pair) >= 4 and pair["acc"].nunique() > 1 and pair["contam"].nunique() > 1:
            rho = float(pair["acc"].corr(pair["contam"], method="spearman"))
        else:
            rho = float("nan")
        corr_rows.append(
            {
                "model": SHORT.get(str(model), str(model)),
                "n": int(len(pair)),
                "spearman_p2acc_disjunction_vs_contam_before": rho,
                "spearman_p2acc_phase2a_vs_contam_after": "",
                "delta": "",
                "note": "after is blank: phase2a_correct not stored; correlation cannot move",
            }
        )
    pd.DataFrame(corr_rows).to_csv(CORR, index=False)

    print(f"Wrote {OUT} ({len(out)} rows)")
    print(f"Wrote {REPORT}")
    print(f"Wrote {CORR}")
    print("phase2a_values persisted: no")
    print("phase2b_values persisted: no — phase2b_correct requires a re-run")
    print(pd.DataFrame(report).to_string(index=False))
    print(pd.DataFrame(corr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
