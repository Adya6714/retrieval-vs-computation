#!/usr/bin/env python3
"""Run GSM Probe 2: structured planning + stepwise execution (CCI / TEP)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except (TimeoutError, OSError):
    pass

QUESTION_BANK_PATH = Path("data/problems/question_bank_gsm.csv")
RESULTS_PATH = Path("results/raw/GSM_P2_cci.csv")

STEP_LINE_RE = re.compile(
    r"Step\s+(\d+):\s*(.+?)\s*=\s*([\d\.\-]+)",
    flags=re.IGNORECASE,
)
FINAL_ANSWER_RE = re.compile(
    r"Final\s+answer\s*:\s*([\d\.\-]+)",
    flags=re.IGNORECASE,
)
STEP_VALUE_RE = re.compile(
    r"Step\s+(\d+):\s*.+?=\s*([\d\.\-]+)",
    flags=re.IGNORECASE,
)

OUTPUT_COLUMNS = [
    "problem_id",
    "model",
    "contamination_pole",
    "difficulty",
    "phase1_steps_json",
    "phase1_final_answer",
    "phase1_parseable",
    "cci_score",
    "cci_matched",
    "cci_total",
    "inject_at_step",
    "injected_value",
    "true_value_at_injection",
    "tep_score",
    "tep_diverged_steps",
    "tep_total_steps",
    "either_session_correct",
    "phase1_correct",
    "phase2a_correct",
    "phase2b_correct",
    "phase2a_values_json",
    "phase2b_values_json",
    "correct_answer",
]


def _choose_client(model: str, *, dry_run: bool):
    from probes.behavioral.mock_client import MockClient

    if dry_run:
        return MockClient(
            default_response=(
                "Step 1: start = 10\n"
                "Step 2: add = 15\n"
                "Final answer: 15"
            )
        )
    if model.startswith("anthropic/") and os.environ.get("ANTHROPIC_API_KEY"):
        from probes.behavioral.anthropic_client import AnthropicClient

        return AnthropicClient(model=model)
    from probes.behavioral.openai_client import OpenRouterClient

    return OpenRouterClient(model=model)


def _phase1_prompt(problem_text: str) -> str:
    return (
        "Solve the following arithmetic problem by writing out every "
        "computation step explicitly. Each step must show the intermediate "
        "numeric result.\n\n"
        "Format each step exactly as:\n"
        "Step N: [what you are computing] = [numeric result]\n\n"
        "Write all steps then state the final answer as:\n"
        "Final answer: [number]\n\n"
        f"Problem:\n{problem_text}"
    )


def _phase2_prompt(
    problem_text: str,
    *,
    step_k: int,
    prior_steps: list[tuple[int, str, float]],
) -> str:
    if step_k == 1:
        return (
            f"Problem: {problem_text}\n\n"
            "This is step 1. What is the first computation step?\n"
            "Format exactly as: Step 1: [what you are computing] = [numeric result]\n"
            "Give only this one line."
        )

    lines = [f"Step {n}: {desc} = {val}" for n, desc, val in prior_steps]
    prev_n, _prev_desc, prev_val = prior_steps[-1]
    done_block = "\n".join(lines)
    return (
        f"Problem: {problem_text}\n\n"
        "Steps completed so far:\n"
        f"{done_block}\n"
        f"Current value after step {prev_n}: {prev_val}\n\n"
        f"What is step {step_k}?\n"
        f"Format exactly as: Step {step_k}: [what you are computing] = [numeric result]\n"
        "Give only this one line."
    )


def _parse_phase1(raw: str) -> tuple[list[dict], float | None, bool]:
    steps: list[dict] = []
    for m in STEP_LINE_RE.finditer(str(raw)):
        steps.append(
            {
                "step": int(m.group(1)),
                "description": m.group(2).strip(),
                "value": float(m.group(3)),
            }
        )
    steps.sort(key=lambda x: x["step"])

    final_answer: float | None = None
    fm = FINAL_ANSWER_RE.search(str(raw))
    if fm:
        final_answer = float(fm.group(1))

    parseable = bool(steps) and final_answer is not None
    return steps, final_answer, parseable


def _parse_step_value(raw: str, expected_step: int) -> float | None:
    text = str(raw).strip()
    if FINAL_ANSWER_RE.search(text):
        return None
    m = STEP_VALUE_RE.search(text)
    if not m:
        return None
    try:
        if int(m.group(1)) != expected_step:
            return None
        return float(m.group(2))
    except (TypeError, ValueError):
        return None


def _inject_value(true_value: float) -> float:
    injected = true_value * 1.15
    if abs(injected - round(injected)) < 1e-9:
        return float(round(injected))
    return round(injected, 2)


def _critical_step_index(row: pd.Series, phase1_steps: list[dict]) -> int:
    raw_params = str(row.get("difficulty_params", "")).strip()
    if raw_params and raw_params not in ("{}", "null", "nan"):
        try:
            params = json.loads(raw_params)
            if isinstance(params, dict) and "critical_step_index" in params:
                return int(params["critical_step_index"])
        except json.JSONDecodeError:
            pass
    n = len(phase1_steps)
    return max(1, math.floor(n / 2)) if n else 1


def _run_phase2a(
    *,
    client,
    problem_id: str,
    problem_text: str,
    phase1_steps: list[dict],
    max_steps: int,
    inject_at_step: int | None = None,
    injected_value: float | None = None,
) -> list[float | None]:
    values: list[float | None] = []
    prior: list[tuple[int, str, float]] = []
    limit = max(len(phase1_steps) + 2, max_steps)

    for k in range(1, limit + 1):
        if k > len(phase1_steps) + 2:
            break

        if k == 1:
            prompt = _phase2_prompt(problem_text, step_k=1, prior_steps=[])
        else:
            prompt_steps = list(prior)
            if (
                inject_at_step is not None
                and injected_value is not None
                and k == inject_at_step + 1
                and prompt_steps
            ):
                pn, pd, _pv = prompt_steps[-1]
                prompt_steps[-1] = (pn, pd, injected_value)
            prompt = _phase2_prompt(
                problem_text, step_k=k, prior_steps=prompt_steps
            )

        raw = str(client.complete(problem_id, prompt).get("response", "")).strip()
        if FINAL_ANSWER_RE.search(raw):
            break

        val = _parse_step_value(raw, k)
        values.append(val)
        if val is None:
            break

        desc = next(
            (s["description"] for s in phase1_steps if s["step"] == k),
            f"step {k}",
        )
        prior.append((k, desc, val))

        if phase1_steps and k >= len(phase1_steps):
            break

    return values


def _values_close(a: float | None, b: float | None, tol: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _compute_cci(
    phase1_steps: list[dict],
    phase2a_values: list[float | None],
) -> tuple[float, int, int]:
    if not phase1_steps:
        return 0.0, 0, 0
    matched = 0
    total = len(phase1_steps)
    for i, step in enumerate(phase1_steps):
        p1_val = float(step["value"])
        p2_val = phase2a_values[i] if i < len(phase2a_values) else None
        if _values_close(p1_val, p2_val):
            matched += 1
    score = matched / total if total else 0.0
    return score, matched, total


def _compute_tep(
    phase2a_values: list[float | None],
    phase2b_values: list[float | None],
    inject_at_step: int,
) -> tuple[float, int, int]:
    start = inject_at_step
    a_post = phase2a_values[start:] if start < len(phase2a_values) else []
    b_post = phase2b_values[start:] if start < len(phase2b_values) else []
    total = min(len(a_post), len(b_post))
    if total == 0:
        return 0.0, 0, 0
    diverged = sum(
        1 for i in range(total) if not _values_close(a_post[i], b_post[i])
    )
    return diverged / total, diverged, total


def _existing_keys(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(output_path, dtype=str).fillna("")
    if not {"problem_id", "model"}.issubset(df.columns):
        return set()
    return {
        (str(r["problem_id"]), str(r["model"])) for _, r in df.iterrows()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM Probe 2 CCI/TEP.")
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument(
        "--question-bank-path",
        type=str,
        default=str(QUESTION_BANK_PATH),
    )
    parser.add_argument("--output", type=str, default=str(RESULTS_PATH))
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip problem_id+model already in output (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Re-run all rows even if already present",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use MockClient instead of live APIs",
    )
    args = parser.parse_args()

    bank_path = Path(args.question_bank_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not bank_path.exists():
        raise FileNotFoundError(f"Question bank not found: {bank_path}")

    df = pd.read_csv(bank_path, dtype=str).fillna("")
    canonical = df[
        (df["variant_type"].astype(str).str.strip().str.lower() == "canonical")
        & (
            df["problem_family"].astype(str).str.strip().str.lower()
            == "arithmetic_reasoning"
        )
    ].copy()

    done = _existing_keys(output_path) if args.resume else set()
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    client = _choose_client(args.model, dry_run=args.dry_run)

    with output_path.open("a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
            out_f.flush()

        for _, row in canonical.iterrows():
            problem_id = str(row["problem_id"])
            model = str(args.model)
            if args.resume and (problem_id, model) in done:
                continue

            problem_text = str(row["problem_text"])
            correct_answer = str(row["correct_answer"])

            p1_raw = str(
                client.complete(problem_id, _phase1_prompt(problem_text)).get(
                    "response", ""
                )
            )
            phase1_steps, phase1_final, phase1_parseable = _parse_phase1(p1_raw)

            inject_at = _critical_step_index(row, phase1_steps)
            inject_idx = min(max(inject_at - 1, 0), max(len(phase1_steps) - 1, 0))
            true_at_injection = (
                float(phase1_steps[inject_idx]["value"]) if phase1_steps else None
            )
            injected_value = (
                _inject_value(true_at_injection)
                if true_at_injection is not None
                else None
            )

            phase2a_values = _run_phase2a(
                client=client,
                problem_id=problem_id,
                problem_text=problem_text,
                phase1_steps=phase1_steps,
                max_steps=args.max_steps,
            )

            phase2b_values = _run_phase2a(
                client=client,
                problem_id=problem_id,
                problem_text=problem_text,
                phase1_steps=phase1_steps,
                max_steps=args.max_steps,
                inject_at_step=inject_at,
                injected_value=injected_value,
            )

            cci_score, cci_matched, cci_total = _compute_cci(
                phase1_steps, phase2a_values
            )
            tep_score, tep_div, tep_total = _compute_tep(
                phase2a_values, phase2b_values, inject_at
            )

            from probes.contamination.verify import verify_gsm_answer

            phase1_correct = bool(
                phase1_final is not None
                and verify_gsm_answer(str(phase1_final), correct_answer)
            )
            phase2a_final = ""
            if phase2a_values and phase2a_values[-1] is not None:
                phase2a_final = f"Final answer: {phase2a_values[-1]}"
            phase2a_correct = bool(
                phase2a_final and verify_gsm_answer(phase2a_final, correct_answer)
            )
            phase2b_final = ""
            if phase2b_values and phase2b_values[-1] is not None:
                phase2b_final = f"Final answer: {phase2b_values[-1]}"
            phase2b_correct = bool(
                phase2b_final and verify_gsm_answer(phase2b_final, correct_answer)
            )
            either_session_correct = bool(phase2a_correct or phase1_correct)

            writer.writerow(
                {
                    "problem_id": problem_id,
                    "model": model,
                    "contamination_pole": str(row.get("contamination_pole", "")),
                    "difficulty": str(row.get("difficulty", "")),
                    "phase1_steps_json": json.dumps(phase1_steps),
                    "phase1_final_answer": (
                        "" if phase1_final is None else str(phase1_final)
                    ),
                    "phase1_parseable": str(bool(phase1_parseable)),
                    "cci_score": f"{cci_score:.4f}",
                    "cci_matched": str(cci_matched),
                    "cci_total": str(cci_total),
                    "inject_at_step": str(inject_at),
                    "injected_value": (
                        "" if injected_value is None else str(injected_value)
                    ),
                    "true_value_at_injection": (
                        "" if true_at_injection is None else str(true_at_injection)
                    ),
                    "tep_score": f"{tep_score:.4f}",
                    "tep_diverged_steps": str(tep_div),
                    "tep_total_steps": str(tep_total),
                    "either_session_correct": str(bool(either_session_correct)),
                    "phase1_correct": str(bool(phase1_correct)),
                    "phase2a_correct": str(bool(phase2a_correct)),
                    "phase2b_correct": str(bool(phase2b_correct)),
                    "phase2a_values_json": json.dumps(phase2a_values),
                    "phase2b_values_json": json.dumps(phase2b_values),
                    "correct_answer": correct_answer,
                }
            )
            out_f.flush()
            done.add((problem_id, model))

    print(f"Done. output={output_path}")


if __name__ == "__main__":
    main()
