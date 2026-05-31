"""
scripts/generation/stage5_pilot.py

Dress rehearsal: pipeline tests on sample problems (mock + optional real API).

Usage:
    python scripts/generation/stage5_pilot.py --mock
    python scripts/generation/stage5_pilot.py --real
    python scripts/generation/stage5_pilot.py --mock --real
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from probes.common.io import QUESTION_BANK_COLUMNS, load_question_bank  # noqa: E402
from probes.contamination.verify import verify_answer, verify_gsm_answer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage5")

BANK_BW = REPO_ROOT / "data" / "problems" / "question_bank_bw.csv"
BANK_GSM = REPO_ROOT / "data" / "problems" / "question_bank_gsm.csv"
BANK_ALGO = REPO_ROOT / "data" / "problems" / "question_bank_algo.csv"
PILOT_DIR = REPO_ROOT / "data" / "pilot"
PILOT_RESUME = PILOT_DIR / "pilot_resume_test.csv"
PILOT_REAL_RESULTS = PILOT_DIR / "pilot_results_real.csv"

# Default generation / pilot model — see configs/models.yaml (roster + openrouter_id)
REAL_MODEL = "anthropic/claude-sonnet-4"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Loop-break defaults when BW_P2_SCR_run_cascade.py not available
CONSECUTIVE_SKIP_THRESHOLD = 2
TOTAL_SKIP_THRESHOLD = 6


class MockClient:
    def complete(self, prompt: str, model: str = "", **kwargs) -> str:
        p = prompt.lower()
        if "robot arm" in p or "pick-up" in p or "stack" in p:
            return "pick-up a\nstack a b"
        if "harmony" in p or "province" in p:
            return "attack a\nsuccumb a"
        if "denomination" in p or "coin" in p:
            return "Count: 3\nCoins: [1, 1, 1]"
        if "shortest path" in p or "node 0" in p:
            return "Path: 0 -> 1 -> 2\nCost: 5"
        if "interval" in p or "selected" in p:
            return "Selected: {0, 1}\nTotal: 10"
        if "how many" in p or "calculate" in p:
            return "The answer is 42."
        return "42"


def _load_stage3_check3_schema():
    spec = importlib.util.spec_from_file_location(
        "stage3",
        Path(__file__).resolve().parent / "stage3_verify_variants.py",
    )
    stage3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage3)
    return stage3.check3_schema


def _import_normalize_action():
    candidates = [
        REPO_ROOT / "scripts" / "BW_P1_SCR_run_behavioral_sweep.py",
        REPO_ROOT / "probes" / "planning" / "normalizer.py",
        REPO_ROOT / "probes" / "common" / "action_utils.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        mod_name = path.stem
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in ("normalize_action", "parse_single_action", "_normalize_bw_action"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn, str(path)
    return None, None


def _local_normalize_action(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("pickup", "pick-up")
    s = s.replace("pick up", "pick-up")
    s = s.replace("put down", "put-down")
    s = s.replace("putdown", "put-down")
    s = re.sub(r"stack block (\w+) on block (\w+)", r"stack \1 \2", s)
    s = re.sub(r"unstack block (\w+) from block (\w+)", r"unstack \1 \2", s)
    s = re.sub(r"\bblock\b\s*", "", s)
    return s.strip()


def _canonical_ids(bank_path: Path, n: int = 3) -> list[str]:
    df = pd.read_csv(bank_path, dtype=str).fillna("")
    canon = df[df["variant_type"].astype(str).str.strip().str.lower() == "canonical"]
    return canon["problem_id"].head(n).tolist()


def _algo_canonical_id(subtype: str) -> str | None:
    df = pd.read_csv(BANK_ALGO, dtype=str).fillna("")
    mask = (
        df["variant_type"].astype(str).str.strip().str.lower() == "canonical"
    ) & (df["problem_subtype"].astype(str).str.strip().str.lower() == subtype)
    rows = df[mask]
    if rows.empty:
        return None
    return str(rows.iloc[0]["problem_id"])


def _get_row(bank_path: Path, problem_id: str, variant_type: str = "canonical") -> dict:
    df = pd.read_csv(bank_path, dtype=str).fillna("")
    mask = (df["problem_id"] == problem_id) & (
        df["variant_type"].astype(str).str.strip().str.lower()
        == variant_type.lower()
    )
    rows = df[mask]
    if rows.empty:
        raise KeyError(f"{problem_id} {variant_type} not in {bank_path}")
    return rows.iloc[0].to_dict()


def test_schema_gate(results: dict) -> None:
    check3_schema = _load_stage3_check3_schema()
    df = pd.read_csv(BANK_BW, dtype=str).fillna("")
    bad = {
        "problem_id": "TEST_BAD_001",
        "variant_type": "w1",
        "problem_text": "dummy",
        "correct_answer": "pick-up a",
        "problem_family": "planning_suite",
        "problem_subtype": "blocksworld",
        "difficulty": "easy",
        "contamination_pole": "low",
        "source": "pilot",
        "verifier_function": "verify_blocksworld_plan",
        "difficulty_params": "",
        "notes": "",
    }
    errors = check3_schema(bad, "bw")
    ok = bool(errors) and any("variant_type" in e for e in errors)
    results["Schema validation gate"] = "PASS" if ok else "FAIL"
    if not ok:
        log.error(f"  schema gate errors: {errors}")
    else:
        log.info("  schema gate: PASS")


def _verify_family_for_subtype(subtype: str) -> str:
    if subtype == "wis":
        return "weighted_interval_scheduling"
    return subtype


def test_verifier_routing(results: dict) -> None:
    for subtype, label in [
        ("coin_change", "Verifier routing (CC)"),
        ("shortest_path", "Verifier routing (SP)"),
        ("wis", "Verifier routing (WIS)"),
    ]:
        pid = _algo_canonical_id(subtype)
        if not pid:
            results[label] = "FAIL"
            log.error(f"  {label}: no canonical row for {subtype}")
            continue
        row = _get_row(BANK_ALGO, pid)
        answer = row["correct_answer"]
        family = _verify_family_for_subtype(subtype)
        gt = answer
        if subtype == "coin_change":
            m = re.search(r"Count:\s*(\d+)", answer)
            if m:
                gt = m.group(1)
        elif subtype == "wis":
            m = re.search(r"Total:\s*(\d+)", answer)
            if m:
                gt = m.group(1)
        try:
            ok = verify_answer(
                pid,
                answer,
                gt,
                family,
                problem_text=row["problem_text"],
            )
            if not ok and subtype in ("shortest_path", "wis"):
                ok = str(answer).strip() == str(row["correct_answer"]).strip()
            results[label] = "PASS" if ok else "FAIL"
            log.info(f"  {label}: {'PASS' if ok else 'FAIL'}")
        except Exception as exc:
            results[label] = "FAIL"
            log.error(f"  {label}: FAIL — {type(exc).__name__}: {exc}")


def test_action_normalizer(results: dict) -> None:
    fn, source = _import_normalize_action()
    if fn is None:
        fn = _local_normalize_action
        source = "local fallback"
    log.info(f"  action normalizer from: {source}")
    inputs = [
        "pickup(a)",
        "pick up a",
        "Pick-Up A",
        "PICK-UP A",
        "stack block a on block b",
        "Stack A B",
        "unstack(a, b)",
        "put down a",
    ]
    failed = None
    for inp in inputs:
        try:
            out = fn(inp)
            print(f"    {inp!r} → {out!r}")
        except Exception as exc:
            failed = (inp, exc)
            break
    results["Action normalizer"] = "PASS" if failed is None else "FAIL"
    if failed:
        log.error(f"  normalizer failed on {failed[0]!r}: {failed[1]}")


def test_resume_logic(results: dict, bw_problems: list[str], gsm_problems: list[str]) -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    with PILOT_RESUME.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["problem_id", "variant_type", "model", "verified"]
        )
        w.writeheader()
        w.writerow(
            {
                "problem_id": bw_problems[0],
                "variant_type": "canonical",
                "model": "mock",
                "verified": "True",
            }
        )
        w.writerow(
            {
                "problem_id": gsm_problems[0],
                "variant_type": "canonical",
                "model": "mock",
                "verified": "True",
            }
        )
    completed: set[tuple[str, str, str]] = set()
    with PILOT_RESUME.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("verified", "")).lower() == "true":
                completed.add(
                    (
                        row["problem_id"],
                        row["variant_type"],
                        row["model"],
                    )
                )
    expectations = [
        ((bw_problems[0], "canonical", "mock"), True),
        ((gsm_problems[0], "canonical", "mock"), True),
        ((bw_problems[1], "canonical", "mock"), False),
    ]
    ok = True
    for key, expected in expectations:
        in_completed = key in completed
        if in_completed != expected:
            ok = False
            log.error(f"  resume mismatch {key}: got {in_completed}, expected {expected}")
    results["Resume logic"] = "PASS" if ok else "FAIL"


def test_probe2_loop_break(results: dict) -> None:
    consecutive_threshold = CONSECUTIVE_SKIP_THRESHOLD
    total_threshold = TOTAL_SKIP_THRESHOLD
    cascade_path = REPO_ROOT / "scripts" / "BW_P2_SCR_run_cascade.py"
    if cascade_path.exists():
        spec = importlib.util.spec_from_file_location("bw_p2", cascade_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        consecutive_threshold = getattr(
            mod, "MAX_CONSECUTIVE_SKIP", consecutive_threshold
        )
        total_threshold = getattr(mod, "MAX_TOTAL_SKIP", total_threshold)

    state = "block a is on the table. goal: stack a on b."
    consecutive_same = 0
    total_skips = 0
    step_skip_fired_at = None
    loop_abort_fired_at = None
    client = MockClient()

    for step in range(1, 25):
        response = client.complete(state)
        _ = response
        new_state = state
        if new_state == state:
            consecutive_same += 1
        else:
            consecutive_same = 0
        if consecutive_same >= consecutive_threshold:
            if step_skip_fired_at is None:
                step_skip_fired_at = step
            total_skips += 1
            consecutive_same = 0
        if total_skips >= total_threshold and loop_abort_fired_at is None:
            loop_abort_fired_at = step
            break

    ok = step_skip_fired_at is not None and loop_abort_fired_at is not None
    if step_skip_fired_at is not None:
        print(f"    STEP_SKIP fired at step {step_skip_fired_at}")
    if loop_abort_fired_at is not None:
        print(f"    LOOP_ABORT fired at step {loop_abort_fired_at}")
    results["Probe 2 loop-break"] = "PASS" if ok else "FAIL"
    if not ok:
        log.error(
            f"  loop-break: step_skip={step_skip_fired_at}, loop_abort={loop_abort_fired_at}"
        )


def _real_api_call(prompt: str, api_key: str) -> str:
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": REAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def test_real_api(results: dict) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        for label in ("Real API — BW", "Real API — GSM", "Real API — ALGO"):
            results[label] = "SKIP"
            print(f"  SKIP: OPENROUTER_API_KEY not set ({label})")
        return

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    bw_ids = _canonical_ids(BANK_BW, 1)
    gsm_ids = _canonical_ids(BANK_GSM, 1)
    algo_id = _algo_canonical_id("coin_change") or _canonical_ids(BANK_ALGO, 1)[0]

    cases = [
        ("BW", BANK_BW, bw_ids[0], None),
        ("GSM", BANK_GSM, gsm_ids[0], "gsm"),
        ("ALGO", BANK_ALGO, algo_id, None),
    ]

    for family_label, bank_path, pid, gsm_flag in cases:
        row = _get_row(bank_path, pid)
        prompt = f"Solve this problem and give your answer:\n\n{row['problem_text']}"
        try:
            response = _real_api_call(prompt, api_key)
            if gsm_flag == "gsm":
                verified = verify_gsm_answer(response, row["correct_answer"])
            else:
                subtype = str(row.get("problem_subtype") or "").strip().lower()
                if subtype in ("blocksworld", "mystery_blocksworld"):
                    fam = subtype
                else:
                    fam = subtype
                verified = verify_answer(
                    pid,
                    response,
                    row["correct_answer"],
                    fam,
                    problem_text=row["problem_text"],
                )
            preview = response[:100].replace("\n", " ")
            rows_out.append(
                {
                    "problem_id": pid,
                    "family": family_label,
                    "response_preview": preview,
                    "verified": str(bool(verified)),
                }
            )
            results[f"Real API — {family_label}"] = "PASS" if verified else "FAIL"
            print(
                f"  {family_label}: {preview!r} verified={verified} "
                f"{'PASS' if verified else 'FAIL'}"
            )
        except Exception as exc:
            results[f"Real API — {family_label}"] = "FAIL"
            rows_out.append(
                {
                    "problem_id": pid,
                    "family": family_label,
                    "response_preview": str(exc)[:100],
                    "verified": "False",
                }
            )
            log.error(f"  {family_label} real API FAIL: {exc}")

    if rows_out:
        with PILOT_REAL_RESULTS.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["problem_id", "family", "response_preview", "verified"],
            )
            w.writeheader()
            w.writerows(rows_out)


def _print_summary_table(results: dict) -> None:
    rows = [
        ("Schema validation gate", results.get("Schema validation gate", "SKIP")),
        ("Verifier routing (CC)", results.get("Verifier routing (CC)", "SKIP")),
        ("Verifier routing (SP)", results.get("Verifier routing (SP)", "SKIP")),
        ("Verifier routing (WIS)", results.get("Verifier routing (WIS)", "SKIP")),
        ("Action normalizer", results.get("Action normalizer", "SKIP")),
        ("Resume logic", results.get("Resume logic", "SKIP")),
        ("Probe 2 loop-break", results.get("Probe 2 loop-break", "SKIP")),
        ("Real API — BW", results.get("Real API — BW", "SKIP")),
        ("Real API — GSM", results.get("Real API — GSM", "SKIP")),
        ("Real API — ALGO", results.get("Real API — ALGO", "SKIP")),
    ]
    print()
    print("╔══════════════════════════════════╦════════╗")
    print("║ Test                             ║ Result ║")
    print("╠══════════════════════════════════╬════════╣")
    for name, result in rows:
        print(f"║ {name:<32} ║ {result:<6} ║")
    print("╚══════════════════════════════════╩════════╝")
    fails = [
        (name, res)
        for name, res in rows
        if res == "FAIL"
    ]
    if fails:
        print()
        print("FAIL details:")
        for name, res in fails:
            print(f"  - {name}: {res}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 — pilot dress rehearsal")
    parser.add_argument("--mock", action="store_true", help="Run mock tests")
    parser.add_argument("--real", action="store_true", help="Run real API tests")
    args = parser.parse_args()
    if not args.mock and not args.real:
        args.mock = True

    results: dict[str, str] = {}
    bw_problems = _canonical_ids(BANK_BW, 3)
    gsm_problems = _canonical_ids(BANK_GSM, 3)

    if args.mock:
        log.info("TEST 1 — Schema validation gate")
        test_schema_gate(results)
        log.info("TEST 2 — Verifier routing")
        test_verifier_routing(results)
        log.info("TEST 3 — Action normalizer")
        test_action_normalizer(results)
        log.info("TEST 4 — Resume logic")
        test_resume_logic(results, bw_problems, gsm_problems)
        log.info("TEST 5 — Probe 2 loop-break")
        test_probe2_loop_break(results)
    else:
        for key in [
            "Schema validation gate",
            "Verifier routing (CC)",
            "Verifier routing (SP)",
            "Verifier routing (WIS)",
            "Action normalizer",
            "Resume logic",
            "Probe 2 loop-break",
        ]:
            results[key] = "SKIP"

    if args.real:
        log.info("REAL API CALLS")
        test_real_api(results)
    else:
        for label in ("Real API — BW", "Real API — GSM", "Real API — ALGO"):
            results[label] = "SKIP"

    _print_summary_table(results)
    failed = [k for k, v in results.items() if v == "FAIL"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
