#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generation.utils.bank_writer import (  # noqa: E402
    max_id_number,
    next_problem_id,
    read_existing_bank,
    used_source_keys,
    write_rows,
)
from scripts.generation.utils.pddl_parser import (  # noqa: E402
    PddlProblem,
    parse_problem,
    render_bw_prompt,
    render_mbw_prompt,
    verify_nl_format_matches_bank,
)


FAST_DOWNWARD = REPO_ROOT / "tools/fast-downward/fast-downward.py"
QUESTION_BANK = REPO_ROOT / "data/problems/question_bank_bw.csv"
OUT_CSV = REPO_ROOT / "data/staging/bw_canonical.csv"


def _difficulty(plan_len: int) -> str:
    if plan_len <= 6:
        return "easy"
    if plan_len <= 12:
        return "medium"
    return "hard"


def _parse_plan_lines(text: str) -> list[str]:
    actions: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+:\s*\(([^)]+)\)\s*$", line.strip().lower())
        if not m:
            continue
        action = " ".join(m.group(1).split())
        actions.append(action)
    return actions


def solve_with_fd(domain: Path, problem: Path, timeout_s: int = 30) -> list[str] | None:
    with tempfile.TemporaryDirectory(prefix="fd_bw_") as tmp:
        plan_file = Path(tmp) / "sas_plan"
        cmd = [
            sys.executable,
            str(FAST_DOWNWARD),
            "--plan-file",
            str(plan_file),
            str(domain),
            str(problem),
            "--search",
            "astar(lmcut())",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None

        if plan_file.exists():
            raw = plan_file.read_text(encoding="utf-8")
            lines = [
                " ".join(x.strip().strip("()").split()).lower()
                for x in raw.splitlines()
                if x.strip() and not x.strip().startswith(";")
            ]
            return lines if lines else None

        merged = f"{proc.stdout}\n{proc.stderr}"
        lines = _parse_plan_lines(merged)
        return lines if lines else None


def _find_first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No existing path in candidates: {[str(x) for x in candidates]}")


def _source_for_planbench(kind: str, rel_path: str, filename: str) -> str:
    return f"type={kind} | dataset=planbench | filename={filename} | path={rel_path}"


def _make_answer(plan_actions: list[str]) -> str:
    return "\n".join(plan_actions)


def _bw_row(
    *,
    problem_id: str,
    text: str,
    answer: str,
    difficulty: str,
    source: str,
    subtype: str,
    verifier: str,
    contamination: str,
    notes: str,
    n_blocks: int,
) -> dict:
    return {
        "problem_id": problem_id,
        "variant_type": "canonical",
        "problem_text": text,
        "correct_answer": answer,
        "problem_family": "planning_suite",
        "problem_subtype": subtype,
        "difficulty": difficulty,
        "contamination_pole": contamination,
        "source": source,
        "verifier_function": verifier,
        "difficulty_params": f"num_blocks={n_blocks}",
        "notes": notes,
        "status": "ok",
        "selection_reason": "stage1_canonical_extraction",
    }


def _procedural_bwe_problem(seed: int, n_blocks: int, out_path: Path) -> None:
    rng = random.Random(seed)
    blocks = [chr(ord("a") + i) for i in range(12)]
    chosen = rng.sample(blocks, n_blocks)
    rng.shuffle(chosen)
    goal_chain = [(chosen[i], chosen[i + 1]) for i in range(len(chosen) - 1)]
    goal_lines = "\n".join(f"(on {x} {y})" for x, y in goal_chain)
    obj = " ".join(chosen)
    pddl = f"""(define (problem BW-E-{seed})
(:domain blocksworld-4ops)
(:objects {obj})
(:init
(handempty)
{chr(10).join(f"(ontable {b})" for b in chosen)}
{chr(10).join(f"(clear {b})" for b in chosen)}
)
(:goal
(and
{goal_lines}
))
)
"""
    out_path.write_text(pddl, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 BW canonical extraction")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print selected candidates only")
    mode.add_argument("--run", action="store_true", help="Write data/staging/bw_canonical.csv")
    args = parser.parse_args()

    if args.run:
        sanity_problem = PddlProblem(
            name="format-sanity",
            domain="blocksworld-4ops",
            objects=["a", "b", "c", "d"],
            init_atoms=[
                ("handempty", []),
                ("ontable", ["a"]),
                ("ontable", ["b"]),
                ("ontable", ["c"]),
                ("ontable", ["d"]),
                ("clear", ["a"]),
                ("clear", ["b"]),
                ("clear", ["c"]),
                ("clear", ["d"]),
            ],
            goal_atoms=[("on", ["a", "b"]), ("on", ["b", "c"])],
        )
        sanity_text = render_bw_prompt(sanity_problem)
        if not verify_nl_format_matches_bank(sanity_text, str(QUESTION_BANK)):
            raise RuntimeError(
                "NL format sanity check failed: generated BW prompt structure does not match question_bank_bw.csv."
            )

    bw_problem_dir = _find_first_existing(
        [
            REPO_ROOT / "data/sources/planbench/blocksworld/problem_files",
            REPO_ROOT / "data/sources/planbench/plan-bench/instances/blocksworld/generated",
            REPO_ROOT / "data/sources/planbench/llm_planning_analysis/instances/blocksworld/generated_basic",
        ]
    )
    mbw_problem_dir = _find_first_existing(
        [
            REPO_ROOT / "data/sources/planbench/plan-bench/instances/blocksworld/mystery/generated",
            REPO_ROOT / "data/sources/planbench/llm_planning_analysis/instances/blocksworld/mystery/generated_basic",
        ]
    )
    bw_domain = _find_first_existing(
        [
            REPO_ROOT / "data/sources/planbench/llm_planning_analysis/instances/blocksworld/generated_domain.pddl",
            REPO_ROOT / "data/sources/planbench/plan-bench/instances/blocksworld/generated_domain.pddl",
        ]
    )
    mbw_domain = _find_first_existing(
        [
            REPO_ROOT / "data/sources/planbench/llm_planning_analysis/instances/blocksworld/mystery/generated_domain.pddl",
            REPO_ROOT / "data/sources/planbench/plan-bench/instances/blocksworld/mystery/generated_domain.pddl",
        ]
    )

    bank_df = read_existing_bank(QUESTION_BANK)
    used_sources = used_source_keys(bank_df)

    bw_max = max_id_number(bank_df, "BW")
    mbw_max = max_id_number(bank_df, "MBW")
    bwe_max = max_id_number(bank_df, "BW_E")

    selected: list[dict] = []
    bw_targets = {"easy": 5, "medium": 10, "hard": 5}
    bw_counts = {k: 0 for k in bw_targets}

    for pddl_file in sorted(bw_problem_dir.glob("*.pddl")):
        if all(bw_counts[k] >= bw_targets[k] for k in bw_targets):
            break
        if pddl_file.name in used_sources or str(pddl_file) in used_sources:
            continue
        rel_source = pddl_file.as_posix().split("data/sources/planbench/")[-1]
        if rel_source in used_sources:
            continue

        plan = solve_with_fd(bw_domain, pddl_file, timeout_s=30)
        if not plan:
            continue
        diff = _difficulty(len(plan))
        if bw_counts[diff] >= bw_targets[diff]:
            continue
        problem = parse_problem(pddl_file)
        bw_max += 1
        pid = next_problem_id("BW", bw_max - 1)
        source = _source_for_planbench("planbench_original", rel_source, pddl_file.name)
        row = _bw_row(
            problem_id=pid,
            text=render_bw_prompt(problem),
            answer=_make_answer(plan),
            difficulty=diff,
            source=source,
            subtype="blocksworld",
            verifier="verify_blocksworld_plan",
            contamination="high",
            notes="Stage1 PlanBench canonical extraction",
            n_blocks=len(problem.objects),
        )
        selected.append(row)
        bw_counts[diff] += 1
        used_sources.add(pddl_file.name)
        used_sources.add(rel_source)

    mbw_count = 0
    for pddl_file in sorted(mbw_problem_dir.glob("*.pddl")):
        if mbw_count >= 5:
            break
        if pddl_file.name in used_sources or str(pddl_file) in used_sources:
            continue
        rel_source = pddl_file.as_posix().split("data/sources/planbench/")[-1]
        if rel_source in used_sources:
            continue
        plan = solve_with_fd(mbw_domain, pddl_file, timeout_s=30)
        if not plan:
            continue
        problem = parse_problem(pddl_file)
        diff = _difficulty(len(plan))
        mbw_max += 1
        pid = next_problem_id("MBW", mbw_max - 1)
        source = _source_for_planbench("planbench_mystery", rel_source, pddl_file.name)
        row = _bw_row(
            problem_id=pid,
            text=render_mbw_prompt(problem),
            answer=_make_answer(plan),
            difficulty=diff,
            source=source,
            subtype="mystery_blocksworld",
            verifier="verify_mystery_blocksworld_plan",
            contamination="high",
            notes="Stage1 PlanBench MBW canonical extraction",
            n_blocks=len(problem.objects),
        )
        selected.append(row)
        mbw_count += 1
        used_sources.add(pddl_file.name)
        used_sources.add(rel_source)

    bwe_count = 0
    bwe_seed = 20260520
    with tempfile.TemporaryDirectory(prefix="bwe_gen_") as tmp:
        tmp_dir = Path(tmp)
        while bwe_count < 5 and bwe_seed < 20261520:
            pddl_path = tmp_dir / f"bwe-seed-{bwe_seed}.pddl"
            n_blocks = random.Random(bwe_seed).choice([4, 5, 6, 7, 8, 9])
            _procedural_bwe_problem(bwe_seed, n_blocks, pddl_path)

            source = f"type=procedural_bw_e | dataset=generated | seed={bwe_seed} | filename={pddl_path.name}"
            if source in used_sources or pddl_path.name in used_sources:
                bwe_seed += 1
                continue

            plan = solve_with_fd(bw_domain, pddl_path, timeout_s=30)
            if not plan:
                bwe_seed += 1
                continue
            problem = parse_problem(pddl_path)
            diff = _difficulty(len(plan))
            bwe_max += 1
            pid = next_problem_id("BW_E", bwe_max - 1)
            row = _bw_row(
                problem_id=pid,
                text=render_bw_prompt(problem),
                answer=_make_answer(plan),
                difficulty=diff,
                source=source,
                subtype="blocksworld",
                verifier="verify_blocksworld_plan",
                contamination="low",
                notes=f"Procedural BW_E generated with seed {bwe_seed}",
                n_blocks=len(problem.objects),
            )
            selected.append(row)
            bwe_count += 1
            used_sources.add(source)
            bwe_seed += 1

    by_prefix = {"BW": 0, "MBW": 0, "BW_E": 0}
    for row in selected:
        pid = row["problem_id"]
        if pid.startswith("BW_E_"):
            by_prefix["BW_E"] += 1
        elif pid.startswith("MBW_"):
            by_prefix["MBW"] += 1
        elif pid.startswith("BW_"):
            by_prefix["BW"] += 1

    print("Selection summary:")
    print(f"  BW standard: {by_prefix['BW']} (easy={bw_counts['easy']}, medium={bw_counts['medium']}, hard={bw_counts['hard']})")
    print(f"  MBW: {by_prefix['MBW']}")
    print(f"  BW_E: {by_prefix['BW_E']}")
    print()
    for row in selected:
        print(f"{row['problem_id']} | {row['difficulty']} | {row['source']}")

    if args.run:
        write_rows(selected, OUT_CSV)
        print(f"\nWrote {len(selected)} rows to {OUT_CSV}")
    else:
        print("\nDry-run mode: no files written.")


if __name__ == "__main__":
    main()
