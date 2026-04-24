"""Run contamination triage over Probe 1 instances."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

from tqdm import tqdm
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from probes.contamination import infinigram_client as _ig
from probes.contamination.score import score_problem
from probes.common.io import QUESTION_BANK_PATH

INPUT_PATH = Path(QUESTION_BANK_PATH)
OUTPUT_PATH = Path("results/BW_P3_RES_contamination_triage.csv")

OUTPUT_COLUMNS = [
    "problem_id",
    "problem_family",
    "problem_subtype",
    "problem_text",
    "correct_answer",
    "max_ngram_length",
    "max_ngram_count",
    "contamination_score",
    "template_contamination_score",
    "instance_contamination_score",
    "difficulty_numeric",
]


def _existing_problem_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {str(row.get("problem_id", "")).strip() for row in reader if row.get("problem_id")}


def run_triage(
    limit: int | None = None,
    family: str | None = None,
    resume: bool = True,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    max_ngram: int | None = None,
    decompose_contamination: bool = False,
) -> None:
    # Comparability guardrail:
    # - contamination_score must remain the legacy score_problem(problem_text, ...)
    # - decomposition fields are additive only
    # - no family-specific shortcut scoring paths are introduced here
    print(
        f"Infini-gram endpoint {_ig.API_URL!r} index={_ig.INDEX_NAME!r} "
        f"ssl_verify={_ig.SSL_VERIFY}",
        flush=True,
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = _existing_problem_ids(output_path) if resume else set()
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if rows:
        required_columns = {
            "problem_id",
            "variant_type",
            "problem_text",
            "correct_answer",
            "problem_family",
            "problem_subtype",
            "difficulty_params",
        }
        missing_cols = required_columns - set(rows[0].keys())
        if missing_cols:
            raise ValueError(
                f"Question bank missing required columns: {sorted(missing_cols)}"
            )

    # Triage should run over canonical rows only.
    rows = [
        row
        for row in rows
        if str(row.get("variant_type", "")).strip().lower() == "canonical"
    ]

    if family is not None:
        family_norm = family.strip().lower()
        algo_family_aliases = {"algorithmic", "algorithmic_suite", "algorithmic suit"}
        rows = [
            row for row in rows
            if (
                row.get("problem_subtype", "").strip().lower() == family_norm
                or row.get("problem_family", "").strip().lower() == family_norm
                or (
                    family_norm in algo_family_aliases
                    and row.get("problem_subtype", "").strip().lower() in {"coin_change", "shortest_path", "wis"}
                )
            )
        ]

    if limit is not None:
        rows = rows[:limit]

    template_queries = {
        "coin_change": "minimum number of coins to make change for",
        "shortest_path": "find the shortest path in a weighted graph",
        "wis": "weighted interval scheduling maximum weight independent set",
    }

    def _score_query_with_retry(query: str, fam: str, retries: int = 2) -> float:
        if not query.strip():
            raise ValueError("Contamination query is empty.")
        last_err: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                return float(
                    score_problem(
                        query,
                        family=fam,
                        max_ngram=max_ngram,
                    )["contamination_score"]
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(
                    f"ERROR: contamination query failed attempt={attempt} query={query!r} err={e}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(0.5)
        # Explicit sentinel value; not silently skipped.
        print(
            f"ERROR: contamination query permanently failed; marking score=-1.0 query={query!r} err={last_err}",
            file=sys.stderr,
            flush=True,
        )
        return -1.0

    def _parse_difficulty_params(problem_id: str, raw: str) -> dict:
        if not str(raw).strip():
            raise ValueError(f"{problem_id}: missing difficulty_params")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{problem_id}: invalid difficulty_params JSON: {e}") from e

    def _build_instance_query_and_difficulty(problem_id: str, subtype: str, params: dict) -> tuple[str, int]:
        if subtype == "coin_change":
            if "denominations" not in params or "target" not in params:
                raise ValueError(f"{problem_id}: coin_change requires denominations + target in difficulty_params")
            denoms = params["denominations"]
            target = params["target"]
            if not isinstance(denoms, list) or not denoms:
                raise ValueError(f"{problem_id}: denominations must be non-empty list")
            denoms_int = [int(x) for x in denoms]
            target_int = int(target)
            query = f"denominations {', '.join(str(x) for x in denoms_int)} target {target_int}"
            return query, len(denoms_int)
        if subtype == "shortest_path":
            if "graph" not in params:
                raise ValueError(f"{problem_id}: shortest_path requires graph in difficulty_params")
            graph = params["graph"]
            if not isinstance(graph, list) or not graph:
                raise ValueError(f"{problem_id}: graph must be non-empty list")
            triples: list[str] = []
            for e in graph:
                if not isinstance(e, dict) or not {"u", "v", "w"}.issubset(set(e.keys())):
                    raise ValueError(f"{problem_id}: graph edges must be dicts with u,v,w")
                triples.append(f"{int(e['u'])} {int(e['v'])} {int(e['w'])}")
            query = ", ".join(triples)
            # Use edge-count consistently for SP difficulty_numeric.
            return query, len(graph)
        if subtype == "wis":
            if "intervals" not in params:
                raise ValueError(f"{problem_id}: wis requires intervals in difficulty_params")
            intervals = params["intervals"]
            if not isinstance(intervals, list) or not intervals:
                raise ValueError(f"{problem_id}: intervals must be non-empty list")
            triples: list[str] = []
            for it in intervals:
                if isinstance(it, dict):
                    if not {"start", "end", "weight"}.issubset(set(it.keys())):
                        raise ValueError(f"{problem_id}: wis interval dict must include start,end,weight")
                    s, e, w = int(it["start"]), int(it["end"]), int(it["weight"])
                elif isinstance(it, (list, tuple)) and len(it) >= 3:
                    s, e, w = int(it[0]), int(it[1]), int(it[2])
                else:
                    raise ValueError(f"{problem_id}: wis interval must be dict or [start,end,weight]")
                triples.append(f"({s},{e},{w})")
            query = ", ".join(triples)
            return query, len(intervals)
        raise ValueError(f"{problem_id}: unsupported problem_subtype {subtype!r}")

    gradient_rows: list[dict] = []

    with output_path.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)

        if write_header:
            writer.writeheader()
            outfile.flush()

        print(f"Processing {len(rows)} problems...")
        for row in tqdm(rows, desc="Contamination Triage"):
            problem_id = str(row.get("problem_id", "")).strip()
            if problem_id and problem_id in processed_ids:
                continue

            problem_text = str(row.get("problem_text", "")).strip().strip('"')
            problem_subtype = str(row.get("problem_subtype", "")).strip().lower()
            params = _parse_difficulty_params(problem_id, str(row.get("difficulty_params", "")))
            score = score_problem(
                problem_text,
                family=str(row.get("problem_family", "")),
                max_ngram=max_ngram,
            )

            template_score = ""
            instance_score = ""
            difficulty_numeric: int | str = ""
            if decompose_contamination:
                if problem_subtype not in template_queries:
                    raise ValueError(
                        f"{problem_id}: unsupported problem_subtype for decomposition: {problem_subtype!r}"
                    )
                template_query = template_queries[problem_subtype]
                instance_query, difficulty_numeric = _build_instance_query_and_difficulty(
                    problem_id, problem_subtype, params
                )
                template_score = _score_query_with_retry(
                    template_query, fam=str(row.get("problem_family", ""))
                )
                instance_score = _score_query_with_retry(
                    instance_query, fam=str(row.get("problem_family", ""))
                )
                gradient_rows.append(
                    {
                        "problem_id": problem_id,
                        "subtype": problem_subtype,
                        "instance_type": str(params.get("instance_type", "")).strip().lower(),
                        "instance_contamination_score": instance_score,
                    }
                )

            output_row = {
                "problem_id": problem_id,
                "problem_family": row.get("problem_family", row.get("problem_subtype", "")),
                "problem_subtype": row.get("problem_subtype", ""),
                "problem_text": problem_text,
                "correct_answer": row.get("correct_answer", ""),
                "max_ngram_length": score["max_ngram_length"],
                "max_ngram_count": score["max_ngram_count"],
                "contamination_score": score["contamination_score"],
                "template_contamination_score": template_score,
                "instance_contamination_score": instance_score,
                "difficulty_numeric": difficulty_numeric,
            }
            writer.writerow(output_row)
            outfile.flush()

            if problem_id:
                processed_ids.add(problem_id)

    if decompose_contamination and gradient_rows:
        cc_std = [
            float(r["instance_contamination_score"])
            for r in gradient_rows
            if r["subtype"] == "coin_change" and r["instance_type"] == "standard" and float(r["instance_contamination_score"]) >= 0
        ]
        cc_adv = [
            float(r["instance_contamination_score"])
            for r in gradient_rows
            if r["subtype"] == "coin_change" and r["instance_type"] == "adversarial" and float(r["instance_contamination_score"]) >= 0
        ]
        wis_std = [
            float(r["instance_contamination_score"])
            for r in gradient_rows
            if r["subtype"] == "wis" and r["instance_type"] == "standard" and float(r["instance_contamination_score"]) >= 0
        ]
        wis_adv = [
            float(r["instance_contamination_score"])
            for r in gradient_rows
            if r["subtype"] == "wis" and r["instance_type"] == "adversarial" and float(r["instance_contamination_score"]) >= 0
        ]
        gradient_ok = True
        if cc_std and cc_adv and (sum(cc_std) / len(cc_std) <= sum(cc_adv) / len(cc_adv)):
            gradient_ok = False
        if wis_std and wis_adv and (sum(wis_std) / len(wis_std) <= sum(wis_adv) / len(wis_adv)):
            gradient_ok = False
        if not gradient_ok:
            print(
                "Contamination gradient not observed — adversarial instances may be contaminated",
                file=sys.stderr,
                flush=True,
            )

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run contamination triage")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only first N problems then stop"
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Process only problems where problem_family matches this value",
    )
    parser.add_argument(
        "--question-bank-path",
        type=str,
        default=str(INPUT_PATH),
        help="Path to question bank CSV (legacy alias; prefer --bank-path)",
    )
    parser.add_argument(
        "--bank-path",
        type=str,
        default=None,
        help="Path to question bank CSV (required for new pipeline usage)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        default=None,
        help="Override maximum n-gram length used in contamination scoring",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip problem_ids already present in the output CSV (default)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not skip problem_ids already present",
    )
    parser.add_argument(
        "--decompose-contamination",
        action="store_true",
        help="Compute template_contamination_score + instance_contamination_score + difficulty_numeric",
    )

    args = parser.parse_args()
    if not args.bank_path:
        raise ValueError("--bank-path is required")
    if not args.output:
        raise ValueError("--output is required")

    run_triage(
        limit=args.limit,
        family=args.family,
        resume=args.resume,
        input_path=Path(args.bank_path or args.question_bank_path),
        output_path=Path(args.output),
        max_ngram=args.max_ngram,
        decompose_contamination=args.decompose_contamination,
    )
