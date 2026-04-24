#!/usr/bin/env python3
"""Strict audit for algorithmic question bank rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "problem_id",
    "variant_type",
    "problem_text",
    "correct_answer",
    "problem_family",
    "problem_subtype",
    "difficulty",
    "contamination_pole",
    "source",
    "verifier_function",
    "difficulty_params",
    "notes",
]
VALID_VARIANTS = {"canonical", "W1", "W2", "W3", "W4", "W6"}
VALID_POLES = {"High", "Medium", "Low"}


def fail(fails: list[dict], row: pd.Series, check_name: str, detail: str) -> None:
    fails.append(
        {
            "problem_id": str(row.get("problem_id", "")),
            "variant_type": str(row.get("variant_type", "")),
            "check_name": check_name,
            "status": "FAIL",
            "detail": detail,
        }
    )


def normalize_params_for_compare(params: dict) -> dict:
    x = dict(params)
    x.pop("node_mapping", None)
    x.pop("item_mapping", None)
    return x


def parse_cc_answer(answer: str) -> tuple[int, list[int]]:
    m = re.search(
        r"(?:Count|Total)\s*:\s*(\d+)\s*[\n, ]+\s*(?:Coins|Scoops)\s*:\s*\[([^\]]*)\]",
        answer,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError("Invalid CC answer format.")
    count = int(m.group(1))
    coins = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
    return count, coins


def parse_sp_answer(answer: str) -> tuple[list[str], int]:
    m = re.search(r"Path:\s*(.+?)\s*,\s*Cost:\s*(-?\d+)\s*$", answer.strip())
    if not m:
        raise ValueError("Invalid SP answer format.")
    path = [x.strip() for x in re.split(r"\s*→\s*|\s*->\s*", m.group(1)) if x.strip()]
    return path, int(m.group(2))


def parse_wis_answer(answer: str) -> tuple[list[str], int]:
    m = re.search(r"Selected:\s*\{([^}]*)\}\s*,\s*Total:\s*(\d+)\s*$", answer.strip())
    if not m:
        raise ValueError("Invalid WIS answer format.")
    items = [x.strip() for x in m.group(1).split(",") if x.strip()]
    return items, int(m.group(2))


def audit_row(row: pd.Series, fails: list[dict], total_checks: Counter) -> None:
    subtype_prefix = str(row["problem_id"]).split("_", 1)[0]
    total_checks["structural"] += 1

    if str(row["variant_type"]) not in VALID_VARIANTS:
        fail(fails, row, "variant_type", "variant_type must be one of canonical/W1/W2/W3/W4/W6.")
    if str(row["variant_type"]) != "canonical" and str(row["variant_type"]).islower():
        fail(fails, row, "variant_type_case", "variant_type must not be lowercase.")
    if str(row["contamination_pole"]) not in VALID_POLES:
        fail(fails, row, "contamination_pole", "contamination_pole must be High/Medium/Low.")
    if not str(row["correct_answer"]).strip():
        fail(fails, row, "correct_answer_non_empty", "correct_answer is empty.")

    params_obj: dict
    try:
        params_obj = json.loads(str(row["difficulty_params"]))
        if not isinstance(params_obj, dict):
            raise ValueError("difficulty_params must decode to JSON object.")
    except Exception as exc:
        fail(fails, row, "difficulty_params_json", f"invalid JSON: {exc}")
        return

    if "greedy_answer" not in params_obj or not str(params_obj.get("greedy_answer", "")).strip():
        fail(
            fails,
            row,
            "greedy_answer_required",
            "All ALGO rows must include non-empty greedy_answer.",
        )
    if "difficulty_numeric" not in params_obj or str(params_obj.get("difficulty_numeric", "")).strip() == "":
        fail(
            fails,
            row,
            "difficulty_numeric_required",
            "All ALGO rows must include difficulty_numeric.",
        )
    if subtype_prefix in {"CC", "WIS"} and "greedy_succeeds" not in params_obj:
        fail(fails, row, "greedy_succeeds_required", "CC/WIS must include greedy_succeeds.")
    if "critical_step_index" not in params_obj:
        fail(
            fails,
            row,
            "critical_step_index_required",
            "All ALGO rows must include critical_step_index.",
        )
    else:
        try:
            csi = int(params_obj.get("critical_step_index"))
            inst = str(params_obj.get("instance_type", "")).strip().lower()
            if inst == "standard" and csi != -1:
                fail(
                    fails,
                    row,
                    "critical_step_index_standard",
                    "Standard rows must set critical_step_index=-1.",
                )
            if inst == "adversarial" and csi < 0:
                fail(
                    fails,
                    row,
                    "critical_step_index_adversarial",
                    "Adversarial rows must set 0-indexed critical_step_index >= 0.",
                )
        except Exception:
            fail(
                fails,
                row,
                "critical_step_index_type",
                "critical_step_index must be an integer.",
            )

    if str(row["variant_type"]) == "W6":
        if str(row["contamination_pole"]) != "Low":
            fail(fails, row, "w6_contamination", "W6 contamination_pole must be Low.")
        if not str(row["source"]).startswith("procedural_seed_"):
            fail(fails, row, "w6_source", "W6 source must start with procedural_seed_.")

    total_checks["content"] += 1
    try:
        if subtype_prefix == "CC":
            denoms = list(params_obj["denominations"])
            target = int(params_obj["target"])
            count, coins = parse_cc_answer(str(row["correct_answer"]))
            if len(coins) != count:
                raise ValueError("Count does not match number of coins.")
            if sum(coins) != target:
                raise ValueError("Coin sum does not match target.")
            if any(c not in denoms for c in coins):
                raise ValueError("Answer includes coin not in denomination set.")
        elif subtype_prefix == "SP":
            edges = params_obj["graph"]
            edge_cost = {(int(e["u"]), int(e["v"])): int(e["w"]) for e in edges}
            src = int(params_obj["source"])
            tgt = int(params_obj["target"])
            node_mapping = {v: int(k) for k, v in params_obj.get("node_mapping", {}).items()}

            path_tokens, cost = parse_sp_answer(str(row["correct_answer"]))
            if node_mapping:
                nodes = [node_mapping[t] for t in path_tokens]
            else:
                nodes = [int(t) for t in path_tokens]
            if not nodes or nodes[0] != src or nodes[-1] != tgt:
                raise ValueError("SP path endpoints do not match source/target.")
            total = 0
            for a, b in zip(nodes[:-1], nodes[1:]):
                if (a, b) not in edge_cost:
                    raise ValueError(f"Missing edge in path: {(a, b)}")
                total += edge_cost[(a, b)]
            if total != cost:
                raise ValueError("SP path cost does not equal summed edge weights.")
        elif subtype_prefix == "WIS":
            intervals = params_obj["intervals"]
            item_mapping = {v: int(k) for k, v in params_obj.get("item_mapping", {}).items()}
            by_id = {int(x["id"]): x for x in intervals}
            items, total = parse_wis_answer(str(row["correct_answer"]))
            chosen = [item_mapping[t] for t in items] if item_mapping else [int(t) for t in items]
            for idx in chosen:
                if idx not in by_id:
                    raise ValueError(f"Selected interval id not found: {idx}")
            for i in range(len(chosen)):
                a = by_id[chosen[i]]
                for j in range(i + 1, len(chosen)):
                    b = by_id[chosen[j]]
                    if not (int(a["end"]) <= int(b["start"]) or int(b["end"]) <= int(a["start"])):
                        raise ValueError("Selected intervals overlap.")
            computed = sum(int(by_id[idx]["weight"]) for idx in chosen)
            if computed != total:
                raise ValueError("WIS total does not equal sum of selected interval weights.")
    except Exception as exc:
        fail(fails, row, f"{subtype_prefix}_content", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank_path = Path(args.bank)
    out_path = Path(args.output)
    if not bank_path.exists():
        raise FileNotFoundError(f"Missing bank CSV: {bank_path}")

    df = pd.read_csv(bank_path)
    fails: list[dict] = []
    checks = Counter()

    for col in REQUIRED_COLUMNS:
        checks["structural"] += 1
        if col not in df.columns:
            fails.append(
                {
                    "problem_id": "",
                    "variant_type": "",
                    "check_name": "required_columns",
                    "status": "FAIL",
                    "detail": f"Missing column: {col}",
                }
            )
    if fails:
        pd.DataFrame(fails).to_csv(out_path, index=False)
        print(f"total rows: {len(df)}")
        print(f"total checks: {sum(checks.values())}")
        print("failures per category: structural=1")
        print("Audit failed. Exit 1.")
        raise SystemExit(1)

    # Focus audit on algorithmic suite ids.
    algo_df = df[df["problem_id"].astype(str).str.match(r"^(CC|SP|WIS)_")].copy()

    for _, row in algo_df.iterrows():
        audit_row(row, fails, checks)

    # Cross-variant checks.
    for pid, group in algo_df.groupby("problem_id"):
        checks["cross_variant"] += 1
        if group["problem_subtype"].nunique() != 1:
            fails.append(
                {
                    "problem_id": pid,
                    "variant_type": "*",
                    "check_name": "problem_subtype_consistency",
                    "status": "FAIL",
                    "detail": "problem_subtype differs across variants.",
                }
            )
        if group["difficulty"].nunique() != 1:
            fails.append(
                {
                    "problem_id": pid,
                    "variant_type": "*",
                    "check_name": "difficulty_consistency",
                    "status": "FAIL",
                    "detail": "difficulty differs across variants.",
                }
            )
        if group["verifier_function"].nunique() != 1:
            fails.append(
                {
                    "problem_id": pid,
                    "variant_type": "*",
                    "check_name": "verifier_function_consistency",
                    "status": "FAIL",
                    "detail": "verifier_function differs across variants.",
                }
            )

        canonical = group[group["variant_type"] == "canonical"]
        if len(canonical) == 1:
            canonical_params = json.loads(str(canonical.iloc[0]["difficulty_params"]))
            canonical_norm = normalize_params_for_compare(canonical_params)
            for _, row in group.iterrows():
                params = json.loads(str(row["difficulty_params"]))
                if normalize_params_for_compare(params) != canonical_norm:
                    fails.append(
                        {
                            "problem_id": pid,
                            "variant_type": str(row["variant_type"]),
                            "check_name": "difficulty_params_consistency",
                            "status": "FAIL",
                            "detail": "difficulty_params differs from canonical (beyond mapping fields).",
                        }
                    )
            w6 = group[group["variant_type"] == "W6"]
            if len(w6) == 1 and str(w6.iloc[0]["correct_answer"]) == str(canonical.iloc[0]["correct_answer"]):
                fails.append(
                    {
                        "problem_id": pid,
                        "variant_type": "W6",
                        "check_name": "w6_answer_difference",
                        "status": "FAIL",
                        "detail": "W6 correct_answer must differ from canonical.",
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        fails, columns=["problem_id", "variant_type", "check_name", "status", "detail"]
    ).to_csv(out_path, index=False)

    category_counts = Counter(item["check_name"].split("_")[0] for item in fails)
    print(f"total rows: {len(algo_df)}")
    print(f"total checks: {sum(checks.values())}")
    if category_counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(category_counts.items()))
    else:
        summary = "none"
    print(f"failures per category: {summary}")

    if fails:
        print("Audit failed. Exit 1.")
        raise SystemExit(1)
    print("All checks passed. Exit 0.")


if __name__ == "__main__":
    main()
