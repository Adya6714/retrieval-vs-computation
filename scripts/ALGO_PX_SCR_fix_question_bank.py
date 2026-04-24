#!/usr/bin/env python3
"""Apply targeted ALGO fixes and normalize difficulty_params JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_BANK = Path("data/problems/question_bank_algo.csv")


def require_single_row(df: pd.DataFrame, problem_id: str, variant_type: str) -> int:
    mask = (df["problem_id"] == problem_id) & (df["variant_type"] == variant_type)
    matches = df.index[mask].tolist()
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 row for ({problem_id}, {variant_type}); found {len(matches)}."
        )
    return matches[0]


def set_field(
    df: pd.DataFrame,
    row_idx: int,
    field: str,
    new_value: str,
    changes: dict[int, list[str]],
) -> None:
    old_value = str(df.at[row_idx, field])
    if old_value == new_value:
        return
    df.at[row_idx, field] = new_value
    changes[row_idx].append(f"{field}: {old_value!r} -> {new_value!r}")


def replace_one_of(
    df: pd.DataFrame,
    row_idx: int,
    field: str,
    old_values: tuple[str, ...],
    new_value: str,
    changes: dict[int, list[str]],
) -> None:
    cur = str(df.at[row_idx, field])
    if new_value in cur:
        return
    for old in old_values:
        if old in cur:
            updated = cur.replace(old, new_value)
            if updated != cur:
                df.at[row_idx, field] = updated
                changes[row_idx].append(f"{field}: replaced {old!r} with {new_value!r}")
            return
    raise ValueError(f"Expected one of {old_values!r} in {field} for row index {row_idx}.")


def parse_key_equals_blob(blob: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(blob).split("|"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_cc_from_text(text: str) -> tuple[list[int], int]:
    den_match = re.search(r"denominations:\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
    tgt_match = re.search(r"exact change for\s*(\d+)", text, flags=re.IGNORECASE)
    if not den_match or not tgt_match:
        raise ValueError("Unable to parse CC denominations/target from problem_text.")
    denoms = [int(x.strip()) for x in den_match.group(1).split(",") if x.strip()]
    target = int(tgt_match.group(1))
    return denoms, target


def parse_sp_edges(text: str) -> list[tuple[int, int, int]]:
    found: list[tuple[int, int, int]] = []
    triples = re.findall(r"\((\d+)\s*,\s*(\d+)\s*,\s*(-?\d+)\)", text)
    found.extend((int(u), int(v), int(w)) for u, v, w in triples)
    line_edges = re.findall(r"\((\d+)\)[^:\n]*\((\d+)\):[^\d-]*(-?\d+)", text)
    found.extend((int(u), int(v), int(w)) for u, v, w in line_edges)
    plain_edges = re.findall(
        r"\b(\d+)\b[^:\n]*?\bto\b[^:\n]*?\b(\d+)\b:[^\d-]*(-?\d+)",
        text,
        flags=re.IGNORECASE,
    )
    found.extend((int(u), int(v), int(w)) for u, v, w in plain_edges)
    dedup = list(dict.fromkeys(found))
    return dedup


def parse_sp_src_tgt(answer: str) -> tuple[int, int]:
    path_match = re.search(r"Path:\s*([^,]+),\s*Cost:", answer)
    if not path_match:
        raise ValueError(f"Could not parse SP path from answer: {answer!r}")
    nodes = [int(x) for x in re.findall(r"\d+", path_match.group(1))]
    if len(nodes) < 2:
        raise ValueError("SP path must contain at least 2 nodes.")
    return nodes[0], nodes[-1]


def parse_wis_weights(text: str) -> dict[int, int]:
    weights: dict[int, int] = {}
    # e.g., "Plot 0: 8", "Server 1: 6", etc.
    for idx, val in re.findall(r"\b(?:Plot|District|Server|Tower|Center|Node|House)\s+(\d+):\s*(\d+)", text):
        weights[int(idx)] = int(val)
    # fallback for formal map "0->8"
    for idx, val in re.findall(r"(\d+)\s*->\s*(\d+)", text):
        weights.setdefault(int(idx), int(val))
    return weights


def parse_bool(value: str, default: bool = False) -> bool:
    v = str(value).strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return default


def to_label(i: int) -> str:
    return f"Hub {chr(65 + i)}"


def build_canonical_structures(df: pd.DataFrame) -> dict[str, dict]:
    structures: dict[str, dict] = {}
    canon = df[df["variant_type"] == "canonical"]
    for _, row in canon.iterrows():
        pid = str(row["problem_id"])
        text = str(row["problem_text"])
        ans = str(row["correct_answer"])
        old = parse_key_equals_blob(str(row.get("difficulty_params", "")))

        if pid.startswith("CC_"):
            denoms, target = parse_cc_from_text(text)
            greedy_succeeds = parse_bool(old.get("greedy_succeeds"), default=False)
            instance_type = str(old.get("instance_type", "standard")).strip() or "standard"
            greedy_answer = str(old.get("greedy_answer", ans)).strip()
            structures[pid] = {
                "subtype": "CC",
                "denominations": denoms,
                "target": target,
                "greedy_succeeds": greedy_succeeds,
                "instance_type": instance_type,
                "greedy_answer": greedy_answer,
            }
        elif pid.startswith("SP_"):
            # Parse edges from canonical; fallback to W4 variant text.
            edges = parse_sp_edges(text)
            if not edges:
                w4_idx = require_single_row(df, pid, "W4")
                edges = parse_sp_edges(str(df.at[w4_idx, "problem_text"]))
            if not edges:
                raise ValueError(f"Unable to parse SP graph edges for {pid}.")
            src, tgt = parse_sp_src_tgt(ans)
            directed = parse_bool(old.get("directed"), default=True)
            n_vertices = int(old.get("num_vertices", max(max(u, v) for u, v, _ in edges) + 1))
            instance_type = str(old.get("instance_type", "standard")).strip() or "standard"
            greedy_answer = str(old.get("greedy_answer", ans)).strip()
            structures[pid] = {
                "subtype": "SP",
                "directed": directed,
                "source": src,
                "target": tgt,
                "graph": [{"u": u, "v": v, "w": w} for u, v, w in edges],
                "num_vertices": n_vertices,
                "instance_type": instance_type,
                "greedy_answer": greedy_answer,
            }
        elif pid.startswith("WIS_"):
            weights = parse_wis_weights(text)
            if not weights:
                w4_idx = require_single_row(df, pid, "W4")
                weights = parse_wis_weights(str(df.at[w4_idx, "problem_text"]))
            if not weights:
                raise ValueError(f"Unable to parse WIS weights for {pid}.")
            greedy_succeeds = parse_bool(old.get("greedy_succeeds"), default=False)
            instance_type = str(old.get("instance_type", "standard")).strip() or "standard"
            greedy_answer = str(old.get("greedy_answer", ans)).strip()
            intervals = [
                {"id": i, "start": 2 * i, "end": 2 * i + 1, "weight": int(weights[i])}
                for i in sorted(weights)
            ]
            structures[pid] = {
                "subtype": "WIS",
                "intervals": intervals,
                "greedy_succeeds": greedy_succeeds,
                "instance_type": instance_type,
                "greedy_answer": greedy_answer,
            }
    return structures


def map_sp_answer_with_mapping(answer: str, node_mapping: dict[str, str]) -> str:
    match = re.search(r"Path:\s*(.+?)\s*,\s*Cost:\s*(-?\d+)", answer)
    if not match:
        raise ValueError(f"Bad SP answer format: {answer!r}")
    nodes = [x.strip() for x in re.split(r"\s*[→\-]+>\s*|\s*→\s*", match.group(1)) if x.strip()]
    mapped = []
    for node in nodes:
        key = str(int(node))
        mapped.append(node_mapping[key])
    return f"Path: {' → '.join(mapped)}, Cost: {match.group(2)}"


def map_wis_answer_with_mapping(answer: str, item_mapping: dict[str, str]) -> str:
    match = re.search(r"Selected:\s*\{([^}]*)\}\s*,\s*Total:\s*(\d+)", answer)
    if not match:
        raise ValueError(f"Bad WIS answer format: {answer!r}")
    ids = [x.strip() for x in match.group(1).split(",") if x.strip()]
    mapped = [item_mapping[str(int(x))] for x in ids]
    return f"Selected: {{{', '.join(mapped)}}}, Total: {match.group(2)}"


def apply_targeted_fixes(df: pd.DataFrame, changes: dict[int, list[str]]) -> None:
    idx = require_single_row(df, "CC_04", "canonical")
    set_field(df, idx, "correct_answer", "Count: 2\nCoins: [17, 10]", changes)
    idx = require_single_row(df, "CC_04", "W1")
    set_field(df, idx, "correct_answer", "Count: 2\nCoins: [17, 10]", changes)
    idx = require_single_row(df, "CC_04", "W2")
    set_field(df, idx, "correct_answer", "Count: 2\nCoins: [17, 10]", changes)
    idx = require_single_row(df, "CC_04", "W3")
    set_field(df, idx, "correct_answer", "Total: 2\nScoops: [17, 10]", changes)

    idx = require_single_row(df, "CC_04", "W4")
    set_field(df, idx, "correct_answer", "Count: 2\nCoins: [17, 10]", changes)
    replace_one_of(
        df,
        idx,
        "problem_text",
        ("(CurrentSum + c) <= 143", r"(\text{CurrentSum} + c) \le 143"),
        "(CurrentSum + c) <= 27",
        changes,
    )
    replace_one_of(
        df,
        idx,
        "problem_text",
        ("reach CurrentSum = 143", r"reach $\text{CurrentSum} = 143$"),
        "reach CurrentSum = 27",
        changes,
    )

    idx = require_single_row(df, "CC_05", "W4")
    replace_one_of(
        df,
        idx,
        "problem_text",
        ("(CurrentSum + c) <= 1431", r"(\text{CurrentSum} + c) \le 1431"),
        "(CurrentSum + c) <= 14",
        changes,
    )
    replace_one_of(
        df,
        idx,
        "problem_text",
        ("reach CurrentSum = 1431", r"reach $\text{CurrentSum} = 1431$"),
        "reach CurrentSum = 14",
        changes,
    )

    for variant in ("canonical", "W1", "W2", "W4"):
        idx = require_single_row(df, "CC_07", variant)
        set_field(df, idx, "correct_answer", "Count: 2\nCoins: [8, 4]", changes)
    idx = require_single_row(df, "CC_07", "W3")
    set_field(df, idx, "correct_answer", "Total: 2\nScoops: [8, 4]", changes)
    replace_one_of(df, idx, "problem_text", ("2 g, 5 g",), "4 g, 5 g", changes)

    idx = require_single_row(df, "WIS_004", "W4")
    replace_one_of(df, idx, "problem_text", ("{4,11}",), "{4,1}", changes)
    replace_one_of(df, idx, "problem_text", ("{1,10}",), "{8,1}", changes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=str(DEFAULT_BANK))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bank_path = Path(args.bank)
    out_path = Path(args.output) if args.output else bank_path
    if not bank_path.exists():
        raise FileNotFoundError(f"Bank not found: {bank_path}")

    df = pd.read_csv(bank_path)
    required = {"problem_id", "variant_type", "problem_text", "correct_answer", "difficulty_params"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    changes: dict[int, list[str]] = defaultdict(list)

    apply_targeted_fixes(df, changes)

    structures = build_canonical_structures(df)
    for idx, row in df.iterrows():
        pid = str(row["problem_id"])
        vtype = str(row["variant_type"])
        if pid not in structures:
            continue
        payload = dict(structures[pid])
        if vtype == "W3":
            if pid.startswith("SP_"):
                n = int(payload.get("num_vertices", 0))
                payload["node_mapping"] = {str(i): to_label(i) for i in range(n)}
            elif pid.startswith("WIS_"):
                n = len(payload["intervals"])
                payload["item_mapping"] = {str(i): f"Item {chr(65 + i)}" for i in range(n)}
        new_params = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        set_field(df, idx, "difficulty_params", new_params, changes)

    # Normalize contamination pole casing and verifier function consistency.
    pole_map = {"low": "Low", "medium": "Medium", "high": "High"}
    for idx, row in df.iterrows():
        pole = str(row["contamination_pole"]).strip()
        if pole.lower() in pole_map:
            set_field(df, idx, "contamination_pole", pole_map[pole.lower()], changes)
        pid = str(row["problem_id"]).strip()
        if pid.startswith("CC_"):
            set_field(df, idx, "problem_subtype", "coin_change", changes)
        elif pid.startswith("SP_"):
            set_field(df, idx, "problem_subtype", "shortest_path", changes)
        elif pid.startswith("WIS_"):
            set_field(df, idx, "problem_subtype", "wis", changes)
    for pid, grp in df.groupby("problem_id"):
        canon_rows = grp[grp["variant_type"] == "canonical"]
        if len(canon_rows) != 1:
            continue
        canon_verifier = str(canon_rows.iloc[0]["verifier_function"])
        for idx in grp.index:
            set_field(df, idx, "verifier_function", canon_verifier, changes)

    # Enforce Option A globally: W3 SP/WIS answers must be in mapped label space.
    for idx, row in df.iterrows():
        if str(row["variant_type"]) != "W3":
            continue
        pid = str(row["problem_id"])
        params = json.loads(str(row["difficulty_params"]))
        if pid.startswith("SP_") and "node_mapping" in params:
            base_ans = str(df.at[require_single_row(df, pid, "canonical"), "correct_answer"])
            mapped = map_sp_answer_with_mapping(base_ans, params["node_mapping"])
            set_field(df, idx, "correct_answer", mapped, changes)
        if pid.startswith("WIS_") and "item_mapping" in params:
            base_ans = str(df.at[require_single_row(df, pid, "canonical"), "correct_answer"])
            mapped = map_wis_answer_with_mapping(base_ans, params["item_mapping"])
            set_field(df, idx, "correct_answer", mapped, changes)

    # Propagation fixes.
    sp005_answer = "Path: 0 → 4 → 9 → 11 → 14 → 15, Cost: 32"
    for variant in ("canonical", "W1", "W2", "W3", "W4"):
        idx = require_single_row(df, "SP_005", variant)
        if variant == "W3":
            params = json.loads(str(df.at[idx, "difficulty_params"]))
            mapped = map_sp_answer_with_mapping(sp005_answer, params["node_mapping"])
            set_field(df, idx, "correct_answer", mapped, changes)
        else:
            set_field(df, idx, "correct_answer", sp005_answer, changes)

    wis005_answer = "Selected: {2, 5, 7, 8, 11, 14}, Total: 67"
    for variant in ("canonical", "W1", "W2", "W3", "W4"):
        idx = require_single_row(df, "WIS_005", variant)
        if variant == "W3":
            params = json.loads(str(df.at[idx, "difficulty_params"]))
            mapped = map_wis_answer_with_mapping(wis005_answer, params["item_mapping"])
            set_field(df, idx, "correct_answer", mapped, changes)
        else:
            set_field(df, idx, "correct_answer", wis005_answer, changes)

    # Manual flag: enforce expected contamination for WIS_001 W3, then assert.
    idx = require_single_row(df, "WIS_001", "W3")
    set_field(df, idx, "contamination_pole", "High", changes)

    # Assertions.
    idx = require_single_row(df, "WIS_001", "W3")
    if str(df.at[idx, "contamination_pole"]) != "High":
        raise AssertionError("WIS_001 W3 contamination_pole must be 'High'.")

    idx = require_single_row(df, "WIS_004", "W4")
    text = str(df.at[idx, "problem_text"])
    if "{4,1}" not in text or "{8,1}" not in text:
        raise AssertionError("WIS_004 W4 required edge corrections missing.")

    sp005_canonical = json.loads(str(df.at[require_single_row(df, "SP_005", "canonical"), "difficulty_params"]))
    sp005_w2 = json.loads(str(df.at[require_single_row(df, "SP_005", "W2"), "difficulty_params"]))
    if sp005_w2.get("graph") != sp005_canonical.get("graph"):
        raise AssertionError("SP_005 W2 graph must match SP_005 canonical graph.")

    # Also enforce SP_001 W3 mapped answer per Option A.
    idx = require_single_row(df, "SP_001", "W3")
    params = json.loads(str(df.at[idx, "difficulty_params"]))
    base_ans = str(df.at[require_single_row(df, "SP_001", "canonical"), "correct_answer"])
    mapped_ans = map_sp_answer_with_mapping(base_ans, params["node_mapping"])
    set_field(df, idx, "correct_answer", mapped_ans, changes)
    mapped_answer = str(df.at[idx, "correct_answer"])
    path_part = re.search(r"Path:\s*(.+?)\s*,\s*Cost:", mapped_answer)
    if not path_part or re.search(r"\d", path_part.group(1)):
        raise AssertionError("SP_001 W3 mapping failed: path must not contain digits.")

    df.to_csv(out_path, index=False)

    print(f"Applied fixes to: {out_path}")
    print(f"Rows changed: {len(changes)}")
    for row_idx in sorted(changes):
        print(f"- {df.at[row_idx, 'problem_id']} / {df.at[row_idx, 'variant_type']}")
        for item in changes[row_idx]:
            print(f"  * {item}")


if __name__ == "__main__":
    main()
