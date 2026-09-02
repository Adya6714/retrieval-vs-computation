#!/usr/bin/env python3
"""L3: Quantify oracle bias direction for verifier defect fixes."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

csv.field_size_limit(sys.maxsize)

from probes.common.variants import normalize_variant  # noqa: E402
from probes.contamination.verify import parse_action_mapping_from_notes, verify_answer  # noqa: E402
from probes.contamination.verify_algo import verify_algo  # noqa: E402

RAW = REPO_ROOT / "results/raw"
DER = REPO_ROOT / "results/derived"
OUT = DER / "oracle_bias_summary.csv"

BANKS = {
    "ALGO": REPO_ROOT / "data/problems/question_bank_algo.csv",
    "BW": REPO_ROOT / "data/problems/question_bank_bw.csv",
}


def _parse_blocksworld_state_legacy(problem_text: str) -> tuple[set[tuple], set[tuple]] | None:
    text = str(problem_text)
    current_match = re.search(r"current state:\s*(.*?)\s*goal:", text, re.IGNORECASE | re.DOTALL)
    goal_match = re.search(r"goal:\s*(.*?)(?:respond with|$)", text, re.IGNORECASE | re.DOTALL)
    if not current_match or not goal_match:
        return None
    current = current_match.group(1).lower()
    goal = goal_match.group(1).lower()
    state: set[tuple] = set()
    goal_facts: set[tuple] = set()
    m = re.search(r"blocks?\s+(.+?)\s+are clear and on the table", current)
    if m:
        blocks = [b.strip() for b in re.split(r",| and ", m.group(1)) if b.strip()]
        for b in blocks:
            state.add(("clear", b))
            state.add(("ontable", b))
    for b in re.findall(r"block\s+([a-z0-9]+)\s+is clear and on the table", current):
        state.add(("clear", b))
        state.add(("ontable", b))
    for x, y in re.findall(r"block\s+([a-z0-9]+)\s+is on block\s+([a-z0-9]+)", current):
        state.add(("on", x, y))
    if "hand is empty" in current:
        state.add(("handempty",))
    for x, y in re.findall(r"block\s+([a-z0-9]+)\s+is on block\s+([a-z0-9]+)", goal):
        goal_facts.add(("on", x, y))
    for b in re.findall(r"block\s+([a-z0-9]+)\s+is on the table", goal):
        goal_facts.add(("ontable", b))
    for b in re.findall(r"block\s+([a-z0-9]+)\s+is clear", goal):
        goal_facts.add(("clear", b))
    if not state and not goal_facts:
        return None
    return state, goal_facts


def _load_bank() -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for fam, path in BANKS.items():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (
                    fam,
                    str(row.get("problem_id", "")).strip(),
                    normalize_variant(row.get("variant_type", "")),
                )
                out[key] = row
    return out


def _model_answer(raw: dict[str, str]) -> str:
    for col in ("raw_response", "model_answer", "model_response", "response", "answer"):
        if raw.get(col):
            return str(raw[col])
    return ""


def _bool_col(val: object) -> bool | None:
    s = str(val or "").strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _score_algo_mapping_only(
    raw: dict[str, str], bank: dict[str, str], *, mode: str
) -> bool | None:
    import probes.contamination.verify_algo as va

    def _params_only(params, notes=None, problem_text=None):
        existing = va._parse_params(params).get("node_mapping") or {}
        if isinstance(existing, dict):
            return {str(k): str(v) for k, v in existing.items()}
        return {}

    kwargs = dict(
        problem_id=raw.get("problem_id", ""),
        model_answer=_model_answer(raw),
        ground_truth=bank.get("correct_answer", ""),
        problem_subtype=bank.get("problem_subtype", ""),
        variant_type=normalize_variant(raw.get("variant_type", "")),
        difficulty_params=bank.get("difficulty_params", ""),
        notes=bank.get("notes"),
        problem_text=bank.get("problem_text"),
    )
    if mode == "before":
        with patch("probes.contamination.verify_algo.resolve_sp_node_mapping", side_effect=_params_only):
            ok, _, _ = verify_algo(**kwargs)
    else:
        ok, _, _ = verify_algo(**kwargs)
    return ok


def _score_bw(
    raw: dict[str, str],
    bank: dict[str, str],
    *,
    use_action_mapping: bool,
    parser_mode: str,
) -> bool | None:
    import probes.contamination.verify as vmod

    mapping = parse_action_mapping_from_notes(bank.get("notes")) if use_action_mapping else None
    parser = (
        _parse_blocksworld_state_legacy
        if parser_mode == "before"
        else vmod._parse_blocksworld_state
    )
    with patch.object(vmod, "_parse_blocksworld_state", parser):
        ok = verify_answer(
            raw.get("problem_id", ""),
            _model_answer(raw),
            bank.get("correct_answer", ""),
            "blocksworld",
            problem_text=bank.get("problem_text", ""),
            action_mapping=mapping,
        )
    return ok


def _load_raw_rows() -> list[dict]:
    rows: list[dict] = []
    for pattern in ("BW_P1_behavioral*.csv", "ALGO_P1_behavioral_*.csv"):
        for path in sorted(RAW.glob(pattern)):
            if "review" in path.name.lower():
                continue
            fam = "ALGO" if path.name.startswith("ALGO_") else "BW"
            with path.open(newline="", encoding="utf-8") as f:
                for raw in csv.DictReader(f):
                    raw["_family"] = fam
                    rows.append(raw)
    return rows


def _load_rescored_sp() -> list[tuple[bool | None, bool | None, str]]:
    """Full J1 SP W3 repair: raw verified vs rescored_correct (mapping + Path line)."""
    import pandas as pd

    out: list[tuple[bool | None, bool | None, str]] = []
    bank = _load_bank()
    for path in sorted(DER.glob("ALGO_P1_*_rescored.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[df["included"].str.lower().eq("true")]
        df = df[df["variant_type"].map(normalize_variant).eq("W3")]
        for _, row in df.iterrows():
            pid = str(row["problem_id"])
            key = ("ALGO", pid, "W3")
            if key not in bank:
                continue
            if str(bank[key].get("problem_subtype", "")).lower() != "shortest_path":
                continue
            before = _bool_col(row.get("old_verified", row.get("verified", "")))
            after = _bool_col(row.get("rescored_correct", ""))
            if before is None:
                before = _bool_col(row.get("behavioral_correct", ""))
            out.append((before, after, "W3"))
        # canonical SP rows from same files
        df2 = pd.read_csv(path, dtype=str).fillna("")
        df2 = df2[df2["included"].str.lower().eq("true")]
        df2 = df2[df2["variant_type"].map(normalize_variant).eq("canonical")]
        for _, row in df2.iterrows():
            pid = str(row["problem_id"])
            key = ("ALGO", pid, "canonical")
            if key not in bank:
                continue
            if str(bank[key].get("problem_subtype", "")).lower() != "shortest_path":
                continue
            before = _bool_col(row.get("old_verified", row.get("verified", "")))
            after = _bool_col(row.get("rescored_correct", ""))
            out.append((before, after, "canonical"))
    return out


def _summarize(
    defect: str,
    family: str,
    variant_scope: str,
    rows: list[tuple[bool | None, bool | None, str]],
    note: str = "",
) -> dict:
    pert = [r for r in rows if r[2] != "canonical"]
    can = [r for r in rows if r[2] == "canonical"]
    if defect in {"SP_W3_node_mapping", "BW_W3_action_mapping"}:
        pert = [r for r in rows if r[2] == "W3"]
        can = [r for r in rows if r[2] == "canonical"]
    affected = [r for r in rows if r[0] is not None and r[1] is not None and r[0] != r[1]]

    def _acc(group: list[tuple], idx: int) -> float | None:
        vals = [r[idx] for r in group if r[idx] is not None]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    before_p, after_p = _acc(pert, 0), _acc(pert, 1)
    before_c, after_c = _acc(can, 0), _acc(can, 1)
    delta_p = None if before_p is None or after_p is None else after_p - before_p
    delta_c = None if before_c is None or after_c is None else after_c - before_c
    sign = ""
    if delta_p is not None:
        sign = "up" if delta_p > 0 else ("down" if delta_p < 0 else "flat")
    return {
        "defect": defect,
        "family": family,
        "variant_scope": variant_scope,
        "rows_affected": len(affected),
        "rows_scored": len(rows),
        "perturbed_n": len(pert),
        "canonical_n": len(can),
        "acc_before_perturbed": round(before_p, 4) if before_p is not None else "",
        "acc_after_perturbed": round(after_p, 4) if after_p is not None else "",
        "delta_perturbed": round(delta_p, 4) if delta_p is not None else "",
        "acc_before_canonical": round(before_c, 4) if before_c is not None else "",
        "acc_after_canonical": round(after_c, 4) if after_c is not None else "",
        "delta_canonical": round(delta_c, 4) if delta_c is not None else "",
        "sign_perturbed": sign,
        "false_to_true": sum(1 for b, a, _ in affected if b is False and a is True),
        "true_to_false": sum(1 for b, a, _ in affected if b is True and a is False),
        "note": note,
    }


def main() -> None:
    bank = _load_bank()
    bw_map_rows: list[tuple[bool | None, bool | None, str]] = []
    bw_parser_rows: list[tuple[bool | None, bool | None, str]] = []

    for raw in _load_raw_rows():
        fam = raw["_family"]
        pid = str(raw.get("problem_id", "")).strip()
        variant = normalize_variant(raw.get("variant_type", ""))
        key = (fam, pid, variant)
        if key not in bank or not _model_answer(raw).strip():
            continue
        brow = bank[key]
        if fam == "BW":
            if str(brow.get("problem_subtype", "")).lower() != "blocksworld":
                continue
            if pid.upper().startswith("MBW"):
                continue
            b_map = _score_bw(raw, brow, use_action_mapping=False, parser_mode="after")
            a_map = _score_bw(raw, brow, use_action_mapping=True, parser_mode="after")
            if variant in {"W3", "canonical"}:
                bw_map_rows.append((b_map, a_map, variant))
            bw_parser_rows.append(
                (
                    _score_bw(raw, brow, use_action_mapping=True, parser_mode="before"),
                    _score_bw(raw, brow, use_action_mapping=True, parser_mode="after"),
                    variant,
                )
            )

    sp_rescore_rows = _load_rescored_sp()
    out_rows = [
        _summarize(
            "SP_W3_node_mapping",
            "ALGO",
            "W3_shortest_path",
            sp_rescore_rows,
            note="verified→rescored_correct; includes notes mapping + trailing Path preference (J1)",
        ),
        _summarize(
            "BW_W3_action_mapping",
            "BW",
            "W3_blocksworld",
            bw_map_rows,
            note="action_mapping=None vs notes mapping; canonical unchanged",
        ),
        _summarize(
            "BW_state_parser",
            "BW",
            "all_blocksworld_variants",
            bw_parser_rows,
            note="legacy regex parser vs released prose/table parser",
        ),
    ]
    DER.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {OUT}")
    for r in out_rows:
        print(r)


if __name__ == "__main__":
    main()
