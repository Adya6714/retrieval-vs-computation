from __future__ import annotations

from dataclasses import dataclass
import csv
import difflib
import random
import re
from pathlib import Path


@dataclass
class PddlProblem:
    name: str
    domain: str
    objects: list[str]
    init_atoms: list[tuple[str, list[str]]]
    goal_atoms: list[tuple[str, list[str]]]


def _strip_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _extract_block(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return ""
    start = text.find("(", idx)
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _extract_simple(marker: str, text: str) -> str:
    m = re.search(marker, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _parse_atoms(block: str) -> list[tuple[str, list[str]]]:
    atoms: list[tuple[str, list[str]]] = []
    for pred, args in re.findall(r"\(\s*([^\s()]+)([^()]*)\)", block):
        p = pred.strip().lower()
        if p in {"and", ":init", ":goal"}:
            continue
        arg_list = [a.strip().lower() for a in args.split() if a.strip()]
        atoms.append((p, arg_list))
    return atoms


def parse_problem(problem_path: Path) -> PddlProblem:
    raw = problem_path.read_text(encoding="utf-8")
    text = _strip_comments(raw)
    name = _extract_simple(r"\(define\s+\(problem\s+([^)]+)\)", text) or problem_path.stem
    domain = _extract_simple(r"\(:domain\s+([^)]+)\)", text)

    objects_block = _extract_block(text, "(:objects")
    objects = []
    if objects_block:
        body = re.sub(r"^\(\s*:objects", "", objects_block, flags=re.IGNORECASE).rstrip(")")
        objects = [tok.lower() for tok in body.split() if tok and tok != "-"]

    init_block = _extract_block(text, "(:init")
    goal_block = _extract_block(text, "(:goal")
    init_atoms = _parse_atoms(init_block)
    goal_atoms = _parse_atoms(goal_block)
    return PddlProblem(
        name=name.lower(),
        domain=domain.lower(),
        objects=objects,
        init_atoms=init_atoms,
        goal_atoms=goal_atoms,
    )


def _oxford(items: list[str], prefix: str = "") -> str:
    if not items:
        return ""
    rendered = [f"{prefix}{x}" for x in items]
    if len(rendered) == 1:
        return rendered[0]
    if len(rendered) == 2:
        return f"{rendered[0]} and {rendered[1]}"
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}"


def _sentence_join(parts: list[str]) -> str:
    return ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 1 else parts[0]


def render_bw_prompt(problem: PddlProblem) -> str:
    on_pairs = [(a[0], a[1]) for p, a in problem.init_atoms if p == "on" and len(a) == 2]
    ontable = sorted(a[0] for p, a in problem.init_atoms if p == "ontable" and len(a) == 1)
    clear = set(a[0] for p, a in problem.init_atoms if p == "clear" and len(a) == 1)
    handempty = any(p == "handempty" for p, _ in problem.init_atoms)

    state_parts: list[str] = []
    if ontable:
        clear_on_table = [b for b in ontable if b in clear]
        if clear_on_table:
            state_parts.append(f"Blocks {_oxford(clear_on_table)} are clear and on the table")
        else:
            state_parts.append(f"Blocks {_oxford(ontable)} are on the table")
    if on_pairs:
        for x, y in on_pairs:
            state_parts.append(f"block {x} is on block {y}")
    if handempty:
        state_parts.append("the hand is empty")
    current_state = _sentence_join(state_parts).capitalize() + "."

    goal_on = [(a[0], a[1]) for p, a in problem.goal_atoms if p == "on" and len(a) == 2]
    goal_parts = [f"block {x} is on block {y}" for x, y in goal_on]
    goal_text = _sentence_join(goal_parts).capitalize() + "."

    return (
        "You are a robot arm. Available actions: pick-up X (X must be clear and on the table, hand must be empty), "
        "put-down X (place X on the table), stack X Y (place X on Y; Y must be clear, you must be holding X), "
        "unstack X Y (pick up X from Y; X must be clear, hand must be empty). You can hold one block at a time. "
        f"Current state: {current_state} Goal: {goal_text} "
        "Respond with a numbered list of actions only. Each action must be exactly one of: pick-up X / put-down X / "
        "stack X Y / unstack X Y. No explanation. No extra text."
    )


def render_mbw_prompt(problem: PddlProblem) -> str:
    harmony = any(p == "harmony" for p, _ in problem.init_atoms)
    province = sorted(a[0] for p, a in problem.init_atoms if p == "province" and len(a) == 1)
    planet = set(a[0] for p, a in problem.init_atoms if p == "planet" and len(a) == 1)
    pain = sorted(a[0] for p, a in problem.init_atoms if p == "pain" and len(a) == 1)
    init_craves = [(a[0], a[1]) for p, a in problem.init_atoms if p == "craves" and len(a) == 2]
    goal_craves = [(a[0], a[1]) for p, a in problem.goal_atoms if p == "craves" and len(a) == 2]

    init_parts: list[str] = []
    if harmony:
        init_parts.append("harmony is true")
    paired = [b for b in province if b in planet]
    if paired:
        init_parts.append(f"planet and province are true for blocks {_oxford(paired)}")
    if pain:
        init_parts.append(f"pain is true for blocks {_oxford(pain)}")
    if init_craves:
        init_parts.append(
            f"craves facts {_oxford([f'{x} {y}' for x, y in init_craves], prefix='')} are true"
        )
    init_text = _sentence_join(init_parts).capitalize() + "."

    goal_items = [f"craves {x} {y}" for x, y in goal_craves]
    goal_text = _sentence_join(goal_items).capitalize() + " are true."

    return (
        "You are a robot arm. Available actions: attack X (requires harmony, province X, planet X to be true), "
        "succumb X (requires pain X to be true), overcome X Y (requires province Y and pain X to be true), "
        "feast X Y (requires craves X Y, province X, harmony to be true). "
        f"Current state: {init_text} Goal: {goal_text} "
        "Respond with a numbered list of actions only. Each action must be exactly one of: attack X / succumb X / "
        "overcome X Y / feast X Y. No explanation. No extra text."
    )


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def _normalize_sentence_structure(sentence: str) -> str:
    s = sentence.lower().strip().strip('"')
    s = re.sub(r"\b\d+\b", "<num>", s)
    # BW/MBW entities in this bank are typically single-letter symbols.
    s = re.sub(r"\b[a-z]\b", "<ent>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pattern_lines(problem_text: str) -> list[str]:
    return [_normalize_sentence_structure(s) for s in _split_sentences(problem_text)]


def _pattern_signature(problem_text: str) -> list[str]:
    signature: list[str] = []
    for s in _pattern_lines(problem_text):
        if s.startswith("you are <ent> robot arm."):
            signature.append("INTRO")
        elif s.startswith("available actions:"):
            signature.append("ACTIONS")
        elif s.startswith("you can hold one block at <ent> time."):
            signature.append("CAPACITY")
        elif s.startswith("current state:"):
            signature.append("CURRENT_STATE")
        elif s == "the hand is empty.":
            signature.append("CURRENT_DETAIL")
        elif re.match(r"^block <ent> is clear\.$", s):
            signature.append("CURRENT_DETAIL")
        elif s.startswith("goal:"):
            signature.append("GOAL_STATE")
        elif s.startswith("respond with <ent> numbered list of actions only."):
            signature.append("OUTPUT_REQ")
        elif s.startswith("each action must be exactly one of:"):
            signature.append("ACTION_SCHEMA")
        elif s == "no explanation.":
            signature.append("NO_EXPLANATION")
        elif s == "no extra text.":
            signature.append("NO_EXTRA_TEXT")
        else:
            signature.append(f"OTHER:{s}")
    return signature


def _compressed_signature(signature: list[str]) -> list[str]:
    # Allow legacy rows that split current-state details into extra sentences.
    return [tok for tok in signature if tok != "CURRENT_DETAIL"]


def verify_nl_format_matches_bank(generated_text: str, bank_csv_path: str) -> bool:
    """One-time sanity check that generated NL prompt matches BW bank structure."""
    csv_path = Path(bank_csv_path)
    if not csv_path.exists():
        print(f"[format-check] missing bank CSV: {csv_path}")
        return False

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("variant_type", "")).strip().lower() != "canonical":
                continue
            if str(row.get("problem_subtype", "")).strip().lower() != "blocksworld":
                continue
            text = str(row.get("problem_text", "")).strip()
            if text:
                rows.append(row)

    if len(rows) < 3:
        print(f"[format-check] not enough canonical BW rows in {csv_path} (need >= 3)")
        return False

    sample_rows = random.Random(42).sample(rows, 3)
    expected_patterns = [_pattern_lines(r["problem_text"]) for r in sample_rows]
    expected_signatures = [_compressed_signature(_pattern_signature(r["problem_text"])) for r in sample_rows]
    generated_pattern = _pattern_lines(generated_text)
    generated_signature = _compressed_signature(_pattern_signature(generated_text))

    for expected in expected_signatures:
        if generated_signature == expected:
            return True

    print("[format-check] generated problem_text structure does not match sampled bank rows.")
    for idx, expected in enumerate(expected_patterns, start=1):
        print(f"\n--- expected sample {idx} ---")
        print("\n".join(expected))
        print(f"--- generated ---")
        print("\n".join(generated_pattern))
        print(f"--- expected signature {idx} ---")
        print("\n".join(expected_signatures[idx - 1]))
        print("--- generated signature ---")
        print("\n".join(generated_signature))
        print("--- diff ---")
        diff = difflib.unified_diff(
            expected_signatures[idx - 1],
            generated_signature,
            fromfile=f"bank_sample_{idx}",
            tofile="generated_text",
            lineterm="",
        )
        for line in diff:
            print(line)
    return False
