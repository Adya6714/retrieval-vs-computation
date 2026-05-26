"""Shared helpers for W3/W5/W6 variant generation (substitutions, PDDL, Fast Downward)."""

from __future__ import annotations

import random
import re
import subprocess
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FD_SCRIPT = _REPO_ROOT / "tools" / "fast-downward" / "fast-downward.py"
_BW_DOMAIN_CANDIDATES = [
    _REPO_ROOT
    / "data/sources/planbench/plan-bench/instances/blocksworld/generated/domain.pddl",
    _REPO_ROOT
    / "data/sources/planbench/plan-bench/instances/blocksworld/generated_domain.pddl",
    _REPO_ROOT
    / "data/sources/planbench/llm_planning_analysis/instances/blocksworld/generated_domain.pddl",
]

_BW_DOMAIN_CACHE: str | None = None


def build_substitution_regex(mapping: dict) -> re.Pattern:
    keys = sorted(mapping.keys(), key=len, reverse=True)
    escaped = [re.escape(k) for k in keys]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern)


# WHAT THIS DOES (build_substitution_regex):
# Builds one regex that finds whole words to replace (longer words like
# "unstack" are tried before shorter ones like "stack" so they don't overlap).


def apply_mapping(text: str, mapping: dict) -> str:
    if not mapping:
        return text
    pattern = build_substitution_regex(mapping)
    return pattern.sub(lambda m: mapping[m.group(0)], text)


# WHAT THIS DOES (apply_mapping):
# Replaces every mapped word in the text in a single pass, so shorter
# keys cannot corrupt longer words (e.g. replacing "a" inside "stack").


def make_inverse_mapping(mapping: dict) -> dict:
    return {v: k for k, v in mapping.items()}


# WHAT THIS DOES (make_inverse_mapping):
# Swaps keys and values so renamed tokens can be mapped back to originals.


def verify_w3_roundtrip(
    w3_text: str, canonical_text: str, full_mapping: dict
) -> tuple[bool, str]:
    inverse = make_inverse_mapping(full_mapping)
    recovered = apply_mapping(w3_text, inverse)
    normalized_recovered = " ".join(recovered.split())
    normalized_canonical = " ".join(canonical_text.split())
    if normalized_recovered != normalized_canonical:
        return False, "roundtrip_mismatch"

    if full_mapping:
        time_math_words = {
            "months",
            "month",
            "year",
            "years",
            "day",
            "days",
            "week",
            "weeks",
            "hour",
            "hours",
            "minute",
            "minutes",
            "second",
            "seconds",
            "percent",
            "percentage",
            "per",
            "twice",
            "triple",
            "quadruple",
            "half",
            "quarter",
        }
        source_pattern = build_substitution_regex(full_mapping)
        found = sorted(set(source_pattern.findall(w3_text)))
        domain_survivors = [s for s in found if s.lower() not in time_math_words]
        if domain_survivors:
            return False, f"source_tokens_survived: {set(domain_survivors)}"

    return True, "ok"


# WHAT THIS DOES (verify_w3_roundtrip):
# Checks that W3 text can be inverted to match the canonical text exactly,
# and that no original (pre-rename) tokens are still present in W3.


def run_fast_downward(
    domain_pddl: str, problem_pddl: str, timeout: int = 30
) -> tuple[str | None, str]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            domain_path = tmp / "domain.pddl"
            problem_path = tmp / "problem.pddl"
            plan_path = tmp / "plan.txt"
            domain_path.write_text(domain_pddl, encoding="utf-8")
            problem_path.write_text(problem_pddl, encoding="utf-8")

            cmd = [
                "python",
                str(_FD_SCRIPT),
                "--plan-file",
                str(plan_path),
                str(domain_path),
                str(problem_path),
                "--search",
                "astar(lmcut())",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(_REPO_ROOT),
                )
            except subprocess.TimeoutExpired:
                return None, "timeout"

            if plan_path.exists():
                action_lines = []
                for line in plan_path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith(";"):
                        action_lines.append(stripped)
                return "\n".join(action_lines), "ok"

            stderr = result.stderr or ""
            return None, f"no_plan:{stderr[-300:]}"
    except Exception as exc:
        return None, f"error:{str(exc)}"


# WHAT THIS DOES (run_fast_downward):
# Writes domain and problem PDDL to a temp folder, runs Fast Downward,
# and returns the plan lines (or an error tag if planning failed).


def w6_seed(problem_id: str) -> int:
    return hash(problem_id + "W6") % (2**32)


# WHAT THIS DOES (w6_seed):
# Turns a problem_id into a fixed random seed so W6 generation is repeatable.


def _extract_pddl_block(text: str, marker: str) -> str:
    idx = text.lower().find(marker.lower())
    if idx < 0:
        raise ValueError(f"Missing {marker} section in PDDL")
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
    raise ValueError(f"Unbalanced parentheses in PDDL near {marker}")


def _section_body(section_block: str, keyword: str) -> str:
    m = re.match(
        rf"\(\s*:{keyword}\s+(.*)\)\s*$",
        section_block.strip(),
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Could not parse :{keyword} section")
    return m.group(1).strip()


def _strip_goal_and_wrapper(goal_body: str) -> str:
    body = goal_body.strip()
    m = re.match(r"\(\s*and\s+(.*)\)\s*$", body, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return body


def _wrap_init_as_goal(init_body: str) -> str:
    body = init_body.strip()
    if re.match(r"\(\s*and\s", body, flags=re.IGNORECASE):
        return body
    return f"(and {body})"


def extract_section(
    text: str, keyword: str
) -> tuple[int | None, int | None, str | None]:
    start = text.find(f"(:{keyword}")
    if start == -1:
        return None, None, None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1, text[start : i + 1]
    return None, None, None


def derive_full_bw_init(on_pairs: list[tuple[str, str]], all_blocks: list[str]) -> str:
    has_something_below = {upper for upper, lower in on_pairs}
    has_something_above = {lower for upper, lower in on_pairs}

    on_table = sorted([b for b in all_blocks if b not in has_something_below])
    clear = sorted([b for b in all_blocks if b not in has_something_above])

    lines = []
    for upper, lower in on_pairs:
        lines.append(f"(on {upper} {lower})")
    for b in on_table:
        lines.append(f"(ontable {b})")
    for b in clear:
        lines.append(f"(clear {b})")
    lines.append("(handempty)")

    return "\n".join(lines)


# WHAT THIS DOES (derive_full_bw_init):
# In Blocksworld, the goal only says which blocks go on which other blocks. But the
# actual starting state needs every block accounted for — on-table, clear, handempty.
# This function derives all those missing facts from the tower structure.


def parse_on_pairs_from_init(init_content: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\(on\s+(\w+)\s+(\w+)\)")
    return pattern.findall(init_content)


# WHAT THIS DOES (parse_on_pairs_from_init):
# Extract all (on x y) pairs from a PDDL :init or :goal content string.


def parse_all_blocks_from_objects(pddl_text: str) -> list[str]:
    match = re.search(r"\(:objects\s+(.*?)\)", pddl_text, re.DOTALL)
    if not match:
        return []
    return match.group(1).split()


# WHAT THIS DOES (parse_all_blocks_from_objects):
# Read block names from the (:objects ...) line in a PDDL problem file.


def _strip_outer_section(section: str, keyword: str) -> str:
    inner = section[len(f"(:{keyword}") :].strip()
    if inner.endswith(")"):
        inner = inner[:-1].strip()
    return inner


def inspect_w5_goal_tower(
    pddl_text: str,
) -> tuple[list[tuple[str, str]], list[str], str]:
    """Return (on_pairs, all_blocks, derived_init_content) for W5 diagnostics."""
    all_blocks = parse_all_blocks_from_objects(pddl_text)
    _start, _end, goal_section = extract_section(pddl_text, "goal")
    if not goal_section:
        return [], all_blocks, ""
    goal_inner = _strip_outer_section(goal_section, "goal")
    if goal_inner.strip().startswith("(and"):
        goal_content = _strip_goal_and_wrapper(goal_inner)
    else:
        goal_content = goal_inner.strip()
    on_pairs = parse_on_pairs_from_init(goal_content)
    return on_pairs, all_blocks, derive_full_bw_init(on_pairs, all_blocks)


def swap_pddl_init_goal(pddl_text: str) -> str:
    init_start, init_end, init_section = extract_section(pddl_text, "init")
    goal_start, goal_end, goal_section = extract_section(pddl_text, "goal")

    if init_section is None or goal_section is None:
        raise ValueError("Could not find :init or :goal in PDDL")

    all_blocks = parse_all_blocks_from_objects(pddl_text)

    goal_inner = _strip_outer_section(goal_section, "goal")
    if goal_inner.strip().startswith("(and"):
        goal_content = _strip_goal_and_wrapper(goal_inner)
    else:
        goal_content = goal_inner.strip()

    on_pairs = parse_on_pairs_from_init(goal_content)
    new_init_content = derive_full_bw_init(on_pairs, all_blocks)

    init_content = _strip_outer_section(init_section, "init")

    new_init_section = f"(:init\n{new_init_content})"
    new_goal_section = f"(:goal (and\n{init_content}\n))"

    return (
        pddl_text[:init_start]
        + new_init_section
        + pddl_text[init_end:goal_start]
        + new_goal_section
        + pddl_text[goal_end:]
    )


# WHAT THIS DOES (swap_pddl_init_goal):
# W5 reversal: start from the original goal tower (full init facts), plan to the
# original flat init state. Derives ontable/clear/handempty for a valid :init.


def _english_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _parse_bw_facts(facts_text: str) -> tuple[set[str], dict[str, str], set[str], bool]:
    on_table: set[str] = set()
    on: dict[str, str] = {}
    clear: set[str] = set()
    handempty = False

    for pred, args_blob in re.findall(r"\(\s*([^\s()]+)([^()]*)\)", facts_text):
        p = pred.strip().lower()
        args = [a.strip() for a in args_blob.split() if a.strip()]
        if p == "ontable" and len(args) == 1:
            on_table.add(args[0])
        elif p == "on" and len(args) == 2:
            on[args[0]] = args[1]
        elif p == "clear" and len(args) == 1:
            clear.add(args[0])
        elif p in {"handempty", "arm-empty"}:
            handempty = True

    return on_table, on, clear, handempty


def _positions_from_facts(
    on_table: set[str], on: dict[str, str]
) -> dict[str, str]:
    positions: dict[str, str] = {b: "table" for b in on_table}
    positions.update(on)
    return positions


def _describe_init_from_facts(on_table: set[str], on: dict[str, str], handempty: bool) -> str:
    positions = _positions_from_facts(on_table, on)
    if positions and all(v == "table" for v in positions.values()):
        sentence = (
            f"Blocks {_english_list(sorted(positions.keys()))} are clear and on the table."
        )
        if handempty:
            sentence += " The hand is empty."
        return sentence

    sentences: list[str] = []
    for block in sorted(positions.keys()):
        support = positions[block]
        if support == "table":
            sentences.append(f"Block {block} is on the table.")
        else:
            sentences.append(f"Block {block} is on block {support}.")
    if handempty:
        sentences.append("The hand is empty.")
    return " ".join(sentences)


def _describe_goal_from_facts(on_table: set[str], on: dict[str, str]) -> str:
    positions = _positions_from_facts(on_table, on)
    clauses: list[str] = []
    for block in sorted(positions.keys()):
        support = positions[block]
        if support == "table":
            clauses.append(f"block {block} is on the table")
        else:
            clauses.append(f"block {block} is on block {support}")
    if not clauses:
        return ""
    joined = ", ".join(clauses)
    return joined[0].upper() + joined[1:] + "."


def pddl_to_natural_language(pddl_text: str, num_blocks: int) -> str:
    init_block = _extract_pddl_block(pddl_text, "(:init")
    goal_block = _extract_pddl_block(pddl_text, "(:goal")
    init_body = _section_body(init_block, "init")
    goal_body = _section_body(goal_block, "goal")
    goal_facts = _strip_goal_and_wrapper(goal_body)

    init_on_table, init_on, _init_clear, init_handempty = _parse_bw_facts(init_body)
    goal_on_table, goal_on, _goal_clear, _goal_handempty = _parse_bw_facts(goal_facts)

    init_desc = _describe_init_from_facts(init_on_table, init_on, init_handempty)
    goal_desc = _describe_goal_from_facts(goal_on_table, goal_on)

    _ = num_blocks  # reserved for callers that pass expected block count

    return (
        "You are a robot arm. Available actions: pick-up X (X must be clear and on "
        "the table, hand must be empty), put-down X (place X on the table), stack X Y "
        "(place X on Y; Y must be clear, you must be holding X), unstack X Y (pick up "
        "X from Y; X must be clear, hand must be empty). You can hold one block at a "
        f"time. Current state: {init_desc} Goal: {goal_desc} "
        "Respond with a numbered list of actions only. Each action must be exactly one "
        "of: pick-up X / put-down X / stack X Y / unstack X Y. No explanation. No "
        "extra text."
    )


# WHAT THIS DOES (pddl_to_natural_language):
# Converts PDDL init/goal facts into the same English prompt style as bw_canonical.csv.


def fd_plan_to_bw_format(fd_output: str) -> str:
    lines: list[str] = []
    for raw in fd_output.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        line = line.strip("()").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


# WHAT THIS DOES (fd_plan_to_bw_format):
# Strips parentheses from Fast Downward plan lines to match bank answer format.


def load_bw_domain() -> str:
    global _BW_DOMAIN_CACHE
    if _BW_DOMAIN_CACHE is not None:
        return _BW_DOMAIN_CACHE
    domain_path = next((p for p in _BW_DOMAIN_CANDIDATES if p.exists()), None)
    if domain_path is None:
        tried = "\n  ".join(str(p) for p in _BW_DOMAIN_CANDIDATES)
        raise FileNotFoundError(
            f"Blocksworld domain not found. Tried:\n  {tried}"
        )
    _BW_DOMAIN_CACHE = domain_path.read_text(encoding="utf-8")
    return _BW_DOMAIN_CACHE


# WHAT THIS DOES (load_bw_domain):
# Loads the blocksworld domain PDDL once and caches it for reuse.


def generate_random_bw_pddl(n_blocks: int, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    blocks = [chr(ord("a") + i) for i in range(n_blocks)]

    init_facts = ["(handempty)"]
    for block in blocks:
        init_facts.append(f"(ontable {block})")
        init_facts.append(f"(clear {block})")

    shuffled = blocks[:]
    rng.shuffle(shuffled)
    goal_facts: list[str] = []
    goal_facts.append(f"(ontable {shuffled[0]})")
    for i in range(1, len(shuffled)):
        goal_facts.append(f"(on {shuffled[i]} {shuffled[i - 1]})")
    goal_facts.append(f"(clear {shuffled[-1]})")

    objects_str = " ".join(blocks)
    init_str = "\n    ".join(init_facts)
    goal_str = "\n    ".join(goal_facts)
    problem_pddl = f"""\
(define (problem bw-w6-{seed})
  (:domain blocksworld-4ops)
  (:objects {objects_str})
  (:init
    {init_str})
  (:goal (and
    {goal_str}))
)
"""
    domain_pddl = load_bw_domain()
    return domain_pddl, problem_pddl


# WHAT THIS DOES (generate_random_bw_pddl):
# Builds a random but valid blocksworld problem (flat start, single tower goal)
# plus the domain text, using a fixed seed for reproducibility.


def _run_unit_tests() -> None:
    tests: list[tuple[str, callable[[], None]]] = []

    def test_substring_safety() -> None:
        mapping = {
            "a": "alice",
            "b": "bob",
            "pick-up": "hire",
            "stack": "promote",
            "unstack": "demote",
            "put-down": "fire",
        }
        text = "stack a b\nunstack b a\npick-up a\nput-down b"
        result = apply_mapping(text, mapping)
        assert "promote alice bob" in result, f"stack broken: {result}"
        assert "demote bob alice" in result, f"unstack broken: {result}"
        assert "stalice" not in result, f"substring collision: {result}"
        assert "hire alice" in result, f"pick-up broken: {result}"
        assert "fire bob" in result, f"put-down broken: {result}"

    def test_unstack_before_stack() -> None:
        mapping = {"stack": "promote", "unstack": "demote"}
        assert apply_mapping("unstack a b", mapping) == "demote a b"
        assert apply_mapping("stack a b", mapping) == "promote a b"
        assert "undemote" not in apply_mapping("unstack a b", mapping)

    def test_roundtrip_passes() -> None:
        mapping = {
            "a": "alice",
            "b": "bob",
            "stack": "promote",
            "unstack": "demote",
            "pick-up": "hire",
            "put-down": "fire",
        }
        canonical = "stack a b\npick-up a"
        w3 = apply_mapping(canonical, mapping)
        passed, reason = verify_w3_roundtrip(w3, canonical, mapping)
        assert passed, f"roundtrip should pass: {reason}"

    def test_roundtrip_fails_on_corruption() -> None:
        partial_mapping = {"a": "alice"}
        text = "stack a b"
        pattern = re.compile(r"\b(a)\b")
        w3_partial = pattern.sub("alice", text)
        passed, _ = verify_w3_roundtrip(w3_partial, text, partial_mapping)
        assert passed

        w3_corrupted = "promote alice alice"
        passed_corrupted, _ = verify_w3_roundtrip(
            w3_corrupted,
            "promote a b",
            {"a": "alice", "b": "alice"},
        )
        assert not passed_corrupted, "should fail on non-bijective corruption"

    def test_derive_full_bw_init() -> None:
        on_pairs = [("j", "h"), ("h", "l"), ("l", "i")]
        all_blocks = ["j", "h", "l", "i"]
        result = derive_full_bw_init(on_pairs, all_blocks)
        assert "(on j h)" in result
        assert "(on h l)" in result
        assert "(on l i)" in result
        assert "(ontable i)" in result
        assert "(clear j)" in result
        assert "(handempty)" in result
        assert "(ontable j)" not in result
        assert "(clear i)" not in result

    tests.extend(
        [
            ("TEST 1 — substring safety", test_substring_safety),
            ("TEST 2 — unstack before stack", test_unstack_before_stack),
            ("TEST 3 — roundtrip passes", test_roundtrip_passes),
            ("TEST 4 — roundtrip fails when token survives", test_roundtrip_fails_on_corruption),
            ("TEST 5 — derive_full_bw_init", test_derive_full_bw_init),
        ]
    )

    passed_count = 0
    for name, fn in tests:
        print(name)
        try:
            fn()
            print("PASSED")
            passed_count += 1
        except AssertionError as exc:
            print(f"FAILED: {exc}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    total = len(tests)
    if passed_count == total:
        print(f"All {total} unit tests passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_unit_tests()
