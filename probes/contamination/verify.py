"""Answer verification with family-specific handlers and orchestration."""

from __future__ import annotations

import re


def _extract_actions(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [m.group(0).strip().lower() for m in pattern.finditer(str(text).lower())]


def _verify_numeric(model_answer, ground_truth) -> bool:
    match = re.search(r"[-+]?\d*\.?\d+", str(model_answer))
    if not match:
        return False
    try:
        model_val = float(match.group())
        gt_val = float(ground_truth)
        return abs(model_val - gt_val) <= 1e-6
    except ValueError:
        return False


def _verify_shortest_path(model_answer, ground_truth) -> bool:
    parts = re.split(r"[,\- \>]+", str(model_answer).strip().upper())
    model_path = "".join([p for p in parts if len(p) == 1 and p.isalpha()])
    gt_path = "".join([p for p in str(ground_truth).strip().upper() if p.isalpha()])
    return model_path == gt_path


def _parse_blocksworld_state(problem_text: str) -> tuple[set[tuple], set[tuple]] | None:
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
    return state, goal_facts


def _apply_blocksworld_action(state: set[tuple], action: str) -> bool:
    parts = action.split()
    if not parts:
        return False
    verb = parts[0]
    if verb in {"pick-up", "put-down"} and len(parts) != 2:
        return False
    if verb in {"stack", "unstack"} and len(parts) != 3:
        return False

    if verb == "pick-up":
        x = parts[1]
        pre = {("clear", x), ("ontable", x), ("handempty",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("clear", x), ("ontable", x), ("handempty",)})
        state.add(("holding", x))
        return True
    if verb == "put-down":
        x = parts[1]
        pre = {("holding", x)}
        if not pre.issubset(state):
            return False
        state.remove(("holding", x))
        state.update({("ontable", x), ("clear", x), ("handempty",)})
        return True
    if verb == "stack":
        x, y = parts[1], parts[2]
        pre = {("holding", x), ("clear", y)}
        if not pre.issubset(state):
            return False
        state.difference_update({("holding", x), ("clear", y)})
        state.update({("on", x, y), ("clear", x), ("handempty",)})
        return True
    if verb == "unstack":
        x, y = parts[1], parts[2]
        pre = {("on", x, y), ("clear", x), ("handempty",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("on", x, y), ("clear", x), ("handempty",)})
        state.update({("holding", x), ("clear", y)})
        return True
    return False


def _verify_blocksworld_state_machine(model_answer, problem_text) -> bool | None:
    parsed = _parse_blocksworld_state(problem_text)
    if parsed is None:
        return None
    state, goal = parsed
    action_pattern = re.compile(r"(pick-up|put-down|stack|unstack)\s+[a-z0-9]+(\s+[a-z0-9]+)?", re.IGNORECASE)
    actions = _extract_actions(model_answer, action_pattern)
    if not actions:
        return False
    for action in actions:
        if not _apply_blocksworld_action(state, action):
            return False
    return goal.issubset(state)


def _parse_mystery_state(problem_text: str) -> tuple[set[tuple], set[tuple]] | None:
    text = str(problem_text).lower()
    current_match = re.search(r"current state:\s*(.*?)\s*goal:", text, re.IGNORECASE | re.DOTALL)
    goal_match = re.search(r"goal:\s*(.*?)(?:respond with|$)", text, re.IGNORECASE | re.DOTALL)
    if not current_match or not goal_match:
        return None
    current = current_match.group(1)
    goal = goal_match.group(1)
    state: set[tuple] = set()
    goals: set[tuple] = set()

    if "harmony is true" in current:
        state.add(("harmony",))

    m = re.search(r"planet and province are true for blocks?\s+(.+?)(?:\.|$)", current)
    if m:
        blocks = [b.strip() for b in re.split(r",| and ", m.group(1)) if b.strip()]
        for b in blocks:
            state.add(("planet", b))
            state.add(("province", b))

    for x, y in re.findall(r"craves\s+([a-z0-9]+)\s+([a-z0-9]+)", goal):
        goals.add(("craves", x, y))
    return state, goals


def _apply_mystery_action(state: set[tuple], action: str) -> bool:
    parts = action.split()
    if not parts:
        return False
    verb = parts[0]
    if verb in {"attack", "succumb"} and len(parts) != 2:
        return False
    if verb in {"overcome", "feast"} and len(parts) != 3:
        return False

    if verb == "attack":
        x = parts[1]
        pre = {("province", x), ("planet", x), ("harmony",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("province", x), ("planet", x), ("harmony",)})
        state.add(("pain", x))
        return True
    if verb == "succumb":
        x = parts[1]
        pre = {("pain", x)}
        if not pre.issubset(state):
            return False
        state.remove(("pain", x))
        state.update({("province", x), ("planet", x), ("harmony",)})
        return True
    if verb == "overcome":
        x, y = parts[1], parts[2]
        pre = {("pain", x), ("province", y)}
        if not pre.issubset(state):
            return False
        state.difference_update({("pain", x), ("province", y)})
        state.update({("harmony",), ("province", x), ("craves", x, y)})
        return True
    if verb == "feast":
        x, y = parts[1], parts[2]
        pre = {("craves", x, y), ("province", x), ("harmony",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("craves", x, y), ("province", x), ("harmony",)})
        state.update({("pain", x), ("province", y)})
        return True
    return False


def _verify_mystery_state_machine(model_answer, problem_text) -> bool | None:
    parsed = _parse_mystery_state(problem_text)
    if parsed is None:
        return None
    state, goals = parsed
    mystery_pattern = re.compile(
        r"(attack|succumb|overcome|feast)\s+[a-z0-9]+(\s+[a-z0-9]+)?",
        re.IGNORECASE,
    )
    actions = _extract_actions(model_answer, mystery_pattern)
    if not actions:
        return False
    for action in actions:
        if not _apply_mystery_action(state, action):
            return False
    return goals.issubset(state)


def verify_answer(problem_id, model_answer, ground_truth, family, problem_text=None):
    numeric_families = {
        "gsm", 
        "shortest_path", 
        "weighted_interval_scheduling", 
        "coin_change", 
        "knapsack"
    }
    plan_families = {"blocksworld", "logistics", "mystery_blocksworld"}

    if family == "shortest_path":
        return _verify_shortest_path(model_answer, ground_truth)

    elif family in numeric_families:
        return _verify_numeric(model_answer, ground_truth)

    elif family == "mystery_blocksworld":
        sim_ok = _verify_mystery_state_machine(model_answer, problem_text)
        if sim_ok is not None:
            return sim_ok
        mystery_pattern = re.compile(
            r'(attack|succumb|overcome|feast)\s+[a-z0-9]+(\s+[a-z0-9]+)?',
            re.IGNORECASE
        )
        model_matches = _extract_actions(model_answer, mystery_pattern)
        gt_matches = _extract_actions(ground_truth, mystery_pattern)
        if not model_matches or not gt_matches:
            return str(model_answer).strip().lower() == str(ground_truth).strip().lower()
        return model_matches == gt_matches

    elif family in plan_families:
        sim_ok = _verify_blocksworld_state_machine(model_answer, problem_text)
        if sim_ok is not None:
            return sim_ok
        # Extract actions like 'pick-up A' or 'unstack A B'
        action_pattern = re.compile(r'(pick-up|put-down|stack|unstack)\s+[a-z0-9]+(\s+[a-z0-9]+)?', re.IGNORECASE)
        model_matches = _extract_actions(model_answer, action_pattern)
        gt_matches = _extract_actions(ground_truth, action_pattern)
        
        if not model_matches or not gt_matches:
            # Fallback to simple string comparison if parser fails
            return str(model_answer).strip().lower() == str(ground_truth).strip().lower()
            
        return model_matches == gt_matches

    else:
        valid_families = numeric_families | plan_families | {"shortest_path"}
        raise ValueError(f"Unrecognized family: '{family}'. Expected one of {sorted(valid_families)}")
