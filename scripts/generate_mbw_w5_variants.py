import csv
import random
from collections import deque


SEED_CONFIG = [
    ("MBW_001", "hard", 8, 9901),
    ("MBW_10", "easy", 4, 9902),
    ("MBW_100", "easy", 4, 9903),
    ("MBW_127", "easy", 4, 9904),
    ("MBW_185", "hard", 8, 9905),
]

CSV_PATH = "data/problems/question_bank.csv"


def chain_goal(blocks, length):
    return tuple((blocks[i], blocks[i + 1]) for i in range(length))


def sample_problem(seed, num_blocks, difficulty):
    rng = random.Random(seed)
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    blocks = rng.sample(alphabet, num_blocks)
    rng.shuffle(blocks)

    init_on = {b: None for b in blocks}
    if difficulty == "hard":
        goal_len = num_blocks - 1
    else:
        goal_len = max(2, num_blocks // 2)
    goal_pairs = chain_goal(blocks, goal_len)
    return blocks, init_on, goal_pairs


def clear_blocks(on_map):
    supported = set(v for v in on_map.values() if v is not None)
    return [b for b in on_map if b not in supported]


def state_key(on_map, holding):
    return (tuple(sorted(on_map.items())), holding)


def goal_satisfied(on_map, goal_pairs):
    for top, bottom in goal_pairs:
        if on_map[top] != bottom:
            return False
    return True


def legal_actions(on_map, holding):
    clears = set(clear_blocks(on_map))
    actions = []
    if holding is None:
        for b, below in on_map.items():
            if b in clears and below is None:
                actions.append(("attack", b, None))
        for b, below in on_map.items():
            if b in clears and below is not None:
                actions.append(("feast", b, below))
    else:
        x = holding
        actions.append(("succumb", x, None))
        for y in clears:
            if y != x:
                actions.append(("overcome", x, y))
    return actions


def apply_action(on_map, holding, action):
    verb, x, y = action
    next_on = dict(on_map)
    next_holding = holding

    if verb == "attack":
        if holding is not None:
            return None
        if on_map[x] is not None:
            return None
        if x not in clear_blocks(on_map):
            return None
        next_holding = x
    elif verb == "succumb":
        if holding != x:
            return None
        next_holding = None
        next_on[x] = None
    elif verb == "feast":
        if holding is not None:
            return None
        if on_map[x] != y:
            return None
        if x not in clear_blocks(on_map):
            return None
        next_holding = x
        next_on[x] = None
    elif verb == "overcome":
        if holding != x:
            return None
        if y not in clear_blocks(on_map):
            return None
        next_holding = None
        next_on[x] = y
    else:
        return None
    return next_on, next_holding


def solve_bfs(init_on, goal_pairs):
    start = (dict(init_on), None)
    start_key = state_key(start[0], start[1])
    queue = deque([(start[0], start[1], [])])
    seen = {start_key}

    while queue:
        on_map, holding, path = queue.popleft()
        if goal_satisfied(on_map, goal_pairs):
            return path
        for action in legal_actions(on_map, holding):
            nxt = apply_action(on_map, holding, action)
            if nxt is None:
                continue
            nxt_on, nxt_holding = nxt
            key = state_key(nxt_on, nxt_holding)
            if key in seen:
                continue
            seen.add(key)
            queue.append((nxt_on, nxt_holding, path + [action]))
    return None


def plan_to_text(plan):
    lines = []
    for verb, x, y in plan:
        if y is None:
            lines.append(f"{verb} {x}")
        else:
            lines.append(f"{verb} {x} {y}")
    return "\n".join(lines)


def problem_text(blocks, goal_pairs):
    block_list = ", ".join(blocks[:-1]) + f", and {blocks[-1]}" if len(blocks) > 1 else blocks[0]
    goal_lines = []
    for top, bottom in goal_pairs:
        goal_lines.append(f"craves({top}, {bottom})")
    goals = ", ".join(goal_lines[:-1]) + f", and {goal_lines[-1]}" if len(goal_lines) > 1 else goal_lines[0]
    return (
        "You are in Mystery Blocksworld. Available actions: "
        "attack(x) (requires province(x), planet(x), harmony), "
        "succumb(x) (requires pain(x)), "
        "overcome(x,y) (requires pain(x), province(y)), "
        "feast(x,y) (requires craves(x,y), province(x), harmony). "
        f"Initial state: {block_list} are all province and planet; harmony is true. "
        f"Goal state: {goals}. "
        "Respond with a numbered list of actions only, using exactly attack/succumb/overcome/feast."
    )


def validate_plan(init_on, goal_pairs, plan):
    on_map = dict(init_on)
    holding = None
    for action in plan:
        nxt = apply_action(on_map, holding, action)
        if nxt is None:
            return False
        on_map, holding = nxt
    return holding is None and goal_satisfied(on_map, goal_pairs)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, fieldnames = read_rows(CSV_PATH)
    if not fieldnames:
        raise RuntimeError("question_bank.csv has no header.")

    to_add = []
    for canonical_id, difficulty, num_blocks, seed in SEED_CONFIG:
        blocks, init_on, goal_pairs = sample_problem(seed, num_blocks, difficulty)
        plan = solve_bfs(init_on, goal_pairs)
        if not plan:
            raise RuntimeError(f"No BFS plan found for {canonical_id} seed {seed}.")
        if not validate_plan(init_on, goal_pairs, plan):
            raise RuntimeError(f"Invalid BFS plan for {canonical_id} seed {seed}.")

        problem_id = f"{canonical_id}_W5"
        to_add.append(
            {
                "problem_id": problem_id,
                "variant_type": "W5",
                "problem_text": problem_text(blocks, goal_pairs),
                "correct_answer": plan_to_text(plan),
                "problem_family": "planning_suite",
                "problem_subtype": "mystery_blocksworld",
                "difficulty": difficulty,
                "contamination_pole": "low",
                "source": f"generated_seed_{seed}",
                "verifier_function": "verify_mystery_blocksworld_plan",
                "difficulty_params": "",
                "notes": "W5 forward-backward reversal",
            }
        )

    existing_ids = {row.get("problem_id", "") for row in to_add}
    kept = [row for row in rows if row.get("problem_id", "") not in existing_ids]
    kept.extend(to_add)
    write_rows(CSV_PATH, kept, fieldnames)

    for row in to_add:
        print(f"Added {row['problem_id']} ({row['source']})")


if __name__ == "__main__":
    main()
