import sys, os, re, json, csv, argparse, copy
sys.path.insert(0, ".")

from cci_pipeline import (
    parse_pddl, execute_action,
    make_turn1_prompt, make_followup_prompt,
    goal_reached, state_to_narrative,
)
from probes.behavioral.cci import compute_cci
from probes.behavioral.model_client import ModelClient
import pandas as pd


def build_narrative(state, objects):
    return state_to_narrative(state, objects)


def build_goal_narrative(goal):
    parts = []
    for top, bot in goal.items():
        if bot is None:
            parts.append(f"block {top} on the table")
        else:
            parts.append(f"block {top} on block {bot}")
    return "; ".join(parts) if parts else "(empty goal)"


def normalize_action(s):
    import re
    s = s.strip().lower().rstrip('.')
    s = s.replace('(', ' ').replace(')', '').replace(',', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove "block " prefix before block names
    s = re.sub(r'\bblock\s+', '', s)
    # pick up / pickup -> pick-up
    s = re.sub(r'^pick\s*[-_]?\s*up\s+', 'pick-up ', s)
    s = re.sub(r'^pickup\s+', 'pick-up ', s)
    # put down / putdown -> put-down
    s = re.sub(r'^put\s*[-_]?\s*down\s+', 'put-down ', s)
    s = re.sub(r'^putdown\s+', 'put-down ', s)
    # place X on Y -> stack X Y
    m = re.match(r'^place\s+(\w+)\s+on\s+(\w+)$', s)
    if m:
        return f'stack {m.group(1)} {m.group(2)}'
    # place X under Y -> stack X Y (W3 HR variant)
    m = re.match(r'^place\s+(\w+)\s+under\s+(\w+)$', s)
    if m:
        return f'stack {m.group(1)} {m.group(2)}'
    # select X -> pick-up X (W3 HR variant)
    m = re.match(r'^select\s+(\w+)$', s)
    if m:
        return f'pick-up {m.group(1)}'
    return s.strip()


PREAMBLE_PREFIXES = (
    "i'll", "i will", "here is", "here's", "the next",
    "let me", "to solve", "first,", "now,", "okay",
    "sure", "great", "certainly", "to reach", "since",
    "the goal", "we need", "we must", "step", "action",
)


def parse_single_action(response_text):
    import re
    for line in str(response_text).strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if any(lower.startswith(p) for p in PREAMBLE_PREFIXES):
            continue
        line = re.sub(r'^\d+[\.\)\:]\s*', '', line)
        line = re.sub(r'^step\s+\d+[\.\:\)]?\s*', '', line,
                      flags=re.IGNORECASE)
        line = line.strip()
        if not line:
            continue
        normalized = normalize_action(line)
        if normalized:
            return normalized
    return ""


KNOWN_VERBS = {"pick-up", "put-down", "stack", "unstack",
               "pickup", "putdown", "pick up", "put down",
               "place", "select", "unstack"}


def profile_precondition_violations(executed_steps, pddl_path):
    """
    For each executed action, classify why it failed precondition check.
    Categories:
      'valid'            - action executed successfully
      'hand_not_empty'   - tried pick-up or unstack while holding something
      'block_not_clear'  - tried to pick-up/unstack a block with something on it
      'block_not_on_table' - tried pick-up but block not on table
      'wrong_stack_source' - tried unstack X Y but X is not on Y
      'target_not_clear' - tried stack but target block not clear
      'format_error'     - action string not recognized by executor at all
    Returns list of dicts: {step, action, category}
    """
    import copy
    try:
        objects, state, goal = parse_pddl(pddl_path)
    except Exception:
        return []

    profile = []
    for i, action in enumerate(executed_steps):
        parts = action.strip().split()
        if not parts:
            profile.append({"step": i, "action": action,
                            "category": "format_error"})
            continue

        verb = parts[0].lower()
        try:
            new_state = execute_action(copy.deepcopy(state), action)
            profile.append({"step": i, "action": action,
                            "category": "valid"})
            state = new_state
        except ValueError as e:
            err = str(e).lower()
            if "hand not empty" in err:
                cat = "hand_not_empty"
            elif "not on table" in err or "not in on_table" in err:
                cat = "block_not_on_table"
            elif "not clear" in err and "target" not in err:
                cat = "block_not_clear"
            elif "not clear" in err and "target" in err:
                cat = "target_not_clear"
            elif "is on" in err and "unstack" in err:
                cat = "wrong_stack_source"
            elif "unknown" in err or "malformed" in err:
                cat = "format_error"
            else:
                cat = "other_illegal"
            profile.append({"step": i, "action": action, "category": cat})

    return profile


def semantic_validity_rate(executed_steps):
    """
    Fraction of outputs that are recognizable as planning-domain
    language even if format is wrong or preconditions violated.
    An action is semantically valid if:
      - its first token is a known action verb (after normalization), AND
      - it references at least one single-letter or short block name
    """
    import re
    if not executed_steps:
        return None
    sem_valid = 0
    for action in executed_steps:
        s = action.strip().lower()
        has_verb = any(s.startswith(v) for v in KNOWN_VERBS)
        has_block = bool(re.search(r'\b[a-z]\b|\b[a-z]{1,3}\b', s))
        if has_verb and has_block:
            sem_valid += 1
    return round(sem_valid / len(executed_steps), 4)


def goal_proximity(state, goal):
    """
    Fraction of goal on-relations already satisfied in current state.
    Distinct from goal_reached (which requires ALL relations satisfied).
    """
    if not goal:
        return None
    met = sum(
        1 for top, bottom in goal.items()
        if state["on"].get(top) == bottom
    )
    return round(met / len(goal), 4)


def run_cci_session(problem_id, pddl_path, generated_plan,
                    client, max_steps=50):
    try:
        objects, state, goal = parse_pddl(pddl_path)
    except Exception as e:
        return {"problem_id": problem_id, "error": str(e),
                "cci": None, "goal_reached": False}

    executed_steps = []
    illegal_count  = 0

    last_action = ""
    for step in range(max_steps):
        narrative = build_narrative(state, objects)
        goal_nar  = build_goal_narrative(goal)

        if step == 0:
            prompt = make_turn1_prompt(narrative, goal_nar)
        else:
            prompt = make_followup_prompt(narrative, goal_nar, last_action)

        try:
            response = client.complete(prompt)
        except Exception as e:
            print(f"    API error at step {step}: {e}")
            break

        action = parse_single_action(response)
        if not action:
            break

        try:
            new_state = execute_action(copy.deepcopy(state), action)
            executed_steps.append(action)
            last_action = action
            state = new_state
            if goal_reached(state, goal):
                break
        except ValueError:
            executed_steps.append(action)
            last_action = action
            illegal_count += 1
            # state unchanged on illegal action

    # Repetition Rate: fraction of consecutive identical actions
    rr = 0.0
    if len(executed_steps) > 1:
        repeats = sum(
            1 for i in range(1, len(executed_steps))
            if executed_steps[i] == executed_steps[i-1]
        )
        rr = round(repeats / (len(executed_steps) - 1), 4)

    # First Illegal Step: index of first action that fails execution
    fis = None
    try:
        objects_fis, state_fis, goal_fis = parse_pddl(pddl_path)
        for fis_idx, act in enumerate(executed_steps):
            try:
                state_fis = execute_action(copy.deepcopy(state_fis), act)
            except ValueError:
                fis = fis_idx
                break
    except Exception:
        pass

    # Partial Goal Achievement: fraction of goal on-relations satisfied
    # after all execution steps
    goals_total = len(goal)
    goals_met = sum(
        1 for top, bottom in goal.items()
        if state["on"].get(top) == bottom
    ) if goals_total > 0 else 0
    pga = round(goals_met / goals_total, 4) if goals_total > 0 else None

    violation_profile = profile_precondition_violations(
        executed_steps, pddl_path
    )

    from collections import Counter
    vcounts = Counter(v["category"] for v in violation_profile)

    svr = semantic_validity_rate(executed_steps)

    cci_result = compute_cci(problem_id, generated_plan, executed_steps)

    return {
        "problem_id":            problem_id,
        "cci":                   cci_result["cci"],
        "matched_steps":         cci_result["matched_steps"],
        "total_steps_compared":  cci_result["total_steps_compared"],
        "generated_plan_length": len(generated_plan),
        "executed_length":       len(executed_steps),
        "illegal_action_count":  illegal_count,
        "repetition_rate":       rr,
        "first_illegal_step":    fis,
        "partial_goal_achievement": pga,
        "goals_met":             goals_met,
        "violation_hand_not_empty":    vcounts.get("hand_not_empty", 0),
        "violation_block_not_clear":   vcounts.get("block_not_clear", 0),
        "violation_block_not_on_table": vcounts.get("block_not_on_table", 0),
        "violation_wrong_stack_source": vcounts.get("wrong_stack_source", 0),
        "violation_target_not_clear":  vcounts.get("target_not_clear", 0),
        "violation_format_error":      vcounts.get("format_error", 0),
        "violation_other":             vcounts.get("other_illegal", 0),
        "violation_profile_json":      json.dumps(violation_profile),
        "semantic_validity_rate":      svr,
        "goal_reached":          goal_reached(state, goal),
        "executed_steps_json":   json.dumps(executed_steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["anthropic/claude-3.7-sonnet",
                                 "openai/gpt-4o"])
    parser.add_argument("--output",    default="results/BW_RES_P2_probe2a_cci.csv")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "OPENROUTER_API_KEY is not set. Export your OpenRouter API key, e.g.:",
            file=sys.stderr,
        )
        print("  export OPENROUTER_API_KEY='...'", file=sys.stderr)
        sys.exit(1)

    plans_path = "results/BW_RES_P2_phase1_plans.csv"
    if not os.path.exists(plans_path):
        plans_path = "results/phase1_plans.csv"
    plans = pd.read_csv(plans_path)
    plans = plans[plans["plan_length"] > 0].reset_index(drop=True)
    print(f"Loaded {len(plans)} plan rows")

    done = set()
    if args.resume and os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        done = set(zip(existing["problem_id"], existing["model"]))
        print(f"Resuming — {len(done)} pairs already done")

    fieldnames = [
        "problem_id", "model", "difficulty", "contamination_pole",
        "cci", "matched_steps", "total_steps_compared",
        "generated_plan_length", "executed_length",
        "illegal_action_count",
        "repetition_rate", "first_illegal_step",
        "partial_goal_achievement", "goals_met",
        "violation_hand_not_empty",
        "violation_block_not_clear",
        "violation_block_not_on_table",
        "violation_wrong_stack_source",
        "violation_target_not_clear",
        "violation_format_error",
        "violation_other",
        "violation_profile_json",
        "semantic_validity_rate",
        "goal_reached", "executed_steps_json",
    ]

    write_header = not (args.resume and os.path.exists(args.output))
    out_file = open(args.output, "a", newline="")
    writer   = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for model_str in args.models:
        model_tail  = model_str.split("/")[-1]
        model_plans = plans[plans["model"].str.contains(
            model_tail, case=False, na=False
        )]

        if len(model_plans) == 0:
            print(f"WARNING: no plans for '{model_tail}' — skipping")
            continue

        print(f"\n--- {model_str} | {len(model_plans)} problems ---")
        client = ModelClient(model_str, temperature=0.0)

        for _, row in model_plans.iterrows():
            pid = row["problem_id"]
            if (pid, model_str) in done:
                print(f"  {pid} | skipped")
                continue

            generated_plan = json.loads(row["parsed_plan_json"])
            print(f"  {pid} | plan_len={len(generated_plan)} | "
                  f"running...", end=" ", flush=True)

            result = run_cci_session(
                pid, row["pddl_path"], generated_plan,
                client, args.max_steps
            )

            out_row = {
                "problem_id":         pid,
                "model":              model_str,
                "difficulty":         row["difficulty"],
                "contamination_pole": row["contamination_pole"],
                **result,
            }
            for f in fieldnames:
                out_row.setdefault(f, None)

            writer.writerow({f: out_row[f] for f in fieldnames})
            out_file.flush()

            print(f"CCI={result.get('cci')} | "
                  f"exec={result.get('executed_length')} steps | "
                  f"illegal={result.get('illegal_action_count')} | "
                  f"goal={result.get('goal_reached')}")

    out_file.close()
    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
