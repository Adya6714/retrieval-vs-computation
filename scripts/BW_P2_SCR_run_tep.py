import sys, os, re, json, csv, argparse, copy
sys.path.insert(0, ".")

from cci_pipeline import (
    parse_pddl, execute_action,
    make_turn1_prompt, make_followup_prompt,
    goal_reached, seeded_inject_error, state_to_narrative,
)
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


def classify_action(action, true_state, displayed_state):
    try:
        execute_action(copy.deepcopy(true_state), action)
        valid_true = True
    except ValueError:
        valid_true = False

    try:
        execute_action(copy.deepcopy(displayed_state), action)
        valid_disp = True
    except ValueError:
        valid_disp = False

    if valid_disp and valid_true:
        return "ambiguous"
    if valid_disp and not valid_true:
        return "adapted"
    if valid_true and not valid_disp:
        return "resistant"
    return "illegal_both"


def injection_schedule(plan_length):
    if plan_length < 4:
        return [2] if plan_length > 2 else []
    candidates = sorted(set([
        2,
        max(3, plan_length // 3),
        max(4, plan_length // 2),
        max(5, 2 * plan_length // 3),
    ]))
    return [k for k in candidates if k < plan_length - 1]


def compute_tep(cascade_sequence):
    adapted  = sum(1 for s in cascade_sequence
                   if s["classification"] == "adapted")
    resistant = sum(1 for s in cascade_sequence
                    if s["classification"] == "resistant")
    denom = adapted + resistant
    return round(adapted / denom, 4) if denom > 0 else None


def run_tep_session(problem_id, pddl_path, inject_at_step,
                    client, max_steps=50):
    try:
        objects, initial_state, goal = parse_pddl(pddl_path)
    except Exception as e:
        return {"error": str(e), "inject_at_step": inject_at_step}

    true_state      = copy.deepcopy(initial_state)
    displayed_state = copy.deepcopy(initial_state)

    injection_desc   = None
    cascade_sequence = []
    pre_steps        = []

    last_action = ""
    for step in range(max_steps):

        if step == inject_at_step and injection_desc is None:
            displayed_state, injection_desc = seeded_inject_error(
                displayed_state, objects,
                seed_str=f"{problem_id}_{inject_at_step}"
            )

        narrative = build_narrative(displayed_state, objects)
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

        if injection_desc is not None and step > inject_at_step:
            cls = classify_action(action, true_state, displayed_state)
            cascade_sequence.append({
                "step": step, "action": action, "classification": cls,
            })
        else:
            pre_steps.append(action)

        last_action = action

        try:
            true_state = execute_action(copy.deepcopy(true_state), action)
        except ValueError:
            pass

        try:
            displayed_state = execute_action(
                copy.deepcopy(displayed_state), action)
        except ValueError:
            pass

        if goal_reached(true_state, goal):
            break

    adapted      = sum(1 for s in cascade_sequence
                       if s["classification"] == "adapted")
    resistant    = sum(1 for s in cascade_sequence
                       if s["classification"] == "resistant")
    ambiguous    = sum(1 for s in cascade_sequence
                       if s["classification"] == "ambiguous")
    illegal_both = sum(1 for s in cascade_sequence
                       if s["classification"] == "illegal_both")

    tep = compute_tep(cascade_sequence)
    first_class = (cascade_sequence[0]["classification"]
                   if cascade_sequence else None)

    return {
        "inject_at_step":        inject_at_step,
        "injection_desc":        injection_desc,
        "tep":                   tep,
        "adapted_count":         adapted,
        "resistant_count":       resistant,
        "ambiguous_count":       ambiguous,
        "illegal_both_count":    illegal_both,
        "first_response_class":  first_class,
        "steps_after_injection": len(cascade_sequence),
        "goal_reached_true":     goal_reached(true_state, goal),
        "cascade_sequence_json": json.dumps(cascade_sequence),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["anthropic/claude-3.7-sonnet",
                                 "openai/gpt-4o"])
    parser.add_argument("--output",    default="results/BW_P2_RES_tep.csv")
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

    plans_path = "results/BW_P2_RES_phase1_plans.csv"
    if not os.path.exists(plans_path):
        plans_path = "results/phase1_plans.csv"
    plans = pd.read_csv(plans_path)
    plans = plans[plans["plan_length"] > 0].reset_index(drop=True)
    print(f"Loaded {len(plans)} plan rows")

    done = set()
    if args.resume and os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        done = set(zip(existing["problem_id"], existing["model"],
                       existing["inject_at_step"].astype(str)))
        print(f"Resuming — {len(done)} sessions already done")

    fieldnames = [
        "problem_id", "model", "difficulty", "contamination_pole",
        "plan_length", "inject_at_step", "injection_desc",
        "tep", "adapted_count", "resistant_count",
        "ambiguous_count", "illegal_both_count",
        "first_response_class", "steps_after_injection",
        "goal_reached_true", "cascade_sequence_json",
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
            pid      = row["problem_id"]
            plan_len = int(row["plan_length"])
            schedule = injection_schedule(plan_len)

            if not schedule:
                print(f"  {pid} | plan too short ({plan_len}) — skipping")
                continue

            for k in schedule:
                if (pid, model_str, str(k)) in done:
                    print(f"  {pid} | inject@{k} | skipped")
                    continue

                print(f"  {pid} | inject@{k}/{plan_len} | "
                      f"running...", end=" ", flush=True)

                result = run_tep_session(
                    pid, row["pddl_path"], k, client, args.max_steps
                )

                out_row = {
                    "problem_id":         pid,
                    "model":              model_str,
                    "difficulty":         row["difficulty"],
                    "contamination_pole": row["contamination_pole"],
                    "plan_length":        plan_len,
                    **result,
                }
                for f in fieldnames:
                    out_row.setdefault(f, None)

                writer.writerow({f: out_row[f] for f in fieldnames})
                out_file.flush()

                print(f"TEP={result.get('tep')} | "
                      f"first={result.get('first_response_class')} | "
                      f"A={result.get('adapted_count')} "
                      f"R={result.get('resistant_count')} "
                      f"X={result.get('illegal_both_count')}")

    out_file.close()
    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
