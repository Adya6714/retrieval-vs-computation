import sys, os, re, json, csv, argparse, copy
sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from probes.behavioral.bw_cci_pipeline import (
    parse_pddl, execute_action,
    make_turn1_prompt, make_followup_prompt,
    goal_reached, seeded_inject_error, state_to_narrative,
)
from probes.behavioral.model_client import ModelClient
import pandas as pd

TEP_FIELDNAMES = [
    "problem_id", "model", "difficulty", "contamination_pole",
    "plan_length", "inject_at_step", "injection_desc",
    "session_status", "skip_count",
    "tep", "adapted_count", "resistant_count",
    "ambiguous_count", "illegal_both_count",
    "first_response_class", "steps_after_injection",
    "goal_reached_true", "cascade_sequence_json",
]


def _migrate_tep_output(path: str, fieldnames: list[str] | None = None) -> None:
    """Normalize legacy TEP CSVs and drop malformed append rows (schema mismatch)."""
    fieldnames = fieldnames or TEP_FIELDNAMES
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return
        header = [h.strip() for h in header]
        n_expected = len(header)
        rows: list[dict[str, str]] = []
        dropped = 0
        for row in reader:
            if len(row) != n_expected:
                dropped += 1
                continue
            rows.append(dict(zip(header, row)))
    if dropped:
        print(f"TEP migrate: dropped {dropped} malformed row(s) from {path}")
    out_rows = []
    for row in rows:
        out = {k: row.get(k, "") for k in fieldnames}
        if "session_status" not in row:
            out["session_status"] = row.get("session_status", "")
        if "skip_count" not in row:
            out["skip_count"] = row.get("skip_count", "")
        out_rows.append(out)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)


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


def _safe_json_loads(value, default):
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return parsed if parsed is not None else default
    except Exception:
        return default


def _load_plan_lookup(plans_path):
    if not os.path.exists(plans_path):
        return {}
    plans = pd.read_csv(plans_path, dtype=str).fillna("")
    lookup = {}
    for _, row in plans.iterrows():
        plan = _safe_json_loads(row.get("parsed_plan_json", ""), [])
        if not isinstance(plan, list):
            plan = []
        plan_norm = [normalize_action(str(x)) for x in plan]
        lookup[(str(row.get("problem_id", "")), str(row.get("model", "")))] = plan_norm
    return lookup


def recompute_tep_from_existing_csv(input_path, output_path=None, plans_path="results/raw/BW_P2_plans.csv"):
    df = pd.read_csv(input_path, dtype=str).fillna("")
    plan_lookup = _load_plan_lookup(plans_path)

    recomputed_rows = []
    for _, row in df.iterrows():
        cascade = _safe_json_loads(row.get("cascade_sequence_json", "[]"), [])
        if not isinstance(cascade, list):
            cascade = []

        model = str(row.get("model", ""))
        problem_id = str(row.get("problem_id", ""))
        plan_actions = plan_lookup.get((problem_id, model), [])

        adapted = 0
        resistant = 0
        ambiguous = 0
        illegal_both = 0
        inferred_cascade = []

        for step_entry in cascade:
            if not isinstance(step_entry, dict):
                continue
            step_idx = step_entry.get("step", None)
            try:
                step_idx = int(step_idx)
            except Exception:
                step_idx = None

            action = normalize_action(str(step_entry.get("action", "")))
            cls = str(step_entry.get("classification", "")).strip().lower()

            planned = None
            if step_idx is not None and 0 <= step_idx < len(plan_actions):
                planned = plan_actions[step_idx]
            differs = (planned is not None and action != "" and action != planned)

            if cls not in {"adapted", "resistant", "ambiguous", "illegal_both"}:
                if planned is None or action == "":
                    cls = "ambiguous"
                else:
                    cls = "adapted" if differs else "resistant"

            if cls == "adapted":
                adapted += 1
            elif cls == "resistant":
                resistant += 1
            elif cls == "ambiguous":
                ambiguous += 1
            else:
                illegal_both += 1

            inferred_cascade.append({"step": step_idx, "action": action, "classification": cls})

        denom = adapted + resistant + ambiguous
        tep = round(adapted / denom, 4) if denom > 0 else None
        first_class = inferred_cascade[0]["classification"] if inferred_cascade else None

        row["adapted_count"] = str(adapted)
        row["resistant_count"] = str(resistant)
        row["ambiguous_count"] = str(ambiguous)
        row["illegal_both_count"] = str(illegal_both)
        row["steps_after_injection"] = str(len(inferred_cascade))
        row["first_response_class"] = first_class if first_class is not None else ""
        row["tep"] = "" if tep is None else f"{tep:.4f}"
        row["cascade_sequence_json"] = json.dumps(inferred_cascade)
        recomputed_rows.append(row)

    out_df = pd.DataFrame(recomputed_rows, columns=df.columns)
    out_path = output_path or input_path
    out_df.to_csv(out_path, index=False)
    return out_df, out_path


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
    skip_count = 0
    error_count = 0
    last_error = None
    session_status = "complete"

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
            current_error = "parse_error"
            if current_error == last_error:
                error_count += 1
            else:
                error_count = 1
                last_error = current_error

            if error_count >= 2:
                cascade_sequence.append({
                    "step": step,
                    "action": "STEP_SKIP",
                    "classification": "illegal_both",
                })
                skip_count += 1
                error_count = 0
                last_error = None
                if skip_count > 5:
                    session_status = "aborted: excessive illegal steps"
                    break
            continue

        if injection_desc is not None and step > inject_at_step:
            cls = classify_action(action, true_state, displayed_state)
            current_error = f"illegal_both:{action}" if cls == "illegal_both" else None
            if current_error:
                if current_error == last_error:
                    error_count += 1
                else:
                    error_count = 1
                    last_error = current_error

                if error_count >= 2:
                    cascade_sequence.append({
                        "step": step,
                        "action": "STEP_SKIP",
                        "classification": "illegal_both",
                    })
                    skip_count += 1
                    error_count = 0
                    last_error = None
                    if skip_count > 5:
                        session_status = "aborted: excessive illegal steps"
                        break
                    continue
            else:
                error_count = 0
                last_error = None
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

    tep = None if session_status.startswith("aborted:") else compute_tep(cascade_sequence)
    first_class = (cascade_sequence[0]["classification"]
                   if cascade_sequence else None)

    return {
        "inject_at_step":        inject_at_step,
        "injection_desc":        injection_desc,
        "session_status":        session_status,
        "skip_count":            skip_count,
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
                        default=["anthropic/claude-sonnet-4",
                                 "openai/gpt-4o"])
    parser.add_argument("--output",    default="results/raw/BW_P2_tep.csv")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--problem-ids", nargs="+", default=None)
    parser.add_argument("--recompute-from-cascade", action="store_true")
    parser.add_argument("--input", default="results/raw/BW_P2_tep.csv")
    parser.add_argument("--plans-path", default="results/raw/BW_P2_plans.csv")
    args = parser.parse_args()

    if args.recompute_from_cascade:
        out_df, out_path = recompute_tep_from_existing_csv(
            input_path=args.input,
            output_path=args.output,
            plans_path=args.plans_path,
        )
        tep_vals = pd.to_numeric(out_df.get("tep", pd.Series([], dtype=float)), errors="coerce")
        valid_tep = int(tep_vals.notna().sum())
        total = int(len(out_df))
        print(f"Recomputed TEP from cascade JSON: {out_path}")
        print(f"valid_tep: {valid_tep}/{total}")
        if valid_tep > 0:
            out_df = out_df.copy()
            out_df["tep_num"] = pd.to_numeric(out_df["tep"], errors="coerce")
            means = out_df.groupby("model", dropna=False)["tep_num"].mean(numeric_only=True).dropna()
            print("mean TEP per model:")
            for model, val in means.items():
                print(f"  {model}: {val:.4f}")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "OPENROUTER_API_KEY is not set. Export your OpenRouter API key, e.g.:",
            file=sys.stderr,
        )
        print("  export OPENROUTER_API_KEY='...'", file=sys.stderr)
        sys.exit(1)

    plans_path = "results/raw/BW_P2_plans.csv"
    if not os.path.exists(plans_path):
        plans_path = "results/phase1_plans.csv"
    plans = pd.read_csv(plans_path)
    plans = plans[plans["plan_length"] > 0].reset_index(drop=True)
    print(f"Loaded {len(plans)} plan rows")

    fieldnames = TEP_FIELDNAMES

    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        _migrate_tep_output(args.output, fieldnames)

    done = set()
    if args.resume and os.path.exists(args.output):
        existing = pd.read_csv(args.output)
        done = set(zip(existing["problem_id"], existing["model"],
                       existing["inject_at_step"].astype(str)))
        print(f"Resuming — {len(done)} sessions already done")

    write_header = not (args.resume and os.path.exists(args.output) and os.path.getsize(args.output) > 0)
    out_file = open(args.output, "a", newline="")
    writer   = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for model_str in args.models:
        model_tail  = model_str.split("/")[-1]
        model_plans = plans[plans["model"].str.contains(
            model_tail, case=False, na=False
        )]
        if args.problem_ids:
            allowed = set(str(x) for x in args.problem_ids)
            model_plans = model_plans[
                model_plans["problem_id"].astype(str).isin(allowed)
            ]

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
