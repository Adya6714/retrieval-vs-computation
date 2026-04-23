import re
import os
import json
import sys
import pandas as pd

SWEEP_PATH = "results/BW_P1_RES_behavioral_sweep.csv"
BANK_PATH  = "data/problems/question_bank.csv"
OUT_PATH   = "results/BW_P2_RES_phase1_plans.csv"
PDDL_ROOT  = "/Users/adya/Desktop/LLMs-Planning"

PROBE2_MODELS = {
    "anthropic/claude-3.7-sonnet",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3-8b-instruct",
}


def extract_pddl_path(source_str):
    match = re.search(r'path=([^\|]+)', str(source_str))
    if not match:
        return None
    relative = match.group(1).strip()
    return os.path.join(PDDL_ROOT, relative)


def parse_plan(raw_text):
    if not raw_text or str(raw_text).strip() in ("", "nan"):
        return []
    lines = str(raw_text).strip().split("\n")
    actions = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+[\.\)\:]\s*', '', line)
        line = re.sub(r'^step\s+\d+[\.\:\)]?\s*', '', line,
                      flags=re.IGNORECASE)
        line = line.strip().lower()
        if line:
            actions.append(line)
    return actions


def main():
    sweep_path = SWEEP_PATH if os.path.exists(SWEEP_PATH) else "results/BW_P1_RES_behavioral_sweep.csv"
    sweep = pd.read_csv(sweep_path)
    bank  = pd.read_csv(BANK_PATH)

    bw_bank = bank[
        (bank["problem_subtype"] == "blocksworld") &
        (bank["variant_type"]    == "canonical")
    ][["problem_id", "source", "difficulty", "contamination_pole"]].copy()

    bw_bank["pddl_path"] = bw_bank["source"].apply(extract_pddl_path)

    missing = bw_bank[~bw_bank["pddl_path"].apply(
        lambda p: os.path.exists(p) if isinstance(p, str) and p.strip() else False
    )]
    if len(missing):
        print("WARNING — PDDL files not found:")
        print(missing[["problem_id", "pddl_path"]].to_string())

    sweep_bw = sweep[
        (sweep["variant_type"] == "canonical") &
        (sweep["model"].isin(PROBE2_MODELS))
    ]

    merged = sweep_bw.merge(bw_bank, on="problem_id", how="inner")
    print(f"Matched rows: {len(merged)}  "
          f"({merged['problem_id'].nunique()} problems x "
          f"{merged['model'].nunique()} models)")

    rows = []
    for _, row in merged.iterrows():
        raw    = row["raw_response"]
        parsed = parse_plan(raw)
        rows.append({
            "problem_id":         row["problem_id"],
            "model":              row["model"],
            "pddl_path":          row["pddl_path"],
            "difficulty":         row["difficulty"],
            "contamination_pole": row["contamination_pole"],
            "raw_plan":           str(raw),
            "parsed_plan_json":   json.dumps(parsed),
            "plan_length":        len(parsed),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(f"Non-empty plans: {len(out[out['plan_length'] > 0])} / {len(out)}")
    print()
    print(out[["problem_id", "model", "plan_length", "difficulty"]].to_string(index=False))


if __name__ == "__main__":
    main()
