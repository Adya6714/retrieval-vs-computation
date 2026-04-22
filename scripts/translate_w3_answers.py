#!/usr/bin/env python3
"""Translate canonical blocksworld correct_answer to W3 (HR scenario) in question_bank.csv."""

import csv
import re
from pathlib import Path

DEFAULT_CSV = Path("data/problems/question_bank.csv")

# Standard word exclusions so we don't accidentally pick them up as names.
EXCLUSIONS = {
    "You", "HR", "Available", "X", "Y", "Current", "Goal", "Respond", "Each", "No", "Manager", "The"
}

def get_name_mapping(w3_text: str) -> dict[str, str]:
    """Dynamically infer the block-to-name mapping from the W3 problem text."""
    names = re.findall(r"\b[A-Z][a-z]+\b", w3_text)
    mapping = {}
    for name in names:
        if name not in EXCLUSIONS:
            mapping[name[0].lower()] = name
    return mapping

def translate_action(pddl_action: str, mapping: dict[str, str]) -> str:
    parts = pddl_action.lower().split()
    if not parts:
        return pddl_action
    
    verb = parts[0]
    args = parts[1:]
    
    # Map the arguments to names, default to the argument if not found
    names = [mapping.get(arg, arg) for arg in args]

    if verb == "pick-up" and len(names) == 1:
        return f"select {names[0]}"
    elif verb == "put-down" and len(names) == 1:
        return f"release {names[0]}"
    elif verb == "stack" and len(names) == 2:
        return f"place {names[0]} under {names[1]}"
    elif verb == "unstack" and len(names) == 2:
        return f"remove {names[0]} from {names[1]}"
    
    return pddl_action

def main() -> None:
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {DEFAULT_CSV}")

    with DEFAULT_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV has no header.")
        rows = list(reader)

    # 1. Grab all canonical blocksworld answers
    canonical_answers: dict[str, str] = {}
    for row in rows:
        pid = str(row.get("problem_id", "")).strip()
        vtype = str(row.get("variant_type", "")).strip().lower()
        if pid and vtype == "canonical":
            canonical_answers[pid] = str(row.get("correct_answer", ""))

    updated = 0
    for row in rows:
        pid = str(row.get("problem_id", "")).strip()
        vtype = str(row.get("variant_type", "")).strip().upper()
        
        # Only process W3 rows that belong to blocksworld
        subtype = str(row.get("problem_subtype", "")).strip().lower()
        if vtype != "W3" or subtype != "blocksworld":
            continue
            
        if pid not in canonical_answers:
            continue
            
        canonical_ans = canonical_answers[pid]
        current_ans = str(row.get("correct_answer", ""))
        
        # Skip if already translated
        if current_ans and "select" in current_ans:
            continue
            
        if not canonical_ans:
            continue

        w3_text = str(row.get("problem_text", ""))
        mapping = get_name_mapping(w3_text)
        
        canonical_actions = canonical_ans.strip().split("\n")
        translated_actions = []
        
        for i, action in enumerate(canonical_actions):
            # Optional: handle if canonical had numbers (e.g., "1. pick-up a")
            action_clean = re.sub(r"^\d+\.\s*", "", action)
            translated = translate_action(action_clean, mapping)
            translated_actions.append(f"{i+1}. {translated}")
            
        translated_answer = "\n".join(translated_actions)
        row["correct_answer"] = translated_answer
        updated += 1
        print(f"Translated W3 for problem_id {pid}")

    print(f"Updated {updated} W3 rows.")

    with DEFAULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
