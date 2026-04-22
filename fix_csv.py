#!/usr/bin/env python3
import csv
import os


def _csv_path() -> str:
    # Spec says "question_bank.csv"; support both common run locations.
    if os.path.exists("question_bank.csv"):
        return "question_bank.csv"
    return os.path.join("data", "problems", "question_bank.csv")


def main() -> None:
    path = _csv_path()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"No CSV header found in {path}")

    changed_row_keys: set[tuple[str, str]] = set()

    def mark_changed(problem_id: str, variant_type: str, what: str) -> None:
        changed_row_keys.add((problem_id, variant_type))
        print(f"[FIXED] {problem_id} | {variant_type} | {what}")

    # FIX 9 mapping for BW_E canonical source standardization
    bwe_source_map = {
        "BW_E002": "type=planbench_generated | dataset=planbench_basic | filename=instance-basic-3-2.pddl",
        "BW_E015": "type=planbench_generated | dataset=planbench_basic | filename=instance-basic-3-15.pddl",
        "BW_E017": "type=planbench_generated | dataset=planbench_basic | filename=instance-basic-3-17.pddl",
        "BW_E019": "type=planbench_generated | dataset=planbench_basic | filename=instance-basic-3-19.pddl",
        "BW_E100": "type=planbench_generated | dataset=planbench_basic | filename=instance-basic-3-100.pddl",
    }

    fix8_old = (
        "Respond with a numbered list of actions only. Each action must be exactly\n"
        "one of: pick-up X / put-down X / stack X Y / unstack X Y.\n"
        "No explanation. No extra text."
    )
    fix8_new = (
        "Respond with a numbered list of actions only. Each action must be exactly one of: "
        "pick-up X / put-down X / stack X Y / unstack X Y. No explanation. No extra text."
    )

    for r in rows:
        pid = (r.get("problem_id") or "").strip()
        vtype = (r.get("variant_type") or "").strip()

        # FIX 1 — MBW W3 correct_answer: wrong action name
        if (r.get("problem_subtype") or "").strip() == "mystery_blocksworld" and vtype == "W3":
            ca = r.get("correct_answer") or ""
            new_ca = ca.replace("overcome", "broker")
            if new_ca != ca:
                r["correct_answer"] = new_ca
                mark_changed(pid, vtype, "correct_answer: replaced overcome -> broker")

        # FIX 2 — BW_E series lowercase variant_type
        if pid.startswith("BW_E") and vtype in {"w2", "w3", "w4"}:
            new_v = vtype.upper()
            r["variant_type"] = new_v
            mark_changed(pid, vtype, f"variant_type: {vtype} -> {new_v}")
            vtype = new_v  # keep local in sync for subsequent fix checks

        # FIX 3 — BW_E series contamination_pole
        if pid.startswith("BW_E"):
            old = r.get("contamination_pole") or ""
            if old != "low":
                r["contamination_pole"] = "low"
                mark_changed(pid, vtype, f"contamination_pole: {old!r} -> 'low'")

        # FIX 4 — BW_080 W5 missing base block statement
        if pid == "BW_080" and vtype == "W5":
            pt = r.get("problem_text") or ""
            old = "and block j is on block f. The hand is empty."
            new = "and block j is on block f. Block f is on the table. The hand is empty."
            if old in pt:
                r["problem_text"] = pt.replace(old, new)
                mark_changed(pid, vtype, "problem_text: added 'Block f is on the table.'")

        # FIX 5 — BW_227 trailing pipe in difficulty_params
        if pid.startswith("BW_227"):
            dp = r.get("difficulty_params") or ""
            dp_strip = dp.strip()
            if dp_strip.endswith("|") and dp_strip == "num_blocks=9 | requires_unstacking=false |":
                r["difficulty_params"] = "num_blocks=9 | requires_unstacking=false | max_initial_stack_depth=1"
                mark_changed(pid, vtype, "difficulty_params: replaced trailing-pipe value")

        # FIX 6 — MBW_127 W4 stray backtick fence
        if pid == "MBW_127" and vtype == "W4":
            pt = r.get("problem_text") or ""
            if "```" in pt:
                r["problem_text"] = pt.replace("```", "")
                mark_changed(pid, vtype, "problem_text: removed ``` fences")

        # FIX 7 — BW_E W4 domain label
        if pid.startswith("BW_E") and vtype in {"w4", "W4"}:
            pt = r.get("problem_text") or ""
            new_pt = pt.replace("PEOPLE:", "BLOCKS:").replace("P = {", "B = {")
            if new_pt != pt:
                r["problem_text"] = new_pt
                mark_changed(pid, vtype, "problem_text: PEOPLE->BLOCKS and P={ -> B={")

        # FIX 8 — BW_172 and BW_080 W2 line break in output instruction
        if pid in {"BW_172", "BW_080"} and vtype == "W2":
            pt = r.get("problem_text") or ""
            if fix8_old in pt:
                r["problem_text"] = pt.replace(fix8_old, fix8_new)
                mark_changed(pid, vtype, "problem_text: collapsed output instruction line breaks")

        # FIX 9 — BW_E source field standardization (canonical only)
        if pid.startswith("BW_E") and vtype == "canonical" and pid in bwe_source_map:
            old = r.get("source") or ""
            new = bwe_source_map[pid]
            if old != new:
                r["source"] = new
                mark_changed(pid, vtype, "source: standardized planbench_basic filename")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"total rows changed: {len(changed_row_keys)}")


if __name__ == "__main__":
    main()

