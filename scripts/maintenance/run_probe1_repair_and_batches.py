#!/usr/bin/env python3
import csv
import os
from collections import defaultdict

from probes.behavioral.model_client import ModelClient
from probes.contamination.verify import verify_answer


TARGET_MODELS = [
    "anthropic/claude-3.7-sonnet",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
]


def resolve_paths():
    sweep_candidates = [
        "results/BW_RES_P1_behavioral_sweep.csv",
        "data/behavioral_sweep.csv",
        "results/behavioral_sweep.csv",
    ]
    sweep_path = None
    for p in sweep_candidates:
        if os.path.exists(p):
            sweep_path = p
            break
    if sweep_path is None:
        raise RuntimeError("Could not find behavioral_sweep.csv in data/ or results/.")

    qb_path = "data/problems/question_bank.csv"
    if not os.path.exists(qb_path):
        raise RuntimeError("question_bank.csv not found at data/problems/question_bank.csv")
    return sweep_path, qb_path


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fields = list(r.fieldnames or [])
    return rows, fields


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def already_run(sweep_df, problem_id, variant_type, model):
    """
    Returns True if this exact (problem_id, variant_type, model)
    combination already exists in behavioral_sweep.csv.
    Use this before every single API call to avoid re-spending money.
    variant_type matching must be CASE-INSENSITIVE.
    """
    for row in sweep_df:
        if (
            (row.get("problem_id") == problem_id)
            and ((row.get("variant_type", "") or "").upper() == variant_type.upper())
            and (row.get("model") == model)
        ):
            return True
    return False


def append_result(sweep_path, row_dict):
    """
    Appends a single result row to behavioral_sweep.csv.
    row_dict must have keys: problem_id, problem_family, variant_type,
    model, raw_response, behavioral_correct, notes
    Saves immediately after each row — do not batch, save on every result
    so partial runs are not lost.
    """
    rows, fieldnames = load_csv(sweep_path)
    out = {k: "" for k in fieldnames}
    out.update(row_dict)
    rows.append(out)
    save_csv(sweep_path, rows, fieldnames)


def get_qb_row(qb_rows, pid, variant):
    for r in qb_rows:
        if r.get("problem_id") == pid and (r.get("variant_type", "") or "").upper() == variant.upper():
            return r
    return None


def get_status(row):
    return (row.get("status", "ok") or "ok").strip().lower()


def clean_sweep(sweep_path):
    rows, fields = load_csv(sweep_path)
    kept = []
    for r in rows:
        pid = r.get("problem_id", "") or ""
        model = r.get("model", "") or ""
        if "_W5_TEMP" in pid:
            continue
        if model == "meta-llama/llama-3-8b-instruct":
            continue
        kept.append(r)
    save_csv(sweep_path, kept, fields)
    print(f"Cleaned sweep: {len(kept)} rows remaining")


def reverify_rows(sweep_path, qb_rows):
    sweep_rows, fields = load_csv(sweep_path)
    qb_index = {(r.get("problem_id"), (r.get("variant_type", "") or "").upper()): r for r in qb_rows}

    # 3A + 3B
    targets = set()
    for pid in ["MBW_001", "MBW_10", "MBW_100", "MBW_127", "MBW_185"]:
        targets.add((pid, "W3"))
    for pid in ["BW_E015", "BW_E017"]:
        targets.add((pid, "W3"))

    updated = 0
    for r in sweep_rows:
        key = (r.get("problem_id"), (r.get("variant_type", "") or "").upper())
        model = r.get("model")
        if key not in targets or model not in TARGET_MODELS:
            continue
        qb = qb_index.get(key)
        if not qb:
            continue
        old = str(r.get("behavioral_correct"))
        fam = qb.get("problem_subtype", qb.get("problem_family", ""))
        new = verify_answer(
            qb.get("problem_id"),
            r.get("raw_response", ""),
            qb.get("correct_answer", ""),
            fam,
            qb.get("problem_text", ""),
        )
        r["behavioral_correct"] = str(bool(new))
        updated += 1
        label = "MBW W3 re-verified" if key[0].startswith("MBW_") else "BW_E W3 re-verified"
        print(f"{label}: {key[0]} | {model} | {old} → {r['behavioral_correct']}")

    save_csv(sweep_path, sweep_rows, fields)
    print(f"Re-verified rows updated: {updated}")

    # 3C existence check
    c = 0
    bwe = ["BW_E002", "BW_E015", "BW_E017", "BW_E019", "BW_E100"]
    for r in sweep_rows:
        if r.get("problem_id") in bwe and (r.get("variant_type", "") or "").upper() in {"W2", "W4"} and r.get("model") in TARGET_MODELS:
            c += 1
    print(f"BW_E W2/W4 rows present: {c} (expected 30)")


def verifier_gate(qb_rows):
    targets = []
    for pid in ["MBW_001", "MBW_10", "MBW_100", "MBW_127", "MBW_185"]:
        targets.append((pid, "W3"))
    for pid in ["BW_E017", "BW_E015"]:
        targets.append((pid, "W3"))
    for pid in ["MBW_001", "MBW_10", "MBW_100", "MBW_127", "MBW_185"]:
        targets.append((pid, "W5"))
    for pid in ["BW_E002_W6", "BW_E015_W6", "BW_E017_W6", "BW_E019_W6", "BW_E100_W6"]:
        targets.append((pid, "W6"))

    qb_index = {(r.get("problem_id"), (r.get("variant_type", "") or "").upper()): r for r in qb_rows}
    for pid, var in targets:
        row = qb_index.get((pid, var))
        if not row:
            raise RuntimeError(f"VERIFIER GATE missing row: {pid} {var}")
        fam = row.get("problem_subtype", row.get("problem_family", ""))
        ok = verify_answer(pid, row.get("correct_answer", ""), row.get("correct_answer", ""), fam, row.get("problem_text", ""))
        print(f"VERIFIER GATE: {pid} {var} → {ok}")
        if not ok:
            raise RuntimeError(f"VERIFIER GATE FAILED: {pid} {var}")


def add_status_column(qb_path):
    rows, fields = load_csv(qb_path)
    if "status" not in fields:
        fields.append("status")
        for r in rows:
            r["status"] = "ok"
    for r in rows:
        if r.get("problem_id") == "MBW_100" and (r.get("variant_type", "") or "").upper() == "W1":
            r["status"] = "broken"
        if r.get("problem_id") == "MBW_10" and (r.get("variant_type", "") or "").upper() == "W1":
            r["status"] = "broken"
        if not r.get("status"):
            r["status"] = "ok"
    save_csv(qb_path, rows, fields)
    print("Status column added. Broken rows marked: MBW_100 W1, MBW_10 W1")
    return rows


def run_eval_batch(sweep_path, qb_rows, batch_name, problem_ids, variant_type, family_fallback):
    summary = defaultdict(int)
    for pid in problem_ids:
        qb = get_qb_row(qb_rows, pid, variant_type)
        if qb is None:
            print(f"MISSING {variant_type} QB ROW: {pid}")
            summary["missing_qb"] += len(TARGET_MODELS)
            continue
        if get_status(qb) == "broken":
            print(f"SKIP BROKEN: {pid} {variant_type}")
            summary["broken_skip"] += len(TARGET_MODELS)
            continue
        problem_text = qb.get("problem_text", "")
        correct_answer = qb.get("correct_answer", "")
        fam = qb.get("problem_subtype", qb.get("problem_family", family_fallback))
        for model in TARGET_MODELS:
            sweep_rows, _ = load_csv(sweep_path)
            if already_run(sweep_rows, pid, variant_type, model):
                print(f"SKIP: {pid} {variant_type} {model} already run")
                summary["already_run"] += 1
                continue
            client = ModelClient(model, temperature=0.0)
            raw = client.complete(problem_text)
            ok = verify_answer(pid, raw, correct_answer, fam, problem_text)
            row = {
                "problem_id": pid,
                "problem_family": fam,
                "variant_type": variant_type,
                "model": model,
                "raw_response": raw,
                "behavioral_correct": str(bool(ok)),
                "notes": "",
            }
            append_result(sweep_path, row)
            print(f"RAN: {pid} {variant_type} {model} → {bool(ok)}")
            summary["ran"] += 1
            if ok:
                summary["correct"] += 1
    print(f"{batch_name} summary: {dict(summary)}")
    return dict(summary)


def run_section_4c(sweep_path, qb_rows):
    target_ids = ["BW_E002_W6", "BW_E015_W6", "BW_E017_W6", "BW_E019_W6", "BW_E100_W6"]
    sweep_rows, _ = load_csv(sweep_path)
    c = 0
    for r in sweep_rows:
        if r.get("problem_id") in target_ids and r.get("model") in TARGET_MODELS:
            c += 1
    if c == 15:
        print("BW_E W6: already complete, skipping")
    else:
        for pid in target_ids:
            qb = get_qb_row(qb_rows, pid, "W6")
            if not qb:
                continue
            if get_status(qb) == "broken":
                continue
            fam = qb.get("problem_subtype", qb.get("problem_family", "blocksworld"))
            for model in TARGET_MODELS:
                sweep_rows, _ = load_csv(sweep_path)
                if already_run(sweep_rows, pid, "W6", model):
                    print(f"SKIP: {pid} W6 {model} already run")
                    continue
                client = ModelClient(model, temperature=0.0)
                raw = client.complete(qb.get("problem_text", ""))
                ok = verify_answer(pid, raw, qb.get("correct_answer", ""), fam, qb.get("problem_text", ""))
                append_result(sweep_path, {
                    "problem_id": pid,
                    "problem_family": fam,
                    "variant_type": "W6",
                    "model": model,
                    "raw_response": raw,
                    "behavioral_correct": str(bool(ok)),
                    "notes": "",
                })
                print(f"RAN: {pid} W6 {model} → {bool(ok)}")

    # new BW_E W6 rows not in suffix list
    extra = []
    for r in qb_rows:
        pid = r.get("problem_id", "")
        if (r.get("variant_type", "") or "").upper() != "W6":
            continue
        if pid in target_ids:
            continue
        if not pid.startswith("BW_E"):
            continue
        if (r.get("contamination_pole", "") or "").lower() != "low":
            continue
        extra.append(r)

    for qb in extra:
        pid = qb.get("problem_id")
        if get_status(qb) == "broken":
            continue
        fam = qb.get("problem_subtype", qb.get("problem_family", "blocksworld"))
        for model in TARGET_MODELS:
            sweep_rows, _ = load_csv(sweep_path)
            if already_run(sweep_rows, pid, "W6", model):
                print(f"SKIP: {pid} W6 {model} already run")
                continue
            client = ModelClient(model, temperature=0.0)
            raw = client.complete(qb.get("problem_text", ""))
            ok = verify_answer(pid, raw, qb.get("correct_answer", ""), fam, qb.get("problem_text", ""))
            append_result(sweep_path, {
                "problem_id": pid,
                "problem_family": fam,
                "variant_type": "W6",
                "model": model,
                "raw_response": raw,
                "behavioral_correct": str(bool(ok)),
                "notes": "",
            })
            print(f"RAN: {pid} W6 {model} → {bool(ok)}")


def run_section_4e(sweep_path, qb_rows):
    mbw_w6 = [r for r in qb_rows if (r.get("problem_subtype") == "mystery_blocksworld" and (r.get("variant_type", "") or "").upper() == "W6")]
    if not mbw_w6:
        print("MBW W6: no QB rows found, skipping")
        return
    ids = sorted({r.get("problem_id") for r in mbw_w6})
    run_eval_batch(sweep_path, qb_rows, "4E_MBW_W6", ids, "W6", "mystery_blocksworld")


def main():
    sweep_path, qb_path = resolve_paths()
    clean_sweep(sweep_path)

    qb_rows = add_status_column(qb_path)
    reverify_rows(sweep_path, qb_rows)
    verifier_gate(qb_rows)

    summaries = {}
    summaries["4A"] = run_eval_batch(
        sweep_path, qb_rows, "4A_MBW_W5",
        ["MBW_001", "MBW_10", "MBW_100", "MBW_127", "MBW_185"],
        "W5", "mystery_blocksworld",
    )
    summaries["4B"] = run_eval_batch(
        sweep_path, qb_rows, "4B_W1",
        [
            "BW_001", "BW_010", "BW_011", "BW_080", "BW_120", "BW_137", "BW_155", "BW_172", "BW_227", "BW_467",
            "BW_E002", "BW_E015", "BW_E017", "BW_E019", "BW_E100",
            "MBW_001", "MBW_127", "MBW_185", "MBW_100", "MBW_10",
        ],
        "W1", "blocksworld",
    )
    run_section_4c(sweep_path, qb_rows)
    summaries["4D"] = run_eval_batch(
        sweep_path, qb_rows, "4D_BW_080_W5",
        ["BW_080"],
        "W5", "blocksworld",
    )
    run_section_4e(sweep_path, qb_rows)

    rows, _ = load_csv(sweep_path)
    by_model = defaultdict(int)
    by_variant = defaultdict(int)
    for r in rows:
        m = r.get("model", "")
        if m == "mock":
            continue
        by_model[m] += 1
        by_variant[(m, (r.get("variant_type", "") or "").upper())] += 1

    print("\n=== POST-RUN VALIDATION SUMMARY ===")
    print(f"Sweep path: {sweep_path}")
    print(f"Total rows: {len(rows)}")
    print(f"Rows excluding mock: {sum(1 for r in rows if r.get('model') != 'mock')}")
    print("By model (excluding mock):")
    for m in TARGET_MODELS:
        print(f"  {m}: {by_model[m]}")
    print("Batch summaries:")
    for k, v in summaries.items():
        print(f"  {k}: {v}")
    print("Variant counts by model (excluding mock):")
    for m in TARGET_MODELS:
        for v in ["CANONICAL", "W1", "W2", "W3", "W4", "W5", "W6"]:
            print(f"  {m} {v}: {by_variant[(m, v)]}")


if __name__ == "__main__":
    main()

