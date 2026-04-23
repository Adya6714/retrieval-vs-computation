"""Recompute behavioral_correct using current verify_answer (no API calls)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from probes.common.io import QUESTION_BANK_PATH, QUESTION_BANK_COLUMNS
from probes.contamination.verify import verify_answer


def _bank_lookup() -> dict[tuple[str, str], dict[str, str]]:
    df = pd.read_csv(QUESTION_BANK_PATH, dtype=str)
    missing = set(QUESTION_BANK_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Question bank missing columns: {sorted(missing)}")
    out: dict[tuple[str, str], dict[str, str]] = {}
    for _, row in df.iterrows():
        pid = str(row.get("problem_id", "")).strip()
        vt = str(row.get("variant_type", "")).strip().lower()
        if not pid:
            continue
        sub = str(row.get("problem_subtype", "")).strip().lower()
        fam = str(row.get("problem_family", "")).strip().lower()
        out[(pid, vt)] = {
            "problem_text": "" if pd.isna(row.get("problem_text")) else str(row.get("problem_text")),
            "correct_answer": "" if pd.isna(row.get("correct_answer")) else str(row.get("correct_answer")),
            "family": sub or fam or "blocksworld",
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score behavioral_sweep.csv without API calls")
    parser.add_argument(
        "--input",
        type=str,
        default="results/behavioral_sweep.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Default: overwrite --input (with .bak backup)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    lookup = _bank_lookup()
    df = pd.read_csv(inp, dtype=str)

    out_rows = []
    n_ok = 0
    for _, row in df.iterrows():
        pid = str(row.get("problem_id", "")).strip()
        vt = str(row.get("variant_type", "")).strip().lower()
        raw = row.get("raw_response", "")
        raw = "" if pd.isna(raw) else str(raw)
        model = str(row.get("model", "")).strip()

        if raw.startswith("ERROR:"):
            out_rows.append(dict(row))
            continue

        meta = lookup.get((pid, vt), {})
        problem_text = meta.get("problem_text", "")
        correct_answer = meta.get("correct_answer", "")
        family = meta.get("family", str(row.get("problem_family", "blocksworld")).strip().lower())

        try:
            is_correct = verify_answer(
                pid, raw, correct_answer, family, problem_text=problem_text or None
            )
        except ValueError:
            is_correct = False

        r = dict(row)
        r["behavioral_correct"] = bool(is_correct)
        out_rows.append(r)
        if is_correct is True:
            n_ok += 1

    out_df = pd.DataFrame(out_rows)
    outp = Path(args.output) if args.output else inp
    if args.output is None:
        bak = inp.with_suffix(inp.suffix + ".bak")
        shutil.copy2(inp, bak)
        print(f"Backup: {bak}")

    out_df.to_csv(outp, index=False)
    true_ct = (out_df["behavioral_correct"].astype(str).str.lower().isin(("true", "1"))).sum()
    print(f"Wrote {outp}  rows={len(out_df)}  behavioral_correct=True count={true_ct}")


if __name__ == "__main__":
    main()
