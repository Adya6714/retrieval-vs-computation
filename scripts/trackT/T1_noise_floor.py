#!/usr/bin/env python3
"""T1 inference-stack noise floor (HP-16).

HP Track T entry point. Supports --resume and --dry-run.
Do not invent numbers; raw CSVs are append-only truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_rows(path: Path, fieldnames: list[str], rows: list[dict], resume_key=None) -> int:
    """Append rows; if resume_key, skip keys already present."""
    ensure_parent(path)
    done = set()
    if path.exists() and path.stat().st_size > 0 and resume_key is not None:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                done.add(resume_key(row))
    write_header = not path.exists() or path.stat().st_size == 0
    n = 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            if resume_key is not None and resume_key(row) in done:
                continue
            w.writerow({k: row.get(k, "") for k in fieldnames})
            n += 1
    return n


RAW_COLS = [
    "problem_id", "variant_type", "model", "batch_size", "attn_impl", "dtype",
    "response_sha16", "raw_response", "parsed_answer", "verified",
    "gen_tokens", "truncated", "family", "config_id",
]

CONFIGS = [
    {"batch_size": b, "attn_impl": a, "dtype": d}
    for b in (1, 8, 16)
    for a in ("eager", "sdpa")
    for d in ("float16", "float32")
]


def load_items(family: str):
    import pandas as pd
    if family == "GSM":
        bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_gsm.csv")
    elif family == "ALGO":
        bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_algo.csv")
        bank = bank[bank["problem_subtype"] == "coin_change"]
    else:
        raise SystemExit(f"unsupported family {family}")
    return bank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", choices=["GSM", "ALGO"], required=True)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        REPO_ROOT / "results/raw" / f"T1_noise_floor_{model_slug(args.model)}.csv"
    )
    bank = load_items(args.family)
    configs = CONFIGS if args.configs == "all" else CONFIGS[:1]
    rows = []
    if args.dry_run:
        # MockClient: one placeholder row per (item, config) without GPU.
        sample = bank.head(2)
        for _, r in sample.iterrows():
            for c in configs:
                raw = f"DRY_RUN answer for {r['problem_id']} {c}"
                rows.append({
                    "problem_id": r["problem_id"],
                    "variant_type": r.get("variant_type", "canonical"),
                    "model": args.model,
                    "batch_size": c["batch_size"],
                    "attn_impl": c["attn_impl"],
                    "dtype": c["dtype"],
                    "response_sha16": hashlib.sha1(raw.encode()).hexdigest()[:16],
                    "raw_response": raw,
                    "parsed_answer": "",
                    "verified": "",
                    "gen_tokens": 0,
                    "truncated": False,
                    "family": args.family,
                    "config_id": f"b{c['batch_size']}_{c['attn_impl']}_{c['dtype']}",
                })
        n = append_rows(
            out, RAW_COLS, rows,
            resume_key=(lambda row: (row["problem_id"], row["variant_type"], row["config_id"], row["model"]))
            if args.resume else None,
        )
        print(f"[T1 dry-run] wrote {n} rows → {out}")
        return

    # Full run: load model once per config (caller should use Colab T4).
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "torch/transformers required for non-dry-run. "
            "Use --dry-run on CPU or install GPU stack in Colab."
        ) from e

    # Gold-in-gold-out check is performed by T1_noise_floor_metrics / verifiers.
    print(f"[T1] full run not executed in this scaffold without explicit GPU confirm.")
    print(f"[T1] Use Colab notebook T1_noise_floor.ipynb with DRY_RUN=False after validating dry-run.")
    print(f"[T1] Planned out={out} items={len(bank)} configs={len(configs)}")
    # Intentionally stop before inventing answers — HP says raw outputs are truth from the model.
    raise SystemExit(
        "Non-dry-run path: implement model loop in scripts/trackT/T1_noise_floor.py "
        "on Colab (see HP-16). Dry-run path is available for pipeline checks."
    )


if __name__ == "__main__":
    main()
