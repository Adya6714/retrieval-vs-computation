#!/usr/bin/env python3
"""Greedy-decode Llama-3.1-8B-Instruct on ALGO canonical; score with verify_algo.

Uses the same DP-optimal verifier as Probe 1 (probes/contamination/verify_algo.py).
Generation: chat template + do_sample=False (greedy).

Usage (GPU box):

    python3 scripts/algo_llama_greedy_accuracy.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --output results/raw/ALGO_llama31_8b_greedy_canonical.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify_algo import verify_algo

FAMILY_INSTRUCTION = (
    "Solve the problem. Output ONLY the first action of the solution "
    "(no numbering, no explanation)."
)


def existing_done(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str)
    if "problem_id" not in df.columns:
        return set()
    return set(df["problem_id"].astype(str).str.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/problems/question_bank_algo.csv")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument(
        "--output",
        default="results/raw/ALGO_llama31_8b_greedy_canonical.csv",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.bank, dtype=str).fillna("")
    # Match Probe-1 ALGO suite filter
    df = df[df["problem_id"].str.match(r"^(CC|SP|WIS)_")].copy()
    df = df[df["variant_type"].str.strip() == "canonical"].copy()
    if args.limit is not None:
        df = df.head(args.limit)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_done(out) if args.resume else set()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    print(f"[load] {args.model} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    fields = [
        "problem_id",
        "problem_subtype",
        "variant_type",
        "model",
        "model_answer",
        "ground_truth",
        "verified",
        "parse_status",
        "reason",
        "correct_alternative",
    ]
    write_header = not out.exists() or out.stat().st_size == 0 or not args.resume
    if not args.resume and out.exists():
        out.write_text("")  # truncate
        write_header = True

    n_ok = 0
    n_done = 0
    # recount from file if resume
    if args.resume and out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, dtype=str)
        n_ok = int(prev["verified"].astype(str).str.lower().isin(["true", "1"]).sum())
        n_done = len(prev)

    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()

        for i, row in df.iterrows():
            pid = str(row["problem_id"]).strip()
            if pid in done:
                continue
            subtype = str(row["problem_subtype"]).strip()
            problem_text = str(row["problem_text"])
            ground_truth = str(row["correct_answer"])
            difficulty_params = json.loads(str(row["difficulty_params"]))

            user_msg = f"{FAMILY_INSTRUCTION}\n\nProblem:\n{problem_text}"
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # greedy
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            verified, reason, meta = verify_algo(
                pid,
                answer,
                ground_truth,
                subtype,
                "canonical",
                difficulty_params,
            )
            parse_status = meta.get("parse_status", "")
            alt = bool(meta.get("alternative_path") or meta.get("alternative_set"))

            w.writerow(
                {
                    "problem_id": pid,
                    "problem_subtype": subtype,
                    "variant_type": "canonical",
                    "model": args.model,
                    "model_answer": answer,
                    "ground_truth": ground_truth,
                    "verified": verified,
                    "parse_status": parse_status,
                    "reason": reason,
                    "correct_alternative": alt,
                }
            )
            f.flush()
            n_done += 1
            n_ok += int(bool(verified))
            print(
                f"[{n_done}/{len(df)}] {pid} verified={verified} parse={parse_status} "
                f"acc_so_far={n_ok}/{n_done}={n_ok/n_done:.3f}"
            )

    print("\n=== DONE ===")
    print(f"Output: {out}")
    print(f"ALGO canonical greedy accuracy: {n_ok}/{n_done} = {n_ok/n_done:.4f}")
    # subtype breakdown
    final = pd.read_csv(out, dtype=str)
    final["ok"] = final["verified"].astype(str).str.lower().isin(["true", "1"])
    print(final.groupby("problem_subtype")["ok"].agg(["sum", "count", "mean"]).to_string())


if __name__ == "__main__":
    main()
