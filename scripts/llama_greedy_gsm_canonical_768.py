#!/usr/bin/env python3
"""G4: GSM canonical greedy rerun with max_new_tokens=768.

Matches the Colab llama_greedy_behavioural protocol (Appendix N Probe-1
template + chat wrapper + verify_gsm_answer). Writes a NEW raw file; does
not modify results/raw/llama_greedy_p1.csv.

Expected: canonical accuracy recovers toward ~0.78 if the original 15.9%
cell was truncation at MAX_NEW=128.

Usage (GPU + HF gated weights):
  python scripts/llama_greedy_gsm_canonical_768.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify import verify_gsm_answer  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DEFAULT = REPO_ROOT / "results/raw/llama_greedy_p1_gsm_canonical_768.csv"
BANK = REPO_ROOT / "data/problems/question_bank_gsm.csv"

PROBE1_TEMPLATE = (
    "Solve the following problem exactly and provide only the final answer "
    "in the required output format. Problem: {problem}. Format instruction: "
    "{family_specific_output_format}."
)
GSM_FORMAT = (
    "Write the final numerical answer on its own line as #### <number>. "
    "No other text after that tag."
)


def build_prompt(problem_text: str) -> str:
    return PROBE1_TEMPLATE.format(
        problem=problem_text.strip(),
        family_specific_output_format=GSM_FORMAT,
    )


def wrap_chat(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.resolve() == (REPO_ROOT / "results/raw/llama_greedy_p1.csv").resolve():
        raise SystemExit("Refusing to overwrite llama_greedy_p1.csv")

    bank = pd.read_csv(BANK, dtype=str).fillna("")
    items = bank[
        (bank["variant_type"].astype(str).str.lower() == "canonical")
        & (
            bank["problem_family"].astype(str).str.strip().str.lower()
            == "arithmetic_reasoning"
        )
    ].copy()
    if args.limit is not None:
        items = items.head(args.limit)

    done: set[str] = set()
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        prev = pd.read_csv(out_path, dtype=str)
        done = set(prev["problem_id"].astype(str))

    print(f"[load] {args.model} max_new_tokens={args.max_new_tokens}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem_id",
        "family",
        "variant",
        "model_answer",
        "correct",
        "n_chars",
        "max_new_tokens",
        "model",
    ]
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    n_ok = n_done = 0
    if args.resume and out_path.exists() and out_path.stat().st_size > 0:
        prev = pd.read_csv(out_path, dtype=str)
        n_done = len(prev)
        n_ok = int(prev["correct"].astype(str).str.lower().isin(["true", "1"]).sum())

    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for _, row in items.iterrows():
            pid = str(row["problem_id"]).strip()
            if pid in done:
                continue
            user = build_prompt(str(row["problem_text"]))
            prompt = wrap_chat(tokenizer, user)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new = out[0, inputs["input_ids"].shape[1] :]
            ans = tokenizer.decode(new, skip_special_tokens=True).strip()
            correct = bool(verify_gsm_answer(ans, str(row["correct_answer"])))
            w.writerow(
                {
                    "problem_id": pid,
                    "family": "GSM",
                    "variant": "canonical",
                    "model_answer": ans,
                    "correct": str(correct),
                    "n_chars": str(len(ans)),
                    "max_new_tokens": str(args.max_new_tokens),
                    "model": args.model,
                }
            )
            f.flush()
            n_done += 1
            n_ok += int(correct)
            print(f"[{n_done}/{len(items)}] {pid} correct={correct} acc={n_ok}/{n_done}")

    print(f"Wrote {out_path}  canonical acc={n_ok}/{n_done}")


if __name__ == "__main__":
    main()
