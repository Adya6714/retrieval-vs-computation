#!/usr/bin/env python3
"""Diagnose mechanistic read position + gold-in-prompt leakage.

Replays the EXACT prompt wrap + HF encode path from
scripts/run_mechanistic_sweep_7b.py (no changes to that reading logic).

Examples:
  # (1)(2) Llama prompts / leakage (tokenizer only — no GPU needed)
  python3 scripts/diagnose_mechanistic_read_position.py \\
      --model meta-llama/Llama-3.1-8B-Instruct --prompt-mode chat-direct \\
      --family algo --n 5 --tokenizer-only

  # (3) Qwen ALGO median final rank via identical metric path (needs GPU)
  python3 scripts/diagnose_mechanistic_read_position.py \\
      --model Qwen/Qwen2.5-7B --prompt-mode raw-qa \\
      --family algo --n 100 --compute-ranks \\
      --report-median
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd


def _first_action(answer: str) -> str:
    lines = [l.strip() for l in str(answer).split("\n") if l.strip()]
    if not lines:
        return ""
    return re.sub(r"^\d+[\.\)]\s*", "", lines[0])


def _family_instruction(family: str) -> str:
    f = (family or "").strip().lower()
    if f in ("gsm", "arithmetic_reasoning"):
        return "Solve the problem. Output ONLY the final numerical answer, nothing else."
    if f in ("bw", "blocksworld", "blocks world", "planning_suite"):
        return "Solve the planning problem. Output ONLY the first plan step (action + arguments), no numbering, no explanation."
    return "Solve the problem. Output ONLY the first action of the solution (no numbering, no explanation)."


def make_wrap(tokenizer, prompt_mode: str):
    """Same wrap logic as run_mechanistic_sweep_7b.py."""

    def _wrap(text: str, family: str = "") -> str:
        if prompt_mode == "raw":
            return text
        if prompt_mode == "raw-qa":
            return f"Problem: {text}\n\nAnswer:\n"
        user_msg = text
        if prompt_mode == "chat-direct":
            user_msg = f"{_family_instruction(family)}\n\nProblem:\n{text}"
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True,
            tokenize=False,
        )

    return _wrap


def resolve_target_token(tokenizer, prompt: str, answer: str):
    """Identical to run_mechanistic_sweep_7b._resolve_target_token."""

    def _hf_encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    if not answer.strip():
        return -1, "", "", []
    prompt_ids = _hf_encode(prompt)
    candidates = []
    for sep in ("", " "):
        joint = _hf_encode(prompt + sep + answer)
        if len(joint) <= len(prompt_ids):
            continue
        if joint[: len(prompt_ids)] != prompt_ids:
            continue
        tid = int(joint[len(prompt_ids)])
        candidates.append((sep, tid, len(joint)))
    if not candidates:
        bare = _hf_encode(answer)
        if not bare:
            return -1, "", "FALLBACK", prompt_ids
        tid = int(bare[0])
        return tid, tokenizer.decode([tid]), "FALLBACK", prompt_ids
    candidates.sort(key=lambda c: c[2])
    sep, tid, _ = candidates[0]
    return tid, tokenizer.decode([tid]), repr(sep), prompt_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-mode", default="chat-direct")
    ap.add_argument("--family", default="algo", choices=["algo", "gsm", "bw"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--tokenizer-only", action="store_true")
    ap.add_argument("--compute-ranks", action="store_true")
    ap.add_argument("--report-median", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="If set with --report-median, also print median final rank from this CSV (artifact check).",
    )
    args = ap.parse_args()

    bank = {
        "algo": "data/problems/question_bank_algo.csv",
        "gsm": "data/problems/question_bank_gsm.csv",
        "bw": "data/problems/question_bank_bw.csv",
    }[args.family]
    df = pd.read_csv(bank, dtype=str).fillna("")
    canon = df[(df.variant_type == "canonical") & (df.problem_family.str.contains(
        {"algo": "algorithmic|coin", "gsm": "arithmetic", "bw": "planning|block"}[args.family],
        case=False,
        regex=True,
        na=False,
    ))]
    # Prefer algorithmic over coin change for algo samples
    if args.family == "algo":
        algo_only = canon[canon.problem_family.str.lower() == "algorithmic"]
        if len(algo_only):
            canon = algo_only
    rows = canon.head(args.n)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    _wrap = make_wrap(tokenizer, args.prompt_mode)

    print("=" * 72)
    print(f"model={args.model}  prompt_mode={args.prompt_mode}  family={args.family}  n={len(rows)}")
    print("Read position: residual @ LAST prompt token  →  python index = len(prompt_ids)-1")
    print("               (cache[k][0, -1, :] in run_mechanistic_sweep_7b._last_token_resid)")
    print("=" * 72)

    records = []
    for _, row in rows.iterrows():
        pid = str(row.problem_id).strip()
        family = str(row.problem_family)
        text = str(row.problem_text)
        action = _first_action(row.correct_answer)
        prompt = _wrap(text, family)
        tid, tdec, sep, prompt_ids = resolve_target_token(tokenizer, prompt, action)
        read_pos = len(prompt_ids) - 1  # last token index
        # occurrences of gold token id anywhere in prompt (positions ≤ read_pos)
        hit_positions = [i for i, t in enumerate(prompt_ids) if t == tid]
        gold_in_prompt = len(hit_positions) > 0
        # also check decoded substring (informational)
        gold_str_in_prompt = (tdec in prompt) if tdec else False

        rec = {
            "problem_id": pid,
            "prompt": prompt,
            "prompt_n_tokens": len(prompt_ids),
            "read_token_index": read_pos,
            "read_token_id": prompt_ids[read_pos] if prompt_ids else None,
            "read_token_decoded": tokenizer.decode([prompt_ids[read_pos]]) if prompt_ids else None,
            "gold_action": action,
            "gold_token_id": tid,
            "gold_token_decoded": tdec,
            "sep": sep,
            "gold_token_id_in_prompt": gold_in_prompt,
            "gold_token_id_positions": hit_positions,
            "gold_decoded_substring_in_prompt": gold_str_in_prompt,
        }
        records.append(rec)

        print(f"\n--- {pid} ---")
        print(f"gold_action: {action!r}")
        print(f"gold_token_id={tid}  decoded={tdec!r}  sep={sep}")
        print(f"prompt_n_tokens={len(prompt_ids)}  read_token_index={read_pos}  "
              f"(0-based last position; this is where rank is read)")
        print(f"token_at_read_pos id={rec['read_token_id']} decoded={rec['read_token_decoded']!r}")
        print(f"gold_token_id appears in prompt at positions ≤ read_pos? {gold_in_prompt}  positions={hit_positions}")
        print(f"decoded gold substring appears in prompt text? {gold_str_in_prompt}")
        print("EXACT PROMPT STRING BEGIN")
        print(prompt)
        print("EXACT PROMPT STRING END")

    n_leak = sum(1 for r in records if r["gold_token_id_in_prompt"])
    print("\n" + "=" * 72)
    print(f"LEAKAGE SUMMARY: {n_leak}/{len(records)} samples have gold token id in prompt at/before read position")
    print("=" * 72)

    if args.from_csv and args.report_median:
        cdf = pd.read_csv(args.from_csv)
        def parse(x):
            return ast.literal_eval(x) if isinstance(x, str) else x
        sub = cdf[(cdf.problem_family == "algorithmic") & (cdf.variant_type == "canonical")]
        finals = [parse(x)[-1] for x in sub.target_rank_per_layer]
        import statistics
        med = statistics.median(finals)
        print(f"\n[CSV artifact] {args.from_csv}")
        print(f"  model={sub.model.iloc[0] if len(sub) else '?'}  n={len(finals)}  median_final_rank={med}")

    if not args.compute_ranks:
        out = Path("results/raw/diagnose_read_position_prompts.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        # store without huge dumps if needed — keep full prompts
        out.write_text(json.dumps(records, indent=2))
        print(f"\nWrote {out}")
        return

    # --- identical rank path (needs GPU + TL) ---
    import gc
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    torch.set_grad_enabled(False)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    print(f"\n[model] loading {args.model} for rank recompute…")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype], low_cpu_mem_usage=True
    )
    model = HookedTransformer.from_pretrained(
        args.model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype_map[args.dtype],
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    n_layers = model.cfg.n_layers

    def _hf_encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def final_rank(text: str, action: str, family: str) -> int:
        prompt = _wrap(text, family)
        tid, _, _, _ = resolve_target_token(tokenizer, prompt, action)
        ids = _hf_encode(prompt)
        tokens = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        acts = cache[f"blocks.{n_layers-1}.hook_resid_post"][0, -1, :].float()
        logits = acts @ model.unembed.W_U.float() + model.unembed.b_U.float()
        rank = int((logits > logits[tid]).sum().item()) + 1
        del cache, acts, logits
        torch.cuda.empty_cache()
        return rank

    # For median report use full ALGO can bank (not just --n samples) when --report-median
    eval_rows = canon if args.report_median else rows

    ranks = []
    for _, row in eval_rows.iterrows():
        r = final_rank(str(row.problem_text), _first_action(row.correct_answer), str(row.problem_family))
        ranks.append(r)
        print(f"  {row.problem_id}: final_rank={r}")

    if args.report_median and ranks:
        import statistics
        med = statistics.median(ranks)
        print(f"\n[LIVE identical path] n={len(ranks)}  median_final_rank={med}")
        print(f"  published paper value: 22472")
        print(f"  match (abs diff ≤ 1)? {abs(float(med) - 22472) <= 1}")


if __name__ == "__main__":
    main()
