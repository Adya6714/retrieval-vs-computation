"""Headless 7B mechanistic sweep — paper-aligned.

Computes per-problem:
  - crystallization_layer (logit-lens top-5; -1 if never)
  - layer_cosine_similarities (canonical vs W2/W3/W4 mean per layer)
  - W6 is run as its own row (W6 text → CL on W6 answer; cosine vs same-pid canonical)

Resume-safe: skips (problem_id, variant_type) already present in the output CSV.

Source banks: LOCAL repo only:
  - data/problems/question_bank_gsm.csv
  - data/problems/question_bank_algo.csv
  - data/problems/question_bank_bw.csv   (optional; skip by default)

Output: results/raw/mechanistic_sweep_7b.csv

Usage (T4/A40/A100, fp16):

    python3 scripts/run_mechanistic_sweep_7b.py \\
        --families gsm algo \\
        --variants canonical w6 \\
        --output results/raw/mechanistic_sweep_7b.csv

Headless tip — run inside tmux/screen, log to file:

    tmux new -s mech
    python3 scripts/run_mechanistic_sweep_7b.py \\
        --families gsm algo --variants canonical w6 2>&1 \\
        | tee logs/mechanistic_sweep_7b_$(date +%F).log
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Default repo-spec'd mechanistic model (see configs/models.yaml: mechanistic: true)
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

BANK_PATHS = {
    "gsm": "data/problems/question_bank_gsm.csv",
    "algo": "data/problems/question_bank_algo.csv",
    "bw": "data/problems/question_bank_bw.csv",
}

OUT_COLS = [
    "problem_id",
    "problem_family",
    "variant_type",
    "model",
    "crystallization_layer",
    "n_layers_processed",
    "layer_cosine_similarities",
    # added in graded-metric pass: rank-of-target & log-prob-of-target per layer
    # (lets us recompute CL at any top-K threshold post-hoc and regress depth
    # against contamination)
    "target_rank_per_layer",
    "target_logprob_per_layer",
    "target_token_id",
    "target_token_decoded",
]


def _build_variants_lookup(banks: dict[str, pd.DataFrame]) -> dict[str, list[tuple[str, str]]]:
    """problem_id -> list of (variant_type, problem_text) for W2/W3/W4 only."""
    variants: dict[str, list[tuple[str, str]]] = {}
    for df in banks.values():
        for _, row in df.iterrows():
            vtype = str(row.get("variant_type", "")).strip()
            if vtype in ("W2", "W3", "W4"):
                pid = str(row["problem_id"]).strip()
                variants.setdefault(pid, []).append((vtype, str(row["problem_text"])))
    return variants


def _build_w6_lookup(banks: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """problem_id -> W6 row (if present)."""
    w6: dict[str, pd.Series] = {}
    for df in banks.values():
        for _, row in df.iterrows():
            vtype = str(row.get("variant_type", "")).strip()
            if vtype == "W6":
                w6[str(row["problem_id"]).strip()] = row
    return w6


def _canonical_rows(banks: dict[str, pd.DataFrame], families: list[str]) -> list[tuple[str, pd.Series]]:
    """Ordered list of (family_tag, row) for all canonical rows across requested families."""
    out: list[tuple[str, pd.Series]] = []
    for fam in families:
        df = banks.get(fam)
        if df is None:
            continue
        canon = df[df["variant_type"] == "canonical"]
        for _, row in canon.iterrows():
            out.append((fam, row))
    return out


def _first_action(answer: str) -> str:
    """First non-empty line of correct_answer, stripped of leading '1. '/'1) '."""
    lines = [l.strip() for l in str(answer).split("\n") if l.strip()]
    if not lines:
        return ""
    return re.sub(r"^\d+[\.\)]\s*", "", lines[0])


def _load_existing(output_path: Path) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Return (existing df, done set keyed by (problem_id, variant_type))."""
    if output_path.exists() and output_path.stat().st_size > 0:
        df = pd.read_csv(output_path)
        if "model" in df.columns:
            # drop GPT-2 or 0.5B pilot artefacts if present
            df = df[~df["model"].isin(["gpt2", "Qwen/Qwen2.5-0.5B-Instruct"])].copy()
        done: set[tuple[str, str]] = set()
        for _, r in df.iterrows():
            done.add((str(r["problem_id"]).strip(), str(r.get("variant_type", "canonical")).strip()))
        return df, done
    return pd.DataFrame(columns=OUT_COLS), set()


def _write_row(output_path: Path, results: list[dict]) -> None:
    pd.DataFrame(results, columns=OUT_COLS).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 7B mechanistic sweep (local banks only).")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["gsm", "algo"],
        choices=["gsm", "algo", "bw"],
        help="Which families to sweep (default: gsm algo).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["canonical", "w6"],
        choices=["canonical", "w6"],
        help="Which variants to sweep (default: canonical w6).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"HF model id (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model + activations (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16).",
    )
    parser.add_argument(
        "--output",
        default="results/raw/mechanistic_sweep_7b.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N problems per (family, variant) pass — useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model load; emit per-row placeholder rows with CL=-1 and sims=[] for pipeline check.",
    )
    parser.add_argument(
        "--prompt-mode",
        default="chat-direct",
        choices=["chat-direct", "chat", "raw", "raw-qa"],
        help="'chat-direct' (default): chat template + family-aware direct-answer instruction "
             "(intended for instruct-tuned models — but RLHF distortion usually still suppresses CL). "
             "'chat' = chat template only. "
             "'raw' = feed problem_text directly (oldest behaviour; reproduces the original cosine CSV). "
             "'raw-qa' (recommended for BASE models): wrap in 'Problem: <text>\\n\\nAnswer: ' so the "
             "last position is the completion-prediction slot — this is the standard prompt for "
             "logit-lens on pre-trained (non-RLHF) models.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    print(f"[setup] cwd: {os.getcwd()}")

    # Validate banks
    banks: dict[str, pd.DataFrame] = {}
    for fam in args.families:
        path = BANK_PATHS[fam]
        if not Path(path).exists():
            sys.exit(f"[setup] missing bank: {path}")
        banks[fam] = pd.read_csv(path, dtype=str).fillna("")
        print(f"[setup] loaded bank '{fam}': {len(banks[fam])} rows ({len(banks[fam][banks[fam].variant_type=='canonical'])} canonical)")

    # Build variant + W6 lookups across requested families
    variants_lookup = _build_variants_lookup(banks)
    w6_lookup = _build_w6_lookup(banks)
    print(f"[setup] W2/W3/W4 lookup keys: {len(variants_lookup)} | W6 lookup keys: {len(w6_lookup)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_existing, done = _load_existing(output_path)
    results: list[dict] = df_existing.to_dict("records")
    print(f"[setup] existing rows: {len(df_existing)} | done keys (pid, variant): {len(done)}")

    # Load model + tokenizer
    if not args.dry_run:
        # Reduce fragmentation on the GPU during the transient .clone() inside
        # transformer_lens weight processing. Must be set before importing torch.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        import torch

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"[gpu] {torch.cuda.get_device_name(0)} | free={free/1e9:.1f} GB / {total/1e9:.1f} GB")
        else:
            print("[gpu] WARNING: cuda not available; this run will be very slow.")

        print(f"[model] loading {args.model} in {args.dtype} on {args.device}")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from transformer_lens import HookedTransformer
        import torch.nn.functional as F

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # Load HF model on CPU first to avoid a double-resident copy on GPU when
        # HookedTransformer subsequently pulls weights and clones during processing.
        # We free `hf_model` after TL has consumed its state dict.
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            low_cpu_mem_usage=True,
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
        # Free the (now-redundant) HF model
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            print(f"[gpu] post-load: free={free/1e9:.1f} GB / {total/1e9:.1f} GB")

        n_layers = model.cfg.n_layers
        print(f"[model] loaded | n_layers={n_layers} | d_model={model.cfg.d_model}")
        print(f"[prompt] mode = {args.prompt_mode}")

        def _family_instruction(family: str) -> str:
            f = (family or "").strip().lower()
            if f in ("gsm", "arithmetic_reasoning"):
                return "Solve the problem. Output ONLY the final numerical answer, nothing else."
            if f in ("bw", "blocksworld", "blocks world"):
                return "Solve the planning problem. Output ONLY the first plan step (action + arguments), no numbering, no explanation."
            return "Solve the problem. Output ONLY the first action of the solution (no numbering, no explanation)."

        def _wrap(text: str, family: str = "") -> str:
            """Build the model input according to args.prompt_mode."""
            if args.prompt_mode == "raw":
                return text
            if args.prompt_mode == "raw-qa":
                # Completion-style prompt; standard for logit-lens on base models.
                # Trailing newline gives a clean BPE boundary — the next token
                # the model predicts is the answer's content character (e.g. '5',
                # 'Un', 'Take'), not a literal space or a space-prefixed merge.
                return f"Problem: {text}\n\nAnswer:\n"
            user_msg = text
            if args.prompt_mode == "chat-direct":
                user_msg = f"{_family_instruction(family)}\n\nProblem:\n{text}"
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                add_generation_prompt=True,
                tokenize=False,
            )

        def _hf_encode(text: str) -> list[int]:
            """Tokenize via HF tokenizer (no special tokens added). Consistent
            with target-token resolution below — TL's model.to_tokens may not
            match HF in edge cases involving chat-template special tokens."""
            return tokenizer.encode(text, add_special_tokens=False)

        def _last_token_resid(text: str, family: str = "") -> "torch.Tensor":
            wrapped = _wrap(text, family)
            ids = _hf_encode(wrapped)
            tokens = torch.tensor([ids], device=args.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            keys = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
            acts = torch.stack([cache[k][0, -1, :].float() for k in keys])
            del cache
            torch.cuda.empty_cache()
            gc.collect()
            return acts

        def _resolve_target_token(prompt: str, answer: str) -> tuple[int, str, str]:
            """Find the token id that would NATURALLY follow `prompt` if the next
            text were `answer`. This is THE key correctness fix: tokenize prompt+answer
            jointly (with the natural separator) and read off the first id after
            the prompt's tokens. BPE tokens differ between e.g. '5' and ' 5'; the
            model predicts the one that matches its prompt context, not the bare
            answer string.

            Returns (target_id, target_decoded, separator_used).
            """
            if not answer.strip():
                return -1, "", ""
            prompt_ids = _hf_encode(prompt)

            # Try both no-separator and single-space; pick whichever (a) keeps
            # prompt_ids as a strict prefix of the joint encoding, and (b) yields
            # the SHORTEST joint tokenization (= BPE-natural).
            candidates: list[tuple[str, int, int]] = []
            for sep in ("", " "):
                joint = _hf_encode(prompt + sep + answer)
                if len(joint) <= len(prompt_ids):
                    continue
                if joint[: len(prompt_ids)] != prompt_ids:
                    # boundary changed by BPE merge — skip this sep
                    continue
                tid = int(joint[len(prompt_ids)])
                candidates.append((sep, tid, len(joint)))
            if not candidates:
                # fallback: tokenize answer in isolation (old behaviour)
                bare = _hf_encode(answer)
                if not bare:
                    return -1, "", ""
                tid = int(bare[0])
                return tid, tokenizer.decode([tid]), "FALLBACK"
            candidates.sort(key=lambda c: c[2])
            sep, tid, _ = candidates[0]
            return tid, tokenizer.decode([tid]), repr(sep)

        def logit_lens_metrics(text: str, target_action: str, family: str = "") -> dict:
            """Return all per-layer metrics for the answer-token logit-lens probe.

            Key correctness fix: target_id is the token that NATURALLY follows
            the actual prompt (computed via _resolve_target_token), not the first
            token of `target_action` in isolation. This matters because BPE
            tokenizers give different ids to '5' vs ' 5' vs '\\n5'.

            Computes in one forward pass:
              - crystallization_layer: earliest layer at which target ID enters top-5
              - target_rank_per_layer: list[int] of size n_layers (1 = top-1)
              - target_logprob_per_layer: list[float] log-softmax probability of target ID
            """
            if not target_action.strip():
                return {"cl": -1, "ranks": [], "logprobs": [], "target_id": -1, "target_decoded": ""}
            prompt = _wrap(text, family)
            target_id, target_decoded, _sep = _resolve_target_token(prompt, target_action)
            if target_id < 0:
                return {"cl": -1, "ranks": [], "logprobs": [], "target_id": -1, "target_decoded": ""}
            acts = _last_token_resid(text, family)
            W_U = model.unembed.W_U.float()
            b_U = model.unembed.b_U.float()
            cl = -1
            ranks: list[int] = []
            logprobs: list[float] = []
            for layer_i in range(n_layers):
                logits = acts[layer_i] @ W_U + b_U
                # rank: how many logits are STRICTLY greater than target's logit (+1)
                target_logit = logits[target_id]
                rank = int((logits > target_logit).sum().item()) + 1
                # log-softmax of target
                lp = float(F.log_softmax(logits, dim=-1)[target_id].item())
                ranks.append(rank)
                logprobs.append(round(lp, 4))
                if cl == -1 and rank <= 5:
                    cl = layer_i
            del acts, W_U, b_U
            torch.cuda.empty_cache()
            gc.collect()
            return {
                "cl": cl,
                "ranks": ranks,
                "logprobs": logprobs,
                "target_id": target_id,
                "target_decoded": target_decoded,
            }

        def cosine_sims(canonical_text: str, variant_texts: list[str], family: str = "") -> list[float]:
            if not variant_texts:
                return []
            canon_acts = _last_token_resid(canonical_text, family)
            sims_per_var = []
            for vtext in variant_texts:
                var_acts = _last_token_resid(vtext, family)
                sims = F.cosine_similarity(canon_acts, var_acts, dim=1)
                sims_per_var.append(sims)
                del var_acts
                gc.collect()
            del canon_acts
            torch.cuda.empty_cache()
            gc.collect()
            mean_sims = (sum(sims_per_var) / len(sims_per_var)).tolist()
            return [round(float(x), 4) for x in mean_sims]
    else:
        print("[dry-run] skipping model load — will write placeholder rows.")
        n_layers = 28

        def logit_lens_metrics(text: str, target_action: str, family: str = "") -> dict:
            return {
                "cl": -1,
                "ranks": [0] * n_layers,
                "logprobs": [0.0] * n_layers,
                "target_id": -1,
                "target_decoded": "",
            }

        def cosine_sims(canonical_text: str, variant_texts: list[str], family: str = "") -> list[float]:
            return [0.0] * n_layers if variant_texts else []

    # ===== CANONICAL PASS =====
    if "canonical" in args.variants:
        rows = _canonical_rows(banks, args.families)
        if args.limit is not None:
            rows = rows[: args.limit]
        print(f"[canonical] queue: {len(rows)} rows | already done: {sum(1 for fam, r in rows if (str(r['problem_id']).strip(), 'canonical') in done)}")
        t0 = time.time()
        for i, (fam, row) in enumerate(rows):
            pid = str(row["problem_id"]).strip()
            if (pid, "canonical") in done:
                continue
            problem_text = str(row["problem_text"])
            family = str(row.get("problem_family", fam))
            first_action = _first_action(row.get("correct_answer", ""))

            try:
                m = logit_lens_metrics(problem_text, first_action, family)
                cl, ranks, logprobs = m["cl"], m["ranks"], m["logprobs"]
                target_id, target_decoded = m["target_id"], m["target_decoded"]
            except Exception as e:
                print(f"  ! logit_lens_metrics failed for {pid}: {e}")
                cl, ranks, logprobs = -1, [], []
                target_id, target_decoded = -1, ""

            variant_texts = [vt for _, vt in variants_lookup.get(pid, [])]
            try:
                sims = cosine_sims(problem_text, variant_texts, family)
            except Exception as e:
                print(f"  ! cosine_sims failed for {pid}: {e}")
                sims = []

            results.append({
                "problem_id": pid,
                "problem_family": family,
                "variant_type": "canonical",
                "model": args.model,
                "crystallization_layer": cl,
                "n_layers_processed": n_layers,
                "layer_cosine_similarities": json.dumps(sims),
                "target_rank_per_layer": json.dumps(ranks),
                "target_logprob_per_layer": json.dumps(logprobs),
                "target_token_id": target_id,
                "target_token_decoded": target_decoded,
            })
            done.add((pid, "canonical"))
            _write_row(output_path, results)
            elapsed = time.time() - t0
            min_rank = min(ranks) if ranks else -1
            print(f"[{i+1}/{len(rows)}] {pid} fam={fam} CL={cl} min_rank={min_rank} sims_len={len(sims)} | elapsed={elapsed:.0f}s")

    # ===== W6 PASS =====
    if "w6" in args.variants:
        w6_queue: list[tuple[str, pd.Series]] = []
        for fam in args.families:
            df = banks.get(fam)
            if df is None:
                continue
            w6_rows = df[df["variant_type"].isin(["W6", "w6"])]
            for _, row in w6_rows.iterrows():
                w6_queue.append((fam, row))
        if args.limit is not None:
            w6_queue = w6_queue[: args.limit]
        print(f"[w6] queue: {len(w6_queue)} rows | already done: {sum(1 for fam, r in w6_queue if (str(r['problem_id']).strip() + '_W6', 'W6') in done)}")

        t0 = time.time()
        for i, (fam, row) in enumerate(w6_queue):
            base_pid = str(row["problem_id"]).strip()
            tagged_pid = base_pid + "_W6"
            if (tagged_pid, "W6") in done:
                continue
            w6_text = str(row["problem_text"])
            family = str(row.get("problem_family", fam))
            first_action = _first_action(row.get("correct_answer", ""))

            try:
                m = logit_lens_metrics(w6_text, first_action, family)
                cl, ranks, logprobs = m["cl"], m["ranks"], m["logprobs"]
                target_id, target_decoded = m["target_id"], m["target_decoded"]
            except Exception as e:
                print(f"  ! logit_lens_metrics failed for {tagged_pid}: {e}")
                cl, ranks, logprobs = -1, [], []
                target_id, target_decoded = -1, ""

            canon = banks[fam][
                (banks[fam]["problem_id"] == base_pid) & (banks[fam]["variant_type"] == "canonical")
            ]
            if not canon.empty:
                try:
                    sims = cosine_sims(str(canon.iloc[0]["problem_text"]), [w6_text], family)
                except Exception as e:
                    print(f"  ! cosine_sims failed for {tagged_pid}: {e}")
                    sims = []
            else:
                sims = []

            results.append({
                "problem_id": tagged_pid,
                "problem_family": family,
                "variant_type": "W6",
                "model": args.model,
                "crystallization_layer": cl,
                "n_layers_processed": n_layers,
                "layer_cosine_similarities": json.dumps(sims),
                "target_rank_per_layer": json.dumps(ranks),
                "target_logprob_per_layer": json.dumps(logprobs),
                "target_token_id": target_id,
                "target_token_decoded": target_decoded,
            })
            done.add((tagged_pid, "W6"))
            _write_row(output_path, results)
            elapsed = time.time() - t0
            min_rank = min(ranks) if ranks else -1
            print(f"[w6 {i+1}/{len(w6_queue)}] {tagged_pid} fam={fam} CL={cl} min_rank={min_rank} sims_len={len(sims)} | elapsed={elapsed:.0f}s")

    print("\n=== DONE ===")
    df_final = pd.read_csv(output_path)
    print(f"Output: {output_path}")
    print(f"Rows: {len(df_final)}")
    for fam in sorted(df_final["problem_family"].unique()):
        sub = df_final[df_final["problem_family"] == fam]
        never = (sub["crystallization_layer"] == -1).sum()
        cl_vals = sub[sub["crystallization_layer"] != -1]["crystallization_layer"]
        print(f"  {fam}: n={len(sub)} | never_crystallized={never}/{len(sub)} | mean_CL_when_crystallized={cl_vals.mean():.1f}")


if __name__ == "__main__":
    main()
