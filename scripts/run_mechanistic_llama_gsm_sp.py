#!/usr/bin/env python3
"""Appendix H follow-up: Llama-3.1-8B-Instruct mechanistic probe on GSM + SP.

Fixes four pilot flaws:
  1. Content-gold tokens (not Path:/Count: scaffolding)
  2. Single consistent Instruct + chat-direct config (no base/raw-QA mix)
  3. Behavior-internals link on the *same* HF backbone (greedy do_sample=False)
  4. Adequate power: GSM 44 can/W3 (+24 W6) and SP 55 can/W3 (+50 W6)

Scope: GSM + ALGO shortest_path only (WIS/CC/BW excluded — degenerate golds).

Output (long form): results/raw/mechanistic_llama_gsm_sp_raw.csv
  one row per (problem_id, variant_type, layer)

Then run:
  python3 scripts/analyze_mechanistic_llama_gsm_sp.py

Usage (GPU / Colab T4):
  python3 scripts/run_mechanistic_llama_gsm_sp.py --resume
  python3 scripts/run_mechanistic_llama_gsm_sp.py --families gsm --limit 2  # smoke
  python3 scripts/run_mechanistic_llama_gsm_sp.py --dry-run --limit 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify import verify_gsm_answer  # noqa: E402
from probes.contamination.verify_algo import verify_algo  # noqa: E402

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUT = "results/raw/mechanistic_llama_gsm_sp_raw.csv"

GSM_BANK = "data/problems/question_bank_gsm.csv"
ALGO_BANK = "data/problems/question_bank_algo.csv"
GSM_P1_LLAMA = "results/raw/GSM_P1_behavioral_llama.csv"

RAW_COLS = [
    "problem_id",
    "family",
    "subtype",
    "variant_type",
    "layer",
    "n_layers",
    "rank",
    "logprob",
    "cosine_to_gold_unembed",
    "gold_value",
    "gold_token_ids",
    "gold_token_decoded",
    "gold_first_token_note",
    "model",
    "model_answer",
    "model_correct",
    "p1_behavioral_correct",
    "verify_reason",
]


def _first_action(answer: str) -> str:
    lines = [ln.strip() for ln in str(answer).split("\n") if ln.strip()]
    if not lines:
        return ""
    return re.sub(r"^\d+[\.\)]\s*", "", lines[0])


def _gold_content_answer(answer: str, family: str) -> str:
    """Content-bearing gold span (first token of this span is ranked)."""
    action = _first_action(answer)
    if not action:
        return ""
    fam = family.lower()
    if fam in ("sp", "shortest_path", "algo", "algorithmic") or action.lower().startswith("path:"):
        m = re.match(r"^Path:\s*(.+)$", action, flags=re.I)
        content = m.group(1).strip() if m else action
        # Drop trailing cost annotation for token targeting; keep full string as gold_value separately.
        content = re.split(r",\s*Cost\s*:", content, maxsplit=1, flags=re.I)[0].strip()
        return content
    return action.strip()


def _family_instruction(family: str) -> str:
    if family == "gsm":
        return "Solve the problem. Output ONLY the final numerical answer, nothing else."
    return (
        "Solve the problem. Output ONLY the first action of the solution "
        "(no numbering, no explanation)."
    )


def _sp_subtype_label(instance_type: str) -> str:
    it = str(instance_type).strip().lower()
    if it == "adversarial":
        return "SP-chall"
    if it == "standard":
        return "SP-std"
    return f"SP-{it or 'unknown'}"


def _resolve_target_token(tokenizer, prompt: str, answer: str) -> tuple[int, str, str, list[int]]:
    """Prompt-aware first-token id for `answer` following `prompt`."""
    if not answer.strip():
        return -1, "", "", []

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prompt_ids = enc(prompt)
    answer_ids_bare = enc(answer)
    candidates: list[tuple[str, int, int, list[int]]] = []
    for sep in ("", " "):
        joint = enc(prompt + sep + answer)
        if len(joint) <= len(prompt_ids):
            continue
        if joint[: len(prompt_ids)] != prompt_ids:
            continue
        rest = joint[len(prompt_ids) :]
        tid = int(rest[0])
        candidates.append((sep, tid, len(joint), rest))
    if not candidates:
        if not answer_ids_bare:
            return -1, "", "FALLBACK", []
        tid = int(answer_ids_bare[0])
        return tid, tokenizer.decode([tid]), "FALLBACK", answer_ids_bare
    candidates.sort(key=lambda c: c[2])
    sep, tid, _, rest = candidates[0]
    return tid, tokenizer.decode([tid]), repr(sep), rest


def _load_queue(families: list[str], variants: list[str], limit: int | None) -> list[dict]:
    rows: list[dict] = []

    def _variant_norm(v: str) -> str:
        return "canonical" if str(v).strip().lower() == "canonical" else str(v).strip().upper()

    def _select_variant(df: pd.DataFrame, vkey: str) -> pd.DataFrame:
        if vkey == "canonical":
            return df[df["variant_type"].astype(str).str.strip() == "canonical"]
        return df[df["variant_type"].astype(str).str.strip().str.upper() == vkey]

    if "gsm" in families:
        gsm = pd.read_csv(REPO_ROOT / GSM_BANK, dtype=str).fillna("")
        for v in variants:
            vkey = _variant_norm(v)
            sub = _select_variant(gsm, vkey)
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "problem_id": str(r["problem_id"]).strip(),
                        "family": "gsm",
                        "subtype": "GSM",
                        "variant_type": vkey,
                        "problem_text": str(r["problem_text"]),
                        "correct_answer": str(r["correct_answer"]),
                        "difficulty_params": str(r.get("difficulty_params", "{}")),
                        "problem_subtype": "gsm",
                    }
                )

    if "sp" in families:
        algo = pd.read_csv(REPO_ROOT / ALGO_BANK, dtype=str).fillna("")
        algo = algo[algo["problem_subtype"].astype(str).str.strip() == "shortest_path"].copy()
        for v in variants:
            vkey = _variant_norm(v)
            sub = _select_variant(algo, vkey)
            for _, r in sub.iterrows():
                rows.append(
                    {
                        "problem_id": str(r["problem_id"]).strip(),
                        "family": "sp",
                        "subtype": _sp_subtype_label(r.get("instance_type", "")),
                        "variant_type": vkey,
                        "problem_text": str(r["problem_text"]),
                        "correct_answer": str(r["correct_answer"]),
                        "difficulty_params": str(r.get("difficulty_params", "{}")),
                        "problem_subtype": "shortest_path",
                        "instance_type": str(r.get("instance_type", "")),
                    }
                )

    # Stable order: family, variant, problem_id
    variant_order = {v: i for i, v in enumerate(["canonical", "W3", "W6"])}
    fam_order = {"gsm": 0, "sp": 1}
    rows.sort(
        key=lambda d: (
            fam_order.get(d["family"], 9),
            variant_order.get(d["variant_type"], 9),
            d["problem_id"],
        )
    )
    if limit is not None:
        # Limit applies per (family, variant) so smoke tests cover both arms.
        capped: list[dict] = []
        counts: dict[tuple[str, str], int] = {}
        for d in rows:
            key = (d["family"], d["variant_type"])
            if counts.get(key, 0) >= limit:
                continue
            counts[key] = counts.get(key, 0) + 1
            capped.append(d)
        rows = capped
    return rows


def _load_p1_gsm_labels() -> dict[tuple[str, str], bool | None]:
    path = REPO_ROOT / GSM_P1_LLAMA
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    bank = pd.read_csv(REPO_ROOT / GSM_BANK, dtype=str)
    bank_ids = set(bank[bank["variant_type"] == "canonical"]["problem_id"].astype(str))
    df = df[df["problem_id"].astype(str).isin(bank_ids)].copy()
    out: dict[tuple[str, str], bool | None] = {}
    for _, r in df.iterrows():
        pid = str(r["problem_id"]).strip()
        vt = str(r["variant_type"]).strip()
        if vt.lower() == "w6":
            vt = "W6"
        elif vt.upper().startswith("W") and vt != "canonical":
            vt = vt.upper()
        val = str(r.get("behavioral_correct", "")).strip().lower()
        if val in {"true", "1", "yes"}:
            out[(pid, vt)] = True
        elif val in {"false", "0", "no"}:
            out[(pid, vt)] = False
        else:
            out[(pid, vt)] = None
    return out


def _existing_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        return set()
    done = set()
    for _, r in df.iterrows():
        done.add((str(r["problem_id"]).strip(), str(r["variant_type"]).strip()))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--families", nargs="+", default=["gsm", "sp"], choices=["gsm", "sp"])
    ap.add_argument("--variants", nargs="+", default=["canonical", "W3", "W6"], choices=["canonical", "W3", "W6"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="Max instances per (family, variant).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="No model load; write placeholder layer rows.")
    ap.add_argument("--n-layers-dry", type=int, default=32, help="Placeholder layer count for --dry-run.")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    queue = _load_queue(args.families, args.variants, args.limit)
    p1_labels = _load_p1_gsm_labels()
    done = _existing_done(out) if args.resume else set()
    if not args.resume and out.exists():
        out.write_text("")

    print(f"[setup] model={args.model} dtype={args.dtype} families={args.families} variants={args.variants}")
    print(f"[setup] queue={len(queue)} already_done={len(done)} output={out}")
    for fam in args.families:
        for vt in args.variants:
            vnorm = "canonical" if vt == "canonical" else vt.upper()
            n = sum(1 for d in queue if d["family"] == fam and d["variant_type"] == vnorm)
            print(f"  {fam}/{vnorm}: {n}")

    write_header = not out.exists() or out.stat().st_size == 0

    if args.dry_run:
        tokenizer = None
        model = None
        n_layers = args.n_layers_dry
        device = "cpu"
        print("[dry-run] skipping model load")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.set_grad_enabled(False)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        print(f"[model] loading {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            device_map="auto",
        )
        model.eval()
        n_layers = int(model.config.num_hidden_layers)
        device = next(model.parameters()).device
        print(f"[model] n_layers={n_layers} device={device}")

    t0 = time.time()
    n_written = 0

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLS)
        if write_header:
            writer.writeheader()

        for i, item in enumerate(queue):
            pid = item["problem_id"]
            vt = item["variant_type"]
            if (pid, vt) in done:
                continue

            family = item["family"]
            gold_value_full = str(item["correct_answer"])
            gold_content = _gold_content_answer(gold_value_full, family)
            # Prefer content span as gold_value for audit; keep note if Path stripped.
            gold_value = gold_content if gold_content else gold_value_full

            if args.dry_run:
                gold_token_ids = [0]
                gold_decoded = "<dry>"
                ranks = list(range(1, n_layers + 1))
                logprobs = [-1.0] * n_layers
                cosines = [0.0] * n_layers
                model_answer = ""
                model_correct = False
                verify_reason = "dry_run"
                note = "first_token_of_content_span"
            else:
                import torch
                import torch.nn.functional as F

                user_msg = f"{_family_instruction(family)}\n\nProblem:\n{item['problem_text']}"
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                tid, gold_decoded, _sep, gold_token_ids = _resolve_target_token(
                    tokenizer, prompt, gold_content
                )
                note = (
                    "first_token_of_content_span"
                    if len(gold_token_ids) <= 1
                    else f"first_of_{len(gold_token_ids)}_tokens"
                )
                if tid < 0:
                    ranks = [-1] * n_layers
                    logprobs = [float("nan")] * n_layers
                    cosines = [float("nan")] * n_layers
                else:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out_fwd = model(
                            **inputs,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                    # hidden_states: (embed + n_layers); score post-layer states 1..n
                    hidden_states = out_fwd.hidden_states[1:]  # skip embedding
                    # Unembedding direction: lm_head weight row for gold token
                    W_U = model.lm_head.weight.detach().float()  # [vocab, d]
                    u = W_U[tid]
                    u_norm = F.normalize(u.unsqueeze(0), dim=-1)

                    ranks = []
                    logprobs = []
                    cosines = []
                    for layer_h in hidden_states:
                        h = layer_h[0, -1, :].float()  # last prompt position
                        logits = h @ W_U.T
                        target_logit = logits[tid]
                        rank = int((logits > target_logit).sum().item()) + 1
                        lp = float(F.log_softmax(logits, dim=-1)[tid].item())
                        cos = float(F.cosine_similarity(h.unsqueeze(0), u_norm, dim=-1).item())
                        ranks.append(rank)
                        logprobs.append(round(lp, 6))
                        cosines.append(round(cos, 6))
                    del out_fwd, hidden_states, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Greedy generation for behavior label (same backbone)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = gen[0, inputs["input_ids"].shape[1] :]
                model_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                del inputs, gen
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if family == "gsm":
                    model_correct = bool(verify_gsm_answer(model_answer, item["correct_answer"]))
                    verify_reason = "gsm_match" if model_correct else "gsm_mismatch"
                else:
                    verified, reason, _meta = verify_algo(
                        pid,
                        model_answer,
                        item["correct_answer"],
                        item["problem_subtype"],
                        vt,
                        item["difficulty_params"],
                    )
                    model_correct = bool(verified)
                    verify_reason = reason

            p1 = p1_labels.get((pid, vt))
            p1_str = "" if p1 is None else str(bool(p1))

            for layer_i in range(n_layers):
                writer.writerow(
                    {
                        "problem_id": pid,
                        "family": family,
                        "subtype": item["subtype"],
                        "variant_type": vt,
                        "layer": layer_i,
                        "n_layers": n_layers,
                        "rank": ranks[layer_i] if layer_i < len(ranks) else "",
                        "logprob": logprobs[layer_i] if layer_i < len(logprobs) else "",
                        "cosine_to_gold_unembed": cosines[layer_i] if layer_i < len(cosines) else "",
                        "gold_value": gold_value,
                        "gold_token_ids": json.dumps([int(x) for x in gold_token_ids]),
                        "gold_token_decoded": gold_decoded,
                        "gold_first_token_note": note,
                        "model": args.model,
                        "model_answer": model_answer,
                        "model_correct": model_correct,
                        "p1_behavioral_correct": p1_str,
                        "verify_reason": verify_reason,
                    }
                )
            f.flush()
            done.add((pid, vt))
            n_written += 1
            final_rank = ranks[-1] if ranks else None
            elapsed = time.time() - t0
            print(
                f"[{i+1}/{len(queue)}] {family}/{vt} {pid} "
                f"final_rank={final_rank} correct={model_correct} "
                f"gold={gold_decoded!r} | wrote={n_written} elapsed={elapsed:.0f}s"
            )

    print("\n=== DONE ===")
    print(f"Output: {out}")
    if out.exists() and out.stat().st_size > 0:
        df = pd.read_csv(out)
        print(f"Rows: {len(df)} (instance×layer)")
        inst = df.drop_duplicates(["problem_id", "variant_type"])
        print(inst.groupby(["family", "variant_type"]).size().to_string())
        print(
            "model_correct rates:\n",
            inst.groupby("family")["model_correct"]
            .apply(lambda s: f"{s.astype(str).str.lower().isin(['true','1']).sum()}/{len(s)}")
            .to_string(),
        )


if __name__ == "__main__":
    main()
