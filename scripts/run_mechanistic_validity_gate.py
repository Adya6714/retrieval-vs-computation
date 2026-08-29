#!/usr/bin/env python3
"""Validity gate: can the Llama mechanistic lens separate memorized vs novel?

Reuses layer-by-layer gold-token rank extraction from
``scripts/run_mechanistic_llama_gsm_sp.py`` (same chat-direct Instruct prompt,
content-gold first token, final-layer rank).

Groups:
  HIGH-EXPOSURE (n=10): trivial arithmetic (obvious pretraining exposure)
  LOW-EXPOSURE  (n=10): GSM canonical with lowest Infini-gram proximity
                        from results/raw/GSM_P3_contamination.csv
                        (``contamination_score`` = GSM c_T / proximity)

Outputs (default under results/raw/mechanistic/):
  mechanistic_validity_gate.csv
  mechanistic_validity_gate_summary.csv

Usage:
  python3 scripts/run_mechanistic_validity_gate.py
  python3 scripts/run_mechanistic_validity_gate.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse prior GSM/SP mechanistic helpers — do not rewrite extraction.
from scripts.run_mechanistic_llama_gsm_sp import (  # noqa: E402
    DEFAULT_MODEL,
    _family_instruction,
    _gold_content_answer,
    _resolve_target_token,
)

CONTAM_PATH = REPO_ROOT / "results/raw/GSM_P3_contamination.csv"
GSM_BANK = REPO_ROOT / "data/problems/question_bank_gsm.csv"

# Trivial arithmetic: answers are single integers (content-gold = digit string).
HIGH_EXPOSURE_ITEMS: list[dict] = [
    {"problem_id": "HIGH_01", "problem_text": "What is 2 + 2?", "correct_answer": "4"},
    {"problem_id": "HIGH_02", "problem_text": "What is 10 times 10?", "correct_answer": "100"},
    {"problem_id": "HIGH_03", "problem_text": "What is half of 100?", "correct_answer": "50"},
    {"problem_id": "HIGH_04", "problem_text": "What is 5 + 5?", "correct_answer": "10"},
    {"problem_id": "HIGH_05", "problem_text": "What is 3 times 3?", "correct_answer": "9"},
    {"problem_id": "HIGH_06", "problem_text": "What is 100 minus 1?", "correct_answer": "99"},
    {"problem_id": "HIGH_07", "problem_text": "What is 7 + 3?", "correct_answer": "10"},
    {"problem_id": "HIGH_08", "problem_text": "What is 9 + 1?", "correct_answer": "10"},
    {"problem_id": "HIGH_09", "problem_text": "What is 6 times 2?", "correct_answer": "12"},
    {"problem_id": "HIGH_10", "problem_text": "What is 8 + 8?", "correct_answer": "16"},
]

PER_ITEM_COLS = [
    "problem_id",
    "group",
    "problem_text",
    "correct_answer",
    "c_T",
    "final_layer_rank",
    "n_layers",
    "gold_token_decoded",
    "gold_token_ids",
    "gold_first_token_note",
    "model",
    "device",
    "dtype",
]


def _select_low_exposure(n: int = 10) -> list[dict]:
    """Lowest Infini-gram proximity GSM canonicals.

    GSM_P3_contamination.csv has ``contamination_score`` (no column named c_T).
    That score is the GSM proximity / Infini-gram measure used elsewhere as c_T.
    Strict bottom decile of n=44 is only ~4 items (all score 0); we take the
    lowest ``n`` by contamination_score so the LOW arm has n=10 as requested.
    """
    contam = pd.read_csv(CONTAM_PATH)
    if "contamination_score" not in contam.columns:
        raise SystemExit(f"Missing contamination_score in {CONTAM_PATH}")
    contam["c_T"] = pd.to_numeric(contam["contamination_score"], errors="coerce")
    contam = contam.dropna(subset=["c_T"]).sort_values(["c_T", "problem_id"])
    n_decile = max(1, int(len(contam) * 0.1))
    bottom_decile_ids = set(contam.head(n_decile)["problem_id"].astype(str))
    low = contam.head(n).copy()

    bank = pd.read_csv(GSM_BANK, dtype=str).fillna("")
    bank = bank[bank["variant_type"].astype(str).str.strip() == "canonical"]
    bank_map = {str(r.problem_id): r for _, r in bank.iterrows()}

    items = []
    for _, r in low.iterrows():
        pid = str(r["problem_id"]).strip()
        if pid not in bank_map:
            raise KeyError(f"LOW-EXPOSURE {pid} missing from GSM bank canonical")
        br = bank_map[pid]
        items.append(
            {
                "problem_id": pid,
                "group": "LOW_EXPOSURE",
                "problem_text": str(br["problem_text"]),
                "correct_answer": str(br["correct_answer"]),
                "c_T": float(r["c_T"]),
                "in_strict_bottom_decile": pid in bottom_decile_ids,
            }
        )
    return items


def _build_queue() -> list[dict]:
    high = [
        {
            **h,
            "group": "HIGH_EXPOSURE",
            "c_T": "",  # not applicable
            "in_strict_bottom_decile": "",
        }
        for h in HIGH_EXPOSURE_ITEMS
    ]
    low = _select_low_exposure(10)
    return high + low


def _extract_final_rank(
    *,
    model,
    tokenizer,
    device,
    problem_text: str,
    correct_answer: str,
    n_layers: int,
) -> dict:
    import torch
    import torch.nn.functional as F

    gold_content = _gold_content_answer(str(correct_answer), "gsm")
    if not gold_content:
        # Numeric gold may be "4.0" from bank — strip to content digit string.
        gold_content = str(correct_answer).strip()
        if gold_content.endswith(".0") and gold_content.replace(".", "", 1).replace("-", "", 1).isdigit():
            gold_content = str(int(float(gold_content)))

    user_msg = f"{_family_instruction('gsm')}\n\nProblem:\n{problem_text}"
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
        return {
            "final_layer_rank": -1,
            "gold_token_decoded": gold_decoded,
            "gold_token_ids": json.dumps([]),
            "gold_first_token_note": "RESOLVE_FAILED",
            "n_layers": n_layers,
        }

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_fwd = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = out_fwd.hidden_states[1:]  # skip embedding; index 0 = layer 0
    W_U = model.lm_head.weight.detach().float()
    # Final layer only (same scoring as prior run's last layer row).
    h = hidden_states[-1][0, -1, :].float()
    logits = h @ W_U.T
    target_logit = logits[tid]
    rank = int((logits > target_logit).sum().item()) + 1
    del out_fwd, hidden_states, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "final_layer_rank": rank,
        "gold_token_decoded": gold_decoded,
        "gold_token_ids": json.dumps([int(x) for x in gold_token_ids]),
        "gold_first_token_note": note,
        "n_layers": n_layers,
    }


def _mann_whitney(high_ranks: list[float], low_ranks: list[float]) -> dict:
    """Two-sided MWU; also one-sided HIGH < LOW (memorized more accessible)."""
    # scipy: alternative='less' means high_ranks stochastically less than low_ranks
    u_two, p_two = stats.mannwhitneyu(high_ranks, low_ranks, alternative="two-sided")
    u_less, p_less = stats.mannwhitneyu(high_ranks, low_ranks, alternative="less")
    return {
        "n_high": len(high_ranks),
        "n_low": len(low_ranks),
        "median_rank_high": float(pd.Series(high_ranks).median()),
        "median_rank_low": float(pd.Series(low_ranks).median()),
        "mean_rank_high": float(pd.Series(high_ranks).mean()),
        "mean_rank_low": float(pd.Series(low_ranks).mean()),
        "U_two_sided": float(u_two),
        "p_two_sided": float(p_two),
        "U_high_less_than_low": float(u_less),
        "p_high_less_than_low": float(p_less),
        "separation_detected_alpha_05": bool(p_two < 0.05),
        "direction_matches_memorization_h1": bool(
            pd.Series(high_ranks).median() < pd.Series(low_ranks).median()
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out-dir", default="results/raw/mechanistic")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "mechanistic_validity_gate.csv"
    sum_path = out_dir / "mechanistic_validity_gate_summary.csv"

    queue = _build_queue()
    print(f"[setup] model={args.model} dtype={args.dtype} n={len(queue)}")
    for g in ("HIGH_EXPOSURE", "LOW_EXPOSURE"):
        sub = [x for x in queue if x["group"] == g]
        print(f"  {g}: {[x['problem_id'] for x in sub]}")

    if args.dry_run:
        rows = []
        for item in queue:
            rows.append(
                {
                    "problem_id": item["problem_id"],
                    "group": item["group"],
                    "problem_text": item["problem_text"],
                    "correct_answer": item["correct_answer"],
                    "c_T": item.get("c_T", ""),
                    "final_layer_rank": 1 if item["group"] == "HIGH_EXPOSURE" else 100,
                    "n_layers": 32,
                    "gold_token_decoded": "<dry>",
                    "gold_token_ids": "[]",
                    "gold_first_token_note": "dry_run",
                    "model": args.model,
                    "device": "dry",
                    "dtype": args.dtype,
                }
            )
        device_str = "dry"
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
        dtype = dtype_map[args.dtype]
        # Prefer CUDA, then MPS, then CPU — same extraction either way.
        if torch.cuda.is_available():
            device_map = "auto"
            device_str = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_map = None
            device_str = "mps"
            # bfloat16 on MPS can be flaky; fall back to float16 if needed at load.
        else:
            device_map = None
            device_str = "cpu"

        print(f"[model] loading {args.model} device={device_str} dtype={args.dtype}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {"torch_dtype": dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        except Exception as exc:
            if device_str == "mps" and args.dtype == "bfloat16":
                print(f"[model] bfloat16 load failed ({exc}); retry float16 on MPS")
                dtype = torch.float16
                args.dtype = "float16"
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, torch_dtype=dtype
                )
            else:
                raise
        if device_map is None:
            model = model.to(device_str)
        model.eval()
        n_layers = int(model.config.num_hidden_layers)
        device = next(model.parameters()).device
        print(f"[model] n_layers={n_layers} param_device={device}")

        rows = []
        t0 = time.time()
        for i, item in enumerate(queue):
            # Normalize GSM bank answers like "51.0" → content used by prior run.
            ans = str(item["correct_answer"]).strip()
            try:
                if "." in ans:
                    f = float(ans)
                    if f == int(f):
                        ans = str(int(f))
            except ValueError:
                pass
            ext = _extract_final_rank(
                model=model,
                tokenizer=tokenizer,
                device=device,
                problem_text=item["problem_text"],
                correct_answer=ans,
                n_layers=n_layers,
            )
            row = {
                "problem_id": item["problem_id"],
                "group": item["group"],
                "problem_text": item["problem_text"],
                "correct_answer": item["correct_answer"],
                "c_T": item.get("c_T", ""),
                "final_layer_rank": ext["final_layer_rank"],
                "n_layers": ext["n_layers"],
                "gold_token_decoded": ext["gold_token_decoded"],
                "gold_token_ids": ext["gold_token_ids"],
                "gold_first_token_note": ext["gold_first_token_note"],
                "model": args.model,
                "device": str(device),
                "dtype": args.dtype,
            }
            rows.append(row)
            print(
                f"[{i+1}/{len(queue)}] {item['group']} {item['problem_id']} "
                f"rank={ext['final_layer_rank']} gold={ext['gold_token_decoded']!r} "
                f"elapsed={time.time()-t0:.0f}s"
            )

    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_ITEM_COLS)
        w.writeheader()
        w.writerows(rows)

    high_ranks = [float(r["final_layer_rank"]) for r in rows if r["group"] == "HIGH_EXPOSURE"]
    low_ranks = [float(r["final_layer_rank"]) for r in rows if r["group"] == "LOW_EXPOSURE"]
    mw = _mann_whitney(high_ranks, low_ranks)

    # Gate rule (pre-registered intent): need separation AND memorization direction
    # (HIGH ranks < LOW ranks). Report null/messy as measured — do not reframe.
    gate_pass = bool(
        mw["separation_detected_alpha_05"] and mw["direction_matches_memorization_h1"]
    )
    summary = {
        **mw,
        "gate_pass": gate_pass,
        "c_T_column_used": "contamination_score (GSM_P3_contamination.csv; GSM proxy for c_T)",
        "strict_bottom_decile_n": max(1, int(44 * 0.1)),
        "low_selection": "lowest_10_by_contamination_score",
        "high_ids": ";".join(r["problem_id"] for r in rows if r["group"] == "HIGH_EXPOSURE"),
        "low_ids": ";".join(r["problem_id"] for r in rows if r["group"] == "LOW_EXPOSURE"),
        "high_ranks_raw": ";".join(str(int(x)) for x in high_ranks),
        "low_ranks_raw": ";".join(str(int(x)) for x in low_ranks),
        "model": args.model,
        "note": (
            "PASS only if two-sided p<0.05 AND median(HIGH)<median(LOW). "
            "If FAIL, Prompt 3 mechanistic angle must not be trusted on this lens."
        ),
    }
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("\n=== VALIDITY GATE ===")
    print(f"median HIGH={mw['median_rank_high']} LOW={mw['median_rank_low']}")
    print(f"MWU two-sided U={mw['U_two_sided']:.4g} p={mw['p_two_sided']:.4g}")
    print(f"MWU HIGH<LOW   U={mw['U_high_less_than_low']:.4g} p={mw['p_high_less_than_low']:.4g}")
    print(f"gate_pass={gate_pass}")
    print(f"wrote {per_path}")
    print(f"wrote {sum_path}")
    if not gate_pass:
        print(
            "GATE FAIL / NO SEPARATION (or wrong direction): "
            "do not proceed to Prompt 3 on this mechanistic lens."
        )
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
