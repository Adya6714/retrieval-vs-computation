#!/usr/bin/env python3
"""Tier-1: Llama-3.1-8B-Instruct greedy behavioral re-run (BW / GSM / ALGO).

Re-scores the exact published item sets under do_sample=False so we can test
whether the BW rename (−22.0pp) and GSM/ALGO W3 effects survive determinism.

Outputs (results/raw/):
  llama_greedy_rerun_bw.csv
  llama_greedy_rerun_gsm.csv
  llama_greedy_rerun_algo.csv
  llama_determinism_check.csv

Usage (GPU):
  python3 scripts/llama_greedy_behavioral_rerun.py --families bw gsm algo --resume
  python3 scripts/llama_greedy_behavioral_rerun.py --determinism-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify import verify_answer  # noqa: E402
from probes.contamination.verify_algo import verify_algo  # noqa: E402

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_OR = "meta-llama/llama-3.1-8b-instruct"

INSTR = {
    "gsm": "Solve the problem. Output ONLY the final numerical answer, nothing else.",
    "algo": (
        "Solve the problem. Output ONLY the first action of the solution "
        "(no numbering, no explanation)."
    ),
    "bw": (
        "Solve the planning problem. Output ONLY the plan as a numbered list "
        "of actions, nothing else."
    ),
}

OUT_SCHEMA = ["problem_id", "variant_type", "model", "behavioral_correct"]
OUT_EXTRA = ["raw_response", "family", "correct_answer", "verify_family"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort; some ops still non-deterministic on GPU.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    if v.lower() == "canonical":
        return "canonical"
    return v.upper()


def _load_banks() -> dict[tuple[str, str], dict]:
    """(problem_id, variant_type) -> row dict with problem_text, correct_answer, ..."""
    lookup: dict[tuple[str, str], dict] = {}
    for path, default_fam in [
        (REPO_ROOT / "data/problems/question_bank_bw.csv", "blocksworld"),
        (REPO_ROOT / "data/problems/question_bank_gsm.csv", "gsm"),
        (REPO_ROOT / "data/problems/question_bank_algo.csv", "algorithmic"),
    ]:
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, r in df.iterrows():
            pid = str(r["problem_id"]).strip()
            vt = _norm_vt(r["variant_type"])
            subtype = str(r.get("problem_subtype", "")).strip().lower()
            fam = str(r.get("problem_family", default_fam)).strip().lower()
            # Must match OpenRouter P1 routing in BW_P1_SCR_run_behavioral_sweep:
            # bank problem_family "arithmetic_reasoning" → verify_gsm_answer (last number).
            # Labeling GSM as "gsm" incorrectly hits _verify_numeric (first number) → 0/44.
            if subtype == "mystery_blocksworld":
                verifier_family = "mystery_blocksworld"
            elif subtype == "blocksworld" or fam in {"planning_suite", "blocksworld"}:
                verifier_family = "blocksworld"
            elif fam == "arithmetic_reasoning" or "arith" in fam or pid.startswith("GSM"):
                verifier_family = "arithmetic_reasoning"
            elif subtype in {"shortest_path", "coin_change", "wis", "wis_independent_set"}:
                verifier_family = "algo"
            else:
                verifier_family = (
                    "arithmetic_reasoning"
                    if pid.startswith("GSM")
                    else ("blocksworld" if pid.startswith(("BW", "MBW")) else "algo")
                )
            lookup[(pid, vt)] = {
                "problem_id": pid,
                "variant_type": vt,
                "problem_text": str(r["problem_text"]),
                "correct_answer": str(r["correct_answer"]),
                "problem_family": fam,
                "problem_subtype": subtype or str(r.get("problem_subtype", "")),
                "difficulty_params": str(r.get("difficulty_params", "{}")),
                "verifier_family": verifier_family,
            }
    return lookup


def _bw_paired_items(lookup: dict) -> list[dict]:
    """Llama can∩W5 pairs from BW_P1_behavioral.csv, restricted to BW_/MBW_.

    NOTE: The raw BW_P1 file also contains GSM rows under the Llama model filter.
    Including those inflated OpenRouter "BW" can accuracy to ~0.321; local greedy
    scored them with the GSM verifier and pulled the pooled rate to ~0.028.
    """
    bw = pd.read_csv(REPO_ROOT / "results/raw/BW_P1_behavioral.csv", dtype=str).fillna("")
    llama = bw[bw["model"].astype(str).str.lower().str.contains("llama-3.1-8b")].copy()
    llama["variant_type"] = llama["variant_type"].map(_norm_vt)
    llama = llama.drop_duplicates(["problem_id", "variant_type"], keep="last")
    can = set(llama.loc[llama.variant_type == "canonical", "problem_id"].astype(str))
    w5 = set(llama.loc[llama.variant_type == "W5", "problem_id"].astype(str))
    paired = sorted(
        pid
        for pid in (can & w5)
        if str(pid).startswith(("BW_", "MBW_"))
    )
    items = []
    for pid in paired:
        for vt in ("canonical", "W5"):
            key = (pid, vt)
            if key not in lookup:
                raise KeyError(f"Missing bank text for BW-pool item {key}")
            items.append(dict(lookup[key]))
    return items


def _family_items(lookup: dict, family: str, variants: list[str]) -> list[dict]:
    if family == "gsm":
        bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_gsm.csv", dtype=str)
        pids = set(bank.loc[bank.variant_type == "canonical", "problem_id"].astype(str))
        prefix_ok = lambda pid: pid in pids
    elif family == "algo":
        bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_algo.csv", dtype=str)
        pids = set(bank.loc[bank.variant_type == "canonical", "problem_id"].astype(str))
        prefix_ok = lambda pid: pid in pids
    else:
        raise ValueError(family)

    items = []
    for pid in sorted(pids):
        for vt in variants:
            vt = _norm_vt(vt)
            key = (pid, vt)
            if key not in lookup:
                continue
            if not prefix_ok(pid):
                continue
            items.append(dict(lookup[key]))
    return items


def _wrap_prompt(tokenizer, text: str, family_key: str) -> str:
    user = f"{INSTR[family_key]}\n\nProblem:\n{text}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _family_key(item: dict) -> str:
    vf = item["verifier_family"]
    if vf in {"gsm", "arithmetic_reasoning"}:
        return "gsm"
    if vf == "algo" or item["problem_subtype"] in {
        "shortest_path",
        "coin_change",
        "wis",
        "wis_independent_set",
    }:
        return "algo"
    return "bw"


def _verify(item: dict, answer: str) -> bool:
    vf = item["verifier_family"]
    if vf == "algo" or item.get("problem_subtype") in {
        "shortest_path",
        "coin_change",
        "wis",
        "wis_independent_set",
    }:
        ok, _reason, _meta = verify_algo(
            item["problem_id"],
            answer,
            item["correct_answer"],
            item["problem_subtype"],
            item["variant_type"],
            item["difficulty_params"],
        )
        return bool(ok)
    return bool(
        verify_answer(
            item["problem_id"],
            answer,
            item["correct_answer"],
            vf,
            problem_text=item["problem_text"],
        )
    )


def _existing_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str)
    return {
        (str(r.problem_id).strip(), _norm_vt(r.variant_type))
        for _, r in df.iterrows()
    }


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def run_determinism_check(
    model, tokenizer, lookup: dict, out_path: Path, n_prompts: int, repeats: int, max_new: int, device
) -> None:
    """3× greedy on n sample prompts; flag non-identical outputs."""
    # Prefer diverse prompts from published sets
    samples = []
    for fam, variants in [("gsm", ["canonical"]), ("algo", ["canonical"]), ("bw", ["canonical"])]:
        if fam == "bw":
            pool = _bw_paired_items(lookup)
            pool = [x for x in pool if x["variant_type"] == "canonical"]
        else:
            pool = _family_items(lookup, fam, variants)
        samples.extend(pool[: max(1, n_prompts // 3 + 1)])
    samples = samples[:n_prompts]

    fields = [
        "sample_idx",
        "problem_id",
        "variant_type",
        "repeat",
        "response_hash",
        "raw_response",
        "identical_across_repeats",
    ]
    rows = []
    print(f"[determinism] {len(samples)} prompts × {repeats} repeats")
    for i, item in enumerate(samples):
        fk = _family_key(item)
        prompt = _wrap_prompt(tokenizer, item["problem_text"], fk)
        hashes = []
        texts = []
        for rep in range(repeats):
            ans = _generate(model, tokenizer, prompt, max_new, device)
            h = hashlib.sha256(ans.encode("utf-8")).hexdigest()[:16]
            hashes.append(h)
            texts.append(ans)
            rows.append(
                {
                    "sample_idx": i,
                    "problem_id": item["problem_id"],
                    "variant_type": item["variant_type"],
                    "repeat": rep,
                    "response_hash": h,
                    "raw_response": ans,
                    "identical_across_repeats": "",  # filled below
                }
            )
        identical = len(set(hashes)) == 1
        for r in rows[-(repeats):]:
            r["identical_across_repeats"] = identical
        print(
            f"  [{i+1}/{len(samples)}] {item['problem_id']} identical={identical} hashes={hashes}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for i in range(len(samples)) if rows[i * repeats]["identical_across_repeats"])
    print(f"[determinism] {n_ok}/{len(samples)} prompts bitwise-identical across {repeats} runs")
    print(f"[determinism] wrote {out_path}")


def run_family(
    family: str,
    items: list[dict],
    model,
    tokenizer,
    out_path: Path,
    resume: bool,
    max_new: int,
    device,
    model_name: str,
) -> None:
    done = _existing_done(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.write_text("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    fields = OUT_SCHEMA + OUT_EXTRA

    n_ok = 0
    n_done = 0
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        prev = pd.read_csv(out_path, dtype=str)
        n_ok = int(prev["behavioral_correct"].astype(str).str.lower().isin(["true", "1"]).sum())
        n_done = len(prev)

    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()

        for i, item in enumerate(items):
            key = (item["problem_id"], item["variant_type"])
            if key in done:
                continue
            fk = _family_key(item)
            prompt = _wrap_prompt(tokenizer, item["problem_text"], fk)
            ans = _generate(model, tokenizer, prompt, max_new, device)
            correct = _verify(item, ans)
            w.writerow(
                {
                    "problem_id": item["problem_id"],
                    "variant_type": item["variant_type"],
                    "model": model_name,
                    "behavioral_correct": correct,
                    "raw_response": ans,
                    "family": fk,
                    "correct_answer": item["correct_answer"],
                    "verify_family": item["verifier_family"],
                }
            )
            f.flush()
            n_done += 1
            n_ok += int(correct)
            print(
                f"[{family} {i+1}/{len(items)}] {item['problem_id']} {item['variant_type']} "
                f"correct={correct} acc={n_ok}/{n_done}={n_ok/max(n_done,1):.3f}"
            )

    print(f"\n=== {family} DONE → {out_path}  acc={n_ok}/{n_done} ===")
    # quick can vs rewrite summary
    df = pd.read_csv(out_path, dtype=str)
    df["ok"] = df["behavioral_correct"].astype(str).str.lower().isin(["true", "1"])
    for vt in sorted(df.variant_type.unique()):
        sub = df[df.variant_type == vt]
        print(f"  {vt}: {sub.ok.mean():.3f} (n={len(sub)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--families", nargs="+", default=["bw", "gsm", "algo"], choices=["bw", "gsm", "algo"])
    ap.add_argument("--out-dir", default="results/raw")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--determinism-only", action="store_true")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--determinism-prompts", type=int, default=10)
    ap.add_argument("--determinism-repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="Cap items per family (smoke test).")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Override variants per family (e.g. canonical only for spot-check).",
    )
    ap.add_argument(
        "--out-name",
        default=None,
        help="Override output CSV basename (e.g. llama_greedy_rerun_gsm_v2.csv).",
    )
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    _seed_everything(args.seed)

    lookup = _load_banks()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    print(f"[load] {args.model} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    det_path = out_dir / "llama_determinism_check.csv"
    if not args.skip_determinism:
        run_determinism_check(
            model,
            tokenizer,
            lookup,
            det_path,
            args.determinism_prompts,
            args.determinism_repeats,
            min(64, args.max_new_tokens),
            device,
        )
    if args.determinism_only:
        return

    for fam in args.families:
        if fam == "bw":
            items = _bw_paired_items(lookup)
            if args.variants is not None:
                want = {_norm_vt(v) for v in args.variants}
                items = [x for x in items if x["variant_type"] in want]
            out = out_dir / (args.out_name or "llama_greedy_rerun_bw.csv")
        elif fam == "gsm":
            variants = args.variants or ["canonical", "W3"]
            items = _family_items(lookup, "gsm", variants)
            out = out_dir / (args.out_name or "llama_greedy_rerun_gsm.csv")
        else:
            variants = args.variants or ["canonical", "W3"]
            items = _family_items(lookup, "algo", variants)
            out = out_dir / (args.out_name or "llama_greedy_rerun_algo.csv")
        if args.limit is not None:
            items = items[: args.limit]
        print(f"[queue] {fam}: {len(items)} items → {out}")
        run_family(
            fam,
            items,
            model,
            tokenizer,
            out,
            args.resume,
            args.max_new_tokens,
            device,
            args.model,
        )


if __name__ == "__main__":
    main()
