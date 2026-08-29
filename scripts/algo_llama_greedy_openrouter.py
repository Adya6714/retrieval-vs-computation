#!/usr/bin/env python3
"""Greedy (temperature=0) ALGO canonical accuracy for Llama-3.1-8B via OpenRouter.

Scores with probes.contamination.verify_algo (DP-optimal).
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
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify_algo import verify_algo

load_dotenv(REPO_ROOT / ".env")

FAMILY_INSTRUCTION = (
    "Solve the problem. Output ONLY the first action of the solution "
    "(no numbering, no explanation)."
)


def complete_greedy(model: str, prompt: str, max_tokens: int = 512) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        },
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
    data = r.json()
    return (data["choices"][0]["message"].get("content") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/problems/question_bank_algo.csv")
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument(
        "--output",
        default="results/raw/ALGO_llama31_8b_greedy_canonical.csv",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    df = pd.read_csv(args.bank, dtype=str).fillna("")
    df = df[df["problem_id"].str.match(r"^(CC|SP|WIS)_")].copy()
    df = df[df["variant_type"].str.strip() == "canonical"].copy()
    if args.limit is not None:
        df = df.head(args.limit)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and out.exists() and out.stat().st_size > 0:
        done = set(pd.read_csv(out, dtype=str)["problem_id"].astype(str))
    elif out.exists() and not args.resume:
        out.unlink()

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
        "decoding",
    ]
    write_header = not out.exists() or out.stat().st_size == 0
    n_ok = n_done = 0
    if args.resume and out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, dtype=str)
        n_ok = int(prev["verified"].astype(str).str.lower().isin(["true", "1"]).sum())
        n_done = len(prev)

    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for _, row in df.iterrows():
            pid = str(row["problem_id"]).strip()
            if pid in done:
                continue
            subtype = str(row["problem_subtype"]).strip()
            prompt = f"{FAMILY_INSTRUCTION}\n\nProblem:\n{row['problem_text']}"
            try:
                answer = complete_greedy(args.model, prompt, max_tokens=args.max_tokens)
            except Exception as e:
                answer = f"[API_ERROR] {e}"
                verified, reason, meta = False, f"api_error: {e}", {"parse_status": "parse_failed"}
            else:
                verified, reason, meta = verify_algo(
                    pid,
                    answer,
                    str(row["correct_answer"]),
                    subtype,
                    "canonical",
                    json.loads(str(row["difficulty_params"])),
                )
            alt = bool(meta.get("alternative_path") or meta.get("alternative_set"))
            w.writerow(
                {
                    "problem_id": pid,
                    "problem_subtype": subtype,
                    "variant_type": "canonical",
                    "model": args.model,
                    "model_answer": answer,
                    "ground_truth": row["correct_answer"],
                    "verified": verified,
                    "parse_status": meta.get("parse_status", ""),
                    "reason": reason,
                    "correct_alternative": alt,
                    "decoding": "greedy_temp0",
                }
            )
            f.flush()
            n_done += 1
            n_ok += int(bool(verified))
            print(
                f"[{n_done}/{len(df)}] {pid} verified={verified} "
                f"acc={n_ok}/{n_done}={n_ok/n_done:.3f}",
                flush=True,
            )
            time.sleep(0.15)

    print("\n=== DONE ===")
    print(f"accuracy: {n_ok}/{n_done} = {n_ok / max(n_done,1):.4f}")
    final = pd.read_csv(out, dtype=str)
    final["ok"] = final["verified"].astype(str).str.lower().isin(["true", "1"])
    print(final.groupby("problem_subtype")["ok"].agg(["sum", "count", "mean"]).to_string())


if __name__ == "__main__":
    main()
