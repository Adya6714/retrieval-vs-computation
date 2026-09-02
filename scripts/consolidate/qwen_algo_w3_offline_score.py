#!/usr/bin/env python3
"""Offline W3 correctness for Qwen2.5-1.5B on frozen 61 ALGO adversarial instances."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify_algo import verify_algo  # noqa: E402

BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
P2 = REPO_ROOT / "results/raw/ALGO_P2_phase1_claude_new.csv"
OUT = REPO_ROOT / "results/derived/N3_qwen_algo_w3_scores.csv"
OLLAMA_MODEL = "qwen2.5:1.5b-instruct"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def _ollama_generate(prompt: str, *, max_tokens: int = 128) -> str:
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    return str(data.get("response", "")).strip()


def main() -> None:
    adv = set(
        pd.read_csv(P2, dtype=str)
        .loc[lambda d: d["instance_type"].str.lower() == "adversarial", "problem_id"]
        .astype(str),
    )
    bank = pd.read_csv(BANK, dtype=str).fillna("")
    bank = bank[
        (bank["variant_type"].str.lower() == "w3")
        & (bank["problem_id"].astype(str).isin(adv))
    ].copy()

    if OUT.exists() and len(pd.read_csv(OUT)) >= len(bank):
        print(f"Cache exists: {OUT}")
        return

    rows: list[dict] = []
    for i, (_, row) in enumerate(bank.iterrows(), start=1):
        pid = str(row["problem_id"])
        prompt = str(row["problem_text"])
        try:
            answer = _ollama_generate(prompt)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama not reachable at {OLLAMA_URL}. Start with: ollama serve"
            ) from exc
        ok, reason, _meta = verify_algo(
            pid,
            answer,
            str(row["correct_answer"]),
            str(row.get("problem_subtype", "")),
            str(row.get("variant_type", "W3")),
            str(row.get("difficulty_params", "")),
            notes=str(row.get("notes", "")) or None,
            problem_text=prompt,
        )
        rows.append(
            {
                "problem_id": pid,
                "model": f"Qwen/{OLLAMA_MODEL}",
                "w3_ok": bool(ok),
                "verify_reason": reason,
                "model_answer": answer[:500],
            }
        )
        print(f"[{i}/{len(bank)}] {pid}: {ok} ({reason})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
