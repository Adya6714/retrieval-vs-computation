#!/usr/bin/env python3
"""Diagnose llama_greedy_p1.csv GSM truncation. Does not rewrite results/raw/.

The 15.9% GSM canonical cell is not a model-quality result: 23/37 wrong
canonical answers are cut mid-token at the ~128-token cap. This script records
that diagnosis, the ALGO 8=8 coincidence, BW W3-vs-canonical (already below
the 0.30 retention floor), and WIS_017–020 golds.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW = REPO_ROOT / "results/raw" / "llama_greedy_p1.csv"
DERIVED = REPO_ROOT / "results/derived"
ALGO_BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
BW_BANK = REPO_ROOT / "data/problems/question_bank_bw.csv"

TRUNC_OUT = DERIVED / "llama_greedy_p1_gsm_truncation_audit.csv"
COIN_OUT = DERIVED / "llama_greedy_p1_coincidence_audit.csv"
SUMMARY_OUT = DERIVED / "llama_greedy_p1_g4_diagnosis.csv"


def _ok(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def ends_mid_token(text: str) -> bool:
    s = str(text or "").rstrip()
    if not s:
        return False
    last = s[-1]
    if last.isalnum():
        return True
    if last in {"=", "*", "+", "/", "-", "(", ",", ":", "$"}:
        return True
    return False


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW, dtype=str).fillna("")
    gsm = df[df["family"].astype(str).str.upper() == "GSM"].copy()
    can = gsm[gsm["variant"].astype(str).str.lower() == "canonical"].copy()
    can["correct_bool"] = can["correct"].map(_ok)
    can["n_chars"] = can["model_answer"].str.len()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    def n_tokens(text: str) -> int:
        return len(tokenizer.encode(str(text), add_special_tokens=False))

    def last_tok(text: str) -> str:
        ids = tokenizer.encode(str(text), add_special_tokens=False)
        return tokenizer.decode([ids[-1]]) if ids else ""

    can["n_tokens"] = can["model_answer"].map(n_tokens)
    can["last_token"] = can["model_answer"].map(last_tok)
    can["hit_token_cap"] = can["n_tokens"] >= 127
    can["ends_mid_token"] = can["model_answer"].map(ends_mid_token)
    can["has_hash_tag"] = can["model_answer"].str.contains(r"####", regex=True)
    wrong = can[~can["correct_bool"]]
    right = can[can["correct_bool"]]

    rows = []
    for _, r in can.iterrows():
        rows.append(
            {
                "problem_id": r["problem_id"],
                "correct": str(bool(r["correct_bool"])),
                "n_chars": int(r["n_chars"]),
                "n_tokens": int(r["n_tokens"]),
                "last_token": r["last_token"],
                "hit_token_cap": str(bool(r["hit_token_cap"])),
                "ends_mid_token": str(bool(r["ends_mid_token"])),
                "has_hash_tag": str(bool(r["has_hash_tag"])),
                "answer_tail": str(r["model_answer"])[-80:].replace("\n", "\\n"),
            }
        )
    with TRUNC_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_wrong = int(len(wrong))
    n_mid = int(wrong["ends_mid_token"].sum())
    n_cap = int(wrong["hit_token_cap"].sum())
    n_cap128 = int((wrong["n_tokens"] == 128).sum())
    orig_acc = float(can["correct_bool"].mean()) if len(can) else float("nan")

    algo = df[df["family"].astype(str).str.upper() == "ALGO"].copy()
    bw = df[df["family"].astype(str).str.upper() == "BW"].copy()

    def overlap(fam_df: pd.DataFrame) -> dict:
        piv = fam_df.pivot_table(
            index="problem_id",
            columns="variant",
            values="correct",
            aggfunc="first",
        )
        for col in piv.columns:
            piv[col] = piv[col].map(_ok)
        can_c = piv.get("canonical", pd.Series(dtype=bool)).fillna(False)
        w3_c = piv.get("W3", pd.Series(dtype=bool)).fillna(False)
        both = can_c & w3_c
        can_only = can_c & ~w3_c
        w3_only = ~can_c & w3_c
        neither = ~can_c & ~w3_c
        return {
            "n": int(len(piv)),
            "both": int(both.sum()),
            "canonical_only": int(can_only.sum()),
            "w3_only": int(w3_only.sum()),
            "neither": int(neither.sum()),
            "canonical_acc": float(can_c.mean()) if len(can_c) else float("nan"),
            "w3_acc": float(w3_c.mean()) if len(w3_c) else float("nan"),
            "canonical_correct_ids": ",".join(sorted(can_c[can_c].index.astype(str))),
            "w3_correct_ids": ",".join(sorted(w3_c[w3_c].index.astype(str))),
            "w3_only_ids": ",".join(sorted(w3_only[w3_only].index.astype(str))),
        }

    algo_o = overlap(algo)
    bw_o = overlap(bw)

    algo_bank = pd.read_csv(ALGO_BANK, dtype=str).fillna("")
    wis_rows = []
    for pid in ["WIS_017", "WIS_018", "WIS_019", "WIS_020"]:
        sub = algo_bank[
            (algo_bank["problem_id"] == pid)
            & (algo_bank["variant_type"].astype(str).str.lower() == "canonical")
        ]
        gold = str(sub.iloc[0]["correct_answer"]) if len(sub) else ""
        params = str(sub.iloc[0]["difficulty_params"]) if len(sub) else ""
        wis_rows.append(
            {
                "problem_id": pid,
                "canonical_gold": gold,
                "same_template": (
                    "six-interval chain_overlap clones; gold always Selected: {4, 5}; "
                    "only interval 4/5 weights change"
                ),
                "difficulty_params_head": params[:180],
            }
        )

    bw_bank = pd.read_csv(BW_BANK, dtype=str).fillna("")
    w3_only_ids = [x for x in bw_o["w3_only_ids"].split(",") if x]
    rename_domains = []
    for pid in w3_only_ids:
        sub = bw_bank[
            (bw_bank["problem_id"] == pid)
            & (bw_bank["variant_type"].astype(str).str.upper() == "W3")
        ]
        if sub.empty:
            continue
        text = str(sub.iloc[0]["problem_text"])
        subtype = str(sub.iloc[0].get("problem_subtype", ""))
        notes = str(sub.iloc[0].get("notes", ""))
        # Mystery BW uses attack/succumb/overcome/feast; classic uses pick-up/stack.
        if re.search(r"\b(attack|succumb|overcome|feast)\b", text, re.I):
            domain = "mystery_blocksworld"
        elif re.search(r"\b(pick-up|put-down|unstack|stack)\b", text, re.I):
            domain = "blocksworld"
        else:
            domain = subtype or "unknown"
        rename_domains.append(
            {
                "problem_id": pid,
                "subtype": subtype,
                "rename_domain": domain,
                "notes_head": notes[:120],
            }
        )
    domain_counts = Counter(r["rename_domain"] for r in rename_domains)

    coin_rows = [
        {
            "family": "ALGO",
            **{k: algo_o[k] for k in algo_o},
            "retention_floor_0_30": "n/a (not a W3 retention cell in this file's GSM sense)",
            "note": "identical 0.0727 is 8 and 8 on different IDs, not identical prompts",
        },
        {
            "family": "BW",
            **{k: bw_o[k] for k in bw_o},
            "retention_floor_0_30": "suppressed (canonical 4/65=0.0615 < 0.30)",
            "note": (
                f"W3-only n={bw_o['w3_only']} vs canonical-only n={bw_o['canonical_only']}; "
                f"W3-only rename domains={dict(domain_counts)}"
            ),
        },
    ]
    with COIN_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coin_rows[0].keys()))
        w.writeheader()
        w.writerows(coin_rows)

    wis_path = DERIVED / "llama_greedy_p1_wis_017_020_gold.csv"
    with wis_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "problem_id",
                "canonical_gold",
                "same_template",
                "difficulty_params_head",
            ],
        )
        w.writeheader()
        w.writerows(wis_rows)

    w3_path = DERIVED / "llama_greedy_p1_bw_w3_only.csv"
    if rename_domains:
        with w3_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rename_domains[0].keys()))
            w.writeheader()
            w.writerows(rename_domains)

    summary = [
        {
            "item": "gsm_canonical_n",
            "value": str(len(can)),
        },
        {
            "item": "gsm_canonical_correct",
            "value": str(int(can["correct_bool"].sum())),
        },
        {
            "item": "gsm_canonical_accuracy_original",
            "value": f"{orig_acc:.4f}",
        },
        {
            "item": "gsm_canonical_wrong",
            "value": str(n_wrong),
        },
        {
            "item": "gsm_wrong_ends_mid_token_char_heuristic",
            "value": str(n_mid),
        },
        {
            "item": "gsm_wrong_hit_token_cap_ge_127",
            "value": str(n_cap),
        },
        {
            "item": "gsm_wrong_exactly_128_tokens",
            "value": str(n_cap128),
        },
        {
            "item": "gsm_correct_median_chars",
            "value": str(float(right["n_chars"].median()) if len(right) else ""),
        },
        {
            "item": "gsm_wrong_median_chars",
            "value": str(float(wrong["n_chars"].median()) if len(wrong) else ""),
        },
        {
            "item": "gsm_wrong_max_chars",
            "value": str(int(wrong["n_chars"].max()) if len(wrong) else ""),
        },
        {
            "item": "gsm_canonical_accuracy_768_rerun",
            "value": "not_executed",
        },
        {
            "item": "rerun_status",
            "value": "not_executed: no CUDA, Llama-3.1-8B-Instruct weights not cached, machine RAM 8.6GB, ollama has no llama3.1 model",
        },
        {
            "item": "max_new_tokens_original",
            "value": "128",
        },
        {
            "item": "max_new_tokens_required",
            "value": "768",
        },
        {
            "item": "expected_recovery_if_truncation",
            "value": "~0.78 canonical (not measured here)",
        },
    ]
    with SUMMARY_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "value"])
        w.writeheader()
        w.writerows(summary)

    print(f"Wrote {TRUNC_OUT}")
    print(f"Wrote {COIN_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"GSM canonical {int(can['correct_bool'].sum())}/{len(can)}={orig_acc:.4f}")
    print(f"wrong mid-token heuristic {n_mid}/{n_wrong}; hit cap>=127 {n_cap}; exactly 128 {n_cap128}")
    print(
        "correct median chars",
        float(right["n_chars"].median()) if len(right) else None,
        "wrong median",
        float(wrong["n_chars"].median()) if len(wrong) else None,
        "wrong max",
        int(wrong["n_chars"].max()) if len(wrong) else None,
    )
    print("ALGO overlap", {k: algo_o[k] for k in ("both", "canonical_only", "w3_only", "neither", "canonical_acc", "w3_acc")})
    print("BW overlap", {k: bw_o[k] for k in ("both", "canonical_only", "w3_only", "neither", "canonical_acc", "w3_acc")})
    print("WIS golds", wis_rows)
    print("BW W3-only domains", dict(domain_counts))


if __name__ == "__main__":
    main()
