#!/usr/bin/env python3
"""G5: ALGO gold-token degeneracy under the Appendix H >50% rule.

Same rule as colab/_build_notebooks.py: family is degenerate if the modal
gold token is >50% of all items OR of canonical items. Gold is answer
content (Cost:/Count:/Total:), not Path:/Selected: scaffolding.

Does not write results/raw/. Does not call any model API.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALGO_BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
DERIVED = REPO_ROOT / "results/derived"
OUT = DERIVED / "ALGO_gold_token_degeneracy.csv"

DEGEN_FRAC = 0.5

# Frozen adversarial pool (colab/_build_notebooks.py ALGO_ADV). Not bank instance_type.
ALGO_ADV = {
    "CC": [f"CC_{i:02d}" for i in range(1, 11)],
    "SP": [
        "SP_003", "SP_004", "SP_005", "SP_019", "SP_020", "SP_021", "SP_023",
        "SP_024", "SP_026", "SP_027", "SP_028", "SP_029", "SP_030", "SP_037",
        "SP_038", "SP_039", "SP_040", "SP_042", "SP_044", "SP_045", "SP_046",
        "SP_047", "SP_048", "SP_062", "SP_063", "SP_064", "SP_065", "SP_066",
        "SP_068", "SP_069", "SP_070", "SP_071", "SP_072", "SP_073",
    ],
    "WIS": [
        "WIS_003", "WIS_004", "WIS_013", "WIS_014", "WIS_015", "WIS_016",
        "WIS_017", "WIS_018", "WIS_019", "WIS_020", "WIS_023", "WIS_024",
        "WIS_025", "WIS_026", "WIS_027", "WIS_028", "WIS_029",
    ],
}
ALGO_ADV_IDS = ALGO_ADV["CC"] + ALGO_ADV["SP"] + ALGO_ADV["WIS"]
assert len(ALGO_ADV_IDS) == 61, len(ALGO_ADV_IDS)


def algo_gold_content(problem_id: str, correct_answer: str) -> str:
    """Copied from colab/_build_notebooks.py — answer-content numeric token."""
    s = str(correct_answer)
    pid = str(problem_id).strip().upper()
    if pid.startswith("SP"):
        m = re.search(r"Cost\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Cost: gold in {s!r}")
        return m.group(1)
    if pid.startswith("CC"):
        m = re.search(r"(?:Count|Total)\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Count:/Total: gold in {s!r}")
        return m.group(1)
    if pid.startswith("WIS"):
        m = re.search(r"Total\s*:\s*(-?\d+)", s, flags=re.I)
        if not m:
            raise ValueError(f"{problem_id}: no Total: gold in {s!r}")
        return m.group(1)
    raise ValueError(f"{problem_id}: unknown ALGO subtype")


def degeneracy(toks: list[str]) -> tuple[bool, str, int, int]:
    n = len(toks)
    if n == 0:
        return True, "", 0, 0
    vc = Counter(toks)
    modal, n_m = vc.most_common(1)[0]
    return (n_m / n) > DEGEN_FRAC, modal, n_m, n


def _universe(df: pd.DataFrame, pids: list[str] | None, variants: list[str] | None) -> pd.DataFrame:
    out = df
    if pids is not None:
        out = out[out["problem_id"].isin(pids)]
    if variants is not None:
        want = {v.lower() for v in variants}
        out = out[out["variant_type"].astype(str).str.lower().isin(want)]
    return out


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ALGO_BANK, dtype=str).fillna("")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant_type"] = df["variant_type"].astype(str).str.strip()
    df.loc[df["variant_type"].str.lower() == "canonical", "variant_type"] = "canonical"
    df["gold_content"] = [
        algo_gold_content(pid, ans)
        for pid, ans in zip(df["problem_id"], df["correct_answer"])
    ]

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    except Exception:
        tokenizer = None

    def gold_token(gold: str) -> str:
        if tokenizer is None:
            return gold
        ids = tokenizer.encode(gold, add_special_tokens=False)
        return tokenizer.decode([ids[0]]) if ids else gold

    token_source = (
        "llama3.1_first_bpe of algo_gold_content"
        if tokenizer is not None
        else "content_string_heuristic (Llama tokenizer not available; single-digit golds are one token in Llama-3 BPE)"
    )

    df["gold_token"] = [gold_token(g) for g in df["gold_content"]]

    universes = [
        ("all_bank_all_variants", None, None),
        ("all_bank_canonical", None, ["canonical"]),
        ("frozen61_canonical_w3", ALGO_ADV_IDS, ["canonical", "W3"]),
        ("frozen61_canonical", ALGO_ADV_IDS, ["canonical"]),
    ]

    rows = []
    for name, pids, variants in universes:
        sub = _universe(df, pids, variants)
        toks = [str(x) for x in sub["gold_token"].tolist()]
        flagged, modal, n_m, n = degeneracy(toks)
        can = sub[sub["variant_type"] == "canonical"]
        can_flag, can_modal, can_n, can_N = degeneracy(
            [str(x) for x in can["gold_token"].tolist()]
        )
        family_flag = flagged or can_flag
        top5 = Counter(toks).most_common(5)
        rows.append(
            {
                "universe": name,
                "n_items": n,
                "n_canonical": can_N,
                "n_distinct_gold": len(set(toks)),
                "modal_gold": modal,
                "modal_n": n_m,
                "modal_share": round(n_m / n, 4) if n else "",
                "canonical_modal_gold": can_modal,
                "canonical_modal_n": can_n,
                "canonical_modal_share": round(can_n / can_N, 4) if can_N else "",
                "degen_frac_rule": DEGEN_FRAC,
                "degenerate": str(bool(family_flag)),
                "passes_degeneracy_rule": str(not family_flag),
                "top5": json.dumps(top5),
                "token_source": token_source,
            }
        )

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT}")
    print(f"token_source: {token_source}")
    for r in rows:
        print(
            f"  {r['universe']}: modal={r['modal_gold']!r} {r['modal_n']}/{r['n_items']}"
            f"={r['modal_share']} distinct={r['n_distinct_gold']} "
            f"canonical {r['canonical_modal_gold']!r} {r['canonical_modal_share']} "
            f"PASS={r['passes_degeneracy_rule']}"
        )


if __name__ == "__main__":
    main()
