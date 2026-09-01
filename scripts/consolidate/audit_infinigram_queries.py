#!/usr/bin/env python3
"""G3: Audit Infini-gram query construction. Does not re-query the API.

Reports index, n-gram window, whether max_ngram_count is raw, and whether
instance-level / GSM queries share the old BW keyword-dump defect.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination import infinigram_client as ig  # noqa: E402
from probes.contamination import score as score_mod  # noqa: E402

DERIVED = REPO_ROOT / "results/derived"
OUT = DERIVED / "P3_infinigram_query_audit.csv"

GSM_P3 = REPO_ROOT / "scripts/GSM_P3_SCR_run_contamination_triage.py"
ALGO_P3 = REPO_ROOT / "scripts/ALGO_P3_SCR_run_contamination_triage.py"
BW_P3 = REPO_ROOT / "scripts/BW_P3_SCR_run_contamination_triage.py"


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    gsm_src = _src(GSM_P3)
    algo_src = _src(ALGO_P3)
    bw_src = _src(BW_P3)

    gsm_query = "full problem_text only (no template/instance split)"
    algo_template = (
        "score_problem(full problem_text, family='gsm') — same full text as "
        "contamination_score, but forced through the GSM n=8 cap"
    )
    algo_instance = "correct_answer (gold string), not instance parameters / graph text"
    bw_template_bw = (
        "You are a robot arm. Available actions: pick-up X, put-down X, "
        "stack X Y, unstack X Y. You can hold one block at a time."
    )
    bw_template_mbw = "Available actions: attack X, succumb X, overcome X Y, feast X Y."
    bw_instance = (
        "goal snippet after 'Goal:' / 'Objective:' if present; else fallback "
        "'blocksworld num_blocks {n}'"
    )

    rows = [
        {
            "family": "shared_scorer",
            "script": "probes/contamination/score.py + infinigram_client.py",
            "index_queried": ig.INDEX_NAME,
            "api_url": ig.API_URL,
            "min_n": score_mod.MIN_NGRAM,
            "max_n_default": score_mod.DEFAULT_MAX_NGRAM,
            "max_n_arithmetic": score_mod.ARITHMETIC_MAX_NGRAM,
            "window_comparable_across_families": "no — GSM/arithmetic capped at 8; ALGO/BW default 13",
            "max_ngram_count_meaning": "raw Infini-gram count at the longest n with count>0 (not normalized)",
            "contamination_score_meaning": "best_len / whitespace_token_len (length fraction, normalized)",
            "query_construction": "whitespace tokens; binary search longest n in [5, max_n] with count>0; final count uses stride 1",
            "keyword_dump_defect": "n/a",
            "notes": f"cache={ig.CACHE_PATH}; env INFINIGRAM_INDEX overrides default v4_rpj_llama_s4",
        },
        {
            "family": "GSM",
            "script": "scripts/GSM_P3_SCR_run_contamination_triage.py",
            "index_queried": ig.INDEX_NAME,
            "api_url": ig.API_URL,
            "min_n": score_mod.MIN_NGRAM,
            "max_n_default": score_mod.ARITHMETIC_MAX_NGRAM,
            "max_n_arithmetic": score_mod.ARITHMETIC_MAX_NGRAM,
            "window_comparable_across_families": "no",
            "max_ngram_count_meaning": "raw count",
            "contamination_score_meaning": "best_len / token_len",
            "query_construction": gsm_query,
            "keyword_dump_defect": "no — full word-problem text, not a keyword bag. Also not decomposed into template vs instance.",
            "notes": "family=arithmetic_reasoning triggers ARITHMETIC_MAX_NGRAM=8. No template_contamination_score column.",
        },
        {
            "family": "ALGO",
            "script": "scripts/ALGO_P3_SCR_run_contamination_triage.py",
            "index_queried": ig.INDEX_NAME,
            "api_url": ig.API_URL,
            "min_n": score_mod.MIN_NGRAM,
            "max_n_default": score_mod.DEFAULT_MAX_NGRAM,
            "max_n_arithmetic": score_mod.ARITHMETIC_MAX_NGRAM,
            "window_comparable_across_families": "no — contamination_score uses family=algorithmic (n=13); template/instance _score_query forces family='gsm' (n=8)",
            "max_ngram_count_meaning": "raw count",
            "contamination_score_meaning": "best_len / token_len",
            "query_construction": (
                f"contamination_score=full problem_text (n=13); "
                f"template={algo_template}; instance={algo_instance}"
            ),
            "keyword_dump_defect": "yes for instance — gold answer is not instance text. template is a second pass over the same full problem_text, not a template stem.",
            "notes": "ALGO decompose columns are miswired. Do not treat template vs instance as a real split.",
        },
        {
            "family": "BW",
            "script": "scripts/BW_P3_SCR_run_contamination_triage.py",
            "index_queried": ig.INDEX_NAME,
            "api_url": ig.API_URL,
            "min_n": score_mod.MIN_NGRAM,
            "max_n_default": score_mod.DEFAULT_MAX_NGRAM,
            "max_n_arithmetic": score_mod.ARITHMETIC_MAX_NGRAM,
            "window_comparable_across_families": "no vs GSM (13 vs 8); yes vs ALGO contamination_score",
            "max_ngram_count_meaning": "raw count",
            "contamination_score_meaning": "best_len / token_len on full problem_text",
            "query_construction": (
                f"template blocksworld={bw_template_bw!r}; "
                f"template mystery={bw_template_mbw!r}; "
                f"instance={bw_instance}"
            ),
            "keyword_dump_defect": (
                "template: no (grammatical stems after C1). "
                "instance fallback: yes — 'blocksworld num_blocks N' is a keyword dump. "
                "instance when Goal: is present: no (goal sentence)."
            ),
            "notes": "Old BW template keyword dump was replaced. Fallback instance query still has the defect. GSM does not share the dump; ALGO instance=gold is a different defect.",
        },
    ]

    # Confirm source still has the ALGO family='gsm' wiring and GSM full-text-only.
    assert "family=\"gsm\"" in algo_src or "family='gsm'" in algo_src
    assert "score_problem(" in gsm_src
    assert "blocksworld num_blocks" in bw_src
    assert "You are a robot arm" in bw_src

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}")
    for r in rows:
        print(f"\n[{r['family']}] index={r['index_queried']} max_n_default={r['max_n_default']}")
        print(f"  query: {r['query_construction'][:200]}")
        print(f"  keyword_dump: {r['keyword_dump_defect']}")
        print(f"  count: {r['max_ngram_count_meaning']}")


if __name__ == "__main__":
    main()
