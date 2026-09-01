#!/usr/bin/env python3
"""Audit that W1–W6 actually transform canonical problem_text / answers.

One row per (bank, problem_id, variant). Does not write results/raw/.
Does not call any model API.

transform_status (first match wins):
  identical_to_canonical  problem_text equals canonical after whitespace collapse
  answer_mismatch         W1–W4 gold differs from canonical, or W5/W6 gold is identical
  near_duplicate          W6 only: token Jaccard >= NEAR_DUP_JACCARD vs canonical
                          or vs any other W6 in the same bank
  transformed             otherwise
"""

from __future__ import annotations

import csv
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.variants import normalize_variant  # noqa: E402

DATA = REPO_ROOT / "data/problems"
DERIVED = REPO_ROOT / "results/derived"
OUT = DERIVED / "variant_transform_audit.csv"
COUNTS = DERIVED / "variant_transform_audit_counts.csv"

BANKS = {
    "ALGO": DATA / "question_bank_algo.csv",
    "BW": DATA / "question_bank_bw.csv",
    "GSM": DATA / "question_bank_gsm.csv",
}

# Pre-specified. Do not tune to change which cells survive.
NEAR_DUP_JACCARD = 0.85
ANSWER_SAME_VARIANTS = {"W1", "W2", "W3", "W4"}
ANSWER_DIFF_VARIANTS = {"W5", "W6"}

csv.field_size_limit(sys.maxsize)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _norm_answer(text: str) -> str:
    return _norm_ws(text).lower()


def _tokens(text: str) -> list[str]:
    return _norm_ws(text).split()


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_ws(a), _norm_ws(b)).ratio()


def load_bank(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []

    for bank_name, path in BANKS.items():
        rows = load_bank(path)
        by_pid: dict[str, dict[str, dict[str, str]]] = {}
        for row in rows:
            pid = str(row.get("problem_id", "")).strip()
            vt = normalize_variant(row.get("variant_type"))
            if not pid or not vt:
                continue
            by_pid.setdefault(pid, {})[vt] = row

        w6_texts: dict[str, str] = {}
        for pid, variants in by_pid.items():
            if "W6" in variants:
                w6_texts[pid] = str(variants["W6"].get("problem_text", ""))

        for pid, variants in by_pid.items():
            can = variants.get("canonical")
            if can is None:
                continue
            can_text = str(can.get("problem_text", ""))
            can_ans = str(can.get("correct_answer", ""))
            for vt, row in variants.items():
                if vt == "canonical":
                    continue
                v_text = str(row.get("problem_text", ""))
                v_ans = str(row.get("correct_answer", ""))
                text_identical = _norm_ws(v_text) == _norm_ws(can_text)
                ans_identical = _norm_answer(v_ans) == _norm_answer(can_ans)
                jac = token_jaccard(v_text, can_text)
                sim = char_similarity(v_text, can_text)
                tok_can, tok_v = _tokens(can_text), _tokens(v_text)

                near_vs_can = False
                near_vs_w6_id = ""
                if vt == "W6":
                    near_vs_can = (not text_identical) and jac >= NEAR_DUP_JACCARD
                    for other_pid, other_text in w6_texts.items():
                        if other_pid == pid:
                            continue
                        if token_jaccard(v_text, other_text) >= NEAR_DUP_JACCARD:
                            near_vs_w6_id = other_pid
                            break

                if vt in ANSWER_SAME_VARIANTS:
                    ans_ok = ans_identical
                    ans_expected = "identical"
                elif vt in ANSWER_DIFF_VARIANTS:
                    ans_ok = not ans_identical
                    ans_expected = "different"
                else:
                    ans_ok = True
                    ans_expected = "unspecified"

                if text_identical:
                    status = "identical_to_canonical"
                elif not ans_ok:
                    status = "answer_mismatch"
                elif vt == "W6" and (near_vs_can or near_vs_w6_id):
                    status = "near_duplicate"
                else:
                    status = "transformed"

                rows_out.append(
                    {
                        "bank": bank_name,
                        "problem_id": pid,
                        "variant": vt,
                        "problem_family": str(row.get("problem_family", "")),
                        "problem_subtype": str(row.get("problem_subtype", "")),
                        "char_len_canonical": len(_norm_ws(can_text)),
                        "char_len_variant": len(_norm_ws(v_text)),
                        "char_similarity": round(sim, 4),
                        "n_tokens_canonical": len(tok_can),
                        "n_tokens_variant": len(tok_v),
                        "token_jaccard": round(jac, 4),
                        "text_identical_to_canonical": str(text_identical),
                        "answer_identical_to_canonical": str(ans_identical),
                        "answer_expected": ans_expected,
                        "near_duplicate_vs_canonical": str(near_vs_can),
                        "near_duplicate_w6_peer": near_vs_w6_id,
                        "near_dup_jaccard_floor": NEAR_DUP_JACCARD,
                        "transform_status": status,
                    }
                )

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    counts: dict[tuple[str, str, str], int] = {}
    for r in rows_out:
        key = (r["bank"], r["variant"], r["transform_status"])
        counts[key] = counts.get(key, 0) + 1
    count_rows = [
        {"bank": b, "variant": v, "transform_status": s, "n": n}
        for (b, v, s), n in sorted(counts.items())
    ]
    with COUNTS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["bank", "variant", "transform_status", "n"]
        )
        writer.writeheader()
        writer.writerows(count_rows)

    print(f"Wrote {OUT} ({len(rows_out)} pairs)")
    print(f"Wrote {COUNTS}")
    from collections import Counter

    print("by variant × status:")
    vc = Counter((r["variant"], r["transform_status"]) for r in rows_out)
    for (vt, st), n in sorted(vc.items()):
        print(f"  {vt:12} {st:24} {n}")


if __name__ == "__main__":
    main()
