"""Contamination scoring via longest matched n-gram search."""

from __future__ import annotations

from probes.contamination.infinigram_client import get_ngram_count

MIN_NGRAM = 5
START_NGRAM = 20


def _max_count_for_length(problem_text: str, n: int) -> int:
    """Return max infinigram count among all n-grams of length n in text."""
    tokens = problem_text.split()
    if len(tokens) < n:
        return 0

    max_count = 0
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        count = int(get_ngram_count(ngram))
        if count > max_count:
            max_count = count
    return max_count


def score_problem(problem_text: str) -> dict[str, float | int]:
    """Score a problem by longest matched n-gram."""

    tokens = problem_text.split()
    if not tokens:
        return {
            "max_ngram_length": 0,
            "max_ngram_count": 0,
            "contamination_score": 0.0,
        }

    max_len = len(tokens)
    if max_len < MIN_NGRAM:
        return {
            "max_ngram_length": 0,
            "max_ngram_count": 0,
            "contamination_score": 0.0,
        }

    start_len = min(START_NGRAM, max_len)
    best_len = 0
    best_count = 0

    start_count = _max_count_for_length(problem_text, start_len)
    if start_count > 0:
        best_len = start_len
        best_count = start_count
        current_len = start_len + 1
        while current_len <= max_len:
            current_count = _max_count_for_length(problem_text, current_len)
            if current_count == 0:
                break
            best_len = current_len
            best_count = current_count
            current_len += 1
    else:
        current_len = start_len - 1
        while current_len >= MIN_NGRAM:
            current_count = _max_count_for_length(problem_text, current_len)
            if current_count > 0:
                best_len = current_len
                best_count = current_count
                break
            current_len -= 1

    return {
        "max_ngram_length": best_len,
        "max_ngram_count": best_count,
        "contamination_score": round(best_len / max(len(tokens), 1), 4),
    }
