"""Contamination scoring via longest matched n-gram search."""

from __future__ import annotations

from probes.contamination.infinigram_client import get_ngram_count

MIN_NGRAM = 5
DEFAULT_MAX_NGRAM = 13
ARITHMETIC_MAX_NGRAM = 8
STRIDE = 3  # Skip tokens to reduce API calls while still likely catching significant contamination


def _max_count_for_length(problem_text: str, n: int, stop_at_one: bool = False) -> int:
    """Return max infinigram count among all n-grams of length n in text.
    
    If stop_at_one is True, returns 1 as soon as any n-gram is found in the corpus.
    """
    tokens = problem_text.split()
    if len(tokens) < n:
        return 0

    max_count = 0
    # Use a stride to reduce the number of queries. 
    # If a long sequence exists in the corpus, checking every 3rd starting position
    # is extremely likely to still find a (slightly shorter) match or 
    # a match of the same length if the sequence is long enough.
    for i in range(0, len(tokens) - n + 1, STRIDE):
        ngram = " ".join(tokens[i : i + n])
        count = int(get_ngram_count(ngram))
        if count > 0 and stop_at_one:
            return 1
        if count > max_count:
            max_count = count
    return max_count


def score_problem(
    problem_text: str,
    family: str | None = None,
    max_ngram: int | None = None,
) -> dict[str, float | int]:
    """Score a problem by longest matched n-gram using binary search for length."""

    tokens = problem_text.split()
    if not tokens:
        return {
            "max_ngram_length": 0,
            "max_ngram_count": 0,
            "contamination_score": 0.0,
        }

    token_len = len(tokens)
    if token_len < MIN_NGRAM:
        return {
            "max_ngram_length": 0,
            "max_ngram_count": 0,
            "contamination_score": 0.0,
        }

    if max_ngram is None:
        fam = (family or "").strip().lower()
        if fam == "arithmetic_reasoning":
            max_ngram = ARITHMETIC_MAX_NGRAM
        else:
            max_ngram = DEFAULT_MAX_NGRAM
    max_ngram = max(int(max_ngram), MIN_NGRAM)
    max_len = min(token_len, max_ngram)

    # Binary search for the longest n-gram that exists (count > 0)
    low = MIN_NGRAM
    high = max_len
    best_len = 0

    while low <= high:
        mid = (low + high) // 2
        # We use stop_at_one=True to speed up the check
        if _max_count_for_length(problem_text, mid, stop_at_one=True) > 0:
            best_len = mid
            low = mid + 1
        else:
            high = mid - 1

    # Now that we have the best_len, find the actual max_count for that length
    best_count = 0
    if best_len > 0:
        # For the final results, we check every position to be precise (stride=1)
        best_count = 0
        tokens = problem_text.split()
        for i in range(len(tokens) - best_len + 1):
            ngram = " ".join(tokens[i : i + best_len])
            count = int(get_ngram_count(ngram))
            if count > best_count:
                best_count = count

    return {
        "max_ngram_length": best_len,
        "max_ngram_count": best_count,
        "contamination_score": round(best_len / max(token_len, 1), 4),
    }
