"""Minimal Infini-gram count client with retry and local caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

API_URL = "https://api.infini-gram.io/"
INDEX_NAME = "v4_pileval_gpt2"
QUERY_TYPE = "count"
CACHE_PATH = Path("data/infinigram_cache.json")
REQUEST_TIMEOUT_SECONDS = 30

_CACHE: dict[str, int] | None = None


def _load_cache() -> dict[str, int]:
    if not CACHE_PATH.exists():
        return {}

    with CACHE_PATH.open("r", encoding="utf-8") as f:
        raw: Any = json.load(f)
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items()}


def _save_cache(cache: dict[str, int]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=True, indent=2, sort_keys=True)


def _get_cache() -> dict[str, int]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_cache()
    return _CACHE


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_count(query: str) -> int:
    response = requests.post(
        API_URL,
        json={
            "index": INDEX_NAME,
            "query_type": QUERY_TYPE,
            "query": query,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    print(f"DEBUG infinigram query: {query!r}")
    print(f"DEBUG response status: {response.status_code}")
    print(f"DEBUG response body: {response.text[:300]}")
    response.raise_for_status()
    payload = response.json()
    if "count" not in payload:
        raise ValueError("Infini-gram response missing 'count'.")
    return int(payload["count"])


def get_ngram_count(query: str) -> int:
    """Return Infini-gram count for a query string, using disk cache."""
    if not query:
        return 0

    cache = _get_cache()
    if query in cache:
        return cache[query]

    count = _fetch_count(query)
    cache[query] = count
    _save_cache(cache)
    return count
