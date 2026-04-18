"""Minimal Infini-gram count client with retry and local caching.

If https://api.infini-gram.io/ returns 403 (some networks block it), use the
mini endpoint and a mini index, for example::

    export INFINIGRAM_API_URL=https://api.infini-gram-mini.io/
    export INFINIGRAM_INDEX=v2_dclm_all

If you see ``SSL: CERTIFICATE_VERIFY_FAILED`` / hostname mismatch, a proxy is
often intercepting TLS. Prefer fixing trust store / proxy; as a last resort::

    export INFINIGRAM_SSL_VERIFY=0

If you hit HTTP 403 after many queries (rate limit), wait a few minutes and
re-run triage (resume skips finished ``problem_id``s). Optionally throttle::

    export INFINIGRAM_THROTTLE_SEC=1.0

Scores are comparable within a fixed (URL, index) pair, not across corpora.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
import urllib3
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

API_URL = os.environ.get("INFINIGRAM_API_URL", "https://api.infini-gram.io/").rstrip("/") + "/"
INDEX_NAME = os.environ.get("INFINIGRAM_INDEX", "v4_rpj_llama_s4")
QUERY_TYPE = "count"
CACHE_PATH = Path("data/infinigram_cache.json")
REQUEST_TIMEOUT_SECONDS = 10
_LEGACY_INDEX = "v4_rpj_llama_s4"

_SSL_VERIFY_RAW = os.environ.get("INFINIGRAM_SSL_VERIFY", "1").strip().lower()
SSL_VERIFY = _SSL_VERIFY_RAW not in ("0", "false", "no", "off")

_DEBUG = os.environ.get("INFINIGRAM_DEBUG", "").strip().lower() in ("1", "true", "yes")

# Space out requests to reduce403 rate limits from the public API (seconds).
_THROTTLE_SEC = float(os.environ.get("INFINIGRAM_THROTTLE_SEC", "0") or 0)
_last_request_mono: float | None = None

_CACHE: dict[str, dict[str, int]] | None = None
_INSECURE_WARNED = False


def _retryable_infini(exc: BaseException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (403, 429, 502, 503, 504)
    return False


def _throttle() -> None:
    global _last_request_mono
    if _THROTTLE_SEC <= 0:
        return
    now = time.monotonic()
    if _last_request_mono is not None:
        gap = _THROTTLE_SEC - (now - _last_request_mono)
        if gap > 0:
            time.sleep(gap)
    _last_request_mono = time.monotonic()


def _normalize_disk_cache(raw: Any) -> dict[str, dict[str, int]]:
    """Load nested {index: {query: count}}; migrate legacy flat {query: count}."""
    if not isinstance(raw, dict) or not raw:
        return {}
    first_val = next(iter(raw.values()), None)
    if isinstance(first_val, int):
        return {_LEGACY_INDEX: {str(k): int(v) for k, v in raw.items()}}
    out: dict[str, dict[str, int]] = {}
    for idx, m in raw.items():
        if not isinstance(m, dict):
            continue
        out[str(idx)] = {str(k): int(v) for k, v in m.items()}
    return out


def _load_cache() -> dict[str, dict[str, int]]:
    if not CACHE_PATH.exists():
        return {}

    with CACHE_PATH.open("r", encoding="utf-8") as f:
        raw: Any = json.load(f)
    return _normalize_disk_cache(raw)


def _save_cache(cache: dict[str, dict[str, int]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=True, indent=2, sort_keys=True)


def _get_cache() -> dict[str, dict[str, int]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_cache()
    return _CACHE


def _cache_get(cache: dict[str, dict[str, int]], index: str, query: str) -> int | None:
    bucket = cache.get(index)
    if bucket and query in bucket:
        return bucket[query]
    return None


def _cache_set(cache: dict[str, dict[str, int]], index: str, query: str, count: int) -> None:
    if index not in cache:
        cache[index] = {}
    cache[index][query] = count


@retry(
    retry=retry_if_exception(_retryable_infini),
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=8, max=180),
    reraise=True,
)
def _fetch_count(query: str) -> int:
    global _INSECURE_WARNED
    if _DEBUG:
        print(f"DEBUG infinigram query: {query!r}", flush=True)
    _throttle()
    if not SSL_VERIFY and not _INSECURE_WARNED:
        print(
            "WARNING: INFINIGRAM_SSL_VERIFY=0 — TLS certificate verification is disabled.",
            file=sys.stderr,
            flush=True,
        )
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        _INSECURE_WARNED = True
    response = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "rvc-probes/1.0 (contamination triage)",
        },
        json={
            "index": INDEX_NAME,
            "query_type": QUERY_TYPE,
            "query": query,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        verify=SSL_VERIFY,
    )
    if not response.ok:
        detail = (response.text or "")[:800]
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} for {response.url!r} "
            f"(index={INDEX_NAME!r}). Body: {detail}",
            response=response,
        )
    payload = response.json()
    if "error" in payload:
        raise ValueError(f"Infini-gram API error: {payload.get('error')!r}")
    if "count" not in payload:
        raise ValueError("Infini-gram response missing 'count'.")
    return int(payload["count"])


def get_ngram_count(query: str) -> int:
    """Return Infini-gram count for a query string, using disk cache."""
    if not query:
        return 0

    cache = _get_cache()
    hit = _cache_get(cache, INDEX_NAME, query)
    if hit is not None:
        return hit

    count = _fetch_count(query)
    _cache_set(cache, INDEX_NAME, query, count)
    _save_cache(cache)
    return count
