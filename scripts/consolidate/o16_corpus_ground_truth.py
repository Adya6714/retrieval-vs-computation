#!/usr/bin/env python3
"""O16 Part A: Corpus ground truth via Infini-gram on The Pile and Dolma.

Searches every canonical GSM/ALGO/BW problem statement in:
  - v4_piletrain_llama  (Pythia / The Pile)
  - v4_dolma-v1_7_llama  (OLMo lineage / Dolma)

Records exact match, longest matching whitespace n-gram, corpus document
count, and matching document IDs (via find + get_doc_by_rank).

Near-exact = not exact, but longest n-gram length >= max(10, ceil(0.4 * n_tokens)).
Ground-truth member = exact OR near-exact.

Paper note: this calibration exists only for open-corpus models (Pythia, OLMo).
It cannot be done for Claude, GPT-4o, Gemini, o4-mini, or DeepSeek.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)
load_dotenv(REPO_ROOT / ".env")

# Prefer faster retries for bulk O16 sweeps (must be set before client import).
os.environ.setdefault("INFINIGRAM_FAST", "1")

# Pile/Dolma indexes live on the full Infini-gram API, not the mini endpoint
# (mini often only hosts a DCLM shard). Prefer explicit O16 override, else full API.
os.environ["INFINIGRAM_API_URL"] = os.environ.get(
    "O16_INFINIGRAM_API_URL",
    "https://api.infini-gram.io/",
)

from probes.contamination import infinigram_client as _ig_mod  # noqa: E402

# Reload URL constants after env override (module may have been imported elsewhere).
_ig_mod.API_URL = os.environ["INFINIGRAM_API_URL"].rstrip("/") + "/"

from probes.contamination.infinigram_client import (  # noqa: E402
    find_matching_docs,
    get_ngram_count,
)

DER = REPO_ROOT / "results" / "derived"
BANK = REPO_ROOT / "data" / "problems"
OUT_CSV = DER / "O16_corpus_ground_truth.csv"
OUT_JSONL = DER / "O16_corpus_ground_truth.jsonl"

CORPORA = [
    {
        "corpus": "pile",
        "index": "v4_piletrain_llama",
        "model_lineage": "Pythia (The Pile)",
    },
    {
        "corpus": "dolma",
        "index": "v4_dolma-v1_7_llama",
        "model_lineage": "OLMo (Dolma)",
    },
]

MIN_NGRAM = 5
STRIDE_PROBE = 5
MAX_DOCS = 3
NEAR_EXACT_FRAC = 0.4
NEAR_EXACT_MIN_LEN = 10
# Cap binary-search length: enough for near-exact without scanning 150-token BW prompts.
MAX_NGRAM_SEARCH = 25


def _norm_vt(v: str) -> str:
    v = str(v).strip()
    return "canonical" if v.lower() == "canonical" else v.upper()


def _strip_quotes(text: str) -> str:
    s = str(text)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


def load_canonicals(limit: int | None = None) -> list[dict]:
    specs = [
        ("GSM", BANK / "question_bank_gsm.csv"),
        ("ALGO", BANK / "question_bank_algo.csv"),
        ("BW", BANK / "question_bank_bw.csv"),
    ]
    items: list[dict] = []
    for family, path in specs:
        df = pd.read_csv(path, dtype=str).fillna("")
        df["variant_type"] = df["variant_type"].map(_norm_vt)
        can = df[df["variant_type"] == "canonical"].copy()
        for _, row in can.iterrows():
            text = _strip_quotes(str(row["problem_text"])).strip()
            if not text:
                continue
            items.append(
                {
                    "family": family,
                    "problem_id": str(row["problem_id"]).strip(),
                    "problem_text": text,
                    "n_tokens": len(text.split()),
                }
            )
        if limit is not None:
            fam_items = [x for x in items if x["family"] == family]
            keep_ids = {x["problem_id"] for x in fam_items[:limit]}
            items = [
                x
                for x in items
                if x["family"] != family or x["problem_id"] in keep_ids
            ]
    return items


def _any_ngram_hit(text: str, n: int, index: str) -> tuple[bool, int, str]:
    """Window probe: dense for short n-grams, sparse for long ones."""
    tokens = text.split()
    if len(tokens) < n:
        return False, 0, ""
    last = len(tokens) - n
    if n <= 8 or last <= 20:
        positions = list(range(0, last + 1, max(1, STRIDE_PROBE // 2 or 1)))
        if last not in positions:
            positions.append(last)
    else:
        positions = sorted({0, last, max(0, last // 2), max(0, last // 4), max(0, (3 * last) // 4)})
    best_count = 0
    best_ngram = ""
    for i in positions:
        ngram = " ".join(tokens[i : i + n])
        count = get_ngram_count(ngram, index=index)
        if count > best_count:
            best_count = count
            best_ngram = ngram
        if count > 0:
            return True, count, ngram
    return best_count > 0, best_count, best_ngram


def longest_match(text: str, index: str) -> tuple[int, int, str, bool]:
    """Membership-first match with a small query budget.

    Returns (best_len, best_count, best_ngram, near_exact).
    Near-exact is decided by a direct threshold probe (not full binary search),
    which keeps Infini-gram calls within rate limits for ~400 cells.
    """
    tokens = text.split()
    token_len = len(tokens)
    if token_len < MIN_NGRAM:
        return 0, 0, "", False

    thresh = max(
        NEAR_EXACT_MIN_LEN,
        int(math.ceil(NEAR_EXACT_FRAC * min(token_len, MAX_NGRAM_SEARCH))),
    )
    thresh = min(thresh, token_len, MAX_NGRAM_SEARCH)

    hit, count, ngram = _any_ngram_hit(text, thresh, index)
    if hit:
        best_len, best_count, best_ngram = thresh, count, ngram
        # Opportunistically try a couple of longer lengths (≤2 extra probes).
        for n in (min(token_len, MAX_NGRAM_SEARCH),):
            if n <= best_len:
                continue
            h2, c2, g2 = _any_ngram_hit(text, n, index)
            if h2:
                best_len, best_count, best_ngram = n, c2, g2
        return best_len, best_count, best_ngram, True

    # Not near-exact: record a short-floor length if any 5-gram hits.
    h5, c5, g5 = _any_ngram_hit(text, MIN_NGRAM, index)
    if h5:
        return MIN_NGRAM, c5, g5, False
    return 0, 0, "", False


def near_exact(n_tokens: int, best_len: int, exact: bool) -> bool:
    if exact or best_len <= 0 or n_tokens <= 0:
        return False
    # Cap threshold by MAX_NGRAM_SEARCH so BW items remain classifiable.
    thresh = max(
        NEAR_EXACT_MIN_LEN,
        int(math.ceil(NEAR_EXACT_FRAC * min(n_tokens, MAX_NGRAM_SEARCH))),
    )
    return best_len >= thresh


def search_one(item: dict, corpus_meta: dict, *, phase: str = "full") -> dict:
    text = item["problem_text"]
    n_tokens = item["n_tokens"]
    index = corpus_meta["index"]
    # Infini-gram API rejects queries > 1000 characters.
    truncated = len(text) > 1000
    query_text = text[:1000] if truncated else text
    exact_count = get_ngram_count(query_text, index=index) if n_tokens >= 1 else 0
    # Full-string exact only if we did not truncate.
    exact = (exact_count > 0) and (not truncated)

    if exact:
        best_len, best_count, best_ngram = n_tokens, exact_count, text
        is_near = False
    elif phase == "exact":
        # Defer near-exact probes to --phase near (rate-limit friendly).
        best_len, best_count, best_ngram, is_near = 0, 0, "", False
    else:
        best_len, best_count, best_ngram, is_near = longest_match(text, index)

    member = exact or is_near

    doc_query = text if exact else (best_ngram if is_near else "")
    docs: list[dict] = []
    if doc_query and phase != "exact":
        try:
            docs = find_matching_docs(
                doc_query,
                index=index,
                max_docs=MAX_DOCS if exact else 1,
            )
        except Exception as exc:  # noqa: BLE001
            docs = [{"error": str(exc)[:240]}]
    elif exact and phase == "exact":
        # Still try to grab doc IDs for exact hits (1 doc).
        try:
            docs = find_matching_docs(query_text, index=index, max_docs=1)
        except Exception as exc:  # noqa: BLE001
            docs = [{"error": str(exc)[:240]}]

    doc_ids = []
    for d in docs:
        if d.get("metadata_id"):
            doc_ids.append(str(d["metadata_id"]))
        elif d.get("doc_ix") is not None:
            doc_ids.append(f"doc_ix:{d['doc_ix']}")

    return {
        "family": item["family"],
        "problem_id": item["problem_id"],
        "corpus": corpus_meta["corpus"],
        "index_name": index,
        "model_lineage": corpus_meta["model_lineage"],
        "n_tokens": n_tokens,
        "exact_match_found": exact,
        "exact_match_count": int(exact_count),
        "longest_ngram_length": int(best_len),
        "longest_ngram_count": int(best_count),
        "longest_ngram_text": best_ngram[:500],
        "near_exact_match_found": is_near,
        "ground_truth_member": member,
        "matched_contamination_score": round(best_len / max(n_tokens, 1), 6),
        "corpus_document_count": int(best_count if best_len > 0 else 0),
        "matching_document_ids": "|".join(doc_ids),
        "matching_documents_json": json.dumps(docs, ensure_ascii=False)[:4000],
        "search_phase": phase,
        "exact_query_truncated": truncated,
        "near_exact_rule": (
            f"longest>=max({NEAR_EXACT_MIN_LEN},ceil({NEAR_EXACT_FRAC}*n_tokens)); "
            f"max_ngram_search={MAX_NGRAM_SEARCH}"
        ),
        "closed_model_note": (
            "Ground-truth corpus membership is only identifiable for open-corpus "
            "models (Pythia/Pile, OLMo/Dolma). Impossible for Claude, GPT-4o, "
            "Gemini, o4-mini, DeepSeek."
        ),
    }


def _done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    keys: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.add((rec["family"], rec["problem_id"], rec["corpus"]))
    return keys


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(["family", "problem_id", "corpus"], keep="last")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Per-family smoke limit")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--family",
        nargs="*",
        default=None,
        help="Optional family filter, e.g. --family GSM ALGO",
    )
    ap.add_argument(
        "--phase",
        choices=["exact", "near", "full"],
        default="exact",
        help="exact=1 query/cell; near=enrich non-exact rows; full=exact+near",
    )
    ap.add_argument(
        "--throttle",
        type=float,
        default=None,
        help="Override INFINIGRAM_THROTTLE_SEC",
    )
    args = ap.parse_args()

    if args.throttle is not None:
        os.environ["INFINIGRAM_THROTTLE_SEC"] = str(args.throttle)
        import probes.contamination.infinigram_client as ig

        ig._THROTTLE_SEC = float(args.throttle)

    DER.mkdir(parents=True, exist_ok=True)
    items = load_canonicals(args.limit)
    if args.family:
        keep = {f.upper() for f in args.family}
        items = [x for x in items if x["family"] in keep]
    print(
        f"[O16A] phase={args.phase} | {len(items)} canonicals × {len(CORPORA)} corpora"
    )

    existing: dict[tuple[str, str, str], dict] = {}
    if OUT_JSONL.exists() and not args.no_resume:
        with OUT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                existing[(rec["family"], rec["problem_id"], rec["corpus"])] = rec
    print(f"[O16A] existing keys={len(existing)}")

    n_new = 0

    def _handle_error(exc: Exception) -> None:
        print(f"    ERROR {type(exc).__name__}: {exc}", flush=True)
        if "403" in str(exc) or "429" in str(exc) or "Forbidden" in str(exc):
            print("    sleeping 180s after rate limit...", flush=True)
            time.sleep(180)
        else:
            time.sleep(5)

    if args.phase == "near":
        # Start from existing; update in place after each probe so progress survives kills.
        by_key = dict(existing)
        for item in items:
            for corpus_meta in CORPORA:
                key = (item["family"], item["problem_id"], corpus_meta["corpus"])
                prev = by_key.get(key)
                if prev is None:
                    continue
                exact = prev.get("exact_match_found") in (True, "true", "True")
                already = (
                    prev.get("near_probed")
                    or prev.get("search_phase") in {"near", "full"}
                    # Early pre-phase rows already ran a full/sparse longest probe.
                    or (
                        prev.get("search_phase") not in {"exact", "near"}
                        and int(prev.get("longest_ngram_length") or 0) >= 0
                        and "longest_ngram_length" in prev
                        and prev.get("search_phase") is None
                    )
                )
                # Re-probe only exact-phase rows that deferred near (longest still 0).
                if prev.get("search_phase") == "exact" and not exact:
                    already = False
                if prev.get("near_probed"):
                    already = True
                if exact or already:
                    continue
                print(
                    f"  [near] {key[0]} {key[1]} @ {key[2]} "
                    f"(n_tokens={item['n_tokens']})",
                    flush=True,
                )
                try:
                    rec = search_one(item, corpus_meta, phase="near")
                    rec["near_probed"] = True
                    by_key[key] = rec
                    n_new += 1
                    print(
                        f"    exact={rec['exact_match_found']} near={rec['near_exact_match_found']} "
                        f"longest={rec['longest_ngram_length']}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    _handle_error(exc)
                    continue
                # Checkpoint after every successful probe.
                with OUT_JSONL.open("w", encoding="utf-8") as f:
                    for rec in by_key.values():
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        mode = "w" if args.no_resume else "a"
        with OUT_JSONL.open(mode, encoding="utf-8") as f:
            for item in items:
                for corpus_meta in CORPORA:
                    key = (item["family"], item["problem_id"], corpus_meta["corpus"])
                    if key in existing and not args.no_resume:
                        continue
                    phase = args.phase  # exact or full
                    print(
                        f"  [{phase}] {key[0]} {key[1]} @ {key[2]} "
                        f"(n_tokens={item['n_tokens']})",
                        flush=True,
                    )
                    try:
                        rec = search_one(item, corpus_meta, phase=phase)
                    except Exception as exc:  # noqa: BLE001
                        _handle_error(exc)
                        continue
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_new += 1
                    print(
                        f"    exact={rec['exact_match_found']} near={rec['near_exact_match_found']} "
                        f"longest={rec['longest_ngram_length']} count={rec['corpus_document_count']}",
                        flush=True,
                    )

    df = jsonl_to_csv(OUT_JSONL, OUT_CSV)
    print(f"[O16A] wrote {n_new} new/updated rows; total {len(df)} → {OUT_CSV}")
    if not df.empty:
        print(
            df.groupby(["family", "corpus"])["ground_truth_member"]
            .agg(["sum", "count", "mean"])
            .to_string()
        )


if __name__ == "__main__":
    main()
