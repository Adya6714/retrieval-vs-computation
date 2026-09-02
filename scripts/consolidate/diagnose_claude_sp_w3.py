#!/usr/bin/env python3
"""I1: Diagnose Claude shortest_path W3. No model API calls.

1. Gold-in-gold-out: all SP W3 bank golds through verify_algo.
2. Classify every Claude SP W3 failure from stored raw responses.
3. Test whether the verifier expects numeric IDs while the model uses
   renamed labels (BW W3 shape in a different family).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify_algo import verify_algo  # noqa: E402

BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
CLAUDE = REPO_ROOT / "results/raw/ALGO_P1_behavioral_claude.csv"
DERIVED = REPO_ROOT / "results/derived"
OUT_ROWS = DERIVED / "I1_claude_sp_w3_failures.csv"
OUT_SUM = DERIVED / "I1_claude_sp_w3_summary.csv"

CLAUDE_MODEL = "anthropic/claude-sonnet-4"
REFUSAL = re.compile(
    r"\b(i\s+can'?t|i\s+cannot|i\s+am\s+unable|i'?m\s+not\s+able|"
    r"as\s+an\s+ai|i\s+won'?t\s+answer|i\s+must\s+refuse|i\s+refuse)\b",
    re.I,
)


def _params(raw: object) -> dict:
    try:
        p = json.loads(str(raw or ""))
        return p if isinstance(p, dict) else {}
    except json.JSONDecodeError:
        return {}


def _remap_labels_to_ids(text: str, params: dict) -> str:
    mapping = params.get("node_mapping") or {}
    items = sorted(mapping.items(), key=lambda kv: len(str(kv[1])), reverse=True)
    out = str(text)
    for kid, label in items:
        out = re.sub(re.escape(str(label)), str(kid), out, flags=re.IGNORECASE)
    return out


def _answer_uses_labels(text: str, params: dict) -> bool:
    low = str(text).lower()
    labels = [str(v).strip().lower() for v in (params.get("node_mapping") or {}).values() if str(v).strip()]
    hits = sum(1 for lab in labels if lab and lab in low)
    return hits >= 2


PATH_BLOCK_RE = re.compile(
    r"(Path\s*:\s*.{0,400}?(?:Cost|Total)\s*[:=]\s*-?\d+)",
    re.I | re.S,
)


def _last_path_block(text: str) -> str:
    hits = list(PATH_BLOCK_RE.finditer(str(text or "")))
    if hits:
        return hits[-1].group(1)
    hits2 = list(re.finditer(r"Path\s*:\s*[^\n]{0,300}", str(text or ""), re.I))
    return hits2[-1].group(0) if hits2 else ""


def _n_arrow_chunks(text: str) -> int:
    low = str(text).lower().replace("→", "->")
    return len(
        re.findall(r"([a-z0-9][a-z0-9 ]*(?:\s*->\s*[a-z0-9][a-z0-9 ]*)+)", low)
    )


def _answer_uses_numeric_path(text: str) -> bool:
    arrows = re.findall(
        r"((?:-?\d+)(?:\s*(?:→|->|,)\s*-?\d+){1,})",
        str(text).replace("→", "->"),
    )
    return bool(arrows)


def classify(
    ans: str,
    ok: bool,
    reason: str,
    meta: dict,
    last_ok: bool,
    remapped_ok: bool,
) -> str:
    raw = str(ans or "")
    stripped = raw.strip()
    if stripped == "" or stripped.lower() in {"nan", "none"}:
        return "empty"
    if stripped.upper().startswith("ERROR"):
        return "api_error"
    if REFUSAL.search(stripped):
        return "refusal"
    if ok:
        return "correct"
    # Final Path: line (or a numeric remap of the same answer) verifies, but
    # the full CoT does not — the verifier rejected a renamed/city-name form.
    if last_ok or remapped_ok:
        return "renamed_form_rejected"
    parse_status = str(meta.get("parse_status") or "")
    if parse_status == "parse_failed" or str(reason).startswith("parse_failed"):
        return "unparseable"
    return "wrong_path"


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    bank = pd.read_csv(BANK, dtype=str).fillna("")
    sp_w3 = bank[
        (bank["problem_subtype"].str.strip() == "shortest_path")
        & (bank["variant_type"].str.strip().str.lower() == "w3")
    ].copy()
    gold_pass = gold_fail = 0
    gold_fail_ids = []
    for _, r in sp_w3.iterrows():
        ok, reason, _meta = verify_algo(
            str(r["problem_id"]),
            str(r["correct_answer"]),
            str(r["correct_answer"]),
            "shortest_path",
            "W3",
            r["difficulty_params"],
        )
        if ok:
            gold_pass += 1
        else:
            gold_fail += 1
            gold_fail_ids.append((str(r["problem_id"]), reason))
    n_gold = len(sp_w3)
    print(f"SP W3 gold roundtrip: {gold_pass}/{n_gold} pass  ({gold_fail} fail)")
    for pid, reason in gold_fail_ids[:8]:
        print(f"  gold fail {pid}: {reason}")

    claude = pd.read_csv(CLAUDE, dtype=str).fillna("")
    claude = claude[claude["model"] == CLAUDE_MODEL]
    claude["variant_type"] = claude["variant_type"].astype(str).str.strip()
    claude["variant_type"] = claude["variant_type"].where(
        ~claude["variant_type"].str.lower().eq("w3"), "W3"
    )
    rows = claude[
        claude["problem_id"].isin(set(sp_w3["problem_id"]))
        & (claude["variant_type"] == "W3")
    ].drop_duplicates(["problem_id"], keep="last")
    bank_idx = sp_w3.set_index("problem_id")

    out_rows = []
    counts: Counter[str] = Counter()
    stored_true = 0
    recompute_true = 0
    for _, r in rows.iterrows():
        pid = str(r["problem_id"])
        b = bank_idx.loc[pid]
        params = _params(b["difficulty_params"])
        ans = str(r.get("model_answer") or r.get("raw_response") or "")
        gold = str(b["correct_answer"])
        ok, reason, meta = verify_algo(pid, ans, gold, "shortest_path", "W3", b["difficulty_params"])
        remapped = _remap_labels_to_ids(ans, params)
        rok, rreason, _rmeta = verify_algo(
            pid, remapped, gold, "shortest_path", "W3", b["difficulty_params"]
        )
        last_block = _last_path_block(ans)
        if last_block:
            lok, lreason, lmeta = verify_algo(
                pid, last_block, gold, "shortest_path", "W3", b["difficulty_params"]
            )
        else:
            lok, lreason, lmeta = False, "no_path_block", {}
        stored = str(r.get("verified", "")).strip().lower() == "true"
        if stored:
            stored_true += 1
        if ok:
            recompute_true += 1
        empty_map = not bool(params.get("node_mapping"))
        label = classify(ans, bool(ok), reason, meta, bool(lok), bool(rok))
        counts[label] += 1
        out_rows.append(
            {
                "problem_id": pid,
                "stored_verified": stored,
                "recomputed_ok": bool(ok),
                "reason": reason,
                "parse_status": meta.get("parse_status", ""),
                "path_provided": meta.get("path_provided", ""),
                "failure_class": label,
                "empty_node_mapping": empty_map,
                "n_arrow_chunks": _n_arrow_chunks(ans),
                "uses_mapping_labels": _answer_uses_labels(ans, params),
                "uses_numeric_arrow_path": _answer_uses_numeric_path(ans),
                "last_path_block_ok": bool(lok),
                "last_path_block_reason": lreason,
                "last_path_parse_status": (lmeta or {}).get("parse_status", ""),
                "remapped_labels_to_ids_ok": bool(rok),
                "remapped_reason": rreason,
                "last_path_preview": re.sub(r"\s+", " ", last_block)[:200],
                "answer_preview": re.sub(r"\s+", " ", ans)[:240],
            }
        )
    pd.DataFrame(out_rows).to_csv(OUT_ROWS, index=False)

    n = len(out_rows)
    acc_stored = stored_true / n if n else float("nan")
    acc_re = recompute_true / n if n else float("nan")
    summary = [
        {"metric": "sp_w3_gold_roundtrip_pass", "value": gold_pass, "n": n_gold, "note": "verify_algo(gold, gold)"},
        {"metric": "sp_w3_gold_roundtrip_fail", "value": gold_fail, "n": n_gold, "note": ""},
        {"metric": "claude_sp_w3_n", "value": n, "n": n, "note": CLAUDE_MODEL},
        {"metric": "claude_sp_w3_stored_acc", "value": round(acc_stored, 6), "n": n, "note": "verified column"},
        {"metric": "claude_sp_w3_recomputed_acc", "value": round(acc_re, 6), "n": n, "note": "verify_algo on raw model_answer"},
    ]
    for cls in ("correct", "wrong_path", "renamed_form_rejected", "unparseable", "refusal", "empty", "api_error"):
        summary.append(
            {
                "metric": f"claude_sp_w3_{cls}",
                "value": int(counts[cls]),
                "n": n,
                "note": "taxonomy on full CoT; renamed_form_rejected = last Path: line verifies",
            }
        )
    n_empty_map = sum(1 for r in out_rows if r["empty_node_mapping"])
    n_last_ok = sum(1 for r in out_rows if r["last_path_block_ok"])
    n_cost_only = sum(1 for r in out_rows if str(r["last_path_block_reason"]) == "correct_cost_only")
    n_remap_rescue = sum(1 for r in out_rows if r["remapped_labels_to_ids_ok"] and not r["recomputed_ok"])
    summary.extend(
        [
            {"metric": "sp_w3_empty_node_mapping", "value": n_empty_map, "n": n_gold, "note": "bank difficulty_params.node_mapping missing"},
            {"metric": "claude_sp_w3_last_path_block_ok", "value": n_last_ok, "n": n, "note": "verify_algo on trailing Path:/Cost: only"},
            {"metric": "claude_sp_w3_last_path_cost_only", "value": n_cost_only, "n": n, "note": "city names unmapped; cost matches Dijkstra"},
            {"metric": "claude_sp_w3_remap_rescues", "value": n_remap_rescue, "n": n, "note": "string-replace mapping labels with IDs in full CoT"},
        ]
    )
    pd.DataFrame(summary).to_csv(OUT_SUM, index=False)

    print(f"Claude SP W3 n={n} stored_acc={acc_stored:.3f} recomputed_acc={acc_re:.3f}")
    print("failure_class counts (report before interpretation):")
    for cls, c in counts.most_common():
        print(f"  {cls:24s} {c}")
    print(f"empty node_mapping in bank: {n_empty_map}/{n_gold}")
    print(f"last Path: block verifies: {n_last_ok}/{n}")
    print(f"last Path: cost-only (unmapped names): {n_cost_only}/{n}")
    print(f"remap-labels-to-ids rescues a failure: {n_remap_rescue}")
    print(f"Wrote {OUT_ROWS}")
    print(f"Wrote {OUT_SUM}")


if __name__ == "__main__":
    main()
