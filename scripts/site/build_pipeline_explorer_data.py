#!/usr/bin/env python3
"""
Build site/data/pipeline_explorer.json for the website's Pipeline Explorer.

Reads ONLY committed artefacts (operating rule 1: raw/derived CSVs are truth;
this script derives, never invents). For each exemplar item it emits:
  - every surface variant (canonical, W1..W6) with text + gold + verifier name
  - per-model Probe 1 verdicts (rescored_correct, included, exclusion_reason)
  - Probe 3 proximity scores (within-family exposure proxy)
  - Probe 2 per-instance CCI where it exists (ALGO)
  - programme routing: which phases/claims this item feeds and their status

Usage:
  PYTHONPATH=. python scripts/site/build_pipeline_explorer_data.py
  PYTHONPATH=. python scripts/site/build_pipeline_explorer_data.py --ids GSM_001 ALGO_CC_001 BW_001
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BANKS = {
    "GSM": ROOT / "data/problems/question_bank_gsm.csv",
    "BW": ROOT / "data/problems/question_bank_bw.csv",
    "ALGO": ROOT / "data/problems/question_bank_algo.csv",
}
P3 = {
    "GSM": ROOT / "results/raw/GSM_P3_contamination.csv",
    "BW": ROOT / "results/raw/BW_P3_contamination.csv",
    "ALGO": ROOT / "results/raw/ALGO_P3_contamination.csv",
}
CCI = ROOT / "results/derived/ALGO_P2_per_instance_cci.csv"
OUT = ROOT / "site/data/pipeline_explorer.json"

MODEL_LABELS = {
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "meta-llama/llama-3.1-8b-instruct": "Llama-3.1-8B",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "R1-distill-70B",
}
VARIANT_ORDER = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
VARIANT_MEANING = {
    "canonical": "Original benchmark item.",
    "W1": "Paraphrase. Same entities, same numbers, new wording.",
    "W2": "Structural reformat (table, matrix, equation set).",
    "W3": "Cover-story isomorph: entities and framing moved to a new real-word domain (W3b). True nonce rename (W3a) is being added, see EF-07.",
    "W4": "Formal notation (LaTeX, PDDL predicates, edge list).",
    "W5": "Direction reversal (initial and goal, source and destination).",
    "W6": "Procedural regeneration: same algorithm, new numbers.",
}
GOLD_RULE = {v: "fixed" for v in ["canonical", "W1", "W2", "W3", "W4"]}
GOLD_RULE.update({"W5": "re-derived", "W6": "re-derived"})
VERIFIER_KIND = {
    "GSM": "Numeric match against gold",
    "BW": "PDDL state simulator replays the plan",
    "ALGO": "Exact optimal solver recomputes the optimum",
}
PHASE_ROUTING = [
    {"phase": "0", "name": "Foundations", "status": "complete",
     "uses_item": "Probe 1-3 behavioural layer, intrusion errors, mixture IRT, G-theory.",
     "claims": ["CC-1"]},
    {"phase": "1", "name": "Calibration + mechanism", "status": "blocked (GPU)",
     "uses_item": "D1-lite seen/unseen LoRA: does this item's label track known exposure? Gate G1.",
     "claims": ["H1", "CC-2", "H5"]},
    {"phase": "1T", "name": "Open-weight T4 track (amendment A1)", "status": "ready",
     "uses_item": "Noise floor, precision sweep, base/Coder/Math and distillation contrasts on the same item.",
     "claims": ["H8", "H9", "H10", "H11", "H12", "H13"]},
    {"phase": "2", "name": "Architecture claim", "status": "designed",
     "uses_item": "Symbolic-pathway occupancy on canonical vs W3; commitment depth. Gate G2.",
     "claims": ["AC-1", "AC-2", "H6"]},
    {"phase": "3", "name": "Laws", "status": "designed",
     "uses_item": "Distance ladders, direction, diversity-vs-dose training arms incl. RLVR. Gate G3.",
     "claims": ["H3", "H7", "H7b", "AC-3"]},
    {"phase": "4", "name": "Ecology and development", "status": "planned",
     "uses_item": "OLMo checkpoint sweep: when does this item become rename-invariant?",
     "claims": ["CC-3", "H4"]},
]


def parse_mapping(notes):
    """Extract the persisted entity mapping from the bank's notes column, if any."""
    if not isinstance(notes, str) or "{" not in notes:
        return None
    try:
        obj = json.loads(notes[notes.index("{"):])
    except Exception:
        return None
    out = {}
    if isinstance(obj, dict):
        if "chosen_domain" in obj:
            out["domain"] = obj["chosen_domain"]
        if isinstance(obj.get("entity_mapping"), dict):
            out["mapping"] = obj["entity_mapping"]
    return out or None


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (bool,)):
        return v
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return v


def load_p1(family: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / f"results/derived/{family}_P1_behavioral*_rescored.csv")))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        keep = [c for c in ["problem_id", "variant_type", "variant_type_normalized", "model",
                            "rescored_correct", "included", "exclusion_reason", "verify_method"] if c in d.columns]
        frames.append(d[keep])
    d = pd.concat(frames, ignore_index=True)
    d = d[d["model"].isin(MODEL_LABELS)]  # drops 'mock' and malformed rows
    vcol = "variant_type_normalized" if "variant_type_normalized" in d.columns else "variant_type"
    d["variant"] = d[vcol].fillna(d["variant_type"])
    return d


def auto_pick(bank: pd.DataFrame, p1: pd.DataFrame, family: str, subtype: str | None = None) -> str:
    """Pick the most instructive item: canonical solved by most models, W3 split."""
    ids = bank[bank["variant_type"] == "canonical"]
    if subtype:
        ids = ids[ids["problem_subtype"] == subtype]
    best, best_score = None, -1.0
    inc = p1[p1["included"] == True]  # noqa: E712
    for pid in ids["problem_id"]:
        if family == "BW" and (pid.startswith("BW_E") or pid.startswith("MBW_")):
            continue  # BW_E* lack PDDL for triangulation; MBW is obfuscated (shared-hard)
        sub = inc[inc["problem_id"] == pid]
        can = sub[sub["variant"] == "canonical"]["rescored_correct"].dropna()
        w3 = sub[sub["variant"] == "W3"]["rescored_correct"].dropna()
        nvar = bank[bank["problem_id"] == pid]["variant_type"].nunique()
        if len(can) < 3 or len(w3) < 3:
            continue
        split = 1 - abs(w3.mean() - 0.5) * 2  # 1.0 when W3 splits models evenly
        w3_notes = bank[(bank["problem_id"] == pid) & (bank["variant_type"] == "W3")]["notes"]
        has_map = bool(len(w3_notes)) and parse_mapping(w3_notes.iloc[0]) is not None
        score = can.mean() + split + 0.1 * nvar + (0.5 if has_map else 0.0)
        if score > best_score:
            best, best_score = pid, score
    return best


def build_item(pid: str, family: str, bank: pd.DataFrame, p1: pd.DataFrame,
               p3: pd.DataFrame, cci: pd.DataFrame | None) -> dict:
    rows = bank[bank["problem_id"] == pid].copy()
    rows["order"] = rows["variant_type"].map({v: i for i, v in enumerate(VARIANT_ORDER)})
    rows = rows.sort_values("order")
    item_p1 = p1[p1["problem_id"] == pid]
    variants = []
    for _, r in rows.iterrows():
        v = r["variant_type"]
        res = []
        for m, lab in MODEL_LABELS.items():
            hit = item_p1[(item_p1["model"] == m) & (item_p1["variant"] == v)]
            if hit.empty:
                continue
            h = hit.iloc[-1]
            res.append({
                "model": lab,
                "correct": _clean(h.get("rescored_correct")),
                "included": _clean(h.get("included")),
                "exclusion_reason": _clean(h.get("exclusion_reason")),
            })
        variants.append({
            "variant": v,
            "meaning": VARIANT_MEANING.get(v, ""),
            "text": _clean(r["problem_text"]),
            "gold": _clean(str(r["correct_answer"])),
            "gold_rule": GOLD_RULE.get(v, "fixed"),
            "verifier_function": _clean(r.get("verifier_function")),
            "mapping": parse_mapping(r.get("notes")) if v == "W3" else None,
            "probe1": res,
        })
    prox = p3[p3["problem_id"] == pid]
    proximity = None
    if not prox.empty:
        x = prox.iloc[0]
        proximity = {k: _clean(x[k]) for k in ["contamination_score", "max_ngram_length", "max_ngram_count",
                                                "template_contamination_score", "instance_contamination_score"]
                     if k in prox.columns}
        fam_scores = p3["contamination_score"].dropna()
        if proximity.get("contamination_score") is not None and len(fam_scores):
            proximity["family_percentile"] = round(float((fam_scores < proximity["contamination_score"]).mean()), 3)
        proximity["family_scores"] = [round(float(s), 4) for s in fam_scores.tolist()]
    probe2 = None
    if cci is not None:
        c = cci[cci["problem_id"] == pid]
        if not c.empty:
            probe2 = [{"model": MODEL_LABELS.get(m, m), "cci": _clean(round(float(v), 3)) if pd.notna(v) else None}
                      for m, v in zip(c["model"], c["cci_composite"])]
    canon = rows[rows["variant_type"] == "canonical"].iloc[0]
    return {
        "problem_id": pid,
        "family": family,
        "subtype": _clean(canon.get("problem_subtype")),
        "difficulty": _clean(canon.get("difficulty")),
        "source": _clean(canon.get("source")),
        "verifier_kind": VERIFIER_KIND[family],
        "variants": variants,
        "probe3_proximity": proximity,
        "probe2_cci": probe2,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    cci = pd.read_csv(CCI) if CCI.exists() else None
    items = []
    picks = {"GSM": [None], "BW": [None], "ALGO": ["coin_change", "wis"]}
    for family, subtypes in picks.items():
        bank = pd.read_csv(BANKS[family])
        p1 = load_p1(family)
        p3 = pd.read_csv(P3[family])
        chosen = []
        if args.ids:
            chosen = [i for i in args.ids if i in set(bank["problem_id"])]
        else:
            for st in subtypes:
                pid = auto_pick(bank, p1, family, st)
                if pid:
                    chosen.append(pid)
        for pid in chosen:
            items.append(build_item(pid, family, bank, p1, p3, cci))
    payload = {
        "generated_by": "scripts/site/build_pipeline_explorer_data.py",
        "sources": ["data/problems/question_bank_*.csv",
                    "results/derived/*_P1_behavioral*_rescored.csv",
                    "results/raw/*_P3_contamination.csv",
                    "results/derived/ALGO_P2_per_instance_cci.csv"],
        "phase_routing": PHASE_ROUTING,
        "items": items,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str))
    print(f"wrote {OUT} with {len(items)} items: {[i['problem_id'] for i in items]}")


if __name__ == "__main__":
    main()
