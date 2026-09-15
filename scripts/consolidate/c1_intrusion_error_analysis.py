#!/usr/bin/env python3
"""C1 / DS-02: Intrusion-error analysis on Probe-1 (no new API calls).

Pre-registration: results/derived/C1_INTRUSION_PREREG.md
Methods note:     results/derived/C1_INTRUSION_METHODS.md

Outputs:
  results/derived/C1_intrusion_errors.csv
  results/derived/C1_intrusion_rates.csv
  results/derived/C1_intrusion_vs_contamination.csv
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import algo_cluster_map  # noqa: E402
from probes.common.cluster_inference import cluster_bootstrap_assoc  # noqa: E402
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.stats import cluster_bootstrap_ci  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402
from probes.contamination.verify import (  # noqa: E402
    parse_action_mapping_from_notes,
    parse_entity_mapping_from_notes,
)
from probes.contamination.verify_algo import resolve_sp_node_mapping  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
BANK = {
    "ALGO": REPO_ROOT / "data/problems/question_bank_algo.csv",
    "BW": REPO_ROOT / "data/problems/question_bank_bw.csv",
    "GSM": REPO_ROOT / "data/problems/question_bank_gsm.csv",
}

OUT_ERRORS = DER / "C1_intrusion_errors.csv"
OUT_RATES = DER / "C1_intrusion_rates.csv"
OUT_CORR = DER / "C1_intrusion_vs_contamination.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
VARIANTS = ("W3", "W6")
PARTIAL_THRESH = 0.5
N_BOOT_RATE = 10000
N_BOOT_ASSOC = 5000
SEED = 42
GSM_TOL = 0.01

_PLAN_LINE_PREFIX = re.compile(r"^\s*\d+[\).\s]+")
_SP_PATH = re.compile(r"Path\s*:\s*(.+?)(?:,\s*Cost\s*:|$)", re.I)
_CC_COUNT = re.compile(r"(?:Count|Total)\s*:\s*(-?\d+)", re.I)
_CC_LIST = re.compile(r"(?:Coins|Scoops)\s*:\s*\[([^\]]*)\]", re.I)
_WIS_SEL = re.compile(r"Selected\s*:\s*\{([^}]*)\}", re.I)
_NUM = re.compile(r"(?<![\w])\$?-?[\d,]+(?:\.\d+)?(?![\w])")
_HASH_NUM = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _blocker_check() -> None:
    """Require raw response text in Probe-1 sources. Stop if only booleans."""
    samples = [
        ("ALGO", next(DER.glob("ALGO_P1_behavioral_*_rescored.csv"))),
        ("BW", DER / "BW_P1_behavioral_rescored.csv"),
        ("GSM", next(DER.glob("GSM_P1_behavioral_*_rescored.csv"))),
    ]
    for fam, path in samples:
        df = pd.read_csv(path, nrows=5, dtype=str)
        cols = set(df.columns)
        has_text = bool(cols & {"raw_response", "model_answer"})
        if not has_text:
            raise SystemExit(
                f"BLOCKER: {fam} Probe-1 CSV {path.name} has no raw_response/"
                f"model_answer — only verdict columns. Cannot run DS-02."
            )
        text_col = "model_answer" if "model_answer" in cols else "raw_response"
        nonempty = df[text_col].fillna("").astype(str).str.strip().ne("").any()
        if not nonempty:
            raise SystemExit(
                f"BLOCKER: {fam} Probe-1 text column {text_col!r} is empty."
            )
    print("[blocker] PASS — raw response text present for ALGO/BW/GSM")


# ---------------------------------------------------------------------------
# Clone families
# ---------------------------------------------------------------------------

def _clone_map_all() -> dict[tuple[str, str], str]:
    path = DER / "bank_clone_audit.csv"
    out: dict[tuple[str, str], str] = {}
    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, r in df.iterrows():
            fam = str(r["family"]).strip().upper()
            pid = str(r["problem_id"]).strip()
            cid = str(r.get("clone_family_id") or "").strip()
            out[(fam, pid)] = cid if cid else f"SINGLETON_{pid}"
    # ALGO fallback from probes.common.clones
    for pid, cid in algo_cluster_map().items():
        out.setdefault(("ALGO", pid), cid)
    return out


def clone_family_for(fam: str, pid: str, cmap: dict[tuple[str, str], str]) -> str:
    return cmap.get((fam, str(pid)), f"SINGLETON_{pid}")


# ---------------------------------------------------------------------------
# Bank + mappings
# ---------------------------------------------------------------------------

def _parse_notes_json(notes: str | None) -> dict:
    text = str(notes or "")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            blob, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict):
            return blob
    return {}


def load_banks() -> dict[str, pd.DataFrame]:
    banks = {}
    for fam, path in BANK.items():
        df = pd.read_csv(path, dtype=str).fillna("")
        df["variant"] = df["variant_type"].map(normalize_variant)
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
        banks[fam] = df
    return banks


def bank_gold_index(banks: dict[str, pd.DataFrame]) -> dict[tuple[str, str, str], str]:
    """(family, problem_id, variant) -> correct_answer."""
    idx: dict[tuple[str, str, str], str] = {}
    for fam, df in banks.items():
        for _, r in df.iterrows():
            key = (fam, str(r["problem_id"]), str(r["variant"]))
            idx[key] = str(r.get("correct_answer") or "")
    return idx


def bank_row(banks: dict[str, pd.DataFrame], fam: str, pid: str, variant: str) -> dict:
    df = banks[fam]
    sub = df[(df["problem_id"] == pid) & (df["variant"] == variant)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Structured extractors
# ---------------------------------------------------------------------------

def extract_sp_path(text: str) -> tuple[str, ...]:
    s = str(text or "")
    matches = list(_SP_PATH.finditer(s))
    if not matches:
        return tuple()
    blob = matches[-1].group(1).split("\n")[0]
    if re.search(r"→|->", blob):
        parts = re.split(r"\s*(?:→|->)\s*", blob)
        toks = [re.sub(r"[^A-Za-z0-9]", "", p.strip()) for p in parts]
        toks = [t for t in toks if t and t.lower() not in {"path", "cost"}]
        if len(toks) >= 2:
            return tuple(t.upper() for t in toks)
    nums = re.findall(r"\b\d+\b", blob)
    return tuple(nums) if len(nums) >= 2 else tuple()


def extract_cc(text: str) -> tuple[int | None, tuple[int, ...]]:
    s = str(text or "")
    cm = _CC_COUNT.search(s)
    count = int(cm.group(1)) if cm else None
    lm = _CC_LIST.search(s)
    coins: tuple[int, ...] = tuple()
    if lm:
        coins = tuple(sorted(int(x) for x in re.findall(r"-?\d+", lm.group(1))))
    return count, coins


def extract_wis(text: str) -> frozenset[str]:
    s = str(text or "")
    sm = _WIS_SEL.search(s)
    if not sm:
        return frozenset()
    toks = [t.strip().upper() for t in re.split(r"[,\s]+", sm.group(1)) if t.strip()]
    return frozenset(toks)


def extract_gsm_final(text: str) -> float | None:
    s = str(text or "")
    tagged = _HASH_NUM.search(s)
    if tagged:
        try:
            return float(tagged.group(1).replace(",", ""))
        except ValueError:
            return None
    nums = _NUM.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace("$", "").replace(",", ""))
    except ValueError:
        return None


def extract_gsm_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM.finditer(str(text or "")):
        try:
            out.append(float(m.group(0).replace("$", "").replace(",", "")))
        except ValueError:
            continue
    return out


def _strip_lines(text: str) -> list[str]:
    out = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _PLAN_LINE_PREFIX.sub("", line).strip()
        if line:
            out.append(line)
    return out


def extract_bw_actions_raw(text: str) -> tuple[str, ...]:
    """Extract actions allowing both canonical and arbitrary leading verbs."""
    acts: list[str] = []
    verb = (
        r"pick-up|put-down|stack|unstack|attack|succumb|overcome|broker|feast|"
        r"[a-z][a-z0-9_-]*"
    )
    pat = re.compile(
        rf"^({verb})\s+([A-Za-z0-9_-]+)(?:\s+([A-Za-z0-9_-]+))?\s*$",
        re.I,
    )
    for line in _strip_lines(text):
        norm = re.sub(r"\bblock\s+", "", line.strip(), flags=re.I).lower()
        m = pat.match(norm)
        if not m:
            continue
        parts = [m.group(1).lower(), m.group(2).lower()]
        if m.group(3):
            parts.append(m.group(3).lower())
        acts.append(" ".join(parts))
    return tuple(acts)


def invert_map(mapping: dict[str, str] | None) -> dict[str, str]:
    if not mapping:
        return {}
    out: dict[str, str] = {}
    for k, v in mapping.items():
        kk, vv = str(k).strip(), str(v).strip()
        if not vv:
            continue
        for key in {vv, vv.lower(), vv.upper(), vv.replace("_", "-"), vv.replace("-", "_")}:
            out[key] = kk
            out[key.lower()] = kk
            out[key.upper()] = kk
    return out


def apply_token_map(tok: str, inv: dict[str, str]) -> str:
    t = str(tok).strip()
    if t in inv:
        return inv[t]
    if t.lower() in inv:
        return inv[t.lower()]
    if t.upper() in inv:
        return inv[t.upper()]
    # Hub A / Item A style: try full string
    compact = re.sub(r"\s+", " ", t)
    if compact in inv:
        return inv[compact]
    if compact.lower() in inv:
        return inv[compact.lower()]
    return t


def rewrite_sp_path(path: tuple[str, ...], inv: dict[str, str]) -> tuple[str, ...]:
    if not path or not inv:
        return path
    out: list[str] = []
    for t in path:
        mapped = apply_token_map(t, inv)
        out.append(mapped if re.fullmatch(r"\d+", str(mapped)) else str(mapped).upper())
    return tuple(out)


def rewrite_wis(selected: frozenset[str], inv: dict[str, str]) -> frozenset[str]:
    if not selected or not inv:
        return selected
    return frozenset(apply_token_map(t, inv).upper() for t in selected)


def rewrite_bw_actions(
    actions: tuple[str, ...],
    action_inv: dict[str, str],
    entity_inv: dict[str, str],
) -> tuple[str, ...]:
    out: list[str] = []
    for act in actions:
        parts = act.split()
        if not parts:
            continue
        verb = parts[0]
        if action_inv:
            verb = apply_token_map(verb, action_inv).lower()
            # also try underscore/hyphen variants
            verb = action_inv.get(verb, verb)
        objs = [apply_token_map(o, entity_inv).lower() for o in parts[1:]]
        out.append(" ".join([verb] + objs))
    return tuple(out)


def jaccard_counter(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return float(inter) / float(union) if union else 0.0


def is_subsequence(needle: tuple, hay: tuple) -> bool:
    if not needle:
        return False
    n, h = len(needle), len(hay)
    if n > h:
        return False
    for i in range(h - n + 1):
        if hay[i : i + n] == needle:
            return True
    return False


def nums_close(a: float | None, b: float | None, tol: float = GSM_TOL) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < tol


def surface_matches_variant(fam: str, pid: str, response: str, var_gt: str) -> bool:
    """True if response equals variant gold in surface form (no reverse-map)."""
    if fam == "GSM":
        return nums_close(extract_gsm_final(response), extract_gsm_final(var_gt))
    if fam == "BW":
        ra, va = extract_bw_actions_raw(response), extract_bw_actions_raw(var_gt)
        return bool(ra) and bool(va) and ra == va
    sub = algo_subtype(pid)
    if sub == "SP":
        mp, vp = extract_sp_path(response), extract_sp_path(var_gt)
        return bool(mp) and bool(vp) and mp == vp
    if sub == "CC":
        _mc, mcoins = extract_cc(response)
        _vc, vcoins = extract_cc(var_gt)
        return bool(mcoins) and bool(vcoins) and mcoins == vcoins
    if sub == "WIS":
        ms, vs = extract_wis(response), extract_wis(var_gt)
        return bool(ms) and bool(vs) and ms == vs
    return str(response).strip() == str(var_gt).strip()


def is_surface_incorrect(fam: str, pid: str, response: str, var_gt: str) -> bool:
    return not surface_matches_variant(fam, pid, response, var_gt)


# ---------------------------------------------------------------------------
# Family structured gold + classification
# ---------------------------------------------------------------------------

def algo_subtype(pid: str) -> str:
    p = str(pid).upper()
    if p.startswith("SP"):
        return "SP"
    if p.startswith("CC"):
        return "CC"
    if p.startswith("WIS"):
        return "WIS"
    return "OTHER"


def structured_gold_algo(text: str, pid: str):
    sub = algo_subtype(pid)
    if sub == "SP":
        return ("SP", extract_sp_path(text))
    if sub == "CC":
        return ("CC", extract_cc(text))
    if sub == "WIS":
        return ("WIS", extract_wis(text))
    return ("RAW", str(text).strip())


def golds_differ(fam: str, pid: str, can_gt: str, var_gt: str) -> bool:
    if fam == "GSM":
        return not nums_close(extract_gsm_final(can_gt), extract_gsm_final(var_gt))
    if fam == "BW":
        return extract_bw_actions_raw(can_gt) != extract_bw_actions_raw(var_gt)
    # ALGO
    kind_c, sc = structured_gold_algo(can_gt, pid)
    kind_v, sv = structured_gold_algo(var_gt, pid)
    if kind_c != kind_v:
        return True
    return sc != sv


def _w3_maps(fam: str, row: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return (entity_inv label→id, action_inv renamed→canonical)."""
    notes = row.get("notes")
    entity = parse_entity_mapping_from_notes(notes) or {}
    action = parse_action_mapping_from_notes(notes) or {}
    blob = _parse_notes_json(notes)
    if not entity and isinstance(blob.get("entity_mapping"), dict):
        entity = {str(k): str(v) for k, v in blob["entity_mapping"].items()}
    if fam == "ALGO" and algo_subtype(str(row.get("problem_id", ""))) == "SP":
        try:
            entity = resolve_sp_node_mapping(
                row.get("difficulty_params") or "{}",
                notes=notes,
                problem_text=row.get("problem_text"),
            ) or entity
        except Exception:
            pass
    return invert_map(entity), invert_map(action)


def classify_error(
    fam: str,
    pid: str,
    variant: str,
    response: str,
    can_gt: str,
    var_gt: str,
    bank_variant_row: dict,
) -> tuple[str, str]:
    """Return (error_class, response_normalized)."""
    resp = str(response or "")
    entity_inv, action_inv = ({}, {})
    if variant == "W3":
        entity_inv, action_inv = _w3_maps(fam, bank_variant_row)

    if fam == "GSM":
        final = extract_gsm_final(resp)
        can_n = extract_gsm_final(can_gt)
        var_n = extract_gsm_final(var_gt)
        norm = "" if final is None else f"{final}"
        if nums_close(final, can_n) and not nums_close(final, var_n):
            return "INTRUSION", norm
        chain = extract_gsm_numbers(resp)
        if can_n is not None and any(nums_close(x, can_n) for x in chain):
            # canonical number appears in chain but final is not (or equals var)
            if not nums_close(final, can_n):
                return "PARTIAL_INTRUSION", norm
        return "OTHER_ERROR", norm

    if fam == "BW":
        raw_acts = extract_bw_actions_raw(resp)
        can_acts = extract_bw_actions_raw(can_gt)
        var_acts = extract_bw_actions_raw(var_gt)
        mapped = (
            rewrite_bw_actions(raw_acts, action_inv, entity_inv)
            if (action_inv or entity_inv)
            else raw_acts
        )
        norm = " | ".join(mapped) if mapped else " | ".join(raw_acts)
        # Do not label as intrusion if the raw response already equals variant gold.
        if raw_acts and var_acts and raw_acts == var_acts:
            return "OTHER_ERROR", norm
        for cand in (mapped, raw_acts):
            if cand and can_acts and cand == can_acts:
                return "INTRUSION", norm
        if can_acts and (
            is_subsequence(can_acts, mapped)
            or is_subsequence(can_acts, raw_acts)
            or jaccard_counter(Counter(mapped), Counter(can_acts)) > PARTIAL_THRESH
            or jaccard_counter(Counter(raw_acts), Counter(can_acts)) > PARTIAL_THRESH
        ):
            return "PARTIAL_INTRUSION", norm
        return "OTHER_ERROR", norm

    # ALGO
    sub = algo_subtype(pid)
    if sub == "SP":
        mp = extract_sp_path(resp)
        cp = extract_sp_path(can_gt)
        vp = extract_sp_path(var_gt)
        mapped = rewrite_sp_path(mp, entity_inv) if entity_inv else mp
        # After reverse map, normalize digit-like
        def _norm_path(p: tuple[str, ...]) -> tuple[str, ...]:
            out = []
            for t in p:
                tt = str(t).strip()
                if re.fullmatch(r"\d+", tt):
                    out.append(tt)
                else:
                    out.append(tt.upper())
            return tuple(out)

        mapped_n, cp_n, vp_n, mp_n = map(_norm_path, (mapped, cp, vp, mp))
        norm = " -> ".join(mapped_n) if mapped_n else " -> ".join(mp_n)
        # Raw already equals variant gold → not an intrusion (correct isomorphic answer).
        if mp_n and vp_n and mp_n == vp_n:
            return "OTHER_ERROR", norm
        for cand in (mapped_n, mp_n):
            if cand and cp_n and cand == cp_n:
                return "INTRUSION", norm
        if cp_n and (
            is_subsequence(cp_n, mapped_n)
            or is_subsequence(cp_n, mp_n)
            or jaccard_counter(Counter(mapped_n), Counter(cp_n)) > PARTIAL_THRESH
            or jaccard_counter(Counter(mp_n), Counter(cp_n)) > PARTIAL_THRESH
        ):
            return "PARTIAL_INTRUSION", norm
        return "OTHER_ERROR", norm

    if sub == "CC":
        mc, mcoins = extract_cc(resp)
        cc, ccoins = extract_cc(can_gt)
        vc, vcoins = extract_cc(var_gt)
        norm = f"Count:{mc}; Coins:{list(mcoins)}"
        if mcoins and vcoins and mcoins == vcoins:
            return "OTHER_ERROR", norm
        if mcoins and ccoins and mcoins == ccoins:
            return "INTRUSION", norm
        if ccoins and jaccard_counter(Counter(mcoins), Counter(ccoins)) > PARTIAL_THRESH:
            return "PARTIAL_INTRUSION", norm
        return "OTHER_ERROR", norm

    if sub == "WIS":
        ms = extract_wis(resp)
        cs = extract_wis(can_gt)
        vs = extract_wis(var_gt)
        mapped = rewrite_wis(ms, entity_inv) if entity_inv else ms
        norm = "{" + ", ".join(sorted(mapped if mapped else ms)) + "}"
        if ms and vs and ms == vs:
            return "OTHER_ERROR", norm
        for cand in (mapped, ms):
            if cand and cs and cand == cs:
                return "INTRUSION", norm
        if cs and (
            jaccard_counter(Counter(mapped), Counter(cs)) > PARTIAL_THRESH
            or jaccard_counter(Counter(ms), Counter(cs)) > PARTIAL_THRESH
        ):
            return "PARTIAL_INTRUSION", norm
        return "OTHER_ERROR", norm

    # fallback raw string
    norm = resp.strip()[:500]
    if norm and norm == str(can_gt).strip() and norm != str(var_gt).strip():
        return "INTRUSION", norm
    return "OTHER_ERROR", norm


def matches_intrusion_against_gold(
    fam: str,
    pid: str,
    variant: str,
    response: str,
    can_gt: str,
    var_gt: str,
    bank_variant_row: dict,
) -> bool:
    cls, _ = classify_error(fam, pid, variant, response, can_gt, var_gt, bank_variant_row)
    return cls == "INTRUSION"


# ---------------------------------------------------------------------------
# Load P1
# ---------------------------------------------------------------------------

def load_p1() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("*_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        name = path.name
        if name.startswith("ALGO_"):
            fam = "ALGO"
        elif name.startswith("BW_"):
            fam = "BW"
        elif name.startswith("GSM_"):
            fam = "GSM"
        else:
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        df = filter_excluded(df, family=fam)
        df["family"] = fam
        df["model_short"] = df["model"].map(PAPER_MODELS)
        df = df[df["model_short"].isin(PAPER_MODELS.values())].copy()
        df["variant"] = df["variant_type"].map(normalize_variant)
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
        text_col = "model_answer" if "model_answer" in df.columns else "raw_response"
        df["response"] = df[text_col].astype(str)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        parts.append(
            df[
                [
                    "family",
                    "problem_id",
                    "variant",
                    "model",
                    "model_short",
                    "response",
                    "ok",
                ]
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(
        ["family", "problem_id", "variant", "model_short"], keep="last"
    )


def load_contamination() -> pd.DataFrame:
    rows = []
    for fam, path in [
        ("GSM", RAW / "GSM_P3_contamination.csv"),
        ("BW", RAW / "BW_P3_contamination.csv"),
        ("ALGO", RAW / "ALGO_P3_contamination.csv"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df["family"] = fam
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
        score_col = (
            "contamination_score"
            if "contamination_score" in df.columns
            else "instance_contamination_score"
        )
        df["contamination_score"] = pd.to_numeric(df[score_col], errors="coerce")
        rows.append(df[["family", "problem_id", "contamination_score"]])
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        ["family", "problem_id"], keep="first"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_error_rows(
    p1: pd.DataFrame,
    banks: dict[str, pd.DataFrame],
    gold_idx: dict[tuple[str, str, str], str],
    cmap: dict[tuple[str, str], str],
) -> pd.DataFrame:
    rows = []
    for fam in ("ALGO", "BW", "GSM"):
        for variant in VARIANTS:
            sub = p1[(p1["family"] == fam) & (p1["variant"] == variant)]
            for _, r in sub.iterrows():
                pid = str(r["problem_id"])
                can_gt = gold_idx.get((fam, pid, "canonical"), "")
                var_gt = gold_idx.get((fam, pid, variant), "")
                if not can_gt or not var_gt:
                    continue
                if not golds_differ(fam, pid, can_gt, var_gt):
                    continue
                if not is_surface_incorrect(fam, pid, r["response"], var_gt):
                    continue
                brow = bank_row(banks, fam, pid, variant)
                cls, norm = classify_error(
                    fam, pid, variant, r["response"], can_gt, var_gt, brow
                )
                rows.append(
                    {
                        "family": fam,
                        "model": r["model_short"],
                        "variant": variant,
                        "problem_id": pid,
                        "error_class": cls,
                        "canonical_gold": can_gt,
                        "variant_gold": var_gt,
                        "response_normalized": norm,
                        "clone_family": clone_family_for(fam, pid, cmap),
                        "rescored_correct": bool(r["ok"]),
                        "response_raw": str(r["response"])[:2000],
                    }
                )
    return pd.DataFrame(rows)


def add_chance_baseline(errors: pd.DataFrame, p1: pd.DataFrame, banks, gold_idx) -> pd.DataFrame:
    """Per-row chance_i: fraction of other surface-incorrect answers matching this canonical gold."""
    if errors.empty:
        errors["chance_i"] = []
        return errors

    # Pool = other surface-incorrect responses in same family×model×variant
    # (rebuild from p1 + gold; avoid depending on errors alone for pool size).
    pools: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for _, r in p1[p1["variant"].isin(VARIANTS)].iterrows():
        fam, variant, model, pid = (
            r["family"],
            r["variant"],
            r["model_short"],
            str(r["problem_id"]),
        )
        var_gt = gold_idx.get((fam, pid, variant), "")
        if not var_gt:
            continue
        if not is_surface_incorrect(fam, pid, r["response"], var_gt):
            continue
        pools[(fam, model, variant)].append((pid, str(r["response"])))

    chance_vals = []
    for _, e in errors.iterrows():
        key = (e["family"], e["model"], e["variant"])
        pool = pools.get(key, [])
        others = [(pid, resp) for pid, resp in pool if pid != e["problem_id"]]
        if not others:
            chance_vals.append(float("nan"))
            continue
        focal_bank = bank_row(banks, e["family"], e["problem_id"], e["variant"])
        hits = 0
        for _pid, resp in others:
            if matches_intrusion_against_gold(
                e["family"],
                e["problem_id"],
                e["variant"],
                resp,
                e["canonical_gold"],
                e["variant_gold"],
                focal_bank,
            ):
                hits += 1
        chance_vals.append(hits / len(others))
    out = errors.copy()
    out["chance_i"] = chance_vals
    return out


def compute_rates(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, model, variant), g in errors.groupby(["family", "model", "variant"]):
        n = len(g)
        n_int = int((g["error_class"] == "INTRUSION").sum())
        n_part = int((g["error_class"] == "PARTIAL_INTRUSION").sum())
        n_other = int((g["error_class"] == "OTHER_ERROR").sum())
        rate = n_int / n if n else float("nan")
        flags = (g["error_class"] == "INTRUSION").astype(float).tolist()
        clusters = g["clone_family"].astype(str).tolist()
        lo, hi = cluster_bootstrap_ci(flags, clusters, n_resamples=N_BOOT_RATE, seed=SEED)
        chance_series = pd.to_numeric(g["chance_i"], errors="coerce")
        chance = float(chance_series.mean()) if chance_series.notna().any() else float("nan")
        rows.append(
            {
                "family": fam,
                "model": model,
                "variant": variant,
                "n_errors": n,
                "n_INTRUSION": n_int,
                "n_PARTIAL_INTRUSION": n_part,
                "n_OTHER_ERROR": n_other,
                "intrusion_rate": rate,
                "ci_low": lo,
                "ci_high": hi,
                "chance_baseline": chance,
                "intrusion_minus_chance": (rate - chance) if n and np.isfinite(chance) else float("nan"),
                "n_clusters": g["clone_family"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "variant", "model"]).reset_index(drop=True)


def compute_corr(errors: pd.DataFrame, contam: pd.DataFrame) -> pd.DataFrame:
    merged = errors.merge(contam, on=["family", "problem_id"], how="left")
    merged["is_intrusion"] = (merged["error_class"] == "INTRUSION").astype(float)
    rows = []
    for fam in ("ALGO", "BW", "GSM"):
        for model, g in merged[merged["family"] == fam].groupby("model"):
            sub = g.dropna(subset=["contamination_score"])
            if len(sub) < 3:
                rows.append(
                    {
                        "family": fam,
                        "model": model,
                        "n": int(len(sub)),
                        "n_clusters": int(sub["clone_family"].nunique()) if len(sub) else 0,
                        "spearman_rho": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "p_clustered": float("nan"),
                        "n_intrusions": int(sub["is_intrusion"].sum()) if len(sub) else 0,
                        "verdict": "insufficient_n",
                    }
                )
                continue
            res = cluster_bootstrap_assoc(
                sub["contamination_score"],
                sub["is_intrusion"],
                sub["clone_family"].astype(str).tolist(),
                kind="spearman",
                n_boot=N_BOOT_ASSOC,
                seed=SEED,
            )
            rho = res["estimate"]
            lo, hi = res["ci_low"], res["ci_high"]
            # Pre-registered reading: positive association expected under retrieval.
            if not np.isfinite(rho):
                verdict = "undefined"
            elif lo > 0:
                verdict = "positive_supported"
            elif hi < 0:
                verdict = "negative_supported"
            else:
                verdict = "null_ci_includes_0"
            rows.append(
                {
                    "family": fam,
                    "model": model,
                    "n": res["n"],
                    "n_clusters": res["n_clusters"],
                    "spearman_rho": rho,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_clustered": res["p_clustered"],
                    "n_intrusions": int(sub["is_intrusion"].sum()),
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "model"]).reset_index(drop=True)


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    _blocker_check()

    print("[prereg] C1_INTRUSION_PREREG.md / C1_INTRUSION_METHODS.md (frozen)")
    banks = load_banks()
    gold_idx = bank_gold_index(banks)
    cmap = _clone_map_all()
    p1 = load_p1()
    print(f"[load] P1 rows={len(p1)} models={sorted(p1.model_short.unique())}")

    errors = build_error_rows(p1, banks, gold_idx, cmap)
    print(f"[errors] eligible gold-diff incorrect rows={len(errors)}")
    if errors.empty:
        print("No eligible errors; writing empty outputs.")
        errors.to_csv(OUT_ERRORS, index=False)
        pd.DataFrame().to_csv(OUT_RATES, index=False)
        pd.DataFrame().to_csv(OUT_CORR, index=False)
        return

    errors = add_chance_baseline(errors, p1, banks, gold_idx)

    # Export schema without raw response dump (keep normalized)
    export_cols = [
        "family",
        "model",
        "variant",
        "problem_id",
        "error_class",
        "canonical_gold",
        "variant_gold",
        "response_normalized",
        "clone_family",
        "rescored_correct",
        "chance_i",
    ]
    errors[export_cols].to_csv(OUT_ERRORS, index=False)
    print(f"[write] {OUT_ERRORS} ({len(errors)} rows)")

    rates = compute_rates(errors)
    rates.to_csv(OUT_RATES, index=False)
    print(f"[write] {OUT_RATES} ({len(rates)} rows)")

    contam = load_contamination()
    corr = compute_corr(errors, contam)
    corr.to_csv(OUT_CORR, index=False)
    print(f"[write] {OUT_CORR} ({len(corr)} rows)")

    # Console summary (null is publishable)
    print("\n=== intrusion rates (INTRUSION / errors) vs chance ===")
    for _, r in rates.iterrows():
        print(
            f"  {r.family:4} {r.model:8} {r.variant}: "
            f"{int(r.n_INTRUSION)}/{int(r.n_errors)} = {r.intrusion_rate:.3f} "
            f"[{r.ci_low:.3f},{r.ci_high:.3f}] chance={r.chance_baseline:.4f}"
        )
    print("\n=== intrusion vs contamination (Spearman; null if CI∋0) ===")
    for _, r in corr.iterrows():
        print(
            f"  {r.family:4} {r.model:8}: rho={r.spearman_rho} "
            f"CI=[{r.ci_low},{r.ci_high}] p={r.p_clustered} → {r.verdict}"
        )


if __name__ == "__main__":
    main()
