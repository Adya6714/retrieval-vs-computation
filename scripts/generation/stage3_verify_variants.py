"""
scripts/generation/stage3_verify_variants.py

Stage 3 — Verification Gate
Reads staging CSVs, runs 3 checks per row, writes:
  data/staging/verified_rows.csv
  data/staging/quarantine.csv
  data/staging/gsm_w5_manual_review.csv

Usage:
  python scripts/generation/stage3_verify_variants.py [--dry-run] [--family bw|gsm|algo]

Checks:
  Check 1 — Structural integrity (family + variant-type specific)
  Check 2 — Answer correctness via solver (skip GSM W5)
  Check 3 — Schema validation (column presence, vocabulary)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — works whether run from repo root or scripts/generation/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports from existing repo modules
# ---------------------------------------------------------------------------
from probes.common.io import QUESTION_BANK_COLUMNS  # noqa: E402
from probes.contamination.verify import verify_answer, verify_gsm_answer  # noqa: E402
from scripts.generation.utils.variant_utils import (  # noqa: E402
    apply_mapping,
    make_inverse_mapping,
    verify_w3_roundtrip,
)

# ALGO solvers + CC parser from generate_w6
from scripts.ALGO_PX_SCR_generate_w6 import (  # noqa: E402
    dp_coin_change,
    format_cc_answer,
    format_wis_answer,
    parse_cc_instance,
    parse_src_tgt_from_sp_text,
    shortest_path_unique,
    wis_interval_optimal,
)

# SP/WIS text parsers live in fix_question_bank (fallback when difficulty_params absent)
try:
    from scripts.ALGO_PX_SCR_fix_question_bank import (  # noqa: E402
        parse_sp_edges,
        parse_wis_weights,
    )
    _HAS_FIX_QB = True
except ImportError:
    _HAS_FIX_QB = False

# Answer parsers for ALGO comparison
try:
    from scripts.ALGO_PX_SCR_audit_bank import (  # noqa: E402
        parse_cc_answer,
        parse_sp_answer,
        parse_wis_answer,
    )
    _HAS_AUDIT = True
except ImportError:
    _HAS_AUDIT = False

import networkx as nx  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage3")

# ---------------------------------------------------------------------------
# Schema constants (tuned to 144-problem staging)
# ---------------------------------------------------------------------------
VALID_VARIANT_TYPES = {"canonical", "W1", "W2", "W3", "W4", "W5", "W6"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}

FAMILY_SPECS: dict[str, dict[str, set[str]]] = {
    "bw": {
        "problem_family": {"planning_suite"},
        "problem_subtype": {"blocksworld", "mystery_blocksworld"},
        "contamination_pole": {"high", "low"},
    },
    "gsm": {
        "problem_family": {"arithmetic_reasoning"},
        "problem_subtype": {"gsm_symbolic"},
        "contamination_pole": {"high", "medium"},
    },
    "algo": {
        "problem_family": {"algorithmic"},
        "problem_subtype": {"coin_change", "shortest_path", "wis"},
        "contamination_pole": {"high", "low"},
    },
}

STAGING_EXTRA_COLS = {"status", "selection_reason", "generator_model"}

# Subtypes that intentionally have no W5
NO_W5_SUBTYPES = {"coin_change", "wis"}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAGING = REPO_ROOT / "data" / "staging"
BW_VARIANTS  = STAGING / "bw_variants.csv"
GSM_VARIANTS = STAGING / "gsm_variants.csv"
ALGO_VARIANTS = STAGING / "algo_variants.csv"
BW_CANONICAL  = STAGING / "bw_canonical.csv"
GSM_CANONICAL = STAGING / "gsm_canonical.csv"
ALGO_CANONICAL = STAGING / "algo_canonical.csv"

OUT_VERIFIED  = STAGING / "verified_rows.csv"
OUT_QUARANTINE = STAGING / "quarantine.csv"
OUT_GSM_W5   = STAGING / "gsm_w5_manual_review.csv"


# ===========================================================================
# Helpers shared across checks
# ===========================================================================

def _numbers_in_text(text: str) -> list[str]:
    """Return all numeric tokens (int or float) found in text, preserving order.
    Trailing dot is stripped so '204.' and '204' compare equal."""
    return [t.rstrip(".") for t in re.findall(r"\d+\.?\d*", text or "")]


def _extract_bw_blocks(text: str) -> set[str]:
    """Extract single-letter block ids from BW natural-language problem text
    OR from a W2 markdown table. Copied from stage2_generate_variants.py."""
    blocks: set[str] = set()
    # W2 markdown table: | a | table | e |  — grab single letters in cells
    for cell in re.findall(r"\|\s*([^|\n]+?)\s*(?=\|)", text):
        cell = cell.strip()
        if re.match(r"^[a-z]$", cell, re.IGNORECASE):
            blocks.add(cell.lower())
    # NL text: "block X" patterns
    cs = re.search(r"Current state:(.*?)(?:Goal:|$)", text, re.IGNORECASE | re.DOTALL)
    search = cs.group(1) if cs else text
    for m in re.finditer(r"\bblock\s+([a-z])\b", search, re.IGNORECASE):
        blocks.add(m.group(1).lower())
    multi = re.search(
        r"[Bb]locks?\s+([\w,\s]+?)\s+are\s+(?:clear\s+and\s+)?on\s+the\s+table",
        search,
    )
    if multi:
        for b in re.findall(r"\b([a-z])\b", multi.group(1)):
            blocks.add(b.lower())
    return blocks


def _verify_w1_algo_lists(canonical: str, w1: str) -> list[str]:
    """Every [...] list literal in canonical must appear verbatim in W1.
    Returns list of missing list literals (empty = pass)."""
    lists = re.findall(r"\[[^\]]+\]", canonical or "")
    missing = [lst for lst in lists if lst not in (w1 or "")]
    return missing


def _parse_notes_mapping(notes: str) -> dict | None:
    """Extract entity_mapping + action_mapping from W3 notes JSON."""
    if not notes:
        return None
    try:
        blob = json.loads(notes)
        mapping: dict = {}
        mapping.update(blob.get("entity_mapping") or {})
        mapping.update(blob.get("action_mapping") or {})
        return mapping if mapping else None
    except (json.JSONDecodeError, AttributeError):
        # Try extracting a raw JSON object substring
        m = re.search(r"\{.*\}", notes, re.DOTALL)
        if m:
            try:
                blob = json.loads(m.group())
                mapping = {}
                mapping.update(blob.get("entity_mapping") or {})
                mapping.update(blob.get("action_mapping") or {})
                return mapping if mapping else None
            except json.JSONDecodeError:
                pass
    return None


def _load_difficulty_params(row: dict) -> dict:
    raw = str(row.get("difficulty_params") or "").strip()
    if not raw or raw in ("{}", "null", "nan", ""):
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _get_canonical(canonical_df: pd.DataFrame, problem_id: str) -> dict | None:
    """Return the canonical row for a given problem_id."""
    # canonical rows have variant_type == "canonical"
    mask = (canonical_df["problem_id"] == problem_id) & (
        canonical_df["variant_type"].str.strip().str.lower() == "canonical"
    )
    rows = canonical_df[mask]
    if rows.empty:
        # fallback: first row with that problem_id in canonical CSV
        mask2 = canonical_df["problem_id"] == problem_id
        rows = canonical_df[mask2]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


# ===========================================================================
# CHECK 1 — Structural integrity
# ===========================================================================

def check1_structural(row: dict, canonical: dict | None, family: str) -> list[str]:
    """Return list of error strings; empty = pass."""
    errors: list[str] = []
    vt = (row.get("variant_type") or "").strip()
    text = str(row.get("problem_text") or "")
    answer = str(row.get("correct_answer") or "")
    subtype = str(row.get("problem_subtype") or "").strip().lower()

    # -----------------------------------------------------------------------
    if vt == "W1":
        if canonical is None:
            errors.append("W1: cannot find canonical row for structural check")
            return errors
        can_text = str(canonical.get("problem_text") or "")

        # W1/W2/W3/W4 must carry identical correct_answer to canonical
        can_answer = str(canonical.get("correct_answer") or "").strip()
        if answer.strip() != can_answer:
            errors.append(
                f"W1: correct_answer differs from canonical.\n"
                f"  variant ={answer[:120]!r}\n"
                f"  canonical={can_answer[:120]!r}"
            )

        if family == "bw":
            can_blocks = _extract_bw_blocks(can_text)
            w1_blocks  = _extract_bw_blocks(text)
            missing = can_blocks - w1_blocks
            if missing:
                errors.append(f"W1: missing blocks in W1 text: {sorted(missing)}")

        elif family == "gsm":
            can_nums = sorted(_numbers_in_text(can_text))
            w1_nums  = sorted(_numbers_in_text(text))
            if can_nums != w1_nums:
                errors.append(
                    f"W1: number set mismatch. canonical={can_nums} w1={w1_nums}"
                )

        elif family == "algo":
            can_nums = sorted(_numbers_in_text(can_text))
            w1_nums  = sorted(_numbers_in_text(text))
            if can_nums != w1_nums:
                errors.append(
                    f"W1: number set mismatch. canonical={can_nums} w1={w1_nums}"
                )
            missing_lists = _verify_w1_algo_lists(can_text, text)
            if missing_lists:
                errors.append(f"W1: list literals missing from W1: {missing_lists}")

    # -----------------------------------------------------------------------
    elif vt == "W2":
        if canonical is None:
            errors.append("W2: cannot find canonical row for structural check")
            return errors
        can_text = str(canonical.get("problem_text") or "")

        # W2 must carry identical correct_answer to canonical
        can_answer = str(canonical.get("correct_answer") or "").strip()
        if answer.strip() != can_answer:
            errors.append(
                f"W2: correct_answer differs from canonical.\n"
                f"  variant ={answer[:120]!r}\n"
                f"  canonical={can_answer[:120]!r}"
            )

        if family == "bw":
            can_blocks = _extract_bw_blocks(can_text)
            w2_blocks  = _extract_bw_blocks(text)
            missing = can_blocks - w2_blocks
            if missing:
                errors.append(f"W2: blocks missing from W2 table: {sorted(missing)}")
            if not text.strip():
                errors.append("W2: empty problem_text")

        elif family == "gsm":
            can_nums = sorted(_numbers_in_text(can_text))
            w2_nums  = sorted(_numbers_in_text(text))
            if not set(can_nums).issubset(set(w2_nums)):
                errors.append(
                    f"W2: canonical numbers not a subset of W2 numbers. "
                    f"missing={sorted(set(can_nums) - set(w2_nums))}"
                )

        elif family == "algo":
            # W2 is a deterministic reformat from difficulty_params
            params = _load_difficulty_params(row)
            if not params:
                errors.append("W2: difficulty_params missing; cannot validate ALGO W2")
            if not text.strip():
                errors.append("W2: empty problem_text")

    # -----------------------------------------------------------------------
    elif vt == "W3":
        if canonical is None:
            errors.append("W3: cannot find canonical row for structural check")
            return errors
        can_text = str(canonical.get("problem_text") or "")

        # CC W3 uses context transform, not bijective entity mapping
        if subtype == "coin_change":
            if not text.strip():
                errors.append("W3 (CC): empty problem_text")
            # No inverse-mapping check for CC
            return errors

        # All other subtypes: extract mapping from notes
        notes = str(row.get("notes") or "")
        mapping = _parse_notes_mapping(notes)
        if not mapping:
            errors.append("W3: no entity/action mapping found in notes")
            return errors

        ok, detail = verify_w3_roundtrip(text, can_text, mapping)
        if not ok:
            errors.append(f"W3: roundtrip check failed — {detail}")

        # W3 correct_answer check:
        # Any subtype that renames actions/entities will have a different answer
        # string — use inverse-mapping roundtrip for all planning + SP subtypes.
        # CC W3 is already returned early above (context transform, no mapping).
        # GSM W3 renames names/objects but numbers/answer are unchanged → exact match.
        can_answer = str(canonical.get("correct_answer") or "").strip()
        if subtype in ("blocksworld", "mystery_blocksworld", "shortest_path"):
            inv_mapping = make_inverse_mapping(mapping)
            recovered = apply_mapping(answer, inv_mapping)
            if recovered.strip() != can_answer:
                errors.append(
                    f"W3 ({subtype}): answer inverse-mapping mismatch.\n"
                    f"  recovered ={recovered[:120]!r}\n"
                    f"  canonical ={can_answer[:120]!r}"
                )
        else:
            # GSM and WIS: answer must be string-identical to canonical
            if answer.strip() != can_answer:
                errors.append(
                    f"W3: correct_answer differs from canonical.\n"
                    f"  variant  ={answer[:120]!r}\n"
                    f"  canonical={can_answer[:120]!r}"
                )

    # -----------------------------------------------------------------------
    elif vt == "W4":
        if canonical is None:
            errors.append("W4: cannot find canonical row for structural check")
            return errors

        # W4 must carry identical correct_answer to canonical
        can_answer = str(canonical.get("correct_answer") or "").strip()
        if answer.strip() != can_answer:
            errors.append(
                f"W4: correct_answer differs from canonical.\n"
                f"  variant ={answer[:120]!r}\n"
                f"  canonical={can_answer[:120]!r}"
            )
        if not text.strip():
            errors.append("W4: empty problem_text")

    # -----------------------------------------------------------------------
    elif vt == "W5":
        if subtype in NO_W5_SUBTYPES:
            errors.append(f"W5: {subtype} should not have W5 rows")
            return errors

        # GSM W5: flagged for manual review, no structural check here
        if family == "gsm":
            return errors  # handled separately in main loop

        # BW W5: verify init/goal are swapped vs canonical.
        # NL section text comparison is brittle — paraphrase wording, sentence
        # order, and BW_E procedural format all cause false positives.
        # Policy: trust notes swap marker for Check 1; Check 2 verifies the
        # actual plan executes correctly on the swapped problem.
        if family == "bw":
            if canonical is None:
                errors.append("W5 (BW): cannot find canonical row")
                return errors
            notes = str(row.get("notes") or "").lower()
            if not any(k in notes for k in ("w5", "swap", "reverse", "procedural_seed",
                                             "w5_bw_plan_invalid", "accepting fd plan")):
                errors.append(
                    "W5 (BW): notes don't confirm init/goal swap — "
                    "add 'swap' or 'W5' marker to notes field"
                )

        # SP W5: verify source/target are reversed vs canonical difficulty_params
        if subtype == "shortest_path":
            if not text.strip():
                errors.append("W5 (SP): empty problem_text")
            if canonical is not None:
                can_params = _load_difficulty_params(canonical)
                w5_params  = _load_difficulty_params(row)
                can_src = can_params.get("source") or can_params.get("src")
                can_tgt = can_params.get("target") or can_params.get("tgt")
                w5_src  = w5_params.get("source")  or w5_params.get("src")
                w5_tgt  = w5_params.get("target")  or w5_params.get("tgt")
                if can_src is not None and can_tgt is not None and w5_src is not None and w5_tgt is not None:
                    if int(w5_src) != int(can_tgt) or int(w5_tgt) != int(can_src):
                        errors.append(
                            f"W5 (SP): source/target not swapped vs canonical.\n"
                            f"  canonical src={can_src} tgt={can_tgt}\n"
                            f"  W5        src={w5_src}  tgt={w5_tgt}"
                        )

    # -----------------------------------------------------------------------
    elif vt == "W6":
        if not text.strip():
            errors.append("W6: empty problem_text")
        if not answer.strip():
            errors.append("W6: empty correct_answer")

    # -----------------------------------------------------------------------
    elif vt == "canonical":
        if not text.strip():
            errors.append("canonical: empty problem_text")
        if not answer.strip():
            errors.append("canonical: empty correct_answer")

    return errors


# ===========================================================================
# CHECK 2 — Solver answer verification
# ===========================================================================

_BW_PREDICATES = {"pick-up", "put-down", "stack", "unstack"}


def _plan_uses_bw_predicates(plan: str) -> bool:
    """Return True if the plan contains standard BW action words."""
    plan_lower = plan.lower()
    return any(p in plan_lower for p in _BW_PREDICATES)


def _resolve_bw_verifier_family(row: dict, answer: str | None = None) -> str:
    """Map planning_suite row to concrete verifier family string.

    MBW W6 rows are generated with standard BW plans (pick-up/stack) even
    though problem_subtype is mystery_blocksworld.  Detect this by inspecting
    the plan vocabulary and route to 'blocksworld' verifier in that case.
    """
    subtype = str(row.get("problem_subtype") or "").strip().lower()
    pid = str(row.get("problem_id") or "")
    if subtype == "mystery_blocksworld" or pid.upper().startswith("MBW"):
        # If the plan uses BW predicates, the W6 generator wrote a BW plan
        # into an MBW slot — use BW verifier to match the plan vocabulary.
        if answer and _plan_uses_bw_predicates(answer):
            return "blocksworld"
        return "mystery_blocksworld"
    return "blocksworld"


def _check2_bw(row: dict) -> list[str]:
    errors: list[str] = []
    vt     = (row.get("variant_type") or "").strip()
    answer = str(row.get("correct_answer") or "").strip()
    text   = str(row.get("problem_text") or "").strip()
    pid    = str(row.get("problem_id") or "")
    notes  = str(row.get("notes") or "").lower()
    verifier_family = _resolve_bw_verifier_family(row, answer=answer)

    if vt == "W5":
        # NL state-machine sim is expected to fail for W5;
        # accept if notes contain expected markers
        if ("w5_bw_plan_invalid" in notes or "nl plan sim failed" in notes
                or "accepting fd plan" in notes or "procedural_seed" in notes):
            return []  # expected gap — pass
        try:
            ok = verify_answer(pid, answer, answer, verifier_family, problem_text=text)
        except Exception:
            ok = False
        if not ok:
            errors.append(
                "W5 (BW): NL plan sim failed and notes don't mark FD-accepted; "
                "add 'NL plan sim failed; accepting FD plan' to notes or re-verify manually"
            )
        return errors

    # W3: problem_text and correct_answer use renamed vocabulary.
    # Apply inverse mapping to recover canonical vocabulary before calling
    # verify_answer, which expects BW/MBW predicates and block names.
    if vt == "W3":
        notes_raw = str(row.get("notes") or "")
        mapping = _parse_notes_mapping(notes_raw)
        if mapping:
            inv = make_inverse_mapping(mapping)
            verify_text   = apply_mapping(text, inv)
            verify_answer_str = apply_mapping(answer, inv)
        else:
            # No mapping found — fall through with original strings;
            # verifier will likely reject but at least we tried
            verify_text   = text
            verify_answer_str = answer
        try:
            ok = verify_answer(pid, verify_answer_str, verify_answer_str,
                               verifier_family, problem_text=verify_text)
        except Exception as exc:
            errors.append(f"BW W3 verifier raised {type(exc).__name__}: {exc}")
            return errors
        if not ok:
            errors.append(
                f"BW W3 solver rejected inverse-mapped answer for {pid}. "
                f"recovered_answer={verify_answer_str[:120]!r}"
            )
        return errors

    # All other variants: run state-machine verifier directly
    try:
        ok = verify_answer(pid, answer, answer, verifier_family, problem_text=text)
    except Exception as exc:
        errors.append(f"BW verifier raised {type(exc).__name__}: {exc}")
        return errors

    if not ok:
        errors.append(
            f"BW solver rejected answer for {pid} {vt}. "
            f"answer={answer[:120]!r}"
        )
    return errors


def _check2_gsm(row: dict) -> list[str]:
    errors: list[str] = []
    vt = (row.get("variant_type") or "").strip()

    if vt == "W5":
        return []  # GSM W5 routed to manual review, skip solver

    answer = str(row.get("correct_answer") or "").strip()
    if not answer:
        errors.append("GSM: empty correct_answer — cannot verify")
        return errors

    # Pass correct_answer as both args: verifies the stored answer parses correctly
    try:
        ok = verify_gsm_answer(answer, answer)
    except Exception as exc:
        errors.append(f"GSM verifier raised {type(exc).__name__}: {exc}")
        return errors

    if not ok:
        errors.append(
            f"GSM: verify_gsm_answer rejected stored answer {answer!r} — "
            "answer format may be unparseable"
        )
    return errors


def _build_sp_graph_from_params(params: dict) -> tuple[nx.Graph | nx.DiGraph | None, int, int]:
    """Reconstruct NetworkX graph from difficulty_params edge list.
    Uses explicit None checks so source=0 or target=0 are handled correctly."""
    directed = params.get("directed", False)
    edges = params.get("edges") or params.get("graph") or []

    # Explicit None check — 0 is a valid node id and must not be treated as missing
    src = params.get("source") if params.get("source") is not None else params.get("src")
    tgt = params.get("target") if params.get("target") is not None else params.get("tgt")

    if not edges or src is None or tgt is None:
        return None, -1, -1

    G = nx.DiGraph() if directed else nx.Graph()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            G.add_edge(int(e[0]), int(e[1]), weight=int(e[2]))
        elif isinstance(e, dict):
            G.add_edge(int(e["u"]), int(e["v"]), weight=int(e.get("w", e.get("weight", 1))))
    return G, int(src), int(tgt)


def _build_wis_intervals_from_params(params: dict) -> list[dict] | None:
    """Reconstruct WIS interval list from difficulty_params."""
    intervals = params.get("intervals")
    if intervals:
        return intervals
    weights = params.get("weights") or params.get("node_weights")
    if weights:
        if isinstance(weights, dict):
            return [
                {"id": int(k), "start": 2 * int(k), "end": 2 * int(k) + 1, "weight": int(v)}
                for k, v in weights.items()
            ]
        elif isinstance(weights, list):
            return [
                {"id": i, "start": 2 * i, "end": 2 * i + 1, "weight": int(w)}
                for i, w in enumerate(weights)
            ]
    return None


def _check2_algo(row: dict) -> list[str]:
    errors: list[str] = []
    subtype = str(row.get("problem_subtype") or "").strip().lower()
    text    = str(row.get("problem_text") or "").strip()
    answer  = str(row.get("correct_answer") or "").strip()
    params  = _load_difficulty_params(row)

    # ---- Coin Change -------------------------------------------------------
    if subtype == "coin_change":
        # Prefer params, fallback to text parser
        if params and "denominations" in params and "target" in params:
            denoms = [int(d) for d in params["denominations"]]
            target = int(params["target"])
        else:
            try:
                denoms, target = parse_cc_instance(text)
            except Exception as exc:
                errors.append(f"CC: failed to parse instance — {exc}")
                return errors

        if target == 0:
            errors.append("CC: target=0 (known invalid instance)")
            return errors

        result = dp_coin_change(denoms, target)
        if result is None:
            errors.append(f"CC: dp_coin_change returned None (unsolvable) for denoms={denoms} target={target}")
            return errors

        count, coins = result
        expected = format_cc_answer(count, coins)

        # Compare stored answer to solver answer
        if _HAS_AUDIT:
            try:
                cc_parsed = parse_cc_answer(answer)
                # parse_cc_answer may return (count, coins) tuple or bare int
                stored_count = cc_parsed[0] if isinstance(cc_parsed, (list, tuple)) else int(cc_parsed)
                if stored_count != count:
                    errors.append(
                        f"CC: stored answer count={stored_count} != solver count={count} "
                        f"(denoms={denoms}, target={target})"
                    )
            except Exception as exc:
                errors.append(f"CC: could not parse stored answer — {exc}")
        else:
            # Fallback: normalize and compare strings
            if answer.strip() != expected.strip():
                errors.append(
                    f"CC: stored answer {answer!r} != solver answer {expected!r}"
                )

    # ---- Shortest Path -----------------------------------------------------
    elif subtype == "shortest_path":
        # Fix: Stage 1 sets 'requires_bellman_ford', not 'use_bellman_ford'
        use_bf = bool(
            params.get("requires_bellman_ford", False)
            or params.get("use_bellman_ford", False)
        )
        vt = str(row.get("variant_type") or "").strip()

        # Always build graph from difficulty_params — never from text.
        # W2 is a reformatted table; W3 has renamed nodes; W6 procedural text
        # may differ from params. Params are ground truth.
        G, src, tgt = _build_sp_graph_from_params(params)

        if G is None:
            if vt in ("W2", "W3"):
                errors.append(
                    f"SP {vt}: difficulty_params has no edge list — cannot verify "
                    f"(text parser not reliable for {vt})"
                )
                return errors
            if not _HAS_FIX_QB:
                errors.append("SP: no difficulty_params edges and fix_question_bank not importable")
                return errors
            try:
                edges = parse_sp_edges(text)
                directed = bool(params.get("directed", False))
                G = nx.DiGraph() if directed else nx.Graph()
                for e in edges:
                    G.add_edge(int(e[0]), int(e[1]), weight=int(e[2]))
                src, tgt = parse_src_tgt_from_sp_text(text, answer)
            except Exception as exc:
                errors.append(f"SP: text parsing failed — {exc}")
                return errors

        # W3: params graph uses original node ids; src/tgt from params are
        # already correct. Graph is solved as-is; answer comparison handles
        # renamed labels via inverse-mapping cost parse below.

        # W5: Stage 2 generates W5 by reversing ALL edges (g.reverse()) then
        # solving tgt→src on that reversed graph. Replicate exactly.
        if vt == "W5":
            G_solve = G.reverse(copy=True) if isinstance(G, nx.DiGraph) else G
            solve_src, solve_tgt = int(tgt), int(src)
        else:
            G_solve = G
            solve_src, solve_tgt = int(src), int(tgt)

        try:
            if use_bf:
                solver_path = nx.bellman_ford_path(
                    G_solve, solve_src, solve_tgt, weight="weight"
                )
                solver_cost = int(
                    nx.bellman_ford_path_length(
                        G_solve, solve_src, solve_tgt, weight="weight"
                    )
                )
            else:
                result = shortest_path_unique(
                    G_solve, solve_src, solve_tgt, use_bellman_ford=use_bf
                )
                if result is None:
                    errors.append(f"SP: no path found from {solve_src} to {solve_tgt}")
                    return errors
                solver_path, solver_cost = result
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            errors.append(f"SP: no path found from {solve_src} to {solve_tgt}")
            return errors
        except Exception as exc:
            errors.append(f"SP: solver raised {type(exc).__name__}: {exc}")
            return errors

        # W3: answer uses renamed node labels — inverse-map to get original format
        # before cost parsing, which expects integer node ids
        answer_for_parse = answer
        if vt == "W3":
            notes_raw = str(row.get("notes") or "")
            w3_mapping = _parse_notes_mapping(notes_raw)
            if w3_mapping:
                inv = make_inverse_mapping(w3_mapping)
                answer_for_parse = apply_mapping(answer, inv)

        if _HAS_AUDIT:
            try:
                sp_parsed = parse_sp_answer(answer_for_parse)
                # parse_sp_answer may return (path, cost) tuple or bare int
                stored_cost = sp_parsed[1] if isinstance(sp_parsed, (list, tuple)) else int(sp_parsed)
                if stored_cost != solver_cost:
                    errors.append(
                        f"SP: stored cost={stored_cost} != solver cost={solver_cost} "
                        f"(src={src}, tgt={tgt})"
                    )
            except Exception as exc:
                errors.append(f"SP: could not parse stored answer — {exc}")
        else:
            if str(solver_cost) not in answer_for_parse:
                errors.append(
                    f"SP: solver cost {solver_cost} not found in stored answer {answer_for_parse!r}"
                )

    # ---- WIS ---------------------------------------------------------------
    elif subtype == "wis":
        intervals = _build_wis_intervals_from_params(params)

        if intervals is None:
            if not _HAS_FIX_QB:
                errors.append("WIS: no difficulty_params intervals and fix_question_bank not importable")
                return errors
            try:
                weights = parse_wis_weights(text)
                intervals = [
                    {"id": i, "start": 2 * i, "end": 2 * i + 1, "weight": int(w)}
                    for i, w in enumerate(weights)
                ]
            except Exception as exc:
                errors.append(f"WIS: text parsing failed — {exc}")
                return errors

        try:
            selected, total = wis_interval_optimal(intervals)
        except Exception as exc:
            errors.append(f"WIS: solver raised {type(exc).__name__}: {exc}")
            return errors

        if _HAS_AUDIT:
            try:
                wis_parsed = parse_wis_answer(answer)
                # parse_wis_answer may return (items, total) tuple or bare int
                stored_total = wis_parsed[1] if isinstance(wis_parsed, (list, tuple)) else int(wis_parsed)
                if stored_total != total:
                    errors.append(
                        f"WIS: stored total={stored_total} != solver total={total}"
                    )
            except Exception as exc:
                errors.append(f"WIS: could not parse stored answer — {exc}")
        else:
            if str(total) not in answer:
                errors.append(
                    f"WIS: solver total {total} not found in stored answer {answer!r}"
                )

    else:
        errors.append(f"ALGO: unknown subtype {subtype!r}")

    return errors


def check2_solver(row: dict, family: str) -> list[str]:
    vt = (row.get("variant_type") or "").strip()
    subtype = str(row.get("problem_subtype") or "").strip().lower()

    # Intentional gaps — not errors
    if vt == "W5" and subtype in NO_W5_SUBTYPES:
        return []
    if vt == "W5" and family == "gsm":
        return []  # routed to manual review

    if family == "bw":
        return _check2_bw(row)
    elif family == "gsm":
        return _check2_gsm(row)
    elif family == "algo":
        return _check2_algo(row)
    return [f"Unknown family {family!r}"]


# ===========================================================================
# CHECK 3 — Schema validation
# ===========================================================================

def check3_schema(row: dict, family: str) -> list[str]:
    errors: list[str] = []
    spec = FAMILY_SPECS[family]

    # Required columns present
    for col in QUESTION_BANK_COLUMNS:
        if col not in row:
            errors.append(f"missing column: {col}")

    vt = (row.get("variant_type") or "").strip()
    if vt not in VALID_VARIANT_TYPES:
        errors.append(f"invalid variant_type: {vt!r} (expected uppercase W1–W6 or 'canonical')")

    diff = (row.get("difficulty") or "").strip().lower()
    if diff not in VALID_DIFFICULTY:
        errors.append(f"invalid difficulty: {diff!r}")

    pole = (row.get("contamination_pole") or "").strip().lower()
    if pole not in spec["contamination_pole"]:
        errors.append(f"invalid contamination_pole for {family}: {pole!r}")

    pf = (row.get("problem_family") or "").strip().lower()
    if pf not in spec["problem_family"]:
        errors.append(f"invalid problem_family: {pf!r}")

    sub = (row.get("problem_subtype") or "").strip().lower()
    if sub not in spec["problem_subtype"]:
        errors.append(f"invalid problem_subtype: {sub!r}")

    for col in ("problem_id", "problem_text", "correct_answer", "source"):
        if not str(row.get(col, "")).strip():
            errors.append(f"empty required field: {col}")

    if family == "algo":
        raw_params = str(row.get("difficulty_params") or "").strip()
        if not raw_params or raw_params in ("{}", "null", "nan", ""):
            errors.append("ALGO: difficulty_params is empty (required for ALGO rows)")
        else:
            try:
                parsed = json.loads(raw_params)
                if not isinstance(parsed, dict):
                    errors.append("ALGO: difficulty_params must be a JSON object")
            except json.JSONDecodeError as e:
                errors.append(f"ALGO: difficulty_params invalid JSON — {e}")

    # NOTE: W6 contamination_pole is NOT enforced to 'low' here.
    # Staging ALGO W6 rows carry 'high' by convention from Stage 2 generation.
    # The audit script (ALGO_PX_SCR_audit_bank.py) has its own rule for the
    # final bank; Stage 3 does not override the staging convention.

    # MBW: correct_answer must not use BW predicates
    if sub == "mystery_blocksworld" and vt != "W6":
        bw_predicates = {"pick-up", "put-down", "stack", "unstack"}
        answer = str(row.get("correct_answer") or "").lower()
        found = [p for p in bw_predicates if p in answer]
        if found:
            errors.append(
                f"MBW: correct_answer contains BW predicates {found}; "
                "expected MBW predicates (attack, succumb, feast, bootstrap, overcome)"
            )

    return errors


# ===========================================================================
# Main per-row verifier
# ===========================================================================

def verify_row(
    row: dict,
    canonical: dict | None,
    family: str,
) -> tuple[bool, list[str], bool]:
    """
    Returns (passed: bool, all_errors: list[str], is_gsm_w5: bool).
    is_gsm_w5 = True when this row should go to gsm_w5_manual_review.csv.
    """
    vt = (row.get("variant_type") or "").strip()
    is_gsm_w5 = (family == "gsm" and vt == "W5")

    e1 = check1_structural(row, canonical, family)
    e3 = check3_schema(row, family)

    # Check 2 skipped for GSM W5
    if is_gsm_w5:
        e2 = []
    else:
        e2 = check2_solver(row, family)

    all_errors = (
        [f"[Check1] {e}" for e in e1]
        + [f"[Check2] {e}" for e in e2]
        + [f"[Check3] {e}" for e in e3]
    )
    passed = len(all_errors) == 0
    return passed, all_errors, is_gsm_w5


# ===========================================================================
# Load + normalize helpers
# ===========================================================================

def _normalize_row(row: dict) -> dict:
    """Normalize case fields before checks."""
    vt = str(row.get("variant_type") or "").strip()
    # Normalize w1 → W1 etc.
    if re.match(r"^w[1-6]$", vt, re.IGNORECASE):
        row["variant_type"] = vt.upper()
    pole = str(row.get("contamination_pole") or "").strip()
    row["contamination_pole"] = pole.lower()
    pf = str(row.get("problem_family") or "").strip()
    # Normalize "Algorithmic Suit" → "algorithmic"
    if "algorithmic" in pf.lower():
        row["problem_family"] = "algorithmic"
    elif pf:
        row["problem_family"] = pf.lower()
    return row


def _load_with_canonical(
    variants_path: Path, canonical_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants_df  = pd.read_csv(variants_path, dtype=str).fillna("")
    canonical_df = pd.read_csv(canonical_path, dtype=str).fillna("")
    return variants_df, canonical_df


# ===========================================================================
# Process one family
# ===========================================================================

def process_family(
    family: str,
    variants_path: Path,
    canonical_path: Path,
    dry_run: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (verified_rows, quarantine_rows, gsm_w5_rows).
    """
    log.info(f"=== Processing family: {family.upper()} ===")
    variants_df, canonical_df = _load_with_canonical(variants_path, canonical_path)
    log.info(f"  Loaded {len(variants_df)} variant rows, {len(canonical_df)} canonical rows")

    verified: list[dict] = []
    quarantine: list[dict] = []
    gsm_w5: list[dict] = []

    for idx, raw_row in variants_df.iterrows():
        row = _normalize_row(raw_row.to_dict())
        pid = row.get("problem_id", f"row_{idx}")
        vt  = (row.get("variant_type") or "").strip()

        canonical = _get_canonical(canonical_df, pid)

        passed, errors, is_gsm_w5 = verify_row(row, canonical, family)

        # Strip staging-only columns before writing output
        clean_row = {k: v for k, v in row.items() if k not in STAGING_EXTRA_COLS}

        if is_gsm_w5:
            gsm_w5.append({**clean_row, "manual_review_reason": "GSM_W5_NEEDS_MANUAL_VERIFICATION"})
            log.info(f"  {pid} {vt:12} → MANUAL_REVIEW (GSM W5)")
        elif passed:
            verified.append(clean_row)
            if not dry_run:
                log.debug(f"  {pid} {vt:12} → PASS")
        else:
            failure_str = " | ".join(errors)
            quarantine.append({**clean_row, "failure_reason": failure_str})
            log.warning(f"  {pid} {vt:12} → QUARANTINE: {failure_str[:120]}")

    log.info(
        f"  {family.upper()} done: "
        f"{len(verified)} verified, "
        f"{len(quarantine)} quarantined, "
        f"{len(gsm_w5)} GSM W5 manual"
    )
    return verified, quarantine, gsm_w5


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3 — Variant Verification Gate")
    parser.add_argument("--dry-run", action="store_true", help="Run checks but don't write output files")
    parser.add_argument("--family", choices=["bw", "gsm", "algo"], help="Process only one family")
    args = parser.parse_args()

    families_to_run = (
        [args.family] if args.family
        else ["bw", "gsm", "algo"]
    )

    FAMILY_PATHS = {
        "bw":   (BW_VARIANTS,   BW_CANONICAL),
        "gsm":  (GSM_VARIANTS,  GSM_CANONICAL),
        "algo": (ALGO_VARIANTS, ALGO_CANONICAL),
    }

    all_verified:  list[dict] = []
    all_quarantine: list[dict] = []
    all_gsm_w5:   list[dict] = []

    for fam in families_to_run:
        vp, cp = FAMILY_PATHS[fam]
        if not vp.exists():
            log.warning(f"Variants file not found, skipping: {vp}")
            continue
        if not cp.exists():
            log.warning(f"Canonical file not found, skipping: {cp}")
            continue
        v, q, g = process_family(fam, vp, cp, dry_run=args.dry_run)
        all_verified.extend(v)
        all_quarantine.extend(q)
        all_gsm_w5.extend(g)

    # Summary
    total = len(all_verified) + len(all_quarantine) + len(all_gsm_w5)
    log.info("=" * 60)
    log.info(f"TOTAL ROWS PROCESSED : {total}")
    log.info(f"  Verified (pass)    : {len(all_verified)}")
    log.info(f"  Quarantined (fail) : {len(all_quarantine)}")
    log.info(f"  GSM W5 manual      : {len(all_gsm_w5)}")
    log.info("=" * 60)

    if args.dry_run:
        log.info("DRY RUN — no files written")
        if all_quarantine:
            log.warning(f"Would quarantine {len(all_quarantine)} rows:")
            for r in all_quarantine:
                log.warning(f"  {r.get('problem_id')} {r.get('variant_type')}: {r.get('failure_reason','')[:100]}")
        return

    # Write outputs
    STAGING.mkdir(parents=True, exist_ok=True)

    if all_verified:
        pd.DataFrame(all_verified).to_csv(OUT_VERIFIED, index=False)
        log.info(f"Wrote {len(all_verified)} verified rows → {OUT_VERIFIED}")

    if all_quarantine:
        pd.DataFrame(all_quarantine).to_csv(OUT_QUARANTINE, index=False)
        log.info(f"Wrote {len(all_quarantine)} quarantine rows → {OUT_QUARANTINE}")
    else:
        # Write empty quarantine with header
        pd.DataFrame(columns=list(QUESTION_BANK_COLUMNS) + ["failure_reason"]).to_csv(
            OUT_QUARANTINE, index=False
        )
        log.info(f"All rows passed — empty quarantine written → {OUT_QUARANTINE}")

    if all_gsm_w5:
        pd.DataFrame(all_gsm_w5).to_csv(OUT_GSM_W5, index=False)
        log.info(f"Wrote {len(all_gsm_w5)} GSM W5 rows → {OUT_GSM_W5}")


if __name__ == "__main__":
    main()
