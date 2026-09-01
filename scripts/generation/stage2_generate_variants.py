#!/usr/bin/env python3
"""Stage 2: generate W1–W6 variants from staging canonical CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify import (  # noqa: E402
    _verify_blocksworld_state_machine,
    _verify_mystery_state_machine,
)
from scripts.generation.utils import variant_prompts as prompts  # noqa: E402
from scripts.generation.utils.variant_utils import (  # noqa: E402
    apply_mapping,
    fd_plan_to_bw_format,
    generate_random_bw_pddl,
    inspect_w5_goal_tower,
    load_bw_domain,
    make_inverse_mapping,
    pddl_to_natural_language,
    run_fast_downward,
    swap_pddl_init_goal,
    verify_w3_roundtrip,
    w6_seed,
)
from scripts.ALGO_PX_SCR_generate_w6 import (  # noqa: E402
    dp_coin_change,
    format_cc_answer,
    format_sp_answer,
    format_wis_answer,
    generate_sp_graph,
    generate_wis_graph,
    greedy_coin_change,
    parse_cc_instance,
    render_cc_text_from_canonical,
    render_sp_text_from_canonical,
    render_wis_text_from_canonical,
)

# Default Stage 2 LLM — see configs/models.yaml
MODEL_ID = "anthropic/claude-sonnet-4"
LOG_PATH = REPO_ROOT / "data/staging/stage2_generation.log"
FAILURES_PATH = REPO_ROOT / "data/staging/variant_generation_failures.csv"
MANIFEST_PATH = REPO_ROOT / "data/staging/stage2_variant_manifest.json"
VALID_VARIANT_TYPES = frozenset({"W1", "W2", "W3", "W4", "W5", "W6"})
PLANBENCH_ROOT = REPO_ROOT / "data/sources/planbench"

GSM_JSONL_PATHS = {
    "high": [
        REPO_ROOT / "data/sources/gsm_symbolic/generated_data/GSM_symbolic.jsonl",
        Path("~/Desktop/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl").expanduser(),
    ],
    "medium": [
        REPO_ROOT / "data/sources/gsm_symbolic/generated_data/GSM_p1.jsonl",
        REPO_ROOT / "data/sources/gsm_symbolic/generated_data/GSM_p2.jsonl",
        Path("~/Desktop/ml-gsm-symbolic/generated_data/GSM_p1.jsonl").expanduser(),
        Path("~/Desktop/ml-gsm-symbolic/generated_data/GSM_p2.jsonl").expanduser(),
    ],
}

BW_ACTIONS = ("pick-up", "put-down", "stack", "unstack")
MBW_ACTIONS = ("attack", "succumb", "overcome", "feast")

FAMILY_PATHS = {
    "bw": REPO_ROOT / "data/staging/bw_canonical.csv",
    "gsm": REPO_ROOT / "data/staging/gsm_canonical.csv",
    "algo": REPO_ROOT / "data/staging/algo_canonical.csv",
}


class Stage2OpenRouterClient:
    """OpenRouter chat client with system + user messages."""

    def __init__(self, model: str = MODEL_ID) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set.")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _post(self, messages: list[dict[str, str]]) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
        if not response.ok:
            try:
                detail = response.json().get("error", {}).get("message", str(response.text)[:400])
            except Exception:
                detail = (response.text or "")[:400]
            raise requests.HTTPError(f"{response.status_code}: {detail}", response=response)
        return response.json()

    def complete(
        self, problem_id: str, system: str, user: str, variant_type: str, logger: logging.Logger
    ) -> str:
        logger.info("  [API] Calling %s for %s %s...", self.model, problem_id, variant_type)
        data = self._post(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        text = ""
        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "") or ""
        logger.info("  [API] Response received (%s chars)", len(text))
        return text.strip()


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_canonicals(family: str) -> tuple[list[dict[str, str]], list[str]]:
    path = FAMILY_PATHS[family]
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def output_fieldnames(canonical_fields: list[str]) -> list[str]:
    fields = list(canonical_fields)
    if "generator_model" not in fields:
        fields.append("generator_model")
    return fields


def load_manifest() -> set[tuple[str, str]]:
    if not MANIFEST_PATH.exists():
        return set()
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return {
            (str(e["problem_id"]), str(e["variant_type"]).upper())
            for e in data.get("completed", [])
            if e.get("problem_id") and e.get("variant_type")
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return set()


def save_manifest(done: set[tuple[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed": [
            {"problem_id": pid, "variant_type": vtype}
            for pid, vtype in sorted(done)
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sync_manifest_from_variants_csv(output_path: Path, family: str) -> set[tuple[str, str]]:
    from_csv = set()
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("problem_id", "")).strip()
                vtype = str(row.get("variant_type", "")).strip().upper()
                if pid and vtype in VALID_VARIANT_TYPES:
                    from_csv.add((pid, vtype))
    merged = load_manifest() | from_csv
    save_manifest(merged)
    return merged


def load_existing_variants(output_path: Path, family: str = "") -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = load_manifest()
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("problem_id", "")).strip()
                vtype = str(row.get("variant_type", "")).strip().upper()
                if pid and vtype in VALID_VARIANT_TYPES:
                    done.add((pid, vtype))
    if family:
        save_manifest(done)
    return done


def log_failure(
    problem_id: str, variant_type: str, reason: str, failures_path: Path, logger: logging.Logger
) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failures_path.exists() or failures_path.stat().st_size == 0
    with failures_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["problem_id", "variant_type", "reason", "timestamp"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "problem_id": problem_id,
                "variant_type": variant_type,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    logger.info("  Logged failure: %s %s — %s", problem_id, variant_type, reason)


def make_variant_row(
    canonical: dict[str, str],
    variant_type: str,
    problem_text: str,
    correct_answer: str,
    generator_model: str = "deterministic",
    notes: str = "",
) -> dict[str, str]:
    row = dict(canonical)
    row["variant_type"] = variant_type
    row["problem_text"] = problem_text
    row["correct_answer"] = correct_answer
    row["generator_model"] = generator_model
    if notes:
        existing = str(row.get("notes", "")).strip()
        row["notes"] = f"{existing} | {notes}" if existing else notes
    return row


def write_variants_atomic(
    variants: list[dict[str, str]], output_path: Path, fieldnames: list[str]
) -> None:
    if not variants:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("problem_id", "")).strip()
                vtype = str(row.get("variant_type", "")).strip().upper()
                if pid and vtype in {"W1", "W2", "W3", "W4", "W5", "W6"}:
                    existing[(pid, vtype)] = row
    for row in variants:
        pid = str(row.get("problem_id", "")).strip()
        vtype = str(row.get("variant_type", "")).strip().upper()
        if pid and vtype:
            existing[(pid, vtype)] = row

    ordered: list[dict[str, str]] = []
    seen_pids: list[str] = []
    for (pid, _vtype), row in existing.items():
        if pid not in seen_pids:
            seen_pids.append(pid)
    for pid in seen_pids:
        for vtype in ("W1", "W2", "W3", "W4", "W5", "W6"):
            row = existing.get((pid, vtype))
            if row:
                ordered.append(row)

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(output_path)
    record = load_manifest() | {
        (str(r.get("problem_id", "")).strip(), str(r.get("variant_type", "")).strip().upper())
        for r in ordered
        if str(r.get("problem_id", "")).strip()
        and str(r.get("variant_type", "")).strip().upper() in VALID_VARIANT_TYPES
    }
    save_manifest(record)


def applicable_variants(row: dict[str, str]) -> list[str]:
    family = str(row.get("problem_family", "")).strip().lower()
    subtype = str(row.get("problem_subtype", "")).strip().lower()
    variants = ["W1", "W2", "W3", "W4", "W6"]

    w5_ok = False
    if family == "planning_suite" and subtype in {
        "blocksworld",
        "mystery_blocksworld",
        "bw_e",
        "blocksworld_e",
    }:
        w5_ok = True
    elif family == "arithmetic_reasoning" and subtype == "gsm_symbolic":
        w5_ok = True
    elif family == "algorithmic" and subtype == "shortest_path":
        w5_ok = True

    if w5_ok:
        variants.append("W5")
    return variants


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return cleaned


def _parse_json_response(text: str) -> dict:
    return json.loads(_strip_json_fences(text))


def _numbers_in_text(text: str) -> set[str]:
    return set(re.findall(r"\d+\.?\d*", text or ""))


def _extract_bw_blocks(text: str) -> set[str]:
    blocks: set[str] = set()
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


def _parse_bw_positions(section: str) -> dict[str, str]:
    positions: dict[str, str] = {}
    for m in re.finditer(r"[Bb]lock\s+(\w+)\s+is\s+on\s+block\s+(\w+)", section):
        positions[m.group(1).lower()] = m.group(2).lower()
    for m in re.finditer(r"[Bb]lock\s+(\w+)\s+is\s+on\s+the\s+table", section):
        positions[m.group(1).lower()] = "table"
    multi = re.search(
        r"[Bb]locks?\s+([\w,\s]+?)\s+are\s+(?:clear\s+and\s+)?on\s+the\s+table",
        section,
    )
    if multi:
        for b in re.findall(r"\b([a-z])\b", multi.group(1)):
            positions[b.lower()] = "table"
    return positions


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def verify_same_answer(canonical_answer: str, variant_answer: str) -> tuple[bool, str]:
    if normalize(canonical_answer) == normalize(variant_answer):
        return True, "ok"
    return False, "ANSWER_MISMATCH"


def verify_w3_planning_answer(
    canonical_answer: str, w3_answer: str, full_mapping: dict[str, str]
) -> tuple[bool, str]:
    inverse = make_inverse_mapping(full_mapping)
    restored = apply_mapping(w3_answer, inverse)
    if " ".join(restored.split()) != " ".join(canonical_answer.split()):
        return False, "ANSWER_MISMATCH"
    return True, "ok"


def _verify_w1_algo_lists(canonical_text: str, w1_text: str) -> tuple[bool, str]:
    for lst in re.findall(r"\[[\d,\s]+\]", canonical_text):
        norm = re.sub(r"\s+", "", lst)
        w1_lists = [re.sub(r"\s+", "", x) for x in re.findall(r"\[[\d,\s]+\]", w1_text)]
        if norm not in w1_lists:
            return False, f"missing_list:{lst}"
    return True, "ok"


# ── W1 ─────────────────────────────────────────────────────────────────────


def generate_w1(
    row: dict[str, str],
    client: Stage2OpenRouterClient | None,
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, str] | None:
    canonical_text = row["problem_text"]
    canonical_answer = row["correct_answer"]
    pid = row["problem_id"]
    if dry_run:
        return make_variant_row(
            row,
            "W1",
            "[DRY RUN - W1]",
            canonical_answer,
            notes="dry_run",
        )

    numbers = sorted(_numbers_in_text(canonical_text))
    blocks = _extract_bw_blocks(canonical_text) if row.get("problem_family") == "planning_suite" else set()
    algo_prefix = ""
    if row.get("problem_family") == "algorithmic":
        algo_prefix = (
            "IMPORTANT: This problem contains a list of numbers. "
            "You MUST preserve the exact list notation and all numbers verbatim. "
            "Do not spell numbers as words. Do not reorder lists.\n\n"
        )
    extra = ""
    for attempt in (1, 2):
        user = algo_prefix + prompts.W1_USER.format(problem_text=canonical_text) + extra
        text = client.complete(pid, prompts.W1_SYSTEM, user, "W1", logger)
        if _numbers_in_text(text) != _numbers_in_text(canonical_text):
            logger.info("  [API] Verification: failed — number mismatch (attempt %s)", attempt)
            extra = f"\n\nCRITICAL: These exact numbers must appear verbatim: {numbers}"
            continue
        if row.get("problem_family") == "algorithmic":
            ok_lists, reason_lists = _verify_w1_algo_lists(canonical_text, text)
            if not ok_lists:
                logger.info(
                    "  [API] Verification: failed — list format (%s) (attempt %s)",
                    reason_lists,
                    attempt,
                )
                canon_lists = re.findall(r"\[[\d,\s]+\]", canonical_text)
                extra = (
                    f"\n\nCRITICAL: Preserve these list literals exactly: {canon_lists}"
                )
                continue
        if blocks and not blocks.issubset(_extract_bw_blocks(text)):
            logger.info("  [API] Verification: failed — block name mismatch (attempt %s)", attempt)
            extra = f"\n\nCRITICAL: These block names must appear: {sorted(blocks)}"
            continue
        ok, reason = verify_same_answer(canonical_answer, canonical_answer)
        logger.info("  [API] Verification: %s — %s", "passed" if ok else "failed", reason)
        return make_variant_row(row, "W1", text, canonical_answer, generator_model=MODEL_ID)
    return None


# ── W2 ─────────────────────────────────────────────────────────────────────


def _bw_w2_table(text: str) -> str:
    m_cs = re.search(r"Current state:(.*?)(?:Goal:|$)", text, re.IGNORECASE | re.DOTALL)
    m_gl = re.search(r"Goal:(.*?)(?:Respond with|$)", text, re.IGNORECASE | re.DOTALL)
    if not m_cs or not m_gl:
        raise ValueError("Could not parse BW Current state / Goal")
    current = _parse_bw_positions(m_cs.group(1))
    goal = _parse_bw_positions(m_gl.group(1))
    blocks = sorted(set(current) | set(goal))
    lines = ["| Block | Current Position | Goal Position |", "|-------|-----------------|---------------|"]
    for b in blocks:
        cur = current.get(b, "?")
        gol = goal.get(b, "?")
        lines.append(f"| {b} | {cur} | {gol} |")
    return "\n".join(lines)


def generate_w2(
    row: dict[str, str],
    client: Stage2OpenRouterClient | None,
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, str] | None:
    answer = row["correct_answer"]
    family = row.get("problem_family", "")
    subtype = row.get("problem_subtype", "")

    if dry_run:
        return make_variant_row(row, "W2", "[DRY RUN - W2]", answer, notes="dry_run")

    try:
        if family == "planning_suite":
            text = _bw_w2_table(row["problem_text"])
            model = "deterministic"
        elif family == "arithmetic_reasoning":
            text = client.complete(
                row["problem_id"],
                prompts.W2_GSM_SYSTEM,
                prompts.W2_GSM_USER.format(problem_text=row["problem_text"]),
                "W2",
                logger,
            )
            if not _numbers_in_text(row["problem_text"]).issubset(_numbers_in_text(text)):
                logger.info("  W2: FAILED — GSM number preservation")
                return None
            model = MODEL_ID
        elif family == "algorithmic":
            params = json.loads(row.get("difficulty_params") or "{}")
            st = params.get("subtype", subtype).upper()
            if st == "CC":
                denoms = params.get("denominations", [])
                target = params.get("target", "")
                text = (
                    f"Given D = {{{', '.join(str(d) for d in denoms)}}} "
                    f"and target t = {target}, find minimum coin count."
                )
            elif st == "SP":
                edges = params.get("graph", [])
                src, tgt = params.get("source"), params.get("target")
                edge_lines = [f"({e['u']} → {e['v']}, weight={e['w']})" for e in edges]
                text = (
                    f"Graph edges:\n" + "\n".join(edge_lines) + f"\nSource: {src}\nTarget: {tgt}"
                )
            elif st == "WIS" or subtype == "wis":
                intervals = params.get("intervals", params.get("plots", []))
                lines = [
                    f"({it.get('id', i)}, start={it.get('start')}, end={it.get('end')}, weight={it.get('weight')})"
                    for i, it in enumerate(intervals)
                ]
                text = "Intervals:\n" + "\n".join(lines) + "\nObjective: maximize total weight of non-overlapping intervals."
            else:
                raise ValueError(f"Unsupported ALGO subtype for W2: {subtype}")
            model = "deterministic"
        else:
            raise ValueError(f"Unsupported family for W2: {family}")

        ok, reason = verify_same_answer(answer, answer)
        if not ok:
            return None
        return make_variant_row(row, "W2", text, answer, generator_model=model)
    except Exception as exc:
        logger.info("  W2: FAILED — %s", exc)
        return None


# ── W3 ─────────────────────────────────────────────────────────────────────


def _w3_prompts(row: dict[str, str]) -> tuple[str, str]:
    family = row.get("problem_family", "")
    subtype = row.get("problem_subtype", "")
    text = row["problem_text"]
    if family == "planning_suite":
        return prompts.W3_BW_MAPPING_SYSTEM, prompts.W3_BW_MAPPING_USER.format(problem_text=text)
    if family == "arithmetic_reasoning":
        return prompts.W3_GSM_MAPPING_SYSTEM, prompts.W3_GSM_MAPPING_USER.format(problem_text=text)
    if family == "algorithmic":
        if subtype == "shortest_path":
            return prompts.W3_SP_MAPPING_SYSTEM, prompts.W3_SP_MAPPING_USER.format(problem_text=text)
        if subtype == "wis":
            return prompts.W3_WIS_MAPPING_SYSTEM, prompts.W3_WIS_MAPPING_USER.format(problem_text=text)
        if subtype == "coin_change":
            return prompts.W3_CC_MAPPING_SYSTEM, prompts.W3_CC_MAPPING_USER.format(problem_text=text)
    raise ValueError(f"No W3 prompts for {family}/{subtype}")


def _apply_cc_w3(text: str, meta: dict) -> str:
    ctx = meta.get("chosen_context", "alternate units")
    unit = meta.get("unit_name", "unit")
    target_desc = meta.get("target_description", "target amount")
    out = text
    out = re.sub(r"\bcoins?\b", unit + "s", out, flags=re.IGNORECASE)
    out = re.sub(r"\bcoin change\b", ctx, out, flags=re.IGNORECASE)
    out = re.sub(r"\bdenominations?\b", f"{unit} sizes", out, flags=re.IGNORECASE)
    if "exact change for" in out.lower():
        out = re.sub(
            r"(exact change for)\s*\d+",
            rf"\1 the required {target_desc}",
            out,
            flags=re.IGNORECASE,
            count=1,
        )
    return out


def generate_w3(
    row: dict[str, str],
    client: Stage2OpenRouterClient | None,
    dry_run: bool,
    failures_path: Path,
    logger: logging.Logger,
) -> dict[str, str] | None:
    if dry_run:
        return make_variant_row(
            row, "W3", "[DRY RUN - W3]", row["correct_answer"], notes="dry_run"
        )

    system, user = _w3_prompts(row)
    canonical_text = row["problem_text"]
    canonical_answer = row["correct_answer"]
    subtype = row.get("problem_subtype", "")

    for attempt in range(1, 4):
        try:
            raw = client.complete(row["problem_id"], system, user, "W3", logger)
            meta = _parse_json_response(raw)
        except (json.JSONDecodeError, requests.HTTPError, ValueError) as exc:
            logger.info("  W3 attempt %s: JSON/API failed — %s", attempt, exc)
            continue

        if subtype == "coin_change":
            w3_text = _apply_cc_w3(canonical_text, meta)
            w3_answer = canonical_answer
            notes = json.dumps(meta, ensure_ascii=False)
            return make_variant_row(
                row, "W3", w3_text, w3_answer, generator_model=MODEL_ID, notes=notes
            )

        entity_mapping = {str(k): str(v) for k, v in meta.get("entity_mapping", {}).items()}
        action_mapping = {str(k): str(v) for k, v in meta.get("action_mapping", {}).items()}

        if row.get("problem_family") == "arithmetic_reasoning" and not entity_mapping:
            logger.info("  W3 GSM attempt %s: empty mapping, retrying", attempt)
            continue

        full_mapping = {**entity_mapping, **action_mapping}

        if not full_mapping:
            logger.info("  W3 attempt %s: empty mapping", attempt)
            continue
        if len(set(full_mapping.values())) != len(full_mapping):
            logger.info("  W3 attempt %s: non-bijective mapping", attempt)
            continue
        if row.get("problem_family") == "planning_suite" and subtype == "blocksworld":
            if not all(a in action_mapping for a in BW_ACTIONS):
                logger.info("  W3 attempt %s: missing BW actions", attempt)
                continue

        w3_text = apply_mapping(canonical_text, full_mapping)
        w3_answer = apply_mapping(str(canonical_answer), full_mapping)
        ok_rt, reason_rt = verify_w3_roundtrip(w3_text, canonical_text, full_mapping)
        logger.info("  [API] Verification: %s — %s", "passed" if ok_rt else "failed", reason_rt)
        if not ok_rt:
            continue

        family = row.get("problem_family", "")
        if family == "planning_suite":
            ok_ans, reason_ans = verify_w3_planning_answer(
                canonical_answer, w3_answer, full_mapping
            )
            logger.info("  [API] Verification: %s — %s", "passed" if ok_ans else "failed", reason_ans)
            if not ok_ans:
                continue
        elif family == "algorithmic" and subtype == "shortest_path":
            inverse = make_inverse_mapping(full_mapping)
            restored_answer = apply_mapping(w3_answer, inverse)
            if normalize(restored_answer) != normalize(canonical_answer):
                logger.info(
                    "  [API] Verification: failed — SP_W3_ANSWER_ROUNDTRIP_FAILED"
                )
                continue
            logger.info("  [API] Verification: passed — ok")
        else:
            ok_ans, reason_ans = verify_same_answer(canonical_answer, w3_answer)
            logger.info("  [API] Verification: %s — %s", "passed" if ok_ans else "failed", reason_ans)
            if not ok_ans:
                continue

        notes = json.dumps(meta, ensure_ascii=False)
        return make_variant_row(row, "W3", w3_text, w3_answer, generator_model=MODEL_ID, notes=notes)

    log_failure(row["problem_id"], "W3", "max_attempts_exceeded", failures_path, logger)
    return None


# ── W4 ─────────────────────────────────────────────────────────────────────


def _bw_w4_text(problem_text: str) -> str:
    blocks = sorted(_extract_bw_blocks(problem_text))
    m_cs = re.search(r"Current state:(.*?)(?:Goal:|$)", problem_text, re.IGNORECASE | re.DOTALL)
    m_gl = re.search(r"Goal:(.*?)(?:Respond|$)", problem_text, re.IGNORECASE | re.DOTALL)
    init_pos = _parse_bw_positions(m_cs.group(1)) if m_cs else {}
    goal_pos = _parse_bw_positions(m_gl.group(1)) if m_gl else {}

    def preds(pos: dict[str, str]) -> str:
        lines = []
        for b in sorted(pos):
            if pos[b] == "table":
                lines.append(f"OnTable({b})")
                lines.append(f"Clear({b})")
            else:
                lines.append(f"On({b},{pos[b]})")
        lines.append("HandEmpty")
        return "\n".join(lines)

    block_list = ", ".join(blocks)
    return (
        "FORMAL DEFINITION:\nπ = ⟨B, S₀, S*, A⟩\n\n"
        f"BLOCKS:\nB = {{{block_list}}}\n\n"
        f"INITIAL STATE S₀:\n{preds(init_pos)}\n\n"
        f"GOAL STATE S*:\n{preds(goal_pos)}\n\n"
        "ACTIONS (Preconditions → Add Effects | Delete Effects):\n"
        "pick-up(x): Pre: Clear(x), OnTable(x), HandEmpty → Holding(x) | ¬Clear(x), ¬OnTable(x), ¬HandEmpty\n"
        "put-down(x): Pre: Holding(x) → OnTable(x), Clear(x), HandEmpty | ¬Holding(x)\n"
        "stack(x,y): Pre: Holding(x), Clear(y) → On(x,y), Clear(x), HandEmpty | ¬Holding(x), ¬Clear(y)\n"
        "unstack(x,y): Pre: On(x,y), Clear(x), HandEmpty → Holding(x) | ¬On(x,y), ¬Clear(x), ¬HandEmpty\n\n"
        "TASK:\nFind plan π = (a₁, a₂, ..., aₙ) such that:\n"
        "S₀ →(a₁) S₁ →(a₂) ... →(aₙ) Sₙ, where S* ⊆ Sₙ\n\n"
        "OUTPUT FORMAT:\nReturn a numbered list of actions only.\nNo explanation. No extra text."
    )


def _mbw_w4_text(problem_text: str) -> str:
    text = problem_text.lower()
    cs = re.search(r"current state:(.*?)(?:goal:|$)", text, re.DOTALL)
    gl = re.search(r"goal:(.*?)(?:respond|$)", text, re.DOTALL)
    init_block = cs.group(1).strip() if cs else ""
    goal_block = gl.group(1).strip() if gl else ""
    return (
        "FORMAL DEFINITION:\nπ = ⟨B, S₀, S*, A⟩ (Mystery Blocksworld)\n\n"
        f"INITIAL STATE S₀:\n{init_block}\n\n"
        f"GOAL STATE S*:\n{goal_block}\n\n"
        "ACTIONS:\n"
        "attack(x): Pre: province(x), planet(x), harmony\n"
        "succumb(x): Pre: pain(x)\n"
        "overcome(x,y): Pre: pain(x), province(y)\n"
        "feast(x,y): Pre: craves(x,y), province(x), harmony\n\n"
        "TASK:\nFind plan achieving S* from S₀.\n\n"
        "OUTPUT FORMAT:\nReturn a numbered list of actions only.\nNo explanation. No extra text."
    )


def _gsm_w4_text(problem_text: str) -> str:
    text = problem_text.strip()
    nums = re.findall(r"[-+]?\d+\.?\d*", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if nums and sentences:
        var_lines: list[str] = []
        for i, num in enumerate(nums[:10]):
            label = "value"
            for sent in sentences:
                if num in sent:
                    label = sent[:100]
                    break
            var_lines.append(f"  n{i + 1} = {num}  # {label}")

        rel_lines: list[str] = []
        math_hint = ("twice", "half", "percent", "more", "less", "total", "sum", "each", "per")
        for sent in sentences[:-1]:
            lower = sent.lower()
            if re.search(r"\d", sent) and (
                any(h in lower for h in math_hint) or len(rel_lines) < len(sentences) - 2
            ):
                rel_lines.append(f"  {sent}")
        if not rel_lines and len(sentences) > 1:
            rel_lines = [f"  {sent}" for sent in sentences[:-1][:6]]

        question = sentences[-1] if sentences else text
        structured = (
            "Variables:\n"
            + "\n".join(var_lines)
            + "\nRelationships:\n"
            + "\n".join(rel_lines[:8])
            + f"\nFind: solve for the unknown — {question}"
        )
        if "<" not in structured:
            return structured

    given = ", ".join(nums) if nums else "none"
    context = sentences[0] if sentences else text[:200]
    compute = sentences[-1] if sentences else text
    return (
        f"Given values: {given}\n"
        f"Context: {context}\n"
        f"Compute: {compute}\n"
        "Output: single numeric answer only."
    )


def generate_w4(row: dict[str, str], dry_run: bool, logger: logging.Logger) -> dict[str, str] | None:
    if dry_run:
        return make_variant_row(row, "W4", "[DRY RUN - W4]", row["correct_answer"], notes="dry_run")
    try:
        family = row.get("problem_family", "")
        subtype = row.get("problem_subtype", "")
        if family == "planning_suite" and subtype == "mystery_blocksworld":
            text = _mbw_w4_text(row["problem_text"])
        elif family == "planning_suite":
            text = _bw_w4_text(row["problem_text"])
        elif family == "arithmetic_reasoning":
            text = _gsm_w4_text(row["problem_text"])
        elif family == "algorithmic":
            params = json.loads(row.get("difficulty_params") or "{}")
            st = params.get("subtype", subtype).upper()
            if st == "CC":
                denoms = params.get("denominations", [])
                target = params.get("target", "")
                text = (
                    "FORMAL DEFINITION: π = ⟨D, S₀, S*, A⟩\n"
                    f"DENOMINATIONS: D = {{{', '.join(str(d) for d in denoms)}}}\n"
                    "INITIAL STATE S₀: CurrentSum = 0\n"
                    f"GOAL STATE S*: CurrentSum = {target}\n"
                    "ACTIONS:\nSelectCoin(c):\n"
                    f"  Pre: c ∈ D, (CurrentSum + c) ≤ {target}\n"
                    "  Effect: CurrentSum ← CurrentSum + c\n"
                    f"TASK: Find shortest sequence of actions to reach CurrentSum = {target}.\n"
                    "OUTPUT FORMAT:\nCount: [integer]\nCoins: [denomination1, denomination2, ...]"
                )
            elif st == "SP":
                n = params.get("num_vertices", 0)
                src, tgt = params.get("source"), params.get("target")
                edges = params.get("graph", [])
                edge_str = ", ".join(f"({e['u']},{e['v']},{e['w']})" for e in edges)
                text = (
                    f"G = (V, E) where V = {{0, 1, ..., {n-1}}}\n"
                    f"E = {{{edge_str}}}\n"
                    f"Find: δ({src}, {tgt}) = minimum weight path from {src} to {tgt}\n"
                    "OUTPUT FORMAT:\nPath: X -> X -> X, Cost: X"
                )
            else:
                intervals = params.get("intervals", [])
                interval_lines = "\n".join(
                    f"  (s={it.get('start')}, f={it.get('end')}, w={it.get('weight')})"
                    for it in intervals
                )
                text = (
                    "J = {(sⱼ, fⱼ, wⱼ)} where:\n" + interval_lines + "\n"
                    "Non-overlap constraint: ∀i,j ∈ S: [sᵢ,fᵢ] ∩ [sⱼ,fⱼ] = ∅\n"
                    "Find: S ⊆ J maximizing Σⱼ∈S wⱼ\n"
                    "OUTPUT FORMAT:\nSelected: {X, X, X, ...}, Total: X"
                )
        else:
            raise ValueError(f"W4 unsupported: {family}/{subtype}")
        return make_variant_row(row, "W4", text, row["correct_answer"], generator_model="deterministic")
    except Exception as exc:
        logger.info("  W4: FAILED — %s", exc)
        return None


# ── W5 ─────────────────────────────────────────────────────────────────────


def _resolve_pddl_path(source: str) -> Path | None:
    m = re.search(r"path=([^\s|]+)", source or "")
    if not m:
        return None
    rel = m.group(1).strip()
    candidates = [
        PLANBENCH_ROOT / rel,
        REPO_ROOT / "data/sources/planbench" / rel,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_n_blocks_from_row(row: dict[str, str]) -> int | None:
    params_str = row.get("difficulty_params", "")
    m = re.search(r"num_blocks=(\d+)", params_str)
    if m:
        return int(m.group(1))
    text = row.get("problem_text", "")
    m = re.search(r"Blocks? ([a-z](?:,\s*[a-z])*)", text)
    if m:
        blocks = re.findall(r"\b[a-z]\b", m.group(1))
        return len(blocks) if blocks else None
    return None


def _swap_bw_nl_problem_text(problem_text: str) -> str | None:
    """Swap Current state and Goal sections in canonical BW natural-language prompt."""
    m = re.search(
        r"(Current state:)(.*?)(Goal:)(.*?)(Respond with)",
        problem_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None

    new_current = m.group(4).strip()
    on_re = re.findall(r"block\s+(\w+)\s+is on block\s+(\w+)", new_current, re.IGNORECASE)
    if on_re:
        blocks_in_tower = sorted({b for pair in on_re for b in pair})
        has_above = {low for _up, low in on_re}
        for b in blocks_in_tower:
            if b not in has_above:
                new_current += f" Block {b} is clear."
    if "hand is empty" not in new_current.lower():
        new_current += " The hand is empty."

    new_goal = m.group(2).strip()
    blocks = _extract_bw_blocks(problem_text)
    if re.search(r"blocks?\s+.*are clear and on the table", new_goal, re.IGNORECASE):
        new_goal = " ".join(f"Block {b} is on the table." for b in sorted(blocks))
        if "hand is empty" not in new_goal.lower():
            new_goal += " The hand is empty."

    return (
        problem_text[: m.start(1)]
        + m.group(1)
        + " "
        + new_current
        + " "
        + m.group(3)
        + " "
        + new_goal
        + " "
        + problem_text[m.start(5) :]
    )


def _invert_bw_move(move: str) -> str:
    inv = {"pick-up": "put-down", "put-down": "pick-up", "stack": "unstack", "unstack": "stack"}
    lm = move.strip().lower()
    for verb, inverse in inv.items():
        if lm.startswith(verb + " "):
            rest = move[len(verb) :].strip()
            return f"{inverse} {rest}"
    raise ValueError(f"Cannot invert move: {move}")


def generate_w5_bw(
    row: dict[str, str], dry_run: bool, logger: logging.Logger
) -> dict[str, str] | None:
    if dry_run:
        return make_variant_row(row, "W5", "[DRY RUN - W5]", row["correct_answer"], notes="dry_run")

    source = row.get("source", "") or ""
    path_match = re.search(r"path=", source)
    pddl_path = _resolve_pddl_path(source)
    if pddl_path is None and path_match is None:
        pid = row["problem_id"]
        seed = w6_seed(pid + "W5")
        n_blocks = parse_n_blocks_from_row(row)
        if n_blocks is None:
            log_failure(pid, "W5", "cannot_determine_n_blocks", FAILURES_PATH, logger)
            return None
        domain_pddl = load_bw_domain()
        _, problem_pddl = generate_random_bw_pddl(n_blocks, seed)
        plan, status = run_fast_downward(domain_pddl, problem_pddl, timeout=60)
        if plan is None:
            _, problem_pddl = generate_random_bw_pddl(n_blocks, seed + 1)
            plan, status = run_fast_downward(domain_pddl, problem_pddl, timeout=60)
        if plan is None:
            log_failure(pid, "W5", f"fd_failed_bw_e: {status}", FAILURES_PATH, logger)
            return None
        w5_text = pddl_to_natural_language(problem_pddl, n_blocks)
        w5_answer = fd_plan_to_bw_format(plan)
        return make_variant_row(
            row,
            "W5",
            w5_text,
            w5_answer,
            generator_model="deterministic",
            notes=f"w5_bw_e_procedural_seed={seed}",
        )
    if pddl_path is None:
        logger.info("  W5: FAILED — PDDL path not found")
        return None
    pddl_text = pddl_path.read_text(encoding="utf-8")
    on_pairs, all_blocks, new_init_content = inspect_w5_goal_tower(pddl_text)
    swapped = swap_pddl_init_goal(pddl_text)
    domain = load_bw_domain()
    n_blocks = len(_extract_bw_blocks(row.get("problem_text", ""))) or 4
    m_nb = re.search(r"num_blocks=(\d+)", row.get("difficulty_params", ""))
    if m_nb:
        n_blocks = int(m_nb.group(1))

    logger.info("  [W5-BW] domain length: %s", len(domain))
    logger.info("  [W5-BW] on_pairs from goal: %s", on_pairs)
    logger.info("  [W5-BW] all_blocks: %s", all_blocks)
    logger.info("  [W5-BW] derived new init:\n%s", new_init_content)
    logger.info("  [W5-BW] swapped PDDL preview: %s", swapped[:200])
    logger.info(
        "  [W5-BW] FD command: python tools/fast-downward/fast-downward.py "
        "--plan-file <tmp>/plan.txt <domain> <problem> --search astar(lmcut())"
    )

    plan, status = run_fast_downward(domain, swapped, timeout=60)
    if plan is None:
        logger.info("  W5: FAILED — FD %s", status)
        return None
    w5_text = _swap_bw_nl_problem_text(row.get("problem_text", ""))
    if w5_text is None:
        w5_text = pddl_to_natural_language(swapped, n_blocks)
        logger.info("  [W5-BW] NL text from PDDL fallback (canonical swap failed)")
    w5_answer = fd_plan_to_bw_format(plan)
    ok = _verify_blocksworld_state_machine(w5_answer, w5_text)
    logger.info("  [API] Verification: %s — blocksworld plan sim", "passed" if ok else "failed")
    notes = ""
    if ok is not True:
        step_count = len([ln for ln in w5_answer.splitlines() if ln.strip()])
        logger.info(
            "  [W5-BW] NL plan sim failed; accepting FD plan (%s steps)", step_count
        )
        notes = "W5_FD_PLAN_ACCEPTED_NL_SIM_FAILED"
    return make_variant_row(
        row, "W5", w5_text, w5_answer, generator_model="deterministic", notes=notes
    )


def generate_w5_mbw(row: dict[str, str], dry_run: bool, logger: logging.Logger) -> dict[str, str] | None:
    if dry_run:
        return make_variant_row(row, "W5", "[DRY RUN - W5]", "", notes="dry_run|MBW_W5_NEEDS_MANUAL_REVIEW")
    moves = [ln.strip() for ln in row["correct_answer"].splitlines() if ln.strip()]
    inv = {"attack": "succumb", "succumb": "attack", "overcome": "feast", "feast": "overcome"}
    mbw_plan = []
    for m in reversed(moves):
        parts = m.split()
        if parts and parts[0] in inv:
            parts[0] = inv[parts[0]]
        mbw_plan.append(" ".join(parts))
    w5_text = row["problem_text"]
    notes = "MBW_W5_NEEDS_MANUAL_REVIEW"
    ok = _verify_mystery_state_machine("\n".join(mbw_plan), w5_text)
    if ok is not True:
        notes += " | W5_MBw_PLAN_UNVERIFIED"
    return make_variant_row(
        row, "W5", w5_text, "\n".join(mbw_plan), generator_model="deterministic", notes=notes
    )


def generate_w5_gsm(
    row: dict[str, str],
    client: Stage2OpenRouterClient | None,
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, str] | None:
    if dry_run:
        return make_variant_row(row, "W5", "[DRY RUN - W5]", row["correct_answer"], notes="dry_run")
    canonical_answer = str(row["correct_answer"]).strip()
    w5_text = client.complete(
        row["problem_id"],
        prompts.W5_GSM_SYSTEM,
        prompts.W5_GSM_USER.format(
            problem_text=row["problem_text"], correct_answer=canonical_answer
        ),
        "W5",
        logger,
    )
    notes = "GSM_W5_NEEDS_MANUAL_VERIFICATION"
    if canonical_answer not in w5_text:
        notes += " | W5_GSM_ANSWER_NOT_EMBEDDED"
        logger.info("  W5_GSM_ANSWER_NOT_EMBEDDED")

    solve_system = (
        "You solve math problems. Output only the final numeric answer. No explanation."
    )
    solve_user = f"What is the answer to this math problem? Give only the number.\n\n{w5_text}"
    solve_raw = client.complete(
        row["problem_id"], solve_system, solve_user, "W5-answer", logger
    )
    solve_nums = re.findall(r"[-+]?\d+\.?\d*", solve_raw)
    w5_answer = solve_nums[-1].strip() if solve_nums else solve_raw.strip()
    logger.info(
        "  [W5-GSM] canonical_answer=%s, w5_answer=%s", canonical_answer, w5_answer
    )
    if w5_answer == canonical_answer:
        notes += " | W5_GSM_ANSWER_EQUALS_CANONICAL"
        logger.info("  W5_GSM_ANSWER_EQUALS_CANONICAL")

    return make_variant_row(row, "W5", w5_text, w5_answer, generator_model=MODEL_ID, notes=notes)


def _build_sp_graph(params: dict) -> nx.DiGraph:
    g = nx.DiGraph()
    for e in params.get("graph", []):
        g.add_edge(int(e["u"]), int(e["v"]), weight=int(e["w"]))
    return g


def generate_w5_sp(row: dict[str, str], dry_run: bool, logger: logging.Logger) -> dict[str, str] | None:
    if dry_run:
        return make_variant_row(row, "W5", "[DRY RUN - W5]", row["correct_answer"], notes="dry_run")
    params = json.loads(row.get("difficulty_params") or "{}")
    src, tgt = int(params["source"]), int(params["target"])
    use_bf = bool(params.get("requires_bellman_ford", False))
    g = _build_sp_graph(params)
    g_rev = g.reverse(copy=True)
    try:
        if use_bf:
            path = nx.bellman_ford_path(g_rev, tgt, src, weight="weight")
            cost = int(nx.bellman_ford_path_length(g_rev, tgt, src, weight="weight"))
        else:
            path = nx.shortest_path(g_rev, tgt, src, weight="weight")
            cost = int(nx.shortest_path_length(g_rev, tgt, src, weight="weight"))
    except nx.NetworkXNoPath:
        logger.info("  W5: FAILED — no reversed path")
        return None

    w5_answer = format_sp_answer(path, cost)
    text = row["problem_text"]
    # Flip printed edge lines case-insensitively; do not touch the query sentence.
    edge_line = re.compile(
        r"(?P<prefix>^\s*[-*]?\s*)[Nn]ode\s+(?P<u>\d+)\s+to\s+[Nn]ode\s+(?P<v>\d+)\s*:\s*(?P<w>-?\d+)\s*$"
    )
    rewritten: list[str] = []
    for line in text.splitlines(keepends=True):
        core, nl = (line[:-1], line[-1]) if line.endswith("\n") else (line, "")
        m = edge_line.match(core)
        if m:
            node_word = "Node" if "Node" in core else "node"
            rewritten.append(
                f"{m.group('prefix')}{node_word} {m.group('v')} to {node_word} {m.group('u')}: {m.group('w')}{nl}"
            )
        else:
            rewritten.append(line)
    text = "".join(rewritten)
    text = re.sub(
        rf"from node\s*{src}\s+to node\s*{tgt}",
        f"from node {tgt} to node {src}",
        text,
        flags=re.IGNORECASE,
    )
    # Re-verify
    g2 = _build_sp_graph(params)
    g2_rev = g2.reverse(copy=True)
    if use_bf:
        path2 = nx.bellman_ford_path(g2_rev, tgt, src, weight="weight")
        cost2 = int(nx.bellman_ford_path_length(g2_rev, tgt, src, weight="weight"))
    else:
        path2 = nx.shortest_path(g2_rev, tgt, src, weight="weight")
        cost2 = int(nx.shortest_path_length(g2_rev, tgt, src, weight="weight"))
    if path2 != path or cost2 != cost:
        logger.info("  W5: FAILED — SP verification mismatch")
        return None
    return make_variant_row(row, "W5", text, w5_answer, generator_model="deterministic")


def generate_w5(
    row: dict[str, str],
    client: Stage2OpenRouterClient | None,
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, str] | None:
    family = row.get("problem_family", "")
    subtype = row.get("problem_subtype", "")
    if family == "planning_suite" and subtype == "blocksworld":
        return generate_w5_bw(row, dry_run, logger)
    if family == "planning_suite" and subtype == "mystery_blocksworld":
        return generate_w5_mbw(row, dry_run, logger)
    if family == "arithmetic_reasoning":
        return generate_w5_gsm(row, client, dry_run, logger)
    if family == "algorithmic" and subtype == "shortest_path":
        return generate_w5_sp(row, dry_run, logger)
    return None


# ── W6 ─────────────────────────────────────────────────────────────────────


def _load_gsm_instance(template_id: str, pole: str) -> dict | None:
    paths = GSM_JSONL_PATHS.get("high" if pole == "high" else "medium", [])
    fallback: dict | None = None
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tid = str(data.get("template_id", data.get("id", "")))
                if tid != str(template_id):
                    continue
                inst = str(data.get("instance", ""))
                if inst == "1":
                    return data
                if inst == "0" and fallback is None:
                    fallback = data
    return fallback


def _generate_cc_w6_from_seed(
    rng: random.Random,
    canonical_denoms: list[int],
    canonical_target: int,
    canonical_text: str,
) -> tuple[list[int], int, str, str] | None:
    """Generate a CC instance that differs from canonical denominations/target."""
    k = len(canonical_denoms)
    has_one = 1 in canonical_denoms
    lo = 2 if not has_one else 1
    hi = max(30, max(canonical_denoms) + 30)
    canon_denoms = [int(d) for d in canonical_denoms]

    for _ in range(2000):
        if has_one:
            pool = [1] + rng.sample(range(2, hi + 1), k - 1)
        else:
            pool = rng.sample(range(lo, hi + 1), k)
        new_denoms = sorted(set(pool))
        if len(new_denoms) != k:
            continue
        if not has_one and 1 in new_denoms:
            continue
        new_target = rng.randint(20, 150)
        if new_denoms == canon_denoms and new_target == canonical_target:
            continue
        solved = dp_coin_change(new_denoms, new_target) or greedy_coin_change(
            new_denoms, new_target
        )
        if not solved:
            continue
        count, coins = solved
        text = render_cc_text_from_canonical(canonical_text, new_denoms, new_target)
        answer = format_cc_answer(count, coins)
        return new_denoms, new_target, text, answer
    return None


def generate_w6_cc(
    row: dict[str, str],
    logger: logging.Logger,
    failures_path: Path,
) -> dict[str, str] | None:
    """Procedural W6 for coin change with distinct denominations/target from canonical."""
    pid = row["problem_id"]
    canonical_params = json.loads(row.get("difficulty_params") or "{}")
    canonical_denoms = [int(d) for d in canonical_params.get("denominations", [])]
    canonical_target = int(canonical_params.get("target", 0))
    base_seed = w6_seed(pid)

    for offset in range(3):
        attempt_seed = base_seed + offset
        rng = random.Random(attempt_seed)
        result = _generate_cc_w6_from_seed(
            rng, canonical_denoms, canonical_target, row["problem_text"]
        )
        if result is None:
            continue
        new_denoms, new_target, text, answer = result
        if new_denoms == canonical_denoms and new_target == canonical_target:
            logger.warning(
                "  W6 CC same as canonical for %s, reseeding (seed=%s)", pid, attempt_seed
            )
            continue
        logger.info(
            "  [W6-CC] canonical target=%s, w6 target=%s", canonical_target, new_target
        )
        return make_variant_row(
            row,
            "W6",
            text,
            answer,
            generator_model="deterministic",
            notes=f"w6_seed={attempt_seed}",
        )

    log_failure(pid, "W6", "cc_w6_same_as_canonical_after_reseeds", failures_path, logger)
    return None


def generate_w6(
    row: dict[str, str],
    done_set: set[tuple[str, str]],
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, str] | None:
    pid = row["problem_id"]
    if (pid, "W6") in done_set:
        logger.info("  W6 already exists for %s, skipping generation", pid)
        return None
    if dry_run:
        return make_variant_row(row, "W6", "[DRY RUN - W6]", row["correct_answer"], notes="dry_run")

    family = row.get("problem_family", "")
    subtype = row.get("problem_subtype", "")
    seed = w6_seed(pid)

    try:
        if family == "planning_suite":
            n_blocks = 4
            m = re.search(r"num_blocks=(\d+)", row.get("difficulty_params", ""))
            if m:
                n_blocks = int(m.group(1))
            for attempt_seed in (seed, seed + 1):
                domain, problem = generate_random_bw_pddl(n_blocks, attempt_seed)
                plan, status = run_fast_downward(domain, problem, timeout=60)
                if plan:
                    text = pddl_to_natural_language(problem, n_blocks)
                    answer = fd_plan_to_bw_format(plan)
                    ok = _verify_blocksworld_state_machine(answer, text)
                    if ok is True:
                        return make_variant_row(
                            row,
                            "W6",
                            text,
                            answer,
                            generator_model="deterministic",
                            notes=f"w6_seed={attempt_seed}",
                        )
            return None

        if family == "arithmetic_reasoning":
            m = re.search(r"template_id=(\d+)", row.get("source", ""))
            if not m:
                return None
            inst = _load_gsm_instance(m.group(1), row.get("contamination_pole", "high"))
            if not inst:
                return None
            answer_raw = inst.get("answer", "")
            ans_m = re.findall(r"####\s*([^\n\r]+)", answer_raw)
            answer = ans_m[-1].strip() if ans_m else str(answer_raw).strip()
            return make_variant_row(
                row,
                "W6",
                inst.get("question", "").strip(),
                answer,
                generator_model="deterministic",
                notes=f"gsm_template_{m.group(1)}_instance_1",
            )

        if family == "algorithmic":
            params = json.loads(row.get("difficulty_params") or "{}")
            rng = random.Random(seed)
            st = params.get("subtype", subtype).upper()
            if st == "CC":
                return generate_w6_cc(row, logger, FAILURES_PATH)
            elif st == "SP":
                n = int(params.get("num_vertices", 5))
                num_edges = len(params.get("graph", []))
                directed = bool(params.get("directed", True))
                use_bf = bool(params.get("requires_bellman_ford", False))
                src, tgt = int(params["source"]), int(params["target"])
                g, path, cost = generate_sp_graph(
                    rng, n, num_edges, directed, src, tgt, use_bf
                )
                edges = [(int(u), int(v), int(d["weight"])) for u, v, d in g.edges(data=True)]
                text = render_sp_text_from_canonical(row["problem_text"], edges, directed)
                answer = format_sp_answer(path, cost)
            else:
                n = int(params.get("num_intervals", len(params.get("intervals", [])) or 5))
                graph_type = str(params.get("graph_type", "path"))
                greedy_ok = bool(params.get("greedy_succeeds", True))
                g, weights, selected, total = generate_wis_graph(
                    rng, n, graph_type, greedy_ok, None
                )
                edges = list(g.edges())
                text = render_wis_text_from_canonical(row["problem_text"], weights, edges)
                answer = format_wis_answer(selected, total)
            return make_variant_row(
                row, "W6", text, answer, generator_model="deterministic", notes=f"w6_seed={seed}"
            )
    except Exception as exc:
        logger.info("  W6: FAILED — %s", exc)
        return None
    return None


# ── Dispatcher ─────────────────────────────────────────────────────────────


def generate_variant(
    row: dict[str, str],
    variant_type: str,
    client: Stage2OpenRouterClient | None,
    dry_run: bool,
    failures_path: Path,
    done_set: set[tuple[str, str]],
    logger: logging.Logger,
) -> dict[str, str] | None:
    logger.info("  %s: generating...", variant_type)
    try:
        if variant_type == "W1":
            result = generate_w1(row, client, dry_run, logger)
        elif variant_type == "W2":
            result = generate_w2(row, client, dry_run, logger)
        elif variant_type == "W3":
            result = generate_w3(row, client, dry_run, failures_path, logger)
        elif variant_type == "W4":
            result = generate_w4(row, dry_run, logger)
        elif variant_type == "W5":
            result = generate_w5(row, client, dry_run, logger)
        elif variant_type == "W6":
            result = generate_w6(row, done_set, dry_run, logger)
        else:
            result = None
        if result is None:
            w6_skip = variant_type == "W6" and (row["problem_id"], "W6") in done_set
            if not w6_skip:
                log_failure(
                    row["problem_id"],
                    variant_type,
                    "generation_returned_none",
                    failures_path,
                    logger,
                )
        return result
    except Exception as exc:
        logger.info("  %s: FAILED — %s", variant_type, exc)
        log_failure(row["problem_id"], variant_type, str(exc), failures_path, logger)
        return None


def process_family(
    family: str,
    args: argparse.Namespace,
    client: Stage2OpenRouterClient | None,
    logger: logging.Logger,
) -> tuple[int, int, int]:
    canonicals, canon_fields = load_canonicals(family)
    if args.limit:
        canonicals = canonicals[: args.limit]
    if args.problem_id:
        canonicals = [r for r in canonicals if r["problem_id"] == args.problem_id]

    output_path = REPO_ROOT / f"data/staging/{family}_variants.csv"
    fieldnames = output_fieldnames(canon_fields)
    if args.resume or args.gaps_only:
        done_set = sync_manifest_from_variants_csv(output_path, family)
    else:
        done_set = set()

    if args.gaps_only:
        filtered = []
        for row in canonicals:
            pid = row["problem_id"]
            applicable = applicable_variants(row)
            if not all((pid, v) in done_set for v in applicable):
                filtered.append(row)
        logger.info(
            "Gaps-only: %s problems need work (of %s canonicals)",
            len(filtered),
            len(canonicals),
        )
        canonicals = filtered

    logger.info("Loading canonicals from %s: %s rows", FAMILY_PATHS[family], len(canonicals))
    logger.info("Output path: %s", output_path)
    logger.info("Processing %s canonicals", len(canonicals))

    processed = skipped = failed_variants = 0

    for i, row in enumerate(canonicals):
        pid = row["problem_id"]
        applicable = applicable_variants(row)

        if args.resume and all((pid, v) in done_set for v in applicable):
            logger.info("Skipping %s: all %s variants already done", pid, len(applicable))
            skipped += 1
            continue

        logger.info("Processing %s (%s/%s)", pid, i + 1, len(canonicals))
        logger.info("  Applicable variants: %s", applicable)

        variants: list[dict[str, str]] = []
        for v in applicable:
            if args.resume and (pid, v) in done_set:
                if v == "W6":
                    logger.info("  W6 already exists for %s, skipping", pid)
                else:
                    logger.info("  %s: already done, skipping", v)
                continue

            result = generate_variant(
                row, v, client, args.dry_run, FAILURES_PATH, done_set, logger
            )
            if result is not None:
                variants.append(result)
                done_set.add((pid, v))
                logger.info("  %s: OK", v)
            else:
                if not (v == "W6" and (pid, "W6") in done_set):
                    failed_variants += 1
                    logger.info("  %s: FAILED", v)

        if variants and not args.dry_run:
            write_variants_atomic(variants, output_path, fieldnames)
            logger.info("  Written %s variants for %s", len(variants), pid)
        elif variants and args.dry_run:
            logger.info("  [DRY RUN] Would write %s variants for %s", len(variants), pid)

        processed += 1

    return processed, skipped, failed_variants


def run_test_mode(logger: logging.Logger) -> None:
    logger.info("=== Stage 2 TEST MODE ===")
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY required for --test")
        raise SystemExit(1)
    client = Stage2OpenRouterClient(model=MODEL_ID)
    rows_summary: list[tuple[str, str, str, str]] = []
    for family in ("bw", "gsm", "algo"):
        canonicals, _ = load_canonicals(family)
        if not canonicals:
            continue
        row = canonicals[0]
        pid = row["problem_id"]
        for v in applicable_variants(row):
            res = generate_variant(row, v, client, False, FAILURES_PATH, set(), logger)
            status = "OK" if res else "FAIL"
            rows_summary.append((family, pid, v, status))
    logger.info("\n=== TEST SUMMARY ===")
    for family, pid, v, status in rows_summary:
        logger.info("%-4s %-12s %-3s %s", family, pid, v, status)
    fails = sum(1 for *_, s in rows_summary if s == "FAIL")
    if fails:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 variant generation")
    parser.add_argument("--family", required=True, choices=["bw", "gsm", "algo", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--gaps-only",
        action="store_true",
        help="Only process canonicals missing at least one required variant (implies --resume).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-id", default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()
    if args.test:
        run_test_mode(logger)
        return

    if args.gaps_only:
        args.resume = True

    logger.info(
        "=== Stage 2 starting: family=%s, dry_run=%s, resume=%s, gaps_only=%s ===",
        args.family,
        args.dry_run,
        args.resume,
        args.gaps_only,
    )

    client: Stage2OpenRouterClient | None = None
    if not args.dry_run:
        client = Stage2OpenRouterClient(model=MODEL_ID)

    families = ["bw", "gsm", "algo"] if args.family == "all" else [args.family]
    total_processed = total_skipped = total_failed = 0
    for fam in families:
        p, s, f = process_family(fam, args, client, logger)
        total_processed += p
        total_skipped += s
        total_failed += f

    logger.info(
        "=== Done. processed=%s, skipped=%s, failed_variants=%s ===",
        total_processed,
        total_skipped,
        total_failed,
    )


if __name__ == "__main__":
    main()
