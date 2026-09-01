"""Canonical paths under results/. All run scripts should import from here."""

from __future__ import annotations

import re
from pathlib import Path

RESULTS = Path("results")
RAW = RESULTS / "raw"
DERIVED = RESULTS / "derived"
PAPER = RESULTS / "paper"
FIGURES = RESULTS / "figures"
ARCHIVE = RESULTS / "archive"
DEPRECATED = RESULTS / "deprecated"

MODEL_SLUGS = ("claude", "gpt4o", "llama")


def model_slug(model: str) -> str:
    m = model.lower()
    if "claude" in m:
        return "claude"
    if "gpt-4o" in m or "gpt4o" in m:
        return "gpt4o"
    if "llama" in m:
        return "llama"
    return re.sub(r"[^a-z0-9]+", "_", m).strip("_") or "model"


def ensure_dirs() -> None:
    for d in (RAW, DERIVED, PAPER, FIGURES):
        d.mkdir(parents=True, exist_ok=True)


# --- ALGO ---
def algo_p1_behavioral(model: str) -> Path:
    return RAW / f"ALGO_P1_behavioral_{model_slug(model)}.csv"


ALGO_P1_BEHAVIORAL_CLAUDE = RAW / "ALGO_P1_behavioral_claude.csv"
ALGO_P1_BEHAVIORAL_GPT4O = RAW / "ALGO_P1_behavioral_gpt4o.csv"
ALGO_P1_BEHAVIORAL_LLAMA = RAW / "ALGO_P1_behavioral_llama.csv"
ALGO_P1_BEHAVIORAL_ALL = [ALGO_P1_BEHAVIORAL_CLAUDE, ALGO_P1_BEHAVIORAL_GPT4O, ALGO_P1_BEHAVIORAL_LLAMA]

ALGO_P1_REVIEW_QUEUE = RAW / "ALGO_P1_review_queue.csv"
# Authoritative Phase 1 files are the 110-row `_new` (Claude/GPT-4o/Llama)
# overlays and the 110-row Gemini file. The unsuffixed gpt4o/llama CSVs are
# 20-row pilots and must not be concatenated with the overlays.
ALGO_P2_PHASE1_CLAUDE = RAW / "ALGO_P2_phase1_claude_new.csv"
ALGO_P2_PHASE1_GPT4O = RAW / "ALGO_P2_phase1_gpt4o_new.csv"
ALGO_P2_PHASE1_LLAMA = RAW / "ALGO_P2_phase1_llama_new.csv"
ALGO_P2_PHASE1_GEMINI = RAW / "ALGO_P2_phase1_gemini.csv"
ALGO_P2_PHASE1_PILOT_GPT4O = RAW / "ALGO_P2_phase1_gpt4o.csv"
ALGO_P2_PHASE1_PILOT_LLAMA = RAW / "ALGO_P2_phase1_llama.csv"


def algo_p2_phase1_files() -> list[Path]:
    return [
        ALGO_P2_PHASE1_CLAUDE,
        ALGO_P2_PHASE1_GPT4O,
        ALGO_P2_PHASE1_LLAMA,
        ALGO_P2_PHASE1_GEMINI,
    ]
ALGO_P2_PHASE2_NORMAL = RAW / "ALGO_P2_phase2_normal.csv"
ALGO_P2_PHASE2_INJECTED = RAW / "ALGO_P2_phase2_injected.csv"
ALGO_P3_CONTAMINATION = RAW / "ALGO_P3_contamination.csv"
ALGO_P3_MECHANISTIC = RAW / "ALGO_P3_mechanistic.csv"

ALGO_P1_METRICS = DEPRECATED / "ALGO_P1_metrics.csv"
ALGO_P2_METRICS = DERIVED / "ALGO_P2_metrics.csv"
ALGO_P3_TRIANGULATION = DERIVED / "ALGO_P3_triangulation.csv"

# --- GSM ---
def gsm_p1_behavioral(model: str) -> Path:
    return RAW / f"GSM_P1_behavioral_{model_slug(model)}.csv"


GSM_P1_BEHAVIORAL_CLAUDE = RAW / "GSM_P1_behavioral_claude.csv"
GSM_P1_BEHAVIORAL_GPT4O = RAW / "GSM_P1_behavioral_gpt4o.csv"
GSM_P1_BEHAVIORAL_LLAMA = RAW / "GSM_P1_behavioral_llama.csv"
GSM_P1_BEHAVIORAL_ALL = [GSM_P1_BEHAVIORAL_CLAUDE, GSM_P1_BEHAVIORAL_GPT4O, GSM_P1_BEHAVIORAL_LLAMA]

GSM_P2_CCI = RAW / "GSM_P2_cci.csv"
GSM_P2_REVIEW_QUEUE = RAW / "GSM_P2_review_queue.csv"
GSM_P3_CONTAMINATION = RAW / "GSM_P3_contamination.csv"
GSM_P3_MECHANISTIC = RAW / "GSM_P3_mechanistic.csv"

GSM_P1_METRICS = DEPRECATED / "GSM_P1_metrics.csv"
GSM_P2_METRICS = DERIVED / "GSM_P2_metrics.csv"
GSM_P3_TRIANGULATION_CLAUDE = DERIVED / "GSM_P3_triangulation_claude.csv"
GSM_P3_TRIANGULATION_GPT4O = DERIVED / "GSM_P3_triangulation_gpt4o.csv"

# --- BW ---
BW_P1_BEHAVIORAL = RAW / "BW_P1_behavioral.csv"
BW_P2_PLANS = RAW / "BW_P2_plans.csv"
BW_P2_CCI = RAW / "BW_P2_cci.csv"
BW_P2_TEP = RAW / "BW_P2_tep.csv"
BW_P3_CONTAMINATION = RAW / "BW_P3_contamination.csv"
BW_P3_MECHANISTIC = RAW / "BW_P3_mechanistic.csv"

BW_P1_METRICS = DEPRECATED / "BW_P1_metrics.csv"
BW_P3_TRIANGULATION_CLAUDE = DERIVED / "BW_P3_triangulation_claude.csv"
BW_P3_TRIANGULATION_GPT4O = DERIVED / "BW_P3_triangulation_gpt4o.csv"
BW_P3_TRIANGULATION_LLAMA = DERIVED / "BW_P3_triangulation_llama.csv"

# --- Paper ---
TABLE1_CROSS_FAMILY = PAPER / "TABLE1_cross_family.csv"
CROSS_FAMILY_REGRESSION = PAPER / "cross_family_regression.csv"
PROBE2_CONSOLIDATED = PAPER / "PROBE2_consolidated.csv"
