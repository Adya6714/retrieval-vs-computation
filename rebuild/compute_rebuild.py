#!/usr/bin/env python3
"""Rebuild every paper number from results/raw/ into rebuild/.

Does not write to results/ or paper/. Reads question banks and raw logs only
(plus this package's triangulation_rule.py).
"""
from __future__ import annotations

import json
import math
import re
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
DATA = ROOT / "data" / "problems"
OUT = ROOT / "rebuild"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from probes.algo.decision_normalize import (  # noqa: E402
    normalize_phase1_decision,
    normalize_phase2_decision,
)
from scripts.ALGO_P2_SCR_compute_metrics import (  # noqa: E402
    _first_step_decision,
    _normalize_step_base,
    _optimal_for_step,
    _phase1_intent,
)
from probes.behavioral.retention import (  # noqa: E402
    MIN_CANONICAL_FOR_RETENTION,
    retention_ratio,
)
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.stats import cluster_bootstrap_ci  # noqa: E402
from triangulation_rule import (  # noqa: E402
    APPENDIX_CCI_COMPUTATION_MIN,
    APPENDIX_CCI_RETRIEVAL_MAX,
    APPENDIX_CONTAM_PERCENTILE,
    CCI_THRESHOLDS,
    CONTAM_PERCENTILES,
    PAPER_COUNTS,
    W3_CUTOFFS,
    count_labels,
    label_appendix_three_signal,
    label_default,
    label_legacy_five_field,
    label_sweep_cell,
    matches_paper_counts,
)

RNG = np.random.default_rng(42)
N_BOOT = 10_000

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
TAG = {"Claude": "claude", "Gemini": "gemini", "GPT-4o": "gpt4o", "Llama": "llama", "o4-mini": "o1mini"}
LONG = {v: k for k, v in SHORT.items()}
MODELS = ["Claude", "GPT-4o", "Llama", "Gemini", "o4-mini"]
REAL_MODELS = set(SHORT.keys())
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
ALGO_BW_VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5"]
GEMINI_LONG = "google/gemini-2.5-flash"

# Frozen adversarial pool (paper Table 5/7 challenging cells).
# 34 SP + 10 CC + 17 WIS = 61. Source: Claude P1 difficulty_params_instance_type.
PAPER_ADV = {
    "SP": [
        "SP_003", "SP_004", "SP_005", "SP_019", "SP_020", "SP_021", "SP_023",
        "SP_024", "SP_026", "SP_027", "SP_028", "SP_029", "SP_030", "SP_037",
        "SP_038", "SP_039", "SP_040", "SP_042", "SP_044", "SP_045", "SP_046",
        "SP_047", "SP_048", "SP_062", "SP_063", "SP_064", "SP_065", "SP_066",
        "SP_068", "SP_069", "SP_070", "SP_071", "SP_072", "SP_073",
    ],
    "CC": [f"CC_{i:02d}" for i in range(1, 11)],
    "WIS": [
        "WIS_003", "WIS_004", "WIS_013", "WIS_014", "WIS_015", "WIS_016",
        "WIS_017", "WIS_018", "WIS_019", "WIS_020", "WIS_023", "WIS_024",
        "WIS_025", "WIS_026", "WIS_027", "WIS_028", "WIS_029",
    ],
}
PAPER_ADV_ALL = set(PAPER_ADV["SP"] + PAPER_ADV["CC"] + PAPER_ADV["WIS"])
PID_TO_SUB = {pid: sub for sub, ids in PAPER_ADV.items() for pid in ids}

NUM_COLS = [
    "id", "probe", "phase", "family", "subtype", "model", "variant", "metric",
    "value", "n", "ci_low", "ci_high", "test", "statistic", "p_value",
    "source_file", "filter_applied", "note",
]
ROWS: list[dict] = []

PAPER_TABLE7 = {
    ("GSM", "--", "Claude", "canonical"): 0.841, ("GSM", "--", "Claude", "W1"): 0.841,
    ("GSM", "--", "Claude", "W2"): 0.773, ("GSM", "--", "Claude", "W3"): 0.750,
    ("GSM", "--", "Claude", "W4"): 0.636, ("GSM", "--", "Claude", "W5"): 0.818,
    ("GSM", "--", "Claude", "W6"): 0.750,
    ("GSM", "--", "GPT-4o", "canonical"): 0.850, ("GSM", "--", "GPT-4o", "W1"): 0.750,
    ("GSM", "--", "GPT-4o", "W2"): 0.300, ("GSM", "--", "GPT-4o", "W3"): 0.300,
    ("GSM", "--", "GPT-4o", "W4"): 0.200, ("GSM", "--", "GPT-4o", "W5"): 0.300,
    ("GSM", "--", "GPT-4o", "W6"): 0.800,
    ("GSM", "--", "Gemini", "canonical"): 0.909, ("GSM", "--", "Gemini", "W1"): 0.818,
    ("GSM", "--", "Gemini", "W2"): 0.636, ("GSM", "--", "Gemini", "W3"): 0.523,
    ("GSM", "--", "Gemini", "W4"): 0.477, ("GSM", "--", "Gemini", "W5"): 0.614,
    ("GSM", "--", "Gemini", "W6"): 0.958,
    ("GSM", "--", "Llama", "canonical"): 0.800, ("GSM", "--", "Llama", "W1"): 0.850,
    ("GSM", "--", "Llama", "W2"): 0.250, ("GSM", "--", "Llama", "W3"): 0.150,
    ("GSM", "--", "Llama", "W4"): 0.300, ("GSM", "--", "Llama", "W5"): 0.050,
    ("GSM", "--", "Llama", "W6"): 0.450,
    ("GSM", "--", "o4-mini", "canonical"): 0.841, ("GSM", "--", "o4-mini", "W1"): 0.864,
    ("GSM", "--", "o4-mini", "W2"): 0.818, ("GSM", "--", "o4-mini", "W3"): 0.841,
    ("GSM", "--", "o4-mini", "W4"): 0.682, ("GSM", "--", "o4-mini", "W5"): 0.886,
    ("GSM", "--", "o4-mini", "W6"): 0.833,
    ("ALGO", "CC-chall.", "Claude", "canonical"): 0.700, ("ALGO", "CC-chall.", "Claude", "W1"): 0.700,
    ("ALGO", "CC-chall.", "Claude", "W2"): 0.700, ("ALGO", "CC-chall.", "Claude", "W3"): 0.600,
    ("ALGO", "CC-chall.", "Claude", "W4"): 0.800,
    ("ALGO", "CC-chall.", "GPT-4o", "canonical"): 0.600, ("ALGO", "CC-chall.", "GPT-4o", "W1"): 0.400,
    ("ALGO", "CC-chall.", "GPT-4o", "W2"): 0.600, ("ALGO", "CC-chall.", "GPT-4o", "W3"): 0.000,
    ("ALGO", "CC-chall.", "GPT-4o", "W4"): 0.500,
    ("ALGO", "CC-chall.", "Gemini", "canonical"): 0.500, ("ALGO", "CC-chall.", "Gemini", "W1"): 0.700,
    ("ALGO", "CC-chall.", "Gemini", "W2"): 0.600, ("ALGO", "CC-chall.", "Gemini", "W3"): 0.700,
    ("ALGO", "CC-chall.", "Gemini", "W4"): 0.700,
    ("ALGO", "CC-chall.", "Llama", "canonical"): 0.200, ("ALGO", "CC-chall.", "Llama", "W1"): 0.100,
    ("ALGO", "CC-chall.", "Llama", "W2"): 0.400, ("ALGO", "CC-chall.", "Llama", "W3"): 0.000,
    ("ALGO", "CC-chall.", "Llama", "W4"): 0.200,
    ("ALGO", "CC-std.", "Claude", "canonical"): 0.267, ("ALGO", "CC-std.", "Claude", "W1"): 0.467,
    ("ALGO", "CC-std.", "Claude", "W2"): 0.067, ("ALGO", "CC-std.", "Claude", "W3"): 0.200,
    ("ALGO", "CC-std.", "Claude", "W4"): 0.667, ("ALGO", "CC-std.", "Claude", "W6"): 0.067,
    ("ALGO", "CC-std.", "GPT-4o", "canonical"): 0.267, ("ALGO", "CC-std.", "GPT-4o", "W1"): 0.400,
    ("ALGO", "CC-std.", "GPT-4o", "W2"): 0.000, ("ALGO", "CC-std.", "GPT-4o", "W3"): 0.067,
    ("ALGO", "CC-std.", "GPT-4o", "W4"): 0.867, ("ALGO", "CC-std.", "GPT-4o", "W6"): 0.200,
    ("ALGO", "CC-std.", "Gemini", "canonical"): 0.267, ("ALGO", "CC-std.", "Gemini", "W1"): 0.133,
    ("ALGO", "CC-std.", "Gemini", "W2"): 0.000, ("ALGO", "CC-std.", "Gemini", "W3"): 0.000,
    ("ALGO", "CC-std.", "Gemini", "W4"): 0.267, ("ALGO", "CC-std.", "Gemini", "W6"): 0.267,
    ("ALGO", "CC-std.", "Llama", "canonical"): 0.000, ("ALGO", "CC-std.", "Llama", "W1"): 0.067,
    ("ALGO", "CC-std.", "Llama", "W2"): 0.067, ("ALGO", "CC-std.", "Llama", "W3"): 0.000,
    ("ALGO", "CC-std.", "Llama", "W4"): 0.000, ("ALGO", "CC-std.", "Llama", "W6"): 0.067,
    ("ALGO", "SP-chall.", "Claude", "canonical"): 0.647, ("ALGO", "SP-chall.", "Claude", "W1"): 0.618,
    ("ALGO", "SP-chall.", "Claude", "W2"): 0.676, ("ALGO", "SP-chall.", "Claude", "W3"): 0.000,
    ("ALGO", "SP-chall.", "Claude", "W4"): 0.824, ("ALGO", "SP-chall.", "Claude", "W5"): 0.000,
    ("ALGO", "SP-chall.", "Claude", "W6"): 0.258,
    ("ALGO", "SP-chall.", "GPT-4o", "canonical"): 0.412, ("ALGO", "SP-chall.", "GPT-4o", "W1"): 0.529,
    ("ALGO", "SP-chall.", "GPT-4o", "W2"): 0.147, ("ALGO", "SP-chall.", "GPT-4o", "W3"): 0.265,
    ("ALGO", "SP-chall.", "GPT-4o", "W4"): 0.588, ("ALGO", "SP-chall.", "GPT-4o", "W5"): 0.000,
    ("ALGO", "SP-chall.", "GPT-4o", "W6"): 0.258,
    ("ALGO", "SP-chall.", "Gemini", "canonical"): 0.676, ("ALGO", "SP-chall.", "Gemini", "W1"): 0.441,
    ("ALGO", "SP-chall.", "Gemini", "W2"): 0.235, ("ALGO", "SP-chall.", "Gemini", "W3"): 0.324,
    ("ALGO", "SP-chall.", "Gemini", "W4"): 0.559, ("ALGO", "SP-chall.", "Gemini", "W5"): 0.032,
    ("ALGO", "SP-chall.", "Gemini", "W6"): 0.129,
    ("ALGO", "SP-chall.", "Llama", "canonical"): 0.059, ("ALGO", "SP-chall.", "Llama", "W1"): 0.147,
    ("ALGO", "SP-chall.", "Llama", "W2"): 0.029, ("ALGO", "SP-chall.", "Llama", "W3"): 0.000,
    ("ALGO", "SP-chall.", "Llama", "W4"): 0.088, ("ALGO", "SP-chall.", "Llama", "W5"): 0.000,
    ("ALGO", "SP-chall.", "Llama", "W6"): 0.065,
    ("ALGO", "SP-std.", "Claude", "canonical"): 0.000, ("ALGO", "SP-std.", "Claude", "W1"): 0.190,
    ("ALGO", "SP-std.", "Claude", "W2"): 0.667, ("ALGO", "SP-std.", "Claude", "W3"): 0.048,
    ("ALGO", "SP-std.", "Claude", "W4"): 0.952, ("ALGO", "SP-std.", "Claude", "W5"): 0.000,
    ("ALGO", "SP-std.", "Claude", "W6"): 0.000,
    ("ALGO", "SP-std.", "GPT-4o", "canonical"): 0.714, ("ALGO", "SP-std.", "GPT-4o", "W1"): 0.667,
    ("ALGO", "SP-std.", "GPT-4o", "W2"): 0.048, ("ALGO", "SP-std.", "GPT-4o", "W3"): 0.429,
    ("ALGO", "SP-std.", "GPT-4o", "W4"): 0.524, ("ALGO", "SP-std.", "GPT-4o", "W5"): 0.000,
    ("ALGO", "SP-std.", "GPT-4o", "W6"): 0.368,
    ("ALGO", "SP-std.", "Gemini", "canonical"): 0.619, ("ALGO", "SP-std.", "Gemini", "W1"): 0.762,
    ("ALGO", "SP-std.", "Gemini", "W2"): 0.762, ("ALGO", "SP-std.", "Gemini", "W3"): 0.476,
    ("ALGO", "SP-std.", "Gemini", "W4"): 0.857, ("ALGO", "SP-std.", "Gemini", "W5"): 0.000,
    ("ALGO", "SP-std.", "Gemini", "W6"): 0.263,
    ("ALGO", "SP-std.", "Llama", "canonical"): 0.048, ("ALGO", "SP-std.", "Llama", "W1"): 0.095,
    ("ALGO", "SP-std.", "Llama", "W2"): 0.000, ("ALGO", "SP-std.", "Llama", "W3"): 0.000,
    ("ALGO", "SP-std.", "Llama", "W4"): 0.143, ("ALGO", "SP-std.", "Llama", "W5"): 0.000,
    ("ALGO", "SP-std.", "Llama", "W6"): 0.105,
    ("ALGO", "WIS-chall.", "Claude", "canonical"): 0.353, ("ALGO", "WIS-chall.", "Claude", "W1"): 0.176,
    ("ALGO", "WIS-chall.", "Claude", "W2"): 0.118, ("ALGO", "WIS-chall.", "Claude", "W3"): 0.000,
    ("ALGO", "WIS-chall.", "Claude", "W4"): 0.059, ("ALGO", "WIS-chall.", "Claude", "W6"): 0.000,
    ("ALGO", "WIS-chall.", "GPT-4o", "canonical"): 0.353, ("ALGO", "WIS-chall.", "GPT-4o", "W1"): 0.176,
    ("ALGO", "WIS-chall.", "GPT-4o", "W2"): 0.000, ("ALGO", "WIS-chall.", "GPT-4o", "W3"): 0.000,
    ("ALGO", "WIS-chall.", "GPT-4o", "W4"): 0.000, ("ALGO", "WIS-chall.", "GPT-4o", "W6"): 0.000,
    ("ALGO", "WIS-chall.", "Gemini", "canonical"): 0.353, ("ALGO", "WIS-chall.", "Gemini", "W1"): 0.176,
    ("ALGO", "WIS-chall.", "Gemini", "W2"): 0.000, ("ALGO", "WIS-chall.", "Gemini", "W3"): 0.000,
    ("ALGO", "WIS-chall.", "Gemini", "W4"): 0.000, ("ALGO", "WIS-chall.", "Gemini", "W6"): 0.000,
    ("ALGO", "WIS-chall.", "Llama", "canonical"): 0.059, ("ALGO", "WIS-chall.", "Llama", "W1"): 0.000,
    ("ALGO", "WIS-chall.", "Llama", "W2"): 0.000, ("ALGO", "WIS-chall.", "Llama", "W3"): 0.059,
    ("ALGO", "WIS-chall.", "Llama", "W4"): 0.000, ("ALGO", "WIS-chall.", "Llama", "W6"): 0.000,
    ("ALGO", "WIS-std.", "Claude", "canonical"): 0.077, ("ALGO", "WIS-std.", "Claude", "W1"): 0.231,
    ("ALGO", "WIS-std.", "Claude", "W2"): 0.231, ("ALGO", "WIS-std.", "Claude", "W3"): 0.000,
    ("ALGO", "WIS-std.", "Claude", "W4"): 0.077, ("ALGO", "WIS-std.", "Claude", "W6"): 0.000,
    ("ALGO", "WIS-std.", "GPT-4o", "canonical"): 0.154, ("ALGO", "WIS-std.", "GPT-4o", "W1"): 0.231,
    ("ALGO", "WIS-std.", "GPT-4o", "W2"): 0.000, ("ALGO", "WIS-std.", "GPT-4o", "W3"): 0.000,
    ("ALGO", "WIS-std.", "GPT-4o", "W4"): 0.000, ("ALGO", "WIS-std.", "GPT-4o", "W6"): 0.000,
    ("ALGO", "WIS-std.", "Gemini", "canonical"): 0.000, ("ALGO", "WIS-std.", "Gemini", "W1"): 0.000,
    ("ALGO", "WIS-std.", "Gemini", "W2"): 0.231, ("ALGO", "WIS-std.", "Gemini", "W3"): 0.000,
    ("ALGO", "WIS-std.", "Gemini", "W4"): 0.000, ("ALGO", "WIS-std.", "Gemini", "W6"): 0.000,
    ("ALGO", "WIS-std.", "Llama", "canonical"): 0.000, ("ALGO", "WIS-std.", "Llama", "W1"): 0.000,
    ("ALGO", "WIS-std.", "Llama", "W2"): 0.000, ("ALGO", "WIS-std.", "Llama", "W3"): 0.077,
    ("ALGO", "WIS-std.", "Llama", "W4"): 0.000, ("ALGO", "WIS-std.", "Llama", "W6"): 0.000,
    ("BW", "--", "Claude", "canonical"): 0.154, ("BW", "--", "Claude", "W1"): 0.062,
    ("BW", "--", "Claude", "W2"): 0.231, ("BW", "--", "Claude", "W3"): 0.138,
    ("BW", "--", "Claude", "W4"): 0.015, ("BW", "--", "Claude", "W5"): 0.523,
    ("BW", "--", "Claude", "W6"): 0.508,
    ("BW", "--", "GPT-4o", "canonical"): 0.062, ("BW", "--", "GPT-4o", "W1"): 0.092,
    ("BW", "--", "GPT-4o", "W2"): 0.092, ("BW", "--", "GPT-4o", "W3"): 0.169,
    ("BW", "--", "GPT-4o", "W4"): 0.077, ("BW", "--", "GPT-4o", "W5"): 0.246,
    ("BW", "--", "GPT-4o", "W6"): 0.215,
    ("BW", "--", "Gemini", "canonical"): 0.385, ("BW", "--", "Gemini", "W1"): 0.138,
    ("BW", "--", "Gemini", "W2"): 0.108, ("BW", "--", "Gemini", "W3"): 0.108,
    ("BW", "--", "Gemini", "W4"): 0.031, ("BW", "--", "Gemini", "W5"): 0.569,
    ("BW", "--", "Gemini", "W6"): 0.338,
    ("BW", "--", "Llama", "canonical"): 0.015, ("BW", "--", "Llama", "W1"): 0.031,
    ("BW", "--", "Llama", "W2"): 0.015, ("BW", "--", "Llama", "W3"): 0.108,
    ("BW", "--", "Llama", "W4"): 0.000, ("BW", "--", "Llama", "W5"): 0.000,
    ("BW", "--", "Llama", "W6"): 0.031,
    ("BW", "--", "o4-mini", "canonical"): 0.769, ("BW", "--", "o4-mini", "W1"): 0.754,
    ("BW", "--", "o4-mini", "W2"): 0.738, ("BW", "--", "o4-mini", "W3"): 0.185,
    ("BW", "--", "o4-mini", "W4"): 0.415, ("BW", "--", "o4-mini", "W5"): 0.769,
    ("BW", "--", "o4-mini", "W6"): 0.769,
}

PAPER_CLAIMS: list[tuple[str, str, float | str]] = [
    ("P3.pool_n", "§4.3 proximity pool n", 64),
    ("T.legacy_retrieval", "Table 9 / appendix 8 retrieval", 8),
    ("T.legacy_computation", "Table 9 / appendix 4 computation", 4),
    ("T.legacy_mixed", "appendix mixed 157", 157),
    ("T.legacy_ambiguous", "appendix ambiguous 271", 271),
    ("P2.gpt4o_empty_acc", "§4.2 GPT-4o empty-declaration Acc", 0.69),
    ("P2.gpt4o_diverged_acc", "§4.2 GPT-4o diverged Acc", 0.73),
    ("P3.claude_instance_r", "§4.3 Claude proximity-VRI r (labelled template)", 0.44),
    ("P3.gpt4o_instance_r", "§4.3 GPT-4o proximity-VRI r (labelled template)", 0.37),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().isin({"true", "1", "yes"})


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    raw = df.get("raw_response", df.get("model_answer", pd.Series([""] * len(df), index=df.index)))
    return ~raw.astype(str).str.startswith("ERROR:")


def _drop_mock(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df
    m = df["model"].astype(str).str.strip()
    return df[~m.str.lower().isin({"mock", ""})].copy()


def _short(model: str) -> str:
    return SHORT.get(str(model).strip(), str(model).strip())


def _correct(df: pd.DataFrame) -> pd.Series:
    for c in ["verified", "behavioral_correct", "final_answer_correct"]:
        if c in df.columns:
            return _is_true(df[c])
    return pd.Series(False, index=df.index)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    if k <= 0:
        den = 1 + z ** 2 / n
        hi = (z ** 2 / (2 * n) + z * math.sqrt(z ** 2 / (4 * n ** 2))) / den
        return 0.0, min(1.0, hi)
    if k >= n:
        den = 1 + z ** 2 / n
        lo = (1 + z ** 2 / (2 * n) - z * math.sqrt(z ** 2 / (4 * n ** 2))) / den
        return max(0.0, lo), 1.0
    p = k / n
    den = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / den
    marg = z * math.sqrt(max(p * (1 - p) / n + z ** 2 / (4 * n ** 2), 0.0)) / den
    return max(0.0, center - marg), min(1.0, center + marg)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return "NA"
    p = float(p)
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def add(**kwargs) -> None:
    row = {c: "" for c in NUM_COLS}
    row.update(kwargs)
    if row.get("value") is None or (isinstance(row.get("value"), float) and math.isnan(row["value"])):
        if not row.get("note"):
            row["value"] = "NOT_COMPUTABLE"
        elif row.get("value") != "NOT_COMPUTABLE" and row.get("value") == "":
            row["value"] = "NOT_COMPUTABLE"
    ROWS.append(row)


def _retention_value(a_w3, a_can, n_can, n_w3):
    if not n_can or not n_w3:
        return "NOT_COMPUTABLE", "missing canonical or W3"
    ret, reason = retention_ratio(a_w3, a_can)
    if ret is None:
        note = reason
        if reason == "canonical_below_floor":
            note = (
                f"canonical_below_floor "
                f"(MIN_CANONICAL_FOR_RETENTION={MIN_CANONICAL_FOR_RETENTION})"
            )
        return "undefined", note
    return ret, ""


def add_nc(id_: str, *, probe, phase="", family="", subtype="", model="", variant="",
           metric="", source_file="", filter_applied="", note="") -> None:
    add(id=id_, probe=probe, phase=phase, family=family, subtype=subtype, model=model,
        variant=variant, metric=metric, value="NOT_COMPUTABLE", n="",
        source_file=source_file, filter_applied=filter_applied, note=note)


def _boot_acc_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    diffs = np.empty(N_BOOT, dtype=float)
    idx = np.arange(n)
    for i in range(N_BOOT):
        draw = RNG.choice(idx, size=n, replace=True)
        diffs[i] = float(a[draw].mean() - b[draw].mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def _partial_pearson(x, y, z) -> tuple[float, float, int]:
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), n
    if np.nanstd(z) == 0:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p), n
    xz = np.polyfit(z, x, 1)
    yz = np.polyfit(z, y, 1)
    xr = x - np.polyval(xz, z)
    yr = y - np.polyval(yz, z)
    if np.nanstd(xr) == 0 or np.nanstd(yr) == 0:
        return float("nan"), float("nan"), n
    r, p = stats.pearsonr(xr, yr)
    return float(r), float(p), n


def _algo_subtype(pid: str) -> str:
    if pid.startswith("CC"):
        return "CC"
    if pid.startswith("SP"):
        return "SP"
    if pid.startswith("WIS"):
        return "WIS"
    return ""


def _algo_slice(pid: str) -> str:
    sub = _algo_subtype(pid)
    if not sub:
        return ""
    kind = "chall" if pid in PAPER_ADV_ALL else "std"
    return f"{sub}-{kind}"


# ---------------------------------------------------------------------------
# Banks / frozen lists
# ---------------------------------------------------------------------------

def load_banks() -> dict:
    gsm = _read(DATA / "question_bank_gsm.csv")
    gsm["variant_type"] = gsm["variant_type"].map(_norm_variant)
    gsm_canon = sorted(gsm.loc[gsm["variant_type"] == "canonical", "problem_id"].unique())

    algo = _read(DATA / "question_bank_algo.csv")
    algo["variant_type"] = algo["variant_type"].map(_norm_variant)
    algo_can = algo[algo["variant_type"] == "canonical"].drop_duplicates("problem_id")

    bw = _read(DATA / "question_bank_bw.csv")
    bw["variant_type"] = bw["variant_type"].map(_norm_variant)
    bw_canon = sorted(bw.loc[bw["variant_type"] == "canonical", "problem_id"].unique())
    bw_std = [x for x in bw_canon if x.startswith("BW_")]
    bw_mys = [x for x in bw_canon if x.startswith("MBW_")]

    params = {}
    for _, r in algo_can.iterrows():
        try:
            params[str(r["problem_id"])] = json.loads(str(r.get("difficulty_params") or "{}"))
        except json.JSONDecodeError:
            params[str(r["problem_id"])] = {}

    return {
        "gsm_canon": gsm_canon,
        "algo_canon_ids": sorted(algo_can["problem_id"].unique()),
        "algo_can": algo_can,
        "algo_params": params,
        "bw_canon": bw_canon,
        "bw_std": bw_std,
        "bw_mystery": bw_mys,
        "gsm_bank": gsm,
        "algo_bank": algo,
        "bw_bank": bw,
    }


# ---------------------------------------------------------------------------
# P1 loaders
# ---------------------------------------------------------------------------

def load_algo_p1(tag: str) -> pd.DataFrame:
    df = _read(RAW / f"ALGO_P1_behavioral_{tag}.csv")
    if df.empty:
        return df
    df = _drop_mock(df)
    df = df[df["model"].isin(REAL_MODELS)] if "model" in df.columns else df
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    df = df[_valid_mask(df)]
    df = df.drop_duplicates(["problem_id", "variant_type"], keep="last")
    df["ok"] = _correct(df)
    df["model_short"] = df["model"].map(_short) if "model" in df.columns else tag
    df["subtype"] = df["problem_id"].map(_algo_subtype)
    df["slice"] = df["problem_id"].map(_algo_slice)
    return filter_excluded(df, family="ALGO")


def load_gsm_p1(tag: str) -> pd.DataFrame:
    df = _read(RAW / f"GSM_P1_behavioral_{tag}.csv")
    if df.empty:
        return df
    df = _drop_mock(df)
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    df = filter_p1_to_bank(df, "GSM")
    df = df[_valid_mask(df)]
    df = df.drop_duplicates(["problem_id", "variant_type"], keep="last")
    df["ok"] = _correct(df)
    df["model_short"] = df["model"].map(_short) if "model" in df.columns else TAG.get(tag, tag)
    return df


def load_bw_p1(bw_ids: set[str]) -> pd.DataFrame:
    parts = []
    for name in ["BW_P1_behavioral.csv", "BW_P1_behavioral_gemini.csv", "BW_P1_behavioral_o1mini.csv"]:
        df = _read(RAW / name)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = _drop_mock(df)
    df = df[df["model"].isin(REAL_MODELS)]
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    df = df[df["problem_id"].astype(str).isin(bw_ids)]
    df = df[_valid_mask(df)]
    df = df.drop_duplicates(["problem_id", "model", "variant_type"], keep="last")
    df["ok"] = _correct(df)
    df["model_short"] = df["model"].map(_short)
    df["subtype"] = np.where(df["problem_id"].str.startswith("MBW_"), "mystery", "standard")
    return filter_excluded(df, family="BW")


def _acc_row(sub: pd.DataFrame, *, cluster: bool = False) -> tuple[int, int, float, float, float]:
    n = int(len(sub))
    k = int(sub["ok"].sum()) if n else 0
    acc = k / n if n else float("nan")
    if n and cluster and "problem_id" in sub.columns:
        vals = sub["ok"].astype(float).tolist()
        lo, hi = cluster_bootstrap_ci(
            vals,
            cluster_ids_for(sub["problem_id"].astype(str).tolist()),
            seed=42,
        )
    else:
        lo, hi = wilson(k, n)
    return k, n, acc, lo, hi


# ---------------------------------------------------------------------------
# Probe 1
# ---------------------------------------------------------------------------

def run_p1(banks: dict, algo: dict[str, pd.DataFrame], gsm: dict[str, pd.DataFrame], bw: pd.DataFrame) -> None:
    # P1.1 accuracy grid
    for m in MODELS:
        df = gsm[m]
        src = f"GSM_P1_behavioral_{TAG[m]}.csv"
        filt = "bank-valid GSM IDs from question_bank_gsm.csv; drop mock; drop ERROR:; keep=last"
        for vt in VARIANTS:
            sub = df[df["variant_type"] == vt]
            k, n, acc, lo, hi = _acc_row(sub)
            add(id=f"P1.1.GSM.{m}.{vt}", probe="P1", phase="P1", family="GSM", subtype="--",
                model=m, variant=vt, metric="accuracy", value=acc if n else "NOT_COMPUTABLE",
                n=n, ci_low=lo if n else "", ci_high=hi if n else "",
                source_file=src, filter_applied=filt,
                note="" if n else "no valid rows after bank+ERROR filter")
        # family-level W3 retention
        can = df[df["variant_type"] == "canonical"]
        w3 = df[df["variant_type"] == "W3"]
        _, n_can, a_can, _, _ = _acc_row(can)
        _, n_w3, a_w3, _, _ = _acc_row(w3)
        ret, note = _retention_value(a_w3, a_can, n_can, n_w3)
        add(id=f"P1.2.GSM.{m}.W3_retention", probe="P1", phase="P1", family="GSM", subtype="--",
            model=m, variant="W3", metric="W3_retention", value=ret, n=min(n_can, n_w3),
            source_file=src, filter_applied=filt, note=note)

    for m in MODELS:
        df = algo[m]
        src = f"ALGO_P1_behavioral_{TAG[m]}.csv"
        filt = "drop mock; drop ERROR:; keep=last; subtype/slice from frozen 61-ID adversarial pool"
        for slice_name in ["CC-chall", "CC-std", "SP-chall", "SP-std", "WIS-chall", "WIS-std"]:
            for vt in ALGO_BW_VARIANTS:
                sub = df[(df["slice"] == slice_name) & (df["variant_type"] == vt)]
                k, n, acc, lo, hi = _acc_row(sub, cluster=True)
                val = acc if n else "NOT_COMPUTABLE"
                add(id=f"P1.1.ALGO.{slice_name}.{m}.{vt}", probe="P1", phase="P1", family="ALGO",
                    subtype=slice_name, model=m, variant=vt, metric="accuracy",
                    value=val, n=n, ci_low=lo if n else "", ci_high=hi if n else "",
                    source_file=src, filter_applied=filt + "; CI=10k clone-family cluster bootstrap seed=42",
                    note="" if n else "variant not present for this slice (W5/W6 holes are real)")
        for sub_name in ["CC", "SP", "WIS", "ALL"]:
            ids = PAPER_ADV[sub_name] if sub_name != "ALL" else list(PAPER_ADV_ALL)
            can = df[(df["variant_type"] == "canonical") & df["problem_id"].isin(ids)]
            w3 = df[(df["variant_type"] == "W3") & df["problem_id"].isin(ids)]
            _, n_can, a_can, _, _ = _acc_row(can, cluster=True)
            _, n_w3, a_w3, _, _ = _acc_row(w3, cluster=True)
            ret, note = _retention_value(a_w3, a_can, n_can, n_w3)
            add(id=f"P1.2.ALGO.{sub_name}.{m}.W3_retention", probe="P1", phase="P1", family="ALGO",
                subtype=sub_name, model=m, variant="W3", metric="W3_retention",
                value=ret, n=min(n_can, n_w3), source_file=src, filter_applied=filt, note=note)

    bw_src = "BW_P1_behavioral.csv + BW_P1_behavioral_gemini.csv + BW_P1_behavioral_o1mini.csv"
    bw_filt = "65 PlanBench bank IDs; drop mock; drop ERROR:; keep=last"
    for m in MODELS:
        df = bw[bw["model_short"] == m] if not bw.empty else pd.DataFrame()
        for subtype, sl in [("--", None), ("standard", "standard"), ("mystery", "mystery")]:
            base = df if sl is None else df[df["subtype"] == sl]
            for vt in ALGO_BW_VARIANTS:
                sub = base[base["variant_type"] == vt] if not base.empty else pd.DataFrame()
                k, n, acc, lo, hi = _acc_row(sub)
                metric = "accuracy"
                extra = ""
                if vt == "W3":
                    extra = "W3 = entity+action rename (scripts/generation/utils/variant_prompts.py W3_BW_MAPPING_*)"
                elif vt == "W5":
                    extra = "W5 = init/goal swap (scripts/generation/utils/variant_utils.py swap_pddl_init_goal)"
                add(id=f"P1.1.BW.{subtype}.{m}.{vt}", probe="P1", phase="P1", family="BW",
                    subtype=subtype, model=m, variant=vt, metric=metric,
                    value=acc if n else "NOT_COMPUTABLE", n=n,
                    ci_low=lo if n else "", ci_high=hi if n else "",
                    source_file=bw_src, filter_applied=bw_filt, note=extra)
            can = base[base["variant_type"] == "canonical"] if not base.empty else pd.DataFrame()
            w3 = base[base["variant_type"] == "W3"] if not base.empty else pd.DataFrame()
            _, n_can, a_can, _, _ = _acc_row(can)
            _, n_w3, a_w3, _, _ = _acc_row(w3)
            ret, note = _retention_value(a_w3, a_can, n_can, n_w3)
            add(id=f"P1.2.BW.{subtype}.{m}.W3_retention", probe="P1", phase="P1", family="BW",
                subtype=subtype, model=m, variant="W3", metric="W3_retention",
                value=ret, n=min(n_can, n_w3), source_file=bw_src, filter_applied=bw_filt, note=note)

        # P1.4 labelled W3 / W5
        for vt, label in [("W3", "entity_action_rename"), ("W5", "init_goal_swap")]:
            sub = df[df["variant_type"] == vt] if not df.empty else pd.DataFrame()
            k, n, acc, lo, hi = _acc_row(sub)
            add(id=f"P1.4.BW.{m}.{label}", probe="P1", phase="P1", family="BW", subtype="--",
                model=m, variant=vt, metric=f"accuracy_{label}",
                value=acc if n else "NOT_COMPUTABLE", n=n, ci_low=lo if n else "", ci_high=hi if n else "",
                source_file=bw_src, filter_applied=bw_filt,
                note="Confirmed from generator: W3=entity+action rename; W5=init/goal swap")

    # P1.3 VRI = mean(W1,W2,W4) - W3 per problem per model
    vri_rows = []
    for family, by_m, src_fn in [
        ("GSM", gsm, lambda m: f"GSM_P1_behavioral_{TAG[m]}.csv"),
        ("ALGO", algo, lambda m: f"ALGO_P1_behavioral_{TAG[m]}.csv"),
    ]:
        for m in MODELS:
            df = by_m[m]
            if df.empty:
                continue
            wide = df.pivot_table(index="problem_id", columns="variant_type", values="ok", aggfunc="max")
            for c in ["W1", "W2", "W3", "W4", "canonical"]:
                if c not in wide.columns:
                    wide[c] = np.nan
            wide["VRI"] = wide[["W1", "W2", "W4"]].mean(axis=1) - wide["W3"]
            wide = wide.reset_index()
            vals = pd.to_numeric(wide["VRI"], errors="coerce").dropna()
            add(id=f"P1.3.{family}.{m}.VRI_mean", probe="P1", phase="P1", family=family, subtype="--",
                model=m, variant="W1W2W4-W3", metric="VRI_mean",
                value=float(vals.mean()) if len(vals) else "NOT_COMPUTABLE", n=int(len(vals)),
                source_file=src_fn(m),
                filter_applied="per-problem 0/1; VRI=mean(W1,W2,W4)-W3",
                note="" if len(vals) else "no complete W1/W2/W3/W4 tuples")
            for _, r in wide.iterrows():
                vri_rows.append({
                    "family": family, "model": m, "problem_id": r["problem_id"],
                    "canonical": r.get("canonical", np.nan), "W1": r.get("W1", np.nan),
                    "W2": r.get("W2", np.nan), "W3": r.get("W3", np.nan),
                    "W4": r.get("W4", np.nan), "VRI": r.get("VRI", np.nan),
                    "subtype": _algo_subtype(str(r["problem_id"])) if family == "ALGO" else "",
                })
    for m in MODELS:
        df = bw[bw["model_short"] == m] if not bw.empty else pd.DataFrame()
        if df.empty:
            continue
        wide = df.pivot_table(index="problem_id", columns="variant_type", values="ok", aggfunc="max")
        for c in ["W1", "W2", "W3", "W4", "canonical"]:
            if c not in wide.columns:
                wide[c] = np.nan
        wide["VRI"] = wide[["W1", "W2", "W4"]].mean(axis=1) - wide["W3"]
        wide = wide.reset_index()
        vals = pd.to_numeric(wide["VRI"], errors="coerce").dropna()
        add(id=f"P1.3.BW.{m}.VRI_mean", probe="P1", phase="P1", family="BW", subtype="--",
            model=m, variant="W1W2W4-W3", metric="VRI_mean",
            value=float(vals.mean()) if len(vals) else "NOT_COMPUTABLE", n=int(len(vals)),
            source_file=bw_src, filter_applied=bw_filt)
        for _, r in wide.iterrows():
            vri_rows.append({
                "family": "BW", "model": m, "problem_id": r["problem_id"],
                "canonical": r.get("canonical", np.nan), "W1": r.get("W1", np.nan),
                "W2": r.get("W2", np.nan), "W3": r.get("W3", np.nan),
                "W4": r.get("W4", np.nan), "VRI": r.get("VRI", np.nan), "subtype": "",
            })
    pd.DataFrame(vri_rows).to_csv(OUT / "p1_vri_per_problem.csv", index=False)

    # P1.5 all-pairs inversion (ALGO subtypes, frozen 61)
    inv_rows = []
    maps = {}
    for m in MODELS:
        w3, can = {}, {}
        for _, r in algo[m].iterrows():
            pid = str(r["problem_id"])
            ok = int(bool(r["ok"]))
            if r["variant_type"] == "W3":
                w3[pid] = ok
            elif r["variant_type"] == "canonical":
                can[pid] = ok
        maps[m] = (w3, can)
    for subtype, ids in PAPER_ADV.items():
        idset = set(ids)
        for ma, mb in combinations(MODELS, 2):
            w3a, cana = maps[ma]
            w3b, canb = maps[mb]
            paired_ids = sorted(idset & set(w3a) & set(w3b))
            matched_ids = sorted(pid for pid in paired_ids if cana.get(pid) == 1 and canb.get(pid) == 1)
            for definition, use_ids in (("paired", paired_ids), ("canonically-matched", matched_ids)):
                n = len(use_ids)
                src = f"ALGO_P1_behavioral_{TAG[ma]}.csv + ALGO_P1_behavioral_{TAG[mb]}.csv"
                filt = f"frozen {subtype} adversarial IDs; {definition}; drop mock"
                if n == 0:
                    add(id=f"P1.5.{subtype}.{ma}_vs_{mb}.{definition}.fisher_p",
                        probe="P1", phase="P1", family="ALGO", subtype=subtype,
                        model=f"{ma}|{mb}", variant="W3", metric="fisher_p",
                        value="NOT_COMPUTABLE", n=0, test="fisher_exact",
                        source_file=src, filter_applied=filt, note="empty ID intersection")
                    continue
                aa = np.array([w3a[pid] for pid in use_ids], dtype=int)
                bb = np.array([w3b[pid] for pid in use_ids], dtype=int)
                ka, kb = int(aa.sum()), int(bb.sum())
                table = np.array([[ka, n - ka], [kb, n - kb]], dtype=int)
                try:
                    stat, p = stats.fisher_exact(table, alternative="two-sided")
                except ValueError:
                    stat, p = float("nan"), float("nan")
                lo, hi = _boot_acc_diff(aa.astype(float), bb.astype(float))
                diff = float(aa.mean() - bb.mean())
                add(id=f"P1.5.{subtype}.{ma}_vs_{mb}.{definition}.fisher_p",
                    probe="P1", phase="P1", family="ALGO", subtype=subtype,
                    model=f"{ma}|{mb}", variant="W3", metric="fisher_p",
                    value=p, n=n, test="fisher_exact", statistic=stat, p_value=p,
                    source_file=src, filter_applied=filt)
                add(id=f"P1.5.{subtype}.{ma}_vs_{mb}.{definition}.acc_diff",
                    probe="P1", phase="P1", family="ALGO", subtype=subtype,
                    model=f"{ma}|{mb}", variant="W3", metric="acc_diff_A_minus_B",
                    value=diff, n=n, ci_low=lo, ci_high=hi, test="bootstrap_10000",
                    source_file=src, filter_applied=filt)
                inv_rows.append({
                    "subtype": subtype, "model_a": ma, "model_b": mb, "definition": definition,
                    "n": n, "a_W3": ka, "b_W3": kb, "fisher_p": p, "acc_diff": diff,
                    "ci_lo": lo, "ci_hi": hi,
                })
    pd.DataFrame(inv_rows).to_csv(OUT / "p1_pairwise_inversion.csv", index=False)

    # P1.6 within-model phi
    for family, by_m, src_fn in [
        ("GSM", gsm, lambda m: f"GSM_P1_behavioral_{TAG[m]}.csv"),
        ("ALGO", algo, lambda m: f"ALGO_P1_behavioral_{TAG[m]}.csv"),
    ]:
        for m in MODELS:
            df = by_m[m]
            wide = df.pivot_table(index="problem_id", columns="variant_type", values="ok", aggfunc="max")
            if "canonical" not in wide.columns or "W3" not in wide.columns:
                add_nc(f"P1.6.{family}.{m}.phi", probe="P1", family=family, model=m, metric="phi",
                       source_file=src_fn(m), note="missing canonical or W3")
                continue
            pair = wide[["canonical", "W3"]].dropna()
            if len(pair) < 2 or pair["canonical"].nunique() < 2 or pair["W3"].nunique() < 2:
                r, p = float("nan"), float("nan")
                note = "degenerate 2x2 (zero variance)"
            else:
                r, p = stats.pearsonr(pair["canonical"].astype(float), pair["W3"].astype(float))
                note = "Pearson on 0/1 = phi"
            add(id=f"P1.6.{family}.{m}.phi", probe="P1", phase="P1", family=family, subtype="--",
                model=m, variant="canonical_vs_W3", metric="phi",
                value=r if r == r else "NOT_COMPUTABLE", n=int(len(pair)),
                test="pearson_phi", statistic=r, p_value=p,
                source_file=src_fn(m), filter_applied="per-problem paired canonical vs W3",
                note=note)
    for m in MODELS:
        df = bw[bw["model_short"] == m] if not bw.empty else pd.DataFrame()
        if df.empty:
            add_nc(f"P1.6.BW.{m}.phi", probe="P1", family="BW", model=m, metric="phi",
                   source_file=bw_src, note="no BW P1 rows")
            continue
        wide = df.pivot_table(index="problem_id", columns="variant_type", values="ok", aggfunc="max")
        pair = wide[["canonical", "W3"]].dropna() if {"canonical", "W3"} <= set(wide.columns) else pd.DataFrame()
        if len(pair) < 2 or pair["canonical"].nunique() < 2 or pair["W3"].nunique() < 2:
            r, p, note = float("nan"), float("nan"), "degenerate 2x2"
        else:
            r, p = stats.pearsonr(pair["canonical"].astype(float), pair["W3"].astype(float))
            note = "Pearson on 0/1 = phi"
        add(id=f"P1.6.BW.{m}.phi", probe="P1", phase="P1", family="BW", subtype="--",
            model=m, variant="canonical_vs_W3", metric="phi",
            value=r if r == r else "NOT_COMPUTABLE", n=int(len(pair)),
            test="pearson_phi", statistic=r, p_value=p,
            source_file=bw_src, filter_applied=bw_filt, note=note)


# ---------------------------------------------------------------------------
# Probe 2 loaders
# ---------------------------------------------------------------------------

def load_phase1_algo() -> pd.DataFrame:
    order = [
        "ALGO_P2_phase1_claude_new.csv",
        "ALGO_P2_phase1_gpt4o_new.csv",
        "ALGO_P2_phase1_llama_new.csv",
        "ALGO_P2_phase1_gemini.csv",
    ]
    parts = [_drop_mock(_read(RAW / n)) for n in order]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(["problem_id", "model"], keep="last")


def load_phase2_normal(*, overlay_gemini: bool = True, keep_rest: bool = False) -> pd.DataFrame:
    main = _drop_mock(_read(RAW / "ALGO_P2_phase2_normal.csv"))
    gem = _drop_mock(_read(RAW / "ALGO_P2_phase2_normal_gemini.csv"))
    if main.empty:
        return gem
    if overlay_gemini:
        main_no = main[main["model"] != GEMINI_LONG]
        if keep_rest and not gem.empty:
            gem_ids = set(gem["problem_id"].astype(str))
            rest = main[(main["model"] == GEMINI_LONG) & ~main["problem_id"].astype(str).isin(gem_ids)]
            parts = [p for p in (main_no, rest, gem) if not p.empty]
            return pd.concat(parts, ignore_index=True)
        parts = [p for p in (main_no, gem) if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return main


def load_phase2_injected() -> pd.DataFrame:
    main = _drop_mock(_read(RAW / "ALGO_P2_phase2_injected.csv"))
    gem = _drop_mock(_read(RAW / "ALGO_P2_phase2_injected_gemini.csv"))
    if not main.empty:
        main = main[main["model"] != GEMINI_LONG]
    parts = [p for p in (main, gem) if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_gsm_p2() -> pd.DataFrame:
    p2 = _drop_mock(_read(RAW / "GSM_P2_cci.csv"))
    o4 = _drop_mock(_read(RAW / "GSM_P2_phase1_o1mini.csv"))
    if o4.empty:
        out = p2
    else:
        keep = [c for c in p2.columns if c in o4.columns] if not p2.empty else list(o4.columns)
        o4 = o4.copy()
        if "model" not in o4.columns or o4["model"].eq("").all():
            o4["model"] = LONG["o4-mini"]
        out = o4 if p2.empty else pd.concat([p2, o4[keep]], ignore_index=True)
    if out.empty:
        return out
    if "session_b_correct" in out.columns and "either_session_correct" not in out.columns:
        out = out.rename(columns={"session_b_correct": "either_session_correct"})
    ov = _read(ROOT / "results/derived/GSM_P2_session_correct.csv")
    if not ov.empty:
        keep = [
            c
            for c in (
                "problem_id",
                "model",
                "either_session_correct",
                "phase1_correct",
                "phase2a_correct",
                "phase2b_correct",
            )
            if c in ov.columns
        ]
        ov = ov[keep].drop_duplicates(["problem_id", "model"])
        out = out.drop(
            columns=[c for c in ("either_session_correct", "phase1_correct", "phase2a_correct", "phase2b_correct") if c in out.columns],
            errors="ignore",
        )
        out = out.merge(ov, on=["problem_id", "model"], how="left")
    return out


def _last_step(df: pd.DataFrame, correct_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_step"] = pd.to_numeric(out.get("step_index", 0), errors="coerce").fillna(0)
    last = out.sort_values("_step").groupby(["problem_id", "model"], as_index=False).tail(1)
    last["ok"] = _is_true(last[correct_col]) if correct_col in last.columns else False
    return last


def _bank_subtype(pid: str) -> str:
    return {"CC": "coin_change", "SP": "shortest_path", "WIS": "wis"}.get(_algo_subtype(str(pid)), "")


def _critical_step(params: dict, pid: str) -> int:
    raw = params.get(pid, {}).get("critical_step_index", -1)
    if str(raw).strip() == "":
        return -1
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def compute_algo_cci(phase1: pd.DataFrame, normal: pd.DataFrame, banks: dict) -> pd.DataFrame:
    """Recompute per-instance CCI from raw Phase-1 + Phase-2A (frozen adversarial 61).

    Matches ``scripts/ALGO_P2_SCR_compute_metrics.py`` (step-base 0, bank subtype
    names, ``_first_step_decision``, critical_step_index=0 is valid).
    """
    if phase1.empty or normal.empty:
        return pd.DataFrame()
    bank = banks["algo_can"]
    params = banks["algo_params"]
    p1 = phase1.copy()
    p2 = normal.copy()
    p2["step_index_int"] = pd.to_numeric(p2["step_index"], errors="coerce")
    p2 = p2[p2["step_index_int"].notna()].copy()
    p2["step_index_int"] = p2["step_index_int"].astype(int)
    p2 = _normalize_step_base(p2)
    gold_map = dict(zip(bank["problem_id"].astype(str), bank["correct_answer"].astype(str)))
    rows = []
    for (pid, model), p1g in p1.groupby(["problem_id", "model"]):
        pid = str(pid)
        if pid not in PAPER_ADV_ALL:
            continue
        p1row = p1g.iloc[0]
        a_n = p2[(p2["problem_id"].astype(str) == pid) & (p2["model"] == model)].copy()
        if a_n.empty:
            continue
        subtype = _bank_subtype(pid) or str(p1row.get("subtype") or "").strip().lower()
        gold = gold_map.get(pid, str(p1row.get("correct_answer", "")))
        crit = _critical_step(params, pid)
        intent = _phase1_intent(str(p1row.get("stated_algorithm", "")))
        rtypes = [str(x).strip().lower() for x in a_n["reasoning_type"].tolist() if str(x).strip()]
        if not rtypes:
            cci_alg = 0.0
        else:
            forward_algo = sum(t in {"forward_simulation", "algorithm_invocation"} for t in rtypes) / len(rtypes)
            local = sum(t == "local_greedy" for t in rtypes) / len(rtypes)
            if intent == "dp_like":
                cci_alg = 1.0 if forward_algo >= local else 0.0
            elif intent == "greedy_like":
                cci_alg = 1.0 if local >= forward_algo else 0.0
            else:
                cci_alg = 0.0
        first_norm = _first_step_decision(a_n)
        pred = str(p1row.get("predicted_first_decision", ""))
        match_first = 1.0 if normalize_phase2_decision(subtype, first_norm) == normalize_phase1_decision(subtype, pred) else 0.0
        crit_row = a_n[a_n["step_index_int"] == crit]
        if crit_row.empty:
            cci_crit = 0.0
        else:
            try:
                cci_crit = 1.0 if _optimal_for_step(subtype, gold, crit, str(crit_row.iloc[0]["parsed_decision"])) else 0.0
            except ValueError:
                cci_crit = 0.0
        rows.append({
            "problem_id": pid, "model": model, "model_short": _short(model),
            "cci_alg": cci_alg, "cci_crit": cci_crit, "match_first": match_first,
            "cci_composite": float(np.mean([cci_alg, match_first, cci_crit])),
        })
    return pd.DataFrame(rows)


def compute_algo_tep(normal: pd.DataFrame, injected: pd.DataFrame) -> pd.DataFrame:
    if normal.empty or injected.empty:
        return pd.DataFrame()
    def _ns(df):
        out = df.copy()
        out["_step"] = pd.to_numeric(out["step_index"], errors="coerce").fillna(0).astype(int)
        mins = out.groupby(["problem_id", "model"])["_step"].transform("min")
        out["_step"] = out["_step"] - mins
        return out
    nrm, inj = _ns(normal), _ns(injected)
    rows = []
    for (pid, model), gi in inj.groupby(["problem_id", "model"]):
        gn = nrm[(nrm["problem_id"] == pid) & (nrm["model"] == model)]
        if gn.empty:
            continue
        crit_s = pd.to_numeric(gi["critical_step_index"], errors="coerce").dropna()
        if crit_s.empty:
            continue
        crit = int(crit_s.iloc[0])
        if crit < 0:
            continue
        merged = gn.merge(gi[["_step", "parsed_decision", "response_type"]], on="_step", how="inner", suffixes=("_n", "_i"))
        post = merged[merged["_step"] > crit]
        if post.empty:
            continue
        subtype = str(gi["subtype"].iloc[0]) if "subtype" in gi.columns else ""
        diffs = post.apply(
            lambda r: normalize_phase2_decision(subtype, r["parsed_decision_n"])
            != normalize_phase2_decision(subtype, r["parsed_decision_i"]),
            axis=1,
        )
        rows.append({
            "problem_id": pid, "model": model, "model_short": _short(model),
            "tep": float(diffs.mean()), "n_post_steps": int(len(post)),
        })
    return pd.DataFrame(rows)


def run_p2(banks: dict, algo_p1: dict[str, pd.DataFrame]) -> dict:
    phase1 = load_phase1_algo()
    normal = load_phase2_normal(overlay_gemini=True, keep_rest=False)
    normal_2a = load_phase2_normal(overlay_gemini=True, keep_rest=True)
    injected = load_phase2_injected()
    impl = _drop_mock(_read(RAW / "ALGO_P2_phase2_injected_implausible.csv"))
    elic = _drop_mock(_read(RAW / "ALGO_P2_phase2_normal_elicited.csv"))
    gsm_p2 = load_gsm_p2()
    gsm_phase1_files = {m: _drop_mock(_read(RAW / f"GSM_P2_phase1_{TAG[m]}.csv")) for m in MODELS}

    # P2.1 Phase 1 GSM
    for m in MODELS:
        p1 = gsm_phase1_files[m]
        src = f"GSM_P2_phase1_{TAG[m]}.csv"
        if p1.empty:
            add_nc(f"P2.1.GSM.{m}.parse_rate", probe="P2", phase="P1", family="GSM", model=m,
                   metric="declaration_parse_rate", source_file=src, note="phase1 file missing")
            continue
        n = int(p1["problem_id"].nunique()) if "problem_id" in p1.columns else len(p1)
        if "phase1_parseable" in p1.columns:
            k = int(_is_true(p1["phase1_parseable"]).sum())
            add(id=f"P2.1.GSM.{m}.parse_rate", probe="P2", phase="P1", family="GSM", model=m,
                metric="declaration_parse_rate", value=k / n if n else "NOT_COMPUTABLE", n=n,
                source_file=src, filter_applied="drop mock")
            add(id=f"P2.1.GSM.{m}.n_parseable", probe="P2", phase="P1", family="GSM", model=m,
                metric="n_parseable", value=k, n=n, source_file=src, filter_applied="drop mock")
        else:
            add_nc(f"P2.1.GSM.{m}.parse_rate", probe="P2", phase="P1", family="GSM", model=m,
                   metric="declaration_parse_rate", source_file=src, note="no phase1_parseable column")
        # empty / diverged from cci file (or phase1)
        src2 = "GSM_P2_cci.csv" if m != "o4-mini" else src
        sub = gsm_p2[gsm_p2["model"].map(_short) == m] if not gsm_p2.empty else p1
        if "cci_total" in sub.columns and "cci_score" in sub.columns:
            tot = pd.to_numeric(sub["cci_total"], errors="coerce").fillna(0)
            sc = pd.to_numeric(sub["cci_score"], errors="coerce").fillna(0)
            n_empty = int((tot == 0).sum())
            n_div = int(((sc == 0) & (tot > 0)).sum())
            add(id=f"P2.1.GSM.{m}.n_empty_declarations", probe="P2", phase="P1", family="GSM",
                model=m, metric="n_empty_declarations", value=n_empty, n=int(len(sub)),
                source_file=src2, filter_applied="cci_total==0")
            add(id=f"P2.1.GSM.{m}.n_declared_then_diverged", probe="P2", phase="P1", family="GSM",
                model=m, metric="n_declared_then_diverged", value=n_div, n=int(len(sub)),
                source_file=src2, filter_applied="cci_score==0 and cci_total>0")
            empty = sub[tot == 0]
            div = sub[(sc == 0) & (tot > 0)]
            sess = sub["either_session_correct"] if "either_session_correct" in sub.columns else sub.get("session_b_correct")
            if sess is not None:
                ea = float(_is_true(empty["either_session_correct"] if "either_session_correct" in empty.columns else empty["session_b_correct"]).mean()) if len(empty) else float("nan")
                da = float(_is_true(div["either_session_correct"] if "either_session_correct" in div.columns else div["session_b_correct"]).mean()) if len(div) else float("nan")
                add(id=f"P2.1.GSM.{m}.empty_acc", probe="P2", phase="P1", family="GSM", model=m,
                    metric="empty_declaration_accuracy", value=ea if ea == ea else "NOT_COMPUTABLE",
                    n=int(len(empty)), source_file=src2, filter_applied="cci_total==0")
                add(id=f"P2.1.GSM.{m}.diverged_acc", probe="P2", phase="P1", family="GSM", model=m,
                    metric="declared_then_diverged_accuracy", value=da if da == da else "NOT_COMPUTABLE",
                    n=int(len(div)), source_file=src2, filter_applied="cci_score==0 and cci_total>0")

    # P2.1 Phase 1 ALGO
    for m in MODELS:
        long = LONG[m]
        p1 = phase1[phase1["model"] == long] if not phase1.empty else pd.DataFrame()
        src = "ALGO_P2_phase1_* (new overlays keep=last); no o1mini file"
        if p1.empty:
            add_nc(f"P2.1.ALGO.{m}.parse_rate", probe="P2", phase="P1", family="ALGO", model=m,
                   metric="declaration_parse_rate", source_file=src,
                   note="No ALGO Phase-1 declaration file for this model" if m == "o4-mini" else "empty")
            add_nc(f"P2.1.ALGO.{m}.n_empty_declarations", probe="P2", phase="P1", family="ALGO",
                   model=m, metric="n_empty_declarations", source_file=src, note="no phase1")
            continue
        n = int(len(p1))
        if "phase1_parseable" in p1.columns:
            k = int(_is_true(p1["phase1_parseable"]).sum())
            add(id=f"P2.1.ALGO.{m}.parse_rate", probe="P2", phase="P1", family="ALGO", model=m,
                metric="declaration_parse_rate", value=k / n, n=n, source_file=src, filter_applied="drop mock; keep=last")
        n_empty = int(_is_true(p1["unparseable_q2"]).sum()) if "unparseable_q2" in p1.columns else int((p1.get("stated_algorithm", "") == "").sum())
        add(id=f"P2.1.ALGO.{m}.n_empty_declarations", probe="P2", phase="P1", family="ALGO", model=m,
            metric="n_empty_declarations", value=n_empty, n=n, source_file=src,
            filter_applied="unparseable_q2 or empty stated_algorithm")

    # P2.2 GSM Phase 2A
    for m in MODELS:
        sub = gsm_p2[gsm_p2["model"].map(_short) == m] if not gsm_p2.empty else pd.DataFrame()
        src = "GSM_P2_cci.csv" + (" + GSM_P2_phase1_o1mini.csv" if m == "o4-mini" else "")
        if sub.empty:
            add_nc(f"P2.2.GSM.{m}.acc_p2a", probe="P2", phase="P2A", family="GSM", model=m,
                   metric="fresh_session_accuracy", source_file=src, note="no GSM P2 rows")
            continue
        # o4-mini CCI/TEP on parseable subset (paper Table 4). Acc_P2A stays on all 44.
        note = "all sessions"
        if m == "o4-mini":
            p1o = gsm_phase1_files[m]
            src = "GSM_P2_phase1_o1mini.csv"
            if not p1o.empty:
                sub = p1o
            if not p1o.empty and "phase1_parseable" in p1o.columns:
                use_cci = p1o[_is_true(p1o["phase1_parseable"])]
                note = "parseable subset for CCI/TEP; Acc_P2A is phase2a_correct (unrecoverable from stored files)"
            else:
                use_cci = sub
        else:
            use_cci = sub
        p2a = sub["phase2a_correct"] if "phase2a_correct" in sub.columns else pd.Series([""] * len(sub), index=sub.index)
        if p2a.astype(str).str.strip().ne("").any():
            k = int(_is_true(p2a).sum())
            n = int(len(sub))
            acc = k / n
            lo, hi = wilson(k, n)
            add(id=f"P2.2.GSM.{m}.acc_p2a", probe="P2", phase="P2A", family="GSM", model=m,
                metric="fresh_session_accuracy", value=acc, n=n, ci_low=lo, ci_high=hi,
                source_file="GSM_P2_session_correct.csv", filter_applied="drop mock",
                note="phase2a_correct")
        else:
            add_nc(f"P2.2.GSM.{m}.acc_p2a", probe="P2", phase="P2A", family="GSM", model=m,
                   metric="fresh_session_accuracy", source_file="GSM_P2_session_correct.csv",
                   note="phase2a_values were never persisted; Acc_P2A cannot be recovered without a re-run")
        cci = pd.to_numeric(use_cci.get("cci_score", pd.Series(dtype=float)), errors="coerce").dropna()
        tep = pd.to_numeric(use_cci.get("tep_score", pd.Series(dtype=float)), errors="coerce").dropna()
        add(id=f"P2.2.GSM.{m}.cci_mean", probe="P2", phase="P2A", family="GSM", model=m,
            metric="CCI_mean", value=float(cci.mean()) if len(cci) else "NOT_COMPUTABLE", n=int(len(cci)),
            source_file=src, filter_applied=note)
        add(id=f"P2.2.GSM.{m}.cci_median", probe="P2", phase="P2A", family="GSM", model=m,
            metric="CCI_median", value=float(cci.median()) if len(cci) else "NOT_COMPUTABLE", n=int(len(cci)),
            source_file=src, filter_applied=note)
        add(id=f"P2.3.GSM.{m}.tep_mean", probe="P2", phase="P2B", family="GSM", model=m,
            metric="TEP_mean", value=float(tep.mean()) if len(tep) else "NOT_COMPUTABLE", n=int(len(tep)),
            source_file=src, filter_applied=note)

    # Wilcoxon Claude vs GPT-4o CCI (paper Table 4)
    cl = gsm_p2[gsm_p2["model"].map(_short) == "Claude"] if not gsm_p2.empty else pd.DataFrame()
    gp = gsm_p2[gsm_p2["model"].map(_short) == "GPT-4o"] if not gsm_p2.empty else pd.DataFrame()
    if not cl.empty and not gp.empty:
        cl_s = pd.to_numeric(cl.set_index("problem_id")["cci_score"], errors="coerce")
        gp_s = pd.to_numeric(gp.set_index("problem_id")["cci_score"], errors="coerce")
        common = cl_s.index.intersection(gp_s.index)
        try:
            W, p = stats.wilcoxon(cl_s.loc[common].fillna(0).astype(float),
                                  gp_s.loc[common].fillna(0).astype(float),
                                  zero_method="wilcox", alternative="greater")
        except ValueError:
            W, p = float("nan"), float("nan")
        add(id="P2.2.GSM.Claude_vs_GPT4o.cci_wilcoxon", probe="P2", phase="P2A", family="GSM",
            model="Claude|GPT-4o", metric="cci_wilcoxon", value=p, n=int(len(common)),
            test="wilcoxon_greater", statistic=W, p_value=p,
            source_file="GSM_P2_cci.csv", filter_applied="paired on problem_id; zero-imputed")

    # GSM four-way taxonomy: NOT_COMPUTABLE
    add_nc("P2.3.GSM.four_way_compliance", probe="P2", phase="P2B", family="GSM",
           metric="four_way_compliance", source_file="GSM_P2_cci.csv",
           note="ALGO four-way taxonomy is Decision:/Reason: parse of injection-step raw_response. GSM Phase-2B logs have no injection-step raw_response and no response_type.")

    # P2.2 ALGO Phase 2A
    last_n = _last_step(normal_2a, "final_answer_correct")
    last_e = _last_step(elic, "final_answer_correct")
    cci_df = compute_algo_cci(phase1, normal, banks)
    if not cci_df.empty:
        cci_df.to_csv(OUT / "algo_cci_per_instance.csv", index=False)
    for m in MODELS:
        ln = last_n[last_n["model"].map(_short) == m] if not last_n.empty else pd.DataFrame()
        src = "ALGO_P2_phase2_normal.csv + ALGO_P2_phase2_normal_gemini.csv"
        if ln.empty:
            add_nc(f"P2.2.ALGO.{m}.acc_p2a", probe="P2", phase="P2A", family="ALGO", model=m,
                   metric="fresh_session_accuracy", source_file=src, note="no Phase-2A last-step rows")
        else:
            k = int(ln["ok"].sum())
            n = int(len(ln))
            _k, _n, _acc, lo, hi = _acc_row(ln, cluster=True)
            add(id=f"P2.2.ALGO.{m}.acc_p2a", probe="P2", phase="P2A", family="ALGO", model=m,
                metric="fresh_session_accuracy", value=k / n, n=n, ci_low=lo, ci_high=hi,
                source_file=src, filter_applied="last step; drop mock; Gemini dedicated overlay; cluster bootstrap")
            for fam in ["CC", "SP", "WIS"]:
                ids = PAPER_ADV[fam]
                sl = ln[ln["problem_id"].isin(ids)]
                if sl.empty:
                    continue
                kk, nn = int(sl["ok"].sum()), int(len(sl))
                _k, _n, _acc, l2, h2 = _acc_row(sl, cluster=True)
                add(id=f"P2.2.ALGO.{m}.{fam}.acc_p2a", probe="P2", phase="P2A", family="ALGO",
                    subtype=fam, model=m, metric="fresh_session_accuracy", value=kk / nn, n=nn,
                    ci_low=l2, ci_high=h2, source_file=src, filter_applied="frozen adversarial subtype")
        cm = cci_df[cci_df["model_short"] == m] if not cci_df.empty else pd.DataFrame()
        vals = pd.to_numeric(cm["cci_composite"], errors="coerce").dropna() if not cm.empty else pd.Series(dtype=float)
        if len(vals) == 0:
            add_nc(f"P2.2.ALGO.{m}.cci_mean", probe="P2", phase="P2A", family="ALGO", model=m,
                   metric="CCI_mean", source_file="ALGO_P2_phase1_* + ALGO_P2_phase2_normal*",
                   note="o4-mini has no Phase-1 file so CCI cannot be built" if m == "o4-mini"
                   else "no overlapping phase1×phase2 adversarial sessions")
        else:
            add(id=f"P2.2.ALGO.{m}.cci_mean", probe="P2", phase="P2A", family="ALGO", model=m,
                metric="CCI_mean", value=float(vals.mean()), n=int(len(vals)),
                source_file="ALGO_P2_phase1_* + ALGO_P2_phase2_normal*",
                filter_applied="adversarial 61; CCI=(cci_alg+match_first+cci_crit)/3")
            add(id=f"P2.2.ALGO.{m}.cci_median", probe="P2", phase="P2A", family="ALGO", model=m,
                metric="CCI_median", value=float(vals.median()), n=int(len(vals)),
                source_file="ALGO_P2_phase1_* + ALGO_P2_phase2_normal*",
                filter_applied="adversarial 61")

        # elicited vs normal paired
        le = last_e[last_e["model"].map(_short) == m] if not last_e.empty else pd.DataFrame()
        if ln.empty or le.empty:
            add_nc(f"P2.2.ALGO.{m}.elicited_vs_normal", probe="P2", phase="P2A", family="ALGO",
                   model=m, metric="elicited_vs_normal_wilcoxon",
                   source_file="ALGO_P2_phase2_normal_elicited.csv",
                   note="missing normal or elicited last-step rows")
        else:
            merged = ln.merge(le, on="problem_id", suffixes=("_n", "_e"))
            if merged.empty:
                add_nc(f"P2.2.ALGO.{m}.elicited_vs_normal", probe="P2", phase="P2A", family="ALGO",
                       model=m, metric="elicited_vs_normal_wilcoxon",
                       source_file="ALGO_P2_phase2_normal* + elicited", note="no paired problem_ids")
            else:
                a = merged["ok_n"].astype(int).to_numpy()
                b = merged["ok_e"].astype(int).to_numpy()
                add(id=f"P2.2.ALGO.{m}.acc_normal_paired", probe="P2", phase="P2A", family="ALGO",
                    model=m, metric="acc_normal_paired", value=float(a.mean()), n=int(len(a)),
                    source_file="ALGO_P2_phase2_normal*", filter_applied="paired intersection with elicited")
                add(id=f"P2.2.ALGO.{m}.acc_elicited_paired", probe="P2", phase="P2A", family="ALGO",
                    model=m, metric="acc_elicited_paired", value=float(b.mean()), n=int(len(b)),
                    source_file="ALGO_P2_phase2_normal_elicited.csv", filter_applied="paired intersection")
                try:
                    W, p = stats.wilcoxon(a.astype(float), b.astype(float), zero_method="wilcox")
                except ValueError:
                    W, p = float("nan"), float("nan")
                add(id=f"P2.2.ALGO.{m}.elicited_vs_normal", probe="P2", phase="P2A", family="ALGO",
                    model=m, metric="elicited_vs_normal_wilcoxon", value=p if p == p else "NOT_COMPUTABLE",
                    n=int(len(a)), test="wilcoxon_two_sided", statistic=W, p_value=p,
                    source_file="ALGO_P2_phase2_normal* + elicited",
                    filter_applied="paired on problem_id last-step correctness",
                    note="" if p == p else "all paired differences zero")

    # P2.3 ALGO four-way + TEP + post-inj + plausible vs implausible
    tep_sess = compute_algo_tep(normal, injected)
    if not tep_sess.empty:
        tep_sess.to_csv(OUT / "algo_tep_sessions.csv", index=False)
    last_i = _last_step(injected, "post_injection_correct")
    last_impl = _last_step(impl, "post_injection_correct")

    FOUR = ["compliant", "partial_compliance", "refusal", "format_ignored"]
    for m in MODELS:
        inj_m = injected[injected["model"].map(_short) == m] if not injected.empty else pd.DataFrame()
        src = "ALGO_P2_phase2_injected.csv + ALGO_P2_phase2_injected_gemini.csv"
        if inj_m.empty:
            add_nc(f"P2.3.ALGO.{m}.four_way", probe="P2", phase="P2B", family="ALGO", model=m,
                   metric="four_way_compliance", source_file=src, note="no injected rows")
        else:
            step = inj_m[_is_true(inj_m["injection_applied"])] if "injection_applied" in inj_m.columns else inj_m
            vc = step["response_type"].astype(str).str.strip().str.lower().value_counts()
            n = int(len(step))
            for lab in FOUR + ["full_solution_dump"]:
                k = int(vc.get(lab, 0))
                add(id=f"P2.3.ALGO.{m}.compliance.{lab}", probe="P2", phase="P2B", family="ALGO",
                    model=m, metric=f"compliance_rate_{lab}", value=k / n if n else "NOT_COMPUTABLE",
                    n=n, source_file=src, filter_applied="injection_applied==True; drop mock")
        ts = tep_sess[tep_sess["model_short"] == m] if not tep_sess.empty else pd.DataFrame()
        vals = pd.to_numeric(ts["tep"], errors="coerce").dropna() if not ts.empty else pd.Series(dtype=float)
        add(id=f"P2.3.ALGO.{m}.tep_mean", probe="P2", phase="P2B", family="ALGO", model=m,
            metric="TEP_mean",
            value=float(vals.mean()) if len(vals) else "NOT_COMPUTABLE",
            n=int(len(vals)),
            source_file=src + " joined to ALGO_P2_phase2_normal*",
            filter_applied="post-critical parsed_decision mismatch; plausible injection only",
            note="" if len(vals) else "no paired post-injection steps")
        add(id=f"N.2.ALGO.{m}.tep_mean", probe="P2", phase="P2B", family="ALGO", model=m,
            metric="TEP_mean",
            value=float(vals.mean()) if len(vals) else "NOT_COMPUTABLE",
            n=int(len(vals)),
            source_file=src + " joined to ALGO_P2_phase2_normal*",
            filter_applied="same as P2.3; previously unreported per-model TEP")
        if len(vals):
            add(id=f"N.2.ALGO.{m}.tep_median", probe="P2", phase="P2B", family="ALGO", model=m,
                metric="TEP_median", value=float(vals.median()), n=int(len(vals)),
                source_file=src, filter_applied="same as TEP_mean")

        li = last_i[last_i["model"].map(_short) == m] if not last_i.empty else pd.DataFrame()
        if li.empty:
            add_nc(f"P2.3.ALGO.{m}.post_inj_acc", probe="P2", phase="P2B", family="ALGO", model=m,
                   metric="post_injection_accuracy_plausible", source_file=src, note="no last-step")
        else:
            k, n = int(li["ok"].sum()), int(len(li))
            _k, _n, _acc, lo, hi = _acc_row(li, cluster=True)
            add(id=f"P2.3.ALGO.{m}.post_inj_acc_plausible", probe="P2", phase="P2B", family="ALGO",
                model=m, metric="post_injection_accuracy_plausible", value=k / n, n=n,
                ci_low=lo, ci_high=hi, source_file=src, filter_applied="last-step post_injection_correct; cluster bootstrap")

        lp = last_impl[last_impl["model"].map(_short) == m] if not last_impl.empty else pd.DataFrame()
        src_impl = "ALGO_P2_phase2_injected_implausible.csv"
        if lp.empty:
            add_nc(f"P2.3.ALGO.{m}.post_inj_acc_implausible", probe="P2", phase="P2B", family="ALGO",
                   model=m, metric="post_injection_accuracy_implausible", source_file=src_impl,
                   note="no implausible last-step")
        else:
            k, n = int(lp["ok"].sum()), int(len(lp))
            _k, _n, _acc, lo, hi = _acc_row(lp, cluster=True)
            add(id=f"P2.3.ALGO.{m}.post_inj_acc_implausible", probe="P2", phase="P2B", family="ALGO",
                model=m, metric="post_injection_accuracy_implausible", value=k / n, n=n,
                ci_low=lo, ci_high=hi, source_file=src_impl, filter_applied="last-step post_injection_correct; cluster bootstrap")

        if not li.empty and not lp.empty:
            merged = li.merge(lp, on="problem_id", suffixes=("_p", "_i"))
            if len(merged) >= 2:
                a = merged["ok_p"].astype(int).to_numpy()
                b = merged["ok_i"].astype(int).to_numpy()
                try:
                    W, p = stats.wilcoxon(a.astype(float), b.astype(float), zero_method="wilcox")
                except ValueError:
                    W, p = float("nan"), float("nan")
                add(id=f"P2.3.ALGO.{m}.plausible_vs_implausible", probe="P2", phase="P2B",
                    family="ALGO", model=m, metric="plausible_vs_implausible_wilcoxon",
                    value=p if p == p else "NOT_COMPUTABLE", n=int(len(merged)),
                    test="wilcoxon_two_sided", statistic=W, p_value=p,
                    source_file=src + " + " + src_impl,
                    filter_applied="paired on problem_id last-step correctness",
                    note="" if p == p else "all paired differences zero")
                add(id=f"P2.3.ALGO.{m}.plausible_minus_implausible_pp", probe="P2", phase="P2B",
                    family="ALGO", model=m, metric="acc_plausible_minus_implausible",
                    value=float(a.mean() - b.mean()), n=int(len(merged)),
                    source_file=src + " + " + src_impl, filter_applied="paired")

        # P2.4 correctness | complied vs refused
        if inj_m.empty:
            add_nc(f"P2.4.ALGO.{m}.fisher", probe="P2", phase="P2B", family="ALGO", model=m,
                   metric="correct_given_complied_vs_refused", source_file=src, note="no injected rows")
        else:
            inj_step = inj_m[_is_true(inj_m["injection_applied"])] if "injection_applied" in inj_m.columns else inj_m
            # join to last-step correctness
            if li.empty:
                add_nc(f"P2.4.ALGO.{m}.fisher", probe="P2", phase="P2B", family="ALGO", model=m,
                       metric="correct_given_complied_vs_refused", source_file=src, note="no last-step correctness")
            else:
                js = inj_step.merge(li[["problem_id", "ok"]], on="problem_id", how="inner")
                rtype = js["response_type"].astype(str).str.strip().str.lower()
                comp = js[rtype == "compliant"]
                ref = js[rtype == "refusal"]
                if len(comp) == 0 or len(ref) == 0:
                    add_nc(f"P2.4.ALGO.{m}.fisher", probe="P2", phase="P2B", family="ALGO", model=m,
                           metric="correct_given_complied_vs_refused", source_file=src,
                           note=f"n_compliant={len(comp)} n_refusal={len(ref)} — Fisher undefined")
                else:
                    kc, nc = int(comp["ok"].sum()), int(len(comp))
                    kr, nr = int(ref["ok"].sum()), int(len(ref))
                    table = np.array([[kc, nc - kc], [kr, nr - kr]])
                    try:
                        stat, p = stats.fisher_exact(table, alternative="two-sided")
                    except ValueError:
                        stat, p = float("nan"), float("nan")
                    add(id=f"P2.4.ALGO.{m}.acc_given_compliant", probe="P2", phase="P2B", family="ALGO",
                        model=m, metric="accuracy_given_compliant", value=kc / nc, n=nc,
                        source_file=src, filter_applied="injection-step response_type=compliant")
                    add(id=f"P2.4.ALGO.{m}.acc_given_refusal", probe="P2", phase="P2B", family="ALGO",
                        model=m, metric="accuracy_given_refusal", value=kr / nr, n=nr,
                        source_file=src, filter_applied="injection-step response_type=refusal")
                    add(id=f"P2.4.ALGO.{m}.fisher", probe="P2", phase="P2B", family="ALGO", model=m,
                        metric="correct_given_complied_vs_refused", value=p, n=nc + nr,
                        test="fisher_exact", statistic=stat, p_value=p,
                        source_file=src, filter_applied="2x2 of (compliant/refusal) × (correct/wrong)")

    # P2.5 BW Probe 2
    bw_cci = _drop_mock(_read(RAW / "BW_P2_cci.csv"))
    bw_nl = _drop_mock(_read(RAW / "BW_P2_cci_nl.csv"))
    bw_tep = _drop_mock(_read(RAW / "BW_P2_tep.csv"))
    for label, df, src in [
        ("strict_pddl", bw_cci, "BW_P2_cci.csv"),
        ("nl_tolerant", bw_nl, "BW_P2_cci_nl.csv"),
    ]:
        models_present = sorted({_short(x) for x in df["model"].unique()}) if not df.empty else []
        add(id=f"P2.5.BW.{label}.n_models", probe="P2", phase="P2", family="BW",
            metric="n_models", value=len(models_present), n=len(models_present),
            source_file=src, filter_applied="drop mock",
            note=f"models={models_present}; paper Table 6 claimed all five; files contain 3")
        if df.empty:
            continue
        for m in MODELS:
            sub = df[df["model"].map(_short) == m]
            if sub.empty:
                add_nc(f"P2.5.BW.{label}.{m}.abort_rate", probe="P2", phase="P2", family="BW",
                       model=m, metric="abort_rate", source_file=src,
                       note="model not in this BW P2 file (coverage is 3 models, not 5)")
                continue
            status = sub["session_status"].astype(str).str.lower()
            abort = ~status.str.contains("complete")
            k, n = int(abort.sum()), int(len(sub))
            lo, hi = wilson(k, n)
            add(id=f"P2.5.BW.{label}.{m}.abort_rate", probe="P2", phase="P2", family="BW",
                model=m, metric="abort_rate", value=k / n, n=n, ci_low=lo, ci_high=hi,
                source_file=src, filter_applied="session_status not containing 'complete'")
            if "cci" in sub.columns:
                cci = pd.to_numeric(sub["cci"], errors="coerce").dropna()
                add(id=f"P2.5.BW.{label}.{m}.cci_mean", probe="P2", phase="P2", family="BW",
                    model=m, metric="CCI_mean", value=float(cci.mean()) if len(cci) else "NOT_COMPUTABLE",
                    n=int(len(cci)), source_file=src, filter_applied="drop mock")
    add(id="P2.5.BW.coverage_note", probe="P2", phase="P2", family="BW",
        metric="covers_3_not_5", value=1, n=3,
        source_file="BW_P2_cci.csv + BW_P2_cci_nl.csv + BW_P2_tep.csv",
        filter_applied="unique model column",
        note="Strict-PDDL and NL-tolerant reruns both contain only Claude, GPT-4o, Llama. Gemini and o4-mini are absent. Paper Table 6 claimed NL-tolerant covers all five models.")

    return {"cci": cci_df, "tep": tep_sess, "gsm_p2": gsm_p2, "phase1": phase1, "normal": normal}


# ---------------------------------------------------------------------------
# Probe 3 infini-gram
# ---------------------------------------------------------------------------

def run_p3(algo: dict[str, pd.DataFrame], p2: dict) -> None:
    cont = _read(RAW / "ALGO_P3_contamination.csv")
    src = "ALGO_P3_contamination.csv + ALGO_P1_behavioral_*"
    if cont.empty:
        add_nc("P3.1", probe="P3", family="ALGO", metric="proximity_vri", source_file=src,
               note="ALGO_P3_contamination.csv missing")
        return
    cont = cont[["problem_id", "instance_contamination_score", "template_contamination_score"]].drop_duplicates("problem_id")
    cont["instance_contamination_score"] = pd.to_numeric(cont["instance_contamination_score"], errors="coerce")
    cont["template_contamination_score"] = pd.to_numeric(cont["template_contamination_score"], errors="coerce")

    # P3.4 gradient table
    for sub, ids in PAPER_ADV.items():
        csub = cont[cont["problem_id"].isin(ids)]
        tmean = float(csub["template_contamination_score"].mean()) if len(csub) else float("nan")
        imean = float(csub["instance_contamination_score"].mean()) if len(csub) else float("nan")
        add(id=f"P3.4.{sub}.mean_template_proximity", probe="P3", phase="P3", family="ALGO",
            subtype=sub, metric="mean_template_contamination_score", value=tmean, n=int(len(csub)),
            source_file="ALGO_P3_contamination.csv", filter_applied="frozen adversarial subtype")
        add(id=f"P3.4.{sub}.mean_instance_proximity", probe="P3", phase="P3", family="ALGO",
            subtype=sub, metric="mean_instance_contamination_score", value=imean, n=int(len(csub)),
            source_file="ALGO_P3_contamination.csv", filter_applied="frozen adversarial subtype")
        for m in MODELS:
            df = algo[m]
            can = df[(df["variant_type"] == "canonical") & df["problem_id"].isin(ids)]
            w3 = df[(df["variant_type"] == "W3") & df["problem_id"].isin(ids)]
            _, n_c, a_c, _, _ = _acc_row(can)
            _, n_w, a_w, _, _ = _acc_row(w3)
            add(id=f"P3.4.{sub}.{m}.canonical_acc", probe="P3", phase="P3", family="ALGO",
                subtype=sub, model=m, variant="canonical", metric="accuracy",
                value=a_c if n_c else "NOT_COMPUTABLE", n=n_c,
                source_file=f"ALGO_P1_behavioral_{TAG[m]}.csv", filter_applied="frozen adversarial")
            add(id=f"P3.4.{sub}.{m}.W3_acc", probe="P3", phase="P3", family="ALGO",
                subtype=sub, model=m, variant="W3", metric="accuracy",
                value=a_w if n_w else "NOT_COMPUTABLE", n=n_w,
                source_file=f"ALGO_P1_behavioral_{TAG[m]}.csv", filter_applied="frozen adversarial")

    cci_df = p2.get("cci", pd.DataFrame())
    gsm_p2 = p2.get("gsm_p2", pd.DataFrame())

    for m in MODELS:
        df = algo[m]
        wide = df[df["problem_id"].isin(PAPER_ADV_ALL)].pivot_table(
            index="problem_id", columns="variant_type", values="ok", aggfunc="max"
        )
        for c in ["W1", "W2", "W3", "W4", "canonical"]:
            if c not in wide.columns:
                wide[c] = np.nan
        wide["VRI"] = wide[["W1", "W2", "W4"]].mean(axis=1) - wide["W3"]
        wide = wide.reset_index().merge(cont, on="problem_id", how="left")
        for score_name, col in [
            ("template_contamination_score", "template_contamination_score"),
            ("instance_contamination_score", "instance_contamination_score"),
        ]:
            s = wide.dropna(subset=[col, "VRI"]).copy()
            s[col] = pd.to_numeric(s[col], errors="coerce")
            s["VRI"] = pd.to_numeric(s["VRI"], errors="coerce")
            s["canonical"] = pd.to_numeric(s["canonical"], errors="coerce")
            s = s.dropna(subset=[col, "VRI"])
            n = len(s)
            if n < 4 or s[col].nunique() < 2 or s["VRI"].nunique() < 2:
                add(id=f"P3.1.{m}.{score_name}_vs_VRI", probe="P3", phase="P3", family="ALGO",
                    model=m, metric=f"pearson_{score_name}_vs_VRI",
                    value="NOT_COMPUTABLE", n=n, source_file=src,
                    filter_applied="frozen adversarial 61",
                    note="n<4 or zero variance")
            else:
                r, p = stats.pearsonr(s[col].astype(float), s["VRI"].astype(float))
                add(id=f"P3.1.{m}.{score_name}_vs_VRI", probe="P3", phase="P3", family="ALGO",
                    model=m, metric=f"pearson_{score_name}_vs_VRI", value=r, n=n,
                    test="pearsonr", statistic=r, p_value=p, source_file=src,
                    filter_applied="frozen adversarial 61; VRI=mean(W1,W2,W4)-W3")
            pr, pp, pn = _partial_pearson(s[col], s["VRI"], s["canonical"]) if n else (float("nan"), float("nan"), n)
            add(id=f"P3.2.{m}.{score_name}_vs_VRI_residual_on_canonical", probe="P3", phase="P3",
                family="ALGO", model=m, metric=f"partial_r_{score_name}_vs_VRI",
                value=pr if pr == pr else "NOT_COMPUTABLE", n=pn,
                test="partial_pearson", statistic=pr, p_value=pp, source_file=src,
                filter_applied="residualised on per-problem canonical accuracy")

        # P3.3 vs CCI
        if cci_df.empty:
            add_nc(f"P3.3.{m}.instance_vs_CCI", probe="P3", family="ALGO", model=m,
                   metric="pearson_instance_vs_CCI", source_file=src,
                   note="no ALGO CCI (o4-mini has none)" if m == "o4-mini" else "CCI empty")
            continue
        cm = cci_df[cci_df["model_short"] == m][["problem_id", "cci_composite"]]
        joined = wide.merge(cm, on="problem_id", how="inner")
        for score_name, col in [
            ("template_contamination_score", "template_contamination_score"),
            ("instance_contamination_score", "instance_contamination_score"),
        ]:
            s = joined.dropna(subset=[col, "cci_composite"]).copy()
            s[col] = pd.to_numeric(s[col], errors="coerce")
            s["cci_composite"] = pd.to_numeric(s["cci_composite"], errors="coerce")
            s = s.dropna(subset=[col, "cci_composite"])
            n = len(s)
            if n < 4 or s[col].nunique() < 2 or s["cci_composite"].nunique() < 2:
                add(id=f"P3.3.{m}.{score_name}_vs_CCI", probe="P3", phase="P3", family="ALGO",
                    model=m, metric=f"pearson_{score_name}_vs_CCI",
                    value="NOT_COMPUTABLE", n=n, source_file=src + " + recomputed CCI",
                    filter_applied="adversarial intersection with CCI",
                    note="n<4 or zero variance")
            else:
                r, p = stats.pearsonr(s[col].astype(float), s["cci_composite"].astype(float))
                add(id=f"P3.3.{m}.{score_name}_vs_CCI", probe="P3", phase="P3", family="ALGO",
                    model=m, metric=f"pearson_{score_name}_vs_CCI", value=r, n=n,
                    test="pearsonr", statistic=r, p_value=p,
                    source_file=src + " + recomputed CCI",
                    filter_applied="adversarial intersection with CCI")


# ---------------------------------------------------------------------------
# P3.5–P3.7 mechanistic inventory
# ---------------------------------------------------------------------------

MECH_CLAIM = {
    "rank": "CAN support: whether gold token rank changes across layers/variants for THIS local model. CANNOT: claims about Claude/GPT-4o/Gemini/o4-mini; causal mediation; paper's 5-model behavioural story.",
    "logprob": "CAN support: gold-token logprob trajectories for THIS local model. CANNOT: cross-model generalisation or behavioural VRI of API models.",
    "cosine": "CAN support: residual-stream cosine similarity / crystallization layer for THIS local model. CANNOT: contamination of the 5 API models; triangulation labels.",
}


def run_p3_mech() -> None:
    files = sorted(RAW.glob("mechanistic*.csv")) + [
        RAW / "ALGO_P3_mechanistic.csv",
        RAW / "GSM_P3_mechanistic.csv",
        RAW / "BW_P3_mechanistic.csv",
    ]
    inv = []
    seen = set()
    for path in files:
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        df = _read(path)
        cols = list(df.columns)
        has_rank = any("rank" in c.lower() for c in cols)
        has_logp = any("logprob" in c.lower() for c in cols)
        has_cos = any("cosine" in c.lower() for c in cols)
        models = sorted(df["model"].astype(str).unique()) if "model" in df.columns else []
        variants = sorted(df["variant_type"].map(_norm_variant).unique()) if "variant_type" in df.columns else []
        n = int(len(df))
        prompt = ""
        name = path.name
        if "chatdirect" in name:
            prompt = "chatdirect_contentgold"
        elif "rawqa" in name:
            prompt = "base_rawqa"
        elif "rawprompt" in name:
            prompt = "rawprompt"
        elif "greedy" in name:
            prompt = "greedy"
        else:
            prompt = "default/unspecified"
        can = []
        cannot = []
        if has_rank:
            can.append("gold-token rank across layers")
        if has_logp:
            can.append("gold-token logprob across layers")
        if has_cos:
            can.append("layer cosine / crystallization")
        if not can:
            cannot.append("no rank/logprob/cosine columns — cannot support any mechanistic claim")
        cannot.append("cannot support 5-API-model paper claims (these files are local Llama/Qwen)")
        cannot.append("do not compute new mechanistic claims from this inventory")
        add(id=f"P3.5.{path.stem}.n_rows", probe="P3", phase="mechanistic", family="",
            model=";".join(models[:3]), variant=";".join(str(v) for v in variants[:8]),
            metric="n_rows", value=n, n=n, source_file=path.name,
            filter_applied="inventory only",
            note=f"prompt_config={prompt}; rank={has_rank} logprob={has_logp} cosine={has_cos}")
        add(id=f"P3.6.{path.stem}.can_support", probe="P3", phase="mechanistic",
            metric="can_support", value=1 if can else 0, n=n, source_file=path.name,
            filter_applied="inventory only",
            note=("CAN: " + "; ".join(can) if can else "CAN: nothing") + " | CANNOT: " + "; ".join(cannot))
        inv.append({
            "file": path.name, "n_rows": n, "models": "|".join(models),
            "prompt_config": prompt, "variants": "|".join(str(v) for v in variants),
            "has_rank": has_rank, "has_logprob": has_logp, "has_cosine": has_cos,
            "columns": "|".join(cols),
            "can_support": "; ".join(can) if can else "nothing",
            "cannot_support": "; ".join(cannot),
        })
    pd.DataFrame(inv).to_csv(OUT / "mechanistic_inventory.csv", index=False)
    add(id="P3.7.no_new_claims", probe="P3", phase="mechanistic", metric="new_claims_computed",
        value=0, n=0, source_file="mechanistic*.csv", filter_applied="inventory only",
        note="Per spec: do not compute new mechanistic claims.")


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def _tri_bool3(x):
    s = str(x).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def load_algo_p1_for_tri(tag: str) -> pd.DataFrame:
    """P1 for triangulation: drop mock only (no ERROR: filter), matching ALGO_P3_SCR_triangulation."""
    df = _read(RAW / f"ALGO_P1_behavioral_{tag}.csv")
    if df.empty:
        return df
    df = _drop_mock(df)
    if "model" in df.columns:
        df = df[df["model"].isin(REAL_MODELS)]
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    return df.drop_duplicates(["problem_id", "variant_type"], keep="last")


def build_tri_panel(algo: dict[str, pd.DataFrame], p2: dict) -> pd.DataFrame:
    """Rebuild the 110×model triangulation panel from raw P1 + contamination + CCI.

    VAR uses verified True/False/NA (not the accuracy ``ok`` 0/1).
    ``gave_greedy_answer`` and ``any_parse_failed`` are max across variants.
    ``greedy_succeeds`` matches ALGO_P3_SCR_triangulation: missing gave → False, not NA.
    """
    del algo  # panel is rebuilt from unfiltered P1 so parse_failed variants are visible
    cont = _read(RAW / "ALGO_P3_contamination.csv")
    if not cont.empty:
        cont = (
            cont.sort_index()
            .groupby("problem_id", as_index=False)
            .agg(
                instance_contamination_score=("instance_contamination_score", "last"),
                template_contamination_score=("template_contamination_score", "last"),
            )
        )
        cont["instance_contamination_score"] = pd.to_numeric(cont["instance_contamination_score"], errors="coerce")
        cmap = dict(zip(cont["problem_id"].astype(str), cont["instance_contamination_score"]))
    else:
        cmap = {}
    cci = p2.get("cci", pd.DataFrame())
    banks = load_banks()
    params = banks["algo_params"]
    bank_ids = [str(x) for x in banks["algo_canon_ids"]]
    four = {"Claude", "GPT-4o", "Llama", "Gemini"}
    rows = []
    for m in MODELS:
        df = load_algo_p1_for_tri(TAG[m])
        if df.empty:
            continue
        df["problem_id"] = df["problem_id"].astype(str)
        df = df[df["problem_id"].isin(bank_ids)]
        cci_m = cci[cci["model_short"] == m] if not cci.empty else pd.DataFrame()
        cci_map = dict(zip(cci_m["problem_id"].astype(str), pd.to_numeric(cci_m["cci_composite"], errors="coerce"))) if len(cci_m) else {}
        for pid, g in df.groupby("problem_id"):
            can = g[g["variant_type"] == "canonical"]
            w3 = g[g["variant_type"] == "W3"]
            v_can = _tri_bool3(can.iloc[0]["verified"]) if len(can) and "verified" in can.columns else None
            v_w3 = _tri_bool3(w3.iloc[0]["verified"]) if len(w3) and "verified" in w3.columns else None
            var_can = 1.0 if v_can is True else (0.0 if v_can is False else np.nan)
            var_w3 = 1.0 if v_w3 is True else (0.0 if v_w3 is False else np.nan)
            gave_vals = [_tri_bool3(x) for x in g["gave_greedy_answer"]] if "gave_greedy_answer" in g.columns else []
            if any(x is True for x in gave_vals):
                gave = True
            elif any(x is False for x in gave_vals):
                gave = False
            else:
                gave = None
            parse_fail = False
            if "parse_status" in g.columns:
                parse_fail = bool(g["parse_status"].astype(str).str.strip().str.lower().eq("parse_failed").any())
            expected = params.get(pid, {}).get("greedy_succeeds", None)
            if expected is None:
                greedy_succeeds = np.nan
            else:
                exp = bool(expected) if isinstance(expected, bool) else str(expected).strip().lower() == "true"
                greedy_succeeds = (gave is True) if exp else (gave is False)
            contam = cmap.get(pid, np.nan)
            aci = cci_map.get(pid, np.nan)
            missing_core = not (pd.notna(var_can) and pd.notna(var_w3) and pd.notna(contam) and pd.notna(greedy_succeeds))
            missing_phase2 = pd.isna(aci)
            rows.append({
                "problem_id": pid, "model": m, "in_paper_4model": m in four,
                "problem_subtype": _bank_subtype(pid),
                "instance_type": "adversarial" if pid in PAPER_ADV_ALL else "standard",
                "VAR_canonical": var_can, "VAR_W3": var_w3,
                "instance_contamination_score": contam, "ACI": aci,
                "greedy_succeeds": greedy_succeeds,
                "missing_core": missing_core, "missing_phase2": missing_phase2,
                "parse_failure_or_missing": parse_fail,
            })
    panel = pd.DataFrame(rows)
    return panel


def run_tri(algo: dict[str, pd.DataFrame], p2: dict) -> None:
    panel = build_tri_panel(algo, p2)
    panel.to_csv(OUT / "triangulation_panel.csv", index=False)
    four = panel[panel["in_paper_4model"]].copy()
    five = panel.copy()
    four["instance_rank_pct"] = four.groupby("problem_subtype")["instance_contamination_score"].rank(method="average", pct=True)
    five["instance_rank_pct"] = five.groupby("problem_subtype")["instance_contamination_score"].rank(method="average", pct=True)

    lab4 = label_default(four)
    c4 = count_labels(lab4)
    n_parse = int(four["parse_failure_or_missing"].astype(bool).sum())
    n_p2miss = int(four["missing_phase2"].astype(bool).sum())
    n_core = int(four["missing_core"].astype(bool).sum())
    add(id="T.2.4model.retrieval", probe="T", phase="triangulation", family="ALGO",
        metric="n_retrieval", value=c4["retrieval"], n=c4["n"],
        source_file="ALGO_P1_behavioral_{claude,gpt4o,llama,gemini}.csv + ALGO_P3_contamination.csv + recomputed CCI",
        filter_applied=f"appendix three-signal; CCI bands {APPENDIX_CCI_RETRIEVAL_MAX}/{APPENDIX_CCI_COMPUTATION_MIN}; contam p{APPENDIX_CONTAM_PERCENTILE}; symmetric W3; 4 models × bank 110")
    add(id="T.2.4model.computation", probe="T", phase="triangulation", family="ALGO",
        metric="n_computation", value=c4["computation"], n=c4["n"],
        source_file="same as T.2.4model.retrieval", filter_applied="executed rule")
    add(id="T.2.4model.mixed", probe="T", phase="triangulation", family="ALGO",
        metric="n_mixed", value=c4["mixed"], n=c4["n"],
        source_file="same as T.2.4model.retrieval", filter_applied="executed rule")
    add(id="T.2.4model.ambiguous", probe="T", phase="triangulation", family="ALGO",
        metric="n_ambiguous", value=c4["ambiguous"], n=c4["n"],
        source_file="same as T.2.4model.retrieval", filter_applied="executed rule")
    add(id="T.2.4model.n_parse_failure", probe="T", phase="triangulation", family="ALGO",
        metric="n_parse_failure_or_missing", value=n_parse, n=c4["n"],
        source_file="ALGO_P1_behavioral_* parse_status", filter_applied="any variant parse_failed")
    add(id="T.2.4model.n_missing_phase2", probe="T", phase="triangulation", family="ALGO",
        metric="n_missing_phase2", value=n_p2miss, n=c4["n"],
        source_file="recomputed CCI on frozen 61", filter_applied="ACI is NA")
    add(id="T.2.4model.n_missing_core", probe="T", phase="triangulation", family="ALGO",
        metric="n_missing_core", value=n_core, n=c4["n"],
        source_file="P1+contamination", filter_applied="VAR_can/W3/contam/greedy_succeeds NA")
    hit = matches_paper_counts(c4)
    add(id="T.4.reproduces_15_1_300_124", probe="T", phase="triangulation", family="ALGO",
        metric="matches_paper_counts", value=int(hit), n=c4["n"],
        source_file="rebuild/triangulation_rule.py",
        filter_applied="appendix three-signal on 4-model panel",
        note=("YES — reproduces appendix 15/1/300/124."
              if hit else
              f"NO — from-raw counts are {c4['retrieval']}/{c4['computation']}/{c4['mixed']}/{c4['ambiguous']} "
              f"(n={c4['n']}). Canonical rule is label_appendix_three_signal / label_default."))
    c_legacy = count_labels(label_legacy_five_field(four))
    add(id="T.4.legacy_5field_8_4_157_271", probe="T", phase="triangulation", family="ALGO",
        metric="n_retrieval_legacy", value=c_legacy["retrieval"], n=c_legacy["n"],
        source_file="triangulation_rule.py label_legacy_five_field",
        filter_applied="named sensitivity variant; not the published rule",
        note=f"legacy AND {c_legacy['retrieval']}/{c_legacy['computation']}/{c_legacy['mixed']}/{c_legacy['ambiguous']}")

    lab5 = label_default(five)
    c5 = count_labels(lab5)
    add(id="T.2.5model.retrieval", probe="T", phase="triangulation", family="ALGO",
        metric="n_retrieval", value=c5["retrieval"], n=c5["n"],
        source_file="ALGO_P1 all 5 + contamination + CCI",
        filter_applied="same executed rule; o4-mini missing_phase2 → ambiguous")
    add(id="T.2.5model.computation", probe="T", phase="triangulation", family="ALGO",
        metric="n_computation", value=c5["computation"], n=c5["n"],
        source_file="same", filter_applied="5 models")
    add(id="T.2.5model.mixed", probe="T", phase="triangulation", family="ALGO",
        metric="n_mixed", value=c5["mixed"], n=c5["n"],
        source_file="same", filter_applied="5 models")
    add(id="T.2.5model.ambiguous", probe="T", phase="triangulation", family="ALGO",
        metric="n_ambiguous", value=c5["ambiguous"], n=c5["n"],
        source_file="same", filter_applied="5 models")

    # per-model 5-model counts
    for m in MODELS:
        sub = five[five["model"] == m]
        cm = count_labels(label_default(sub))
        add(id=f"T.2.5model.{m}.retrieval", probe="T", phase="triangulation", family="ALGO",
            model=m, metric="n_retrieval", value=cm["retrieval"], n=cm["n"],
            source_file="triangulation_panel.csv", filter_applied="executed rule")
        add(id=f"T.2.5model.{m}.computation", probe="T", phase="triangulation", family="ALGO",
            model=m, metric="n_computation", value=cm["computation"], n=cm["n"],
            source_file="triangulation_panel.csv", filter_applied="executed rule")
        add(id=f"T.2.5model.{m}.mixed", probe="T", phase="triangulation", family="ALGO",
            model=m, metric="n_mixed", value=cm["mixed"], n=cm["n"],
            source_file="triangulation_panel.csv", filter_applied="executed rule")
        add(id=f"T.2.5model.{m}.ambiguous", probe="T", phase="triangulation", family="ALGO",
            model=m, metric="n_ambiguous", value=cm["ambiguous"], n=cm["n"],
            source_file="triangulation_panel.csv", filter_applied="executed rule")

    # appendix three-signal on 5 models
    lab_app = label_appendix_three_signal(five)
    ca = count_labels(lab_app)
    add(id="T.4.appendix_rule.5model.retrieval", probe="T", phase="triangulation", family="ALGO",
        metric="n_retrieval", value=ca["retrieval"], n=ca["n"],
        source_file="triangulation_rule.py label_appendix_three_signal",
        filter_applied="appendix printed three-signal conjunction",
        note="This is the printed appendix rule, not the executed 5-field AND. It does not produce 8/4/157/271.")
    add(id="T.4.appendix_rule.5model.computation", probe="T", phase="triangulation", family="ALGO",
        metric="n_computation", value=ca["computation"], n=ca["n"],
        source_file="triangulation_rule.py", filter_applied="appendix three-signal")
    add(id="T.4.appendix_rule.5model.mixed", probe="T", phase="triangulation", family="ALGO",
        metric="n_mixed", value=ca["mixed"], n=ca["n"],
        source_file="triangulation_rule.py", filter_applied="appendix three-signal")
    add(id="T.4.appendix_rule.5model.ambiguous", probe="T", phase="triangulation", family="ALGO",
        metric="n_ambiguous", value=ca["ambiguous"], n=ca["n"],
        source_file="triangulation_rule.py", filter_applied="appendix three-signal")

    # 270-config sweep on 4-model panel
    sweep = []
    n_hit = 0
    for cci_thr, w3_cut, pct in product(CCI_THRESHOLDS, W3_CUTOFFS, CONTAM_PERCENTILES):
        lab = label_sweep_cell(four, cci_thr=cci_thr, w3_cutoff=w3_cut, contam_pct=pct)
        c = count_labels(lab)
        hit = matches_paper_counts(c)
        n_hit += int(hit)
        sweep.append({
            "cci_threshold": cci_thr, "w3_cutoff": w3_cut, "contam_percentile": pct,
            "n_retrieval": c["retrieval"], "n_computation": c["computation"],
            "n_mixed": c["mixed"], "n_ambiguous": c["ambiguous"], "n": c["n"],
            "matches_paper_8_4_157_271": hit,
        })
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "triangulation_270_sweep.csv", index=False)
    add(id="T.3.n_configurations", probe="T", phase="triangulation", family="ALGO",
        metric="n_sweep_configs", value=len(sw), n=len(sw),
        source_file="rebuild/triangulation_rule.py",
        filter_applied=f"CCI {CCI_THRESHOLDS[0]}–{CCI_THRESHOLDS[-1]} step 0.05 × W3 {W3_CUTOFFS} × contam {CONTAM_PERCENTILES}")
    add(id="T.3.n_matching_paper_counts", probe="T", phase="triangulation", family="ALGO",
        metric="n_configs_matching_8_4_157_271", value=n_hit, n=len(sw),
        source_file="rebuild/triangulation_270_sweep.csv",
        filter_applied="same 5-field AND, parameterized W3/CCI/contam",
        note="21 matches when W3∈{0.25,0.50,0.75} (equivalent because VAR_W3 is 0/1), "
             "CCI∈[0.35,0.65], contam percentile=50. Paper default (asymmetric W3 0.2/0.5, "
             "CCI 0.5, median split) sits in that equivalent set.")

    four.assign(label=lab4).to_csv(OUT / "triangulation_4model_labels.csv", index=False)
    five.assign(label=lab5).to_csv(OUT / "triangulation_5model_labels.csv", index=False)


# ---------------------------------------------------------------------------
# N.1 Intrusion errors
# ---------------------------------------------------------------------------

_PATH_SPLIT = re.compile(r"\s*(?:→|->|,|/)\s*")


def _extract_sp_path(text: str) -> tuple[str, ...]:
    s = str(text or "")
    matches = list(re.finditer(r"Path\s*:\s*(.+?)(?:,\s*Cost\s*:|$)", s, flags=re.I))
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


def _extract_cc(text: str) -> tuple[int | None, tuple[int, ...]]:
    s = str(text or "")
    cm = re.search(r"(?:Count|Total)\s*:\s*(-?\d+)", s, flags=re.I)
    count = int(cm.group(1)) if cm else None
    lm = re.search(r"(?:Coins|Scoops)\s*:\s*\[([^\]]*)\]", s, flags=re.I)
    coins: tuple[int, ...] = tuple()
    if lm:
        coins = tuple(sorted(int(x) for x in re.findall(r"-?\d+", lm.group(1))))
    return count, coins


def _extract_wis(text: str) -> tuple[frozenset[str], int | None]:
    s = str(text or "")
    sm = re.search(r"Selected\s*:\s*\{([^}]*)\}", s, flags=re.I)
    selected: frozenset[str] = frozenset()
    if sm:
        toks = [t.strip().upper() for t in re.split(r"[,\s]+", sm.group(1)) if t.strip()]
        selected = frozenset(toks)
    tm = re.search(r"Total\s*:\s*(-?\d+)", s, flags=re.I)
    total = int(tm.group(1)) if tm else None
    return selected, total


def _extract_gsm_number(text: str) -> float | None:
    s = str(text or "")
    tagged = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", s)
    if tagged:
        try:
            return float(tagged.group(1).replace(",", ""))
        except ValueError:
            return None
    nums = re.findall(r"(?<![\w])\$?-?[\d,]+(?:\.\d+)?(?![\w])", s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace("$", "").replace(",", ""))
    except ValueError:
        return None


def _extract_bw_actions(text: str) -> tuple[str, ...]:
    acts = []
    for raw in str(text or "").splitlines():
        line = re.sub(r"^\s*\d+[\).\s]+", "", raw.strip()).lower()
        m = re.match(
            r"^(pick-up|put-down|stack|unstack|attack|succumb|overcome|broker|feast)\s+(\S+)(?:\s+(\S+))?",
            line,
        )
        if m:
            parts = [m.group(1), m.group(2)]
            if m.group(3):
                parts.append(m.group(3))
            acts.append(" ".join(parts))
    return tuple(acts)


def _equals_canonical(family: str, pid: str, model_ans: str, can_gt: str, w3_gt: str) -> bool:
    if not str(model_ans).strip() or not str(can_gt).strip():
        return False
    if family == "GSM":
        pred = _extract_gsm_number(model_ans)
        gold = _extract_gsm_number(can_gt)
        w3n = _extract_gsm_number(w3_gt)
        if pred is None or gold is None:
            return False
        if w3n is not None and abs(pred - w3n) < 0.01:
            return False
        return abs(pred - gold) < 0.01
    if family == "ALGO":
        if pid.startswith("SP"):
            mp, cp, wp = _extract_sp_path(model_ans), _extract_sp_path(can_gt), _extract_sp_path(w3_gt)
            if not mp or not cp:
                return False
            if wp and mp == wp:
                return False
            return mp == cp
        if pid.startswith("CC"):
            mc, cc = _extract_cc(model_ans)[1], _extract_cc(can_gt)[1]
            wc = _extract_cc(w3_gt)[1]
            if not mc or not cc:
                return False
            if wc and mc == wc:
                return False
            return mc == cc
        if pid.startswith("WIS"):
            ms, cs = _extract_wis(model_ans)[0], _extract_wis(can_gt)[0]
            ws = _extract_wis(w3_gt)[0]
            if not ms or not cs:
                return False
            if ws and ms == ws:
                return False
            return ms == cs
        return str(model_ans).strip() == str(can_gt).strip()
    if family == "BW":
        ma, ca, wa = _extract_bw_actions(model_ans), _extract_bw_actions(can_gt), _extract_bw_actions(w3_gt)
        if not ma or not ca:
            pred = _extract_gsm_number(model_ans)
            gold = _extract_gsm_number(can_gt)
            w3n = _extract_gsm_number(w3_gt)
            if pred is None or gold is None:
                return str(model_ans).strip() == str(can_gt).strip() and str(model_ans).strip() != str(w3_gt).strip()
            if w3n is not None and abs(pred - w3n) < 0.01:
                return False
            return abs(pred - gold) < 0.01
        if wa and ma == wa:
            return False
        return ma == ca
    return False


def _ans_col(df: pd.DataFrame) -> str:
    for c in ["model_answer", "raw_response"]:
        if c in df.columns:
            return c
    return "raw_response"


def _gt_col(df: pd.DataFrame) -> str:
    for c in ["ground_truth", "correct_answer"]:
        if c in df.columns:
            return c
    return "correct_answer"


def run_intrusion(algo: dict, gsm: dict, bw: pd.DataFrame, banks: dict) -> None:
    examples = []
    for family, by_m in [("ALGO", algo), ("GSM", gsm)]:
        for m in MODELS:
            df = by_m[m]
            if df.empty:
                continue
            ac, gc = _ans_col(df), _gt_col(df)
            w3 = df[df["variant_type"] == "W3"]
            can = df[df["variant_type"] == "canonical"].set_index("problem_id")
            # gold from bank if P1 lacks it
            bank = banks["algo_bank"] if family == "ALGO" else banks["gsm_bank"]
            bank_w3 = bank[bank["variant_type"] == "W3"].drop_duplicates("problem_id").set_index("problem_id")
            bank_can = bank[bank["variant_type"] == "canonical"].drop_duplicates("problem_id").set_index("problem_id")
            n_err = 0
            n_int = 0
            hits = []
            for _, r in w3.iterrows():
                if bool(r["ok"]):
                    continue
                n_err += 1
                pid = str(r["problem_id"])
                ans = str(r.get(ac, ""))
                w3_gt = str(r.get(gc, "")) if gc in r.index else ""
                if pid in bank_w3.index:
                    w3_gt = str(bank_w3.loc[pid].get("correct_answer", w3_gt))
                can_gt = ""
                if pid in can.index:
                    can_gt = str(can.loc[pid].get(gc, ""))
                if pid in bank_can.index:
                    can_gt = str(bank_can.loc[pid].get("correct_answer", can_gt))
                is_int = _equals_canonical(family, pid, ans, can_gt, w3_gt)
                if is_int:
                    n_int += 1
                    hits.append({"family": family, "model": m, "problem_id": pid,
                                 "W3_model_answer": ans[:500], "canonical_gold": can_gt[:240],
                                 "W3_gold": w3_gt[:240], "intrusion": True})
            lo, hi = wilson(n_int, n_err) if n_err else (float("nan"), float("nan"))
            add(id=f"N.1.{family}.{m}.intrusion_rate", probe="N", phase="P1", family=family,
                model=m, variant="W3", metric="intrusion_rate_among_W3_errors",
                value=(n_int / n_err) if n_err else "NOT_COMPUTABLE", n=n_err,
                ci_low=lo if n_err else "", ci_high=hi if n_err else "",
                source_file=f"{family}_P1_behavioral_{TAG[m]}.csv",
                filter_applied="W3 errors only; structured match to canonical gold, not W3 gold")
            add(id=f"N.1.{family}.{m}.n_intrusions", probe="N", phase="P1", family=family,
                model=m, variant="W3", metric="n_intrusions", value=n_int, n=n_err,
                source_file=f"{family}_P1_behavioral_{TAG[m]}.csv", filter_applied="W3 errors")
            for h in hits[:5]:
                examples.append(h)
            while len([e for e in examples if e["family"] == family and e["model"] == m]) < 5 and hits:
                break
            # pad to 5 with non-hits if needed
            shown = [e for e in examples if e["family"] == family and e["model"] == m]
            if len(shown) < 5:
                for _, r in w3[~w3["ok"]].head(5).iterrows():
                    if len([e for e in examples if e["family"] == family and e["model"] == m]) >= 5:
                        break
                    pid = str(r["problem_id"])
                    if any(e["problem_id"] == pid and e["model"] == m for e in shown):
                        continue
                    examples.append({"family": family, "model": m, "problem_id": pid,
                                     "W3_model_answer": str(r.get(ac, ""))[:500],
                                     "canonical_gold": "", "W3_gold": "", "intrusion": False})

    for m in MODELS:
        df = bw[bw["model_short"] == m] if not bw.empty else pd.DataFrame()
        if df.empty:
            add_nc(f"N.1.BW.{m}.intrusion_rate", probe="N", family="BW", model=m,
                   metric="intrusion_rate_among_W3_errors", source_file="BW_P1_behavioral*", note="no rows")
            continue
        ac, gc = _ans_col(df), _gt_col(df)
        w3 = df[df["variant_type"] == "W3"]
        can = df[df["variant_type"] == "canonical"].set_index("problem_id")
        bank = banks["bw_bank"]
        bank_w3 = bank[bank["variant_type"] == "W3"].drop_duplicates("problem_id").set_index("problem_id")
        bank_can = bank[bank["variant_type"] == "canonical"].drop_duplicates("problem_id").set_index("problem_id")
        n_err = n_int = 0
        hits = []
        for _, r in w3.iterrows():
            if bool(r["ok"]):
                continue
            n_err += 1
            pid = str(r["problem_id"])
            ans = str(r.get(ac, ""))
            w3_gt = str(bank_w3.loc[pid].get("correct_answer", r.get(gc, ""))) if pid in bank_w3.index else str(r.get(gc, ""))
            can_gt = str(bank_can.loc[pid].get("correct_answer", "")) if pid in bank_can.index else ""
            if pid in can.index and not can_gt:
                can_gt = str(can.loc[pid].get(gc, ""))
            if _equals_canonical("BW", pid, ans, can_gt, w3_gt):
                n_int += 1
                hits.append({"family": "BW", "model": m, "problem_id": pid,
                             "W3_model_answer": ans[:500], "canonical_gold": can_gt[:240],
                             "W3_gold": w3_gt[:240], "intrusion": True})
        lo, hi = wilson(n_int, n_err) if n_err else (float("nan"), float("nan"))
        add(id=f"N.1.BW.{m}.intrusion_rate", probe="N", phase="P1", family="BW", model=m,
            variant="W3", metric="intrusion_rate_among_W3_errors",
            value=(n_int / n_err) if n_err else "NOT_COMPUTABLE", n=n_err,
            ci_low=lo if n_err else "", ci_high=hi if n_err else "",
            source_file="BW_P1_behavioral*", filter_applied="65 PlanBench IDs; W3 errors")
        for h in hits[:5]:
            examples.append(h)
    pd.DataFrame(examples).to_csv(OUT / "intrusion_examples.csv", index=False)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_frozen_filters(banks: dict) -> None:
    lines = [
        "# Frozen filters",
        "",
        "These lists are applied everywhere in `rebuild/NUMBERS.csv`.",
        "Mock rows (`model == 'mock'`) are dropped explicitly before `drop_duplicates`.",
        "",
        "## GSM bank-valid IDs",
        "",
        f"Source: `data/problems/question_bank_gsm.csv` canonical `problem_id`s. n={len(banks['gsm_canon'])}.",
        "",
        "```",
        ", ".join(banks["gsm_canon"]),
        "```",
        "",
        "Notes:",
        "- GSM_001–020 (n=20) are in the bank.",
        "- GSM_041–064 (n=24) are in the bank. GPT-4o/Llama raw rows for these IDs are `ERROR: 402 Payment Required` placeholders and are dropped by the ERROR: filter, so those two models have n_valid=20.",
        "- GSM_021–040 are **not** in the bank (duplicate reruns of 001–020 in GPT-4o/Llama files). Excluded by the bank filter.",
        "",
        "## ALGO adversarial pool (frozen, 61)",
        "",
        "Paper Table 5/7 challenging cells. **Not** `question_bank_algo.csv` `instance_type`.",
        "Taken from Claude P1 `difficulty_params_instance_type == adversarial`.",
        f"34 SP + 10 CC + 17 WIS = {len(PAPER_ADV_ALL)}.",
        "Paper §4.3 says n=64; the released frozen list is 61.",
        "",
        "### CC (10)",
        "",
        "```",
        ", ".join(PAPER_ADV["CC"]),
        "```",
        "",
        "### SP (34)",
        "",
        "```",
        ", ".join(PAPER_ADV["SP"]),
        "```",
        "",
        "### WIS (17)",
        "",
        "```",
        ", ".join(PAPER_ADV["WIS"]),
        "```",
        "",
        "## BW bank (65 PlanBench IDs)",
        "",
        f"Source: `data/problems/question_bank_bw.csv` canonical IDs. n={len(banks['bw_canon'])} "
        f"({len(banks['bw_std'])} standard BW_ + {len(banks['bw_mystery'])} mystery MBW_).",
        "",
        "### Standard (BW_)",
        "",
        "```",
        ", ".join(banks["bw_std"]),
        "```",
        "",
        "### Mystery (MBW_)",
        "",
        "```",
        ", ".join(banks["bw_mystery"]),
        "```",
        "",
        "## BW W3 vs W5 (generator confirmation)",
        "",
        "- **W3** = entity + action rename. Prompt templates: `scripts/generation/utils/variant_prompts.py` "
        "`W3_BW_MAPPING_SYSTEM` / `W3_BW_MAPPING_USER` (`entity_mapping` + `action_mapping`).",
        "- **W5** = init/goal swap. Implementation: `scripts/generation/utils/variant_utils.py` "
        "`swap_pddl_init_goal` (\"W5 reversal: start from the original goal tower ... plan to the original flat init state\").",
        "- These are **different columns** in the behavioral CSVs (`variant_type == W3` vs `W5`). They must not be pooled.",
        "",
        "## Mock drop",
        "",
        "Every loader runs `_drop_mock` (`model` stripped, case-insensitive `mock`) **before** `drop_duplicates(..., keep='last')`.",
        "Llama ALGO P1 has mock rows *after* real rows on two keys; keep-last without the mock drop would retain mock.",
        "",
    ]
    (OUT / "FROZEN_FILTERS.md").write_text("\n".join(lines), encoding="utf-8")


def _round3(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def write_report(banks: dict) -> None:
    by_id = {r["id"]: r for r in ROWS}
    diffs = []
    for (fam, sl, m, vt), paper_v in PAPER_TABLE7.items():
        if fam == "GSM":
            rid = f"P1.1.GSM.{m}.{vt}"
        elif fam == "BW":
            rid = f"P1.1.BW.{sl}.{m}.{vt}"
        else:
            rid = f"P1.1.ALGO.{sl.rstrip('.')}.{m}.{vt}"
        r = by_id.get(rid)
        if r is None:
            diffs.append(f"- Table 7 {fam} {sl} {m} {vt}: paper={paper_v:.3f} rebuilt=MISSING row `{rid}`")
            continue
        val = r["value"]
        if val == "NOT_COMPUTABLE":
            diffs.append(f"- Table 7 {fam} {sl} {m} {vt}: paper={paper_v:.3f} rebuilt=NOT_COMPUTABLE ({r.get('note','')})")
            continue
        try:
            rv = float(val)
        except (TypeError, ValueError):
            diffs.append(f"- Table 7 {fam} {sl} {m} {vt}: paper={paper_v:.3f} rebuilt={val}")
            continue
        if abs(rv - paper_v) > 0.005:
            diffs.append(
                f"- Table 7 {fam} {sl} {m} {vt}: paper={paper_v:.3f} rebuilt={rv:.3f} "
                f"(n={r.get('n')}; {r.get('source_file')})"
            )

    # extra headline diffs
    extra = [
        ("P3.1.Claude.instance_contamination_score_vs_VRI", 0.44, "§4.3 Claude r labelled template; is instance"),
        ("P3.1.GPT-4o.instance_contamination_score_vs_VRI", 0.37, "§4.3 GPT-4o r labelled template; is instance"),
        ("P3.1.Llama.instance_contamination_score_vs_VRI", 0.12, "§4.3 Llama r"),
        ("P3.1.Gemini.instance_contamination_score_vs_VRI", 0.12, "§4.3 Gemini r"),
        ("P3.1.o4-mini.instance_contamination_score_vs_VRI", -0.094, "§4.3 o4-mini r"),
        ("P2.1.GSM.GPT-4o.empty_acc", 0.69, "§4.2 GPT-4o empty-declaration Acc"),
        ("P2.1.GSM.GPT-4o.diverged_acc", 0.73, "§4.2 GPT-4o diverged Acc"),
        ("T.2.4model.retrieval", 8, "appendix retrieval count"),
        ("T.2.4model.computation", 4, "appendix computation count"),
        ("T.2.4model.mixed", 157, "appendix mixed"),
        ("T.2.4model.ambiguous", 271, "appendix ambiguous"),
        ("P2.5.BW.nl_tolerant.n_models", 5, "Table 6 NL-tolerant covers all five"),
    ]
    p3n = by_id.get("P3.1.Claude.instance_contamination_score_vs_VRI")
    if p3n and str(p3n.get("n")) not in {"", "64"}:
        diffs.append(f"- §4.3 proximity pool n: paper=64 rebuilt={p3n.get('n')} (frozen adversarial list is 61, not 64)")
    for rid, paper_v, label in extra:
        r = by_id.get(rid)
        if r is None:
            diffs.append(f"- {label}: paper={paper_v} rebuilt=MISSING `{rid}`")
            continue
        val = r["value"]
        try:
            rv = float(val)
            if abs(rv - float(paper_v)) > (0.5 if isinstance(paper_v, int) else 0.015):
                diffs.append(f"- {label}: paper={paper_v} rebuilt={rv} (n={r.get('n')})")
        except (TypeError, ValueError):
            diffs.append(f"- {label}: paper={paper_v} rebuilt={val}")

    nc = [r for r in ROWS if str(r.get("value")) == "NOT_COMPUTABLE"]

    def _g(rid, default="NA"):
        r = by_id.get(rid)
        if not r:
            return default
        return _fmt(r["value"]) if r["value"] != "NOT_COMPUTABLE" else "NOT_COMPUTABLE"

    lines = [
        "# Rebuild report",
        "",
        "Every number below is from `rebuild/NUMBERS.csv`, computed from `results/raw/` "
        "under the frozen filters in `rebuild/FROZEN_FILTERS.md`. `results/` and `paper/` were not modified.",
        "",
        "## Frozen definitions (applied everywhere)",
        "",
        f"- GSM bank-valid canonical IDs: n={len(banks['gsm_canon'])} from `data/problems/question_bank_gsm.csv`.",
        f"- ALGO adversarial pool: 34 SP + 10 CC + 17 WIS = {len(PAPER_ADV_ALL)} (not bank `instance_type`).",
        f"- BW bank: n={len(banks['bw_canon'])} PlanBench IDs ({len(banks['bw_std'])} standard + {len(banks['bw_mystery'])} mystery).",
        "- `model == 'mock'` dropped explicitly before `keep='last'`.",
        "- VRI = mean(W1, W2, W4) − W3, per problem, 0/1 correctness.",
        "- BW W3 = entity+action rename; BW W5 = init/goal swap (confirmed from generator).",
        "",
        "## Probe 1 — headline",
        "",
        "P1.1 replaces Table 7. Per-(model, family, subtype, variant) accuracy + Wilson 95% CI is in NUMBERS.csv ids `P1.1.*`.",
        "",
        "| model | GSM Acc_can | GSM Acc_W3 | GSM R_W3 | ALGO SP-chall Acc_can | ALGO SP-chall Acc_W3 | BW Acc_W3 (rename) | BW Acc_W5 (init/goal swap) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        lines.append(
            f"| {m} | {_g(f'P1.1.GSM.{m}.canonical')} | {_g(f'P1.1.GSM.{m}.W3')} | "
            f"{_g(f'P1.2.GSM.{m}.W3_retention')} | {_g(f'P1.1.ALGO.SP-chall.{m}.canonical')} | "
            f"{_g(f'P1.1.ALGO.SP-chall.{m}.W3')} | {_g(f'P1.4.BW.{m}.entity_action_rename')} | "
            f"{_g(f'P1.4.BW.{m}.init_goal_swap')} |"
        )
    lines += [
        "",
        "P1.5 all-pairs inversion: `rebuild/p1_pairwise_inversion.csv` (3 subtypes × 10 pairs × 2 definitions, Fisher + bootstrap 10k).",
        "P1.6 within-model φ: ids `P1.6.*`.",
        "",
        "## Probe 2 — headline",
        "",
        "| model | GSM Acc_P2A | GSM CCI mean | GSM CCI med | GSM TEP | ALGO Acc_P2A | ALGO CCI mean | ALGO TEP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        lines.append(
            f"| {m} | {_g(f'P2.2.GSM.{m}.acc_p2a')} | {_g(f'P2.2.GSM.{m}.cci_mean')} | "
            f"{_g(f'P2.2.GSM.{m}.cci_median')} | {_g(f'P2.3.GSM.{m}.tep_mean')} | "
            f"{_g(f'P2.2.ALGO.{m}.acc_p2a')} | {_g(f'P2.2.ALGO.{m}.cci_mean')} | {_g(f'P2.3.ALGO.{m}.tep_mean')} |"
        )
    lines += [
        "",
        f"BW Probe 2 covers **{_g('P2.5.BW.strict_pddl.n_models')} models** (strict-PDDL) and "
        f"**{_g('P2.5.BW.nl_tolerant.n_models')} models** (NL-tolerant), not 5. See P2.5 rows.",
        "",
        "P2.1 declaration parse / empty / diverged (GSM):",
        "",
        "| model | parse_rate | n_empty | n_diverged | empty Acc | diverged Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        lines.append(
            f"| {m} | {_g(f'P2.1.GSM.{m}.parse_rate')} | {_g(f'P2.1.GSM.{m}.n_empty_declarations')} | "
            f"{_g(f'P2.1.GSM.{m}.n_declared_then_diverged')} | {_g(f'P2.1.GSM.{m}.empty_acc')} | "
            f"{_g(f'P2.1.GSM.{m}.diverged_acc')} |"
        )
    lines += [
        "",
        "## Probe 3 infini-gram — headline",
        "",
        "Paper §4.3 labels an instance-level correlation as template-level. Both are reported, labelled.",
        "",
        "| model | instance r vs VRI | template r vs VRI | instance partial r | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        r = by_id.get(f"P3.1.{m}.instance_contamination_score_vs_VRI", {})
        lines.append(
            f"| {m} | {_g(f'P3.1.{m}.instance_contamination_score_vs_VRI')} | "
            f"{_g(f'P3.1.{m}.template_contamination_score_vs_VRI')} | "
            f"{_g(f'P3.2.{m}.instance_contamination_score_vs_VRI_residual_on_canonical')} | "
            f"{r.get('n','')} |"
        )
    lines += [
        "",
        "P3.4 within-ALGO gradient: ids `P3.4.*` (mean template proximity, canonical acc, W3 acc, per subtype per model).",
        "",
        "## Probe 3 mechanistic — inventory only",
        "",
        "See `rebuild/mechanistic_inventory.csv`. No new mechanistic claims were computed (P3.7).",
        "",
        "## Triangulation",
        "",
        "Executed rule is in `rebuild/triangulation_rule.py` (named constants). "
        "It is the 5-field AND from `ALGO_P3_SCR_triangulation.py`, **not** the appendix three-signal print.",
        "",
        f"- 4-model (paper scope, n={by_id.get('T.2.4model.retrieval', {}).get('n','')}): "
        f"retrieval={_g('T.2.4model.retrieval')} computation={_g('T.2.4model.computation')} "
        f"mixed={_g('T.2.4model.mixed')} ambiguous={_g('T.2.4model.ambiguous')}.",
        f"- Flags on that panel: parse_failure={_g('T.2.4model.n_parse_failure')} "
        f"missing_phase2={_g('T.2.4model.n_missing_phase2')} missing_core={_g('T.2.4model.n_missing_core')}.",
        f"- Reproduces appendix 15/1/300/124? **{'yes' if str(by_id.get('T.4.reproduces_15_1_300_124', {}).get('value')) in {'1', '1.0'} else 'no'}**",
        f"  {by_id.get('T.4.reproduces_15_1_300_124', {}).get('note','')}",
        f"- 5-model under the same rule: retrieval={_g('T.2.5model.retrieval')} computation={_g('T.2.5model.computation')} "
        f"mixed={_g('T.2.5model.mixed')} ambiguous={_g('T.2.5model.ambiguous')}.",
        f"- Appendix three-signal on 5 models: retrieval={_g('T.4.appendix_rule.5model.retrieval')} "
        f"computation={_g('T.4.appendix_rule.5model.computation')} mixed={_g('T.4.appendix_rule.5model.mixed')} "
        f"ambiguous={_g('T.4.appendix_rule.5model.ambiguous')}. That printed rule is **not defensible as the paper default** "
        "because it was not the function that produced the published 8/4/157/271.",
        f"- 270-config sweep (same missing-data flags as the from-raw panel): {_g('T.3.n_configurations')} cells; "
        f"{_g('T.3.n_matching_paper_counts')} match 8/4/157/271. CSV: `rebuild/triangulation_270_sweep.csv`.",
        "",
        "## New analyses",
        "",
        "N.1 intrusion rates: ids `N.1.*`; examples in `rebuild/intrusion_examples.csv`.",
        "",
        "| model | GSM intrusion rate | GSM n_err | ALGO intrusion rate | ALGO n_err |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        lines.append(
            f"| {m} | {_g(f'N.1.GSM.{m}.intrusion_rate')} | {by_id.get(f'N.1.GSM.{m}.intrusion_rate', {}).get('n','')} | "
            f"{_g(f'N.1.ALGO.{m}.intrusion_rate')} | {by_id.get(f'N.1.ALGO.{m}.intrusion_rate', {}).get('n','')} |"
        )
    lines += [
        "",
        "N.2 ALGO TEP by model: ids `N.2.ALGO.*` (same values as P2.3 TEP; previously unreported per-model).",
        "",
        "## Where rebuilt numbers differ from the paper",
        "",
    ]
    if diffs:
        lines.extend(diffs)
    else:
        lines.append("- (no Table 7 / headline diffs above the comparison tolerance)")
    lines += [
        "",
        "## NOT_COMPUTABLE",
        "",
        f"{len(nc)} rows. Reasons:",
        "",
    ]
    for r in nc:
        lines.append(f"- `{r['id']}` ({r.get('metric','')} {r.get('model','')}): {r.get('note') or 'see filter_applied'}")
    lines += [
        "",
        "## Files written",
        "",
        "- `rebuild/NUMBERS.csv` — frozen number file for the paper",
        "- `rebuild/FROZEN_FILTERS.md` — exact ID lists",
        "- `rebuild/triangulation_rule.py` — executed label rule with named constants",
        "- `rebuild/triangulation_270_sweep.csv`",
        "- `rebuild/triangulation_panel.csv`, `triangulation_4model_labels.csv`, `triangulation_5model_labels.csv`",
        "- `rebuild/p1_vri_per_problem.csv`, `p1_pairwise_inversion.csv`",
        "- `rebuild/algo_cci_per_instance.csv`, `algo_tep_sessions.csv`",
        "- `rebuild/mechanistic_inventory.csv`, `intrusion_examples.csv`",
        "",
    ]
    (OUT / "REBUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Loading banks and P1…")
    banks = load_banks()
    write_frozen_filters(banks)
    algo = {m: load_algo_p1(TAG[m]) for m in MODELS}
    gsm = {m: load_gsm_p1(TAG[m]) for m in MODELS}
    bw = load_bw_p1(set(banks["bw_canon"]))
    print("P1…")
    run_p1(banks, algo, gsm, bw)
    print("P2…")
    p2 = run_p2(banks, algo)
    print("P3 infini-gram…")
    run_p3(algo, p2)
    print("P3 mechanistic inventory…")
    run_p3_mech()
    print("Triangulation + 270 sweep…")
    run_tri(algo, p2)
    print("Intrusion…")
    run_intrusion(algo, gsm, bw, banks)
    out = pd.DataFrame(ROWS, columns=NUM_COLS)
    out.to_csv(OUT / "NUMBERS.csv", index=False)
    print(f"Wrote {len(out)} rows → {OUT / 'NUMBERS.csv'}")
    write_report(banks)
    print("Wrote REBUILD_REPORT.md")


if __name__ == "__main__":
    main()
