#!/usr/bin/env python3
"""Audit 2026-08 follow-up analyses A–G. Existing raw logs only. No API calls.

Writes CSVs and short markdown notes under audit_2026_08/new/.
"""
from __future__ import annotations

import math
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/adya/Desktop/rvc")
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"
DATA = ROOT / "data" / "problems"
OUT = ROOT / "audit_2026_08" / "new"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from probes.algo.decision_normalize import normalize_phase2_decision  # noqa: E402
from scripts.runs.coverage_audit import filter_p1_to_bank, _norm_variant  # noqa: E402

RNG = np.random.default_rng(42)
N_BOOT = 10_000

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
TAG = {
    "Claude": "claude",
    "Gemini": "gemini",
    "GPT-4o": "gpt4o",
    "Llama": "llama",
    "o4-mini": "o1mini",
}
LONG = {v: k for k, v in SHORT.items()}
MODELS = ["Claude", "GPT-4o", "Llama", "Gemini", "o4-mini"]
REAL_MODELS = set(SHORT.keys())
GEMINI_LONG = "google/gemini-2.5-flash"

# Frozen adversarial pool (paper Table 5/7 challenging cells): 34 SP + 10 CC + 17 WIS = 61.
# Paper §4.3 says n=64; the released frozen list is 61.
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
PID_TO_SUB = {}
for sub, ids in PAPER_ADV.items():
    for pid in ids:
        PID_TO_SUB[pid] = sub

ALGO_NAME_PATTERNS = [
    ("dijkstra", re.compile(r"\bdijkstra'?s?\b", re.I)),
    ("bellman_ford", re.compile(r"\bbellman[-\s]?ford\b", re.I)),
    ("floyd_warshall", re.compile(r"\bfloyd(?:[-\s]?warshall)?\b", re.I)),
    ("astar", re.compile(r"\ba(?:\s*[-*]\s*star|\*)\b", re.I)),
    ("bfs", re.compile(r"\b(?:bfs|breadth[-\s]?first)\b", re.I)),
    ("dfs", re.compile(r"\b(?:dfs|depth[-\s]?first)\b", re.I)),
    ("dp", re.compile(r"\b(?:dynamic programming|\bdp\b|memoization|knapsack)\b", re.I)),
    ("greedy", re.compile(r"\bgreedy\b", re.I)),
    ("backtracking", re.compile(r"\bbacktracking\b", re.I)),
    ("interval_scheduling", re.compile(r"\b(?:weighted )?interval scheduling\b", re.I)),
]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().isin({"true", "1", "yes"})


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    raw = df.get("raw_response", df.get("model_answer", pd.Series([""] * len(df))))
    return ~raw.astype(str).str.startswith("ERROR:")


def _drop_mock(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df
    m = df["model"].astype(str).str.strip()
    return df[~m.str.lower().isin({"mock", ""}) & m.isin(REAL_MODELS)].copy()


def _short(model: str) -> str:
    return SHORT.get(str(model).strip(), str(model).strip())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    if k <= 0:
        # Wilson upper bound with p=0; force lower to exact 0
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


def _md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_algo_p1(tag: str) -> pd.DataFrame:
    df = _read(RAW / f"ALGO_P1_behavioral_{tag}.csv")
    if df.empty:
        return df
    df = _drop_mock(df)
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    df["model_short"] = df["model"].map(_short)
    df = df[_valid_mask(df)]
    df = df.drop_duplicates(["problem_id", "variant_type"], keep="last")
    return df


def algo_p1_all() -> dict[str, pd.DataFrame]:
    return {m: load_algo_p1(TAG[m]) for m in MODELS}


def load_gsm_p1(tag: str) -> pd.DataFrame:
    df = _read(RAW / f"GSM_P1_behavioral_{tag}.csv")
    if df.empty:
        return df
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    df = filter_p1_to_bank(df, "GSM")
    df = df[_valid_mask(df)]
    df = df.drop_duplicates(["problem_id", "variant_type"], keep="last")
    df["model_short"] = df["model"].map(_short) if "model" in df.columns else TAG.get(tag, tag)
    if "model" not in df.columns or df["model"].eq("").all():
        df["model"] = LONG.get(df["model_short"].iloc[0], df["model_short"].iloc[0]) if len(df) else ""
    return df


def load_bw_p1() -> pd.DataFrame:
    parts = []
    for name in ["BW_P1_behavioral.csv", "BW_P1_behavioral_gemini.csv", "BW_P1_behavioral_o1mini.csv"]:
        df = _read(RAW / name)
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = _drop_mock(df)
    df = df[df["problem_id"].astype(str).str.startswith(("BW_", "MBW_"))]
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    df = df[_valid_mask(df)]
    df["model_short"] = df["model"].map(_short)
    df = df.drop_duplicates(["problem_id", "model", "variant_type"], keep="last")
    return df


def load_phase2_normal(*, overlay_gemini_dedicated: bool = True, keep_main_gemini_rest: bool = False) -> pd.DataFrame:
    """Gemini dedicated rerun covers the 61 adversarial problems.

    TEP/P2B: drop main-file Gemini (use dedicated only).
    Phase-2A agreement: overlay dedicated on the 61, keep main-file Gemini for the other 49.
    """
    main = _drop_mock(_read(RAW / "ALGO_P2_phase2_normal.csv"))
    gem = _drop_mock(_read(RAW / "ALGO_P2_phase2_normal_gemini.csv"))
    if main.empty:
        return gem
    if overlay_gemini_dedicated:
        main_no_gem = main[main["model"] != GEMINI_LONG]
        if keep_main_gemini_rest and not gem.empty:
            gem_ids = set(gem["problem_id"].astype(str))
            rest = main[(main["model"] == GEMINI_LONG) & ~main["problem_id"].astype(str).isin(gem_ids)]
            parts = [p for p in (main_no_gem, rest, gem) if not p.empty]
            return pd.concat(parts, ignore_index=True)
        parts = [p for p in (main_no_gem, gem) if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return main


def load_phase2_injected() -> pd.DataFrame:
    """Plausible injection (P2B). Exclude implausible (different condition)."""
    main = _drop_mock(_read(RAW / "ALGO_P2_phase2_injected.csv"))
    gem = _drop_mock(_read(RAW / "ALGO_P2_phase2_injected_gemini.csv"))
    if not main.empty:
        main = main[main["model"] != GEMINI_LONG]
    parts = [p for p in (main, gem) if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_phase1_declarations() -> pd.DataFrame:
    """Prefer *_new files by concatenating old then new (keep=last)."""
    order = [
        "ALGO_P2_phase1_claude_new.csv",
        "ALGO_P2_phase1_gpt4o_new.csv",
        "ALGO_P2_phase1_llama_new.csv",
        "ALGO_P2_phase1_gemini.csv",
    ]
    parts = []
    for name in order:
        df = _drop_mock(_read(RAW / name))
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["problem_id", "model"], keep="last")
    return df


# ---------------------------------------------------------------------------
# A. All-pairs rename inversion
# ---------------------------------------------------------------------------

def _w3_canonical_maps(p1: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    w3, can = {}, {}
    for _, r in p1.iterrows():
        pid = str(r["problem_id"])
        ok = int(_is_true(pd.Series([r.get("verified", "")])).iloc[0])
        vt = r["variant_type"]
        if vt == "W3":
            w3[pid] = ok
        elif vt == "canonical":
            can[pid] = ok
    return w3, can


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


def run_a(p1_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    maps = {m: _w3_canonical_maps(df) for m, df in p1_by_model.items()}
    rows = []
    for subtype, ids in PAPER_ADV.items():
        idset = set(ids)
        for ma, mb in combinations(MODELS, 2):
            w3a, cana = maps[ma]
            w3b, canb = maps[mb]
            paired_ids = sorted(idset & set(w3a) & set(w3b))
            matched_ids = sorted(pid for pid in paired_ids if cana.get(pid) == 1 and canb.get(pid) == 1)
            for definition, use_ids in (("paired", paired_ids), ("canonically-matched", matched_ids)):
                n = len(use_ids)
                if n == 0:
                    rows.append({
                        "subtype": subtype, "model_a": ma, "model_b": mb,
                        "definition": definition, "n": 0,
                        "a_W3_correct": 0, "b_W3_correct": 0,
                        "fisher_p_two_sided": float("nan"),
                        "acc_diff_a_minus_b": float("nan"),
                        "acc_diff_ci95_lo": float("nan"),
                        "acc_diff_ci95_hi": float("nan"),
                    })
                    continue
                aa = np.array([w3a[pid] for pid in use_ids], dtype=int)
                bb = np.array([w3b[pid] for pid in use_ids], dtype=int)
                ka, kb = int(aa.sum()), int(bb.sum())
                table = np.array([[ka, n - ka], [kb, n - kb]], dtype=int)
                if (table == 0).all() or table.min() < 0:
                    p = float("nan")
                else:
                    try:
                        _, p = stats.fisher_exact(table, alternative="two-sided")
                    except ValueError:
                        p = float("nan")
                lo, hi = _boot_acc_diff(aa.astype(float), bb.astype(float))
                rows.append({
                    "subtype": subtype, "model_a": ma, "model_b": mb,
                    "definition": definition, "n": n,
                    "a_W3_correct": ka, "b_W3_correct": kb,
                    "fisher_p_two_sided": float(p) if p == p else float("nan"),
                    "acc_diff_a_minus_b": float(aa.mean() - bb.mean()),
                    "acc_diff_ci95_lo": lo,
                    "acc_diff_ci95_hi": hi,
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "pairwise_inversion.csv", index=False)
    return out


def note_a(df: pd.DataFrame) -> None:
    sig = df[(df["definition"] == "canonically-matched") & (df["fisher_p_two_sided"] < 0.05)]
    lines = [
        "# A. All-pairs rename inversion",
        "",
        f"Frozen adversarial pool: **{len(PAPER_ADV['SP'])} SP + {len(PAPER_ADV['CC'])} CC + {len(PAPER_ADV['WIS'])} WIS = {len(PAPER_ADV_ALL)}** problems (paper §4.3 says n=64; released frozen list is 61).",
        "Fisher exact is the 2×2 of (W3-correct, W3-wrong) counts for model A vs model B. Bootstrap 95% CI is a paired resample of `problem_id` (10,000) on accuracy difference (A−B).",
        "",
        f"**Rows:** {len(df)} (3 subtypes × 10 pairs × 2 definitions).",
        f"**Canonically-matched pairs with two-sided Fisher p<0.05:** {len(sig)}.",
        "",
    ]
    if len(sig):
        lines.append("| subtype | A | B | n | A W3 | B W3 | p | acc diff [CI] |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|")
        for _, r in sig.sort_values("fisher_p_two_sided").iterrows():
            lines.append(
                f"| {r.subtype} | {r.model_a} | {r.model_b} | {int(r.n)} | "
                f"{int(r.a_W3_correct)} | {int(r.b_W3_correct)} | {_fmt_p(r.fisher_p_two_sided)} | "
                f"{_fmt(r.acc_diff_a_minus_b, 3)} [{_fmt(r.acc_diff_ci95_lo, 3)}, {_fmt(r.acc_diff_ci95_hi, 3)}] |"
            )
        lines.append("")
    # highlight SP Claude vs GPT-4o which the paper reported
    sub = df[(df.subtype == "SP") & (df.model_a == "Claude") & (df.model_b == "GPT-4o")]
    lines.append("Claude vs GPT-4o (paper’s reported inversion pair):")
    for _, r in sub.iterrows():
        lines.append(
            f"- {r.definition}: n={int(r.n)}, Claude W3={int(r.a_W3_correct)}, "
            f"GPT-4o W3={int(r.b_W3_correct)}, p={_fmt_p(r.fisher_p_two_sided)}, "
            f"diff={_fmt(r.acc_diff_a_minus_b, 3)} [{_fmt(r.acc_diff_ci95_lo, 3)}, {_fmt(r.acc_diff_ci95_hi, 3)}]"
        )
    lines += [
        "",
        "**Flags:** none — all 10 pairs × 3 subtypes × 2 definitions computed from ALGO P1 `verified` after dropping `mock` and `ERROR:` rows. Pairwise n is the intersection of problems both models actually logged.",
    ]
    _md(OUT / "A_pairwise_inversion.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# B. ALGO TEP
# ---------------------------------------------------------------------------

def _norm_steps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_step"] = pd.to_numeric(out["step_index"], errors="coerce").fillna(0).astype(int)
    mins = out.groupby(["problem_id", "model"])["_step"].transform("min")
    out["_step"] = out["_step"] - mins
    return out


def compute_algo_tep(normal: pd.DataFrame, injected: pd.DataFrame) -> pd.DataFrame:
    if normal.empty or injected.empty:
        return pd.DataFrame(columns=["problem_id", "model", "model_short", "tep", "n_post_steps"])
    normal = _norm_steps(normal)
    injected = _norm_steps(injected)
    rows = []
    keys = injected.groupby(["problem_id", "model"])
    for (pid, model), gi in keys:
        gn = normal[(normal["problem_id"] == pid) & (normal["model"] == model)]
        if gn.empty:
            continue
        crit_s = pd.to_numeric(gi["critical_step_index"], errors="coerce").dropna()
        if crit_s.empty:
            continue
        crit = int(crit_s.iloc[0])
        if crit < 0:
            continue
        merged = gn.merge(
            gi[["_step", "parsed_decision", "response_type"]],
            on="_step",
            how="inner",
            suffixes=("_n", "_i"),
        )
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
            "problem_id": pid,
            "model": model,
            "model_short": _short(model),
            "tep": float(diffs.mean()),
            "n_post_steps": int(len(post)),
        })
    return pd.DataFrame(rows)


def run_b(normal: pd.DataFrame, injected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inj_cols = list(_read(RAW / "ALGO_P2_phase2_injected.csv").columns) if (RAW / "ALGO_P2_phase2_injected.csv").exists() else []
    missing_tep_col = "tep_score" not in inj_cols
    sess = compute_algo_tep(normal, injected)
    rows = []
    for m in MODELS:
        sub = sess[sess["model_short"] == m] if not sess.empty else pd.DataFrame()
        vals = pd.to_numeric(sub["tep"], errors="coerce").dropna() if not sub.empty else pd.Series(dtype=float)
        rows.append({
            "model": m,
            "n": int(len(vals)),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "median": float(vals.median()) if len(vals) else float("nan"),
            "tep_score_column_in_injected": (not missing_tep_col),
            "uninjected_join_used": True,
        })
    by_model = pd.DataFrame(rows)
    by_model.to_csv(OUT / "algo_tep_by_model.csv", index=False)
    if not sess.empty:
        sess.to_csv(OUT / "algo_tep_sessions.csv", index=False)
    return by_model, sess


def note_b(by_model: pd.DataFrame, sess: pd.DataFrame) -> None:
    inj_cols = list(_read(RAW / "ALGO_P2_phase2_injected.csv").columns)
    lines = [
        "# B. ALGO TEP per model",
        "",
        "Appendix D: TEP = fraction of **post-injection** steps whose numeric/symbolic content differs from the uninjected run.",
        "",
        f"`ALGO_P2_phase2_injected*.csv` columns: `{', '.join(inj_cols)}`.",
        "",
        "**Is ALGO TEP computable?** Yes, but **not from the injected file alone**.",
        "- Injected files have **no `tep_score` column** (that is the GSM P2 column name).",
        "- `diverged_from_normal` is **not** TEP: it is `step >= critical_step_index` (a Boolean “after injection” flag written at run time, not a content comparison).",
        "- TEP **is** computable by joining `parsed_decision` on `(problem_id, model, step)` against `ALGO_P2_phase2_normal.csv` + `ALGO_P2_phase2_normal_gemini.csv`. That join is what this file reports.",
        "",
        "| model | n sessions | mean TEP | median TEP |",
        "|---|---:|---:|---:|",
    ]
    for _, r in by_model.iterrows():
        lines.append(f"| {r.model} | {int(r.n)} | {_fmt(r['mean'], 3)} | {_fmt(r['median'], 3)} |")
    n_tot = int(by_model["n"].sum())
    lines += [
        "",
        f"Session-level rows: {0 if sess is None else len(sess)} (plausible injection only; implausible file excluded as a different condition). Gemini taken from the dedicated rerun, not the mixed main file.",
        "",
        f"**Flags:** `{n_tot}` sessions with at least one paired post-injection step. Sessions with `critical_step_index < 0` or no overlapping post-injection steps are dropped (TEP undefined), not imputed.",
    ]
    _md(OUT / "B_algo_tep.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# C. GSM Phase-2B compliance
# ---------------------------------------------------------------------------

def run_c() -> pd.DataFrame:
    p2 = _read(RAW / "GSM_P2_cci.csv")
    phase1_files = {
        "Claude": RAW / "GSM_P2_phase1_claude.csv",
        "GPT-4o": RAW / "GSM_P2_phase1_gpt4o.csv",
        "Llama": RAW / "GSM_P2_phase1_llama.csv",
        "Gemini": RAW / "GSM_P2_phase1_gemini.csv",
        "o4-mini": RAW / "GSM_P2_phase1_o1mini.csv",
    }
    p2_fields = list(p2.columns) if not p2.empty else []
    rows = []
    for m, path in phase1_files.items():
        df = _read(path)
        fields = list(df.columns) if not df.empty else []
        n_cci = int((p2["model"].map(_short) == m).sum()) if not p2.empty else 0
        n_phase1 = int(df["problem_id"].nunique()) if not df.empty else 0
        rows.append({
            "model": m,
            "n_gsm_p2_cci": n_cci,
            "n_phase1": n_phase1,
            "compliant": float("nan"),
            "partial": float("nan"),
            "refusal": float("nan"),
            "format_ignored": float("nan"),
            "computable": False,
            "reason": "ALGO four-way taxonomy is Decision:/Reason: parse of injection-step raw_response (response_type). GSM Phase-2B logs have no injection-step raw_response and no response_type.",
            "gsm_p2_cci_fields": "|".join(p2_fields),
            "gsm_phase1_fields": "|".join(fields),
            "phase1_file_present": path.exists(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gsm_p2b_compliance.csv", index=False)
    return out


def note_c(df: pd.DataFrame) -> None:
    p2_fields = df["gsm_p2_cci_fields"].iloc[0] if len(df) else ""
    p1_fields = df["gsm_phase1_fields"].iloc[0] if len(df) else ""
    lines = [
        "# C. GSM Phase-2B compliance",
        "",
        "**Not computable.** The four-way taxonomy (compliant / partial / refusal / format-ignored) is the ALGO Phase-2B `response_type` classifier in `scripts/ALGO_P2_SCR_run_phase2.py:parse_decision_reason` (Decision:/Reason: format). It does not exist for GSM.",
        "",
        "GSM Phase-2B raw injection-step text is not in the released logs, so the ALGO classifier cannot be reapplied.",
        "",
        "**GSM 2B fields that ARE available** (`results/raw/GSM_P2_cci.csv`):",
        f"`{p2_fields.replace('|', ', ')}`",
        "",
        "**Additional Phase-1/session fields** (`GSM_P2_phase1_*.csv`, including o4-mini):",
        f"`{p1_fields.replace('|', ', ')}`",
        "",
        "| model | n in GSM_P2_cci.csv | n in phase1 file | phase1 file |",
        "|---|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.model} | {int(r.n_gsm_p2_cci)} | {int(r.n_phase1)} | "
            f"{'yes' if r.phase1_file_present else 'NO'} |"
        )
    lines += [
        "",
        "o4-mini is in `GSM_P2_phase1_o1mini.csv` (n=44) but **not** in `GSM_P2_cci.csv` (the 4-model table the paper plots).",
        "",
        "**Flags:** no `response_type`, `raw_response` (2B), or `parse_status` on any GSM P2 file. Closest 2B outcomes already stored: `tep_score`, `tep_diverged_steps`, `tep_total_steps`, `session_b_correct`, `inject_at_step`, `injected_value`.",
    ]
    _md(OUT / "C_gsm_p2b_compliance.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# D. Declared vs executed
# ---------------------------------------------------------------------------

def _algo_families(text: str) -> set[str]:
    s = str(text or "")
    hits = {name for name, pat in ALGO_NAME_PATTERNS if pat.search(s)}
    return hits


def run_d(phase1: pd.DataFrame, normal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    detail = []
    for m in MODELS:
        long = LONG[m]
        p1 = phase1[phase1["model"] == long] if not phase1.empty else pd.DataFrame()
        p2 = normal[normal["model"] == long] if not normal.empty else pd.DataFrame()
        if p1.empty:
            rows.append({
                "model": m, "n": 0, "agreement_rate": float("nan"),
                "n_any_invocation": 0, "n_match": 0,
                "agree_correct": 0, "agree_incorrect": 0,
                "disagree_correct": 0, "disagree_incorrect": 0,
                "fisher_p": float("nan"),
                "note": "No ALGO Phase-1 declaration file for this model" if m == "o4-mini"
                else "Phase-1 file empty",
            })
            continue
        if p2.empty:
            rows.append({
                "model": m, "n": 0, "agreement_rate": float("nan"),
                "n_any_invocation": 0, "n_match": 0,
                "agree_correct": 0, "agree_incorrect": 0,
                "disagree_correct": 0, "disagree_incorrect": 0,
                "fisher_p": float("nan"),
                "note": "No Phase-2A steps for this model",
            })
            continue
        last = (
            p2.assign(_step=pd.to_numeric(p2["step_index"], errors="coerce"))
            .sort_values("_step")
            .groupby("problem_id", as_index=False)
            .tail(1)[["problem_id", "final_answer_correct"]]
        )
        last["final_ok"] = _is_true(last["final_answer_correct"])
        inv = p2[p2["reasoning_type"].astype(str).str.strip().str.lower() == "algorithm_invocation"]
        n = 0
        n_inv = 0
        n_match = 0
        ac = ai = dc = di = 0
        for _, r in p1.iterrows():
            pid = str(r["problem_id"])
            p2pid = p2[p2["problem_id"] == pid]
            if p2pid.empty:
                continue
            declared = _algo_families(r.get("stated_algorithm", "")) | _algo_families(r.get("raw_response", ""))
            steps = inv[inv["problem_id"] == pid]
            any_inv = len(steps) > 0
            exec_names = set()
            for _, s in steps.iterrows():
                exec_names |= _algo_families(s.get("reasoning_text", "")) | _algo_families(s.get("raw_response", ""))
            matched = bool(any_inv and declared and (declared & exec_names))
            agree = bool(any_inv and matched)
            fin = last[last["problem_id"] == pid]
            ok = bool(fin["final_ok"].iloc[0]) if len(fin) else False
            n += 1
            n_inv += int(any_inv)
            n_match += int(matched)
            if agree and ok:
                ac += 1
            elif agree and not ok:
                ai += 1
            elif (not agree) and ok:
                dc += 1
            else:
                di += 1
            detail.append({
                "model": m, "problem_id": pid,
                "stated_algorithm": r.get("stated_algorithm", ""),
                "declared_families": "|".join(sorted(declared)),
                "any_algorithm_invocation": any_inv,
                "executed_families": "|".join(sorted(exec_names)),
                "agreement": agree,
                "final_answer_correct": ok,
            })
        table = np.array([[ac, ai], [dc, di]], dtype=int)
        try:
            _, fp = stats.fisher_exact(table, alternative="two-sided")
        except ValueError:
            fp = float("nan")
        rows.append({
            "model": m, "n": n,
            "agreement_rate": (n_match / n) if n else float("nan"),
            "n_any_invocation": n_inv, "n_match": n_match,
            "agree_correct": ac, "agree_incorrect": ai,
            "disagree_correct": dc, "disagree_incorrect": di,
            "fisher_p": float(fp) if fp == fp else float("nan"),
            "note": "agreement = ≥1 Phase-2A algorithm_invocation whose named algorithm family intersects Phase-1 stated_algorithm",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "declared_vs_executed.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "declared_vs_executed_detail.csv", index=False)
    return out


def note_d(df: pd.DataFrame) -> None:
    lines = [
        "# D. Declared-vs-executed algorithm agreement",
        "",
        "Phase 1: `stated_algorithm`. Phase 2A: `reasoning_type == algorithm_invocation` on uninjected steps (`ALGO_P2_phase2_normal*.csv`).",
        "Match = the invocation step names an algorithm in the same family as the Phase-1 declaration (Dijkstra / DP / greedy / …).",
        "2×2: agreement × Phase-2A `final_answer_correct` (last step), Fisher two-sided.",
        "",
        "| model | n | any invocation | match | agreement rate | 2×2 (agree✓, agree✗, dis✓, dis✗) | Fisher p |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.model} | {int(r.n)} | {int(r.n_any_invocation)} | {int(r.n_match)} | "
            f"{_fmt(r.agreement_rate, 3)} | {int(r.agree_correct)}/{int(r.agree_incorrect)}/"
            f"{int(r.disagree_correct)}/{int(r.disagree_incorrect)} | {_fmt_p(r.fisher_p)} |"
        )
    o4 = df[df.model == "o4-mini"].iloc[0]
    lines += [
        "",
        f"**Flags:** o4-mini agreement **not computable** — {o4.note}. "
        "There is no `ALGO_P2_phase1_o1mini.csv`. o4-mini *does* have Phase-2A steps in `ALGO_P2_phase2_normal.csv` (including a few `algorithm_invocation` rows), but nothing to match against.",
        "GPT-4o has Phase-1 declarations but **zero** `algorithm_invocation` steps in Phase 2A, so agreement rate is 0 by construction.",
        "Invocation is rare overall (paper Table 8 / appendix cases); most sessions never name an algorithm at execution time.",
    ]
    _md(OUT / "D_declared_vs_executed.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# E. Instance-level proximity vs VRI
# ---------------------------------------------------------------------------

def _per_problem_vri(p1: pd.DataFrame) -> pd.DataFrame:
    df = p1[p1["problem_id"].isin(PAPER_ADV_ALL)].copy()
    df["_ok"] = _is_true(df["verified"]).astype(int)
    pivot = df.pivot_table(index="problem_id", columns="variant_type", values="_ok", aggfunc="mean")
    for c in ("canonical", "W1", "W2", "W3", "W4"):
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot["VRI"] = pivot[["W1", "W2", "W4"]].mean(axis=1) - pivot["W3"]
    return pivot.reset_index()


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


def _boot_corr(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    rs = np.empty(N_BOOT, dtype=float)
    idx = np.arange(n)
    for i in range(N_BOOT):
        d = RNG.choice(idx, size=n, replace=True)
        xs, ys = x[d], y[d]
        if np.nanstd(xs) == 0 or np.nanstd(ys) == 0:
            rs[i] = np.nan
        else:
            rs[i] = stats.pearsonr(xs, ys)[0]
    rs = rs[np.isfinite(rs)]
    if len(rs) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(rs, [0.025, 0.975])
    return float(lo), float(hi)


def run_e(p1_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cont = _read(RAW / "ALGO_P3_contamination.csv")
    cont = cont[["problem_id", "instance_contamination_score", "template_contamination_score"]].drop_duplicates("problem_id")
    cont["instance_contamination_score"] = pd.to_numeric(cont["instance_contamination_score"], errors="coerce")
    cont["template_contamination_score"] = pd.to_numeric(cont["template_contamination_score"], errors="coerce")
    rows = []
    for m in MODELS:
        vri = _per_problem_vri(p1_by_model[m])
        merged = vri.merge(cont, on="problem_id", how="left")
        s = merged.dropna(subset=["instance_contamination_score", "VRI"])
        n = len(s)
        if n < 4:
            rows.append({
                "model": m, "n": n, "n_paper_claimed": 64, "pool": "frozen_adversarial_61",
                "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_rho": float("nan"), "spearman_p": float("nan"),
                "partial_r_residualized_on_canonical": float("nan"),
                "partial_p": float("nan"),
                "pearson_boot_ci95_lo": float("nan"), "pearson_boot_ci95_hi": float("nan"),
            })
            continue
        r, p = stats.pearsonr(s["instance_contamination_score"], s["VRI"])
        rho, sp = stats.spearmanr(s["instance_contamination_score"], s["VRI"])
        pr, pp, _ = _partial_pearson(s["instance_contamination_score"], s["VRI"], s["canonical"])
        lo, hi = _boot_corr(s["instance_contamination_score"], s["VRI"])
        st = s.dropna(subset=["template_contamination_score", "VRI"])
        if len(st) >= 4 and st["template_contamination_score"].nunique() > 1:
            tr, tp = stats.pearsonr(st["template_contamination_score"], st["VRI"])
            tpr, tpp, _ = _partial_pearson(st["template_contamination_score"], st["VRI"], st["canonical"])
        else:
            tr = tp = tpr = tpp = float("nan")
        rows.append({
            "model": m, "n": n, "n_paper_claimed": 64, "pool": "frozen_adversarial_61",
            "pearson_r": float(r), "pearson_p": float(p),
            "spearman_rho": float(rho), "spearman_p": float(sp),
            "partial_r_residualized_on_canonical": float(pr),
            "partial_p": float(pp),
            "pearson_boot_ci95_lo": lo, "pearson_boot_ci95_hi": hi,
            "template_pearson_r": float(tr), "template_pearson_p": float(tp),
            "template_partial_r": float(tpr), "template_partial_p": float(tpp),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "proximity_instance_level.csv", index=False)
    return out


def note_e(df: pd.DataFrame) -> None:
    lines = [
        "# E. Instance-level proximity vs VRI",
        "",
        "Repeats §4.3 (proximity vs VRI, raw Pearson and residualized on per-problem canonical accuracy) with **`instance_contamination_score`** instead of template proximity.",
        "VRI = mean(W1, W2, W4) − W3, per problem, 0/1 correctness.",
        "",
        f"Pool: frozen adversarial **n={len(PAPER_ADV_ALL)}** (34 SP + 10 CC + 17 WIS). Paper text says n=64; that count is not recoverable as a unique-ID list from released files.",
        "",
        "| model | n | instance r (p) | instance partial r (p) | template r (p) | instance bootstrap 95% CI |",
        "|---|---:|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.model} | {int(r.n)} | {_fmt(r.pearson_r, 2)} ({_fmt_p(r.pearson_p)}) | "
            f"{_fmt(r.partial_r_residualized_on_canonical, 2)} ({_fmt_p(r.partial_p)}) | "
            f"{_fmt(r.template_pearson_r, 2)} ({_fmt_p(r.template_pearson_p)}) | "
            f"[{_fmt(r.pearson_boot_ci95_lo, 2)}, {_fmt(r.pearson_boot_ci95_hi, 2)}] |"
        )
    lines += [
        "",
        "Paper §4.3 headlines (claimed template, n=64): Claude r=+0.44, GPT-4o r=+0.37, Llama/Gemini ~0.12, o4-mini r=−0.094; partial Claude +0.41 / GPT-4o +0.39.",
        "Those published r values match **instance** scores on the frozen 61, not template scores. Template r on the same 61 is much weaker (see `template_pearson_r` in the CSV). The figure script `contam_vri_pearson()` already correlated `instance_contamination_score` vs VRI.",
        "",
        "**Flags:** n is 61 not 64. All five models computed. Instance scores are 0 for many problems (floor mass).",
    ]
    _md(OUT / "E_proximity_instance_level.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# F. 5-model triangulation with raw W3
# ---------------------------------------------------------------------------

def run_f(p1_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cont = _read(RAW / "ALGO_P3_contamination.csv")
    cont = cont[["problem_id", "instance_contamination_score"]].drop_duplicates("problem_id")
    cont["instance_contamination_score"] = pd.to_numeric(cont["instance_contamination_score"], errors="coerce")
    p75 = float(cont["instance_contamination_score"].quantile(0.75))
    floor = float(cont["instance_contamination_score"].min())

    cci = _read(DER / "ALGO_P2_per_instance_cci.csv")
    if not cci.empty:
        cci["cci_composite"] = pd.to_numeric(cci["cci_composite"], errors="coerce")
        cci["model_short"] = cci["model"].map(_short)

    # 110 ALGO problems = union of P1 canonical IDs
    all_ids = sorted({pid for df in p1_by_model.values() for pid in df["problem_id"].unique()})
    detail = []
    for m in MODELS:
        w3, can = _w3_canonical_maps(p1_by_model[m])
        cci_m = cci[cci["model_short"] == m] if not cci.empty else pd.DataFrame()
        cci_map = dict(zip(cci_m["problem_id"], cci_m["cci_composite"])) if len(cci_m) else {}
        for pid in all_ids:
            if pid not in w3:
                continue
            w3v = int(w3[pid])
            cs = cont.loc[cont["problem_id"] == pid, "instance_contamination_score"]
            contam = float(cs.iloc[0]) if len(cs) else float("nan")
            cci_v = cci_map.get(pid, float("nan"))
            # signal directions: +1 computation, -1 retrieval, 0 neither/missing
            sig_w3 = 1 if w3v == 1 else -1
            if cci_v == cci_v:
                if cci_v <= 0.10:
                    sig_cci = -1
                elif cci_v >= 0.67:
                    sig_cci = 1
                else:
                    sig_cci = 0
            else:
                sig_cci = 0
            if contam == contam:
                if contam >= p75:
                    sig_c = -1
                elif abs(contam - floor) <= 1e-12:
                    sig_c = 1
                else:
                    sig_c = 0
            else:
                sig_c = 0
            sigs = [sig_w3, sig_cci, sig_c]
            has_ret = any(s == -1 for s in sigs)
            has_comp = any(s == 1 for s in sigs)
            cci_ok = cci_v == cci_v
            if cci_ok and sig_w3 == -1 and sig_cci == -1 and sig_c == -1:
                label = "retrieval_consistent"
            elif cci_ok and sig_w3 == 1 and sig_cci == 1 and sig_c == 1:
                label = "computation_consistent"
            elif has_ret and has_comp:
                label = "mixed"
            else:
                label = "ambiguous"
            detail.append({
                "model": m, "problem_id": pid, "W3": w3v, "CCI": cci_v,
                "instance_contamination_score": contam,
                "contam_p75": p75, "contam_floor": floor,
                "cci_available": cci_ok, "label": label,
            })
    det = pd.DataFrame(detail)
    counts = []
    for m in MODELS:
        sub = det[det["model"] == m]
        vc = sub["label"].value_counts()
        counts.append({
            "model": m,
            "n": int(len(sub)),
            "retrieval_consistent": int(vc.get("retrieval_consistent", 0)),
            "computation_consistent": int(vc.get("computation_consistent", 0)),
            "mixed": int(vc.get("mixed", 0)),
            "ambiguous": int(vc.get("ambiguous", 0)),
            "n_cci_available": int(sub["cci_available"].sum()),
            "contam_p75": p75,
            "contam_floor": floor,
            "rule": "appendix three-signal: W3 in {0,1}; CCI<=0.10 or >=0.67; contamination at floor or >=p75. Conjunction for strong labels. Mixed = conflicting directions. Ambiguous = else (incl. missing CCI).",
        })
    out = pd.DataFrame(counts)
    out.to_csv(OUT / "triangulation_5model.csv", index=False)
    det.to_csv(OUT / "triangulation_5model_detail.csv", index=False)
    return out


def note_f(df: pd.DataFrame) -> None:
    p75 = df["contam_p75"].iloc[0] if len(df) else float("nan")
    floor = df["contam_floor"].iloc[0] if len(df) else float("nan")
    tot = df[["retrieval_consistent", "computation_consistent", "mixed", "ambiguous"]].sum()
    lines = [
        "# F. Five-model triangulation under a non-retention rule",
        "",
        "Paper default used **retention** (W3/canonical, undefined if canonical=0) and dropped o4-mini as “degenerate at ceiling canonical accuracy”.",
        "This rerun uses **raw W3 correctness (0/1)** on all five models × 110 ALGO problems.",
        "",
        "Appendix label rules (three-signal conjunction):",
        "- W3 at 0 (retrieval) or 1 (computation)",
        "- CCI ≤ 0.10 (retrieval) or ≥ 0.67 (computation), per-instance `cci_composite`",
        f"- instance contamination at floor ({_fmt(floor, 3)}) or ≥ 75th percentile ({_fmt(p75, 3)}) of the 110-problem scores",
        "",
        "Strong retrieval / computation require **all three** signals aligned. Mixed = at least one retrieval-direction and one computation-direction signal. Ambiguous = remainder (including missing CCI).",
        "",
        "| model | n | retrieval | computation | mixed | ambiguous | CCI available |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.model} | {int(r.n)} | {int(r.retrieval_consistent)} | {int(r.computation_consistent)} | "
            f"{int(r.mixed)} | {int(r.ambiguous)} | {int(r.n_cci_available)} |"
        )
    lines += [
        "",
        f"**Pooled (5 models):** retrieval={int(tot.retrieval_consistent)}, computation={int(tot.computation_consistent)}, "
        f"mixed={int(tot.mixed)}, ambiguous={int(tot.ambiguous)}.",
        "",
        "Paper 4-model legacy counts for comparison: retrieval=8, computation=4, mixed=157, ambiguous=271 (440 instances). Those used the 5-field AND in `ALGO_P3_SCR_triangulation.py` (canonical>0.5, W3<0.2, contamination top-half, greedy_succeeds, plus missing-data → ambiguous), **not** the three thresholds printed in the appendix. Replacing retention with raw W3 on the *printed* three-signal rule does not reproduce 8/4.",
        "",
        "**Flags:** o4-mini has **no per-instance CCI** (`ALGO_P2_per_instance_cci.csv` is 4 models; no Phase-1 file to build `cci_composite`). o4-mini rows therefore cannot form a strong conjunction and land in mixed/ambiguous from W3×contamination only. CCI itself exists for only 61 adversarial problems × 4 models, so most of the 110×5 grid is CCI-missing → ambiguous.",
    ]
    _md(OUT / "F_triangulation_5model.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# G. Intrusion errors
# ---------------------------------------------------------------------------

_PATH_SPLIT = re.compile(r"\s*(?:→|->|,|/)\s*")


def _extract_sp_path(text: str) -> tuple[str, ...]:
    """Only from an explicit Path: ... line (last one), never from edge listings."""
    s = str(text or "")
    matches = list(re.finditer(r"Path\s*:\s*(.+?)(?:,\s*Cost\s*:|$)", s, flags=re.I))
    if not matches:
        return tuple()
    blob = matches[-1].group(1)
    blob = blob.split("\n")[0]
    if re.search(r"→|->", blob):
        parts = re.split(r"\s*(?:→|->)\s*", blob)
        toks = [re.sub(r"[^A-Za-z0-9]", "", p.strip()) for p in parts]
        toks = [t for t in toks if t and t.lower() not in {"path", "cost"}]
        if len(toks) >= 2:
            return tuple(t.upper() for t in toks)
    nums = re.findall(r"\b\d+\b", blob)
    if len(nums) >= 2:
        return tuple(nums)
    return tuple()


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


def _match_span(family: str, pid: str, model_ans: str) -> str:
    """Short span that triggered the canonical-answer match (for examples)."""
    if family == "ALGO" and pid.startswith("SP"):
        ms = list(re.finditer(r"Path\s*:\s*(.+?)(?:,\s*Cost\s*:|$)", str(model_ans), flags=re.I))
        return ms[-1].group(0)[:240] if ms else ""
    if family == "ALGO" and pid.startswith("WIS"):
        m = re.search(r"Selected\s*:\s*\{[^}]*\}(?:\s*,\s*Total\s*:\s*-?\d+)?", str(model_ans), flags=re.I)
        return m.group(0)[:240] if m else ""
    if family == "ALGO" and pid.startswith("CC"):
        m = re.search(r"(?:Count|Total)\s*:\s*-?\d+.{0,80}(?:Coins|Scoops)\s*:\s*\[[^\]]*\]", str(model_ans), flags=re.I | re.S)
        return (m.group(0)[:240] if m else "")[:240]
    return str(model_ans)[:240]


def _equals_canonical(family: str, pid: str, model_ans: str, can_gt: str, w3_gt: str) -> bool:
    """True if the W3 response encodes the canonical gold answer (pre-rename)."""
    if not str(model_ans).strip() or not str(can_gt).strip():
        return False
    # refuse if it actually matches W3 gold (then it's just correct-format W3, not intrusion)
    if family == "GSM":
        pred = _extract_gsm_number(model_ans)
        gold = _extract_gsm_number(can_gt)
        w3n = _extract_gsm_number(w3_gt)
        if pred is None or gold is None:
            return False
        if w3n is not None and abs(pred - w3n) < 0.01:
            return False  # equals W3 gold; not an intrusion among errors unless W3 gold ≠ canonical
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
            # numeric BW answers
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


def _intrusion_rows_algo(p1_by_model: dict[str, pd.DataFrame]) -> list[dict]:
    out = []
    for m, df in p1_by_model.items():
        can = df[df["variant_type"] == "canonical"][["problem_id", "ground_truth"]].drop_duplicates("problem_id")
        w3 = df[df["variant_type"] == "W3"][["problem_id", "ground_truth", "model_answer", "verified"]]
        merged = w3.merge(can, on="problem_id", suffixes=("_w3", "_can"))
        merged = merged[~_is_true(merged["verified"])]
        for _, r in merged.iterrows():
            pid = str(r["problem_id"])
            flag = _equals_canonical("ALGO", pid, r["model_answer"], r["ground_truth_can"], r["ground_truth_w3"])
            out.append({
                "family": "ALGO", "model": m, "problem_id": pid,
                "canonical_answer": r["ground_truth_can"],
                "W3_model_answer": r["model_answer"],
                "W3_ground_truth": r["ground_truth_w3"],
                "intrusion": flag,
            })
    return out


def _intrusion_rows_gsm() -> list[dict]:
    out = []
    for m in MODELS:
        df = load_gsm_p1(TAG[m])
        if df.empty:
            continue
        can = df[df["variant_type"] == "canonical"][["problem_id", "correct_answer"]].drop_duplicates("problem_id")
        w3 = df[df["variant_type"] == "W3"][["problem_id", "correct_answer", "raw_response", "behavioral_correct"]]
        merged = w3.merge(can, on="problem_id", suffixes=("_w3", "_can"))
        merged = merged[~_is_true(merged["behavioral_correct"])]
        for _, r in merged.iterrows():
            flag = _equals_canonical(
                "GSM", str(r["problem_id"]), r["raw_response"], r["correct_answer_can"], r["correct_answer_w3"]
            )
            out.append({
                "family": "GSM", "model": m, "problem_id": str(r["problem_id"]),
                "canonical_answer": r["correct_answer_can"],
                "W3_model_answer": r["raw_response"],
                "W3_ground_truth": r["correct_answer_w3"],
                "intrusion": flag,
            })
    return out


def _intrusion_rows_bw(bw: pd.DataFrame) -> list[dict]:
    out = []
    if bw.empty:
        return out
    for m, df in bw.groupby("model_short"):
        can = df[df["variant_type"] == "canonical"][["problem_id", "correct_answer"]].drop_duplicates("problem_id")
        w3 = df[df["variant_type"] == "W3"][["problem_id", "correct_answer", "raw_response", "behavioral_correct"]]
        merged = w3.merge(can, on="problem_id", suffixes=("_w3", "_can"))
        merged = merged[~_is_true(merged["behavioral_correct"])]
        for _, r in merged.iterrows():
            flag = _equals_canonical(
                "BW", str(r["problem_id"]), r["raw_response"],
                r["correct_answer_can"], r["correct_answer_w3"],
            )
            out.append({
                "family": "BW", "model": m, "problem_id": str(r["problem_id"]),
                "canonical_answer": r["correct_answer_can"],
                "W3_model_answer": r["raw_response"],
                "W3_ground_truth": r["correct_answer_w3"],
                "intrusion": flag,
            })
    return out


def run_g(p1_by_model: dict[str, pd.DataFrame], bw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = _intrusion_rows_algo(p1_by_model) + _intrusion_rows_gsm() + _intrusion_rows_bw(bw)
    det = pd.DataFrame(detail)
    rows = []
    if det.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "intrusion_rates.csv", index=False)
        return out, det
    for (m, fam), g in det.groupby(["model", "family"]):
        n_err = len(g)
        k = int(g["intrusion"].sum())
        lo, hi = wilson(k, n_err)
        rows.append({
            "model": m, "family": fam,
            "n_W3_errors": n_err,
            "intrusion_count": k,
            "intrusion_rate": (k / n_err) if n_err else float("nan"),
            "wilson_ci95_lo": lo, "wilson_ci95_hi": hi,
        })
    rates = pd.DataFrame(rows).sort_values(["family", "model"])
    rates.to_csv(OUT / "intrusion_rates.csv", index=False)

    examples = []
    for m in MODELS:
        hits = det[(det["model"] == m) & (det["intrusion"])].drop_duplicates(["family", "problem_id"])
        pad = det[(det["model"] == m) & (~det["intrusion"])].drop_duplicates(["family", "problem_id"])
        take_parts = []
        remaining = 5
        for fam in ["ALGO", "GSM", "BW"]:
            sub = hits[hits["family"] == fam]
            n_take = min(2, remaining, len(sub))
            if n_take:
                take_parts.append(sub.head(n_take))
                remaining -= n_take
        take = pd.concat(take_parts, ignore_index=True) if take_parts else hits.head(0)
        if len(take) < 5:
            extra_hits = hits[~hits["problem_id"].isin(set(take["problem_id"]))] if len(take) else hits
            need = 5 - len(take)
            take = pd.concat([take, extra_hits.head(need)], ignore_index=True)
        if len(take) < 5:
            take = pd.concat([take, pad.head(5 - len(take))], ignore_index=True)
        take = take.head(5)
        for _, r in take.iterrows():
            examples.append({
                "model": m,
                "family": r["family"],
                "problem_id": r["problem_id"],
                "canonical_answer": str(r["canonical_answer"])[:500],
                "W3_model_answer": str(r["W3_model_answer"])[:500],
                "intrusion": bool(r["intrusion"]),
                "match_span": _match_span(str(r["family"]), str(r["problem_id"]), str(r["W3_model_answer"])),
            })
    ex = pd.DataFrame(examples)
    ex.to_csv(OUT / "intrusion_examples.csv", index=False)
    return rates, ex


def note_g(rates: pd.DataFrame, examples: pd.DataFrame) -> None:
    lines = [
        "# G. Intrusion errors",
        "",
        "Among W3 **errors**, count cases where the model’s answer equals the **canonical** gold (the pre-rename answer), and does **not** equal the W3 gold.",
        "",
        "GSM W3 preserves the numeric gold (name substitution only), so matching the canonical number while being a W3 error is almost impossible unless the verifier and the extractor disagree. ALGO W3 often relabels nodes/items (0,1,2 → Hub A,B,C / Item A,B,C); producing the numeric/canonical identifiers on a renamed instance is the intrusion.",
        "",
        "| family | model | W3 errors | intrusions | rate | Wilson 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    if rates is None or rates.empty:
        lines.append("")
        lines.append("**Flags:** no W3 error rows found.")
        _md(OUT / "G_intrusion_errors.md", "\n".join(lines))
        return
    for _, r in rates.iterrows():
        lines.append(
            f"| {r.family} | {r.model} | {int(r.n_W3_errors)} | {int(r.intrusion_count)} | "
            f"{_fmt(r.intrusion_rate, 3)} | [{_fmt(r.wilson_ci95_lo, 3)}, {_fmt(r.wilson_ci95_hi, 3)}] |"
        )
    algo = rates[rates.family == "ALGO"]
    k_algo = int(algo.intrusion_count.sum()) if len(algo) else 0
    n_algo = int(algo.n_W3_errors.sum()) if len(algo) else 0
    gsm = rates[rates.family == "GSM"]
    k_gsm = int(gsm.intrusion_count.sum()) if len(gsm) else 0
    lines += [
        "",
        f"**ALGO pooled:** {k_algo}/{n_algo} W3 errors are canonical-answer intrusions.",
        f"**GSM pooled:** {k_gsm} intrusions (expected near zero because W3 gold = canonical gold).",
        "",
        "Five example rows per model are in `intrusion_examples.csv` (`W3_model_answer` truncated to 500 chars; `match_span` is the Path:/Selected: line that actually matched). Rows with `intrusion=False` are fillers when a model had fewer than 5 true hits. The matching line is often at the **end** of a long CoT, so the truncated `W3_model_answer` alone can look like a false positive.",
        "",
        "**Flags:** GSM bank filter applied (GPT-4o/Llama n_valid=20). BW uses `BW_`/`MBW_` IDs from the three P1 files; GSM rows that leaked into `BW_P1_behavioral.csv` are dropped. Answer equality is structured (path / coin multiset / selected set / last number / action list), not raw-string equality — Claude W3 errors never matched canonical GT as a raw string.",
    ]
    _md(OUT / "G_intrusion_errors.md", "\n".join(lines))


def main() -> None:
    print("Loading P1…")
    p1 = algo_p1_all()
    bw = load_bw_p1()
    print("Loading P2…")
    normal_tep = load_phase2_normal(overlay_gemini_dedicated=True, keep_main_gemini_rest=False)
    normal_p2a = load_phase2_normal(overlay_gemini_dedicated=True, keep_main_gemini_rest=True)
    injected = load_phase2_injected()
    phase1 = load_phase1_declarations()

    print("A pairwise inversion")
    a = run_a(p1)
    note_a(a)

    print("B ALGO TEP")
    b, sess = run_b(normal_tep, injected)
    note_b(b, sess)

    print("C GSM 2B compliance")
    c = run_c()
    note_c(c)

    print("D declared vs executed")
    d = run_d(phase1, normal_p2a)
    note_d(d)

    print("E instance proximity")
    e = run_e(p1)
    note_e(e)

    print("F triangulation")
    f = run_f(p1)
    note_f(f)

    print("G intrusion")
    g, ex = run_g(p1, bw)
    note_g(g, ex)

    print("Wrote", OUT)


if __name__ == "__main__":
    main()
