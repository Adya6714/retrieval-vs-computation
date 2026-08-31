#!/usr/bin/env python3
"""Solidify analyses T1–T3 from existing rebuild / raw data.

Writes only under rebuild/solidify/. Does not touch results/ or paper/.
"""
from __future__ import annotations

import math
import re
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
REBUILD = ROOT / "rebuild"
OUT = REBUILD / "solidify"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REBUILD))
sys.path.insert(0, str(ROOT))

import compute_rebuild as cr  # noqa: E402
from triangulation_rule import (  # noqa: E402
    CCI_THRESHOLDS,
    CONTAM_PERCENTILES,
    W3_CUTOFFS,
    count_labels,
    label_default,
    label_sweep_cell,
)

RNG = np.random.default_rng(42)
N_BOOT = 10_000
ALPHA = 0.05
MODELS = cr.MODELS
FAMILIES = ("ALGO", "GSM", "BW")


def _fmt(x, nd=3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(xf):
        return "NA"
    if math.isinf(xf):
        return "∞" if xf > 0 else "-∞"
    return f"{xf:.{nd}f}"


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return "NA"
    p = float(p)
    if p < 1e-300:
        return "<1e-300"
    if p < 0.0001:
        return f"{p:.2e}"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def _fmt_ci(lo, hi, nd=3) -> str:
    if lo is None or hi is None:
        return "NA"
    if isinstance(lo, float) and (math.isnan(lo) or math.isnan(hi)):
        return "NA"
    return f"[{_fmt(lo, nd)}, {_fmt(hi, nd)}]"


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _confident_rate(counts: dict) -> float:
    n = counts["n"]
    if n == 0:
        return float("nan")
    return (counts["retrieval"] + counts["computation"]) / n


# ---------------------------------------------------------------------------
# T1 — complete-case triangulation
# ---------------------------------------------------------------------------

def _load_four_panel() -> pd.DataFrame:
    df = pd.read_csv(REBUILD / "triangulation_4model_labels.csv")
    for c in ("VAR_canonical", "VAR_W3", "instance_contamination_score", "ACI", "instance_rank_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("greedy_succeeds", "missing_core", "missing_phase2", "parse_failure_or_missing", "in_paper_4model"):
        if c in df.columns:
            df[c] = _as_bool(df[c])
    return df


def run_t1() -> dict:
    four = _load_four_panel()
    assert len(four) == 440, f"expected 440-row 4-model panel, got {len(four)}"

    n_parse = int(four["parse_failure_or_missing"].sum())
    n_p2 = int(four["missing_phase2"].sum())
    n_core = int(four["missing_core"].sum())

    w3_ok = four["VAR_W3"].notna()
    cci_ok = four["ACI"].notna()
    prox_ok = four["instance_contamination_score"].notna()
    three_signals = w3_ok & cci_ok & prox_ok
    parse_ok = ~four["parse_failure_or_missing"]

    # Primary complete-case: all three signals present AND W3 parse succeeded.
    # Equivalent to ~ambiguous under the executed rule given missing_core = 0.
    complete = three_signals & parse_ok
    n_three = int(three_signals.sum())
    n_complete = int(complete.sum())
    n_three_but_parse = int((three_signals & ~parse_ok).sum())

    lab_full = label_default(four)
    c_full = count_labels(lab_full)
    rate_full = _confident_rate(c_full)

    three = four.loc[three_signals].copy()
    lab_3 = label_default(three)
    c_3 = count_labels(lab_3)
    rate_3 = _confident_rate(c_3)

    cc = four.loc[complete].copy()
    # Inherited ranks from the 440 panel (contamination rank is a problem property).
    lab_cc = label_default(cc)
    c_cc = count_labels(lab_cc)
    rate_cc = _confident_rate(c_cc)

    # Sensitivity: re-rank contamination within subtype on the complete-case subset.
    cc_rerank = cc.copy()
    cc_rerank["instance_rank_pct"] = cc_rerank.groupby("problem_subtype")[
        "instance_contamination_score"
    ].rank(method="average", pct=True)
    lab_cc_rr = label_default(cc_rerank)
    c_cc_rr = count_labels(lab_cc_rr)
    rate_cc_rr = _confident_rate(c_cc_rr)

    def _sweep(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for cci_thr, w3_cut, pct in product(CCI_THRESHOLDS, W3_CUTOFFS, CONTAM_PERCENTILES):
            lab = label_sweep_cell(df, cci_thr=cci_thr, w3_cutoff=w3_cut, contam_pct=pct)
            c = count_labels(lab)
            rows.append({
                "cci_threshold": cci_thr,
                "w3_cutoff": w3_cut,
                "contam_percentile": pct,
                "n_retrieval": c["retrieval"],
                "n_computation": c["computation"],
                "n_mixed": c["mixed"],
                "n_ambiguous": c["ambiguous"],
                "n": c["n"],
                "n_confident": c["retrieval"] + c["computation"],
                "confident_label_rate": _confident_rate(c),
            })
        return pd.DataFrame(rows)

    sw_full = _sweep(four)
    sw_cc = _sweep(cc)
    sw_cc_rr = _sweep(cc_rerank)

    def _max_row(sw: pd.DataFrame) -> pd.Series:
        return sw.loc[sw["confident_label_rate"].idxmax()]

    mx_full = _max_row(sw_full)
    mx_cc = _max_row(sw_cc)
    mx_cc_rr = _max_row(sw_cc_rr)

    counts_rows = [
        {
            "panel": "full_440",
            "n": c_full["n"],
            "n_retrieval": c_full["retrieval"],
            "n_computation": c_full["computation"],
            "n_mixed": c_full["mixed"],
            "n_ambiguous": c_full["ambiguous"],
            "confident_label_rate": rate_full,
            "n_parse_failure": n_parse,
            "n_missing_phase2": n_p2,
            "n_missing_core": n_core,
            "note": "executed 5-field AND; missing-data flags force ambiguous",
        },
        {
            "panel": "three_signals",
            "n": c_3["n"],
            "n_retrieval": c_3["retrieval"],
            "n_computation": c_3["computation"],
            "n_mixed": c_3["mixed"],
            "n_ambiguous": c_3["ambiguous"],
            "confident_label_rate": rate_3,
            "n_parse_failure": int(three["parse_failure_or_missing"].sum()),
            "n_missing_phase2": int(three["missing_phase2"].sum()),
            "n_missing_core": int(three["missing_core"].sum()),
            "note": "W3 + CCI + proximity present; parse_failure still forces ambiguous",
        },
        {
            "panel": "complete_case",
            "n": c_cc["n"],
            "n_retrieval": c_cc["retrieval"],
            "n_computation": c_cc["computation"],
            "n_mixed": c_cc["mixed"],
            "n_ambiguous": c_cc["ambiguous"],
            "confident_label_rate": rate_cc,
            "n_parse_failure": 0,
            "n_missing_phase2": 0,
            "n_missing_core": int(cc["missing_core"].sum()),
            "note": "W3 + CCI + proximity present, parse succeeded; ranks inherited from 440",
        },
        {
            "panel": "complete_case_reranked",
            "n": c_cc_rr["n"],
            "n_retrieval": c_cc_rr["retrieval"],
            "n_computation": c_cc_rr["computation"],
            "n_mixed": c_cc_rr["mixed"],
            "n_ambiguous": c_cc_rr["ambiguous"],
            "confident_label_rate": rate_cc_rr,
            "n_parse_failure": 0,
            "n_missing_phase2": 0,
            "n_missing_core": int(cc_rerank["missing_core"].sum()),
            "note": "sensitivity: contamination rank recomputed on complete-case subset",
        },
    ]
    counts_df = pd.DataFrame(counts_rows)
    counts_df.to_csv(OUT / "T1_complete_case_counts.csv", index=False)
    sw_cc.to_csv(OUT / "T1_270_sweep_complete_case.csv", index=False)
    sw_cc_rr.to_csv(OUT / "T1_270_sweep_complete_case_reranked.csv", index=False)

    four.assign(
        three_signals=three_signals,
        complete_case=complete,
        label_default=lab_full,
    ).to_csv(OUT / "T1_panel_flags.csv", index=False)

    md = [
        "# T1 — Triangulation complete-case",
        "",
        "The executed rule (`rebuild/triangulation_rule.py`) marks an instance **ambiguous** when any of `missing_core`, `parse_failure_or_missing`, or `missing_phase2` is true. Retrieval and computation are assigned only on the complement. Mixed is everything else. That means the published 8 / 4 / 157 / 271 split **confounds signal disagreement with signal absence**.",
        "",
        "## Complete-case definition",
        "",
        "An instance is complete-case when all three named signals are present **and** W3 actually parsed:",
        "",
        "- W3 correctness: `VAR_W3` not NA",
        "- CCI: `ACI` not NA (`missing_phase2` is false)",
        "- proximity: `instance_contamination_score` not NA",
        "- parse succeeded: `parse_failure_or_missing` is false",
        "",
        f"On the 440-row 4-model panel: parse_failure = {n_parse}, missing_phase2 = {n_p2}, missing_core = {n_core}.",
        f"Three-signal intersection (W3 + CCI + proximity): n = {n_three}.",
        f"Of those, {n_three_but_parse} still have a parse failure.",
        f"**n_complete = {n_complete}** after requiring a successful parse — the subset on which the executed rule can assign mixed / retrieval / computation.",
        "",
        "Contamination ranks are inherited from the 440-panel (a problem-level property). A reranked sensitivity is reported below.",
        "",
        "## Default rule — full panel vs complete-case",
        "",
        "Confident-label rate = (n_retrieval + n_computation) / n.",
        "",
        "| panel | n | retrieval | computation | mixed | ambiguous | confident-label rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| full 440 (signals never collected **or** disagree) | {c_full['n']} | {c_full['retrieval']} | {c_full['computation']} | {c_full['mixed']} | {c_full['ambiguous']} | {_fmt(rate_full, 4)} ({c_full['retrieval'] + c_full['computation']}/{c_full['n']}) |",
        f"| three signals present (parse flag still applied) | {c_3['n']} | {c_3['retrieval']} | {c_3['computation']} | {c_3['mixed']} | {c_3['ambiguous']} | {_fmt(rate_3, 4)} ({c_3['retrieval'] + c_3['computation']}/{c_3['n']}) |",
        f"| complete-case (parse succeeded; remaining mixed = disagreement) | {c_cc['n']} | {c_cc['retrieval']} | {c_cc['computation']} | {c_cc['mixed']} | {c_cc['ambiguous']} | {_fmt(rate_cc, 4)} ({c_cc['retrieval'] + c_cc['computation']}/{c_cc['n']}) |",
        "",
        f"Side by side (headline): **{_fmt(rate_full, 4)}** on the full panel vs **{_fmt(rate_cc, 4)}** on complete-case.",
        "",
        "Retrieval and computation counts do not move (8 and 4). Those labels already required all flags clear. What moves is the denominator: 271 of 440 were never eligible because a signal was missing. On complete-case, ambiguous drops to 0 and the 157 mixed labels are genuine three-signal disagreement.",
        "",
        "## 270-config sweep — maximum confident-label rate",
        "",
        "Same grid as the rebuild: CCI ∈ {0.05, 0.10, …, 0.90} × W3 cutoff ∈ {0.0, 0.25, 0.5, 0.75, 1.0} × contamination percentile ∈ {50, 75, 90}. Missing-data flags still force ambiguous, so on the full panel they pin n_ambiguous = 271 in every cell. On complete-case that pin is gone; the sweep only reallocates the 169 collected instances among retrieval / computation / mixed.",
        "",
        "| panel | n | default confident rate | **max** confident rate over 270 | config at max (CCI, W3, contam pct) | retrieval / computation / mixed / ambiguous at max |",
        "|---|---:|---:|---:|---|---|",
        (
            f"| full 440 | {c_full['n']} | {_fmt(rate_full, 4)} | **{_fmt(float(mx_full['confident_label_rate']), 4)}** "
            f"({int(mx_full['n_confident'])}/{int(mx_full['n'])}) | "
            f"{mx_full['cci_threshold']}, {mx_full['w3_cutoff']}, {int(mx_full['contam_percentile'])} | "
            f"{int(mx_full['n_retrieval'])} / {int(mx_full['n_computation'])} / {int(mx_full['n_mixed'])} / {int(mx_full['n_ambiguous'])} |"
        ),
        (
            f"| complete-case | {c_cc['n']} | {_fmt(rate_cc, 4)} | **{_fmt(float(mx_cc['confident_label_rate']), 4)}** "
            f"({int(mx_cc['n_confident'])}/{int(mx_cc['n'])}) | "
            f"{mx_cc['cci_threshold']}, {mx_cc['w3_cutoff']}, {int(mx_cc['contam_percentile'])} | "
            f"{int(mx_cc['n_retrieval'])} / {int(mx_cc['n_computation'])} / {int(mx_cc['n_mixed'])} / {int(mx_cc['n_ambiguous'])} |"
        ),
        "",
        f"Side by side: **max confident-label rate { _fmt(float(mx_full['confident_label_rate']), 4)}** (full panel) vs **{_fmt(float(mx_cc['confident_label_rate']), 4)}** (complete-case).",
        "",
        "Even at the most generous cell of the 270-grid, complete-case confident labels stay a minority. The dominant complete-case label is mixed: the three signals were collected and they disagree.",
        "",
        "## Sensitivity: re-rank contamination on the subset",
        "",
        f"Default rule after re-ranking: retrieval={c_cc_rr['retrieval']}, computation={c_cc_rr['computation']}, mixed={c_cc_rr['mixed']}, ambiguous={c_cc_rr['ambiguous']}, confident rate={_fmt(rate_cc_rr, 4)}.",
        (
            f"Sweep max after re-ranking: {_fmt(float(mx_cc_rr['confident_label_rate']), 4)} "
            f"({int(mx_cc_rr['n_confident'])}/{int(mx_cc_rr['n'])}) at CCI={mx_cc_rr['cci_threshold']}, "
            f"W3={mx_cc_rr['w3_cutoff']}, contam pct={int(mx_cc_rr['contam_percentile'])}."
        ),
        "",
        "Primary numbers use inherited 440-panel ranks.",
        "",
        "## Files",
        "",
        "- `T1_complete_case_counts.csv`",
        "- `T1_270_sweep_complete_case.csv`",
        "- `T1_270_sweep_complete_case_reranked.csv`",
        "- `T1_panel_flags.csv`",
        "",
    ]
    (OUT / "T1_complete_case.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "n_complete": n_complete,
        "n_three": n_three,
        "n_parse": n_parse,
        "n_p2": n_p2,
        "c_full": c_full,
        "c_3": c_3,
        "c_cc": c_cc,
        "c_cc_rr": c_cc_rr,
        "rate_full": rate_full,
        "rate_3": rate_3,
        "rate_cc": rate_cc,
        "rate_cc_rr": rate_cc_rr,
        "mx_full": mx_full,
        "mx_cc": mx_cc,
        "mx_cc_rr": mx_cc_rr,
    }


# ---------------------------------------------------------------------------
# T2 — DS-02 intrusion
# ---------------------------------------------------------------------------

def _match_span(family: str, pid: str, model_ans: str) -> str:
    if family == "ALGO" and pid.startswith("SP"):
        ms = list(re.finditer(r"Path\s*:\s*(.+?)(?:,\s*Cost\s*:|$)", str(model_ans), flags=re.I))
        return ms[-1].group(0).strip() if ms else ""
    if family == "ALGO" and pid.startswith("WIS"):
        m = re.search(r"Selected\s*:\s*\{[^}]*\}(?:\s*,\s*Total\s*:\s*-?\d+)?", str(model_ans), flags=re.I)
        return m.group(0).strip() if m else ""
    if family == "ALGO" and pid.startswith("CC"):
        m = re.search(
            r"(?:Count|Total)\s*:\s*-?\d+.{0,80}(?:Coins|Scoops)\s*:\s*\[[^\]]*\]",
            str(model_ans),
            flags=re.I | re.S,
        )
        return (m.group(0).strip() if m else "")[:400]
    return ""


def _collect_intrusions(algo, gsm, bw, banks) -> pd.DataFrame:
    rows = []
    packs = [
        ("ALGO", algo, banks["algo_bank"], lambda m, df: df),
        ("GSM", gsm, banks["gsm_bank"], lambda m, df: df),
    ]
    for family, by_m, bank, _ in packs:
        bank_w3 = bank[bank["variant_type"] == "W3"].drop_duplicates("problem_id").set_index("problem_id")
        bank_can = bank[bank["variant_type"] == "canonical"].drop_duplicates("problem_id").set_index("problem_id")
        for m in MODELS:
            df = by_m[m]
            if df.empty:
                continue
            ac, gc = cr._ans_col(df), cr._gt_col(df)
            w3 = df[df["variant_type"] == "W3"]
            can = df[df["variant_type"] == "canonical"].set_index("problem_id")
            for _, r in w3.iterrows():
                if bool(r["ok"]):
                    continue
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
                is_int = cr._equals_canonical(family, pid, ans, can_gt, w3_gt)
                rows.append({
                    "family": family,
                    "model": m,
                    "problem_id": pid,
                    "canonical_gold": can_gt,
                    "W3_gold": w3_gt,
                    "W3_model_answer": ans,
                    "match_span": _match_span(family, pid, ans) if is_int else "",
                    "intrusion": bool(is_int),
                })

    bank = banks["bw_bank"]
    bank_w3 = bank[bank["variant_type"] == "W3"].drop_duplicates("problem_id").set_index("problem_id")
    bank_can = bank[bank["variant_type"] == "canonical"].drop_duplicates("problem_id").set_index("problem_id")
    for m in MODELS:
        df = bw[bw["model_short"] == m] if not bw.empty else pd.DataFrame()
        if df.empty:
            continue
        ac, gc = cr._ans_col(df), cr._gt_col(df)
        w3 = df[df["variant_type"] == "W3"]
        can = df[df["variant_type"] == "canonical"].set_index("problem_id")
        for _, r in w3.iterrows():
            if bool(r["ok"]):
                continue
            pid = str(r["problem_id"])
            ans = str(r.get(ac, ""))
            w3_gt = str(bank_w3.loc[pid].get("correct_answer", r.get(gc, ""))) if pid in bank_w3.index else str(r.get(gc, ""))
            can_gt = str(bank_can.loc[pid].get("correct_answer", "")) if pid in bank_can.index else ""
            if pid in can.index and not can_gt:
                can_gt = str(can.loc[pid].get(gc, ""))
            is_int = cr._equals_canonical("BW", pid, ans, can_gt, w3_gt)
            rows.append({
                "family": "BW",
                "model": m,
                "problem_id": pid,
                "canonical_gold": can_gt,
                "W3_gold": w3_gt,
                "W3_model_answer": ans,
                "match_span": _match_span("BW", pid, ans) if is_int else "",
                "intrusion": bool(is_int),
            })
    return pd.DataFrame(rows)


def run_t2(algo, gsm, bw, banks) -> dict:
    detail = _collect_intrusions(algo, gsm, bw, banks)
    detail.to_csv(OUT / "T2_intrusion_detail.csv", index=False)

    rate_rows = []
    for family in FAMILIES:
        for m in MODELS:
            sub = detail[(detail["family"] == family) & (detail["model"] == m)]
            n_err = int(len(sub))
            n_int = int(sub["intrusion"].sum()) if n_err else 0
            rate = (n_int / n_err) if n_err else float("nan")
            lo, hi = cr.wilson(n_int, n_err) if n_err else (float("nan"), float("nan"))
            rate_rows.append({
                "model": m,
                "family": family,
                "n_W3_errors": n_err,
                "n_intrusions": n_int,
                "intrusion_rate": rate,
                "wilson_ci95_lo": lo,
                "wilson_ci95_hi": hi,
            })
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(OUT / "T2_intrusion_rates.csv", index=False)

    fisher_rows = []
    algo_rates = rates[rates["family"] == "ALGO"].set_index("model")
    o4_k = int(algo_rates.loc["o4-mini", "n_intrusions"])
    o4_n = int(algo_rates.loc["o4-mini", "n_W3_errors"])
    for m in MODELS:
        if m == "o4-mini":
            continue
        k = int(algo_rates.loc[m, "n_intrusions"])
        n = int(algo_rates.loc[m, "n_W3_errors"])
        table = np.array([[o4_k, o4_n - o4_k], [k, n - k]], dtype=int)
        try:
            oddsr, p = stats.fisher_exact(table, alternative="two-sided")
        except ValueError:
            oddsr, p = float("nan"), float("nan")
        fisher_rows.append({
            "comparison": f"o4-mini vs {m}",
            "o4mini_intrusions": o4_k,
            "o4mini_W3_errors": o4_n,
            "other_intrusions": k,
            "other_W3_errors": n,
            "table_o4_int": o4_k,
            "table_o4_non": o4_n - o4_k,
            "table_other_int": k,
            "table_other_non": n - k,
            "odds_ratio": float(oddsr),
            "fisher_p_two_sided": float(p),
        })
    fisher = pd.DataFrame(fisher_rows)
    fisher.to_csv(OUT / "T2_intrusion_fisher_algo.csv", index=False)

    ex_rows = []
    for m in MODELS:
        hits = detail[(detail["model"] == m) & (detail["intrusion"])].copy()
        # Prefer ALGO (the only family with structural rename intrusions).
        hits["fam_rank"] = hits["family"].map({"ALGO": 0, "GSM": 1, "BW": 2})
        hits = hits.sort_values(["fam_rank", "family", "problem_id"])
        taken = hits.head(5)
        for _, r in taken.iterrows():
            ex_rows.append({
                "model": m,
                "family": r["family"],
                "problem_id": r["problem_id"],
                "canonical_gold": r["canonical_gold"],
                "W3_gold": r["W3_gold"],
                "match_span": r["match_span"],
                "W3_model_answer": r["W3_model_answer"],
            })
    examples = pd.DataFrame(ex_rows)
    examples.to_csv(OUT / "T2_intrusion_examples.csv", index=False)

    md = [
        "# T2 — DS-02 intrusion errors (paper-ready)",
        "",
        "Canonical-answer intrusion: among W3 **errors**, the model’s W3 response encodes the **canonical** gold (pre-rename identifiers) and does **not** encode the W3 gold. Matching is structured (SP path tokens, CC coin multiset, WIS selected set, GSM last number, BW action list), same detector as `rebuild/compute_rebuild.py`.",
        "",
        "GSM W3 is name-substitution that preserves the numeric gold, so a W3 error that still equals the canonical number is almost impossible unless verifier and extractor disagree. ALGO W3 relabels nodes/items (0,1,2 → Hub A,B,C); emitting the canonical identifiers on the renamed instance is the intrusion.",
        "",
        "## Rates",
        "",
        "| model | family | n_W3_errors | n_intrusions | intrusion_rate | Wilson 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, r in rates.iterrows():
        md.append(
            f"| {r.model} | {r.family} | {int(r.n_W3_errors)} | {int(r.n_intrusions)} | "
            f"{_fmt(r.intrusion_rate, 3)} | {_fmt_ci(r.wilson_ci95_lo, r.wilson_ci95_hi)} |"
        )

    algo = rates[rates["family"] == "ALGO"]
    k_algo = int(algo["n_intrusions"].sum())
    n_algo = int(algo["n_W3_errors"].sum())
    gsm_r = rates[rates["family"] == "GSM"]
    k_gsm = int(gsm_r["n_intrusions"].sum())
    n_gsm = int(gsm_r["n_W3_errors"].sum())
    bw_r = rates[rates["family"] == "BW"]
    k_bw = int(bw_r["n_intrusions"].sum())
    n_bw = int(bw_r["n_W3_errors"].sum())
    md += [
        "",
        f"ALGO pooled: **{k_algo}/{n_algo}**. GSM pooled: **{k_gsm}/{n_gsm}**. BW pooled: **{k_bw}/{n_bw}**.",
        "",
        "## Fisher exact: o4-mini vs each other model (ALGO W3 errors)",
        "",
        "2×2 of (intrusion, non-intrusion) among W3 errors. Two-sided Fisher. Odds ratio > 1 means o4-mini has a higher intrusion odds than the comparison model.",
        "",
        "| comparison | o4-mini | other | OR | Fisher p |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in fisher.iterrows():
        md.append(
            f"| {r.comparison} | {int(r.o4mini_intrusions)}/{int(r.o4mini_W3_errors)} | "
            f"{int(r.other_intrusions)}/{int(r.other_W3_errors)} | "
            f"{_fmt(r.odds_ratio, 2)} | {_fmt_p(r.fisher_p_two_sided)} |"
        )

    md += [
        "",
        "## Verbatim examples (true intrusions only)",
        "",
        "Up to five true hits per model. Llama has none. Full W3 traces are in `T2_intrusion_examples.csv`; below: canonical gold, then the span of the W3 response that reproduced it.",
        "",
    ]
    for m in MODELS:
        hits = examples[examples["model"] == m] if len(examples) else pd.DataFrame()
        md.append(f"### {m} (n={len(hits)})")
        md.append("")
        if hits.empty:
            md.append("No canonical-answer intrusions.")
            md.append("")
            continue
        for i, r in enumerate(hits.itertuples(index=False), 1):
            span = r.match_span or "(match span not extracted; see full W3_model_answer in CSV)"
            md.append(f"**{i}. {r.family} {r.problem_id}**")
            md.append("")
            md.append(f"- Canonical gold: `{r.canonical_gold}`")
            md.append(f"- W3 gold: `{r.W3_gold}`")
            md.append(f"- W3 response (reproducing span): `{span}`")
            md.append("")
    md += [
        "Unparsed or empty W3 answers are counted as errors and as non-intrusions (never labelled intrusion without a structured canonical match).",
        "",
        "## Files",
        "",
        "- `T2_intrusion_rates.csv`",
        "- `T2_intrusion_fisher_algo.csv`",
        "- `T2_intrusion_examples.csv` (full W3 traces)",
        "- `T2_intrusion_detail.csv` (every W3 error)",
        "",
    ]
    (OUT / "T2_intrusion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"rates": rates, "fisher": fisher, "examples": examples, "detail": detail}


# ---------------------------------------------------------------------------
# T3 — DS-14 double dissociation
# ---------------------------------------------------------------------------

def _boot_ci(vals: np.ndarray) -> tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def _boot_acc_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), np.array([])
    diffs = np.empty(N_BOOT, dtype=float)
    idx = np.arange(n)
    for i in range(N_BOOT):
        draw = RNG.choice(idx, size=n, replace=True)
        diffs[i] = float(a[draw].mean() - b[draw].mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi), diffs


def _boot_dod(a1, b1, a2, b2) -> tuple[float, float, float, float]:
    """Bootstrap CI and two-sided p for difference-of-differences.

    DoD = (acc_a - acc_b)_S1 - (acc_a - acc_b)_S2.
    Items are resampled independently within each subtype (different ID sets).
    Pairing within subtype is preserved (same draw index for a and b).
    """
    n1, n2 = len(a1), len(a2)
    dods = np.empty(N_BOOT, dtype=float)
    idx1, idx2 = np.arange(n1), np.arange(n2)
    for i in range(N_BOOT):
        d1 = RNG.choice(idx1, size=n1, replace=True)
        d2 = RNG.choice(idx2, size=n2, replace=True)
        dods[i] = (float(a1[d1].mean() - b1[d1].mean()) - float(a2[d2].mean() - b2[d2].mean()))
    lo, hi = np.quantile(dods, [0.025, 0.975])
    obs = float(a1.mean() - b1.mean()) - float(a2.mean() - b2.mean())
    p = float(min(1.0, 2 * min(np.mean(dods <= 0), np.mean(dods >= 0))))
    return obs, float(lo), float(hi), p


def _loglinear_threeway(tables: list[np.ndarray]) -> tuple[float, float, str]:
    """LR test of the three-way interaction in a 2×2×2 Poisson log-linear model.

    tables: two 2×2 arrays, each [[a_correct, a_incorrect], [b_correct, b_incorrect]].
    Returns (G2, p, note).
    """
    rows = []
    for s, t in enumerate(tables):
        for m in (0, 1):
            for outcome, col in ((1, 0), (0, 1)):
                rows.append({
                    "subtype": s,
                    "model": m,
                    "correct": outcome,
                    "count": int(t[m, col]),
                })
    df = pd.DataFrame(rows)
    if (df["count"] == 0).any():
        df = df.copy()
        df["count"] = df["count"] + 0.5
        note = "Haldane-Anscombe +0.5 continuity (zero cell)"
    else:
        note = ""
    try:
        sat = smf.glm(
            "count ~ C(model)*C(subtype)*C(correct)",
            data=df,
            family=sm.families.Poisson(),
        ).fit(disp=0)
        no3 = smf.glm(
            "count ~ (C(model)+C(subtype)+C(correct))**2",
            data=df,
            family=sm.families.Poisson(),
        ).fit(disp=0)
        g2 = float(2 * (sat.llf - no3.llf))
        p = float(stats.chi2.sf(max(g2, 0.0), 1))
        return g2, p, note
    except Exception as e:  # noqa: BLE001
        return float("nan"), float("nan"), f"log-linear failed: {e}"


def _breslow_day(tables: list[np.ndarray]) -> tuple[float, float, str]:
    try:
        st = StratifiedTable(tables)
        res = st.test_equal_odds()
        stat = float(np.asarray(res.statistic).reshape(-1)[0])
        p = float(np.asarray(res.pvalue).reshape(-1)[0])
        return stat, p, ""
    except Exception as e:  # noqa: BLE001
        return float("nan"), float("nan"), str(e)


def run_t3(algo: dict[str, pd.DataFrame]) -> dict:
    maps: dict[str, tuple[dict[str, int], dict[str, int]]] = {}
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

    inv = pd.read_csv(REBUILD / "p1_pairwise_inversion.csv")
    inv_cm = inv[inv["definition"] == "canonically-matched"].copy()

    pair_rows = []
    crossover_rows = []
    subtypes = ["SP", "CC", "WIS"]

    for ma, mb in combinations(MODELS, 2):
        w3a, cana = maps[ma]
        w3b, canb = maps[mb]
        vecs: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        stats_sub: dict[str, dict] = {}
        for sub in subtypes:
            idset = set(cr.PAPER_ADV[sub])
            paired_ids = sorted(idset & set(w3a) & set(w3b))
            matched_ids = [pid for pid in paired_ids if cana.get(pid) == 1 and canb.get(pid) == 1]
            n = len(matched_ids)
            if n == 0:
                stats_sub[sub] = {
                    "n": 0, "a_W3": 0, "b_W3": 0, "acc_a": float("nan"), "acc_b": float("nan"),
                    "acc_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "fisher_p": float("nan"), "odds_ratio": float("nan"), "ids": [],
                }
                continue
            aa = np.array([w3a[pid] for pid in matched_ids], dtype=float)
            bb = np.array([w3b[pid] for pid in matched_ids], dtype=float)
            ka, kb = int(aa.sum()), int(bb.sum())
            table = np.array([[ka, n - ka], [kb, n - kb]], dtype=int)
            try:
                oddsr, p = stats.fisher_exact(table, alternative="two-sided")
            except ValueError:
                oddsr, p = float("nan"), float("nan")
            lo, hi, _ = _boot_acc_diff(aa, bb)
            diff = float(aa.mean() - bb.mean())
            vecs[sub] = (aa, bb, matched_ids)
            stats_sub[sub] = {
                "n": n, "a_W3": ka, "b_W3": kb,
                "acc_a": float(aa.mean()), "acc_b": float(bb.mean()),
                "acc_diff": diff, "ci_lo": lo, "ci_hi": hi,
                "fisher_p": float(p), "odds_ratio": float(oddsr), "ids": matched_ids,
            }
            csv_hit = inv_cm[
                (inv_cm["subtype"] == sub) & (inv_cm["model_a"] == ma) & (inv_cm["model_b"] == mb)
            ]
            csv_p = float(csv_hit["fisher_p"].iloc[0]) if len(csv_hit) else float("nan")
            pair_rows.append({
                "model_a": ma, "model_b": mb, "subtype": sub, "n": n,
                "a_W3": ka, "b_W3": kb, "acc_diff": diff,
                "ci_lo": lo, "ci_hi": hi, "fisher_p": float(p),
                "odds_ratio": float(oddsr),
                "rebuild_csv_fisher_p": csv_p,
            })

        for s1, s2 in combinations(subtypes, 2):
            st1, st2 = stats_sub[s1], stats_sub[s2]
            if st1["n"] == 0 or st2["n"] == 0:
                continue
            d1, d2 = st1["acc_diff"], st2["acc_diff"]
            opposite = (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)
            if not opposite:
                continue
            both_sig = (
                st1["fisher_p"] == st1["fisher_p"] and st2["fisher_p"] == st2["fisher_p"]
                and st1["fisher_p"] < ALPHA and st2["fisher_p"] < ALPHA
            )
            ci1_excl = st1["ci_lo"] > 0 or st1["ci_hi"] < 0
            ci2_excl = st2["ci_lo"] > 0 or st2["ci_hi"] < 0
            both_ci = ci1_excl and ci2_excl

            t1 = np.array([[st1["a_W3"], st1["n"] - st1["a_W3"]],
                           [st1["b_W3"], st1["n"] - st1["b_W3"]]], dtype=int)
            t2 = np.array([[st2["a_W3"], st2["n"] - st2["a_W3"]],
                           [st2["b_W3"], st2["n"] - st2["b_W3"]]], dtype=int)
            bd_stat, bd_p, bd_note = _breslow_day([t1, t2])
            g2, g2_p, g2_note = _loglinear_threeway([t1, t2])

            aa1, bb1, _ = vecs[s1]
            aa2, bb2, _ = vecs[s2]
            dod, dod_lo, dod_hi, dod_p = _boot_dod(aa1, bb1, aa2, bb2)
            dod_excl = dod_lo > 0 or dod_hi < 0
            interaction_sig = (
                (bd_p == bd_p and bd_p < ALPHA)
                or (g2_p == g2_p and g2_p < ALPHA)
                or dod_excl
            )

            if both_sig and both_ci and interaction_sig:
                verdict = "strict_crossover"
            elif both_sig:
                verdict = "suggestive_both_fisher"
            elif (st1["fisher_p"] < ALPHA) or (st2["fisher_p"] < ALPHA):
                verdict = "suggestive_single_fisher"
            else:
                verdict = "suggestive_direction_only"

            crossover_rows.append({
                "model_a": ma, "model_b": mb,
                "subtype_1": s1, "subtype_2": s2,
                "n_1": st1["n"], "n_2": st2["n"],
                "a_W3_1": st1["a_W3"], "b_W3_1": st1["b_W3"],
                "a_W3_2": st2["a_W3"], "b_W3_2": st2["b_W3"],
                "acc_diff_1": d1, "ci_lo_1": st1["ci_lo"], "ci_hi_1": st1["ci_hi"],
                "acc_diff_2": d2, "ci_lo_2": st2["ci_lo"], "ci_hi_2": st2["ci_hi"],
                "fisher_p_1": st1["fisher_p"], "fisher_p_2": st2["fisher_p"],
                "odds_ratio_1": st1["odds_ratio"], "odds_ratio_2": st2["odds_ratio"],
                "breslow_day_stat": bd_stat, "breslow_day_p": bd_p, "breslow_day_note": bd_note,
                "loglinear_G2": g2, "loglinear_p": g2_p, "loglinear_note": g2_note,
                "dod": dod, "dod_ci_lo": dod_lo, "dod_ci_hi": dod_hi, "dod_boot_p": dod_p,
                "both_fisher_p05": both_sig,
                "both_bootstrap_ci_exclude_0": both_ci,
                "interaction_sig": interaction_sig,
                "verdict": verdict,
            })

    pairs_df = pd.DataFrame(pair_rows)
    xo = pd.DataFrame(crossover_rows)
    pairs_df.to_csv(OUT / "T3_canonically_matched_cells.csv", index=False)
    xo.to_csv(OUT / "T3_crossover.csv", index=False)

    strict = xo[xo["verdict"] == "strict_crossover"] if len(xo) else xo
    sugg = xo[xo["verdict"] != "strict_crossover"] if len(xo) else xo

    def _arm_line(r, which: str) -> str:
        s = r[f"subtype_{which}"]
        n = int(r[f"n_{which}"])
        ka = int(r[f"a_W3_{which}"])
        kb = int(r[f"b_W3_{which}"])
        return (
            f"{s} (n={n}): {r.model_a} {ka}/{n} vs {r.model_b} {kb}/{n}, "
            f"Δ = {_fmt(r[f'acc_diff_{which}'], 3)} {_fmt_ci(r[f'ci_lo_{which}'], r[f'ci_hi_{which}'])}, "
            f"Fisher p = {_fmt_p(r[f'fisher_p_{which}'])}, OR = {_fmt(r[f'odds_ratio_{which}'], 2)}"
        )

    md = [
        "# T3 — DS-14 double dissociation (formal)",
        "",
        "Source: `rebuild/p1_pairwise_inversion.csv`, recomputed at item level from the same frozen ALGO adversarial IDs and P1 loaders as the rebuild. **Definition = canonically-matched**: the ID must be in the frozen subtype list and both models must be canonical-correct. Effect size is W3 accuracy of model A minus model B; 95% CI is a paired bootstrap (10,000), same as the rebuild.",
        "",
        "A **genuine crossover** is a model pair with significant effects in **opposite directions** on two different subtypes. Single dissociations (one subtype significant, the other not) can be a difficulty artefact; a crossover cannot.",
        "",
        "## Strict vs suggestive",
        "",
        "**Strict crossover** (all of):",
        "",
        "1. Canonically-matched definition.",
        "2. Two different subtypes.",
        "3. Opposite signs of Δ (acc_A − acc_B).",
        "4. Fisher exact two-sided p < 0.05 on **both** subtypes.",
        "5. Bootstrap 95% CI of Δ excludes 0 on **both** subtypes.",
        "6. Combined 2×2×2 interaction is significant: Breslow–Day p < 0.05 **or** log-linear three-way G² p < 0.05 **or** bootstrap 95% CI of the difference-of-differences excludes 0.",
        "",
        "The 2×2×2 table is model (A, B) × subtype (S1, S2) × W3 outcome (correct, incorrect). Breslow–Day tests homogeneity of the two subtype odds ratios. The Poisson log-linear G² tests the three-way term against the all-two-way model (1 df). Zero cells get Haldane–Anscombe +0.5 before the log-linear fit. Difference-of-differences = Δ_S1 − Δ_S2, items resampled independently within subtype, pairing within subtype preserved.",
        "",
        "**Suggestive**: opposite-signed Δ on two subtypes, but (4)–(6) not all met (one Fisher non-significant, a CI covering 0, or the interaction not significant).",
        "",
        "α = 0.05, uncorrected. 10 model pairs × 3 subtype-pair slots = 30 implicit tests.",
        "",
        f"Opposite-signed subtype pairs found: **{len(xo)}**. Strict: **{len(strict)}**. Suggestive: **{len(sugg)}**.",
        "",
    ]

    if len(strict):
        md.append("## Pairs that meet the strict crossover criterion")
        md.append("")
        for _, r in strict.iterrows():
            worse = r.subtype_1 if r.acc_diff_1 < 0 else r.subtype_2
            better = r.subtype_2 if r.acc_diff_1 < 0 else r.subtype_1
            md.append(f"### {r.model_a} vs {r.model_b} — {r.subtype_1} × {r.subtype_2}")
            md.append("")
            md.append(f"- {_arm_line(r, '1')}")
            md.append(f"- {_arm_line(r, '2')}")
            md.append(
                f"- Combined 2×2×2: Breslow–Day = {_fmt(r.breslow_day_stat, 2)}, p = {_fmt_p(r.breslow_day_p)}"
                + (f" ({r.breslow_day_note})" if r.breslow_day_note else "")
                + f"; log-linear G² = {_fmt(r.loglinear_G2, 2)}, p = {_fmt_p(r.loglinear_p)}"
                + (f" ({r.loglinear_note})" if r.loglinear_note else "")
                + f"; ΔΔ = {_fmt(r.dod, 3)} {_fmt_ci(r.dod_ci_lo, r.dod_ci_hi)}."
            )
            md.append("")
            md.append(
                f"**Verdict: strict crossover.** Relative to {r.model_b}, {r.model_a} is worse on {worse} "
                f"and better on {better}; both simple effects and the interaction survive the criterion. "
                "Caveat: canonically-matched n on CC is small (matched IDs require both models canonical-correct)."
            )
            md.append("")
    else:
        md.append("## Pairs that meet the strict crossover criterion")
        md.append("")
        md.append("None.")
        md.append("")

    md.append("## Pairs that are only suggestive")
    md.append("")
    if len(sugg):
        md.append("| pair | subtypes | Δ_1 [CI] | Fisher p_1 | Δ_2 [CI] | Fisher p_2 | Breslow–Day p | log-linear p | ΔΔ [CI] | why not strict |")
        md.append("|---|---|---|---:|---|---:|---:|---:|---|---|")
        why = {
            "suggestive_both_fisher": "both Fisher < 0.05 but CI or interaction short",
            "suggestive_single_fisher": "only one subtype Fisher-significant",
            "suggestive_direction_only": "opposite signs, neither Fisher < 0.05",
        }
        for _, r in sugg.iterrows():
            md.append(
                f"| {r.model_a} vs {r.model_b} | {r.subtype_1} × {r.subtype_2} | "
                f"{_fmt(r.acc_diff_1, 3)} {_fmt_ci(r.ci_lo_1, r.ci_hi_1)} | {_fmt_p(r.fisher_p_1)} | "
                f"{_fmt(r.acc_diff_2, 3)} {_fmt_ci(r.ci_lo_2, r.ci_hi_2)} | {_fmt_p(r.fisher_p_2)} | "
                f"{_fmt_p(r.breslow_day_p)} | {_fmt_p(r.loglinear_p)} | "
                f"{_fmt(r.dod, 3)} {_fmt_ci(r.dod_ci_lo, r.dod_ci_hi)} | {why.get(r.verdict, r.verdict)} |"
            )
        md.append("")
        for _, r in sugg.iterrows():
            md.append(f"### {r.model_a} vs {r.model_b} — {r.subtype_1} × {r.subtype_2} ({r.verdict})")
            md.append("")
            md.append(f"- {_arm_line(r, '1')}")
            md.append(f"- {_arm_line(r, '2')}")
            md.append(
                f"- Combined 2×2×2: Breslow–Day p = {_fmt_p(r.breslow_day_p)}; "
                f"log-linear G² p = {_fmt_p(r.loglinear_p)}; "
                f"ΔΔ = {_fmt(r.dod, 3)} {_fmt_ci(r.dod_ci_lo, r.dod_ci_hi)}."
            )
            md.append("")
    else:
        md.append("No other model pair has opposite-signed canonically-matched Δ on two subtypes.")
        md.append("")

    md += [
        "## Pairs with no opposite-signed subtype pair",
        "",
        "Every other canonically-matched pair is either same-sign across subtypes, zero on the second subtype, or missing a matched ID intersection. Those are single dissociations or nulls, not crossovers. Full cell table: `T3_canonically_matched_cells.csv`.",
        "",
        "## Files",
        "",
        "- `T3_crossover.csv`",
        "- `T3_canonically_matched_cells.csv`",
        "",
    ]
    (OUT / "T3_double_dissociation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"cells": pairs_df, "crossover": xo, "strict": strict, "suggestive": sugg}


def write_index(t1, t2, t3) -> None:
    c_full, c_cc = t1["c_full"], t1["c_cc"]
    mx_full, mx_cc = t1["mx_full"], t1["mx_cc"]
    rates, fisher = t2["rates"], t2["fisher"]
    xo = t3["crossover"]
    n_strict = int((xo["verdict"] == "strict_crossover").sum()) if len(xo) else 0

    algo_o4 = rates[(rates["family"] == "ALGO") & (rates["model"] == "o4-mini")].iloc[0]
    lines = [
        "# Solidify report",
        "",
        "Three analyses from existing data. Script: `rebuild/solidify/compute_solidify.py`. Frozen filters unchanged.",
        "",
        "## T1 — Complete-case triangulation",
        "",
        f"n_complete = **{t1['n_complete']}** (of 440). Default labels on that subset: retrieval={c_cc['retrieval']}, computation={c_cc['computation']}, mixed={c_cc['mixed']}, ambiguous={c_cc['ambiguous']}.",
        "",
        "| | n | retrieval | computation | mixed | ambiguous | confident-label rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| full panel (missing **or** disagree) | {c_full['n']} | {c_full['retrieval']} | {c_full['computation']} | {c_full['mixed']} | {c_full['ambiguous']} | {_fmt(t1['rate_full'], 4)} |",
        f"| complete-case (disagree only) | {c_cc['n']} | {c_cc['retrieval']} | {c_cc['computation']} | {c_cc['mixed']} | {c_cc['ambiguous']} | {_fmt(t1['rate_cc'], 4)} |",
        "",
        f"270-sweep **maximum** confident-label rate: **{_fmt(float(mx_full['confident_label_rate']), 4)}** (full) vs **{_fmt(float(mx_cc['confident_label_rate']), 4)}** (complete-case).",
        "",
        "Details: `T1_complete_case.md`.",
        "",
        "## T2 — DS-02 intrusion",
        "",
        f"o4-mini ALGO: {int(algo_o4.n_intrusions)}/{int(algo_o4.n_W3_errors)} = {_fmt(algo_o4.intrusion_rate, 3)} {_fmt_ci(algo_o4.wilson_ci95_lo, algo_o4.wilson_ci95_hi)}.",
        "Fisher vs other models on ALGO:",
        "",
    ]
    for _, r in fisher.iterrows():
        lines.append(f"- {r.comparison}: OR={_fmt(r.odds_ratio, 2)}, p={_fmt_p(r.fisher_p_two_sided)}")
    lines += [
        "",
        "Details: `T2_intrusion.md`.",
        "",
        "## T3 — DS-14 double dissociation",
        "",
        f"Canonically-matched opposite-sign subtype pairs: {len(xo)}. Strict crossovers: **{n_strict}**. Suggestive: **{len(xo) - n_strict}**.",
        "",
        "Details: `T3_double_dissociation.md`.",
        "",
    ]
    (OUT / "SOLIDIFY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("T1 complete-case triangulation…")
    t1 = run_t1()
    print(f"  n_complete={t1['n_complete']}  default_cc_rate={t1['rate_cc']:.4f}  sweep_max_cc={float(t1['mx_cc']['confident_label_rate']):.4f}")

    print("Loading P1 for T2/T3…")
    banks = cr.load_banks()
    algo = {m: cr.load_algo_p1(cr.TAG[m]) for m in MODELS}
    gsm = {m: cr.load_gsm_p1(cr.TAG[m]) for m in MODELS}
    bw = cr.load_bw_p1(set(banks["bw_canon"]))

    print("T2 intrusion…")
    t2 = run_t2(algo, gsm, bw, banks)
    print(f"  ALGO intrusions: {int(t2['detail'][(t2['detail'].family=='ALGO') & t2['detail'].intrusion].shape[0])}")

    print("T3 double dissociation…")
    t3 = run_t3(algo)
    print(f"  opposite-sign pairs={len(t3['crossover'])} strict={len(t3['strict'])}")

    write_index(t1, t2, t3)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
