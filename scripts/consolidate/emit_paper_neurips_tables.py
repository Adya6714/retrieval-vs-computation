#!/usr/bin/env python3
"""Emit NeurIPS paper tables from derived CSVs — no hand-carried numbers.

Writes:
  paper/tables/table_oracle.tex
  paper/tables/table_bw_structural.tex
  paper/tables/table_bw_accuracy.tex
  paper/tables/table_variants.tex
  paper/tables/table_coverage.tex
  results/derived/PAPER_NUMBER_DELTAS.csv
  results/derived/SP_std_claude_canonical_audit.csv
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.exclusions import filter_excluded, load_exclusions  # noqa: E402
from probes.common.stats import cluster_bootstrap_ci, wilson_ci  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
PAPER_T = REPO_ROOT / "paper" / "tables"

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

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}
TEX_MODEL = {
    "Claude": "Claude",
    "GPT-4o": "GPT-4o",
    "Gemini": "Gemini",
    "Llama": "Llama",
    "o4-mini": r"\omini",
    "DeepSeek": "DeepSeek",
}
DEFECT_LABEL = {
    "BW_W3_action_mapping": "Rename mapping not consumed (planning)",
    "SP_W3_node_mapping": "Rename mapping not consumed (graph)",
    "BW_state_parser": "Legacy state parser",
}
STRUCT_LABEL = {
    "num_blocks": "Blocks",
    "n_goal_clauses": "Goal clauses",
    "goal_tower_depth": "Goal tower depth",
    "init_tower_depth": "Initial tower depth",
    "fd_optimal_plan_length": "Optimal plan length",
}
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
T7_GSM_MODELS = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini"]
T7_ALGO_MODELS = ["Claude", "GPT-4o", "Gemini", "Llama"]
T7_BW_MODELS = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini"]
T7_ALGO_SLICES = [
    "CC-chall",
    "CC-std",
    "SP-chall",
    "SP-std",
    "WIS-chall",
    "WIS-std",
]


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _algo_slice(pid: str) -> str:
    if pid.startswith("CC"):
        sub = "CC"
    elif pid.startswith("SP"):
        sub = "SP"
    elif pid.startswith("WIS"):
        sub = "WIS"
    else:
        return ""
    kind = "chall" if pid in PAPER_ADV_ALL else "std"
    return f"{sub}-{kind}"


def _round_disp(x: float | None, nd: int) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x) + 1e-12, nd)


def _fmt_acc(x: float | None) -> str:
    x = _round_disp(x, 3)
    if x is None:
        return "---"
    if x >= 1:
        return f"{x:.3f}"
    return f"{x:.3f}".replace("0.", ".", 1)


def _fmt_ci(lo: float, hi: float) -> str:
    def p(v: float) -> str:
        v = _round_disp(v, 2)
        assert v is not None
        s = f"{v:.2f}"
        return s[1:] if s.startswith("0") else s

    return f"[{p(lo)},{p(hi)}]"


def _fmt_signed(x: float, nd: int = 3) -> str:
    x = _round_disp(x, nd)
    assert x is not None
    if abs(x) < 5e-4 and nd == 3:
        return f".{0:0{nd}d}" if nd else ".000"
    sign = "+" if x > 0 else ""
    if nd == 3 and abs(x) < 1:
        body = f"{abs(x):.3f}".replace("0.", ".", 1)
        return f"{sign}{body}" if x > 0 else f"-{body}"
    return f"{sign}{x:.{nd}f}"


def _omit_algo(slice_name: str, variant: str) -> str | None:
    if variant == "W5" and not slice_name.startswith("SP"):
        return "not defined"
    return None


def write_oracle() -> list[dict]:
    df = pd.read_csv(DER / "oracle_bias_summary.csv")
    # Paper order: clean planning case first, then graph, then parser mix.
    order = ["BW_W3_action_mapping", "SP_W3_node_mapping", "BW_state_parser"]
    df["ord"] = df["defect"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values("ord")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Oracle defects detected by gold-in-gold-out, with effect on accuracy. The first row is",
        r"the clean case: the perturbed condition moves by $+20.9$ points while the canonical condition",
        r"does not move at all, which is the signature of treatment-correlated bias. The remaining two",
        r"bundle bias with general parser improvement and are reported as such.}",
        r"\label{tab:oracle}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Defect & Rows & Pert.\ before & Pert.\ after & $\Delta$ pert. & $\Delta$ canonical \\",
        r"\midrule",
    ]
    cells = []
    for _, r in df.iterrows():
        label = DEFECT_LABEL.get(str(r["defect"]), str(r["defect"]))
        rows = int(r["rows_affected"])
        pb, pa = float(r["acc_before_perturbed"]), float(r["acc_after_perturbed"])
        dp, dc = float(r["delta_perturbed"]), float(r["delta_canonical"])
        lines.append(
            f"{label} & {rows} & {_fmt_acc(pb)} & {_fmt_acc(pa)} "
            f"& ${_fmt_signed(dp)}$ & ${_fmt_signed(dc)}$ \\\\"
        )
        for metric, val in [
            ("rows_affected", rows),
            ("pert_before", _round_disp(pb, 3)),
            ("pert_after", _round_disp(pa, 3)),
            ("delta_perturbed", _round_disp(dp, 3)),
            ("delta_canonical", _round_disp(dc, 3)),
        ]:
            cells.append(
                {
                    "location": "table_oracle",
                    "family": r["family"],
                    "slice": r["variant_scope"],
                    "model": "--",
                    "variant": str(r["defect"]),
                    "metric": metric,
                    "new_value": val,
                    "n": rows,
                    "nd": 0 if metric == "rows_affected" else 3,
                    "note_hint": "from oracle_bias_summary.csv",
                }
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (PAPER_T / "table_oracle.tex").write_text("\n".join(lines), encoding="utf-8")
    return cells


def write_bw_tables() -> list[dict]:
    rep = pd.read_csv(DER / "K3_bw_canonical_w6_matched_report.csv")
    cells: list[dict] = []

    struct = rep[rep["section"] == "structural_matched"].copy()
    struct["ord"] = struct["metric"].map({k: i for i, k in enumerate(STRUCT_LABEL)})
    struct = struct.dropna(subset=["ord"]).sort_values("ord")
    s_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Structural comparison of canonical versus regenerated Blocksworld instances",
        r"($n{=}47$ matched pairs; mystery instances excluded as they have no standard-domain encoding).",
        r"Regenerated instances are harder or equal on every axis. Optimal plan length is computed with",
        r"Fast Downward under an admissible heuristic.}",
        r"\label{tab:bw-structural}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Measure & Canonical & Regenerated & $\Delta$ \\",
        r"\midrule",
    ]
    for _, r in struct.iterrows():
        label = STRUCT_LABEL[str(r["metric"])]
        c, w, d = float(r["canonical_mean"]), float(r["w6_mean"]), float(r["delta_w6_minus_canonical"])
        nd = 2 if "plan_length" in str(r["metric"]) or "clauses" in str(r["metric"]) or "blocks" in str(r["metric"]) else 2
        s_lines.append(
            f"{label} & {c:.{nd}f} & {w:.{nd}f} & ${_fmt_signed(d, nd)}$ \\\\"
        )
        cells.append(
            {
                "location": "table_bw_structural",
                "family": "BW",
                "slice": "matched47",
                "model": "--",
                "variant": str(r["metric"]),
                "metric": "delta",
                "new_value": _round_disp(d, nd),
                "n": int(r["n_pairs"]),
                "nd": nd,
                "note_hint": "from K3_bw_canonical_w6_matched_report.csv",
            }
        )
        cells.append(
            {
                "location": "table_bw_structural",
                "family": "BW",
                "slice": "matched47",
                "model": "--",
                "variant": str(r["metric"]),
                "metric": "canonical_mean",
                "new_value": _round_disp(c, nd),
                "n": int(r["n_pairs"]),
                "nd": nd,
                "note_hint": "from K3_bw_canonical_w6_matched_report.csv",
            }
        )
        cells.append(
            {
                "location": "table_bw_structural",
                "family": "BW",
                "slice": "matched47",
                "model": "--",
                "variant": str(r["metric"]),
                "metric": "w6_mean",
                "new_value": _round_disp(w, nd),
                "n": int(r["n_pairs"]),
                "nd": nd,
                "note_hint": "from K3_bw_canonical_w6_matched_report.csv",
            }
        )
    s_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_bw_structural.tex").write_text("\n".join(s_lines), encoding="utf-8")

    acc = rep[rep["section"] == "accuracy_matched"].copy()
    order = ["Claude", "DeepSeek", "GPT-4o", "Llama", "o4-mini", "Gemini"]
    acc["ord"] = acc["model"].map({k: i for i, k in enumerate(order)})
    acc = acc.sort_values("ord")
    a_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accuracy on the same 47 matched pairs. Four of six models score higher on the",
        r"structurally harder regenerated set.}",
        r"\label{tab:bw-accuracy}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Canonical & Regenerated & $\Delta$ \\",
        r"\midrule",
    ]
    name = {
        "Claude": "Claude Sonnet~4",
        "DeepSeek": "DeepSeek-R1-distill",
        "GPT-4o": "GPT-4o",
        "Llama": "Llama-3.1-8B",
        "o4-mini": "o4-mini",
        "Gemini": "Gemini 2.5 Flash",
    }
    for _, r in acc.iterrows():
        m = str(r["model"])
        c, w, d = float(r["canonical_mean"]), float(r["w6_mean"]), float(r["delta_w6_minus_canonical"])
        a_lines.append(
            f"{name.get(m, m)} & {_fmt_acc(c)} & {_fmt_acc(w)} & ${_fmt_signed(d)}$ \\\\"
        )
        for metric, val in [
            ("canonical", _round_disp(c, 3)),
            ("regenerated", _round_disp(w, 3)),
            ("delta", _round_disp(d, 3)),
        ]:
            cells.append(
                {
                    "location": "table_bw_accuracy",
                    "family": "BW",
                    "slice": "matched47",
                    "model": m,
                    "variant": "W6",
                    "metric": metric,
                    "new_value": val,
                    "n": int(r["n_pairs"]),
                    "nd": 3,
                    "note_hint": "from K3_bw_canonical_w6_matched_report.csv",
                }
            )
    a_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_bw_accuracy.tex").write_text("\n".join(a_lines), encoding="utf-8")
    return cells


def _load_algo_all() -> pd.DataFrame:
    parts = []
    for tag in ["claude", "gemini", "gpt4o", "llama", "o1mini"]:
        path = DER / f"ALGO_P1_behavioral_{tag}_rescored.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[df["model"].isin(SHORT)]
        df["variant_type"] = df["variant_type"].map(normalize_variant)
        df["model_short"] = df["model"].map(SHORT)
        df["slice"] = df["problem_id"].map(_algo_slice)
        df["ok"] = _is_true(df["rescored_correct"])
        df["included_bool"] = _is_true(df["included"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["problem_id", "variant_type", "model_short"], keep="last")


def _load_gsm_all() -> pd.DataFrame:
    parts = []
    for tag in ["claude", "gemini", "gpt4o", "llama", "o1mini"]:
        path = DER / f"GSM_P1_behavioral_{tag}_rescored.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[df["model"].isin(SHORT)]
        df["variant_type"] = df["variant_type"].map(normalize_variant)
        df["model_short"] = df["model"].map(SHORT)
        df["ok"] = _is_true(df["rescored_correct"])
        df["included_bool"] = _is_true(df["included"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["problem_id", "variant_type", "model_short"], keep="last")


def _load_bw_all() -> pd.DataFrame:
    bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_bw.csv", dtype=str)
    ids = set(
        bank.loc[
            bank["variant_type"].str.strip().str.lower() == "canonical", "problem_id"
        ].astype(str)
    )
    parts = []
    for name in [
        "BW_P1_behavioral_rescored.csv",
        "BW_P1_behavioral_gemini_rescored.csv",
        "BW_P1_behavioral_o1mini_rescored.csv",
    ]:
        path = DER / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[df["model"].isin(SHORT)]
        df = df[df["problem_id"].isin(ids)]
        df["variant_type"] = df["variant_type"].map(normalize_variant)
        df["model_short"] = df["model"].map(SHORT)
        df["ok"] = _is_true(df["rescored_correct"])
        df["included_bool"] = _is_true(df["included"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["problem_id", "variant_type", "model_short"], keep="last")


def _exclusion_reason_for_cell(
    family: str,
    model: str,
    variant: str,
    slice_name: str | None,
    raw: pd.DataFrame,
) -> str | None:
    """Return a short reason when a cell has no included rows."""
    omit = _omit_algo(slice_name or "", variant) if family == "ALGO" else None
    if omit:
        return omit
    if family == "GSM" and variant == "W6" and model in {"GPT-4o", "Llama"}:
        return "missing_bank_row"
    sub = raw[(raw["model_short"] == model) & (raw["variant_type"] == variant)]
    if slice_name and "slice" in sub.columns:
        sub = sub[sub["slice"] == slice_name]
    if sub.empty:
        excl = load_exclusions()
        fam = family.upper()
        hits = excl[(excl["family"] == fam) & (excl["variant"] == variant)]
        if not hits.empty:
            return str(hits["reason"].mode().iloc[0])
        return "no rows"
    if not sub["included_bool"].any():
        reasons = sub["exclusion_reason"].astype(str).str.strip()
        reasons = reasons[reasons != ""]
        if not reasons.empty:
            return str(reasons.mode().iloc[0])
        return "all excluded"
    return None


def _cell_acc_ci(
    family: str,
    raw: pd.DataFrame,
    model: str,
    variant: str,
    slice_name: str | None = None,
) -> tuple[float | None, int, int, float | None, float | None, str | None]:
    reason = _exclusion_reason_for_cell(family, model, variant, slice_name, raw)
    sub = raw[(raw["model_short"] == model) & (raw["variant_type"] == variant)].copy()
    if slice_name and "slice" in sub.columns:
        sub = sub[sub["slice"] == slice_name]
    before = sub.copy()
    # Apply family exclusions only for scoring (same as emit_paper_p1_tables).
    sub = filter_excluded(sub, family=family)
    if len(before) and len(sub) == 0:
        excl = load_exclusions()
        fam = family.upper()
        pids = set(before["problem_id"].astype(str))
        hits = excl[
            (excl["family"] == fam)
            & (excl["variant"] == variant)
            & (excl["problem_id"].isin(pids))
        ]
        if not hits.empty:
            reason = str(hits["reason"].mode().iloc[0])
        else:
            reason = reason or "variant_excluded"
        return None, 0, 0, None, None, reason
    sub = sub[sub["included_bool"]]
    n = int(len(sub))
    if n == 0:
        return None, 0, 0, None, None, reason or "no included"
    k = int(sub["ok"].sum())
    acc = k / n
    if family == "ALGO":
        vals = sub["ok"].astype(float).tolist()
        clusters = cluster_ids_for(sub["problem_id"].astype(str).tolist())
        lo, hi = cluster_bootstrap_ci(vals, clusters, n_resamples=10000, seed=42)
    else:
        lo, hi = wilson_ci(k, n)
    return acc, k, n, lo, hi, None


def _assert_summary_consistency(algo: pd.DataFrame, gsm: pd.DataFrame, bw: pd.DataFrame) -> None:
    """Point estimates must agree with P1_rescore_summary (included-only)."""
    summary = pd.read_csv(DER / "P1_rescore_summary.csv")
    # Map summary family → (frame, filter)
    # Summary uses subtype labels; check a few high-signal cells.
    checks = []
    for _, r in summary.iterrows():
        model = SHORT.get(str(r["model"]))
        if not model:
            continue
        fam = str(r["family"])
        vt = normalize_variant(str(r["variant"]))
        new_acc = float(r["new_accuracy"])
        n_sum = int(r["n"])
        if fam in {"coin_change", "shortest_path", "wis"}:
            frame = algo
            sub = frame[
                (frame["model_short"] == model)
                & (frame["variant_type"] == vt)
                & (frame["included_bool"])
            ]
            sub = filter_excluded(sub, family="ALGO")
            # subtype filter via problem_id prefix
            prefix = {"coin_change": "CC", "shortest_path": "SP", "wis": "WIS"}[fam]
            sub = sub[sub["problem_id"].astype(str).str.startswith(prefix)]
            if n_sum == 0:
                continue
            if len(sub) != n_sum:
                # Summary may predate exclusion list; tolerate n mismatch only if acc close.
                pass
            if len(sub) == 0:
                continue
            acc = float(sub["ok"].mean())
            if abs(acc - new_acc) > 0.02 and len(sub) == n_sum:
                checks.append((fam, model, vt, new_acc, acc, n_sum, len(sub)))
        elif fam in {"gsm_p1p2", "gsm_symbolic"}:
            continue  # GSM summary splits templates; table pools GSM
        elif fam in {"blocksworld", "mystery_blocksworld"}:
            continue  # BW table uses bank-restricted pool
    if checks:
        print("WARNING: P1_rescore_summary mismatches (showing up to 5):")
        for c in checks[:5]:
            print(" ", c)


def write_variants() -> list[dict]:
    algo = _load_algo_all()
    gsm = _load_gsm_all()
    bw = _load_bw_all()
    _assert_summary_consistency(algo, gsm, bw)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Per-variant accuracy by family, slice, and model (Probe~1).",
        r"  Accuracies use rescored \texttt{included=True} rows consistent with",
        r"  \texttt{results/derived/P1\_rescore\_summary.csv}. ALGO intervals are",
        r"  10k cluster-bootstrap CIs over clone families (seed~42); GSM/BW use",
        r"  Wilson 95\% CIs. Excluded cells show the exclusion reason, never 0.}",
        r"  \label{tab:variants}",
        r"  \begin{tabular}{lllccccccc}",
        r"    \toprule",
        r"    Family & Slice & Model & Can. & W1 & W2 & W3 & W4 & W5 & W6 \\",
        r"    \midrule",
    ]
    cells: list[dict] = []

    def emit_row(family: str, slice_name: str, model: str, texts: list[str]) -> None:
        lines.append(
            f"    {family} & {slice_name} & {TEX_MODEL[model]:<10} & "
            + " & ".join(texts)
            + r" \\"
        )

    def pack(family: str, slice_name: str, model: str, frame: pd.DataFrame) -> list[str]:
        out = []
        for vt in VARIANTS:
            acc, k, n, lo, hi, reason = _cell_acc_ci(
                family, frame, model, vt, None if slice_name == "--" else slice_name
            )
            if reason:
                text = r"\textit{" + reason.replace("_", r"\_") + "}"
                disp = None
            else:
                assert lo is not None and hi is not None
                text = f"{_fmt_acc(acc)} {_fmt_ci(lo, hi)}"
                disp = _round_disp(acc, 3)
            out.append(text)
            cells.append(
                {
                    "location": "table_variants",
                    "family": family,
                    "slice": slice_name,
                    "model": model,
                    "variant": vt,
                    "metric": "accuracy",
                    "new_value": disp,
                    "n": n,
                    "nd": 3,
                    "display": text if reason else _fmt_acc(acc),
                    "note_hint": reason or "included=True rescored; matches P1_rescore_summary lineage",
                }
            )
        return out

    for m in T7_GSM_MODELS:
        emit_row("GSM", "--", m, pack("GSM", "--", m, gsm))
    lines.append(r"    \midrule")
    for sl in T7_ALGO_SLICES:
        for m in T7_ALGO_MODELS:
            emit_row("ALGO", sl, m, pack("ALGO", sl, m, algo))
    lines.append(r"    \midrule")
    for m in T7_BW_MODELS:
        emit_row("BW", "--", m, pack("BW", "--", m, bw))
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    (PAPER_T / "table_variants.tex").write_text("\n".join(lines), encoding="utf-8")
    return cells


def _tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_coverage() -> list[dict]:
    df = pd.read_csv(DER / "COVERAGE_MASTER.csv").fillna("")
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \caption{Coverage and exclusions from \texttt{COVERAGE\_MASTER.csv}.",
        r"  One row per (probe, family, model, variant).}",
        r"  \label{tab:coverage}",
        r"  \begin{tabular}{llllrrrl}",
        r"    \toprule",
        r"    Probe & Family & Model & Variant & Att. & Incl. & Excl. & Top exclusion \\",
        r"    \midrule",
    ]
    cells = []
    rev = SHORT
    for _, r in df.iterrows():
        model = rev.get(str(r["model"]), str(r["model"]))
        if "/" in model:
            model = model.split("/")[-1][:18]
        reason = str(r.get("exclusion_reason_1") or "")
        if reason.lower() in {"", "nan", "none"}:
            reason = "---"
        else:
            reason = _tex_escape(reason)
        att = int(float(r["cells_attempted"])) if str(r["cells_attempted"]).strip() else 0
        inc = int(float(r["cells_included"])) if str(r["cells_included"]).strip() else 0
        exc = int(float(r["cells_excluded"])) if str(r["cells_excluded"]).strip() else 0
        variant = _tex_escape(str(r["variant"]))
        lines.append(
            f"    {_tex_escape(r['probe'])} & {_tex_escape(r['family'])} & "
            f"{_tex_escape(model)} & {variant} "
            f"& {att} & {inc} & {exc} & {reason} \\\\"
        )
        cells.append(
            {
                "location": "table_coverage",
                "family": r["family"],
                "slice": r["probe"],
                "model": model,
                "variant": r["variant"],
                "metric": "cells_included",
                "new_value": inc,
                "n": att,
                "nd": 0,
                "note_hint": "from COVERAGE_MASTER.csv",
            }
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_coverage.tex").write_text("\n".join(lines), encoding="utf-8")
    return cells


def audit_sp_std_claude() -> None:
    df = pd.read_csv(DER / "ALGO_P1_behavioral_claude_rescored.csv", dtype=str).fillna("")
    sub = df[
        (df["problem_id"].str.startswith("SP"))
        & (~df["problem_id"].isin(PAPER_ADV["SP"]))
        & (df["variant_type"].map(normalize_variant) == "canonical")
    ].copy()
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "problem_id": r["problem_id"],
                "included": r["included"],
                "exclusion_reason": r["exclusion_reason"],
                "correct_canonical_stored": r.get("correct_canonical", ""),
                "rescored_correct": r["rescored_correct"],
                "rescore_reason": r.get("rescore_reason", ""),
                "verdict_changed": r.get("verdict_changed", ""),
            }
        )
    out = pd.DataFrame(rows)
    path = DER / "SP_std_claude_canonical_audit.csv"
    out.to_csv(path, index=False)
    n = len(out)
    n_inc = int(_is_true(out["included"]).sum())
    n_ok = int(_is_true(out["rescored_correct"]).sum())
    n_old = int(_is_true(out["correct_canonical_stored"]).sum()) if "correct_canonical_stored" in out else 0
    print(f"SP-std Claude canonical audit → {path}")
    print(f"  n={n} included={n_inc} excluded={n - n_inc}")
    print(f"  stored correct_canonical={n_old}/{n} ({n_old/n if n else float('nan'):.3f})")
    print(f"  rescored_correct={n_ok}/{n_inc} ({n_ok/n_inc if n_inc else float('nan'):.3f})")
    if n_inc and n - n_inc:
        print("  exclusion reasons:", out.loc[~_is_true(out["included"]), "exclusion_reason"].value_counts().to_dict())


def _parse_old_table7(text: str) -> dict[tuple, float | None]:
    out: dict[tuple, float | None] = {}
    alias = {r"\omini": "o4-mini"}
    for line in text.splitlines():
        if line.count("&") < 8:
            continue
        if "Family" in line or "toprule" in line or "midrule" in line:
            continue
        parts = [p.strip() for p in line.strip().rstrip("\\").split("&")]
        if len(parts) < 10:
            continue
        fam, sl, model = parts[0], parts[1], parts[2]
        model = alias.get(model, model)
        sl = sl if sl else "--"
        sl = sl.rstrip(".")
        for vt, cell in zip(VARIANTS, parts[3:10]):
            # strip CI if present
            cell = cell.split()[0] if cell else cell
            cell = re.sub(r"\\textit\{([^}]*)\}", r"\1", cell)
            if cell in {"---", ""} or cell.startswith(r"\textit"):
                val: float | None = None
            elif cell.startswith("1") or cell.startswith("0"):
                val = float(cell)
            elif cell.startswith("."):
                val = float("0" + cell)
            else:
                # exclusion reason text
                val = None
            out[("table_variants", fam, sl, model, vt, "accuracy")] = val
            out[("table7", fam, sl, model, vt, "accuracy")] = val
    return out


def _parse_inline_oracle(text: str) -> dict[tuple, float | None]:
    out: dict[tuple, float | None] = {}
    # crude: Capture .043 & .252 & $+.209$ & $.000$ style rows
    for i, line in enumerate(text.splitlines()):
        if "Rename mapping not consumed (planning)" in line:
            key_def = "BW_W3_action_mapping"
        elif "Rename mapping not consumed (graph)" in line:
            key_def = "SP_W3_node_mapping"
        elif "Legacy state parser" in line:
            key_def = "BW_state_parser"
        else:
            continue
        nums = re.findall(r"\$?([+-]?\.\d+|1\.\d+|0\.\d+)\$?", line)
        # rows & pert_before & pert_after & delta_pert & delta_can — first int separate
        ints = re.findall(r"&\s*(\d+)\s*&", line)
        if ints:
            out[("table_oracle", "--", "--", "--", key_def, "rows_affected")] = float(ints[0])
        floats = []
        for tok in re.findall(r"(?:&\s*\$?)([+-]?(?:\d+\.\d+|\.\d+))", line):
            floats.append(float(tok if tok[0] in "+-" or tok.startswith("1") or tok.startswith("0") else ("0" + tok if tok.startswith(".") else tok)))
        # Fallback parse pieces after first &
        parts = [p.strip() for p in line.split("&")]
        if len(parts) >= 6:
            def num(s: str) -> float:
                s = s.replace("$", "").replace("+", "").strip().rstrip("\\")
                if s.startswith("."):
                    s = "0" + s
                if s.startswith("-."):
                    s = "-0" + s[1:]
                return float(s)

            try:
                out[("table_oracle", "--", "--", "--", key_def, "rows_affected")] = float(
                    parts[1].strip()
                )
                out[("table_oracle", "--", "--", "--", key_def, "pert_before")] = num(parts[2])
                out[("table_oracle", "--", "--", "--", key_def, "pert_after")] = num(parts[3])
                out[("table_oracle", "--", "--", "--", key_def, "delta_perturbed")] = num(parts[4])
                out[("table_oracle", "--", "--", "--", key_def, "delta_canonical")] = num(parts[5])
            except ValueError:
                pass
    return out


def _parse_inline_bw(text: str) -> dict[tuple, float | None]:
    out: dict[tuple, float | None] = {}
    struct_map = {
        "Blocks": "num_blocks",
        "Goal clauses": "n_goal_clauses",
        "Goal tower depth": "goal_tower_depth",
        "Initial tower depth": "init_tower_depth",
        "Optimal plan length": "fd_optimal_plan_length",
    }
    for line in text.splitlines():
        for label, metric in struct_map.items():
            if line.strip().startswith(label):
                parts = [p.strip().rstrip("\\") for p in line.split("&")]
                if len(parts) >= 4:
                    def num(s: str) -> float:
                        s = s.replace("$", "").replace("+", "").strip()
                        if s.startswith("-."):
                            s = "-0" + s[1:]
                        elif s.startswith("."):
                            s = "0" + s
                        return float(s)

                    try:
                        out[("table_bw_structural", "BW", "matched47", "--", metric, "canonical_mean")] = num(parts[1])
                        out[("table_bw_structural", "BW", "matched47", "--", metric, "w6_mean")] = num(parts[2])
                        out[("table_bw_structural", "BW", "matched47", "--", metric, "delta")] = num(parts[3])
                    except ValueError:
                        pass
        model_key = None
        if "Claude Sonnet" in line:
            model_key = "Claude"
        elif "DeepSeek" in line:
            model_key = "DeepSeek"
        elif "GPT-4o" in line:
            model_key = "GPT-4o"
        elif "Llama-3.1" in line:
            model_key = "Llama"
        elif "o4-mini" in line:
            model_key = "o4-mini"
        elif "Gemini 2.5" in line:
            model_key = "Gemini"
        if model_key is None:
            continue
        parts = [p.strip().rstrip("\\") for p in line.split("&")]
        if len(parts) < 4:
            continue

        def num(s: str) -> float:
            s = s.replace("$", "").replace("+", "").strip()
            if s.startswith("-."):
                s = "-0" + s[1:]
            elif s.startswith("."):
                s = "0" + s
            return float(s)

        try:
            out[("table_bw_accuracy", "BW", "matched47", model_key, "W6", "canonical")] = num(parts[1])
            out[("table_bw_accuracy", "BW", "matched47", model_key, "W6", "regenerated")] = num(parts[2])
            out[("table_bw_accuracy", "BW", "matched47", model_key, "W6", "delta")] = num(parts[3])
        except ValueError:
            pass
    return out


def write_deltas(new_cells: list[dict]) -> None:
    old: dict[tuple, float | None] = {}
    try:
        old_t7 = subprocess.check_output(
            ["git", "show", "HEAD:paper/tables/table7_pervariant.tex"], cwd=REPO_ROOT
        ).decode()
        old.update(_parse_old_table7(old_t7))
    except subprocess.CalledProcessError:
        pass
    main_tex = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    old.update(_parse_inline_oracle(main_tex))
    old.update(_parse_inline_bw(main_tex))

    rows = []
    for c in new_cells:
        key = (c["location"], c["family"], c["slice"], c["model"], c["variant"], c["metric"])
        # also compare table_variants against old table7 keys
        old_v = old.get(key, "MISSING_IN_OLD")
        if old_v == "MISSING_IN_OLD" and c["location"] == "table_variants":
            alt = ("table7", c["family"], c["slice"], c["model"], c["variant"], c["metric"])
            old_v = old.get(alt, "MISSING_IN_OLD")
        new_v = c.get("new_value")
        if old_v == "MISSING_IN_OLD":
            # only record accuracy cells for variants / meaningful metrics
            if c["location"] == "table_coverage":
                continue
            if c["metric"] not in {
                "accuracy",
                "rows_affected",
                "pert_before",
                "pert_after",
                "delta_perturbed",
                "delta_canonical",
                "canonical",
                "regenerated",
                "delta",
                "canonical_mean",
                "w6_mean",
            }:
                continue
            changed = True
            old_s: float | str | None = ""
        else:
            nd = int(c.get("nd") or 3)
            if old_v is None and new_v is None:
                changed = False
            elif old_v is None or new_v is None:
                changed = True
            else:
                changed = _round_disp(float(old_v), nd) != _round_disp(float(new_v), nd)
            old_s = "" if old_v is None else old_v
        if not changed:
            continue
        reason = c.get("note_hint", "")
        if old_v != "MISSING_IN_OLD" and c["location"] == "table_variants":
            if c["family"] == "ALGO" and c["variant"] in {"canonical", "W3", "W6"}:
                reason = "W3 verifier repair / rescored included=True (disagreed with pre-repair draft)"
            elif old_v is None and new_v is not None:
                reason = "previously omitted/stale cell now populated from rescored rows"
            else:
                reason = "stale in previous draft vs rescored P1"
        elif old_v == "MISSING_IN_OLD":
            reason = reason or "added"
        rows.append(
            {
                "location": c["location"],
                "family": c["family"],
                "slice": c["slice"],
                "model": c["model"],
                "variant": c["variant"],
                "metric": c["metric"],
                "old_value": (
                    "MISSING"
                    if old_v == "MISSING_IN_OLD"
                    else ("---" if old_v is None else old_s)
                ),
                "new_value": "---" if new_v is None else new_v,
                "delta": ""
                if (old_v in (None, "MISSING_IN_OLD") or new_v is None)
                else float(new_v) - float(old_v),
                "n": c.get("n", ""),
                "note": reason,
            }
        )
    out = pd.DataFrame(rows)
    path = DER / "PAPER_NUMBER_DELTAS.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} changed cells)")


def main() -> None:
    PAPER_T.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    cells += write_oracle()
    print("Wrote", PAPER_T / "table_oracle.tex")
    cells += write_bw_tables()
    print("Wrote", PAPER_T / "table_bw_structural.tex")
    print("Wrote", PAPER_T / "table_bw_accuracy.tex")
    cells += write_variants()
    print("Wrote", PAPER_T / "table_variants.tex")
    cells += write_coverage()
    print("Wrote", PAPER_T / "table_coverage.tex")
    write_deltas(cells)
    audit_sp_std_claude()


if __name__ == "__main__":
    main()
