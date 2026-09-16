#!/usr/bin/env python3
"""Emit additional NeurIPS paper tables (N1, cleaned coverage, Q5 results)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DER = REPO_ROOT / "results" / "derived"
PAPER_T = REPO_ROOT / "paper" / "tables"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
    "openai/o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b",
}
SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "---"
    x = round(float(x) + 1e-12, nd)
    if abs(x) >= 1:
        return f"{x:.{nd}f}"
    s = f"{x:.{nd}f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") or s.startswith("-0.") else s


def _signed(x: float, nd: int = 3) -> str:
    if abs(x) < 5 * 10 ** (-(nd + 1)):
        return f".{'0'*nd}"
    sign = "+" if x > 0 else ""
    body = _fmt(abs(x), nd)
    return f"{sign}{body}" if x > 0 else f"-{body}"


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


def write_n1() -> None:
    n1 = pd.read_csv(DER / "N1_bw_w6_stratified_accuracy.csv")
    matched = n1[n1["subset"] == "matched_naming_convention"].copy()
    twobytwo = n1[n1["subset"] == "naming_x_variant_2x2"].copy()
    order = ["Claude", "DeepSeek", "GPT-4o", "Llama", "o4-mini", "Gemini"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Blocksworld canonical vs regenerated accuracy stratified by block-naming",
        r"convention (\texttt{N1\_bw\_w6\_stratified\_accuracy.csv}). Top: pairs matched on",
        r"naming ($n{=}13$). Bottom: full naming $\times$ variant $2\times 2$ with per-cell $n$.",
        r"The sequential-canonical cell is $n{=}3$ and is reported only as a confound check,",
        r"not as an effect estimate. Unmatched Claude $\Delta{=}+0.468$ shrinks to within-stratum",
        r"$+0.090$ (sequential) and $+0.064$ (scattered).}",
        r"\label{tab:bw-stratified}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\multicolumn{5}{l}{\emph{Matched naming convention}} \\",
        r"Model & $n$ & Canonical & Regenerated & $\Delta$ \\",
        r"\midrule",
    ]
    matched["ord"] = matched["model"].map({m: i for i, m in enumerate(order)})
    for _, r in matched.sort_values("ord").iterrows():
        lines.append(
            f"{r['model']} & {int(r['n_pairs'])} & {_fmt(r['canonical_accuracy'])} "
            f"& {_fmt(r['w6_accuracy'])} & ${_signed(float(r['delta_w6_minus_canonical']))}$ \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Naming $\times$ variant $2\times 2$ (accuracy, $n$)}} \\",
        r"Model & Seq./can. & Seq./W6 & Scat./can. & Scat./W6 \\",
        r"\midrule",
    ]
    for m in order:
        cells = {}
        for _, r in twobytwo[twobytwo["model"] == m].iterrows():
            key = (str(r["naming"]), str(r["variant"]))
            cells[key] = (float(r["accuracy"]), int(r["n_pairs"]))
        def cell(naming: str, variant: str) -> str:
            a, n = cells[(naming, variant)]
            return f"{_fmt(a)} ($n{{=}}{n}$)"
        lines.append(
            f"{m} & {cell('sequential','canonical')} & {cell('sequential','W6')} "
            f"& {cell('scattered','canonical')} & {cell('scattered','W6')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_n1_stratified.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote table_n1_stratified.tex")


def write_coverage() -> None:
    df = pd.read_csv(DER / "COVERAGE_MASTER.csv").fillna("")
    # Drop mock fixtures and non-model names.
    mock_mask = (
        df["model"].astype(str).str.contains(r"mock|answer is|MOCK", case=False, na=False)
        | df["variant"].astype(str).str.fullmatch(r"MOCK", case=False, na=False)
        | df["exclusion_reason_1"].astype(str).str.fullmatch(r"mock_row", case=False, na=False)
        | df["exclusion_reason_breakdown"].astype(str).str.contains("mock_row", case=False, na=False)
    )
    df = df[~mock_mask].copy()
    # Keep only known paper models (plus short names already mapped).
    def keep_model(m: str) -> bool:
        m = str(m).strip()
        if m in PAPER_MODELS or m in SHORT.values():
            return True
        if m in SHORT:
            return True
        # allow already-shortened provider suffixes only if in SHORT values
        return False

    df = df[df["model"].map(keep_model)].copy()
    lines = [
        r"\begin{longtable}{llllrrrl}",
        r"\caption{Coverage and exclusions from \texttt{COVERAGE\_MASTER.csv}",
        r"(mock\_row fixtures and non-model names removed). One row per",
        r"(probe, family, model, variant).}",
        r"\label{tab:coverage}\\",
        r"\toprule",
        r"Probe & Family & Model & Variant & Att. & Incl. & Excl. & Top exclusion \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Probe & Family & Model & Variant & Att. & Incl. & Excl. & Top exclusion \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{\emph{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r in df.iterrows():
        model = SHORT.get(str(r["model"]), str(r["model"]))
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
        lines.append(
            f"{_tex_escape(r['probe'])} & {_tex_escape(r['family'])} & "
            f"{_tex_escape(model)} & {_tex_escape(r['variant'])} "
            f"& {att} & {inc} & {exc} & {reason} \\\\"
        )
    lines += [r"\end{longtable}", ""]
    (PAPER_T / "table_coverage.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote table_coverage.tex ({len(df)} rows after mock filter)")


def write_q5_tables() -> None:
    # ALGO CCI
    a = pd.read_csv(DER / "ALGO_P2_cci.csv")
    a["cci"] = pd.to_numeric(a["cci_score"], errors="coerce")
    a["model_short"] = a["model"].map(SHORT)
    rows = []
    for m, g in a.groupby("model_short"):
        valid = g["cci"].dropna()
        rows.append((m, float(valid.mean()) if len(valid) else float("nan"), int(len(valid)), int(len(g))))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{ALGO Probe~2 CCI (\texttt{ALGO\_P2\_cci.csv}). Gemini has no usable",
        r"CCI scores in this file.}",
        r"\label{tab:algo-cci}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Mean CCI & $n$ scored & $n$ rows \\",
        r"\midrule",
    ]
    for m, mean, ns, nr in sorted(rows, key=lambda x: str(x[0])):
        lines.append(f"{m} & {_fmt(mean)} & {ns} & {nr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_algo_cci.tex").write_text("\n".join(lines), encoding="utf-8")

    # O10 primary variance components (ALGO max_cells + GSM full_7)
    vc = pd.read_csv(DER / "O10_variance_components.csv")
    vc = vc[vc["is_primary"] == True]  # noqa: E712
    gcoef = pd.read_csv(DER / "O10_generalizability_coefficients.csv")
    gcoef = gcoef[gcoef["is_primary"] == True]  # noqa: E712
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Primary G-theory variance components",
        r"(\texttt{O10\_variance\_components.csv}) and coefficients",
        r"(\texttt{O10\_generalizability\_coefficients.csv}).}",
        r"\label{tab:o10}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family & Design & Prop.\ model & Prop.\ item & $G_{\mathrm{item}}$ & $\phi_{\mathrm{item}}$ \\",
        r"\midrule",
    ]
    for _, r in gcoef.iterrows():
        fam, design = r["family"], r["design"]
        sub = vc[(vc["family"] == fam) & (vc["design"] == design)]
        prop_m = float(sub.loc[sub["component"] == "model", "proportion"].iloc[0]) if (sub["component"] == "model").any() else float("nan")
        prop_i = float(r["prop_item"])
        lines.append(
            f"{fam} & {_tex_escape(design)} & {_fmt(prop_m)} & {_fmt(prop_i)} "
            f"& {_fmt(float(r['G_item_over_variant_model']))} & {_fmt(float(r['phi_item_absolute']))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_o10.tex").write_text("\n".join(lines), encoding="utf-8")

    # C1 intrusion W3
    c1 = pd.read_csv(DER / "C1_intrusion_rates.csv")
    c1w3 = c1[c1["variant"] == "W3"].copy()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Canonical-token intrusion among W3 errors",
        r"(\texttt{C1\_intrusion\_rates.csv}).}",
        r"\label{tab:c1-intrusion}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Family & Model & $n$ errors & Intrusion rate & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in c1w3.iterrows():
        lines.append(
            f"{r['family']} & {r['model']} & {int(r['n_errors'])} & {_fmt(float(r['intrusion_rate']))} "
            f"& [{_fmt(float(r['ci_low']),2)},{_fmt(float(r['ci_high']),2)}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_c1_intrusion.tex").write_text("\n".join(lines), encoding="utf-8")

    # C8 null summary
    null = pd.read_csv(DER / "C8_null_expectation.csv")
    rel = null[null["analysis"] == "rate_reliable_crossovers"].copy()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Reliable dissociation crossovers vs additive null",
        r"(\texttt{C8\_null\_expectation.csv}; pairs in",
        r"\texttt{C8\_dissociation\_crossovers.csv}).}",
        r"\label{tab:c8}",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Family & Observed & Null mean & $p(\ge\mathrm{obs})$ & Verdict \\",
        r"\midrule",
    ]
    for _, r in rel.iterrows():
        lines.append(
            f"{r['family']} & {int(r['observed_crossovers'])} & {_fmt(float(r['null_mean']),2)} "
            f"& {_fmt(float(r['null_p_ge_obs']),3)} & {_tex_escape(r['verdict'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_c8.tex").write_text("\n".join(lines), encoding="utf-8")

    # Variant ordering
    vo = pd.read_csv(DER / "P1_variant_ordering.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Stability of W1--W6 difficulty ordering",
        r"(\texttt{P1\_variant\_ordering.csv}; Kendall $W$, 10k permutations).}",
        r"\label{tab:variant-order}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Analysis & Scope & $W$ & $p$ & $n$ rankers \\",
        r"\midrule",
    ]
    for _, r in vo.iterrows():
        scope = r["family"] if str(r["family"]).strip() not in {"", "--"} else SHORT.get(str(r["model"]), str(r["model"]))
        lines.append(
            f"{_tex_escape(r['analysis'])} & {_tex_escape(scope)} & {_fmt(float(r['kendall_W']))} "
            f"& {_fmt(float(r['p_value']),3)} & {int(r['n_rankers'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_variant_order.tex").write_text("\n".join(lines), encoding="utf-8")

    # N5 GSM cells
    n5 = pd.read_csv(DER / "N5_contamination_vs_retention.csv")
    n5 = n5[n5["family"] == "GSM"].copy()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{GSM Infini-gram proximity vs W3 retention",
        r"(\texttt{N5\_contamination\_vs\_retention.csv}; canonical-correct only).",
        r"All GSM intervals span zero.}",
        r"\label{tab:n5}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & $n$ & $\rho$ & 95\% CI & $p$ \\",
        r"\midrule",
    ]
    for _, r in n5.iterrows():
        lines.append(
            f"{r['model']} & {int(r['n'])} & {_fmt(float(r['spearman_rho']))} "
            f"& [{_fmt(float(r['ci_low']))},{_fmt(float(r['ci_high']))}] "
            f"& {_fmt(float(r['p_value']),3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_n5.tex").write_text("\n".join(lines), encoding="utf-8")

    # N3 not estimable
    n3 = pd.read_csv(DER / "N3_algo_mech_behavior_link.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Mechanistic rank-shift vs W3 correctness",
        r"(\texttt{N3\_algo\_mech\_behavior\_link.csv}). Outcome degeneracy makes",
        r"$\rho$ not estimable.}",
        r"\label{tab:n3}",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Model & $n$ & W3-correct & Status \\",
        r"\midrule",
    ]
    for _, r in n3.iterrows():
        lines.append(
            f"{_tex_escape(r['model'])} & {int(r['n'])} & {int(r['n_w3_correct'])}/{int(r['n'])} "
            f"& NOT ESTIMABLE \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_n3.tex").write_text("\n".join(lines), encoding="utf-8")

    # C2 IRT
    c2 = pd.read_csv(DER / "C2_irt_fit_comparison.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Mixture IRT / Rasch model comparison",
        r"(\texttt{C2\_irt\_fit\_comparison.csv}). Strategy posteriors in",
        r"\texttt{strategy\_posteriors.csv} ($n{=}72$ GSM rows, $K{=}1$).}",
        r"\label{tab:c2-irt}",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Family & Model & $K$ & BIC & Selected & Note \\",
        r"\midrule",
    ]
    for _, r in c2.iterrows():
        k = "---" if pd.isna(r["K"]) else str(int(r["K"]))
        bic = "---" if pd.isna(r["bic"]) else _fmt(float(r["bic"]), 1)
        sel = "yes" if bool(r["selected"]) else "no"
        note = _tex_escape(str(r["note"])[:60]).replace("∈", r"$\in$").replace("—", "---")
        lines.append(
            f"{r['family']} & {_tex_escape(r['model'])} & {k} & {bic} & {sel} & {note} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_c2_irt.tex").write_text("\n".join(lines), encoding="utf-8")

    # P2/P1 convergence summary table
    conv = pd.read_csv(DER / "P2_P1_convergence.csv")
    conv = conv[conv["analysis"] == "pointbiserial_cci_w3_correct"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{CCI vs rename survival point-biserial",
        r"(\texttt{P2\_P1\_convergence.csv}). CIs and $p$ from the same",
        r"cluster bootstrap (seed~42, 5k resamples).}",
        r"\label{tab:p2p1}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Scope & $r$ & 95\% CI & $p$ & $n$ \\",
        r"\midrule",
    ]
    for _, r in conv.iterrows():
        lines.append(
            f"{_tex_escape(r['scope'])} & {_fmt(float(r['statistic']))} "
            f"& [{_fmt(float(r['ci_low']))},{_fmt(float(r['ci_high']))}] "
            f"& {_fmt(float(r['p_value']),3)} & {int(r['n'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER_T / "table_p2p1.tex").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote Q5 tables")


def patch_algo_w5_exclusions() -> None:
    """Rewrite ALGO SP W5 cells with heavy parse_failed exclusions."""
    path = PAPER_T / "table_variants.tex"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    # Map: for SP-chall/SP-std rows, replace W5 accuracy for GPT-4o/Gemini with parse_failed.
    # Also ensure we document o4-mini if present.
    # Compute inclusion from rescored files.
    reasons = {}
    for tag in ["claude", "gemini", "gpt4o", "llama", "o1mini"]:
        df = pd.read_csv(DER / f"ALGO_P1_behavioral_{tag}_rescored.csv", dtype=str).fillna("")
        w5 = df[(df["variant_type"].str.upper() == "W5") & (df["problem_id"].str.startswith("SP"))]
        for m, g in w5.groupby("model"):
            if m not in SHORT:
                continue
            n = len(g)
            inc = int((g["included"].str.lower() == "true").sum())
            top = g["exclusion_reason"].value_counts()
            # prefer non-ok reason
            non_ok = top.drop(labels=[x for x in top.index if x == "in_bank_ok"], errors="ignore")
            reason = str(non_ok.index[0]) if len(non_ok) else "in_bank_ok"
            reasons[SHORT[m]] = (inc, n, reason)
    out_lines = []
    for line in text.splitlines():
        if "ALGO & SP-" in line and ("GPT-4o" in line or "Gemini" in line or r"\omini" in line):
            model = "GPT-4o" if "GPT-4o" in line else ("Gemini" if "Gemini" in line else "o4-mini")
            inc, n, reason = reasons.get(model, (0, 0, "parse_failed"))
            if inc < n and reason != "in_bank_ok":
                parts = [p.strip() for p in line.rstrip("\\").split("&")]
                # Family Slice Model Can W1 W2 W3 W4 W5 W6
                if len(parts) >= 10:
                    parts[8] = r"\textit{" + _tex_escape(f"{reason} ({inc}/{n})") + "}"
                    line = "    " + " & ".join(parts) + r" \\"
        out_lines.append(line)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print("Patched ALGO SP W5 exclusion cells:", {k: v for k, v in reasons.items() if v[0] < v[1]})


def main() -> None:
    PAPER_T.mkdir(parents=True, exist_ok=True)
    write_n1()
    write_coverage()
    write_q5_tables()
    patch_algo_w5_exclusions()


if __name__ == "__main__":
    main()
