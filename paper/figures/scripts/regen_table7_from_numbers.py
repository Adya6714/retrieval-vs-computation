#!/usr/bin/env python3
"""Regenerate paper/tables/table7_pervariant.tex from rebuild/NUMBERS.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NUMBERS = ROOT / "rebuild" / "NUMBERS.csv"
OUT = ROOT / "paper" / "tables" / "table7_pervariant.tex"

MODELS = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini"]
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
VAR_HEADERS = ["Can.", "W1", "W2", "W3", "W4", "W5", "W6"]


def fmt(v) -> str:
    if pd.isna(v) or str(v) == "NOT_COMPUTABLE":
        return "---"
    try:
        return f"{float(v):.3f}".lstrip("0") if float(v) < 1 else f"{float(v):.3f}"
    except Exception:
        return "---"


def fmt_acc(v) -> str:
    if pd.isna(v) or str(v) == "NOT_COMPUTABLE":
        return "---"
    x = float(v)
    s = f"{x:.3f}"
    if s.startswith("0"):
        s = s[1:]
    return s


def cell(df: pd.DataFrame, family: str, slice_: str, model: str, variant: str) -> str:
    q = (
        (df["family"] == family)
        & (df["model"] == model)
        & (df["variant"] == variant)
        & (df["metric"] == "accuracy")
    )
    if slice_ and "slice" in df.columns:
        q &= df["slice"].fillna("--") == slice_
    elif slice_ == "--" and "slice" in df.columns:
        q &= df["slice"].fillna("--").isin(["--", "", family]) | (df["slice"].isna())
    rows = df[q]
    if rows.empty and "subtype" in df.columns:
        q2 = (
            (df["family"] == family)
            & (df["model"] == model)
            & (df["variant"] == variant)
            & (df["metric"] == "accuracy")
        )
        if slice_ not in (None, "--", ""):
            q2 &= df["subtype"].fillna("") == slice_
        rows = df[q2]
    if rows.empty:
        # try id pattern
        sid = slice_ if slice_ not in (None, "--") else ""
        pat = f"P1.1.{family}."
        if sid:
            pat += f"{sid}."
        pat += f"{model}.{variant}"
        rows = df[df["id"] == pat]
    if rows.empty:
        return "---"
    return fmt_acc(rows.iloc[0]["value"])


def main() -> None:
    df = pd.read_csv(NUMBERS)
    # Normalize column names if needed
    cols = {c.lower(): c for c in df.columns}
    # Keep as-is; NUMBERS uses lowercase-ish names from rebuild

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append("  \\setlength{\\tabcolsep}{4pt}")
    lines.append(
        "  \\caption{Per-variant accuracy under frozen rebuild filters "
        "(\\texttt{rebuild/NUMBERS.csv}, \\texttt{rebuild/FROZEN\\_FILTERS.md}). "
        "GSM bank-valid $n{=}44$ (GPT-4o/Llama Probe-1 currently $n{=}20$ on "
        "GSM\\_001--020; W6 for those models is NOT\\_COMPUTABLE after bank+ERROR "
        "filter). ALGO slices use the frozen adversarial/standard split "
        "(SP+CC+WIS challenging $n{=}61$). BW bank $n{=}65$ PlanBench IDs. "
        "Dashes mark absent variants or non-computable cells (API/credit gaps "
        "or design holes), not zero accuracy.}"
    )
    lines.append("  \\label{tab:pervariant}")
    lines.append("  \\begin{tabular}{lllccccccc}")
    lines.append("    \\toprule")
    lines.append("    Family & Slice & Model & Can. & W1 & W2 & W3 & W4 & W5 & W6 \\\\")
    lines.append("    \\midrule")

    # Discover available (family, slice) from ids
    p1 = df[df["id"].astype(str).str.startswith("P1.1.")].copy()

    def lookup(family: str, slice_name: str, model: str, variant: str) -> str:
        if slice_name in (None, "", "--"):
            iid = f"P1.1.{family}.{model}.{variant}"
        else:
            iid = f"P1.1.{family}.{slice_name}.{model}.{variant}"
        rows = p1[p1["id"] == iid]
        if rows.empty:
            return "---"
        return fmt_acc(rows.iloc[0]["value"])

    # GSM
    for model in MODELS:
        vals = [lookup("GSM", "--", model, v) for v in VARIANTS]
        # GSM ids are P1.1.GSM.Model.variant without slice
        vals = []
        for v in VARIANTS:
            rows = p1[p1["id"] == f"P1.1.GSM.{model}.{v}"]
            vals.append(fmt_acc(rows.iloc[0]["value"]) if len(rows) else "---")
        model_tex = "\\omini" if model == "o4-mini" else model
        lines.append(
            "    GSM & -- & "
            + model_tex.ljust(10)
            + " & "
            + " & ".join(vals)
            + " \\\\"
        )
    lines.append("    \\midrule")

    algo_slices = [
        "CC-chall",
        "CC-std",
        "SP-chall",
        "SP-std",
        "WIS-chall",
        "WIS-std",
    ]
    for sl in algo_slices:
        for model in ["Claude", "GPT-4o", "Gemini", "Llama"]:
            vals = []
            for v in VARIANTS:
                rows = p1[p1["id"] == f"P1.1.ALGO.{sl}.{model}.{v}"]
                vals.append(fmt_acc(rows.iloc[0]["value"]) if len(rows) else "---")
            lines.append(
                f"    ALGO & {sl} & {model.ljust(10)} & "
                + " & ".join(vals)
                + " \\\\"
            )
    lines.append("    \\midrule")

    for model in MODELS:
        vals = []
        for v in VARIANTS:
            rows = p1[p1["id"] == f"P1.1.BW.--.{model}.{v}"]
            if rows.empty:
                rows = p1[p1["id"] == f"P1.1.BW.{model}.{v}"]
            vals.append(fmt_acc(rows.iloc[0]["value"]) if len(rows) else "---")
        model_tex = "\\omini" if model == "o4-mini" else model
        lines.append(
            "    BW & -- & "
            + model_tex.ljust(10)
            + " & "
            + " & ".join(vals)
            + " \\\\"
        )

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    OUT.write_text("\n".join(lines) + "\n")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
