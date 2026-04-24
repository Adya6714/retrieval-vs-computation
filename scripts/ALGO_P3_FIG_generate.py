#!/usr/bin/env python3
"""Generate ALGO Probe 3 figures (red/coral family)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

RESULTS_DIR = Path("results")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

P3_PRIMARY = "#B91C1C"
P3_ACCENT = "#EF4444"
GREY = "#9CA3AF"

MODEL_ORDER = [
    "anthropic/claude-3.7-sonnet",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
]
MODEL_LABEL = {
    "anthropic/claude-3.7-sonnet": "Claude 3.7",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
}
SUBTYPE_ORDER = ["coin_change", "shortest_path", "wis"]
SUBTYPE_LABEL = {"coin_change": "CC", "shortest_path": "SP", "wis": "WIS"}


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_triangulation() -> pd.DataFrame:
    p = RESULTS_DIR / "ALGO_P3_RES_triangulation.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p, dtype=str).fillna("")
    d["VAR_canonical"] = pd.to_numeric(d["VAR_canonical"], errors="coerce")
    d["instance_contamination_score"] = pd.to_numeric(d["instance_contamination_score"], errors="coerce")
    d["greedy_succeeds_num"] = pd.to_numeric(d["greedy_succeeds"], errors="coerce")
    d["problem_subtype"] = d["problem_subtype"].str.strip().str.lower()
    return d


def _parse_bw_regression_files() -> list[dict]:
    files = [
        ("BW-Claude", RESULTS_DIR / "BW_P3_RES_contamination_regression_claude37.txt"),
        ("BW-GPT4o", RESULTS_DIR / "BW_P3_RES_contamination_regression_gpt4o.txt"),
        ("BW-Llama", RESULTS_DIR / "BW_P3_RES_contamination_regression_llama8b.txt"),
    ]
    rows = []
    pat = re.compile(r"contamination_score\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)")
    for label, path in files:
        if not path.exists():
            continue
        txt = path.read_text(encoding="utf-8")
        m = pat.search(txt)
        if not m:
            continue
        coef = float(m.group(1))
        se = float(m.group(2))
        rows.append({"label": label, "coef": coef, "lo": coef - 1.96 * se, "hi": coef + 1.96 * se, "family": "BW"})
    return rows


def _bootstrap_coef(reg: pd.DataFrame, n_boot: int = 10000) -> tuple[float, float, float]:
    fit = smf.ols("VAR_canonical ~ instance_contamination_score + greedy_succeeds_num", data=reg).fit()
    coef = float(fit.params.get("instance_contamination_score", np.nan))
    n = len(reg)
    samples = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        b = reg.iloc[idx].reset_index(drop=True)
        try:
            bf = smf.ols("VAR_canonical ~ instance_contamination_score + greedy_succeeds_num", data=b).fit()
            samples.append(float(bf.params.get("instance_contamination_score", np.nan)))
        except Exception:
            continue
    if not samples:
        return coef, np.nan, np.nan
    return coef, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def plot_regression_forest(tri: pd.DataFrame) -> None:
    rows = _parse_bw_regression_files()
    for model in MODEL_ORDER:
        for subtype in SUBTYPE_ORDER:
            sub = tri[(tri["model"] == model) & (tri["problem_subtype"] == subtype)][
                ["VAR_canonical", "instance_contamination_score", "greedy_succeeds_num"]
            ].dropna()
            if len(sub) < 5 or sub["instance_contamination_score"].nunique() < 2:
                rows.append(
                    {
                        "label": f"{SUBTYPE_LABEL[subtype]}-{MODEL_LABEL[model]}",
                        "coef": np.nan,
                        "lo": np.nan,
                        "hi": np.nan,
                        "family": "ALGO",
                    }
                )
                continue
            coef, lo, hi = _bootstrap_coef(sub, n_boot=10000)
            rows.append(
                {
                    "label": f"{SUBTYPE_LABEL[subtype]}-{MODEL_LABEL[model]}",
                    "coef": coef,
                    "lo": lo,
                    "hi": hi,
                    "family": "ALGO",
                }
            )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["coef"], how="all")
    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.38)))
    y = np.arange(len(df))
    colors = [GREY if f == "BW" else P3_PRIMARY for f in df["family"]]
    ax.scatter(df["coef"], y, color=colors, s=45, zorder=3)
    for yi, lo, hi, c in zip(y, df["lo"], df["hi"], colors):
        if pd.notna(lo) and pd.notna(hi):
            ax.hlines(yi, lo, hi, color=c, linewidth=2, alpha=0.9)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Coefficient on instance_contamination_score")
    ax.set_title("Figure 8 — Regression coefficient forest plot (95% bootstrap CI)")
    _save(fig, "ALGO_P3_FIG_08_regression_forest")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9})
    tri = _load_triangulation()
    plot_regression_forest(tri)
    print("Generated ALGO Probe 3 figures in results/figures/")


if __name__ == "__main__":
    main()
