#!/usr/bin/env python3
"""Generate ALGO Probe 2 figures (purple family) plus Step-K figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

RESULTS_DIR = Path("results")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

P2_PRIMARY = "#6D28D9"
P2_ACCENT = "#8B5CF6"
P2_LIGHT = "#EDE9FE"

RED = "#EF4444"
GREEN = "#10B981"
BLUE = "#3B82F6"
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


def _load_phase2_normal() -> pd.DataFrame:
    p = RESULTS_DIR / "ALGO_P2_RES_phase2_normal.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p, dtype=str).fillna("")
    for c in ["step_index"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _load_phase2_injected() -> pd.DataFrame:
    p = RESULTS_DIR / "ALGO_P2_RES_phase2_injected.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p, dtype=str).fillna("")
    for c in ["step_index", "critical_step_index"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def _load_phase1() -> pd.DataFrame:
    paths = [
        RESULTS_DIR / "ALGO_P2_RES_phase1_claude.csv",
        RESULTS_DIR / "ALGO_P2_RES_phase1_gpt4o.csv",
        RESULTS_DIR / "ALGO_P2_RES_phase1_llama.csv",
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    return pd.concat([pd.read_csv(p, dtype=str).fillna("") for p in paths], ignore_index=True)


def _compute_instance_tep(normal: pd.DataFrame, injected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = sorted(set(zip(injected["problem_id"], injected["model"], injected["subtype"])))
    for pid, model, subtype in keys:
        i = injected[(injected["problem_id"] == pid) & (injected["model"] == model) & (injected["subtype"] == subtype)].copy()
        n = normal[(normal["problem_id"] == pid) & (normal["model"] == model) & (normal["subtype"] == subtype)].copy()
        if i.empty or n.empty:
            continue
        crit = int(pd.to_numeric(i["critical_step_index"], errors="coerce").dropna().iloc[0]) if i["critical_step_index"].notna().any() else -1
        m = n.merge(
            i[["step_index", "response_type", "parsed_decision"]],
            on="step_index",
            how="inner",
            suffixes=("_n", "_i"),
        )
        m = m[(m["step_index"] > crit) & (m["response_type_n"] == "compliant") & (m["response_type_i"] == "compliant")]
        if len(m) == 0:
            tep = np.nan
        else:
            tep = float((m["parsed_decision_n"].astype(str).str.strip() != m["parsed_decision_i"].astype(str).str.strip()).mean())
        rows.append({"problem_id": pid, "model": model, "subtype": subtype, "tep": tep})
    return pd.DataFrame(rows)


def _compute_instance_cci_proxy(normal: pd.DataFrame) -> pd.DataFrame:
    # Proxy CCI from reasoning composition per instance:
    # ratio of algorithm_invocation/forward_simulation steps.
    rows = []
    for (pid, model, subtype), g in normal.groupby(["problem_id", "model", "subtype"]):
        r = g["reasoning_type"].astype(str).str.strip().str.lower()
        if len(r) == 0:
            cci = np.nan
        else:
            cci = float(((r == "algorithm_invocation") | (r == "forward_simulation")).mean())
        rows.append({"problem_id": pid, "model": model, "subtype": subtype, "cci": cci})
    return pd.DataFrame(rows)


def plot_cci_tep(normal: pd.DataFrame, injected: pd.DataFrame) -> None:
    cci = _compute_instance_cci_proxy(normal)
    tep = _compute_instance_tep(normal, injected)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, metric_df, metric_name in [(axes[0], cci, "CCI"), (axes[1], tep, "TEP")]:
        pos = np.arange(len(SUBTYPE_ORDER))
        width = 0.22
        for i, model in enumerate(MODEL_ORDER):
            vals_by_sub = []
            for subtype in SUBTYPE_ORDER:
                vals = metric_df[(metric_df["model"] == model) & (metric_df["subtype"] == subtype)][metric_name.lower()].dropna().astype(float).tolist()
                vals_by_sub.append(vals)
            # violin per subtype/model offset
            for j, vals in enumerate(vals_by_sub):
                x = pos[j] + (i - 1) * width
                if vals:
                    v = ax.violinplot([vals], positions=[x], widths=width * 0.9, showmeans=True)
                    for b in v["bodies"]:
                        b.set_facecolor(P2_ACCENT if i == 1 else (P2_PRIMARY if i == 0 else P2_LIGHT))
                        b.set_alpha(0.5)
                    ax.scatter(np.full(len(vals), x), vals, s=12, color=P2_PRIMARY, alpha=0.6, edgecolor="black", linewidth=0.2)
        ax.set_xticks(pos)
        ax.set_xticklabels([SUBTYPE_LABEL[s] for s in SUBTYPE_ORDER])
        ax.set_ylim(0, 1.0)
        ax.set_title(metric_name)
        ax.set_ylabel("Score")
    axes[0].set_title("CCI by subtype/model")
    axes[1].set_title("TEP by subtype/model")
    fig.suptitle("Figure 6 — CCI and TEP side by side")
    _save(fig, "ALGO_P2_FIG_06_cci_tep_side_by_side")


def plot_tep_interpretation(tep: pd.DataFrame) -> None:
    agg = tep.groupby("subtype", as_index=False)["tep"].mean()
    labels = {
        "coin_change": "state-reactive\n(computation signal)",
        "shortest_path": "shortcut-reactive\n(ambiguous)",
        "wis": "local-greedy-reactive\n(retrieval signal)",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SUBTYPE_ORDER))
    y = [float(agg[agg["subtype"] == s]["tep"].mean()) for s in SUBTYPE_ORDER]
    bars = ax.bar(x, y, color=P2_PRIMARY, alpha=0.9)
    for i, (b, yy) in enumerate(zip(bars, y)):
        ax.text(b.get_x() + b.get_width() / 2, yy + 0.03, f"{yy:.2f}", ha="center", fontsize=9)
        ax.text(b.get_x() + b.get_width() / 2, 0.02, labels[SUBTYPE_ORDER[i]], ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBTYPE_LABEL[s] for s in SUBTYPE_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("TEP")
    ax.set_title("Figure 7 — TEP by subtype (interpretation map)")
    _save(fig, "ALGO_P2_FIG_07_tep_interpretation")


def plot_reasoning_distribution(normal: pd.DataFrame) -> None:
    n = normal.copy()
    n["reasoning_type"] = n["reasoning_type"].astype(str).str.strip().str.lower()
    keep = ["local_greedy", "forward_simulation", "algorithm_invocation", "unclear"]
    n.loc[~n["reasoning_type"].isin(keep), "reasoning_type"] = "unclear"
    combos = []
    for subtype in SUBTYPE_ORDER:
        for inst in ["standard", "adversarial"]:
            sub = n[(n["subtype"] == subtype) & (n["instance_type"] == inst)]
            total = max(1, len(sub))
            frac = {k: float((sub["reasoning_type"] == k).sum() / total) for k in keep}
            combos.append({"label": f"{SUBTYPE_LABEL[subtype]}-{inst[:3].upper()}", **frac})
    d = pd.DataFrame(combos)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    color_map = {"local_greedy": RED, "forward_simulation": GREEN, "algorithm_invocation": BLUE, "unclear": GREY}
    for k in keep:
        vals = d[k].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, color=color_map[k], label=k)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("Figure 9 — Reasoning type distribution by instance type")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, "ALGO_P2_FIG_09_reasoning_distribution")


def plot_adc_matrix(phase1: pd.DataFrame, normal: pd.DataFrame) -> None:
    p1 = phase1.copy()
    p1["adc"] = p1["greedy_assessment_correct"].astype(str).str.strip().str.lower().map({"true": "yes", "false": "no"})
    # final correctness from last step in normal
    n = normal.copy()
    n["step_index"] = pd.to_numeric(n["step_index"], errors="coerce")
    last = n.sort_values("step_index").groupby(["problem_id", "model"], as_index=False).tail(1)
    last["final"] = last["final_answer_correct"].astype(str).str.strip().str.lower().map({"true": "yes", "false": "no"})
    m = p1.merge(last[["problem_id", "model", "final"]], on=["problem_id", "model"], how="left")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = m[m["model"] == model]
        mat = np.zeros((2, 2), dtype=int)  # rows adc yes/no, cols final yes/no
        for i, adc in enumerate(["yes", "no"]):
            for j, final in enumerate(["yes", "no"]):
                mat[i, j] = int(((sub["adc"] == adc) & (sub["final"] == final)).sum())
        im = ax.imshow(mat, cmap=plt.cm.Blues, vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["final yes", "final no"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["adc yes", "adc no"])
        ax.set_title(MODEL_LABEL[model], fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Figure 10 — ADC × final correctness (2×2)")
    _save(fig, "ALGO_P2_FIG_10_adc_matrix")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9})
    normal = _load_phase2_normal()
    injected = _load_phase2_injected()
    phase1 = _load_phase1()
    plot_cci_tep(normal, injected)
    tep = _compute_instance_tep(normal, injected)
    plot_tep_interpretation(tep)
    plot_reasoning_distribution(normal)
    plot_adc_matrix(phase1, normal)
    print("Generated ALGO Probe 2 figures in results/figures/")


if __name__ == "__main__":
    main()
