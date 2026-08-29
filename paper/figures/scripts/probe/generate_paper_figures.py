#!/usr/bin/env python3
"""Generate four paper figures at 300 DPI PDF."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color-blind safe (Okabe-Ito)
C_CLAUDE = "#0072B2"
C_GPT = "#E69F00"
C_LLAMA = "#009E73"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_BLUE = "#0072B2"
C_GRAY = "#999999"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1"])


def _parse_curve(raw: str) -> np.ndarray:
    if pd.isna(raw):
        return np.array([])
    text = str(raw).strip()
    try:
        return np.array(json.loads(text))
    except json.JSONDecodeError:
        return np.array(ast.literal_eval(text))


def fig1_inversion() -> None:
    gsm_c = pd.read_csv(ROOT / "results/raw/GSM_P1_behavioral_claude.csv")
    gsm_g = pd.read_csv(ROOT / "results/raw/GSM_P1_behavioral_gpt4o.csv")
    trian = pd.read_csv(ROOT / "results/derived/ALGO_P3_triangulation.csv")

    def gsm_acc(df: pd.DataFrame, vt: str) -> float:
        sub = df[df["variant_type"].str.upper() == vt.upper()]
        return float(_bool_series(sub["behavioral_correct"]).mean())

    def algo_adv_acc(subtype: str, model: str, vt: str) -> float:
        d = trian[
            (trian["problem_subtype"] == subtype)
            & (trian["instance_type"] == "adversarial")
            & (trian["model"] == model)
        ]
        col = "VAR_canonical" if vt.lower() == "canonical" else "VAR_W3"
        return float(d[col].mean())

    groups = ["GSM", "CC-adv", "SP-adv"]
    data = {
        "Claude canonical": [
            gsm_acc(gsm_c, "canonical"),
            algo_adv_acc("coin_change", "anthropic/claude-sonnet-4", "canonical"),
            algo_adv_acc("shortest_path", "anthropic/claude-sonnet-4", "canonical"),
        ],
        "Claude W3": [
            gsm_acc(gsm_c, "W3"),
            algo_adv_acc("coin_change", "anthropic/claude-sonnet-4", "W3"),
            algo_adv_acc("shortest_path", "anthropic/claude-sonnet-4", "W3"),
        ],
        "GPT-4o canonical": [
            gsm_acc(gsm_g, "canonical"),
            algo_adv_acc("coin_change", "openai/gpt-4o", "canonical"),
            algo_adv_acc("shortest_path", "openai/gpt-4o", "canonical"),
        ],
        "GPT-4o W3": [
            gsm_acc(gsm_g, "W3"),
            algo_adv_acc("coin_change", "openai/gpt-4o", "W3"),
            algo_adv_acc("shortest_path", "openai/gpt-4o", "W3"),
        ],
    }

    x = np.arange(len(groups))
    width = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = [C_CLAUDE, C_CLAUDE, C_GPT, C_GPT]
    hatches = ["", "//", "", "//"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (label, vals) in enumerate(data.items()):
        bars = ax.bar(
            x + offsets[i] * width,
            vals,
            width,
            label=label,
            color=colors[i],
            alpha=0.85 if "W3" in label else 1.0,
            hatch=hatches[i],
            edgecolor="white",
            linewidth=0.5,
        )

    # Highlight SP-adv inversion zone
    sp_idx = 2
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (sp_idx - 0.42, 0.0),
            0.84,
            0.72,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor=C_RED,
            facecolor="none",
            linestyle="--",
        )
    )
    ax.annotate(
        "Inversion:\nClaude canon > GPT canon\nClaude W3 ≪ GPT W3",
        xy=(sp_idx, 0.55),
        xytext=(sp_idx + 0.55, 0.78),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
        color=C_RED,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy–robustness inversion across families")
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig1_inversion.pdf")


def fig2_nine_problems() -> None:
    problems = ["CC_01", "CC_02", "CC_03", "CC_05", "SP_024", "SP_044", "SP_068", "SP_069", "SP_070"]
    cols = ["Claude\ncanon", "Claude\nW3", "GPT-4o\ncanon", "GPT-4o\nW3"]
    algo_c = pd.read_csv(ROOT / "results/raw/ALGO_P1_behavioral_claude.csv")
    algo_g = pd.read_csv(ROOT / "results/raw/ALGO_P1_behavioral_gpt4o.csv")

    grid = np.zeros((len(problems), 4))
    for i, pid in enumerate(problems):
        for j, (df, vt) in enumerate(
            [(algo_c, "canonical"), (algo_c, "W3"), (algo_g, "canonical"), (algo_g, "W3")]
        ):
            row = df[(df["problem_id"] == pid) & (df["variant_type"].str.upper() == vt.upper())]
            if len(row):
                grid[i, j] = 1.0 if _bool_series(row["verified"]).iloc[0] else 0.0
            else:
                grid[i, j] = np.nan

    cmap = sns.color_palette(["#D55E00", "#009E73"])  # red, green
    fig, ax = plt.subplots(figsize=(5.5, 6))
    sns.heatmap(
        grid,
        ax=ax,
        cmap=cmap,
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        xticklabels=cols,
        yticklabels=problems,
        vmin=0,
        vmax=1,
    )
    for i in range(len(problems)):
        for j in range(4):
            val = grid[i, j]
            if np.isnan(val):
                text = "—"
            else:
                text = "✓" if val == 1 else "✗"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    ax.set_title("Visual double dissociation (9 exemplar problems)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _save(fig, "fig2_9problems.pdf")


def fig3_crossprobe() -> None:
    gsm_cci = pd.read_csv(ROOT / "results/raw/GSM_P2_cci.csv")
    gsm_contam = pd.read_csv(ROOT / "results/raw/GSM_P3_contamination.csv")
    bw_plans = pd.read_csv(ROOT / "results/raw/BW_P2_plans.csv")
    bw_contam = pd.read_csv(ROOT / "results/raw/BW_P3_contamination.csv")
    phase2 = pd.read_csv(ROOT / "results/raw/ALGO_P2_phase2_normal.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))

    # Panel A
    ax = axes[0]
    merged = gsm_cci.merge(
        gsm_contam[["problem_id", "contamination_score"]].drop_duplicates("problem_id"),
        on="problem_id",
        how="inner",
    )
    model_colors = {
        "anthropic/claude-sonnet-4": C_CLAUDE,
        "openai/gpt-4o": C_GPT,
        "meta-llama/llama-3.1-8b-instruct": C_LLAMA,
    }
    model_labels = {"anthropic/claude-sonnet-4": "Claude", "openai/gpt-4o": "GPT-4o", "meta-llama/llama-3.1-8b-instruct": "Llama"}
    for model, color in model_colors.items():
        sub = merged[merged["model"] == model].dropna(subset=["contamination_score", "cci_score"])
        ax.scatter(sub["contamination_score"], sub["cci_score"], s=28, alpha=0.75, color=color, label=model_labels[model])
        if len(sub) > 5:
            z = np.polyfit(sub["contamination_score"], sub["cci_score"], 1)
            xs = np.linspace(sub["contamination_score"].min(), sub["contamination_score"].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color=color, lw=1.5, alpha=0.9)
            r, p = stats.pearsonr(sub["contamination_score"], sub["cci_score"])
            if model == "anthropic/claude-sonnet-4":
                ax.text(0.04, 0.94, f"Claude: r={r:.2f}, p={p:.3f}", transform=ax.transAxes, fontsize=8, color=C_CLAUDE)
            elif model == "openai/gpt-4o":
                ax.text(0.04, 0.86, f"GPT-4o: r={r:.2f}, p={p:.2f}", transform=ax.transAxes, fontsize=8, color=C_GPT)
            else:
                ax.text(0.04, 0.78, f"Llama: r={r:.2f}, p={p:.2f}", transform=ax.transAxes, fontsize=8, color=C_LLAMA)

    ax.set_xlabel("Contamination score")
    ax.set_ylabel("CCI")
    ax.set_title("A  GSM: contamination × coupling")
    ax.legend(frameon=False, loc="lower right")

    # Panel B
    ax = axes[1]
    bw_m = bw_plans.merge(
        bw_contam[["problem_id", "contamination_score"]].drop_duplicates("problem_id"),
        on="problem_id",
        how="left",
    )
    for model, color in model_colors.items():
        sub = bw_m[(bw_m["model"] == model) & bw_m["contamination_score"].notna()]
        ax.scatter(sub["contamination_score"], sub["plan_length"], s=22, alpha=0.6, color=color, label=model_labels[model])
        if len(sub) > 5:
            z = np.polyfit(sub["contamination_score"], sub["plan_length"], 1)
            xs = np.linspace(sub["contamination_score"].min(), sub["contamination_score"].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color=color, lw=1.5)
            rho, p = stats.spearmanr(sub["contamination_score"], sub["plan_length"])
            short = model.split("/")[-1][:6]
            ax.text(
                0.03,
                0.92 - 0.08 * list(model_colors.keys()).index(model),
                f"{model_labels[model]}: ρ={rho:.2f}",
                transform=ax.transAxes,
                fontsize=7.5,
                color=color,
            )
    ax.set_xlabel("Contamination score")
    ax.set_ylabel("Plan length")
    ax.set_title("B  BW: contamination × plan length")

    # Panel C
    ax = axes[2]
    phase2 = phase2.copy()
    phase2["correct"] = phase2["final_answer_correct"].astype(str).str.lower() == "true"
    rt = (
        phase2.groupby("reasoning_type")["correct"]
        .mean()
        .reindex(["algorithm_invocation", "unclear", "local_greedy", "forward_simulation", "backtracking"])
        .dropna()
    )
    show = rt.loc[["algorithm_invocation", "unclear", "local_greedy"]]
    labels = ["algorithm\ninvocation", "unclear", "local\ngreedy"]
    bars = ax.bar(range(len(show)), show.values * 100, color=[C_CLAUDE, C_GPT, C_LLAMA], edgecolor="white")
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("C  ALGO phase-2 reasoning type")
    for b, v in zip(bars, show.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig3_crossprobe.pdf")


def fig4_mechanistic() -> None:
    mech = pd.read_csv(ROOT / "results/raw/ALGO_P3_mechanistic.csv")
    mech["curve"] = mech["layer_cosine_similarities"].apply(_parse_curve)
    mech = mech[mech["curve"].apply(len) == 24]

    def mean_curve(prefix: str) -> np.ndarray:
        sub = mech[mech["problem_id"].str.startswith(f"{prefix}_")]
        return np.vstack(sub["curve"].tolist()).mean(axis=0)

    cc_all = mech[mech["problem_id"].str.startswith("CC_")]
    cc_std = cc_all["curve"].apply(lambda c: np.std(c))
    flat_ids = cc_std.nsmallest(3).index
    cc_w6_proxy = np.vstack(cc_all.loc[flat_ids, "curve"].tolist()).mean(axis=0)

    layers = np.arange(1, 25)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, cc_w6_proxy, color=C_BLUE, lw=2, ls="--", label="CC high-similarity (W6 proxy)")
    ax.plot(layers, mean_curve("CC"), color=C_BLUE, lw=2, label="CC canonical (mean)")
    ax.plot(layers, mean_curve("SP"), color=C_GPT, lw=2, label="SP canonical (mean)")
    ax.plot(layers, mean_curve("WIS"), color=C_LLAMA, lw=2, label="WIS canonical (mean)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity to final-layer activation")
    ax.set_title("Layer-wise activation similarity (preliminary)")
    ax.set_ylim(0.35, 1.0)
    ax.legend(frameon=True, loc="lower right")
    ax.text(
        0.02,
        0.02,
        "Preliminary: Qwen2.5-0.5B-Instruct\nPending 7B validation",
        transform=ax.transAxes,
        fontsize=8,
        color=C_BLUE,
        bbox=dict(boxstyle="round", facecolor="#DBEAFE", alpha=0.8),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig4_mechanistic.pdf")


def main() -> None:
    fig1_inversion()
    fig2_nine_problems()
    fig3_crossprobe()
    fig4_mechanistic()
    print("All four paper figures generated.")


if __name__ == "__main__":
    main()
