#!/usr/bin/env python3
"""Generate ALGO Probe 1 figures (blue family)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.stats import bootstrap_ci

np.random.seed(42)

RESULTS_DIR = Path("results")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

P1_PRIMARY = "#1D4ED8"
P1_ACCENT = "#3B82F6"
P1_LIGHT = "#DBEAFE"
GOLD = "#F59E0B"
RED = "#EF4444"

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
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W6"]


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_behavioral() -> pd.DataFrame:
    paths = [
        RESULTS_DIR / "ALGO_P1_RES_behavioral_sweep_claude.csv",
        RESULTS_DIR / "ALGO_P1_RES_behavioral_sweep_gpt4o.csv",
        RESULTS_DIR / "ALGO_P1_RES_behavioral_sweep_llama.csv",
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    df = pd.concat([pd.read_csv(p, dtype=str).fillna("") for p in paths], ignore_index=True)
    required = {"problem_id", "variant_type", "model", "verified", "correct_canonical", "gave_greedy_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"behavioral files missing columns: {sorted(missing)}")
    df["verified_bool"] = df["verified"].astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    return df


def _load_bank_meta() -> pd.DataFrame:
    bank = pd.read_csv(RESULTS_DIR.parent / "data/problems/question_bank_algo.csv", dtype=str).fillna("")
    bank = bank[bank["variant_type"].str.strip().str.lower() == "canonical"].copy()
    bank["problem_subtype"] = bank["problem_subtype"].str.strip().str.lower()

    def parse_params(raw: str) -> dict:
        if not raw.strip():
            raise ValueError("missing difficulty_params")
        return json.loads(raw)

    bank["params"] = bank["difficulty_params"].map(parse_params)
    bank["instance_type"] = bank["params"].map(lambda p: str(p.get("instance_type", "")).strip().lower())
    bank["greedy_succeeds_expected"] = bank["params"].map(lambda p: p.get("greedy_succeeds", None))
    return bank[["problem_id", "problem_subtype", "instance_type", "greedy_succeeds_expected"]]


def _load_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "ALGO_P1_RES_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    m = pd.read_csv(path, dtype=str).fillna("")
    m["metric_value_num"] = pd.to_numeric(m["metric_value"], errors="coerce")
    m["ci_lower_num"] = pd.to_numeric(m["ci_lower"], errors="coerce")
    m["ci_upper_num"] = pd.to_numeric(m["ci_upper"], errors="coerce")
    return m


def _load_contam() -> pd.DataFrame:
    path = RESULTS_DIR / "ALGO_P3_RES_contamination.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    c = pd.read_csv(path, dtype=str).fillna("")
    c["problem_subtype"] = c["problem_subtype"].str.strip().str.lower()
    c["instance_contamination_score"] = pd.to_numeric(c["instance_contamination_score"], errors="coerce")
    # dedup from resumed runs
    c = c.groupby(["problem_id", "problem_subtype"], as_index=False).agg({"instance_contamination_score": "last"})
    return c


def _bootstrap_mean(vals: list[float], n: int = 10000) -> tuple[float, float, float]:
    if not vals:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    lo, hi = bootstrap_ci(vals, n_resamples=n)
    return mean, float(lo), float(hi)


def plot_var_heatmap(beh: pd.DataFrame, bank: pd.DataFrame) -> None:
    df = beh.merge(bank, on="problem_id", how="left", validate="many_to_one")
    cmap = mcolors.LinearSegmentedColormap.from_list("white_blue_algo", ["#FFFFFF", P1_ACCENT])

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharey=True)
    for r, inst in enumerate(["standard", "adversarial"]):
        for c, subtype in enumerate(SUBTYPE_ORDER):
            ax = axes[r, c]
            sub = df[(df["instance_type"] == inst) & (df["problem_subtype"] == subtype)]
            mat = np.full((len(MODEL_ORDER), len(VARIANTS)), np.nan)
            for i, model in enumerate(MODEL_ORDER):
                for j, variant in enumerate(VARIANTS):
                    vals = sub[(sub["model"] == model) & (sub["variant_type"] == variant)]["verified_bool"].dropna().astype(float).tolist()
                    if vals:
                        mat[i, j] = float(np.mean(vals))
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap=cmap)
            for i in range(len(MODEL_ORDER)):
                for j in range(len(VARIANTS)):
                    if not np.isnan(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(VARIANTS)))
            ax.set_xticklabels(VARIANTS, rotation=45, ha="right")
            ax.set_yticks(range(len(MODEL_ORDER)))
            ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=9)
            ax.set_title(f"{SUBTYPE_LABEL[subtype]} — {inst.title()}")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("VAR")
    fig.suptitle("Figure 1 — VAR Heatmap by subtype and instance type", fontsize=13)
    _save(fig, "ALGO_P1_FIG_01_var_heatmap")


def plot_gss_bar(beh: pd.DataFrame, bank: pd.DataFrame) -> None:
    df = beh.merge(bank, on="problem_id", how="left", validate="many_to_one")
    df = df[df["variant_type"] == "canonical"].copy()
    rows = []
    for model in MODEL_ORDER:
        for subtype in SUBTYPE_ORDER:
            sub = df[(df["model"] == model) & (df["problem_subtype"] == subtype)]
            good = sub[sub["greedy_succeeds_expected"].astype(str).str.lower() == "true"]["verified_bool"].dropna().astype(float).tolist()
            bad = sub[sub["greedy_succeeds_expected"].astype(str).str.lower() == "false"]["verified_bool"].dropna().astype(float).tolist()
            if good and bad:
                vals = []
                nboot = 10000
                n = max(len(good), len(bad))
                for _ in range(nboot):
                    g = np.random.choice(good, size=n, replace=True).mean()
                    b = np.random.choice(bad, size=n, replace=True).mean()
                    vals.append(float(g - b))
                mean = float(np.mean(vals))
                lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
            else:
                mean, lo, hi = np.nan, np.nan, np.nan
            rows.append({"model": model, "subtype": subtype, "gss": mean, "lo": lo, "hi": hi})
    g = pd.DataFrame(rows)
    x = np.arange(len(SUBTYPE_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(MODEL_ORDER):
        sub = g[g["model"] == model].set_index("subtype")
        ys = [sub.loc[s, "gss"] if s in sub.index else np.nan for s in SUBTYPE_ORDER]
        lo = [sub.loc[s, "lo"] if s in sub.index else np.nan for s in SUBTYPE_ORDER]
        hi = [sub.loc[s, "hi"] if s in sub.index else np.nan for s in SUBTYPE_ORDER]
        yerr = np.array([
            [max(0.0, y - l) if pd.notna(y) and pd.notna(l) else 0.0 for y, l in zip(ys, lo)],
            [max(0.0, h - y) if pd.notna(y) and pd.notna(h) else 0.0 for y, h in zip(ys, hi)],
        ])
        bars = ax.bar(x + (i - 1) * width, ys, width=width, yerr=yerr, capsize=3, label=MODEL_LABEL[model], alpha=0.9)
        for b, y in zip(bars, ys):
            if pd.notna(y):
                ax.text(b.get_x() + b.get_width() / 2, y + (0.03 if y >= 0 else -0.05), f"{y:.2f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBTYPE_LABEL[s] for s in SUBTYPE_ORDER])
    ax.set_ylabel("GSS")
    ax.set_ylim(-1, 1)
    ax.set_title("Figure 2 — GSS by subtype and model (bootstrap 95% CI)")
    ax.legend(fontsize=9)
    _save(fig, "ALGO_P1_FIG_02_gss_bar")


def plot_vri_gap(metrics: pd.DataFrame) -> None:
    m = metrics[metrics["metric_name"].isin(["VRI_structural", "VRI_vocabulary"])].copy()
    if m.empty:
        return
    agg = m.groupby(["subtype", "metric_name"], as_index=False)["metric_value_num"].mean()
    x = np.arange(len(SUBTYPE_ORDER))
    width = 0.34
    structural = [float(agg[(agg["subtype"] == s) & (agg["metric_name"] == "VRI_structural")]["metric_value_num"].mean()) for s in SUBTYPE_ORDER]
    vocab = [float(agg[(agg["subtype"] == s) & (agg["metric_name"] == "VRI_vocabulary")]["metric_value_num"].mean()) for s in SUBTYPE_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, structural, width=width, color=GOLD, label="Structural")
    b2 = ax.bar(x + width / 2, vocab, width=width, color=RED, label="Vocabulary")
    for bars in [b1, b2]:
        for b in bars:
            y = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, y + 0.02, f"{y:.2f}", ha="center", fontsize=8)
    gaps = np.array(structural) - np.array(vocab)
    for i, g in enumerate(gaps):
        ax.text(x[i], max(structural[i], vocab[i]) + 0.08, f"gap={g:.2f}", ha="center", fontsize=8, color="#111827")
    if abs(gaps[2]) < 0.05:
        ax.text(2, 0.95, "WIS gap≈0 (cross-subtype finding)", ha="center", fontsize=9, color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBTYPE_LABEL[s] for s in SUBTYPE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("VRI")
    ax.set_title("Figure 3 — VRI Structural vs Vocabulary")
    ax.legend()
    _save(fig, "ALGO_P1_FIG_03_vri_structural_vs_vocabulary")


def plot_contamination_scatter(beh: pd.DataFrame, bank: pd.DataFrame, contam: pd.DataFrame) -> None:
    df = beh[beh["variant_type"] == "canonical"].copy()
    df = df.merge(bank[["problem_id", "problem_subtype"]], on="problem_id", how="left", validate="many_to_one")
    df = df.merge(contam, on=["problem_id", "problem_subtype"], how="left", validate="many_to_one")
    df["instance_contamination_score"] = pd.to_numeric(df["instance_contamination_score"], errors="coerce")
    df = df.dropna(subset=["verified_bool", "instance_contamination_score"])

    colors = {"coin_change": "#2563EB", "shortest_path": "#1D4ED8", "wis": "#60A5FA"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for subtype in SUBTYPE_ORDER:
        sub = df[df["problem_subtype"] == subtype]
        ax.scatter(sub["instance_contamination_score"], sub["verified_bool"], s=28, alpha=0.7, color=colors[subtype], label=SUBTYPE_LABEL[subtype], edgecolor="black", linewidth=0.2)
    # Regression + bootstrap band (overall)
    x = df["instance_contamination_score"].to_numpy(float)
    y = df["verified_bool"].to_numpy(float)
    if len(x) >= 2 and np.unique(x).size > 1:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 150)
        ys = b1 * xs + b0
        preds = []
        n = len(x)
        for _ in range(3000):
            idx = np.random.randint(0, n, size=n)
            xb, yb = x[idx], y[idx]
            if np.unique(xb).size < 2:
                continue
            bb1, bb0 = np.polyfit(xb, yb, 1)
            preds.append(bb1 * xs + bb0)
        if preds:
            arr = np.vstack(preds)
            lo = np.percentile(arr, 2.5, axis=0)
            hi = np.percentile(arr, 97.5, axis=0)
            ax.fill_between(xs, lo, hi, color=P1_PRIMARY, alpha=0.15)
        ax.plot(xs, ys, color=P1_PRIMARY, linewidth=2, label="OLS trend")
    ax.set_xlabel("instance_contamination_score")
    ax.set_ylabel("VAR(canonical)")
    ax.set_title("Figure 4 — Contamination vs canonical VAR")
    ax.legend()
    _save(fig, "ALGO_P1_FIG_04_contamination_scatter")


def plot_cross_family(beh: pd.DataFrame, bank: pd.DataFrame) -> None:
    # Algo GSS-like score per model.
    algo = beh[beh["variant_type"] == "canonical"].merge(bank, on="problem_id", how="left", validate="many_to_one")
    algo_rows = []
    for model in MODEL_ORDER:
        sub = algo[algo["model"] == model]
        good = sub[sub["greedy_succeeds_expected"].astype(str).str.lower() == "true"]["verified_bool"].dropna().astype(float)
        bad = sub[sub["greedy_succeeds_expected"].astype(str).str.lower() == "false"]["verified_bool"].dropna().astype(float)
        val = float(good.mean() - bad.mean()) if len(good) and len(bad) else np.nan
        algo_rows.append((model, val))
    algo_df = pd.DataFrame(algo_rows, columns=["model", "algo_gss"])

    # BW PDAS proxy from BW behavioral: W5 - canonical accuracy.
    bw_path = RESULTS_DIR / "BW_P1_RES_behavioral_sweep.csv"
    bw = pd.read_csv(bw_path, dtype=str).fillna("")
    bw["behavioral_correct_bool"] = bw["behavioral_correct"].astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    bw_rows = []
    for model in MODEL_ORDER:
        sub = bw[bw["model"] == model]
        can = sub[sub["variant_type"] == "canonical"]["behavioral_correct_bool"].dropna().astype(float)
        w5 = sub[sub["variant_type"] == "W5"]["behavioral_correct_bool"].dropna().astype(float)
        p = float(w5.mean() - can.mean()) if len(can) and len(w5) else np.nan
        bw_rows.append((model, p))
    bw_df = pd.DataFrame(bw_rows, columns=["model", "bw_pdas"])

    merged = algo_df.merge(bw_df, on="model", how="outer")
    x = np.arange(4)  # BW,CC,SP,WIS
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(MODEL_ORDER):
        g = merged[merged["model"] == model]
        vals = [
            float(g["bw_pdas"].iloc[0]) if len(g) else np.nan,
            float(g["algo_gss"].iloc[0]) if len(g) else np.nan,
            float(g["algo_gss"].iloc[0]) if len(g) else np.nan,
            float(g["algo_gss"].iloc[0]) if len(g) else np.nan,
        ]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=[P1_LIGHT, P1_PRIMARY, P1_PRIMARY, P1_PRIMARY],
            alpha=0.9,
            label=MODEL_LABEL[model],
        )
        for b, y in zip(bars, vals):
            if pd.notna(y):
                ax.text(b.get_x() + b.get_width() / 2, y + 0.02, f"{y:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["BW (PDAS)", "CC (GSS)", "SP (GSS)", "WIS (GSS)"])
    ax.set_ylabel("Score")
    ax.set_title("Figure 5 — Cross-family heuristic signal (BW vs ALGO)")
    ax.legend(fontsize=8)
    _save(fig, "ALGO_P1_FIG_05_cross_family_comparison")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9})
    beh = _load_behavioral()
    bank = _load_bank_meta()
    metrics = _load_metrics()
    contam = _load_contam()
    plot_var_heatmap(beh, bank)
    plot_gss_bar(beh, bank)
    plot_vri_gap(metrics)
    plot_contamination_scatter(beh, bank, contam)
    plot_cross_family(beh, bank)
    print("Generated ALGO Probe 1 figures in results/figures/")


if __name__ == "__main__":
    main()
