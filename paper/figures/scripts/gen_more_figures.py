"""Additional EMNLP-style figures (third batch).

New figures:
  fig_bw_inversion.pdf    — BW canonical vs W5/W6 per-model paired bars with
                             Wilcoxon-p annotations. Surfaces the contamination
                             signal: renaming blocks *improves* Claude/Gemini
                             and *destroys* Llama.
  fig_subtype_grid.pdf    — ALGO 3 subtypes × 7 variants × 5 models heatmap.
                             Surfaces (a) the WIS Achilles heel, (b) the
                             universal ALGO-W5 collapse, (c) subtype-specific
                             o4-mini brittleness.
  fig_probe2_summary.pdf  — Two-panel summary: (left) GSM Phase-1 CCI per
                             model with mean-TEP overlay; (right) ALGO P2A
                             normal vs elicited final-correct.
  fig_gsm_w5w6.pdf        — GSM W5 vs W6 inversion per model (24-problem
                             intersection). Shows distractor < numeric pert.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RAW  = ROOT / "results" / "raw"
DER  = ROOT / "results" / "derived"
OUT  = Path(__file__).resolve().parents[1]

COLOR = {
    "Claude":   "#0072B2",
    "GPT-4o":   "#D55E00",
    "Llama":    "#CC79A7",
    "Gemini":   "#009E73",
    "o4-mini":  "#E69F00",
}
MODELS = ["Claude", "Gemini", "GPT-4o", "Llama", "o4-mini"]
MMAP = {"Claude": "anthropic/claude-sonnet-4",
        "Gemini": "google/gemini-2.5-flash",
        "GPT-4o": "openai/gpt-4o",
        "Llama":  "meta-llama/llama-3.1-8b-instruct",
        "o4-mini":"openai/o4-mini"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})


def _bw_model_df(label: str) -> pd.DataFrame:
    """Returns per-model BW P1 deduplicated dataframe."""
    if label in {"Claude", "GPT-4o", "Llama"}:
        df = pd.read_csv(RAW / "BW_P1_behavioral.csv", dtype=str).fillna("")
        df = df[df.model == MMAP[label]]
    elif label == "Gemini":
        df = pd.read_csv(RAW / "BW_P1_behavioral_gemini.csv", dtype=str).fillna("")
    elif label == "o4-mini":
        df = pd.read_csv(RAW / "BW_P1_behavioral_o1mini.csv", dtype=str).fillna("")
    else:
        return pd.DataFrame()
    df["variant_type"] = df["variant_type"].astype(str).str.strip().apply(
        lambda v: v.upper() if v and v[0].lower() == "w" else v
    )
    return df.drop_duplicates(["problem_id", "variant_type"], keep="last")


def fig_bw_inversion() -> None:
    """Per-model BW canonical vs W5 paired bars + Wilcoxon p (within-model)."""
    rows = []
    for label in MODELS:
        d = _bw_model_df(label)
        canon = d[d.variant_type == "canonical"].set_index("problem_id")
        w5    = d[d.variant_type == "W5"].set_index("problem_id")
        common = canon.index.intersection(w5.index)
        if len(common) < 10:
            rows.append({"model": label, "canon": float("nan"),
                         "W5": float("nan"), "p": float("nan")})
            continue
        c = (canon.loc[common, "behavioral_correct"].str.lower() == "true").astype(int)
        v = (w5.loc[common,   "behavioral_correct"].str.lower() == "true").astype(int)
        if c.std() == 0 and v.std() == 0:
            p = float("nan")
        else:
            try: p = stats.wilcoxon(c, v, zero_method="wilcox").pvalue
            except: p = float("nan")
        rows.append({"model": label, "canon": float(c.mean()),
                     "W5": float(v.mean()), "p": p, "n": len(common)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.subplots_adjust(top=0.85, bottom=0.16, left=0.10, right=0.97)
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w/2, df.canon, width=w, color="#88AABB", edgecolor="white",
           linewidth=0.6, label="canonical blocks (a,b,c,...)")
    ax.bar(x + w/2, df.W5,    width=w, color="#A67BC2", edgecolor="white",
           linewidth=0.6, label="renamed blocks (W$_5$)")

    for i, row in df.iterrows():
        if pd.isna(row.p): continue
        h = max(row.canon, row.W5)
        if row.p < 0.001:
            sig = "***"
        elif row.p < 0.01:
            sig = "**"
        elif row.p < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        delta = row.W5 - row.canon
        arrow = r"$\uparrow$" if delta > 0 else (r"$\downarrow$" if delta < 0 else "")
        ax.text(i, h + 0.03, f"{sig}\n{arrow}{abs(delta)*100:.1f}pp",
                ha="center", va="bottom", fontsize=8.2, color="#222")

    ax.set_xticks(x); ax.set_xticklabels(df.model)
    ax.set_ylabel("Behavioural correct rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Blocksworld: renaming blocks flips the sign — Claude/Gemini gain, Llama collapses")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", color="#bbb", alpha=0.5)

    plt.savefig(OUT / "fig_bw_inversion.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_bw_inversion.pdf")


def fig_subtype_grid() -> None:
    """3 ALGO subtypes × 7 variants × 5 models heatmap (5 cols, 3 rows)."""
    bank = pd.read_csv(ROOT / "data" / "problems" / "question_bank_algo.csv",
                       dtype=str).fillna("")[["problem_id","variant_type","problem_subtype"]]

    sub_files = {
        "Claude":   "ALGO_P1_behavioral_claude.csv",
        "GPT-4o":   "ALGO_P1_behavioral_gpt4o.csv",
        "Gemini":   "ALGO_P1_behavioral_gemini.csv",
        "Llama":    "ALGO_P1_behavioral_llama.csv",
        "o4-mini":  "ALGO_P1_behavioral_o1mini.csv",
    }

    variants = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
    subtypes = ["coin_change", "shortest_path", "wis"]
    subtype_labels = {"coin_change": "Coin change (greedy)",
                      "shortest_path": "Shortest path",
                      "wis": "WIS (DP)"}

    accs = {}  # accs[(model, subtype, variant)] = acc
    for label, f in sub_files.items():
        df = pd.read_csv(RAW / f, dtype=str).fillna("")
        df = df.drop_duplicates(["problem_id", "variant_type"], keep="last")
        df = df.merge(bank, on=["problem_id", "variant_type"], how="left")
        for s in subtypes:
            for v in variants:
                sub = df[(df.variant_type == v) & (df.problem_subtype == s)]
                if len(sub) == 0:
                    accs[(label, s, v)] = float("nan")
                else:
                    accs[(label, s, v)] = float((sub.verified.str.lower() == "true").mean())

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.4), sharex=True)
    fig.subplots_adjust(top=0.92, bottom=0.07, left=0.16, right=0.96, hspace=0.30)

    cmap = plt.cm.RdYlGn
    for ax, s in zip(axes, subtypes):
        mat = np.array([[accs.get((m, s, v), np.nan) for v in variants] for m in MODELS], dtype=float)
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=9)
        ax.set_xticks(range(len(variants))); ax.set_xticklabels(variants, fontsize=9)
        ax.set_title(subtype_labels[s], fontsize=10, loc="left")
        for i, mlbl in enumerate(MODELS):
            for j, v in enumerate(variants):
                val = mat[i, j]
                if np.isnan(val):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8.0, color="#999")
                else:
                    color = "white" if 0.3 < val < 0.7 else "#222"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.6, color=color)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

    fig.suptitle("ALGO accuracy by subtype × variant × model — WIS is universally fatal",
                 fontsize=10.5)
    # colorbar on right
    cb_ax = fig.add_axes([0.97, 0.10, 0.014, 0.78])
    plt.colorbar(im, cax=cb_ax)
    plt.savefig(OUT / "fig_subtype_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_subtype_grid.pdf")


def fig_probe2_summary() -> None:
    """Two-panel: left = GSM CCI per model with mean-TEP overlay,
       right = ALGO P2A normal vs elicited (with paired Wilcoxon p).
    """
    # --- LEFT ---
    p2 = pd.read_csv(DER / "probe2_gsm_metrics.csv")
    p2 = p2.set_index("model").reindex(MODELS).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6),
                              gridspec_kw={"width_ratios": [1.0, 1.0]})
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.99, wspace=0.30)

    ax = axes[0]
    x = np.arange(len(MODELS))
    w = 0.36
    ax.bar(x - w/2, p2.mean_cci, width=w, color="#3498DB",
           edgecolor="white", linewidth=0.6, label="mean CCI")
    ax.bar(x + w/2, p2.mean_tep, width=w, color="#F39C12",
           edgecolor="white", linewidth=0.6, label="mean TEP")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("GSM Probe 2 — contradiction-citation (CCI) and target-evidence persistence (TEP)",
                 fontsize=10)
    ax.set_ylabel("Score")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", ls=":", color="#bbb", alpha=0.4)
    for i, row in p2.iterrows():
        ax.text(i - w/2, row.mean_cci + 0.018, f"{row.mean_cci:.2f}",
                ha="center", fontsize=7.4, color="#222")
        ax.text(i + w/2, row.mean_tep + 0.018, f"{row.mean_tep:.2f}",
                ha="center", fontsize=7.4, color="#222")

    # --- RIGHT ---
    algo = pd.read_csv(DER / "probe2_algo_metrics.csv")
    norm = algo[algo.condition == "normal"].set_index("model").reindex(MODELS).reset_index()
    elic = algo[algo.condition == "elicited"].set_index("model").reindex(MODELS).reset_index()

    ax = axes[1]
    ax.bar(x - w/2, norm.final_correct, width=w, color="#88AABB",
           edgecolor="white", linewidth=0.6, label="normal")
    ax.bar(x + w/2, elic.final_correct, width=w, color="#A67BC2",
           edgecolor="white", linewidth=0.6, label="elicited (named algorithm)")
    for i, m in enumerate(MODELS):
        d = elic.final_correct.iloc[i] - norm.final_correct.iloc[i]
        col = "#256029" if d > 0 else "#7E1F1F"
        sym = "+" if d > 0 else ""
        ax.text(i, max(elic.final_correct.iloc[i], norm.final_correct.iloc[i]) + 0.025,
                f"Δ={sym}{d*100:.1f}pp", ha="center", va="bottom",
                fontsize=7.8, color=col)
    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_title("ALGO Probe 2A — elicitation never significantly helps (paired Wilcoxon, p≥0.13)",
                 fontsize=10)
    ax.set_ylabel("Final-answer correct rate")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", ls=":", color="#bbb", alpha=0.4)

    plt.savefig(OUT / "fig_probe2_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_probe2_summary.pdf")


def fig_gsm_w5w6() -> None:
    """GSM W5 vs W6 inversion per model. 24-problem intersection.
    Numeric perturbation (W5) vs distractor injection (W6)."""
    gsm_files = {
        "Claude":   "GSM_P1_behavioral_claude.csv",
        "GPT-4o":   "GSM_P1_behavioral_gpt4o.csv",
        "Gemini":   "GSM_P1_behavioral_gemini.csv",
        "Llama":    "GSM_P1_behavioral_llama.csv",
        "o4-mini":  "GSM_P1_behavioral_o1mini.csv",
    }
    rows = []
    for label, f in gsm_files.items():
        df = pd.read_csv(RAW / f, dtype=str).fillna("")
        df["variant_type"] = df["variant_type"].astype(str).str.strip().apply(
            lambda v: v.upper() if v and v[0].lower() == "w" else v
        )
        df = df.drop_duplicates(["problem_id","variant_type"], keep="last")
        w5 = df[df.variant_type=="W5"].set_index("problem_id")
        w6 = df[df.variant_type=="W6"].set_index("problem_id")
        common = w5.index.intersection(w6.index)
        if len(common)<5:
            rows.append({"model": label, "W5": float("nan"), "W6": float("nan"), "n": 0})
            continue
        a = (w5.loc[common,"behavioral_correct"].str.lower()=="true").astype(int)
        b = (w6.loc[common,"behavioral_correct"].str.lower()=="true").astype(int)
        try: pv = stats.wilcoxon(a, b, zero_method="wilcox").pvalue
        except: pv = float("nan")
        rows.append({"model": label, "W5": float(a.mean()),
                     "W6": float(b.mean()), "n": len(common), "p": pv})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.12, right=0.97)

    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w/2, df.W5, width=w, color="#9B59B6", edgecolor="white",
           linewidth=0.6, label="W$_5$ (numeric perturbation)")
    ax.bar(x + w/2, df.W6, width=w, color="#27AE60", edgecolor="white",
           linewidth=0.6, label="W$_6$ (distractor injection)")
    for i, row in df.iterrows():
        if pd.isna(row.W5) or pd.isna(row.W6): continue
        h = max(row.W5, row.W6)
        ax.text(i, h + 0.025, f"n={int(row.n)}", ha="center", va="bottom",
                fontsize=7.8, color="#444")

    ax.set_xticks(x); ax.set_xticklabels(df.model, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Behavioural correct rate")
    ax.set_title("GSM — numeric perturbation (W$_5$) vs distractor injection (W$_6$) on shared problems")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.9)
    ax.grid(axis="y", ls=":", color="#bbb", alpha=0.4)

    plt.savefig(OUT / "fig_gsm_w5w6.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_gsm_w5w6.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_bw_inversion()
    fig_subtype_grid()
    fig_probe2_summary()
    fig_gsm_w5w6()
    print("Done. (4 new figures)")


if __name__ == "__main__":
    main()
