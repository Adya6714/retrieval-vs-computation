# scripts/figures/fig3_cci_tep.py
# Figure 3: CCI (left) and TEP (right) by model, GSM, n=44 — loaded from data
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
RAW = ROOT / "results/raw"
FIG.mkdir(exist_ok=True)

ORDER = [
    ("Claude", "claude", "anthropic/claude-sonnet-4"),
    ("GPT-4o", "gpt4o", "openai/gpt-4o"),
    ("Llama-8B", "llama", "meta-llama/llama-3.1-8b-instruct"),
    ("Gemini-2.5", "gemini", "google/gemini-2.5-flash"),
]
colors = ["#0072B2", "#D55E00", "#CC79A7", "#009E73"]


def load_gsm_p2() -> pd.DataFrame:
    cci = RAW / "GSM_P2_cci.csv"
    if cci.exists() and len(pd.read_csv(cci)) >= 160:
        return pd.read_csv(cci)
    frames = [pd.read_csv(RAW / f"GSM_P2_phase1_{short}.csv") for _, short, _ in ORDER]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_gsm_p2()
    labels, cci, cci_m, tep, acc = [], [], [], [], []
    for label, short, mid in ORDER:
        if short == "gpt4o":
            s = df[df.model.astype(str).str.contains("gpt-4o", case=False)]
        else:
            s = df[df.model.astype(str).str.contains(short, case=False)]
        labels.append(label)
        cci.append(float(s.cci_score.mean()))
        cci_m.append(float(s.cci_score.median()))
        tep.append(float(s.tep_score.mean()))
        acc.append(float(s.session_b_correct.astype(str).str.lower().eq("true").mean()))

    cl = df[df.model.astype(str).str.contains("claude", case=False)].set_index("problem_id")["cci_score"]
    gp = df[df.model.astype(str).str.contains("gpt-4o", case=False)].set_index("problem_id")["cci_score"]
    ids = sorted(set(cl.index) | set(gp.index))
    w, p_wilcox = stats.wilcoxon(
        [cl.get(i, 0.0) for i in ids],
        [gp.get(i, 0.0) for i in ids],
        alternative="greater",
    )

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, sharey=False)

    ax = axes[0]
    ax.set_title("(a) Plan-Execution Coupling (CCI)\nGSM, n=44", fontsize=10)
    ax.bar(x, cci, color=colors, alpha=0.85, width=0.55, edgecolor="white", lw=0.5)
    for i, (xi, med) in enumerate(zip(x, cci_m)):
        ax.hlines(med, xi - 0.22, xi + 0.22, colors="black", lw=2, zorder=5, label="Median" if i == 0 else "")
        if med == 0.0:
            ax.text(xi, 0.008, "med=0", ha="center", fontsize=7.5, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Mean CCI (numeric step agreement)", fontsize=9)
    ax.set_ylim(0, max(0.32, max(cci) * 1.15))
    ax.annotate("", xy=(1, cci[0]), xytext=(0, cci[0]), arrowprops=dict(arrowstyle="<->", color="black", lw=1))
    ax.text(0.5, cci[0] + 0.02, f"p={p_wilcox:.3f}", ha="center", fontsize=7.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.set_title("(b) Trajectory Error Propagation (TEP)\nGSM, n=44", fontsize=10)
    ax2.bar(x, tep, color=colors, alpha=0.85, width=0.55, edgecolor="white", lw=0.5)
    ax2b = ax2.twinx()
    ax2b.plot(x, acc, "k--o", lw=1.5, ms=5, label="Phase-2 accuracy", zorder=5)
    ax2b.set_ylabel("GSM session accuracy", fontsize=9, color="black")
    ax2b.set_ylim(0.2, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("Mean TEP", fontsize=10)
    ax2.set_ylim(0, 0.95)
    ax2b.legend(fontsize=8, loc="upper left", framealpha=0.7)
    ax2.spines[["top"]].set_visible(False)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(FIG / f"fig3_cci_tep.{ext}", bbox_inches="tight")
    print(f"Figure 3 saved to {FIG} (Wilcoxon p={p_wilcox:.4f})")


if __name__ == "__main__":
    main()
