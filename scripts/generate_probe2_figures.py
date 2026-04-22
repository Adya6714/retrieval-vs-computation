import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
import os

os.makedirs("results/figures", exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────
P2_PRIMARY  = "#6D28D9"
P2_ACCENT   = "#8B5CF6"
P2_LIGHT    = "#EDE9FE"
P3_PRIMARY  = "#B91C1C"
P3_ACCENT   = "#EF4444"
GREEN       = "#10B981"
GREEN_DARK  = "#065F46"
AMBER       = "#F59E0B"
GRAY        = "#9CA3AF"
DIFFICULTY_COLORS = {"easy": GREEN, "medium": AMBER, "hard": P3_ACCENT}
MODEL_LABELS = {
    "anthropic/claude-3.7-sonnet": "Claude 3.7",
    "openai/gpt-4o":               "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
    "meta-llama/llama-3-8b-instruct":   "Llama 3 8B",
}

def short_model(m):
    return MODEL_LABELS.get(m, m.split("/")[-1])


# ── Load data ──────────────────────────────────────────────────────────────
cci  = pd.read_csv("results/probe2a_cci.csv")
tep  = pd.read_csv("results/probe2b_tep.csv")
val  = pd.read_csv("results/probe2_validity_comparison.csv")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Phase 1 vs Phase 2 validity + CCI grouped bar chart
# ══════════════════════════════════════════════════════════════════════════
def fig1_validity_comparison():
    models = [m for m in cci["model"].unique() if m in MODEL_LABELS]
    x      = np.arange(len(models))
    width  = 0.25

    p1_vals, p2_vals, cci_vals = [], [], []
    for m in models:
        vrow = val[val["model"] == m]
        p1_vals.append(vrow["p1_validity_rate"].mean() if len(vrow) else 0)
        p2_vals.append(vrow["p2_validity_rate"].mean() if len(vrow) else 0)
        cci_vals.append(cci[cci["model"] == m]["cci"].mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width, p1_vals, width, label="Phase 1 validity",
                color=P2_PRIMARY, alpha=0.9)
    b2 = ax.bar(x,          p2_vals, width, label="Phase 2 validity",
                color=P2_ACCENT, alpha=0.9)
    b3 = ax.bar(x + width,  cci_vals, width, label="CCI",
                color=P2_LIGHT, alpha=0.9, edgecolor=P2_PRIMARY, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in models], fontsize=11)
    ax.set_ylabel("Rate (0–1)", fontsize=11)
    ax.set_title("Figure 1 — Phase 1 vs Phase 2 Validity & CCI by Model\n"
                 "(Probe 2: Plan–Execution Coupling)", fontsize=12)
    ax.set_ylim(0, max(max(p1_vals) * 1.3, 0.05))
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    for bar in [b1, b2, b3]:
        for rect in bar:
            h = rect.get_height()
            if h > 0.001:
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/figures/fig1_validity_comparison.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig1_validity_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 1")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Action type distribution heatmap
# ══════════════════════════════════════════════════════════════════════════
def fig2_action_heatmap():
    ACTION_CATS = ["pick-up", "put-down", "stack", "unstack", "other"]

    def categorize(action):
        a = str(action).strip().lower()
        for cat in ["pick-up", "put-down", "stack", "unstack"]:
            if a.startswith(cat):
                return cat
        return "other"

    models = [m for m in cci["model"].unique() if m in MODEL_LABELS]
    fig, axes = plt.subplots(1, len(models),
                             figsize=(5 * len(models), 7), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        mrows = cci[cci["model"] == model].copy()
        pids  = sorted(mrows["problem_id"].unique())
        matrix = []
        for pid in pids:
            row  = mrows[mrows["problem_id"] == pid].iloc[0]
            val = row.get("executed_steps_json")
            if isinstance(val, str) and val.strip():
                try:
                    acts = json.loads(val)
                except Exception:
                    acts = []
            else:
                acts = []
            counts = Counter(categorize(a) for a in acts)
            total  = len(acts) if acts else 1
            matrix.append([counts.get(c, 0) / total for c in ACTION_CATS])

        mat = np.array(matrix)
        im  = ax.imshow(mat, aspect="auto", vmin=0, vmax=1,
                        cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                            "p2", ["white", P2_PRIMARY]))
        ax.set_xticks(range(len(ACTION_CATS)))
        ax.set_xticklabels(ACTION_CATS, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels(pids, fontsize=8)
        ax.set_title(short_model(model), fontsize=11)

        for i in range(len(pids)):
            for j in range(len(ACTION_CATS)):
                val_cell = mat[i, j]
                color = "white" if val_cell > 0.5 else "black"
                ax.text(j, i, f"{val_cell:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.suptitle("Figure 2 — Action Type Distribution Heatmap (Phase 2)\n"
                 "Fraction of steps per action category", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/figures/fig2_action_heatmap.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig2_action_heatmap.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 2")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — TEP cascade for BW_E problems where injection engaged
# ══════════════════════════════════════════════════════════════════════════
def fig3_tep_cascade():
    CLASS_Y = {"adapted": 3, "ambiguous": 2, "resistant": 1,
               "illegal_both": 0}
    CLASS_LABELS = {0: "illegal_both", 1: "resistant",
                    2: "ambiguous", 3: "adapted"}

    real_injections = tep[
        tep["injection_desc"].notna() &
        ~tep["injection_desc"].str.startswith("NO ERROR")
    ]
    if len(real_injections) == 0:
        print("Figure 3 skipped — no real injections found in TEP data")
        return

    models = real_injections["model"].unique()
    pids   = real_injections["problem_id"].unique()[:3]  # max 3 problems

    fig, axes = plt.subplots(len(pids), 1,
                             figsize=(10, 4 * len(pids)), squeeze=False)

    injection_colors = [P2_PRIMARY, P2_ACCENT, AMBER, GREEN]

    for row_idx, pid in enumerate(pids):
        ax = axes[row_idx][0]
        pid_rows = real_injections[real_injections["problem_id"] == pid]

        for color_idx, (_, r) in enumerate(pid_rows.iterrows()):
            val = r.get("cascade_sequence_json")
            if isinstance(val, str) and val.strip():
                try:
                    seq = json.loads(val)
                except Exception:
                    seq = []
            else:
                seq = []
            if not seq:
                continue
            steps = [s["step"] for s in seq]
            ys    = [CLASS_Y.get(s["classification"], 0) for s in seq]
            k     = int(r["inject_at_step"])
            col   = injection_colors[color_idx % len(injection_colors)]
            ax.plot(steps, ys, "o-", color=col, alpha=0.8, linewidth=1.5,
                    markersize=5,
                    label=f"inject@step{k} ({r['injection_desc'][:30]})")
            ax.axvline(x=k, color=P3_PRIMARY, linestyle="--",
                       alpha=0.6, linewidth=1)

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["illegal_both", "resistant",
                            "ambiguous", "adapted"], fontsize=9)
        ax.set_xlabel("Execution step", fontsize=10)
        ax.set_title(f"{pid} — TEP cascade by injection point", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure 3 — Trajectory Error Propagation Cascade\n"
                 "Model behavior after state injection (red dashed = injection)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("results/figures/fig3_tep_cascade.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig3_tep_cascade.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 3")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Repetition Rate by model and difficulty
# ══════════════════════════════════════════════════════════════════════════
def fig4_repetition_rate():
    if "repetition_rate" not in cci.columns:
        print("Figure 4 skipped — repetition_rate column not in CCI data yet")
        return

    models = [m for m in cci["model"].unique() if m in MODEL_LABELS]
    diffs  = ["easy", "medium", "hard"]
    x      = np.arange(len(diffs))
    width  = 0.25
    colors = [P2_PRIMARY, P2_ACCENT, P2_LIGHT]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        mdata = cci[cci["model"] == model]
        vals  = []
        for d in diffs:
            sub = mdata[mdata["difficulty"] == d]["repetition_rate"].dropna()
            vals.append(sub.mean() if len(sub) else 0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      label=short_model(model),
                      color=colors[i % len(colors)],
                      alpha=0.9,
                      edgecolor=P2_PRIMARY if colors[i] == P2_LIGHT else "none",
                      linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Medium", "Hard"], fontsize=11)
    ax.set_ylabel("Repetition Rate (0–1)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color=GRAY, linestyle=":", linewidth=1)
    ax.set_title("Figure 4 — Repetition Rate by Model and Difficulty\n"
                 "(RR=1.0 means model repeated same action every step)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/figures/fig4_repetition_rate.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig4_repetition_rate.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 4")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Precondition violation breakdown stacked bar
# ══════════════════════════════════════════════════════════════════════════
def fig5_violation_breakdown():
    viol_cols = [
        "violation_hand_not_empty",
        "violation_block_not_clear",
        "violation_block_not_on_table",
        "violation_wrong_stack_source",
        "violation_target_not_clear",
        "violation_format_error",
        "violation_other",
    ]
    if not all(c in cci.columns for c in viol_cols):
        print("Figure 5 skipped — violation columns not in CCI data yet")
        return

    viol_labels = [
        "Hand not empty",
        "Block not clear",
        "Block not on table",
        "Wrong unstack source",
        "Target not clear",
        "Format error",
        "Other illegal",
    ]
    viol_colors = [
        P2_PRIMARY, P2_ACCENT, "#A78BFA",
        "#7C3AED", "#5B21B6", GRAY, P2_LIGHT
    ]

    models = [m for m in cci["model"].unique() if m in MODEL_LABELS]
    fig, axes = plt.subplots(1, len(models),
                             figsize=(6 * len(models), 6), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax_idx, model in enumerate(models):
        mdata = cci[cci["model"] == model].copy()
        pids  = sorted(mdata["problem_id"].unique())
        bottoms = np.zeros(len(pids))

        for col, label, color in zip(viol_cols, viol_labels, viol_colors):
            vals = [
                mdata[mdata["problem_id"] == pid][col].sum()
                for pid in pids
            ]
            totals = [
                mdata[mdata["problem_id"] == pid]["executed_length"].sum()
                for pid in pids
            ]
            fracs = [v / t if t > 0 else 0 for v, t in zip(vals, totals)]
            axes[ax_idx].barh(range(len(pids)), fracs, left=bottoms,
                             color=color, label=label, alpha=0.9)
            bottoms += np.array(fracs)

        # Valid steps
        valid_fracs = [
            mdata[mdata["problem_id"] == pid]["p2_valid_steps"].sum() /
            max(mdata[mdata["problem_id"] == pid]["executed_length"].sum(), 1)
            for pid in pids
        ] if "p2_valid_steps" in mdata.columns else [0] * len(pids)
        axes[ax_idx].barh(range(len(pids)), valid_fracs, left=bottoms,
                         color=GREEN, label="Valid", alpha=0.9)

        axes[ax_idx].set_yticks(range(len(pids)))
        axes[ax_idx].set_yticklabels(pids, fontsize=8)
        axes[ax_idx].set_xlabel("Fraction of steps", fontsize=10)
        axes[ax_idx].set_title(short_model(model), fontsize=11)
        axes[ax_idx].spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(viol_colors + [GREEN],
                               viol_labels + ["Valid"])]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Figure 5 — Precondition Violation Profile per Problem\n"
                 "(Probe 2: mechanistic failure breakdown)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/figures/fig5_violation_breakdown.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig5_violation_breakdown.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 5")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — First Illegal Step distribution (violin + strip)
# ══════════════════════════════════════════════════════════════════════════
def fig6_first_illegal_step():
    if "first_illegal_step" not in cci.columns:
        print("Figure 6 skipped — first_illegal_step column not in CCI data")
        return

    models = [m for m in cci["model"].unique() if m in MODEL_LABELS]
    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_model = []
    labels = []
    for m in models:
        vals = cci[cci["model"] == m]["first_illegal_step"].dropna().tolist()
        data_by_model.append(vals)
        labels.append(short_model(m))

    parts = ax.violinplot(data_by_model, positions=range(len(models)),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(P2_ACCENT)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color(P2_PRIMARY)
    parts["cmedians"].set_linewidth(2)

    for i, vals in enumerate(data_by_model):
        jitter = np.random.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=P2_PRIMARY, alpha=0.5, s=25, zorder=3)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Step index of first illegal action", fontsize=11)
    ax.set_title("Figure 6 — First Illegal Step Distribution\n"
                 "(median ≈ 1 = hand-state tracking failure from step 1)",
                 fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/figures/fig6_first_illegal_step.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig("results/figures/fig6_first_illegal_step.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved Figure 6")


# ══════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Probe 2 figures...")
    fig1_validity_comparison()
    fig2_action_heatmap()
    fig3_tep_cascade()
    fig4_repetition_rate()
    fig5_violation_breakdown()
    fig6_first_illegal_step()
    print(f"\nAll figures saved to results/figures/")

