from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import GREEN, GRAY, MODEL_LABELS, MODEL_ORDER, P1_ACCENT_BLUE, P1_PRIMARY_BLUE, P3_GOLD
from analysis.figures._common import load_behavioral, output_dir


def classify_row(raw: str, ok: bool):
    txt = (raw or "").strip()
    if not txt:
        return "refused_empty"
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if ok:
        return "valid_correct_goal"
    if not lines:
        return "refused_empty"
    # Approx proxy without full simulator: first line looks action-like?
    first = lines[0].lower()
    verbs = ("pick-up", "put-down", "stack", "unstack", "attack", "succumb", "overcome", "broker", "feast")
    if any(first.startswith(v) for v in verbs):
        # If many steps likely semantically valid but wrong goal.
        return "valid_wrong_goal" if len(lines) > 3 else "invalid_le3"
    return "invalid_le3"


def main():
    df = load_behavioral()
    cats = [
        "valid_correct_goal",
        "valid_wrong_goal",
        "invalid_le3",
        "invalid_gt3",
        "refused_empty",
    ]
    colors = {
        "valid_correct_goal": GREEN,
        "valid_wrong_goal": P3_GOLD,
        "invalid_le3": P1_PRIMARY_BLUE,
        "invalid_gt3": P1_ACCENT_BLUE,
        "refused_empty": GRAY,
    }
    labels = {
        "valid_correct_goal": "Valid plan + correct goal",
        "valid_wrong_goal": "Valid plan + wrong goal",
        "invalid_le3": "Invalid at step ≤ 3",
        "invalid_gt3": "Invalid at step > 3",
        "refused_empty": "Refused / empty",
    }

    per_model = {}
    for m in MODEL_ORDER:
        d = df[df["model"] == m].copy()
        counts = {k: 0 for k in cats}
        for _, r in d.iterrows():
            ok = str(r.get("behavioral_correct", "")).strip().lower() == "true"
            raw = str(r.get("raw_response", "") or "")
            cat = classify_row(raw, ok)
            if cat == "invalid_le3":
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                if len(lines) > 3:
                    cat = "invalid_gt3"
            counts[cat] += 1
        total = sum(counts.values()) or 1
        per_model[m] = {k: counts[k] / total for k in cats}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODEL_ORDER))
    bottom = np.zeros(len(MODEL_ORDER))
    for c in cats:
        vals = [per_model[m][c] for m in MODEL_ORDER]
        ax.bar(x, vals, bottom=bottom, color=colors[c], label=labels[c])
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of responses")
    ax.set_title("Figure 5 — Failure Mode Taxonomy")
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = output_dir()
    fig.savefig(out / "BW_FIG_P2_failure_taxonomy.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "BW_FIG_P2_failure_taxonomy.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
