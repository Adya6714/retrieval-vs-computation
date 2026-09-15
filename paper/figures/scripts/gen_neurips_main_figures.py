#!/usr/bin/env python3
"""Generate NeurIPS main-text figures fig1–fig4 from derived CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
OUT = REPO_ROOT / "paper" / "figures"
RAW = REPO_ROOT / "results" / "raw"

MODEL_MAP = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}
COLORS = {
    "Claude": "#1f77b4",
    "GPT-4o": "#ff7f0e",
    "Gemini": "#2ca02c",
    "Llama": "#d62728",
    "o4-mini": "#9467bd",
    "DeepSeek": "#8c564b",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def fig1_confounds() -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))

    # (a) oracle deltas
    ax = axes[0]
    ora = pd.read_csv(DER / "oracle_bias_summary.csv")
    order = ["BW_W3_action_mapping", "SP_W3_node_mapping", "BW_state_parser"]
    labels = {
        "BW_W3_action_mapping": "BW W3 map",
        "SP_W3_node_mapping": "SP W3 map",
        "BW_state_parser": "BW parser",
    }
    ora["ord"] = ora["defect"].map({k: i for i, k in enumerate(order)})
    ora = ora.sort_values("ord")
    x = np.arange(len(ora))
    w = 0.35
    ax.bar(x - w / 2, ora["delta_perturbed"], width=w, color="#4C78A8", label=r"$\Delta$ perturbed")
    ax.bar(x + w / 2, ora["delta_canonical"], width=w, color="#F58518", label=r"$\Delta$ canonical")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[d] for d in ora["defect"]], rotation=15, ha="right")
    ax.set_ylabel("Accuracy delta")
    ax.set_title("(a) Oracle repair")
    ax.legend(frameon=False, loc="upper right")

    # (b) difficulty slope + accuracy overlay
    ax = axes[1]
    rep = pd.read_csv(DER / "K3_bw_canonical_w6_matched_report.csv")
    struct = rep[rep["section"] == "structural_matched"]
    metrics = [
        ("num_blocks", "blocks"),
        ("n_goal_clauses", "clauses"),
        ("goal_tower_depth", "goal depth"),
        ("fd_optimal_plan_length", "opt. plan"),
    ]
    for i, (m, lab) in enumerate(metrics):
        row = struct[struct["metric"] == m].iloc[0]
        ax.plot([0, 1], [row["canonical_mean"], row["w6_mean"]], "-o", color="0.35", ms=4, lw=1.2)
        ax.text(-0.02, row["canonical_mean"], lab, ha="right", va="center", fontsize=7, color="0.3")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Canonical", "Regenerated"])
    ax.set_ylabel("Structural mean")
    ax.set_title("(b) BW difficulty (47 pairs)")
    ax2 = ax.twinx()
    acc = rep[rep["section"] == "accuracy_matched"]
    for _, r in acc.iterrows():
        c = COLORS.get(str(r["model"]), "0.5")
        ax2.plot(
            [0, 1],
            [r["canonical_mean"], r["w6_mean"]],
            "--",
            color=c,
            lw=1.0,
            alpha=0.85,
            label=str(r["model"]),
        )
    ax2.set_ylabel("Accuracy", color="0.25")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.12, 0.5))

    # (c) iid vs cluster interval widths
    ax = axes[2]
    i2 = pd.read_csv(DER / "I2_algo_cluster_bootstrap.csv")
    sub = i2[i2["pool"].astype(str).str.contains("bank110|table7|frozen", case=False, na=False)].copy()
    sub = sub[(sub["iid_width"] > 0) & (sub["cluster_width"] > 0)]
    ax.scatter(sub["iid_width"], sub["cluster_width"], s=12, alpha=0.55, c="#4C78A8", edgecolors="none")
    lim = max(sub["iid_width"].max(), sub["cluster_width"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ":", color="0.5", lw=1)
    med = float(sub["width_ratio_cluster_over_iid"].median())
    ax.text(0.05, 0.95, f"median width ratio {med:.2f}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("IID interval width")
    ax.set_ylabel("Cluster interval width")
    ax.set_title("(c) Clone inflation (ALGO)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_confounds.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_confounds.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_confounds")


def fig2_bw_inversion() -> None:
    _style()
    inst = pd.read_csv(DER / "K3_bw_canonical_w6_instances.csv")
    matched = inst[~inst["byte_identical_text"].astype(str).str.lower().isin(["true", "1"])].copy()
    assert len(matched) == 47, len(matched)
    models = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini", "DeepSeek"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rng = np.random.default_rng(42)
    for m in models:
        can = matched[f"{m}_canonical_correct"].astype(str).str.lower().isin(["true", "1"]).astype(float)
        w6 = matched[f"{m}_w6_correct"].astype(str).str.lower().isin(["true", "1"]).astype(float)
        delta = w6 - can
        x = matched["canonical_fd_optimal_plan_length"].astype(float) + rng.normal(0, 0.12, size=len(matched))
        y = delta + rng.normal(0, 0.03, size=len(matched))
        ax.scatter(x, y, s=28, alpha=0.75, label=m, c=COLORS[m], edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Canonical optimal plan length (Fast Downward)")
    ax.set_ylabel(r"Accuracy delta (regenerated $-$ canonical)")
    ax.set_title("Blocksworld 47-pair inversion")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["can only", "tie / both", "regen only"])
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_bw_inversion.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_bw_inversion.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_bw_inversion")


def fig3_locus() -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))

    fp = pd.read_csv(DER / "P1_failure_patterns.csv")
    sh = fp[
        (fp["section"] == "shared_hard_canonical")
        & (fp["key"] == "fail_all_five_paper_models")
    ].copy()
    ax = axes[0]
    order = ["ALGO", "GSM", "BW"]
    counts = []
    dens = []
    for fam in order:
        row = sh[sh["family"] == fam]
        c = int(row["count"].iloc[0]) if not row.empty else 0
        n = int(row["n_problems"].iloc[0]) if not row.empty else 0
        counts.append(c)
        dens.append(f"{c}/{n}")
    bars = ax.bar(order, counts, color=["#4C78A8", "#72B7B2", "#F58518"])
    for b, lab in zip(bars, dens):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, lab, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Shared-hard count (all 5 models fail)")
    ax.set_title("(a) Shared-hard canonical items")
    ax.set_ylim(0, max(counts + [1]) * 1.35)

    ax = axes[1]
    phi = pd.read_csv(DER / "P1_phi_canonical_w3.csv")
    families = ["ALGO", "GSM", "BW"]
    models = ["Claude", "GPT-4o", "Gemini", "Llama", "o4-mini"]
    mat = np.full((len(families), len(models)), np.nan)
    lo = np.full_like(mat, np.nan)
    hi = np.full_like(mat, np.nan)
    for i, fam in enumerate(families):
        for j, m in enumerate(models):
            row = phi[(phi["family"] == fam) & (phi["model"] == m)]
            if row.empty:
                continue
            mat[i, j] = float(row["phi"].iloc[0]) if str(row["phi"].iloc[0]) not in {"", "nan"} else np.nan
            if str(row["phi_ci_low"].iloc[0]) not in {"", "nan"}:
                lo[i, j] = float(row["phi_ci_low"].iloc[0])
                hi[i, j] = float(row["phi_ci_high"].iloc[0])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)
    for i in range(len(families)):
        for j in range(len(models)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)
                continue
            txt = f"{mat[i, j]:.2f}"
            if not np.isnan(lo[i, j]):
                txt += f"\n[{lo[i, j]:.2f},{hi[i, j]:.2f}]"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="black")
    ax.set_title(r"(b) $\phi$ (canonical vs $W_3$) with CIs")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_locus.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_locus.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_locus")


def _load_gsm_p1() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("GSM_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[df["included"].astype(str).str.lower().isin(["true", "1"])]
        df = df[df["model"].isin(MODEL_MAP)]
        df["variant"] = df["variant_type"].map(normalize_variant)
        df["ok"] = df["rescored_correct"].astype(str).str.lower().isin(["true", "1"])
        df["model_short"] = df["model"].map(MODEL_MAP)
        parts.append(df)
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["problem_id", "variant", "model_short"], keep="last"
    )


def fig4_construct() -> None:
    _style()
    p2_path = RAW / "GSM_P2_cci.csv"
    if not p2_path.exists():
        raise FileNotFoundError(p2_path)
    p2 = pd.read_csv(p2_path, dtype=str).fillna("")
    p2["model_short"] = p2["model"].map(MODEL_MAP)
    p2["cci"] = pd.to_numeric(p2["cci_score"], errors="coerce")
    p1 = _load_gsm_p1()
    w3 = p1[p1["variant"] == "W3"][["problem_id", "model_short", "ok"]].rename(columns={"ok": "w3_ok"})
    merged = p2.merge(w3, on=["problem_id", "model_short"], how="inner")
    merged = merged[merged["cci"].notna()].copy()
    merged["w3_correct"] = merged["w3_ok"].astype(int)
    n = len(merged)
    assert n == 128, f"expected n=128 GSM CCI×W3 pairs, got {n}"

    conv = pd.read_csv(DER / "P2_P1_convergence.csv")
    row = conv[
        (conv["analysis"] == "pointbiserial_cci_w3_correct")
        & (conv["scope"] == "GSM_all_instances_with_p2")
    ].iloc[0]
    r = float(row["statistic"])
    p = float(row["p_value"])
    ci_lo, ci_hi = float(row["ci_low"]), float(row["ci_high"])

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # jitter binary y for visibility
    rng = np.random.default_rng(0)
    y = merged["w3_correct"].to_numpy(dtype=float) + rng.normal(0, 0.03, size=n)
    x = merged["cci"].to_numpy(dtype=float)
    ax.scatter(x, y, s=22, alpha=0.55, c="#4C78A8", edgecolors="white", linewidths=0.3)

    # logistic-style linear fit on continuous for display band (OLS on 0/1)
    slope, intercept, _, _, _ = stats.linregress(x, merged["w3_correct"].to_numpy(dtype=float))
    xs = np.linspace(max(0, x.min() - 0.02), min(1, x.max() + 0.02), 100)
    ys = intercept + slope * xs
    # residual bootstrap band
    resid = merged["w3_correct"].to_numpy(dtype=float) - (intercept + slope * x)
    band = []
    for _ in range(1000):
        sample = intercept + slope * xs + rng.choice(resid, size=len(xs), replace=True)
        band.append(sample)
    band = np.asarray(band)
    lo = np.percentile(band, 2.5, axis=0)
    hi = np.percentile(band, 97.5, axis=0)
    ax.fill_between(xs, lo, hi, color="#4C78A8", alpha=0.2, linewidth=0)
    ax.plot(xs, ys, color="#E45756", lw=1.8)
    ax.set_xlabel("CCI (plan–execution consistency)")
    ax.set_ylabel("Rename survival ($W_3$ correct)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fail", "survive"])
    ax.set_title("CCI vs rename survival")
    ax.text(
        0.03,
        0.97,
        f"$r={r:.3f}$ [{ci_lo:.3f},{ci_hi:.3f}]\n$p={p:.3f}$, $n={n}$",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8"),
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig4_construct.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_construct.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig4_construct")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_confounds()
    fig2_bw_inversion()
    fig3_locus()
    fig4_construct()


if __name__ == "__main__":
    main()
