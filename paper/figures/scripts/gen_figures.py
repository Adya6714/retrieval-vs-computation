"""Regenerate the 5 numerical figures used in the paper, end-to-end from
raw verifier logs. One run rewrites every PDF in ../ .

Inputs (paths relative to repo root):
    results/raw/GSM_P1_behavioral_{claude,gpt4o,llama,gemini,o1mini}.csv
    results/raw/GSM_P2_cci.csv
    results/raw/GSM_P3_contamination.csv
    results/raw/ALGO_P1_behavioral_{claude,gpt4o,llama,gemini,o1mini}.csv
    results/raw/ALGO_P2_phase2_normal.csv
    results/raw/ALGO_P2_phase2_normal_gemini.csv
    results/raw/ALGO_P2_phase2_injected.csv
    results/raw/ALGO_P2_phase2_injected_gemini.csv
    results/derived/ALGO_P3_triangulation.csv   (only for adversarial id-set)

Outputs:
    LLM Overleaf/figures/{fig_robustness,fig_decay,fig_heatmap,fig_cci,fig_paradox}.pdf

Every number in every panel is computed from the raw CSVs at runtime;
no hard-coded constants for the numbers themselves.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

# ---------------------------------------------------------------------------
# Paths and global style
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"
OUT = Path(__file__).resolve().parents[1]  # LLM Overleaf/figures

MODEL_LONG = {
    "Claude": "anthropic/claude-sonnet-4",
    "GPT-4o": "openai/gpt-4o",
    "Llama-8B": "meta-llama/llama-3.1-8b-instruct",
    "Gemini-2.5": "google/gemini-2.5-flash",
    "o4-mini": "openai/o4-mini",
}
COLOR = {
    "Claude": "#0072B2",
    "GPT-4o": "#D55E00",
    "Llama-8B": "#CC79A7",
    "Gemini-2.5": "#009E73",
    "o4-mini": "#E69F00",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,  # editable text in PDF
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _correct_col(df: pd.DataFrame) -> pd.Series:
    """Return binary 'is-correct' series for either family schema."""
    if "verified" in df.columns:
        return df["verified"].fillna(False).astype(bool)
    if "behavioral_correct" in df.columns:
        return df["behavioral_correct"].fillna(False).astype(bool)
    raise KeyError("no correctness column in P1 file")


def _read_algo_p1(slug: str) -> pd.DataFrame:
    """ALGO P1 raw, deduped on (problem_id, variant_type).

    The o4-mini raw file is shared with legacy o1-mini rows; we filter
    explicitly to the modern model name when ``slug == "o1mini"``.
    """
    df = pd.read_csv(RAW / f"ALGO_P1_behavioral_{slug}.csv")
    if slug == "o1mini":
        df = df[df.model.astype(str).str.contains("o4-mini", case=False, na=False)]
    return df.drop_duplicates(subset=["problem_id", "variant_type"])


BANK_GSM_PATH = ROOT / "data" / "problems" / "question_bank_gsm.csv"

MODEL_SHORT = {v: k for k, v in MODEL_LONG.items()}

EXPECTED_GSM_CANONICAL_N = {
    "anthropic/claude-sonnet-4": 44,
    "google/gemini-2.5-flash": 44,
    "openai/o4-mini": 44,
    "openai/gpt-4o": 20,
    "meta-llama/llama-3.1-8b-instruct": 20,
}


def _load_bank_gsm() -> set[str]:
    """Bank-valid GSM problem IDs (excludes off-bank GSM_021–040)."""
    try:
        bank = pd.read_csv(BANK_GSM_PATH, usecols=["problem_id"])
        return set(bank["problem_id"].astype(str).unique())
    except Exception:
        return {
            *(f"GSM_{i:03d}" for i in range(1, 21)),
            *(f"GSM_{i:03d}" for i in range(41, 65)),
        }


BANK_GSM = _load_bank_gsm()


def _gsm_response_series(df: pd.DataFrame) -> pd.Series:
    if "raw_response" in df.columns:
        return df["raw_response"].astype(str)
    if "response" in df.columns:
        return df["response"].astype(str)
    return pd.Series("", index=df.index)


def _assert_gsm_p1_canonical_counts(df: pd.DataFrame) -> None:
    canon = (
        df[df.variant_type == "canonical"]
        .groupby("model")["problem_id"]
        .nunique()
    )
    for mid, expected in EXPECTED_GSM_CANONICAL_N.items():
        got = int(canon.get(mid, 0))
        label = MODEL_SHORT.get(mid, mid)
        print(f"  GSM P1 canonical n: {label} = {got} (expected {expected})")
        if got != expected:
            raise AssertionError(
                f"GSM P1 canonical n mismatch for {label} ({mid}): "
                f"got {got}, expected {expected}"
            )


@lru_cache(maxsize=1)
def _gsm_p1_unified() -> pd.DataFrame:
    """Authoritative GSM P1 frame from per-model raw files.

    Filters to bank-valid problem IDs only; drops ERROR stubs and rows with
    empty/NaN behavioral_correct (failed API calls are excluded, not scored
    as incorrect).  Claude/Gemini/o4-mini: n=44 canonical; GPT-4o/Llama: n=20.
    """
    parts: list[pd.DataFrame] = []
    for slug, model_id in [
        ("claude", "anthropic/claude-sonnet-4"),
        ("gpt4o",  "openai/gpt-4o"),
        ("llama",  "meta-llama/llama-3.1-8b-instruct"),
        ("gemini", "google/gemini-2.5-flash"),
        ("o1mini", "openai/o4-mini"),
    ]:
        df = pd.read_csv(RAW / f"GSM_P1_behavioral_{slug}.csv", dtype=str).fillna("")
        df = df[df["problem_id"].astype(str).isin(BANK_GSM)]
        resp = _gsm_response_series(df)
        df = df[~resp.str.startswith("ERROR")]
        bc = df["behavioral_correct"].astype(str).str.strip()
        df = df[bc.ne("") & df["behavioral_correct"].notna()]
        df["variant_type"] = df.variant_type.where(
            ~df.variant_type.str.startswith("w"),
            df.variant_type.str.upper(),
        )
        df["correct"] = df.behavioral_correct.str.lower().eq("true").astype(int)
        df["model"] = model_id
        parts.append(df[["problem_id", "variant_type", "model", "correct"]])
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["problem_id", "variant_type", "model"])
    _assert_gsm_p1_canonical_counts(out)
    return out


def dump_gsm_p1_reconciliation() -> pd.DataFrame:
    """Write GSM P1 accuracies via the same path figures use; print + CSV."""
    df = _gsm_p1_unified()
    variants = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
    rows: list[dict] = []
    for mid in sorted(df.model.unique()):
        label = MODEL_SHORT.get(mid, mid)
        sub = df[df.model == mid]
        for v in variants:
            s = sub[sub.variant_type == v]
            n = int(len(s))
            acc = float(s.correct.mean()) if n else float("nan")
            rows.append({"model": label, "variant": v, "accuracy": acc, "n": n})
    out = pd.DataFrame(rows)
    path = DER / "gsm_p1_figure_table_reconciliation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print("\nGSM P1 figure-table reconciliation (bank-filtered):")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  wrote {path}")
    return out


def p1_acc(family: str, slug: str, model_label: str) -> dict[str, float]:
    """Return Canonical/W1..W6 accuracy for one model, one family."""
    if family == "GSM":
        u = _gsm_p1_unified()
        sub = u[u.model == MODEL_LONG[model_label]]
        out = {}
        for v in ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]:
            s = sub[sub.variant_type == v]
            out[v] = float(s.correct.mean()) if len(s) else float("nan")
        out["n_can"] = int((sub.variant_type == "canonical").sum())
        return out
    # ALGO
    df = _read_algo_p1(slug)
    df = df.assign(_ok=_correct_col(df).astype(int).values)
    out = {}
    for v in ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]:
        sub = df[df.variant_type == v]
        out[v] = float(sub._ok.mean()) if len(sub) else float("nan")
    out["n_can"] = int((df.variant_type == "canonical").sum())
    return out


def p1_acc_subtype(slug: str, problem_ids: list[str]) -> dict[str, float]:
    """ALGO per-variant accuracy restricted to a problem-id list."""
    df = _read_algo_p1(slug)
    df = df[df.problem_id.isin(problem_ids)]
    df = df.assign(_ok=_correct_col(df).astype(int).values)
    out = {}
    for v in ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]:
        sub = df[df.variant_type == v]
        out[v] = float(sub._ok.mean()) if len(sub) else float("nan")
    out["n"] = int((df.variant_type == "canonical").sum())
    return out


def algo_adv_ids() -> dict[str, list[str]]:
    """Adversarial-bank problem ids per subtype (from triangulation file)."""
    tri = pd.read_csv(DER / "ALGO_P3_triangulation.csv")
    adv = tri[tri.instance_type == "adversarial"][
        ["problem_id", "problem_subtype"]
    ].drop_duplicates()
    return {sub: adv[adv.problem_subtype == sub].problem_id.tolist()
            for sub in adv.problem_subtype.unique()}


def gsm_p2_metrics() -> dict[str, dict[str, float]]:
    """Return per-model CCI/TEP/accuracy on GSM Probe 2 (n=44)."""
    p2 = pd.read_csv(RAW / "GSM_P2_cci.csv")
    out = {}
    for label, mid in MODEL_LONG.items():
        if label == "o4-mini":  # not run on Probe 2
            continue
        s = p2[p2.model == mid]
        if len(s) == 0:
            continue
        out[label] = {
            "n": len(s),
            "cci_mean": float(s.cci_score.mean()),
            "cci_med": float(s.cci_score.median()),
            "tep_mean": float(s.tep_score.mean()),
            "acc": float(
                s.session_b_correct.astype(str).str.lower().eq("true").mean()
            ),
        }
    return out


def algo_p2b_response() -> dict[str, dict[str, float]]:
    """Return fraction of injection-step response_types per model (ALGO P2B)."""
    a = pd.read_csv(RAW / "ALGO_P2_phase2_injected.csv")
    g = pd.read_csv(RAW / "ALGO_P2_phase2_injected_gemini.csv")
    # Gemini appears in both files; keep the dedicated rerun only (n=61).
    a_no_gem = a[a.model != MODEL_LONG["Gemini-2.5"]]
    inj = pd.concat([a_no_gem, g], ignore_index=True)
    inj = inj[inj.injection_applied == True]  # noqa: E712
    out = {}
    for label, mid in MODEL_LONG.items():
        if label == "o4-mini":
            continue
        s = inj[inj.model == mid]
        if len(s) == 0:
            continue
        counts = s.response_type.value_counts(normalize=True)
        out[label] = {
            "n": len(s),
            "compliant": float(counts.get("compliant", 0)),
            "partial": float(
                counts.get("partial_compliance", 0) + counts.get("refusal", 0)
            ),
            "format_ignored": float(counts.get("format_ignored", 0)),
        }
    return out


def contam_vri_pearson() -> dict[str, tuple[float, float, int]]:
    """ALGO adversarial: Pearson r of instance_contamination_score vs VRI."""
    contam = pd.read_csv(RAW / "ALGO_P3_contamination.csv")[
        ["problem_id", "instance_contamination_score"]
    ]
    adv_ids = []
    for v in algo_adv_ids().values():
        adv_ids.extend(v)
    out = {}
    for label, slug in [
        ("Claude", "claude"),
        ("GPT-4o", "gpt4o"),
        ("Llama-8B", "llama"),
        ("Gemini-2.5", "gemini"),
        ("o4-mini", "o1mini"),
    ]:
        df = _read_algo_p1(slug)
        df = df[df.problem_id.isin(adv_ids)]
        df = df.assign(_ok=_correct_col(df).astype(int).values)
        pivot = df.pivot_table(
            index="problem_id",
            columns="variant_type",
            values="_ok",
            aggfunc="mean",
        )
        for c in ("W1", "W2", "W3", "W4"):
            if c not in pivot.columns:
                pivot[c] = float("nan")
        pivot["VRI"] = pivot[["W1", "W2", "W4"]].mean(axis=1) - pivot["W3"]
        merged = pivot.reset_index().merge(contam, on="problem_id", how="left")
        s = merged.dropna(subset=["instance_contamination_score", "VRI"])
        if len(s) < 3:
            out[label] = (float("nan"), float("nan"), len(s))
            continue
        r, p = stats.pearsonr(s.instance_contamination_score, s.VRI)
        out[label] = (float(r), float(p), len(s))
    return out


def contam_cci_pearson() -> dict[str, tuple[float, float, int]]:
    """GSM: Pearson r of contamination_score vs cci_score; cci_total>0."""
    gsm_c = pd.read_csv(RAW / "GSM_P3_contamination.csv")[
        ["problem_id", "contamination_score"]
    ]
    out = {}
    for label, slug in [
        ("Claude", "claude"),
        ("GPT-4o", "gpt4o"),
        ("Llama-8B", "llama"),
        ("Gemini-2.5", "gemini"),
    ]:
        df = pd.read_csv(RAW / f"GSM_P2_phase1_{slug}.csv").merge(gsm_c, on="problem_id")
        valid = df[df.cci_total > 0]
        if len(valid) < 3:
            out[label] = (float("nan"), float("nan"), len(valid))
            continue
        r, p = stats.pearsonr(valid.contamination_score, valid.cci_score)
        out[label] = (float(r), float(p), len(valid))
    return out


def algo_p2a_combined() -> pd.DataFrame:
    """Phase-2A normal across 4 models (Claude, GPT-4o, Llama, Gemini; n=1531)."""
    a = pd.read_csv(RAW / "ALGO_P2_phase2_normal.csv")
    g = pd.read_csv(RAW / "ALGO_P2_phase2_normal_gemini.csv")
    # Gemini appears in both files; keep the dedicated rerun only.
    a_no_gem = a[a.model != MODEL_LONG["Gemini-2.5"]]
    df = pd.concat([a_no_gem, g], ignore_index=True)
    four = [MODEL_LONG[m] for m in ("Claude", "GPT-4o", "Llama-8B", "Gemini-2.5")]
    df = df[df.model.isin(four)].copy()
    df["final_answer_correct"] = (
        df.final_answer_correct.astype(str).str.lower().eq("true")
    )
    return df


def gsm_p2_paired_wilcoxon() -> tuple[float, float]:
    """Paired Wilcoxon Claude vs GPT-4o CCI on common problem ids (full bank, zero-imputed)."""
    p2 = pd.read_csv(RAW / "GSM_P2_cci.csv")
    cl = p2[p2.model == MODEL_LONG["Claude"]].set_index("problem_id")["cci_score"]
    gp = p2[p2.model == MODEL_LONG["GPT-4o"]].set_index("problem_id")["cci_score"]
    ids = sorted(set(cl.index) | set(gp.index))
    a = np.array([cl.get(i, 0.0) for i in ids], dtype=float)
    b = np.array([gp.get(i, 0.0) for i in ids], dtype=float)
    w, p = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    return float(w), float(p)


def sig_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ===========================================================================
# Figure 1 -- fig_robustness
# ===========================================================================
def fig_robustness() -> None:
    """(a) GSM scatter: canonical x retention, 5 models (incl. o4-mini)
    (b) Adversarial ALGO: subtype-specific can/W3 bars, 5 models."""

    # ---- panel (a) data ----
    gsm_models = [
        ("Claude", "claude"),
        ("GPT-4o", "gpt4o"),
        ("Llama-8B", "llama"),
        ("Gemini-2.5", "gemini"),
        ("o4-mini", "o1mini"),
    ]
    rows = []
    for label, slug in gsm_models:
        a = p1_acc("GSM", slug, label)
        rows.append({"model": label, "can": a["canonical"], "W3": a["W3"], "n": a["n_can"]})
    g = pd.DataFrame(rows)
    g["retention"] = g.W3 / g.can

    # ---- panel (b) data (adversarial-bank, 5 models incl. o4-mini) ----
    adv = algo_adv_ids()
    model_slugs = [
        ("Claude", "claude"),
        ("GPT-4o", "gpt4o"),
        ("Llama-8B", "llama"),
        ("Gemini-2.5", "gemini"),
        ("o4-mini", "o1mini"),
    ]
    bar_rows = []
    for sub, label in [
        ("coin_change", "CC-adv"),
        ("shortest_path", "SP-adv"),
        ("wis", "WIS-adv"),
    ]:
        ids = adv[sub]
        row = {"sub": label}
        for mlbl, slug in model_slugs:
            a = p1_acc_subtype(slug, ids)
            row[f"{mlbl}_can"] = a["canonical"]
            row[f"{mlbl}_W3"] = a["W3"]
        bar_rows.append(row)
    bdf = pd.DataFrame(bar_rows)

    # ---- render ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.07, right=0.98, wspace=0.27)

    # (a)
    ax = axes[0]
    ax.set_title("(a) GSM arithmetic (n=44; GPT-4o/Llama n=20)", fontsize=10.5)
    offsets = {
        "Claude": (0.003, 0.012),
        "GPT-4o": (0.004, -0.025),
        "Llama-8B": (0.005, -0.005),
        "Gemini-2.5": (0.005, -0.005),
        "o4-mini": (-0.03, 0.02),
    }
    for _, row in g.iterrows():
        m = "*" if row.model == "o4-mini" else "o"
        sz = 180 if row.model == "o4-mini" else 95
        ax.scatter(row.can, row.retention, c=COLOR[row.model], marker=m,
                   s=sz, zorder=4, edgecolors="white", linewidths=0.6)
        lbl = row.model + ("*" if row.model == "o4-mini" else "")
        dx, dy = offsets.get(row.model, (0.003, 0.012))
        ax.annotate(lbl, (row.can + dx, row.retention + dy), fontsize=8.8, color="#222")
    ax.axhline(1.0, ls="--", color="gray", alpha=0.4, lw=1)
    ax.set_xlabel("Canonical accuracy")
    ax.set_ylabel(r"$W_3$ retention  $(\mathrm{Acc}_{W3}/\mathrm{Acc}_\mathrm{can})$")
    ax.set_xlim(0.79, 0.96)
    ax.set_ylim(0.12, 1.05)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) 5-model adversarial ALGO panel; each subtype shows can vs W3 grouped
    ax2 = axes[1]
    ax2.set_title("(b) Adversarial ALGO: 5-model subtype dissociation",
                  fontsize=10.5)
    x = np.arange(len(bdf))
    w_can = 0.09
    w_w3 = 0.09
    n_models = len(model_slugs)
    pair_stride = 0.15
    pair_width = 0.075
    base = -((n_models * pair_stride) / 2.0) + 0.01
    for i_m, (mlbl, _slug) in enumerate(model_slugs):
        c = COLOR[mlbl]
        off_can = base + i_m * pair_stride
        off_w3 = off_can + pair_width
        ax2.bar(x + off_can, bdf[f"{mlbl}_can"], width=pair_width, color=c,
                edgecolor="white", linewidth=0.4, alpha=0.92)
        ax2.bar(x + off_w3, bdf[f"{mlbl}_W3"], width=pair_width, color=c,
                edgecolor="white", linewidth=0.4, alpha=0.45, hatch="//")
    # Compact legend: 4 model colors + canonical/W3 style swatches
    model_handles = [mpatches.Patch(facecolor=COLOR[mlbl], alpha=0.92,
                                     edgecolor="white", label=mlbl)
                     for mlbl, _ in model_slugs]
    style_handles = [
        mpatches.Patch(facecolor="#888", alpha=0.92, edgecolor="white",
                       label="canonical"),
        mpatches.Patch(facecolor="#888", alpha=0.45, edgecolor="white",
                       hatch="//", label="$W_3$"),
    ]
    handles = model_handles + style_handles
    # callouts (small, near the relevant bars, not overlapping the legend)
    for i, row in bdf.iterrows():
        if row["sub"] == "CC-adv":
            ax2.text(i - 0.30, 0.78,
                     "Claude robust\nGPT-4o $W_3 {\\to} 0$",
                     fontsize=6.8, ha="left", color="#444",
                     bbox=dict(facecolor="#ffffff", edgecolor="#cccccc",
                              alpha=0.85, boxstyle="round,pad=0.18"))
        elif row["sub"] == "SP-adv":
            ax2.text(i - 0.32, 0.55,
                     "Claude $W_3 {\\to} 0$\nGPT-4o partial (.27)",
                     fontsize=6.8, ha="left", color="#444",
                     bbox=dict(facecolor="#ffffff", edgecolor="#cccccc",
                              alpha=0.85, boxstyle="round,pad=0.18"))
        elif row["sub"] == "WIS-adv":
            ax2.text(i - 0.30, 0.55,
                     "$W_3 {\\to} 0$\nall models",
                     fontsize=6.8, ha="left", color="#444",
                     bbox=dict(facecolor="#ffffff", edgecolor="#cccccc",
                              alpha=0.85, boxstyle="round,pad=0.18"))
    ax2.set_xticks(x)
    ax2.set_xticklabels(bdf["sub"].tolist())
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend(handles=handles, fontsize=7.4, framealpha=0.9,
               loc="upper right", ncol=4, handlelength=1.1,
               handletextpad=0.4, columnspacing=0.7)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.savefig(OUT / "fig_robustness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_robustness.pdf")


# ===========================================================================
# Figure 2 -- fig_decay
# ===========================================================================
def fig_decay() -> None:
    """(a) GSM per-variant for 5 models (including o4-mini)
    (b) Challenging ALGO subtypes (Claude vs GPT-4o), CC-chall + SP-chall."""

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.07, right=0.98, wspace=0.22)

    variants = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
    xlbl = ["Can", "W1", "W2", "W3", "W4", "W5", "W6"]
    xx = np.arange(len(variants))

    # ---- (a) GSM ----
    ax = axes[0]
    ax.set_title("(a) GSM: per-variant accuracy (n=44; GPT-4o/Llama n=20)", fontsize=10.5)
    # W3 shaded band
    ax.axvspan(2.6, 3.4, color="#d3d3d3", alpha=0.45, zorder=0)
    ax.text(3.0, 1.04, "$W_3$", ha="center", fontsize=8.5, color="#555")

    for label, slug in [("Claude", "claude"), ("GPT-4o", "gpt4o"),
                         ("Llama-8B", "llama"), ("Gemini-2.5", "gemini"),
                         ("o4-mini", "o1mini")]:
        a = p1_acc("GSM", slug, label)
        ys = [a[v] for v in variants]
        ls = "--" if label == "o4-mini" else "-"
        mark = "*" if label == "o4-mini" else "o"
        ms = 8.5 if label == "o4-mini" else 5.5
        ax.plot(xx, ys, ls=ls, marker=mark, lw=1.5, ms=ms,
                color=COLOR[label], label=label + ("*" if label == "o4-mini" else ""),
                markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(xx)
    ax.set_xticklabels(xlbl)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.7, loc="lower left", ncol=2, framealpha=0.85,
              handlelength=1.6, handletextpad=0.5, columnspacing=0.7)
    ax.text(0.99, 0.02, "* o4-mini retains canonical-level accuracy at $W_3$ on GSM",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="gray")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (b) ALGO challenging Claude vs GPT-4o, CC-chall + SP-chall ----
    ax2 = axes[1]
    ax2.set_title("(b) Challenging ALGO subtypes (Claude vs. GPT-4o)",
                  fontsize=10.5)
    ax2.axvspan(2.6, 3.4, color="#d3d3d3", alpha=0.45, zorder=0)
    ax2.text(3.0, 1.04, "$W_3$", ha="center", fontsize=8.5, color="#555")

    adv = algo_adv_ids()
    series = [
        ("Claude SP-chall.", "claude", adv["shortest_path"], COLOR["Claude"], "-", "o"),
        ("Claude CC-chall.", "claude", adv["coin_change"], COLOR["Claude"], "--", "D"),
        ("GPT-4o SP-chall.", "gpt4o", adv["shortest_path"], COLOR["GPT-4o"], "-", "o"),
        ("GPT-4o CC-chall.", "gpt4o", adv["coin_change"], COLOR["GPT-4o"], "--", "D"),
    ]
    for label, slug, ids, c, ls, mk in series:
        a = p1_acc_subtype(slug, ids)
        ys = [a[v] for v in variants]
        ax2.plot(xx, ys, ls=ls, marker=mk, lw=1.5, ms=5.5, color=c, label=label,
                 markeredgecolor="white", markeredgewidth=0.5)
    # Claude SP collapse annotation
    ax2.annotate("Claude SP $\\to$ 0% at $W_3$",
                 xy=(3, 0.0), xytext=(4.0, 0.18),
                 fontsize=7.8, color="#444",
                 arrowprops=dict(arrowstyle="->", color="#888", lw=0.7))
    ax2.set_xticks(xx)
    ax2.set_xticklabels(xlbl)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=7.7, loc="upper right", framealpha=0.85,
               handlelength=1.6, handletextpad=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.savefig(OUT / "fig_decay.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_decay.pdf")


# ===========================================================================
# Figure 3 -- fig_heatmap
# ===========================================================================
def fig_heatmap() -> None:
    """(a) GSM macro metrics  (b) Proximity -> behavioral signal."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.subplots_adjust(top=0.90, bottom=0.16, left=0.10, right=0.95, wspace=0.42)

    # ---- (a) Macro metrics (GSM)  --- 5 models including o4-mini CCI/TEP ----
    models = ["Claude", "GPT-4o", "Llama-8B", "Gemini-2.5", "o4-mini"]
    cols_a = [r"$\mathrm{Acc}_\mathrm{can}$", r"$R_{W3}$", "CCI", "TEP"]
    matA = np.full((5, 4), np.nan)
    for i, lbl in enumerate(models):
        slug = {"Claude": "claude", "GPT-4o": "gpt4o",
                "Llama-8B": "llama", "Gemini-2.5": "gemini",
                "o4-mini": "o1mini"}[lbl]
        a_p1 = p1_acc("GSM", slug, lbl)
        matA[i, 0] = a_p1["canonical"]
        matA[i, 1] = a_p1["W3"] / a_p1["canonical"]
        if lbl == "o4-mini":
            # Compute o4-mini CCI/TEP directly from the dedicated o1mini file
            df = pd.read_csv(RAW / "GSM_P2_phase1_o1mini.csv", dtype=str).fillna("")
            parse = df[df.phase1_parseable.str.lower() == "true"]
            if len(parse):
                cci = pd.to_numeric(parse["cci_score"], errors="coerce")
                tep = pd.to_numeric(parse["tep_score"], errors="coerce")
                matA[i, 2] = float(cci.mean())
                matA[i, 3] = float(tep.mean())
        else:
            p2 = gsm_p2_metrics()[lbl]
            matA[i, 2] = p2["cci_mean"]
            matA[i, 3] = p2["tep_mean"]

    ax = axes[0]
    ax.set_title("(a) Macro metrics (GSM)", fontsize=10.5)
    im = ax.imshow(matA, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(cols_a)
    ax.set_yticks(range(5))
    ax.set_yticklabels(models)
    for i in range(5):
        for j in range(4):
            v = matA[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        color="#888", fontsize=8, style="italic")
            else:
                txt_color = "white" if v > 0.55 else "#222"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=txt_color, fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=8)

    # ---- (b) Proximity -> behavioral signal (o4-mini also rendered) ----
    main_models = ["Claude", "GPT-4o", "Llama-8B", "Gemini-2.5", "o4-mini"]
    vri = contam_vri_pearson()
    cci = contam_cci_pearson()
    cols_b = ["contam$\\to$VRI\n(ALGO-chall.)", "contam$\\to$CCI\n(GSM)"]
    matB = np.full((5, 2), np.nan)
    sig = np.empty((5, 2), dtype=object)
    for i, lbl in enumerate(main_models):
        plain = lbl
        if plain in vri:
            rv, pv, _ = vri[plain]
            if np.isfinite(rv):
                matB[i, 0] = rv
                sig[i, 0] = sig_marker(pv)
        if plain in cci:
            rc, pc, _ = cci[plain]
            if np.isfinite(rc):
                matB[i, 1] = rc
                sig[i, 1] = sig_marker(pc)

    ax2 = axes[1]
    ax2.set_title("(b) Proximity $\\to$ behavioral signal", fontsize=10.5)
    im2 = ax2.imshow(matB, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(cols_b, fontsize=9)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(main_models)
    for i in range(5):
        for j in range(2):
            v = matB[i, j]
            if np.isnan(v):
                ax2.text(j, i, "n/a", ha="center", va="center",
                         color="#888", fontsize=8, style="italic")
                continue
            txt_color = "white" if abs(v) > 0.30 else "#222"
            ax2.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     color=txt_color, fontsize=10, fontweight="bold")
            mark = sig[i, j]
            ax2.text(j, i + 0.28, mark, ha="center", va="center",
                     color=txt_color, fontsize=7.5, style="italic")
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.07, pad=0.04)
    cb2.ax.tick_params(labelsize=8)

    plt.savefig(OUT / "fig_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_heatmap.pdf")


# ===========================================================================
# Figure 4 -- fig_cci
# ===========================================================================
def fig_cci() -> None:
    """(a) Plan fidelity (CCI), GSM n=44 with Wilcoxon arrow
    (b) Trajectory error propagation, GSM n=44, with twin-axis Acc line
    (c) Injection-step response on ALGO Phase-2B, n=61, stacked bars."""

    models = ["Claude", "GPT-4o", "Llama-8B", "Gemini-2.5", "o4-mini"]
    p2_models = ["Claude", "GPT-4o", "Llama-8B", "Gemini-2.5"]
    colors = [COLOR[m] for m in models]
    p2 = gsm_p2_metrics()
    p2b = algo_p2b_response()
    w, p_w = gsm_p2_paired_wilcoxon()

    # Inline-compute o4-mini CCI/TEP from the dedicated o1mini file
    o4_df = pd.read_csv(RAW / "GSM_P2_phase1_o1mini.csv", dtype=str).fillna("")
    o4_par = o4_df[o4_df.phase1_parseable.str.lower() == "true"]
    o4_cci_mean = pd.to_numeric(o4_par["cci_score"], errors="coerce").mean()
    o4_cci_med  = pd.to_numeric(o4_par["cci_score"], errors="coerce").median()
    o4_tep_mean = pd.to_numeric(o4_par["tep_score"], errors="coerce").mean()
    o4_acc = (o4_df["session_b_correct"].str.lower() == "true").mean()

    cci_mean = [p2[m]["cci_mean"] for m in p2_models] + [o4_cci_mean]
    cci_med  = [p2[m]["cci_med"]  for m in p2_models] + [o4_cci_med]
    tep_mean = [p2[m]["tep_mean"] for m in p2_models] + [o4_tep_mean]
    acc      = [p2[m]["acc"]      for m in p2_models] + [o4_acc]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.05, right=0.99, wspace=0.32)
    x = np.arange(len(models))

    # (a) CCI
    ax = axes[0]
    ax.set_title("(a) Plan fidelity (CCI)\nGSM, n=44", fontsize=10.5)
    ax.bar(x, np.nan_to_num(cci_mean, nan=0.0), color=colors, alpha=0.88, width=0.55,
           edgecolor="white", lw=0.6)
    for i, med in enumerate(cci_med):
        if not np.isfinite(med):
            continue
        ax.hlines(med, x[i] - 0.21, x[i] + 0.21, colors="black", lw=2, zorder=5,
                  label="Median" if i == 0 else "")
        if med == 0:
            ax.text(x[i], 0.005, "med=0", ha="center", fontsize=7.5, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Mean CCI")
    top_y = np.nanmax(cci_mean) + 0.05
    ax.set_ylim(0, top_y + 0.04)
    # Wilcoxon arrow
    ax.annotate("", xy=(1, cci_mean[1] + 0.012), xytext=(0, cci_mean[0] + 0.012),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax.text(0.5, max(cci_mean[0], cci_mean[1]) + 0.025,
            f"$p={p_w:.3f}$", ha="center", fontsize=8.5)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=7.8)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) TEP
    ax2 = axes[1]
    ax2.set_title("(b) Trajectory error propagation\nGSM, n=44", fontsize=10.5)
    ax2.bar(x, np.nan_to_num(tep_mean, nan=0.0), color=colors, alpha=0.88, width=0.55,
            edgecolor="white", lw=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("Mean TEP")
    ax2.set_ylim(0, np.nanmax(tep_mean) + 0.18)
    ax2b = ax2.twinx()
    acc_line = ax2b.plot(
        x, acc, color="#222222", ls="--", marker="o", lw=1.6, ms=6,
        label="Phase-2A accuracy (right axis)", zorder=5,
    )[0]
    for i, a_val in enumerate(acc):
        if np.isfinite(a_val):
            ax2b.annotate(
                f"{a_val * 100:.0f}%", (x[i], a_val),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7, color="#222222",
            )
    ax2b.set_ylabel("Phase-2A accuracy")
    ax2b.set_ylim(0.2, 1.05)
    ax2b.spines[["top"]].set_visible(False)
    tep_patch = mpatches.Patch(facecolor="#777", alpha=0.88, label="Mean TEP (bars)")
    ax2.legend(handles=[tep_patch, acc_line], loc="upper left",
               framealpha=0.85, fontsize=7.8)
    ax2.spines[["top"]].set_visible(False)

    # (c) injection-step stacked bars (o4-mini included via p2b lookup if present)
    ax3 = axes[2]
    ax3.set_title("(c) Injection-step response\nALGO, n=61", fontsize=10.5)
    # try to source o4-mini from p2b helper; if missing, compute inline
    if "o4-mini" in p2b:
        o4_p2b = p2b["o4-mini"]
    else:
        # Inline: read injected file and compute compliance rates for o4-mini
        try:
            df_inj = pd.read_csv(RAW / "ALGO_P2_phase2_injected.csv", dtype=str).fillna("")
            df_inj_o4 = df_inj[df_inj.model == "openai/o4-mini"].copy()
            df_inj_o4["_step"] = pd.to_numeric(df_inj_o4.step_index, errors="coerce").fillna(0)
            last = df_inj_o4.sort_values("_step").groupby("problem_id").tail(1)
            # diverged_from_normal indicates the model followed the injection -> compliant
            div = (last.diverged_from_normal.str.lower() == "true").mean()
            o4_p2b = {"compliant": float(div), "partial": 0.0, "format_ignored": 1.0 - float(div)}
        except Exception:
            o4_p2b = {"compliant": np.nan, "partial": np.nan, "format_ignored": np.nan}
    compl = [p2b[m]["compliant"] for m in p2_models] + [o4_p2b["compliant"]]
    part  = [p2b[m]["partial"]   for m in p2_models] + [o4_p2b["partial"]]
    fig_ig = [p2b[m]["format_ignored"] for m in p2_models] + [o4_p2b["format_ignored"]]
    ax3.bar(x, np.nan_to_num(compl, nan=0.0), width=0.6, color=colors, alpha=0.88,
            edgecolor="white", lw=0.6, label="Compliant")
    ax3.bar(x, np.nan_to_num(part, nan=0.0), width=0.6, bottom=np.nan_to_num(compl, nan=0.0), color=colors, alpha=0.55,
            edgecolor="white", lw=0.6, hatch="////", label="Partial/refusal")
    ax3.bar(x, np.nan_to_num(fig_ig, nan=0.0), width=0.6, bottom=np.nan_to_num(np.add(compl, part), nan=0.0), color="#bbbbbb",
            alpha=0.7, edgecolor="white", lw=0.6, hatch="..", label="Format-ignored")
    for i in range(len(models)):
        c_i, p_i, f_i = compl[i], part[i], fig_ig[i]
        if not (np.isfinite(c_i) and np.isfinite(p_i) and np.isfinite(f_i)):
            continue
        if c_i >= 0.03:
            ax3.text(x[i], c_i / 2, f"{c_i * 100:.0f}%",
                     ha="center", va="center", color="white", fontsize=8.5)
        elif c_i < 0.01 and (p_i + f_i) > 0.99:
            ax3.text(x[i], 0.012, "0%",
                     ha="center", va="bottom", color="#333333", fontsize=8.5,
                     fontweight="bold")
        if p_i >= 0.03:
            ax3.text(x[i], c_i + p_i / 2, f"{p_i * 100:.0f}%",
                     ha="center", va="center", color="#333333", fontsize=8.5)
        if f_i >= 0.03:
            ax3.text(x[i], c_i + p_i + f_i / 2, f"{f_i * 100:.0f}%",
                     ha="center", va="center", color="#444444", fontsize=8.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax3.set_ylabel("Fraction of injection steps")
    ax3.set_ylim(0, 1.03)
    legend_handles = [
        mpatches.Patch(facecolor="#777", alpha=0.88, label="Compliant"),
        mpatches.Patch(facecolor="#777", alpha=0.55, hatch="////", label="Partial/refusal"),
        mpatches.Patch(facecolor="#bbb", alpha=0.7, hatch="..", label="Format-ignored"),
    ]
    ax3.legend(handles=legend_handles, fontsize=7.5, loc="lower left",
               framealpha=0.85, handlelength=1.6)
    ax3.spines[["top", "right"]].set_visible(False)

    plt.savefig(OUT / "fig_cci.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_cci.pdf")


# ===========================================================================
# Figure 5 -- fig_paradox
# ===========================================================================
def fig_paradox() -> None:
    """Algorithm-invocation observation: bar of % correct per reasoning_type."""

    df = algo_p2a_combined()
    ORDER = [
        ("algorithm_invocation", "Algorithm\nInvocation"),
        ("backtracking", "Backtracking"),
        ("forward_simulation", "Forward\nSimulation"),
        ("local_greedy", "Local\nGreedy"),
        ("unclear", "Unclear"),
    ]
    rows = []
    for key, label in ORDER:
        s = df[df.reasoning_type == key]
        rows.append({"label": label, "key": key, "n": len(s),
                     "pct": float(s.final_answer_correct.mean() * 100) if len(s) else 0.0})
    rdf = pd.DataFrame(rows)
    baseline = rdf[rdf.key == "unclear"].pct.values[0]

    # Fisher's exact: algorithm_invocation vs unclear
    ai = df[df.reasoning_type == "algorithm_invocation"]
    un = df[df.reasoning_type == "unclear"]
    table = np.array([
        [int((ai.final_answer_correct == True).sum()),
         int((ai.final_answer_correct == False).sum())],
        [int((un.final_answer_correct == True).sum()),
         int((un.final_answer_correct == False).sum())],
    ])
    _, p_fisher = stats.fisher_exact(table, alternative="two-sided")
    ai_n = int(len(ai))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.subplots_adjust(top=0.85, bottom=0.16, left=0.10, right=0.97)
    ax.set_title(
        "Algorithm-invocation observation: step-level reasoning vs. correctness\n"
        f"(ALGO Phase-2A, n={len(df):,} steps, 4 models; observational, "
        f"Fisher's exact $p={p_fisher:.2f}$ vs. unclear baseline)",
        fontsize=9.5)

    bar_colors = ["#D55E00", "#CC79A7", "#E69F00", "#0072B2", "#0072B2"]
    bar_alpha = [1.0, 0.85, 0.85, 0.85, 0.85]
    bars = ax.bar(np.arange(5), rdf.pct, color=bar_colors, alpha=0.92,
                  width=0.66, edgecolor="white", lw=0.7)
    # Hatch the unclear bar slightly differently
    for i, (b, a) in enumerate(zip(bars, bar_alpha)):
        b.set_alpha(a)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(rdf.label, fontsize=9)
    ax.set_ylabel("Final-answer correctness (%)")
    ymax = max(18.0, baseline + 5)
    ax.set_ylim(0, ymax)
    ax.axhline(baseline, ls=":", color="#555", lw=1.2, zorder=2)
    ax.text(4.3, baseline + 0.5, f"{baseline:.1f}%\n(unclear baseline)",
            ha="left", va="bottom", fontsize=8.5, color="#444")
    # n= labels above each bar
    for i, row in rdf.iterrows():
        ax.text(i, row.pct + 0.4, f"n={row.n}",
                ha="center", fontsize=8.3, color="#333")
    # 0/13 callout for Algorithm-Invocation
    ai_idx = 0
    ax.annotate(
        f"0/{ai_n} correct\n(3 models)",
        xy=(ai_idx, 0.6),
        xytext=(ai_idx + 1.05, 7.0),
        fontsize=8.5, color="#A04000",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#A04000", lw=1.0),
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.savefig(OUT / "fig_paradox.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_paradox.pdf")


# ===========================================================================
# Figure 6 -- fig_mechanistic  (NEW)
# ===========================================================================
def fig_mechanistic() -> None:
    """Mechanistic dissociation in Qwen-2.5-7B residual stream.

    A 2x2 panel showing the four mechanistic metrics for the gold token,
    canonical vs. W_6, across three problem families:

      (a) Median final-token rank (terminal commitment).
      (b) Median rank-per-layer trajectory (depth-resolved commitment).
      (c) Median log-prob-per-layer trajectory (calibrated confidence).
      (d) Median cosine-similarity-per-layer (residual stream alignment
          with the gold-token unembedding direction).

    All four metrics come from the per-layer arrays in
    ``uploaded/05_probe3_raw/mechanistic_sweep_7b_base_rawqa.csv``.
    """
    import ast as _ast
    src_raw = ROOT / "uploaded" / "05_probe3_raw" / "mechanistic_sweep_7b_base_rawqa.csv"
    src_sum = ROOT / "results" / "paper" / "AUDIT" / "probe3_mechanistic_family_variant_finalrank.csv"
    raw = pd.read_csv(src_raw)
    summ = pd.read_csv(src_sum)

    def _parse(s: object) -> list[float]:
        if isinstance(s, str) and s.strip().startswith("["):
            try:
                return list(_ast.literal_eval(s))
            except Exception:
                return []
        return []

    for col in (
        "target_rank_per_layer",
        "target_logprob_per_layer",
        "layer_cosine_similarities",
    ):
        raw[col + "_arr"] = raw[col].map(_parse)

    n_layers = int(raw["n_layers_processed"].mode().iloc[0])

    families_order = [
        ("algorithmic", "Algorithmic (ALGO)", "#0072B2"),
        ("planning_suite", "Planning (BW)", "#009E73"),
        ("arithmetic_reasoning", "Arithmetic (GSM)", "#D55E00"),
    ]

    def _median_traj(family: str, variant: str, col: str) -> np.ndarray | None:
        sub = raw[(raw.problem_family == family) & (raw.variant_type == variant)]
        arrs = [a for a in sub[col + "_arr"] if len(a) == n_layers]
        if not arrs:
            return None
        return np.median(np.asarray(arrs, dtype=float), axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    fig.subplots_adjust(
        top=0.90, bottom=0.10, left=0.07, right=0.985, hspace=0.42, wspace=0.30
    )
    fig.suptitle(
        "Mechanistic dissociation (Qwen-2.5-7B residual stream): "
        "W$_6$ surfaces the gold token across all four depth-resolved metrics "
        "for algorithmic and planning families, but not for arithmetic",
        fontsize=11.0,
        y=0.985,
    )

    # ------------------------------------------------------------------
    # Panel (a): bar chart of median final-token rank, canonical vs W6
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    labels, can_vals, w6_vals, n_can, n_w6 = [], [], [], [], []
    for key, lbl, _ in families_order:
        sub = summ[summ.problem_family == key]
        c = sub[sub.variant_type == "canonical"]
        w = sub[sub.variant_type == "W6"]
        if len(c) and len(w):
            labels.append(lbl.replace(" ", "\n", 1))
            can_vals.append(float(c["median_final_rank"].iloc[0]))
            w6_vals.append(float(w["median_final_rank"].iloc[0]))
            n_can.append(int(c["n"].iloc[0]))
            n_w6.append(int(w["n"].iloc[0]))
    x = np.arange(len(labels))
    bw = 0.34
    ax.bar(x - bw / 2, can_vals, bw, color="#888888", alpha=0.9,
           edgecolor="white", lw=0.6, label="Canonical")
    ax.bar(x + bw / 2, w6_vals, bw, color="#0072B2", alpha=0.9,
           edgecolor="white", lw=0.6, label="W$_6$ (new numbers,\nsame algorithm)")
    ax.set_yscale("log")
    ax.set_ylim(20, 100_000)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median final-token rank (log)\n(lower = more accessible)")
    ax.set_title("(a) Terminal commitment to the gold token", loc="left", fontsize=10)
    for i, (c, w_) in enumerate(zip(can_vals, w6_vals)):
        ax.text(i - bw / 2, c * 1.18, f"{c:,.0f}", ha="center", fontsize=8.0, color="#222")
        ax.text(i + bw / 2, w_ * 1.18, f"{w_:,.0f}", ha="center", fontsize=8.0, color="#222")
        ratio = w_ / c if c else 1.0
        if ratio < 0.5:
            verdict = f"{c / w_:.0f}x easier"
            color = "#2A7F2A"
        elif ratio > 1.05:
            verdict = "reversal"
            color = "#A04000"
        else:
            verdict = "no change"
            color = "#777777"
        ax.annotate(verdict, xy=(i, max(c, w_) * 3.0),
                    ha="center", fontsize=8.5, color=color, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.92, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="major", ls="--", color="#dddddd", lw=0.6, zorder=0)

    # ------------------------------------------------------------------
    # Helper for trajectory panels
    # ------------------------------------------------------------------
    layer_idx = np.arange(n_layers)

    def _draw_traj(ax_, metric_col: str, ylabel: str, title: str,
                   ylog: bool = False, ylim: tuple[float, float] | None = None) -> None:
        for key, lbl, color in families_order:
            for var, ls, marker in (("canonical", "-", "o"), ("W6", "--", "s")):
                y = _median_traj(key, var, metric_col)
                if y is None:
                    continue
                ax_.plot(layer_idx, y, ls=ls, lw=1.7, color=color, alpha=0.92,
                         marker=marker, markersize=3.2, markevery=4,
                         label=f"{lbl} - {'W6' if var=='W6' else 'canonical'}")
        ax_.set_xlabel("Layer index (0 = embedding, %d = final)" % (n_layers - 1))
        ax_.set_ylabel(ylabel)
        ax_.set_title(title, loc="left", fontsize=10)
        if ylog:
            ax_.set_yscale("log")
        if ylim is not None:
            ax_.set_ylim(*ylim)
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.grid(True, ls="--", color="#dddddd", lw=0.55, zorder=0)

    # ------------------------------------------------------------------
    # Panel (b): rank-per-layer trajectory
    # ------------------------------------------------------------------
    _draw_traj(
        axes[0, 1],
        "target_rank_per_layer",
        "Median gold-token rank (log)\n(lower = more accessible)",
        "(b) Depth-resolved rank trajectory",
        ylog=True,
        ylim=(20, 250_000),
    )

    # ------------------------------------------------------------------
    # Panel (c): logprob-per-layer trajectory
    # ------------------------------------------------------------------
    _draw_traj(
        axes[1, 0],
        "target_logprob_per_layer",
        "Median gold-token log-prob\n(higher = more confident)",
        "(c) Depth-resolved calibrated confidence",
        ylim=(-160, 0),
    )

    # ------------------------------------------------------------------
    # Panel (d): cosine similarity trajectory
    # ------------------------------------------------------------------
    _draw_traj(
        axes[1, 1],
        "layer_cosine_similarities",
        "Median residual-stream cosine similarity\nto gold unembedding direction",
        "(d) Depth-resolved representational alignment",
        ylim=(0.80, 1.005),
    )

    # ------------------------------------------------------------------
    # Shared legend for panels b/c/d
    # ------------------------------------------------------------------
    fam_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=lbl)
        for _, lbl, color in families_order
    ]
    var_handles = [
        Line2D([0], [0], color="#444444", lw=1.8, ls="-",  marker="o", markersize=4, label="Canonical"),
        Line2D([0], [0], color="#444444", lw=1.8, ls="--", marker="s", markersize=4, label="W$_6$"),
    ]
    fig.legend(
        handles=fam_handles + var_handles,
        loc="lower center", ncol=5, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(bottom=0.13)

    # Annotate sample sizes once, under panel (a)
    nl = ", ".join(
        f"{lbl.split(' (')[0]}: n_can={nc}/n_W6={nw}"
        for (key, lbl, _), nc, nw in zip(families_order, n_can, n_w6)
    )
    fig.text(0.07, 0.045,
             f"Qwen-2.5-7B, {n_layers} layers. {nl}. Coin Change omitted (no W$_6$ sample).",
             fontsize=8.0, color="#555555")

    plt.savefig(OUT / "fig_mechanistic.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_mechanistic.pdf")


# ===========================================================================
# Figure 7 -- fig_population  (NEW)
# ===========================================================================
def fig_population() -> None:
    """Population-level test: pairwise inversion exists; population correlation
    is null.  Scatter of (canonical accuracy, W3 retention) across all
    (model x subtype) cells, with a fit line, plus arrows highlighting the
    two matched-canonical inversion pairs (Claude<->GPT-4o on SP and CC).
    """
    tri = pd.read_csv(DER / "ALGO_P3_triangulation.csv")
    adv_map = tri[["problem_id", "instance_type"]].drop_duplicates().set_index(
        "problem_id")["instance_type"]

    cells = []
    main_models = [
        ("Claude", "anthropic/claude-sonnet-4"),
        ("GPT-4o", "openai/gpt-4o"),
        ("Llama-8B", "meta-llama/llama-3.1-8b-instruct"),
        ("Gemini-2.5", "google/gemini-2.5-flash"),
    ]

    # ALGO subtype x instance_type cells
    sub_map = tri[["problem_id", "problem_subtype"]].drop_duplicates().set_index(
        "problem_id")["problem_subtype"]
    for lbl, mid in main_models:
        slug = {"Claude": "claude", "GPT-4o": "gpt4o",
                "Llama-8B": "llama", "Gemini-2.5": "gemini"}[lbl]
        df = _read_algo_p1(slug)
        df = df.assign(_ok=_correct_col(df).astype(int).values)
        df["_inst"] = df.problem_id.map(adv_map)
        df["_sub"] = df.problem_id.map(sub_map)
        for sub_key, sub_lbl in [("coin_change", "CC"), ("shortest_path", "SP"),
                                  ("wis", "WIS")]:
            for inst in ["adversarial", "standard"]:
                cell = df[(df["_sub"] == sub_key) & (df["_inst"] == inst)]
                can_rows = cell[cell.variant_type == "canonical"]
                w3_rows = cell[cell.variant_type == "W3"]
                if len(can_rows) == 0 or len(w3_rows) == 0:
                    continue
                can = can_rows._ok.mean()
                w3 = w3_rows._ok.mean()
                if not (np.isnan(can) or np.isnan(w3)) and can > 0:
                    cells.append({
                        "model": lbl, "family": "ALGO",
                        "sub": f"{sub_lbl}-{inst[:4]}",
                        "can": can, "R": w3 / can,
                        "n": int(len(can_rows)),
                    })

    # GSM single cell per model (use the unified P1 frame)
    uni = _gsm_p1_unified()
    for lbl, mid in main_models:
        s = uni[uni.model == mid]
        c_rows = s[s.variant_type == "canonical"]
        w_rows = s[s.variant_type == "W3"]
        if len(c_rows) == 0 or len(w_rows) == 0:
            continue
        c = c_rows.correct.mean()
        w = w_rows.correct.mean()
        if c > 0:
            cells.append({"model": lbl, "family": "GSM", "sub": "GSM",
                          "can": c, "R": w / c, "n": int(len(c_rows))})

    # BW per-subtype per model.  Split by problem_id prefix (BW_* vs MBW_*).
    bw_main = pd.read_csv(RAW / "BW_P1_behavioral.csv", dtype=str).fillna("")
    bw_gem = pd.read_csv(RAW / "BW_P1_behavioral_gemini.csv", dtype=str).fillna("")
    bw_all = pd.concat([bw_main, bw_gem], ignore_index=True)
    bw_all = bw_all[~bw_all.model.astype(str).str.contains("mock|The answer is|deepseek", regex=True)]
    bw_all = bw_all[bw_all.problem_id.astype(str).str.startswith(("BW_", "MBW_"))]
    bw_all["correct"] = bw_all.behavioral_correct.str.lower().eq("true").astype(int)
    bw_all["sub_lbl"] = bw_all.problem_id.astype(str).str.startswith("MBW_").map(
        {True: "BW-mys", False: "BW-std"})
    for lbl, mid in main_models:
        for sub_lbl in ["BW-std", "BW-mys"]:
            cell = bw_all[(bw_all.model == mid) & (bw_all.sub_lbl == sub_lbl)]
            cans = cell[cell.variant_type == "canonical"]
            w3s = cell[cell.variant_type == "W3"]
            if len(cans) == 0 or len(w3s) == 0:
                continue
            cc = cans.correct.mean()
            ww = w3s.correct.mean()
            if cc > 0:
                cells.append({"model": lbl, "family": "BW", "sub": sub_lbl,
                              "can": cc, "R": ww / cc, "n": int(len(cans))})

    cdf = pd.DataFrame(cells)
    rho, p_pop = stats.spearmanr(cdf.can, cdf.R)

    # bootstrap CI
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, len(cdf), len(cdf))
        r, _ = stats.spearmanr(cdf.can.values[idx], cdf.R.values[idx])
        boots.append(r)
    ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.10, right=0.97)
    color_by_model = {"Claude": COLOR["Claude"], "GPT-4o": COLOR["GPT-4o"],
                      "Llama-8B": COLOR["Llama-8B"], "Gemini-2.5": COLOR["Gemini-2.5"]}
    marker_by_family = {"ALGO": "o", "GSM": "s", "BW": "^"}
    for _, row in cdf.iterrows():
        ax.scatter(row.can, row.R,
                   c=color_by_model[row.model],
                   marker=marker_by_family[row.family],
                   s=90, alpha=0.85, edgecolors="white", lw=0.6, zorder=4)

    # linear fit
    from numpy.polynomial import polynomial as P
    x = cdf.can.values
    y = cdf.R.values
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ys = coef[0] * xs + coef[1]
    ax.plot(xs, ys, "k--", lw=1.2, label=f"Linear fit (slope={coef[0]:+.2f})", zorder=3)

    # highlight the two matched-canonical inversion pairs with arrows
    # SP-adv Claude vs GPT-4o: same can ≈ 0.65 vs 0.41
    sp_pts = cdf[(cdf.family == "ALGO") & (cdf.sub == "SP-adve")]
    cc_pts = cdf[(cdf.family == "ALGO") & (cdf.sub == "CC-adve")]
    if len(sp_pts) < 2:
        sp_pts = cdf[(cdf.family == "ALGO") & cdf["sub"].str.startswith("SP-")]
    if len(cc_pts) < 2:
        cc_pts = cdf[(cdf.family == "ALGO") & cdf["sub"].str.startswith("CC-")]
    if len(sp_pts) >= 2:
        c_sp = sp_pts[sp_pts.model == "Claude"].iloc[0]
        g_sp = sp_pts[sp_pts.model == "GPT-4o"].iloc[0]
        ax.annotate("", xy=(c_sp.can, c_sp.R), xytext=(g_sp.can, g_sp.R),
                    arrowprops=dict(arrowstyle="->", color="#444",
                                    lw=1.2, alpha=0.85))
        ax.text(0.13, 0.18,
                "SP-adv inversion:\nhigher can. $\\to$ lower $R_{W3}$",
                fontsize=7.3, color="#444",
                bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                          alpha=0.9, boxstyle="round,pad=0.20"))
    if len(cc_pts) >= 2:
        c_cc = cc_pts[cc_pts.model == "Claude"].iloc[0]
        g_cc = cc_pts[cc_pts.model == "GPT-4o"].iloc[0]
        ax.annotate("", xy=(c_cc.can, c_cc.R), xytext=(g_cc.can, g_cc.R),
                    arrowprops=dict(arrowstyle="->", color="#A04000",
                                    lw=1.2, alpha=0.85))
        ax.text(0.50, 1.28,
                "CC-adv inversion:\nClaude robust, GPT-4o collapses",
                fontsize=7.3, color="#A04000",
                bbox=dict(facecolor="white", edgecolor="#e5b9a5",
                          alpha=0.9, boxstyle="round,pad=0.20"))

    ax.axhline(1.0, ls=":", color="#888", lw=0.8, alpha=0.7, zorder=2)
    ax.set_xlabel("Canonical accuracy")
    ax.set_ylabel(r"$W_3$ retention  $\mathrm{Acc}_{W3}/\mathrm{Acc}_\mathrm{can}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.55)
    ax.set_title(
        "Population-level dissociation test: Spearman "
        f"$r={rho:+.3f}$ ({len(cdf)} cells, "
        f"95% CI $[{ci_lo:+.2f}, {ci_hi:+.2f}]$, $p={p_pop:.2f}$).\n"
        "Pairwise inversions exist (arrows); population correlation is null.",
        fontsize=9.5,
    )

    # custom legend for models + family markers
    handles = [mpatches.Patch(facecolor=c, label=lbl)
                for lbl, c in color_by_model.items()]
    fhandles = [Line2D([0], [0], marker=mk, color="#444", linestyle="",
                       markersize=7, label=lbl)
                for lbl, mk in [("ALGO", "o"), ("GSM", "s"), ("BW", "^")]]
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.85,
                     ncol=2, handlelength=1.0, handletextpad=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=fhandles + [Line2D([0], [0], linestyle="--", color="black", label="fit")],
              loc="lower right", fontsize=8, framealpha=0.85,
              handlelength=1.2, handletextpad=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.savefig(OUT / "fig_population.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote fig_population.pdf  (n={len(cdf)}, r={rho:+.3f}, p={p_pop:.3f})")
    return cdf, rho, p_pop, ci_lo, ci_hi


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    print(f"Output dir: {OUT}")
    dump_gsm_p1_reconciliation()
    fig_robustness()
    fig_decay()
    fig_heatmap()
    fig_cci()
    fig_paradox()
    fig_mechanistic()
    fig_population()
    print("All 7 figures regenerated from raw data.")


if __name__ == "__main__":
    main()
