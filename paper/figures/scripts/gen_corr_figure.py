"""Cross-probe correlation matrix (per model) — find structure across probes.

For each model, builds a per-problem feature vector across all probes:
  P1: canonical, W1, W2, W3, W4, W5, W6 accuracies (binary per problem)
  P2: cci, tep, p2a final-correct, injection-resistance (where available)
  P3: contamination score, max_ngram_length

Then computes Spearman correlation matrices and plots them as a small
multiples heatmap. P2.acc is phase2a_correct only (not the old
either_session_correct disjunction). If phase2a was never persisted, P2.acc
is absent and cannot correlate with contamination.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"
OUT = Path(__file__).resolve().parents[1]
BANK_GSM_PATH = ROOT / "data" / "problems" / "question_bank_gsm.csv"

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash":   "Gemini",
    "openai/gpt-4o":             "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini":            "o4-mini",
}

EXPECTED_GSM_CANONICAL_N = {
    "Claude": 44,
    "Gemini": 44,
    "o4-mini": 44,
    "GPT-4o": 20,
    "Llama": 20,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "pdf.fonttype": 42,
})


def _load_bank_gsm() -> set[str]:
    try:
        bank = pd.read_csv(BANK_GSM_PATH, usecols=["problem_id"])
        return set(bank["problem_id"].astype(str).unique())
    except Exception:
        return {
            *(f"GSM_{i:03d}" for i in range(1, 21)),
            *(f"GSM_{i:03d}" for i in range(41, 65)),
        }


BANK_GSM = _load_bank_gsm()


def _norm_v(v: str) -> str:
    v = str(v).strip()
    return v.upper() if v and v[0].lower() == "w" else v


def gsm_per_problem(model_short: str, model_full: str) -> pd.DataFrame:
    """Per-problem GSM feature vector for one model: variants + P2 metrics
    + P3 contamination."""
    tag = {"Claude": "claude", "Gemini": "gemini", "GPT-4o": "gpt4o",
           "Llama": "llama", "o4-mini": "o1mini"}[model_short]
    p = RAW / f"GSM_P1_behavioral_{tag}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype=str).fillna("")
    if "model" in df.columns:
        df = df[df["model"] == model_full]
    df = df[df["problem_id"].astype(str).isin(BANK_GSM)]
    resp_col = "raw_response" if "raw_response" in df.columns else "response"
    if resp_col in df.columns:
        df = df[~df[resp_col].astype(str).str.startswith("ERROR")]
    bc = df["behavioral_correct"].astype(str).str.strip()
    df = df[bc.ne("") & df["behavioral_correct"].notna()]
    df["variant_type"] = df["variant_type"].apply(_norm_v)
    df["ok"] = df["behavioral_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    p1 = df.pivot_table(
        index="problem_id", columns="variant_type", values="ok", aggfunc="last",
    ).reset_index()
    p1.columns.name = None
    # P2 CCI
    p2 = pd.read_csv(RAW / "GSM_P2_cci.csv", dtype=str).fillna("")
    p2 = p2[p2["model"] == model_full]
    ov_path = DER / "GSM_P2_session_correct.csv"
    if ov_path.exists() and not p2.empty:
        ov = pd.read_csv(ov_path, dtype=str).fillna("")
        ov = ov[ov["model"] == model_full]
        keep = [c for c in ("problem_id", "phase2a_correct", "either_session_correct") if c in ov.columns]
        p2 = p2.merge(ov[keep].drop_duplicates("problem_id"), on="problem_id", how="left", suffixes=("", "_ov"))
        if "either_session_correct_ov" in p2.columns:
            p2["either_session_correct"] = p2["either_session_correct_ov"]
    if not p2.empty and "cci_score" in p2.columns:
        p2_sub = p2[["problem_id", "cci_score", "tep_score"]].copy()
        for c in ["cci_score", "tep_score"]:
            p2_sub[c] = pd.to_numeric(p2_sub[c], errors="coerce")
        if "phase2a_correct" in p2.columns and p2["phase2a_correct"].astype(str).str.strip().ne("").any():
            p2_sub["P2.acc"] = p2["phase2a_correct"].astype(str).str.lower().eq("true").astype(int)
        merged = p1.merge(p2_sub, on="problem_id", how="left")
    else:
        merged = p1
    # P3 contamination
    p3 = pd.read_csv(RAW / "GSM_P3_contamination.csv", dtype=str).fillna("")
    if "contamination_score" in p3.columns:
        p3["contamination_score"] = pd.to_numeric(p3["contamination_score"], errors="coerce")
    if "max_ngram_length" in p3.columns:
        p3["max_ngram_length"] = pd.to_numeric(p3["max_ngram_length"], errors="coerce")
    cols_keep = [c for c in ["problem_id", "contamination_score", "max_ngram_length"] if c in p3.columns]
    merged = merged.merge(p3[cols_keep], on="problem_id", how="left")
    return merged


def corr_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols_present = [c for c in cols if c in df.columns]
    sub = df[cols_present].apply(pd.to_numeric, errors="coerce")
    return sub.corr(method="spearman")


def fig_cross_probe_corr() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.06, right=0.99, wspace=0.32)
    models = [("Claude", "anthropic/claude-sonnet-4"),
              ("Gemini", "google/gemini-2.5-flash"),
              ("GPT-4o", "openai/gpt-4o"),
              ("Llama", "meta-llama/llama-3.1-8b-instruct"),
              ("o4-mini", "openai/o4-mini")]
    cols = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6",
             "cci_score", "tep_score", "P2.acc",
             "contamination_score", "max_ngram_length"]
    short_labels = {"canonical": "can", "W1": "W1", "W2": "W2", "W3": "W3", "W4": "W4",
                    "W5": "W5", "W6": "W6", "cci_score": "CCI", "tep_score": "TEP",
                    "P2.acc": "P2.acc", "contamination_score": "contam",
                    "max_ngram_length": "n-gram"}

    im = None
    for ax, (mlabel, mfull) in zip(axes, models):
        df = gsm_per_problem(mlabel, mfull)
        n_can = int(df["canonical"].notna().sum()) if "canonical" in df.columns else 0
        expected = EXPECTED_GSM_CANONICAL_N[mlabel]
        if n_can != expected:
            raise AssertionError(
                f"GSM corr matrix {mlabel}: canonical n={n_can}, expected {expected}"
            )
        if df.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(mlabel)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        cm = corr_matrix(df, cols)
        if cm.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(mlabel)
            continue
        im = ax.imshow(cm.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        labels = [short_labels.get(c, c) for c in cm.columns]
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        title_n = f"n={n_can}" if mlabel in ("GPT-4o", "Llama") else f"n={n_can}"
        ax.set_title(f"{mlabel}\n({title_n})", fontsize=9)

    cax = fig.add_axes([0.92, 0.20, 0.01, 0.66])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Spearman $\rho$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Cross-probe Spearman correlation matrix per model — GSM "
        "(n=44; GPT-4o/Llama n=20). Sub-probes that share a model's "
        "fragility cluster on the diagonal blocks.",
        fontsize=10.0,
    )
    plt.savefig(OUT / "fig_corr_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_corr_matrix.pdf")


if __name__ == "__main__":
    fig_cross_probe_corr()
