#!/usr/bin/env python3
"""Step 11 — Cross-probe correlation analysis (P1 × P2 × P3).

Connects P1 W3 fragility, P2 process (CCI), and P3 contamination on shared instances.

Outputs:
    results/derived/cross_probe_instance_frame.csv
    results/derived/cross_probe_spearman_by_model.csv
    results/derived/cross_probe_spearman_model_level.csv
    results/derived/cross_probe_agreement_instances.csv
    results/derived/cross_probe_acc_vs_w3retention.csv  (refreshed)
    results/derived/cross_probe_correlation_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runs.triangulation_v2 import SHORT as FULL_TO_SHORT  # noqa: E402

EXCLUDE_MODELS = {"mock", "deepseek/deepseek-r1-distill-llama-70b"}
SHORT_TO_FULL = {v: k for k, v in FULL_TO_SHORT.items()}

P1_W3_DROP_THRESH = 0.5  # canonical ok, w3 fail
P2_CCI_LOW = 0.3
P3_CONTAM_HIGH = 0.6


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return None


def _norm_model(s: str) -> str:
    s = str(s).strip()
    if s in FULL_TO_SHORT:
        return FULL_TO_SHORT[s]
    if s in SHORT_TO_FULL:
        return s
    for full, short in FULL_TO_SHORT.items():
        if s == full or s.endswith(full.split("/")[-1]):
            return short
    return s


def _spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    xv = pd.to_numeric(x[mask], errors="coerce")
    yv = pd.to_numeric(y[mask], errors="coerce")
    ok = xv.notna() & yv.notna()
    xv, yv = xv[ok], yv[ok]
    if len(xv) < 5 or xv.nunique() < 2 or yv.nunique() < 2:
        return float("nan"), float("nan"), int(len(xv))
    rho, p = stats.spearmanr(xv, yv)
    return float(rho), float(p), int(len(xv))


def load_contam_lookup() -> pd.DataFrame:
    rows: list[dict] = []
    specs = [
        ("GSM", RAW / "GSM_P3_contamination.csv", "contamination_score"),
        ("ALGO", RAW / "ALGO_P3_contamination.csv", "instance_contamination_score"),
        ("BW", RAW / "BW_P3_contamination.csv", "instance_contamination_score"),
    ]
    for fam, path, col in specs:
        df = _safe_read(path)
        if df is None or col not in df.columns:
            continue
        for pid, g in df.groupby("problem_id"):
            val = pd.to_numeric(g[col], errors="coerce").mean()
            rows.append({"family": fam, "problem_id": pid, "p3_contam": val})
    return pd.DataFrame(rows)


def load_p2_cci_lookup() -> pd.DataFrame:
    rows: list[dict] = []
    from scripts.runs.coverage_audit import load_gsm_p2_merged

    gsm = load_gsm_p2_merged()
    if not gsm.empty and "cci_score" in gsm.columns:
        for _, r in gsm.iterrows():
            rows.append(
                {
                    "family": "GSM",
                    "problem_id": r["problem_id"],
                    "model": _norm_model(r.get("model", "")),
                    "p2_cci": pd.to_numeric(r["cci_score"], errors="coerce"),
                }
            )
    bw = _safe_read(RAW / "BW_P2_cci.csv")
    if bw is not None and "cci" in bw.columns:
        for _, r in bw.iterrows():
            rows.append(
                {
                    "family": "BW",
                    "problem_id": r["problem_id"],
                    "model": _norm_model(r.get("model", "")),
                    "p2_cci": pd.to_numeric(r["cci"], errors="coerce"),
                }
            )
    algo = _safe_read(DER / "ALGO_P2_per_instance_cci.csv")
    if algo is not None and "cci_composite" in algo.columns:
        for _, r in algo.iterrows():
            rows.append(
                {
                    "family": "ALGO",
                    "problem_id": r["problem_id"],
                    "model": _norm_model(r.get("model", "")),
                    "p2_cci": pd.to_numeric(r["cci_composite"], errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["family", "problem_id", "model", "p2_cci"])
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["family", "problem_id", "model"], keep="last")


def build_instance_frame() -> pd.DataFrame:
    lab = pd.read_csv(DER / "triangulation_v2_labels.csv")
    lab = lab[~lab["model"].isin(EXCLUDE_MODELS)].copy()
    lab["model_short"] = lab["model"].map(_norm_model)

    lab["var_canonical"] = pd.to_numeric(lab["var_canonical"], errors="coerce")
    lab["var_w3"] = pd.to_numeric(lab["var_w3"], errors="coerce")
    lab["vri_gap"] = pd.to_numeric(lab.get("vri_gap", np.nan), errors="coerce")
    lab["p3_contam"] = pd.to_numeric(lab.get("contam", np.nan), errors="coerce")
    lab["p2_cci"] = pd.to_numeric(lab.get("cci_effective", np.nan), errors="coerce")

    # P1 W3 drop: 1 when canonical correct & W3 wrong; else 0; NaN if canonical unknown
    lab["p1_w3_drop"] = np.where(
        lab["var_canonical"].notna() & lab["var_w3"].notna(),
        np.maximum(lab["var_canonical"] - lab["var_w3"], 0),
        np.nan,
    )
    lab["p1_w3_retention"] = np.where(
        lab["var_canonical"] == 1,
        lab["var_w3"],
        np.where(lab["var_canonical"] == 0, np.nan, np.nan),
    )

    contam = load_contam_lookup()
    if not contam.empty:
        lab = lab.merge(contam, on=["family", "problem_id"], how="left", suffixes=("", "_bank"))
        lab["p3_contam"] = lab["p3_contam"].fillna(lab.get("p3_contam_bank", np.nan))
        if "p3_contam_bank" in lab.columns:
            lab = lab.drop(columns=["p3_contam_bank"])

    p2 = load_p2_cci_lookup()
    if not p2.empty:
        lab = lab.merge(
            p2.rename(columns={"model": "model_short", "p2_cci": "p2_cci_file"}),
            on=["family", "problem_id", "model_short"],
            how="left",
        )
        lab["p2_cci"] = lab["p2_cci"].fillna(lab["p2_cci_file"])
        lab = lab.drop(columns=["p2_cci_file"], errors="ignore")

    # Signal flags for agreement
    lab["flag_p1_fragile"] = (lab["p1_w3_drop"] >= P1_W3_DROP_THRESH) | (lab["vri_gap"] > 0.5)
    lab["flag_p2_low_cci"] = lab["p2_cci"] <= P2_CCI_LOW
    lab["flag_p3_high_contam"] = lab["p3_contam"] >= P3_CONTAM_HIGH
    lab["flag_p1_robust"] = (lab["p1_w3_retention"] >= 0.8) & (lab["var_canonical"] == 1)
    lab["flag_p2_high_cci"] = lab["p2_cci"] >= 0.5
    lab["flag_p3_low_contam"] = lab["p3_contam"] <= 0.4

    lab["triple_retrieval_agree"] = (
        lab["flag_p1_fragile"] & lab["flag_p2_low_cci"] & lab["flag_p3_high_contam"]
    )
    lab["triple_computation_agree"] = (
        lab["flag_p1_robust"] & lab["flag_p2_high_cci"] & lab["flag_p3_low_contam"]
    )
    lab["triple_any_disagree"] = (
        lab[["flag_p1_fragile", "flag_p2_low_cci", "flag_p3_high_contam"]].notna().all(axis=1)
        & ~(
            (lab["flag_p1_fragile"] & lab["flag_p2_low_cci"] & lab["flag_p3_high_contam"])
            | (lab["flag_p1_robust"] & lab["flag_p2_high_cci"] & lab["flag_p3_low_contam"])
        )
    )
    return lab


def spearman_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("p1_w3_drop", "p2_cci", "P1_W3_drop vs P2_CCI"),
        ("p1_w3_drop", "p3_contam", "P1_W3_drop vs P3_contam"),
        ("p2_cci", "p3_contam", "P2_CCI vs P3_contam"),
        ("p1_w3_retention", "p2_cci", "P1_W3_retention vs P2_CCI"),
        ("vri_gap", "p3_contam", "VRI_gap vs P3_contam"),
    ]
    rows: list[dict] = []
    for (fam, model), g in frame.groupby(["family", "model_short"]):
        for xcol, ycol, label in pairs:
            if xcol not in g.columns or ycol not in g.columns:
                continue
            rho, p, n = _spearman_safe(g[xcol], g[ycol])
            rows.append(
                {
                    "family": fam,
                    "model": model,
                    "pair": label,
                    "x": xcol,
                    "y": ycol,
                    "n": n,
                    "spearman_rho": rho,
                    "p_value": p,
                    "corr_eligible": n >= 5 and not pd.isna(rho),
                }
            )
    return pd.DataFrame(rows)


def spearman_model_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Across-model Spearman using per-model means (5 points per family)."""
    metrics = ["p1_w3_drop", "p1_w3_retention", "p2_cci", "p3_contam", "vri_gap"]
    rows: list[dict] = []
    for fam, g in frame.groupby("family"):
        agg = g.groupby("model_short")[[m for m in metrics if m in g.columns]].mean(numeric_only=True)
        if len(agg) < 3:
            continue
        pairs = [
            ("p1_w3_drop", "p2_cci"),
            ("p1_w3_drop", "p3_contam"),
            ("p2_cci", "p3_contam"),
            ("p1_w3_retention", "p2_cci"),
        ]
        for xcol, ycol in pairs:
            if xcol not in agg.columns or ycol not in agg.columns:
                continue
            rho, p, n = _spearman_safe(agg[xcol], agg[ycol])
            rows.append(
                {
                    "family": fam,
                    "pair": f"{xcol} vs {ycol}",
                    "n_models": n,
                    "spearman_rho": rho,
                    "p_value": p,
                    "models_used": ",".join(sorted(agg.index.tolist())),
                }
            )
    return pd.DataFrame(rows)


def agreement_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    valid = frame[
        frame["p1_w3_drop"].notna() | frame["p2_cci"].notna() | frame["p3_contam"].notna()
    ]
    for (fam, model), g in valid.groupby(["family", "model_short"]):
        n = len(g)
        rows.append(
            {
                "family": fam,
                "model": model,
                "n_instances": n,
                "triple_retrieval_agree_n": int(g["triple_retrieval_agree"].sum()),
                "triple_retrieval_agree_pct": float(g["triple_retrieval_agree"].mean()),
                "triple_computation_agree_n": int(g["triple_computation_agree"].sum()),
                "triple_computation_agree_pct": float(g["triple_computation_agree"].mean()),
                "triple_mixed_n": int(g["triple_any_disagree"].sum()),
                "triple_mixed_pct": float(g["triple_any_disagree"].mean()),
                "p2_cci_valid_n": int(g["p2_cci"].notna().sum()),
                "p3_contam_valid_n": int(g["p3_contam"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def refresh_acc_w3retention(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fam, g in frame.groupby("family"):
        agg = g.groupby("model_short").agg(
            canonical=("var_canonical", "mean"),
            w3_ret=("p1_w3_retention", "mean"),
        )
        if len(agg) < 3:
            continue
        rho, p, n = _spearman_safe(agg["canonical"], agg["w3_ret"])
        rows.append(
            {
                "probe": fam,
                "n_models": n,
                "spearman_rho": rho,
                "p_value": p,
                "models_used": ",".join(sorted(agg.index.tolist())),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    frame: pd.DataFrame,
    by_model: pd.DataFrame,
    model_level: pd.DataFrame,
    agreement: pd.DataFrame,
    acc_w3: pd.DataFrame,
) -> None:
    lines = [
        "# Cross-probe correlation analysis (Step 11)",
        "",
        "Links P1 W3 fragility, P2 CCI, and P3 contamination on shared instances.",
        "",
        f"**Instance rows (excl. mock):** {len(frame)}",
        "",
        "## Model-level Spearman (5 models × family means)",
        "",
        "```",
    ]
    if not model_level.empty:
        lines.append(
            model_level.sort_values("family")[
                ["family", "pair", "n_models", "spearman_rho", "p_value"]
            ].round(4).to_string(index=False)
        )
    lines.append("```")
    lines.append("")

    lines.extend(["## Per-model instance Spearman (selected pairs)", ""])
    show = by_model[by_model["corr_eligible"]].copy()
    show["abs_rho"] = show["spearman_rho"].abs()
    show = show.sort_values("abs_rho", ascending=False)
    if not show.empty:
        lines.append("```")
        lines.append(
            show.head(20)[["family", "model", "pair", "n", "spearman_rho", "p_value"]].round(4).to_string(index=False)
        )
        lines.append("```")
    else:
        lines.append("_No eligible instance-level correlations (need n≥5 and variation)._")
    lines.append("")

    lines.extend(["## Triple-probe agreement (threshold flags)", ""])
    lines.append(
        f"- P1 fragile: w3_drop≥{P1_W3_DROP_THRESH} or vri_gap>0.5 · P2 low: cci≤{P2_CCI_LOW} · P3 high: contam≥{P3_CONTAM_HIGH}"
    )
    lines.append("")
    if not agreement.empty:
        lines.append("```")
        lines.append(
            agreement.sort_values(["family", "model"])[
                [
                    "family",
                    "model",
                    "n_instances",
                    "triple_retrieval_agree_pct",
                    "triple_computation_agree_pct",
                    "triple_mixed_pct",
                    "p2_cci_valid_n",
                ]
            ].round(3).to_string(index=False)
        )
        lines.append("```")
    lines.append("")

    tot_ret = int(frame["triple_retrieval_agree"].sum())
    tot_comp = int(frame["triple_computation_agree"].sum())
    tot_n = len(frame)
    lines.extend(
        [
            f"**All-family totals:** retrieval-agree {tot_ret}/{tot_n} ({100*tot_ret/tot_n:.1f}%) · "
            f"computation-agree {tot_comp}/{tot_n} ({100*tot_comp/tot_n:.1f}%)",
            "",
            "## Accuracy vs W3 retention (across models, refreshed)",
            "",
            "```",
        ]
    )
    if not acc_w3.empty:
        lines.append(acc_w3.round(4).to_string(index=False))
    lines.append("```")
    lines.extend(
        [
            "",
            "## Coverage caveats",
            "",
            "- **BW:** P2 CCI sparse in triangulation labels — merged from `BW_P2_cci.csv` where available.",
            "- **GSM:** P3 contam missing for ~40 instances; P2 CCI missing for ~84.",
            "- **ALGO:** P2 CCI only on adversarial subset (~244/550 instances).",
            "",
            "## Files",
            "",
            "- `cross_probe_instance_frame.csv`",
            "- `cross_probe_spearman_by_model.csv`",
            "- `cross_probe_spearman_model_level.csv`",
            "- `cross_probe_agreement_instances.csv`",
            "- `cross_probe_acc_vs_w3retention.csv`",
            "",
        ]
    )
    (DER / "cross_probe_correlation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    frame = build_instance_frame()
    by_model = spearman_by_model(frame)
    model_level = spearman_model_level(frame)
    agreement = agreement_summary(frame)
    acc_w3 = refresh_acc_w3retention(frame)

    # Instance-level export: rows with any triple agreement flag
    inst_agree = frame[
        frame["triple_retrieval_agree"] | frame["triple_computation_agree"] | frame["triple_any_disagree"]
    ][
        [
            "family",
            "problem_id",
            "model_short",
            "p1_w3_drop",
            "p1_w3_retention",
            "p2_cci",
            "p3_contam",
            "triple_retrieval_agree",
            "triple_computation_agree",
            "triple_any_disagree",
        ]
    ].copy()

    frame.to_csv(DER / "cross_probe_instance_frame.csv", index=False)
    by_model.to_csv(DER / "cross_probe_spearman_by_model.csv", index=False)
    model_level.to_csv(DER / "cross_probe_spearman_model_level.csv", index=False)
    agreement.to_csv(DER / "cross_probe_agreement_instances.csv", index=False)
    inst_agree.to_csv(DER / "cross_probe_triple_flagged_instances.csv", index=False)
    acc_w3.to_csv(DER / "cross_probe_acc_vs_w3retention.csv", index=False)
    write_summary(frame, by_model, model_level, agreement, acc_w3)

    print("Wrote cross-probe correlation pack:")
    for name in [
        "cross_probe_instance_frame.csv",
        "cross_probe_spearman_by_model.csv",
        "cross_probe_spearman_model_level.csv",
        "cross_probe_agreement_instances.csv",
        "cross_probe_triple_flagged_instances.csv",
        "cross_probe_acc_vs_w3retention.csv",
        "cross_probe_correlation_summary.md",
    ]:
        print(f"  results/derived/{name}")


if __name__ == "__main__":
    main()
