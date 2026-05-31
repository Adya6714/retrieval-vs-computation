#!/usr/bin/env python3
"""Step 9 — VRI / rename-type analysis (existing P1 data only).

VRI = mean(Acc_W1, Acc_W2, Acc_W4) - Acc_W3

Outputs:
    results/derived/vri_by_model.csv
    results/derived/vri_by_subtype.csv
    results/derived/vri_proximity_correlation.csv
    results/derived/vri_analysis_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DER = ROOT / "results" / "derived"
AUD = ROOT / "results" / "paper" / "AUDIT"

SHORT = {
    "Claude": "Claude",
    "Gemini": "Gemini",
    "GPT-4o": "GPT-4o",
    "Llama": "Llama",
    "o4-mini": "o4-mini",
}

PARTIAL_BANK = {
    ("GSM", "GPT-4o"): "20/44 bank-valid",
    ("GSM", "Llama"): "20/44 bank-valid",
}


def _vri(acc: dict[str, float]) -> float:
    w1 = acc.get("W1", float("nan"))
    w2 = acc.get("W2", float("nan"))
    w3 = acc.get("W3", float("nan"))
    w4 = acc.get("W4", float("nan"))
    if any(pd.isna(x) for x in (w1, w2, w3, w4)):
        return float("nan")
    return float((w1 + w2 + w4) / 3 - w3)


def load_family_model_vri() -> pd.DataFrame:
    p1 = pd.read_csv(DER / "probe1_per_model_variant.csv")
    w3 = pd.read_csv(DER / "probe1_w3_retention.csv")
    rows: list[dict] = []
    for (probe, model), g in p1.groupby(["probe", "model"]):
        acc = dict(zip(g["variant"], g["accuracy"]))
        canon_n = int(g.loc[g["variant"] == "canonical", "n_valid"].iloc[0]) if "canonical" in acc else 0
        bank_note = PARTIAL_BANK.get((probe, model), "")
        rows.append(
            {
                "family": probe,
                "model": model,
                "acc_canonical": acc.get("canonical", float("nan")),
                "acc_w3": acc.get("W3", float("nan")),
                "vri": _vri(acc),
                "canonical_n_valid": canon_n,
                "bank_status": bank_note or "full_bank",
            }
        )
    df = pd.DataFrame(rows)
    w3_small = w3.rename(columns={"W3_retention": "w3_retention"})
    df = df.merge(
        w3_small[["probe", "model", "w3_retention"]],
        left_on=["family", "model"],
        right_on=["probe", "model"],
        how="left",
    ).drop(columns=["probe"])
    return df


def load_subtype_vri() -> pd.DataFrame:
    path = DER / "P1_metrics_by_model_subtype.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows: list[dict] = []
    for _, r in df.iterrows():
        acc = {
            "W1": r.get("acc_W1", float("nan")),
            "W2": r.get("acc_W2", float("nan")),
            "W3": r.get("acc_W3", float("nan")),
            "W4": r.get("acc_W4", float("nan")),
        }
        rows.append(
            {
                "family": r.get("family", ""),
                "model": str(r.get("model", "")).split("/")[-1],
                "subtype": r.get("subtype", ""),
                "vri": _vri(acc),
                "acc_w3": acc["W3"],
                "acc_canonical": r.get("acc_canonical", float("nan")),
                "n_problems": r.get("n_problems", ""),
            }
        )
    return pd.DataFrame(rows)


def proximity_vri_algo() -> pd.DataFrame:
    tri_path = DER / "ALGO_P3_triangulation_v2.csv"
    if not tri_path.exists():
        tri_path = ROOT / "results" / "derived" / "ALGO_P3_triangulation_v2.csv"
    if not tri_path.exists():
        return pd.DataFrame()
    tri = pd.read_csv(tri_path)
    tri = tri[tri.get("instance_type", "").astype(str).str.lower() == "adversarial"].copy()
    if tri.empty or "VRI_gap" not in tri.columns:
        return pd.DataFrame()
    tri["proximity"] = pd.to_numeric(tri.get("instance_contamination_score", ""), errors="coerce")
    tri["vri_gap"] = pd.to_numeric(tri["VRI_gap"], errors="coerce")
    rows: list[dict] = []
    for model, g in tri.groupby("model"):
        gv = g[g["proximity"].notna() & g["vri_gap"].notna()]
        if len(gv) < 5:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = stats.spearmanr(gv["proximity"], gv["vri_gap"])
        rows.append(
            {
                "family": "ALGO",
                "model": model.split("/")[-1] if "/" in str(model) else model,
                "n_adversarial": len(gv),
                "spearman_proximity_vri_gap": float(rho),
                "p_value": float(p),
                "mean_vri_gap": float(gv["vri_gap"].mean()),
            }
        )
    pooled = tri[tri["proximity"].notna() & tri["vri_gap"].notna()]
    if len(pooled) >= 5:
        rho, p = stats.spearmanr(pooled["proximity"], pooled["vri_gap"])
        rows.append(
            {
                "family": "ALGO",
                "model": "pooled",
                "n_adversarial": len(pooled),
                "spearman_proximity_vri_gap": float(rho),
                "p_value": float(p),
                "mean_vri_gap": float(pooled["vri_gap"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_summary(by_model: pd.DataFrame, by_sub: pd.DataFrame, prox: pd.DataFrame) -> None:
    lines = [
        "# VRI analysis (Step 9)",
        "",
        "VRI = (Acc_W1 + Acc_W2 + Acc_W4) / 3 − Acc_W3. Positive VRI → W3 hurts more than vocabulary-preserving variants.",
        "",
        "**Data quality:** GSM GPT-4o/Llama computed on **20/44 bank-valid** IDs only — interpret with caution.",
        "",
        "## VRI vs W3 retention (by family × model)",
        "",
        "```",
        by_model.sort_values(["family", "model"])[
            ["family", "model", "acc_canonical", "acc_w3", "vri", "w3_retention", "bank_status"]
        ].round(3).to_string(index=False),
        "```",
        "",
    ]
    if not by_sub.empty:
        lines.extend(["## VRI by subtype (ALGO)", "", "```"])
        algo = by_sub[by_sub["family"] == "ALGO"].sort_values(["model", "subtype"])
        lines.append(algo.round(3).to_string(index=False))
        lines.append("```")
        lines.append("")

    if not prox.empty:
        lines.extend(["## Proximity × VRI_gap (ALGO adversarial instances)", "", "```"])
        lines.append(prox.round(4).to_string(index=False))
        lines.append("```")
        lines.append("")

    top = by_model.dropna(subset=["vri"]).sort_values("vri", ascending=False).head(5)
    lines.extend(["## Highest VRI (rename-specific fragility)", ""])
    for _, r in top.iterrows():
        lines.append(f"- **{r['family']} {r['model']}**: VRI={r['vri']:.3f} (W3 retention={r['w3_retention']:.3f})")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `vri_by_model.csv`",
            "- `vri_by_subtype.csv`",
            "- `vri_proximity_correlation.csv`",
            "",
        ]
    )
    (DER / "vri_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    by_model = load_family_model_vri()
    by_sub = load_subtype_vri()
    prox = proximity_vri_algo()

    by_model.to_csv(DER / "vri_by_model.csv", index=False)
    by_sub.to_csv(DER / "vri_by_subtype.csv", index=False)
    prox.to_csv(DER / "vri_proximity_correlation.csv", index=False)
    write_summary(by_model, by_sub, prox)

    print("Wrote VRI analysis:")
    for name in ["vri_by_model.csv", "vri_by_subtype.csv", "vri_proximity_correlation.csv", "vri_analysis_summary.md"]:
        print(f"  results/derived/{name}")


if __name__ == "__main__":
    main()
