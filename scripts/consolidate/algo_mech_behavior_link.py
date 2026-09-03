#!/usr/bin/env python3
"""N3: ALGO mechanistic-behavioral validation (Llama-3.1-8B + Qwen2.5-1.5B only).

Correlate canonical→W3 final-layer gold-token rank shift against W3 correctness.
Framed as validation, not mechanism.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.cluster_inference import cluster_bootstrap_assoc  # noqa: E402
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
MECH_IN = REPO_ROOT / "Mechanistic Frequency Controlled Algorithm.csv"
OUT = DER / "N3_algo_mech_behavior_link.csv"
INST_OUT = DER / "N3_algo_mech_behavior_instances.csv"
QWEN_SCORES = DER / "N3_qwen_algo_w3_scores.csv"
QWEN_SCORER = REPO_ROOT / "scripts/consolidate/qwen_algo_w3_offline_score.py"

MECH_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "label": "Llama-3.1-8B",
        "p1_model": "meta-llama/llama-3.1-8b-instruct",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "label": "Qwen2.5-1.5B",
        "p1_model": None,
    },
}
FROZEN_P2 = REPO_ROOT / "results/raw/ALGO_P2_phase1_claude_new.csv"
N_BOOT = 5000
SEED = 42


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_w3_scores() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("ALGO_P1_*rescored.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[_is_true(df["included"])].copy()
        df = filter_excluded(df, family="ALGO")
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["w3_ok"] = _is_true(ok)
        parts.append(df[df["variant"] == "W3"][["problem_id", "model", "w3_ok"]])
    llama = pd.concat(parts, ignore_index=True).drop_duplicates(["problem_id", "model"], keep="last")
    llama = llama[llama["model"] == "meta-llama/llama-3.1-8b-instruct"].copy()

    if not QWEN_SCORES.exists():
        print("Running offline Qwen W3 scorer (first time)...")
        subprocess.run([sys.executable, str(QWEN_SCORER)], check=True)
    qwen = pd.read_csv(QWEN_SCORES, dtype=str).fillna("")
    qwen["w3_ok"] = qwen["w3_ok"].astype(str).str.lower().isin({"true", "1", "yes"})
    qwen = qwen[["problem_id", "w3_ok"]].copy()
    qwen["mech_model_key"] = "Qwen/Qwen2.5-1.5B-Instruct"

    llama["mech_model_key"] = "meta-llama/Llama-3.1-8B-Instruct"
    return pd.concat([llama[["problem_id", "w3_ok", "mech_model_key"]], qwen], ignore_index=True)


def _final_ranks(mech: pd.DataFrame) -> pd.DataFrame:
    mech = mech.copy()
    mech["layer"] = pd.to_numeric(mech["layer"], errors="coerce")
    mech["rank"] = pd.to_numeric(mech["rank"], errors="coerce")
    mech["n_layers"] = pd.to_numeric(mech["n_layers"], errors="coerce")
    mech["final_layer"] = mech["n_layers"] - 1
    fin = mech[mech["layer"] == mech["final_layer"]].copy()
    fin = fin[fin["variant"].isin(["canonical", "W3"])].copy()
    can = fin[fin["variant"] == "canonical"][["model", "problem_id", "rank"]].rename(columns={"rank": "rank_canonical"})
    w3 = fin[fin["variant"] == "W3"][["model", "problem_id", "rank"]].rename(columns={"rank": "rank_w3"})
    merged = can.merge(w3, on=["model", "problem_id"], how="inner")
    merged["rank_shift_canonical_minus_w3"] = merged["rank_canonical"] - merged["rank_w3"]
    return merged


def main() -> None:
    if not MECH_IN.exists():
        raise FileNotFoundError(MECH_IN)
    adv = set(
        pd.read_csv(FROZEN_P2, dtype=str)
        .loc[lambda d: d["instance_type"].str.lower() == "adversarial", "problem_id"]
        .astype(str),
    )

    mech = pd.read_csv(MECH_IN, dtype=str).fillna("")
    mech = mech[mech["model"].isin(MECH_MODELS.keys())].copy()
    mech = mech[mech["problem_id"].isin(adv)].copy()

    ranks = _final_ranks(mech)
    w3 = _load_w3_scores()
    ranks = ranks.merge(w3, left_on=["model", "problem_id"], right_on=["mech_model_key", "problem_id"], how="inner")
    ranks = ranks.drop(columns=["mech_model_key"])
    ranks["cluster_id"] = cluster_ids_for(ranks["problem_id"].astype(str).tolist())

    label_map = {k: v["label"] for k, v in MECH_MODELS.items()}
    inst = ranks.rename(columns={"model": "mech_model_key"}).copy()
    inst["model"] = inst["mech_model_key"].map(label_map)
    inst.to_csv(INST_OUT, index=False)

    rows: list[dict] = []
    for mech_model, meta in MECH_MODELS.items():
        sub = inst[inst["mech_model_key"] == mech_model].copy()
        if sub.empty:
            continue
        sub["w3_ok"] = sub["w3_ok"].astype(bool).astype(int)
        if sub["w3_ok"].nunique() < 2 or sub["rank_shift_canonical_minus_w3"].nunique() < 2:
            rows.append(
                {
                    "model": meta["label"],
                    "n": len(sub),
                    "n_clusters": sub["cluster_id"].nunique(),
                    "spearman_rho": "",
                    "ci_low": "",
                    "ci_high": "",
                    "p_value": "",
                    "p_value_method": "cluster_bootstrap_two_sided",
                    "x": "rank_shift_canonical_minus_w3",
                    "y": "w3_correct",
                    "bootstrap": "cluster_by_clone_family",
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "note": "insufficient variation — correlation undefined",
                }
            )
            continue
        res = cluster_bootstrap_assoc(
            sub["rank_shift_canonical_minus_w3"],
            sub["w3_ok"],
            sub["cluster_id"],
            kind="spearman",
            n_boot=N_BOOT,
            seed=SEED,
        )
        rows.append(
            {
                "model": meta["label"],
                "n": res["n"],
                "n_clusters": res["n_clusters"],
                "spearman_rho": round(res["estimate"], 4) if res["estimate"] == res["estimate"] else "",
                "ci_low": round(res["ci_low"], 4) if res["ci_low"] == res["ci_low"] else "",
                "ci_high": round(res["ci_high"], 4) if res["ci_high"] == res["ci_high"] else "",
                "p_value": round(res["p_clustered"], 4) if res["p_clustered"] == res["p_clustered"] else "",
                "p_value_method": "cluster_bootstrap_two_sided",
                "x": "rank_shift_canonical_minus_w3",
                "y": "w3_correct",
                "bootstrap": "cluster_by_clone_family",
                "n_boot": N_BOOT,
                "seed": SEED,
                "note": "Validation only — rank shift vs W3 correctness; not a mechanism claim",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {INST_OUT} ({len(inst)} rows)")
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
