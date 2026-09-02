#!/usr/bin/env python3
"""I2: Cluster-bootstrap ALGO CIs, resampling clone families not problems.

Does not write results/raw/. Does not call any model API.

Old interval: 10k iid percentile bootstrap on problem-level 0/1 (probes.common.stats.bootstrap_ci).
New interval: 10k cluster bootstrap (same mean estimand, families resampled).
Wilson is also reported because the paper/appendix name Wilson while Probe-1
metric code uses the 10k bootstrap; rebuild/NUMBERS.csv uses Wilson.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.stats import bootstrap_ci, cluster_bootstrap_ci, wilson_ci  # noqa: E402

# Frozen adversarial pool (paper Table 5/7 challenging cells).
# 34 SP + 10 CC + 17 WIS = 61. Source: Claude P1 difficulty_params_instance_type.
PAPER_ADV = {
    "SP": [
        "SP_003", "SP_004", "SP_005", "SP_019", "SP_020", "SP_021", "SP_023",
        "SP_024", "SP_026", "SP_027", "SP_028", "SP_029", "SP_030", "SP_037",
        "SP_038", "SP_039", "SP_040", "SP_042", "SP_044", "SP_045", "SP_046",
        "SP_047", "SP_048", "SP_062", "SP_063", "SP_064", "SP_065", "SP_066",
        "SP_068", "SP_069", "SP_070", "SP_071", "SP_072", "SP_073",
    ],
    "CC": [f"CC_{i:02d}" for i in range(1, 11)],
    "WIS": [
        "WIS_003", "WIS_004", "WIS_013", "WIS_014", "WIS_015", "WIS_016",
        "WIS_017", "WIS_018", "WIS_019", "WIS_020", "WIS_023", "WIS_024",
        "WIS_025", "WIS_026", "WIS_027", "WIS_028", "WIS_029",
    ],
}
PAPER_ADV_ALL = set(PAPER_ADV["SP"] + PAPER_ADV["CC"] + PAPER_ADV["WIS"])
SUBTYPE_SHORT = {"coin_change": "CC", "shortest_path": "SP", "wis": "WIS"}

RAW = REPO_ROOT / "results/raw"
DERIVED = REPO_ROOT / "results/derived"
BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
CLONES = DERIVED / "bank_clone_audit.csv"
OUT = DERIVED / "I2_algo_cluster_bootstrap.csv"
OUT_POOL = DERIVED / "I2_frozen61_cluster_effective_n.csv"

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
P1_FILES = {
    "Claude": RAW / "ALGO_P1_behavioral_claude.csv",
    "GPT-4o": RAW / "ALGO_P1_behavioral_gpt4o.csv",
    "Llama": RAW / "ALGO_P1_behavioral_llama.csv",
    "Gemini": RAW / "ALGO_P1_behavioral_gemini.csv",
    "o4-mini": RAW / "ALGO_P1_behavioral_o1mini.csv",
}
N_BOOT = 10_000
SEED = 42
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5"]


def _ok(df: pd.DataFrame) -> pd.Series:
    if "verified" in df.columns:
        return df["verified"].astype(str).str.strip().str.lower().eq("true")
    if "behavioral_correct" in df.columns:
        return df["behavioral_correct"].astype(str).str.strip().str.lower().eq("true")
    raise KeyError("no verified column")


def _load_p1() -> pd.DataFrame:
    parts = []
    for label, path in P1_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[df["model"].astype(str).str.lower() != "mock"]
        long = next(k for k, v in SHORT.items() if v == label)
        df = df[df["model"] == long]
        df["model_short"] = label
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    vt = out["variant_type"].astype(str).str.strip()
    vt = vt.where(~vt.str.lower().eq("canonical"), "canonical")
    vt = vt.where(~vt.str.lower().str.fullmatch(r"w[1-6]"), vt.str.upper())
    out["variant_type"] = vt
    out = out.drop_duplicates(["model_short", "problem_id", "variant_type"], keep="last")
    out = filter_excluded(out, family="ALGO")
    out["ok"] = _ok(out).astype(float)
    return out


def _cluster_map() -> dict[str, str]:
    clones = pd.read_csv(CLONES, dtype=str).fillna("")
    clones = clones[clones["family"] == "ALGO"]
    m = {}
    for _, r in clones.iterrows():
        pid = str(r["problem_id"]).strip()
        cid = str(r["clone_family_id"]).strip()
        m[pid] = cid if cid else f"SINGLETON_{pid}"
    return m


def _ci_row(
    *,
    cell: str,
    model: str,
    subtype: str,
    variant: str,
    pool: str,
    vals: list[float],
    clusters: list[str],
) -> dict:
    n = len(vals)
    k = int(sum(vals))
    acc = (k / n) if n else float("nan")
    n_fam = len(set(clusters))
    np.random.seed(SEED)
    iid_lo, iid_hi = bootstrap_ci(vals, n_resamples=N_BOOT)
    cl_lo, cl_hi = cluster_bootstrap_ci(vals, clusters, n_resamples=N_BOOT, seed=SEED)
    w_lo, w_hi = wilson_ci(k, n)
    iid_w = (iid_hi - iid_lo) if n else float("nan")
    cl_w = (cl_hi - cl_lo) if n else float("nan")
    w_w = (w_hi - w_lo) if n else float("nan")
    ratio = (cl_w / iid_w) if iid_w and iid_w > 0 else float("nan")
    return {
        "cell": cell,
        "model": model,
        "subtype": subtype,
        "variant": variant,
        "pool": pool,
        "k": k,
        "n": n,
        "n_clone_families": n_fam,
        "acc": acc,
        "iid_bootstrap_lo": iid_lo,
        "iid_bootstrap_hi": iid_hi,
        "iid_width": iid_w,
        "cluster_bootstrap_lo": cl_lo,
        "cluster_bootstrap_hi": cl_hi,
        "cluster_width": cl_w,
        "width_ratio_cluster_over_iid": ratio,
        "wilson_lo": w_lo,
        "wilson_hi": w_hi,
        "wilson_width": w_w,
        "note": "iid=10k problem bootstrap (metric scripts); cluster=10k family resample; wilson=paper name",
    }


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    p1 = _load_p1()
    bank = pd.read_csv(BANK, dtype=str).fillna("")
    can = bank[bank["variant_type"].astype(str).str.lower() == "canonical"][
        ["problem_id", "problem_subtype"]
    ].drop_duplicates("problem_id")
    if "problem_subtype" in p1.columns:
        p1 = p1.drop(columns=["problem_subtype"])
    p1 = p1.merge(can, on="problem_id", how="left")
    cmap = _cluster_map()
    p1["cluster"] = p1["problem_id"].map(lambda x: cmap.get(str(x), f"SINGLETON_{x}"))

    # Frozen 61 effective n
    pool_ids = list(PAPER_ADV_ALL)
    pool_clusters = {pid: cmap.get(pid, f"SINGLETON_{pid}") for pid in pool_ids}
    fams = sorted(set(pool_clusters.values()))
    sizes = pd.Series(list(pool_clusters.values())).value_counts()
    pool_rows = [
        {
            "pool": "frozen_61_adversarial",
            "n_ids": len(pool_ids),
            "effective_n_clone_families": len(fams),
            "n_ids_in_multi_member_families": int(sum(1 for pid, c in pool_clusters.items() if sizes[c] > 1)),
            "largest_family_in_pool": int(sizes.max()) if len(sizes) else 0,
            "note": "34 SP + 10 CC + 17 WIS; family IDs from bank_clone_audit.csv restricted to the 61",
        }
    ]
    for sub, ids in PAPER_ADV.items():
        sub_c = [cmap.get(pid, f"SINGLETON_{pid}") for pid in ids]
        pool_rows.append(
            {
                "pool": f"frozen_61_{sub}",
                "n_ids": len(ids),
                "effective_n_clone_families": len(set(sub_c)),
                "n_ids_in_multi_member_families": int(
                    sum(1 for pid, c in zip(ids, sub_c) if sub_c.count(c) > 1)
                ),
                "largest_family_in_pool": int(pd.Series(sub_c).value_counts().max()) if sub_c else 0,
                "note": "",
            }
        )
    pd.DataFrame(pool_rows).to_csv(OUT_POOL, index=False)

    rows: list[dict] = []
    # Full bank: model × variant (pooled subtypes) and model × subtype × variant
    for model, g_m in p1.groupby("model_short"):
        for variant in VARIANTS:
            g = g_m[g_m["variant_type"] == variant]
            if g.empty:
                continue
            rows.append(
                _ci_row(
                    cell=f"{model}|ALL|{variant}|bank110",
                    model=str(model),
                    subtype="ALL",
                    variant=variant,
                    pool="bank_110",
                    vals=g["ok"].astype(float).tolist(),
                    clusters=g["cluster"].astype(str).tolist(),
                )
            )
        for subtype, g_s in g_m.groupby("problem_subtype"):
            for variant in VARIANTS:
                g = g_s[g_s["variant_type"] == variant]
                if g.empty:
                    continue
                rows.append(
                    _ci_row(
                        cell=f"{model}|{subtype}|{variant}|bank110",
                        model=str(model),
                        subtype=str(subtype),
                        variant=variant,
                        pool="bank_110_subtype",
                        vals=g["ok"].astype(float).tolist(),
                        clusters=g["cluster"].astype(str).tolist(),
                    )
                )
            # Table 7 chall / std slices
            short = SUBTYPE_SHORT.get(str(subtype), str(subtype))
            chall_ids = set(PAPER_ADV.get(short, []))
            for variant in VARIANTS:
                g = g_s[g_s["variant_type"] == variant]
                g_ch = g[g["problem_id"].isin(chall_ids)]
                g_st = g[~g["problem_id"].isin(chall_ids)]
                if not g_ch.empty:
                    rows.append(
                        _ci_row(
                            cell=f"{model}|{short}|{variant}|table7_chall",
                            model=str(model),
                            subtype=short,
                            variant=variant,
                            pool="table7_chall",
                            vals=g_ch["ok"].astype(float).tolist(),
                            clusters=g_ch["cluster"].astype(str).tolist(),
                        )
                    )
                if not g_st.empty:
                    rows.append(
                        _ci_row(
                            cell=f"{model}|{short}|{variant}|table7_std",
                            model=str(model),
                            subtype=short,
                            variant=variant,
                            pool="table7_std",
                            vals=g_st["ok"].astype(float).tolist(),
                            clusters=g_st["cluster"].astype(str).tolist(),
                        )
                    )
        # Frozen 61 overall
        g61 = g_m[g_m["problem_id"].isin(PAPER_ADV_ALL)]
        for variant in VARIANTS:
            g = g61[g61["variant_type"] == variant]
            if g.empty:
                continue
            rows.append(
                _ci_row(
                    cell=f"{model}|ALL|{variant}|frozen61",
                    model=str(model),
                    subtype="ALL",
                    variant=variant,
                    pool="frozen_61",
                    vals=g["ok"].astype(float).tolist(),
                    clusters=g["cluster"].astype(str).tolist(),
                )
            )
        for sub, ids in PAPER_ADV.items():
            g_sub = g_m[g_m["problem_id"].isin(ids)]
            for variant in VARIANTS:
                g = g_sub[g_sub["variant_type"] == variant]
                if g.empty:
                    continue
                rows.append(
                    _ci_row(
                        cell=f"{model}|{sub}|{variant}|frozen61",
                        model=str(model),
                        subtype=sub,
                        variant=variant,
                        pool="frozen_61_subtype",
                        vals=g["ok"].astype(float).tolist(),
                        clusters=g["cluster"].astype(str).tolist(),
                    )
                )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} cells)")
    print(f"Wrote {OUT_POOL}")
    print(pd.DataFrame(pool_rows).to_string(index=False))
    sub = out[out["pool"] == "bank_110"]
    print(
        "bank_110 width ratio cluster/iid: "
        f"median={sub['width_ratio_cluster_over_iid'].median():.3f} "
        f"mean={sub['width_ratio_cluster_over_iid'].mean():.3f} "
        f"max={sub['width_ratio_cluster_over_iid'].max():.3f}"
    )
    sub61 = out[out["pool"] == "frozen_61"]
    print(
        "frozen_61 overall width ratio cluster/iid: "
        f"median={sub61['width_ratio_cluster_over_iid'].median():.3f} "
        f"mean={sub61['width_ratio_cluster_over_iid'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
