"""Analysis from scored.csv. Items (problems) are the sampling unit; samples estimate within-item noise.

Every interval is a cluster bootstrap over problems. Every variant effect is paired with the same
problem's canonical. Tables are written for the full usable bank and for the verified tier.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

FLOOR, CEIL, B = 0.30, 0.95, 4000
RNG = np.random.default_rng(20261113)


def _boot_mean(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _signflip_p(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    signs = RNG.choice([-1.0, 1.0], (B, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + (null >= abs(d.mean()) - 1e-12).sum()) / (B + 1))


def _holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (len(p) - rank)))
        adj[i] = run
    return adj


def _pass_all(c: int, n: int, k: int) -> float:
    """Unbiased estimate of P(all k of k fresh samples correct) from c correct out of n."""
    return comb(c, k) / comb(n, k) if n >= k else np.nan


def exclusions(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["experiment", "model", "thinking", "instruction"], dropna=False)
    out = g.apply(lambda x: pd.Series({
        "calls": len(x), "api_error": int((x.exclude_reason == "api_error").sum()),
        "truncated": int((x.exclude_reason == "truncated").sum()), "usable": int(x.usable.sum()),
        "marker_rate": x.loc[x.usable, "marker"].mean() if x.usable.any() else np.nan,
        "format_fail_rate": (x.loc[x.usable, "parse_status"] != "parsed").mean() if x.usable.any() else np.nan,
        "median_completion_tokens": pd.to_numeric(x.completion_tokens, errors="coerce").median(),
    }), include_groups=False)
    return out.reset_index()


def surface_table(df: pd.DataFrame, score: str = "correct", group: str = "subtype") -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["model", "thinking", "instruction", group]):
        per = g.groupby(["problem_id", "variant"])[score].agg(["mean", "sum", "count"]).reset_index()
        acc = per.pivot(index="problem_id", columns="variant", values="mean")
        if "canonical" not in acc:
            continue
        block = []
        for var in [c for c in acc.columns]:
            x = acc[var].dropna().to_numpy()
            lo, hi = _boot_mean(x)
            sub = per[per.variant == var]
            k = int(sub["count"].min()) if len(sub) else 0
            row = dict(zip(["model", "thinking", "instruction", group], keys), variant=var, n_problems=len(x),
                       samples_per_item=k, accuracy=x.mean() if len(x) else np.nan, ci_lo=lo, ci_hi=hi,
                       pass_all_k=np.mean([_pass_all(int(s), int(n), k) for s, n in zip(sub["sum"], sub["count"])]) if k else np.nan,
                       pass_any_k=(sub["sum"] > 0).mean() if len(sub) else np.nan)
            if var != "canonical":
                pair = acc[["canonical", var]].dropna()
                d = (pair[var] - pair["canonical"]).to_numpy()
                dlo, dhi = _boot_mean(d)
                canon = pair["canonical"].mean() if len(pair) else np.nan
                row.update(n_paired=len(pair), canonical_paired=canon, diff_vs_canonical=d.mean() if len(d) else np.nan,
                           diff_ci_lo=dlo, diff_ci_hi=dhi, p_signflip=_signflip_p(d),
                           retention=(pair[var].mean() / canon) if len(pair) and canon >= FLOOR else np.nan,
                           note="canonical_below_floor" if len(pair) and canon < FLOOR else ("canonical_at_ceiling" if len(pair) and canon >= CEIL else ""))
            block.append(row)
        ps = [r["p_signflip"] for r in block if "p_signflip" in r]
        for r, a in zip([r for r in block if "p_signflip" in r], _holm(ps) if ps else []):
            r["p_holm"] = a
        rows += block
    return pd.DataFrame(rows)


def instruction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Does the variant effect depend on the instruction? Difference-in-differences against 'none'."""
    rows = []
    for (model, thinking, sub), g in df.groupby(["model", "thinking", "subtype"]):
        acc = g.groupby(["problem_id", "instruction", "variant"]).correct.mean().unstack(["instruction", "variant"])
        for ins in sorted({c[0] for c in acc.columns}):
            for var in sorted({c[1] for c in acc.columns} - {"canonical"}):
                need = [(ins, var), (ins, "canonical")]
                if not all(c in acc.columns for c in need):
                    continue
                eff = (acc[(ins, var)] - acc[(ins, "canonical")])
                row = dict(model=model, thinking=thinking, subtype=sub, instruction=ins, variant=var,
                           acc_canonical=acc[(ins, "canonical")].mean(), acc_variant=acc[(ins, var)].mean())
                e = eff.dropna().to_numpy()
                row.update(n=len(e), variant_effect=e.mean() if len(e) else np.nan, **dict(zip(["ci_lo", "ci_hi"], _boot_mean(e))))
                if ins != "none" and all(c in acc.columns for c in [("none", var), ("none", "canonical")]):
                    did = (eff - (acc[("none", var)] - acc[("none", "canonical")])).dropna().to_numpy()
                    row.update(did_vs_none=did.mean() if len(did) else np.nan, did_p=_signflip_p(did),
                               **dict(zip(["did_ci_lo", "did_ci_hi"], _boot_mean(did))))
                rows.append(row)
    return pd.DataFrame(rows)


def noise_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, thinking, fam), g in df.groupby(["model", "thinking", "family"]):
        per = g.groupby("problem_id").correct.agg(["sum", "count"])
        p = per["sum"] / per["count"]
        n = int(per["count"].min())
        row = dict(model=model, thinking=thinking, family=fam, n_items=len(per), samples_per_item=n, accuracy=p.mean(),
                   unstable_items=((p > 0) & (p < 1)).mean(), mean_within_item_sd=np.sqrt(p * (1 - p)).mean())
        for k in (1, 4, 8):
            row[f"pass_all_{k}"] = np.mean([_pass_all(int(s), int(c), k) for s, c in zip(per["sum"], per["count"])])
        # smallest paired accuracy difference detectable at 80% power with this many items and k samples
        for k in (1, 5):
            se = np.sqrt(2 * (p * (1 - p)).mean() / k / max(len(per), 1) + 1e-12)
            row[f"min_detectable_diff_k{k}"] = 2.8 * se
        rows.append(row)
    return pd.DataFrame(rows)


def p2_table(plan: pd.DataFrame, cont: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, thinking), g in plan.groupby(["model", "thinking"]):
        c = cont[(cont.model == model) & (cont.thinking == thinking)]
        row = dict(model=model, thinking=thinking, plan_sessions=len(g), plan_correct=g.correct.mean(),
                   eligible_sessions=c.groupby(["item_id", "sample"]).ngroups)
        acc = c.groupby(["problem_id", "arm"]).correct.mean().unstack("arm")
        for arm in acc.columns:
            x = acc[arm].dropna().to_numpy()
            row[f"acc_{arm}"] = x.mean()
            row[f"acc_{arm}_ci"] = "[%.3f, %.3f]" % _boot_mean(x)
        if {"control_true", "inject_false"} <= set(acc.columns):
            d = (acc["inject_false"] - acc["control_true"]).dropna().to_numpy()
            row.update(carry_forward_effect=d.mean(), carry_p=_signflip_p(d))
        if {"inject_cued", "inject_false"} <= set(acc.columns):
            d = (acc["inject_cued"] - acc["inject_false"]).dropna().to_numpy()
            row.update(cue_recovery_effect=d.mean(), cue_p=_signflip_p(d))
        rows.append(row)
    return pd.DataFrame(rows)


def run(d: Path) -> Path:
    df = pd.read_csv(d / "scored.csv", keep_default_na=False, na_values=[""])
    for col in ("usable", "correct", "correct_strict", "marker"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    out = d / "analysis"
    out.mkdir(exist_ok=True)
    exclusions(df).to_csv(out / "exclusions.csv", index=False)
    use = df[df.usable].copy()
    md = [f"# Analysis: {d.name}", "", "Items are the unit. Intervals are 95% cluster bootstrap over problems.",
          f"Retention is reported only when paired canonical accuracy is at least {FLOOR}.", ""]

    def emit(name: str, table: pd.DataFrame, title: str):
        if table is None or table.empty:
            return
        table.to_csv(out / f"{name}.csv", index=False)
        md.extend([f"## {title}", "", table.round(3).to_markdown(index=False), ""])

    emit("exclusions", exclusions(df), "Call accounting")
    p1 = use[use.experiment == "p1_surface"]
    if len(p1):
        emit("p1_by_subtype", surface_table(p1), "P1 surface variants, by subtype")
        emit("p1_by_family", surface_table(p1, group="family"), "P1 surface variants, by family")
        emit("p1_verified_only", surface_table(p1[p1.tier == "verified"]), "P1, verified tier only (sensitivity)")
        emit("p1_strict_marker", surface_table(p1, score="correct_strict"), "P1, strict scoring: marker required (sensitivity)")
    e1 = use[use.experiment == "e1_instruction"]
    if len(e1):
        emit("e1_instruction", instruction_table(e1), "E1 instruction x variant")
    e2 = use[use.experiment == "e2_noise"]
    if len(e2):
        emit("e2_noise", noise_table(e2), "E2 noise floor and pass^k")
    pa, pb = use[use.experiment == "p2_plan"], use[use.experiment == "p2_continue"]
    if len(pa) and len(pb):
        emit("p2_false_state", p2_table(pa, pb), "P2 continuation under a false intermediate value")
    (out / "REPORT.md").write_text("\n".join(md))
    return out / "REPORT.md"
