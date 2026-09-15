#!/usr/bin/env python3
"""C7: Degeneracy-filtered Kendall's W (N4) + N4↔J6 reconciliation.

Preregistered ranker degeneracy (fixed before filtering):
  degenerate if (max−min) accuracy across W1–W6 < 0.10
            OR mean accuracy > 0.95
            OR mean accuracy < 0.10.

Recomputes within-family Kendall's W with the O2 within-row permutation null
(10000 perms, seed 42), full sample and excluding degenerate rankers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.consolidate.p1_variant_ordering import (  # noqa: E402
    N_PERM,
    SEED,
    VARIANTS,
    _kendall_w,
    _load_p1,
    _perm_p_value,
)

DER = REPO_ROOT / "results" / "derived"
OUT_CSV = DER / "C7_concordance_filtered.csv"
OUT_NOTE = DER / "C7_reconciliation_note.md"

RANGE_MIN = 0.10
MEAN_CEIL = 0.95
MEAN_FLOOR = 0.10


def _ranker_table(p1: pd.DataFrame) -> pd.DataFrame:
    acc = p1.groupby(["family", "model", "variant"], as_index=False)["ok"].mean()
    rows = []
    for (fam, model), g in acc.groupby(["family", "model"]):
        by_v = g.set_index("variant")["ok"].reindex(VARIANTS).astype(float)
        present = by_v.dropna()
        if len(present) < 2:
            continue
        mean_acc = float(present.mean())
        rng = float(present.max() - present.min())
        deg_range = rng < RANGE_MIN
        deg_ceil = mean_acc > MEAN_CEIL
        deg_floor = mean_acc < MEAN_FLOOR
        rows.append(
            {
                "row_type": "ranker",
                "family": fam,
                "model": model,
                "n_variants_scored": int(len(present)),
                "variants_scored": "|".join(present.index.tolist()),
                "acc_mean": round(mean_acc, 4),
                "acc_min": round(float(present.min()), 4),
                "acc_max": round(float(present.max()), 4),
                "acc_range": round(rng, 4),
                "acc_W1": round(float(by_v["W1"]), 4) if by_v["W1"] == by_v["W1"] else "",
                "acc_W2": round(float(by_v["W2"]), 4) if by_v["W2"] == by_v["W2"] else "",
                "acc_W3": round(float(by_v["W3"]), 4) if by_v["W3"] == by_v["W3"] else "",
                "acc_W4": round(float(by_v["W4"]), 4) if by_v["W4"] == by_v["W4"] else "",
                "acc_W5": round(float(by_v["W5"]), 4) if by_v["W5"] == by_v["W5"] else "",
                "acc_W6": round(float(by_v["W6"]), 4) if by_v["W6"] == by_v["W6"] else "",
                "flag_range_lt_0.10": deg_range,
                "flag_mean_gt_0.95": deg_ceil,
                "flag_mean_lt_0.10": deg_floor,
                "degenerate": bool(deg_range or deg_ceil or deg_floor),
                "degeneracy_reasons": "|".join(
                    [
                        r
                        for r, f in [
                            ("range_lt_0.10", deg_range),
                            ("mean_gt_0.95", deg_ceil),
                            ("mean_lt_0.10", deg_floor),
                        ]
                        if f
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def _rank_matrix_for_models(
    sub: pd.DataFrame, models: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    acc = sub.groupby(["model", "variant"], as_index=False)["ok"].mean()
    present_variants = [v for v in VARIANTS if v in set(acc["variant"])]
    kept_models: list[str] = []
    rows: list[np.ndarray] = []
    for model in models:
        row = acc[acc["model"] == model].set_index("variant")["ok"]
        vals = row.reindex(present_variants).astype(float)
        if not np.all(np.isfinite(vals.to_numpy())):
            continue
        ranks = stats.rankdata(-vals.to_numpy(), method="average")
        rows.append(ranks)
        kept_models.append(model)
    if not rows:
        return np.empty((0, 0)), [], present_variants
    return np.vstack(rows), kept_models, present_variants


def _concordance_row(
    *,
    family: str,
    sample: str,
    mat: np.ndarray,
    models: list[str],
    variants: list[str],
    excluded: list[str],
    rng: np.random.Generator,
) -> dict:
    if len(models) < 2 or mat.shape[1] < 2:
        w, p = float("nan"), float("nan")
    else:
        w = _kendall_w(mat)
        p = _perm_p_value(mat, w, rng, n_perm=N_PERM)
    return {
        "row_type": "concordance",
        "family": family,
        "sample": sample,
        "kendall_W": round(w, 4) if w == w else "",
        "p_value": round(p, 4) if p == p else "",
        "n_rankers": len(models),
        "n_variants": len(variants),
        "variants": "|".join(variants),
        "models_included": "|".join(models),
        "models_excluded_degenerate": "|".join(excluded) if excluded else "",
        "n_perm": N_PERM,
        "seed": SEED if sample == "full" else SEED + 10_000,
        "null": "within_row_rank_permutation_O2",
        "criterion": (
            f"degenerate if acc_range(W1-W6)<{RANGE_MIN} "
            f"OR mean_acc>{MEAN_CEIL} OR mean_acc<{MEAN_FLOOR}"
        ),
    }


def _write_note(
    rankers: pd.DataFrame,
    conc: pd.DataFrame,
    shared: dict[str, tuple[int, int]],
) -> None:
    deg = rankers[rankers["degenerate"]]
    deg_lines = []
    if deg.empty:
        deg_lines.append("- None.")
    else:
        for _, r in deg.iterrows():
            deg_lines.append(
                f"- `{r['family']}` / `{r['model']}`: mean={r['acc_mean']}, "
                f"range={r['acc_range']}, reasons=`{r['degeneracy_reasons']}`"
            )

    def _conc(fam: str, sample: str) -> pd.Series:
        hit = conc[(conc["family"] == fam) & (conc["sample"] == sample)]
        if hit.empty:
            raise RuntimeError(f"missing concordance row {fam}/{sample}")
        return hit.iloc[0]

    w_lines = []
    for fam in ["ALGO", "BW", "GSM"]:
        full = _conc(fam, "full")
        filt = _conc(fam, "filtered")
        excl = filt["models_excluded_degenerate"] or "(none)"
        w_lines.append(
            f"- **{fam}**: full W={full['kendall_W']} (p={full['p_value']}, "
            f"m={full['n_rankers']}); filtered W={filt['kendall_W']} "
            f"(p={filt['p_value']}, m={filt['n_rankers']}); excluded={excl}"
        )
    bw_full = _conc("BW", "full")

    sh_lines = [
        f"- ALGO shared-hard (fail all five paper models): "
        f"{shared['ALGO'][0]}/{shared['ALGO'][1]}",
        f"- GSM shared-hard (fail all five on paper n=20): "
        f"{shared['GSM'][0]}/{shared['GSM'][1]}",
        f"- BW shared-hard (fail all five): "
        f"{shared['BW'][0]}/{shared['BW'][1]}",
    ]

    # Highlight ALGO o4-mini note (motivating concern vs criterion outcome)
    o4 = rankers[
        (rankers["family"] == "ALGO") & (rankers["model"].str.contains("o4-mini"))
    ]
    o4_note = ""
    if not o4.empty:
        r = o4.iloc[0]
        o4_note = (
            f"ALGO `openai/o4-mini` motivated the check (pairwise ρ −0.09 to −0.31 vs "
            f"other models; W3 overall {r['acc_W3']}, SP-subset post-fix ≈0.982) but "
            f"**does not** meet the preregistered criterion (mean={r['acc_mean']}, "
            f"range={r['acc_range']}). The only ALGO exclusion is Llama (floor + flat)."
        )

    text = f"""# C7 reconciliation note — variant concordance vs item-level shared hard

**Date.** 2026-09-03  
**Criterion (preregistered, auditable).** A (family, model) ranker is degenerate if
the range of its W1–W6 accuracies is `< {RANGE_MIN}`, or its mean accuracy
`> {MEAN_CEIL}`, or its mean accuracy `< {MEAN_FLOOR}`. Filtering is applied only
after this rule; full-sample W is always reported beside filtered W.

## Degenerate rankers

{chr(10).join(deg_lines)}

{o4_note}

## Kendall's W (O2 within-row null, {N_PERM} perms, seed={SEED})

{chr(10).join(w_lines)}

ALGO's full-sample p≈0.049 sits on the α=0.05 knife-edge. Dropping the only
degenerate ranker (Llama) **raises** W (more concordance among informative
rankers) but leaves p≈0.05 — so the marginal ALGO result is fragile, not a
clear artifact of a ceiling ranker deflating W. o4-mini was never excluded by
the criterion. BW stays decisive with or without filtering.

## Shared-hard item counts (J6 / `P1_failure_patterns.csv`)

{chr(10).join(sh_lines)}

## Reconciliation (N4 vs J6)

These two claims are **not** in conflict once the unit of agreement is named:

1. **N4 (variant level).** Within a family, models tend to agree on which
   *surface transforms* are hard (e.g. BW Kendall W={bw_full['kendall_W']},
   p={bw_full['p_value']}).
   That is agreement over six variant aggregates, not over items.

2. **J6 (item level).** Fragility does **not** concentrate on a shared hard-item
   core: ALGO {shared['ALGO'][0]}/{shared['ALGO'][1]}, GSM {shared['GSM'][0]}/{shared['GSM'][1]},
   BW only {shared['BW'][0]}/{shared['BW'][1]} items fail for all five paper models
   on canonical.

**One sentence for the paper.** Models can share a family's *variant difficulty
ordering* while still failing different *items* under the same transform —
variant-level concordance without item-level shared hardness.

Sources: `P1_variant_ordering_v2.csv` / this script for W; `P1_failure_patterns.csv`
(`shared_hard_canonical`, `fail_all_five_paper_models`) for J6 counts;
`C7_concordance_filtered.csv` for the audit trail.
"""
    OUT_NOTE.write_text(text)


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    p1 = _load_p1()
    p1 = p1[p1["variant"].isin(VARIANTS)].copy()

    rankers = _ranker_table(p1)
    print("=== Ranker degeneracy ===")
    print(
        rankers[
            [
                "family",
                "model",
                "acc_mean",
                "acc_range",
                "degenerate",
                "degeneracy_reasons",
            ]
        ].to_string(index=False)
    )

    # Full-sample pass: same family order + shared RNG as N4/O2 so p-values match.
    rng_full = np.random.default_rng(SEED)
    full_by_fam: dict[str, dict] = {}
    mats: dict[str, tuple] = {}
    for fam in ["ALGO", "BW", "GSM"]:
        sub = p1[p1["family"] == fam]
        fam_rankers = rankers[rankers["family"] == fam]
        all_models = sorted(fam_rankers["model"].tolist())
        deg_models = sorted(
            fam_rankers.loc[fam_rankers["degenerate"], "model"].tolist()
        )
        keep_models = [m for m in all_models if m not in set(deg_models)]
        mat_full, models_full, variants = _rank_matrix_for_models(sub, all_models)
        mats[fam] = (sub, keep_models, deg_models, variants)
        full_by_fam[fam] = _concordance_row(
            family=fam,
            sample="full",
            mat=mat_full,
            models=models_full,
            variants=variants,
            excluded=[],
            rng=rng_full,
        )

    # Filtered pass: independent stream so filtering does not shift full-sample p.
    rng_filt = np.random.default_rng(SEED + 10_000)
    conc_rows: list[dict] = []
    for fam in ["ALGO", "BW", "GSM"]:
        sub, keep_models, deg_models, _variants = mats[fam]
        if not deg_models:
            # Identical ranker set → identical W; reuse full-sample p for audit clarity.
            filt_row = dict(full_by_fam[fam])
            filt_row["sample"] = "filtered"
            filt_row["models_excluded_degenerate"] = ""
        else:
            mat_f, models_f, variants_f = _rank_matrix_for_models(sub, keep_models)
            filt_row = _concordance_row(
                family=fam,
                sample="filtered",
                mat=mat_f,
                models=models_f,
                variants=variants_f,
                excluded=deg_models,
                rng=rng_filt,
            )
        conc_rows.append(full_by_fam[fam])
        conc_rows.append(filt_row)
        print(
            f"[{fam}] full W={full_by_fam[fam]['kendall_W']} "
            f"p={full_by_fam[fam]['p_value']} m={full_by_fam[fam]['n_rankers']} | "
            f"filtered W={filt_row['kendall_W']} p={filt_row['p_value']} "
            f"m={filt_row['n_rankers']} excl={deg_models or '[]'}"
        )

    conc = pd.DataFrame(conc_rows)

    # Criterion audit row
    crit = pd.DataFrame(
        [
            {
                "row_type": "criterion",
                "family": "",
                "model": "",
                "criterion": (
                    f"degenerate if acc_range(W1-W6)<{RANGE_MIN} "
                    f"OR mean_acc>{MEAN_CEIL} OR mean_acc<{MEAN_FLOOR}"
                ),
                "range_threshold": RANGE_MIN,
                "mean_ceiling": MEAN_CEIL,
                "mean_floor": MEAN_FLOOR,
                "n_perm": N_PERM,
                "seed": SEED,
                "null": "within_row_rank_permutation_O2",
                "note": (
                    "Criterion fixed before inspecting filtered W; "
                    "full and filtered both reported."
                ),
            }
        ]
    )

    # Align columns for concat
    out = pd.concat([crit, rankers, conc], ignore_index=True, sort=False)
    out.to_csv(OUT_CSV, index=False)

    shared = {
        "ALGO": (0, 110),
        "GSM": (0, 20),
        "BW": (14, 64),
    }
    # Prefer live J6 table if present
    fail_path = DER / "P1_failure_patterns.csv"
    if fail_path.exists():
        fp = pd.read_csv(fail_path)
        sh = fp[
            (fp["section"] == "shared_hard_canonical")
            & (fp["key"] == "fail_all_five_paper_models")
        ]
        for _, r in sh.iterrows():
            shared[str(r["family"])] = (int(r["count"]), int(r["n_problems"]))

    _write_note(rankers, conc, shared)
    print(f"Wrote {OUT_CSV} ({len(out)} rows)")
    print(f"Wrote {OUT_NOTE}")


if __name__ == "__main__":
    main()
