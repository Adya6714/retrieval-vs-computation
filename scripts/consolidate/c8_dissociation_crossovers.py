#!/usr/bin/env python3
"""C8 / DS-14: Double-dissociation crossover battery (BX-01).

Searches for item×model crossovers on Probe-1 cells:
  M1 solves A robustly and fails B; M2 solves B robustly and fails A.

Rates are mean correctness over variants (n≥MIN_N). A candidate uses
thresholds TAU_HI / TAU_LO. Each leg is tested against a Bernoulli null
with common p = pooled accuracy on the two items (exact discrete Binomial
enumeration over all outcomes — equivalent to noise-range simulation at
n≤7); both legs must fall outside the one-sided 95% noise range.

Null expectation: simulate rate matrices from an additive LPM
(item + model; no interaction) with matched trial counts, apply the same
candidate + leg filters, 1000 replicates.

Also reports raw binary cell-level crossover counts per family×variant
vs additive-Bernoulli null (single latent dimension).

Outputs:
  results/derived/C8_dissociation_crossovers.csv
  results/derived/C8_null_expectation.csv
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.consolidate.p1_variant_ordering import _load_p1  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
OUT_XO = DER / "C8_dissociation_crossovers.csv"
OUT_NULL = DER / "C8_null_expectation.csv"

# Preregistered thresholds (auditable)
TAU_HI = 2.0 / 3.0  # "solves robustly"
TAU_LO = 1.0 / 3.0  # "fails"
MIN_N = 4  # min variants for a rate cell
LEG_Q = 0.95  # one-sided quantile: obs diff must exceed this under pooled-p null
N_LEG_SIM = 5000
N_NULL = 1000
SEED = 42
EPS = 1e-6

# Keep only family-native item IDs (BW rescored files also carry GSM_* rows).
FAMILY_ID_PREFIXES = {
    "ALGO": ("CC_", "SP_", "WIS_"),
    "BW": ("BW_", "MBW_"),
    "GSM": ("GSM_",),
}


def _family_items(df: pd.DataFrame, family: str) -> pd.DataFrame:
    prefixes = FAMILY_ID_PREFIXES[family]
    return df[df["problem_id"].astype(str).str.startswith(prefixes)].copy()


def _rates(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = (
        sub.groupby(["problem_id", "model"], as_index=False)["ok"]
        .agg(rate="mean", n="count")
    )
    g = g[g["n"] >= MIN_N]
    R = g.pivot(index="problem_id", columns="model", values="rate")
    N = g.pivot(index="problem_id", columns="model", values="n")
    return R, N


def _additive_from_rates(R: np.ndarray) -> np.ndarray:
    """LPM additive fit on rates; NaNs ignored in margins."""
    item = np.nanmean(R, axis=1)
    model = np.nanmean(R, axis=0)
    grand = np.nanmean(R)
    P = item[:, None] + model[None, :] - grand
    return np.clip(P, EPS, 1.0 - EPS)


def _leg_beyond_noise(
    s_hi: float,
    n_hi: int,
    s_lo: float,
    n_lo: int,
    rng: np.random.Generator | None = None,
    n_sim: int = N_LEG_SIM,
) -> tuple[bool, float, float, float]:
    """Test whether (high on A, low on B) exceeds Bernoulli noise under pooled p.

    Exact discrete null: all Bin(n_hi,p)×Bin(n_lo,p) outcomes (n≤7).
    Beyond-noise iff observed diff strictly exceeds the LEG_Q quantile of null diffs.
    Returns (beyond, p_one_sided, obs_diff, noise_q95).
    `rng`/`n_sim` kept for API compatibility; unused when exact path applies.
    """
    del rng, n_sim  # exact path
    n_hi = int(n_hi)
    n_lo = int(n_lo)
    if n_hi < 1 or n_lo < 1:
        return False, 1.0, float("nan"), float("nan")
    obs_diff = s_hi / n_hi - s_lo / n_lo
    p_pool = float(np.clip((s_hi + s_lo) / (n_hi + n_lo), EPS, 1.0 - EPS))

    # pmf of K~Bin(n_hi,p), M~Bin(n_lo,p)
    k = np.arange(n_hi + 1, dtype=float)
    m = np.arange(n_lo + 1, dtype=float)
    # log-pmf stable enough at these n
    from math import comb

    pk = np.array(
        [comb(n_hi, int(x)) * (p_pool**x) * ((1 - p_pool) ** (n_hi - x)) for x in k]
    )
    pm = np.array(
        [comb(n_lo, int(x)) * (p_pool**x) * ((1 - p_pool) ** (n_lo - x)) for x in m]
    )
    # joint grid
    diff = (k[:, None] / n_hi) - (m[None, :] / n_lo)
    pr = pk[:, None] * pm[None, :]
    flat_d = diff.ravel()
    flat_p = pr.ravel()
    # sort by diff for quantile
    order = np.argsort(flat_d, kind="mergesort")
    flat_d = flat_d[order]
    flat_p = flat_p[order]
    cdf = np.cumsum(flat_p)
    # smallest d such that cdf >= LEG_Q
    idx = int(np.searchsorted(cdf, LEG_Q, side="left"))
    idx = min(idx, len(flat_d) - 1)
    q = float(flat_d[idx])
    p_os = float(flat_p[flat_d >= obs_diff - 1e-15].sum())
    beyond = obs_diff > q + 1e-15
    return beyond, p_os, float(obs_diff), q


def _enumerate_rate_crossovers(
    R: pd.DataFrame,
    N: pd.DataFrame,
    family: str,
    rng: np.random.Generator,
    apply_leg_filter: bool,
) -> list[dict]:
    items = list(R.index)
    models = list(R.columns)
    Rm = R.to_numpy(dtype=float)
    Nm = N.to_numpy(dtype=float)
    item_ix = {it: i for i, it in enumerate(items)}
    model_ix = {m: j for j, m in enumerate(models)}
    rows: list[dict] = []

    for A, B in combinations(items, 2):
        ia, ib = item_ix[A], item_ix[B]
        for M1, M2 in combinations(models, 2):
            ja, jb = model_ix[M1], model_ix[M2]
            cells = [(ia, ja), (ib, ja), (ia, jb), (ib, jb)]
            if any(np.isnan(Rm[i, j]) or np.isnan(Nm[i, j]) for i, j in cells):
                continue
            r11, r21 = Rm[ia, ja], Rm[ib, ja]
            r12, r22 = Rm[ia, jb], Rm[ib, jb]
            n11, n21 = Nm[ia, ja], Nm[ib, ja]
            n12, n22 = Nm[ia, jb], Nm[ib, jb]

            # Orient 1: M1 solves A fails B; M2 solves B fails A
            o1 = r11 >= TAU_HI and r21 <= TAU_LO and r12 <= TAU_LO and r22 >= TAU_HI
            o2 = r12 >= TAU_HI and r22 <= TAU_LO and r11 <= TAU_LO and r21 >= TAU_HI
            if not (o1 or o2):
                continue
            if o1:
                m_hi_a, m_hi_b = M1, M2
                ra_hi, rb_lo, n_a_hi, n_b_lo = r11, r21, n11, n21
                ra_lo, rb_hi, n_a_lo, n_b_hi = r12, r22, n12, n22
                s_a_hi, s_b_lo = r11 * n11, r21 * n21
                s_a_lo, s_b_hi = r12 * n12, r22 * n22
                orient = "M1_solves_A_fails_B__M2_solves_B_fails_A"
            else:
                m_hi_a, m_hi_b = M2, M1
                ra_hi, rb_lo, n_a_hi, n_b_lo = r12, r22, n12, n22
                ra_lo, rb_hi, n_a_lo, n_b_hi = r11, r21, n11, n21
                s_a_hi, s_b_lo = r12 * n12, r22 * n22
                s_a_lo, s_b_hi = r11 * n11, r21 * n21
                orient = "M2_solves_A_fails_B__M1_solves_B_fails_A"

            if apply_leg_filter:
                ok1, p1, d1, q1 = _leg_beyond_noise(
                    s_a_hi, int(n_a_hi), s_b_lo, int(n_b_lo), rng
                )
                ok2, p2, d2, q2 = _leg_beyond_noise(
                    s_b_hi, int(n_b_hi), s_a_lo, int(n_a_lo), rng
                )
                reliable = bool(ok1 and ok2)
            else:
                ok1 = ok2 = True
                p1 = p2 = d1 = d2 = q1 = q2 = float("nan")
                reliable = True

            rows.append(
                {
                    "family": family,
                    "item_a": A,
                    "item_b": B,
                    "model_solves_a": m_hi_a,
                    "model_solves_b": m_hi_b,
                    "orientation": orient,
                    "rate_solves_a_on_a": round(float(ra_hi), 4),
                    "rate_solves_a_on_b": round(float(rb_lo), 4),
                    "rate_solves_b_on_a": round(float(ra_lo), 4),
                    "rate_solves_b_on_b": round(float(rb_hi), 4),
                    "n_solves_a_on_a": int(n_a_hi),
                    "n_solves_a_on_b": int(n_b_lo),
                    "n_solves_b_on_a": int(n_a_lo),
                    "n_solves_b_on_b": int(n_b_hi),
                    "leg_a_beyond_noise": ok1,
                    "leg_b_beyond_noise": ok2,
                    "leg_a_p_one_sided": round(p1, 4) if p1 == p1 else "",
                    "leg_b_p_one_sided": round(p2, 4) if p2 == p2 else "",
                    "leg_a_obs_diff": round(d1, 4) if d1 == d1 else "",
                    "leg_b_obs_diff": round(d2, 4) if d2 == d2 else "",
                    "leg_a_noise_q95": round(q1, 4) if q1 == q1 else "",
                    "leg_b_noise_q95": round(q2, 4) if q2 == q2 else "",
                    "reliable": reliable,
                    "tau_hi": round(TAU_HI, 4),
                    "tau_lo": round(TAU_LO, 4),
                    "min_n_variants": MIN_N,
                }
            )
    return rows


def _count_reliable_from_arrays(
    R: np.ndarray,
    N: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Count threshold candidates and reliable crossovers from dense arrays (NaN-aware)."""
    ni, nm = R.shape
    n_cand = 0
    n_rel = 0
    for i in range(ni):
        for j in range(i + 1, ni):
            for a in range(nm):
                for b in range(a + 1, nm):
                    vals = [R[i, a], R[j, a], R[i, b], R[j, b]]
                    ns = [N[i, a], N[j, a], N[i, b], N[j, b]]
                    if any(np.isnan(vals)) or any(np.isnan(ns)):
                        continue
                    r11, r21, r12, r22 = vals
                    n11, n21, n12, n22 = ns
                    o1 = r11 >= TAU_HI and r21 <= TAU_LO and r12 <= TAU_LO and r22 >= TAU_HI
                    o2 = r12 >= TAU_HI and r22 <= TAU_LO and r11 <= TAU_LO and r21 >= TAU_HI
                    if not (o1 or o2):
                        continue
                    n_cand += 1
                    if o1:
                        ok1, _, _, _ = _leg_beyond_noise(
                            r11 * n11, int(n11), r21 * n21, int(n21), rng
                        )
                        ok2, _, _, _ = _leg_beyond_noise(
                            r22 * n22, int(n22), r12 * n12, int(n12), rng
                        )
                    else:
                        ok1, _, _, _ = _leg_beyond_noise(
                            r12 * n12, int(n12), r22 * n22, int(n22), rng
                        )
                        ok2, _, _, _ = _leg_beyond_noise(
                            r21 * n21, int(n21), r11 * n11, int(n11), rng
                        )
                    if ok1 and ok2:
                        n_rel += 1
    return n_cand, n_rel


def _simulate_null_rate_counts(
    R: pd.DataFrame,
    N: pd.DataFrame,
    rng: np.random.Generator,
    n_null: int = N_NULL,
) -> tuple[np.ndarray, np.ndarray]:
    """Null: rates ~ Bin(n, P_additive)/n; count candidates & reliable."""
    Rm = R.to_numpy(dtype=float)
    Nm = N.to_numpy(dtype=float)
    P = _additive_from_rates(Rm)
    # Where missing in R, keep missing in sims
    miss = np.isnan(Rm) | np.isnan(Nm)
    cand_counts = np.empty(n_null, dtype=int)
    rel_counts = np.empty(n_null, dtype=int)
    for t in range(n_null):
        Rs = np.full_like(Rm, np.nan, dtype=float)
        for i in range(Rm.shape[0]):
            for j in range(Rm.shape[1]):
                if miss[i, j]:
                    continue
                n_ij = int(Nm[i, j])
                Rs[i, j] = rng.binomial(n_ij, float(P[i, j])) / n_ij
        c, r = _count_reliable_from_arrays(Rs, Nm, rng)
        cand_counts[t] = c
        rel_counts[t] = r
    return cand_counts, rel_counts


def _binary_crossover_count(Y: np.ndarray) -> int:
    """Count item-pair × model-pair binary crossovers on a complete 0/1 matrix."""
    ni, nm = Y.shape
    cnt = 0
    for a, b in combinations(range(nm), 2):
        only_a = (Y[:, a] == 1) & (Y[:, b] == 0)
        only_b = (Y[:, a] == 0) & (Y[:, b] == 1)
        cnt += int(only_a.sum() * only_b.sum())
    return cnt


def _binary_family_analysis(
    sub: pd.DataFrame, family: str, rng: np.random.Generator
) -> list[dict]:
    """Raw binary crossover counts vs additive-Bernoulli null, per variant + pooled."""
    rows = []
    pooled_obs = 0
    pooled_null = []
    for variant, sv in sorted(sub.groupby("variant"), key=lambda x: str(x[0])):
        wide = sv.groupby(["problem_id", "model"])["ok"].mean().unstack("model")
        wide = wide.dropna(axis=0, how="any")
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue
        Y = wide.to_numpy().astype(float)
        obs = _binary_crossover_count(Y.astype(int))
        pooled_obs += obs
        item = Y.mean(axis=1)
        model = Y.mean(axis=0)
        grand = Y.mean()
        P = np.clip(item[:, None] + model[None, :] - grand, EPS, 1 - EPS)
        null_counts = np.empty(N_NULL, dtype=int)
        for t in range(N_NULL):
            Ys = (rng.random(Y.shape) < P).astype(int)
            null_counts[t] = _binary_crossover_count(Ys)
        pooled_null.append(null_counts)
        rows.append(
            {
                "analysis": "binary_cell_crossovers",
                "family": family,
                "variant": variant,
                "n_items": int(Y.shape[0]),
                "n_models": int(Y.shape[1]),
                "observed_crossovers": obs,
                "null_mean": round(float(null_counts.mean()), 2),
                "null_sd": round(float(null_counts.std(ddof=1)), 2),
                "null_ci_low": round(float(np.quantile(null_counts, 0.025)), 2),
                "null_ci_high": round(float(np.quantile(null_counts, 0.975)), 2),
                "null_p_ge_obs": round(
                    float((1 + np.sum(null_counts >= obs)) / (1 + N_NULL)), 4
                ),
                "verdict": (
                    "at_or_below_chance"
                    if obs <= np.quantile(null_counts, 0.975)
                    else "above_chance"
                ),
                "n_null_sims": N_NULL,
                "seed": SEED,
                "note": "Additive LPM item+model; Y~Bern(P); no interaction.",
            }
        )
    if pooled_null:
        # sum of independent variant nulls is conservative for pooled; use mean of sums
        null_sum = np.sum(np.vstack(pooled_null), axis=0)
        rows.append(
            {
                "analysis": "binary_cell_crossovers",
                "family": family,
                "variant": "ALL_VARIANTS_SUM",
                "n_items": "",
                "n_models": "",
                "observed_crossovers": pooled_obs,
                "null_mean": round(float(null_sum.mean()), 2),
                "null_sd": round(float(null_sum.std(ddof=1)), 2),
                "null_ci_low": round(float(np.quantile(null_sum, 0.025)), 2),
                "null_ci_high": round(float(np.quantile(null_sum, 0.975)), 2),
                "null_p_ge_obs": round(
                    float((1 + np.sum(null_sum >= pooled_obs)) / (1 + N_NULL)), 4
                ),
                "verdict": (
                    "at_or_below_chance"
                    if pooled_obs <= np.quantile(null_sum, 0.975)
                    else "above_chance"
                ),
                "n_null_sims": N_NULL,
                "seed": SEED,
                "note": "Sum of per-variant crossover counts; null sums paired sims.",
            }
        )
    return rows


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    p1 = _load_p1()
    rng = np.random.default_rng(SEED)

    xo_rows: list[dict] = []
    null_rows: list[dict] = []

    print(
        f"Criteria: TAU_HI={TAU_HI:.4f} TAU_LO={TAU_LO:.4f} MIN_N={MIN_N} "
        f"leg_q={LEG_Q} N_NULL={N_NULL}"
    )

    for fam in ["ALGO", "BW", "GSM"]:
        sub = _family_items(p1[p1["family"] == fam], fam)
        R, N = _rates(sub)
        # Drop models/items that are entirely empty after MIN_N filter
        R = R.dropna(axis=0, how="all").dropna(axis=1, how="all")
        N = N.reindex(index=R.index, columns=R.columns)

        # Observed candidates (list all; mark reliable)
        found = _enumerate_rate_crossovers(R, N, fam, rng, apply_leg_filter=True)
        xo_rows.extend(found)
        n_cand = len(found)
        n_rel = int(sum(1 for r in found if r["reliable"]))
        print(
            f"[{fam}] rate candidates={n_cand} reliable={n_rel} "
            f"(items={R.shape[0]}, models={R.shape[1]})"
        )

        # Null expectation for rate-based reliable count
        null_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        cand_null, rel_null = _simulate_null_rate_counts(R, N, null_rng, N_NULL)
        for label, obs, arr in [
            ("rate_threshold_candidates", n_cand, cand_null),
            ("rate_reliable_crossovers", n_rel, rel_null),
        ]:
            p_ge = float((1 + np.sum(arr >= obs)) / (1 + N_NULL))
            at_chance = obs <= np.quantile(arr, 0.975)
            null_rows.append(
                {
                    "analysis": label,
                    "family": fam,
                    "variant": "rate_over_variants",
                    "n_items": int(R.shape[0]),
                    "n_models": int(R.shape[1]),
                    "observed_crossovers": obs,
                    "null_mean": round(float(arr.mean()), 2),
                    "null_sd": round(float(arr.std(ddof=1)), 2),
                    "null_ci_low": round(float(np.quantile(arr, 0.025)), 2),
                    "null_ci_high": round(float(np.quantile(arr, 0.975)), 2),
                    "null_p_ge_obs": round(p_ge, 4),
                    "verdict": (
                        "at_or_below_chance"
                        if at_chance
                        else "above_chance"
                    ),
                    "n_null_sims": N_NULL,
                    "seed": SEED,
                    "tau_hi": round(TAU_HI, 4),
                    "tau_lo": round(TAU_LO, 4),
                    "min_n_variants": MIN_N,
                    "leg_noise_quantile": LEG_Q,
                    "note": (
                        "Null = Bin(n, P)/n with P=additive LPM(item+model) on observed "
                        "rates; same TAU and exact Bernoulli pooled-p leg filter "
                        f"(diff > q{LEG_Q:.2f} under equal-p). "
                        "Family-native IDs only (ALGO: CC/SP/WIS; BW: BW/MBW; GSM: GSM). "
                        "at_or_below_chance ⇒ single latent dimension + sampling noise "
                        "suffices (consistent with one-factor EFA / O10 model-dominant VC). "
                        "GSM rate-reliable ABOVE chance is the exception — report as "
                        "candidate behavioral double-dissociation evidence on GSM only."
                    ),
                }
            )
            print(
                f"  {label}: obs={obs} null={arr.mean():.1f} "
                f"[{np.quantile(arr,0.025):.0f},{np.quantile(arr,0.975):.0f}] "
                f"p_ge={p_ge:.3f} → {'chance' if at_chance else 'ABOVE'}"
            )

        # Binary cell-level (enumeration diagnostic)
        bin_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        bin_rows = _binary_family_analysis(sub, fam, bin_rng)
        null_rows.extend(bin_rows)
        pooled = [r for r in bin_rows if r["variant"] == "ALL_VARIANTS_SUM"]
        if pooled:
            r = pooled[0]
            print(
                f"  binary sum: obs={r['observed_crossovers']} "
                f"null={r['null_mean']} → {r['verdict']}"
            )

    xo = pd.DataFrame(xo_rows)
    # Put reliable first for readability
    if len(xo):
        xo = xo.sort_values(
            ["reliable", "family", "item_a", "item_b"],
            ascending=[False, True, True, True],
        )
    xo.to_csv(OUT_XO, index=False)

    null = pd.DataFrame(null_rows)
    null.to_csv(OUT_NULL, index=False)
    print(f"Wrote {OUT_XO} ({len(xo)} rows)")
    print(f"Wrote {OUT_NULL} ({len(null)} rows)")

    # Headline
    print("\n=== Headline (rate reliable vs null) ===")
    sub = null[null["analysis"] == "rate_reliable_crossovers"]
    print(sub[["family", "observed_crossovers", "null_mean", "null_ci_low",
               "null_ci_high", "null_p_ge_obs", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
