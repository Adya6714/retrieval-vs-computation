"""Statistical testing and confidence interval utilities.

All Probe-1 metric CIs from bootstrap_ci use 10000 resamples by default.
Paper/appendix text often names Wilson 95% CIs; rebuild NUMBERS.csv uses
wilson_ci. cluster_bootstrap_ci resamples clone families, not problems.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy import stats


def bootstrap_ci(values: list[float], n_resamples: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    if not values:
        return (float('nan'), float('nan'))
    
    values_arr = np.array(values)
    n = len(values_arr)
    
    # Resample with replacement
    resamples = np.random.choice(values_arr, size=(n_resamples, n), replace=True)
    means = np.mean(resamples, axis=1)
    
    alpha = 1.0 - ci
    lower_perc = (alpha / 2.0) * 100
    upper_perc = (1.0 - alpha / 2.0) * 100
    
    lower = float(np.percentile(means, lower_perc))
    upper = float(np.percentile(means, upper_perc))
    
    return (lower, upper)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (float(lo), float(hi))


def cluster_bootstrap_ci(
    values: list[float],
    cluster_ids: list[str],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile CI for the problem-level mean, resampling clone families.

    Each unique cluster is drawn with replacement. All observations in a
    drawn cluster are included (clusters of size s keep weight s).
    """
    if not values or len(values) != len(cluster_ids):
        return (float("nan"), float("nan"))
    grouped: dict[str, list[float]] = defaultdict(list)
    for v, c in zip(values, cluster_ids):
        grouped[str(c)].append(float(v))
    fams = list(grouped.keys())
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci
    means = np.empty(n_resamples, dtype=float)
    n_f = len(fams)
    for i in range(n_resamples):
        draw = rng.integers(0, n_f, size=n_f)
        concat: list[float] = []
        for j in draw:
            concat.extend(grouped[fams[j]])
        means[i] = float(np.mean(concat))
    lo = float(np.percentile(means, (alpha / 2.0) * 100))
    hi = float(np.percentile(means, (1.0 - alpha / 2.0) * 100))
    return (lo, hi)
    

def wilcoxon_test(a: list[float], b: list[float]) -> dict:
    if len(a) != len(b):
        raise ValueError("Paired samples a and b must have the same length.")
        
    diffs = [abs(x - y) for x, y in zip(a, b)]
    if sum(diffs) == 0.0:
        result = {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False
        }
    else:
        res = stats.wilcoxon(a, b)
        result = {
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "significant": bool(res.pvalue < 0.05)
        }
    
    if len(a) < 10:
        result["warning"] = "sample size < 10, interpret with caution"
        
    return result


def effect_size_r(statistic: float, n: int) -> float:
    return float(statistic / math.sqrt(n))
