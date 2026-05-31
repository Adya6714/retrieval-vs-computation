"""
This module provides statistical testing and confidence interval utilities.
All confidence intervals use 10000 resamples by default per the statistical 
requirements in the research design.
"""

from __future__ import annotations

import math
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
