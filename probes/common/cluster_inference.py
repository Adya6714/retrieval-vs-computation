"""Cluster-bootstrap association inference (CI primary, p secondary).

Percentile CI and two-sided bootstrap p for H0: theta = 0 are computed from the
same cluster-resampled null, so they cannot contradict each other:
  p = 2 * min(mean(boots <= 0), mean(boots >= 0)), floored at 1/B.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats

AssocKind = Literal["spearman", "pointbiserial"]


def _assoc(x: np.ndarray, y: np.ndarray, kind: AssocKind) -> float:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    if kind == "spearman":
        r, _ = stats.spearmanr(x, y)
    else:
        if len(x) < 3:
            return float("nan")
        r, _ = stats.pointbiserialr(y.astype(float), x.astype(float))
    return float(r)


def bootstrap_p_two_sided(boots: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p for H0: theta=0 (coherent with equal-tailed CI)."""
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return float("nan")
    left = float(np.mean(boots <= 0.0))
    right = float(np.mean(boots >= 0.0))
    p = 2.0 * min(left, right)
    return float(max(p, 1.0 / len(boots)))


def cluster_bootstrap_assoc(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    cluster_ids: list[str] | pd.Series,
    *,
    kind: AssocKind = "spearman",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """Return estimate, percentile CI, and cluster-bootstrap p for association."""
    xv = np.asarray(pd.Series(x).astype(float), dtype=float)
    yv = np.asarray(pd.Series(y).astype(float), dtype=float)
    cids = [str(c) for c in list(cluster_ids)]
    if len(xv) != len(yv) or len(xv) != len(cids):
        raise ValueError("x, y, cluster_ids must have equal length")

    estimate = _assoc(xv, yv, kind)
    clusters = sorted(set(cids))
    grouped = {c: [i for i, cid in enumerate(cids) if cid == c] for c in clusters}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        idx = [j for c in draw for j in grouped[c]]
        boots[i] = _assoc(xv[idx], yv[idx], kind)
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_clustered": float("nan"),
            "n": int(len(xv)),
            "n_clusters": len(clusters),
        }
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_clustered": bootstrap_p_two_sided(finite),
        "n": int(len(xv)),
        "n_clusters": len(clusters),
    }


def iid_bootstrap_assoc(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    kind: AssocKind = "spearman",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """IID bootstrap when the sampling unit is already the independent unit (e.g. models)."""
    xv = np.asarray(pd.Series(x).astype(float), dtype=float)
    yv = np.asarray(pd.Series(y).astype(float), dtype=float)
    n = len(xv)
    estimate = _assoc(xv, yv, kind)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[i] = _assoc(xv[idx], yv[idx], kind)
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_clustered": float("nan"),
            "n": n,
            "n_clusters": n,
        }
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_clustered": bootstrap_p_two_sided(finite),
        "n": n,
        "n_clusters": n,
    }


def sig_at(p: float | None, alpha: float = 0.05) -> bool | None:
    if p is None or p != p:
        return None
    return bool(p < alpha)


def ci_excludes_zero(lo: float, hi: float) -> bool | None:
    if lo != lo or hi != hi:
        return None
    return bool(lo > 0 or hi < 0)
