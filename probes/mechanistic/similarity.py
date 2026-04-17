"""Layer-wise activation similarity utilities."""

from __future__ import annotations

import numpy as np


def layer_cosine_similarity(acts_a: np.ndarray, acts_b: np.ndarray) -> np.ndarray:
    """Compute layer-wise cosine similarity at the last token position."""
    if acts_a.ndim != 3 or acts_b.ndim != 3:
        raise ValueError("Both inputs must have shape [num_layers, seq_len, d_model].")
    if acts_a.shape != acts_b.shape:
        raise ValueError("Input activation arrays must have the same shape.")
    if acts_a.shape[1] < 1:
        raise ValueError("seq_len must be at least 1.")

    vecs_a = acts_a[:, -1, :]
    vecs_b = acts_b[:, -1, :]

    dot = np.sum(vecs_a * vecs_b, axis=1)
    norm_a = np.linalg.norm(vecs_a, axis=1)
    norm_b = np.linalg.norm(vecs_b, axis=1)
    denom = norm_a * norm_b

    similarities = np.zeros(acts_a.shape[0], dtype=np.float64)
    valid = denom > 0
    similarities[valid] = dot[valid] / denom[valid]
    return similarities
