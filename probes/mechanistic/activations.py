"""Activation extraction utilities for mechanistic probing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from transformer_lens import HookedTransformer

ACTIVATIONS_DIR = Path("results/activations")


def extract_activations(
    model: HookedTransformer,
    prompt: str,
    problem_id: str,
    variant_id: str,
) -> np.ndarray:
    """Extract per-layer resid_post activations and save as .npz."""
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    layer_acts: list[np.ndarray] = []
    for layer in range(model.cfg.n_layers):
        resid_post = cache[f"blocks.{layer}.hook_resid_post"]
        # remove batch dimension: [1, seq_len, d_model] -> [seq_len, d_model]
        layer_acts.append(resid_post[0].detach().cpu().numpy())

    activations = np.stack(layer_acts, axis=0)

    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ACTIVATIONS_DIR / f"{problem_id}_{variant_id}.npz"
    np.savez_compressed(output_path, activations=activations)

    return activations
