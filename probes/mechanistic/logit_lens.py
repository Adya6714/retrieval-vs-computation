"""Logit lens utilities for per-layer token predictions."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def run_logit_lens(model: HookedTransformer, prompt: str) -> list[dict]:
    """Run logit lens at each layer and return top-1 token predictions."""
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    results: list[dict] = []
    for layer in range(model.cfg.n_layers):
        resid_post = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
        normed = model.ln_final(resid_post)
        logits = model.unembed(normed)
        probs = torch.softmax(logits, dim=-1)

        top_prob, top_idx = torch.max(probs, dim=-1)
        token_id = int(top_idx.item())
        top_token = model.tokenizer.decode([token_id])

        results.append(
            {
                "layer": layer,
                "top_token": top_token,
                "top_prob": float(top_prob.item()),
            }
        )

    return results
