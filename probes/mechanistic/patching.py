"""
Activation patching for Probe 1 mechanistic analysis.
Every patching experiment must include a random-position control.
patch_minus_control is the key diagnostic: positive values indicate the
patched layer carries variant-specific information.
"""

from __future__ import annotations

import random
from typing import Any


def patch_activations(
    model: Any,
    source_tokens: Any,
    target_tokens: Any,
    layer: int,
    position: int,
) -> dict:
    hook_name = f"blocks.{layer}.hook_resid_post"

    # a. Run source to get cache
    source_logits, source_cache = model.run_with_cache(source_tokens, names_filter=hook_name)
    source_act = source_cache[hook_name]

    # Run target without patch to obtain original predicted token
    target_logits = model(target_tokens)

    # b. Target run with patching hook
    def patching_hook(resid: Any, hook: Any) -> Any:
        resid[:, position, :] = source_act[:, position, :]
        return resid

    patched_logits = model.run_with_hooks(
        target_tokens,
        fwd_hooks=[(hook_name, patching_hook)]
    )

    source_top_token = int(source_logits[0, position].argmax().item())
    target_top_token = int(target_logits[0, position].argmax().item())
    patched_top_token = int(patched_logits[0, position].argmax().item())

    patch_success = bool(patched_top_token == source_top_token)

    return {
        "source_top_token": source_top_token,
        "target_top_token": target_top_token,
        "patched_top_token": patched_top_token,
        "patch_success": patch_success,
    }


def random_position_control(
    model: Any,
    source_tokens: Any,
    target_tokens: Any,
    layer: int,
    true_position: int,
) -> dict:
    # Handle potentially batched tensors or nested lists flexibly
    seq_len = source_tokens.shape[-1] if hasattr(source_tokens, "shape") else len(source_tokens[0])

    if seq_len <= 1:
        raise ValueError("source_tokens must have length > 1 for random_position_control")

    # Select a random position ignoring the true position
    available_positions = [p for p in range(seq_len) if p != true_position]
    control_position = random.choice(available_positions)

    hook_name = f"blocks.{layer}.hook_resid_post"

    source_logits, source_cache = model.run_with_cache(source_tokens, names_filter=hook_name)
    source_act = source_cache[hook_name]

    target_logits = model(target_tokens)

    def control_hook(resid: Any, hook: Any) -> Any:
        # Patch from random position in source run to true_position in target run
        resid[:, true_position, :] = source_act[:, control_position, :]
        return resid

    patched_logits = model.run_with_hooks(
        target_tokens,
        fwd_hooks=[(hook_name, control_hook)]
    )

    source_top_token = int(source_logits[0, true_position].argmax().item())
    target_top_token = int(target_logits[0, true_position].argmax().item())
    patched_top_token = int(patched_logits[0, true_position].argmax().item())

    patch_success = bool(patched_top_token == source_top_token)

    return {
        "source_top_token": source_top_token,
        "target_top_token": target_top_token,
        "patched_top_token": patched_top_token,
        "patch_success": patch_success,
        "control_position": control_position,
    }


def run_patching_experiment(
    model: Any,
    canonical_tokens: Any,
    variant_tokens: Any,
    layers: list[int],
    position: int,
) -> list[dict]:
    results = []
    
    for layer in layers:
        patch_result = patch_activations(
            model=model,
            source_tokens=canonical_tokens,
            target_tokens=variant_tokens,
            layer=layer,
            position=position,
        )
        
        control_result = random_position_control(
            model=model,
            source_tokens=canonical_tokens,
            target_tokens=variant_tokens,
            layer=layer,
            true_position=position,
        )
        
        patch_minus_control = float(patch_result["patch_success"]) - float(control_result["patch_success"])

        results.append({
            "layer": layer,
            "patch_result": patch_result,
            "control_result": control_result,
            "patch_minus_control": patch_minus_control,
        })
        
    return results
