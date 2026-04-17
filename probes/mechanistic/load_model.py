"""Load Qwen2.5-7B with TransformerLens."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen2.5-7B"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device: str | None = None, load_in_8bit: bool = False) -> HookedTransformer:
    """Load and return Qwen2.5-7B via TransformerLens."""
    resolved_device = device or _default_device()

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=resolved_device,
        load_in_8bit=load_in_8bit,
    )

    print(f"Loaded model: {MODEL_NAME}")
    print(f"Device: {resolved_device} | load_in_8bit: {load_in_8bit}")
    print(f"Num layers: {model.cfg.n_layers}")
    print(f"Hidden size: {model.cfg.d_model}")
    print(f"Num heads: {model.cfg.n_heads}")

    return model
