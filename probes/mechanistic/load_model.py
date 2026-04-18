"""Load Qwen2.5-7B with TransformerLens."""

from __future__ import annotations

import torch
try:
    from transformer_lens import HookedTransformer
except ImportError as e:
    raise ImportError(
        "transformer_lens is not installed. "
        "Uncomment it in requirements.txt and run: pip install -r requirements.txt. "
        "GPU access required for real runs."
    ) from e

MODEL_NAME = "Qwen/Qwen2.5-7B"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str | None = None,
    load_in_8bit: bool = False
) -> HookedTransformer:
    """Load and return Qwen2.5-7B via TransformerLens."""
    resolved_device = device or _default_device()

    model = HookedTransformer.from_pretrained(
        model_name,
        device=resolved_device,
        load_in_8bit=load_in_8bit,
    )

    print(f"Loaded model: {model_name}")
    print(f"Device: {resolved_device} | load_in_8bit: {load_in_8bit}")
    print(f"Num layers: {model.cfg.n_layers}")
    print(f"Hidden size: {model.cfg.d_model}")
    print(f"Num heads: {model.cfg.n_heads}")

    return model
