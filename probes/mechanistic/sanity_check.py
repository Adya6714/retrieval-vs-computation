"""Sanity checks for mechanistic probing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probes.mechanistic.activations import extract_activations
from probes.mechanistic.load_model import load_model
from probes.mechanistic.logit_lens import run_logit_lens
from probes.mechanistic.similarity import layer_cosine_similarity

ACTS_DIR = Path("results/activations")


def _test_activation_consistency(model) -> None:
    prompt = "The quick brown fox"
    extract_activations(model, prompt, problem_id="test", variant_id="a")
    extract_activations(model, prompt, problem_id="test", variant_id="b")

    acts_a_path = ACTS_DIR / "test_a.npz"
    acts_b_path = ACTS_DIR / "test_b.npz"

    acts_a = np.load(acts_a_path)["activations"]
    acts_b = np.load(acts_b_path)["activations"]

    sims = layer_cosine_similarity(acts_a, acts_b)
    passed = bool(np.all(sims > 0.999))

    if passed:
        print("Test 1 (activation consistency): PASS")
    else:
        print("Test 1 (activation consistency): FAIL")
        print(f"Minimum cosine similarity: {float(np.min(sims)):.6f}")


def _test_logit_lens(model) -> None:
    prompt = "The capital of France is"
    rows = run_logit_lens(model, prompt)

    first_paris_layer: int | None = None
    for row in rows:
        token_text = str(row["top_token"]).strip()
        if first_paris_layer is None and token_text.lower() == "paris":
            first_paris_layer = int(row["layer"])

    print("Test 2 (logit lens inspection):")
    for row in rows:
        layer = int(row["layer"])
        token_text = str(row["top_token"]).strip().replace("\n", "\\n")
        top_prob = float(row["top_prob"])
        marker = " <== first 'Paris'" if first_paris_layer == layer else ""
        print(f"layer={layer:02d} top_token='{token_text}' top_prob={top_prob:.6f}{marker}")

    if first_paris_layer is None:
        print("Paris did not appear as top-1 in any layer.")
    else:
        print(f"Paris first appears as top-1 at layer {first_paris_layer}.")


def main() -> None:
    model = load_model()
    _test_activation_consistency(model)
    _test_logit_lens(model)


if __name__ == "__main__":
    main()
