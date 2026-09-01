"""Canonical variant_type labels used on every bank join."""

from __future__ import annotations


def normalize_variant(value: object) -> str:
    """Uppercase variant labels (w6 → W6). Keep canonical as canonical."""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() == "canonical":
        return "canonical"
    return text.upper()
