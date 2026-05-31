"""Normalize ALGO Probe 2 phase1 prose vs phase2 structured decisions."""

from __future__ import annotations

import re


def normalize_subtype(subtype: str) -> str:
    s = str(subtype).strip().lower()
    if s in {"cc", "coin_change"}:
        return "coin_change"
    if s in {"sp", "shortest_path"}:
        return "shortest_path"
    if s == "wis":
        return "wis"
    return s


def normalize_phase2_decision(subtype: str, text: str) -> str:
    """Normalize structured phase2 `parsed_decision` values."""
    st = normalize_subtype(subtype)
    s = str(text).strip()
    if not s:
        return ""
    if st == "wis":
        m = re.search(r"\b(SELECT|RULE OUT)\s+(-?\d+)\b", s, flags=re.IGNORECASE)
        return f"{m.group(1).upper()} {int(m.group(2))}" if m else s.upper()
    m = re.search(r"-?\d+", s)
    return str(int(m.group(0))) if m else s


def normalize_phase1_decision(subtype: str, text: str) -> str:
    """Extract a comparable token from free-text phase1 `predicted_first_decision`."""
    st = normalize_subtype(subtype)
    s = str(text).strip()
    if not s:
        return ""

    if st == "wis":
        m = re.search(r"\b(SELECT|RULE OUT)\s+(-?\d+)\b", s, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1).upper()} {int(m.group(2))}"
        for pat in (
            r"(?:select|include|choose|start(?:ing)?\s+with)\s+(?:plot|district|server|interval|tower|item)?\s*#?\s*(-?\d+)",
            r"(?:interval|plot|district|server|tower)\s+(-?\d+)",
        ):
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                return f"SELECT {int(m.group(1))}"
        return s.upper()

    if st == "shortest_path":
        for pat in (
            r"(?:move|go(?:ing)?|travel|visit|head)\s+(?:to\s+)?(?:node|waypoint|vertex|camp|lodge|house)?\s*(-?\d+)",
            r"0\s+to\s+(-?\d+)",
            r"from\s+(?:node|waypoint)?\s*\d+.*?(?:to|visit)\s+(?:node|waypoint|camp)?\s*(-?\d+)",
            r"(?:edge|path)\s+(?:to|toward)\s+(?:node|waypoint)?\s*(-?\d+)",
        ):
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                return str(int(m.groups()[-1]))
        return normalize_phase2_decision(st, s)

    if st == "coin_change":
        for pat in (
            r"first coin choice.*?(?:be|is|will be)\s+(?:the\s+)?(?:denomination\s+)?(-?\d+)",
            r"(?:choose|select|pick|use)\s+(?:the\s+)?(?:coin\s+)?(?:denomination\s+)?(-?\d+)",
            r"(?:denomination|coin)\s+(-?\d+)",
            r"\bcoin\s+(-?\d+)\b",
        ):
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                return str(int(m.group(1)))
        return normalize_phase2_decision(st, s)

    return normalize_phase2_decision(st, s)


def normalize_decision(subtype: str, text: str, *, source: str = "phase2") -> str:
    """Back-compat wrapper used by metric scripts."""
    if source == "phase1":
        return normalize_phase1_decision(subtype, text)
    return normalize_phase2_decision(subtype, text)


def decisions_match(subtype: str, phase1_text: str, phase2_text: str) -> bool:
    p1 = normalize_phase1_decision(subtype, phase1_text)
    p2 = normalize_phase2_decision(subtype, phase2_text)
    return bool(p1) and p1 == p2
