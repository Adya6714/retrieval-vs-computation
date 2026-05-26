"""NL-tolerant action remapping for Mystery Blocksworld.

MBW action verbs (attack, succumb, overcome, feast) have NO natural-English
synonyms. The mechanistic prediction from §11 (canonical rank 9087 for MBW vs
313 for standard BW) implies models will fail to commit to these verbs
regardless of parser tolerance.

This parser therefore focuses on cleaning up preamble and obvious shape-fixes,
not on accepting synonyms. The interesting paper claim is that even with this
tolerant parser, MBW abort/loop rates stay high - confirming the mechanistic
prediction behaviourally.
"""

from __future__ import annotations

import re

CANONICAL_VERBS_MBW = {"attack", "succumb", "overcome", "feast"}


EXTRA_PREAMBLE_PREFIXES_MBW = (
    "i need",
    "i should",
    "i can",
    "i'm going",
    "i am going",
    "i have to",
    "i must",
    "i'll",
    "i will",
    "i would",
    "i could",
    "to put",
    "to do",
    "to achieve",
    "to make",
    "to get",
    "this means",
    "this requires",
    "the plan",
    "the sequence",
    "looking at",
    "based on",
    "first ",
    "next ",
    "then ",
    "after",
    "before",
    "let me",
    "let's",
    "now,",
    "now ",
    "okay",
    "sure",
    "great",
    "certainly",
    "given",
)


def is_preamble_mbw(line: str) -> bool:
    s = (line or "").strip().lower()
    if not s:
        return True
    if any(s.startswith(p) for p in EXTRA_PREAMBLE_PREFIXES_MBW):
        return True
    # If the line contains no MBW verb at all, treat as preamble.
    if not any(v in s for v in CANONICAL_VERBS_MBW):
        return True
    return False


def remap_to_canonical_mbw(action: str) -> str | None:
    """Map an MBW action string to the canonical form.

    Returns one of:
      "attack X" / "succumb X" / "overcome X Y" / "feast X Y"
    or None if unmappable.
    """
    s = (action or "").strip().lower()
    if not s:
        return None
    s = s.rstrip(".;,!?")
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r",", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    # Strip "block " mid-string
    s = re.sub(r"\bblock\s+", "", s, flags=re.IGNORECASE)
    # Strip wrapping like "perform attack on X" -> "attack X"
    s = re.sub(r"^perform\s+", "", s)
    s = re.sub(r"^execute\s+", "", s)
    s = re.sub(r"^action[:\s]+", "", s)
    # "attack on X" -> "attack X"  (no args between verb and "on")
    for v in CANONICAL_VERBS_MBW:
        s = re.sub(rf"^{v}\s+on\s+", f"{v} ", s)
    # "overcome X on Y" / "feast X on Y" -> "overcome X Y" (strip connective between args)
    s = re.sub(r"^(overcome|feast)\s+(\w+)\s+(?:on|onto|to)\s+(\w+)", r"\1 \2 \3", s)

    # Direct canonical
    m = re.match(r"^(attack|succumb)\s+(\w+)\s*$", s)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = re.match(r"^(overcome|feast)\s+(\w+)\s+(\w+)\s*$", s)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    # 2-arg attack/succumb (model confused) -> probably overcome
    m = re.match(r"^attack\s+(\w+)\s+(\w+)\s*$", s)
    if m:
        # ambiguous; cannot recover safely
        return None
    return None
