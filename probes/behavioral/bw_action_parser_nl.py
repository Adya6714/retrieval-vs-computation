"""NL-tolerant action remapping for Blocksworld Phase-2 reparse.

Maps semantically-clear natural-language variants to the 4 canonical PDDL verbs
(pick-up X / put-down X / stack X Y / unstack X Y).

Empirical motivation: in `results/raw/BW_P2_cci.csv`, the dominant abort-causing
action shapes were `put-down X Y` (n=119), `put X on Y` (n=151 with "put" verb),
`put-down X on top of Y` (n=121), `pick-up X from Y` (n=128). All are
semantically clear; the model is using natural-English compositions.

This module is parser-only. It does not change the executor (`execute_action`)
or the state machine. It rewrites the action string before it reaches the executor.

Tests are in `scripts/audit/test_bw_action_remap.py`.
"""

from __future__ import annotations

import re

CANONICAL_VERBS = {"pick-up", "put-down", "stack", "unstack"}

# Extra preamble starters that aren't in the original PREAMBLE_PREFIXES.
# These cause the parser to treat preamble as the action.
EXTRA_PREAMBLE_PREFIXES = (
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
    "to stack",
    "to pick",
    "to make",
    "to get",
    "to achieve",
    "to accomplish",
    "to reach",
    "to do",
    "to build",
    "to construct",
    "to move",
    "this means",
    "this requires",
    "this is",
    "that means",
    "looking at",
    "based on",
    "considering",
    "examining",
    "analyzing",
    "the plan",
    "the sequence",
    "the actions",
    "the answer",
    "the correct",
    "first i",
    "next i",
    "then i",
    "after",
    "before",
    "given",
)


def is_preamble(line: str) -> bool:
    """Return True if a line looks like model preamble/reasoning, not an action."""
    s = line.strip().lower()
    if not s:
        return True
    if any(s.startswith(p) for p in EXTRA_PREAMBLE_PREFIXES):
        return True
    # No canonical verb anywhere in the line => probably preamble
    if not any(v in s for v in ("pick", "put", "stack", "unstack", "place", "move", "set ", "lift", "take", "remove", "grab")):
        # Heuristic: a sentence that doesn't even mention a block-manipulation verb
        return True
    return False


def _strip_block_prefix(text: str) -> str:
    return re.sub(r"\bblock\s+", "", text, flags=re.IGNORECASE)


def _strip_connectives(text: str) -> str:
    """Remove English connectives that come between block names."""
    s = text
    s = re.sub(r"\s+on\s+top\s+of\s+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+onto\s+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+on\s+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+from\s+(?:on\s+top\s+of\s+|on\s+|top\s+of\s+)?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+off\s+(?:of\s+)?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+to\s+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+the\s+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_action(s: str) -> str:
    """NL remap first; fall back to stripped lowercase if unmappable.

    Strict runners call this before ``execute_action``. Precondition checking
    stays strict; only the string that reaches the executor is rewritten.
    """
    remapped = remap_to_canonical(s)
    if remapped:
        return remapped
    return (s or "").strip().lower().rstrip(".")


def remap_to_canonical(action: str) -> str | None:
    """Try to rewrite `action` to one of the 4 canonical PDDL forms.

    Returns the canonical action string, or None if unmappable.
    The returned string is one of:
      "pick-up X"   (1 single-token block)
      "put-down X"  (1 single-token block, only if intent is "place on table")
      "stack X Y"
      "unstack X Y"
    """
    s = (action or "").strip().lower()
    if not s:
        return None
    # Remove trailing punctuation
    s = s.rstrip(".;,!?")
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r",", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    # Remove "block " mid-string
    s = _strip_block_prefix(s)
    # "put it on table" etc.
    is_table = bool(re.search(r"\btable\b", s))
    # Normalise hyphenation variants
    s = re.sub(r"\bpick\s*[-_]?\s*up\b", "pick-up", s)
    s = re.sub(r"\bput\s*[-_]?\s*down\b", "put-down", s)
    s = re.sub(r"\bset\s*[-_]?\s*down\b", "put-down", s)
    # "set X down" pattern -> "put-down X"
    m_set_x_down = re.match(r"^set\s+(\w+)\s+down(\b.*)?$", s)
    if m_set_x_down:
        rest = (m_set_x_down.group(2) or "").strip()
        if not rest or re.match(r"^(?:on\s+|onto\s+)?(?:the\s+)?table\b", rest):
            return f"put-down {m_set_x_down.group(1)}"
        # "set X down on Y" => stack X Y
        m_xy = re.match(r"^(?:on\s+(?:top\s+of\s+)?|onto\s+)(\w+)$", rest)
        if m_xy and m_xy.group(1) != "table":
            return f"stack {m_set_x_down.group(1)} {m_xy.group(1)}"
    # "set X down on the table" pattern explicit handler
    m_set_table = re.match(
        r"^put-down\s+(\w+)\s+(?:on\s+|onto\s+)?(?:the\s+)?table\b", s
    )
    if m_set_table:
        return f"put-down {m_set_table.group(1)}"
    # Normalise "off of" -> "off" before further parsing
    s = re.sub(r"\boff\s+of\b", "off", s)

    # --- table-destination first ---
    # "put-down X on the table" / "put X on table" / "set X on table" => put-down X
    if is_table:
        # extract a single token that comes right after pick/put/set/move/place
        m = re.match(
            r"^(?:put-down|put|place|move|set|drop)\s+(\w+)\s+(?:on|onto|to|down|on\s+top\s+of)?\s*(?:the\s+)?table\b",
            s,
        )
        if m:
            return f"put-down {m.group(1)}"
    # --- unstack semantics: pick/lift/take/remove X from Y ---
    m = re.match(
        r"^(?:pick-up|pick|lift|take|remove|grab|unstack)\s+(\w+)\s+(?:up\s+)?(?:from|off|off\s+of)\s+(?:the\s+top\s+of\s+)?(\w+)",
        s,
    )
    if m:
        return f"unstack {m.group(1)} {m.group(2)}"

    # --- stack semantics: put/place/set/stack/move X on(to)(top of) Y ---
    m = re.match(
        r"^(?:put-down|put|place|set|move|stack)\s+(\w+)\s+(?:on\s+top\s+of|onto|on|to)\s+(\w+)",
        s,
    )
    if m and m.group(2) != "table":
        return f"stack {m.group(1)} {m.group(2)}"

    # --- pick-up X (single block, no source) ---
    m = re.match(r"^(?:pick-up|pick|lift|grab)\s+(?:up\s+)?(\w+)\s*$", s)
    if m:
        return f"pick-up {m.group(1)}"

    # --- put-down X (single block, table implied) ---
    m = re.match(r"^(?:put-down|put|place|set|drop)\s+(\w+)\s*$", s)
    if m:
        return f"put-down {m.group(1)}"

    # --- 2-arg verb-confusion: put-down X Y / put X Y / move X Y ---
    # If the model wrote `put-down X Y` (2 token args), intent is almost certainly stack.
    m = re.match(r"^(?:put-down|put|move|place|set)\s+(\w+)\s+(\w+)\s*$", s)
    if m and m.group(2) != "table":
        return f"stack {m.group(1)} {m.group(2)}"

    # --- stack X Y already ---
    m = re.match(r"^stack\s+(\w+)\s+(\w+)\s*$", s)
    if m:
        return f"stack {m.group(1)} {m.group(2)}"

    # --- unstack X Y already ---
    m = re.match(r"^unstack\s+(\w+)\s+(\w+)\s*$", s)
    if m:
        return f"unstack {m.group(1)} {m.group(2)}"

    # --- unstack X (no source given - try to recover) ---
    m = re.match(r"^unstack\s+(\w+)\s*$", s)
    if m:
        # Cannot fully canonicalise without state; mark as ambiguous
        return None

    return None


def classify_action(action: str) -> str:
    """Return a coarse-grained classification of the action string.
    Used for reporting which categories were recovered by the NL remap.
    """
    s = (action or "").strip().lower()
    if not s:
        return "empty"
    if is_preamble(s):
        return "preamble"
    canon = remap_to_canonical(s)
    if canon is None:
        return "unmappable"
    # Determine what the original verb was
    parts = s.split()
    orig_verb = parts[0]
    canon_verb = canon.split()[0]
    if orig_verb == canon_verb and len(parts) - 1 == len(canon.split()) - 1:
        return f"canonical:{canon_verb}"
    return f"remapped:{orig_verb}->{canon_verb}"
