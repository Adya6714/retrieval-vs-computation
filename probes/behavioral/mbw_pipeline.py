"""Mystery Blocksworld (MBW) pipeline.

MBW is standard Blocksworld with renamed actions and predicates:
  attack X    ↔ pick-up X     (preconds: harmony, province X, planet X)
  succumb X   ↔ put-down X    (preconds: pain X)
  overcome X Y ↔ stack X Y    (preconds: province Y, pain X)
  feast X Y   ↔ unstack X Y   (preconds: craves X Y, province X, harmony)

Predicates:
  harmony       ↔ hand-empty
  province X    ↔ clear X
  planet X      ↔ on-table X
  pain X        ↔ holding X
  craves X Y    ↔ X is on Y (the on relation)

State representation:
  {
    "harmony": bool,
    "province": set[str],
    "planet": set[str],
    "pain": str | None,          # the block currently being held; mirror of "holding"
    "craves": dict[str, str],    # craves[X] = Y means "block X is on block Y"
  }

This module is the MBW equivalent of probes/behavioral/bw_cci_pipeline.py.
"""

from __future__ import annotations

import copy
import re
from typing import Any


def parse_state_from_text_mbw(problem_text: str) -> tuple[list[str], dict, dict]:
    """Parse an MBW problem_text into (objects, init_state, goal).

    Expected format examples:
      "Current state: harmony is true. planet and province are true for blocks d, e, a, b, and f."
      "Goal: craves d e, craves e a, craves a b, and craves b f are true."
    """
    text = str(problem_text or "")
    cs_m = re.search(
        r"current state[:\s]+(.*?)\s*(?:goal[:\s]|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    gl_m = re.search(
        r"goal[:\s]+(.*?)(?:\s*respond|\s*each action|\s*no explanation|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not cs_m:
        raise ValueError("Could not parse Current state for MBW problem")
    if not gl_m:
        raise ValueError("Could not parse Goal for MBW problem")
    cs_text = cs_m.group(1).strip()
    gl_text = gl_m.group(1).strip()

    state: dict[str, Any] = {
        "harmony": False,
        "province": set(),
        "planet": set(),
        "pain": None,
        "craves": {},
    }
    objects: set[str] = set()

    if re.search(r"\bharmony\s+is\s+true\b", cs_text, re.IGNORECASE):
        state["harmony"] = True
    # "planet and province are true for blocks X, Y, Z"
    pp_m = re.search(
        r"planet\s+and\s+province\s+are\s+true\s+for\s+blocks?\s+([\w\s,\-]+?)\s*\.?$",
        cs_text,
        re.IGNORECASE | re.DOTALL,
    )
    # Allow trailing punctuation / continuation
    if not pp_m:
        pp_m = re.search(
            r"planet\s+and\s+province\s+are\s+true\s+for\s+blocks?\s+([\w\s,\-]+?)[\.\n]",
            cs_text + ".",
            re.IGNORECASE | re.DOTALL,
        )
    if pp_m:
        block_str = pp_m.group(1)
        names = [tok.strip().strip(",.").lower() for tok in re.split(r",|\band\b", block_str) if tok.strip()]
        for n in names:
            if not n or n == "and":
                continue
            state["province"].add(n)
            state["planet"].add(n)
            objects.add(n)
    # Standalone "province" / "planet" mentions (defensive)
    for m in re.finditer(r"province\s+is\s+true\s+for\s+block\s+(\w+)", cs_text, re.IGNORECASE):
        b = m.group(1).lower()
        state["province"].add(b)
        objects.add(b)
    for m in re.finditer(r"planet\s+is\s+true\s+for\s+block\s+(\w+)", cs_text, re.IGNORECASE):
        b = m.group(1).lower()
        state["planet"].add(b)
        objects.add(b)
    # "pain X is true"
    pain_m = re.search(r"pain\s+(?:is\s+true\s+for\s+block\s+)?(\w+)", cs_text, re.IGNORECASE)
    if pain_m and pain_m.group(1).lower() not in ("is",):
        state["pain"] = pain_m.group(1).lower()
        objects.add(state["pain"])
    # "craves X Y" in initial state (rare)
    for m in re.finditer(r"craves\s+(\w+)\s+(\w+)", cs_text, re.IGNORECASE):
        x, y = m.group(1).lower(), m.group(2).lower()
        state["craves"][x] = y
        objects.update({x, y})

    # Goal parsing: "craves X Y, craves Y Z, ..."
    goal: dict[str, str] = {}
    for m in re.finditer(r"craves\s+(\w+)\s+(\w+)", gl_text, re.IGNORECASE):
        x, y = m.group(1).lower(), m.group(2).lower()
        goal[x] = y
        objects.update({x, y})

    return sorted(objects), state, goal


def state_to_narrative_mbw(state: dict, objects: list[str]) -> str:
    """Render MBW state in the same style as the problem_text."""
    parts: list[str] = []
    parts.append("harmony is true" if state["harmony"] else "harmony is false")
    if state["province"]:
        parts.append(f"province is true for blocks {', '.join(sorted(state['province']))}")
    else:
        parts.append("province is true for no blocks")
    if state["planet"]:
        parts.append(f"planet is true for blocks {', '.join(sorted(state['planet']))}")
    else:
        parts.append("planet is true for no blocks")
    if state["pain"] is not None:
        parts.append(f"pain is true for block {state['pain']}")
    if state["craves"]:
        c_parts = [f"craves {x} {y}" for x, y in sorted(state["craves"].items())]
        parts.append("the following craves relations are true: " + ", ".join(c_parts))
    return ". ".join(parts) + "."


def goal_reached_mbw(state: dict, goal: dict) -> bool:
    for x, y in goal.items():
        if state["craves"].get(x) != y:
            return False
    return True


def execute_action_mbw(state: dict, action: str) -> dict:
    """Apply one MBW action string; return new state. Raises ValueError on illegal."""
    s = copy.deepcopy(state)
    parts = (action or "").strip().split()
    if not parts:
        raise ValueError("Empty action")
    verb = parts[0].lower()

    if verb == "attack" and len(parts) == 2:
        x = parts[1].lower()
        # preconds: harmony, province X, planet X
        if not s["harmony"]:
            raise ValueError(f"attack {x}: harmony false")
        if x not in s["province"]:
            raise ValueError(f"attack {x}: province {x} false")
        if x not in s["planet"]:
            raise ValueError(f"attack {x}: planet {x} false")
        # effects: pain X true; harmony, province X, planet X false
        s["pain"] = x
        s["harmony"] = False
        s["province"].discard(x)
        s["planet"].discard(x)
        return s

    if verb == "succumb" and len(parts) == 2:
        x = parts[1].lower()
        if s["pain"] != x:
            raise ValueError(f"succumb {x}: pain {x} false")
        s["harmony"] = True
        s["province"].add(x)
        s["planet"].add(x)
        s["pain"] = None
        return s

    if verb == "overcome" and len(parts) == 3:
        x, y = parts[1].lower(), parts[2].lower()
        if y not in s["province"]:
            raise ValueError(f"overcome {x} {y}: province {y} false")
        if s["pain"] != x:
            raise ValueError(f"overcome {x} {y}: pain {x} false")
        if x == y:
            raise ValueError(f"overcome {x} {y}: x and y identical")
        s["harmony"] = True
        s["province"].add(x)
        s["craves"][x] = y
        s["province"].discard(y)
        s["pain"] = None
        return s

    if verb == "feast" and len(parts) == 3:
        x, y = parts[1].lower(), parts[2].lower()
        if s["craves"].get(x) != y:
            raise ValueError(f"feast {x} {y}: craves {x} {y} false")
        if x not in s["province"]:
            raise ValueError(f"feast {x} {y}: province {x} false")
        if not s["harmony"]:
            raise ValueError(f"feast {x} {y}: harmony false")
        s["pain"] = x
        s["province"].add(y)
        s["craves"].pop(x, None)
        s["province"].discard(x)
        s["harmony"] = False
        return s

    raise ValueError(f"Unknown or malformed MBW action: {action!r}")


# -------- prompt templates (mirror BW shapes, just with MBW vocabulary) --------


def make_phase1_prompt_mbw(narrative: str, goal_narrative: str) -> str:
    return (
        "You are solving a problem with these actions: "
        "attack X (requires harmony, province X, planet X), "
        "succumb X (requires pain X), "
        "overcome X Y (requires province Y, pain X), "
        "feast X Y (requires craves X Y, province X, harmony).\n\n"
        f"Current state: {narrative}\n\n"
        f"Goal: {goal_narrative}\n\n"
        "Respond with a numbered list of actions, one per line. "
        "Each action must be exactly one of: attack X / succumb X / overcome X Y / feast X Y."
    )


def make_turn1_prompt_mbw(narrative: str, goal_narrative: str) -> str:
    return make_phase1_prompt_mbw(narrative, goal_narrative)


def make_followup_prompt_mbw(
    narrative: str,
    goal_narrative: str,
    last_action: str,
    error_note: str = "",
) -> str:
    extra = f"\nNote: {error_note}\n" if error_note else ""
    return (
        f"Current state after your last action ({last_action}): {narrative}\n\n"
        f"Goal remains: {goal_narrative}\n"
        f"{extra}\n"
        "What is the single next action? Reply with one line only. "
        "Use exactly one of: attack X / succumb X / overcome X Y / feast X Y."
    )
