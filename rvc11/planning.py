"""Blocksworld / Mystery Blocksworld: formal state, simulator, plan parsing, reference planner.

All planning items are stored in canonical Blocksworld facts. Mystery Blocksworld is the same
domain under a fixed renaming, so one simulator serves both:

    attack=pick-up  succumb=put-down  overcome=stack  feast=unstack
    harmony=handempty  planet=ontable  province=clear  pain=holding  craves=on
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CANON_VERBS = ("pick-up", "put-down", "stack", "unstack")
ARITY = {"pick-up": 1, "put-down": 1, "stack": 2, "unstack": 2}
MYSTERY_VERBS = {"pick-up": "attack", "put-down": "succumb", "stack": "overcome", "unstack": "feast"}
MYSTERY_PRED = {"harmony": "handempty", "planet": "ontable", "province": "clear", "pain": "holding", "craves": "on"}

Fact = tuple


def mystery_to_bw(facts: set[Fact]) -> set[Fact]:
    return {(MYSTERY_PRED.get(f[0], f[0]),) + tuple(f[1:]) for f in facts}


def close_state(on_pairs: set[tuple[str, str]], blocks: set[str]) -> set[Fact]:
    """Physical closure of a hand-empty state from its ``on`` relation and block set."""
    blocks = set(blocks) | {a for a, _ in on_pairs} | {b for _, b in on_pairs}
    below = dict(on_pairs)
    covered = set(below.values())
    state: set[Fact] = {("handempty",)}
    for b in blocks:
        state.add(("on", b, below[b]) if b in below else ("ontable", b))
        if b not in covered:
            state.add(("clear", b))
    return state


def state_is_physical(state: set[Fact]) -> tuple[bool, str]:
    """True when the state is a legal hand-empty Blocksworld configuration (towers, no cycles)."""
    on = [(f[1], f[2]) for f in state if f[0] == "on"]
    blocks = {f[1] for f in state if f[0] in ("ontable", "clear")} | {x for p in on for x in p}
    if not blocks:
        return False, "no blocks"
    if ("handempty",) not in state:
        return False, "hand not empty"
    if len({a for a, _ in on}) != len(on):
        return False, "block on two things"
    if len({b for _, b in on}) != len(on):
        return False, "two blocks on one block"
    if close_state(set(on), blocks) != state:
        return False, "explicit facts differ from physical closure"
    below = dict(on)
    for b in blocks:
        seen, cur = set(), b
        while cur in below:
            if cur in seen:
                return False, "cycle"
            seen.add(cur)
            cur = below[cur]
    return True, "ok"


def apply_action(state: set[Fact], verb: str, args: tuple[str, ...]) -> str | None:
    """Apply one canonical action in place. Returns None on success, else the failure reason."""
    if verb not in ARITY or len(args) != ARITY[verb]:
        return f"bad arity for {verb}"
    if verb == "pick-up":
        (x,) = args
        pre = {("clear", x), ("ontable", x), ("handempty",)}
        if not pre <= state:
            return f"precondition failed: pick-up {x}"
        state -= pre
        state.add(("holding", x))
    elif verb == "put-down":
        (x,) = args
        if ("holding", x) not in state:
            return f"precondition failed: put-down {x}"
        state.discard(("holding", x))
        state |= {("ontable", x), ("clear", x), ("handempty",)}
    elif verb == "stack":
        x, y = args
        if x == y or not {("holding", x), ("clear", y)} <= state:
            return f"precondition failed: stack {x} {y}"
        state -= {("holding", x), ("clear", y)}
        state |= {("on", x, y), ("clear", x), ("handempty",)}
    else:
        x, y = args
        pre = {("on", x, y), ("clear", x), ("handempty",)}
        if not pre <= state:
            return f"precondition failed: unstack {x} {y}"
        state -= pre
        state |= {("holding", x), ("clear", y)}
    return None


def simulate(init: set[Fact], goal: set[Fact], plan: list[tuple[str, tuple[str, ...]]]) -> tuple[bool, str]:
    state = set(init)
    for i, (verb, args) in enumerate(plan, 1):
        err = apply_action(state, verb, args)
        if err:
            return False, f"step {i}: {err}"
    missing = goal - state
    if missing:
        return False, f"goal not reached: {len(missing)} goal fact(s) false"
    return True, "ok"


def reference_plan(init: set[Fact], goal: set[Fact]) -> list[tuple[str, tuple[str, ...]]]:
    """A valid (not optimal) plan: put every block on the table, then build the goal towers.

    Used as a second, independent witness that a spec is solvable and to replace legacy gold
    plans that do not simulate. Raises ValueError when the goal is not a consistent tower set.
    """
    state = set(init)
    plan: list[tuple[str, tuple[str, ...]]] = []
    goal_on = {f[1]: f[2] for f in goal if f[0] == "on"}
    goal_table = {f[1] for f in goal if f[0] == "ontable"}
    goal_clear = {f[1] for f in goal if f[0] == "clear"}
    if len(set(goal_on.values())) != len(goal_on):
        raise ValueError("goal puts two blocks on one block")

    def below(b):
        return next((f[2] for f in state if f[0] == "on" and f[1] == b), None)

    def in_place(b, seen=()):
        if b in seen:
            raise ValueError("goal cycle")
        if b in goal_on:
            return below(b) == goal_on[b] and in_place(goal_on[b], seen + (b,))
        if b in goal_table:
            return below(b) is None
        return True  # unconstrained support: wherever it is

    def act(verb, *args):
        err = apply_action(state, verb, args)
        if err:
            raise ValueError(f"reference planner bug: {err}")
        plan.append((verb, args))

    blocks = {f[1] for f in state if f[0] in ("ontable", "clear", "on")} | {f[2] for f in state if f[0] == "on"}
    # Phase 1: dismantle every tower, top-down. Always legal from a hand-empty state.
    while True:
        top = next((b for b in sorted(blocks) if ("clear", b) in state and below(b) is not None), None)
        if top is None:
            break
        act("unstack", top, below(top))
        act("put-down", top)
    # Phase 2: build goal towers bottom-up.
    progress = True
    while progress:
        progress = False
        for b in sorted(goal_on):
            tgt = goal_on[b]
            if below(b) == tgt:
                continue
            if in_place(tgt) and ("clear", tgt) in state and ("clear", b) in state and below(b) is None:
                act("pick-up", b)
                act("stack", b, tgt)
                progress = True
    ok, why = simulate(init, goal, plan)
    if not ok:
        raise ValueError(f"reference planner could not solve: {why}")
    return plan


# ----------------------------------------------------------------------------------------------
# Surface vocabulary: how a variant names the four actions and the objects.
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Vocab:
    """templates: canonical verb -> surface template using X / Y, e.g. ``place X under Y``."""

    templates: dict
    entities: dict  # canonical object -> surface name

    def render(self, verb: str, args: tuple[str, ...]) -> str:
        out = self.templates[verb]
        names = [self.entities.get(a, a) for a in args]
        out = re.sub(r"\bX\b", lambda _m: names[0], out, count=1)
        if len(names) > 1:
            out = re.sub(r"\bY\b", lambda _m: names[1], out, count=1)
        return out

    def render_plan(self, plan) -> str:
        return "\n".join(f"{i}. {self.render(v, a)}" for i, (v, a) in enumerate(plan, 1))

    def schema(self) -> str:
        return " / ".join(self.templates[v] for v in CANON_VERBS)

    def _regexes(self):
        out = []
        for verb in CANON_VERBS:
            pat = re.escape(self.templates[verb].lower())
            pat = pat.replace(r"\ ", r"\s+")
            pat = re.sub(r"(?<![a-z0-9_\\-])x(?![a-z0-9_-])", r"(?P<x>[a-z0-9_]+)", pat, count=1)
            pat = re.sub(r"(?<![a-z0-9_\\-])y(?![a-z0-9_-])", r"(?P<y>[a-z0-9_]+)", pat, count=1)
            out.append((verb, re.compile(rf"^{pat}$")))
        return out

    def parse_line(self, line: str):
        """Return (verb, args) in canonical names, or None if the line is not a well-formed action."""
        norm = _normalise_action_line(line)
        if not norm:
            return None
        back = {str(v).lower(): k for k, v in self.entities.items()}
        for verb, rx in self._regexes():
            m = rx.match(norm)
            if not m:
                continue
            raw = [m.group("x")] + ([m.group("y")] if ARITY[verb] == 2 else [])
            return verb, tuple(back.get(a, a) for a in raw)
        return None


_LIST_PREFIX = re.compile(r"^\s*(?:step\s*)?\(?\d+\s*[\).:\-]\s*|^\s*[-*•]\s+", re.IGNORECASE)


def _normalise_action_line(line: str) -> str:
    s = _LIST_PREFIX.sub("", str(line)).strip()
    s = s.strip("`*_ \t").rstrip(".;,")
    s = s.lower()
    s = re.sub(r"\(([^()]*)\)", lambda m: " " + m.group(1).replace(",", " "), s)  # stack(a, b) -> stack a b
    s = re.sub(r"\b(?:the\s+)?blocks?\s+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_plan(answer_block: str, vocab: Vocab):
    """Parse a final-answer block into canonical actions.

    Strict by design: every non-empty line must be a well-formed action. Returns
    (plan, status, detail) with status in {parsed, empty, malformed_line}.
    """
    lines = [ln for ln in str(answer_block).splitlines() if ln.strip() and not re.fullmatch(r"\s*`{3}.*", ln)]
    if not lines:
        return [], "empty", "no lines in final answer"
    plan = []
    for ln in lines:
        act = vocab.parse_line(ln)
        if act is None:
            return plan, "malformed_line", f"not an action: {ln.strip()[:80]!r}"
        plan.append(act)
    return plan, "parsed", ""
