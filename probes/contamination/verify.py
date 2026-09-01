"""Answer verification with family-specific handlers and orchestration."""

from __future__ import annotations

import json
import re

_PLAN_LINE_PREFIX = re.compile(r"^\s*\d+[\).\s]+")


def parse_action_mapping_from_notes(notes: str | None) -> dict[str, str] | None:
    """Extract W3 ``action_mapping`` from a question-bank ``notes`` field.

    ``generate_w3`` persists the mapping as a JSON object in ``notes``
    (sometimes after a `` | `` prefix). Returns None when no mapping is stored.
    """
    text = str(notes or "").strip()
    if not text or "action_mapping" not in text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            blob, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if not isinstance(blob, dict):
            continue
        raw = blob.get("action_mapping")
        if isinstance(raw, dict) and raw:
            return {str(k): str(v) for k, v in raw.items()}
    return None


_BW_MAPPING_KEYS = {"pick-up", "put-down", "stack", "unstack"}

_MYSTERY_ACTION_SIG = re.compile(
    r"([A-Za-z][A-Za-z0-9_-]*)\s+X(?:\s+Y)?\s*\(([^)]*)\)"
)


def recover_mystery_action_mapping(problem_text: str | None) -> dict[str, str] | None:
    """Recover canonical→renamed mystery verbs from the Available-actions preamble.

    Maps by arity + precondition keywords onto attack/succumb/overcome/feast.
    """
    text = str(problem_text or "")
    section = text
    m = re.search(
        r"Available actions:\s*(.+?)(?:Current |Present |INITIAL |Goal |Objective )",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        section = m.group(1)
    mapping: dict[str, str] = {}
    for verb, pre in _MYSTERY_ACTION_SIG.findall(section):
        name = verb.strip().lower()
        pre_l = pre.lower()
        arity2 = bool(re.search(rf"{re.escape(verb)}\s+X\s+Y\s*\(", section, flags=re.IGNORECASE))
        if "craves" in pre_l or "alliance" in pre_l:
            mapping["feast"] = name
        elif arity2 and (
            "pain" in pre_l or "tension" in pre_l or "province" in pre_l or "sovereignty" in pre_l
        ):
            mapping["overcome"] = name
        elif (not arity2) and ("pain" in pre_l or "tension" in pre_l):
            mapping["succumb"] = name
        elif (not arity2) and (
            "harmony" in pre_l or "goodwill" in pre_l or "planet" in pre_l or "influence" in pre_l
        ):
            mapping["attack"] = name
    return mapping or None


def mystery_action_mapping(
    notes: str | None, problem_text: str | None, explicit: dict[str, str] | None = None
) -> dict[str, str] | None:
    """Prefer an explicit mapping unless it is a Blocksworld pick-up/stack dict."""
    if explicit:
        keys = {str(k).strip().lower() for k in explicit}
        if keys & _BW_MAPPING_KEYS:
            recovered = recover_mystery_action_mapping(problem_text)
            return recovered or explicit
        return explicit
    from_notes = parse_action_mapping_from_notes(notes)
    if from_notes:
        keys = {str(k).strip().lower() for k in from_notes}
        if not (keys & _BW_MAPPING_KEYS):
            return from_notes
    return recover_mystery_action_mapping(problem_text)


def _canonical_verb_aliases(action_mapping: dict[str, str] | None) -> dict[str, str]:
    """Invert pick-up→renamed mapping to renamed→canonical, with _/- aliases."""
    if not action_mapping:
        return {}
    out: dict[str, str] = {}
    for canonical, renamed in action_mapping.items():
        canon = str(canonical).strip().lower()
        raw = str(renamed).strip().lower()
        for key in {raw, raw.replace("_", "-"), raw.replace("-", "_")}:
            if key:
                out[key] = canon
    return out


def _rewrite_leading_verb(norm: str, verb_aliases: dict[str, str]) -> str:
    if not verb_aliases or not norm:
        return norm
    parts = norm.split()
    if not parts:
        return norm
    canon = verb_aliases.get(parts[0])
    if canon is None:
        return norm
    parts[0] = canon
    return " ".join(parts)


def _strip_numbered_plan_lines(text: str) -> list[str]:
    """One logical line per list item; strip leading ``1.`` / ``2)`` etc."""
    out: list[str] = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _PLAN_LINE_PREFIX.sub("", line).strip()
        if line:
            out.append(line)
    return out


_BLOCKSWORLD_LINE = re.compile(
    r"^(pick-up|put-down|stack|unstack)\s+([a-z0-9_-]+)(?:\s+([a-z0-9_-]+))?\s*$",
    re.IGNORECASE,
)


def _extract_blocksworld_actions_line_based(
    text: str, action_mapping: dict[str, str] | None = None
) -> list[str]:
    """Parse blocksworld plans line-by-line so numbered lists do not break regex.

    When ``action_mapping`` is the W3 rename dict (canonical → renamed), invert
    it and rewrite the leading verb before matching the four primitives.
    """
    aliases = _canonical_verb_aliases(action_mapping)
    actions: list[str] = []
    for line in _strip_numbered_plan_lines(text):
        norm = line.strip().lower()
        norm = re.sub(r"\bblock\s+", "", norm)
        if aliases:
            norm = _rewrite_leading_verb(norm, aliases)
        norm = re.sub(r"^select\s+([a-z0-9]+)$", r"pick-up \1", norm)
        norm = re.sub(r"^release\s+([a-z0-9]+)$", r"put-down \1", norm)
        norm = re.sub(r"^place\s+([a-z0-9]+)\s+under\s+([a-z0-9]+)$", r"stack \1 \2", norm)
        norm = re.sub(r"^place\s+([a-z0-9]+)\s+on\s+([a-z0-9]+)$", r"stack \1 \2", norm)
        norm = re.sub(r"^remove\s+([a-z0-9]+)\s+from\s+([a-z0-9]+)$", r"unstack \1 \2", norm)

        m = _BLOCKSWORLD_LINE.match(norm)
        if not m:
            continue
        parts = [m.group(1).lower(), m.group(2).lower()]
        if m.group(3):
            parts.append(m.group(3).lower())
        actions.append(" ".join(parts))
    return actions


_MYSTERY_LINE = re.compile(
    r"^(attack|succumb|overcome|broker|feast)\s+([a-z0-9_-]+)(?:\s+([a-z0-9_-]+))?\s*$",
    re.IGNORECASE,
)


def _extract_mystery_actions_line_based(
    text: str, action_mapping: dict[str, str] | None = None
) -> list[str]:
    aliases = _canonical_verb_aliases(action_mapping)
    actions: list[str] = []
    for line in _strip_numbered_plan_lines(text):
        norm = line.strip().lower()
        if aliases:
            norm = _rewrite_leading_verb(norm, aliases)
        m = _MYSTERY_LINE.match(norm)
        if not m:
            continue
        parts = [m.group(1).lower(), m.group(2).lower()]
        if m.group(3):
            parts.append(m.group(3).lower())
        actions.append(" ".join(parts))
    return actions


def _extract_actions(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [m.group(0).strip().lower() for m in pattern.finditer(str(text).lower())]


def _verify_numeric(model_answer, ground_truth) -> bool:
    match = re.search(r"[-+]?\d*\.?\d+", str(model_answer))
    if not match:
        return False
    try:
        model_val = float(match.group())
        gt_val = float(ground_truth)
        return abs(model_val - gt_val) <= 1e-6
    except ValueError:
        return False


def verify_gsm_answer(model_response: str, correct_answer) -> bool:
    """Verify GSM-style numeric answers with tolerant extraction."""
    try:
        gt_val = float(str(correct_answer).replace(",", "").strip())
    except (TypeError, ValueError):
        return False

    response = str(model_response or "")

    tagged = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", response)
    if tagged:
        try:
            pred_val = float(tagged.group(1).replace(",", ""))
            return abs(pred_val - gt_val) < 0.01
        except ValueError:
            return False

    numbers = re.findall(r"(?<![\w])\$?-?[\d,]+(?:\.\d+)?(?![\w])", response)
    if not numbers:
        return False

    candidate = numbers[-1].replace("$", "").replace(",", "")
    try:
        pred_val = float(candidate)
    except ValueError:
        return False
    return abs(pred_val - gt_val) < 0.01


LAST_VERIFY_META: dict[str, str] = {}

VERIFY_STATE_MACHINE = "state_machine"
VERIFY_EXACT_SEQUENCE = "exact_sequence"
VERIFY_STRING_EQUALITY = "string_equality"


def _set_verify_meta(*, verify_method: str) -> None:
    LAST_VERIFY_META.clear()
    LAST_VERIFY_META["verify_method"] = verify_method


def _verify_shortest_path(model_answer, ground_truth) -> bool:
    parts = re.split(r"[,\- \>]+", str(model_answer).strip().upper())
    model_path = "".join([p for p in parts if len(p) == 1 and p.isalpha()])
    gt_path = "".join([p for p in str(ground_truth).strip().upper() if p.isalpha()])
    return model_path == gt_path


_BW_NAME = (
    r"(?:(?!(?:and|the|block|blocks|which|with|your|currently|atop|over|above|"
    r"labeled|all|must|be|on|top|of|table|surface|nations?)\b)[a-z][a-z0-9_-]*)"
)
_BW_STOP = {
    "block", "blocks", "the", "and", "table", "surface", "which", "with", "your",
    "currently", "atop", "over", "above", "labeled", "all", "must", "be", "on",
    "top", "of",
}
_NAME_LIST = (
    rf"((?:{_BW_NAME})(?:\s*,\s*(?:{_BW_NAME}))*(?:,?\s+and\s+(?:{_BW_NAME}))?)"
)
_CURRENT_START = re.compile(
    r"(?:current\s+(?:state|configuration|situation)|present\s+(?:situation|state)|"
    r"initial\s+state(?:\s+s[0₀])?|INITIAL\s+STATE\s+S[0₀])\s*:?",
    re.IGNORECASE,
)
_GOAL_START = re.compile(
    r"(?:goal\s+state(?:\s+s[*＊])?|GOAL\s+STATE\s+S[*＊]|target\s+configuration|"
    r"objective|goal|target)\s*:?",
    re.IGNORECASE,
)
_SECTION_END = re.compile(
    r"(?:respond with|reply with|reply using|provide only|what sequence|"
    r"actions\s*\(|task:|output format:|include no explanation|"
    r"each action must|every action (?:must|should))",
    re.IGNORECASE,
)


def _extract_current_and_goal(problem_text: str) -> tuple[str, str] | None:
    text = str(problem_text or "")
    if not text.strip():
        return None
    cur_m = _CURRENT_START.search(text)
    if not cur_m:
        return None
    after_cur = text[cur_m.end() :]
    goal_m = _GOAL_START.search(after_cur)
    if not goal_m:
        return None
    current = after_cur[: goal_m.start()]
    rest = after_cur[goal_m.end() :]
    end_m = _SECTION_END.search(rest)
    goal = rest[: end_m.start()] if end_m else rest
    return current, goal


def _split_name_list(blob: str) -> list[str]:
    names: list[str] = []
    for part in re.split(r",|\band\b", blob, flags=re.IGNORECASE):
        part = re.sub(
            r"^(?:blocks?|the\s+blocks?|nations?|the\s+nations?|labeled)\s+",
            "",
            part.strip(),
            flags=re.IGNORECASE,
        )
        m = re.match(rf"^({_BW_NAME})\b", part.strip(), flags=re.IGNORECASE)
        if m and m.group(1).lower() not in _BW_STOP:
            names.append(m.group(1).lower())
    return names


def _parse_markdown_tables(text: str) -> list[dict[str, object]]:
    tables: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            if current is not None:
                tables.append(current)
                current = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-+:?", c.replace(" ", "") or "-") for c in cells):
            continue
        if current is None:
            current = {"headers": [c.lower() for c in cells], "rows": []}
        else:
            current["rows"].append(cells)  # type: ignore[index]
    if current is not None:
        tables.append(current)
    return tables


def _facts_from_bw_tables(section: str) -> set[tuple]:
    facts: set[tuple] = set()
    for table in _parse_markdown_tables(section):
        headers = [str(h) for h in table["headers"]]  # type: ignore[index]
        rows = table["rows"]  # type: ignore[index]
        joined = " ".join(headers)
        for row in rows:  # type: ignore[union-attr]
            if not row:
                continue
            block = str(row[0]).strip().lower()
            if not re.fullmatch(_BW_NAME, block):
                continue
            loc = str(row[1]).strip().lower() if len(row) > 1 else ""
            extra = str(row[2]).strip().lower() if len(row) > 2 else ""
            if "location" in joined or "position" in joined or "must be on" in joined or "target" in joined:
                loc_norm = re.sub(r"^on\s+", "", loc).strip()
                if loc_norm in {"", "—", "-", "n/a"}:
                    continue
                if loc_norm in {"table", "on table"} or loc_norm.endswith("table"):
                    facts.add(("ontable", block))
                    facts.add(("clear", block))
                elif re.fullmatch(_BW_NAME, loc_norm):
                    facts.add(("on", block, loc_norm))
            if "clear" in joined:
                if extra in {"yes", "y", "true"} or loc in {"yes", "y", "true"}:
                    facts.add(("clear", block))
                elif extra in {"no", "n", "false"}:
                    facts.discard(("clear", block))
    return facts


def _parse_bw_predicates(section: str) -> set[tuple]:
    text = str(section)
    facts: set[tuple] = set()
    for b in re.findall(rf"ontable\s*\(\s*({_BW_NAME})\s*\)", text, flags=re.IGNORECASE):
        facts.add(("ontable", b.lower()))
    for b in re.findall(rf"clear\s*\(\s*({_BW_NAME})\s*\)", text, flags=re.IGNORECASE):
        facts.add(("clear", b.lower()))
    for x, y in re.findall(
        rf"on\s*\(\s*({_BW_NAME})\s*,\s*({_BW_NAME})\s*\)", text, flags=re.IGNORECASE
    ):
        facts.add(("on", x.lower(), y.lower()))
    if re.search(r"\bhandempty\b", text, flags=re.IGNORECASE):
        facts.add(("handempty",))
    return facts


def _parse_bw_prose_facts(section: str) -> set[tuple]:
    text = " ".join(str(section).lower().split())
    facts: set[tuple] = set()

    if re.search(
        r"hand(?:s)? (?:is |are |currently )?(?:empty|free|vacant|unoccupied)"
        r"|gripper (?:is |currently )?(?:vacant|free|unoccupied|empty)"
        r"|your hands are free|hand being unoccupied",
        text,
    ):
        facts.add(("handempty",))

    list_pats_clear_table = (
        rf"(?:(?:the )?blocks?(?:\s+labeled)?|all blocks labeled)\s+{_NAME_LIST}\s+"
        rf"are (?:all )?(?:clear|unobstructed|accessible) and "
        rf"(?:on|resting on|sitting on|positioned on) (?:the )?(?:table|surface)",
        rf"(?:(?:the )?blocks?(?:\s+labeled)?|all blocks labeled)\s+{_NAME_LIST}\s+"
        rf"are (?:all )?unassigned and available",
        rf"(?:the )?blocks?\s+{_NAME_LIST}\s+(?:are|sit|sits) (?:all )?(?:unobstructed|clear|accessible)"
        rf" (?:and )?(?:resting |sitting |positioned )?on (?:the )?(?:table|surface)",
        rf"^({_NAME_LIST})\s+are all unassigned and available",
        rf"(?:the )?blocks?\s+{_NAME_LIST}\s+sit unobstructed on (?:the )?(?:table|surface)",
        rf"on the table sit blocks?\s+{_NAME_LIST}",
    )
    for pat in list_pats_clear_table:
        for m in re.finditer(pat, text):
            names = _split_name_list(m.group(1))
            for b in names:
                facts.add(("clear", b))
                facts.add(("ontable", b))

    for m in re.finditer(
        rf"(?:the )?blocks?\s+{_NAME_LIST}\s+are (?:all )?(?:on the table|on the surface|on the table surface)\b",
        text,
    ):
        for b in _split_name_list(m.group(1)):
            facts.add(("ontable", b))

    for x, y in re.findall(
        rf"block\s+({_BW_NAME})\s+(?:is on|rests atop|sits atop|should be placed on|"
        rf"is placed on|is stacked on|is positioned on|sits on|rests on)\s+(?:block\s+)?({_BW_NAME})",
        text,
    ):
        if (
            x in _BW_STOP
            or y in _BW_STOP
            or x in {"block", "blocks"}
            or y in {"block", "blocks", "the", "table", "surface"}
        ):
            continue
        facts.add(("on", x, y))
    for x, y in re.findall(rf"({_BW_NAME})\s+reports to\s+({_BW_NAME})", text):
        facts.add(("on", x, y))
    for x, y in re.findall(rf"({_BW_NAME})\s+is reporting to\s+({_BW_NAME})", text):
        facts.add(("on", x, y))
    for x, y in re.findall(
        rf"(?:position|place|set)\s+block\s+({_BW_NAME})\s+(?:atop|over|above)\s+block\s+({_BW_NAME})",
        text,
    ):
        if x not in _BW_STOP and y not in _BW_STOP:
            facts.add(("on", x, y))
    for x, y in re.findall(
        rf"block\s+({_BW_NAME})\s+should be (?:on top of|placed on|on)\s+block\s+({_BW_NAME})",
        text,
    ):
        if x not in _BW_STOP and y not in _BW_STOP:
            facts.add(("on", x, y))

    if "which" in text:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if "which" not in sentence:
                continue
            names = re.findall(rf"block\s+({_BW_NAME})", sentence)
            if len(names) >= 2:
                for a, b in zip(names, names[1:]):
                    facts.add(("on", a, b))

    for b in re.findall(rf"block\s+({_BW_NAME})\s+is on the (?:table|surface)", text):
        facts.add(("ontable", b))
    for b in re.findall(rf"block\s+({_BW_NAME})\s+is (?:on the (?:table|surface) and )?(?:clear|unobstructed)", text):
        facts.add(("clear", b))
    for b in re.findall(
        rf"block\s+({_BW_NAME})\s+is on the (?:table|surface) and (?:clear|unobstructed)", text
    ):
        facts.add(("ontable", b))
        facts.add(("clear", b))
    for m in re.finditer(
        rf"(?:the )?blocks?\s+{_NAME_LIST}\s+are (?:all )?(?:clear|unobstructed)\b",
        text,
    ):
        for b in _split_name_list(m.group(1)):
            facts.add(("clear", b))
    for b in re.findall(rf"({_BW_NAME})\s+is unassigned(?: and available)?", text):
        facts.add(("clear", b))
        facts.add(("ontable", b))
        facts.add(("clear", b))
        facts.add(("ontable", b))

    return facts


def _infer_bw_current_derived(state: set[tuple]) -> None:
    """Fill clear/ontable from on-relations so stacked W5 initials are simulable."""
    on_of: dict[str, str] = {}
    blocks: set[str] = set()
    for fact in list(state):
        if fact[0] == "on" and len(fact) == 3:
            on_of[fact[1]] = fact[2]
            blocks.add(fact[1])
            blocks.add(fact[2])
        elif fact[0] in {"clear", "ontable"} and len(fact) == 2:
            blocks.add(fact[1])
    if not on_of:
        return
    occupied = set(on_of.values())
    for b in blocks:
        if b not in occupied:
            state.add(("clear", b))
        if b not in on_of:
            state.add(("ontable", b))


def _bw_state_usable(parsed: tuple[set[tuple], set[tuple]] | None) -> bool:
    if not parsed:
        return False
    state, goal = parsed
    content = {f[0] for f in state} | {f[0] for f in goal}
    return bool(content & {"on", "ontable", "clear"})


def _parse_blocksworld_state(problem_text: str) -> tuple[set[tuple], set[tuple]] | None:
    sections = _extract_current_and_goal(problem_text)
    if sections is None:
        return None
    current, goal = sections
    state: set[tuple] = set()
    goal_facts: set[tuple] = set()
    state.update(_parse_bw_predicates(current))
    state.update(_parse_bw_prose_facts(current))
    state.update(_facts_from_bw_tables(current))
    goal_facts.update(_parse_bw_predicates(goal))
    goal_facts.update(_parse_bw_prose_facts(goal))
    goal_facts.update(_facts_from_bw_tables(goal))
    goal_facts = {f for f in goal_facts if f[0] in {"on", "ontable", "clear"}}
    _infer_bw_current_derived(state)
    if ("handempty",) not in state and not any(f[0] == "holding" for f in state):
        state.add(("handempty",))
    if not state and not goal_facts:
        return None
    return state, goal_facts


def _apply_blocksworld_action(state: set[tuple], action: str) -> bool:
    parts = action.split()
    if not parts:
        return False
    verb = parts[0]
    if verb in {"pick-up", "put-down"} and len(parts) != 2:
        return False
    if verb in {"stack", "unstack"} and len(parts) != 3:
        return False

    if verb == "pick-up":
        x = parts[1]
        pre = {("clear", x), ("ontable", x), ("handempty",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("clear", x), ("ontable", x), ("handempty",)})
        state.add(("holding", x))
        return True
    if verb == "put-down":
        x = parts[1]
        pre = {("holding", x)}
        if not pre.issubset(state):
            return False
        state.remove(("holding", x))
        state.update({("ontable", x), ("clear", x), ("handempty",)})
        return True
    if verb == "stack":
        x, y = parts[1], parts[2]
        pre = {("holding", x), ("clear", y)}
        if not pre.issubset(state):
            return False
        state.difference_update({("holding", x), ("clear", y)})
        state.update({("on", x, y), ("clear", x), ("handempty",)})
        return True
    if verb == "unstack":
        x, y = parts[1], parts[2]
        pre = {("on", x, y), ("clear", x), ("handempty",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("on", x, y), ("clear", x), ("handempty",)})
        state.update({("holding", x), ("clear", y)})
        return True
    return False


def _verify_blocksworld_state_machine(
    model_answer, problem_text, action_mapping: dict[str, str] | None = None
) -> bool | None:
    parsed = _parse_blocksworld_state(problem_text)
    if parsed is None:
        return None
    state, goal = parsed
    actions = _extract_blocksworld_actions_line_based(
        model_answer, action_mapping=action_mapping
    )
    if not actions:
        return False
    for action in actions:
        if not _apply_blocksworld_action(state, action):
            return False
    return goal.issubset(state)


def _parse_mystery_predicates(section: str) -> set[tuple]:
    text = str(section)
    facts: set[tuple] = set()
    if re.search(r"\bharmony\b", text, flags=re.IGNORECASE) and re.search(
        r"harmony is true|\bharmony\b", text, flags=re.IGNORECASE
    ):
        if re.search(r"harmony is true|\bharmony,", text, flags=re.IGNORECASE) or re.search(
            r"^harmony\s*$", text.strip(), flags=re.IGNORECASE | re.MULTILINE
        ):
            facts.add(("harmony",))
    for kind, pred in (("planet", "planet"), ("province", "province")):
        for b in re.findall(rf"{kind}\s*\(\s*({_BW_NAME})\s*\)", text, flags=re.IGNORECASE):
            facts.add((pred, b.lower()))
    for x, y in re.findall(
        rf"craves\s*\(\s*({_BW_NAME})\s*,\s*({_BW_NAME})\s*\)", text, flags=re.IGNORECASE
    ):
        facts.add(("craves", x.lower(), y.lower()))
    return facts


def _parse_mystery_prose_facts(section: str) -> set[tuple]:
    text = " ".join(str(section).lower().split())
    facts: set[tuple] = set()
    if re.search(r"harmony is true|goodwill is true", text):
        facts.add(("harmony",))
    for m in re.finditer(
        r"(?:planet and province|influence and sovereignty) are true for "
        r"(?:blocks?|nations?)?\s*(.+?)(?:\.|$)",
        text,
    ):
        for b in _split_name_list(m.group(1)):
            facts.add(("planet", b))
            facts.add(("province", b))
    for x, y in re.findall(rf"craves\s+({_BW_NAME})\s+({_BW_NAME})", text):
        facts.add(("craves", x, y))
    for x, y in re.findall(rf"({_BW_NAME})\s+allies with\s+({_BW_NAME})", text):
        facts.add(("craves", x, y))
    return facts


def _facts_from_mystery_tables(section: str) -> set[tuple]:
    facts: set[tuple] = set()
    for table in _parse_markdown_tables(section):
        headers = [str(h) for h in table["headers"]]  # type: ignore[index]
        rows = table["rows"]  # type: ignore[index]
        joined = " ".join(headers)
        for row in rows:  # type: ignore[union-attr]
            if not row:
                continue
            block = str(row[0]).strip().lower()
            if not re.fullmatch(_BW_NAME, block):
                continue
            if "planet" in joined or "province" in joined:
                cells = [str(c).strip().lower() for c in row[1:]]
                # Position column may pack "province, planet"
                packed = " ".join(cells)
                if "planet" in packed or (len(row) > 1 and str(row[1]).lower() in {"yes", "y", "true"}):
                    if "planet" in joined or "planet" in packed:
                        facts.add(("planet", block))
                if "province" in packed or (
                    "province" in joined and len(row) > 2 and str(row[2]).lower() in {"yes", "y", "true"}
                ):
                    facts.add(("province", block))
                if str(row[1]).lower() in {"yes", "y", "true"} and "planet" in joined:
                    facts.add(("planet", block))
                if len(row) > 2 and str(row[2]).lower() in {"yes", "y", "true"} and "province" in joined:
                    facts.add(("province", block))
            if "crave" in joined or "target" in joined or "position" in joined:
                loc = str(row[1]).strip().lower() if len(row) > 1 else ""
                loc = re.sub(r"^on\s+", "", loc).strip()
                if loc and re.fullmatch(_BW_NAME, loc):
                    facts.add(("craves", block, loc))
    return facts


def _infer_mystery_current_derived(state: set[tuple]) -> None:
    craves_of: dict[str, str] = {}
    blocks: set[str] = set()
    for fact in list(state):
        if fact[0] == "craves" and len(fact) == 3:
            craves_of[fact[1]] = fact[2]
            blocks.add(fact[1])
            blocks.add(fact[2])
        elif fact[0] in {"planet", "province"} and len(fact) == 2:
            blocks.add(fact[1])
    for b in blocks:
        state.add(("planet", b))
        state.add(("province", b))
    if craves_of:
        occupied = set(craves_of.values())
        for b in blocks:
            if b in occupied:
                state.discard(("province", b))
        if ("harmony",) not in state:
            state.add(("harmony",))


def _mystery_state_usable(parsed: tuple[set[tuple], set[tuple]] | None) -> bool:
    if not parsed:
        return False
    state, goals = parsed
    content = {f[0] for f in state} | {f[0] for f in goals}
    return bool(content & {"planet", "province", "craves", "harmony"})


def _parse_mystery_state(problem_text: str) -> tuple[set[tuple], set[tuple]] | None:
    sections = _extract_current_and_goal(problem_text)
    if sections is None:
        return None
    current, goal = sections
    state: set[tuple] = set()
    goals: set[tuple] = set()
    state.update(_parse_mystery_predicates(current))
    state.update(_parse_mystery_prose_facts(current))
    state.update(_facts_from_mystery_tables(current))
    goals.update(_parse_mystery_predicates(goal))
    goals.update(_parse_mystery_prose_facts(goal))
    goals.update(_facts_from_mystery_tables(goal))
    _infer_mystery_current_derived(state)
    if not state and not goals:
        return None
    return state, goals


def _apply_mystery_action(state: set[tuple], action: str) -> bool:
    parts = action.split()
    if not parts:
        return False
    verb = parts[0]
    if verb in {"attack", "succumb"} and len(parts) != 2:
        return False
    if verb in {"overcome", "broker", "feast"} and len(parts) != 3:
        return False

    if verb == "attack":
        x = parts[1]
        pre = {("province", x), ("planet", x), ("harmony",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("province", x), ("planet", x), ("harmony",)})
        state.add(("pain", x))
        return True
    if verb == "succumb":
        x = parts[1]
        pre = {("pain", x)}
        if not pre.issubset(state):
            return False
        state.remove(("pain", x))
        state.update({("province", x), ("planet", x), ("harmony",)})
        return True
    if verb in {"overcome", "broker"}:
        x, y = parts[1], parts[2]
        pre = {("pain", x), ("province", y)}
        if not pre.issubset(state):
            return False
        state.difference_update({("pain", x), ("province", y)})
        state.update({("harmony",), ("province", x), ("craves", x, y)})
        return True
    if verb == "feast":
        x, y = parts[1], parts[2]
        pre = {("craves", x, y), ("province", x), ("harmony",)}
        if not pre.issubset(state):
            return False
        state.difference_update({("craves", x, y), ("province", x), ("harmony",)})
        state.update({("pain", x), ("province", y)})
        return True
    return False


def _verify_mystery_state_machine(
    model_answer, problem_text, action_mapping: dict[str, str] | None = None
) -> bool | None:
    parsed = _parse_mystery_state(problem_text)
    if parsed is None:
        return None
    state, goals = parsed
    actions = _extract_mystery_actions_line_based(
        model_answer, action_mapping=action_mapping
    )
    if not actions:
        return False
    for action in actions:
        if not _apply_mystery_action(state, action):
            return False
    return goals.issubset(state)


def _explicit_sequence_or_string(
    model_answer,
    ground_truth,
    extract_fn,
    fallback_pattern: re.Pattern[str],
) -> bool:
    model_matches = extract_fn(model_answer)
    gt_matches = extract_fn(ground_truth)
    if model_matches and gt_matches:
        _set_verify_meta(verify_method=VERIFY_EXACT_SEQUENCE)
        return model_matches == gt_matches
    model_matches = _extract_actions(model_answer, fallback_pattern)
    gt_matches = _extract_actions(ground_truth, fallback_pattern)
    if model_matches and gt_matches:
        _set_verify_meta(verify_method=VERIFY_EXACT_SEQUENCE)
        return model_matches == gt_matches
    _set_verify_meta(verify_method=VERIFY_STRING_EQUALITY)
    return str(model_answer).strip().lower() == str(ground_truth).strip().lower()


def verify_answer(
    problem_id,
    model_answer,
    ground_truth,
    family,
    problem_text=None,
    action_mapping=None,
):
    numeric_families = {
        "gsm", 
        "shortest_path", 
        "weighted_interval_scheduling", 
        "coin_change", 
        "knapsack"
    }
    plan_families = {"blocksworld", "logistics", "mystery_blocksworld"}
    has_problem_text = bool(str(problem_text or "").strip()) and str(problem_text).strip() != "None"

    if family == "shortest_path":
        return _verify_shortest_path(model_answer, ground_truth)

    elif family == "arithmetic_reasoning":
        return verify_gsm_answer(model_answer, ground_truth)

    elif family in numeric_families:
        return _verify_numeric(model_answer, ground_truth)

    elif family == "mystery_blocksworld":
        action_mapping = mystery_action_mapping(None, problem_text, explicit=action_mapping)
        mystery_parsed = _parse_mystery_state(problem_text)
        if _mystery_state_usable(mystery_parsed):
            sim_ok = _verify_mystery_state_machine(
                model_answer, problem_text, action_mapping=action_mapping
            )
            _set_verify_meta(verify_method=VERIFY_STATE_MACHINE)
            return bool(sim_ok is True)
        if has_problem_text and _bw_state_usable(_parse_blocksworld_state(problem_text)):
            sim_ok = _verify_blocksworld_state_machine(
                model_answer, problem_text, action_mapping=action_mapping
            )
            _set_verify_meta(verify_method=VERIFY_STATE_MACHINE)
            if sim_ok is True:
                return True
            if sim_ok is False:
                return False
            return None
        if has_problem_text:
            _set_verify_meta(verify_method=VERIFY_STATE_MACHINE)
            return None
        mystery_pattern = re.compile(
            r"(attack|succumb|overcome|broker|feast)\s+[a-z0-9_-]+(\s+[a-z0-9_-]+)?",
            re.IGNORECASE,
        )
        return _explicit_sequence_or_string(
            model_answer,
            ground_truth,
            lambda t: _extract_mystery_actions_line_based(t, action_mapping=action_mapping),
            mystery_pattern,
        )

    elif family in plan_families:
        parsed = _parse_blocksworld_state(problem_text)
        if _bw_state_usable(parsed):
            sim_ok = _verify_blocksworld_state_machine(
                model_answer, problem_text, action_mapping=action_mapping
            )
            _set_verify_meta(verify_method=VERIFY_STATE_MACHINE)
            return bool(sim_ok is True)
        if has_problem_text:
            _set_verify_meta(verify_method=VERIFY_STATE_MACHINE)
            return None
        action_pattern = re.compile(
            r"(pick-up|put-down|stack|unstack)\s+[a-z0-9_-]+(\s+[a-z0-9_-]+)?",
            re.IGNORECASE,
        )
        return _explicit_sequence_or_string(
            model_answer,
            ground_truth,
            lambda t: _extract_blocksworld_actions_line_based(t, action_mapping=action_mapping),
            action_pattern,
        )

    else:
        valid_families = numeric_families | plan_families | {"shortest_path"}
        raise ValueError(f"Unrecognized family: '{family}'. Expected one of {sorted(valid_families)}")
