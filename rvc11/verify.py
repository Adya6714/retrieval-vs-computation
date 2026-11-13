"""Single scoring path: final-answer extraction + spec-based verification."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from . import algo, arith, planning
from .prompts import MARKER

_MARK = re.compile(r"\**\s*final\s+answer\s*\**\s*[:：]\s*\**", re.IGNORECASE)


@dataclass
class Verdict:
    correct: bool            # primary score (lenient extraction)
    correct_strict: bool     # marker present AND correct
    marker: bool             # the model used the FINAL ANSWER marker
    parse_status: str        # parsed | format_fail | empty | malformed_line | no_answer
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def extract_block(response: str) -> tuple[str, bool]:
    """Text after the last FINAL ANSWER marker. Without a marker, the whole response is returned."""
    text = str(response or "")
    hits = list(_MARK.finditer(text))
    if hits:
        return text[hits[-1].end():].strip(), True
    return text.strip(), False


def _tail_plan(text: str, vocab) -> str:
    """Lenient fallback for unmarked plans: the last unbroken run of well-formed action lines."""
    run: list[str] = []
    for ln in reversed([l for l in text.splitlines() if l.strip()]):
        if vocab.parse_line(ln) is None:
            if run:
                break
            continue
        run.append(ln)
    return "\n".join(reversed(run))


def _check(block: str, item: dict, marked: bool) -> tuple[bool, str, str]:
    spec = item["spec"]
    kind = spec["kind"]
    if not block.strip():
        return False, "no_answer", "empty response"
    if kind == "plan":
        vocab = planning.Vocab(spec["templates"], spec["entities"])
        if not marked:
            block = _tail_plan(block, vocab)
        plan, status, detail = planning.parse_plan(block, vocab)
        if status != "parsed":
            return False, status, detail
        ok, why = planning.simulate({tuple(f) for f in spec["init"]}, {tuple(f) for f in spec["goal"]}, plan)
        return ok, "parsed", why
    if kind == "numeric":
        if not marked:
            lines = [l for l in block.splitlines() if l.strip()]
            nums = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?", lines[-1]) if lines else []
            block = nums[-1] if nums else ""
        return arith.verify_numeric(block, spec)
    fn = {"sp": algo.verify_sp, "wis": algo.verify_wis, "cc": algo.verify_cc}[kind]
    return fn(block, spec)


def score(response: str, item: dict) -> Verdict:
    block, marked = extract_block(response)
    ok, status, reason = _check(block, item, marked)
    return Verdict(correct=bool(ok), correct_strict=bool(ok and marked), marker=marked, parse_status=status, reason=reason)


def gold_response(item: dict) -> str:
    return f"{MARKER}\n{item['gold_block']}"
