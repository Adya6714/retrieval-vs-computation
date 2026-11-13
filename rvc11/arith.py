"""Arithmetic word problems: numeric answer verification and text-level checks."""

from __future__ import annotations

import re

_NUM = re.compile(r"(?<![\w.])[-+]?\$?\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+(?![\d.]))?")

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _to_float(tok: str) -> float | None:
    t = tok.replace("$", "").replace(",", "").replace(" ", "")
    try:
        if "/" in t:
            a, b = t.split("/")
            return float(a) / float(b)
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def numbers_in(text: str) -> list[float]:
    """All numbers in a text, including spelled-out numbers below 100 ('thirty two', 'thirty-two')."""
    s = str(text).lower()
    out: list[float] = []
    tens = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
    units = "one|two|three|four|five|six|seven|eight|nine"
    for m in re.finditer(rf"\b({tens})[\s-]({units})\b", s):
        out.append(float(_WORDS[m.group(1)] + _WORDS[m.group(2)]))
    s = re.sub(rf"\b({tens})[\s-]({units})\b", " ", s)
    for m in re.finditer(r"\b(" + "|".join(_WORDS) + r")\b", s):
        out.append(float(_WORDS[m.group(1)]))
    for m in _NUM.finditer(s):
        v = _to_float(m.group(0))
        if v is not None:
            out.append(abs(v))
    return out


def verify_numeric(block: str, spec: dict) -> tuple[bool, str, str]:
    gold = float(spec["answer"])
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(block))
    vals = [v for v in (_to_float(m.group(0)) for m in _NUM.finditer(text)) if v is not None]
    distinct = sorted({round(v, 6) for v in vals})
    if not distinct:
        return False, "format_fail", "no number in final answer"
    if len(distinct) > 1:
        return False, "format_fail", f"final answer holds {len(distinct)} different numbers"
    pred = distinct[0]
    ok = abs(pred - gold) <= 1e-6 * max(1.0, abs(gold)) or round(pred, 2) == round(gold, 2)
    return ok, "parsed", "ok" if ok else f"predicted {pred:g}, expected {gold:g}"
