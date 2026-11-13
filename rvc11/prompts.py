"""Prompt construction. One code path for every family, variant, model and experiment.

prompt = task_text + ANSWER FORMAT block + INSTRUCTION block

* task_text comes from the bank and contains no instruction of any kind.
* The answer-format block depends only on the subtype (and, for planning, on the action names
  that the variant itself uses). Its wording is identical across variants.
* The instruction block is the experimental factor of the instruction-drift experiment.
"""

from __future__ import annotations

from .planning import CANON_VERBS

MARKER = "FINAL ANSWER:"

INSTRUCTIONS = {
    # level -> text appended after the answer-format block
    "none": "",
    "show_work": "Work through the problem step by step before you give the final answer.",
    "answer_only": "Do not show any working or explanation. Output only the final answer section.",
}

_FORMAT = {
    "gsm": "a single number with no units and no other text",
    "shortest_path": "exactly two lines:\nPath: <first location> -> <next location> -> ... -> <last location>\nCost: <number>\nName each location exactly as the problem names it.",
    "wis": "exactly two lines:\nSelected: <item>, <item>, ...\nTotal: <number>\nName each item exactly as the problem names it.",
    "coin_change": "exactly two lines:\nCount: <integer>\nValues: <value>, <value>, ...\nList one value for every unit you use, with repeats.",
}


def answer_format(item: dict) -> str:
    sub = item["subtype"]
    if item["family"] == "planning":
        schema = " / ".join(item["spec"]["templates"][v] for v in CANON_VERBS)
        body = ("a numbered list with one action per line and nothing else. Each action must be "
                f"exactly one of: {schema}")
    else:
        body = _FORMAT[sub]
    return f"End your response with a line that says exactly \"{MARKER}\" followed by {body}"


def build_prompt(item: dict, instruction: str = "none") -> str:
    if instruction not in INSTRUCTIONS:
        raise KeyError(f"unknown instruction level {instruction!r}")
    parts = [item["task_text"].strip(), answer_format(item)]
    if INSTRUCTIONS[instruction]:
        parts.append(INSTRUCTIONS[instruction])
    return "\n\n".join(parts)


def scaffold(item: dict, instruction: str = "none") -> str:
    """Everything in the prompt that is not task text, with variant-specific action names masked.

    QA asserts this string is identical for all variants of a subtype: any difference between two
    prompts of the same problem is then, by construction, inside task_text."""
    text = build_prompt({**item, "task_text": ""}, instruction)
    if item["family"] == "planning":
        schema = " / ".join(item["spec"]["templates"][v] for v in CANON_VERBS)
        text = text.replace(schema, "<ACTION SCHEMA>")
    return text
