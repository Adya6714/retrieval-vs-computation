"""Call plans. An experiment is a deterministic list of call specs built from the bank and study.yaml.

Call spec keys: experiment, model, thinking, instruction, item_id, sample, arm, prompt.
"""

from __future__ import annotations

import hashlib
import re

from . import prompts, verify

PLAN_MARK = "Step"


def _order(items, salt: str):
    return sorted(items, key=lambda i: hashlib.sha256((salt + i["problem_id"]).encode()).hexdigest())


def call_id(spec: dict) -> str:
    key = "|".join(str(spec[k]) for k in ("experiment", "model", "thinking", "instruction", "item_id", "arm", "sample"))
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _thinking_modes(model_cfg: dict, wanted: list[str]) -> list[str]:
    return [t for t in wanted if (model_cfg.get("thinking") or {}).get(t) is not None]


def _spec(exp, model, thinking, instruction, item, sample, arm="main", prompt=None):
    s = {"experiment": exp, "model": model, "thinking": thinking, "instruction": instruction,
         "item_id": item["item_id"], "sample": sample, "arm": arm,
         "prompt": prompt if prompt is not None else prompts.build_prompt(item, instruction)}
    s["call_id"] = call_id(s)
    s["prompt_sha"] = hashlib.sha256(s["prompt"].encode()).hexdigest()[:16]
    return s


def _limit(items, n_problems):
    if not n_problems:
        return items
    keep = set()
    for sub in sorted({i["subtype"] for i in items}):
        pids = [i["problem_id"] for i in _order([i for i in items if i["subtype"] == sub and i["variant"] == "canonical"], "limit")]
        keep |= set(pids[:n_problems])
    return [i for i in items if i["problem_id"] in keep]


def build_plan(exp: str, study: dict, models: dict, bank: list[dict], model_keys: list[str], limit: int | None = None) -> list[dict]:
    cfg = study["experiments"][exp]
    items = [i for i in bank if i["status"] == "ok"]
    out: list[dict] = []
    if exp == "p1_surface":
        items = _limit(items, limit)
        for mk in model_keys:
            for th in _thinking_modes(models[mk], cfg["thinking"]):
                for it in items:
                    for s in range(cfg["samples"]):
                        out.append(_spec(exp, mk, th, "none", it, s))
    elif exp == "e1_instruction":
        items = _limit([i for i in items if i["variant"] in cfg["variants"]], limit)
        for mk in model_keys:
            for th in _thinking_modes(models[mk], cfg["thinking"]):
                for ins in cfg["instructions"]:
                    for it in items:
                        for s in range(cfg["samples"]):
                            out.append(_spec(exp, mk, th, ins, it, s))
    elif exp == "e2_noise":
        canon = [i for i in items if i["variant"] == "canonical" and i["tier"] == "verified"]
        chosen = []
        for fam in sorted({i["family"] for i in canon}):
            chosen += _order([i for i in canon if i["family"] == fam], "noise")[: (limit or cfg["problems_per_family"])]
        for mk in model_keys:
            for th in _thinking_modes(models[mk], cfg["thinking"]):
                for it in chosen:
                    for s in range(cfg["samples"]):
                        out.append(_spec(exp, mk, th, "none", it, s))
    elif exp == "p2_plan":
        canon = _order([i for i in items if i["subtype"] == "gsm" and i["variant"] == "canonical"], "p2")
        canon = canon[: limit] if limit else canon
        for mk in model_keys:
            for th in _thinking_modes(models[mk], cfg["thinking"]):
                for it in canon:
                    for s in range(cfg["samples"]):
                        out.append(_spec(exp, mk, th, "none", it, s, arm="plan", prompt=p2_plan_prompt(it)))
    else:
        raise KeyError(exp)
    return out


# ------------------------------------------------------------------ P2: plan, then continue under a false state
def p2_plan_prompt(item: dict) -> str:
    return (
        f"{item['task_text'].strip()}\n\n"
        "Solve the problem by writing every computation as its own numbered step, in this form:\n"
        "Step 1: <what you compute> = <number>\nStep 2: <what you compute> = <number>\n"
        "Use as many steps as you need. Each step must end with '= <number>'.\n\n"
        f"{prompts.answer_format(item)}"
    )


_STEP = re.compile(r"^\s*\**\s*Step\s+(\d+)\s*\**\s*[:.]\s*(.+?)=\s*\$?\s*(-?[\d,]*\.?\d+)\s*[^=\n]*$", re.IGNORECASE | re.MULTILINE)


def p2_parse_steps(response: str) -> list[tuple[int, str, float]]:
    body = verify._MARK.split(str(response))[0]
    steps = []
    for m in _STEP.finditer(body):
        try:
            steps.append((int(m.group(1)), m.group(2).strip().rstrip("=").strip(), float(m.group(3).replace(",", ""))))
        except ValueError:
            return []
    if [s[0] for s in steps] != list(range(1, len(steps) + 1)):
        return []
    return steps


def p2_false_value(v: float) -> float:
    """A wrong value of the same type and magnitude: +50%, never equal to the original."""
    w = v * 1.5 if v != 0 else 7.0
    if float(v).is_integer():
        w = float(round(w))
        if w == v:
            w = v + 1
    else:
        w = round(w, 2)
    return w


def _fmt(v: float) -> str:
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


def p2_continue_prompt(item: dict, steps, j: int, arm: str) -> str:
    shown = []
    for n, desc, val in steps[:j]:
        v = p2_false_value(val) if (n == j and arm in ("inject_false", "inject_cued")) else val
        shown.append(f"Step {n}: {desc} = {_fmt(v)}")
    cue = "\nOne of the steps above may contain an arithmetic error. Check them before you continue." if arm == "inject_cued" else ""
    return (
        f"{item['task_text'].strip()}\n\n"
        "The solution has been started below. Continue it from the next step, in the same form, until the problem is solved.\n\n"
        + "\n".join(shown) + cue + "\n\n" + prompts.answer_format(item)
    )


P2_ARMS = ("control_true", "inject_false", "inject_cued")


def build_p2_continue(plan_calls: list[dict], scored: dict, bank_by_id: dict) -> list[dict]:
    """Stage-B call specs from stage-A (arm=plan) results. Only correct, parseable plans with >=2
    steps are eligible; eligibility is reported by the analysis, never silently dropped."""
    out = []
    for c in plan_calls:
        if c.get("status") != "ok" or not scored.get(c["call_id"], {}).get("correct"):
            continue
        steps = p2_parse_steps(c["response"])
        if len(steps) < 2:
            continue
        j = (len(steps) + 1) // 2
        item = bank_by_id[c["item_id"]]
        for arm in P2_ARMS:
            s = {"experiment": "p2_continue", "model": c["model"], "thinking": c["thinking"], "instruction": "none",
                 "item_id": c["item_id"], "sample": c["sample"], "arm": arm,
                 "prompt": p2_continue_prompt(item, steps, j, arm)}
            s["call_id"] = call_id(s)
            s["prompt_sha"] = hashlib.sha256(s["prompt"].encode()).hexdigest()[:16]
            s["p2"] = {"n_steps": len(steps), "inject_step": j, "true_value": steps[j - 1][2], "false_value": p2_false_value(steps[j - 1][2])}
            out.append(s)
    return out
